import { decodeAudioBlob, float16BitsToFloat32 } from "../src/runtime/audio.js";
import { DiCoSeSeparator, type SeparatorTiming } from "../src/runtime/separator.js";
import { DICOSE_STEM_NAMES, type DiCoSeStemName } from "../src/worker-protocol.js";

interface SampledStemReference {
  readonly leftF32LeBase64: string;
  readonly rightF32LeBase64: string;
}

interface WaveformReference {
  readonly fixture: {
    readonly sampleRate: number;
    readonly samples: number;
  };
  readonly sampling: {
    readonly offset: number;
    readonly stride: number;
    readonly countPerChannel: number;
  };
  readonly stems: Readonly<Record<DiCoSeStemName, SampledStemReference>>;
}

interface ReferenceMetrics {
  readonly samples: number;
  readonly nrmse: number;
  readonly snrDb: number;
  readonly cosine: number;
  readonly maxAbs: number;
  readonly referenceRms: number;
  readonly candidateRms: number;
  readonly bitExactFraction?: number;
}

interface StageReference {
  readonly samplesPerStage: number;
  readonly stages: Readonly<Record<string, {
    readonly elements: number;
    readonly valuesF16LeBase64: string;
  }>>;
}

interface ReferenceQualityReport {
  readonly ok: boolean;
  readonly withinEnvelope?: boolean;
  readonly timing?: SeparatorTiming;
  readonly stems?: Readonly<Record<DiCoSeStemName, ReferenceMetrics>>;
  readonly stages?: Readonly<Record<string, ReferenceMetrics>>;
  readonly error?: string;
}

const statusNode = document.querySelector<HTMLElement>("#status");
const resultNode = document.querySelector<HTMLElement>("#result");
const harness = {} as { report?: ReferenceQualityReport };
(globalThis as typeof globalThis & { __DICOSE_BROWSER__?: typeof harness }).__DICOSE_BROWSER__ = harness;

void run();

async function run(): Promise<void> {
  let separator: DiCoSeSeparator | undefined;
  try {
    const query = new URL(location.href).searchParams;
    const source = new URL(query.get("source") ?? "/Mixture_audio_1.wav", location.href);
    const manifest = new URL(query.get("manifestUrl") ?? "/model/manifest.json", location.href).href;
    const [audioResponse, referenceResponse, stageResponse] = await Promise.all([
      fetch(source, { cache: "no-store" }),
      fetch("/test/fixtures/deterministic-waveform-reference.json", { cache: "no-store" }),
      fetch("/test/fixtures/deterministic-stage-reference.json", { cache: "no-store" }),
    ]);
    if (!audioResponse.ok) throw new Error(`Could not fetch quality fixture: HTTP ${audioResponse.status}`);
    if (!referenceResponse.ok) throw new Error(`Could not fetch waveform oracle: HTTP ${referenceResponse.status}`);
    if (!stageResponse.ok) throw new Error(`Could not fetch stage oracle: HTTP ${stageResponse.status}`);
    const reference = await referenceResponse.json() as WaveformReference;
    const stageReference = await stageResponse.json() as StageReference;
    setStatus("Decoding shared fixture…");
    // This oracle deliberately freezes the old input tensor so it isolates
    // model arithmetic. Production decoding has a separate torchaudio-sinc
    // gate and uses that path by default.
    const pcm = await decodeAudioBlob(await audioResponse.blob(), { resampler: "linear" });

    setStatus("Running deterministic WebGPU graph…");
    separator = await DiCoSeSeparator.create({ manifestUrl: manifest, attentionKernel: "flash" });
    const separation = await separator.separatePcm(pcm, {
      outputMode: "deterministic",
      traceSamples: stageReference.samplesPerStage,
    });
    const stems = {} as Record<DiCoSeStemName, ReferenceMetrics>;
    let withinEnvelope = true;
    for (const name of DICOSE_STEM_NAMES) {
      const metrics = compareStem(reference, separation.stems[name], name);
      stems[name] = metrics;
      withinEnvelope &&= passesWaveformEnvelope(metrics, name);
    }
    if (separation.trace === undefined) throw new Error("Deterministic stage trace was not returned");
    const stages: Record<string, ReferenceMetrics> = {};
    for (const [name, expected] of Object.entries(stageReference.stages)) {
      const actual = separation.trace[name];
      if (actual === undefined) throw new Error(`WebGPU trace omitted ${name}`);
      if (actual.elements !== expected.elements) throw new Error(`${name} tensor geometry differs from upstream`);
      stages[name] = compareF16Words(decodeUint16(expected.valuesF16LeBase64), actual.values);
      withinEnvelope &&= passesStageEnvelope(stages[name]!);
    }
    publish({ ok: withinEnvelope, withinEnvelope, timing: separation.timing, stems, stages });
  } catch (error) {
    publish({ ok: false, error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) });
  } finally {
    await separator?.dispose().catch(() => {});
  }
}

function compareF16Words(expected: Uint16Array, actual: Uint16Array): ReferenceMetrics {
  if (expected.length !== actual.length) throw new Error("Sampled tensor lengths differ");
  let errorEnergy = 0;
  let referenceEnergy = 0;
  let candidateEnergy = 0;
  let dot = 0;
  let maxAbs = 0;
  let bitExact = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const expectedWord = expected[index]!;
    const actualWord = actual[index]!;
    const target = float16BitsToFloat32(expectedWord);
    const value = float16BitsToFloat32(actualWord);
    if (!Number.isFinite(target) || !Number.isFinite(value)) {
      throw new Error("Sampled tensor contains a non-finite value");
    }
    const difference = value - target;
    errorEnergy += difference * difference;
    referenceEnergy += target * target;
    candidateEnergy += value * value;
    dot += value * target;
    maxAbs = Math.max(maxAbs, Math.abs(difference));
    if (expectedWord === actualWord) bitExact += 1;
  }
  return {
    samples: expected.length,
    nrmse: Math.sqrt(errorEnergy / Math.max(referenceEnergy, Number.MIN_VALUE)),
    snrDb: 10 * Math.log10(referenceEnergy / Math.max(errorEnergy, Number.MIN_VALUE)),
    cosine: dot / Math.sqrt(Math.max(referenceEnergy * candidateEnergy, Number.MIN_VALUE)),
    maxAbs,
    referenceRms: Math.sqrt(referenceEnergy / expected.length),
    candidateRms: Math.sqrt(candidateEnergy / expected.length),
    bitExactFraction: bitExact / expected.length,
  };
}

function compareStem(
  reference: WaveformReference,
  candidate: { readonly sampleRate: number; readonly length: number; readonly left: Float32Array; readonly right: Float32Array },
  name: DiCoSeStemName,
): ReferenceMetrics {
  if (candidate.sampleRate !== reference.fixture.sampleRate || candidate.length !== reference.fixture.samples) {
    throw new Error(`${name} waveform geometry differs from the upstream reference`);
  }
  const expectedStem = reference.stems[name];
  const expectedChannels = [
    decodeFloat32(expectedStem.leftF32LeBase64),
    decodeFloat32(expectedStem.rightF32LeBase64),
  ] as const;
  const candidateChannels = [candidate.left, candidate.right] as const;
  let errorEnergy = 0;
  let referenceEnergy = 0;
  let candidateEnergy = 0;
  let dot = 0;
  let maxAbs = 0;
  for (let channel = 0; channel < 2; channel += 1) {
    const expected = expectedChannels[channel]!;
    const actual = candidateChannels[channel]!;
    if (expected.length !== reference.sampling.countPerChannel) {
      throw new Error(`${name} reference sample count is malformed`);
    }
    for (let sample = 0; sample < expected.length; sample += 1) {
      const index = reference.sampling.offset + sample * reference.sampling.stride;
      const target = expected[sample]!;
      const value = actual[index]!;
      if (!Number.isFinite(target) || !Number.isFinite(value)) {
        throw new Error(`${name} contains a non-finite sampled waveform value`);
      }
      const difference = value - target;
      errorEnergy += difference * difference;
      referenceEnergy += target * target;
      candidateEnergy += value * value;
      dot += value * target;
      maxAbs = Math.max(maxAbs, Math.abs(difference));
    }
  }
  const samples = reference.sampling.countPerChannel * 2;
  return {
    samples,
    nrmse: Math.sqrt(errorEnergy / Math.max(referenceEnergy, Number.MIN_VALUE)),
    snrDb: 10 * Math.log10(referenceEnergy / Math.max(errorEnergy, Number.MIN_VALUE)),
    cosine: dot / Math.sqrt(Math.max(referenceEnergy * candidateEnergy, Number.MIN_VALUE)),
    maxAbs,
    referenceRms: Math.sqrt(referenceEnergy / samples),
    candidateRms: Math.sqrt(candidateEnergy / samples),
  };
}

function decodeFloat32(base64: string): Float32Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new Float32Array(bytes.buffer);
}

function decodeUint16(base64: string): Uint16Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new Uint16Array(bytes.buffer);
}

function passesWaveformEnvelope(metrics: ReferenceMetrics, name: DiCoSeStemName): boolean {
  if (name === "vocals") return metrics.nrmse <= 0.03 && metrics.maxAbs <= 0.00001;
  return metrics.nrmse <= 0.01 && metrics.cosine >= 0.9999;
}

function passesStageEnvelope(metrics: ReferenceMetrics): boolean {
  return metrics.nrmse <= 0.05 && metrics.cosine >= 0.998;
}

function setStatus(value: string): void {
  if (statusNode !== null) statusNode.textContent = value;
}

function publish(report: ReferenceQualityReport): void {
  harness.report = report;
  setStatus(report.ok ? "Upstream waveform parity measured." : "Upstream waveform parity failed.");
  if (resultNode !== null) resultNode.textContent = JSON.stringify(report, null, 2);
}
