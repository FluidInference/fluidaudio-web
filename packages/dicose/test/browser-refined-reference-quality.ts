import { DiCoSeSeparator, type SeparatorTiming } from "../src/runtime/separator.js";
import { float16BitsToFloat32, SeededGaussian, type StereoPcm } from "../src/runtime/audio.js";
import { DICOSE_STEM_NAMES, type DiCoSeStemName } from "../src/worker-protocol.js";

interface SampledWaveform {
  readonly leftF32LeBase64: string;
  readonly rightF32LeBase64: string;
}

interface RefinedReference {
  readonly schema: "dicose-refined-upstream-reference-v1";
  readonly input: {
    readonly sampleRate: number;
    readonly samples: number;
    readonly sha256PlanarF32Le: string;
    readonly leftF32LeBase64: string;
    readonly rightF32LeBase64: string;
  };
  readonly sampler: {
    readonly seed: number;
    readonly sigmaMax: number;
    readonly cOut: number;
    readonly cSkip: number;
  };
  readonly waveformSampling: {
    readonly formula: "floor(i * samples / count)";
    readonly countPerChannel: number;
  };
  readonly stageSampling: {
    readonly maxSamplesPerCall: number;
  };
  readonly stages: Readonly<Record<string, {
    readonly calls: number;
    readonly elementsPerCall: readonly number[];
    readonly samplesPerCall: readonly number[];
    readonly valuesF16LeBase64: string;
  }>>;
  readonly waveforms: {
    readonly deterministic: Readonly<Record<DiCoSeStemName, SampledWaveform>>;
    readonly cdModelOutput: Readonly<Record<DiCoSeStemName, SampledWaveform>>;
    readonly refined: Readonly<Record<DiCoSeStemName, SampledWaveform>>;
  };
}

interface WaveformMetrics {
  readonly samples: number;
  readonly nrmse: number;
  readonly snrDb: number;
  readonly cosine: number;
  readonly maxAbs: number;
  readonly rmse: number;
  readonly referenceRms: number;
  readonly candidateRms: number;
  readonly bitExactFraction?: number;
}

interface MetricEnvelope {
  readonly maxNrmse: number;
  readonly minCosine: number;
}

const STAGE_ENVELOPES: Readonly<Record<string, MetricEnvelope>> = Object.freeze({
  // These are deliberately stage-specific: f16/WebGPU error grows through the
  // eight-block CD trunk, while the time/mapping seams should remain much
  // tighter. Each limit leaves at least ~3x headroom over the frozen upstream
  // run, yet still fails near the first semantically incorrect operator.
  "cd.timeEmbedding": { maxNrmse: 0.001, minCosine: 0.99999 },
  "cd.mappingInput": { maxNrmse: 0.001, minCosine: 0.99999 },
  "cd.mappingOutput": { maxNrmse: 0.003, minCosine: 0.99998 },
  "cd.stftAdapter": { maxNrmse: 0.001, minCosine: 0.99999 },
  "cd.stftCombined": { maxNrmse: 0.01, minCosine: 0.9999 },
  "cd.bandRaw": { maxNrmse: 0.03, minCosine: 0.999 },
  "cd.bandConditionInput": { maxNrmse: 0.005, minCosine: 0.9999 },
  "cd.bandConditionLinear": { maxNrmse: 0.003, minCosine: 0.99998 },
  "cd.bandConditionGelu": { maxNrmse: 0.003, minCosine: 0.99998 },
  "cd.bandCondition": { maxNrmse: 0.003, minCosine: 0.99998 },
  "cd.bandConditioned": { maxNrmse: 0.03, minCosine: 0.999 },
  "cd.film.layer0.time.attention": { maxNrmse: 0.003, minCosine: 0.99998 },
  "cd.layer0.time": { maxNrmse: 0.025, minCosine: 0.999 },
  "cd.layer0.frequency": { maxNrmse: 0.02, minCosine: 0.999 },
  "cd.layer7.frequency": { maxNrmse: 0.02, minCosine: 0.999 },
  "cd.finalNorm": { maxNrmse: 0.15, minCosine: 0.99 },
  "cd.mask": { maxNrmse: 0.05, minCosine: 0.995 },
});

interface RefinedReferenceReport {
  readonly ok: boolean;
  readonly withinEnvelope?: boolean;
  readonly inputSha256?: string;
  readonly timing?: SeparatorTiming;
  readonly waveforms?: Readonly<Record<"deterministic" | "cdModelOutput" | "refined", Readonly<Record<DiCoSeStemName, WaveformMetrics>>>>;
  readonly stages?: Readonly<Record<string, WaveformMetrics>>;
  readonly error?: string;
}

const statusNode = document.querySelector<HTMLElement>("#status");
const resultNode = document.querySelector<HTMLElement>("#result");
const harness = {} as { report?: RefinedReferenceReport };
(globalThis as typeof globalThis & { __DICOSE_BROWSER__?: typeof harness }).__DICOSE_BROWSER__ = harness;

void run();

async function run(): Promise<void> {
  let separator: DiCoSeSeparator | undefined;
  try {
    const query = new URL(location.href).searchParams;
    const manifest = new URL(query.get("manifestUrl") ?? "/model/manifest.json", location.href).href;
    const response = await fetch("/test/fixtures/refined-upstream-reference.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Could not fetch refined waveform oracle: HTTP ${response.status}`);
    const reference = await response.json() as RefinedReference;
    const pcm = decodeInput(reference);
    const inputSha256 = await planarSha256(pcm);
    if (inputSha256 !== reference.input.sha256PlanarF32Le) {
      throw new Error(`Refined oracle input hash mismatch: ${inputSha256}`);
    }

    setStatus("Running released Full/refined WebGPU graph…");
    separator = await DiCoSeSeparator.create({
      manifestUrl: manifest,
      attentionKernel: "flash",
    });
    const deterministic = await separator.separatePcm(pcm, {
      outputMode: "deterministic",
    });
    const refined = await separator.separatePcm(pcm, {
      outputMode: "refined",
      seed: reference.sampler.seed,
      traceSamples: reference.stageSampling.maxSamplesPerCall,
    });
    const cdModelOutput = deriveCdModelOutput(
      deterministic.stems,
      refined.stems,
      reference.sampler,
    );

    const candidates = {
      deterministic: deterministic.stems,
      cdModelOutput,
      refined: refined.stems,
    } as const;
    const waveforms = {} as Record<keyof typeof candidates, Record<DiCoSeStemName, WaveformMetrics>>;
    let withinEnvelope = true;
    const violations: string[] = [];
    for (const kind of ["deterministic", "cdModelOutput", "refined"] as const) {
      const stems = {} as Record<DiCoSeStemName, WaveformMetrics>;
      for (const name of DICOSE_STEM_NAMES) {
        const metrics = compareStem(reference, reference.waveforms[kind][name], candidates[kind][name], name);
        stems[name] = metrics;
        if (!passesWaveformEnvelope(metrics, kind)) {
          withinEnvelope = false;
          violations.push(`${kind}.${name}`);
        }
      }
      waveforms[kind] = stems;
    }
    if (refined.cdTrace === undefined) throw new Error("Full/refined CD stage trace was not returned");
    const stages: Record<string, WaveformMetrics> = {};
    for (const [name, expected] of Object.entries(reference.stages)) {
      const actual = refined.cdTrace[name];
      if (actual === undefined) throw new Error(`WebGPU trace omitted ${name}`);
      if (
        expected.calls !== expected.elementsPerCall.length ||
        expected.calls !== expected.samplesPerCall.length ||
        actual.elementsPerCall.length !== expected.calls ||
        actual.elementsPerCall.some((elements, index) => elements !== expected.elementsPerCall[index])
      ) {
        throw new Error(`${name} trace call geometry differs from upstream`);
      }
      const metrics = compareF16Words(
        decodeUint16Le(expected.valuesF16LeBase64),
        actual.values,
        expected.samplesPerCall.reduce((total, samples) => total + samples, 0),
        name,
      );
      stages[name] = metrics;
      const expectedWords = decodeUint16Le(expected.valuesF16LeBase64);
      const calls: WaveformMetrics[] = [];
      let offset = 0;
      for (let call = 0; call < expected.calls; call += 1) {
        const samples = expected.samplesPerCall[call]!;
        calls.push(compareF16Words(
          expectedWords.subarray(offset, offset + samples),
          actual.values.subarray(offset, offset + samples),
          samples,
          `${name} call ${call}`,
        ));
        offset += samples;
      }
      const envelope = STAGE_ENVELOPES[name];
      if (envelope === undefined) throw new Error(`No correctness envelope is defined for ${name}`);
      if (!passesMetricEnvelope(metrics, envelope)) {
        withinEnvelope = false;
        violations.push(name);
      }
      for (let call = 0; call < calls.length; call += 1) {
        if (!passesMetricEnvelope(calls[call]!, envelope)) {
          withinEnvelope = false;
          violations.push(`${name}[${call}]`);
        }
      }
    }
    publish({
      ok: withinEnvelope,
      withinEnvelope,
      inputSha256,
      timing: refined.timing,
      waveforms,
      stages,
      ...(violations.length === 0 ? {} : { error: `Envelope violations: ${violations.join(", ")}` }),
    });
  } catch (error) {
    publish({ ok: false, error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) });
  } finally {
    await separator?.dispose().catch(() => {});
  }
}

function passesWaveformEnvelope(
  metrics: WaveformMetrics,
  kind: "deterministic" | "cdModelOutput" | "refined",
): boolean {
  // The bundled excerpt contains a nearly silent vocal stem. Relative error
  // is ill-conditioned there, so require a small absolute error as well as
  // preserving direction. This still rejects a noise-only or zeroed result by
  // orders of magnitude.
  if (metrics.referenceRms < 0.001) {
    return metrics.rmse <= 0.00001 && metrics.maxAbs <= 0.00005 && metrics.cosine >= 0.99;
  }
  if (kind === "deterministic") {
    return metrics.nrmse <= 0.01 && metrics.cosine >= 0.9999 && metrics.maxAbs <= 0.001;
  }
  if (kind === "cdModelOutput") {
    return metrics.nrmse <= 0.03 && metrics.cosine >= 0.999 && metrics.maxAbs <= 0.3;
  }
  return metrics.nrmse <= 0.005 && metrics.cosine >= 0.9999 && metrics.maxAbs <= 0.003;
}

function passesMetricEnvelope(metrics: WaveformMetrics, envelope: MetricEnvelope): boolean {
  return metrics.nrmse <= envelope.maxNrmse && metrics.cosine >= envelope.minCosine;
}

function compareF16Words(
  expected: Uint16Array,
  actual: Uint16Array,
  expectedSamples: number,
  name: string,
): WaveformMetrics {
  if (expected.length !== expectedSamples || actual.length !== expectedSamples) {
    throw new Error(`${name} trace sample count differs from upstream`);
  }
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
      throw new Error(`${name} contains a non-finite stage value`);
    }
    const difference = value - target;
    errorEnergy += difference * difference;
    referenceEnergy += target * target;
    candidateEnergy += value * value;
    dot += target * value;
    maxAbs = Math.max(maxAbs, Math.abs(difference));
    if (expectedWord === actualWord) bitExact += 1;
  }
  return {
    samples: expected.length,
    nrmse: Math.sqrt(errorEnergy / Math.max(referenceEnergy, Number.MIN_VALUE)),
    snrDb: 10 * Math.log10(referenceEnergy / Math.max(errorEnergy, Number.MIN_VALUE)),
    cosine: dot / Math.sqrt(Math.max(referenceEnergy * candidateEnergy, Number.MIN_VALUE)),
    maxAbs,
    rmse: Math.sqrt(errorEnergy / expected.length),
    referenceRms: Math.sqrt(referenceEnergy / expected.length),
    candidateRms: Math.sqrt(candidateEnergy / expected.length),
    bitExactFraction: bitExact / expected.length,
  };
}

function decodeInput(reference: RefinedReference): StereoPcm {
  if (reference.schema !== "dicose-refined-upstream-reference-v1") {
    throw new Error(`Unsupported refined oracle schema: ${String(reference.schema)}`);
  }
  if (reference.waveformSampling.formula !== "floor(i * samples / count)") {
    throw new Error("Unsupported refined waveform sampling formula");
  }
  const left = decodeFloat32Le(reference.input.leftF32LeBase64);
  const right = decodeFloat32Le(reference.input.rightF32LeBase64);
  if (left.length !== reference.input.samples || right.length !== reference.input.samples) {
    throw new Error("Refined oracle input payload has the wrong length");
  }
  return {
    sampleRate: reference.input.sampleRate,
    length: reference.input.samples,
    left,
    right,
    channels: [left, right],
  };
}

function compareStem(
  reference: RefinedReference,
  expected: SampledWaveform,
  candidate: StereoPcm,
  name: DiCoSeStemName,
): WaveformMetrics {
  if (candidate.sampleRate !== reference.input.sampleRate || candidate.length !== reference.input.samples) {
    throw new Error(`${name} refined waveform geometry differs from upstream`);
  }
  const expectedChannels = [
    decodeFloat32Le(expected.leftF32LeBase64),
    decodeFloat32Le(expected.rightF32LeBase64),
  ] as const;
  const candidateChannels = candidate.channels;
  const count = reference.waveformSampling.countPerChannel;
  let errorEnergy = 0;
  let referenceEnergy = 0;
  let candidateEnergy = 0;
  let dot = 0;
  let maxAbs = 0;
  for (let channel = 0; channel < 2; channel += 1) {
    const targets = expectedChannels[channel]!;
    const values = candidateChannels[channel]!;
    if (targets.length !== count) throw new Error(`${name} refined oracle sample count is malformed`);
    for (let sample = 0; sample < count; sample += 1) {
      const index = Math.floor(sample * reference.input.samples / count);
      const target = targets[sample]!;
      const value = values[index]!;
      if (!Number.isFinite(target) || !Number.isFinite(value)) {
        throw new Error(`${name} contains a non-finite refined waveform value`);
      }
      const difference = value - target;
      errorEnergy += difference * difference;
      referenceEnergy += target * target;
      candidateEnergy += value * value;
      dot += target * value;
      maxAbs = Math.max(maxAbs, Math.abs(difference));
    }
  }
  const samples = count * 2;
  return {
    samples,
    nrmse: Math.sqrt(errorEnergy / Math.max(referenceEnergy, Number.MIN_VALUE)),
    snrDb: 10 * Math.log10(referenceEnergy / Math.max(errorEnergy, Number.MIN_VALUE)),
    cosine: dot / Math.sqrt(Math.max(referenceEnergy * candidateEnergy, Number.MIN_VALUE)),
    maxAbs,
    rmse: Math.sqrt(errorEnergy / samples),
    referenceRms: Math.sqrt(referenceEnergy / samples),
    candidateRms: Math.sqrt(candidateEnergy / samples),
  };
}

function deriveCdModelOutput(
  deterministic: Readonly<Record<DiCoSeStemName, StereoPcm>>,
  refined: Readonly<Record<DiCoSeStemName, StereoPcm>>,
  sampler: RefinedReference["sampler"],
): Readonly<Record<DiCoSeStemName, StereoPcm>> {
  const random = new SeededGaussian(sampler.seed);
  const output = {} as Record<DiCoSeStemName, StereoPcm>;
  for (const name of DICOSE_STEM_NAMES) {
    const source = deterministic[name];
    const final = refined[name];
    if (source.length !== final.length) throw new Error(`${name} output lengths differ`);
    const channels = [new Float32Array(source.length), new Float32Array(source.length)] as const;
    for (let channel = 0; channel < 2; channel += 1) {
      const sourceValues = source.channels[channel]!;
      const finalValues = final.channels[channel]!;
      const modelValues = channels[channel]!;
      for (let index = 0; index < source.length; index += 1) {
        const noisy = Math.fround(sourceValues[index]! + sampler.sigmaMax * random.next());
        const finalValue = finalValues[index]!;
        if (Math.abs(finalValue) >= 1) {
          throw new Error(`${name} refined output was clamped; raw CD inversion is ambiguous`);
        }
        modelValues[index] = (finalValue - sampler.cSkip * noisy) / sampler.cOut;
      }
    }
    output[name] = {
      sampleRate: source.sampleRate,
      length: source.length,
      left: channels[0],
      right: channels[1],
      channels,
    };
  }
  return output;
}

function decodeFloat32Le(base64: string): Float32Array {
  const binary = atob(base64);
  if (binary.length % 4 !== 0) throw new Error("Malformed f32 oracle payload");
  const view = new DataView(new ArrayBuffer(binary.length));
  for (let index = 0; index < binary.length; index += 1) {
    view.setUint8(index, binary.charCodeAt(index));
  }
  const output = new Float32Array(binary.length / 4);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = view.getFloat32(index * 4, true);
  }
  return output;
}

function decodeUint16Le(base64: string): Uint16Array {
  const binary = atob(base64);
  if (binary.length % 2 !== 0) throw new Error("Malformed f16 oracle payload");
  const view = new DataView(new ArrayBuffer(binary.length));
  for (let index = 0; index < binary.length; index += 1) {
    view.setUint8(index, binary.charCodeAt(index));
  }
  const output = new Uint16Array(binary.length / 2);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = view.getUint16(index * 2, true);
  }
  return output;
}

async function planarSha256(source: StereoPcm): Promise<string> {
  const bytes = new Uint8Array(source.length * 2 * 4);
  const view = new DataView(bytes.buffer);
  let offset = 0;
  for (const channel of source.channels) {
    for (let index = 0; index < channel.length; index += 1) {
      view.setFloat32(offset, channel[index]!, true);
      offset += 4;
    }
  }
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (value) => value.toString(16).padStart(2, "0")).join("");
}

function setStatus(value: string): void {
  if (statusNode !== null) statusNode.textContent = value;
}

function publish(report: RefinedReferenceReport): void {
  harness.report = report;
  setStatus(report.ok ? "Full/refined upstream waveform parity measured." : "Full/refined upstream waveform parity failed.");
  if (resultNode !== null) resultNode.textContent = JSON.stringify(report, null, 2);
}
