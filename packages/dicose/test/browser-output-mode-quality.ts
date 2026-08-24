import { DiCoSeWorkerClient, type DiCoSeSeparation } from "../src/api.js";
import { decodeAudioBlob, type StereoPcm } from "../src/runtime/audio.js";
import { DICOSE_STEM_NAMES, type DiCoSeStemName } from "../src/worker-protocol.js";

interface WaveformMetrics {
  readonly samples: number;
  readonly nrmse: number;
  readonly snrDb: number;
  readonly cosine: number;
  readonly maxAbs: number;
  readonly maxAbsOverReferencePeak: number;
  readonly rmsRelativeDrift: number;
  readonly peakRelativeDrift: number;
  readonly worstWindowNrmse: number;
  readonly worstWindowCosine: number;
}

interface OutputModeQualityReport {
  readonly ok: boolean;
  readonly deterministicTiming?: Readonly<Record<string, number>>;
  readonly refinedTiming?: Readonly<Record<string, number>>;
  readonly deterministicDiagnosticsExact?: boolean;
  readonly speedup?: number;
  readonly savedMs?: number;
  readonly stems?: Readonly<Record<DiCoSeStemName, WaveformMetrics>>;
  readonly error?: string;
}

const statusNode = document.querySelector<HTMLElement>("#status");
const resultNode = document.querySelector<HTMLElement>("#result");
const harness = {} as { report?: OutputModeQualityReport };
(globalThis as typeof globalThis & { __DICOSE_BROWSER__?: typeof harness }).__DICOSE_BROWSER__ = harness;

void run();

async function run(): Promise<void> {
  let client: DiCoSeWorkerClient | undefined;
  try {
    const query = new URL(location.href).searchParams;
    const source = new URL(query.get("source") ?? "/Mixture_audio_1.wav", location.href);
    const manifest = new URL(query.get("manifestUrl") ?? "/model/manifest.json", location.href).href;
    const response = await fetch(source, { cache: "no-store" });
    if (!response.ok) throw new Error(`Could not fetch quality fixture: HTTP ${response.status}`);
    setStatus("Decoding shared fixture…");
    const pcm = await decodeAudioBlob(await response.blob(), { targetSampleRate: "source" });

    client = new DiCoSeWorkerClient({
      manifestUrl: manifest,
      attentionKernel: "flash",
    });
    setStatus("Initializing shared full-resolution/Flash runtime…");
    await client.initialize();

    setStatus("Running cold deterministic-only output…");
    const deterministic = await client.separatePcm(pcm, {
      outputMode: "deterministic",
      seed: 0xd1c05e,
    });
    setStatus("Running refined output on the same runtime…");
    const refined = await client.separatePcm(pcm, {
      outputMode: "refined",
      seed: 0xd1c05e,
    });

    assertRuntimeContract(pcm, deterministic, refined);
    const stems = {} as Record<DiCoSeStemName, WaveformMetrics>;
    for (const name of DICOSE_STEM_NAMES) {
      stems[name] = compareStem(deterministic, refined, name);
    }
    const deterministicTotal = requireTiming(deterministic, "totalMs");
    const refinedTotal = requireTiming(refined, "totalMs");
    publish({
      ok: true,
      deterministicTiming: deterministic.timing,
      refinedTiming: refined.timing,
      deterministicDiagnosticsExact: true,
      speedup: refinedTotal / deterministicTotal,
      savedMs: refinedTotal - deterministicTotal,
      stems,
    });
  } catch (error) {
    publish({ ok: false, error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) });
  } finally {
    await client?.dispose().catch(() => {});
  }
}

function assertRuntimeContract(
  input: StereoPcm,
  deterministic: DiCoSeSeparation,
  refined: DiCoSeSeparation,
): void {
  if (deterministic.outputMode !== "deterministic") {
    throw new Error(`Deterministic request returned ${deterministic.outputMode}`);
  }
  if (refined.outputMode !== "refined") {
    throw new Error(`Refined request returned ${refined.outputMode}`);
  }
  for (const [label, separation] of [
    ["deterministic", deterministic],
    ["refined", refined],
  ] as const) {
    for (const name of ["prepareMs", "deterministicMs", "mappingMs", "refinementMs", "istftMs", "totalMs"]) {
      const elapsed = requireTiming(separation, name);
      if (elapsed < 0) throw new Error(`${label} timing ${name} is negative`);
    }
    if (requireTiming(separation, "totalMs") <= 0 || requireTiming(separation, "deterministicMs") <= 0) {
      throw new Error(`${label} runtime did not record model execution`);
    }
    assertInstrumentalComplement(input, separation, label);
  }
  if (
    requireTiming(deterministic, "mappingMs") !== 0 ||
    requireTiming(deterministic, "refinementMs") !== 0
  ) {
    throw new Error("Deterministic-only output executed CD mapping or refinement");
  }
  if (deterministic.diagnostics.cdModelOutput !== undefined) {
    throw new Error("Deterministic-only output fabricated CD diagnostics");
  }
  if (refined.diagnostics.cdModelOutput === undefined || requireTiming(refined, "refinementMs") <= 0) {
    throw new Error("Refined output omitted its CD execution evidence");
  }
  const standaloneDiagnostics = deterministic.diagnostics.deterministic;
  const refinedDiagnostics = refined.diagnostics.deterministic;
  if (standaloneDiagnostics === undefined || refinedDiagnostics === undefined) {
    throw new Error("Output-mode panel is missing deterministic diagnostics");
  }
  for (const name of DICOSE_STEM_NAMES) {
    const standalone = standaloneDiagnostics[name];
    const captured = refinedDiagnostics[name];
    if (
      standalone === undefined || captured === undefined ||
      standalone.peak !== captured.peak || standalone.rms !== captured.rms
    ) {
      throw new Error(`${name} deterministic diagnostics changed when CD capture was enabled`);
    }
  }
}

function assertInstrumentalComplement(
  input: StereoPcm,
  separation: DiCoSeSeparation,
  label: string,
): void {
  const { vocals } = separation.stems;
  const { instrumental } = separation;
  if (
    vocals.sampleRate !== input.sampleRate ||
    instrumental.sampleRate !== input.sampleRate ||
    vocals.length !== input.length ||
    instrumental.length !== input.length
  ) {
    throw new Error(`${label} outputs were not restored to the input timeline`);
  }
  for (const [mixtureChannel, vocalsChannel, instrumentalChannel] of [
    [input.left, vocals.left, instrumental.left],
    [input.right, vocals.right, instrumental.right],
  ] as const) {
    for (let index = 0; index < input.length; index += 1) {
      const expected = Math.fround(mixtureChannel[index]! - vocalsChannel[index]!);
      if (instrumentalChannel[index] !== expected) {
        throw new Error(`${label} instrumental is not mixture minus vocals at sample ${index}`);
      }
    }
  }
}

function requireTiming(separation: DiCoSeSeparation, name: string): number {
  const elapsed = separation.timing[name];
  if (elapsed === undefined || !Number.isFinite(elapsed)) {
    throw new Error(`${separation.outputMode} timing ${name} is missing or non-finite`);
  }
  return elapsed;
}

function compareStem(
  deterministic: DiCoSeSeparation,
  refined: DiCoSeSeparation,
  name: DiCoSeStemName,
): WaveformMetrics {
  const reference = deterministic.stems[name];
  const candidate = refined.stems[name];
  if (
    reference.sampleRate !== candidate.sampleRate || reference.length !== candidate.length ||
    reference.left.length !== candidate.left.length || reference.right.length !== candidate.right.length
  ) {
    throw new Error(`${name} waveform geometry changed`);
  }
  let errorEnergy = 0;
  let referenceEnergy = 0;
  let candidateEnergy = 0;
  let dot = 0;
  let referencePeak = 0;
  let candidatePeak = 0;
  let maxAbs = 0;
  let worstWindowNrmse = 0;
  let worstWindowCosine = 1;
  const window = 4_096;
  for (const [referenceChannel, candidateChannel] of [
    [reference.left, candidate.left],
    [reference.right, candidate.right],
  ] as const) {
    for (let start = 0; start < referenceChannel.length; start += window) {
      let windowError = 0;
      let windowReference = 0;
      let windowCandidate = 0;
      let windowDot = 0;
      const end = Math.min(start + window, referenceChannel.length);
      for (let index = start; index < end; index += 1) {
        const expected = referenceChannel[index]!;
        const actual = candidateChannel[index]!;
        if (!Number.isFinite(expected) || !Number.isFinite(actual)) {
          throw new Error(`${name} contains a non-finite sample`);
        }
        const difference = actual - expected;
        const differenceSquared = difference * difference;
        const expectedSquared = expected * expected;
        const actualSquared = actual * actual;
        errorEnergy += differenceSquared;
        referenceEnergy += expectedSquared;
        candidateEnergy += actualSquared;
        dot += expected * actual;
        referencePeak = Math.max(referencePeak, Math.abs(expected));
        candidatePeak = Math.max(candidatePeak, Math.abs(actual));
        maxAbs = Math.max(maxAbs, Math.abs(difference));
        windowError += differenceSquared;
        windowReference += expectedSquared;
        windowCandidate += actualSquared;
        windowDot += expected * actual;
      }
      if (windowReference > 1e-12) {
        worstWindowNrmse = Math.max(worstWindowNrmse, Math.sqrt(windowError / windowReference));
        worstWindowCosine = Math.min(
          worstWindowCosine,
          windowDot / Math.sqrt(Math.max(windowReference * windowCandidate, Number.MIN_VALUE)),
        );
      }
    }
  }
  const samples = reference.length * 2;
  const referenceRms = Math.sqrt(referenceEnergy / samples);
  const candidateRms = Math.sqrt(candidateEnergy / samples);
  const metrics = {
    samples,
    nrmse: Math.sqrt(errorEnergy / Math.max(referenceEnergy, Number.MIN_VALUE)),
    snrDb: 10 * Math.log10(Math.max(referenceEnergy, Number.MIN_VALUE) / Math.max(errorEnergy, Number.MIN_VALUE)),
    cosine: dot / Math.sqrt(Math.max(referenceEnergy * candidateEnergy, Number.MIN_VALUE)),
    maxAbs,
    maxAbsOverReferencePeak: maxAbs / Math.max(referencePeak, Number.MIN_VALUE),
    rmsRelativeDrift: Math.abs(candidateRms - referenceRms) / Math.max(referenceRms, Number.MIN_VALUE),
    peakRelativeDrift: Math.abs(candidatePeak - referencePeak) / Math.max(referencePeak, Number.MIN_VALUE),
    worstWindowNrmse,
    worstWindowCosine,
  };
  if (Object.values(metrics).some((value) => !Number.isFinite(value))) {
    throw new Error(`${name} waveform metrics are non-finite`);
  }
  return metrics;
}

function setStatus(value: string): void {
  if (statusNode !== null) statusNode.textContent = value;
}

function publish(report: OutputModeQualityReport): void {
  harness.report = report;
  setStatus(report.ok ? "Output-mode waveform panel completed." : "Output-mode waveform panel failed.");
  if (resultNode !== null) resultNode.textContent = JSON.stringify(report, null, 2);
}
