import { DiCoSeWorkerClient, type DiCoSeSeparation } from "../src/api.js";
import { decodeAudioBlob } from "../src/runtime/audio.js";
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

interface AttentionQualityReport {
  readonly ok: boolean;
  readonly exactTiming?: Readonly<Record<string, number>>;
  readonly flashTiming?: Readonly<Record<string, number>>;
  readonly stems?: Readonly<Record<DiCoSeStemName, WaveformMetrics>>;
  readonly error?: string;
}

const statusNode = document.querySelector<HTMLElement>("#status");
const resultNode = document.querySelector<HTMLElement>("#result");
const harness = {} as { report?: AttentionQualityReport };
(globalThis as typeof globalThis & { __DICOSE_BROWSER__?: typeof harness }).__DICOSE_BROWSER__ = harness;

void run();

async function run(): Promise<void> {
  let exactClient: DiCoSeWorkerClient | undefined;
  let flashClient: DiCoSeWorkerClient | undefined;
  try {
    const query = new URL(location.href).searchParams;
    const source = new URL(query.get("source") ?? "/Mixture_audio_1.wav", location.href);
    const manifest = new URL(query.get("manifestUrl") ?? "/model/manifest.json", location.href).href;
    const response = await fetch(source, { cache: "no-store" });
    if (!response.ok) throw new Error(`Could not fetch quality fixture: HTTP ${response.status}`);
    setStatus("Decoding shared fixture…");
    const pcm = await decodeAudioBlob(await response.blob());

    setStatus("Running exact Q64 control…");
    exactClient = new DiCoSeWorkerClient({ manifestUrl: manifest, attentionKernel: "q64" });
    const exact = await exactClient.separatePcm(pcm, { seed: 0xd1c05e });
    await exactClient.dispose();
    exactClient = undefined;

    setStatus("Running blockwise Flash candidate…");
    flashClient = new DiCoSeWorkerClient({ manifestUrl: manifest, attentionKernel: "flash" });
    const flash = await flashClient.separatePcm(pcm, { seed: 0xd1c05e });

    const stems = {} as Record<DiCoSeStemName, WaveformMetrics>;
    const failures: string[] = [];
    for (const name of DICOSE_STEM_NAMES) {
      const metrics = compareStem(exact, flash, name);
      stems[name] = metrics;
      if (!passes(metrics)) failures.push(`${name} waveform drift exceeded the Flash quality envelope`);
    }
    const report: AttentionQualityReport = {
      ok: failures.length === 0,
      exactTiming: exact.timing,
      flashTiming: flash.timing,
      stems,
      ...(failures.length === 0 ? {} : { error: failures.join("; ") }),
    };
    publish(report);
  } catch (error) {
    publish({ ok: false, error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) });
  } finally {
    await exactClient?.dispose().catch(() => {});
    await flashClient?.dispose().catch(() => {});
  }
}

function compareStem(
  exact: DiCoSeSeparation,
  flash: DiCoSeSeparation,
  name: DiCoSeStemName,
): WaveformMetrics {
  const reference = exact.stems[name];
  const candidate = flash.stems[name];
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
  return {
    samples,
    nrmse: Math.sqrt(errorEnergy / Math.max(referenceEnergy, Number.MIN_VALUE)),
    snrDb: 10 * Math.log10(referenceEnergy / Math.max(errorEnergy, Number.MIN_VALUE)),
    cosine: dot / Math.sqrt(Math.max(referenceEnergy * candidateEnergy, Number.MIN_VALUE)),
    maxAbs,
    maxAbsOverReferencePeak: maxAbs / Math.max(referencePeak, Number.MIN_VALUE),
    rmsRelativeDrift: Math.abs(candidateRms - referenceRms) / Math.max(referenceRms, Number.MIN_VALUE),
    peakRelativeDrift: Math.abs(candidatePeak - referencePeak) / Math.max(referencePeak, Number.MIN_VALUE),
    worstWindowNrmse,
    worstWindowCosine,
  };
}

function passes(metrics: WaveformMetrics): boolean {
  return metrics.nrmse <= 0.003 && metrics.snrDb >= 50 && metrics.cosine >= 0.9999 &&
    metrics.maxAbsOverReferencePeak <= 0.02 && metrics.rmsRelativeDrift <= 0.005 &&
    metrics.peakRelativeDrift <= 0.01 && metrics.worstWindowNrmse <= 0.01 &&
    metrics.worstWindowCosine >= 0.999;
}

function setStatus(value: string): void {
  if (statusNode !== null) statusNode.textContent = value;
}

function publish(report: AttentionQualityReport): void {
  harness.report = report;
  setStatus(report.ok ? "Attention waveform A/B passed." : "Attention waveform A/B failed.");
  if (resultNode !== null) resultNode.textContent = JSON.stringify(report, null, 2);
}
