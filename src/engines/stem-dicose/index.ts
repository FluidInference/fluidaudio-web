// DiCoSe stem splitter (Hamza Qayyum's DiCoSe.wgsl, vendored at
// packages/dicose): BS-RoFormer + one-step consistency-distilled refinement,
// raw WebGPU WGSL. Splits a full-band mix into drums / bass / other / vocals
// plus a derived instrumental (input − vocals), restored to the input's
// native rate and length.
//
// Default output mode is "deterministic" (fast: skips the CD refiner, ~3× RT
// for a 30 s / 48 kHz clip on Apple Silicon). Pass { outputMode: "refined" }
// for the released full-quality path (~7× slower).

import type { LoadProgress, ProgressCb, SeparateOpts, SeparationEngine, SeparationInput, StemAudio } from "../../core/types.js";
import { checkSupport, decodeAudioBlob, DiCoSeWorkerClient, DICOSE_STEM_NAMES, type DiCoSeOutputMode, type DiCoSeProgress, type StereoPcm } from "dicose-wgsl";
import { DICOSE_DEFAULT_BASE_URL } from "./config.js";

export interface DicoseStemEngineOptions {
  /** Directory serving manifest.json + weights.f16.bin (defaults to the HF mirror). */
  baseUrl?: string;
  /** "deterministic" (fast, default) or "refined" (full CD refinement). */
  outputMode?: DiCoSeOutputMode;
}

export class DicoseStemEngine implements SeparationEngine {
  readonly id = "stem-dicose";
  readonly label = "DiCoSe Stem Splitter";

  private readonly baseUrl: string;
  private readonly outputMode: DiCoSeOutputMode;
  private client: DiCoSeWorkerClient | null = null;
  private onClientProgress: ((p: DiCoSeProgress) => void) | null = null;

  constructor(opts: DicoseStemEngineOptions = {}) {
    this.baseUrl = (opts.baseUrl ?? DICOSE_DEFAULT_BASE_URL).replace(/\/$/, "");
    this.outputMode = opts.outputMode ?? "deterministic";
  }

  async load(onProgress?: ProgressCb): Promise<void> {
    if (this.client) return;
    const support = await checkSupport();
    if (!support.supported) {
      throw new Error(`DiCoSe needs WebGPU features this browser lacks: ${support.errors.join("; ")}`);
    }
    const client = new DiCoSeWorkerClient({
      manifestUrl: `${this.baseUrl}/manifest.json`,
      // The worker lives with the engine so the page bundle stays in charge of
      // bundling/fetch policy (see worker.ts) instead of the package dist.
      createWorker: () => new Worker(new URL("./worker.ts", import.meta.url), { type: "module", name: "dicose-webgpu" }),
      onProgress: (p) => this.onClientProgress?.(p),
    });
    this.onClientProgress = (p) => onProgress?.(loadProgress(p));
    try {
      await client.initialize();
    } catch (err) {
      void client.dispose();
      throw err;
    } finally {
      this.onClientProgress = null;
    }
    this.client = client;
  }

  /** Full-band decode via the vendored decoder: stereo preserved, WAV kept at
   * its container rate, torchaudio-matched sinc resampling inside the runtime. */
  async decodeFile(input: ArrayBuffer): Promise<SeparationInput> {
    const pcm = await decodeAudioBlob(new Blob([input]), { targetSampleRate: "source" });
    return { samples: pcm.left, right: pcm.right, sampleRate: pcm.sampleRate };
  }

  async separate(audio: SeparationInput, opts?: SeparateOpts): Promise<StemAudio[]> {
    const client = this.client;
    if (!client) throw new Error("DicoseStemEngine: call load() first");
    const left = audio.samples;
    const right = audio.right ?? audio.samples; // mono → duplicated channels (inputs are copied before transfer)
    const pcm: StereoPcm = {
      sampleRate: audio.sampleRate,
      length: left.length,
      left,
      right,
      channels: [left, right],
    };
    const totalSeconds = left.length / audio.sampleRate;
    const result = await client.separatePcm(pcm, {
      outputMode: this.outputMode,
      onProgress: (p) => {
        if (p.phase !== "chunk" || !p.total) return;
        const fraction = Math.min(1, (p.completed ?? 0) / p.total);
        opts?.onProgress?.({ processedSeconds: totalSeconds * fraction, totalSeconds, fraction });
      },
    });
    const stems: StemAudio[] = DICOSE_STEM_NAMES.map((name) => toStem(name, result.stems[name]));
    stems.push(toStem("instrumental", result.instrumental));
    return stems;
  }

  async dispose(): Promise<void> {
    const client = this.client;
    this.client = null;
    if (client) await client.dispose();
  }
}

function toStem(name: string, pcm: StereoPcm): StemAudio {
  return { name, samples: pcm.left, right: pcm.right, sampleRate: pcm.sampleRate };
}

function loadProgress(p: DiCoSeProgress): LoadProgress {
  if (p.phase === "weights" && p.total) {
    const fraction = Math.min(1, (p.completed ?? 0) / p.total);
    return { file: "dicose/weights.f16.bin", loaded: p.completed ?? 0, total: p.total, fraction };
  }
  return { file: p.detail ?? p.phase, loaded: 0, total: 0, fraction: 0 };
}
