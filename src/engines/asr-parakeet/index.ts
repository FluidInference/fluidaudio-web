// Parakeet TDT 0.6B v3 — offline transcription.
//
// Uses the `parakeet.js` library, which implements the exact pipeline FluidAudio
// runs in Swift: JS mel frontend → FastConformer encoder (WebGPU) → TDT greedy
// decode with the small decoder+joint (WASM). The `webgpu-hybrid` backend is
// precisely the split the parakeet.js authors found optimal (and matches our
// core/ort.ts policy): heavy encoder on the GPU, tiny dynamic decoder on WASM.
//
// NOTE: parakeet.js is a browser library (Blob URLs, fetch, WebGPU) — it runs
// under `npm run dev` in a real browser. It is a 69★ single-author dependency;
// the intent is to internalize the mel + TDT loop later (its `src/mel.js` +
// `src/parakeet.js` are the readable recipe). Tracked in docs/ARCHITECTURE.md.

import { fromHub } from "parakeet.js";
import { webgpuAvailable } from "../../core/ort";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";

export interface ParakeetOptions {
  /** Model key or HF repo id. Defaults to the multilingual v3. */
  model?: string;
}

export class ParakeetV3Engine implements AsrEngine {
  readonly id = "asr-parakeet";
  readonly label = "Parakeet TDT 0.6B v3";
  private model: any = null;
  private readonly modelKey: string;

  constructor(opts: ParakeetOptions = {}) {
    this.modelKey = opts.model ?? "parakeet-tdt-0.6b-v3";
  }

  async load(onProgress?: ProgressCb): Promise<void> {
    onProgress?.({ file: this.modelKey, loaded: 0, total: 1, fraction: 0.05 });
    this.model = await fromHub(this.modelKey, {
      // int8 is the reliable browser path; encoder still runs on WebGPU.
      encoderQuant: "int8",
      decoderQuant: "int8",
      backend: webgpuAvailable() ? "webgpu-hybrid" : "wasm",
      progress: (p: any) => {
        const frac = typeof p?.progress === "number" ? p.progress / 100 : p?.fraction ?? 0;
        onProgress?.({ file: p?.file ?? this.modelKey, loaded: p?.loaded ?? 0, total: p?.total ?? 0, fraction: frac });
      },
    });
    onProgress?.({ file: this.modelKey, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult> {
    if (!this.model) throw new Error("ParakeetV3Engine.load() not called");
    const res = await this.model.transcribe(audio.samples, audio.sampleRate);
    const text: string = res?.text ?? res?.utterance ?? "";
    const segments = Array.isArray(res?.words)
      ? res.words.map((w: any) => ({ text: w.text ?? w.word ?? "", start: w.start ?? 0, end: w.end ?? 0 }))
      : undefined;
    return { text, segments };
  }

  async dispose(): Promise<void> {
    // parakeet.js sessions are GC'd with the model reference.
    this.model = null;
  }
}
