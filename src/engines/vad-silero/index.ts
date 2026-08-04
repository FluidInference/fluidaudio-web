// Silero VAD — NO onnxruntime. The forward is hand-written (raw-silero.js), driven
// over 512-sample windows by silero.js. Weights (~1.2 MB, extracted from
// silero_vad.onnx's 16 kHz path via scripts/extract-silero-weights.py) are bundled
// with the app, so this engine has zero model download and zero runtime dep.
//
// Parity vs the original ORT model: ~1e-6 on the per-window speech probability
// across a full streaming sequence (scripts/smoke-silero-raw.mjs).

import type { AudioData, ProgressCb, SpeechRange, VadEngine } from "../../core/types";
import { makeSileroWeights, type SileroWeights } from "./raw-silero.js";
import { sileroDetect } from "./silero.js";
import manifest from "./silero-weights.manifest.json";
import weightsUrl from "./silero-weights.bin?url";

export class SileroVadEngine implements VadEngine {
  readonly id = "vad-silero";
  readonly label = "Silero VAD";
  private weights: SileroWeights | null = null;

  async load(onProgress?: ProgressCb): Promise<void> {
    const buf = await (await fetch(weightsUrl)).arrayBuffer();
    const bin = new Float32Array(buf);
    this.weights = makeSileroWeights(bin, manifest as any);
    onProgress?.({ file: "silero-weights.bin", loaded: 1, total: 1, fraction: 1 });
  }

  async detect(audio: AudioData): Promise<SpeechRange[]> {
    if (!this.weights) throw new Error("SileroVadEngine.load() not called");
    return sileroDetect({ weights: this.weights, audio: audio.samples });
  }

  async dispose(): Promise<void> {
    this.weights = null;
  }
}
