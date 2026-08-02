// Whisper ASR (99 languages) via transformers.js — WebGPU with WASM fallback.
// Complements Parakeet (European) with broad multilingual coverage. The pipeline
// handles mel + encoder + autoregressive decode internally; long audio is chunked.

import { pipeline } from "@huggingface/transformers";
import { webgpuAvailable } from "../../core/ort";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";

export interface WhisperOptions {
  /** HF model id; default multilingual base. Use whisper-tiny for speed. */
  model?: string;
}

export class WhisperEngine implements AsrEngine {
  readonly id = "asr-whisper";
  readonly label = "Whisper (99 langs)";
  private asr: any = null;
  private readonly model: string;

  constructor(opts: WhisperOptions = {}) {
    this.model = opts.model ?? "onnx-community/whisper-base";
  }

  async load(onProgress?: ProgressCb): Promise<void> {
    const device = webgpuAvailable() ? "webgpu" : "wasm";
    this.asr = await pipeline("automatic-speech-recognition", this.model, {
      device: device as any,
      dtype: device === "webgpu" ? "fp32" : "q8",
      progress_callback: (p: any) => {
        if (p?.status === "progress") {
          onProgress?.({ file: p.file ?? this.model, loaded: p.loaded ?? 0, total: p.total ?? 0, fraction: (p.progress ?? 0) / 100 });
        }
      },
    });
    onProgress?.({ file: this.model, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult> {
    if (!this.asr) throw new Error("WhisperEngine.load() not called");
    // >30s is chunked automatically; sample must be 16 kHz mono float.
    const out = await this.asr(audio.samples, { chunk_length_s: 30, stride_length_s: 5 });
    return { text: (Array.isArray(out) ? out[0]?.text : out?.text) ?? "" };
  }

  async dispose(): Promise<void> {
    await this.asr?.dispose?.();
    this.asr = null;
  }
}
