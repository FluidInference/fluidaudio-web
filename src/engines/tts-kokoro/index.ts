// Kokoro 82M TTS via kokoro-js (transformers.js under the hood; WebGPU with WASM
// fallback). The FluidAudio CoreML backend is a 7-stage ANE split — that split
// is Apple-specific and irrelevant here; the browser runs the single upstream
// Kokoro ONNX.
//
// English works out of the box. **Chinese is the open item**: kokoro-js's built-in
// phonemizer (misaki) covers English/EN-style G2P well; robust Mandarin needs a
// JS frontend — jieba word segmentation + polyphone (多音字) disambiguation +
// pinyin→IPA — mirroring FluidAudio's separate g2pW CoreML model. Until that's
// wired, `zh` falls back to kokoro-js's own handling (lower polyphone accuracy).

import { KokoroTTS } from "kokoro-js";
import { webgpuAvailable } from "../../core/ort";
import type { AudioData, ProgressCb, TtsEngine } from "../../core/types";

export interface KokoroOptions {
  /** "en" (v1.0) or "zh" (v1.1-zh). */
  lang?: "en" | "zh";
}

export class KokoroTtsEngine implements TtsEngine {
  readonly id: string;
  readonly label: string;
  private tts: KokoroTTS | null = null;
  private readonly modelId: string;

  constructor(private opts: KokoroOptions = {}) {
    const zh = opts.lang === "zh";
    this.id = zh ? "tts-kokoro-zh" : "tts-kokoro-en";
    this.label = zh ? "Kokoro TTS (Chinese)" : "Kokoro TTS (English)";
    this.modelId = zh
      ? "onnx-community/Kokoro-82M-v1.1-zh-ONNX"
      : "onnx-community/Kokoro-82M-v1.0-ONNX";
  }

  async load(onProgress?: ProgressCb): Promise<void> {
    const device = webgpuAvailable() ? "webgpu" : "wasm";
    this.tts = await KokoroTTS.from_pretrained(this.modelId, {
      dtype: device === "webgpu" ? "fp32" : "q8",
      device,
      progress_callback: (p: any) => {
        // kokoro-js emits {status, file, progress, loaded, total}
        if (p?.status === "progress") {
          onProgress?.({
            file: p.file ?? this.modelId,
            loaded: p.loaded ?? 0,
            total: p.total ?? 0,
            fraction: (p.progress ?? 0) / 100,
          });
        }
      },
    });
  }

  async synthesize(text: string, opts?: { voice?: string; speed?: number }): Promise<AudioData> {
    if (!this.tts) throw new Error("KokoroTtsEngine.load() not called");
    const voice = opts?.voice ?? (this.opts.lang === "zh" ? "zf_001" : "af_heart");
    // voice is a wide string here; kokoro-js types it as a per-model literal union.
    const audio = await this.tts.generate(text, { voice: voice as any, speed: opts?.speed ?? 1 });
    // kokoro-js RawAudio: { audio: Float32Array, sampling_rate: number }
    return { samples: audio.audio as Float32Array, sampleRate: audio.sampling_rate };
  }

  async voices(): Promise<string[]> {
    if (!this.tts) return [];
    return Object.keys((this.tts as any).voices ?? {});
  }

  async dispose(): Promise<void> {
    this.tts = null;
  }
}
