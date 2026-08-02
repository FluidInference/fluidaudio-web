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
import { fetchCached } from "../../core/modelCache";
import type { AudioData, ProgressCb, TtsEngine } from "../../core/types";
import { EnglishLexicon } from "./lexicon.js";

export interface KokoroOptions {
  /** "en" (v1.0) or "zh" (v1.1-zh). */
  lang?: "en" | "zh";
}

export class KokoroTtsEngine implements TtsEngine {
  readonly id: string;
  readonly label: string;
  private tts: KokoroTTS | null = null;
  private readonly modelId: string;
  private lexicon: EnglishLexicon | null = null;

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
    // English: prefer the Misaki lexicon (FluidAudio frontend) over espeak.
    if (this.opts.lang !== "zh") {
      try {
        this.lexicon = await EnglishLexicon.load(fetchCached as any);
      } catch {
        this.lexicon = null; // lexicon optional; espeak fallback still works
      }
    }
  }

  async synthesize(text: string, opts?: { voice?: string; speed?: number }): Promise<AudioData> {
    if (!this.tts) throw new Error("KokoroTtsEngine.load() not called");
    const voice = opts?.voice ?? (this.opts.lang === "zh" ? "zf_001" : "af_heart");
    const speed = opts?.speed ?? 1;

    // Lexicon-first English: phonemize via the Misaki lexicon and inject through
    // generate_from_ids (skips espeak). Fall back to kokoro-js generate() when
    // coverage is low (OOV words) — that path uses espeak. Chinese will reuse
    // this same injection with a g2pW-derived phoneme string (see docs/KOKORO_ZH.md).
    if (this.lexicon) {
      const { phonemes, coverage } = this.lexicon.phonemize(text);
      if (coverage >= 0.95 && phonemes) {
        const audio = await this.synthFromPhonemes(phonemes, voice, speed);
        if (audio) return audio;
      }
    }
    const audio = await this.tts.generate(text, { voice: voice as any, speed });
    return { samples: audio.audio as Float32Array, sampleRate: audio.sampling_rate };
  }

  /** Inject a phoneme string straight into Kokoro (tokenizer → generate_from_ids). */
  async synthFromPhonemes(phonemes: string, voice: string, speed = 1): Promise<AudioData | null> {
    const tts = this.tts as any;
    if (!tts) return null;
    const { input_ids } = tts.tokenizer(phonemes, { truncation: true });
    const audio = await tts.generate_from_ids(input_ids, { voice, speed });
    return { samples: audio.audio as Float32Array, sampleRate: audio.sampling_rate };
  }

  async voices(): Promise<string[]> {
    if (!this.tts) return [];
    return Object.keys((this.tts as any).voices ?? {});
  }

  async dispose(): Promise<void> {
    this.tts = null;
    this.lexicon = null;
  }
}
