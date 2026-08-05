// Kokoro 82M TTS — fully ORT-free. No kokoro-js / transformers.js / onnxruntime.
// The StyleTTS2+iSTFTNet pipeline (ALBERT frontend → prosody predictor → SineGen →
// iSTFTNet decoder+generator) is hand-ported to raw WebGPU / WASM-SIMD in
// src/gpu/kokoro-synth.js (parity vs the ONNX model: waveform corr ~0.97, source
// spec exact, decode3 5e-5). Weights: FluidInference/fluidaudio-web/kokoro{,-zh}.
//
// G2P (text → IPA phonemes): English via the Misaki lexicon (lexicon.js); Chinese
// via pinyin-pro → IPA (chinese-g2p.js). Both onnx-free. Out-of-lexicon English
// currently has no espeak fallback (kokoro-js provided that) — a follow-up is
// espeak-ng WASM; until then OOV words are skipped.

import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AudioData, ProgressCb, TtsEngine } from "../../core/types";
import { EnglishLexicon } from "./lexicon.js";
import { chineseToIpa } from "./chinese-g2p.js";
import { loadKokoroBackend } from "./synth-backend.js";
import vocabEn from "./vocab.json";

export interface KokoroOptions {
  /** "en" (v1.0) or "zh" (v1.1-zh). */
  lang?: "en" | "zh";
}

export class KokoroTtsEngine implements TtsEngine {
  readonly id: string;
  readonly label: string;
  private readonly zh: boolean;
  private backend: Awaited<ReturnType<typeof loadKokoroBackend>> | null = null;
  private lexicon: EnglishLexicon | null = null;

  constructor(opts: KokoroOptions = {}) {
    this.zh = opts.lang === "zh";
    this.id = this.zh ? "tts-kokoro-zh" : "tts-kokoro-en";
    this.label = this.zh ? "Kokoro TTS (Chinese)" : "Kokoro TTS (English)";
  }

  async load(onProgress?: ProgressCb): Promise<void> {
    if (this.zh) {
      // Kokoro v1.1-zh has a DIFFERENT architecture (443 vs 554 tensors, different
      // bert namespace) than v1.0 — the raw port (src/gpu/kokoro-synth.js) is
      // validated on v1.0/en only. zh needs its own extraction + validation.
      throw new Error("Kokoro zh: ORT-free port pending (v1.1-zh has a different architecture from the ported v1.0).");
    }
    this.backend = await loadKokoroBackend(fetchCached as any, hfUrl, vocabEn as Record<string, number>, {
      modelDir: this.zh ? "kokoro-zh" : "kokoro",
      voiceRepo: this.zh ? "onnx-community/Kokoro-82M-v1.1-zh-ONNX" : "onnx-community/Kokoro-82M-v1.0-ONNX",
      onProgress,
    });
    if (!this.zh) {
      try {
        this.lexicon = await EnglishLexicon.load(fetchCached as any);
      } catch {
        this.lexicon = null;
      }
    }
  }

  async synthesize(text: string, opts?: { voice?: string; speed?: number }): Promise<AudioData> {
    if (!this.backend) throw new Error("KokoroTtsEngine.load() not called");
    const voice = opts?.voice ?? (this.zh ? "zf_001" : "af_heart");

    let phonemes = "";
    if (this.zh) {
      phonemes = chineseToIpa(text).phonemes ?? "";
    } else if (this.lexicon) {
      phonemes = this.lexicon.phonemize(text).phonemes ?? "";
    }
    if (!phonemes) throw new Error(`Kokoro: no phonemes for input (G2P coverage 0). OOV English needs the espeak-ng follow-up.`);

    const samples = await this.backend.synthFromPhonemes(phonemes, voice);
    return { samples, sampleRate: 24000 };
  }

  async voices(): Promise<string[]> {
    return this.zh ? ["zf_001", "zm_010"] : ["af_heart", "af_bella", "am_michael"];
  }

  async dispose(): Promise<void> {
    this.backend = null;
    this.lexicon = null;
  }
}
