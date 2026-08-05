// Kokoro 82M TTS — fully ORT-free. No kokoro-js / transformers.js / onnxruntime.
// The StyleTTS2+iSTFTNet pipeline (ALBERT frontend → prosody predictor → SineGen →
// iSTFTNet decoder+generator) is hand-ported to raw WebGPU / WASM-SIMD in
// src/gpu/kokoro-synth.js (parity vs the ONNX model: waveform corr ~0.97, source
// spec exact, decode3 5e-5). Weights: FluidInference/fluidaudio-web/kokoro{,-zh}.
//
// G2P (text → phonemes): English via the Misaki lexicon (lexicon.js, IPA); Chinese
// via pinyin-pro → misaki[zh] v1.1 format (zh-frontend-v11.js, Bopomofo + tone
// digits). Both onnx-free. Out-of-lexicon English currently has no espeak fallback
// (kokoro-js provided that) — OOV words are skipped; the gold/silver lexicon tiers
// carry coverage.

import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AudioData, ProgressCb, TtsEngine } from "../../core/types";
import { EnglishLexicon } from "./lexicon.js";
import { chineseToZh11 } from "./zh-frontend-v11.js";
import { loadKokoroBackend } from "./synth-backend.js";
import vocabEn from "./vocab.json";
import vocabZh from "./vocab-zh.json";

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
    this.backend = await loadKokoroBackend(fetchCached as any, hfUrl, (this.zh ? vocabZh : vocabEn) as Record<string, number>, {
      modelDir: this.zh ? "kokoro-zh" : "kokoro",
      voiceRepo: this.zh ? "onnx-community/Kokoro-82M-v1.1-zh-ONNX" : "onnx-community/Kokoro-82M-v1.0-ONNX",
      onProgress,
    });
    if (!this.zh) {
      // Fail loudly: with the espeak fallback gone, English synthesis is impossible
      // without the lexicon — swallowing this error would brick every synthesize().
      this.lexicon = await EnglishLexicon.load(fetchCached as any);
    }
  }

  async synthesize(text: string, opts?: { voice?: string; speed?: number }): Promise<AudioData> {
    if (!this.backend) throw new Error("KokoroTtsEngine.load() not called");
    const voice = opts?.voice ?? (this.zh ? "zf_001" : "af_heart");

    let phonemes = "", coverage = 1;
    if (this.zh) {
      ({ phonemes, coverage } = chineseToZh11(text));
    } else if (this.lexicon) {
      ({ phonemes, coverage } = this.lexicon.phonemize(text));
    }
    if (!phonemes) throw new Error(`Kokoro: no phonemes for input (G2P coverage 0).`);
    // Words the G2P can't cover are omitted from the audio — surface it instead
    // of silently dropping them (no espeak fallback by design; gold/silver
    // lexicon tiers carry coverage).
    if (coverage < 0.95) console.warn(`[kokoro] G2P coverage ${(coverage * 100).toFixed(0)}% — some words will be missing from the audio`);

    const samples = await this.backend.synthFromPhonemes(phonemes, voice, opts?.speed ?? 1);
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
