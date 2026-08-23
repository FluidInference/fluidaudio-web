// VoiceChat-11B TTS ("Aria" voice) — the speech-decoder slice of NVIDIA's
// full-duplex VoiceChat-11B as a standalone TTS engine, fully ORT-free.
// Gemma3 backbone (28L×1152, CFG 0.2 → 2 evals/step) + MoG head + 31-quantizer
// PRVQ codec (12.5 Hz tokens → 22.05 kHz). See docs/VOICECHAT.md phase 2 and
// src/engines/tts-voicechat/model.js for the architecture notes.
//
// Weights are NOT hosted yet — local export only:
//   /tmp/vc-tts-env/bin/python scripts/extract-voicechat-tts.py … (see header)
// and the dev middleware serves models-local/ at /models. The registry probe
// hides the engine when the export is absent.
//
// Backend: forced WASM for now. The per-step loop interleaves host math
// (RMSNorm/RoPE/attention over CPU KV caches, MoG selection, PRVQ search) with
// ComputeContext GEMMs; on WebGPU that pattern is a sync storm, on WASM the
// hops are free. Profiled under node (see PR): the GEMV copies dominate.
// A GPU-resident step loop is deferred work.

import { fetchCached } from "../../core/modelCache.js";
import type { AudioData, ProgressCb, TtsEngine } from "../../core/types.js";
import { createContext } from "../../gpu/context.js";
import { VoicechatTokenizer } from "./tokenizer.js";
import { WeightStore } from "./weights.js";
import { loadVoicechatTtsModel, synthesizeCodes, rvqNorms } from "./model.js";
import { loadVoicechatCodec, dequantize, codecDecode } from "./codec.js";

const MAX_FRAMES = 720; // ~58 s of audio; keeps KV caches + codec batch bounded
const TAIL_KEEP = 6; // silence frames kept after the last speech frame (~0.5 s)

export interface VoicechatTtsOptions {
  /** Weight directory (default: the dev middleware's /models/voicechat-tts). */
  baseUrl?: string;
  /** Argmax-component MoG (no sampling noise) — the parity configuration. */
  deterministic?: boolean;
  seed?: number;
}

export class VoicechatTtsEngine implements TtsEngine {
  readonly id = "tts-voicechat";
  readonly label = "VoiceChat TTS (Aria)";
  private ctx: any = null;
  private model: any = null;
  private codec: any = null;
  private tok: VoicechatTokenizer | null = null;
  private cfg: any = null;
  private charVocab: Map<string, number> | null = null;
  private vocabList: string[] | null = null;
  private rvqNormSq: Float64Array | null = null;

  constructor(private opts: VoicechatTtsOptions = {}) {}

  private base(): string {
    return this.opts.baseUrl ?? `${import.meta.env.BASE_URL}models/voicechat-tts`;
  }

  async load(onProgress?: ProgressCb): Promise<void> {
    // WASM by choice (see header): the host↔context interleave is free there.
    this.ctx = await createContext({ backend: "wasm", onBackend: (b) => console.info(`[tts-voicechat] backend: ${b}`) });
    // Local exports are mutable (extractor re-runs overwrite in place) — skip the cache.
    const bytes = (name: string) => fetchCached(`${this.base()}/${name}`, onProgress, name, { skipCache: true });
    const json = async (name: string) => JSON.parse(new TextDecoder().decode(await bytes(name)));

    this.cfg = await json("config.json");
    const tokJson = await json("tokenizer.json");
    this.tok = new VoicechatTokenizer(tokJson);
    this.vocabList = tokJson.vocab;
    this.charVocab = new Map((tokJson.chars as string[]).map((c: string, i: number) => [c, i]));

    const shards = async (stem: string) => {
      const man = await json(`${stem}.manifest.json`);
      const bins: Uint8Array[] = [];
      for (let i = 0; i < man.shards; i++) bins.push(await bytes(`${stem}.${i}.bin`));
      return new WeightStore(bins, man);
    };
    const ttsW = await shards("tts");
    const codecW = await shards("codec");
    this.model = loadVoicechatTtsModel(this.ctx, ttsW, this.cfg);
    this.codec = loadVoicechatCodec(this.ctx, codecW);
    this.rvqNormSq = rvqNorms(this.codec.rvq, this.cfg.numQuantizers, this.cfg.codebook, this.cfg.latent);
    onProgress?.({ file: this.base(), loaded: 1, total: 1, fraction: 1 });
  }

  /** Char-id decomposition of a token (build_vocabs semantics: chars not in the
   * byte-level char vocab are dropped; may be empty). */
  private tokenChars(id: number): number[] {
    const s = this.vocabList![id] ?? "";
    const out: number[] = [];
    for (const ch of s) {
      const ci = this.charVocab!.get(ch);
      if (ci !== undefined) out.push(ci);
    }
    return out;
  }

  async synthesize(text: string, _opts?: { voice?: string; speed?: number }): Promise<AudioData> {
    if (!this.model || !this.tok) throw new Error("VoicechatTtsEngine.load() not called");
    const cfg = this.cfg;
    const ids = this.tok.encode(text);
    if (!ids.length) throw new Error("tts-voicechat: no tokens for input text");
    // One subword per 80 ms frame: [bos, text…, pad…, eos]. The model speaks at
    // its own pace and pads with silence; ~3.2 frames/token + slack covers
    // normal speech rate, and trailing silence is trimmed after generation.
    const frames = Math.min(MAX_FRAMES, Math.ceil(ids.length * 3.2) + 15);
    if (ids.length + 2 > MAX_FRAMES) throw new Error(`tts-voicechat: text too long (${ids.length} tokens)`);
    const frameTokens = new Array(frames).fill(cfg.textPad);
    frameTokens[0] = cfg.textBos;
    for (let i = 0; i < ids.length; i++) frameTokens[1 + i] = ids[i];
    frameTokens[frames - 1] = cfg.textEos;

    const chars = new Map<number, number[]>();
    for (const t of [...frameTokens, ...cfg.warmSubwordIds]) if (!chars.has(t)) chars.set(t, this.tokenChars(t));

    const { codes } = await synthesizeCodes(this.ctx, this.model, frameTokens, chars, this.codec.rvq, {
      deterministic: this.opts.deterministic ?? false,
      seed: this.opts.seed ?? (Math.random() * 2 ** 31) | 0,
      rvqNormSq: this.rvqNormSq!,
    });

    // Trim trailing all-silence frames (codec time is ~linear in frames).
    const sil = cfg.silenceTokens as number[];
    let last = codes.length - 1;
    while (last >= 0 && codes[last].every((v: number, q: number) => v === sil[q])) last--;
    const keep = Math.min(codes.length, last + 1 + TAIL_KEEP);
    const kept = codes.slice(0, Math.max(keep, 1));

    const latents = dequantize(this.codec.rvq, kept, cfg.latent, cfg.numQuantizers);
    const samples = await codecDecode(this.ctx, this.codec, latents, kept.length);
    return { samples, sampleRate: cfg.sampleRate };
  }

  async voices(): Promise<string[]> {
    return ["aria"];
  }

  async dispose(): Promise<void> {
    this.ctx?.destroy();
    this.ctx = this.model = this.codec = this.cfg = null;
    this.tok = null;
    this.charVocab = null;
    this.vocabList = null;
    this.rvqNormSq = null;
  }
}
