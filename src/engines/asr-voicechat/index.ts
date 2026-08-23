// VoiceChat-11B STT (NVIDIA VoiceChat-11B's user-transcription chain: 609M causal
// FastConformer encoder + RNNT head), fully ORT-free.
//
// The speech chain's perception encoder is a cache-aware streaming FastConformer
// (att_context [70,0] chunked_limited = per-frame causal, 70-frame look-back);
// offline-with-causal-mask and cache-aware streaming compute the same function,
// so both paths share the raw runtime. Everything hand-written (no onnxruntime):
//   • Mel: JsPreprocessor NA log-mel (no CMVN), log guard 2^-24 (NeMo default —
//     the VoiceChat preprocessor config doesn't override it; Nemotron's 1e-10 is
//     that model's own choice).
//   • Encoder: shared raw FastConformer (raw-encoder.js) with the VoiceChat
//     causal config (VOICECHAT_CFG) — causal subsampling pad, causal depthwise
//     (dwK 9), conv-module LayerNorm, per-frame causal mask. fp16 weights.
//   • Decoder+joint: 2-layer LSTM RNNT (raw-decoder-voicechat.js, vocab 1025,
//     blank 1024) in plain JS; the joint's 1024→640 enc projection rides the GPU
//     encode batch. No prompt_kernel — that's Nemotron-only.
// Parity gate: scripts/ci-smoke-voicechat.mjs (sample_general.wav transcript
// matches the CoreML/torch reference from mobius test_e2e_stt.py).
//
// Weights are hosted at FluidInference/fluidaudio-web (voicechat-stt/) like the
// other engines. To regenerate from the source checkpoint, run
//   uv run --with safetensors,numpy,sentencepiece python3 scripts/extract-voicechat-stt.py
// and pass `baseUrl` pointing at the local export (the dev middleware serves
// models-local/ at /models).

import { fetchCached, hfUrl } from "../../core/modelCache.js";
import type { AsrEngine, AsrResult, AudioData, ProgressCb, StreamingAsrEngine } from "../../core/types.js";
import { createContext } from "../../gpu/context.js";
import { loadParakeetEncoder } from "../asr-parakeet/raw-encoder.js";
import { createEncodeStream, encodeStreamPush, encodeStreamFlush, disposeEncodeStream } from "../asr-parakeet/streaming-encoder.js";
import { JsPreprocessor } from "../asr-nemotron/nemotron-mel.js";
import { StreamingMel } from "../asr-nemotron/streaming-mel.js";
import { tokensToWords } from "../../core/captions.js";
import { loadVoicechatDecoder, createVoicechatStream, voicechatDecodeCont } from "./raw-decoder-voicechat.js";
import { VOICECHAT_CFG } from "./config.js";

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";
const PROJ_D = 640; // joint enc projection width (1024→640 GEMM rides the encode batch)
const FRAME_SEC = 0.08; // 10ms mel hop × 8× subsampling
const BATCH_CHUNK = 768; // batch-transcribe pass size (subsampled frames), see EOU sweep
const LOG_GUARD = 2 ** -24; // NeMo AudioToMelSpectrogramPreprocessor default

/** StreamingMel with the VoiceChat log guard (StreamingMel only consults `pre`). */
function makeStreamingMel(): StreamingMel {
  const mel = new StreamingMel(128);
  (mel as any).pre = new (JsPreprocessor as any)({ nMels: 128, logGuard: LOG_GUARD });
  return mel;
}

interface DecodeStream {
  mel: StreamingMel;
  encSt: any;
  dec: any; // RNNT continuation state (pred LSTM + cached predProj)
  ids: number[];
  idTimes: number[];
  subT: number;
  finished: boolean;
  broken: boolean;
}

export class VoicechatSttEngine implements AsrEngine, StreamingAsrEngine {
  readonly id = "asr-voicechat";
  readonly label = "VoiceChat 11B STT";
  private ctx: any = null;
  private enc: any = null;
  private dec: any = null;
  private vocab: Record<string, string> | null = null; // id-string → sentencepiece piece
  private projW: any = null; // joint enc projection 1024→640, GPU-side
  private projB: any = null;
  private stream: DecodeStream | null = null;
  private op: Promise<unknown> = Promise.resolve();

  /** @param opts.baseUrl weight directory override (default: the HF weights repo). */
  constructor(private opts: { baseUrl?: string } = {}) {}

  private serialize<T>(fn: () => Promise<T>): Promise<T> {
    const r = this.op.then(fn, fn);
    this.op = r.then(
      () => undefined,
      () => undefined,
    );
    return r;
  }

  private idsToText(ids: number[]): string {
    return ids
      .map((i) => this.vocab![i] ?? "")
      .filter((tk) => !tk.startsWith("<"))
      .join("")
      .replace(/▁/g, " ")
      .trim();
  }

  async load(onProgress?: ProgressCb): Promise<void> {
    this.ctx = await createContext({ onBackend: (b) => console.info(`[asr-voicechat] backend: ${b}`) });
    const base = this.opts.baseUrl;
    const bytes = (name: string) => fetchCached(base ? `${base}/${name}` : hfUrl(WEIGHTS_REPO, `voicechat-stt/${name}`), onProgress, name);
    const json = async (name: string) => JSON.parse(new TextDecoder().decode(await bytes(name)));

    const encMan = await json("encoder-f16.manifest.json");
    const encBin = await bytes("encoder-f16.bin");
    const decMan = await json("decoder-fp32.manifest.json");
    const decBin = await bytes("decoder-fp32.bin");
    this.vocab = await json("vocab.json");

    this.enc = loadParakeetEncoder(this.ctx, encBin, encMan, VOICECHAT_CFG);
    this.dec = loadVoicechatDecoder(new Float32Array(decBin.buffer, decBin.byteOffset, decBin.byteLength / 4), decMan);
    this.projW = this.ctx.upload(this.dec.encW.slice(), 1024, PROJ_D);
    this.projB = this.ctx.upload(this.dec.encB.slice(), 1, PROJ_D);
    onProgress?.({ file: base ?? `${WEIGHTS_REPO}/voicechat-stt`, loaded: 1, total: 1, fraction: 1 });
  }

  // ── true streaming: per-frame causal (attRight 0) needs no lookahead —
  // emitted frames are final, streaming == offline. ──
  push(chunk: Float32Array): Promise<string> {
    return this.serialize(() => this.pushInner(chunk));
  }

  private async pushInner(chunk: Float32Array): Promise<string> {
    if (!this.enc || !this.dec || !this.vocab) throw new Error("VoicechatSttEngine.load() not called");
    if (this.stream?.finished) throw new Error("finish() already called — reset() to start a new stream");
    if (this.stream?.broken) throw new Error("stream broken by an earlier push failure — reset()");
    if (!this.stream) {
      this.stream = {
        mel: makeStreamingMel(),
        encSt: createEncodeStream(this.ctx, this.enc, { proj: { w: this.projW, b: this.projB } }),
        dec: createVoicechatStream(this.dec),
        ids: [],
        idTimes: [],
        subT: 0,
        finished: false,
        broken: false,
      };
    }
    const s = this.stream;
    try {
      const { data, count } = s.mel.push(chunk);
      if (data && count > 0) {
        const out = await encodeStreamPush(this.ctx, s.encSt, data, count, { maxChunk: 16 });
        if (out) this.consume(s, out);
      }
    } catch (err) {
      s.broken = true;
      throw err;
    }
    return this.idsToText(s.ids);
  }

  finish(): Promise<string> {
    return this.serialize(() => this.finishInner());
  }

  private async finishInner(): Promise<string> {
    if (!this.stream || !this.vocab) return "";
    const s = this.stream;
    if (s.finished || s.broken) return this.idsToText(s.ids);
    s.finished = true;
    const { data, count } = s.mel.flush();
    if (data && count > 0) {
      const out = await encodeStreamPush(this.ctx, s.encSt, data, count, { maxChunk: 16 });
      if (out) this.consume(s, out);
    }
    const tail = await encodeStreamFlush(this.ctx, s.encSt);
    if (tail) this.consume(s, tail);
    return this.idsToText(s.ids);
  }

  reset(): void {
    if (this.stream) {
      disposeEncodeStream(this.ctx, this.stream.encSt);
      this.ctx?.trimPool();
    }
    this.stream = null;
  }

  private consume(st: DecodeStream, framesProj: Float32Array): void {
    const n = framesProj.length / PROJ_D;
    const r = voicechatDecodeCont(this.dec, st.dec, framesProj, n);
    st.ids.push(...r.ids);
    st.idTimes.push(...r.idFrames.map((f) => (st.subT + f) * FRAME_SEC));
    st.subT += n;
  }

  transcribe(audio: AudioData): Promise<AsrResult> {
    // Serialized on the same chain as push/finish (see asr-nemotron: a queued
    // push would otherwise race the batch loop on this.stream).
    return this.serialize(() => this.transcribeInner(audio));
  }

  private async transcribeInner(audio: AudioData): Promise<AsrResult> {
    if (!this.enc || !this.dec || !this.vocab) throw new Error("VoicechatSttEngine.load() not called");
    if (this.stream) throw new Error("a live stream is active — reset() before batch transcribe");
    const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
    const t0 = now();
    // Batch IS a big-chunk stream: linear-cost cache-carrying encode with the
    // joint enc projection riding each chunk batch, one continuous RNNT decode.
    const mel = makeStreamingMel();
    const encSt = createEncodeStream(this.ctx, this.enc, { proj: { w: this.projW, b: this.projB } });
    const st: DecodeStream = {
      mel,
      encSt,
      dec: createVoicechatStream(this.dec),
      ids: [],
      idTimes: [],
      subT: 0,
      finished: false,
      broken: false,
    };
    let melMs = 0;
    let encMs = 0;
    let decMs = 0;
    const consumeB = (frames: Float32Array) => {
      const td = now();
      this.consume(st, frames);
      decMs += now() - td;
    };
    const SLICE = 240 * 16000;
    try {
      for (let off = 0; off < audio.samples.length; off += SLICE) {
        const tm = now();
        const { data, count } = mel.push(audio.samples.subarray(off, Math.min(off + SLICE, audio.samples.length)));
        melMs += now() - tm;
        if (data && count > 0) {
          const te = now();
          const out = await encodeStreamPush(this.ctx, encSt, data, count, { maxChunk: BATCH_CHUNK });
          encMs += now() - te;
          if (out) consumeB(out);
        }
      }
      const tm = now();
      const fl = mel.flush();
      melMs += now() - tm;
      const te = now();
      if (fl.data && fl.count > 0) {
        const out = await encodeStreamPush(this.ctx, encSt, fl.data, fl.count, { maxChunk: BATCH_CHUNK });
        if (out) consumeB(out);
      }
      const tail = await encodeStreamFlush(this.ctx, encSt);
      encMs += now() - te;
      if (tail) consumeB(tail);
    } finally {
      disposeEncodeStream(this.ctx, encSt);
      this.ctx.trimPool();
    }
    return {
      text: this.idsToText(st.ids),
      segments: tokensToWords(st.ids, st.idTimes, this.vocab as Record<number, string>, (id) => (this.vocab![id] ?? "<").startsWith("<")),
      metrics: { melMs: +melMs.toFixed(1), encodeMs: +encMs.toFixed(1), decodeMs: +decMs.toFixed(1), totalMs: +(now() - t0).toFixed(1) },
    };
  }

  async dispose(): Promise<void> {
    this.reset();
    this.ctx?.destroy();
    this.ctx = this.enc = this.dec = this.vocab = null;
  }
}
