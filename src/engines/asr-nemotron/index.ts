// Nemotron 3.5 ASR (multilingual FastConformer-RNNT, 24L d1024), fully ORT-free.
//
// The shipped soniqo model is a cache-aware STREAMING export; here we run the same
// FastConformer OFFLINE (whole clip) on raw WebGPU — cache-aware streaming and
// offline-with-limited-context-mask are equivalent, so we skip the cache plumbing.
// Everything hand-written (no onnxruntime):
//   • Mel: JsPreprocessor NA log-mel (no CMVN).
//   • Encoder: shared raw FastConformer (raw-encoder.js) with Nemotron streaming
//     config — causal subsampling pad, causal depthwise (dwK 9), cache-aware mask
//     (chunk 4, left 56, right 3). int8 weights (600M model is int8-robust, unlike
//     the 120M EOU). Runs on WebGPU.
//   • prompt_kernel: multilingual conditioning MLP (concat conformer_out + language
//     one-hot → 1152→2048→1024) → encoded_output.
//   • Decoder+joint: 2-layer LSTM RNNT (raw-decoder-nemotron.js), plain JS.
// Full offline int8 path == coherent transcript matching the ORT streaming reference.

import { fetchCached, hfUrl } from "../../core/modelCache.js";
import type { AsrEngine, AsrResult, AudioData, ProgressCb, StreamingAsrEngine } from "../../core/types.js";
import { createContext } from "../../gpu/context.js";
import { loadParakeetEncoder, parakeetEncode } from "../asr-parakeet/raw-encoder.js";
import { loadNemotronDecoder, nemotronDecode, loadPromptKernel, applyPromptKernel, createNemotronStream, nemotronDecodeCont } from "./raw-decoder-nemotron.js";
import { createEncodeStream, encodeStreamPush, encodeStreamFlush, disposeEncodeStream } from "../asr-parakeet/streaming-encoder.js";
import { StreamingMel } from "./streaming-mel.js";
import { JsPreprocessor } from "./nemotron-mel.js";

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";
// Nemotron streaming FastConformer config (see raw-encoder.js): causal subsampling
// pad, causal depthwise conv, cache-aware attention (chunk 4, left 56, right 3).
const NEMO_CFG = { melBins: 128, subPad: { t: 2, b: 1, l: 2, r: 1 }, convCausal: true, attChunk: 4, attLeft: 56, attRight: 3 };

const ENC_D = 1024;

export class NemotronEngine implements AsrEngine, StreamingAsrEngine {
  readonly id = "asr-nemotron";
  readonly label = "Nemotron 3.5 multilingual";
  private ctx: any = null;
  private enc: any = null;
  private pk: any = null;
  private dec: any = null;
  private mel = new JsPreprocessor({ nMels: 128 });
  private vocab: string[] | null = null;
  private langMap: Record<string, number> = {};
  private stream: { mel: StreamingMel; encSt: any; decSt: any; ids: number[]; subT: number; finished: boolean; broken: boolean } | null = null;
  private op: Promise<unknown> = Promise.resolve();

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
      .replace(/\u2581/g, " ")
      .trim();
  }

  // ── true streaming (docs/STREAMING.md) — provisional-tail lookahead for the
  // right-context-3 attention: B=4 chunks (1.28s) per the measured decay curve;
  // gate: scripts/streaming-encode-check.mjs (tokens == offline). ──
  push(chunk: Float32Array): Promise<string> {
    return this.serialize(() => this.pushInner(chunk));
  }

  private async pushInner(chunk: Float32Array): Promise<string> {
    if (!this.enc || !this.dec || !this.vocab) throw new Error("NemotronEngine.load() not called");
    if (this.stream?.finished) throw new Error("finish() already called — reset() to start a new stream");
    if (this.stream?.broken) throw new Error("stream broken by an earlier push failure — reset()");
    if (!this.stream) {
      this.stream = {
        mel: new StreamingMel(128),
        encSt: createEncodeStream(this.ctx, this.enc, { lookaheadChunks: 4 }),
        decSt: createNemotronStream(this.dec),
        ids: [],
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
        if (out) this.consume(out);
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
      if (out) this.consume(out);
    }
    const tail = await encodeStreamFlush(this.ctx, s.encSt);
    if (tail) this.consume(tail);
    return this.idsToText(s.ids);
  }

  reset(): void {
    if (this.stream) {
      disposeEncodeStream(this.ctx, this.stream.encSt);
      this.ctx?.trimPool?.();
    }
    this.stream = null;
  }

  private consume(frames: Float32Array): void {
    const s = this.stream!;
    const n = frames.length / ENC_D;
    const langId = this.langMap[this.opts.language ?? "en-US"] ?? 0;
    const enc = applyPromptKernel(this.pk, frames, n, langId);
    s.ids.push(...nemotronDecodeCont(this.dec, s.decSt, enc, n).ids);
    s.subT += n;
  }

  /** @param opts.language e.g. "en-US" / "de" / "zh-CN" / "ja-JP" (default en-US). */
  constructor(private opts: { language?: string } = {}) {}

  async load(onProgress?: ProgressCb): Promise<void> {
    this.ctx = await createContext({ onBackend: (b) => console.info(`[asr-nemotron] backend: ${b}`) });
    const json = async (path: string) => JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path)));
    const bytes = (path: string) => fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path);

    const encMan = await json("nemotron/encoder-int8.manifest.json");
    const encBin = await bytes("nemotron/encoder-int8.bin");
    const decMan = await json("nemotron/decoder-fp32.manifest.json");
    const decBin = await bytes("nemotron/decoder-fp32.bin");
    this.vocab = await json("nemotron/vocab.json");
    const langs = await json("nemotron/languages.json");
    this.langMap = langs.promptDictionary ?? {};

    this.enc = loadParakeetEncoder(this.ctx, encBin, encMan, NEMO_CFG);
    // prompt_kernel weights are fp32 in the encoder blob's fp32 section.
    this.pk = loadPromptKernel(new Float32Array(encBin.buffer, encBin.byteOffset, encBin.byteLength >> 2), encMan);
    this.dec = loadNemotronDecoder(new Float32Array(decBin.buffer, decBin.byteOffset, decBin.byteLength / 4), decMan);
    onProgress?.({ file: WEIGHTS_REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult> {
    if (!this.enc || !this.dec || !this.pk || !this.vocab) throw new Error("NemotronEngine.load() not called");
    if (this.stream) throw new Error("a live stream is active — reset() before batch transcribe");
    const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
    const t0 = now();
    // Long-form: 4-minute segments with per-segment decode (the whole-clip
    // path builds quadratic [T,T] attention buffers — ~8GB at one hour). The
    // model is causal-chunked (left 56, right 3), so a boundary costs ~4.5s of
    // left context; text is concatenated across segments.
    const SEG = 4 * 60 * 16000;
    let melMs = 0;
    let encMs = 0;
    let decMs = 0;
    const texts: string[] = [];
    const langId = this.langMap[this.opts.language ?? "en-US"] ?? 0;
    for (let off = 0; off < audio.samples.length; off += SEG) {
      const slice = audio.samples.subarray(off, Math.min(off + SEG, audio.samples.length));
      if (slice.length < 800) break;
      const tm = now();
      const { features, length } = this.mel.process(slice);
      if (length === 0) continue;
      melMs += now() - tm;
      const te = now();
      const r = await parakeetEncode(this.ctx, this.enc, features, length);
      const conf = await this.ctx.download(r.framesGpu);
      const enc = applyPromptKernel(this.pk, conf, r.Tsub, langId);
      encMs += now() - te;
      const td = now();
      const { ids } = nemotronDecode(this.dec, enc, r.Tsub);
      decMs += now() - td;
      const text = ids
        .map((i: number) => this.vocab![i] ?? "")
        .filter((tk: string) => !tk.startsWith("<"))
        .join("")
        .replace(/▁/g, " ")
        .trim();
      if (text) texts.push(text);
      (this.ctx as any).trimPool?.();
    }
    return {
      text: texts.join(" "),
      metrics: { melMs: +melMs.toFixed(1), encodeMs: +encMs.toFixed(1), decodeMs: +decMs.toFixed(1), totalMs: +(now() - t0).toFixed(1) },
    };
  }

  async dispose(): Promise<void> {
    this.reset();
    this.ctx?.device?.destroy?.();
    this.ctx = this.enc = this.pk = this.dec = this.vocab = null;
  }
}
