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
import type { AsrEngine, AsrResult, AudioData, ProgressCb, StreamingAsrEngine, TranscribeOpts } from "../../core/types.js";
import { makeTranscribeProgress } from "../../core/progress.js";
import { createContext } from "../../gpu/context.js";
import { loadParakeetEncoder } from "../asr-parakeet/raw-encoder.js";
import { loadNemotronDecoder, loadPromptKernel, loadNemoWasmDecoder, nemoWasmDecodeCont, nemoWasmReset } from "./raw-decoder-nemotron.js";
import { createEncodeStream, encodeStreamPush, encodeStreamFlush, disposeEncodeStream } from "../asr-parakeet/streaming-encoder.js";
import { StreamingMel } from "./streaming-mel.js";
import { tokensToWords } from "../../core/captions.js";

const wasmUrl = new URL("../asr-parakeet/parakeet-decoder.wasm", import.meta.url); // cross-bundler asset URL

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";
// Nemotron streaming FastConformer config (see raw-encoder.js): causal subsampling
// pad, causal depthwise conv, cache-aware attention (chunk 4, left 56, right 3).
const NEMO_CFG = { melBins: 128, subPad: { t: 2, b: 1, l: 2, r: 1 }, convCausal: true, attChunk: 4, attLeft: 56, attRight: 3 };

export class NemotronEngine implements AsrEngine, StreamingAsrEngine {
  readonly id = "asr-nemotron";
  readonly label = "Nemotron 3.5 multilingual";
  private ctx: any = null;
  private enc: any = null;
  private pk: any = null;
  private dec: any = null;
  private vocab: Record<string, string> | null = null; // nemotron/vocab.json is an OBJECT keyed by id string, not an array
  private langMap: Record<string, number> = {};
  private wdec: any = null; // wasm-SIMD RNNT decoder (stream state in its instance)
  private projW: any = null; // joint enc projection 1024→640, GPU-side
  private projB: any = null;
  private pk0c: any = null; // prompt-kernel MLP on GPU (lang one-hot → folded bias)
  private pk2w: any = null;
  private pk2b: any = null;
  private langBiasCache = new Map<number, any>();
  private stream: { mel: StreamingMel; encSt: any; ids: number[]; idTimes: number[]; subT: number; finished: boolean; broken: boolean } | null = null;
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
    if (!this.enc || !this.wdec || !this.vocab) throw new Error("NemotronEngine.load() not called");
    if (this.stream?.finished) throw new Error("finish() already called — reset() to start a new stream");
    if (this.stream?.broken) throw new Error("stream broken by an earlier push failure — reset()");
    if (!this.stream) {
      const langId = this.langMap[this.opts.language ?? "en-US"] ?? 0;
      nemoWasmReset(this.wdec);
      this.stream = {
        mel: new StreamingMel(128),
        encSt: createEncodeStream(this.ctx, this.enc, { post: this.postFor(langId), lookaheadChunks: 4 }),
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
      this.ctx?.trimPool();
    }
    this.stream = null;
  }

  private consume(frames: Float32Array): void {
    const st = this.stream!;
    const n = frames.length / 640; // frames arrive prompt-kerneled + projected [n, 640]
    const r = nemoWasmDecodeCont(this.wdec, frames, n);
    st.ids.push(...r.ids);
    st.idTimes.push(...r.idFrames.map((f) => (st.subT + f) * 0.08));
    st.subT += n;
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
    const decF32 = new Float32Array(decBin.buffer, decBin.byteOffset, decBin.byteLength / 4);
    this.dec = loadNemotronDecoder(decF32, decMan);
    // Fast path (task #29): prompt-kernel MLP + 1024→640 projection move to
    // the GPU (were scalar JS per frame — ~400 GFLOP/hour) and decode moves
    // to the wasm-SIMD crate (13088-wide scalar-JS joint was minutes/hour).
    this.wdec = await loadNemoWasmDecoder(await (await fetch(wasmUrl)).arrayBuffer(), decF32, decMan);
    const up2 = (d: Float32Array, r: number, c: number) => (this.ctx.uploadTileMajorF16 ? this.ctx.uploadTileMajorF16(d, r, c) : this.ctx.upload(d, r, c));
    this.pk0c = up2(this.pk.pk0w.subarray(0, 1024 * 2048), 1024, 2048);
    this.pk2w = up2(this.pk.pk2w, 2048, 1024);
    this.pk2b = this.ctx.upload(this.pk.pk2b.slice(), 1, 1024);
    this.projW = this.ctx.upload(this.dec.encW.slice(), 1024, 640);
    this.projB = this.ctx.upload(this.dec.encB.slice(), 1, 640);
    onProgress?.({ file: WEIGHTS_REPO, loaded: 1, total: 1, fraction: 1 });
  }

  // one-hot(lang) @ pk0w == selecting row 1024+langId — fold into the bias.
  private langBias(langId: number): any {
    let b = this.langBiasCache.get(langId);
    if (!b) {
      const fold = new Float32Array(2048);
      const row = this.pk.pk0w.subarray((1024 + langId) * 2048, (1024 + langId + 1) * 2048);
      for (let i = 0; i < 2048; i++) fold[i] = this.pk.pk0b[i] + row[i];
      b = this.ctx.upload(fold, 1, 2048);
      this.langBiasCache.set(langId, b);
    }
    return b;
  }

  /** GPU tail recorded into each chunk batch: prompt-kernel MLP (+folded
   * language bias) then the 1024→640 joint projection. */
  private postFor(langId: number): (ctx: any, x: any) => any {
    const lb = this.langBias(langId);
    return (ctx, x) =>
      ctx.matmul(ctx.matmul(ctx.matmul(x, this.pk0c, { bias: lb, act: "relu" }), this.pk2w, { bias: this.pk2b }), this.projW, { bias: this.projB });
  }

  transcribe(audio: AudioData, opts?: TranscribeOpts): Promise<AsrResult> {
    // Serialized on the same chain as push/finish — a queued-but-unexecuted
    // push would otherwise race the segment loop (TOCTOU on this.stream) and
    // trimPool() mid-stream violates the drained-queue contract.
    return this.serialize(() => this.transcribeInner(audio, opts));
  }

  private async transcribeInner(audio: AudioData, opts?: TranscribeOpts): Promise<AsrResult> {
    if (!this.enc || !this.wdec || !this.pk || !this.vocab) throw new Error("NemotronEngine.load() not called");
    if (this.stream) throw new Error("a live stream is active — reset() before batch transcribe");
    const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
    const t0 = now();
    // Batch IS a big-chunk stream (the EOU recipe, task #29): linear-cost
    // cache-carrying encode with provisional-tail lookahead (attRight=3), GPU
    // prompt kernel + projection riding each chunk batch, one continuous wasm
    // RNNT decode — replaces 4-min quadratic segments + per-frame scalar-JS
    // prompt kernel + scalar-JS 13088-wide joint (minutes per hour).
    const langId = this.langMap[this.opts.language ?? "en-US"] ?? 0;
    const mel = new StreamingMel(128);
    const encSt = createEncodeStream(this.ctx, this.enc, { post: this.postFor(langId), lookaheadChunks: 4 });
    nemoWasmReset(this.wdec);
    let melMs = 0;
    let encMs = 0;
    let decMs = 0;
    let subT = 0;
    const ids: number[] = [];
    const idTimes: number[] = [];
    const consumeB = (frames: Float32Array) => {
      const td = now();
      const n = frames.length / 640;
      const r = nemoWasmDecodeCont(this.wdec, frames, n);
      decMs += now() - td;
      ids.push(...r.ids);
      idTimes.push(...r.idFrames.map((f) => (subT + f) * 0.08));
      subT += n;
    };
    const SLICE = 240 * 16000;
    const BATCH_CHUNK = 768;
    const progress = makeTranscribeProgress(audio.samples.length / 16000, opts?.onProgress);
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
        progress?.update(Math.min(off + SLICE, audio.samples.length) / 16000);
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
    progress?.done();
    return {
      text: this.idsToText(ids),
      segments: tokensToWords(ids, idTimes, this.vocab as Record<number, string>, (id) => (this.vocab![id] ?? "<").startsWith("<")),
      metrics: { melMs: +melMs.toFixed(1), encodeMs: +encMs.toFixed(1), decodeMs: +decMs.toFixed(1), totalMs: +(now() - t0).toFixed(1) },
    };
  }

  async dispose(): Promise<void> {
    this.reset();
    this.ctx?.destroy();
    this.ctx = this.enc = this.pk = this.dec = this.vocab = null;
  }
}
