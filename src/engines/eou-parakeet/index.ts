// Parakeet EOU 120M — end-of-utterance detection + transcription, fully ORT-free.
//
// NVIDIA's `parakeet_realtime_eou_120m-v1`: a streaming FastConformer RNNT with two
// control tokens — <EOU> (end of utterance) and <EOB> — so a voice agent can tell
// when the user finished speaking. Decoded offline (whole clip); the transcript plus
// the <EOU>/<EOB> timestamps come back together.
//
// Everything is hand-written on raw WebGPU + JS (no onnxruntime):
//   • Mel: JsPreprocessor NA log-mel (no CMVN) — this model wants un-normalized mel.
//   • Encoder: the shared FastConformer (raw-encoder.js) with EOU streaming config —
//     causal subsampling pad, causal depthwise conv, conv-module LayerNorm, and a
//     cache-aware chunked attention mask (chunk 2, left context 70). fp16 weights
//     (int8 degrades this 120M RNNT). Runs on WebGPU.
//   • Decoder+joint: 1-layer LSTM RNNT (raw-decoder-eou.js), small enough for plain JS.
//     The exported joint prepends a zero SOS timestep per call (2-step LSTM).
// Full raw path == ORT reference transcript; encoder maxΔ 4.4e-2 (fp16) vs ORT.

import { fetchCached, hfUrl } from "../../core/modelCache.js";
import type { AsrEngine, AsrResult, AudioData, ProgressCb, StreamingAsrEngine } from "../../core/types.js";
import { createContext } from "../../gpu/context.js";
import { loadParakeetEncoder } from "../asr-parakeet/raw-encoder.js";
import { loadEouDecoder, loadEouWasmDecoder, eouWasmDecodeCont, eouWasmReset } from "../asr-parakeet/raw-decoder-eou.js";
import { createEncodeStream, encodeStreamPush, encodeStreamFlush, disposeEncodeStream } from "../asr-parakeet/streaming-encoder.js";
import { JsPreprocessor } from "../asr-nemotron/nemotron-mel.js";
import { StreamingMel } from "../asr-nemotron/streaming-mel.js";
import { tokensToWords } from "../../core/captions.js";
import { makeEouTokenizer } from "./eou-decode.js";
import { EOU_CFG } from "./config.js";

const wasmUrl = new URL("../asr-parakeet/parakeet-decoder.wasm", import.meta.url); // cross-bundler asset URL

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";
const VOCAB_REPO = "ysdede/parakeet-realtime-eou-120m-v1-onnx";
const FRAME_SEC = 0.08; // 10ms mel hop × 8× subsampling
// EOU streaming FastConformer config (see raw-encoder.js): causal subsampling pad,
// causal depthwise conv, chunked-causal attention (chunk 2, left context 70).

const PROJ_D = 640; // joint-projected frame width (512→640 GEMM rides the encode batch)
// Batch-transcribe chunk size (subsampled frames ≈ 61s audio). Swept on the
// 1-hour bench (dawn, M-series): 384→173×, 512→207×, 640→229×, 768→254×,
// 896→208×, 1536→149× — 768 is a solid local optimum (GEMM occupancy vs
// attention-rectangle size), 1.5× the old quadratic-segment path's 167×.
const BATCH_CHUNK = 768;

export class ParakeetEouEngine implements AsrEngine, StreamingAsrEngine {
  readonly id = "eou-parakeet";
  readonly label = "Parakeet EOU 120M";
  private ctx: any = null;
  private enc: any = null;
  private dec: any = null;
  private mel: JsPreprocessor | null = null;
  private tokenizer: ReturnType<typeof makeEouTokenizer> | null = null;
  private wdec: any = null; // wasm-SIMD decoder (holds the stream's RNNT state)
  private projW: any = null; // joint enc projection 512→640, run GPU-side pre-download
  private projB: any = null;
  private stream: {
    mel: StreamingMel;
    encSt: any;
    ids: number[];
    events: { type: string; time: number }[];
    subT: number;
    finished: boolean;
    broken: boolean;
  } | null = null;
  // push()/finish() serialize through this chain: the SDK doesn't force callers
  // to await one push before the next, and interleaved encodeStreamPush loops
  // on one encSt corrupt the FIFO/caches.
  private op: Promise<unknown> = Promise.resolve();

  private serialize<T>(fn: () => Promise<T>): Promise<T> {
    const r = this.op.then(fn, fn);
    this.op = r.then(
      () => undefined,
      () => undefined,
    );
    return r;
  }

  async load(onProgress?: ProgressCb): Promise<void> {
    this.ctx = await createContext({ onBackend: (b) => console.info(`[eou-parakeet] backend: ${b}`) });
    const json = async (path: string) => JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path)));
    const bytes = (path: string) => fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path);

    const encMan = await json("eou/encoder-fp16.manifest.json");
    const encBin = await bytes("eou/encoder-fp16.bin");
    const decMan = await json("eou/decoder-fp32.manifest.json");
    const decBin = await bytes("eou/decoder-fp32.bin");
    const vocab = new TextDecoder().decode(await fetchCached(hfUrl(VOCAB_REPO, "vocab.txt"), onProgress, "vocab.txt"));

    this.enc = loadParakeetEncoder(this.ctx, encBin, encMan, EOU_CFG);
    const decF32 = new Float32Array(decBin.buffer, decBin.byteOffset, decBin.byteLength / 4);
    this.dec = loadEouDecoder(decF32, decMan);
    // Decode runs in the wasm-SIMD crate (JS loop is the hour-scale bottleneck);
    // the joint's 512→640 enc projection rides the encode batch as one GEMM.
    this.wdec = await loadEouWasmDecoder(await (await fetch(wasmUrl)).arrayBuffer(), decF32, decMan);
    this.projW = this.ctx.upload(this.dec.encW.slice(), 512, 640);
    this.projB = this.ctx.upload(this.dec.encB.slice(), 1, 640);
    this.mel = new JsPreprocessor({ nMels: 128 });
    this.tokenizer = makeEouTokenizer(vocab);
    onProgress?.({ file: WEIGHTS_REPO, loaded: 1, total: 1, fraction: 1 });
  }

  transcribe(audio: AudioData): Promise<AsrResult & { events?: { type: string; time: number }[] }> {
    return this.serialize(() => this.transcribeInner(audio)) as Promise<AsrResult & { events?: { type: string; time: number }[] }>;
  }

  private async transcribeInner(audio: AudioData): Promise<AsrResult & { events?: { type: string; time: number }[] }> {
    if (!this.enc || !this.wdec || !this.mel || !this.tokenizer) throw new Error("ParakeetEouEngine.load() not called");
    if (this.stream) throw new Error("a live stream is active — reset() before batch transcribe (shared decoder state)");
    const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
    const t0 = now();
    // Batch IS a big-chunk stream. The old path materialized full [T,T]
    // chunked-attention buffers per 4-minute segment (~288MB of scores each)
    // even though the model only attends 70 frames back; the streaming encoder
    // computes the same function at linear cost (bit-exact — gate:
    // scripts/streaming-encode-check.mjs). Decoder state now carries across
    // the WHOLE clip (no per-segment reset — whole-clip LSTM continuity).
    const mel = new StreamingMel(128);
    const encSt = createEncodeStream(this.ctx, this.enc, { proj: { w: this.projW, b: this.projB } });
    eouWasmReset(this.wdec);
    let melMs = 0;
    let encMs = 0;
    let decMs = 0;
    let subT = 0;
    const ids: number[] = [];
    const idTimes: number[] = [];
    const events: { type: string; time: number }[] = [];
    const consume = (frames: Float32Array) => {
      const td = now();
      const n = frames.length / PROJ_D;
      const r = eouWasmDecodeCont(this.wdec, frames, n, subT);
      decMs += now() - td;
      ids.push(...r.ids);
      idTimes.push(...r.idFrames.map((f: number) => f * FRAME_SEC));
      for (const e of r.events) events.push({ type: e.type, time: +(e.frame * FRAME_SEC).toFixed(2) });
      subT += n;
    };
    const SLICE = 240 * 16000; // feed 4-min slices so chunk passes reach BATCH_CHUNK
    try {
      for (let off = 0; off < audio.samples.length; off += SLICE) {
        const tm = now();
        const { data, count } = mel.push(audio.samples.subarray(off, Math.min(off + SLICE, audio.samples.length)));
        melMs += now() - tm;
        if (data && count > 0) {
          const te = now();
          const out = await encodeStreamPush(this.ctx, encSt, data, count, { maxChunk: BATCH_CHUNK });
          encMs += now() - te;
          if (out) consume(out);
        }
      }
      const tm = now();
      const fl = mel.flush();
      melMs += now() - tm;
      const te = now();
      if (fl.data && fl.count > 0) {
        const out = await encodeStreamPush(this.ctx, encSt, fl.data, fl.count, { maxChunk: BATCH_CHUNK });
        if (out) consume(out);
      }
      const tail = await encodeStreamFlush(this.ctx, encSt);
      encMs += now() - te;
      if (tail) consume(tail);
    } finally {
      disposeEncodeStream(this.ctx, encSt);
      (this.ctx as any).trimPool?.();
    }
    return {
      text: this.tokenizer.decode(ids),
      segments: tokensToWords(ids, idTimes, this.tokenizer.id2token),
      metrics: { melMs: +melMs.toFixed(1), encodeMs: +encMs.toFixed(1), decodeMs: +decMs.toFixed(1), totalMs: +(now() - t0).toFixed(1) },
      events,
    };
  }

  // ── true streaming (docs/STREAMING.md; gate: scripts/streaming-encode-check.mjs) ──
  // Carries conformer K/V + conv caches and the RNNT LSTM state chunk-to-chunk;
  // bit-exact with the offline chunked-causal path, so push() at mic cadence
  // costs one tiny chunk pass instead of a rolling re-decode.

  /** Feed 16 kHz samples; returns the text emitted so far (plus buffered state). */
  push(chunk: Float32Array): Promise<string> {
    return this.serialize(() => this.pushInner(chunk));
  }

  private async pushInner(chunk: Float32Array): Promise<string> {
    if (!this.enc || !this.wdec || !this.tokenizer) throw new Error("ParakeetEouEngine.load() not called");
    if (this.stream?.finished) throw new Error("finish() already called — reset() to start a new stream");
    if (this.stream?.broken) throw new Error("stream broken by an earlier push failure — reset() to start a new stream");
    if (!this.stream) {
      eouWasmReset(this.wdec); // RNNT state lives in the wasm instance
      this.stream = {
        mel: new StreamingMel(128),
        encSt: createEncodeStream(this.ctx, this.enc, { proj: { w: this.projW, b: this.projB } }),
        ids: [],
        events: [],
        subT: 0,
        finished: false,
        broken: false,
      };
    }
    const s = this.stream;
    // The mel FIFO absorbs the chunk BEFORE the encode that can throw — after a
    // failure the stream state is undefined (re-sending the chunk would feed
    // the mel twice). Poison it: the caller must reset() for a new utterance.
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
    return this.tokenizer.decode(s.ids);
  }

  /** Flush the right-padded tail and return the final utterance text. */
  finish(): Promise<string> {
    return this.serialize(() => this.finishInner());
  }

  private async finishInner(): Promise<string> {
    if (!this.stream || !this.tokenizer) return "";
    const s = this.stream;
    if (s.finished || s.broken) return this.tokenizer.decode(s.ids);
    s.finished = true;
    const { data, count } = s.mel.flush();
    if (data && count > 0) {
      const out = await encodeStreamPush(this.ctx, s.encSt, data, count, { maxChunk: 16 });
      if (out) this.consume(out);
    }
    const tail = await encodeStreamFlush(this.ctx, s.encSt);
    if (tail) this.consume(tail);
    return this.tokenizer.decode(s.ids);
  }

  /** <EOU>/<EOB> events seen so far on the current stream (seconds). */
  get streamEvents(): { type: string; time: number }[] {
    return this.stream?.events ?? [];
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
    const n = frames.length / PROJ_D; // stream frames arrive pre-projected [n, 640]
    const { ids, events } = eouWasmDecodeCont(this.wdec, frames, n, s.subT);
    s.ids.push(...ids);
    for (const e of events as { type: string; frame: number }[]) s.events.push({ type: e.type, time: +(e.frame * FRAME_SEC).toFixed(2) });
    s.subT += n;
  }

  async dispose(): Promise<void> {
    this.reset();
    this.ctx?.device?.destroy?.();
    this.ctx = this.enc = this.dec = this.wdec = this.projW = this.projB = this.mel = this.tokenizer = null;
  }
}
