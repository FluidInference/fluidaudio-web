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
import { loadParakeetEncoder, parakeetEncode } from "../asr-parakeet/raw-encoder.js";
import { loadEouDecoder, eouDecode, createEouStream, eouDecodeCont } from "../asr-parakeet/raw-decoder-eou.js";
import { createEncodeStream, encodeStreamPush, encodeStreamFlush, disposeEncodeStream } from "../asr-parakeet/streaming-encoder.js";
import { JsPreprocessor } from "../asr-nemotron/nemotron-mel.js";
import { StreamingMel } from "../asr-nemotron/streaming-mel.js";
import { makeEouTokenizer } from "./eou-decode.js";
import { EOU_CFG } from "./config.js";

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";
const VOCAB_REPO = "ysdede/parakeet-realtime-eou-120m-v1-onnx";
const FRAME_SEC = 0.08; // 10ms mel hop × 8× subsampling
// EOU streaming FastConformer config (see raw-encoder.js): causal subsampling pad,
// causal depthwise conv, chunked-causal attention (chunk 2, left context 70).

const ENC_D = 512;

export class ParakeetEouEngine implements AsrEngine, StreamingAsrEngine {
  readonly id = "eou-parakeet";
  readonly label = "Parakeet EOU 120M";
  private ctx: any = null;
  private enc: any = null;
  private dec: any = null;
  private mel: JsPreprocessor | null = null;
  private tokenizer: ReturnType<typeof makeEouTokenizer> | null = null;
  private stream: {
    mel: StreamingMel;
    encSt: any;
    decSt: any;
    ids: number[];
    events: { type: string; time: number }[];
    subT: number;
    finished: boolean;
  } | null = null;

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
    this.dec = loadEouDecoder(new Float32Array(decBin.buffer, decBin.byteOffset, decBin.byteLength / 4), decMan);
    this.mel = new JsPreprocessor({ nMels: 128 });
    this.tokenizer = makeEouTokenizer(vocab);
    onProgress?.({ file: WEIGHTS_REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult & { events?: { type: string; time: number }[] }> {
    if (!this.enc || !this.dec || !this.mel || !this.tokenizer) throw new Error("ParakeetEouEngine.load() not called");
    const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
    const t0 = now();
    // Long-form: process in 4-minute segments, decoder reset between them.
    // The whole-clip path builds full [T,T] chunked-attention buffers
    // (quadratic — a 1-hour clip attempts ~8GB); the model itself is fully
    // CAUSAL (left context 70 frames ≈ 5.6s), so segmenting only loses that
    // context at each boundary. Event times are offset per segment; text is
    // concatenated. (True chunk-level streaming: docs/STREAMING.md.)
    const SEG_SEC = 4 * 60;
    const SEG = SEG_SEC * 16000;
    let melMs = 0;
    let encMs = 0;
    let decMs = 0;
    const texts: string[] = [];
    const events: { type: string; time: number }[] = [];
    for (let off = 0; off < audio.samples.length; off += SEG) {
      const slice = audio.samples.subarray(off, Math.min(off + SEG, audio.samples.length));
      if (slice.length < 800) break;
      const tm = now();
      const { features, length } = this.mel.process(slice);
      if (length === 0) continue;
      melMs += now() - tm;
      const te = now();
      const r = await parakeetEncode(this.ctx, this.enc, features, length);
      const frames = await this.ctx.download(r.framesGpu);
      encMs += now() - te;
      const td = now();
      // eouDecode resets decoder state per call — exactly the per-segment
      // "decode reset" semantics we want at a segment boundary.
      const { ids, events: evFrames } = eouDecode(this.dec, frames, r.Tsub);
      decMs += now() - td;
      const offSec = off / 16000;
      for (const e of evFrames as { type: string; frame: number }[]) {
        events.push({ type: e.type, time: +(offSec + e.frame * FRAME_SEC).toFixed(2) });
      }
      const text = this.tokenizer.decode(ids);
      if (text) texts.push(text);
      (this.ctx as any).trimPool?.();
    }
    return {
      text: texts.join(" "),
      metrics: { melMs: +melMs.toFixed(1), encodeMs: +encMs.toFixed(1), decodeMs: +decMs.toFixed(1), totalMs: +(now() - t0).toFixed(1) },
      events,
    };
  }

  // ── true streaming (docs/STREAMING.md; gate: scripts/streaming-encode-check.mjs) ──
  // Carries conformer K/V + conv caches and the RNNT LSTM state chunk-to-chunk;
  // bit-exact with the offline chunked-causal path, so push() at mic cadence
  // costs one tiny chunk pass instead of a rolling re-decode.

  /** Feed 16 kHz samples; returns the text emitted so far (plus buffered state). */
  async push(chunk: Float32Array): Promise<string> {
    if (!this.enc || !this.dec || !this.tokenizer) throw new Error("ParakeetEouEngine.load() not called");
    if (this.stream?.finished) throw new Error("finish() already called — reset() to start a new stream");
    if (!this.stream) {
      this.stream = {
        mel: new StreamingMel(128),
        encSt: createEncodeStream(this.ctx, this.enc),
        decSt: createEouStream(this.dec),
        ids: [],
        events: [],
        subT: 0,
        finished: false,
      };
    }
    const s = this.stream;
    const { data, count } = s.mel.push(chunk);
    if (data && count > 0) {
      const out = await encodeStreamPush(this.ctx, s.encSt, data, count, { maxChunk: 16 });
      if (out) this.consume(out);
    }
    return this.tokenizer.decode(s.ids);
  }

  /** Flush the right-padded tail and return the final utterance text. */
  async finish(): Promise<string> {
    if (!this.stream || !this.tokenizer) return "";
    const s = this.stream;
    if (s.finished) return this.tokenizer.decode(s.ids);
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
    const n = frames.length / ENC_D;
    const { ids, events } = eouDecodeCont(this.dec, s.decSt, frames, n, s.subT);
    s.ids.push(...ids);
    for (const e of events as { type: string; frame: number }[]) s.events.push({ type: e.type, time: +(e.frame * FRAME_SEC).toFixed(2) });
    s.subT += n;
  }

  async dispose(): Promise<void> {
    this.reset();
    this.ctx?.device?.destroy?.();
    this.ctx = this.enc = this.dec = this.mel = this.tokenizer = null;
  }
}
