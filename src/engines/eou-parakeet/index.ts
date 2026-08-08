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
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types.js";
import { createContext } from "../../gpu/context.js";
import { loadParakeetEncoder, parakeetEncode } from "../asr-parakeet/raw-encoder.js";
import { loadEouDecoder, eouDecode } from "../asr-parakeet/raw-decoder-eou.js";
import { JsPreprocessor } from "../asr-nemotron/nemotron-mel.js";
import { makeEouTokenizer } from "./eou-decode.js";
import { EOU_CFG } from "./config.js";

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";
const VOCAB_REPO = "ysdede/parakeet-realtime-eou-120m-v1-onnx";
const FRAME_SEC = 0.08; // 10ms mel hop × 8× subsampling
// EOU streaming FastConformer config (see raw-encoder.js): causal subsampling pad,
// causal depthwise conv, chunked-causal attention (chunk 2, left context 70).

export class ParakeetEouEngine implements AsrEngine {
  readonly id = "eou-parakeet";
  readonly label = "Parakeet EOU 120M";
  private ctx: any = null;
  private enc: any = null;
  private dec: any = null;
  private mel: JsPreprocessor | null = null;
  private tokenizer: ReturnType<typeof makeEouTokenizer> | null = null;

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

  async dispose(): Promise<void> {
    this.ctx?.device?.destroy?.();
    this.ctx = this.enc = this.dec = this.mel = this.tokenizer = null;
  }
}
