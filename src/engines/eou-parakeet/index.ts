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

import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";
import { GpuContext, requestGpuDevice } from "../../gpu/compute.js";
import { loadParakeetEncoder, parakeetEncode } from "../asr-parakeet/raw-encoder.js";
import { loadEouDecoder, eouDecode } from "../asr-parakeet/raw-decoder-eou.js";
import { JsPreprocessor } from "../asr-nemotron/nemotron-mel.js";
import { makeEouTokenizer } from "./eou-decode.js";

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";
const VOCAB_REPO = "ysdede/parakeet-realtime-eou-120m-v1-onnx";
const FRAME_SEC = 0.08; // 10ms mel hop × 8× subsampling
// EOU streaming FastConformer config (see raw-encoder.js): causal subsampling pad,
// causal depthwise conv, chunked-causal attention (chunk 2, left context 70).
const EOU_CFG = { melBins: 128, subPad: { t: 2, b: 1, l: 2, r: 1 }, convCausal: true, attChunk: 2, attLeft: 70 };

export class ParakeetEouEngine implements AsrEngine {
  readonly id = "eou-parakeet";
  readonly label = "Parakeet EOU 120M";
  private ctx: any = null;
  private enc: any = null;
  private dec: any = null;
  private mel: JsPreprocessor | null = null;
  private tokenizer: ReturnType<typeof makeEouTokenizer> | null = null;

  async load(onProgress?: ProgressCb): Promise<void> {
    this.ctx = new GpuContext(await requestGpuDevice());
    const json = async (path: string) =>
      JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path)));
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
    const { features, length } = this.mel.process(audio.samples);
    if (length === 0) return { text: "", metrics: { melMs: 0, encodeMs: 0, decodeMs: 0, totalMs: 0 }, events: [] };
    const tMel = now();

    const r = await parakeetEncode(this.ctx, this.enc, features, length);
    const frames = await this.ctx.download(r.framesGpu);
    const tEnc = now();

    const { ids, events: evFrames } = eouDecode(this.dec, frames, r.Tsub);
    const events = evFrames.map((e: { type: string; frame: number }) => ({ type: e.type, time: +(e.frame * FRAME_SEC).toFixed(2) }));
    const tDec = now();

    return {
      text: this.tokenizer.decode(ids),
      metrics: {
        melMs: +(tMel - t0).toFixed(1),
        encodeMs: +(tEnc - tMel).toFixed(1),
        decodeMs: +(tDec - tEnc).toFixed(1),
        totalMs: +(tDec - t0).toFixed(1),
      },
      events,
    };
  }

  async dispose(): Promise<void> {
    this.ctx?.device?.destroy?.();
    this.ctx = this.enc = this.dec = this.mel = this.tokenizer = null;
  }
}
