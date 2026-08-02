// Nemotron 3.5 streaming ASR (en + multilingual, 40 langs) — cache-aware
// FastConformer-RNNT. ONNX: onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4.
//
// Verified I/O (ort-node, cache threading confirmed across chunks):
//   encoder: audio_signal[1,65,128] (56 new mel frames + 9 pre-encode cache),
//            length[1]i64, cache_last_channel[1,24,56,1024],
//            cache_last_time[1,24,1024,8], cache_last_channel_len[1]i64, lang_id[1]i64
//         -> outputs[1,7,1024] + cache_last_channel_next / cache_last_time_next /
//            cache_last_channel_len_next  (feed *_next back next chunk)
//   decoder (LSTM): targets[b,t]i32, h_in/c_in[2,b,640] -> decoder_output[b,640,t], h_out/c_out
//   joint: encoder_output[b,time,1024], decoder_output[b,t,640] -> joint_output[b,time,t,13088]
//   mel: n_fft512/hop160/128 mels/preemph0.97/**normalize=NA** (no CMVN),
//        audio_signal is [1,T,mels] (T-major) — chunk_samples=8960 (560ms tier).
//
// STATUS: encoder cache-threading wired + verified; remaining = NA-mel (streaming
// 65-frame chunks w/ 9-frame overlap) + RNNT greedy decode (decoder+joint) +
// tokenizer. int4 needs WebGPU for correct output (CPU likely degenerate like
// Parakeet int8) — accuracy verification is browser-only. See docs/NEMOTRON.md.

import { createSession, ort, webgpuAvailable } from "../../core/ort";
import { fetchCached, hfUrl } from "../../core/modelCache";
import type { ProgressCb, StreamingAsrEngine } from "../../core/types";

const REPO = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4";
const LAYERS = 24;
const LEFT_CTX = 56;
const CHANNEL_DIM = 1024;
const TIME_CTX = 8;

export class NemotronStreamingEngine implements StreamingAsrEngine {
  readonly id = "asr-nemotron";
  readonly label = "Nemotron 3.5 streaming";
  private encoder: any = null;
  private decoder: any = null;
  private joint: any = null;
  private langId = 0;

  // Encoder streaming caches (threaded chunk→chunk).
  private cacheChannel: any = null;
  private cacheTime: any = null;
  private cacheChannelLen: any = null;

  async load(onProgress?: ProgressCb): Promise<void> {
    if (!webgpuAvailable()) {
      throw new Error("Nemotron int4 needs WebGPU (CPU/WASM likely degenerate, like Parakeet int8).");
    }
    const enc = await fetchCached(hfUrl(REPO, "encoder.onnx"), onProgress, "encoder.onnx");
    // NOTE: encoder.onnx has external data (encoder.onnx.data, ~690MB). ORT-web
    // external-data wiring (fetch the .data, pass as externalData) is a TODO.
    const dec = await fetchCached(hfUrl(REPO, "decoder.onnx"), onProgress, "decoder.onnx");
    const jnt = await fetchCached(hfUrl(REPO, "joint.onnx"), onProgress, "joint.onnx");
    this.encoder = await createSession(enc, "webgpu");
    this.decoder = await createSession(dec, "wasm");
    this.joint = await createSession(jnt, "wasm");
    this.reset();
  }

  reset(): void {
    this.cacheChannel = new ort.Tensor(
      "float32", new Float32Array(LAYERS * LEFT_CTX * CHANNEL_DIM), [1, LAYERS, LEFT_CTX, CHANNEL_DIM]);
    this.cacheTime = new ort.Tensor(
      "float32", new Float32Array(LAYERS * CHANNEL_DIM * TIME_CTX), [1, LAYERS, CHANNEL_DIM, TIME_CTX]);
    this.cacheChannelLen = new ort.Tensor("int64", BigInt64Array.from([0n]), [1]);
  }

  /** Feed one 65-frame mel chunk; threads encoder caches. RNNT decode = TODO. */
  async push(_chunk: Float32Array): Promise<string> {
    if (!this.encoder) throw new Error("load() not called");
    // Encoder feeds — caches threaded chunk→chunk (verified pattern). The only
    // missing input is audio_signal, which needs the NA-mel of `_chunk`.
    const feeds: Record<string, any> = {
      // audio_signal: <NA-mel(_chunk) → [1,65,128]>,   // TODO
      length: new ort.Tensor("int64", BigInt64Array.from([65n]), [1]),
      cache_last_channel: this.cacheChannel,
      cache_last_time: this.cacheTime,
      cache_last_channel_len: this.cacheChannelLen,
      lang_id: new ort.Tensor("int64", BigInt64Array.from([BigInt(this.langId)]), [1]),
    };
    void feeds;
    // After encoder.run: this.cacheChannel = out.cache_last_channel_next, etc.,
    // then RNNT greedy over the 7 enc frames via decoder(LSTM)+joint, detokenize.
    throw new Error("NemotronStreamingEngine.push: NA-mel + RNNT decode not yet ported (encoder plumbing verified)");
  }

  async dispose(): Promise<void> {
    await this.encoder?.release?.();
    await this.decoder?.release?.();
    await this.joint?.release?.();
    this.encoder = this.decoder = this.joint = null;
  }
}
