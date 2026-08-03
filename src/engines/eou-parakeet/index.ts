// Parakeet EOU 120M — end-of-utterance detection + transcription.
//
// NVIDIA's `parakeet_realtime_eou_120m-v1`, exported to ONNX by the asrjs project
// (ysdede/parakeet-realtime-eou-120m-v1-onnx). It's a streaming FastConformer
// RNNT with two extra control tokens — <EOU> (end of utterance) and <EOB> — so a
// voice agent can tell when the user has finished speaking. We decode it offline
// (whole clip) here; the transcript plus the <EOU>/<EOB> timestamps come back
// together.
//
// Two things that cost debugging (see eou-decode.js / docs/EOU.md):
//   • Mel: this model wants UN-normalized (NA) log-mel — the Nemotron frontend,
//     NOT Parakeet's per-feature CMVN. Wrong normalization → the encoder emits
//     content-free frames and the joint predicts blank on every step.
//   • The fused decoder_joint returns a [1,1,2,1027] grid; the last 1027 values
//     are the logits for this step (blank id 1026). Single-layer LSTM state.
//
// The fp32 encoder (~460 MB) decodes correctly on WASM *and* WebGPU (unlike
// Parakeet's int8, which collapses on WASM), so WebGPU is preferred but not
// required. The fused decoder + NA mel run on WASM.

import { createSession, ort, webgpuAvailable } from "../../core/ort";
import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";
import { JsPreprocessor } from "../asr-nemotron/nemotron-mel.js";
import { eouTranscribe, makeEouTokenizer } from "./eou-decode.js";

const REPO = "ysdede/parakeet-realtime-eou-120m-v1-onnx";
const ENCODER = "encoder-model.onnx"; // fp32; int8/fp16 also available
const DECODER = "decoder_joint-model.onnx"; // fused decoder+joint, fp32
const VOCAB = "vocab.txt";

export class ParakeetEouEngine implements AsrEngine {
  readonly id = "eou-parakeet";
  readonly label = "Parakeet EOU 120M";
  private encoder: any = null;
  private decoder: any = null;
  private tokenizer: ReturnType<typeof makeEouTokenizer> | null = null;
  private preprocessor: JsPreprocessor | null = null;

  async load(onProgress?: ProgressCb): Promise<void> {
    const encBytes = await fetchCached(hfUrl(REPO, ENCODER), onProgress, ENCODER);
    const decBytes = await fetchCached(hfUrl(REPO, DECODER), onProgress, DECODER);
    const vocabText = new TextDecoder().decode(await fetchCached(hfUrl(REPO, VOCAB), onProgress, VOCAB));

    // fp32 encoder: WebGPU when available (faster), else WASM (still correct).
    this.encoder = await createSession(encBytes, webgpuAvailable() ? "webgpu" : "wasm");
    this.decoder = await createSession(decBytes, "wasm");
    this.tokenizer = makeEouTokenizer(vocabText);
    this.preprocessor = new JsPreprocessor({ nMels: 128 });
    onProgress?.({ file: REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult & { events?: { type: string; time: number }[] }> {
    if (!this.encoder || !this.decoder || !this.tokenizer || !this.preprocessor) {
      throw new Error("ParakeetEouEngine.load() not called");
    }
    const { text, events, metrics } = await eouTranscribe({
      ort,
      encoder: this.encoder,
      decoder: this.decoder,
      preprocessor: this.preprocessor,
      tokenizer: this.tokenizer,
      audio: audio.samples,
    });
    return { text, metrics, events };
  }

  async dispose(): Promise<void> {
    await this.encoder?.release?.();
    await this.decoder?.release?.();
    this.encoder = this.decoder = this.tokenizer = this.preprocessor = null;
  }
}
