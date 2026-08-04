// Nemotron 3.5 streaming ASR (multilingual, cache-aware FastConformer-RNNT).
//
// Uses the soniqo FP16 export, which is purpose-built to run under
// onnxruntime-web's WebGPU EP (the int4 export can't — no int kernels on WebGPU;
// soniqo also rewrote the 24-input Concat into a ≤6-input tree to stay under the
// 8-storage-buffer WebGPU limit). So the heavy encoder runs on the **GPU** (fast,
// non-blocking); the tiny LSTM decoder + joint run on WASM. mel is NA log-mel in
// JS. Verified headless: exact transcript. See docs/NEMOTRON.md.

import { ort, webgpuAvailable } from "../../core/ort";
import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";
import { JsPreprocessor } from "./nemotron-mel.js";
import { soniqoTranscribe, makeSoniqoTokenizer, soniqoLangPrompt } from "./nemotron-soniqo.js";

const REPO = "soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16";

async function createWithData(name: string, ep: "webgpu" | "wasm", onProgress?: ProgressCb) {
  const model = await fetchCached(hfUrl(REPO, `${name}.onnx`), onProgress, `${name}.onnx`);
  const data = await fetchCached(hfUrl(REPO, `${name}.onnx.data`), onProgress, `${name}.onnx.data`);
  const eps = ep === "webgpu" && webgpuAvailable() ? ["webgpu", "wasm"] : ["wasm"];
  return ort.InferenceSession.create(model, {
    executionProviders: eps as any,
    graphOptimizationLevel: "all",
    externalData: [{ path: `${name}.onnx.data`, data }] as any,
  });
}

export class NemotronEngine implements AsrEngine {
  readonly id = "asr-nemotron";
  readonly label = "Nemotron 3.5 streaming";
  private encoder: any = null;
  private decoder: any = null;
  private joint: any = null;
  private preprocessor = new JsPreprocessor({ nMels: 128 });
  private tokenizer: any = null;
  private languages: any = null;

  /** @param opts.language e.g. "en-US" / "de" / "zh-CN" / "ja-JP" (default en-US). */
  constructor(private opts: { language?: string } = {}) {}

  async load(onProgress?: ProgressCb): Promise<void> {
    // fp16 encoder on WebGPU (the heavy part); LSTM decoder + joint on WASM (tiny).
    this.encoder = await createWithData("encoder", "webgpu", onProgress);
    this.decoder = await createWithData("decoder", "wasm", onProgress);
    this.joint = await createWithData("joint", "wasm", onProgress);
    this.tokenizer = makeSoniqoTokenizer(JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(REPO, "vocab.json"), onProgress, "vocab.json"))));
    this.languages = JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(REPO, "languages.json"), onProgress, "languages.json")));
    onProgress?.({ file: REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult> {
    if (!this.encoder || !this.decoder || !this.joint || !this.tokenizer) {
      throw new Error("NemotronEngine.load() not called");
    }
    const { text } = await soniqoTranscribe({
      ort,
      encoder: this.encoder, decoder: this.decoder, joint: this.joint,
      preprocessor: this.preprocessor, tokenizer: this.tokenizer,
      audio: audio.samples,
      langPrompt: soniqoLangPrompt(this.languages, this.opts.language ?? "en-US"),
    });
    return { text };
  }

  async dispose(): Promise<void> {
    await this.encoder?.release?.();
    await this.decoder?.release?.();
    await this.joint?.release?.();
    this.encoder = this.decoder = this.joint = this.tokenizer = null;
  }
}
