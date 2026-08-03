// Nemotron 3.5 streaming ASR (en + multilingual, 40 langs) — cache-aware
// FastConformer-RNNT. ONNX: onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4.
//
// The int4 encoder runs on **WASM**, NOT WebGPU. It's numerically healthy on WASM
// (encoder std 0.43 — unlike Parakeet's int8, which collapses there), and ORT-web's
// WebGPU EP mishandles the int4 `MatMulNBits` ops → EMPTY transcript in-browser.
// Verified headless (ort-node WASM): correct output. So force WASM for the encoder;
// WebGPU buys nothing here anyway (thin GEMMs — see docs/RAW_WEBGPU.md).
//
// mel is NA log-mel computed in JS (no ONNX mel ships for Nemotron, and the
// parakeet nemo128 mel bakes per-feature CMVN which is wrong here) — the one JS
// DSP stage; could move to an NA-mel ONNX later.

import { configureOrt, ort, webgpuAvailable } from "../../core/ort";
import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";
import { JsPreprocessor } from "./nemotron-mel.js";
import { nemotronTranscribe, makeNemotronTokenizer, makeNemotronLangMap } from "./nemotron-decode.js";

const REPO = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4";

async function createWithData(name: string, ep: "webgpu" | "wasm", onProgress?: ProgressCb) {
  const model = await fetchCached(hfUrl(REPO, `${name}.onnx`), onProgress, `${name}.onnx`);
  const data = await fetchCached(hfUrl(REPO, `${name}.onnx.data`), onProgress, `${name}.onnx.data`);
  configureOrt();
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
  private langMap: Record<string, number> = {};

  /** @param opts.language BCP-47-ish code, e.g. "en-US" / "de" / "zh" (default en-US). */
  constructor(private opts: { language?: string } = {}) {}

  async load(onProgress?: ProgressCb): Promise<void> {
    // Encoder on WASM: int4 is healthy there, but ORT-web's WebGPU EP mishandles
    // the int4 MatMulNBits ops and returns an empty transcript. (WASM is not slower
    // for this thin-GEMM model — WebGPU wouldn't help even if it were correct.)
    this.encoder = await createWithData("encoder", "wasm", onProgress);
    this.decoder = await createWithData("decoder", "wasm", onProgress);
    this.joint = await createWithData("joint", "wasm", onProgress);
    const vocab = new TextDecoder().decode(await fetchCached(hfUrl(REPO, "vocab.txt"), onProgress, "vocab.txt"));
    this.tokenizer = makeNemotronTokenizer(vocab);
    this.langMap = makeNemotronLangMap(vocab);
    onProgress?.({ file: REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult> {
    if (!this.encoder || !this.decoder || !this.joint || !this.tokenizer) {
      throw new Error("NemotronEngine.load() not called");
    }
    const { text } = await nemotronTranscribe({
      ort,
      encoder: this.encoder,
      decoder: this.decoder,
      joint: this.joint,
      preprocessor: this.preprocessor,
      tokenizer: this.tokenizer,
      audio: audio.samples,
      langId: this.langMap[this.opts.language ?? "en-US"] ?? this.langMap["en"] ?? 24,
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
