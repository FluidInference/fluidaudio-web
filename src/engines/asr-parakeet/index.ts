// Parakeet TDT 0.6B v3 — offline transcription, fully internalized (no external
// ASR library). Mel frontend is the vendored NeMo-parity DSP (mel.js); tokenizer
// and TDT greedy decode are our own (tokenizer.js / tdt.js), shared verbatim with
// the headless Node verifier (scripts/smoke-parakeet-internal.mjs).
//
// Backend: the int8 encoder is numerically degenerate on the CPU/WASM EP (it
// collapses to ~0 — verified: encoder output std 0.017 → all-blank), so it MUST
// run on WebGPU. The tiny decoder+joint runs on WASM. Without WebGPU this engine
// throws rather than emit silent garbage.

import { configureOrt, createSession, ort, webgpuAvailable } from "../../core/ort";
import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";
import { OnnxMelPreprocessor } from "./onnxMel.js";
import { ParakeetTokenizer } from "./tokenizer.js";
import { transcribeTdt } from "./tdt.js";

const REPO = "ysdede/parakeet-tdt-0.6b-v3-onnx";
// fp32 encoder (with external .data) — the WebGPU EP has no int8 kernels, so the
// int8 encoder silently falls back to WASM where it decodes to all-blank. fp32
// runs correctly on WebGPU. (~2.4 GB; fp16 would halve it if a fp16 export exists.)
const ENCODER = "encoder-model.onnx";
const ENCODER_DATA = "encoder-model.onnx.data";
const DECODER = "decoder_joint-model.int8.onnx";
const MEL = "nemo128.onnx";
const VOCAB = "vocab.txt";

export class ParakeetV3Engine implements AsrEngine {
  readonly id = "asr-parakeet";
  readonly label = "Parakeet TDT 0.6B v3";
  private encoder: any = null;
  private decoder: any = null;
  private tokenizer: ParakeetTokenizer | null = null;
  private preprocessor: OnnxMelPreprocessor | null = null;

  async load(onProgress?: ProgressCb): Promise<void> {
    if (!webgpuAvailable()) {
      throw new Error("Parakeet v3 needs WebGPU (the fp32 encoder runs there; int8 collapses on WASM).");
    }
    const encBytes = await fetchCached(hfUrl(REPO, ENCODER), onProgress, ENCODER);
    const encData = await fetchCached(hfUrl(REPO, ENCODER_DATA), onProgress, ENCODER_DATA);
    const decBytes = await fetchCached(hfUrl(REPO, DECODER), onProgress, DECODER);
    const melBytes = await fetchCached(hfUrl(REPO, MEL), onProgress, MEL);
    const vocabText = new TextDecoder().decode(await fetchCached(hfUrl(REPO, VOCAB), onProgress, VOCAB));

    // fp32 encoder on WebGPU (external data); mel + decoder on WASM.
    configureOrt();
    this.encoder = await ort.InferenceSession.create(encBytes, {
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "all",
      externalData: [{ path: ENCODER_DATA, data: encData }],
    } as any);
    this.decoder = await createSession(decBytes, "wasm");
    const melSession = await createSession(melBytes, "wasm");
    this.preprocessor = new OnnxMelPreprocessor(ort, melSession, 128);
    this.tokenizer = ParakeetTokenizer.fromVocabText(vocabText);
    onProgress?.({ file: REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult> {
    if (!this.encoder || !this.decoder || !this.tokenizer || !this.preprocessor) {
      throw new Error("ParakeetV3Engine.load() not called");
    }
    const { text, metrics } = await transcribeTdt({
      ort,
      encoder: this.encoder,
      decoder: this.decoder,
      preprocessor: this.preprocessor,
      tokenizer: this.tokenizer,
      audio: audio.samples,
    });
    return { text, metrics };
  }

  async dispose(): Promise<void> {
    await this.encoder?.release?.();
    await this.decoder?.release?.();
    await this.preprocessor?.session?.release?.();
    this.encoder = this.decoder = this.tokenizer = this.preprocessor = null;
  }
}
