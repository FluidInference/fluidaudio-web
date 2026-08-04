// Reference catalog of the ONNX sources each engine pulls. NOTE: this is
// documentation only — engines currently hardcode their own repo/filename
// constants (grep `const REPO`), so keep these entries in sync with the engine
// code by hand. (A future refactor could make the engines consume this.)

export interface ModelFile {
  repo: string;
  path: string;
  revision?: string;
}

export interface ModelSpec {
  /** HF repo(s) + files. */
  files: ModelFile[];
  /** Approx total download, for the UI. */
  approxMB: number;
  license: string;
  note?: string;
}

export const REGISTRY: Record<string, ModelSpec> = {
  // ✅ CONFIRMED — onnx-community, used by kokoro-js by default.
  "tts-kokoro-en": {
    files: [{ repo: "onnx-community/Kokoro-82M-v1.0-ONNX", path: "onnx/model_q8f16.onnx" }],
    approxMB: 90,
    license: "Apache-2.0",
    note: "kokoro-js manages the download itself; listed here for reference.",
  },
  // ✅ CONFIRMED repo exists. Chinese voices ship here; G2P is external (see engine).
  "tts-kokoro-zh": {
    files: [{ repo: "onnx-community/Kokoro-82M-v1.1-zh-ONNX", path: "onnx/model_q8f16.onnx" }],
    approxMB: 90,
    license: "Apache-2.0",
    note: "Acoustic only. Chinese text→phoneme frontend is NOT in the ONNX.",
  },
  // ✅ ORT-FREE — raw WebGPU FastConformer (int8) run offline + JS 2-layer RNNT.
  // Weights from FluidInference/fluidaudio-web (int8 encoder + fp32 decoder).
  "asr-nemotron": {
    files: [
      { repo: "FluidInference/fluidaudio-web", path: "nemotron/encoder-int8.bin" },
      { repo: "FluidInference/fluidaudio-web", path: "nemotron/encoder-int8.manifest.json" },
      { repo: "FluidInference/fluidaudio-web", path: "nemotron/decoder-fp32.bin" },
      { repo: "FluidInference/fluidaudio-web", path: "nemotron/decoder-fp32.manifest.json" },
      { repo: "FluidInference/fluidaudio-web", path: "nemotron/vocab.json" },
      { repo: "FluidInference/fluidaudio-web", path: "nemotron/languages.json" },
    ],
    approxMB: 730,
    license: "nvidia-open-model",
    note: "Multilingual. ORT-free: raw WebGPU int8 FastConformer (offline whole-clip, cache-aware mask) + prompt_kernel language conditioning + JS 2-layer RNN-T. mel = NA log-mel (JS).",
  },
  // ✅ loaded directly by the internalized engine (no ASR library).
  "asr-parakeet-v3": {
    files: [
      { repo: "ysdede/parakeet-tdt-0.6b-v3-onnx", path: "encoder-model.fp16.onnx" },
      { repo: "ysdede/parakeet-tdt-0.6b-v3-onnx", path: "decoder_joint-model.int8.onnx" },
      { repo: "ysdede/parakeet-tdt-0.6b-v3-onnx", path: "nemo128.onnx" },
      { repo: "ysdede/parakeet-tdt-0.6b-v3-onnx", path: "vocab.txt" },
    ],
    approxMB: 1300,
    license: "cc-by-4.0",
    note: "mel (nemo128) + int8 decoder on WASM; fp16 encoder on WebGPU (int8 collapses on WASM, fp32 exceeds the 2GB buffer cap). TDT decode + tokenizer in JS glue.",
  },
  // ✅ ORT-FREE — raw WebGPU int8 FastConformer + raw transformer head (single-chunk offline).
  "diarization-sortformer": {
    files: [
      { repo: "FluidInference/fluidaudio-web", path: "sortformer/encoder-int8.bin" },
      { repo: "FluidInference/fluidaudio-web", path: "sortformer/encoder-int8.manifest.json" },
      { repo: "FluidInference/fluidaudio-web", path: "sortformer/head-fp32.bin" },
      { repo: "FluidInference/fluidaudio-web", path: "sortformer/head-fp32.manifest.json" },
    ],
    approxMB: 140,
    license: "nvidia-open-model",
    note: "ORT-free: raw WebGPU int8 FastConformer encoder + 18-layer transformer head + sigmoid. Single-chunk offline (full pipeline == ORT preds, 1.79e-7 fp32 / 2.3e-3 int8); long-audio needs the streaming spkcache/fifo loop. mel = per_feature (JS).",
  },
  // ✅ CONFIRMED — asrjs export of nvidia/parakeet_realtime_eou_120m-v1.
  // fp32 encoder decodes on WASM *and* WebGPU (no int8-collapse like Parakeet).
  // Wants NA (un-normalized) log-mel — the Nemotron frontend, not per_feature.
  "eou-parakeet": {
    files: [
      { repo: "ysdede/parakeet-realtime-eou-120m-v1-onnx", path: "encoder-model.onnx" },
      { repo: "ysdede/parakeet-realtime-eou-120m-v1-onnx", path: "decoder_joint-model.onnx" },
      { repo: "ysdede/parakeet-realtime-eou-120m-v1-onnx", path: "vocab.txt" },
    ],
    approxMB: 480,
    license: "nvidia-open-model",
    note: "Streaming RNNT with <EOU>/<EOB> control tokens. NA mel (reuses Nemotron frontend). RNNT greedy decode + tokenizer in JS glue.",
  },
  // ✅ CONFIRMED — canonical Silero VAD v5 ONNX, driven directly via core/ort
  // (no @ricky0123/vad-web; its CJS require breaks under Vite).
  "vad-silero": {
    files: [{ repo: "onnx-community/silero-vad", path: "onnx/model.onnx" }],
    approxMB: 2,
    license: "MIT",
    note: "input[1,512]+state[2,1,128]+sr → prob+stateN. 32ms windows, hysteresis + duration guards in JS.",
  },
};
