// Central registry of the ONNX sources each engine pulls. Kept in one place so
// swapping quant levels / repos is a data change, not a code change.
//
// ⚠️ Repos marked TODO must be confirmed against the actual HF tree before the
// engine can load — filenames vary between community exports. Confirmed entries
// were verified against upstream at authoring time.

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
  // ✅ CONFIRMED — soniqo FP16 export, built to run on onnxruntime-web WebGPU.
  // (int4 export can't: WebGPU has no int kernels.) Verified transcript headless.
  "asr-nemotron": {
    files: [
      { repo: "soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16", path: "encoder.onnx" },
      { repo: "soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16", path: "encoder.onnx.data" },
      { repo: "soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16", path: "decoder.onnx" },
      { repo: "soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16", path: "joint.onnx" },
      { repo: "soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16", path: "vocab.json" },
    ],
    approxMB: 1300,
    license: "nvidia-open-model",
    note: "Multilingual. fp16 encoder on WebGPU + LSTM decoder/joint on WASM; 320ms streaming chunks, RNN-T greedy. mel = NA log-mel (JS).",
  },
  // ✅ CONFIRMED — loaded directly by the internalized engine (no ASR library).
  // int8 encoder runs on WebGPU only (CPU/WASM collapses it to all-blank).
  "asr-parakeet-v3": {
    files: [
      { repo: "ysdede/parakeet-tdt-0.6b-v3-onnx", path: "encoder-model.int8.onnx" },
      { repo: "ysdede/parakeet-tdt-0.6b-v3-onnx", path: "decoder_joint-model.int8.onnx" },
      { repo: "ysdede/parakeet-tdt-0.6b-v3-onnx", path: "nemo128.onnx" },
      { repo: "ysdede/parakeet-tdt-0.6b-v3-onnx", path: "vocab.txt" },
    ],
    approxMB: 670,
    license: "cc-by-4.0",
    note: "All ORT: mel (nemo128) + decoder on WASM, encoder int8 on WebGPU (required). TDT decode + tokenizer in JS glue.",
  },
  // ✅ CONFIRMED — sherpa-onnx pretrained diarization set.
  "diarization-pyannote": {
    files: [
      { repo: "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0", path: "model.onnx" },
      // Embedding model: FluidAudio uses wespeaker_v2; sherpa ships 3D-Speaker/NeMo.
      // Pick one embedding export here.
    ],
    approxMB: 80,
    license: "MIT (seg: pyannote CC)",
    note: "Runs through sherpa-onnx WASM, not raw ORT. See engine for the wasm bundle.",
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
