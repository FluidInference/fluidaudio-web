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
  // ✅ CONFIRMED — the model khawjaahmad/nemotron-asr-webgpu loads.
  "asr-nemotron": {
    files: [
      { repo: "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4", path: "onnx/encoder_model.onnx" },
      { repo: "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4", path: "onnx/decoder_joint_model.onnx" },
    ],
    approxMB: 750,
    license: "nvidia-open-model",
    note: "40 langs. Cache-aware streaming; chunk tiers 80/160/320/560/1120ms. Confirm exact file names.",
  },
  // ✅ CONFIRMED — the repo parakeet.js resolves for 'parakeet-tdt-0.6b-v3'.
  // The engine loads via parakeet.js (fromHub), which manages the exact file
  // set; listed here for reference/quant selection.
  "asr-parakeet-v3": {
    files: [{ repo: "ysdede/parakeet-tdt-0.6b-v3-onnx", path: "(managed by parakeet.js)" }],
    approxMB: 600,
    license: "cc-by-4.0",
    note: "Loaded via parakeet.js fromHub('parakeet-tdt-0.6b-v3'), int8 encoder+decoder.",
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
  // ⛔ No public ONNX export of parakeet-realtime-eou-120m yet.
  "eou-parakeet": {
    files: [],
    approxMB: 120,
    license: "cc-by-4.0",
    note: "Greenfield. Needs a NeMo→ONNX export of the streaming EOU encoder first.",
  },
};
