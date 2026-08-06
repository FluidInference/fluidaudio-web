// SDK root. Engines are exposed as SUBPATH exports so consumers only pull the
// code (and lazily, the weights) for what they use:
//   import { ParakeetV3Engine } from "@fluidinference/fluidaudio-web/asr-parakeet";
//   import { KokoroTtsEngine } from "@fluidinference/fluidaudio-web/tts-kokoro";
// The root exports the shared types, the engine registry (id → lazy factory),
// and the audio-decoding helper the demo pages use.

export { ENGINES } from "./engines/registry.js";
export type { EngineEntry, EngineKind } from "./engines/registry.js";
export type { Engine, AsrEngine, TtsEngine, DiarizationEngine, AudioData, AsrResult, DiarSegment, LoadProgress, ProgressCb } from "./core/types.js";
export { decodeToMono16k, pcmToWav } from "./core/audio.js";
export { webgpuAvailable } from "./core/webgpu.js";
