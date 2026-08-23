// Single source of truth for "what engines exist", consumed by the
// playground (main.ts).
//
// `make` is lazy: each engine's module (and its deps) loads only when selected,
// so a broken engine can't take down the whole app at page load.

import type { Engine } from "../core/types.js";

export type EngineKind = "audio" | "text";

export interface EngineEntry {
  id: string;
  /** Display label. */
  label: string;
  kind: EngineKind;
  /** Large weight downloads (hundreds of MB). */
  heavy?: boolean;
  make: () => Promise<Engine>;
}

export const ENGINES: EngineEntry[] = [
  {
    id: "vad-silero",
    label: "Silero VAD",
    kind: "audio",
    make: async () => new (await import("./vad-silero/index.js")).SileroVadEngine(),
  },
  {
    id: "asr-parakeet",
    label: "Parakeet TDT v3",
    kind: "audio",
    make: async () => new (await import("./asr-parakeet/index.js")).ParakeetV3Engine(),
  },
  {
    id: "asr-whisper",
    label: "Whisper (99 langs)",
    kind: "audio",
    make: async () => new (await import("./asr-whisper/index.js")).WhisperEngine(),
  },
  {
    id: "diarization-sortformer",
    label: "Diarization (Sortformer)",
    kind: "audio",
    make: async () => new (await import("./diarization-sortformer/index.js")).SortformerDiarizationEngine(),
  },
  {
    id: "tts-kokoro-en",
    label: "Kokoro TTS — English",
    kind: "text",
    make: async () => new (await import("./tts-kokoro/index.js")).KokoroTtsEngine({ lang: "en" }),
  },
  {
    id: "tts-kokoro-zh",
    label: "Kokoro TTS — Chinese",
    kind: "text",
    heavy: true,
    make: async () => new (await import("./tts-kokoro/index.js")).KokoroTtsEngine({ lang: "zh" }),
  },
  {
    id: "asr-nemotron",
    label: "Nemotron 3.5 (40 langs)",
    kind: "audio",
    heavy: true,
    make: async () => new (await import("./asr-nemotron/index.js")).NemotronEngine(),
  },
  {
    id: "eou-parakeet",
    label: "Parakeet EOU 120M",
    kind: "audio",
    heavy: true,
    make: async () => new (await import("./eou-parakeet/index.js")).ParakeetEouEngine(),
  },
  {
    // Local weights: scripts/extract-voicechat-stt.py → public/models/voicechat-stt/ (gitignored).
    id: "asr-voicechat",
    label: "VoiceChat 11B STT",
    kind: "audio",
    heavy: true,
    make: async () => new (await import("./asr-voicechat/index.js")).VoicechatSttEngine(),
  },
];
