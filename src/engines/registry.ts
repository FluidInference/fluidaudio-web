// Single source of truth for "what engines exist", consumed by the demo
// pages (src/pages/playground.ts, filtered per page by `category`).
//
// `make` is lazy: each engine's module (and its deps) loads only when selected,
// so a broken engine can't take down the whole app at page load.

import type { Engine } from "../core/types.js";

export type EngineKind = "audio" | "text";

/** Which demo page an engine belongs to (pages filter the registry by this). */
export type EngineCategory = "stt" | "tts" | "analysis";

export interface EngineEntry {
  id: string;
  /** Display label. */
  label: string;
  kind: EngineKind;
  category: EngineCategory;
  /** Large weight downloads (hundreds of MB). */
  heavy?: boolean;
  /**
   * Optional availability probe. Engines whose weights may not be deployed
   * (e.g. local-only exports) resolve false to be hidden from pickers instead
   * of failing at load time. Absent = always available.
   */
  available?: () => Promise<boolean>;
  make: () => Promise<Engine>;
}

export const ENGINES: EngineEntry[] = [
  {
    id: "vad-silero",
    label: "Silero VAD",
    kind: "audio",
    category: "analysis",
    make: async () => new (await import("./vad-silero/index.js")).SileroVadEngine(),
  },
  {
    id: "asr-parakeet",
    label: "Parakeet TDT v3",
    kind: "audio",
    category: "stt",
    make: async () => new (await import("./asr-parakeet/index.js")).ParakeetV3Engine(),
  },
  {
    id: "asr-whisper",
    label: "Whisper (99 langs)",
    kind: "audio",
    category: "stt",
    make: async () => new (await import("./asr-whisper/index.js")).WhisperEngine(),
  },
  {
    id: "diarization-sortformer",
    label: "Diarization (Sortformer)",
    kind: "audio",
    category: "analysis",
    make: async () => new (await import("./diarization-sortformer/index.js")).SortformerDiarizationEngine(),
  },
  {
    id: "tts-kokoro-en",
    label: "Kokoro TTS — English",
    kind: "text",
    category: "tts",
    make: async () => new (await import("./tts-kokoro/index.js")).KokoroTtsEngine({ lang: "en" }),
  },
  {
    id: "tts-kokoro-zh",
    label: "Kokoro TTS — Chinese",
    kind: "text",
    category: "tts",
    heavy: true,
    make: async () => new (await import("./tts-kokoro/index.js")).KokoroTtsEngine({ lang: "zh" }),
  },
  {
    id: "asr-nemotron",
    label: "Nemotron 3.5 (40 langs)",
    kind: "audio",
    category: "stt",
    heavy: true,
    make: async () => new (await import("./asr-nemotron/index.js")).NemotronEngine(),
  },
  {
    id: "eou-parakeet",
    label: "Parakeet EOU 120M",
    kind: "audio",
    category: "stt",
    heavy: true,
    make: async () => new (await import("./eou-parakeet/index.js")).ParakeetEouEngine(),
  },
  {
    id: "asr-voicechat",
    label: "VoiceChat 11B STT",
    kind: "audio",
    category: "stt",
    heavy: true,
    make: async () => new (await import("./asr-voicechat/index.js")).VoicechatSttEngine(),
  },
  {
    id: "tts-voicechat",
    label: "VoiceChat TTS (Aria)",
    kind: "text",
    heavy: true,
    // Local-only weights (scripts/extract-voicechat-tts.py → models-local/,
    // served at /models by the dev middleware) — not hosted on HF yet. Probe
    // the export so the picker hides the engine when it is absent.
    available: async () => (await fetch(`${import.meta.env.BASE_URL}models/voicechat-tts/config.json`)).ok,
    make: async () => new (await import("./tts-voicechat/index.js")).VoicechatTtsEngine(),
  },
];
