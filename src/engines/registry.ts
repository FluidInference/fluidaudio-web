// Single source of truth for "what engines exist". Consumed by BOTH pages —
// the playground (main.ts) and the verify page (verify.ts) — so a new engine
// added here automatically appears in both. (They previously kept separate
// hand-maintained lists, which had already drifted.)
//
// `make` is lazy: each engine's module (and its deps) loads only when selected,
// so a broken engine can't take down the whole app at page load.

import type { Engine } from "../core/types";

export type EngineKind = "audio" | "text";

export interface EngineEntry {
  id: string;
  /** Display label. `heavy` engines note their weight class on the verify page. */
  label: string;
  kind: EngineKind;
  /** Large weight downloads (hundreds of MB) — labeled "(heavy)" on the verify page. */
  heavy?: boolean;
  make: () => Promise<Engine>;
}

export const ENGINES: EngineEntry[] = [
  {
    id: "vad-silero",
    label: "Silero VAD",
    kind: "audio",
    make: async () => new (await import("./vad-silero")).SileroVadEngine(),
  },
  {
    id: "asr-parakeet",
    label: "Parakeet TDT v3",
    kind: "audio",
    make: async () => new (await import("./asr-parakeet")).ParakeetV3Engine(),
  },
  {
    id: "asr-whisper",
    label: "Whisper (99 langs)",
    kind: "audio",
    make: async () => new (await import("./asr-whisper")).WhisperEngine(),
  },
  {
    id: "diarization-sortformer",
    label: "Diarization (Sortformer)",
    kind: "audio",
    make: async () => new (await import("./diarization-sortformer")).SortformerDiarizationEngine(),
  },
  {
    id: "tts-kokoro-en",
    label: "Kokoro TTS — English",
    kind: "text",
    make: async () => new (await import("./tts-kokoro")).KokoroTtsEngine({ lang: "en" }),
  },
  {
    id: "tts-kokoro-zh",
    label: "Kokoro TTS — Chinese",
    kind: "text",
    heavy: true,
    make: async () => new (await import("./tts-kokoro")).KokoroTtsEngine({ lang: "zh" }),
  },
  {
    id: "asr-nemotron",
    label: "Nemotron 3.5 (40 langs)",
    kind: "audio",
    heavy: true,
    make: async () => new (await import("./asr-nemotron")).NemotronEngine(),
  },
  {
    id: "eou-parakeet",
    label: "Parakeet EOU 120M",
    kind: "audio",
    heavy: true,
    make: async () => new (await import("./eou-parakeet")).ParakeetEouEngine(),
  },
];
