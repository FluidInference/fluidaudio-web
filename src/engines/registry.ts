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

/**
 * Dev convenience: when the vite middleware serves a local weight export at
 * /models/<dir> (models-local/), prefer it over the HF default — localhost
 * testing shouldn't re-download multi-GB weights that are already on disk.
 * Production hosts 404/HTML here and fall through to the HF default.
 */
export async function localWeightDir(dir: string, probeFile: string): Promise<string | undefined> {
  try {
    const base = (import.meta as any).env?.BASE_URL ?? "/";
    const url = `${base}models/${dir}`;
    const res = await fetch(`${url}/${probeFile}`, { method: "HEAD" });
    if (res.ok && !(res.headers.get("content-type") || "").includes("text/html")) return url;
  } catch {
    /* fall through to hosted default */
  }
  return undefined;
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
    id: "stem-dicose",
    label: "DiCoSe Stem Splitter",
    kind: "audio",
    category: "analysis",
    heavy: true,
    // 623 MB weight package on HF; probe keeps the picker honest if the files
    // are ever unreachable (or on forks without them).
    available: async () => {
      try {
        const res = await fetch("https://huggingface.co/FluidInference/fluidaudio-web/resolve/main/dicose/manifest.json", {
          method: "HEAD",
          referrerPolicy: "no-referrer",
        });
        return res.ok && !(res.headers.get("content-type") || "").includes("text/html");
      } catch {
        return false;
      }
    },
    make: async () => {
      const baseUrl = await localWeightDir("dicose", "manifest.json");
      return new (await import("./stem-dicose/index.js")).DicoseStemEngine(baseUrl ? { baseUrl } : {});
    },
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
    make: async () => {
      const baseUrl = await localWeightDir("voicechat-stt", "decoder-fp32.manifest.json");
      return new (await import("./asr-voicechat/index.js")).VoicechatSttEngine(baseUrl ? { baseUrl } : {});
    },
  },
  {
    id: "tts-voicechat",
    label: "VoiceChat TTS (Aria)",
    kind: "text",
    category: "tts",
    heavy: true,
    // Weights on HF; probe keeps the picker honest if the files are ever
    // unreachable (or on forks without them).
    available: async () => {
      try {
        const res = await fetch("https://huggingface.co/FluidInference/fluidaudio-web/resolve/main/voicechat-tts/config.json", {
          method: "HEAD",
          referrerPolicy: "no-referrer",
        });
        return res.ok && !(res.headers.get("content-type") || "").includes("text/html");
      } catch {
        return false;
      }
    },
    make: async () => {
      const baseUrl = await localWeightDir("voicechat-tts", "config.json");
      return new (await import("./tts-voicechat/index.js")).VoicechatTtsEngine(baseUrl ? { baseUrl } : {});
    },
  },
];
