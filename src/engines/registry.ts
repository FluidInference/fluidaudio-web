// Website catalog: shared SDK engines plus site-only runtimes.
import { ENGINES as SDK_ENGINES, type EngineEntry } from "./sdk-registry.js";
export type { EngineEntry, EngineKind, EngineCategory } from "./sdk-registry.js";

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
  ...SDK_ENGINES,
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
