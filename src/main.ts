// Minimal demo UI wiring. Each engine implements a common interface (core/types),
// so the UI treats them uniformly: pick → load → run. Audio engines take a file;
// TTS takes text.

import { decodeToMono16k, pcmToWav } from "./core/audio";
import { webgpuAvailable } from "./core/ort";
import type { Engine, LoadProgress } from "./core/types";

type Kind = "audio" | "text";
interface Entry {
  label: string;
  kind: Kind;
  // Lazy: each engine's module (and its deps) loads only when selected, so a
  // broken engine can't take down the whole app at page load.
  make: () => Promise<Engine>;
}

const ENTRIES: Record<string, Entry> = {
  "asr-parakeet": {
    label: "Parakeet TDT v3 ✅", kind: "audio",
    make: async () => new (await import("./engines/asr-parakeet")).ParakeetV3Engine(),
  },
  "tts-kokoro-en": {
    label: "Kokoro TTS — English ✅", kind: "text",
    make: async () => new (await import("./engines/tts-kokoro")).KokoroTtsEngine({ lang: "en" }),
  },
  "tts-kokoro-zh": {
    label: "Kokoro TTS — Chinese ✅*", kind: "text",
    make: async () => new (await import("./engines/tts-kokoro")).KokoroTtsEngine({ lang: "zh" }),
  },
  "vad-silero": {
    label: "Silero VAD ✅", kind: "audio",
    make: async () => new (await import("./engines/vad-silero")).SileroVadEngine(),
  },
  "asr-nemotron": {
    label: "Nemotron 3.5 (WebGPU) 🔬", kind: "audio",
    make: async () => new (await import("./engines/asr-nemotron")).NemotronEngine(),
  },
  "diarization-sortformer": {
    label: "Diarization (Sortformer) ✅", kind: "audio",
    make: async () => new (await import("./engines/diarization-sortformer")).SortformerDiarizationEngine(),
  },
  "eou-parakeet": {
    label: "Parakeet EOU ⛔", kind: "audio",
    make: async () => new (await import("./engines/eou-parakeet")).ParakeetEouEngine(),
  },
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const engineSel = $<HTMLSelectElement>("engine");
const status = $<HTMLDivElement>("status");
const output = $<HTMLDivElement>("output");
const progress = $<HTMLProgressElement>("progress");
const runBtn = $<HTMLButtonElement>("run");
const player = $<HTMLAudioElement>("player");

$("gpu").textContent = webgpuAvailable() ? "WebGPU available" : "WASM only (no WebGPU)";

for (const [id, e] of Object.entries(ENTRIES)) {
  const o = document.createElement("option");
  o.value = id;
  o.textContent = e.label;
  engineSel.appendChild(o);
}

let engine: Engine | null = null;
function currentEntry(): Entry { return ENTRIES[engineSel.value]; }

function syncInputs() {
  const kind = currentEntry().kind;
  $("audioInput").hidden = kind !== "audio";
  $("textInput").hidden = kind !== "text";
}
engineSel.addEventListener("change", () => { syncInputs(); runBtn.disabled = true; });
syncInputs();

$("load").addEventListener("click", async () => {
  const entry = currentEntry();
  output.textContent = "";
  progress.hidden = false;
  runBtn.disabled = true;
  try {
    engine = await entry.make();
    status.textContent = `Loading ${entry.label}…`;
    await engine.load((p: LoadProgress) => {
      progress.value = p.fraction || 0;
      status.textContent = `Loading ${p.file} — ${Math.round((p.fraction || 0) * 100)}%`;
    });
    status.textContent = `Ready: ${entry.label}`;
    runBtn.disabled = false;
  } catch (err) {
    status.textContent = `Load failed`;
    output.textContent = String(err);
  } finally {
    progress.hidden = true;
  }
});

runBtn.addEventListener("click", async () => {
  if (!engine) return;
  const entry = currentEntry();
  output.textContent = "Running…";
  player.hidden = true;
  try {
    if (entry.kind === "text") {
      const text = $<HTMLTextAreaElement>("text").value;
      const t0 = performance.now();
      const audio = await (engine as any).synthesize(text);
      const ms = performance.now() - t0;
      const dur = audio.samples.length / audio.sampleRate;
      const blob = pcmToWav(audio.samples, audio.sampleRate);
      player.src = URL.createObjectURL(blob);
      player.hidden = false;
      output.textContent =
        `Synthesized ${dur.toFixed(2)}s @ ${audio.sampleRate}Hz\n` +
        `⏱ ${ms.toFixed(0)}ms · RTFx ${(dur / (ms / 1000)).toFixed(1)}× · ${(text.length / (ms / 1000)).toFixed(0)} chars/s`;
    } else {
      const file = $<HTMLInputElement>("file").files?.[0];
      if (!file) { output.textContent = "Choose an audio file first."; return; }
      const audio = await decodeToMono16k(await file.arrayBuffer());
      const dur = audio.samples.length / audio.sampleRate;
      const t0 = performance.now();
      const result = await runAudioEngine(engine, audio);
      const ms = performance.now() - t0;
      output.textContent =
        `⏱ ${ms.toFixed(0)}ms · audio ${dur.toFixed(1)}s · RTFx ${(dur / (ms / 1000)).toFixed(1)}×\n\n` + result;
    }
  } catch (err) {
    output.textContent = String(err);
  }
});

async function runAudioEngine(eng: Engine, audio: { samples: Float32Array; sampleRate: number }): Promise<string> {
  const any = eng as any;
  if (typeof any.detect === "function") {
    const ranges = await any.detect(audio);
    return `${ranges.length} speech segments:\n` +
      ranges.map((r: any) => `  ${r.start.toFixed(2)}s – ${r.end.toFixed(2)}s`).join("\n");
  }
  if (typeof any.transcribe === "function") {
    const r = await any.transcribe(audio);
    if (r.metrics) {
      const m = r.metrics;
      return `stages: mel ${m.melMs}ms · encode(WebGPU) ${m.encodeMs}ms · decode ${m.decodeMs}ms\n\n${r.text}`;
    }
    return r.text;
  }
  if (typeof any.diarize === "function") {
    const segs = await any.diarize(audio);
    return segs.map((s: any) => `spk${s.speaker}: ${s.start.toFixed(2)}–${s.end.toFixed(2)}`).join("\n");
  }
  if (typeof any.push === "function") {
    // Streaming engines: feed the whole clip in 1s chunks as a smoke test.
    any.reset();
    const step = audio.sampleRate;
    let text = "";
    for (let i = 0; i < audio.samples.length; i += step) {
      text = await any.push(audio.samples.subarray(i, i + step));
    }
    return text;
  }
  return "Engine exposes no known run method.";
}
