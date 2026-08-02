// Minimal demo UI wiring. Each engine implements a common interface (core/types),
// so the UI treats them uniformly: pick → load → run. Audio engines take a file;
// TTS takes text.

import { decodeToMono16k, pcmToWav } from "./core/audio";
import { webgpuAvailable } from "./core/ort";
import type { Engine, LoadProgress } from "./core/types";
import { SileroVadEngine } from "./engines/vad-silero";
import { KokoroTtsEngine } from "./engines/tts-kokoro";
import { ParakeetV3Engine } from "./engines/asr-parakeet";
import { NemotronStreamingEngine } from "./engines/asr-nemotron";
import { PyannoteDiarizationEngine } from "./engines/diarization-pyannote";
import { ParakeetEouEngine } from "./engines/eou-parakeet";

type Kind = "audio" | "text";
interface Entry {
  label: string;
  kind: Kind;
  make: () => Engine;
}

const ENTRIES: Record<string, Entry> = {
  "vad-silero": { label: "Silero VAD ✅", kind: "audio", make: () => new SileroVadEngine() },
  "tts-kokoro-en": { label: "Kokoro TTS — English ✅", kind: "text", make: () => new KokoroTtsEngine({ lang: "en" }) },
  "tts-kokoro-zh": { label: "Kokoro TTS — Chinese ✅*", kind: "text", make: () => new KokoroTtsEngine({ lang: "zh" }) },
  "asr-parakeet": { label: "Parakeet TDT v3 ✅", kind: "audio", make: () => new ParakeetV3Engine() },
  "asr-nemotron": { label: "Nemotron streaming 🚧", kind: "audio", make: () => new NemotronStreamingEngine() },
  "diarization-pyannote": { label: "Diarization (pyannote) 🚧", kind: "audio", make: () => new PyannoteDiarizationEngine() },
  "eou-parakeet": { label: "Parakeet EOU ⛔", kind: "audio", make: () => new ParakeetEouEngine() },
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
    engine = entry.make();
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
      const audio = await (engine as any).synthesize(text);
      const blob = pcmToWav(audio.samples, audio.sampleRate);
      player.src = URL.createObjectURL(blob);
      player.hidden = false;
      output.textContent = `Synthesized ${audio.samples.length / audio.sampleRate | 0}s @ ${audio.sampleRate}Hz`;
    } else {
      const file = $<HTMLInputElement>("file").files?.[0];
      if (!file) { output.textContent = "Choose an audio file first."; return; }
      const audio = await decodeToMono16k(await file.arrayBuffer());
      const result = await runAudioEngine(engine, audio);
      output.textContent = result;
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
    return (await any.transcribe(audio)).text;
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
