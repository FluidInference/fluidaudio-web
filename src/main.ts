// Minimal demo UI wiring. Each engine implements a common interface (core/types),
// so the UI treats them uniformly: pick → load → run. Audio engines take a file;
// TTS takes text.

import { decodeToMono16k, pcmToWav } from "./core/audio.js";
import { webgpuAvailable } from "./core/webgpu.js";
import { ENGINES, type EngineEntry } from "./engines/registry.js";
import { MicCapture } from "./core/mic.js";
import type { Engine, LoadProgress } from "./core/types.js";

// Engine catalog lives in engines/registry.ts (shared with the verify page) —
// a new engine registered there appears in both UIs automatically.
const ENTRIES: Record<string, EngineEntry> = Object.fromEntries(ENGINES.map((e) => [e.id, { ...e, label: `${e.label} ✅` }]));

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const engineSel = $<HTMLSelectElement>("engine");
const status = $<HTMLDivElement>("status");
const output = $<HTMLDivElement>("output");
const progress = $<HTMLProgressElement>("progress");
const runBtn = $<HTMLButtonElement>("run");
const micBtn = $<HTMLButtonElement>("mic");
const player = $<HTMLAudioElement>("player");

$("gpu").textContent = webgpuAvailable() ? "WebGPU available" : "WASM only (no WebGPU)";

for (const [id, e] of Object.entries(ENTRIES)) {
  const o = document.createElement("option");
  o.value = id;
  o.textContent = e.label;
  engineSel.appendChild(o);
}

engineSel.value = "asr-parakeet"; // landing default: the ASR demo, not the first registry entry (VAD)

let engine: Engine | null = null;
function currentEntry(): EngineEntry {
  return ENTRIES[engineSel.value];
}

function syncInputs() {
  const kind = currentEntry().kind;
  $("audioInput").hidden = kind !== "audio";
  $("textInput").hidden = kind !== "text";
  // Custom-vocabulary box: Parakeet only (BK-tree rescorer).
  $("vocabInput").hidden = engineSel.value !== "asr-parakeet";
  // TTS takes text → text box; ASR/VAD/diarization take audio → file picker.
  $("inputLabel").textContent = kind === "text" ? "Text to synthesize" : "Audio file";
}
engineSel.addEventListener("change", () => {
  syncInputs();
  runBtn.disabled = true;
  micBtn.disabled = true;
  if (mic.running) void stopLive();
});
syncInputs();

$("load").addEventListener("click", async () => {
  const entry = currentEntry();
  output.textContent = "";
  progress.hidden = false;
  runBtn.disabled = true;
  try {
    const eng = await entry.make();
    engine = eng;
    status.textContent = `Loading ${entry.label}…`;
    await eng.load((p: LoadProgress) => {
      progress.value = p.fraction || 0;
      status.textContent = `Loading ${p.file} — ${Math.round((p.fraction || 0) * 100)}%`;
    });
    status.textContent = `Ready: ${entry.label}`;
    runBtn.disabled = false;
    micBtn.disabled = !(currentEntry().kind === "audio" && typeof (engine as any)?.transcribe === "function");
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
      if (!file) {
        output.textContent = "Choose an audio file first.";
        return;
      }
      const audio = await decodeToMono16k(await file.arrayBuffer());
      const dur = audio.samples.length / audio.sampleRate;
      const t0 = performance.now();
      const result = await runAudioEngine(engine, audio);
      const ms = performance.now() - t0;
      output.textContent = `⏱ ${ms.toFixed(0)}ms · audio ${dur.toFixed(1)}s · RTFx ${(dur / (ms / 1000)).toFixed(1)}×\n\n` + result;
    }
  } catch (err) {
    output.textContent = String(err);
  }
});

// ── live microphone: rolling-window re-transcription ─────────────────────────
// No engine exposes true incremental streaming yet (the web Nemotron/EOU run
// whole-clip), but at 100×+ RTFx re-decoding a 30 s tail every ~1.5 s costs a
// fraction of a second — the standard "live demo" pattern. On stop, the FULL
// capture is transcribed once for the final transcript.
const mic = new MicCapture();
const LIVE_WINDOW_SEC = 30;
let liveBusy = false;
let liveTimer: ReturnType<typeof setInterval> | null = null;

async function liveTick() {
  if (!engine || liveBusy || mic.seconds < 1) return;
  liveBusy = true;
  try {
    const samples = mic.tail(LIVE_WINDOW_SEC);
    const r = await (engine as any).transcribe({ samples, sampleRate: 16000 });
    const vu = "▁▂▃▄▅▆▇█"[Math.min(7, Math.floor(mic.level * 8))];
    output.textContent = `● LIVE ${vu} ${mic.seconds.toFixed(0)}s (showing last ${Math.min(mic.seconds, LIVE_WINDOW_SEC).toFixed(0)}s)\n\n${r.text}${r.events?.length ? `\n\nevents: ${r.events.map((e: any) => `${e.type}@${e.time}s`).join(" ")}` : ""}`;
  } catch (err) {
    output.textContent = `live error: ${String(err)}`;
  } finally {
    liveBusy = false;
  }
}

async function startLive() {
  output.textContent = "requesting microphone…";
  try {
    mic.clear();
    await mic.start();
  } catch (err) {
    output.textContent = `microphone unavailable: ${String(err)}`;
    return;
  }
  micBtn.textContent = "⏹ Stop";
  runBtn.disabled = true;
  output.textContent = "● LIVE — listening…";
  liveTimer = setInterval(() => void liveTick(), 1500);
}

async function stopLive() {
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
  await mic.stop();
  micBtn.textContent = "🎤 Live";
  runBtn.disabled = false;
  // Final pass over the WHOLE capture (the rolling view only showed the tail).
  if (engine && mic.seconds >= 1) {
    status.textContent = `transcribing full ${mic.seconds.toFixed(0)}s capture…`;
    try {
      const t0 = performance.now();
      const r = await (engine as any).transcribe({ samples: mic.all(), sampleRate: 16000 });
      const ms = performance.now() - t0;
      output.textContent = `■ final transcript (${mic.seconds.toFixed(0)}s captured, ${ms.toFixed(0)}ms, RTFx ${(mic.seconds / (ms / 1000)).toFixed(1)}×)\n\n${r.text}`;
      status.textContent = "Done.";
    } catch (err) {
      output.textContent = String(err);
    }
  }
}

micBtn.addEventListener("click", () => {
  if (mic.running) void stopLive();
  else void startLive();
});

async function runAudioEngine(eng: Engine, audio: { samples: Float32Array; sampleRate: number }): Promise<string> {
  const any = eng as any;
  if (typeof any.detect === "function") {
    const ranges = await any.detect(audio);
    return `${ranges.length} speech segments:\n` + ranges.map((r: any) => `  ${r.start.toFixed(2)}s – ${r.end.toFixed(2)}s`).join("\n");
  }
  if (typeof any.transcribe === "function") {
    if (typeof any.setItn === "function") any.setItn(($("itn") as HTMLInputElement)?.checked ?? false);
    if (typeof any.setVocabulary === "function") {
      const raw = ($("vocab") as HTMLInputElement).value.trim();
      any.setVocabulary(
        raw
          ? raw
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean)
          : [],
      );
    }
    const r = await any.transcribe(audio);
    const events = r.events?.length ? `\n\nevents: ${r.events.map((e: any) => `${e.type}@${e.time}s`).join(" ")}` : "";
    if (r.metrics) {
      const m = r.metrics;
      return `stages: mel ${m.melMs}ms · encode(WebGPU) ${m.encodeMs}ms · decode ${m.decodeMs}ms\n\n${r.text}${events}`;
    }
    return `${r.text}${events}`;
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
