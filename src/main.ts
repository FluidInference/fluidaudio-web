import "./theme/site.css";
import { initSiteTheme } from "./theme/theme.js";
initSiteTheme();

// Minimal demo UI wiring. Each engine implements a common interface (core/types),
// so the UI treats them uniformly: pick → load → run. Audio engines take a file;
// TTS takes text.

import { decodeToMono16k, pcmToWav } from "./core/audio.js";
import { segmentsToSrt, segmentsToVtt } from "./core/captions.js";
import { webgpuAvailable } from "./core/webgpu.js";
import { ENGINES, type EngineEntry } from "./engines/registry.js";
import { MicCapture } from "./core/mic.js";
import type { Engine, LoadProgress, TranscribeProgress } from "./core/types.js";

// Engine catalog lives in engines/registry.ts — a new engine registered there
// appears here automatically.
const ENTRIES: Record<string, EngineEntry> = Object.fromEntries(ENGINES.map((e) => [e.id, { ...e, label: `${e.label} ✅` }]));

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const engineSel = $<HTMLSelectElement>("engine");
const status = $<HTMLDivElement>("status");
const output = $<HTMLDivElement>("output");

// Tile-major direct-B GEMM is DEFAULT ON (browser-verified 282× on the 1hr
// bench). ?tm=0 restores the LDS-staged baseline for A/Bs.
if (new URLSearchParams(location.search).get("tm") === "0") {
  (globalThis as any).__tmGemm = false;
  console.info("[playground] tile-major GEMM disabled (?tm=0)");
}

// Word-timestamp captions from the last file run (SRT/VTT downloads).
let lastSegments: { text: string; start: number; end: number }[] | null = null;
let lastFileName = "";
function renderCaptionLinks() {
  const box = document.getElementById("captionLinks");
  if (!box) return;
  for (const a of Array.from(box.querySelectorAll("a"))) URL.revokeObjectURL((a as HTMLAnchorElement).href);
  box.innerHTML = "";
  if (!lastSegments?.length) return;
  const base = lastFileName.replace(/\.[^.]+$/, "") || "captions";
  box.append("captions: ");
  for (const [ext, mime, body] of [
    ["srt", "application/x-subrip", segmentsToSrt(lastSegments)],
    ["vtt", "text/vtt", segmentsToVtt(lastSegments)],
  ] as const) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([body], { type: mime }));
    a.download = `${base}.${ext}`;
    a.textContent = `⬇ ${ext.toUpperCase()}`;
    a.style.marginRight = "1rem";
    box.appendChild(a);
  }
}
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

// Prune engines whose availability probe fails (e.g. local-only weights not
// deployed on this host) — removes the option instead of 404ing at load time.
void Promise.all(
  ENGINES.filter((e) => e.available).map(async (e) => {
    if (await e.available!().catch(() => false)) return;
    engineSel.querySelector(`option[value="${e.id}"]`)?.remove();
    delete ENTRIES[e.id];
    if (engineSel.value === "") engineSel.value = "asr-parakeet";
  }),
);

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
      lastFileName = file.name;
      const t0 = performance.now();
      const result = await runAudioEngine(engine, audio);
      const ms = performance.now() - t0;
      output.textContent = `⏱ ${ms.toFixed(0)}ms · audio ${dur.toFixed(1)}s · RTFx ${(dur / (ms / 1000)).toFixed(1)}×\n\n` + result;
      renderCaptionLinks();
    }
  } catch (err) {
    output.textContent = String(err);
  }
});

// ── live microphone: rolling-window re-transcription ─────────────────────────
// Engines exposing push() (EOU) stream for real: mic samples feed conformer
// K/V + LSTM caches incrementally (bit-exact with offline — docs/STREAMING.md).
// Batch engines fall back to re-decoding a rolling 30 s tail every ~1.5 s
// (cheap at 100×+ RTFx). On stop: streaming engines flush the tail via
// finish(); batch engines re-transcribe the FULL capture once.
const mic = new MicCapture();
const LIVE_WINDOW_SEC = 30;
let liveBusy = false;
let liveTimer: ReturnType<typeof setInterval> | null = null;
let livePos = 0; // absolute sample index consumed by the streaming path

function isStreaming(
  e: unknown,
): e is { push(c: Float32Array): Promise<string>; finish(): Promise<string>; reset(): void; streamEvents?: { type: string; time: number }[] } {
  return typeof (e as any)?.push === "function";
}

async function liveTick() {
  if (!engine || liveBusy || mic.seconds < 1) return;
  liveBusy = true;
  try {
    const vu = "▁▂▃▄▅▆▇█"[Math.min(7, Math.floor(mic.level * 8))];
    if (isStreaming(engine)) {
      const { samples, total } = mic.since(livePos);
      const text = await engine.push(samples);
      livePos = total; // only after push resolves — a failed push must not skip audio
      mic.dropBefore(livePos); // streaming never re-reads history; keep hours-long sessions bounded
      const ev = engine.streamEvents ?? [];
      output.textContent = `● LIVE ${vu} ${mic.seconds.toFixed(0)}s (true streaming)\n\n${text}${ev.length ? `\n\nevents: ${ev.map((e) => `${e.type}@${e.time}s`).join(" ")}` : ""}`;
    } else {
      const samples = mic.tail(LIVE_WINDOW_SEC);
      const r = await (engine as any).transcribe({ samples, sampleRate: 16000 });
      output.textContent = `● LIVE ${vu} ${mic.seconds.toFixed(0)}s (showing last ${Math.min(mic.seconds, LIVE_WINDOW_SEC).toFixed(0)}s)\n\n${r.text}${r.events?.length ? `\n\nevents: ${r.events.map((e: any) => `${e.type}@${e.time}s`).join(" ")}` : ""}`;
    }
  } catch (err) {
    output.textContent = `live error: ${String(err)}`;
  } finally {
    liveBusy = false;
  }
}

async function startLive() {
  if (stopping) return; // a previous session's flush is still settling
  output.textContent = "requesting microphone…";
  try {
    mic.clear();
    livePos = 0;
    if (engine && isStreaming(engine)) engine.reset();
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

let stopping = false;

async function stopLive() {
  if (stopping) return;
  stopping = true;
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
  // An in-flight tick may be mid-push: pushing/finishing/resetting concurrently
  // would interleave on the same encoder caches. Let it settle first.
  while (liveBusy) await new Promise((r) => setTimeout(r, 25));
  await mic.stop();
  micBtn.textContent = "🎤 Live";
  // (Run stays disabled until the flush below completes.)
  // Capture: the dropdown can reassign the global `engine` while we await —
  // the flush must finish/reset the engine that owned this stream.
  const eng = engine;
  if (eng && isStreaming(eng)) {
    // Streamed all along — just flush the right-padded tail. No re-decode.
    try {
      const { samples, total } = mic.since(livePos);
      livePos = total;
      if (samples.length) await eng.push(samples);
      const text = await eng.finish();
      const ev = eng.streamEvents ?? [];
      output.textContent = `■ final transcript (${mic.seconds.toFixed(0)}s, true streaming)\n\n${text}${ev.length ? `\n\nevents: ${ev.map((e) => `${e.type}@${e.time}s`).join(" ")}` : ""}`;
      status.textContent = "Done.";
    } catch (err) {
      output.textContent = String(err);
    } finally {
      // reset even when the flush failed — a stranded stream blocks every
      // subsequent Run with the stream-active guard.
      try {
        eng.reset();
      } catch {
        /* disposed mid-flight */
      }
    }
  } else if (engine && mic.seconds >= 1) {
    // Final pass over the WHOLE capture (the rolling view only showed the tail).
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
  runBtn.disabled = false; // only after the flush — Run mid-flush hits the stream-active guard
  stopping = false;
}

micBtn.addEventListener("click", () => {
  if (mic.running) void stopLive();
  else void startLive();
});

/** m:ss for the transcription-progress status line (minutes don't wrap: 1h → 60:00). */
function fmtClock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

async function runAudioEngine(eng: Engine, audio: { samples: Float32Array; sampleRate: number }): Promise<string> {
  const any = eng as any;
  if (typeof any.detect === "function") {
    const ranges = await any.detect(audio);
    return `${ranges.length} speech segments:\n` + ranges.map((r: any) => `  ${r.start.toFixed(2)}s – ${r.end.toFixed(2)}s`).join("\n");
  }
  if (typeof any.transcribe === "function") {
    lastSegments = null;
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
    // Determinate progress during long transcriptions (engines emit at their
    // window boundaries). Reuses the model-load bar; status shows a live
    // position + realtime factor, both restored on completion/error.
    const prevStatus = status.textContent;
    const t0 = performance.now();
    let shown = false;
    let r: any;
    try {
      r = await any.transcribe(audio, {
        onProgress: (p: TranscribeProgress) => {
          if (!shown) {
            shown = true;
            progress.value = 0;
            progress.hidden = false;
          }
          progress.value = p.fraction;
          const wall = (performance.now() - t0) / 1000;
          const rt = wall > 0 ? ` (${(p.processedSeconds / wall).toFixed(1)}× RT)` : "";
          status.textContent = `Transcribing… ${fmtClock(p.processedSeconds)} / ${fmtClock(p.totalSeconds)}${rt}`;
        },
      });
    } finally {
      if (shown) {
        progress.hidden = true;
        progress.value = 0;
        status.textContent = prevStatus;
      }
    }
    if (r.segments?.length) lastSegments = r.segments;
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
