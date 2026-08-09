// Live-captions showcase: mic → true-streaming ASR (push()-capable engines).
// EOU finalizes utterances from the model's own <EOU>/<EOB> events; engines
// without events (Nemotron) finalize on a text-stall timeout. The transcript
// is cumulative and append-only, so an utterance = the text delta since the
// last finalize boundary.

import { ENGINES } from "./engines/registry.js";
import { MicCapture } from "./core/mic.js";
import type { Engine } from "./core/types.js";

type StreamingEngine = Engine & {
  push(c: Float32Array): Promise<string>;
  finish(): Promise<string>;
  reset(): void;
  streamEvents?: { type: string; time: number }[];
};

const $ = (id: string) => document.getElementById(id)!;
const mic = new MicCapture();
let engine: StreamingEngine | null = null;
let engineId = "";
let timer: ReturnType<typeof setInterval> | null = null;
let running = false;
let busy = false;
let livePos = 0;
let doneLen = 0; // finalized prefix length of the cumulative transcript
let lastText = "";
let lastEventCount = 0;
let lastChangeAt = 0;
let uttStart = 0; // seconds, for the history timestamp chips

const STALL_FINALIZE_SEC = 2.5; // no-events engines: utterance ends on silence

function finalize(nowText: string) {
  const utt = nowText.slice(doneLen).trim();
  if (utt) {
    const div = document.createElement("div");
    div.className = "utt";
    const t = document.createElement("span");
    t.className = "t";
    const m = Math.floor(uttStart / 60);
    t.textContent = `${m}:${String(Math.floor(uttStart % 60)).padStart(2, "0")}`;
    div.append(t, utt);
    $("history").appendChild(div);
    $("history").scrollTop = $("history").scrollHeight;
  }
  doneLen = nowText.length;
  uttStart = mic.seconds;
}

async function tick() {
  if (!engine || busy || !running) return;
  busy = true;
  try {
    const { samples, total } = mic.since(livePos);
    const text = await engine.push(samples);
    livePos = total;
    mic.dropBefore(livePos); // streaming never re-reads history
    const vu = "▁▂▃▄▅▆▇█"[Math.min(7, Math.floor(mic.level * 8))];
    $("vu").textContent = vu;
    const events = engine.streamEvents;
    if (events) {
      if (events.length > lastEventCount) {
        lastEventCount = events.length;
        finalize(text);
      }
    } else if (text !== lastText) {
      lastChangeAt = performance.now();
    } else if (text.length > doneLen && performance.now() - lastChangeAt > STALL_FINALIZE_SEC * 1000) {
      finalize(text);
    }
    lastText = text;
    $("current").textContent = text.slice(doneLen).trim() || " ";
  } catch (err) {
    $("status").textContent = String(err);
  } finally {
    busy = false;
  }
}

async function start() {
  const id = ($("engine") as HTMLSelectElement).value;
  ($("go") as HTMLButtonElement).disabled = true;
  try {
    if (!engine || engineId !== id) {
      await engine?.dispose();
      engine = null;
      const c = ENGINES.find((e) => e.id === id)!;
      $("status").textContent = `loading ${c.label}…`;
      const e = await c.make();
      await e.load((p) => {
        $("status").textContent = `loading ${p.file ?? ""} ${Math.round((p.fraction || 0) * 100)}%`;
      });
      engine = e as StreamingEngine;
      engineId = id;
    }
    engine.reset();
    $("history").innerHTML = "";
    $("current").textContent = " ";
    livePos = 0;
    doneLen = 0;
    lastText = "";
    lastEventCount = 0;
    lastChangeAt = performance.now();
    uttStart = 0;
    mic.clear();
    await mic.start();
    running = true;
    timer = setInterval(() => void tick(), 300);
    ($("go") as HTMLButtonElement).textContent = "⏹ Stop";
    ($("engine") as HTMLSelectElement).disabled = true;
    $("status").textContent = "listening…";
  } catch (err) {
    $("status").textContent = String(err);
  } finally {
    ($("go") as HTMLButtonElement).disabled = false;
  }
}

async function stop() {
  running = false;
  if (timer) clearInterval(timer);
  timer = null;
  ($("go") as HTMLButtonElement).disabled = true;
  try {
    while (busy) await new Promise((r) => setTimeout(r, 25));
    await mic.stop();
    if (engine) {
      try {
        const { samples, total } = mic.since(livePos);
        if (samples.length) await engine.push(samples);
        livePos = total;
        const text = await engine.finish();
        finalize(text);
        $("current").textContent = " ";
        $("status").textContent = "Stopped.";
      } catch (err) {
        $("status").textContent = String(err);
      } finally {
        try {
          engine.reset();
        } catch {
          /* disposed */
        }
      }
    }
  } finally {
    ($("go") as HTMLButtonElement).textContent = "▶ Start";
    ($("go") as HTMLButtonElement).disabled = false;
    ($("engine") as HTMLSelectElement).disabled = false;
    $("vu").textContent = "";
  }
}

$("go").addEventListener("click", () => {
  if (running) void stop();
  else void start();
});
