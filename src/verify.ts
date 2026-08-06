// Verify: pick (or drop) ONE audio file and every engine runs on it — VAD,
// both ASRs, diarization; TTS engines synthesize a fixed sentence (they consume
// text, not audio). Records load/run timings + RTFx + a short output per engine
// and downloads the results as JSON. This is where you get real WebGPU numbers.
//
// Engine selection: a checkbox per engine, ALL checked by default — untick to
// skip. ?engines=asr-parakeet,vad-silero preselects (only those checked);
// ?noauto=1 skips the JSON auto-download (still shown on page).

import { decodeToMono16k } from "./core/audio";
import { webgpuAvailable } from "./core/webgpu";
import type { Engine } from "./core/types";

const TTS_EN = "The quick brown fox jumps over the lazy dog.";
const TTS_ZH = "今天天气很好，我们一起去公园散步吧。";

interface Case {
  id: string;
  label: string;
  kind: "audio" | "text";
  heavy?: boolean;
  make: () => Promise<Engine>;
  run: (engine: any, audio: { samples: Float32Array; sampleRate: number }) => Promise<any>;
  summarize: (out: any) => string;
}

const CASES: Case[] = [
  {
    id: "vad-silero",
    label: "Silero VAD",
    kind: "audio",
    make: async () => new (await import("./engines/vad-silero")).SileroVadEngine(),
    run: (e, a) => e.detect(a),
    summarize: (o) => `${o.length} speech segments`,
  },
  {
    id: "asr-parakeet",
    label: "Parakeet TDT v3",
    kind: "audio",
    make: async () => new (await import("./engines/asr-parakeet")).ParakeetV3Engine(),
    run: (e, a) => e.transcribe(a),
    summarize: (o) => o.text,
  },
  {
    id: "asr-whisper",
    label: "Whisper (99 langs)",
    kind: "audio",
    make: async () => new (await import("./engines/asr-whisper")).WhisperEngine(),
    run: (e, a) => e.transcribe(a),
    summarize: (o) => o.text,
  },
  {
    id: "diarization-sortformer",
    label: "Sortformer diarization",
    kind: "audio",
    make: async () => new (await import("./engines/diarization-sortformer")).SortformerDiarizationEngine(),
    run: (e, a) => e.diarize(a),
    summarize: (o) => `${new Set(o.map((s: any) => s.speaker)).size} speakers, ${o.length} segments`,
  },
  {
    id: "tts-kokoro-en",
    label: "Kokoro TTS (English)",
    kind: "text",
    make: async () => new (await import("./engines/tts-kokoro")).KokoroTtsEngine({ lang: "en" }),
    run: (e) => e.synthesize(TTS_EN),
    summarize: (o) => `${(o.samples.length / o.sampleRate).toFixed(2)}s audio`,
  },
  {
    id: "tts-kokoro-zh",
    label: "Kokoro TTS (Chinese)",
    kind: "text",
    heavy: true,
    make: async () => new (await import("./engines/tts-kokoro")).KokoroTtsEngine({ lang: "zh" }),
    run: (e) => e.synthesize(TTS_ZH),
    summarize: (o) => `${(o.samples.length / o.sampleRate).toFixed(2)}s audio`,
  },
  {
    // soniqo FP16 export: encoder on WebGPU (purpose-built for it), LSTM decoder +
    // joint on WASM. Correct + fast in-browser (int4 export couldn't run on WebGPU).
    id: "asr-nemotron",
    label: "Nemotron 3.5 (fp16, WebGPU)",
    kind: "audio",
    heavy: true,
    make: async () => new (await import("./engines/asr-nemotron")).NemotronEngine(),
    run: (e, a) => e.transcribe(a),
    summarize: (o) => o.text || "(empty)",
  },
  {
    id: "eou-parakeet",
    label: "Parakeet EOU 120M",
    kind: "audio",
    heavy: true,
    make: async () => new (await import("./engines/eou-parakeet")).ParakeetEouEngine(),
    run: (e, a) => e.transcribe(a),
    summarize: (o) => `${o.text || "(no speech)"}${o.events?.length ? ` · ${o.events.map((e: any) => `${e.type}@${e.time}s`).join(" ")}` : ""}`,
  },
];

const $ = (id: string) => document.getElementById(id)!;

function log(msg: string) {
  const el = $("log");
  el.textContent += msg + "\n";
  el.scrollTop = el.scrollHeight;
}

let running = false;

async function runAll(audioBuf: ArrayBuffer, sourceName: string) {
  if (running) return;
  running = true;
  $("log").textContent = "";
  $("json").textContent = "";
  ($("download") as HTMLAnchorElement).hidden = true;
  try {
    const params = new URLSearchParams(location.search);
    const cases = CASES.filter((c) => ($(`ck-${c.id}`) as HTMLInputElement)?.checked);
    if (!cases.length) {
      $("status").textContent = "No engines selected — tick at least one.";
      return;
    }

    $("status").textContent = `Decoding ${sourceName}…`;
    const audio = await decodeToMono16k(audioBuf);
    const audioSec = audio.samples.length / audio.sampleRate;
    log(`input: ${sourceName} · ${audioSec.toFixed(1)}s @ ${audio.sampleRate}Hz`);

    const results: any = {
      generatedAt: new Date().toISOString(),
      userAgent: navigator.userAgent,
      webgpu: webgpuAvailable(),
      input: { name: sourceName, seconds: +audioSec.toFixed(2), sampleRate: audio.sampleRate },
      engines: [],
    };

    for (const c of cases) {
      log(`\n▶ ${c.label} (${c.id})`);
      const rec: any = { id: c.id, label: c.label, kind: c.kind };
      try {
        let engine: Engine;
        const tLoad = performance.now();
        engine = await c.make();
        await engine.load((p) => {
          $("status").textContent = `${c.label}: loading ${p.file} ${Math.round((p.fraction || 0) * 100)}%`;
        });
        rec.loadMs = +(performance.now() - tLoad).toFixed(0);

        $("status").textContent = `${c.label}: running…`;
        const tRun = performance.now();
        const out = await c.run(engine, audio);
        rec.runMs = +(performance.now() - tRun).toFixed(0);

        if (c.kind === "audio") {
          rec.inputSec = +audioSec.toFixed(2);
          rec.rtfx = +(audioSec / (rec.runMs / 1000)).toFixed(1);
        } else {
          const outSec = out.samples.length / out.sampleRate;
          rec.outputSec = +outSec.toFixed(2);
          rec.rtfx = +(outSec / (rec.runMs / 1000)).toFixed(2);
        }
        if (out?.metrics) rec.stages = out.metrics; // Parakeet per-stage timings
        rec.output = c.summarize(out).slice(0, 200);
        rec.ok = true;
        log(`  ✓ load ${rec.loadMs}ms · run ${rec.runMs}ms · RTFx ${rec.rtfx}× · ${rec.output}`);
        await engine.dispose();
      } catch (err) {
        rec.ok = false;
        rec.error = String(err).slice(0, 300);
        log(`  ✗ ${rec.error}`);
      }
      results.engines.push(rec);
      $("json").textContent = JSON.stringify(results, null, 2);
    }

    $("status").textContent = "Done — drop another file to run again.";
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = $("download") as HTMLAnchorElement;
    a.href = url;
    a.hidden = false;
    if (!params.has("noauto")) {
      a.click(); // best-effort auto-download (may need a click if the browser blocks it)
    }
    console.log("[verify] results:", results);
  } catch (e) {
    $("status").textContent = "Verify run failed";
    log(String(e));
  } finally {
    running = false;
  }
}

// ── engine toggles: one checkbox per engine, all checked by default ──────────
{
  const only = new URLSearchParams(location.search)
    .get("engines")
    ?.split(",")
    .map((s) => s.trim());
  const fs = $("engines");
  for (const c of CASES) {
    const lbl = document.createElement("label");
    lbl.style.cssText = "display:inline-flex;align-items:center;gap:.3rem;margin:.15rem .9rem .15rem 0;cursor:pointer;";
    const ck = document.createElement("input");
    ck.type = "checkbox";
    ck.id = `ck-${c.id}`;
    ck.checked = only ? only.includes(c.id) : true; // default: ALL engines run
    lbl.appendChild(ck);
    lbl.appendChild(document.createTextNode(`${c.label}${c.kind === "text" ? " (text)" : ""}${c.heavy ? " (heavy)" : ""}`));
    fs.appendChild(lbl);
  }
}

// ── file plumbing: input, drag & drop, bundled-sample fallback ───────────────
const fileInput = $("file") as HTMLInputElement;
const drop = $("drop");

fileInput.addEventListener("change", async () => {
  const f = fileInput.files?.[0];
  if (f) runAll(await f.arrayBuffer(), f.name);
  fileInput.value = ""; // allow re-selecting the same file
});

drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("hover");
});
drop.addEventListener("dragleave", () => drop.classList.remove("hover"));
drop.addEventListener("drop", async (e) => {
  e.preventDefault();
  drop.classList.remove("hover");
  const f = e.dataTransfer?.files?.[0];
  if (f) runAll(await f.arrayBuffer(), f.name);
});

$("sampleBtn").addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation(); // don't trigger the surrounding <label>'s file picker
  runAll(await (await fetch("./sample.wav")).arrayBuffer(), "sample.wav (bundled 12s)");
});
