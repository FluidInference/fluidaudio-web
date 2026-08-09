// Verify: pick (or drop) ONE audio file and every engine runs on it — VAD,
// both ASRs, diarization; TTS engines synthesize a fixed sentence (they consume
// text, not audio). Records load/run timings + RTFx + a short output per engine
// and downloads the results as JSON. This is where you get real WebGPU numbers.
//
// Engine selection: a checkbox per engine, ALL checked by default — untick to
// skip. ?engines=asr-parakeet,vad-silero preselects (only those checked);
// ?noauto=1 skips the JSON auto-download (still shown on page).

import { decodeToMono16k } from "./core/audio.js";
import { webgpuAvailable } from "./core/webgpu.js";
import { ENGINES } from "./engines/registry.js";
import { segmentsToSrt, segmentsToVtt } from "./core/captions.js";
import type { Engine } from "./core/types.js";

// Default-enabled engines: the Parakeet ASR pair. Everything else is opt-in
// per run (heavy downloads / secondary engines) — tick to include.
const DEFAULT_ON = new Set(["asr-parakeet", "eou-parakeet"]);

const TTS_TEXT: Record<string, string> = {
  "tts-kokoro-en": "The quick brown fox jumps over the lazy dog.",
  "tts-kokoro-zh": "今天天气很好，我们一起去公园散步吧。",
};

// Engines come from the shared registry (same catalog as the playground);
// run + summarize are duck-typed on the engine interface, so a new registry
// entry appears here with zero verify-side code.
function runEngine(e: any, audio: { samples: Float32Array; sampleRate: number }, id: string): Promise<any> {
  if (typeof e.synthesize === "function") return e.synthesize(TTS_TEXT[id] ?? "Hello from FluidAudio.");
  if (typeof e.detect === "function") return e.detect(audio);
  if (typeof e.diarize === "function") return e.diarize(audio);
  return e.transcribe(audio);
}

function summarize(out: any, id: string): string {
  if (out?.samples && out?.sampleRate) return `${(out.samples.length / out.sampleRate).toFixed(2)}s audio`;
  if (Array.isArray(out)) {
    // Shape alone can't distinguish an EMPTY diarization result from VAD —
    // key the phrasing on the engine id so a silent-diarization bug doesn't
    // read like a VAD result in the persisted JSON.
    if (id.startsWith("diarization")) {
      const speakers = new Set(out.map((s: any) => s.speaker).filter((s: any) => s !== undefined));
      return `${speakers.size} speakers, ${out.length} segments`;
    }
    return `${out.length} speech segments`;
  }
  const events = out?.events?.length ? ` · ${out.events.map((e: any) => `${e.type}@${e.time}s`).join(" ")}` : "";
  return `${out?.text || (id.startsWith("eou") ? "(no speech)" : "(empty)")}${events}`;
}

const $ = (id: string) => document.getElementById(id)!;

function log(msg: string) {
  const el = $("log");
  el.textContent += msg + "\n";
  el.scrollTop = el.scrollHeight;
}

let running = false;
const engineCache = new Map<string, Engine>();
window.addEventListener("pagehide", () => {
  for (const e of engineCache.values()) void e.dispose();
  engineCache.clear();
});

function addCaptionLinks(label: string, segments: { text: string; start: number; end: number }[], sourceName: string) {
  const base = sourceName.replace(/\.[^.]+$/, "") || "captions";
  const row = document.createElement("div");
  row.append(`${label}: `);
  for (const [ext, mime, body] of [
    ["srt", "application/x-subrip", segmentsToSrt(segments)],
    ["vtt", "text/vtt", segmentsToVtt(segments)],
  ] as const) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([body], { type: mime }));
    a.download = `${base}.${ext}`;
    a.textContent = `⬇ ${ext.toUpperCase()}`;
    a.style.marginRight = "1rem";
    row.appendChild(a);
  }
  $("captions").appendChild(row);
}

async function runAll(audioBuf: ArrayBuffer, sourceName: string) {
  if (running) return;
  running = true;
  $("log").textContent = "";
  $("json").textContent = "";
  $("captions").innerHTML = "";
  ($("download") as HTMLAnchorElement).hidden = true;
  try {
    const params = new URLSearchParams(location.search);
    const cases = ENGINES.filter((c) => ($(`ck-${c.id}`) as HTMLInputElement)?.checked);
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

    const keep = ($("keepLoaded") as HTMLInputElement)?.checked ?? false;
    for (const c of cases) {
      log(`\n▶ ${c.label} (${c.id})`);
      const rec: any = { id: c.id, label: c.label, kind: c.kind };
      let engine: Engine | null = null;
      try {
        const tLoad = performance.now();
        const cached = keep ? engineCache.get(c.id) : undefined;
        if (cached) {
          engine = cached;
          rec.loadMs = 0;
          rec.cached = true;
        } else {
          engine = await c.make();
          await engine.load((p) => {
            $("status").textContent = `${c.label}: loading ${p.file} ${Math.round((p.fraction || 0) * 100)}%`;
          });
          rec.loadMs = +(performance.now() - tLoad).toFixed(0);
        }

        $("status").textContent = `${c.label}: running…`;
        const tRun = performance.now();
        const out = await runEngine(engine, audio, c.id);
        rec.runMs = +(performance.now() - tRun).toFixed(0);

        if (c.kind === "audio") {
          rec.inputSec = +audioSec.toFixed(2);
          rec.rtfx = +(audioSec / (rec.runMs / 1000)).toFixed(1);
          rec.rtfxBasis = "input"; // audio-seconds processed per wall-second
        } else {
          const outSec = out.samples.length / out.sampleRate;
          rec.outputSec = +outSec.toFixed(2);
          rec.rtfx = +(outSec / (rec.runMs / 1000)).toFixed(2);
          rec.rtfxBasis = "output"; // TTS: audio-seconds GENERATED per wall-second — not comparable to ASR rtfx
        }
        if (out?.metrics) rec.stages = out.metrics; // Parakeet per-stage timings
        if (out?.segments?.length) {
          // Word timestamps → downloadable captions (segments stay out of the
          // results JSON — a 1-hour file is ~10k words).
          rec.words = out.segments.length;
          addCaptionLinks(c.label, out.segments, sourceName);
        }
        rec.output = summarize(out, c.id).slice(0, 200);
        rec.ok = true;
        log(`  ✓ load ${rec.loadMs}ms · run ${rec.runMs}ms · RTFx ${rec.rtfx}× · ${rec.output}`);
      } catch (err) {
        rec.ok = false;
        rec.error = String(err).slice(0, 300);
        log(`  ✗ ${rec.error}`);
      } finally {
        if (keep && rec.ok && engine) {
          // "Keep models loaded" (opt-in): repeat file drops skip the reload.
          // Costs GPU memory per engine — that's why it is not the default.
          engineCache.set(c.id, engine);
        } else {
          // Evicting a previously-cached engine must dispose it too — deleting
          // the only reference would leak its GPUDevice + weights.
          const stale = engineCache.get(c.id);
          engineCache.delete(c.id);
          try {
            if (stale && stale !== engine) await stale.dispose();
          } catch {
            /* device may already be lost */
          }
          // Dispose even on failure — a leaked GPUDevice + weights would
          // cascade OOM into the remaining engines' runs.
          try {
            await engine?.dispose();
          } catch {
            /* device may already be lost */
          }
        }
      }
      results.engines.push(rec);
      $("json").textContent = JSON.stringify(results, null, 2);
    }

    $("status").textContent = "Done — drop another file to run again.";
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = $("download") as HTMLAnchorElement;
    if (a.href.startsWith("blob:")) URL.revokeObjectURL(a.href); // don't leak the previous run's blob
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
  for (const c of ENGINES) {
    const lbl = document.createElement("label");
    lbl.style.cssText = "display:inline-flex;align-items:center;gap:.3rem;margin:.15rem .9rem .15rem 0;cursor:pointer;";
    const ck = document.createElement("input");
    ck.type = "checkbox";
    ck.id = `ck-${c.id}`;
    ck.checked = only ? only.includes(c.id) : DEFAULT_ON.has(c.id); // default: ASR duo only
    lbl.appendChild(ck);
    lbl.appendChild(document.createTextNode(`${c.label}${c.kind === "text" ? " (text)" : ""}${c.heavy ? " (heavy)" : ""}`));
    fs.appendChild(lbl);
  }
}

// ── file plumbing: input, drag & drop, bundled-sample fallback ───────────────
($("download") as HTMLAnchorElement).addEventListener("click", (e) => {
  // Before any completed run the href is the placeholder — clicking it would
  // save the PAGE ITSELF as "….json". Block until a real blob is attached.
  if (!($("download") as HTMLAnchorElement).href.startsWith("blob:")) e.preventDefault();
});

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
