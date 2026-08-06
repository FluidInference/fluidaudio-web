// Auto-benchmark: on open, runs each engine on a bundled sample, records
// load/run timings + RTFx + a short output, then downloads results as JSON
// (and shows them on the page). This is where you get the real WebGPU numbers.
//
// URL params:
//   ?full=1   also run the heavy engines (Kokoro-zh, Nemotron)
//   ?engines=asr-parakeet,vad-silero   run only these
//   ?noauto=1 don't auto-download the JSON (still shown on page)

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

async function main() {
  const params = new URLSearchParams(location.search);
  const only = params
    .get("engines")
    ?.split(",")
    .map((s) => s.trim());
  const full = params.has("full");
  const cases = CASES.filter((c) => (only ? only.includes(c.id) : full || !c.heavy));

  const audioBuf = await (await fetch("./sample.wav")).arrayBuffer();
  const audio = await decodeToMono16k(audioBuf);
  const audioSec = audio.samples.length / audio.sampleRate;

  const results: any = {
    generatedAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    webgpu: webgpuAvailable(),
    sample: { seconds: +audioSec.toFixed(2), sampleRate: audio.sampleRate },
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

  $("status").textContent = "Done.";
  const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = $("download") as HTMLAnchorElement;
  a.href = url;
  a.hidden = false;
  if (!params.has("noauto")) {
    a.click(); // best-effort auto-download (may need a click if the browser blocks it)
  }
  console.log("[bench] results:", results);
}

main().catch((e) => {
  $("status").textContent = "Benchmark failed";
  log(String(e));
});
