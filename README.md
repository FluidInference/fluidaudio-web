# FluidAudio Web

In-browser inference for FluidAudio's core models via **WebGPU + WebAssembly** —
no server, no upload, everything runs client-side. This is the browser sibling
of the Swift/CoreML [FluidAudio](https://github.com/FluidInference/FluidAudio)
framework and the Rust/WASM [FluidVad](https://github.com/FluidInference/FluidVad).

> **Why a separate repo?** CoreML (`.mlmodelc`) cannot run in a browser — it is
> an Apple-only runtime. The browser runs the **ONNX** source of the same models
> through `onnxruntime-web` (WebGPU EP, WASM fallback), `transformers.js`, and
> `kokoro-js`. So this repo tracks the ONNX exports, not the CoreML bundles.

**Every engine is verified on real data** — no scaffolds. Parakeet v3 = **2.15% WER**
on full LibriSpeech test-clean (matches native FluidAudio ~2.14%). Measured
in-browser on WebGPU (Chrome/macOS, warm/steady-state): VAD **139×**, Sortformer
**128×**, EOU **91×**, Parakeet v3 **47×**, Whisper **33×**, Kokoro **~10×**,
Nemotron **4.1×** — **all 8 engines correct + real-time-plus**, on WebGPU/WASM via
ONNX. (Nemotron uses the soniqo **fp16** export — built to run on ORT-WebGPU, since
the int4 export can't.) First (cold) run is several× slower — WebGPU compiles
shaders. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Model matrix

| Engine | Model | Runtime | Backend | Status |
|---|---|---|---|---|
| `asr-parakeet` | Parakeet TDT 0.6B **v3** | `onnxruntime-web` | **fp16** enc WebGPU + WASM dec | ✅ **2.15% WER**, **46×** (warm); fp16 encoder 1.24 GB (fp32 exceeded the 2 GB buffer cap) |
| `asr-whisper` | Whisper (99 langs) | transformers.js | **WebGPU** / WASM | ✅ **24×** |
| `tts-kokoro` | Kokoro 82M (en + **zh** g2pW) | `kokoro-js` | **WebGPU** / WASM | ✅ **10×** en / **10×** zh (warm) |
| `diarization-sortformer` | NVIDIA Sortformer 4-spk | `onnxruntime-web` | WebGPU / WASM | ✅ **82×** short-audio; long-audio needs streaming loop |
| `asr-nemotron` | Nemotron 3.5 streaming (multilingual) | `onnxruntime-web` | **fp16 enc WebGPU** + WASM dec | ✅ **4.1×**, correct — soniqo fp16 export (built for ORT-WebGPU; int4 can't run there) |
| `vad-silero` | Silero VAD v5 | `onnxruntime-web` | WASM | ✅ **132×** (direct ORT, no `vad-web`) |
| `eou-parakeet` | Parakeet EOU 120M | `onnxruntime-web` | WebGPU / WASM | ✅ **86×** (transcript + `<EOU>`/`<EOB>`) |

✅ = correctness checked (WER / RTFx / output) on real data. Numbers in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

Per-engine deep dives: [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) (integration +
WebGPU-vs-WASM tradeoffs), [`NEMOTRON.md`](docs/NEMOTRON.md) (cache-aware streaming
RNNT), [`EOU.md`](docs/EOU.md) (end-of-utterance detection + the NA-mel gotcha),
[`KOKORO_ZH.md`](docs/KOKORO_ZH.md) (Chinese G2P frontend).

## Quick start

```bash
npm install
npm run dev      # http://localhost:5173  (cross-origin isolated for threaded WASM)
npm run build
```

### Auto-benchmark

Open **`/bench.html`** — it runs each engine on a bundled 12s sample and downloads
a results JSON (load/run ms, RTFx, per-stage timings, output, per-engine errors).
This is where you get the real WebGPU numbers. First run downloads model weights
(cached after). Params: `?full=1` (add the heavy engines — Kokoro-zh, Nemotron,
Parakeet EOU), `?engines=a,b` (pick), `?noauto=1` (don't auto-download).

## Deploy

**Live demo:** https://fluidinference.github.io/fluidaudio-web/ (and
`/bench.html?full=1` for the auto-benchmark). Auto-deploys from `main` via
`.github/workflows/deploy.yml` (one-time: repo **Settings → Pages → Source: GitHub
Actions**).

Static site — `npm run build` outputs `dist/`. Model weights stream from Hugging
Face and cache client-side — no backend.

### Cloudflare Pages (for threaded WASM)

GitHub Pages can't send headers, so WASM runs single-threaded there (WebGPU engines
are unaffected). For **multi-threaded WASM** (`SharedArrayBuffer` — speeds up the
WASM-bound work like Nemotron's decode), deploy to Cloudflare Pages, which serves the
`public/_headers` file that sets COOP/COEP. One-time dashboard setup:

1. **Workers & Pages → Create → Pages → Connect to Git** → pick `FluidInference/fluidaudio-web`.
2. Build settings: **Framework preset** = None (or Vite), **Build command** = `npm run build`, **Build output directory** = `dist`.
3. **Environment variables** → add `NPM_FLAGS = --ignore-scripts` (skips the unused `sharp`/native postinstalls so the install doesn't fail).
4. Deploy. CF sets `CF_PAGES=1`, so Vite uses base `/` (root of the `*.pages.dev` domain), and `public/_headers` turns on cross-origin isolation.

COOP/COEP note: `_headers` uses `require-corp`. The HF model fetches are CORS
(`fetch()` with `access-control-allow-origin: *`), which satisfies COEP — but if a
download is ever blocked, switch the header to `Cross-Origin-Embedder-Policy:
credentialless`. Netlify uses the same `_headers`; Vercel needs `vercel.json`; nginx
`add_header`.

> **First-load weight:** the default `asr-parakeet` uses a ~1.24 GB fp16 encoder — fine
> once cached, heavy on first visit. For a light demo use `?engines=tts-kokoro-en,
> diarization-sortformer,vad-silero` (≈ tens of MB).

WebGPU (Chrome/Edge 121+, Safari 26+) is **optional** — it accelerates the heavy
encoders (Parakeet, Kokoro). Everything else, and every engine's fallback path,
runs on WASM: Silero VAD is WASM-only by design (tiny model), and engines
downgrade automatically where WebGPU op coverage is incomplete. A WASM-only
browser still runs the full matrix, just slower on the big encoders.

## Layout

```
src/
  core/         shared runtime: ORT session factory, model cache, audio I/O, registry
  engines/      one folder per model, all implement a common interface (core/types.ts)
  main.ts       demo UI wiring
docs/           architecture + per-engine notes
```

## Hard-won lessons (things that cost real debugging)

- **WebGPU has no int8/int4 kernels → use fp16 for the browser.** ORT-web's WebGPU EP
  can't run quantized weights: Parakeet's int8 encoder *collapses to all-blank* and
  Nemotron's int4 returns *empty*. The fix for both was the same — an **fp16 encoder**
  (Parakeet 1.24 GB; Nemotron via soniqo's fp16 export, purpose-built for ORT-WebGPU),
  which runs correctly and fast on the GPU. (Check an encoder's output std before
  blaming the decoder.)
- **Task inputs matter more than precision.** Nemotron returned empty *also* because
  `lang_id=0` = Bulgarian; en-US is ordinal **24**. Always verify language/prompt ids.
- **Models vary wildly in size.** Parakeet fp16 encoder ≈ 1.24 GB, Nemotron fp16
  ≈ 1.24 GB, EOU fp32 ≈ 480 MB, Kokoro ≈ 90 MB, Silero VAD ≈ 2 MB. Fetched from
  Hugging Face once, persisted via the Cache API.
- **G2P is the TTS long pole.** The acoustic model is easy; Chinese uses a
  precomputed misaki-exact pinyin→IPA table + `pinyin-pro` (`docs/KOKORO_ZH.md`).
- **onnxruntime-web + Vite:** keep ORT out of `optimizeDeps` (Vite rewrites its
  `jsep.mjs` import and breaks it), and don't pin `wasmPaths` to a mismatched
  version — let ORT self-resolve.
- **Don't fight `@ricky0123/vad-web`.** It's CJS and does a dynamic
  `require("onnxruntime-web/wasm")` that Vite can't resolve once ORT is excluded
  from `optimizeDeps`. Silero's ONNX interface is trivial (512-sample windows +
  a 2×1×128 state), so `vad-silero` drives `silero_vad.onnx` directly through
  `core/ort` and drops the dependency.
- **EOU wants a different mel.** `parakeet-realtime-eou` is a NeMo streaming RNNT
  and expects **NA (un-normalized) log-mel** — the Nemotron frontend, *not*
  Parakeet's per-feature CMVN. Feed it the wrong normalization and the encoder
  emits content-free frames (flat per-frame RMS) while the joint predicts blank
  on every step — looks like a decode bug, is actually the frontend.

## License

MIT (code). Model licenses follow their upstream repos (see registry).
