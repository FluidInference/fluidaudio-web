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
in-browser on WebGPU (Chrome/macOS, warm/steady-state): VAD **132×**, EOU **86×**,
Sortformer **82×**, Whisper **24×**, Parakeet v3 **14×**, Kokoro-en **10×** / zh
**10×** — all 8 engines work in-browser, all on WebGPU/WASM via ONNX. (Nemotron now
uses the soniqo **fp16** export — built to run on ORT-WebGPU, since the int4 export
can't; correct + fast.) First (cold) run is several× slower — WebGPU compiles
shaders. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Model matrix

| Engine | Model | Runtime | Backend | Status |
|---|---|---|---|---|
| `asr-parakeet` | Parakeet TDT 0.6B **v3** | `onnxruntime-web` | **fp16** enc WebGPU + WASM dec | ✅ **2.15% WER**, **46×** (warm); fp16 encoder 1.24 GB (fp32 exceeded the 2 GB buffer cap) |
| `asr-whisper` | Whisper (99 langs) | transformers.js | **WebGPU** / WASM | ✅ **24×** |
| `tts-kokoro` | Kokoro 82M (en + **zh** g2pW) | `kokoro-js` | **WebGPU** / WASM | ✅ **10×** en / **10×** zh (warm) |
| `diarization-sortformer` | NVIDIA Sortformer 4-spk | `onnxruntime-web` | WebGPU / WASM | ✅ **82×** short-audio; long-audio needs streaming loop |
| `asr-nemotron` | Nemotron 3.5 streaming (multilingual) | `onnxruntime-web` | **fp16 enc WebGPU** + WASM dec | ✅ correct + fast — soniqo fp16 export (built for ORT-WebGPU; int4 export can't run there) |
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

Static site — `npm run build` outputs `dist/`. **Threading:** onnxruntime-web's
multi-threaded WASM needs `Cross-Origin-Opener-Policy: same-origin` +
`Cross-Origin-Embedder-Policy: require-corp` (SharedArrayBuffer). GitHub Pages can't
send headers, so the live demo runs **WebGPU (which doesn't need them) at full speed
and WASM single-threaded** — and it deliberately avoids `require-corp`, which would
block the cross-origin Hugging Face model fetches. For threaded WASM, deploy to a
host that sets those two headers: `public/_headers` covers **Netlify / Cloudflare
Pages** automatically (verify HF sends `Cross-Origin-Resource-Policy` under COEP);
Vercel via `vercel.json`, nginx via `add_header`. Model weights stream from Hugging
Face and cache client-side — no backend.

> **First-load weight:** the default `asr-parakeet` uses a ~2.4 GB fp32 encoder — fine
> once cached, heavy on first visit. For a light demo use `?engines=tts-kokoro-en,
> diarization-sortformer,vad-silero` (≈ tens of MB).

WebGPU (Chrome/Edge 121+, Safari 26+) is **optional** — it accelerates the heavy
encoders (Parakeet, Kokoro). Everything else, and every engine's fallback path,
runs on WASM: Silero VAD and Nemotron are WASM-only by design, and engines
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

- **WebGPU has no int8/int4 kernels.** Quantized encoders silently fall back to
  WASM. For Parakeet the int8 encoder then *collapses to all-blank* on WASM
  (output std 0.017) — so it runs the **fp32** encoder on WebGPU instead. Check an
  encoder's output std before blaming the decoder.
- **int-quant is not always degenerate.** Nemotron's int4 encoder is healthy on
  WASM (std 0.43) and runs fine there — no WebGPU required.
- **Task inputs matter more than precision.** Nemotron returned empty not because
  of int4 but because `lang_id=0` = Bulgarian; en-US is ordinal **24**. Always
  verify language/prompt ids.
- **Models vary wildly in size.** Parakeet fp32 encoder ≈ 2.4 GB, Nemotron int4
  ≈ 750 MB, EOU fp32 ≈ 480 MB, Kokoro ≈ 90 MB, Silero VAD ≈ 2 MB. Fetched from
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
