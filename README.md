# FluidAudio Web

In-browser inference for FluidAudio's core models via **WebGPU + WebAssembly** —
no server, no upload, everything runs client-side. This is the browser sibling
of the Swift/CoreML [FluidAudio](https://github.com/FluidInference/FluidAudio)
framework and the Rust/WASM [FluidVad](https://github.com/FluidInference/FluidVad).

> **Why a separate repo?** CoreML (`.mlmodelc`) cannot run in a browser — it is
> an Apple-only runtime. The browser runs the **ONNX** source of the same models
> through `onnxruntime-web` (WebGPU EP, WASM fallback), `transformers.js`, and
> `kokoro-js`. So this repo tracks the ONNX exports, not the CoreML bundles.

**Verified on real data:** Parakeet v3 = **2.15% WER** on full LibriSpeech
test-clean (matches native FluidAudio ~2.14%); Kokoro **~10× RTFx** on WebGPU;
Sortformer **123× RTFx**; Nemotron 3.5 transcribing 40 languages. See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Model matrix

| Engine | Model | Runtime | Backend | Status |
|---|---|---|---|---|
| `asr-parakeet` | Parakeet TDT 0.6B **v3** | `onnxruntime-web` | fp32 enc WebGPU + WASM dec | ✅ **2.15% WER** (full test-clean) |
| `asr-whisper` | Whisper (99 langs) | transformers.js | **WebGPU** / WASM | ✅ verified |
| `tts-kokoro` | Kokoro 82M (en + **zh** g2pW) | `kokoro-js` | **WebGPU** / WASM | ✅ verified (9.99× / 5.26× RTFx) |
| `diarization-sortformer` | NVIDIA Sortformer 4-spk | `onnxruntime-web` | WebGPU / WASM | ✅ verified short-audio (123× RTFx); long-audio needs streaming loop |
| `asr-nemotron` | Nemotron 3.5 streaming (40 langs) | `onnxruntime-web` | WebGPU / WASM | ✅ verified (int4 runs on WASM) |
| `vad-silero` | Silero VAD | `@ricky0123/vad-web` | WASM | ⛔ `vad-web` CJS/Vite issue |
| `eou-parakeet` | Parakeet EOU 120M | `onnxruntime-web` | WASM | ⛔ no public ONNX export |

✅ = correctness checked (WER / RTFx / output) on real data; ⛔ = blocked (reason
in the row). Numbers in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for per-engine integration
notes and the WebGPU-vs-WASM tradeoffs.

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
(cached after). Params: `?full=1` (add Kokoro-zh + Nemotron), `?engines=a,b`
(pick), `?noauto=1` (don't auto-download).

## Deploy

Static site — `npm run build` outputs `dist/`. The host **must** send
`Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy:
require-corp` (threaded WASM needs SharedArrayBuffer). `public/_headers` covers
**Netlify / Cloudflare Pages** automatically; for Vercel add them via
`vercel.json`, for nginx via `add_header`. Model weights stream from Hugging Face
and cache client-side — no backend.

Requires a browser with WebGPU (Chrome/Edge 121+, Safari 26+) — engines fall
back to WASM automatically where WebGPU op coverage is incomplete.

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
- **Models are big.** Parakeet fp32 encoder ≈ 2.4 GB, Nemotron int4 ≈ 750 MB,
  Kokoro ≈ 90 MB. Fetched from Hugging Face once, persisted via the Cache API.
- **G2P is the TTS long pole.** The acoustic model is easy; Chinese uses a
  precomputed misaki-exact pinyin→IPA table + `pinyin-pro` (`docs/KOKORO_ZH.md`).
- **onnxruntime-web + Vite:** keep ORT out of `optimizeDeps` (Vite rewrites its
  `jsep.mjs` import and breaks it), and don't pin `wasmPaths` to a mismatched
  version — let ORT self-resolve.

## License

MIT (code). Model licenses follow their upstream repos (see registry).
