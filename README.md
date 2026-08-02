# FluidAudio Web

In-browser inference for FluidAudio's core models via **WebGPU + WebAssembly** —
no server, no upload, everything runs client-side. This is the browser sibling
of the Swift/CoreML [FluidAudio](https://github.com/FluidInference/FluidAudio)
framework and the Rust/WASM [FluidVad](https://github.com/FluidInference/FluidVad).

> **Why a separate repo?** CoreML (`.mlmodelc`) cannot run in a browser — it is
> an Apple-only runtime. The browser runs the **ONNX** source of the same models
> through `onnxruntime-web` (WebGPU EP, WASM fallback), `kokoro-js`,
> `@ricky0123/vad-web`, and `sherpa-onnx` WASM. So this repo tracks the ONNX
> exports, not the CoreML bundles.

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

"Verified" = correctness checked (WER/RTFx/output) on real data. "Built" = full
pipeline implemented + shape-verified; accuracy pending. See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for numbers.

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

## Cross-cutting reality

- **WebGPU is per-op.** `onnxruntime-web`'s WebGPU EP lacks kernels for many
  dynamic-shape ops; graphs silently fall back to CPU islands. In practice these
  ship **WASM-first, WebGPU where the graph cooperates**. `core/ort.ts` picks the
  backend per engine and lets each override.
- **Models are big.** Nemotron INT4 ≈ 750 MB, Parakeet v3 ≈ 600 MB. First load is
  fetched from Hugging Face and persisted via the Cache API (`core/modelCache.ts`).
- **G2P is the TTS long pole.** The acoustic model is easy; Chinese needs a JS
  frontend (jieba segmentation + polyphone disambiguation + pinyin→IPA).

## License

MIT (code). Model licenses follow their upstream repos (see registry).
