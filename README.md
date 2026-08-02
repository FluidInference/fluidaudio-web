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
| `vad-silero` | Silero VAD | `@ricky0123/vad-web` | WASM | ✅ wired |
| `tts-kokoro` | Kokoro 82M (en + **zh**) | `kokoro-js` | **WebGPU** / WASM | ✅ wired (zh needs G2P — see below) |
| `asr-parakeet` | Parakeet TDT 0.6B **v3** | `onnxruntime-web` | WebGPU enc + WASM dec | 🚧 scaffold |
| `asr-nemotron` | Nemotron 3.5 streaming (en + multilingual) | `onnxruntime-web` | WebGPU / WASM | 🚧 scaffold |
| `diarization-pyannote` | pyannote seg + embedding + clustering | `sherpa-onnx` WASM | WASM | 🚧 scaffold |
| `eou-parakeet` | Parakeet EOU 120M | `onnxruntime-web` | WASM | ⛔ greenfield (no ONNX export yet) |

"Wired" = calls a mature upstream library. "Scaffold" = interface + runtime +
model registry in place, decode loop is the remaining work. "Greenfield" = no
public ONNX export exists; needs export first.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for per-engine integration
notes and the WebGPU-vs-WASM tradeoffs.

## Quick start

```bash
npm install
npm run dev      # http://localhost:5173  (cross-origin isolated for threaded WASM)
npm run build
```

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
