# Architecture

## Goal

Run FluidAudio's core models in the browser with **WebGPU + WebAssembly**, no
server. This is a parallel deployment of the _same models_ FluidAudio ships as
CoreML — but on a hand-written compute stack, because CoreML has no browser
runtime and onnxruntime-web's per-op dispatch left most of the performance on
the table (see [`ORT_REMOVAL.md`](ORT_REMOVAL.md) for that history — production
code has **no onnxruntime anywhere**; `onnxruntime-node` survives only as a
devDependency for offline reference generation).

## Layers

```
UI / SDK consumer (pages/playground.ts, live.ts, @fluidinference/fluidaudio-web)
    ↓
Engine interface (core/types.ts: AsrEngine, StreamingAsrEngine, TtsEngine, …)
    ↓  one folder per model under src/engines/, listed in core/registry.ts
ComputeContext (src/gpu/compute.d.ts — ONE backend-independent op interface)
    ├─ WebGPU: GpuContext (src/gpu/compute.js — facade + op methods)
    │     ├─ kernels/         hand-written WGSL (gemm, attention, conv, …)
    │     ├─ buffer-pool.js   exact-size pool + arena scopes + pinning
    │     ├─ scheduler.js     batching, dispatch, uniform ring, profiler
    │     └─ pipeline-cache.js compiled-pipeline cache
    └─ WASM: WasmContext (src/gpu/wasm-context.js)
          └─ wasm-kernels.wasm  Rust SIMD-128 (rust/wasm-kernels): f32/int8/int4
             GEMM + conv1d over a bump arena; everything else typed-array JS
```

`createContext()` (src/gpu/context.js) prefers WebGPU and falls back to
WASM+SIMD. Both backends implement every required `ComputeContext` member —
engine code never inspects tensor storage (`.buf`/`.data`) and never
feature-probes shared members; only genuinely WebGPU-only capabilities
(`attnFused`, `uploadTileMajorF16`, profiling) are optional. Two gates keep
this honest: `interface-conformance.mjs` (structure, runs in ci:smoke) and
`backend-conformance.mjs` (numeric GPU↔WASM parity over every required op,
`npm run conformance:numeric`).

## Backend policy (the important part)

The old ORT-era rule ("encoders on WebGPU EP, decoders on WASM EP") is gone.
Instead:

- **Encoders** run on the raw WebGPU kernels: GPU-resident tensors, fused
  dispatches, one submit per window group (`withBatch`), arena-scoped buffer
  reuse. The win is fusion + residency, not a faster single GEMM
  ([`RAW_WEBGPU.md`](RAW_WEBGPU.md)).
- **Autoregressive decoders** (RNNT/TDT) run on CPU WASM-SIMD — one result per
  token means a GPU decoder pays a round trip per step. Parakeet/Nemotron/EOU
  ship dedicated Rust decoders (`rust/parakeet-decoder`); Whisper's decoder is
  the exception (KV-cached greedy loop on the GPU, one submit per token).
- No cross-origin isolation is required: there is no SharedArrayBuffer use,
  and COEP would break cross-origin model fetches (see `vite.config.ts` note).
  Workers communicate by transfer.

## Per-engine matrix

| Engine                           | Compute                                                       | Weights (HF: FluidInference/fluidaudio-web)         | Pipeline                                                                                                 |
| -------------------------------- | ------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `asr-parakeet` (TDT 0.6B v3)     | GPU encoder **int8**, WASM decoder fp32                       | `parakeet/encoder-int8`, `decoder-fp32`             | JS mel → FastConformer → GPU joint-projection → WASM TDT; 15s/2s windows, 3-stage pipeline (see below)   |
| `asr-nemotron` (0.6B, 40 langs)  | GPU encoder **int8**, WASM decoder fp32                       | `nemotron/…` + `languages.json`                     | True streaming push()/finish(): cache-aware chunked encoder + GPU prompt-MLP with folded language bias   |
| `asr-whisper` (base, 99 langs)   | GPU fp32 end-to-end                                           | `whisper/…`, vocab from onnx-community              | Sequential 30s chunks; 6-layer PRE-LN encoder; KV-cached greedy GPU decode                               |
| `eou-parakeet` (120M EOU)        | GPU encoder **fp16** (int8 degrades this model), WASM decoder | `eou/…`                                             | Streaming or 240s batch; `<EOU>`/`<EOB>` become timestamped events                                       |
| `tts-kokoro` (82M, en+zh)        | GPU fp32                                                      | `kokoro/`, `kokoro-zh/`, voices from onnx-community | Lexicon G2P (misaki en / pinyin→bopomofo zh) → ALBERT → prosody → iSTFTNet synth (`gpu/kokoro-synth.js`) |
| `vad-silero` (v5)                | WASM only, weights **bundled** (~1.2MB)                       | none (in-repo)                                      | Hand-written forward (`raw-silero.js`), 512-sample windows, hysteresis + merge                           |
| `diarization-sortformer` (4-spk) | GPU encoder **int8**, head fp32                               | `sortformer/…`                                      | Shared FastConformer + 18-layer head; 90s/15s windows stitched by overlap-permutation matching           |
| `diarization-pyannote`           | —                                                             | —                                                   | Scaffold only: the intended path is vendoring sherpa-onnx's WASM diarization bundle, not raw kernels     |

All engines share: `core/modelCache.ts` (HF fetch + Cache API), `core/audio.ts`
(decode to 16k mono), `core/captions.ts` (word timings, SRT/VTT),
`core/textnorm.ts` (opt-in ITN), `core/mic.ts` (live capture).

## The Parakeet pipeline (the template for batch ASR)

`engines/asr-parakeet/pipeline.js` orchestrates a 3-stage software pipeline —
GPU encodes group g+1 while WASM decodes group g and workers compute mel for
g+2. The concerns live in single-purpose modules:

- `windowing.js` — window starts + wb-sized equal-length encoder groups
- `mel-scheduler.js` — mel prefetch via worker pool (unhidden wait timed)
- `decode-sink.js` — GPU-decoder / worker-pool / sync-WASM decode paths,
  strictly in window order
- `stitcher.js` — overlap seam dedup (frame-estimated skip refined by exact
  token match)

The joint projection and the staging readback ride the encoder's own submit
(`post` hook), so the GPU never idles waiting for JS to request a readback.
Worker pools (decode, mel) spawn lazily and fall back to the main thread.
Streaming engines (Nemotron, EOU) use `streaming-encoder.js` instead:
cache-carrying chunked encode, same kernels ([`STREAMING.md`](STREAMING.md)).

## Verification & gates

Layered, mostly hermetic:

| Gate                                                 | Scope                                                         | Command                                               |
| ---------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------- |
| Structural conformance                               | both backends implement ComputeContext (parsed from the d.ts) | in `ci:smoke`                                         |
| Numeric conformance                                  | every required op, GPU vs WASM, seeded inputs                 | `npm run conformance:numeric`                         |
| Kernel parity vs CPU refs                            | WGSL kernels (dawn) / Rust kernels                            | `npm run gpu:verify` / `wasm:verify`                  |
| Engine smokes (e2e, WASM backend, hermetic given HF) | parakeet, eou, kokoro                                         | `npm run ci:smoke`                                    |
| GPU memory gate                                      | pooling/arena invariants over a 120s transcribe               | `scripts/gpu-memory-check.mjs` (needs local fixtures) |
| In-browser                                           | all engines + metrics JSON                                    | `verify.html` / `bench.html`                          |

Node GPU gates run on dawn (`@kmamal/gpu`); dawn keeps the event loop alive,
so gate scripts end with `process.exit`.

## Entry points & SDK

- One page per function, each a thin entry over the shared
  `src/pages/playground.ts` core (registry filtered by `EngineEntry.category`):
  `index.html` / `stt.ts` — speech to text (mic mode, captions, vocab/ITN),
  `tts.html` / `tts.ts` — synthesis, `analyze.html` / `analyze.ts` — VAD +
  diarization. Plus `live.html` / `live.ts` — mic captions with EOU
  finalization, and `music.html` / `music.ts` — ACE-Step music generation.
- SDK: `npm run sdk:build` → `dist-sdk/`, published as
  `@fluidinference/fluidaudio-web` with per-engine subpath exports
  (`…/asr-parakeet` etc.); asset URLs use the cross-bundler
  `new URL("./x.wasm", import.meta.url)` pattern.

## Hosting

Static site; model weights stream from Hugging Face and cache client-side via
the Cache API — no backend, no special headers (COOP/COEP deliberately NOT
set; it gates only SharedArrayBuffer, which nothing uses, and would break
cross-origin model fetches).
