# Porting a FluidAudio model to fluidaudio-web

The repeatable pipeline behind every engine here (Parakeet, Whisper, Kokoro,
Sortformer, Nemotron, EOU, Silero). Follow it and a new model is a checklist,
not archaeology. Reference implementations are cited per step — copy the
closest one.

## The pipeline at a glance

```
ONNX export (the CoreML repo tracks the same upstream checkpoints)
  → 1. extract weights   scripts/extract-<model>.py  →  weights.bin + manifest.json
  → 2. host on HF        FluidInference/fluidaudio-web/<model>/
  → 3. write the forward src/engines/<id>/raw-<model>.js  (on src/gpu kernels)
  → 4. parity-gate it    scripts/<model>-check.mjs  vs an ORT/numpy reference
  → 5. wrap the engine   src/engines/<id>/index.ts  (Engine interface)
  → 6. register it       src/engines/registry.ts    (both pages + SDK pick it up)
  → 7. CI smoke          scripts/ci-smoke-<id>.mjs  (hermetic, word/output asserts)
```

## 1. Weights → flat bin + manifest

One `Float32Array` blob + a JSON manifest of `{dims, offset, len}` per tensor
(offsets in FLOATS, not bytes). Pattern (`extract-parakeet-encoder-weights.py`):

```python
man[k] = {"dims": list(a.shape), "offset": len(blob)//4, "len": int(a.size)}
```

- Name tensors by ROLE (`L{n}_ff1w1`, `q`, `dw`), resolved from ONNX graph
  node names — never by anonymous initializer ids.
- int8: store `i8ByteOffset/scaleOffset/quant:"col"|"row"` per tensor
  (encoder-int8 manifest); f16 source blobs: `dtype:"f16"` + the loader's LUT
  expands (raw-encoder.js `f16lut`). Prefer fp32 for the first port — quantize
  only after parity.
- Fold constants at EXTRACT time when cheap: Parakeet folds `1/√HD` into q,
  the macaron ×0.5 into FF2 weights, xscale into the pre-encode linear.

## 2. Host on Hugging Face

Upload `<model>/weights.bin` + `<model>/manifest.json` to
`FluidInference/fluidaudio-web` (confirm repo before uploading). Engines fetch
via `fetchCached(hfUrl(...))` — Cache-API cached, retried, progress-reported.
No weights in git, no build-time assets.

## 3. The forward on the kernel library

`src/gpu/compute.js` (WebGPU) and `wasm-context.js` (CPU twin) share one
surface — write the forward once, it runs on both (`createContext()` picks).
Available: `matmul` (auto-routes f16/int8/subgroup variants; `{bias, act,
add}` fused epilogues), `conv1d/conv2d` (specialized fast paths self-route),
`lstm`, `layernorm`, `softmax`, `bmmQK/bmmPV/relShiftB` (batched attention),
`transpose/sliceCols/setCols/copyRows/concatRows`, `glu/silu/snake`.

Rules that cost real debugging when ignored:
- **Weights on the B side of matmul** (X@Wᵀ, upload transposed if needed) —
  that's where the f16 fast path lives. Upload big matrices with
  `ctx.uploadF16(...)` (self-falls-back to fp32).
- **Batch submits**: wrap the stack in `ctx.withBatchSync(...)` /
  `withBatch(...)` — per-op submits dominate otherwise. `download()` inside a
  batch flushes + reopens safely.
- **Arena-scope your intermediates**: `pushArena()/popArena(handle)` per
  window/step/synth, `pin()` what outlives the scope. Without this, one run
  can transiently allocate GB (Parakeet was 7.5GB before pooling).
- **Never rely on zero-initialized allocs** — buffers are pooled.
- CPU-side stages (mel, alignment) go in plain JS or the Rust wasm crate
  (`rust/`, wasm32+simd128) when SIMD matters.

## 4. Parity gate BEFORE the engine

A node script (dawn WebGPU via `scripts/gpu-globals.mjs`) comparing your
forward against a reference — ORT-node output, a numpy dump, or the Swift
repo's exported activations. House thresholds: fp32 ~1e-5; with int8 weights
~5e-3 vs the fp32 reference is normal (quantization dominates).

**Gate on OUTPUTS (tokens/segments/waveform corr), not just maxΔ** — and keep
one hermetic word-assert smoke as the anchor: self-relative gates are blind
to deterministic corruption (proven twice in this repo's history).

## 5–6. Engine + registry

Implement the `Engine` interface (core/types.ts): `load(onProgress)`, one of
`transcribe/synthesize/diarize/detect`, `dispose()` (MUST
`ctx.device.destroy()`). Batch ASR takes an optional second argument:
`transcribe(audio, opts?)` with `opts.onProgress` receiving
`TranscribeProgress { processedSeconds, totalSeconds, fraction }` at the
engine's window/slice boundary (`core/progress.ts` keeps it monotonic;
engines that can't estimate progress just never call it). Register in `src/engines/registry.ts` — the
playground and the SDK subpath exports both derive from it
(add the subpath in `scripts/build-sdk.mjs` ENGINE_SUBPATHS when SDK-ready).
Engines must be lazy-importable and safe to construct without WebGPU.

## 7. CI smoke

`scripts/ci-smoke-<id>.mjs`: WASM backend (CI has no GPU), weights from HF
(cache keyed on `scripts/ci-weights.lock`), assert real output words/values,
exit non-zero. Wire into `.github/workflows/ci.yml` engines job.

## Porting-scars index (details in git history / memory)

| Trap | Fix |
|---|---|
| WebGPU errors are async + silent | feature-gate `enable` directives; `uncapturederror` logs; fail-loud guards |
| adapterInfo subgroup sizes unreliable | `probeSubgroups()` runs a real dispatch |
| >8 storage buffers/stage | consolidate weights into one buffer + baked offsets (gpu-decoder.js) |
| barriers after data-dependent branch | `workgroupUniformLoad` |
| >16KB workgroup storage | request `maxComputeWorkgroupStorageSize` from adapter |
| tiny wasm/worklet assets | Blob-URL them (Vite inlines as `data:` URLs, which addModule/Worker reject) |
| isolated kernel benches lie | judge per-dispatch sums in context (`timestamp-query`), never wall-clock under dawn |
| node ESM in the SDK | explicit `.js` import extensions; `new URL(..., import.meta.url)` assets; no top-level await in workers |

## Worked next ports

- **Speaker verification**: extract wespeaker_v2 (ResNet embedding) → conv2d
  stack on the kernels → cosine enroll/verify engine. Closest reference:
  Sortformer (conv-heavy, small head).
- **Canary translation**: encoder = the SHARED FastConformer
  (raw-encoder.js handles Parakeet/Nemotron/EOU/Sortformer configs already —
  extract with `extract-fastconformer-encoder.py`); decoder = Whisper-shaped
  transformer with KV cache (raw-whisper-decoder.js is the template); target
  language = one prompt token (en=64, de=78, fr=71).
