# Architecture

## Goal

Run FluidAudio's core models in the browser with **WebGPU + WebAssembly**, no
server. This is a parallel deployment of the *same models* FluidAudio ships as
CoreML — but from their **ONNX** source, because CoreML has no browser runtime.

## Layers

```
UI (main.ts)  →  Engine interface (core/types.ts)  →  runtime
                                                        ├─ onnxruntime-web  (ASR)
                                                        ├─ kokoro-js        (TTS)
                                                        ├─ @ricky0123/vad-web (VAD)
                                                        └─ sherpa-onnx WASM (diarization)
core/ort.ts        backend selection (WebGPU→WASM), threads/SIMD
core/modelCache.ts HF fetch + Cache API persistence + byte progress
core/audio.ts      file→16kHz mono float, resample, WAV encode
core/registry.ts   ONNX repos/files/quant per engine
```

Everything the UI touches is one of the interfaces in `core/types.ts`
(`AsrEngine`, `StreamingAsrEngine`, `DiarizationEngine`, `TtsEngine`,
`VadEngine`). Adding a model = add a folder under `src/engines/` implementing one
of them + a `registry.ts` entry.

## Backend policy (the important part)

`onnxruntime-web`'s WebGPU EP does **not** have kernels for every op. Dynamic-shape
graphs (RNNT/TDT decoders) fall back to CPU per-op, with GPU↔CPU syncs that can
make WebGPU *slower than WASM-int8* (parakeet.js measured ~15× on some devices).
So the rule encoded in `core/ort.ts`:

- **Encoders** (heavy, mostly static) → request `webgpu` (falls back to wasm).
- **Decoders / joints** (tiny, dynamic, stepwise) → request `wasm` directly.

Cross-origin isolation (COOP/COEP) is required for threaded WASM — set in
`vite.config.ts` for dev; **any production host must send the same headers.**

## Per-engine status & notes

### ✅ `vad-silero` — wired
Wraps `@ricky0123/vad-web` (`NonRealTimeVAD`). The library bundles the Silero
ONNX + wasm. Nothing to port.

### ✅ `tts-kokoro` (en/zh) — wired, zh caveat
Wraps `kokoro-js` (transformers.js; WebGPU with WASM fallback). English is
complete. **Chinese's open item is G2P**: the ONNX is acoustic-only; robust
Mandarin needs a JS frontend (jieba segmentation → polyphone disambiguation →
pinyin→IPA), the browser analog of FluidAudio's separate `g2pW` CoreML model.
Options: port g2pW to ONNX and run it via ORT, or use a JS pinyin lib + a
polyphone table. Until then `zh` relies on kokoro-js's own handling.

### 🚧 `asr-parakeet` (v3) — scaffold
Sessions + registry wired. Remaining: mel frontend + TDT greedy decode. Both are
pure math in FluidAudio (`AudioMelSpectrogram`, `TdtDecoderV3.swift`) — port to
TS. Encoder on WebGPU, decoder+joint on WASM. `parakeet.js` is a working
reference (69★ — read it, don't depend on it).

### 🚧 `asr-nemotron` (en + multilingual) — scaffold
ONNX published: `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`
(40 langs, INT4, ~750 MB). Cache-aware streaming: the encoder carries
`cache_last_channel` / `cache_last_time` / `cache_last_channel_len` between
chunks. Remaining: initialize caches from encoder input metadata, thread them
through `push()`, and run the RNNT greedy loop. Reference:
`khawjaahmad/nemotron-asr-webgpu` (1★ POC).

### 🚧 `diarization-pyannote` — scaffold (use sherpa-onnx, not raw ORT)
`sherpa-onnx` (13.9k★) ships a full offline diarization pipeline compiled to
WASM (`build-wasm-simd-speaker-diarization.sh`,
`SherpaOnnxCreateOfflineSpeakerDiarization`): pyannote segmentation-3.0 +
embedding + clustering. Vendor that Emscripten bundle under `public/sherpa/` and
load it in the engine. FluidAudio uses `wespeaker_v2` embeddings; sherpa ships
3D-Speaker/NeMo — swap the embedding ONNX if you want parity. This is the direct
Senko-Web equivalent.

### ⛔ `eou-parakeet` — greenfield
No public ONNX export of `parakeet-realtime-eou-120m`. Export the streaming EOU
encoder + decision head from NeMo first, then mirror `asr-nemotron` (same
cache-aware family; the model is only 120M).

## Suggested build order

1. **Diarization** — highest leverage per effort (sherpa-onnx does the work; direct Senko parity).
2. **Parakeet v3** — flagship ASR; port mel + TDT loop.
3. **Nemotron** — biggest "wow" (live, 40 langs); cache plumbing is the work.
4. **Kokoro-zh G2P** — unlocks the Chinese TTS demo.
5. **EOU** — after its ONNX export exists.

## Hosting

Static site. Must send `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` (GitHub Pages needs a workaround;
Cloudflare Pages / Netlify can set headers directly). Model weights stream from
Hugging Face and cache client-side; no backend.
