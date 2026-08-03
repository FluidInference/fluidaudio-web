# Architecture

## Goal

Run FluidAudio's core models in the browser with **WebGPU + WebAssembly**, no
server. This is a parallel deployment of the *same models* FluidAudio ships as
CoreML — but from their **ONNX** source, because CoreML has no browser runtime.

## Layers

```
UI (main.ts)  →  Engine interface (core/types.ts)  →  runtime
                                                        ├─ onnxruntime-web  (ASR, VAD, EOU, diarization)
                                                        ├─ kokoro-js        (TTS)
                                                        └─ transformers.js  (Whisper)
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

### ✅ `vad-silero` — reimplemented on `core/ort`
Drives `silero_vad.onnx` (v5, `onnx-community/silero-vad`) directly — **no**
`@ricky0123/vad-web`. That package is CJS and does a dynamic
`require("onnxruntime-web/wasm")` that Vite can't resolve once ORT is excluded
from `optimizeDeps` (which it must be — see backend policy). Silero's interface is
trivial — `input[1,512]` + `state[2,1,128]` + `sr` (int64 scalar) → speech prob +
`stateN` — so `silero.js` runs 512-sample (32 ms) windows, thresholds with
hysteresis, and merges with min-speech / min-silence / pad guards. WASM only (the
model is 2 MB; WebGPU buys nothing). Verified: `scripts/smoke-vad.mjs`.

### ✅ `tts-kokoro` (en/zh) — wired, zh caveat
Wraps `kokoro-js` (transformers.js; WebGPU with WASM fallback). English is
complete. **Chinese's open item is G2P**: the ONNX is acoustic-only; robust
Mandarin needs a JS frontend (jieba segmentation → polyphone disambiguation →
pinyin→IPA), the browser analog of FluidAudio's separate `g2pW` CoreML model.
Options: port g2pW to ONNX and run it via ORT, or use a JS pinyin lib + a
polyphone table. Until then `zh` relies on kokoro-js's own handling.

### ✅ `asr-parakeet` (v3) — internalized (no ASR library), all-ORT compute
Fully in-repo, and **all feature extraction + inference runs on onnxruntime-web
(WebGPU/WASM)** — no heavy JS DSP:
- mel: `nemo128.onnx` (the NeMo log-mel preprocessor, **per-feature CMVN**) run on
  ORT **WASM** (`onnxMel.js`) — replaces the earlier JS FFT.
- encoder: `encoder-model.onnx` (**fp32**, external `.data`) on **WebGPU**.
- decoder+joint: `decoder_joint-model.int8.onnx` on **WASM**.
- `tokenizer.js` (vocab.txt + SentencePiece decode) and the TDT greedy loop in
  `tdt.js` are JS **orchestration** (argmax over logits, loop control, string
  decode) — scalar glue, not tensor compute.

The exact same core runs in the browser engine (`index.ts`, onnxruntime-web) and
the headless verifier (`scripts/smoke-parakeet-internal.mjs`, onnxruntime-node) —
no `parakeet.js` dependency. Repo: `ysdede/parakeet-tdt-0.6b-v3-onnx`.

**Verified transcribing headlessly** (fp32 encoder, ort-node):
`node scripts/smoke-parakeet-internal.mjs /tmp/pk_intro.wav /tmp/pkv3 fp32`
→ "Four Classes That Constitute a Menace from Anti-Suffrage Ten Good Reasons by
Grace Duffield Goodwin" (41 tokens, 0.24 s).

**Why fp32, not int8.** The WebGPU EP has no int8 kernels, so an int8 encoder
silently falls back to WASM — where *this* encoder is numerically degenerate: its
output collapses to ~0 (measured std **0.017** vs ~O(1) healthy) and every frame
decodes to blank → empty transcript. So the engine ships the **fp32** encoder
(`encoder-model.onnx` + `.data`, ~2.4 GB) on WebGPU (throws if WebGPU is absent)
and the int8 decoder on WASM. The mel is confirmed correct independently
(per-feature CMVN output: mean 0, std 1). Full LibriSpeech test-clean → **2.15%
WER**, matching native FluidAudio.

### ✅ `asr-nemotron` (en + multilingual) — working (int4 on WASM)
`onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4` (40 langs, INT4,
~750 MB). Cache-aware streaming: the encoder threads `cache_last_channel` /
`cache_last_time` / `cache_last_channel_len` across 65-frame chunks (9 pre-encode
cache + 56 new), then an RNNT greedy loop over 7 encoder frames. NA log-mel (no
CMVN, unlike Parakeet). int4 is **healthy on WASM** (encoder std 0.43) so no
WebGPU is needed. The bug that looked like int4 degeneracy was `lang_id=0` =
Bulgarian — en-US is ordinal **24** (`makeNemotronLangMap`). See
[`NEMOTRON.md`](NEMOTRON.md). Verified: `scripts/smoke-nemotron.mjs`.

### ✅ `diarization-sortformer` — working (short audio)
NVIDIA `diar_streaming_sortformer_4spk-v2.1` ONNX (fp32, CPU-runnable). Offline
single-chunk path: `chunk[1,T,128]` mel + empty `spkcache`/`fifo` state →
per-frame 4-speaker probs → segments (123× RTFx). Reuses the `nemo128` mel.
Long meetings need the streaming `spkcache`/`fifo` state loop (single-chunk
collapses to the dominant speaker — see [`BENCHMARKS.md`](BENCHMARKS.md)); that's
the open follow-up. (`diarization-pyannote` via sherpa-onnx WASM remains an
alternative if you want the pyannote pipeline.)

### ✅ `eou-parakeet` — working (transcript + `<EOU>`/`<EOB>`)
`ysdede/parakeet-realtime-eou-120m-v1-onnx` (asrjs export of NVIDIA
`parakeet_realtime_eou_120m-v1`). Streaming FastConformer RNNT with `<EOU>`/`<EOB>`
control tokens; fused `decoder_joint`, single-layer LSTM, blank id 1026. Offline
RNNT greedy decode (last-1027 slice of the joint grid); ids ≥ 1024 become
timestamped events, dropped from the transcript. **Wants NA (un-normalized)
log-mel** — the Nemotron frontend, NOT Parakeet's per-feature CMVN; wrong
normalization → content-free encoder frames → all-blank. fp32 encoder decodes on
both WASM and WebGPU (no int8-collapse). See [`EOU.md`](EOU.md). Verified:
`scripts/smoke-eou.mjs`.

## Status

All seven engines are correctness-verified on real data — see the model matrix in
the [README](../README.md) and numbers in [`BENCHMARKS.md`](BENCHMARKS.md). Open
follow-ups are enhancements, not blockers: Sortformer's streaming state loop for
long meetings, true streaming `push()` for Nemotron/EOU (their exports here are
whole-clip), g2pW polyphone accuracy for Kokoro-zh, and fp16 encoders to shrink
downloads.

## Hosting

Static site. Must send `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` (GitHub Pages needs a workaround;
Cloudflare Pages / Netlify can set headers directly). Model weights stream from
Hugging Face and cache client-side; no backend.
