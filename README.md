# FluidAudio Web

Local speech AI in the browser — ASR, TTS, VAD, and speaker diarization on
**hand-written WebGPU (WGSL) + WASM-SIMD kernels**. No onnxruntime-web, no
transformers.js, no server: model weights stream from Hugging Face on first
use, cache client-side, and everything runs on the visitor's machine. This is
the browser sibling of the Swift/CoreML
[FluidAudio](https://github.com/FluidInference/FluidAudio) framework.

**1 hour of audio transcribed in ~12 seconds — 293× real-time — in a Chrome
tab** (Parakeet TDT 0.6B v3, multilingual; verified across three runs on the
1-hour benchmark, Chrome/macOS/WebGPU; ~199× under the node harness).

**Live:** https://fluidaudio-web.hanweng9.workers.dev — playground (one engine
at a time) and [`/verify`](https://fluidaudio-web.hanweng9.workers.dev/verify)
(drop one file, selected engines run on it, results export as JSON).

> **Why hand-written kernels?** The first iteration of this repo ran the same
> models through onnxruntime-web. Rewriting the hot paths as raw WGSL + Rust
> WASM-SIMD (see [`docs/ORT_REMOVAL.md`](docs/ORT_REMOVAL.md) and
> [`docs/RAW_WEBGPU.md`](docs/RAW_WEBGPU.md)) took Parakeet from 33× to **100×+
> real-time in-browser** — batched-window encoding, f16 weight storage _and_
> f16 compute (2× ALU on Apple GPUs), a 3-stage GPU/CPU pipeline, and parallel
> RNNT decode on a Web Worker pool. Every optimization is gated on
> token-identical output.

## SDK

```bash
npm install @fluidinference/fluidaudio-web
```

```ts
import { ParakeetV3Engine } from "@fluidinference/fluidaudio-web/asr-parakeet";
import { decodeToMono16k } from "@fluidinference/fluidaudio-web";

const asr = new ParakeetV3Engine();
await asr.load((p) => console.log(p.file, p.fraction));
asr.setVocabulary(["NVIDIA", "Newrez"]); // optional: fuzzy-correct domain terms
asr.setItn(true); // optional: "twenty one" → "21"
const { text } = await asr.transcribe(await decodeToMono16k(fileArrayBuffer));
await asr.dispose();
```

One tree-shakeable subpath per engine — `/asr-parakeet`, `/asr-whisper`,
`/asr-nemotron`, `/tts-kokoro` (`{ lang: "en" | "zh" }`), `/vad-silero`,
`/diarization-sortformer`, `/eou-parakeet` — plus `/registry` (enumerate
engines, instantiate via `entry.make()`), `/textnorm`, and `/vocab-rescorer`.
Requires a bundler with `new URL(..., import.meta.url)` asset + module-worker
support (Vite, webpack 5 work out of the box). The demo site consumes the
identical source, so every site gate doubles as SDK regression coverage.

Release flow: bump `version` in the root `package.json` → `npm run sdk:pack` →
`cd dist-sdk && npm publish --access public`.

## Engines

Measured in-browser (Chrome/macOS, WebGPU, warm) on a real 284.5s recording via
`/verify` — not a lab clip. RTFx = audio-seconds per wall-second (for TTS:
audio _generated_ per wall-second; not comparable to ASR).

| Engine                   | Model                             | RTFx                                                                                                      | Notes                                                                                                                                                                                      |
| ------------------------ | --------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `asr-whisper`            | Whisper (99 langs)                | re-measuring                                                                                              | KV-cached decode, f16 weights; long-form chunking just landed (the prior 114× was measured on the first-30s-only bug and is retracted)                                                     |
| `asr-parakeet`           | Parakeet TDT 0.6B v3              | **282×** (1hr file)                                                                                       | 2.15% WER LibriSpeech test-clean (core parity-gated vs the reference); worker-pool RNNT decode; opt-in ITN + custom vocabulary                                                             |
| `vad-silero`             | Silero VAD v5                     | 79×                                                                                                       | WASM-SIMD (tiny sequential model by design)                                                                                                                                                |
| `diarization-sortformer` | NVIDIA Sortformer 4-spk           | 79×                                                                                                       | windowed with 24-permutation overlap stitching                                                                                                                                             |
| `tts-kokoro`             | Kokoro 82M (en + zh)              | 4.7× en / 5.6× zh                                                                                         | waveform corr ~0.97 vs reference; en input auto-normalized ("$4.50" is spoken, not dropped)                                                                                                |
| `asr-nemotron`           | Nemotron 3.5 streaming (40 langs) | realtime+                                                                                                 | cache-aware streaming RNNT                                                                                                                                                                 |
| `eou-parakeet`           | Parakeet EOU 120M                 | **297×** browser-verified (1hr in 12.1s; worker-overlapped wasm decode + linear-cost stream-batch encode) | transcript + end-of-utterance events; TRUE streaming push()/finish() (bit-exact cache-carrying encode) + wasm-SIMD RNNT decode; whole-clip batch runs through the same linear-cost encoder |

First (cold) run is several× slower — WebGPU compiles pipelines and weights
download once. WebGPU is optional: every engine falls back to the same math on
WASM-SIMD (slower on the big encoders, identical outputs — cross-backend
parity is CI-gated). History and methodology: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Text processing (WASM)

[`text-processing-rs`](https://github.com/FluidInference/text-processing-rs)
vendored as a 1 MB wasm module (pure Rust, no network):

- **TN** (written → spoken) runs automatically on English Kokoro input:
  numbers/currency/times aren't in the G2P lexicon and used to be silently
  dropped from the audio.
- **ITN** (spoken → written, `"i paid four dollars and fifty cents"` →
  `"i paid $4.50"`) is **opt-in** (`setItn(true)` / the playground checkbox) —
  on everyday speech it also rewrites phrases like "no one" → "no 1".

## Quick start (repo)

```bash
npm install
npm run dev        # http://localhost:5173 — playground; /verify.html for all-engine runs
npm run build      # static site → dist/
npm run sdk:pack   # publishable SDK tarball (dist-sdk/ + .tgz in repo root)
```

`/verify.html` params: `?engines=asr-parakeet,vad-silero` preselects the
checkboxes (all engines run by default), `?noauto=1` skips the JSON
auto-download. "Keep models loaded between runs" makes repeat file drops
instant at the cost of GPU memory.

## Deploy

`main` auto-deploys to **Cloudflare Workers** (static assets, see
`wrangler.jsonc`) via the connected Workers Builds integration — merge and it's
live. Manual: `npm run build && npx wrangler deploy`.

Deliberately **no COOP/COEP**: cross-origin isolation would break the
cross-origin Hugging Face weight fetches, and nothing here needs
`SharedArrayBuffer` — parallelism comes from WebGPU and the decode worker pool
(each worker gets its own weight copy).

## Layout

```
src/
  gpu/          the kernel library: WGSL GEMM/conv/attention/LSTM (compute.js),
                WASM-SIMD twin (wasm-context.js) — one interface, two backends
  engines/      one folder per model on those kernels; registry.ts is the catalog
  core/         audio I/O, model cache, text normalization, shared types
  index.ts      SDK root (engines are subpath exports)
  main.ts / verify.ts   the two demo pages (thin consumers of the registry)
scripts/        node gates: token-identity, kernel parity, per-engine smokes
rust/           parakeet RNNT decoder + kernel lib sources (wasm32+simd128)
docs/           architecture, benchmarks, PORTING.md (add-a-model checklist), the ORT removal story
```

## Hard-won lessons (things that cost real debugging)

- **Wall-clock lies under dawn/node; only `timestamp-query` tells the truth.**
  Every kernel "benchmark" read ~2 ms/op until per-dispatch GPU timestamps
  showed the real distribution — several optimization verdicts flipped.
- **WebGPU errors are async and silent.** A missing `shader-f16` feature
  request turned every f16 GEMM into a no-op: empty transcripts at a
  fake-fast RTFx. Feature-gate every `enable` directive and log
  `uncapturederror`.
- **Synchronous WASM starves microtasks.** A `.then()` holding a GPU readback
  couldn't fire while a 190 ms decode blocked the thread — the GPU idled after
  every batch. Staging copies must ride the producing submit.
- **Measure on the target machine.** The dev box was CPU-bound where user
  machines were GPU-bound and vice versa; the per-stage split in the metrics
  (`mel / encode / decode`) exists because RTFx alone misdiagnosed both.
- **f16 storage ≠ f16 compute.** Halving weight bytes did nothing on a
  compute-bound GPU; switching the inner loop to f16 fma (f32 accumulate per
  8-deep K-tile) bought 1.46× with token-identical output.
- **ITN is not a free win.** English inverse normalization rewrites "no one" →
  "no 1" and deletes words in other languages — it shipped opt-in only because
  a review pass ran the wasm on realistic sentences.
- **Gate on tokens, not maxΔ.** Every perf change here ships with a
  token-identity / parity gate; two of them caught real kernel breakage that
  numeric thresholds would have argued about.

## License

MIT (code). Model weights follow their upstream licenses (see the registry and
Hugging Face model cards).
