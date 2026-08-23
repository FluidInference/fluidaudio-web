# FluidAudio Web

Local speech AI in the browser — ASR, TTS, VAD, speaker diarization, and
music generation on **hand-written WebGPU (WGSL) + WASM-SIMD kernels**. No onnxruntime-web, no
transformers.js, no server: model weights stream from Hugging Face on first
use, cache client-side, and everything runs on the visitor's machine. This is
the browser sibling of the Swift/CoreML
[FluidAudio](https://github.com/FluidInference/FluidAudio) framework.

**1 hour of audio transcribed in ~12 seconds — 293× real-time — in a Chrome
tab** (Parakeet TDT 0.6B v3, multilingual; verified across three runs on the
1-hour benchmark, Chrome/macOS/WebGPU; ~199× under the node harness).

**Live:** https://fluidaudio-web.hanweng9.workers.dev — playground (one engine
at a time), [`/live`](https://fluidaudio-web.hanweng9.workers.dev/live.html)
captions, and [`/music`](https://fluidaudio-web.hanweng9.workers.dev/music.html)
generation.

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
const { text } = await asr.transcribe(await decodeToMono16k(fileArrayBuffer), {
  // optional: transcription progress on long files, emitted at window boundaries
  onProgress: (p) => console.log(`${(p.fraction * 100).toFixed(0)}% — ${p.processedSeconds.toFixed(0)}s / ${p.totalSeconds.toFixed(0)}s`),
});
await asr.dispose();
```

True streaming (EOU / Nemotron) and captions (v0.2.0):

```ts
import { MicCapture, segmentsToSrt } from "@fluidinference/fluidaudio-web";
import { ParakeetEouEngine } from "@fluidinference/fluidaudio-web/eou-parakeet";

const engine = new ParakeetEouEngine();
await engine.load();

// live: feed mic chunks, get cumulative text; <EOU> events + word segments
const mic = new MicCapture();
await mic.start();
let pos = 0;
setInterval(async () => {
  const { samples, total } = mic.since(pos);
  const text = await engine.push(samples); // conformer caches carried — no re-decode
  pos = total;
  console.log(text, engine.streamEvents, engine.streamSegments);
}, 300);
// on stop: const final = await engine.finish(); engine.reset();

// batch: word timestamps → SRT captions
const r = await engine.transcribe({ samples, sampleRate: 16000 });
const srt = segmentsToSrt(r.segments); // also: segmentsToVtt, groupCues
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
the since-removed verify page — not a lab clip. RTFx = audio-seconds per wall-second (for TTS:
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
| `asr-voicechat`          | VoiceChat-11B STT (609M encoder)  | 34.6× (1hr file)                                                                                          | the speech-recognition slice of NVIDIA's full-duplex VoiceChat-11B; fully-causal per-frame streaming, parity byte-identical to the torch reference; weights hosted at [`FluidInference/fluidaudio-web`](https://huggingface.co/FluidInference/fluidaudio-web) like the other engines |
| `musicgen-acestep`       | ACE-Step 1.5 Turbo (3.5B + VAE)   | ~1.9× (180s song in ~95s, M3, warm)                                                                       | full text-to-music on [`/music`](music.html): 8-step DiT + Oobleck VAE in pure WGSL (`packages/acestep`); ~5.7 GB one-time download; requires `shader-f16`; direct mode (optional planner LLM path exists upstream, still being optimized) |
| `tts-voicechat`          | VoiceChat-11B TTS “Aria” (595M backbone + 159M MoG + 763M codec) | ~0.19× node/WASM (185 ms/frame AR loop; codec 0.33×) | the speech-decoder slice of NVIDIA's full-duplex VoiceChat-11B as a standalone TTS voice; audio codes bit-exact vs the torch reference, waveform NRMSE 1.3e-6; local-only weights (`scripts/extract-voicechat-tts.py`, ~3.5 GB) — hidden from the picker unless exported; GPU-resident decode loop deferred |

First (cold) run is several× slower — WebGPU compiles pipelines and weights
download once. WebGPU is optional: every engine falls back to the same math on
WASM-SIMD (slower on the big encoders, identical outputs — cross-backend
parity is CI-gated). History and methodology: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Music generation (ACE-Step)

[`/music.html`](music.html) generates full songs — prompt, optional lyrics, up
to 4 minutes, stereo 48 kHz WAV — entirely client-side. The runtime is
[`packages/acestep`](packages/acestep/), a vendored npm-workspace import of
ace-step-1.5.wgsl by Hamza Qayyum (upstream repo not yet public — his live
demo is at [acestep.narcotic.sh](https://acestep.narcotic.sh); the vendored
source lives in this repo):
~100k lines of TypeScript + WGSL implementing the Qwen3 text encoder, ACE
condition encoder, 24-layer DiT, and Oobleck VAE decoder, with authenticated
streamed model packaging, bounded GPU memory, and cooperative scheduling. It
keeps its own rigorous experiment ledger (`packages/acestep/optimization/`) —
read `packages/acestep/AGENTS.md` before touching kernels there.

The ~5.7 GB of content-addressed model packages currently stream from the
upstream author's public R2 bucket and cache in OPFS; set
`VITE_ACE_MODEL_ORIGIN` to point at a mirror or locally staged packages
(`packages/acestep/model/convert.py --profile production` reproduces the
exact tuple). The optional 0.6B planner ("thinking") path is excluded from
the served manifest until its pending optimization experiments
(OPT-0084/0085/0087) are integrated.

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
npm run dev        # http://localhost:5173 — playground; /live.html, /music.html
npm run build      # static site → dist/
npm run sdk:pack   # publishable SDK tarball (dist-sdk/ + .tgz in repo root)

npm run acestep:check && npm run acestep:test   # ACE-Step runtime (packages/acestep) gates
```

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
  main.ts / live.ts / music.ts   demo pages (thin consumers of the registry / music client)
packages/
  acestep/      vendored ace-step-1.5.wgsl music-gen runtime (own kernels,
                scheduler, tests, and optimization ledger — see its AGENTS.md)
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

## Acknowledgements

The encoder GEMM kernel geometry and the GPU TDT decoder design are adapted
from [parakeet.wgsl](https://github.com/narcotic-sh/parakeet.wgsl) by
Narcotic Software (MIT) — a fast, focused browser Parakeet implementation
that served as both inspiration and reference throughout our optimization
work.

Music generation is built on
ace-step-1.5.wgsl ([live demo](https://acestep.narcotic.sh); upstream repo
not yet public — the full source is vendored at `packages/acestep`) by
**Hamza Qayyum** (Narcotic Software, MIT): he built the complete ACE-Step
1.5 Turbo browser port — correctness-gated WGSL kernels, model packaging,
scheduling, and the Stage-2 optimization program — and handed the project
over for integration here; we took it over, integrated, and are continuing
the optimization work. The `packages/acestep` runtime and the `/music` page's
backend seam are his code.

See [THIRD-PARTY-LICENSES.md](./THIRD-PARTY-LICENSES.md) for the full
list of adapted code and licenses.
