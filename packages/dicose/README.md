# DiCoSe WebGPU

An interactive and automation-ready browser runtime for the released **[DiCoSe BS-RoFormer + one-step
consistency-distilled (CD) refinement](https://arxiv.org/abs/2412.06965)**, using the
[official model weights](https://huggingface.co/karchkha/DiCoSe). The neural graph is raw WGSL/WebGPU:
f16 storage, f32 reductions, converter-native tile-major subgroup GEMM, fused
online attention with producer-rotated K, RMSNorm, FiLM, Conv2d STFT
conditioning, complex masks, and CD affine sampling.
The CPU boundary is intentionally limited to WAV decoding, deterministic
resampling, centered Hann STFT/ISTFT, seeded noise generation, and the final
instrumental complement subtraction.

The public API lives in `src/index.ts`. Browser inference runs in a dedicated
worker and transfers PCM/result buffers instead of blocking the page thread.

## Inference modes

`new DiCoSeWorkerClient()` and `separateAudio(source)` keep the released
full-resolution, one-step refined graph as the default. Fast is an explicit
quality/performance tradeoff that returns the deterministic separator before
CD refinement:

```ts
const fast = new DiCoSeWorkerClient();
const fastResult = await fast.separateAudio(source, {
  outputMode: "deterministic",
});
```

Both modes return the four neural estimates under `result.stems` and a derived
`result.instrumental`. Instrumental is computed as the decoded input mixture
minus the vocal estimate after both have been restored to the uploaded file's
sample rate and exact frame count. It adds no model pass and is deliberately
not computed by summing drums, bass, and other.

On the supplied WAV, deterministic-only had a 5.92-s sustained median. That
number remains useful as a performance measurement, not quality evidence.
Fast uses the released deterministic checkpoint but omits learned refinement.
See `optimization/CORRECTNESS_AUDIT.md` and `optimization/LEDGER.md` for the
current evidence and dispositions.

## Model package

Large checkpoints, download caches, and the generated weight blob are ignored
by Git. Run this command manually whenever the local browser package needs to
be prepared:

```sh
pnpm model:prepare
```

It downloads the two pinned official checkpoints, verifies them, converts the
exact Full/Fast production package into `public/model/`, and verifies the
canonical generated hashes. The source download is about 4.66 GB and is cached
under `model/cache/`. The command requires `uv`; Python 3.13 and all converter
dependencies come from the locked `model/` environment.

## Run locally

```sh
pnpm dev
```

Open `http://127.0.0.1:5173/`, choose or drop a local WAV, select Full or Fast,
and run the separation. The page shows stage timings and
creates the four model stems plus a derived instrumental as five in-memory
stereo WAVs with playback and download controls.
The source file and generated outputs stay in the browser tab; they are not
uploaded. Inputs above 12 seconds are processed as fixed 11-second model items
with reflected context and normalized overlap-add. Full retains the upstream
50% overlap policy. Fast overlaps only the existing 10% fade region. For the
5,608,109-sample model-rate `trust_nobody.wav` input, that changes the plan from
25 chunks in Full to 13 in Fast. A fresh isolated-Chrome sustained panel measured
a 79.61-s median (71.57–113.05 s) for that Fast path; this does not meet the
30-second target, and listening remains the quality gate. Long tracks still
require serial model calls. File-based runs restore each output to the uploaded
WAV's sample rate and exact frame count before playback/download.

For an unattended page invocation, add `?autorun=1`; the result is published
to `window.__DICOSE_BROWSER__.report` and `#result`. `?mode=benchmark` uses a
single persistent worker/model package across its warmup and measured runs.
Neither path opens a save dialog, download, or UI control.

## Checks and isolated browser testing

```sh
pnpm check
pnpm test
pnpm test:reference-quality
pnpm test:refined-reference-quality
pnpm test:output-mode-quality
pnpm test:webgpu
pnpm test:browser
pnpm benchmark:browser
```

The release benchmark accepts an explicit output selector:

```sh
DICOSE_BENCHMARK_OUTPUT_MODE=deterministic pnpm benchmark:browser
```

The browser scripts automatically start Vite and a new headless Chrome process
with a freshly-created temporary `--user-data-dir`, then delete that profile,
stop Chrome, and stop Vite in `finally`. They use CDP to await the automatic
result; no user profile, click, permission prompt, or file save is involved.
`test:browser` additionally enforces the fixture's f16 deterministic-output
envelope against the upstream f32 reference. `test:reference-quality` checks
the released deterministic graph and 30 internal tensor seams;
`test:refined-reference-quality` checks the released one-step CD graph, 17
internal CD seams, its raw model output, and the final refined stems against a
fixed-noise execution of the official PyTorch implementation.

`Mixture_audio_1.wav` is the supplied 22.05 kHz mono fixture. Production decode
duplicates it to stereo and uses the Hann-windowed sinc geometry and defaults
from torchaudio 2.0.2 before processing 1,189 centered-STFT frames. The
deterministic model-arithmetic oracle deliberately replays its older frozen
linear input tensor so resampler and neural-graph regressions remain separate
gates. The CD sampler uses a fixed default noise seed, so an otherwise
identical run is reproducible.
