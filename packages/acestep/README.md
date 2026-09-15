# ace-step-1.5.wgsl

A custom ACE-Step 1.5 Turbo browser runtime by Hamza Qayyum, using WebGPU and
WebAssembly. It powers [FluidAudio Web's music page](../../music.html).

## Features and requirements

- Text prompts, optional lyrics, and deterministic seeds.
- Direct generation or an optional 0.6B planner for metadata and semantic codes.
- Eight denoising evaluations with the pinned Haar DCW sampler.
- Stereo 48 kHz WAV output, normalized to −1 dBFS.
- Inference in a dedicated worker, with progress and cancellation.
- Authenticated model downloads and persistent OPFS caching.

Desktop Chrome on Apple Silicon is the primary validation target. The
production profile requires WebGPU with `shader-f16`; subgroup optimizations
have a portable fallback. Source-audio editing, cover generation, and the VAE
encoder are outside the current scope. Support for phones and other browsers
must be validated on the target device.

The FluidAudio music page uses direct generation and a **5.75 GB** production
model cache. Its optional packed INT8 preview uses **4.43 GB**. The planner is
disabled. The package's development demo also exposes the
planner and uses a different reference manifest. The site's selected model
packages are defined in [config.ts](../../src/engines/musicgen-acestep/config.ts).

## Status and validation

The owner approved the initial browser listening results on August 13, 2026,
and authorized measured optimization. The project remains in development.
Approved audio identities are recorded in [LISTENING_CANDIDATE.md](LISTENING_CANDIDATE.md).

Unit tests and browser kernel checks validate their declared cases; they do
not establish whole-model equivalence to the native upstream implementation.
Native CUDA/XPU captures remain required before making that claim.

[PLAN.md](PLAN.md) defines release criteria, including reliable operation on
the 16 GB M3, stock-Chrome validation, cancellation and cleanup, and a warm-cache
three-minute song in under one minute. Initial download time is measured
separately. See the [optimization ledger](optimization/LEDGER.md) for experiment
results and integration status.

## Build and test

From the FluidAudio Web repository root:

```bash
npm ci
npm run acestep:check
npm run acestep:test
npm run acestep:build
```

These checks do not require downloading the model. For actual GPU validation,
follow the [browser test guide](test/browser/README.md).

To run the public music UI, follow the [root quick start](../../README.md#run-locally).
To prepare weights and run the package's development demo, use `uv` with the
locked Python 3.13 environment. From `packages/acestep/`:

```bash
uv run --frozen --project model --python 3.13 \
  python3 model/convert.py --profile production
npm exec -- vite --host 127.0.0.1 --port 5174
```

Open `http://127.0.0.1:5174/demo/`. See [model preparation](model/README.md) for
disk requirements and package layout. Generated weights, audio, caches, and
large profiles stay out of Git.

## Implementation and source identities

The pipeline combines a Qwen3 text encoder, ACE condition encoder, optional
planner and semantic detokenizer, 24-layer DiT, and Oobleck VAE decoder. Weights
are loaded by phase and audio is decoded in chunks to bound memory use. GPU
submissions are bounded to support cancellation and keep the UI responsive.

The correctness baseline is pinned to:

- ACE-Step source: `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`
- Main model: `19671f406d603126926c1b7e2adc169acbcade22`
- 0.6B planner: `148d8ea0225bdab342ee1ae3a354275ccd60ca80`

[Canonical package hashes](model/canonical-packages.json) identify reproducible
converter outputs. Read [AGENTS.md](AGENTS.md) before changing kernels,
precision, scheduling, or model packaging; it defines the required numerical,
listening, and performance checks.

## License

Original code is MIT licensed. ACE-Step and Qwen artifacts retain their upstream
terms. See [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
