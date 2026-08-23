# Golden reference contract

This directory defines the immutable inputs and tensor taps used to compare the
custom browser graph with the pinned ACE-Step 1.5 implementation. It contains
contracts and small metadata only. Checkpoints, tensor dumps, waveforms, and
spectrograms are local artifacts and must not be committed.

Run the weight-free contract check and its adversarial validator tests with:

```bash
uv run --python 3.13 python3 scripts/validate-golden.py
uv run --python 3.13 python3 -m unittest discover -s scripts -p 'test_*.py'
```

`fixture.schema.json` documents the standalone file shape. The validator is
the executable authority for cross-field geometry, profile resolution, hashes,
and manifest identity; JSON Schema alone cannot express every relationship. The
stdlib-only validator rejects duplicate JSON keys, non-finite numbers, and
unknown keys at every nested fixture level. `--update` refreshes hashes only
after the complete pinned generation and postprocess profile has passed.

## Fixture identity

Each `fixtures/*.json` file contains a `contract` object and its
`contractSha256`. The digest is SHA-256 of UTF-8 JSON serialized with sorted
object keys, no insignificant whitespace, and no ASCII escaping. The outer
`golden/MANIFEST.json` additionally pins every committed contract file byte for
byte. Changing a prompt, one lyric newline, a seed, or a sampler switch creates
a new identity; never edit expected outputs to make a candidate pass.

The four primary fixtures deliberately separate the important paths:

| Fixture | Coverage |
| --- | --- |
| `direct-instrumental-short` | direct Turbo, instrumental, short condition |
| `direct-lyrics-short` | direct Turbo, supplied English lyrics |
| `direct-lyrics-long-condition` | direct Turbo, near-limit lyric conditioning |
| `planner-lyrics-short` | 0.6B planner, two-row code CFG, supplied lyrics |

These are short correctness fixtures, not release-performance benchmarks. A
three-minute fixture should be added before Stage 1.5 memory acceptance, using
the same contract rather than replacing these inexpensive diagnostics.

## Determinism

`request.seed` is the canonical 64-bit lowercase hexadecimal seed. Golden capture does **not** rely
on PyTorch, MPS, JavaScript, or WGSL having interchangeable RNG algorithms.
Instead, the capture harness must use the random-word contract specified in
`prng/ace-seed-v1.json`:

- the `diffusion-noise` Philox domain supplies diffusion random words;
- the `planner-sampling` Philox domain supplies planner random words; and
- counter blocks are independently addressable, never implicit global state.

The transforms are pinned as part of `ace-seed-v1`:

- each diffusion word maps through
  `probit-acklam-binary64-f32-v1`, a fixed-order inverse-normal rational
  approximation whose log and square root contain no host transcendental
  calls, then rounds once to binary32;
- each planner word maps through `u32-midpoint-binary64-cdf-v1`; weights first
  round to binary32, then two ascending-token-ID binary64 passes select the
  first cumulative mass strictly above the midpoint variate; and
- the independent Python 3.13 oracle and TypeScript implementation both check
  every vector in `prng/ace-seed-v1-vectors.json`, including tail branches,
  counter/lane boundaries, zero weights, CDF boundaries, and stream-derived
  values.

The authoritative transforms run on the dedicated worker CPU. That location is
deliberate: WGSL has no portable binary64 arithmetic, and device-specific
transcendentals would make the initial tensor another backend variable. The
worker writes a single `Float32Array` and uploads it; it does not mirror model
weights or retain duplicate song-sized waveforms. A later WASM implementation
may replace the TypeScript execution engine only after it matches every bit in
the committed vectors.

Diffusion consumes exactly one word per flattened `[batch, latent-frame,
channel]` element. Planner draws are continuous across the CoT phase and then
semantic-code phase, ordered by autoregressive step and item. Every emitted
token consumes one word, even when constraints leave one candidate or the
token stops generation. Conditional and unconditional CFG model rows share the
one sampled token and do not consume separate words. Before that draw, planner
logits follow the pinned order: code-phase CFG combination, constrained FSM
mask, repetition penalty, top-k, top-p on untempered FP32 logits, temperature
division, and FP32 softmax. Descending-logit ties resolve by ascending token ID.
The resulting initial-noise tensor and raw planner words are injected into
native capture and hashed. An upstream integer seed alone is not cross-backend
provenance.

## Pinned behavior and explicit defaults

The source behavior is ACE-Step commit
`6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`, main-model snapshot
`19671f406d603126926c1b7e2adc169acbcade22`, and planner snapshot
`148d8ea0225bdab342ee1ae3a354275ccd60ca80`.

The v1 repository API pins the product Turbo path: shift 3, eight Euler ODE
evaluations, no DiT CFG/ADG, no Heun, and no velocity clamp or EMA. The generic
upstream `GenerationParams` dataclass still declares shift 1, but the pinned
Turbo model, Gradio model resolver, and release API resolve to shift 3. DCW is
enabled. Fixtures retain both the ideal resolved schedule and the effective
BF16 sampler schedule. Pinned Turbo first constructs a tensor with
`dtype=context_latents.dtype`, then calls `.item()` for the model timestep,
Euler delta, and DCW; the accepted native oracle therefore actually uses
`[1, 0.953125, 0.8984375, 0.83203125, 0.75, 0.64453125, 0.5,
0.30078125]`. Both browser package profiles preserve those coefficients.
Direct mode resolves to `low=0.05`, `high=0.02`; planner/Think mode
resolves to `low=0.02`, `high=0.06`. Haar DCW is a one-level, zero-padded FP32 temporal DWT
applied to `[B,T,C]` latents. It is output-affecting and therefore is part of
the primary contract, even though upstream comments still call it opt-in in a
few places. A browser Haar DWT/IDWT implementation and matching per-step taps
are required; silently falling back to a no-op is a correctness failure.

Planner-disabled fixtures explicitly turn off `thinking` and all CoT switches.
Otherwise the orchestrator can still invoke the language model merely to fill
metadata or rewrite the caption. The planner fixture intentionally leaves
optional metadata absent so default thinking/CoT is exercised, while keeping
constrained decoding and code-generation settings explicit.

## Capture layout

An actual capture is stored outside Git as:

```text
golden-local/<fixture-id>/<capture-id>/
├── capture.json
├── tensors/
│   ├── <tap-id>.bin
│   └── ...
├── output.wav
└── output.raw-f32le
```

`capture.json` must identify the fixture contract digest, source and model
revisions, exact tensor file length/SHA-256/dtype/shape, backend, package
versions, device, and whether the deterministic browser PRNG was injected.
Tensor payloads use contiguous little-endian storage. Captures lacking a
required tap or using implicit RNG are diagnostic only and cannot become an
accepted golden.

The tap requirements and shape symbols are in `taps.json`. Full tensors can be
large, so local captures may store either the full tensor or a deterministic
slice plus finite-value/statistical summaries as prescribed there. Hashes
recognize one exact reference capture; numerical acceptance still uses declared
per-stage tolerances rather than requiring a raw FP16 WebGPU result to hash to
the BF16/FP32 reference.

## Native capture status

The committed `reference/` harness now authenticates the pinned source, raw
model cache, browser package, fixtures, and deterministic input bundle; injects
the initial noise and planner raw words; installs the declared eager PyTorch
hooks; and writes captures transactionally with strict replay/compare support.
The next required step is to execute it on a supported CUDA/XPU BF16 host. No
native tensor or audio golden has been accepted yet.

An accepted first capture must:

1. check out the pinned ACE commit and verify all pinned model files;
2. force the eager PyTorch attention path and record all package versions;
3. bypass global RNG by injecting the declared initial noise and planner
   categorical words;
4. instrument conditioning, planner/detokenizer, every selected DiT boundary,
   DCW before/after values, final latent, VAE chunks, and raw pre-normalization
   waveform;
5. serialize tensors in the layout above and produce a hashed `capture.json`;
6. replay each capture once in a fresh process and prove identical artifact
   hashes; and
7. leave all large outputs under ignored `golden-local/` storage.

Known upstream wrinkles to test during capture:

- pinned upstream direct text-to-music computes a semantic hint from the
  silence latent through tokenizer/detokenizer, but the all-false `is_covers`
  selection discards it. The scoped browser graph may eliminate that dead
  computation, which is why semantic taps are planner-only;
- planner codes make the handler take its code-conditioned/cover-style context
  branch even though the product request began as text-to-music;
- planner single-item seeding currently touches PyTorch global RNG, so the
  harness must replace sampling rather than assume `manual_seed` parity;
- zero reference audio still traverses the timbre-conditioning path; and
- upstream DCW becomes a no-op if its optional wavelet dependency is absent,
  which is not an acceptable reference environment.
