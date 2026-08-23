# ace-step-1.5.wgsl

An in-progress, fully custom WebGPU and WebAssembly browser port of ACE-Step
1.5 Turbo.

The project aims to generate high-quality music locally on consumer Apple
Silicon without a server or generic browser ML runtime. It is inspired by
[`parakeet.wgsl`](https://github.com/narcotic-sh/parakeet.wgsl), including its
streamed model packaging, bounded GPU memory, specialized WGSL kernels, worker
runtime, and queue-drained cooperative GPU scheduling.

## Project status

Stage 1's browser correctness and human-listening gate passed on 2026-08-13.
The complete, untuned browser pipeline produced owner-approved instrumental and
vocal songs, including the default-CoT planner path. Stage 2 measured
optimization is now authorized; this remains a development baseline rather than
a release.

The first foundation includes the pinned request/sampler and worker contracts,
browser capability reporting, browser-defined Gaussian and categorical random
transforms, authenticated golden-fixture contracts, and a deterministic
model-package converter. The runtime authenticates manifest bytes before strict
parsing, validates the canonical package schema, streams pinned assets through
bounded transactions, provides an integrity-aware resumable OPFS primitive,
and owns GPU graphs with Parakeet-style FIFO cooperative submission. GPU weight
uploads additionally enforce a 64 MiB queued-write high-water mark, drain the
queue before crossing it, and insert a real queue-empty interval between
non-final upload batches.

The untuned correctness kernels now cover GEMM, RMSNorm, authenticated Qwen
RoPE, score-buffer-free full/sliding/causal GQA attention, KV-cache writes,
shard-aware embeddings, transformer and DiT plumbing, fixed FSQ decoding, the
pinned Haar DCW update, and the complete FP32 Oobleck VAE primitive family.
Correctness-first graph composers cover the 28-layer text encoder, Qwen cached
blocks, semantic detokenizer, direct lyric/silence-timbre conditioner, all 24
DiT layers and sampler operations, and the 88-operation VAE decoder. The VAE
decoder is split into bounded FIFO output-domain quanta; even a single large
convolution cannot become one GPU-monopolizing dispatch.

The planner control path reproduces left-padded one/two-row prefill, bounded KV
capacity, deterministic browser-owned filtering/sampling, the normal two-phase
prompt workflow, a PyYAML-compatible output subset, and the pinned metadata
constraint state machine. Independent Python vectors authenticate its field
tries, language choice, caption mask, and upstream's unusual early reasoning
termination. Small CPU-oracle cases for every current graph family pass in
Chrome/WebGPU on the development M3 in both package profiles. Concrete
whole-model owners now connect authenticated model acquisition,
planner-optional conditioning, all eight DiT evaluations, chunked VAE decode,
transactional OPFS audio, and cooperative/cancellable scheduling. Native
production-weight taps remain unclaimed external parity evidence; the browser
implementation is not allowed to serve as its own numerical oracle. The
owner-approved direct and planner-enabled reference-profile WAVs and their
exact settings are recorded in
[`LISTENING_CANDIDATE.md`](./LISTENING_CANDIDATE.md).

The M3 actual-browser safety harness also exercises the largest current tiled
GEMM shape with exact sentinels while continuously checking a UI heartbeat.
Those runs establish bounded cooperative dispatch behavior only; they are not
Stage 2 performance benchmarks.

The implementation program is in [`PLAN.md`](./PLAN.md). Its central rule is:

1. implement and numerically validate the complete scoped browser pipeline;
2. stop for a human listening test; and
3. begin performance optimization only after explicit approval.

The listening packet records measured M3 correctness-run latency. Stage 2 now
replaces those untuned measurements with thermally controlled baselines and a
durable experiment ledger. The headline M3 target is a warm-cache three-minute
song in under one minute without leaving the approved quality, memory, or
responsiveness envelope. M5 performance will be tested later on actual hardware
rather than claimed from scaling alone.

## Initial scope

- ACE-Step 1.5 Turbo, eight denoising evaluations
- text prompts and supplied lyrics
- optional full ACE 5 Hz 0.6B thinking/semantic planner workflow
- direct generation without the planner
- semantic-code FSQ projection and learned detokenizer for the planner path
- upstream-default Haar DCW correction in the eight-step Euler sampler
- default -1 dBFS peak normalization
- stereo 48 kHz WAV output
- Chrome on desktop Apple Silicon first
- uncompressed, phase-staged weights first

Weight compression, larger planners, source-audio workflows, Safari tuning,
and iPhone support are later projects.

The pinned direct, no-cover path evaluates a silence acoustic-tokenizer branch
upstream and then discards it through an all-false `is_covers` selection. The
browser package removes that dead encode-only branch exactly; cover/source
audio remains outside v1 scope.

## Source identities

The correctness baseline is pinned to:

- ACE-Step source commit
  `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`;
- `ACE-Step/Ace-Step1.5` revision
  `19671f406d603126926c1b7e2adc169acbcade22`; and
- `ACE-Step/acestep-5Hz-lm-0.6B` revision
  `148d8ea0225bdab342ee1ae3a354275ccd60ca80`.

The browser model packages are generated reproducibly by `model/convert.py`.
`--profile production` downloads and creates the exact three-package tuple used
by the demo, failing closed unless all committed manifest identities match.
Model weights, conversion caches, and generated audio do not belong in Git.
The initial uncompressed profiles and current production tuple have canonical
manifest hashes recorded in
[`model/canonical-packages.json`](./model/canonical-packages.json).

## Development

The browser-side project uses the pnpm version pinned in `package.json`:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm test
pnpm build
```

Generate the current production weights and run the browser demo at
`http://127.0.0.1:5174/demo/`:

```bash
uv run --frozen --project model --python 3.13 \
  python3 model/convert.py --profile production
pnpm exec vite --host 127.0.0.1 --port 5174
```

The demo defaults to the short direct/instrumental golden request, authenticates
the canonical packed-BF16 reference manifest, uses the cooperative scheduling
profile, and keeps generated audio in transactional OPFS storage until the UI
releases it. The raw-FP16 profile is an explicitly lower-precision diagnostic
until it passes the numerical and listening gates required by `AGENTS.md`.

The model workspace is a frozen uv project on CPython 3.13. Its complete
source-to-package commands and disk requirements are documented in
[`model/README.md`](./model/README.md). The weight-free checks are:

```bash
uv run --python 3.13 python3 scripts/validate-golden.py
uv run --python 3.13 python3 -m unittest discover \
  -s scripts -p 'test_*.py' -v
uv run --frozen --project model --python 3.13 \
  python3 -m unittest discover -s model/tests -v
uv run --frozen --project reference --python 3.13 \
  python3 -m unittest discover -s reference/tests -v
```

The actual-browser kernel harness is documented in
[`test/browser/README.md`](./test/browser/README.md). Passing it proves only the
small declared correctness cases; it is not a performance benchmark.

## License

The original source in this repository is MIT licensed. ACE-Step and Qwen
artifacts retain their respective upstream terms and attribution; see
[`THIRD_PARTY_LICENSES`](./THIRD_PARTY_LICENSES).
