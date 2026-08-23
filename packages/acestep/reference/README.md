# Native reference capture

This workspace turns the pinned ACE-Step implementation into an authenticated,
replayable numerical oracle. It never treats a PyTorch integer seed as
interchangeable with the browser RNG. Accepted captures receive realized
initial Gaussian noise and raw planner words from an external, fixture-pinned
input bundle, run the eager PyTorch model, and serialize the taps declared in
`../golden/taps.json`.

The planner injector implements `u32-midpoint-binary64-cdf-v1` itself. It
consumes one raw uint32 word for every emitted token, continuously across CoT
and semantic generation, including forced and stop tokens. Semantic-logit
probes gather the model's token IDs for `<|audio_code_0|>` through
`<|audio_code_63999|>` in code-value order; these are not vocabulary IDs
`0:64000` in the pinned 217,204-token planner.
Top-k retains the pinned upstream threshold semantics (every tie at the kth
logit survives). Top-p is patched in memory to use descending logits with
ascending token IDs for ties before the upstream shift-one cutoff; captured
logit probes therefore remain the post-filter values actually sampled.

No model code is patched on disk. `instrumentation.py` installs in-memory hooks
at the concrete source sites listed in `upstream-bindings.json`. The capture is
assembled in a hidden partial directory, hash-replayed in place, and promoted
to `golden-local/<fixture>/<capture-id>/` only after every required tap and both
audio artifacts validate.

## Two Python environments

The repository-owned preflight/replay tooling is locked to CPython 3.13:

```bash
uv run --frozen --project reference --python 3.13 \
  python3 -m unittest discover -s reference/tests -v

uv run --frozen --project reference --python 3.13 \
  python3 -m reference.capture preflight \
  --fixture direct-instrumental-short \
  --source /tmp/ace-step-1.5-source
```

The pinned ACE source declares `requires-python = ">=3.11,<3.13"`. The actual
model process must therefore use ACE's own frozen uv lock with CPython 3.12.
That exception is an upstream dependency constraint, not an unlocked fallback.
The capture records the concrete Python, Torch, Transformers, Diffusers,
PyWavelets, and pytorch-wavelets versions.

## Fail-closed preflight

Preflight refuses:

- any ACE checkout not exactly at
  `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`, including a dirty checkout;
- a mutable or abbreviated model revision;
- a raw source artifact whose size differs from `model/source_contract.py`;
- a browser reference package whose manifest differs from
  `model/canonical-packages.json`;
- a golden fixture or tap contract whose committed manifest identity changed;
- an input bundle belonging to another fixture/contract or containing a wrong
  byte length, hash, shape, dtype, non-finite value, or PRNG-oracle mismatch;
- CPU/MPS native execution, because the current tap contract is the upstream
  BF16 reference path; and
- Flash Attention, SDPA, MLX, compilation, quantization, inactive Haar DCW, or
  implicit PyTorch sampling/noise.

The execution view links each declared source-contract file individually.
Parent checkpoint directories are never exposed to upstream loading, so an
extra or symlinked cache file cannot become an unrecorded model input.

The fast preflight checks all schemas, revisions, manifest hashes, file sizes,
and complete file inventories. `--deep-payload-hashes` additionally streams
and hashes every multi-gigabyte raw-model and browser-package payload. Deep
verification is mandatory for `run`.

## Prepare external random inputs

The committed mappings and cross-language vectors are now pinned. Materialize
the realized input bundle with the repository's frozen CPython 3.13 tool; the
loader regenerates every value from `ace-seed-v1` before authorizing it:

```bash
uv run --frozen --project reference --python 3.13 \
  python3 -m reference.capture prepare-inputs \
  --fixture direct-instrumental-short

uv run --frozen --project reference --python 3.13 \
  python3 -m reference.capture prepare-inputs \
  --fixture planner-lyrics-short \
  --planner-word-capacity 4096
```

The ignored layout, described by `input.schema.json`, is:

```text
golden-local/inputs/<fixture-id>/
├── inputs.json
├── initial-noise.f32le
└── planner-words.u32le  # planner only; capacity spans CoT, codes, and EOS
```

Run preflight locally on the M3 without loading Torch:

```bash
uv run --frozen --project reference --python 3.13 \
  python3 -m reference.capture preflight \
  --fixture direct-instrumental-short \
  --inputs golden-local/inputs/direct-instrumental-short/inputs.json \
  --source /tmp/ace-step-1.5-source
```

The exact full-capture command on a supported CUDA/XPU BF16 machine is:

```bash
uv run --frozen --project /tmp/ace-step-1.5-source --python 3.12 \
  python3 -m reference.capture run \
  --fixture direct-instrumental-short \
  --inputs golden-local/inputs/direct-instrumental-short/inputs.json \
  --source /tmp/ace-step-1.5-source \
  --model-cache model/cache \
  --browser-package model/files-reference \
  --deep-payload-hashes \
  --device cuda \
  --capture-id native-cuda-run-01
```

`xpu` is also accepted when the pinned upstream BF16 stack is available.
Capture on this M3 is intentionally limited to preflight/replay because the
pinned PyTorch MPS path runs these models in FP32 while the golden tap contract
is BF16. The browser implementation itself remains targeted at the M3.

## Replay and fresh-process proof

Replay checks strict JSON, exact identities, every file length and SHA-256,
required tap inventory, the committed fixture/tap/golden contract, the hashed
reference tool, source and deep-verified browser package, deterministic input
manifest/payload mappings, exact runtime environment, eager/DCW receipts, and
the absence of unlisted files. There is no unauthenticated replay mode:

```bash
uv run --frozen --project reference --python 3.13 \
  python3 -m reference.capture verify \
  golden-local/direct-instrumental-short/native-cuda-run-01 \
  --fixture direct-instrumental-short \
  --environment-contract \
    golden-local/direct-instrumental-short/native-cuda-run-01.environment.json
```

Run the same fixture and input bundle again in a new process under another
capture ID, then require identical artifact identities:

```bash
uv run --frozen --project reference --python 3.13 \
  python3 -m reference.capture compare \
  golden-local/direct-instrumental-short/native-cuda-run-01 \
  golden-local/direct-instrumental-short/native-cuda-run-02 \
  --fixture direct-instrumental-short \
  --environment-contract \
    golden-local/direct-instrumental-short/native-cuda-run-01.environment.json
```

The two `capture.json` files may record different capture IDs, but every tensor,
raw-audio, WAV, fixture, source, package, and injected-random identity must
match.

Accepted capture forces VAE parameters and floating buffers to FP32 and checks
them again at decode. Tiled decode is pinned to `chunk_size=256`, `overlap=64`,
and `offload_wav_to_cpu=True`; live free-VRAM heuristics are not provenance.

`run` writes the capture atomically, then writes a sibling
`<capture-id>.environment.json` review candidate. It records the exact locked
package versions, accelerator name/capability/memory, CUDA or XPU runtime,
driver, Python, and platform. Review and preserve that file separately before
using it as `--environment-contract`; that independently preserved, reviewed
receipt is the replay root of trust. Replay never derives the trusted
environment from the capture it is checking, so capture-only environment
tampering fails. The automatically written receipt is not signed and does not
by itself prevent a coordinated rewrite of both files.
