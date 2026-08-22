# OPT-0068 Phase-1 native dense gate

This is a standalone benchmark-only Swift/MPS owner.  It does not modify or
select the browser runtime.  It answers one bounded question: can native MPS
execute the nine actual layer-0 M2250 dense operations, including their
required FP32 output and materialization boundary, fast enough to justify a
complete native prototype?

The repository currently has authenticated revision-7 weights but no actual
dense activation/output fixture.  `--mode describe` works now; every other
mode fails closed until `CAPTURE_RECIPE.md` has been completed.  Synthetic
adversarial tensors are correctness screens only and can never become timed
throughput operands.

The missing artifact is exactly six actual FP32 activations, nine accepted
WebGPU FP32 outputs, and the frozen evaluation-0 result.  The fixture generator
and both inspectors reject wrong request/package/evaluation authority,
non-finite data, all-zero data, reused paths, wrong byte hashes, and mismatched
finite/nonzero/range/head/tail evidence.  There is no capture URL today; the
precise isolated browser seam that must exist first is documented in
`CAPTURE_RECIPE.md`.

## Static checks

These commands compile but never instantiate a Metal device because tests do
not invoke the executable:

```sh
swift test --package-path benchmark/opt-0068-native-metal
swift build -c release --package-path benchmark/opt-0068-native-metal
swift run --package-path benchmark/opt-0068-native-metal ace-opt-0068-mps --mode describe
uv run --python 3.13 python -m py_compile benchmark/opt-0068-native-metal/mlx_runner.py
uv run --python 3.13 python -m py_compile benchmark/opt-0068-native-metal/make_fixture_manifest.py
uv run --python 3.13 python -m unittest discover \
  -s benchmark/opt-0068-native-metal -p 'test_*.py'
jq empty benchmark/opt-0068-native-metal/fixture.schema.json
```

## Accepted-run sequence

Use an executable built from a pushed commit.  Finish fixture authentication,
weight repacking, allocations, compilation, independent full CPU contracts,
and warmups before the thermal gate.  GPU execution requires the literal
consent token, preventing an inspection command from accidentally becoming a
benchmark.

```sh
swift run -c release --package-path benchmark/opt-0068-native-metal ace-opt-0068-mps \
  --mode measure \
  --package-dir model/files-fp16-dit-rev7-oracle \
  --fixture-manifest benchmark/artifacts/OPT-0068/actual-m2250/fixture.json \
  --harness-commit OPT0068_COMMIT_SHA \
  --harness-source-root benchmark/opt-0068-native-metal \
  --full-cpu-contract \
  --warmups 2 --samples 5 \
  --thermal-trace benchmark/results/OPT-0068/raw/native-B1-thermal.jsonl \
  --output benchmark/results/OPT-0068/raw/native-B1.json \
  --execute-native-gpu I_UNDERSTAND_OPT_0068_BENCHMARK_ONLY
```

The executable enforces `Mac15,12`, 16 GiB, macOS 26.5.2 build 25F84, then
requires at least 30 continuous seconds of raw thermal level 0.  Polling stays
active through output completion and release of case-owned MPS matrices,
buffers, queue, and device.  Every raw observation is synchronously appended;
the receipt embeds all observations plus the raw trace hash.

Run fresh-process, independently cooled arms in the frozen
WebGPU/native/native/WebGPU order.  The unchanged WebGPU arms remain the
accepted OPT-0009 owner and use the same captured inputs/weights/outputs.  Do
not substitute an old browser median.  The native weighted wall must improve
both paired directions by at least 1.40x and sustain at least 2.40 TFLOP/s.
The native receipt evaluates only the 2.40-TFLOP/s half and deliberately leaves
the paired-speedup fields null/false; closeout may mark the complete dense gate
only after authenticating both independently cooled WebGPU arm receipts.

The separate sustained arm is:

```sh
swift run -c release --package-path benchmark/opt-0068-native-metal ace-opt-0068-mps \
  --mode sustained \
  --package-dir model/files-fp16-dit-rev7-oracle \
  --fixture-manifest benchmark/artifacts/OPT-0068/actual-m2250/fixture.json \
  --harness-commit OPT0068_COMMIT_SHA \
  --harness-source-root benchmark/opt-0068-native-metal \
  --full-cpu-contract --warmups 2 --samples 1 \
  --thermal-trace benchmark/results/OPT-0068/raw/native-sustained-thermal.jsonl \
  --output benchmark/results/OPT-0068/raw/native-sustained.json \
  --execute-native-gpu I_UNDERSTAND_OPT_0068_BENCHMARK_ONLY
```

It repeats the complete 4/2/2/1 mix for at least 60 seconds and reports
five-second intervals.  The final nominal third must retain at least 80% of
the first nominal third.

## Arithmetic and limits

- Inputs are captured FP32 activations rounded once to IEEE FP16 before the
  gate.  Weights are authenticated revision-7 FP16 tensors unpacked from the
  converter-native N256/K32 layout before the gate.
- `MPSMatrixMultiplication` receives FP16 A/B and a requested FP32 C.  Apple's
  API does not expose accumulator selection.  Runtime support and FP32-like
  behavior are therefore premises tested by the full output, CPU, signed-zero,
  cancellation, finite-range, long-K, tail, determinism, complete-write, and
  canary screens.  An unsupported mixed result or screen failure rejects MPS.
  The long-K accumulation probe additionally requires maximum absolute error
  below 0.01; OPT-0009's forbidden native-FP16 accumulator missed by 0.15625,
  so this leaves a 15.6x separation while permitting a different FP32 tree.
  Zero/nonzero class changes are forbidden independently of signed-zero changes.
- Full actual-output CPU comparisons use Accelerate `vDSP_mmul`: the exact
  FP16 operands are widened to FP32 and reduced by an independent CPU FP32
  tree.  The smaller adversarial probes retain a strict scalar source-K-order
  FP32 oracle, so the library and reduction-order premise is not self-tested.
- ACE repeated-layer dense primitives have no bias or activation epilogue;
  receipts report that stage explicitly as absent.  A private-FP32-to-guarded-
  shared blit is included in the authoritative wall and timed separately in a
  diagnostic stage pass.
- A dense pass authorizes only the complete native prototype and later gates
  in the OPT-0068 record.  It is not complete-evaluation, VAE, listening,
  source-to-WAV, under-one-minute, or product-selection evidence.

## Optional MLX corroboration

`mlx_runner.py` authenticates the same fixture before importing MLX and uses
the same 30-second/60-second thermal protocol.  Run it only in a separate,
independently cooled process:

```sh
uv run --python 3.13 --with mlx==0.32.0 --with numpy==2.3.2 \
  python benchmark/opt-0068-native-metal/mlx_runner.py \
  --mode measure \
  --package-dir model/files-fp16-dit-rev7-oracle \
  --fixture-manifest benchmark/artifacts/OPT-0068/actual-m2250/fixture.json \
  --harness-commit OPT0068_COMMIT_SHA --warmups 2 --samples 5 \
  --thermal-trace benchmark/results/OPT-0068/raw/mlx-thermal.jsonl \
  --output benchmark/results/OPT-0068/raw/mlx.json \
  --execute-native-gpu I_UNDERSTAND_OPT_0068_BENCHMARK_ONLY
```

This arm is permanently labelled diagnostic: MLX 0.32's FP16 matmul exposes
an FP16 result before the explicit FP32 cast, does not expose raw GPU command
timestamps through its Python API, and does not duplicate the expensive full
actual-output scalar CPU contract.  It can corroborate a mechanism or reject
itself numerically; it cannot satisfy the primary OPT-0068 gate.
