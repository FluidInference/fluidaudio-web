# OPT-0025 — VAE K1 subgroup GEMM

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Date: 2026-08-15
- Author/agent: Codex
- Risk class: `exact`

## Hypothesis

The VAE K1 family is exactly `[frames,Cin] x [Cin,Cout] + bias`, but production
executes it through the portable 16-row x 8-channel shared-panel Conv1D kernel.
Adapting the already proven fixed32 FP16-operand/FP32-accumulation subgroup GEMM,
with K1 weights stored in its tile-major layout and the existing explicit FP16
output rounding, can remove most barriers and raise the current C4500 family
from 0.123 logical TFLOP/s. OPT-0007 tested the old shared-panel ownership, not
this mechanism.

## Identity

- Baseline commit: `51ced73`
- Execution profile: isolated paired browser A/B over representative production K1 shapes
- Machine: MacBook Air M3, 10 GPU cores, 16 GB
- Browser/API: stock Chrome WebGPU; `shader-f16`; fixed-size-32 subgroups

## Change

Add a benchmark-only K1 subgroup-GEMM kernel, deterministic tile-major weight
packing, and a small paired browser harness. Production selection is unchanged.

## Correctness gate

- Compare every output as raw FP16/U16 against production over all benchmark shapes.
- Require zero mismatches before timing is accepted.
- No listening gate until production/subsystem integration is proposed.

## Benchmark protocol

- One untimed warmup per shape/backend.
- Wait 30 seconds, verify nominal macOS thermal state, then run a balanced in-page A/B.
- Persist every sample; use browser wall timing at the same submit-to-drain boundary.
- Escalate to C512 only if the weighted primitive result projects a material C4500 saving.

## Results

The accepted actual-GPU run followed one warmup per arm and shape, then the
owner-requested 30-second wait with nominal macOS thermal state confirmed
immediately before launch. Compilation, allocation, tile-major packing, upload,
full-output comparison, and warmup were outside the paired timing interval.
Each cell below is the median of four balanced AB/BA browser-wall samples.

| Exact C512-window shape | Current (ms) | Candidate (ms) | Speedup | Candidate TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| 5,120 x 1,024 x 1,024 | 103.250 | 9.950 | 10.377x | 1.079 |
| 30,720 x 512 x 512 | 135.700 | 14.100 | 9.624x | 1.142 |
| 122,880 x 256 x 256 | 128.950 | 12.350 | 10.441x | 1.304 |
| 491,520 x 128 x 128 | 167.400 | 16.250 | 10.302x | 0.991 |
| 983,040 x 128 x 128 | 335.900 | 32.500 | 10.335x | 0.991 |

With each shape occurring three times, one exact C512 decoder window fell from
`2,613.600000500679 ms` to `255.44999957084656 ms`: `10.231356448978277x`,
with candidate aggregate throughput of `1.0718503252299403 TFLOP/s` over
`136,902,082,560` MACs. Scaling the authoritative OPT-0023 C4500 K1 wall by
that paired speedup projects `25,772.300002217293 ms` to
`2,518.9524117098827 ms`, a `23,253.34759050741 ms` saving.

All `241,172,480` raw FP16 outputs across the five full shapes matched the
current kernel bit-for-bit. Every per-shape mismatch count was zero, including
the explicit qNaN-prefill unwritten-output rejection check.

Durable receipt:

- `optimization/results/OPT-0025/result.json`
- SHA-256: `6b2a970a046a53d73c928840f8a345f07b251587ea70634789863b8e89566143`

## Evidence and disposition

This is a strongly positive primitive qualifier. The mechanism removes almost
the complete measured K1 bottleneck while preserving the current bias-first,
increasing-Cin FP32 reduction order and explicit FP16 output rounding.

Disposition remains `benchmark-only`: production still consumes native
`[Cout,1,Cin]` weights and selects the portable K1 Conv1D. Integration must add
the deterministic tile-major package layout and runtime selection, then verify
the complete package-native VAE sequence. No listening gate is required for an
exact integration unless that follow-up changes the arithmetic or precision
contract.
