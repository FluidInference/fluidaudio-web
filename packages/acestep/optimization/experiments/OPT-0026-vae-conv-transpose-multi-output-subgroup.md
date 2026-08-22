# OPT-0026 — VAE ConvTranspose multi-output subgroup

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Date: 2026-08-15
- Author/agent: Codex
- Risk class: `exact`

## Hypothesis

OPT-0022 removed barriers but assigned each lane one output channel with sixteen
row accumulators, limiting input reuse and regressing one important block. A
polyphase kernel in which each fixed32 subgroup owns multiple rows and 128
adjacent output channels, with each lane producing four adjacent channels, can
reuse each input across more outputs while retaining the production
phase/tap/Cin FP32 term order and explicit FP16 rounding.

## Identity

- Baseline commit: `51ced73`
- Execution profile: isolated paired browser A/B over all five production transpose blocks
- Machine: MacBook Air M3, 10 GPU cores, 16 GB
- Browser/API: stock Chrome WebGPU; `shader-f16`; fixed-size-32 subgroups

## Change

Add a benchmark-only packed-polyphase layout, K7-style multi-output subgroup
kernel, inverse-layout verification, and paired browser harness. Production
selection is unchanged.

## Correctness gate

- Verify packed-weight inversion exactly.
- Compare every output as raw FP16/U16 against production across all five block shapes.
- Require zero mismatches before timing is accepted.
- No listening gate until production/subsystem integration is proposed.

## Benchmark protocol

- One untimed warmup per shape/backend.
- Wait 30 seconds, verify nominal macOS thermal state, then run a balanced in-page A/B.
- Persist every sample; use browser wall timing at the same submit-to-drain boundary.
- Escalate only if the weighted result projects a material C4500 saving.

## Results

- Stock Chrome/WebGPU accepted all five generated shaders on the fixed32 Apple
  M3 adapter.
- The native O-K-I -> polyphase -> native round trip compared all `49,610,752`
  weight U16 words with zero mismatches.
- The paired full-shape gate compared all `141,312,000` output U16 words with
  zero mismatches and no remaining output prefill values.
- Four alternating A/B rounds measured these submit-through-matching-drain
  medians:

| Block | Current | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| 0, C2048 -> C1024, stride 10 | 245.40 ms | 74.60 ms | 3.2895x |
| 1, C1024 -> C512, stride 6 | 349.70 ms | 113.25 ms | 3.0879x |
| 2, C512 -> C256, stride 4 | 344.60 ms | 103.50 ms | 3.3295x |
| 3, C256 -> C128, stride 4 | 380.65 ms | 116.50 ms | 3.2674x |
| 4, C128 -> C128, stride 2 | 385.35 ms | 119.80 ms | 3.2166x |
| **Summed medians** | **1,705.70 ms** | **527.65 ms** | **3.232635x** |

The result receipt is
[`optimization/results/OPT-0026/result.json`](../results/OPT-0026/result.json).

The receipt's simple C300-times-15 arithmetic projects `17,670.75 ms` saved.
That is not the preferred long-song estimate because the current authoritative
C4500 family capture measured ConvTranspose at `42,401.0000 ms`, rather than
the C300 result's `25,585.5000 ms` linear projection. Applying only the
measured `3.232635x` primitive ratio to that authoritative wall projects a
`13,116.5431 ms` candidate family and `29,284.4569 ms` saving. This is planning
arithmetic, not a long-window observation.

## Evidence and disposition

Positive benchmark-only evidence. The materially different ownership fixed
OPT-0022's main reuse problem: every block improved by at least `3.0878x`, raw
FP16 output remained bit-exact, and the authoritative-family projection is
large enough to warrant converter/package and production-integration work.
This experiment itself changes no production selection and makes no
long-window, waveform, listening, product-wall, or under-one-minute claim.
