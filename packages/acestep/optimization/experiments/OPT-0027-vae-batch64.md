# OPT-0027 — VAE batch64 submission

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Date: 2026-08-15
- Author/agent: Codex
- Risk class: `exact`

## Hypothesis

The current VAE uses eight graph quanta per command buffer, causing 11,338
decoder drains and about 11.3 seconds of requested queue-empty idle over C4500.
Encoding 64 sequential quanta per command buffer preserves graph order, buffer
aliasing, arithmetic, one-outstanding-buffer ownership, and the final readback,
while removing roughly seven eighths of the drain/idle boundaries. Based on the
current average batch duration, expected command-buffer work remains near
100 ms, keeping cancellation bounded.

## Identity

- Baseline commit: `84b35dc`
- Execution profile: package-native C512 VAE window, batch8 versus batch64
- Machine: MacBook Air M3, 10 GPU cores, 16 GB
- Browser/API: stock Chrome WebGPU

## Change

Parameterize or replace only the decoder quanta-per-command-buffer value. Do
not alter dispatches, bindings, arithmetic, windowing, or queue depth.

## Correctness gate

- Require the exact existing raw FP32 output hash/U32 contents.
- Verify cancellation, destroy, and map cleanup locally.
- Listening is unnecessary because GPU arithmetic and operation order do not change.

## Benchmark protocol

- One untimed warmup per batch size.
- Wait 30 seconds and require nominal thermal state immediately before timing.
- Measure the same package-native C512 fixture at the same stream/window boundary.
- Record wall, decoder submit-through-drain, command-buffer count, and requested idle.

## Results

- The stock-Chrome worker authenticated revision-6 manifest SHA-256
  `94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949`
  and built independent OPT-0028 exact-production backends using the native K7
  path. Both untimed warmups and all four timed executions produced accepted
  output SHA-256
  `893d7c7b3e2b389afbcbe781e76ee24d9f6cd29f90e88311447f26c49c07af47`.
- Each comparison covered `1,966,080` raw FP32/U32 words with zero mismatches
  and zero non-finite values. Runtime events remained empty.
- After the required 30-second nominal thermal gate (level 0), the one
  authorized balanced run measured:

| Order | Arm | Outer window | Decoder submit-through-drain | Decoder CBs | Drains | Requested idle |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | batch8 | 8,561.20 ms | 7,063.20 ms | 982 | 983 | 982 ms |
| 2 | batch64 | 6,188.60 ms | 5,997.80 ms | 123 | 124 | 123 ms |
| 3 | batch64 | 6,219.90 ms | 6,030.40 ms | 123 | 124 | 123 ms |
| 4 | batch8 | 8,397.40 ms | 6,904.60 ms | 982 | 983 | 982 ms |

| Pair / aggregate | Outer saving | Outer speedup | Decoder saving | Decoder speedup |
| --- | ---: | ---: | ---: | ---: |
| Forward, batch8 -> batch64 | 2,372.60 ms | 1.383382x | 1,065.40 ms | 1.177632x |
| Reverse, batch64 -> batch8 | 2,177.50 ms | 1.350086x | 874.20 ms | 1.144966x |
| **Two-sample means** | **2,275.05 ms** | **1.366692x** | **969.80 ms** | **1.161254x** |

- The batch64 mean reduced outer C512 wall from `8,479.30 ms` to
  `6,204.25 ms` (`26.83%`) and decoder submit-through-drain from `6,983.90 ms`
  to `6,014.10 ms` (`13.89%`). It removed 859 decoder command buffers, queue
  drains, and milliseconds of requested idle per C512 execution, an `87.47%`
  reduction in those scheduling counts.
- Cleanup passed: both backend owners, their independent weight phases,
  activation/readback buffers, and the shared device context were destroyed;
  destruction remained idempotent. Chrome reported no warning or error logs.

The canonical receipt is
[`optimization/results/OPT-0027/result.json`](../results/OPT-0027/result.json),
SHA-256 `c6ac3765961979196dbbb8ad4bf22d15f03eacd7859d6c3f0ed1e64b1d341b92`.

## Evidence and disposition

Positive benchmark-only evidence. Batch64 preserved the accepted raw output in
every warmup and measured execution and improved both outer-window and decoder
walls in both paired orders. The receipt deliberately declares no standalone
numeric performance threshold (`performanceGate: null`); its passed status is
the exactness/runtime/cleanup gate, while the paired measurements provide the
performance evidence. Promote batch64 through a separate production change
and full cancellation/responsiveness validation; this experiment itself does
not alter the production default or make a long-window/product-wall claim.
