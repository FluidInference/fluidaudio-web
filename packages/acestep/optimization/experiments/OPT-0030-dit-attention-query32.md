# OPT-0030 — DiT query16/query32 attention

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Date: 2026-08-15
- Risk: reordered-rounding, same as integrated query8

## Hypothesis

Query8 was capped at a 256-thread device request, while the stock M3 WebGPU
adapter exposes 1,024 compute invocations per workgroup. A 512-lane query16 or
1,024-lane query32 workgroup can share each staged 128-wide K/V row across 16
or 32 attention streams, reducing query8's K/V traffic by another 2x or 4x.
Each stream retains the same subgroup QK reduction and ascending-key FP32
online-softmax update as query8.

## Gate

At exact M2250 full attention, compare query8/query16/query32 after one
30-second nominal thermal gate. Require finite complete output, query8-envelope
numerics, and at least 1.35x over query8. Promote the fastest passing geometry
through the existing trajectory/listening gate.

## Identity

- Allocation baseline: `72de722`
- Browser/API: stock Chrome WebGPU, no experimental flags
- Machine: MacBook Air M3, 10 GPU cores, 16 GB

## Results

- Stock Chrome accepted and dispatched all three full-shape shaders, including
  the 1,024-invocation Q32 workgroup. The Apple `metal-3` adapter reported
  fixed-32 subgroups and a 1,024-invocation/workgroup limit.
- Full M2250 output comparison covered `4,608,000` FP32 elements per
  candidate. Q16 and Q32 both matched Q8 bit-for-bit: zero mismatches, zero
  non-finite values, zero maximum absolute error, and the shared output SHA-256
  `2882495eccd1f1971e998b957dfdd12ee517a2019a6ddccd9d76e38bb81c9892`.
- After the required 30-second nominal thermal gate (level 0), the one
  authorized balanced timing run produced:

| Round order | Q8 | Q16 | Q32 |
| --- | ---: | ---: | ---: |
| Q8, Q16, Q32 | 139.60 ms | 219.20 ms | 173.00 ms |
| Q16, Q32, Q8 | 135.60 ms | 222.70 ms | 220.70 ms |
| Q32, Q8, Q16 | 140.80 ms | 202.60 ms | 167.90 ms |
| **Median** | **139.60 ms** | **219.20 ms** | **173.00 ms** |

- Q16 achieved only `0.636861x` Q8 throughput and Q32 only `0.806936x`.
  Although Q32 was the faster candidate, its median was `1.239255x` Q8's
  latency, missing the required `1.35x` speedup decisively.
- The page destroyed all six owned buffers and the device. Chrome reported no
  warning or error logs after completion.

The canonical receipt is
[`optimization/results/OPT-0030/result.json`](../results/OPT-0030/result.json),
SHA-256 `92e2c193ff4ea2f6a6429a1406970d2a32c42b0f23326873c38d9ec9e875c8a2`.

## Evidence and disposition

Negative evidence. On this fixed32 Apple M3 adapter, cutting modeled K/V loads
by 2x or 4x with 512- or 1,024-lane workgroups did not overcome the cost of the
larger workgroups; both candidates regressed exact full-shape latency. Do not
integrate either geometry. Revisit only for a materially different attention
ownership/algorithm, compiler/backend behavior, or GPU architecture—not an
unchanged rerun of Q16/Q32.
