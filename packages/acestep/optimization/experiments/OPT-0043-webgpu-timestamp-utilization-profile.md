# OPT-0043 — WebGPU timestamp and utilization profile

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Risk: diagnostic only; no production arithmetic or selector change

## Hypothesis

Completion-fenced wall time conflates shader execution, command submission,
queue scheduling, callbacks, and requested idle. The stock M3 Chrome adapter
reports the standard `timestamp-query` feature, while macOS exposes rolling AGX
device/renderer/tiler utilization through read-only I/O Registry statistics.
Measuring both around short balanced primitive screens can distinguish an
under-filled/CPU-bound queue from a busy GPU executing low-throughput kernels.

## Frozen scope

- Request only the adapter-reported standard `timestamp-query`, `shader-f16`,
  and subgroup features; no browser flag, native Metal/MPS, or WebNN.
- Instrument unchanged OPT-0009 versus OPT-0032 dense dispatches on the four
  M2250 production shapes. Put timestamp writes immediately around the compute
  pass and resolve/copy results outside the dispatch output.
- Report GPU elapsed time, submit-to-drain wall, their ratio, valid MAC/s, and
  scheduled MAC/s in balanced order. Sample read-only AGX utilization at a
  coarse external cadence and keep it outside every timed interval.
- Reuse existing correctness fixtures and require the same complete finite
  output/numerical gates. No timing result changes OPT-0032's integration
  authority.

## Decision use

If GPU timestamp time closely tracks wall while AGX utilization is high, the
remaining gap is kernel arithmetic/memory efficiency and submission tuning
cannot supply it. If timestamp time is materially below wall or utilization is
low, prioritize graph coalescing, larger independent work batches, or host
overhead. This profile is diagnostic and authorizes no production change by
itself.

## Result

The diagnostic passed on the stock M3 Chrome adapter. One untimed warmup per
arm and full shape preceded two balanced timing rounds. Each sample used one
command buffer, one matching queue drain, and one timestamp-query pair that
bounded the compute pass exactly; output readback was outside timing. The
production-weighted `4/2/2/1` result was:

| Arm | GPU time | Submit-to-drain wall | GPU / wall | Valid GPU throughput | Valid wall throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| OPT-0009 | 191.823872 ms | 211.650000 ms | 0.906326 | 1.37752 TFLOP/s | 1.24848 TFLOP/s |
| OPT-0032 K4 | 139.624448 ms | 153.650000 ms | 0.908718 | 1.89251 TFLOP/s | 1.71976 TFLOP/s |

K4 improved timestamped GPU execution by `1.373856x` and submit-to-drain wall
by `1.377481x`. The nearly identical speedups, with about `91%` of weighted
wall inside the timestamped compute pass for both arms, are direct evidence
that these dense dispatches are GPU-kernel-bound. Only `19.8261 ms` for the
control and `14.0256 ms` for K4 sat outside the timestamped passes. Even
eliminating that entire measured remainder would therefore improve either arm
by only about `1.10x`; submission and queue tuning cannot close the product
target by themselves. Further dense gains must come primarily from kernel
arithmetic, data movement, dependency-chain, and occupancy efficiency.

The external I/O Registry receipt sampled device/renderer/tiler rolling
utilization at `46/43/44%` before browser timing and `46/45/46%` after it.
Those coarse values are outside the browser timing fences and describe a
rolling system statistic, so they are contextual only; they do not override
the direct timestamp-to-wall result.

Correctness remained unchanged from OPT-0032: all `25,344,000` full-shape
outputs and `17,408` adversarial outputs were finite, complete, deterministic,
and within their declared numerical envelopes. Full-shape NRMSE was
`0.0003114215`, SNR `70.1330 dB`, Pearson `0.9999999524`, and maximum absolute
error `0.0144317`; there were no uncaptured GPU errors. Cleanup destroyed all
50 created buffers, left zero live buffers/bytes, and was idempotent.

This is positive diagnostic evidence only. It neither changes OPT-0032's
benchmark-only status nor authorizes a package, runtime selector,
trajectory, listening, or production change.

Receipts:

- [`optimization/results/OPT-0043/result.json`](../results/OPT-0043/result.json),
  SHA-256 `83f6ad1045d6d65fbebbe6ac86197d18bef682ea5e38b148c56e2d75abb7fcb7`.
- [`optimization/results/OPT-0043/external-agx.json`](../results/OPT-0043/external-agx.json),
  SHA-256 `8592ee84f603649f944887754758fc803955070bf36b7324dcdc67aecb264588`.
