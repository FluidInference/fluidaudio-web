# OPT-0035 — VAE C2378 two-window decode

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Risk: exact kernels with quality-affecting chunk-boundary geometry

## Hypothesis

The C512/overlap64 C4500 plan decodes twelve windows and `5,908` latent
frames. With the same 64-frame context, chunk `2,378` has stride `2,250` and
is the smallest chunk that covers C4500 in two windows. The planner produces
two balanced C2314 windows and decodes only `4,628` latent frames: `21.67%`
less convolution work, ten fewer readbacks/window boundaries, and much less
submission overhead.

The exact decoder plan scales linearly. A C2378 FP16 workspace binding is
`1,168,834,560` bytes; three workspaces use `3,506,503,680` bytes. Including
the revision-6 VAE package, ingress, output, and readback keeps expected GPU
residency below 4 GB. The stock M3 adapter advertises
`maxBufferSize=maxStorageBufferBindingSize=4,294,967,292`, so this requires no
experimental browser feature. Allocation success and observed live bytes are
still gates, not assumptions.

## Frozen mechanism

- Change only VAE chunk size from 512 to 2,378; keep overlap 64, hop 1,920,
  exact revision-6 K1/ConvTranspose owners, native exact K7 for the first
  performance screen, and the selected submission batch unchanged.
- For C4500 require exactly two windows: latent `[0,2314)` and
  `[2186,4500)`, cores `[0,2250)` and `[2250,4500)`, with one 64-frame suffix
  and prefix discard respectively.
- Allocate bounded GPU buffers once before timing. Never materialize a second
  full waveform in JS memory and never overlap DiT/VAE residency.
- Request only the adapter's ordinary advertised WebGPU limits. Fail cleanly
  on request/allocation failure; do not retry with hidden flags or another API.
- Preserve every decoder dispatch's arithmetic. The only possible output
  difference comes from fewer chunk boundaries and longer context.

## Gate

First preflight device request, all allocations, exact planner/coverage,
finite output, redzones where applicable, and cleanup. Compare the stitched
C2378 raw waveform with the accepted C512 output using per-channel NRMSE, SNR,
Pearson correlation, peak, seam neighborhoods, determinism, and a listenable
artifact. After one 30-second nominal check, measure complete C4500 VAE wall
and peak live bytes once. Positive evidence requires clean completion, at
least `1.15x` wall speedup, and the existing waveform envelope; owner listening
approval remains mandatory before production selection. Combine with
approximate K7 only in a later separately measured joint gate.

## Result

The authenticated sequential `C512, C2378, C2378, C512` run completed, but
its complete-wall timing is **inconclusive**, not stable negative evidence.
The result receipt mechanically records `status: "failed"` because the
aggregate speedup missed the frozen `1.15x` gate; the underlying four arms do
not provide a stable thermal comparator from which to attribute that miss to
window geometry.

| Sequence | Arm | Complete C4500 VAE wall |
| --- | --- | ---: |
| 1 | C512 | `81,213.90 ms` |
| 2 | C2378 | `131,204.50 ms` |
| 3 | C2378 | `109,622.00 ms` |
| 4 | C512 | `137,486.30 ms` |
| aggregate median | C512 | `109,350.10 ms` |
| aggregate median | C2378 | `120,413.25 ms` |

The aggregate ratio is `0.9081x`, nominally an `11,063.15 ms` regression.
However, the two C512 arms drifted by `+69.3%` from `81,213.90` to
`137,486.30 ms`, while the two C2378 arms moved in the opposite direction,
from `131,204.50` to `109,622.00 ms`. Pairing the first C512/C2378 arms gives
`0.6190x`; pairing the reverse-side C512/C2378 arms gives `1.2542x`. Thus the
direction of the conclusion flips with position in the sequence. The external
30.059-second pre-launch thermal observation was nominal at level 0, but that
single launch gate does not establish equivalent sustained conditions across
four multi-minute fanless arms.

The run nevertheless established the mechanism's correctness, bounded-memory,
coverage, and lifecycle facts:

- C512 and both C2378 executions produced the same SHA-256
  `fb8aae85e21a8a93b39baf738d0f2577e18134c627a05562b710341d0d590f7c`
  over all `17,280,000` raw waveform samples. Joint and per-channel maximum
  absolute error and NRMSE were zero, correlation was one, all samples were
  finite, all C512 and C2378 seam neighborhoods were exact, and the C2378
  repeat was deterministic.
- C2378 decoded `4,628` rather than `5,908` latent frames, a reduction of
  `1,280` frames or `21.6655%`, while covering every output position exactly
  once with the declared two-window plan.
- C2378's observed peak live buffer residency was `3,758,347,792` bytes.
  Across preparation and all four timing arms, all `111` created buffers were
  destroyed exactly once, zero buffers and bytes remained live, and at most
  one arm owner existed at a time.
- Cleanup passed, the production default remained C512, and no listening or
  production-selection approval is claimed.

## Decision and revisit condition

Retain C2378 as benchmark-only and retain C512 as the production default. The
exact work reduction and feasibility remain positive facts, but this receipt
does not establish either a speedup or a stable regression.

Revisit only with shorter paired subsystem screens or interleaved chunk-level
timing whose comparison interval is short enough to avoid the observed
multi-minute thermal/order confounding. Do not repeat the unchanged full
four-arm sequence as the next measurement.

## Artifacts

- [result](../results/OPT-0035/result.json), SHA-256
  `1833087730427396a3a198f20a3d6a159ca54cf1d5e9286f30539a3aac5aafa9`
- [external thermal gate](../results/OPT-0035/thermal.json), SHA-256
  `20e37497f21819f41e9faa3d8792a342cdac6969a4c95216e963ee2fd2b8f564`
