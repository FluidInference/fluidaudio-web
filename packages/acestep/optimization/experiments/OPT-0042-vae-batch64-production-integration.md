# OPT-0042 — VAE batch64 production integration

## Status

- Evidence: `positive`
- Disposition: `integrated`
- Risk: exact scheduling-only change

## Hypothesis

OPT-0027 reproduced the accepted C512 output bit-for-bit in every warmup and
timed arm while reducing mean outer-window wall from `8,479.30 ms` to
`6,204.25 ms` (`1.366692x`). It reduced decoder command buffers, queue drains,
and requested idle intervals from `982` to `123` per C512 window. Selecting the
same batch of 64 in the production C512 path should realize that exact win
without changing any dispatch, kernel, package byte, operation order, or
waveform value.

## Frozen change

- Keep C512/overlap64 windowing and the authenticated OPT-0028 exact-packed
  profile unchanged.
- Select 64 physical decoder quanta per command buffer for ordinary production
  generation, including diagnostic runs. The historical OPT-0023 harness may
  still request batch8 explicitly; production no longer does so merely to keep
  its old homogeneous-family attribution shape.
- Preserve queue depth one, FIFO order, one drain per submitted batch, abort
  checks at every batch boundary, the separate readback buffer, and complete
  backend destruction.
- Do not combine approximate K7, the OPT-0036 selector, or another chunk size
  under this ID.

## Gates

Focused static/runtime tests must prove that ordinary production and trace
capture both pass `quantaPerCommandBuffer=64`, diagnostics reconcile the actual
batch size, cancellation is observed only at a drained batch
boundary, and cleanup remains idempotent. The authenticated C512 browser
receipt from OPT-0027 is the performance and raw-U32 correctness authority.
After integration, a complete C4500 run must reproduce the accepted waveform
identity and report the actual long-stage wall before this is treated as a
product result. No listening gate is required because scheduling cannot change
arithmetic or output order.

## Result

Production and diagnostic construction now explicitly select `64` decoder
quanta per command buffer while retaining queue depth one, FIFO order, one
drain per batch, and the existing batch-boundary abort contract. Family-profile
diagnostics report the selected batch size rather than assuming eight.

Focused pipeline/backend/decoder tests prove ordinary and trace routing,
bounded cancellation, actual command-buffer accounting, and idempotent
cleanup. The OPT-0027 browser authority remains raw-U32 exact and measured
`8,479.30 -> 6,204.25 ms` mean C512 outer wall. OPT-0035 independently ran the
complete C4500 waveform through batch64 with all `17.28 M` samples, seams, and
deterministic outputs exact and zero live resources. Its fixed-order long-wall
comparison was thermally unstable, so this integration makes no C4500 speedup
or under-60-second claim; the short balanced C512 screen is the performance
authority.
