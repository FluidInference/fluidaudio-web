# OPT-0053 — VAE K7 output-major high-tier selector

## Status

- Evidence: `inconclusive`
- Disposition: `superseded`
- Risk: physical-layout-only relative to OPT-0024, but approximate FP16 K4
  partial reduction relative to the current exact scalar-FP32 K7 owner

## First-principles basis

OPT-0047 changed only K7 weight addresses and was raw-U16 exact across all 16
biased operations. It made the high-work C1024 and C512 tiers `2.2919x` and
`1.5442x` faster, while the already-tiny C256/C128 tiers were `0.8478x` and
`0.9583x`. Selecting output-major only for C1024/C512 and retaining OPT-0024
for C256/C128 substitutes a measured production-weighted score of
`2,447.231260 ms` for `4,660.050007 ms` (`1.904213175x`) without accepting a
losing tier.

Applying that ratio only as planning arithmetic to OPT-0024's long K7
projection gives `34.966 -> 18.362 s`, about `16.604 s` potential saving. It
is not a long measurement.

## Integration direction and gate

- Give the VAE package a new authenticated identity. Replace only eligible
  C1024/C512 biased K7 weights with OPT-0047's verified
  `[K7,Cin4,CoutTile128,lane32,output4,CinElement4]` layout. Retain native
  O-K-I for C256/C128 and every unbiased/final K7 operation. Do not duplicate
  either representation.
- Route only exact labels/shapes through output-major. Preserve OPT-0024
  arithmetic, all low-tier owners, K1/ConvTranspose/Snake/Add/ingress owners,
  and batch64 scheduling. Fail closed on identity, feature, subgroup, label,
  shape, or layout mismatch.
- Prove deterministic conversion, complete tensor accounting, exhaustive
  source/layout authentication, exact per-operation owner/count reconciliation,
  and clean lifecycle.
- On the authenticated C512 fixture, require raw-bit-identical complete output
  against a native-layout OPT-0024 K4 arithmetic oracle, deterministic reruns,
  unchanged non-K7 topology, and zero non-finite values. Separately measure the
  candidate against the current revision-6 scalar-FP32 K7 oracle under the
  frozen numerical envelope. Then run balanced AB/BA after one nominal thermal
  check. Both directions must improve homogeneous K7 wall, median K7 speedup
  must reach `1.50x`, and full decoder wall must not regress.
- C4500 requires deterministic complete waveform and seam metrics against the
  exact revision-6 oracle, bounded memory, cancellation, and resource
  destruction. It is not expected to be raw-bit identical to revision 6.

The layout transform is exact relative to OPT-0024, but OPT-0024's FP16 K4
partials are an approximate arithmetic change relative to the current
scalar-FP32 production K7 owner. Production selection therefore remains
subordinate to OPT-0044's activation-trajectory, complete-waveform, 12-second
product listening, and explicit owner-approval gates. Primitive arithmetic or
raw identity against OPT-0024 alone authorizes no package, runtime, long,
quality, or under-60-second claim.

## Closeout — superseded by the stronger selector

The output-major high-tier selector never completed its registered package and
C512 gates, so its evidence is inconclusive. OPT-0051 measured the stronger
row-reuse geometry and OPT-0057 selected that owner by shape, superseding this
direction before production integration. Preserve the OPT-0047 primitive and
planning projection above; no OPT-0053 package or runtime route was selected.
