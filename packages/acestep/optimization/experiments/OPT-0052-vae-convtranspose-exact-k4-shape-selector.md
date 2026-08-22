# OPT-0052 — VAE ConvTranspose exact K4 shape selector

## Status

- Evidence: `inconclusive`
- Disposition: `superseded`
- Risk: exact arithmetic/output, package-layout and routing change

## First-principles basis

OPT-0048's K4 reduction was raw-U16 identical to the exact OPT-0040 owner over
`282,624,000` candidate comparisons. It regressed only block 0 (`0.9320x`)
while improving blocks 1–4 by `1.2729x`, `1.7643x`, `2.4145x`, and
`2.1800x`. Selecting the unchanged exact owner for block 0 and K4 for blocks
1–4 gives a measured-median primitive score of `189.50 ms` versus
`316.20 ms` (`1.668601582x`) without accepting a losing shape.

Applying that ratio only as planning arithmetic to OPT-0040's existing long
projection gives about `8.026 -> 4.810 s`, a further `3.216 s` potential
saving. This is not a long measurement.

## Integration direction and gate

- Give the VAE package a new authenticated identity. Keep block 0's revision-6
  exact polyphase layout and replace only blocks 1–4 with OPT-0048's verified
  `[phase,tap,Cin4,CoutTile,lane,outputWithinLane,K4]` layout. Do not retain
  duplicate copies or browser-repack weights.
- Route block 0 through OPT-0040's exact channel-reuse owner, blocks 1–2 through
  OPT-0048 channel-reuse K4, and blocks 3–4 through OPT-0048 row-reuse K4.
  Preserve every other K1/K7/Snake/Add/ingress owner and batch64 scheduling.
- Prove deterministic converter output, manifest/source/layout authentication,
  complete tensor accounting, exact route/count reconciliation, and clean
  lifecycle. Fail closed on any profile, feature, subgroup, shape, or identity
  mismatch.
- On the authenticated C512 fixture, require raw-bit-identical complete output
  against OPT-0028/0040, deterministic reruns, unchanged seams/topology outside
  ConvTranspose, and zero non-finite values. Then run one balanced timing gate
  after the ordinary single level-0 thermal check. Both paired directions must
  improve homogeneous ConvTranspose wall, median speedup must reach `1.30x`,
  and full decoder wall must not regress.

A pass may replace the current fixed32 VAE package/profile because the output
contract is exact. C4500 still requires complete deterministic waveform,
bounded memory, cancellation, and lifecycle evidence; the primitive ratio
alone makes no long or under-60-second claim.

## Closeout — subsumed by the joint revision-7 gate

OPT-0052 did not complete a standalone authenticated package/C512 receipt, so
its literal evidence remains inconclusive. Its block-0 exact plus blocks-1–4
K4 mechanism was subsumed into OPT-0054 and then reauthenticated and measured
jointly by OPT-0066, with production promotion owned by OPT-0072. This record
is superseded; the original OPT-0048 primitive evidence remains unchanged.
