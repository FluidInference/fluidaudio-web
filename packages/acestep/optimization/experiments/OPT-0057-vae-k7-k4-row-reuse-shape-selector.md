# OPT-0057 — VAE K7 row-reuse K4 shape selector

## Status

- Evidence: `inconclusive`
- Disposition: `superseded`
- Risk: physical-layout-only relative to OPT-0024, but approximate FP16 K4
  partial reduction relative to the current exact scalar-FP32 K7 owner

## First-principles basis

OPT-0051 kept OPT-0024's K4 arithmetic and 32 FP32 running accumulators per
lane while changing ownership from eight rows × four outputs to 16 rows × two
outputs. The candidate was raw-U16 identical to OPT-0024 across all 16 frozen
correctness cases. Its measured tier speedups were `2.332487312x` at C1024,
`1.645454539x` at C512, `0.875x` at C256, and `1.678571432x` at C128. The
universal owner therefore failed only because C256 regressed.

Select row reuse for the measured C1024, C512, and C128 shapes and retain the
unchanged native-layout OPT-0024 owner for C256. Substituting the C256 control
term into OPT-0051's measured score gives exactly:

`2404.04999853 - 423 × 0.5 + 423 × 0.4375 = 2377.61249853 ms`

versus `4754.34374685 ms` for OPT-0024, or `1.9996293549892823x`, without
accepting a losing tier. Applying that ratio only as planning arithmetic to
OPT-0024's existing long-K7 projection gives
`34.966 -> 17.486240593916172 s`, about `17.47975940608383 s` potential
saving. This is not a long measurement.

## Frozen integration direction

- Create a new authenticated VAE package identity and fail-closed runtime
  profile. Replace only the exact eligible C1024/C512/C128 biased K7 weights
  with OPT-0051's verified
  `[K7,Cin4,CoutBand64,lane32,output2,CinElement4]` layout. Retain native O-K-I
  weights for C256 and every other K7 shape. Do not retain duplicate
  representations or perform browser-side repacking.
- Route only authenticated labels whose complete shapes match the frozen
  selector. Preserve increasing K then Cin4 order, native FP16 inputs/weights,
  one FP16 dot per four Cin values, FP32 bias/running state, explicit FP16
  stores, fixed32 WG128, boundary behavior, all non-K7 owners, and batch64
  scheduling. Fail closed on any profile, manifest, feature, subgroup, label,
  shape, or layout mismatch.
- Prove deterministic conversion, complete consumed/excluded tensor
  accounting, exhaustive logical/physical bijections, reproducible package
  bytes and hashes, exact selector-route/count reconciliation, bounded staging,
  cancellation, and exactly-once resource destruction.

## Gates

1. On the authenticated C512 fixture, compare the new mixed-layout/profile
   output against a native-layout OPT-0024 K4 arithmetic oracle. Require raw-U16
   identity for the complete output, deterministic reruns, exact unchanged
   topology outside selected K7 owners, zero non-finite values, intact
   seams/tails, and zero live resources.
2. Separately compare the candidate with the current exact scalar-FP32 K7
   oracle under OPT-0044's unchanged numerical envelope. Raw identity to
   OPT-0024 proves the layout and selector only; it does not erase the K4
   rounding change relative to production.
3. After one nominal level-0 thermal check, run a balanced C512 family plus
   full-decoder AB/BA gate. Both directions must improve homogeneous K7 wall,
   median K7 speedup must reach `1.50x`, and full decoder wall must not regress.
   Report family attribution and outer wall separately; the primitive
   projection is not a substitute for this measurement.
4. If those gates pass, complete OPT-0044's activation-trajectory,
   deterministic C4500 raw-waveform/seam, bounded-memory, lifecycle, and
   12-second product listening gates against the exact production oracle.
   Because K4 partials are approximate, production selection still requires
   explicit owner approval.

No package, profile, production-default, long-song, quality, or under-60-second
claim is authorized by OPT-0051's primitive result or the substituted planning
score alone.

## Closeout — original gate superseded

OPT-0057's first C512 preparation failed before READY because its benchmark
oracle retained scalar revision-6 ConvTranspose arithmetic; that diagnostic
receipt remains preserved. OPT-0066 allocated the corrected complete
same-arithmetic oracle, authenticated the row-reuse mechanism, and passed the
joint C512 quality/timing gate; OPT-0072 owns production promotion. The literal
OPT-0057 result is therefore inconclusive and this record is superseded.
