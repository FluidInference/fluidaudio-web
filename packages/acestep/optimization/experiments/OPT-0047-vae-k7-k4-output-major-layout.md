# OPT-0047 — VAE K7 output-major K4 weight layout

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Risk: physical-layout-only relative to OPT-0024; raw-U16 identity is required

## Hypothesis

OPT-0024 established that the fixed32 8-row × 128-output tile is a strong
arithmetic owner once four adjacent input channels are reduced by one native
FP16 dot and widened into the FP32 running state. Its remaining weight access
is nevertheless inherited from native O-K-I storage. At fixed `(kernel,
cin4)`, adjacent subgroup lanes request the 128 output-channel vectors at a
stride of `7 * (Cin / 4)` `vec4<f16>` records. The workgroup therefore issues a
full logical tile of weights through a warp-wide strided access even though the
same 128 records could be contiguous.

Store each candidate K7 weight as
`[kernel7, cin4, coutTile128, lane32, output4, cinElement4]`. At fixed kernel,
Cin4, and output tile, the four vectors owned by one lane are adjacent and the
128 vectors requested by the subgroup form one contiguous region. This changes
only address arithmetic and transaction locality. Bias, input/output storage,
subgroup ownership, broadcasts, boundary predicates, K-then-Cin4 order,
FP16-dot4 partials, FP32 running accumulation, and FP16 stores remain exactly
OPT-0024.

This is not an unchanged repeat of OPT-0014. That experiment measured an exact
scalar-Cin packed-KIO kernel with different row/channel ownership before the
positive bounded-dot mechanism existed. OPT-0047 tests whether a
dot4-native, output-major `vec4<f16>` layout removes the remaining transaction
pathology in the already-positive OPT-0024 tile.

## Frozen mechanism

- Compare only against OPT-0024 K4 and use the same 16 biased production K7
  shapes. The final unbiased FP32-output `conv2` is outside this experiment.
- Keep WG128, four fixed-32 subgroups, eight rows and 128 output channels per
  subgroup, four adjacent outputs per lane, 32 FP32 scalar accumulators per
  lane, no workgroup storage, and no barriers.
- Keep native NLC input. Pack weights bit-preservingly into
  `[K7, Cin/4, Cout/128, lane32, output4, Cin4]`; every logical FP16 word must
  have a proved forward and inverse address.
- For increasing K then increasing Cin4, load four adjacent weight vectors,
  issue the same four FP16 dot4 operations per row, widen the same partial
  vector once, and add once to the same FP32 state. Do not combine this with
  OPT-0041 K8/K16 partials or OPT-0046 branch-free interiors.
- A benchmark may use a separately timed-outside-the-score pack step. No
  converter, package identity, duplicate resident production weights, or
  production selector is authorized under this primitive ID.

## Gate

Prove exhaustive native-to-packed-to-native raw-U16 identity for every tested
weight and raw-U16 output identity against OPT-0024 on all production
channel/dilation tiers, representative C512 boundaries, deterministic reruns,
complete-write canaries, finite/class checks, and clean lifecycle. Run one
balanced nominal timing screen after one level-0 `notifyutil` observation made
after at least 30 idle seconds. Continue only if every tier is non-slower and
the production-weighted K7 score is at least `1.20x` faster than OPT-0024.

A passing primitive authorizes a new-ID converter-native package and C512
subsystem gate. It does not authorize production routing, a long-song speed
claim, or quality/listening approval.

## Result

The physical transform and arithmetic were exact. Preparation exhaustively
packed and inverted all `30,507,008` FP16 weight words across the 16 biased K7
operations with zero mismatches. Two executions per arm covered every C512
boundary/interior row and compared `16,777,216` output words: zero mismatches,
raw-U16-identical hashes, deterministic repeats, complete qNaN overwrite,
finite/class identity, intact guards, and clean idempotent destruction of all
176 buffers.

One balanced timing screen followed `42.629 s` idle and exactly one
thermal-level-0 observation. Each of four samples repeated its dispatch eight
times to lift the sub-millisecond tiers above timer granularity:

| tier | weight | OPT-0024 median | output-major median | speedup |
| --- | ---: | ---: | ---: | ---: |
| C1024 | 282 | 11.8750 ms | 5.18125 ms | 2.29192x |
| C512 | 423 | 2.18125 ms | 1.41250 ms | 1.54425x |
| C256 | 423 | 0.48750 ms | 0.57500 ms | 0.84783x |
| C128 | 1,269 | 0.14375 ms | 0.15000 ms | 0.95833x |

The output-major score improved `4,660.050007 -> 2,492.174999 ms`
(`1.869872705x`), far above the aggregate threshold, but the two small-channel
tiers were slower. The universal owner therefore fails the frozen
every-tier-non-slower condition and stops before package/subsystem escalation.

This heterogeneous result establishes a much stronger exact follow-up. Keep
OPT-0024 for C256/C128 and select output-major only for C1024/C512. Directly
substituting the measured medians gives `2,447.231260 ms`
(`1.904213175x`) versus general K4. Applying that ratio only as planning
arithmetic to OPT-0024's long K7 projection gives `34.966 -> 18.362 s`, about
`16.604 s` potential saving. It requires a separately registered
converter-native package and authenticated C512/long gate; it is not authority
to reinterpret this negative primitive result.

Compact receipt: [`../results/OPT-0047/result.json`](../results/OPT-0047/result.json),
SHA-256 `b78c577b16a41ad68597da882a92dd07b0d7125a7f72b6a3ce2f1a5ed1cd81af`.
