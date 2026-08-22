# OPT-0046 — VAE K7 branch-free interior dot4

## Status

- Evidence: `inconclusive`
- Disposition: `superseded`
- Risk: same approximate K4 partial arithmetic as OPT-0024; bit-exact versus
  OPT-0024 is required

## Hypothesis

OPT-0043 measured the structurally similar DiT K4 subgroup contraction at
`1.8925` valid GPU TFLOP/s, while OPT-0024's K7 path remains far below that
rate. In the K7 shader, every Cin4 iteration computes eight dynamic tap/row
validity branches and conditionally adds eight partial vectors even though all
seven taps are valid for nearly every interior output row. Splitting boundary
rows from the interior can remove those branches, zero initialization, and
predicate-dependent additions from the dominant region without changing one
arithmetic operation or output bit relative to OPT-0024.

## Frozen mechanism

- Keep OPT-0024's WG128/four-fixed32-subgroup, 8-row × 128-output ownership,
  native NLC/OKI FP16 storage, K-then-Cin4 order, FP16 dot4 partial, one FP32
  widening, FP32 running state, bias seed, and explicit FP16 store.
- Partition each output-time range into prefix boundary, maximal interior, and
  suffix boundary. Interior rows must satisfy every K7 tap by construction;
  its shader performs direct input loads and unconditional partial additions.
- Route prefix/suffix through the unchanged OPT-0024 owner. The planner must
  prove disjoint complete coverage and may emit no empty dispatch. Do not add
  a package layout, repack, workgroup memory, larger accumulator tile, K8/K16
  partial, or production selector under this ID.

## Gate

Compare against OPT-0024 K4, not the older exact K7 kernel, on every production
channel/dilation tier and representative C512 boundaries. Require raw-U16 bit
identity, deterministic complete writes, canaries, no non-finite/class change,
clean lifecycle, and one balanced nominal timing. Continue only if every tier
is non-slower and the weighted K7 score is at least `1.15x` faster than
OPT-0024. The user thermal protocol is one level-0 `notifyutil` observation
after a 30-second wait. This benchmark authorizes no production or quality
change by itself.

## Closeout — superseded before measurement

The branch-free interior candidate never completed its registered correctness
and timing gate, so its evidence is inconclusive. Later output-major and
row-reuse K4 layout work (OPT-0047/0051/0057) attacked the same K7 cost with
measured shape-specific gains and superseded this unmeasured direction. No
OPT-0046 kernel or selector was integrated.
