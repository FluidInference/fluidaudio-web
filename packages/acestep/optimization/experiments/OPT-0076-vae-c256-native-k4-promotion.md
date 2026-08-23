# OPT-0076 — VAE C256 native-layout K4 promotion

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Date: 2026-08-21
- Author/agent: Codex
- Risk class: `additional approximate FP16-dot4 partial reduction`
- Allocation baseline: pushed `main` commit
  `e44788c2945f008f0a16db8bb1b04afbeac3d890`

## First-principles basis

The accepted OPT-0072 production identity uses the physical OPT-0066
revision-7 profile. Its K7 selector sends twelve C1024/C512/C128 residual
labels through OPT-0051 row-reuse K4, but sends the three C256 residual labels
through the native-layout scalar-FP32 owner:

- `block-2-res-1-conv1`, dilation 1;
- `block-2-res-2-conv1`, dilation 3; and
- `block-2-res-3-conv1`, dilation 9.

This is the implemented and validated current truth. It also diverges from
OPT-0051's derived selector and OPT-0057's frozen direction, both of which
named the existing OPT-0024 native-layout K4 owner for C256 after row reuse
regressed there. Later OPT-0066/0072 evidence intentionally validated the
literal scalar implementation and does not authorize silently changing it.
OPT-0076 therefore treats the correction as a new quality-affecting promotion,
not a clerical waiver or a retroactive reinterpretation of those results.

OPT-0024 already supplies the exact physical mechanism needed here. It accepts
the unchanged native O-K-I FP16 weights, uses the shipped WG128/four-subgroup
8-row by 128-output ownership, introduces no repack or duplicate package
payload, advances in increasing K then Cin4 order, computes FP16 dot4 partials,
widens each partial once, and retains FP32 running accumulators. On the frozen
C256/d1 first/interior/tail screen it measured:

`3.6499999166 -> 1.3500000834 ms` (`2.703703475x`).

That timing covered three output ranges. It is prior mechanism evidence, not a
current production result.

## Exact accepted production scope

One C2314 decoder window has `555,360` C256 output frames. Each of the three
target operations therefore owns:

- `142,172,160` FP16 outputs;
- `1,085` batch-64 quanta;
- `34,710` WG128 workgroups; and
- `254,772,510,720` logical multiply-accumulates.

Across the accepted two-C2314 direct path, the promotion covers `6,510`
quanta, `208,260` workgroups, `853,032,960` FP16 outputs, and
`1,528,635,064,320` MACs. The three C256 labels contribute `3,255 / 3,299`
(`98.67%`) of the current native-scalar K7 quanta in each C2314 window. Scaling
the prior three-range saving only by `6,510 / 3` gives about `4.99 s` of
two-window planning headroom. This motivates a current measurement; it is not
a performance claim.

## Frozen implementation direction

1. Add a benchmark-only OPT-0076 selector identity. Route exactly the three
   authenticated C256 labels above to the unchanged
   `AceOpt0024VaeConv1dDirectDot4SubgroupKernel` and expose owner
   `native-k4`. Delegate every other K7 label to the literal current
   OPT-0057 selector with no route, layout, arithmetic, label, or kernel-ID
   change.
2. The selector accepts only batch 1, C256-to-C256, K7, stride 1, the exact
   label-specific dilation/padding, bias present, and FP16 output. It fails
   closed on every mismatch. It must destroy both nested owners exactly once,
   reject use after destruction, and never select a fallback under the new ID.
3. Primitive/selector work is benchmark-only. If its gates pass, add a new
   explicit diagnostic runtime-profile, kernel-set, and precision-map identity
   based on the accepted OPT-0072 physical package. Do not mutate the OPT-0066
   or public OPT-0072 identities, package manifest, or production default.
4. The diagnostic profile changes only the three C256 route owners. It retains
   OPT-0051 row reuse for twelve labels, scalar native owners for ingress and
   final FP32 `conv2`, the accepted transpose selector, K1, Snake, Add,
   batch-64 scheduling, C2314 windowing, seams, normalization, and every
   planner/DiT operation.

## Selector and primitive gate

- Exhaustively reconcile all seventeen K7 operation contracts. Exactly twelve
  remain `row-reuse-k4`, exactly three become `native-k4`, and exactly two
  remain `native-scalar-fp32`; no other production source may import the
  benchmark selector before diagnostic-profile escalation.
- On d1/d3/d9 first/interior/tail C256 ranges, compare selector output against
  a direct OPT-0024 dispatch over deterministic production-local and bounded
  adversarial inputs. Require raw-U16 identity, identical reruns, finite
  outputs, complete qNaN-overwriting writes, intact guards/tails, exact kernel
  and selector IDs, zero GPU errors, and idempotent zero-live cleanup.
- Separately compare against the current scalar owner and report the complete
  OPT-0024 numerical metrics. Apply the unchanged OPT-0024 primitive envelope:
  NRMSE at most `0.001`, SNR at least `60 dB`, Pearson at least `0.99999`, and
  relative maximum absolute error at most `0.01`.
- Under a continuous nominal thermal trace, run at least four balanced
  current/K4 samples across all three dilations and all three range positions.
  Require every paired GPU and fenced-wall score to win and at least `2.0x`
  mean and median for the aggregate C256 bucket. Otherwise stop.

## Complete C512 diagnostic-profile gate

Use one authenticated C512 fixture/package and sequential owners. Compare the
literal accepted production profile A with the diagnostic C256-K4 profile B in
`A/B/B/A` order after correctness and a fresh continuous 30-second level-0
thermal gate.

1. Require the new selector to account for exactly `720 / 730` current native
   K7 quanta and leave all `3,360` row-reuse K7 quanta and all non-K7 owner
   counts unchanged.
2. Require deterministic complete output, all finite values, bounded memory,
   clean cancellation/lifecycle, and the unchanged OPT-0066 complete-waveform
   envelope: NRMSE `<= 0.003`, SNR `>= 50 dB`, Pearson `>= 0.9999`, relative
   RMS/energy drift `<= 0.005`, peak drift `<= 0.01`, DC drift `<= 0.001`, and
   relative maximum error `<= 0.02`.
3. Attribute the three C256 labels separately from total K7. Both paired
   directions must win C256, K7-family, complete decoder, and outer wall;
   median C256 speedup must be at least `2.0x`, and each direction must save at
   least `400 ms` of decoder wall. No non-target family may regress by more
   than the greater of `2%` or `5 ms`.

This gate authorizes only the exact C2314 escalation below.

## C2314 and production-quality gates

Run one independently cooled `A/B` pair and one independently cooled `B/A`
pair on the exact accepted C2314 window. Require both candidate observations
to save at least `1,500 ms` in complete decoder wall and at least `1,500 ms` in
outer window wall, with the same C256 attribution, numerical envelope,
non-target stability, cancellation, memory, and cleanup gates.

If and only if C512 and C2314 pass, execute the complete two-C2314 direct VAE
trajectory from the pinned final latent. Require:

- all `17,280,000` raw stereo samples finite and nonzero;
- deterministic repeat, finite seams, unchanged length/overlap/discard and
  pre-normalization comparison against accepted OPT-0072;
- the unchanged long numerical/energy/peak/DC envelope and ordinary bounded
  `-1 dBFS` normalization;
- zero device loss and exact resource reconciliation; and
- fresh owner blind listening over the approved identity plus the required
  vocal/instrumental coverage.

Only a passing quality/listening gate and a material balanced product wall win
may select a new production profile. Do not infer quality approval from the
existing OPT-0024, OPT-0066, or OPT-0072 results, and do not claim the planning
five-second ceiling as measured performance.

## Benchmark identity

- Machine: local `Mac15,12`, Apple M3 10-GPU-core, 16 GiB unified memory.
- Browser: exact stock Chrome/Codex surface, `shader-f16`, fixed 32-lane
  `subgroups`, and standard timestamps where used.
- Current product authority: public OPT-0072 mapped to physical OPT-0066,
  revision-7 manifest and package, two C2314 windows.
- Allocation baseline, selector/profile/harness commits, source hashes,
  package/fixture identities, every raw sample, continuous thermal traces, and
  through-cleanup lifecycle evidence must be retained.

## Results

The benchmark-only selector and guarded primitive harness were implemented at
`6968ed5f4fc70d43d6701a1d5117842c65c076a0`. The selector reconciled all
seventeen K7 operation contracts: twelve stayed on row reuse, exactly the
three declared C256 labels selected native K4, and ingress/final Conv1D stayed
on the scalar owner.

Correctness passed over six production-local/adversarial cases and all
d1/d3/d9 first/interior/tail probes. The selector matched an independent
OPT-0024 owner over `245,760` raw U16 values with zero first-run or rerun
differences. Every selected output was finite and complete, no dispatch wrote
outside its authenticated range, physical and adjacent guards remained
intact, cleanup reconciled `140/140` buffers with zero live bytes, and there
were no GPU errors or device losses. Versus the current scalar owner, the
aggregate C256 metrics passed the frozen envelope at NRMSE
`0.00022548613920154832`, SNR `72.93760298732087 dB`, Pearson
`0.9999999715712965`, and relative maximum error
`0.0006329113924050633`.

The native K4 mechanism was materially faster in the aggregate. Six balanced
nine-dispatch samples measured mean/median timestamped-GPU speedups of
`3.6038961038961044x` / `3.8019801980198027x` and fenced-wall speedups of
`2.9419355041898876x` / `3.6708860870279425x`. All six GPU pairs won. The
sixth candidate-first sample nevertheless completed `4.063232 ms` of GPU work
in `11.699999928474426 ms` fenced wall versus the paired current sample's
`7.20896 ms` GPU and `10.600000023841858 ms` wall. That single wall reversal
fails the predeclared every-pair rule. The continuous trace stayed nominal for
all `87` observations through cleanup with a `936 ms` maximum poll gap, so it
does not invalidate or waive the result.

Disposition is therefore inconclusive and stopped at the selector/primitive
gate. No diagnostic runtime profile, C512/C2314 decoder run, waveform,
listening gate, package change, or production integration is authorized. See
[`result.json`](../results/OPT-0076/result.json) and the retained browser and
thermal receipts beside it.

## Authority and interactions

- [OPT-0024 native-layout K4 mechanism](OPT-0024-vae-k7-direct-subgroup-fp16-dot4.md)
- [OPT-0051 row-reuse result](../results/OPT-0051/result.json)
- [OPT-0057 intended shape selector](OPT-0057-vae-k7-k4-row-reuse-shape-selector.md)
- [OPT-0066 literal accepted C512 quality gate](OPT-0066-vae-revision7-dual-k4-quality-gate.md)
- [OPT-0072 production quality promotion](OPT-0072-revision7-vae-c2378-production-promotion.md)
- [Final accepted product result](../results/OPT-0073/result.json)

Revisit after a negative result only for changed hardware/compiler behavior,
a materially different C256 arithmetic/dataflow, or a new precision mechanism.
Do not relax the quality or listening gates and do not reclassify approximate
K4 arithmetic as exact.
