# OPT-0078 — DiT dense packed-weight tile multicast

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Date: 2026-08-21
- Author/agent: Codex
- Risk class: `exact increasing-K FP16-operand/FP32-accumulation ownership`
- Allocation baseline: pushed `main` commit
  `9a5f54719208b46f869b69f11271a045ae929047`
- Registration commit: `23e09472eb997c3543d98f07767d08182b572938`
- Candidate/harness commit: `bfd286002654dc67b85d2986686ad917e497d073`

## First-principles basis

The accepted 180-second product spends `54,986.2 ms` in DiT denoising. The
current repeated-layer dense owner is OPT-0009's exact
`M32 x N256 x K32 / WG128` subgroup kernel. Four fixed-32 subgroups own four
eight-row bands over the same 256 output columns. Each subgroup therefore
issues the same 1,024 packed `vec4<u32>` weight-record loads for every K32
tile, while each lane retains 64 independent FP32 accumulators.

OPT-0043 established that this path is GPU-kernel-bound: its weighted
GPU-to-wall ratio was `0.906326`, and timestamped-GPU and wall speedups agreed.
Submission tuning is not the missing mechanism. Historical and current causal
profiles both put the repeated dense path in the tens of seconds: OPT-0019's
exact four-shape current score was `223.2 ms` per layer/evaluation, while the
newer OPT-0067 evaluation slice measured about `3.32-3.37 s` of pure
feed-forward work before dense members hidden inside mixed commands.

The candidate keeps the exact current `M32 x N256 x K32` output tile but uses
WG256 and eight subgroups. Each subgroup owns four rows by eight columns per
lane, reducing private state to 32 FP32 accumulators. The 256 lanes
cooperatively load the existing native packed K32/N256 weight tile exactly
once into a `16,384`-byte workgroup array. All eight subgroups then read the
same packed records from workgroup memory while retaining OPT-0009's subgroup
broadcasts for activations.

This changes neither output ownership nor scheduled arithmetic. It targets
three concrete hardware limits:

- four redundant global/cache weight requests become one cooperative request;
- 64 live accumulators per lane become 32; and
- twice as many independent subgroup row bands can issue while the same 8,192
  outputs remain owned by one workgroup.

The costs are explicit: WG128 becomes WG256, the kernel consumes the device's
full `16 KiB` declared workgroup-storage limit, each K32 tile adds one barrier
after fill and one before overwrite, and all eight subgroups read the shared
tile. Cache multicast may already hide part of the nominal request reduction,
and storage/barriers may reduce occupancy. A bounded primitive screen decides
that causal tradeoff before graph work.

## Novelty and prior results

This is not an unchanged retry of an abandoned dense geometry:

- OPT-0019 used a `M64 x N128 x K16 / WG256` 16x16 thread grid, staged both A
  and B in `6,400` bytes, and did not use subgroup activation broadcasts. It
  was exact and improved every shape by `1.207-1.512x`; its weighted score
  improved `223.2 -> 169.7 ms` (`1.31526x`) but stopped on its frozen `1.55x`
  ratio despite clearing the absolute ten-second planning threshold.
- OPT-0020/0021 changed OPT-0019's K4 arithmetic or panel orientation and were
  negative. OPT-0031 used WG512 with `M128 x N128 x K32` A+B panels and
  regressed. None preserves the current M32/N256 ownership while staging only
  the native packed B tile once across eight subgroups.
- OPT-0032/0037/0038/0050/0056/0058/0074 change reduction precision or operand
  representation and failed numerical, trajectory, or stability gates. This
  experiment does not use their approximate arithmetic.

At the four production multiplicities `4/2/2/1`, current and candidate both
schedule exactly `6,816` workgroups and `133,412,421,632` multiply-adds per
layer/evaluation. A conservative source-request model is:

| weighted per layer/evaluation | current OPT-0009 | candidate |
| --- | ---: | ---: |
| activation request bytes | 2,084,569,088 | 2,084,569,088 |
| weight request bytes | 33,353,105,408 | 8,338,276,352 |
| total operand request bytes | 35,437,674,496 | 10,422,845,440 |
| FP32 accumulators per lane | 64 | 32 |
| declared workgroup storage | 0 | 16,384 B |

The `3.4x` byte ratio is static request accounting, not a wall-speed claim.
The experiment is worthwhile because only a `20.8334 ms` weighted saving per
layer/evaluation projects to four seconds across `24 * 8`, while the distinct
OPT-0019 mechanism measured `53.5 ms`.

## Frozen candidate mechanism

1. Add one benchmark-only kernel with exact geometry
   `M32 x N256 x K32 / WG256`, eight fixed-32 subgroups, four rows and eight
   columns per lane, 32 FP32 accumulators per lane, and exactly two workgroup
   barriers per K32 tile. Do not add a K16/K64, WG128, alternate tile, or
   runtime-tuned geometry sweep under this ID.
2. Consume the existing `dit-gemm-n256-k32-tile-major-v1` FP16 payload. Each
   local lane copies exactly four packed `vec4<u32>` records into
   `array<vec4<u32>, 1024>` workgroup storage. There is no converter, repack,
   package, manifest, or persistent-memory change.
3. For each output, preserve the exact increasing-K expression and boundaries:
   activation FP32 load, conversion to native FP16, packed weight decode,
   widening to FP32, multiplication, and FP32 `acc = acc + product`. Do not use
   `dot`, `fma`, split-K, reassociation, FP16 accumulation, or subgroup matrix
   operations.
4. Preserve the current row-major FP32 output, K/N shape set, binding
   disjointness, and one dispatch/range per dense operation. Keep the
   production selector, graph, sampler, scheduler, package bytes, and OPT-0070
   attention unchanged during the primitive phase.
5. Fail closed unless `shader-f16`, fixed subgroup size 32, WG256, and at least
   `16,384` bytes of workgroup storage are explicitly available. Reject bias,
   non-M2250 rows, non-production K/N shapes, malformed bindings, and any
   layout mismatch without fallback.

## Static and primitive correctness gate

1. Prove exact-once ownership for all `32 * 256` outputs, all eight subgroups,
   the M2250 tail, and every one of the 1,024 packed records per K32 tile.
   Prove the physical packed index is identical to OPT-0009 for every
   N-tile/K-tile/K-in-tile/lane record and that both barriers surround every
   possible overwrite.
2. Reconcile the four exact shapes and multiplicities:
   `M2250/K2048/N2048`, `M2250/K2048/N1024`,
   `M2250/K2048/N6144`, and `M2250/K6144/N2048`. Current and candidate must
   have identical row/column tiles, workgroups, scheduled rows, scheduled
   multiply-adds, valid outputs, and output ranges.
3. Compare current OPT-0009, candidate, and deterministic reruns over every
   output U32 for all four shapes using identical deterministic FP32
   activations and native packed-FP16 weights. Require zero mismatches,
   complete qNaN-prefill overwrite, finite production outputs, intact
   prefix/suffix/adjacent canaries, and explicit coverage of the partial M
   tile.
4. Add bounded signed-zero, subnormal/normal boundary, alternating-cancellation,
   maximum-finite FP16, and activation-rounding fixtures. Require raw-U32 and
   finite-class identity, not a numerical envelope. Retain first/worst
   mismatch coordinates if the gate fails.
5. Require zero uncaptured GPU errors/device losses and balanced created /
   destroyed / mapped resources after ordinary, repeated-destroy, and
   post-destroy rejection paths.

Any raw-word mismatch, missing record, unwritten output, alias, or capability
failure stops before performance timing.

## Primitive performance gate

Compile, warm, and finish correctness before the thermal boundary. Begin
timing only after at least 30 continuous seconds at thermal level 0; poll
through cleanup. Use standard `timestamp-query` plus submit-through-matching-
drain fenced wall. Exclude allocation, upload, compilation, readback, and
serialization.

Run at least eight fixed balanced rounds, alternating current/candidate arm
order and rotating the four shape orders. Each sample is one identical
compute pass, command buffer, submit, and drain. If repetitions are needed to
escape timer quantization, choose and freeze one count before timing and
retain raw totals before division. Retain every sample, paired ratio, GPU
duration, wall duration, thermal observation, and exact order.

Define each arm's complete repeated-dense score as
`4*T2048x2048 + 2*T2048x1024 + 2*T2048x6144 + T6144x2048`.
Also report the feed-forward-only `2*T2048x6144 + T6144x2048` score and each
shape separately. The candidate qualifies only if all of these hold on both
timestamped GPU and fenced wall:

- every shape's mean and median are strictly faster;
- the weighted score wins at least seven of eight same-round pairs;
- weighted mean and median speedup are each at least `1.12x`;
- weighted mean and median absolute saving are each at least
  `20.8334 ms` per layer/evaluation, projecting at least `4,000 ms` across
  `24 * 8`; and
- wall savings remain within `0.75x..1.25x` of the corresponding timestamped
  GPU savings.

A projection below `2,000 ms`, any regressing shape, or a clear storage /
barrier regression closes the exact mechanism as negative. Thermal gaps,
wall/GPU disagreement, or a directionally mixed result is inconclusive. Do
not repeat unchanged timed work, relax the absolute gate, or substitute a
fastest sample.

## Escalation and integration gate

Only a passing primitive may add a distinct diagnostic dense profile using
the same package bytes. Then:

1. Compare independently cooled current/candidate/current/candidate complete
   24-layer M2250 evaluation slices. Require raw-U32 identity at every selected
   dense output in the first diagnostic arm, exact evaluation-result identity
   and deterministic repeat, both paired directions faster, and at least
   `3,000 ms` projected eight-evaluation saving after absolute unrelated-family
   deltas.
2. Reconcile all nine repeated dense routes per layer, their physical ranges,
   dispatches, one FIFO graph owner, unchanged sampler/attention, bounded
   cancellation, injected failure, drain-before-release, idempotent destroy,
   and zero live buffers/bytes/maps. The candidate must not cause a new
   maximum-buffer or heavy-owner residency overlap.
3. Run all eight canonical M2250/C98 evaluations and require raw-U32 identity
   at every sampler tap and the final latent. Exactness failure cannot be
   converted into a quality envelope under this ID.
4. Run one short generic-duration direct product and require identical final
   latent, raw pre-normalization waveform, seams, normalized WAV hash,
   metadata, and cleanup. Raw-bit identity satisfies the quality gate without
   a new subjective listening round.
5. A production pass may select only this exact dense owner. Run one balanced
   current/candidate M2250 confirmation if evaluation-slice materiality is
   ambiguous; do not spend a 180-second product run to rescue a sub-threshold
   primitive. Any final release claim still follows `PLAN.md`'s cumulative
   production and thermal protocol.

This experiment authorizes no approximate K2/K4 arithmetic, head-layout or
RMSNorm/RoPE fusion, attention change, package revision, fewer denoise
evaluations, native Metal path, or under-one-minute claim.

## Authority and revisit boundary

- Current product authority:
  [OPT-0073](OPT-0073-revision7-webcrypto-production-selection.md)
- Dense utilization authority:
  [OPT-0043](OPT-0043-webgpu-timestamp-utilization-profile.md)
- Prior exact cooperative geometry:
  [OPT-0019](OPT-0019-dit-dense-cooperative-panels.md)
- Latest causal evaluation slice:
  [OPT-0067](OPT-0067-dit-quad-query-evaluation-slice-thermal-screen.md)

Do not revisit the unchanged M64/N128 OPT-0019, its K4/vector variants, the
WG512 OPT-0031 tile, or approximate dense owners under this ID. A negative
result closes this exact M32/N256 packed-weight-multicast geometry unless a
material browser/compiler or target-GPU change invalidates its timing. A
different tile, persistent/repacked layout, matrix feature, or changed
arithmetic requires a new experiment ID.

## Results

OPT-0078 is inconclusive and remains benchmark-only. The exact kernel and its
browser gate were implemented at
`bfd286002654dc67b85d2986686ad917e497d073`. Static ownership, all four
production shapes, actual-GPU raw-U32 identity, thermal launch/completion, and
lifecycle gates passed. The candidate was directionally faster throughout,
but it missed the literal frozen mean thresholds narrowly, so no diagnostic
profile, graph integration, package change, trajectory work, or production
selection is authorized.

Correctness compared the current owner, its rerun, the candidate, and its
rerun over all `25,344,000` output words across the four shapes. The three
pairwise comparisons covered `76,032,000` U32 comparisons with zero
differences. Signed zero, subnormal/normal boundaries, alternating
cancellation, bounded maximum-finite FP16, activation rounding boundaries,
qNaN complete-write detection, guards, and the partial M2250 tile all passed.
There were zero uncaptured GPU errors and zero device losses.

Eight balanced rounds alternated arm order and rotated shape order. Every
shape's mean and median was faster on both timestamped GPU and fenced wall,
and the complete weighted score won all `8/8` GPU pairs and all `8/8` wall
pairs. Its complete-score means were:

- GPU `212.443136 -> 191.668224 ms`, a `1.1083899645x` speedup and
  `20.774912 ms` saving; and
- wall `224.937500 -> 202.487500 ms`, a `1.1108710419x` speedup and
  `22.450000 ms` saving.

The medians were materially stronger: GPU
`210.731008 -> 177.143808 ms` (`1.1896041435x`, `33.5872 ms` saved) and wall
`220.700000 -> 187.350000 ms` (`1.1780090764x`, `33.350000 ms` saved).
Wall/GPU savings agreement passed at `1.0806303` for means and `0.9929378`
for medians.

The frozen gate nevertheless requires both mean and median GPU and wall
speedups to reach `1.12x` and both savings to reach `20.8334 ms`. The mean GPU
and wall speedups missed, and mean GPU saving missed by `0.058488 ms` per
layer/evaluation. Its `24 * 8` projection was `3,988.783104 ms`, below the
required `4,000 ms`; mean wall and both median projections exceeded four
seconds, but those passes cannot waive the failed conjuncts. The literal page
decision was therefore inconclusive with no follow-up authorization. There
was no unchanged timing retry.

The external thermal trace contained `84` observations, all level 0, with no
missing values and a `1,016 ms` maximum gap. It began before READY, supplied
`32` continuous nominal gate observations, launched timing `1,054 ms` after
the gate, and covered cleanup. Cleanup reconciled `22/22` buffers and `92/92`
maps/unmaps, peaked at `293,317,664` tracked bytes, passed repeated destroy and
post-destroy rejection, and ended with zero live buffers, bytes, or maps.

The complete `124,834`-byte browser receipt is retained under ignored
`optimization/artifacts` and bound by SHA-256
`27a4b899d469bfb47cf1ff42ec8c95a7623085722b623bb238f86aed4c0b5ffe`.
The `21,662`-byte external JSONL trace is similarly bound by
`ea21d97f514bb2b934acb677f59cf25b74eefd058c696af8947eba3f0d5d0e76`.
See the [compact result](../results/OPT-0078/result.json) for exact samples,
shape summaries, hashes, gates, thermal facts, and resource accounting.

Production retains OPT-0009's dense owner within the accepted OPT-0073 path.
Do not repeat this unchanged benchmark or relax its threshold. A materially
different tile/dataflow, browser/compiler, or target GPU requires a new
experiment ID; this result authorizes no inference-speed or product claim.
