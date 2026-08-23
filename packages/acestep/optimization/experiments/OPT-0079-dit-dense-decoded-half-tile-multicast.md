# OPT-0079 — DiT dense decoded half-tile multicast

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Date: 2026-08-21
- Author/agent: Codex
- Risk class: `exact increasing-K FP16-operand/FP32-accumulation ownership`
- Allocation baseline: pushed `main` commit
  `4084610c5e43dff2d388361965750b0f603400dd`

## First-principles basis

OPT-0078 proved that sharing converter-native packed weights across subgroups
is exact and directionally useful. Its M32xN256xK32/WG256 owner improved every
shape mean and median and won all eight weighted GPU and wall pairs, but its
mean timestamped-GPU projection was `3,988.783104 ms`, just below its frozen
four-second gate. The mechanism used the target device's full `16 KiB`
workgroup allocation, retained one packed-record decode per subgroup/lane/K,
and reused a weight tile across only 32 rows.

The next hardware-scale step is not a barrier or indexing micro-optimization.
It changes all three limiting terms together while retaining the existing
package bytes and exact arithmetic:

- reuse each staged weight half-tile across 64 rows rather than 32;
- stage only 128 columns, consuming `8 KiB` rather than the full `16 KiB`;
- decode each of the 512 packed records once during cooperative fill into
  native `vec4<f16>` workgroup storage, rather than unpacking the same record
  independently in all eight subgroups' inner loops; and
- retain only 32 FP32 accumulators per lane, half of production OPT-0009.

Use WG256 with eight fixed-32 subgroups. Each subgroup owns eight rows and each
lane owns four adjacent columns, producing an exact M64xN128 output tile. The
existing `dit-gemm-n256-k32-tile-major-v1` payload is addressed as its lower or
upper N128 half without repacking. A K32/N128 half contains 512 packed
`vec4<u32>` records. The 256 lanes each load exactly two records, decode their
eight FP16 values into two `vec4<f16>` entries, and collectively fill
`array<vec4<f16>, 1024>`. After a workgroup barrier, every subgroup reads one
four-column vector per lane and K. A second barrier prevents overwrite before
the next K32 tile.

This geometry has a credible occupancy and instruction-throughput advantage:
the shared allocation is halved, the eight subgroups expose the same row-band
parallelism, global weight fill is halved per workgroup, shared read width is
halved per lane, and packed decode work moves from the hot contraction loop to
one cooperative fill. Whether the target compiler/hardware turns those facts
into a material win remains an actual-GPU question.

## Static workload accounting

Use the four exact production shapes and multiplicities
`4/2/2/1`: M2250/K2048/N2048, M2250/K2048/N1024,
M2250/K2048/N6144, and M2250/K6144/N2048.

| weighted per layer/evaluation | production OPT-0009 | OPT-0078 | OPT-0079 candidate |
| --- | ---: | ---: | ---: |
| workgroups | 6,816 | 6,816 | 6,912 |
| scheduled multiply-adds | 133,412,421,632 | 133,412,421,632 | 135,291,469,824 |
| modeled activation requests | 2,084,569,088 B | 2,084,569,088 B | 4,227,858,432 B |
| modeled weight requests | 33,353,105,408 B | 8,338,276,352 B | 4,227,858,432 B |
| modeled operand requests | 35,437,674,496 B | 10,422,845,440 B | 8,455,716,864 B |
| FP32 accumulators per lane | 64 | 32 | 32 |
| declared workgroup storage | 0 | 16,384 B | 8,192 B |

The candidate has `36` M64 row tiles rather than `71` M32 tiles and N128
column counts `16/8/48/16`. The M2250 tail therefore schedules 2,304 rows
rather than 2,272, increasing workgroups and arithmetic by `1.40845%`. The
operand model deliberately charges every scheduled tail row. Its activation
requests double because an N128 tile revisits each row/K value twice as often,
while its weight requests fall by another factor of two versus OPT-0078.
Together they are `18.87%` below OPT-0078 and `76.14%` below production's
static request count. These figures are causal accounting, not speed claims.

## Novelty and closed-space boundary

This is not an unchanged retry of an abandoned dense experiment:

- OPT-0019 used M64xN128xK16/WG256 but staged scalar A and B panels, did not
  use subgroup activation broadcasts, did not consume the native K32 payload,
  and decoded no packed record into a typed FP16 weight panel. It measured an
  exact `1.31526x` weighted improvement but stopped on its older `1.55x` ratio.
- OPT-0021 retained OPT-0019's K16 A+B panel and 16x16 thread ownership while
  changing only panel vectorization; it was not this weights-only K32 path.
- OPT-0078 used M32xN256, a 16 KiB packed-U32 panel, four rows/eight columns per
  lane, and repeated packed decode inside each subgroup's K loop.
- OPT-0031 used WG512 M128xN128 A+B panels and is outside the device's selected
  WG256 limit. Approximate K2/K4, dot, FMA, INT8, and changed-reduction paths
  remain outside this experiment.

An M32/N128, WG128, K16/K64, packed-U32 panel, alternate row tile, or geometry
sweep is not authorized under this ID. A failure closes this exact
M64xN128xK32 decoded-half-tile mechanism on the current browser and GPU.

## Frozen candidate mechanism

1. Add one benchmark-only kernel with exact geometry M64xN128xK32/WG256,
   eight fixed-32 subgroups, eight rows and four columns per lane, and 32 FP32
   accumulators per lane.
2. Consume the existing N256/K32 packed-FP16 payload directly. Derive the
   physical N256 tile and lower/upper N128 half from `workgroup_id.x`; prove
   the physical record index for all N/K tiles and both halves.
3. Each local lane loads exactly two packed records per K32 half-tile. Decode
   each packed record once into two `vec4<f16>` entries in an exactly 8,192-byte
   workgroup array. Use exactly two uniform workgroup barriers per K32 tile.
4. Preserve the exact expression and order for every output: FP32 activation
   load, native-FP16 conversion, native-FP16 weight operand, FP32 widening,
   multiplication, and increasing-K `acc = acc + product`. Do not use `dot`,
   `fma`, reassociation, split K, FP16 accumulation, or matrix extensions.
5. Preserve row-major FP32 output, binding disjointness, one dispatch/range,
   the production package, sampler, graph, attention owner, scheduler, and
   selector. Fail closed unless `shader-f16`, fixed subgroup size 32, WG256,
   and at least 8,192 bytes of workgroup storage are explicitly available.
   Reject bias, non-M2250 rows, non-production shapes, malformed/aligned-short
   bindings, or any layout mismatch without a fallback.

## Static and correctness gate

1. Prove exact-once ownership for all M64xN128 outputs, all eight subgroups,
   every lane's eight-row/four-column microtile, and the partial M2250 tail.
2. Exhaustively prove all 512 logical half-tile record owners and their 1,024
   decoded FP16-vector destinations, plus exact physical lower/upper-half
   indices in every production N/K tile. Both barriers must be uniform and
   surround every possible overwrite.
3. Reconcile the four shapes against OPT-0009: valid output ranges, exact
   weight byte ranges and native layout, per-shape workgroups, scheduled rows,
   scheduled/valid multiply-adds, and the `4/2/2/1` totals above.
4. In the target browser compare current, current rerun, candidate, and
   candidate rerun over every output U32 for all four full M2250 shapes. Require
   zero mismatches, finite-class identity, complete qNaN-prefill overwrite,
   intact prefix/suffix/adjacent guards, and explicit M-tail coverage.
5. Retain the bounded signed-zero, subnormal/normal, alternating-cancellation,
   maximum-finite-FP16, and activation-rounding fixtures. Require raw-U32
   identity rather than a numerical envelope.
6. Require zero uncaptured GPU errors/device losses and balanced resource/map
   accounting after ordinary, repeated-destroy, setup-failure, and post-destroy
   rejection paths.

Any mismatch, missing record, unwritten output, alias, capability failure, or
resource leak stops before timing.

## Primitive performance gate

Compile, warm, and finish correctness before the thermal boundary. Begin
timing only after at least 30 continuous seconds at thermal level 0, using an
external absolute-cadence 1,000 ms trace that continues through cleanup.

Run eight fixed balanced rounds. Alternate current/candidate order and rotate
the four shape orders. Each sample is one identical compute pass, command
buffer, submit, timestamp-query pair, and matching drain. Exclude allocation,
upload, compilation, readback, hashing, and serialization. Retain all raw GPU
and fenced-wall samples and exact orders; perform no unchanged timing retry.

Define the complete score as
`4*T2048x2048 + 2*T2048x1024 + 2*T2048x6144 + T6144x2048` and the
feed-forward score as `2*T2048x6144 + T6144x2048`. The candidate qualifies
only if all conditions hold on both timestamped GPU and fenced wall:

- every shape's mean and median are strictly faster;
- the complete score wins at least seven of eight same-round pairs;
- complete-score mean and median speedup are each at least `1.15x`;
- complete-score mean and median saving are each at least `25.0 ms` per
  layer/evaluation, projecting at least `4,800 ms` over `24 * 8`; and
- wall savings remain within `0.75x..1.25x` of matching GPU savings.

A projection below two seconds, a consistently regressing shape, or a clear
decoded-panel/storage regression is negative. Thermal gaps, wall/GPU
disagreement, or directionally mixed/variance-overlapped evidence is
inconclusive. A non-pass authorizes no graph work and cannot be rescued by
relaxing a conjunct after observation.

## Escalation and integration gate

Only a passing primitive may add a distinct diagnostic dense profile using
the same package bytes and exact kernel. Then:

1. Compare independently cooled current/candidate/current/candidate complete
   24-layer M2250 evaluation slices. Require raw-U32 identity for selected
   actual dense outputs, exact evaluation-result identity/repeat, both paired
   directions faster, and at least `3,500 ms` projected eight-evaluation wall
   saving after unrelated-family deltas.
2. Reconcile all nine repeated dense routes per layer, every physical range,
   FIFO ordering, unchanged queue topology, cancellation/failure cleanup,
   drain-before-release, idempotent destroy, and zero live buffers/bytes/maps.
   The candidate must not introduce a new maximum-buffer or heavy-owner
   residency overlap.
3. Run all eight canonical M2250/C98 evaluations and require raw-U32 identity
   at every sampler tap and the final latent, plus a deterministic repeat.
4. Run one short generic-duration direct product and require identical final
   latent, raw pre-normalization waveform, seams, normalized WAV hash,
   metadata, and cleanup. Exact identity satisfies the quality gate without a
   new subjective listening round.
5. A production pass may select only this exact owner. Final release claims
   remain subject to PLAN.md's cumulative production and thermal protocol.

This experiment authorizes no approximate dense arithmetic, package revision,
attention/sampler/scheduler change, fewer denoise evaluations, native Metal
path, under-one-minute claim, or inference-speed claim before those gates.

## Authority

- Current production selection:
  [OPT-0073](OPT-0073-revision7-webcrypto-production-selection.md)
- Immediate causal predecessor and exact result:
  [OPT-0078](OPT-0078-dit-dense-weight-tile-multicast.md)
- Prior distinct M64/N128 cooperative panel:
  [OPT-0019](OPT-0019-dit-dense-cooperative-panels.md)
- Dense utilization authority:
  [OPT-0043](OPT-0043-webgpu-timestamp-utilization-profile.md)

No candidate code, test, browser execution, GPU work, package change, or
production selection had occurred when this experiment was allocated.

## Results

OPT-0079 is inconclusive and remains benchmark-only. The exact kernel and its
browser gate were implemented at
`aade2c0223383bab99cf477e7697e2394c05a380`, after registration commit
`6313bfbaa61dee91f065d1652ee8eb1446f987b8`. Static ownership, all four
production shapes, actual-GPU raw-U32 identity, thermal launch/completion, and
lifecycle gates passed. The weighted complete score improved directionally,
but the candidate missed every frozen magnitude threshold, had two
shape-statistic regressions, won only six of eight pairs, and failed mean
wall/GPU savings agreement. No diagnostic profile, graph integration, package
change, trajectory work, or production selection is authorized.

Correctness compared the current owner, its rerun, the candidate, and its
rerun over all `25,344,000` output words across the four shapes. The three
pairwise comparisons covered `76,032,000` U32 comparisons with zero
differences. Signed zero, subnormal/normal boundaries, alternating
cancellation, bounded maximum-finite FP16, activation rounding boundaries,
qNaN complete-write detection, guards, and the partial M2250 tile all passed.
The target-browser malformed alias was rejected before allocation or
submission. There were zero uncaptured GPU errors and zero device losses.

Eight balanced rounds alternated arm order and rotated shape order. The
complete weighted-score means were:

- GPU `214.458368 -> 204.267520 ms`, a `1.0498897133x` speedup and
  `10.190848 ms` saving; and
- wall `230.187500 -> 214.450000 ms`, a `1.0733854049x` speedup and
  `15.737500 ms` saving.

Its medians were GPU `210.501632 -> 191.201280 ms` (`1.1009425878x`,
`19.300352 ms` saved) and wall `222.700000 -> 201.400000 ms`
(`1.1057596833x`, `21.300000 ms` saved). The score won only `6/8`
timestamped-GPU pairs and `6/8` fenced-wall pairs versus the required `7/8`.
M2250/K2048/N2048 regressed at mean GPU (`0.9921182266x`), and
M2250/K2048/N1024 regressed at median GPU (`0.9871794872x`); the other two
shapes passed their four directional statistics.

The frozen gate requires every shape statistic to improve, both pair counts
to reach seven, all mean/median GPU and wall speedups to reach `1.15x`, and all
matching savings to reach `25 ms`. None of the four speedup or four saving
thresholds passed. The `24 * 8` projections were mean/median GPU
`1,956.642816/3,705.667584 ms` and mean/median wall
`3,021.600017/4,089.600037 ms`, all below `4,800 ms`. Mean wall saving was
`1.5442777761x` mean GPU saving, outside the permitted `0.75x..1.25x`; the
median ratio was `1.1036068249x`, but one agreeing statistic cannot waive the
failed conjunct. The literal page decision was inconclusive with no follow-up
authorization, and there was no unchanged timing retry.

The external thermal trace contained `67` observations, all level 0, with no
missing values and a `1,010 ms` maximum gap. It began before READY, supplied
`32` continuous nominal gate observations, launched timing `499 ms` after the
gate, and covered cleanup. Cleanup reconciled `22/22` buffers and `92/92`
maps/unmaps, peaked at `293,317,664` tracked bytes, passed repeated destroy and
post-destroy rejection, and ended with zero live buffers, bytes, or maps.

The complete `133,315`-byte browser receipt is retained under ignored
`optimization/artifacts` and bound by SHA-256
`ed4033518620efc5404ef0416238de84bf16631193f330f16e391e5f2b4a1842`.
The `17,276`-byte external JSONL trace is similarly bound by
`499c3ebe469567c81232b2d30707dca65fc550d794eca9d34dc41ccd82a3b9ea`.
See the [compact result](../results/OPT-0079/result.json) for exact samples,
shape summaries, hashes, gates, thermal facts, and resource accounting.

Production retains OPT-0009's dense owner within the accepted OPT-0073 path.
Do not repeat this unchanged benchmark or relax its thresholds. This exact
M64xN128xK32 decoded-half-tile geometry is closed on the current browser and
GPU. A materially different dataflow, browser/compiler, or target GPU requires
a new experiment ID; this result authorizes no inference-speed or product
claim.
