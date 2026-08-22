# OPT-0067 — Thermally isolated quad-query evaluation-slice screen

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Frozen receipt status: `failed` (`passed: false`)
- Decision: `evaluation-slice-non-pass-keep-query8-default`
- Risk: diagnostic attention-owner routing and thermal attribution; no model
  arithmetic, package, or production-default change

## First-principles basis

OPT-0062 established that its fixed-WG256 quad-query owner is production-graph
exact: `442,368,000` actual-layer U32 comparisons had zero mismatches, all
eight evaluation taps and the final latent were raw-U32 exact, and the
independent quad repeat was exact. It also measured a
`1.56074021326695x` aggregate full-self speedup and
`12896.099999785423 ms` aggregate graph saving.

That full-graph performance gate did not pass. The forward graph, DiT stage,
and evaluation-0 slice regressed while the reverse direction improved; both
multi-minute traces spent most observations at thermal levels 1-2, and
unrelated families drifted. Repeating the same two-arm-per-trace full graph
would not better identify the cause.

The smallest production-shaped causal screen is one complete M2250 denoise
evaluation. It contains all `24` transformer layers and exactly `12` selected
odd-layer full-self-attention routes, but only one eighth of the repeated
denoiser workload. Four separately cooled arms control order while ensuring
that one arm never inherits another arm's heating trace.

## Frozen identities and scope

- Use the authenticated revision-7 exact-dense package only: manifest
  `/model/files-fp16-dit-rev7-oracle/manifest.json`, SHA-256
  `d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f`,
  runtime profile `opt-0009-fp16-fp32-dense-v1`, and kernel set
  `opt-0009-n256-k32-fp16-fp32-v1`. Authenticate the main manifest SHA-256
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`.
  Do not use revision-8 K4 dense weights or any approximate dense owner.
- Reuse unchanged query8 control profile
  `fixed32-subgroup-query8-default` and unchanged OPT-0062 candidate profile
  `opt-0062-fixed32-quad-query32-full-self-v1`, kernel set
  `opt-0062-query8-plus-quad-query32-full-self-v1`, and kernel ID
  `opt-0062-fixed32-quad-query32-full-self-v1`. Authenticate quad WGSL SHA-256
  `7b9af88e0f24f96da54dd525850da2432158fb4a7cdaccab1633b961f10911e6`.
- Use the immutable OPT-0062 canonical direct request, SHA-256
  `031e418ac5db37355fe5e265a005cb280e02ce418e560312ac89fa184bb8862f`,
  with its exact C98 conditioning, M2250 shape, initial latent/noise, first
  shift-3 timestep, masks, valid lengths, and DCW inputs. Both arms must consume
  byte-identical tensors; no arm may derive or mutate its own fixture.
- Execute only denoise evaluation `0`, including all `24` layers and its normal
  evaluation-result write. Do not execute evaluations `1..7`, acquire VAE
  weights, decode audio, normalize, write WAV output, invoke the planner, or
  alter scheduling, queue depth, submission topology, package bytes, dense
  owners, sampler math, or any non-attention operation.
- Within that evaluation, quad may own only the exact twelve labels
  `ace-dit-eval-0-layer-L-self-full-attention` for
  `L={1,3,5,7,9,11,13,15,17,19,21,23}`. Query8 remains unchanged for every
  sliding and cross-attention operation. Any label, count, shape, mode,
  fixed-subgroup, descriptor, or owner mismatch must fail closed.
- Query8 remains the public production default throughout this experiment.

## Correctness and measurement-ownership gates

1. Before accepted timing, run an untimed correctness preparation on the same
   authenticated bytes. Require exactly `12` selected routes, each with
   `4,608,000` outputs, for `55,296,000` raw-U32 actual-layer comparisons total.
   Query8 and quad must have zero mismatches, non-finite values, unwritten
   outputs, or changed canaries at every route.
2. Require raw-U32 identity for all `288,000` words of the complete evaluation
   result. Its SHA-256 must reproduce OPT-0062 evaluation 0:
   `d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286`.
   All four timed arms must reproduce that same result; the two query8 arms and
   two quad arms therefore also serve as deterministic repeats.
3. Correctness capture must be outside the accepted timing boundary. The timed
   path must preserve the ordinary evaluation command-buffer graph and record
   `identityExtraSubmitCount = 0` and `identityExtraDrainCount = 0`. No
   timestamp query, diagnostic copy, readback, or attribution mechanism may add
   a measurement-only queue submit or queue drain. Retain explicit ordinary
   submit/drain counts for equality across arms.
4. Each arm must have one FIFO graph owner, sequential non-overlapping package
   ownership, unchanged bounded queue depth, drain-before-release, idempotent
   complete backend/device destruction, no post-cleanup callbacks or work, and
   zero live buffers/bytes. Cancellation and injected-failure paths must clean
   up without executing later evaluations or acquiring VAE weights.

## Four-arm thermal contract

Run exactly four independent arms in ABBA order:

1. `A1`: query8
2. `B1`: OPT-0062 quad
3. `B2`: OPT-0062 quad
4. `A2`: query8

Each arm must initialize the same package and inputs, reach timing-ready, and
then receive its own fresh thermal gate of at least `30` continuous seconds at
documented nominal level 0. The final gate observation must be current when
the arm starts. Begin a distinct raw thermal trace before that arm's gate and
poll continuously through its single evaluation, readback, cleanup, backend
and device disposal. Stop and authenticate that trace before preparing the
next arm. Never execute two arms under one trace, and never reuse a prior arm's
gate.

Retain every gate and trace observation, timestamps, maximum polling gap,
transition, raw-trace SHA-256, initialization/cleanup ordinal, and rejected
setup attempt. A non-nominal transition after a valid start is disclosed under
the repository protocol; directionally mixed, marginal, or
variance-overlapped evidence remains inconclusive rather than being promoted by
choosing favorable samples.

## Performance gate

For every arm retain fenced wall time for the twelve full-self calls, the
complete evaluation, command/drain, requested idle, readback, residual, and
each OPT-0062 family-attribution bucket. No median may replace the four raw
samples.

Treat `A1/B1` as the forward pair and `B2/A2` as the reverse pair. All of the
following are required:

- `B1 < A1` and `B2 < A2` for both full-self wall and complete evaluation
  wall;
- aggregate full-self speedup
  `(A1_full_self + A2_full_self) / (B1_full_self + B2_full_self)` is at least
  `1.30x`;
- the explicitly projected eight-evaluation saving
  `8 * (((A1_eval - B1_eval) + (A2_eval - B2_eval)) / 2)` is at least
  `3,000 ms`; this is a slice-derived projection, not measured full-graph or
  product saving; and
- in both paired directions, every non-full-self family with a nonzero control
  wall and the aggregate non-full-self evaluation wall regress by no more than
  `2%`. Retain exact absolute deltas as well as ratios so small buckets remain
  interpretable; the `2%` gate is not waived for a small bucket.

Failure of any direction, threshold, correctness, thermal-provenance, or
lifecycle condition is a non-pass. Report a clear negative only when the
bounded evidence identifies one; otherwise retain an inconclusive result and
do not repeat the unchanged multi-minute OPT-0062 graph.

## Result

The frozen receipt is a literal non-pass: `status = failed`, `passed = false`.
The causal target result is nevertheless strongly positive. Quad made the
twelve full-self routes faster in both paired directions, made the complete
evaluation faster in both directions, produced a `1.5808922936358476x`
aggregate full-self speedup, and projects
`5828.400000095367 ms` saving over eight evaluations. The experiment failed
only the pre-registered per-family non-full-self `2%` gate. That failure is
retained exactly and is classified as inconclusive overall evidence, not as a
quad-query regression.

### Receipt and run identity

- Receipt schema:
  `ace-opt-0067-quad-query-evaluation0-thermal-gate-v1`; experiment:
  `OPT-0067`; decision:
  `evaluation-slice-non-pass-keep-query8-default`.
- Core and harness commit:
  `e8ea8b0e84bb4c38f8f39cfd881fac102c4b5895`.
- Machine: `Mac15,12`, 10 GPU cores, `17,179,869,184` bytes memory; macOS
  `26.5.2` build `25F84`; Google Chrome `151.0.7922.138`.
- Request SHA-256:
  `031e418ac5db37355fe5e265a005cb280e02ce418e560312ac89fa184bb8862f`
  (`366` canonical JSON bytes). Main manifest SHA-256:
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`;
  dense manifest SHA-256:
  `d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f`.
- Dense runtime profile: `opt-0009-fp16-fp32-dense-v1`. Candidate runtime,
  kernel set, and kernel ID remained
  `opt-0062-fixed32-quad-query32-full-self-v1`,
  `opt-0062-query8-plus-quad-query32-full-self-v1`, and
  `opt-0062-fixed32-quad-query32-full-self-v1`; authenticated WGSL SHA-256:
  `7b9af88e0f24f96da54dd525850da2432158fb4a7cdaccab1633b961f10911e6`.

### Correctness and untimed preparation

- The untimed correctness owner passed. Exactly `12` selected actual-layer
  routes × `4,608,000` outputs produced `55,296,000` raw-U32 comparisons with
  zero mismatches, zero non-finite values, zero changed canaries, one identity
  copy, zero extra command buffers, and zero extra queue drains. The route
  inventory was `96` quad full-self, `96` query8 sliding, `192` query8 cross,
  zero query8-other, and zero unintended quad routes; only the 12
  evaluation-0 routes executed, leaving `84` future quad routes inactive.
- All `288,000` evaluation-result words were raw-U32 exact. The correctness
  owner and all four timed arms reproduced SHA-256
  `d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286`.
- The correctness profile descriptor SHA-256 was
  `d480bde986cba12068e462093169ef1a6cf3ceb45987eabb82ef8c8fe07eca47`
  (`1,886,366` serialized bytes, `6,833` members). It retained `341` graph
  command buffers, one readback command buffer, `342` total command buffers,
  `341` graph drains, `342` total drains, `25` precompute command buffers,
  `316` evaluation command buffers, `341` timings, and `3,069` timing-storage
  bytes.
- Correctness initialization began at epoch millisecond `1786852977153` and
  took `29164.699999928474 ms`. Its correctness-only doubled owner and all
  profile capture were excluded from accepted timing. It retained `158`
  initialization progress events, `74,021` generation progress events, and
  lifecycle ordinals created/init/ready/authorized/checkpoint/cleanup/dispose-
  start/dispose-complete `1/2/0/0/3/4/5/6`.

Every timed arm independently performed preparation before its thermal gate;
none of these values entered the accepted evaluation timing:

| Arm | Owner | Initialize start epoch ms | Initialize wall ms | Pre-timing-ready wall ms | Authorization wait ms | Init/generation events | Lifecycle ordinals created/init/ready/authorized/checkpoint/cleanup/dispose-start/dispose-complete |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A1 | query8 | 1786853022194 | 30924.5 | 6536.799999952316 | 304030.60000002384 | 158 / 70,205 | 7/8/9/10/11/12/13/14 |
| B1 | quad | 1786853397301 | 29522.699999928474 | 5647.700000047684 | 114032.10000002384 | 158 / 73,640 | 15/16/17/18/19/20/21/22 |
| B2 | quad | 1786853577396 | 29815.800000071526 | 5696.699999928474 | 145028.90000009537 | 158 / 73,665 | 23/24/25/26/27/28/29/30 |
| A2 | query8 | 1786853788026 | 29589.400000095367 | 5637.899999976158 | 133677.5 | 158 / 73,886 | 31/32/33/34/35/36/37/38 |

Query8 arms authenticated descriptor SHA-256
`aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76`;
quad arms authenticated
`d480bde986cba12068e462093169ef1a6cf3ceb45987eabb82ef8c8fe07eca47`.
Each arm retained the same `341` graph + one readback = `342` ordinary submits
and `342` ordinary drains, `25` precompute and `316` evaluation command
buffers, `6,833` descriptor members, `341` timings, and `3,069` timing bytes.

### Exact four-arm timing samples

All values are milliseconds and are retained without rounding:

| Arm | Owner | Full-self | Evaluation wall | Non-full-self evaluation | Graph wall | Command/drain | Requested idle | Readback | Residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | query8 | 1421.0999995470047 | 8509.699999928474 | 7088.60000038147 | 8940.299999952316 | 8397.499999523163 | 340 | 26.300000071525574 | 202.80000042915344 |
| B1 | quad | 911.299999833107 | 7821.899999976158 | 6910.600000143051 | 8171.200000047684 | 7654.299999594688 | 340 | 25.300000071525574 | 176.9000004529953 |
| B2 | quad | 863.9000002145767 | 7782.099999904633 | 6918.199999690056 | 8348 | 7827.699999809265 | 340 | 28.40000009536743 | 180.30000019073486 |
| A2 | query8 | 1385.3000001907349 | 8551.399999976158 | 7166.099999785423 | 9122.399999976158 | 8597.800000548363 | 340 | 25.399999976158142 | 184.5999994277954 |

The exact family-attribution samples were:

| Family | A1 query8 | B1 quad | B2 quad | A2 query8 |
| --- | ---: | ---: | ---: | ---: |
| precompute | 0 | 0 | 0 | 0 |
| cross-cache | 0 | 0 | 0 | 0 |
| timestep | 22.799999952316284 | 39.89999997615814 | 42.60000002384186 | 46.89999997615814 |
| input | 44.39999997615814 | 51.5 | 105.39999997615814 | 65.80000007152557 |
| attention-projections | 325.9999997615814 | 258.0000001192093 | 201.89999985694885 | 232.40000021457672 |
| self-full | 1421.0999995470047 | 911.299999833107 | 863.9000002145767 | 1385.3000001907349 |
| self-sliding | 0 | 0 | 0 | 0 |
| cross-attention | 0 | 0 | 0 | 0 |
| feed-forward | 3370.6000002622604 | 3365.8999996185303 | 3324.899999976158 | 3413.300000190735 |
| plumbing | 0 | 0 | 0 | 0 |
| output | 7.300000071525574 | 7.299999952316284 | 6.5 | 6.299999952316284 |
| sampler-dcw | 0.5 | 0.5 | 1 | 0.5 |
| mixed | 2837.500000357628 | 2709.300000190735 | 2759.399999976158 | 2919.899999856949 |

### Frozen performance-gate decision

- Full-self and complete evaluation both improved forward (`A1 -> B1`) and
  reverse (`A2 -> B2`). Aggregate full-self speedup was
  `1.5808922936358476x`, passing the `1.30x` gate. The exact projected
  eight-evaluation saving was `5828.400000095367 ms`, passing the `3,000 ms`
  gate.
- The aggregate non-full-self wall improved in both directions:
  `7088.60000038147 -> 6910.600000143051 ms`, a
  `-178.00000023841858 ms` / `-0.025110741222362565` ratio forward, and
  `7166.099999785423 -> 6918.199999690056 ms`, a
  `-247.90000009536743 ms` / `-0.03459343298346251` ratio reverse.
- The strict individual-family gate nevertheless failed. Forward input was
  `+7.100000023841858 ms` / `+0.15990991053275683`, and forward timestep was
  `+17.100000023841858 ms` / `+0.7500000026142388`. Reverse input was
  `+39.59999990463257 ms` / `+0.6018237061031426`, reverse output was
  `+0.20000004768371582 ms` / `+0.031746039555156313`, and reverse sampler-DCW
  was `+0.5 ms` / `+1`. The sampler maximum is a doubling only because the
  control bucket was `0.5 ms`; the absolute delta remains explicitly retained.
- Consequently `performance.passed = false` and the overall receipt is
  `status = failed`, `passed = false`. The targeted quad evidence is strongly
  positive, but the frozen all-family gate makes this experiment
  inconclusive/non-pass rather than a selection result.

### Thermal gates, traces, and rejected preflights

All four accepted arms had distinct level-0 gates and through-disposal traces,
sampled with `notifyutil -g com.apple.system.thermalpressurelevel`. Every raw
observation is retained in the frozen receipt; the complete gate and trace
identities are:

| Arm | Gate start / complete epoch ms | Gate observations / max gap ms / non-nominal | Trace complete epoch ms | Trace observations / max gap ms / non-nominal | Transition epoch ms:level | Raw trace SHA-256 |
| --- | --- | --- | ---: | --- | --- | --- |
| A1 | 1786853313786 / 1786853363048 | 55 / 921 / 0 | 1786853388024 | 83 / 956 / 0 | 1786853363048:0 | `ce56e0d5fb22ad5a7bdc9c0151d38e4cbc5afe2d6371e731b9d392efa03898d7` |
| B1 | 1786853479545 / 1786853546210 | 74 / 928 / 0 | 1786853568418 | 99 / 928 / 0 | 1786853546210:0 | `c09dd5181e8f59393546ac5bdb19c641be248e4650e06a279994f24e9b132801` |
| B2 | 1786853678227 / 1786853757596 | 88 / 921 / 0 | 1786853780343 | 113 / 921 / 0 | 1786853757596:0 | `97f5f3de91905e4c440dac299c98096cf8d3f56ecf7a0eb554f5c92afdb89562` |
| A2 | 1786853889230 / 1786853955982 | 74 / 1004 / 0 | 1786853984204 | 105 / 1004 / 0 | 1786853955982:0 | `5253bf89272c91e587f9ec0b5f4f6b5066b0094f94226b32299820460b61712b` |

Two A1 setup attempts were rejected before accepted timing, at epoch
milliseconds `1786853202179` and `1786853271975`. Both were thermal-gate
rejections with `OPT-0067 thermal observation is invalid`, and both retained
`timedGpuWorkStarted = false`; neither contributed a timing sample.

### Lifecycle, scope, and decision

The final lifecycle contract passed: one FIFO graph owner per arm, sequential
non-overlapping package ownership, no cross-arm backend/device reuse,
preparation excluded from timing, graph compilation before each gate, one
distinct trace per arm, drain before release, and complete disposal of every
backend and device. Each arm executed exactly one evaluation, encoded zero
later evaluations, and used exactly `342` ordinary submits and `342` ordinary
drains with zero identity-only submits or drains. Every arm destroyed DiT
before checkpoint, awaited generation cleanup before disposal, awaited backend
and device disposal, and never began VAE weight acquisition.

The receipt records `false` for production-default, dense-profile, package,
and scheduler changes; VAE weight acquisition/execution, audio, planner, later
evaluations, full-graph claim, listening claim, and under-one-minute claim are
also all `false`.

Because the frozen performance gate failed, this result authorizes no
diagnostic optimized-stack selection. Query8 remains the production default.
It does not authorize production integration, listening, a full graph, a
three-minute product run, or any product-speed claim.

## Authorization boundary

A pass authorizes selecting OPT-0062 quad only in a separately identified
diagnostic optimized stack for the next bounded stack measurement. It does not
integrate quad into the production default and does not authorize listening,
an eight-evaluation full graph, sampler changes, revision-8/selective dense,
VAE combinations, a three-minute Generate-to-WAV run, or an under-one-minute
claim. Any production selection or broader combined-stack claim requires its
own registered gate.

## Authority

- [OPT-0067 result](../results/OPT-0067/result.json), `150,537` bytes, SHA-256
  `66d96c21ddf9d2dc8c30fb87d8759c5709899b66166268fe91751d75b84a5b95`
- [OPT-0067 raw gates, traces, selected JSONL, and indices](../results/OPT-0067/raw/)
- [OPT-0062 record](OPT-0062-dit-quad-query-attention-integration.md)
- [OPT-0062 result](../results/OPT-0062/result.json), SHA-256
  `cf7db9c02496e216af79bde68387d32955f6eb57649b245e290c62332707db03`
- [OPT-0061 primitive record](OPT-0061-dit-attention-multi-query-wg256.md)
- [OPT-0037 revision-7 package authority](OPT-0037-dit-k4-layout-trajectory-integration.md)

No code, test, browser, GPU, package, runtime, or production-default change
occurred when this experiment was registered.
