# OPT-0080 — DiT depth-two completion epochs

## Status

- Evidence: `positive`
- Disposition: `integrated/product-exact for authenticated direct DiT and exact C2314 VAE scheduling`
- Date: 2026-08-21
- Author/agent: Codex
- Risk class: `exact FIFO scheduling, cancellation, and aliased-resource lifetime`
- Allocation baseline: pushed `main` commit
  `5b636d6c5c652093b5a81d09c7f6fc5fb9955ebf`

## First-principles basis

The accepted OPT-0073 direct 180-second product takes
`94,774.80000007153 ms` Generate-to-WAV, including
`54,986.200000047684 ms` in DiT and `32,586.700000047684 ms` in VAE. It
records `3,803` cooperative GPU queue drains and `3,800 ms` of requested
cooperative idle. At the allocation baseline, the scheduler submitted one
physical command buffer, waited for its cumulative
`queue.onSubmittedWorkDone()` fence, reported progress, and slept for a real
millisecond before creating the next command buffer. That policy is safe and
responsive, but it makes CPU encoding, browser promise dispatch, driver
submission, and GPU execution almost completely serial.

Topology-matched evidence bounds a material opportunity rather than a
micro-optimization:

- the current production-attention DiT graphs in OPT-0062 contain
  `2,552 ms` of explicit idle plus about `1,420--1,425 ms` of host residual
  outside summed submit-to-matching-fence intervals, for an eligible scheduler
  gap of about `3,976.2/3,971.6 ms`;
- the two production-shaped C2314 VAE windows in OPT-0059 expose about
  `723.9 + 726.4 = 1,450.3 ms` of corresponding scheduling gap, including
  `1,110 ms` requested idle; and
- the remaining accepted product stages request another `137 ms` of idle.

The hard zero-gap ceiling is therefore about `5,561.2 ms`. A fixed policy that
keeps at most two singleton command buffers submitted and forces a true
queue-empty boundary after every four completions projects about `2,980 ms`
for DiT and about `4,171 ms` across the currently eligible stack. These are
mechanism budgets, not measured product claims.

Depth two is the smallest queue that can overlap the CPU/driver/GPU pipeline.
An epoch of four preserves frequent true drains and real idle turns while
eliminating three of four depth-one intervals. Increasing depth, sweeping epoch
length, or coalescing commands under this ID would confound the causal result
and expand abort latency without evidence.

## Novelty and closed-space boundary

OPT-0034 combined eight or sixteen graph quanta into one physical command
buffer. Its arithmetic was exact, but the thermal result was inconclusive and
its record permits a later shorter balanced/interleaved evaluation slice.
OPT-0080 is that distinct follow-up: it never coalesces commands, changes no
encoder contents, and retains one cumulative completion fence per physical
command buffer. It changes only how many already-submitted singleton buffers
may exist before the oldest fence is observed.

The control and candidate are frozen:

- `depth1-epoch1`: current production scheduler, maximum outstanding one;
- `opt-0080-depth2-epoch4`: maximum outstanding two, maximum four submitted
  in one phase-aligned epoch, then a true drain and an awaited real 1 ms idle.

Do not add a depth-three arm, alternate epoch lengths, command-buffer
coalescing, timestamp queries, kernel changes, graph fusion, uniform-pool
reuse, VAE selection, or production default under the initial screen. A
failure closes this exact depth-two/four-completion DiT mechanism. A later
geometry or scheduling policy needs a new ID and a new resource audit.

## Frozen exact candidate mechanism

Within each phase-aligned epoch of at most four physical command buffers:

1. Lazily create and submit command 0, then immediately request its cumulative
   completion fence. Do the same for command 1.
2. Await the oldest fence in FIFO order. Report exactly that command's timing
   and progress, check failure and abort, then backfill at most one command.
3. Once four commands have been submitted in the epoch, stop backfilling and
   settle the remaining fences in FIFO order. Because no later work was
   submitted, the final fulfilled fence is a true queue drain.
4. If another epoch remains, start and await the real 1 ms cooperative idle
   only after that true drain. No idle may overlap outstanding GPU work.
5. Preserve every physical command buffer, encoder, dispatch, binding,
   resource, arithmetic operation, FIFO order, and per-submit cumulative
   fence. Maximum submitted-but-unreported work and maximum pending graph
   descriptors are both exactly two.

Every fence promise must be wrapped immediately into a fulfilled/rejected
outcome so a younger rejection cannot become unhandled while the oldest fence
is awaited. Creation is lazy: the implementation may never create more than
the one command needed to fill the second outstanding slot.

On observed abort, command creation/encoding failure, submission failure,
synchronous fence-capture failure, asynchronous fence rejection, callback
failure, idle failure, or device loss:

- stop all backfill and public progress immediately;
- preserve the original failure and submit no later graph work;
- settle every fence belonging to already-submitted work;
- when the last usable fence rejected, attempt one terminal recovery
  `onSubmittedWorkDone()` fence;
- release the FIFO lease and aliased resources only after a terminal fulfilled
  fence or confirmed device loss; and
- rethrow the original failure while retaining secondary settlement failures.

The existing depth-one path and `AceUniformPool` stay unchanged. The pool
rewrites and recycles uniform offsets only after a true drain and is not safe
for this experiment. Conditioning, planner, VAE, and all public production
profiles remain depth one until separately authorized.

## Exact evaluation-0 topology

Use the accepted production tuple and immutable OPT-0067 evaluation fixture:

- main manifest SHA-256
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`;
- revision-7 exact-dense manifest SHA-256
  `d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f`;
- dense profile `opt-0009-fp16-fp32-dense-v1` and accepted OPT-0070 quad
  attention production owner;
- canonical direct-request SHA-256
  `031e418ac5db37355fe5e265a005cb280e02ce418e560312ac89fa184bb8862f`;
- M2250/C98, evaluation 0 only; and
- expected `288,000`-word evaluation result SHA-256
  `d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286`.

The immutable full descriptor table contains `2,553` physical graph command
descriptors and `6,833` graph members. The executed evaluation-0 prefix is the
first `341` descriptors; validate its exact descriptor identity and evaluation
boundaries independently of that table-wide member count:

- indices `0..24`: `25` precompute command buffers;
- indices `25..340`: `316` evaluation command buffers;
- evaluation families: timestep `1`, input `1`, attention projections `24`,
  self-full `48`, feed-forward `72`, mixed `168`, output `1`, and sampler/DCW
  `1`; and
- one separate ordinary result-readback command buffer after the graph.

Force a true epoch boundary after physical index 24. Evaluation work may not
queue behind precompute. The resulting static contract is:

| metric | depth-one control | depth-two candidate |
| --- | ---: | ---: |
| graph command buffers / fences | `341 / 341` | `341 / 341` |
| graph true drains | `341` | `ceil(25/4) + 316/4 = 86` |
| graph idle intervals | `340` | `85` |
| total commands / fences with readback | `342 / 342` | `342 / 342` |
| total true drains with readback | `342` | `87` |
| total requested idle | `341 ms` | `86 ms` |
| maximum outstanding | `1` | `2` |

A future authorized full graph retains `2,553` graph command buffers and
fences. Its candidate topology is `639` true graph drains and `638` graph
idle intervals; the separate ordinary readback makes totals `2,554` fences,
`640` true drains, and `639` idle intervals.

## Graph-owner resource invariant

`AceDitGraphOwner.run()` currently attributes a completed command through one
mutable `active` descriptor. That is invalid once command N+1 can be prepared
before N completes. The candidate must replace it only on its diagnostic path
with a bounded, immutable, command-index-keyed pending map:

- insert `{batch, batchCommandId, entries}` before submitting index N;
- reject duplicate or non-FIFO indices;
- retrieve and remove only index N from the unified completion callback;
- combine timing attribution and graph progress before deletion;
- require map size at most two; and
- on exceptional exit, retain submitted descriptors until terminal settlement,
  then clear all state before resource release.

All graph inputs, control buffers, uniforms, bind groups, and dispatches must be
prepared before `run()`. No `queue.writeBuffer`, mapping, host mutation,
destruction, new GPU buffer, readback, timestamp query, or measurement-only
submission may occur while graph work is outstanding. FIFO queue ordering is
the only dependency required for the existing aliased activation storage.

## Metrics and terminology

A cumulative completion fence is not necessarily a queue drain when a younger
command has already been submitted. Record these separately:

- `commandBuffersSubmitted`;
- `completionFenceRequestedCount`, `completionFenceSettledCount`, and
  `completionFenceRejectedCount`;
- `trueQueueDrainCount` and `completionEpochCount`;
- `requestedCooperativeIdleMs` and `cooperativeIdleTurns`;
- `maximumOutstandingCommandBuffers`; and
- `maximumPendingDescriptorCount`.

Each per-command interval is named `submitThroughCompletionFenceMs`. These
intervals overlap under depth two and are non-additive: no test, result builder,
report, or performance claim may sum them or assign them as exclusive family
wall time. Record disjoint `epochSubmitThroughTrueDrainMs` instead. Only epoch
walls may be summed and reconciled against graph wall. Authoritative timing is
the complete evaluation, graph, and backend fenced wall.

A true drain means the final epoch fence was requested after the epoch's last
submit, no later work was submitted before it fulfilled, and the diagnostic
owner had exclusive use of the queue.

## Static, failure, and lifecycle gates

Before actual-GPU timing, deterministic tests must prove:

1. Exact epoch plans for phase/tail counts one, two, three, four, five, ten,
   the M2250 `7 + 79` evaluation-0 plan, and the full `7 + 8*79` plan.
2. Exactly one fence per submitted command; maximum two outstanding commands
   and pending descriptors; FIFO encode, submit, completion, attribution, and
   progress order; and no more than one lazy lookahead command.
3. No fifth submission before command 4's true drain and awaited idle. Phase
   boundaries and final tails are true drains. Idle begins only at a true
   drain and is awaited before the next submission.
4. Abort and every synchronous/asynchronous failure class stop backfill,
   suppress later progress, settle submitted work, and delay lease/resource
   release. Cover create, encode, submit, fence capture, fence rejection,
   callback, timing, idle, final tail, cleanup, repeated destroy, and device
   loss explicitly.
5. The single mutable `active` descriptor cannot reappear; the pending map is
   exact, bounded, empty after success/failure/cancellation, and every physical
   quantum executes exactly once in original order.
6. Aliased resources are never host-mutated, remapped, destroyed, or returned
   while work is outstanding. Graph/model/runtime destruction occurs exactly
   once after terminal settlement, with zero live buffers, bytes, maps,
   descriptors, callbacks, or leases.
7. The benchmark-only selector cannot affect `AceUniformPool`, conditioning,
   planner, VAE, ordinary production requests, or public API defaults.
8. Result contracts cannot sum overlapping per-fence latencies and reconcile
   every command, fence, true drain, epoch, idle, phase, callback, and resource
   count against the frozen topology.

Any missing command, extra submission, alias mutation, post-failure progress,
early release, unhandled rejection, non-FIFO callback, topology mismatch, or
resource leak stops before timing.

## Actual-GPU correctness and cancellation preflights

Run one untimed A/B correctness preflight on the same authenticated immutable
bytes. Require identical authenticated full-table identity (`2,553`
descriptors and `6,833` members), exact identity and evaluation boundaries for
the executed `341`-descriptor prefix, exactly `341` graph submissions per arm,
zero differences across all `288,000` output U32 words, the expected result
hash, exact deterministic repeats, finite/nonzero output, no uncaptured WebGPU
error, and no device loss. No result guard or canary is allocated or measured
by this preflight, and no such evidence is claimed.

The candidate must report exactly `341` graph completion fences, `86` true
graph drains, `85` graph idles, maximum outstanding two, maximum pending
descriptors two, and one separate ordinary readback command/fence. Correctness
capture is untimed and may add no GPU submissions or drains to either timed
arm.

Run the actual cancellation preflight through the benchmark callback against
the real prepared DiT evaluation-0 graph, with abort observed from a completion
callback while exactly one successor is already submitted. Require exactly
one prefetched tail command, no backfill, no later public progress/checkpoint,
and await scheduler settlement, graph rejection and cleanup, model destruction,
and backend cleanup. Combine that actual lifecycle ordering with the static
tracked-destruction proof before accepting zero live resources. Require no
more than `1,000 ms` from abort observation through rejection and completed
cleanup.

## Thermal evaluation-slice performance gate

Compile and finish correctness/cancellation before accepted timing. Run four
independent, freshly initialized and cooled arms in exact ABBA order:

1. `A1`: `depth1-epoch1`;
2. `B1`: `opt-0080-depth2-epoch4`;
3. `B2`: `opt-0080-depth2-epoch4`;
4. `A2`: `depth1-epoch1`.

Each arm gets a fresh adapter/device/backend/package/graph owner and a separate
external absolute-cadence thermal trace. Begin timing only after at least 30
continuous seconds at level 0 and poll through evaluation readback, cleanup,
backend destruction, and device disposal. Retain every sample and rejected
preflight. Do not rerun unchanged executed timing solely for a better thermal
trace.

Record precompute wall through its boundary idle, evaluation wall from before
creating command 25 through the final graph true drain, graph wall,
graph-to-readback idle, readback fence/map, backend wall, every epoch wall, all
non-additive fence latencies, resource/lifecycle ordinals, and a main-page
50 ms heartbeat.

Responsiveness passes only if no heartbeat gap exceeds `500 ms`, candidate p99
is no worse than `max(100 ms, 1.25 * paired-control p99)`, and candidate maximum
is no worse than `max(500 ms, 1.25 * paired-control maximum)`.

The candidate qualifies for a full-graph confirmation only if all conditions
hold:

- `B1 < A1` and `B2 < A2` for evaluation, graph, and backend wall;
- aggregate evaluation speedup
  `(A1_eval + A2_eval) / (B1_eval + B2_eval)` is at least `1.04x`;
- each paired evaluation saving is at least `250 ms`;
- projected full-graph saving
  `mean(precompute saving) + 8 * mean(evaluation saving)` is at least
  `2,500 ms`;
- every correctness, topology, failure-drain, alias, cancellation, lifecycle,
  thermal-start, and responsiveness gate passes.

Mixed, variance-overlapped, thermally unattributable, or wall-boundary-
inconsistent evidence is inconclusive. A saving below `2,500 ms` is too small
for this scheduler complexity and stops without integration.

## Full-graph confirmation and integration gate

Only a passing evaluation slice may run one separately cooled full-M2250 A/B
confirmation. Untimed correctness must preserve all eight evaluation taps and
the final latent raw-U32 exactly; timed arms use only the ordinary final
readback. Require at least `1.04x` full-graph speedup, at least `2,500 ms`
absolute full-graph saving, exact topology, and all cancellation,
responsiveness, lifecycle, and resource gates before selecting this scheduling
policy for production DiT.

If DiT integrates, screen the identical policy separately on one production
C2314 VAE window in a new ABBA run. The existing one-window topology is `555`
decoder commands plus one readback. The candidate must retain `556` completion
fences, reduce true drains to `139` and idles to `138`, preserve the complete
waveform and guards raw-bit exactly, pass lifecycle/responsiveness gates, and
project at least `800 ms` saving across two windows. A VAE failure retains a
valid DiT-only selection. Do not generalize to the remaining stages,
`AceUniformPool`, conditioning, or planner without a new audit and record.

After any production integration, run a short generic-duration exact direct
request and require identical final latent, raw pre-normalization waveform,
seams, normalized WAV, metadata, cancellation, and cleanup. Reserve another
180-second product run for a later cumulative checkpoint; the present
mechanism cannot by itself establish the under-one-minute target.

This experiment authorizes no model-math, kernel, package, sampler, attention,
dense, VAE, planner, evaluation-count, native Metal, or output-quality change.
Exact identity satisfies its quality gate without a new listening round.

## Evaluation-slice evidence — passed 2026-08-21

The frozen actual-Chromium evaluation-0 screen passed on core commit
`ce14bece806ca66ca9567326686fb79ee1646228` and harness commit
`0dbbe27cb5b486b91fd40355a04a1106f4740ce0`. Every untimed correctness arm
and every timed ABBA arm produced the expected `288,000`-word raw-U32 SHA-256
`d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286`,
with zero output or repeat mismatches. The authenticated descriptor-table
SHA-256 was
`d480bde986cba12068e462093169ef1a6cf3ceb45987eabb82ef8c8fe07eca47`;
all arms retained `6,833` members and the exact `341`-command evaluation-0
prefix.

The timed authoritative walls were:

| arm | profile | evaluation | graph | backend |
| --- | --- | ---: | ---: | ---: |
| `A1` | depth one / epoch one | `7,428.9 ms` | `7,712.5 ms` | `7,770.2 ms` |
| `B1` | depth two / epoch four | `6,507.8 ms` | `6,761.2 ms` | `6,817.8 ms` |
| `B2` | depth two / epoch four | `6,745.4 ms` | `6,956.4 ms` | `7,012.4 ms` |
| `A2` | depth one / epoch one | `9,131.5 ms` | `9,676.4 ms` | `9,737.8 ms` |

Both paired candidate arms improved evaluation, graph, and backend wall.
Evaluation savings were `921.1 ms` forward and `2,386.1 ms` reverse;
aggregate evaluation speedup was `1.249539733802839x`. The frozen projection
was `13,410.850000143051 ms` full-graph saving, well above the `2,500 ms`
escalation threshold. These are evaluation-slice results and a projection,
not a measured full-graph or product claim.

Topology reconciled exactly. Each arm submitted `342` physical command
buffers and requested/settled `342` fences with zero fence rejection. Control
used `341` graph drains, `340` graph idles, and maximum outstanding/pending
counts `1/1`; candidate used `86` graph drains, `85` graph idles, and maximum
counts `2/2`. All pending descriptors were cleared after cleanup. The actual
DiT cancellation preflight observed abort from command 0's completion callback
with exactly one prefetched successor, performed no backfill or later public
progress, completed terminal settlement and destruction before rejection, and
finished abort-through-cleanup in `15.899999976158142 ms` with no unhandled
rejection.

All four accepted arms began after independent nominal thermal gates and
remained at level 0 through cleanup. The 50 ms heartbeat's largest gap was
`52.199999928474426 ms` and every p99 was at most
`52.10000002384186 ms`. One earlier `A1` gate submission was rejected before
timed GPU work; it is retained in the result and is not a sample.

The compact committed
[evaluation-slice result](../results/OPT-0080/evaluation-slice.json) binds the
complete ignored browser receipt and all per-arm raw thermal artifacts by
byte length and SHA-256 while omitting bulky observation, per-fence, and
per-epoch arrays. This pass activates only the separately cooled full-M2250
confirmation described above. Production selection, package bytes, model
math, VAE, planner, and public defaults remain unchanged.

## Full-graph confirmation evidence — passed 2026-08-21

The separately authorized full-M2250 confirmation passed on core and harness
commit `2a9be026c6f36363ca73d44fb3ede9c49b5aaacc`. It authenticated the same
revision-7 production tuple and descriptor table SHA-256
`d480bde986cba12068e462093169ef1a6cf3ceb45987eabb82ef8c8fe07eca47`.
The untimed depth-one and depth-two arms matched all eight accepted evaluation
hashes and the final latent. They compared `2,304,000` evaluation raw-U32 words
with zero mismatches, had zero final-latent mismatch, and matched the final
evaluation tap to the ordinary final readback exactly. Each untimed arm added
eight copies inside existing sampler command buffers but no command buffer or
queue drain. Both timed arms disabled those taps and retained only the ordinary
final readback.

Full topology reconciled exactly across nine phase-aligned segments
`25 + 8 * 316 = 2,553` graph commands:

| metric | depth-one control | depth-two candidate |
| --- | ---: | ---: |
| graph commands / fences | `2,553 / 2,553` | `2,553 / 2,553` |
| graph true drains / epochs | `2,553 / 2,553` | `639 / 639` |
| graph idle intervals | `2,552` | `638` |
| total commands / fences with readback | `2,554 / 2,554` | `2,554 / 2,554` |
| total true drains / requested idle | `2,554 / 2,553 ms` | `640 / 639 ms` |
| maximum outstanding / pending | `1 / 1` | `2 / 2` |

Every completion fence settled with zero rejection and every pending descriptor
was cleared before release. The separately cooled A/B timing used authoritative
walls only; overlapping per-fence intervals remained non-additive:

| arm | profile | eight-evaluation sum | graph | backend |
| --- | --- | ---: | ---: | ---: |
| `A` | depth one / epoch one | `60,529.3 ms` | `61,042.7 ms` | `61,102.2 ms` |
| `B` | depth two / epoch four | `53,362.8 ms` | `53,605.6 ms` | `53,633.1 ms` |

The candidate saved `7,437.100000023842 ms` of measured full graph wall and
reached `1.1387373707231534x`, passing the frozen `2,500 ms` and `1.04x`
gates. Backend wall also improved. The eight-evaluation wall sum saved
`7,166.5 ms` at `1.1342976755343874x`. Wall boundaries reconciled and the
disjoint epoch-wall sums did not exceed graph wall.

Both arms began after distinct independent nominal level-0 gates: `56,788 ms`
and `43,068 ms`, with `63` and `48` observations. Both later heated during the
full graph, reaching levels 1 and 2. The through-cleanup traces retain those
transitions and `52/71` non-nominal observations; the frozen protocol judged
the independently cooled nominal start and did not require an all-level-0
execution. Trace coverage continued `19,759/40,695 ms` past completed cleanup.
The 50 ms heartbeat passed absolute and relative gates: both maximum gaps were
`52.200000047683716 ms`, and both p99 gaps were
`52.10000002384186 ms`.

The actual full-graph candidate cancellation observed abort from command 0's
completion callback with exactly one submitted successor. It performed no
backfill, later public progress, or checkpoint; all submitted work, graph/model
destruction, and backend disposal settled before rejection in
`20.699999928474426 ms`, with no unhandled rejection. Every correctness,
topology, performance, cancellation, responsiveness, lifecycle, resource, and
frozen thermal-start gate therefore passed.

The compact committed
[full-graph result](../results/OPT-0080/full-graph.json) binds the complete
ignored browser receipt plus every raw and parsed per-arm thermal artifact by
byte length and SHA-256 while omitting bulky observations, heartbeat gaps, and
per-fence/per-epoch arrays. This pass authorized selecting the exact
depth-two/four-completion policy for production DiT. At that checkpoint the
integration had not yet occurred; the later integration and product validation
are recorded below.

## Production integration and product gate — passed 2026-08-21

The base production scheduler integration occurred in commit
`7d9ae463aa580d0290421b067467f799cf1b0b75`. Commit
`023bdecbf670b9309db37b6ac3030293ffe3b463` then corrected and finalized the
selector's audited scope while adding the internal product-evidence seam. The
resulting selector is fail-closed to the authenticated reference-subgroup,
OPT-0009 exact-dense, OPT-0070 attention, revision-7 OPT-0072 VAE,
C2378/overlap64, canonical eight-evaluation tuple. Diagnostic seams remain
depth one. The audited experiment scope is direct generation only:
planner-enabled requests deliberately remain depth one until a separate
planner-conditioned/DCW record passes.

The required generic product gate passed in actual in-app Chromium on harness
commit `dfb2a24c979f840f13909b6baee0742bd7ee4f40`. It used the canonical
96-second direct request (`365` bytes, SHA-256
`ecc1d8d0fd7a87e14d0cf827563280fe35853526368becf883d98f4d42cb1ad4`),
C2400/M1200, and the shortest integer production duration with a real C2378
stitch. The exact two-window plan stitched at latent frame `2,250` / audio
frame `4,320,000`; its radius-64 neighborhood contained `491,520` raw-U32
words (`1,966,080` bytes).

The fresh-device order was forced depth-one control, forced depth-two
candidate, ordinary no-override production, then ordinary seam-free
cancellation. Control/candidate and candidate/production both matched every
one of the `153,600` final-latent U32 words, all `9,216,000` raw waveform U32
words, all `491,520` seam U32 words, and every byte of the `36,864,044`-byte
normalized WAV. Both comparison pairs therefore had zero mismatch, zero
non-finite sample, and zero absolute/mean/RMS difference. All three successful
arms shared these identities:

- final latent SHA-256
  `527cdc7e560691f21383f3b06a4a85f7f41ba92e93e6357b7be75f115a5c9e07`;
- raw pre-normalization waveform SHA-256
  `c4152aa56bcf81236b60cb2dbea3976b4a7f4d800af001b9bfbbbd52dda6e82b`;
- normalized WAV SHA-256
  `c088385a6b4dabc30215d122b3a4da8406611f1a7d6d1255eba2846aa7e24e4a`.

The ordinary arm selected and executed `depth2-phase-epoch4`. Total cooperative
queue drains, including unchanged non-DiT product work, were `2,725` for the
forced depth-one control and `1,243` for both depth-two arms. These counts are
topology evidence only; the product gate was intentionally untimed and makes no
thermal or performance claim. Ordinary cancellation aborted synchronously at
the first public Denoising `0/8` event, preserved the exact reason, emitted no
later progress or output, and completed rejection plus disposal in
`4.200000047683716 ms`. Every worker released its committed output before
dispose and termination, only two large outputs coexisted, all four workers
terminated, and the largest 50 ms page-heartbeat gap was
`52.60000014305115 ms`.

Three non-retried failed attempts are retained rather than hidden: a missing
cache-resident revision-7 VAE payload, an over-strict comparison of
policy-dependent progress-event counts, and premature deletion of OPFS-backed
snapshots before page comparison. The authenticated seven-shard VAE cache stage
and the final explicit retain/compare/release handshake resolved those distinct
conditions. The compact
[product result](../results/OPT-0080/product-integration.json) binds every raw
attempt and the passing `51,285`-byte browser receipt by SHA-256.

OPT-0080 is positive and integrated for direct production DiT. No planner,
package, model-math, listening, thermal-product, or under-one-minute claim
follows. The separately required C2314 VAE policy screen passed as recorded
below; at this checkpoint it authorizes a narrowly scoped production
integration but is not itself evidence that the production selector was
changed or that the changed product path passed its required exact gate.

## C2314 VAE scheduler screen — passed 2026-08-21

The separately required actual-C2314 screen passed on core and harness commit
`f8fde273ad57d21c8410710518c63e501acaa1ec`. One persistent actual WebGPU
device, FP16 VAE backend, and package owner authenticated the revision-7
physical OPT-0066 package with manifest SHA-256
`36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7`,
then reused it across untimed correctness, cancellation, post-cancellation
reuse, and all four timed arms. The screen used qpc64 and the committed C2314
fixture SHA-256
`01ec291963276b4784ec0ae3f6b3d7ed80bffd657dfd3b14125729260918783d`.

Untimed depth-one control, depth-two candidate, and the post-cancellation
depth-one probe all reproduced the committed complete C2314 waveform SHA-256
`2a16f0fc4b07661e21628e0b5574c2feeab3882ecef169da52a671c937f36f0c`.
Each bounded pair comparison covered all `8,885,760` raw-U32 words with zero
mismatch; every value was finite and nonzero. Every timed arm also reproduced
that waveform with zero mismatch. Six physical activation/output allocations
carried both an active-C2314-end and a logical-C2378-allocation-end canary:
all `12` regions and `3,072` checked bytes retained SHA-256
`183dead51e555d79ec074ad8acfe08c5f4dffce8392ccadc46bb9da2d5aa413d`.
The separate guard initialization and diagnostic readback fences were excluded
from the scheduler topology and timing.

The one-window topology matched the frozen contract exactly:

| metric | depth-one control | depth-two candidate |
| --- | ---: | ---: |
| decoder commands + ordinary readback | `555 + 1` | `555 + 1` |
| submitted commands / settled fences | `556 / 556` | `556 / 556` |
| true drains / completion epochs | `556 / 556` | `139 / 139` |
| cooperative idle turns | `555` | `138` |
| maximum outstanding commands | `1` | `2` |

No completion fence rejected. The independently cooled ABBA timing retained
authoritative scheduling and complete decode-window walls rather than summing
overlapping per-fence intervals:

| arm | profile | scheduling wall | complete decode window |
| --- | --- | ---: | ---: |
| `A1` | depth one / epoch one | `16,510.60000014305 ms` | `16,525.5 ms` |
| `B1` | depth two / epoch four | `15,176.5 ms` | `15,195.600000143051 ms` |
| `B2` | depth two / epoch four | `15,195.5 ms` | `15,213.200000047684 ms` |
| `A2` | depth one / epoch one | `16,461 ms` | `16,476.5 ms` |

Both candidate directions improved scheduling and complete decode wall. The
complete-window savings were `1,329.8999998569489 ms` forward and
`1,263.2999999523163 ms` reverse, for aggregate speedup
`1.0852779458509707x`. Their frozen two-production-window projection was
`2,593.199999809265 ms`, comfortably above the required `800 ms` gate.

All four arms began after independent nominal level-0 thermal gates and
remained nominal through settlement. The maximum thermal poll gap was
`935 ms`. The 50 ms heartbeat passed both absolute and relative gates; its
largest gap was `52.200000047683716 ms` and largest p99 was
`52.10000014305115 ms`. Actual candidate cancellation aborted from the first
public progress callback, emitted no later progress, scheduling evidence, or
output, preserved the exact reason, drained before rejection, and completed in
`114.09999990463257 ms` with no unhandled rejection. A subsequent complete
depth-one decode on the same owner remained bit-exact. Cleanup destroyed all
`16/16` buffers exactly once, left zero live or mapped resources before device
destruction, and passed idempotent backend destruction.

The compact committed
[VAE scheduler result](../results/OPT-0080/vae-scheduler.json) binds the
`3,038,276`-byte ignored browser receipt (SHA-256
`6a3d97aa4b0360c268415884d5621a0a20d69bbed85548e19187a1d5e4a6ea64`)
plus every per-arm thermal artifact while omitting bulky command, epoch,
heartbeat, canary-span, and thermal-observation arrays.

This positive screen authorizes only a fail-closed production selection for
the exact authenticated direct qpc64 C2314 revision-7/physical-OPT-0066 VAE
window. It does not authorize depth two for a short remainder window,
planner-enabled generation, diagnostics, family tracing, another package or
geometry, or any model-math change. Production remained unchanged at this
evidence checkpoint, and the required post-integration exact product gate was
still pending; this screen is therefore integration authorization, not product
integration evidence.

## VAE production integration and product-exact gate — passed 2026-08-21

The fail-closed VAE selector and its post-integration browser gate were pushed
at core and harness commit
`8d443f6aff4f6a9b06df6db77b3f88b9401123a7`. The selector may choose
`depth2-phase-epoch4` only for the exact authenticated direct production tuple
authorized by the preceding screen, and the backend resolves that policy per
window: exact C2314 uses depth two while every other window remains depth one.
The product gate used the same canonical 96-second direct request recorded
above (SHA-256
`ecc1d8d0fd7a87e14d0cf827563280fe35853526368becf883d98f4d42cb1ad4`),
whose C2400 plan contains a C2314 first window and a C214 remainder.

The fixed, non-retried order was forced depth-one control, forced
C2314-only candidate, seam-free ordinary production, then seam-free ordinary
cancellation. The two forced retained-output arms were exactly identical over
all `153,600` final-latent U32 words, all `9,216,000` raw pre-normalization
waveform U32 words, the `491,520`-word radius-64 stitch neighborhood, and every
byte of the `36,864,044`-byte normalized WAV. Their identities were:

- final latent SHA-256
  `527cdc7e560691f21383f3b06a4a85f7f41ba92e93e6357b7be75f115a5c9e07`;
- raw waveform SHA-256
  `c4152aa56bcf81236b60cb2dbea3976b4a7f4d800af001b9bfbbbd52dda6e82b`;
- stitch neighborhood SHA-256
  `17540647f319a947860bf7402721e3e942f089cfc90a8d8aa20d8191ff889830`;
- normalized WAV SHA-256
  `c088385a6b4dabc30215d122b3a4da8406611f1a7d6d1255eba2846aa7e24e4a`.

The ordinary arm omitted the experiment seam and retained neither final latent
nor raw waveform. Its always-collected timing-free scheduling receipt selected
`opt-0080-c2314-depth2-phase-epoch4`, reported depth two only for C2314 and
depth one for C214, and its public normalized WAV matched the forced candidate
byte for byte. Stable request, metadata, runtime, chunk plan, diagnostic-code,
and progress-stage summaries also matched. This combination proves the
ordinary selector path without misrepresenting unretained ordinary latent or
raw data as directly compared evidence.

The forced-arm evidence preserved both windows' physical command topology.
C2314 kept `555` decoder command buffers plus one readback and all `556`
requested/settled completion fences; depth two changed only
drains/epochs/idles from `556/556/555` to `139/139/138`, with maximum
outstanding commands changing from one to two. The C214 remainder stayed depth
one with `53 + 1` commands, `54` requested/settled fences, `54` drains/epochs,
`53` idles, and maximum outstanding one. The seam-free ordinary receipt
independently reported the same submitted command counts and candidate drain
topology, without claiming forced per-fence evidence for that arm. VAE drain
totals were therefore `610` for control and `193` for both candidate and
ordinary production; complete product drain counts were `1,243` and `826/826`.
These counts are topology evidence, not a timing comparison.

All four fresh workers terminated, every successful arm released its result
before disposal, at most two large retained payloads coexisted, the large
payloads were released before cancellation, and no retry occurred. Ordinary
production cancellation used no OPT-0080 product seam, triggered once,
preserved the exact abort reason, published no later progress, output,
evidence, fatal diagnostic, or unhandled rejection, and left the worker ready
to terminate. Every 50 ms page heartbeat passed the 500 ms responsiveness
bound; the largest observed gap was `52.90000009536743 ms`.

The compact committed
[VAE product-selector result](../results/OPT-0080/vae-product-selector.json)
binds the ignored `61,752`-byte raw receipt at
`optimization/artifacts/OPT-0080/vae-product-selector/raw-receipt.json`
(SHA-256
`bc741387b20be1715eaf80af1661392add8b32318258acce5ff9a8e18cb09225`).
The compact result deliberately omits the receipt's product wall timings.

This gate made no product timing comparison or timing claim, captured no
thermal trace and makes no thermal claim, and ran with the planner disabled;
the planner was never executed and no planner claim follows. The receipt notes
that selector-negative cases remain covered by runtime unit tests, but this
direct browser run is not planner product evidence. It makes no new package,
model-math, listening, quality-envelope, or under-one-minute claim. Within
that literal scope, OPT-0080 VAE scheduling is integrated/product-exact.

## Authority

- Accepted product baseline and drain inventory:
  [OPT-0073](OPT-0073-revision7-webcrypto-production-selection.md)
- Exact evaluation fixture and 341-command topology:
  [OPT-0067](OPT-0067-dit-quad-query-evaluation-slice-thermal-screen.md)
- Current full-graph scheduling attribution:
  [OPT-0062](OPT-0062-dit-quad-query-attention-integration.md)
- Production-shaped VAE scheduling evidence:
  [OPT-0059](OPT-0059-vae-c2378-short-projection-gate.md)
- Prior distinct command-buffer coalescing experiment:
  [OPT-0034](OPT-0034-dit-command-buffer-coalescing.md)

No candidate code, test, browser execution, GPU work, package change, public
profile, or production selection had occurred when this experiment was
allocated.
