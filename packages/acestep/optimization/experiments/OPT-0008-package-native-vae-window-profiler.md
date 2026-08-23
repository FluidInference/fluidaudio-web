# OPT-0008 — Package-native production VAE window profiler

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Date: 2026-08-13
- Author/agent: Codex
- Risk class: `exact`; this experiment adds measurement only and must leave the
  shipped decoder's output bits, graph topology, kernel selection, and
  cooperative scheduling unchanged

## Hypothesis

One authenticated 256-latent-frame window through the shipped package-native
VAE can attribute the current post-integration decode wall time to individual
operations, operation families, selected kernels, and mixed-operation physical
batch boundaries. That evidence can choose whether ConvTranspose or a different
kernel/family is the next useful experiment without another full song or a
speculative optimization threshold.

This is an attribution hypothesis, not a claim that any named primitive is
already the bottleneck. The profiler does not propose or select a new kernel.

## Identity

- Allocation baseline: pushed `main` commit
  `9dbd6e9cb85da211aa9e8224edfc08a2eef3f706`
- Package identity: converter revision 4 reference package, manifest SHA-256
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`
- Package-identity implementation commit:
  `68d7795c616c1520b1d97ddef9f9d3147ab3973e`
- Execution profile: `reference-bf16-subgroups`; VAE weights and arithmetic
  remain the shipped FP32 path
- Scheduling profile: `cooperative`, including the integrated `OPT-0002`
  work-aware ranges and unchanged `OPT-0006` physical batch size eight
- Target machine: MacBook Air `Mac15,12`, Apple M3, 10 GPU cores, 16 GB unified
  memory. The result must record the exact browser, macOS, adapter features and
  limits, harness commit, deterministic-latent identity, and raw artifact hash
  actually used.

## Fixed production boundary

- Authenticate the revision-4 manifest and every consumed VAE shard, then load
  only the VAE heavyweight phase. Do not load the text encoder, condition
  encoder, DiT, or planner.
- Generate one deterministic, nondegenerate 256-frame latent using a committed
  seed/recipe. This is the shipped 10.24-second decoder window, not a synthetic
  collection of isolated primitive shapes.
- Execute the ordinary production decoder graph, current kernel selectors,
  work-aware logical ranges, FIFO ownership, one-outstanding-command-buffer
  policy, real queue-empty interval, and physical batch size eight without
  modification.
- Compile pipelines, authenticate and upload weights, allocate buffers, and run
  the symmetric warmup outside the timed interval. Record those costs
  separately; do not hide them inside an attributed operation bucket.
- The authoritative timed interval covers the complete production decode
  window from the first physical-batch encoding boundary through final GPU
  completion and the production readback boundary. Harness-only hashing and
  report serialization occur after timing.

No model math, package bytes, resource lifetime, logical range, pass boundary,
physical batch boundary, submission, drain, or idle request may change in the
primary measurement.

## Attribution contract

Instrument the existing physical batches in place. Accumulate compact records
inside the worker and publish them after the run rather than adding per-batch
cross-thread messages. Each physical batch records:

- physical-batch index and exact first/last logical quantum;
- operation instance, operation family, and production-selected kernel for
  every enclosed pass;
- pure-versus-mixed classification and the ordered membership of a mixed
  operation-boundary batch;
- logical quantum count, dispatch/pass count, command-buffer count, submission
  count, queue-drain count, and requested cooperative idle milliseconds; and
- encode wall, submit-through-drain wall, actual cooperative-idle wall, and
  total fenced batch wall, using the same monotonic clock as production stage
  timing.

Aggregate pure batches directly by operation instance, family, and selected
kernel. Keep mixed-operation boundary batches in an explicit mixed bucket with
their exact ordered membership; do not fractionally assign their wall time to
members using FLOPs, dispatches, or another model. Reconcile the sum of every
physical batch and unattributed fixed boundary with the full-window wall time,
and reconcile every count with the shipped graph plan.

If, and only if, the mixed bucket is large enough to make the next-kernel
decision ambiguous, a second diagnostic may flush at operation boundaries to
separate those members. That diagnostic must be labeled topology-changing,
must report its altered command-buffer/drain/idle counts, and is not the
production timing baseline. There is no fixed percentage threshold: run it
only when the first trace cannot support a choice.

## Correctness, responsiveness, and cleanup gates

- Preserve the existing full-output validation and record the complete output
  byte length and SHA-256. Require exact U32 identity between the deterministic
  untimed control/warmup output and the timed instrumented output, plus the
  existing complete-write and finite-output checks.
- Record the exact logical-quantum, dispatch, command-buffer, submission,
  drain, requested-idle, and completed-idle totals. Per-batch totals must sum to
  the full-window totals with no missing or duplicate work.
- Retain the production progress stream and a low-overhead event-loop
  heartbeat through warmup, thermal polling, the timed window, readback, and
  cleanup. Report raw maximum gaps and progress counts; do not invent a
  speculative responsiveness threshold.
- Run a separate untimed cancellation probe through the same package-native
  decoder. Request cancellation immediately after the first physical batch has
  drained, require that no later batch is encoded or submitted, and verify FIFO
  drain-before-release and resource destruction.
- Capture validation errors, uncaptured errors, device loss, resource counts,
  and post-destroy state. Any device event, output mismatch, count mismatch,
  submission after cancellation, or cleanup failure invalidates the run.
- Listening is not required because the experiment cannot change arithmetic or
  production output. Any output difference is a failed exactness gate, not a
  candidate for a listening waiver.

## Sparse measurement protocol

1. Add focused static contracts that prove tags derive from the shipped graph
   operations and selected kernels and that instrumentation cannot change
   physical batch construction. Audit exact count reconciliation before using
   Chrome.
2. Authenticate and load only the revision-4 VAE phase, compile all selected
   production pipelines, allocate the normal bounded buffers, and execute one
   untimed deterministic warmup. Continue thermal polling throughout.
3. Begin the timed sample only after at least 30 continuous seconds of nominal
   thermal state, polled every 1,000 ms. A warmup-induced non-nominal reading
   resets this pre-gate.
4. Run one complete 256-frame production window first. Retain the full raw
   trace, thermal log, identity, output hash, count reconciliation, heartbeat,
   memory accounting, and separated load/compile/warmup costs.
5. Repeat a thermally valid full-window sample only if thermal/heartbeat noise,
   batch-level variance, or mixed-boundary attribution makes the operation or
   family ranking ambiguous enough to change the next experiment. Do not
   collect repetitions by default.
6. Run the operation-boundary-flushed diagnostic only under the mixed-bucket
   condition above. Run the one-batch cancellation probe outside all timed
   samples. No song or end-to-end generation is part of OPT-0008.

Current product-path evidence suggests roughly 12.5 seconds for the timed
window. That number is only a directional run-cost estimate, not an acceptance
gate, baseline result, or predicted optimization win. The intended first
measurement budget is one warmup, the 30-second nominal pre-gate, and one timed
window.

## Decision output and risks

The primary output is a ranked, exact attribution of pure-batch wall time by
operation, family, and selected kernel, accompanied by the explicit mixed
bucket and all scheduling/count evidence. The next optimization should follow
the largest credible, addressable production cost while considering kernel
risk and composability. ConvTranspose is one possible result, not a preselected
answer. There is no per-family share or speedup threshold.

Main risks and controls:

- Instrumentation overhead could distort short batches. Keep collection local,
  aggregate after the window, and reconcile profiler wall with the ordinary
  production stage wall.
- A physical batch may cross an operation boundary. Preserve it in the primary
  run, report it explicitly, and use the optional flushed diagnostic only when
  it blocks a decision.
- Compilation, package I/O, and warmup can dominate or heat a short run. Exclude
  them from the attributed interval, report them separately, and require the
  final continuous nominal thermal pre-gate.
- One window is directional evidence. Repeat only when the first trace is
  unstable or ambiguous; never turn the planning estimate into a threshold or
  publish the fastest observation alone.

## Evidence and disposition

- Evidence conclusion and rationale: `positive`. The authenticated shipped
  production window completed in 11,427.100000023842 ms with unchanged
  batch-eight topology, exact warmup/timed output identity, complete count and
  lifetime reconciliation, nominal thermal coverage, and a decision-stable
  attribution. Pure channel-chunked K7 Conv1D alone accounted for 6,028 ms of
  submit-through-drain time, well ahead of portable ConvTranspose at
  1,924.1999998092651 ms and portable K1 Conv1D at
  1,021.3000000119209 ms.
- Code disposition and rationale: `benchmark-only`; OPT-0008 completed the
  measurement and selected a next mechanism, but changed no production kernel,
  graph, package, arithmetic, or scheduling behavior.
- Result JSON: [canonical schema-v2 result](../results/OPT-0008/result.json)
- Interactions: holds integrated `OPT-0002`, `OPT-0004`, `OPT-0005`, and
  `OPT-0006` fixed, uses the package/runtime state closed by `OPT-0003`, and
  does not revisit the abandoned `OPT-0007` geometry.
- Closed on 2026-08-13 with harness commit
  `511a6696e5229894cf4db70554e6bfb4a6d11486`, production commit
  `9dbd6e9cb85da211aa9e8224edfc08a2eef3f706`, package-manifest SHA-256
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`,
  raw-result SHA-256
  `2696530395f43b5440c1131bf2c231881d34b068a41966419873960212bf9b8e`,
  and continuous-thermal-log SHA-256
  `308c6a94b954fb51252d4e4afeb163a9573873adfd1bda5a3f20c973a8a9be79`.

### Completed measurement

The one timed sample decoded the deterministic 256-frame latent into 983,040
FP32 samples (3,932,160 bytes). Both warmup and timed output were fully finite,
fully nonzero, contained zero qNaN-prefill sentinel words, covered every output
word, and hashed to
`30a08c1ec1209ecaa73284e6af98775786b8ad4bd5440bbde32c6c8d6ab482e4`.
The full 983,040-word U32 comparison had zero mismatches. Warmup decode wall was
11,560.100000023842 ms and timed wall was 11,427.100000023842 ms: a 133 ms,
1.1505090786388152% difference relative to warmup, which did not disturb the
ranking.

The timed wall separates into 17.699999928474426 ms decoder encoding,
10,766.599999904633 ms submit-through-drain, 639.1000002622604 ms fenced idle,
1.2999999523162842 ms decoder residual, and 2.399999976158142 ms fixed final
readback, with zero reconciliation delta. “Fenced idle” is specifically the
drain-end-to-next-encode interval and includes the progress callback and
scheduler overhead surrounding the real queue-empty timer; it is not labeled
as pure timer sleep or GPU execution. Package acquisition
(1,546.699999988079 ms), VAE upload (1,452.300000011921 ms), backend compilation
(532.3000000119209 ms), warmup, both qNaN prefills, hashing, cancellation, and
cleanup were outside the attributed timed interval.

Pure-batch submit-through-drain ranking by family was:

| Rank | Family | Pure batches | Quanta / dispatches | Submit-through-drain (ms) | Fenced idle (ms) | Fenced wall (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | K7 Conv1D | 242 | 1,936 / 1,936 | 6,558.89999973774 | 317.1000003814697 | 6,886.1000000834465 |
| 2 | ConvTranspose1D | 35 | 280 / 320 | 1,924.1999998092651 | 45.50000011920929 | 1,971.3000000715256 |
| 3 | K1 Conv1D | 42 | 336 / 336 | 1,021.3000000119209 | 53.900000154972076 | 1,077.1000000834465 |
| 4 | Snake | 71 | 568 / 568 | 121.60000014305115 | 89.59999996423721 | 212.40000009536743 |
| 5 | Add | 30 | 240 / 240 | 55.80000001192093 | 38.39999997615814 | 95 |

Selected-kernel pure submit-through-drain ranking was
channel-chunked Conv1D (6,028 ms), portable ConvTranspose1D
(1,924.1999998092651 ms), portable Conv1D (1,021.3000000119209 ms), tiled
Conv1D (530.8999997377396 ms), portable Snake (121.60000014305115 ms), then
portable Add (55.80000001192093 ms). The largest individual pure operations
were `block-4-res-2-conv1` (872.1000000834465 ms),
`block-4-res-3-conv1` (866.4999998211861 ms), and
`block-1-conv-t1` (643.3000000119209 ms).

The explicit mixed bucket retained 73 production boundary batches, 582 quanta,
588 dispatches, 1,084.8000001907349 ms submit-through-drain, and
1,182.7999997138977 ms fenced wall. Its largest classes were K7+Snake
(383.60000002384186 ms), ConvTranspose+Snake (241.0999999642372 ms), Add+K1
(137.79999995231628 ms), and K1+Snake (134.80000013113022 ms), measured by
submit-through-drain time. No operation-boundary-flush diagnostic was run:
the 6,028 ms pure channel-chunked K7 result already dominated, and even assigning
the entire mixed bucket elsewhere could not make that next-mechanism decision
ambiguous. This follows the record's conditional diagnostic rule and avoids a
topology-changing second run.

The unchanged production plan reconciled exactly: 88 operations, 3,942 logical
quanta/passes, 3,988 primitive dispatches, 493 decoder batch-eight command
buffers plus one final readback, 494 submissions and drains, 493 requested and
completed decoder idles, and 3,943 progress events. All 420 pure and 73 mixed
batches were accounted for. The separate post-drain cancellation probe
completed exactly one eight-quantum batch, one submission/drain/idle, then
rejected with `AbortError`; it prevented every later encode and submission.
Cleanup destroyed all 15 tracked buffers exactly once, left zero live buffers,
made a second destroy harmless, and made post-destroy decode reject with
`InvalidStateError`; no validation, uncaptured-error, or device-loss event was
reported. Worker heartbeat covered preparation through cleanup with 12,721
ticks and a 405.39999997615814 ms maximum timer gap; main-thread heartbeat
recorded 22,038 animation frames, 12,991 timer ticks, a
12.400000000008731 ms maximum frame gap, and an 11.699999988079071 ms maximum
timer gap.

The browser receipt's 58-observation nominal pre-gate was intentionally marked
pre-gate-only. The separately retained continuous logger is the reportable
thermal authority: 89 observations from
`2026-08-14T02:30:07.902775Z` through
`2026-08-14T02:31:35.907943Z`, zero non-nominal readings,
88,004.23787499312 ms monotonic coverage, and a
1,006.4439999987371 ms maximum poll gap. It starts at the required pre-gate
boundary, spans the timed interval from epoch 1786674677005.1 through
1786674688432.2002, and continues through cleanup completion at epoch
1786674688552.

The next tactical direction is an exact fixed-32-subgroup implicit-im2col FP32
K7 Conv1D experiment against the shipped portable counterpart. ConvTranspose
remains the credible later target because its pure cost is second, but Snake
parameter preprocessing is not decision-relevant at only 121.60000014305115 ms
of pure submit-through-drain time. This closure does not allocate another
experiment ID.

That was the measurement-time mechanism choice. The later owner-approved
FP16-first mixed-precision strategy changes the precision foundation of the
successor experiment without invalidating this profile's conclusion that K7
Conv1D is the dominant VAE family.
