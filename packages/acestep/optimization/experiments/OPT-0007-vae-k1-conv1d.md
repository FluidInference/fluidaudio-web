# OPT-0007 — Exact pointwise VAE Conv1D

## Status

- Evidence: `negative`
- Disposition: `abandoned` (fixed geometry not integrated)
- Date: 2026-08-13
- Author/agent: Codex
- Risk class: `exact`; target-browser U32 identity with the accepted scalar
  FP32 Conv1D kernel is the authority

## Hypothesis

After `OPT-0004` and `OPT-0005`, every K7 Conv1D operation has an optimized
production path, but all 15 K1 residual projections remain on the scalar
kernel. K1 is a pointwise matrix projection over frames: a workgroup can reuse
an input-channel slice across multiple output channels and a weight slice
across multiple frames while leaving each invocation solely responsible for
one output and visiting input channels in the original increasing order.

This is a composable optimization hypothesis, not a predicted minimum speedup
or an automatic acceptance threshold. Geometry and integration policy will be
chosen from measured evidence rather than fixed by this allocation record.

## Identity

- Allocation baseline: pushed `main` commit
  `f339266fb91b2e5519fa559b3ca4126361b436ad`
- Model manifest SHA-256:
  `d133b21d55bb6c00ad132aeaa83549ccec1a06c581c9b259268670dcf694fb55`
- Reference fixture manifest SHA-256:
  `cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`
- Execution profile: `reference-bf16-subgroups`; VAE tensors and arithmetic
  remain FP32
- Scheduling baseline: integrated `OPT-0002` work-aware quanta and
  `OPT-0006` bounded batch size eight
- Endpoint/correctness harness: frozen pushed commit
  `3c7b351b48d72e0397ba2d8260f0b52bb98f9442`
- Production-sequence harness: frozen pushed commit
  `1fa164b91a03cf6838fd94812e21949a8e621664`
- Machine: MacBook Air `Mac15,12`, Apple M3, 10 GPU cores, 16 GB unified
  memory; Chrome `151.0.7922.138`, macOS `26.5.2` build `25F84`, WebGPU
  adapter vendor `apple`, architecture `metal-3`

## Exact production scope

The canonical 256-frame window contains 15 K1 Conv1D operations, 414 logical
quanta, 361,758,720 outputs, and 68,451,041,280 valid multiply-accumulates.
This is 10.98% of decoder convolution work. It is the only Conv1D family that
still always selects the portable scalar kernel.

The exact 180-second chunk plan contains 540 K1 operation instances, 14,454
logical quanta, 12,627,640,320 outputs, and 2,389,369,159,680 valid
multiply-accumulates. These counts describe leverage, not an end-to-end speed
projection.

Representative authenticated shapes are:

| Frames | Input/output channels | Quanta per operation | Valid MACs per operation |
| ---: | ---: | ---: | ---: |
| 2,560 | 1,024 | 12 | 2,684,354,560 |
| 15,360 | 512 | 18 | 4,026,531,840 |
| 61,440 | 256 | 18 | 4,026,531,840 |
| 245,760 | 128 | 30 | 4,026,531,840 |
| 491,520 | 128 | 60 | 8,053,063,680 |

## Candidate boundary

- Keep native `[output_channel, kernel, input_channel]` weights and current
  work-aware complete-row ranges. No package conversion or duplicate layout.
- Tile multiple frames and output channels per workgroup, cooperatively stage
  bounded input-channel and weight slices, and let each active invocation own
  exactly one output.
- Initialize from bias or positive zero once, then visit input-channel chunks
  and their real channels in globally increasing order with the same scalar
  expression. Do not perform a cross-lane reduction, change precision, include
  tail placeholders, or move a barrier behind a nonuniform predicate.
- Select the candidate only for safe K1/stride-one/dilation-one complete-row,
  batch-contained ranges within device and U32 limits. Everything else remains
  on the scalar kernel.
- Treat tile and chunk geometry as an empirical choice. Test another geometry
  only when the first screen leaves a decision-relevant attribution question.

## Correctness gate

- Require zero full-range U32 mismatches against the shipped scalar GPU kernel
  on pinned Chrome/M3. Independent CPU indexing and arithmetic sentinels are
  diagnostic support, not a substitute for the GPU authority.
- Exercise bias, multiple batches, frame/output/range tails, input-channel
  chunk boundaries and tails, signed zero, non-finite output prefill,
  contraction-sensitive operands, and cancellation-sensitive mixed
  magnitudes.
- Prove exact-once writes, native weight indexing, globally increasing
  input-channel source order, uniform barriers, dynamic range-control offsets,
  binding/alias/device-limit checks, compilation retry, destruction, and abort
  cleanup before timing.
- Listening is unnecessary while primitive and integrated outputs remain
  bit-identical. Any numerical change requires a new risk declaration and
  quality gate.

## Sparse benchmark protocol

1. Start with static planner/source tests and a complete manageable actual-GPU
   correctness graph. Do not run a VAE window or song.
2. Screen one complete middle production range at the 1,024-channel endpoint
   and one at the 128-channel endpoint against the shipped scalar kernel.
   Use independent qNaN-prefilled outputs, full-range U32 comparison, symmetric
   warmup, balanced paired samples, heartbeat, cancellation, and continuous
   external thermal polling.
3. If either endpoint establishes a useful direction, escalate that useful
   endpoint or shape family with one complete K1 operation or a bounded
   sequence through the production batch-eight topology. Compare another
   geometry only when needed to define safe selector coverage. Choose the
   cheapest boundary that can settle integration; do not run every endpoint
   automatically.
4. Retain raw wall and submit-through-drain samples, range/dispatch/command
   counts, logical throughput, resource bytes, per-batch maxima, responsiveness,
   cancellation, and thermal validity. Compile, allocate, and upload outside
   the timed interval.
5. Integrate only after an exact real-decoder boundary confirms the selector,
   dynamic controls, FIFO dependencies, progress, and batch-eight scheduling.
   No listening run is required for bit-identical output.

Judge the experiment by exactness, a credible positive measured delta,
resource cost, and integration value toward the single warm 180-second
under-60-second product target. There is no per-experiment throughput floor.

## Main risks

- K1 has no temporal halo reuse; weight/input staging and barriers may cost
  more than they save on one channel endpoint.
- The best low-channel and high-channel geometry may differ. Prefer a small
  fail-closed selector over a broad sweep or a forced universal kernel.
- WGSL permits contraction and reassociation. Source order is constrained, but
  actual target-browser scalar-versus-candidate U32 identity remains the
  numerical authority.
- K1 is meaningful but not dominant by itself. A positive local result should
  compose with the existing K7 and scheduling work; it is not an end-to-end
  claim.

## Evidence and disposition

- Complete manageable actual-GPU correctness passed across four cases and
  2,374 outputs with zero scalar-versus-candidate U32 mismatches. The cases
  covered bias/no-bias, multiple batches, input channels 63/64/65, frame and
  output-channel tails, range boundaries, qNaN output prefill, and the declared
  contraction/cancellation arithmetic discriminants.
- The complete middle C1024 production range was non-positive and noisy:
  scalar median wall was 19.55000001192093 ms, candidate median wall was
  22.149999976158142 ms, the ratio of medians was 0.8826185116462391x, and
  the candidate won two of four paired rounds. All 229,376 compared outputs
  remained bit-identical.
- The complete middle C128 production range was encouraging but noisy:
  scalar median wall was 28.750000029802322 ms, candidate median wall was
  13.5 ms, the ratio of medians was 2.129629631837209x, and the candidate won
  three of four paired rounds. This justified one bounded production-shaped
  escalation; it did not establish selector coverage by itself.
- The deciding escalation ran 16 consecutive C128 production ranges through
  the production `OPT-0006` batch-eight topology: 16,777,216 outputs,
  2,147,483,648 valid multiply-adds, 16 passes in two command buffers, two
  drains, two real cooperative idles, and at most one outstanding command
  buffer. The full contiguous output union remained bit-identical and the
  post-timing guards remained qNaN, but the candidate lost all four balanced
  pairs. Scalar median wall was 36.60000002384186 ms versus candidate
  44.099999994039536 ms, a 0.8299319734419193x ratio of medians.
- Cancellation passed at every measured boundary. The sequence cancellation
  drained its first eight-pass batch and real idle, then prevented the later
  batch from being encoded or submitted. All three timed pages passed their
  continuous nominal thermal pre-gates and their external thermal logs span
  the measured result.
- Evidence conclusion and rationale: `negative` for this fixed workgroup-16x8,
  frame-tile-16, output-tile-8, input-chunk-64 geometry. The production-shaped
  aggregate is the most representative bounded evidence and does not support
  integrating it, despite the independently useful one-range C128 signal.
- Code disposition and rationale: `abandoned`; keep production K1 operations
  on the portable scalar kernel. The benchmark implementation and all positive,
  negative, and exactness evidence remain recorded, but no production source or
  package layout changed.
- Result JSON: [`optimization/results/OPT-0007/result.json`](../results/OPT-0007/result.json)
- Interactions: holds `OPT-0002`, `OPT-0004`, `OPT-0005`, and `OPT-0006`
  fixed; changes only K1 Conv1D kernel selection
- Revisit when: a materially different K1 geometry or mechanism addresses the
  measured production-sequence cost; target-browser/GPU behavior changes enough
  to invalidate these timings; or later profiling again makes K1 the best
  decision-relevant bottleneck. Do not repeat the unchanged fixed geometry.
