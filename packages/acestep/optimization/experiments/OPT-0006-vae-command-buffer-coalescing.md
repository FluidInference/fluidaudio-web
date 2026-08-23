# OPT-0006 — Bounded VAE command-buffer coalescing

## Status

- Evidence: `positive`
- Disposition: `integrated`
- Date: 2026-08-13
- Author/agent: Codex
- Risk class: `exact`; dispatches, compute-pass boundaries, dynamic ranges,
  FIFO order, and kernel arithmetic remain unchanged

## Hypothesis

Integrated OPT-0002 bounds one decoder quantum safely, but production still
records, submits, drains, and idles after every quantum. The canonical
256-frame window has 3,942 decoder quanta. A 180-second generation has 36 VAE
windows and exactly 137,665 decoder command buffers plus 36 readbacks under
the current chunk geometry. The required one-millisecond queue-empty interval
therefore requests at least 137.665 seconds before timer overshoot or any GPU
work. That structural floor alone makes the repository's warm 180-second
under-60-second target impossible.

Several consecutive immutable quantum passes can instead be recorded into one
bounded command buffer without changing any dispatch or mathematical order.
The scheduler can retain one outstanding command buffer, drain it fully, then
perform the real queue-empty interval and check cancellation before recording
the next bounded batch.

This is a measured scheduling hypothesis, not a fixed batch-size prescription.
The batch size is chosen from actual responsiveness and wall evidence.

## Identity

- Baseline commit: `86933d573b4967ecfcfb9c7521e65dd40a2bd03d`
- Allocation commit: `1ef7250f133ba4bc21ca71be018969a64f0c1e4d`
- Benchmark harness commit: `4813e3d6b5a548154894ca334ce0f95c0cc16360`
- Production integration commit: `66f69709f7e30a9309e2315de0ea2ad7bae71604`
- Production baseline: `86933d5` (integrated OPT-0005 closure)
- Model manifest SHA-256:
  `d133b21d55bb6c00ad132aeaa83549ccec1a06c581c9b259268670dcf694fb55`
- Reference fixture manifest SHA-256:
  `cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`
- Execution profile: `reference-bf16-subgroups`; VAE FP32; integrated
  OPT-0004/OPT-0005 selector
- Target: MacBook Air `Mac15,12`, Apple M3, 10 GPU cores, 16 GB unified memory,
  pinned local Chrome/WebGPU adapter identity

## Exact static scope

Canonical 256-frame window after OPT-0002:

| Family | Operations | Quanta | Physical dispatches | Outputs | Valid MACs |
| --- | ---: | ---: | ---: | ---: | ---: |
| K7 Conv1D (OPT-0004/5 eligible) | 17 | 2,045 | 2,045 | 363,266,048 | 480,051,590,144 |
| K1 Conv1D | 15 | 414 | 414 | 361,758,720 | 68,451,041,280 |
| ConvTranspose | 5 | 322 | 368 | 120,586,240 | 75,137,122,304 |
| Snake | 36 | 813 | 813 | 844,627,968 | 0 |
| Add | 15 | 348 | 348 | 361,758,720 | 0 |

Total: 3,942 decoder quanta, 3,988 physical dispatches, and one readback.
Coalescing changes none of these logical counts; it changes only physical
command-buffer, drain, and non-final idle counts.

For the exact 180-second chunk plan, static command/idle leverage is:

| Maximum quanta per command buffer | Decoder command buffers | Requested idle |
| ---: | ---: | ---: |
| 1 | 137,665 | 137.665 s |
| 2 | 68,834 | 68.834 s |
| 4 | 34,434 | 34.434 s |
| 8 | 17,217 | 17.217 s |
| 16 | 8,626 | 8.626 s |

These are structural counts, not wall-time predictions. Readback command
buffers and 35 inter-window intervals remain separately accounted.

## Change

- Add a scheduler/backend seam that takes consecutive immutable decoder
  quanta in FIFO order and encodes each quantum into its own compute pass in a
  shared command encoder.
- Submit at most the selected bounded number of quanta per command buffer.
- Keep one command buffer outstanding; await full queue drain; perform the real
  queue-empty timer interval when more decoder work remains; check abort before
  recording/submitting the next batch.
- Preserve per-quantum progress semantics after the containing command buffer
  drains. Never report an individual quantum complete before its batch drains.
- Keep readback separate and preserve the existing inter-window boundary.
- Use the measured batch size of eight in the production VAE backend. The
  internal prepared-resource seam retains explicit batch-size control for the
  batch-one authority and exact integrated boundary tests.

## Correctness gate

- Complete actual-GPU output U32 identity for batch size one versus every
  candidate size on a manageable mixed decoder graph.
- Exercise integrated OPT-0004, OPT-0005, portable K1, ConvTranspose including
  a physical split, Snake, add, dynamic range controls, and operation/range
  transitions.
- Prove identical quantum/pass/dispatch order and counts, exact progress
  completion counts, no duplicate/missing pass, one command buffer outstanding,
  and queue drain before progress/idle/reuse.
- Directly abort after the first drained batch; require no later submission and
  record the actual worst batch drain plus delivery latency.
- Record animation-frame and timer gaps throughout. A size remains a candidate
  only while responsiveness is bounded enough for the product; there is no
  predeclared performance veto.
- Listening is unnecessary because no math, tensors, or output bits change.

## Sparse benchmark protocol

1. Static planner/backend tests for sizes 1, 2, 4, 8, and 16, including final
   partial batches and errors during encode/submit/drain/progress.
2. One complete manageable mixed decoder actual-GPU A/B across those sizes,
   with qNaN-prefilled independent outputs and full-domain bit comparison.
3. One thermally controlled mixed real-production-quantum sequence, balanced
   against size one. Retain raw encode, submit-through-drain, idle, wall,
   heartbeat, cancellation, and command-buffer counts.
4. Choose the largest useful bounded size from the measured Pareto frontier;
   do not automatically sweep further.
5. After integration, run one actual-Chrome exact mixed-decoder boundary. Run a
   full 256-frame VAE window only if smaller evidence cannot determine the next
   production decision. Do not run a song for this experiment.

## Results

Actual Chrome/M3 evidence was positive.

The complete mixed-decoder correctness probe ran the C136-to-C128 one-block
fixture at maximum batch sizes 1, 2, 4, 8, and 16. Every size completed the
same 109 logical quanta and 115 physical dispatches, retained exact
per-quantum progress, and produced the same 12 final FP32 U32 values. The
fixture exercised the integrated tiled and channel-chunked K7 paths, portable
K1 Conv1D, a physically split ConvTranspose, Snake, add, dynamic range
controls, and operation/range transitions. Its direct batch-four cancellation
probe submitted and drained one command buffer, reported four completed
quanta, and prevented later submission.

The decision-changing timing screen used the shipped production
`AceChannelChunkedVaeConv1dKernel` over 16 distinct consecutive C1024/d1
production ranges, indices 32 through 47. Each range contained 32 frames,
32,768 outputs, and 234,881,024 multiply-accumulates. Independent qNaN-filled
outputs for batch sizes 1, 2, 4, 8, and 16 were read back across the complete
524,288-output union and had zero U32 mismatches. Four balanced
forward/reverse/reverse/forward rounds retained every wall sample:

| Maximum quanta per command buffer | Median wall | Speedup vs size 1 | Maximum observed batch drain |
| ---: | ---: | ---: | ---: |
| 1 | 110.95 ms | 1.000x | 14.5 ms |
| 2 | 88.80 ms | 1.249x | 28.5 ms |
| 4 | 85.75 ms | 1.294x | 28.6 ms |
| 8 | 70.00 ms | 1.585x | 39.3 ms |
| 16 | 68.20 ms | 1.627x | 69.2 ms |

Batch size 8 is selected for production integration. Size 16 reduced median
wall only another 2.6% relative to size 8 while increasing the largest
observed uninterrupted batch drain from 39.3 ms to 69.2 ms. Size 8 is therefore
the measured wall-time/responsiveness Pareto choice; this is an engineering
selection from evidence, not a speculative acceptance threshold.

The external thermal trace contains 85 observations across 84.00394 seconds,
has a maximum 1,004.989 ms polling gap, and remained nominal throughout. It
spans the production result recorded at `2026-08-13T22:45:16.789Z`.

Production commit `66f69709f7e30a9309e2315de0ea2ad7bae71604`
integrated batch size 8 into the shipped `AceVaeChunkGpuBackend`. The frozen
commit then passed an actual-Chrome complete C136-to-C128 mixed-decoder
boundary with three independently allocated paths: portable batch 1,
optimized batch 1 authority, and optimized batch 8 production. Portable batch
1 versus optimized batch 1 had zero U32 mismatches, and optimized batch 1
versus production batch 8 also had zero U32 mismatches across all 12 outputs.
Every path retained 109 logical quanta and 115 physical dispatches. Optimized
batch 1 reported 110 command buffers, 110 drains, 109 real idle intervals, and
110 progress events including readback; production batch 8 reported 15, 15,
14, and 110 respectively. Selector coverage remained exactly 7 tiled, 15
channel-chunked, and 18 portable quanta.

This experiment did not run a complete 256-frame VAE window or a song. The
bounded scheduling result establishes a useful local gain and selects the
integrated production batch policy; neither the bounded timing screen nor the
integrated mixed-decoder correctness boundary supports a precise end-to-end
runtime claim.

## Evidence and disposition

- Evidence conclusion and rationale: `positive`; all candidate sizes preserved
  exact output bits and scheduling accounting, while size 8 reduced median
  bounded-screen wall time from 110.95 ms to 70.00 ms (1.585x) with a 39.3 ms
  maximum observed batch drain.
- Code disposition and rationale: `integrated`; production commit `66f6970`
  uses bounded batch size 8, and the actual-Chrome source/backend boundary
  preserved every final output bit and exact scheduling/selector accounting.
- Result JSON: [`../results/OPT-0006/result.json`](../results/OPT-0006/result.json)
- Raw artifacts:
  - production timing/result:
    `optimization/artifacts/OPT-0006/raw/production-ranges.json`, SHA-256
    `ffd795166e5fe038d60deb0c482e791a8e7552809237f5f1bb392f1ede492cf4`
  - external thermal trace:
    `optimization/artifacts/OPT-0006/raw/production-ranges-thermal.jsonl`,
    SHA-256
    `344b3374bfe18869354a73af6731d3011301dd1b68e09a25b70efdd19a6b1a24`
  - complete mixed-decoder correctness:
    `optimization/artifacts/OPT-0006/raw/correctness.json`, SHA-256
    `37f93c720bf94bb9e885d384782c76af7d484b0425fea9d3bc2b40ea5de27cae`
  - integrated production-backend mixed-decoder correctness:
    `optimization/artifacts/OPT-0006/raw/integrated-decoder-correctness.json`,
    1,347 bytes, SHA-256
    `8d04e3b84f4740d50807d012fb7b7944e1907f300ae596f1ce955560cbba045b`
- Interactions: holds OPT-0002 quantum bounds and OPT-0004/OPT-0005 kernels
  fixed; removes only repeated physical submission/drain/idle overhead
- Revisit when: the production source/backend boundary regresses,
  target-browser responsiveness changes materially, or another scheduling
  mechanism supersedes bounded batch size 8
