# OPT-0004 — Operation-native FP32 VAE Conv1D tile

## Status

- Evidence: `positive`
- Disposition: `integrated`
- Date: 2026-08-13
- Author/agent: Codex
- Risk class: `exact`; the arithmetic claim is scoped to bit identity with the
  accepted scalar FP32 GPU kernel on the pinned Chrome/M3 target because WGSL
  permits contraction and reassociation

## Hypothesis

The authenticated 256-latent-frame VAE window spends 548,502,631,424 valid
multiply-accumulates in Conv1D, about 88% of decoder convolution work. The
correctness kernel assigns one output scalar to each invocation, causing every
adjacent output time and channel to reload overlapping input windows.

A workgroup that owns 16 adjacent output times by eight output channels can
stage one shared FP32 input halo plus the current K tap's eight-channel weight
tile. It can then reuse input across channels and neighboring times, and reuse
weights across the 16 time lanes, while keeping the converter-native weight
layout. This should reduce storage traffic without changing precision, output
ownership, or the source reduction order.

The first measured production shape is `block-4-res-1-conv1`: batch one,
491,520 frames, 128 input and output channels, K7, stride/dilation one, padding
three, and bias. It has 62,914,560 outputs and 56,371,249,152 valid MACs.

Expected performance is a hypothesis, not an automatic acceptance threshold.

## Identity

- Baseline commit:
  `4c43b1434fa286251538fc44098b691b05d15da2`
- Candidate and harness commit:
  `eab519804278a0ff1a69616efed0191254983471`
- Production integration commit:
  `48148ad1c791d26653d95418f55720240bafddff`
- Model manifest SHA-256:
  `d133b21d55bb6c00ad132aeaa83549ccec1a06c581c9b259268670dcf694fb55`
- Fixture manifest SHA-256:
  `cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`
- Execution profile: `reference-bf16-subgroups`; VAE tensors and arithmetic
  remain FP32
- Machine: MacBook Air `Mac15,12`, Apple M3, 10 GPU cores, 16 GB unified
  memory; capture exact OS, Chrome, and adapter identity in the result

## Change

The initial candidate is benchmark-only and does not modify `src/`, the model
package, or production dispatch selection.

- Use `@workgroup_size(16, 8, 1)`; each invocation exclusively owns one output
  scalar for one time and one output channel.
- Cooperatively stage the 22-by-128 logical FP32 input halo transposed into a
  `[128,23]` padded workgroup tile. For each K tap, stage its eight-by-128
  weight tile into a padded `[8,129]` workgroup tile before folding input
  channels. Total workgroup storage is 15,904 bytes for the measured shape.
- Keep weights in `[output_channel,kernel,input_channel]` order and initialize
  the accumulator from bias once.
- Visit kernel then input channel in increasing order with the same source
  expression as the scalar baseline. Use no explicit FMA, vector reduction,
  cross-lane partial sum, precision change, or fused epilogue.
- Stage out-of-domain input positions safely, but preserve the baseline's
  validity predicates and skip every invalid kernel tap; multiplying a padding
  placeholder by zero is not equivalent for signed-zero/NaN behavior.
- Keep every lane in the uniform K loop: load weights, barrier, accumulate,
  barrier before overwriting the shared weight tile. Put the input-tile barrier
  before all output/range/channel tail predicates.
- Hold the integrated `OPT-0002` range policy and scheduler topology fixed so
  the paired result attributes the kernel rather than scheduling.

The candidate may initially specialize K7, stride one, dilation one, and
channel-aligned full-row ranges. Unsupported shapes fail closed and continue to
use the scalar production baseline.

## Correctness gate

- Authority: actual scalar-versus-tiled GPU bit identity on pinned Chrome/M3,
  plus independent CPU indexing and arithmetic sentinels.
- Prove exact-once output coverage and correct input/weight indexing for left
  and right padding, time/channel tails, batch boundaries, and range tails.
- Exercise bias/no-bias, signed zero, mixed magnitudes, contraction-sensitive
  values, nonmultiples of both tile dimensions, and a complete manageable
  multi-range graph.
- Prefill outputs with a non-finite sentinel; require complete finite writes,
  zero GPU bit mismatches, fingerprints, and independent edge/range sentinels.
- Validate capability, binding size, alias rejection, lifecycle, compilation
  failure, cancellation, and destruction contracts.
- Listening is unnecessary while all relevant GPU and higher-level outputs
  remain bit-identical. Any numerical difference changes the risk declaration
  and requires appropriate subsystem/waveform/listening evidence.

## Benchmark protocol

- No song. Use the complete first production operation when bounded GPU-side
  comparison/readback makes it practical, plus representative full-range
  quanta and a manageable complete correctness graph.
- Compile and allocate outside timing; warm both kernels symmetrically.
- Use a continuously nominal thermal pre-gate and poll through the sample.
- Run balanced paired baseline/candidate orders in one visible page, retaining
  every fenced wall-time, encode, submit, drain, idle, heartbeat, and memory
  observation.
- One controlled run may establish a clear direction. Repeat independently
  when variance overlaps the decision or production attribution needs it.
- Maintain bounded ranges and test that cancellation prevents submission after
  the active quantum finishes.
- Do not escalate to a full VAE window or song unless integrated evidence makes
  that run decision-relevant.

## Results

The first actual-GPU screen used the exact production channel, kernel, stride,
dilation, padding, weight-layout, range-policy, and scheduler geometry with a
65,536-frame representative rather than the complete 491,520-frame operation.
It covered 8,388,608 outputs, 32 bounded ranges, and 7,515,996,160 valid MACs.
Compilation, allocation, upload, and symmetric warmups were outside timing.

Across four balanced in-page paired rounds:

- scalar active wall samples were 721.5, 649.0, 615.0, and 456.2 ms, with a
  632.0 ms median;
- tiled active wall samples were 223.7, 205.2, 207.8, and 215.2 ms, with a
  211.5 ms median;
- the tiled candidate won all four rounds and improved median active wall time
  by 2.988x;
- median logical throughput rose from 11.901 to 35.548 GMAC/s; and
- the candidate's maximum single-range drain was 20.0 ms. The measured-page
  maximum animation-frame and 10 ms timer gaps were 7.4 and 12.1 ms.

The scalar samples drifted downward from 721.5 to 456.2 ms within the page, so
2.988x is a ratio-of-medians direction estimate rather than an independently
replicated precision claim. The candidate remained comparatively stable, every
same-round pair favored it, and even the least favorable pair was 2.120x.

The external thermal log spans pre-gate, run, and post-run: 92 observations
over 91.336 seconds, maximum poll gap 1,010.035 ms, and zero non-nominal
observations. The in-page gate retained 30.106 continuous nominal seconds.

Actual-GPU correctness passed with zero scalar/tiled bit mismatches across all
three complete manageable preflight graphs, including batch/range boundaries,
padding, time/channel tails, bias/no-bias, signed zero, contraction-sensitive
values, and source-order cancellation. Independent CPU sentinels were exact.
For the production-geometry representative, the first, middle, and last
complete 262,144-element ranges were prefilled independently and compared:
786,432 finite elements, zero bit mismatches, matching fingerprints, and exact
CPU sentinels. The cancellation probe drained the active range and prevented
every later range from being submitted.

This run did not execute the complete 491,520-frame operation, a full VAE
window, a waveform, or a song. It therefore establishes a strong primitive and
scheduler-inclusive integration direction, not end-to-end performance.

Production commit `48148ad1c791d26653d95418f55720240bafddff`
then integrated the proven kernel through a fail-closed decoder selector. Under
the shipped 16 KiB workgroup-storage capability, the canonical 256-frame graph
selects exactly four K7 operations and 365 bounded quanta: `conv1`,
`block-3-res-1-conv1`, `block-4-res-1-conv1`, and `conv2`. K1 operations and
K7 dilation-three/dilation-nine shapes retain the portable scalar path. FIFO
scheduling is unchanged: each quantum is submitted alone, fully drained, and
followed by the real queue-empty interval before the next quantum. Existing
backend cancellation tests remain green.

An actual-Chrome integrated decoder A/B exercised one complete nontrivial
C128 block with both the forced-portable and production-default profiles. The
final FP32 output had zero U32 bit mismatches. The default profile used tiled
`conv1`, `block-0-res-1-conv1`, and `conv2` while the other block Conv1D
operations exercised the fallback path. All twelve final FP32 output values
were also compared against the CPU oracle, with a maximum absolute error of
0.000885. This exact portable/default output
identity closes the applicable integration gate without a new listening test.

The integrated decoder raw artifact is retained locally at
`optimization/artifacts/OPT-0004/raw/integrated-decoder-correctness.json`,
SHA-256
`433473d1863359da83d4917dc8745076885491df8f1c23029aebda7866fdf9e1`.
The production integration does not turn the representative benchmark into a
full-operation or end-to-end claim: the complete 491,520-frame operation, full
VAE window, waveform, and song remain unexecuted for OPT-0004.

Canonical result:
`optimization/results/OPT-0004/result.json`.

Ignored raw artifacts retained locally:

- `optimization/artifacts/OPT-0004/raw/representative-paired-ab.json`, SHA-256
  `6582588efa752d4203238a329be10ad4d36a95503c72db133c451e63b31c211c`;
- `optimization/artifacts/OPT-0004/raw/representative-thermal.jsonl`, SHA-256
  `317ca8806148f9ce375c26d14e98dc512a12d46fb1b4da80badc00965b32999f`.

One earlier actual-Chrome attempt failed before timing because the candidate
WGSL used the reserved identifier `active`. Commit
`eab519804278a0ff1a69616efed0191254983471` renamed it and added compilation
diagnostics; the canonical result above is the frozen post-fix run. That failed
page JSON was not archived, so no raw-artifact identity is claimed for it.

## Evidence and disposition

- Evidence conclusion: `positive`. The exact target-browser candidate produced
  a stable, substantial reduction in representative active wall time and won
  every paired round while passing its declared correctness, responsiveness,
  cancellation, and thermal gates.
- Disposition: `integrated`. The fail-closed production selector, mixed
  tiled/portable decoder graph, exact actual-Chrome final-output comparison,
  bounded FIFO scheduling, resource lifecycle, and existing cancellation
  contract are now in the production path.
- Result JSON: `optimization/results/OPT-0004/result.json`
- Benchmark candidate and harness:
  `eab519804278a0ff1a69616efed0191254983471`
- Production implementation:
  `48148ad1c791d26653d95418f55720240bafddff`
- Interactions: holds integrated `OPT-0002` scheduling fixed and is independent
  of the positive/pending `OPT-0003` DiT GEMM work
- Revisit when: profiling makes additional K7 shapes, a larger workgroup-memory
  profile, or an incompatible better Conv1D kernel decision-relevant.
- Follow-ups: retain this integrated floor and use a larger operation, VAE
  window, waveform, or song only when it can change an optimization or release
  decision.
