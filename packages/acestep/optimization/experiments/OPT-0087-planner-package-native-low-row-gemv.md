# OPT-0087 — Planner package-native low-row GEMV

## Status

- Evidence: `positive` for the actual-browser package gate
- Disposition: `pending-integration`; complete trajectory and planner-enabled
  product gates remain mandatory
- Production integration: not yet authorized
- Date allocated: 2026-08-21
- Author/agent: Codex
- Risk class: exact-order planner dense-kernel ownership and shared graph state
- Allocation baseline: pushed `main` commit
  `84470d877ba69c6ef7821871f9bdb82d8c757747`

No OPT-0087 package selector, package-native A/B harness, timing sample, or
production change existed when this experiment was allocated. The unchanged
direct kernel was implemented and found raw-U32 exact by OPT-0083, but that
primitive experiment is correctly retained as a negative because its
per-sample submit-and-drain wall bandwidth missed its frozen gate.

The registered package-native actual-browser gate later passed. Production
remains on generic A until every integration gate below passes.

## Material change from OPT-0083

OPT-0083 deliberately put one command-buffer submit and full queue drain around
each isolated seven-GEMM layer sample. Its exact direct arm reached
`29.767441860 GB/s` on diagnostic GPU timestamps and won `16/16` wall pairs,
but fixed browser/driver drain cost reduced authoritative wall bandwidth to
`10.917928005 GB/s`, below that experiment's frozen `20 GB/s` threshold.

This experiment does not retry that primitive or reinterpret its result. It
measures the unchanged direct arm inside one authenticated, resident planner:
all 28 real transformer layers, the real tied head, existing layer command
boundaries, cache traffic, attention, readback, and complete-token wall. There
is no added submit/drain boundary around an individual GEMM or layer. The
question is whether eliminating the generic kernel's 16x/8x padded decode
arithmetic produces a material package-level saving after unavoidable graph
costs. That is the revisit condition recorded by OPT-0083.

## First-principles basis

Each decode row logically streams 880,932,864 packed-weight bytes through the
seven dense operations in 28 layers. A full tied head streams another
444,833,792 bytes. The current generic M16 kernel schedules approximately 16x
the logical layer multiply-adds for M1 CoT and 8x for M2 semantic CFG. Those
extra rows do no useful work.

OPT-0010 measured the 28 layer quanta at 133.3--165.5 ms per token, excluding
the separate tied head. OPT-0083 measured the unchanged direct seven-GEMM arm
at 3.605x aggregate fenced-wall speedup and 7.612x diagnostic GPU speedup in
isolation, with exact increasing-K FP32 outputs. Even allowing for attention,
normalization, cache, submission, and readback floors, a roughly 60 ms/token
package saving would remove about one minute from the approximately 1,010
default CoT plus semantic draws. This is a high-leverage system experiment,
not another geometry micro-tune.

## Frozen arms

- **A — package control:** the current reference-BF16 planner, current generic
  M16/N128/K16 `AceCorrectnessGemmKernel`, current depth-one planner scheduler,
  full production tied head and browser-v1 sampler.
- **B — package direct:** the exact OPT-0083 Arm B kernel, unchanged: WG128,
  one lane per output column, direct source-row-major packed-BF16 loads, and an
  explicit increasing-K scalar FP32 accumulator. Route only planner decode
  contractions whose physical row count is one or two and whose token count is
  one. Keep the generic kernel for prefill, rows above two, unsupported shapes,
  raw-FP16, text encoders, DiT/VAE, and every non-planner owner.

No kernel geometry, package layout, weights, accumulation order, scheduler,
head range, sampler, cache layout, command-buffer boundary, cooperative idle,
or downstream model path may change under this ID. Both arms must coexist as
pipeline-only owners over one authenticated weight allocation; weights may not
be duplicated. An internal per-invocation selector is permitted only for the
balanced gate and must be absent from the public declaration surface.

OPT-0084 sampling, OPT-0085 depth-two planner scheduling, OPT-0082 compact
semantic head, and OPT-0086 downstream scheduling remain disabled so this gate
attributes only low-row GEMV ownership.

## Static gates

Deterministic tests must prove:

- only reference-BF16 planner single-token rows one/two can select B;
- every prefill or unsupported tuple retains A, including a one-row batch with
  more than one token;
- all seven layer GEMM roles and every physical tied-head slice select the
  intended arm, including non-multiple N tails;
- A/B share the same authenticated weight bindings without copies or mutation;
- dispatch, quantum, command-buffer, drain, idle, cache, and readback topology
  is otherwise identical;
- selector failure is fail-closed and ordinary production still selects A;
- destroy and partial-construction failure release both pipeline owners exactly
  once without destroying shared weights early; and
- the experiment API is internal and stripped declaration emit exposes no
  OPT-0087 symbol.

## Actual-browser package gate

Use one authenticated current reference-BF16 planner weight owner and the
accepted deterministic fixtures. Compare one middle-cache M1 CoT token and one
middle-cache M2 semantic token. Before every timed arm, recreate the identical
logical phase/cache state with an untimed control prefill; do not compare
successive cache positions. Compile and warm both pipeline owners before
timing.

After at least 30 continuous nominal thermal seconds, run at least eight
balanced interleaved A/B pairs for each row count (16 pairs total), retaining
the external trace through cleanup. Fenced wall intervals are authoritative.
GPU timestamps, when available without altering topology, are diagnostic and
must never replace a missing wall sample.

Require for every sample and deterministic rerun:

- full-logit raw-U32 identity, including all physical head tails;
- identical cache/status bytes, sampled token, Philox word and absolute cursor;
- identical command, dispatch, drain, idle, map, and live-resource accounting;
- no NaN/non-finite regression, uncaptured GPU error, device loss, or leaked
  buffer; and
- bounded cancellation at each experiment boundary and idempotent cleanup.

B must satisfy all of the following performance conditions:

- both M1 and M2 transformer-layer wall medians are below A;
- both M1 and M2 tied-head wall medians are below A;
- aggregate transformer-layer wall speedup is at least `1.50x`;
- at least `14/16` paired model-through-readback wall comparisons favor B;
- every M1/M2 model-through-readback median and complete-token median is below
  A; and
- aggregate model-through-readback median saving is at least `60 ms`, which
  projects at least 60 seconds over 1,010 draws.

Report complete-token wall, model-through-readback wall, transformer layers,
tied head, readback, sampling, logical weight bandwidth, scheduling counts,
directional pairs, medians, means, and dispersion. Do not add overlapping
intervals. A large diagnostic GPU rate does not waive any wall gate, and a
failed package gate is not retried unchanged.

## Actual-browser package result

The frozen implementation and harness at
`980623e0f9ec918f7a537b6a90e341adaad6d4b0` passed on the target M3 in Chrome
151. One authenticated reference-BF16 owner executed eight balanced
same-state pairs each for middle-cache M1 CoT and M2 semantic tokens. Every
full-logit and appended K/V-cache evidence word, actual write-status word,
sampled token, Philox word, cursor boundary, command/dispatch/drain topology,
cancellation checkpoint, and lifecycle check was exact. Cleanup balanced
`3,683/3,683` buffers and `133/133` maps with zero live resources.

The unchanged direct kernel beat generic production A in all `16/16`
model-through-readback pairs and lowered every M1/M2 layer, tied-head, model,
and complete-token median. M1 layer/model/complete medians fell from
`196.050/267.650/363.750 ms` to `82.700/146.050/242.650 ms`; M2 fell from
`217.950/287.400/363.000 ms` to `93.850/158.450/233.250 ms`. Aggregate layer
speedup was `2.229363x`, and aggregate model-through-readback median fell from
`272.150` to `156.200 ms`, saving `115.950 ms/draw`. The frozen projection is
`117.110 s` over 1,010 draws, clearing every package gate.

The 155-observation continuous trace provided a fresh 32-observation nominal
launch interval with at most `1,002 ms` gaps and no missing observations.
Thermal pressure rose to levels 1 and 2 only after the registered launch; all
65 later non-nominal observations are disclosed, and A/B remained balanced
and interleaved. This is a passing, decision-useful result under the frozen
nominal-launch rule, not a continuously nominal claim.

The complete browser receipt is retained in
[`result.json`](../results/OPT-0087/result.json), SHA-256
`2bbb5e27b39cc545d22089e9f66ce93007e30b3106524f03452c215b03d1ec40`,
`2,799,929` bytes. Its ignored raw trace is bound by SHA-256
`193ecc38a9bbfaa044c288c774a72c3a98d3431a67470979307f8912a84ee8bc`
and byte length `25,155`.

## Integration gate

A passing B may become the strict reference-BF16 planner single-token decode
GEMV owner. Ordinary prefill and unsupported shapes remain generic. Run all
focused kernel, encoder, Qwen, planner-model, executor, sampler, and lifecycle
tests; then run:

1. the complete pinned default-CoT trajectory;
2. the complete 150-code-plus-EOS semantic trajectory; and
3. one planner-enabled product correctness gate through conditioning, all
   eight denoise evaluations, VAE decode, normalization, and WAV encoding.

Require identical declared logits checkpoints, every emitted token, Philox
cursor, conditioning input, final latent raw-U32, raw waveform, normalized WAV
bytes, cancellation, progress, topology, peak memory, and resource lifecycle.
Exact product identity requires no listening retest. Record a fresh production
timing only after integration; never add projections from independent
experiments as though they were one measurement.

## Stop conditions

Stop without integration for any raw-U32, token, cursor, cache, waveform, WAV,
topology, cancellation, or lifecycle mismatch; fewer than `14/16` paired wins;
layer speedup below `1.50x`; model-through-readback median saving below `60 ms`;
or projected saving below 60 seconds. Do not tune another direct geometry,
change arithmetic, combine a second optimization, or waive a wall gate under
this ID.

## Authority

- Negative primitive and exact direct mechanism:
  [OPT-0083](OPT-0083-planner-low-row-bf16-gemv.md)
- Planner token attribution:
  [OPT-0010](OPT-0010-package-native-planner-token-profiler.md)
- Direct kernel:
  [`planner-low-row-bf16-gemv.ts`](../../src/webgpu/kernels/planner-low-row-bf16-gemv.ts)
- Planner model: [`planner-model.ts`](../../src/webgpu/planner-model.ts)
- Approved production behavior: [`PLAN.md`](../../PLAN.md)
- Experiment ledger: [`LEDGER.md`](../LEDGER.md)
