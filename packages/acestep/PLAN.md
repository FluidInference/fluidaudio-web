# ACE-Step 1.5 WebGPU project plan

## Current status and authority

This repository contains a custom browser inference implementation of
ACE-Step 1.5 Turbo. Stage 1 browser correctness and human listening were
approved by the repository owner on 2026-08-13. Measured optimization and
release hardening are authorized.

The approved audio identities, quality evidence, and open external-parity
limits are recorded in `LISTENING_CANDIDATE.md` and
`optimization/BASELINE.md`. Git history preserves the detailed Stage 1 build
plan and checklists; they are no longer a live optimization backlog.

The owner has delegated optimization design, experiment selection, benchmark
cadence, implementation details, and integration decisions to the agent. Those
decisions must be driven by current measurements and engineering judgment, not
by predictions written before the implementation existed.

## Product objective

The sole headline performance objective is:

> Generate a three-minute song from a warm model cache in under one minute on
> the local 10-GPU-core, 16 GB Apple M3 machine.

The measured interval starts at Generate and ends when the final WAV is ready.
It includes per-generation cache reads and verification, GPU upload and
compilation that still occurs per generation, planner work when enabled,
conditioning, all eight DiT evaluations, VAE decode, normalization, and output
finalization. Initial network download and first cache population are reported
separately.

Every result must identify whether it used direct or planner-enabled generation
and all other request/runtime settings. There are no independent performance
requirements for GEMM, attention, planner, DiT, VAE, loading, or any other
stage. Per-stage timings and throughput are diagnostic tools for reaching the
end-to-end objective, not acceptance thresholds.

Direct and default-planner generation are separate performance tracks. The
direct path is the first convergence target; the default planner is measured
and optimized as its own workload, and evidence from one track is never used to
claim that the other has reached the headline objective.

## Mission and version-one scope

The product is private, local, high-quality music generation in a normal
browser on consumer Apple Silicon. Version one includes:

- ACE-Step 1.5 Turbo with eight denoising evaluations;
- prompts, supplied lyrics, deterministic seeds, and the ordinary Turbo
  controls;
- direct generation and the optional default two-phase 0.6B planner workflow;
- the Qwen3 text encoder, ACE condition encoder, semantic detokenizer, complete
  24-layer DiT, Haar DCW Euler sampler, and decoder-only Oobleck VAE;
- stereo 48 kHz WAV output with upstream-compatible peak normalization;
- authenticated persistent model caching, bounded phase staging, progress,
  cancellation, diagnostics, and transactional output; and
- Chrome on the local M3 as the first measured production target.

Weight compression, larger planners, source-audio editing/cover workflows, the
VAE encoder, Safari/Firefox tuning, iPhone support, and cosmetic application
work remain later projects unless current evidence makes one of them necessary
to the headline objective.

## Pinned truth

The correctness baseline remains bound to:

- ACE-Step source commit
  `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`;
- ACE main-model revision
  `19671f406d603126926c1b7e2adc169acbcade22`;
- ACE 0.6B planner revision
  `148d8ea0225bdab342ee1ae3a354275ccd60ca80`;
- the authenticated model and fixture manifest identities recorded in
  `optimization/BASELINE.md`; and
- the owner-approved browser artifacts recorded in
  `LISTENING_CANDIDATE.md`.

The accepted product sampler is the pinned Turbo/Gradio path: shift 3, eight
Euler evaluations, Haar DCW in `double` mode, direct low/high strengths
0.05/0.02, and planner/Think strengths 0.02/0.06.

Native CUDA/XPU BF16 taps remain unavailable external parity evidence. The
approved browser baseline is the optimization authority, but no result may be
described as native-upstream numerical identity without the missing native
captures.

## Hard engineering constraints

These are product constraints, not speculative optimization targets.

### Quality and correctness

- Preserve the owner-approved musical quality envelope.
- Keep deterministic fixture, package, seed, sampler, and request identities
  fixed across comparisons.
- Test a change at the smallest boundary capable of detecting its failure, then
  escalate in proportion to its numerical and product risk.
- Exact changes must preserve the relevant accepted bits or explicitly explain
  a target-browser arithmetic contract such as permitted WGSL contraction.
- Changes to precision, accumulation/reduction structure, sampling, attention,
  normalization, VAE math, or final latent require declared numerical
  tolerances and the applicable higher-level and listening checks.
- The optimized M3 profile may require `shader-f16` and use FP16 for heavy
  storage and arithmetic. This is a mixed-precision direction, not permission
  to force every value through FP16: reductions, normalization, softmax, and
  range-sensitive islands remain FP32 wherever current evidence requires it,
  and legitimate overflow is never hidden by clamping.
- The packed-BF16/FP32 profile remains the numerical and listening oracle until
  an FP16-first profile passes its declared primitive, subsystem, latent,
  waveform, instrumental, and vocal gates. Oracle status does not require the
  reference profile to remain a shipped compatibility path.
- Passing a microbenchmark never substitutes for the relevant layer,
  subsystem, final-latent, waveform, or listening evidence.
- Fail closed on non-finite values, malformed packages, unsupported device
  capabilities, and invalid resource geometry.

### Memory and resource lifetime

- The production pipeline must complete reliably on the 16 GB M3 without OOM,
  device loss, or uncontrolled process growth.
- Never mirror the multi-gigabyte package in a monolithic JavaScript or WASM
  allocation.
- Keep downloads, hashing, uploads, staging, readback, and output creation
  bounded and backpressured.
- Retain only resources needed by the active phase; drain before destruction or
  reuse and validate every activation alias against liveness.
- Keep the DiT resident across its eight evaluations, then release it before
  VAE loading.
- Decode and emit audio incrementally rather than retaining avoidable full-song
  copies.
- Measure logical GPU allocations separately from physical browser-process
  footprint when a release or major integration decision needs that evidence.

### Responsiveness and cancellation

- Production inference runs in a dedicated worker and must leave the page,
  compositor, browser, and ordinary desktop use meaningfully responsive.
- GPU submissions remain bounded, observable, and cancellable. Queue depth,
  quantum size, drain policy, and cooperative interval are implementation
  variables selected from current evidence rather than frozen constants.
- Never mutate shared/aliased storage while recorded GPU work can still read it,
  and never release a resource before its final fence.
- Cancellation must stop further work at a bounded scheduling boundary and
  clean up owned GPU, cache, and output resources coherently.

### Reproducibility and provenance

- Source revisions, model snapshots, manifests, and generated package
  transforms are immutable and authenticated.
- `model/convert.py` remains the deterministic, transactional source-to-package
  entry point.
- Every material experiment receives a never-reused ID and records its exact
  commits, model/fixture identity, browser/adapter identity, settings, samples,
  correctness evidence, and code disposition.
- Model weights, caches, audio, raw profiles, temporary environments, and other
  large generated artifacts stay out of Git; commit their hashes and
  reproduction details when they matter.
- Never erase an unsuccessful experiment or rewrite shared history to make the
  optimization path look linear.

## Stage 1 baseline and remaining release evidence

The complete direct and planner-enabled browser paths have produced
owner-approved instrumental and vocal audio. This closes the product listening
gate and authorizes optimization; it does not close every release or external
parity item.

Still required before release:

- a practical empirical three-minute run on the 16 GB M3, including physical
  memory, sustained behavior, output validity, and absence of device loss;
- final stock-Chrome validation of the optimized production path;
- coherent cancellation, cleanup, cache recovery, OOM, and device-loss
  behavior at the optimized high-water mark; and
- native CUDA/XPU evidence before making any upstream-equivalence claim.

The owner explicitly approved deferring the expensive three-minute run until
optimization makes it decision-relevant and practical. Deferral is not a pass.

## Adaptive optimization method

There is no fixed Stage 2 tactic sequence or permanent implementation to-do
list. The current profiler, ledger, and source determine the next work.

The operating loop is:

1. Measure or derive the current dominant end-to-end cost.
2. Form one technically coherent hypothesis with a clear mechanism and risk
   class.
3. Use the cheapest authoritative correctness and performance boundary that can
   decide whether the idea is useful.
4. Preserve a credible positive result as part of the working optimization
   stack when it is composable and its likely benefit justifies integration
   cost.
5. Integrate in stages, validating the relevant real layer, subsystem, latent,
   waveform, or product boundary before production selection.
6. Reprofile after a coherent batch because bottlenecks and worthwhile next
   steps change as optimizations accumulate.
7. Run short or three-minute generations only when they can change an
   integration, convergence, memory, quality, or release decision.

Predicted speedups and throughput figures are hypotheses. Missing a prediction
does not veto a correct positive optimization. An optimization is abandoned
only when evidence shows no credible net benefit, a hard correctness/quality or
safety regression, unreasonable complexity for the measured value, or
domination by an incompatible better implementation.

Several positive candidates may remain pending integration while higher-value
work continues. A later candidate supersedes an earlier one only after it is
actually demonstrated, not because it is imagined to be faster.

## Evidence escalation and benchmark cadence

Benchmark cost is proportional to uncertainty and decision value:

1. static workload/resource analysis;
2. deterministic unit and shader-contract tests;
3. actual-GPU primitive or exact-shape paired measurements;
4. a real repeated layer, block, or bounded subsystem fixture;
5. a short end-to-end generation; and
6. the three-minute product run.

Use balanced in-page A/B ordering when it controls thermal and browser
variance better than separate runs. Record every observation, not only the
fastest one. The owner revised the practical thermal rule on 2026-08-14:
begin attributed measurements only after at least 30 continuous seconds at a
documented nominal thermal state, keep polling through the sample and cleanup,
and disclose every transition. A later non-nominal transition does not by
itself invalidate a balanced same-page comparison whose effect is large and
consistent. Do not repeat an unchanged benchmark solely to obtain an
all-nominal trace. Treat marginal, directionally mixed, or variance-overlapped
results as inconclusive instead.

One well-controlled run can screen a clear positive or negative direction.
Collect independent repetitions when variance overlaps the decision, when
integrating into production, or when publishing a final performance claim.
There is no universal sample count. Short 12–30-second production generations
are ordinary decision tools and may be used whenever they provide clearer
evidence than a narrower harness. The full 180-second generation remains
deliberately infrequent and occurs after shorter evidence makes it worthwhile.

Authoritative timing is fenced wall time at the boundary being evaluated.
Timestamp queries, FLOP/s, per-stage timing, queue-drain timing, memory
breakdowns, and heartbeats are diagnostics that explain the result; none is an
independent product objective.

## Optimization records

`optimization/` is the durable optimization memory:

- `BASELINE.md` records the frozen quality and current integrated baseline;
- `LEDGER.md` indexes every experiment and is the live source of current and
  next optimization work;
- `experiments/` records hypotheses, evidence, decisions, and follow-ups;
- `results/` stores small schema-validated summaries; and
- ignored `artifacts/` retains raw local evidence whose hashes are committed.

Experiment evidence and code disposition are separate dimensions:

- Evidence is `positive`, `negative`, or `inconclusive`.
- Disposition is `benchmark-only`, `pending-integration`, `integrated`,
  `superseded`, or `abandoned`.

`positive / pending-integration` means an isolated improvement is real but its
production work and higher-level gates are unfinished. `negative` means the
hypothesis was refuted or produced no worthwhile net benefit; it does not mean
a useful result missed a speculative target. `integrated` is reserved for code
retained in the production path after its applicable correctness and listening
checks.

## Release criteria

The optimized M3 version is ready for release consideration when:

- a warm-cache three-minute generation completes in under one minute on the
  local M3 under a committed, reproducible protocol;
- the owner-approved quality fixtures and all applicable numerical/browser
  correctness tests still pass;
- the production configurations being shipped complete reliably on the 16 GB
  machine without device loss, OOM, corruption, or resource growth;
- the cooperative default leaves the browser and computer usable, with useful
  progress and cancellation;
- output, cache, cleanup, recovery, and failure paths behave coherently;
- model conversion remains one-command, pinned, deterministic, and documented;
- every material experiment and final benchmark is recorded with complete
  provenance; and
- licenses and attribution are complete and public claims distinguish measured
  results from unsupported hardware or native-equivalence projections.

## Later work

Compression, larger planners, source-audio workflows, additional desktop
browsers, M5 measurements, and mobile profiles are separate evidence-driven
projects. They may be pulled forward only when they are the best route to the
product objective or a release requirement. No M5 or iPhone performance claim
is inferred from M3 success.
