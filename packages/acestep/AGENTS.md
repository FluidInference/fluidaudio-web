# Repository contract

`ace-step-1.5.wgsl` is a custom ACE-Step 1.5 Turbo browser inference runtime.
`PLAN.md` is the canonical implementation program and `main` is the active
branch.

## Current stage

Stage 1's browser correctness and listening gate was approved by the repository
owner on 2026-08-13. Stage 2 (measured optimization and release hardening) is
explicitly authorized.

The hard boundary in `PLAN.md` remains binding. The approved audio identities,
quality envelope, and explicit authorization are recorded in
`LISTENING_CANDIDATE.md` and `optimization/BASELINE.md`. Do not infer approval
for a future quality-affecting change from a benchmark or numerical pass; apply
the experiment's declared correctness and listening gates.

Tiled attention, bounded model staging, chunked VAE decoding, activation
liveness, and cooperative GPU submission are Stage 1 safety architecture, not
permission to change model math for speed.

## Pinned truth

- ACE-Step source behavior:
  `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`
- ACE main model snapshot:
  `19671f406d603126926c1b7e2adc169acbcade22`
- ACE 0.6B planner snapshot:
  `148d8ea0225bdab342ee1ae3a354275ccd60ca80`
- Parakeet implementation reference:
  `../parakeet.wgsl` at
  `7ee112738262a6f5a0efd2f150748a4087432fbb`

Never silently advance a source revision. A revision change requires reviewed
source contracts, new package identities, regenerated reference fixtures, and
a documented reason.

## Model preparation

`model/convert.py` is the sole user-facing raw-source-to-browser-package entry
point. Use its frozen uv environment with CPython 3.13. Generated packages,
download caches, reports, virtual environments, and transactional staging never
belong in Git.

The converter must be deterministic, source-authenticated, complete in its
consumed/excluded tensor accounting, bounded-memory, and transactional. Never
add an undocumented notebook or manual binary-editing step. Future compression
extends the same converter and manifest rather than creating another pipeline.

## Correctness

- Use committed fixture identities and deterministic seeds/initial noise.
- Validate package layouts, primitive kernels, layers, subsystems, each denoise
  step, final latent, VAE chunks/seams, and end-to-end output in that order.
- Keep FP32 reductions/softmax and the packed-BF16 reference profile until a
  lower-precision path has passed the required numerical and listening gates.
- Preserve the pinned Gradio Turbo sampler behavior: shift 3 Euler with DCW
  enabled in `double`/Haar mode. Resolve direct generation to low/high scalers
  0.05/0.02 and Think/planner generation to 0.02/0.06. A DCW-disabled path may
  be diagnostic, but it is not the primary product oracle.
- A planner-enabled product request includes the pinned default CoT phase for
  caption rewrite, language and missing metadata, followed by semantic-code
  generation. A fixture that pre-supplies metadata is useful but is not a full
  planner oracle.
- Validate the raw VAE waveform before applying the ordinary -1 dBFS global
  peak normalization. Implement normalization with bounded storage rather than
  retaining duplicate full-song waveforms.
- Browser output is never its own oracle. Goldens originate from the pinned
  upstream behavior and are reviewed before acceptance.
- Large weights, tensors, audio, and raw profiles remain ignored; commit their
  hashes, settings, small diagnostic slices, and reproduction recipes.

## WebGPU runtime

- Inference lives in a dedicated worker.
- Do not mirror the multi-gigabyte package in JavaScript or WASM memory.
- Do not materialize dense global-attention score matrices or dense local masks.
- Keep the DiT resident for all eight evaluations, then drain and explicitly
  destroy it before loading the VAE.
- Use one FIFO graph owner and a measured, bounded queue depth in the
  cooperative production profile.
- Keep submission quantum, drain policy, queue depth, and cooperative interval
  explicit and benchmarked. The accepted baseline uses one outstanding command
  buffer plus a real queue-empty interval; optimization may change those values
  only with correctness, cancellation, and responsiveness evidence.
- Never mutate aliased shared graph storage with `queue.writeBuffer` while a
  recorded quantum may still read it.
- Treat `shader-f16` as an optimized-profile requirement; subgroup kernels must
  retain a portable workgroup-memory counterpart.

## Optimization history

Every material experiment receives a never-reused ID and a record under
`optimization/` before performance code is changed. Positive, negative, and
inconclusive evidence all remain in the ledger, independently of whether code
is benchmark-only, pending integration, integrated, superseded, or abandoned.
Read the ledger before proposing work and do not repeat an unchanged abandoned
experiment unless its recorded revisit condition is satisfied.

Reportable performance comparisons follow the thermal protocol in `PLAN.md`.
Do not compare a cool baseline against a heated candidate or publish only the
fastest sample.

## Git discipline

Commit and push coherent checkpoints listed in `PLAN.md`. Start risky work from
a pushed checkpoint. Do not mix unrelated refactors with measured changes, and
do not rewrite a shared commit after its SHA appears in a fixture, benchmark,
or optimization record.
