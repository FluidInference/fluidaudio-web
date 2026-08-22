# OPT-0018 — current-production M2250 DiT family profiler

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Date: 2026-08-14
- Author/agent: Codex
- Risk class: `exact`, observational/capture-only
- Registration checkpoint:
  `7c63ea4b6fc36109632f3687bf7652dd9161c72e`
- Implementation: not started at registration

## Hypothesis and context

Attributing every submit-through-drain interval in the exact current
three-minute DiT graph will show whether one current kernel family has a
technically credible path to at least `10,000 ms` of absolute saving. This is a
localization experiment, not an optimization candidate.

Current head already uses authenticated `reference-bf16-subgroups`, DiT
backend `mixed-opt-0009`, dense runtime
`opt-0009-fp16-fp32-dense-v1` / kernel set
`opt-0009-n256-k32-fp16-fp32-v1`, and attention
`fixed32-subgroup-query8`. OPT-0013 measured one isolated exact-M2250 Query8
full-attention primitive at `136.2 ms` versus `1,108.8 ms` portable
(`8.140969x` compute-wall speedup), but current production now already selects
Query8. OPT-0018 measures its remaining production cost; it is not a portable
A/B or an unperformed Query8-integration claim.

The historical three-minute run reported `568.618 s` Generate-to-WAV,
including `125.948 s` DiT and `431.377 s` VAE, under receipt SHA-256
`1a49db615916f4c38ceed7d0dea7cd14075a614247cb2d7249efa7b1dfe4ed70`.
It began after one nominal 30-second gate and ended at thermal level `1`.
That stale `125.948 s` aggregate predates current-head attribution and does not
separate graph drain, idle, readback/map, load/compile, or residual wall. It is
not a current kernel budget, same-run denominator, or under-60-second evidence.

## Frozen authority

Fail closed unless the receipt authenticates:

- ACE source `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0`, fixture manifest
  `cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`,
  main manifest
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`,
  and mixed-dense manifest
  `d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f`;
- the exact canonical 180-second direct-instrumental request bytes/hash:
  `ace-turbo-v1-correctness`, planner disabled, frozen text/metadata/seed,
  shift-3 eight-evaluation Euler, Haar DCW `double`, direct strengths
  `0.05/0.02`;
- capture checkpoint and hashes for the harness and every source controlling
  descriptors, physical composition, runtime selection, or timing; and
- browser, OS, M3/memory/GPU identity, adapter features/limits, and subgroup
  minimum/maximum `32`.

A full-product continuation must also authenticate VAE manifest
`5644bcca87678b4f654b9541459355a73ef136c6bb601aa783b6f50fe2f6dba3`
and profile `opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1`. A
deliberate DiT-only stop does not load or require the VAE.

## Capture-only boundary

Instrumentation may add immutable descriptors and bounded timing collection
only at existing compile, drain, and readback boundaries. It may not change
WGSL/kernel IDs, math/precision, packages/bindings/layouts, graph or sampler,
profile/default/capability selection, primitive/range/pass order, physical
packing/count, queue depth, submit/drain/idle placement, lazy encoding,
progress/cancellation, upload/compile/staging, phase lifetime, VAE, or output.

Build descriptors outside timing. Each drain may perform only a bounded
in-memory numeric write—no per-drain logging, serialization, postMessage,
asynchrony, GPU query, or GPU buffer. Report capture storage/CPU overhead. A
focused contract must prove the disabled path is unchanged and enabled capture
adds observation plumbing only.

## Exact graph and attribution

The exact production graph is batch `1`, `4,500` latent frames, `2,250` DiT
tokens, `98` conditioning tokens, 24 layers, and eight evaluations. The
authenticated tokenizer authority is text `82` plus lyrics `15` plus one
packed no-reference timbre row. The text-token U32LE SHA-256 is
`8067ee5c606e45e54d991364aa82a0ef7303e2a4e98831a01bb974236cafb3b2`;
the lyrics-token U32LE SHA-256 is
`b4b58cd318163b4dfaa02b7ddbf46b18d84a415909c7662f9538c0b9053f3764`.
The earlier `97`-token preflight assumption came from the 12-second fixture;
`180` contributes one more tokenizer token than `12`; the 12-second fixture is
not an admissible authority for this run. `mixed-opt-0009` emits exactly
`2,553` physical graph command buffers. One separate final-latent readback makes
`2,554` total submissions/drains; never count readback as a graph family.

Each graph drain gets one immutable descriptor: physical/logical index and
kind, command ID, subquantum index/count, precompute or evaluation `0..7`,
optional layer `0..23`, ordered primitive/member IDs, backend/kernel identity,
and exactly one family. Cross-family commands use an explicit `mixed` family;
their wall is never split or estimated. Freeze before timing a compact taxonomy
covering precompute/cross-cache, timestep, input, attention projections,
self-full, self-sliding, cross-attention, feed-forward, plumbing, output,
sampler/DCW, and mixed.

The compact raw receipt retains descriptor-table SHA-256 plus an index-ordered
descriptor/timing tuple for every graph drain. It must prove:

- descriptors = graph submissions = graph drains = timings = `2,553`;
- indices `0..2552` occur once with no gap, duplicate, or dropped callback;
- every drain belongs to exactly one family and one precompute/evaluation
  bucket; and
- family sums and precompute-plus-eight-evaluation sums each reconcile to the
  graph submit-through-drain total within a frozen rounding tolerance.

## Run and receipt

Run once after one continuous 30-second nominal start; keep polling through
cleanup, accept the trace, and do not retry unchanged work. Report launch delay
and every later thermal transition.

Two preflight attempts on 2026-08-14 are invalid setup evidence, not timing
runs: both stopped before the first graph command because the capture guard was
stale at `C97`. The first attempt also exposed and corrected lossy failure
serialization; the second preserved the exact guard error. Their external
thermal traces remain ignored as `failed-run-1-thermal.jsonl` (`16,116` bytes,
SHA-256 `3108d30a2286821b1a12bfe03b3e65b83eff86adbf94b1373fdfde935e20369f`)
and `failed-run-2-thermal.jsonl` (`15,755` bytes, SHA-256
`bdb5c649afa4db680807415df2f6dd2f49c7d836c7ac2efc3a9cbd95230e46e2`).
Correcting the authority to exact `C98` does not authorize a retry of executed
graph work because neither attempt executed any graph work.

Authoritative family time is `performance.now()` immediately before each
existing graph `queue.submit` through its matching `onSubmittedWorkDone`.
Report compact family, per-evaluation, and family-by-evaluation totals/counts,
maximum drain, and a bounded slowest list. Separately retain requested
cooperative idle and observed inter-drain/encoding/progress wall,
graph-to-readback idle, readback submit-through-drain and map/detach wall,
complete `dit-denoise` stage wall/residual, DiT load/hash/upload/compile/staging,
and any later VAE/finalization/product wall.

The capture may stop successfully after detached final latent and normal DiT
destruction. State explicitly whether it stopped at DiT or completed the
product; do not run VAE solely for this profiler.

## Correctness and decision

Require the detached `1 x 4,500 x 64` final latent to be exactly `1,152,000`
bytes, entirely finite, nonzero, and raw-SHA-256 recorded; require bit identity
if an identical-request current-head oracle is available. If the product runs
anyway, report WAV structure/finiteness/channels/frames/hash and compare the
identical-request authenticated WAV when available. An unchanged WAV is the
preferred non-perturbation proof, not a reason to run VAE. No listening repeat
is required for capture-only math.

Retain drain-before-release and zero-live cleanup on success, cancellation, and
failure. Any output mismatch, missing drain, reconciliation/schedule failure,
runtime/WebGPU error, device loss, or capture-caused lifecycle change
invalidates attribution.

Pursue a follow-up only if one measured family has a credible mechanism and
attainable floor supporting at least `10,000 ms` absolute saving; a 10-second
family total alone is insufficient. Register that mechanism under a new ID.
Otherwise close the profiler and redirect. OPT-0018 authorizes no optimization
integration, profile/default change, quality/product-speed claim, three-minute
projection, or under-60-second claim.

## Closeout — benchmark attribution retained

The compact profiler and complete thermal trace remain decision-useful, but
the full authenticated browser receipt was never persisted, so the literal
experiment closes inconclusive. This was capture-only benchmark evidence; the
later mechanism-specific DiT experiments supersede the open investigation.
No arithmetic, runtime profile, package, or production default changed here.
