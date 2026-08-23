# OPT-0071 — Warm-cache WebCrypto production integration

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Risk: package trust, bounded host memory, cancellation latency, cache-only
  initialization, and lifecycle; no model math, package, GPU kernel, sampler,
  or output change

## First-principles basis

OPT-0069's literal isolated result remains failed/inconclusive because its
pre-registered component rule allowed at most `20%` read/copy regression and
measured `2.965945018071039`. This new ID does not waive, reinterpret, or
replace that result.

The same evidence identified a narrower product-causal mechanism. Across the
complete `7,325,999,133`-byte physical authentication inventory, sequential
one-file-at-a-time WebCrypto had a `6,700.949999988079 ms` median,
`1,093,277,689.4340403 B/s` throughput, and paired complete-wall savings of
`22,220.94999998808 / 22,603.849999964237 ms`. All `156` digests were exact in
all eight arms, every trace was nominal, and the conservative transient bound
was `365,005,824` bytes.

The read/copy increase is expected mechanism-internal work: `File.arrayBuffer`
materializes one bounded whole-file input so the browser's native SHA-256 owner
can replace approximately `28.2 s` of scalar TypeScript compression. It is not
an independent product stage. The causal product questions are therefore:

1. does complete cache authentication remain exact and bounded at the real
   production seam; and
2. does unchanged cache-only product initialization reach READY at least
   `15 s` sooner in both paired directions without moving cost into any
   unrelated stage?

Those questions require a new integration experiment rather than a post-hoc
change to OPT-0069's gate.

## Frozen integration direction

1. Replace only the current cached-file scalar verifier with sequential
   `File.arrayBuffer()` plus `crypto.subtle.digest("SHA-256", bytes)`.
   Acquisition remains sequential, so exactly one physical payload may be
   materialized and hashed at a time. Release the explicit buffer before
   opening the next file.
2. Preserve exact manifest size and SHA-256 comparison. Publish the immutable
   `File` proof only after both match. Digest aliases reuse that one exact
   physical proof exactly as today; never skip, sample, truncate, or trust a
   cache marker without hashing payload bytes.
3. Check cancellation immediately before the whole-file read, immediately
   before starting WebCrypto, and immediately after its promise completes.
   WebCrypto has no internal `AbortSignal`; disclose and retain the bounded
   one-file cancellation boundary. Do not add a scalar fallback after a
   WebCrypto error or cancellation.
4. Preserve current unreadable/mismatched-cache failure behavior, cache-only
   failure behavior, storage accounting, progress, proof binding, acquisition
   ordering, download behavior, and cleanup. Never mutate an authentic file
   for a corruption screen.
5. Do not change package manifests or bytes, upload behavior, GPU construction,
   queue/drain policy, model residency, planner, conditioning, DiT, VAE,
   normalization, output, or product defaults under this mechanism.
6. Keep diagnostic owner selection confined to the explicit benchmark seam.
   Production may select WebCrypto only after every gate below passes; there
   is no silent runtime A/B selector or compatibility fallback.

## Exact authentication and memory gates

- Authenticate the same three trust roots as OPT-0069: main manifest SHA-256
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`,
  revision-7 dense manifest SHA-256
  `d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f`,
  and revision-6 VAE manifest SHA-256
  `94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949`.
- Require exactly `158` logical records, `156` unique physical digests,
  `7,330,447,819` logical bytes, `7,325,999,133` physical bytes, and inventory
  fingerprint
  `b8fef9059f8d584fd2b17a8d13c21d7e97aa7bdd1b38263a5faa7d3a5e5fb1ce`.
  Also reconcile the diagnostic `102`-file / `5,731,837,696`-byte upload
  subset, but never use it as the ready-wall authority.
- Every control and candidate arm must reproduce all `156` exact manifest
  digests and the `158` logical mappings. Require deterministic repeats,
  correct alias reuse, exact received lengths, no false proof publication,
  and zero cache/package mutation.
- Retain OPT-0069's independent NIST, boundary-length, split-update,
  signed-byte, greater-than-32-bit length, one-bit corruption, short-read,
  pre-abort, and misuse screens. Add injected array-buffer rejection,
  WebCrypto rejection, digest mismatch, success cleanup, and cancellation
  immediately before/during/after the in-flight-file boundary.
- The largest eligible file must remain strictly below `128 MiB`. Exactly one
  payload buffer may be live; maximum explicit live payload remains at most
  `121,668,608` bytes and the conservative logical transient remains strictly
  below `384 MiB` (`402,653,184` bytes). Record exposed browser memory
  telemetry, explicit live-buffer count/bytes, and release before the next
  file.
- Success, cancellation, missing/corrupt cache, injected read/digest failure,
  and dispose must retain idempotent cleanup, no post-cleanup callback/work,
  and zero live payload buffers. No GPU resource may outlive ordinary pipeline
  disposal.

## Balanced production warm-initialization gate

Use unchanged cache-only product initialization and exactly four fresh arms in
`A1 scalar / B1 WebCrypto / B2 WebCrypto / A2 scalar` order. Every arm must use
a fresh worker and a distinct thermal gate after at least `30` continuous
nominal seconds. Continue its raw trace through READY or failure, ordinary
disposal, payload release, and worker termination before preparing the next
arm. Retain all samples, rejected preflights, errors, heartbeats, memory
observations, and thermal transitions.

The authoritative ready wall begins immediately before the ordinary product
worker starts cache-only initialization and ends at the same ordinary READY
publication used by the application. It includes manifest fetch/authentication,
marker-qualified OPFS lookup, complete payload authentication, device setup,
and every other current initialization member. It excludes benchmark setup,
synthetic correctness, the nominal wait, receipt serialization, and any work
after READY. Both owners must use byte-identical manifests/cache entries and
the same browser process/machine identity.

Record separately, without changing the wall boundary:

- complete cache-authentication wall and its read/copy, hash, finalization,
  comparison, and release members;
- manifest fetch/parse, marker lookup, storage planning, device request, GPU
  upload, compilation/construction, queue drain/gap, progress, READY
  publication, disposal, and residual walls; and
- exact download count/bytes, proof inventory, GPU resource/lifecycle counts,
  heartbeat gaps, peak explicit host bytes, and thermal observations.

All of the following are required:

1. `A1_ready - B1_ready >= 15,000 ms` and
   `A2_ready - B2_ready >= 15,000 ms`. Both paired directions must pass; a
   median or favorable direction cannot replace either raw comparison.
2. Candidate complete-authentication median is at most `8,000 ms`, physical
   throughput is at least `915,749,892 B/s`, and complete authentication saves
   at least `15,000 ms` in both paired directions.
3. The known read/copy component may exceed scalar by more than `20%` because
   it is inside the changed authentication mechanism. It must still be
   reported exactly and included in both complete-authentication and READY
   walls. No read/copy-only speed claim is permitted.
4. Every unrelated initialization stage and the aggregate unrelated wall must
   regress by no more than `2%` in either paired direction. Retain exact
   absolute deltas and ratios; do not hide work in residual, setup, cleanup,
   compilation, upload, or READY publication.
5. Correctness, inventory, memory, cancellation, lifecycle, cache-only, and
   thermal provenance gates all pass with zero download bytes and no cache
   mutation in accepted warm arms.

A literal failure of any required direction or safety condition is a
production non-pass. Do not select WebCrypto by appealing to OPT-0069's
isolated aggregate result alone.

## Authority and decision boundary

- [OPT-0069 isolated hash screen](OPT-0069-warm-cache-authentication-hash.md)
- [OPT-0069 failed/inconclusive receipt](../results/OPT-0069/result.json)
- [OPT-0064 warm-start attribution](OPT-0064-direct-request-warm-start-load-overlap.md)
- `src/model/acquire.ts` current verifier and proof owner
- `src/model/cache.ts` marker-qualified OPFS cache owner
- `src/model/sha256.ts` scalar control owner

A complete pass authorizes the bounded WebCrypto verifier as the production
warm-cache authentication owner. It does not authorize skipping
authentication, parallel whole-file payloads, WASM SIMD, package/model changes,
an output-quality claim, or an under-one-minute product claim. After selection,
the cumulative optimized stack still requires its own exact short-output check
and complete 180-second Generate-to-WAV measurement.

No production edit, browser/GPU work, cache mutation, package change, or
performance claim occurred when this experiment was registered.

## Closeout — literal non-pass with decisive causal evidence

The preserved result is
[`result.json`](../results/OPT-0071/result.json), SHA-256
`ae1e63150e1834010a4fdc3a338fdcec0afb7db1af4f83e5eb15a1fc26f4e594`,
with literal status `failed-or-inconclusive`. The complete untimed trust proof
matched all `158/156` logical/unique records and `7,325,999,133` physical
bytes; ordinary timed initialization authenticated its unchanged `151/149`
payload inventory. Every arm was exact, nominal, cache-only, mutation-free,
and lifecycle/cancellation safe. WebCrypto kept one file live, peaked at
`121,668,608` explicit bytes and `365,005,824` conservative transient bytes,
and had a `6,290.35 ms` authentication median.

The causal READY savings were `22,724.3 ms` and `22,098.1 ms`, with
authentication savings of `22,723.0 ms` and `22,114.9 ms`. The literal gate
failed because several tiny unrelated stage ratios exceeded `2%` and the A2
aggregate unrelated wall regressed `16.8 ms` (`3.174%`); the other aggregate
improved `1.3 ms`. Preserve that non-pass without relabelling it. The result is
nevertheless decisive exact/safe evidence that the approximately 22-second
saving is owned by authentication, so the narrower owner-authorized production
selection is allocated under OPT-0073. OPT-0071 itself remains benchmark-only.
