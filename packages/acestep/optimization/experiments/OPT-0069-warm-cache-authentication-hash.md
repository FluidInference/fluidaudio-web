# OPT-0069 — Warm-cache authentication hash acceleration

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Frozen receipt status: `failed-or-inconclusive`
  (`performance.passed: false`)
- Decision: close the literal non-pass without production integration; carry
  the causally positive complete-authentication mechanism into the separately
  registered [OPT-0071 production-seam gate](OPT-0071-warm-cache-webcrypto-production-integration.md)
- Risk: package trust, bounded host memory, cancellation latency, and startup
  responsiveness; no model math, GPU kernel, package, or output change

## First-principles basis

OPT-0064 measured a `29,310.0 ms` cache-only initialization before its
unchanged direct generation. The retained compact failure receipt proves that
`156` cache-authentication operations occurred, but it intentionally omitted
their individual walls; the initialization remainder cannot be assigned by
subtraction. Upload is not the dominant candidate: the later GPU upload of
`5,731,837,696` bytes took only `3,331.1 ms` in total, including OPFS reads,
`writeBuffer`, drains, and cooperative gaps.

The current cache verifier reads each immutable OPFS `File` sequentially and
executes a scalar TypeScript SHA-256 compression loop over at-most-`4 MiB`
chunks. The three exact production acquisition manifests contain:

| inventory | logical records | unique digests | physical bytes |
| --- | ---: | ---: | ---: |
| main reference subset | `103` | `101` | `4,136,399,389` |
| revision-7 dense subset | `48` | `48` | `3,020,808,192` |
| revision-6 VAE subset | `7` | `7` | `168,791,552` |
| complete warm path | `158` | `156` | `7,325,999,133` |

The logical byte total is `7,330,447,819`; aliases are authenticated once and
reuse the exact immutable-`File` proof. The largest physical file is only
`121,668,608` bytes (`116.032 MiB`). Therefore a sequential whole-file
WebCrypto screen can remain bounded to one sub-`128 MiB` input instead of
mirroring a multi-gigabyte package in JavaScript or WASM. This is a materially
different mechanism from skipping authentication: every byte still
contributes to the standard manifest SHA-256.

Standard SHA-256 has a serial block-chaining dependency within one message.
WASM SIMD has no SHA-256 instruction, so a truthful SIMD candidate must hash
independent files in parallel lanes; single-stream `simd128` branding would
not establish a new mechanism. Native browser WebCrypto is the leading arm,
with a bounded four-lane WASM comparison retained only if its implementation
and memory accounting remain exact.

## Frozen experiment direction

1. Add an isolated explicit-button browser benchmark. Phase 1 may not edit
   `src/model/acquire.ts`, `src/model/cache.ts`, `src/model/sha256.ts`, runtime
   selection, manifests, packages, or GPU code.
2. Authenticate the three manifest identities, derive the acquisition subsets
   through the same production helpers, open only marker-qualified OPFS cache
   candidates, and assert the exact inventory above before timing.
3. Arm A is the literal current sequential `File.stream()` plus
   `AceIncrementalSha256` path, including current maximum-`4 MiB` slicing.
4. Arm B is sequential `File.arrayBuffer()` plus
   `crypto.subtle.digest("SHA-256", bytes)`. It may process only one file at a
   time and must release the explicit buffer before opening the next file.
5. An optional arm C may use a dependency-free `wasm32-unknown-unknown`
   implementation with explicit `simd128` across four independent files. It
   must be compared with a four-file scalar-JS control under the identical
   reader schedule, so OPFS concurrency is not mislabelled as a SIMD win.
6. Report the complete `7,325,999,133`-byte path. Also report the exact
   `5,731,837,696`-byte upload subset for OPT-0064 comparability, but never use
   that smaller slice to claim end-to-end initialization savings.
7. Start each arm immediately before opening/reading its first real payload and
   stop only after its final digest comparison and cleanup. Include reads,
   copies, hashing, finalization, synchronization, and release.

## Correctness and safety gates

- Every eligible arm must reproduce all `156` manifest digests exactly, the
  `158` logical records, every byte total, and deterministic repeat results.
- Independently cover NIST SHA-256 vectors and lengths `0`, `1`, `55`, `56`,
  `63`, `64`, `65`, `4 MiB - 1`, `4 MiB`, and `4 MiB + 1`; split updates,
  signed-byte patterns, greater-than-32-bit cumulative length, one-bit
  corruption, short reads, abort, and finalize/reset misuse must fail closed.
- Never mutate an authentic OPFS file for a corruption test. Publish the
  immutable-file proof only after exact size and digest success.
- WebCrypto is eligible only while the largest file remains below `128 MiB`,
  exactly one file is live, and the conservative logical transient bound stays
  below `384 MiB` (three largest-file equivalents). Record memory telemetry
  when the browser exposes it and always retain explicit live-buffer
  accounting. The receipt must disclose that one in-flight WebCrypto digest
  is not internally abortable; cancellation is bounded at a file boundary.
- Preserve main-page and worker heartbeats, raw thermal observations, all
  errors, and cleanup. No GPU device is requested.

## Performance gate

Use balanced complete-inventory arms after at least 30 nominal seconds and
retain every raw sample rather than the fastest run. A candidate is positive
only if:

- median complete authentication wall is at most `8,000 ms`;
- it saves at least `15,000 ms` versus arm A in both paired directions;
- throughput is at least `915,749,892 B/s` over the full physical inventory;
- no read/copy phase or responsiveness regression is hidden by aggregate
  concurrency; and
- all correctness, bounded-memory, abort, and cleanup gates pass.

An isolated pass authorizes only a production-seam integration under this same
ID. The integrated path must then pass a balanced warm-start initialization
comparison with at least `15 s` ready-wall saving, identical proof inventory,
no unrelated stage regression above `2%`, and unchanged lifecycle. Only that
integrated comparison can contribute to an end-to-end product budget.

## Result

The frozen receipt remains a literal non-pass:
`status = failed-or-inconclusive`, `performance.passed = false`. No threshold
is waived or relabelled. The isolated WebCrypto mechanism nevertheless made
the complete authentication path substantially faster under exact, bounded,
all-nominal evidence:

| Frozen aggregate | Result | Gate |
| --- | ---: | ---: |
| Candidate complete-authentication median | `6,700.949999988079 ms` | at most `8,000 ms` — pass |
| Candidate physical throughput | `1,093,277,689.4340403 B/s` | at least `915,749,892 B/s` — pass |
| Forward paired saving | `22,220.94999998808 ms` | at least `15,000 ms` — pass |
| Reverse paired saving | `22,603.849999964237 ms` | at least `15,000 ms` — pass |
| Maximum responsiveness regression | `0.011363639083200816` | at most `0.20` — pass |
| Exact/correct samples | all `8` | required — pass |
| Nominal samples | all `8` | required — pass |
| Maximum read/copy regression | `2.965945018071039` | at most `0.20` — **fail** |

`maximumReadCopyRegression = 2.965945018071039` is the candidate/control
regression fraction, or a `3.965945018071039x` component ratio. It came from
the forward `A1/B1` read/copy walls,
`854.499995470047 -> 3,388.899999976158 ms`. Although WebCrypto moved the much
larger scalar-hash wall into native hashing and reduced the complete wall by
about `22 s`, the frozen OPT-0069 rule independently capped every read/copy
regression at `20%`. Therefore `performanceGate = false` is authoritative.
The positive complete-wall result cannot retrospectively change that rule.

### Run identity and exact inventory

- Core and harness commit:
  `bbc7230d1edbac04b84786eefefbd6d495650b2f`.
- Machine: MacBook Air `Mac15,12`, Apple M3, `8` CPU cores,
  `17,179,869,184` bytes memory; macOS `26.5.2` build `25F84`; Google Chrome
  `151.0.7922.138`.
- Main manifest SHA-256:
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`;
  revision-7 dense manifest SHA-256:
  `d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f`;
  revision-6 VAE manifest SHA-256:
  `94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949`.
- Every arm independently authenticated `158` logical records, `156` unique
  digests, `7,330,447,819` logical bytes, and `7,325,999,133` physical bytes.
  All actual digests matched and every repeat used inventory fingerprint
  `b8fef9059f8d584fd2b17a8d13c21d7e97aa7bdd1b38263a5faa7d3a5e5fb1ce`.
- The secondary OPT-0064 upload slice also reconciled exactly to `102` files
  and `5,731,837,696` bytes, but the complete physical inventory remained the
  decision authority.

All NIST, boundary-length, split-update, signed-byte, greater-than-32-bit
length encoding, one-bit corruption, short-read, pre-abort, finalize, and
post-finalize misuse screens passed. Every authentic cache file remained
immutable. Both owners covered all `156` digests and released their payload
state. No cache/package mutation or GPU device request occurred.

WebCrypto held exactly one payload at a time. Its largest explicit payload was
`121,668,608` bytes and its conservative three-copy transient bound was
`365,005,824` bytes, below the frozen `384 MiB` ceiling. Its in-flight digest
remains internally non-abortable; the accepted contract checks cancellation
before the read, before the digest, and immediately after that one file's
digest completes.

### Exact balanced samples

All values below are milliseconds; no fastest-only subset is selected:

| Arm | Owner | Complete wall | Read/copy | Hash | Upload slice wall |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | scalar stream | `29,054.100000023842` | `854.499995470047` | `28,157.800004959106` | `22,734.900000095367` |
| B1 | WebCrypto | `7,283.400000095367` | `3,388.899999976158` | `3,892.5999999046326` | `5,759.499999880791` |
| B2 | WebCrypto | `6,547.700000047684` | `2,943.5999995470047` | `3,602.0000005960464` | `5,113.700000166893` |
| A2 | scalar stream | `29,234.799999952316` | `907.599999666214` | `28,283.899999260902` | `22,778.00000023842` |
| B3 | WebCrypto | `6,738.899999976158` | `3,107.7999999523163` | `3,629.199999809265` | `5,281.100000023842` |
| A3 | scalar stream | `29,259.5` | `892.8000073432922` | `28,329.899993896484` | `22,844.5` |
| A4 | scalar stream | `29,334.200000047684` | `921.50000166893` | `28,379.599997639656` | `22,858.699999928474` |
| B4 | WebCrypto | `6,663` | `3,061.2999999523163` | `3,599.7000000476837` | `5,198.899999976158` |

### Thermal and persisted evidence

Each accepted arm had its own fresh gate, worker, and through-cleanup trace.
All eight traces contained only level-0 observations. Every trace file equals
the corresponding receipt object, and every authenticated raw-trace digest
equals the SHA-256 of its selected JSONL:

| Arm | Observations | Trace-file SHA-256 | Selected raw JSONL SHA-256 |
| --- | ---: | --- | --- |
| A1 | `172` | `3b697275e7147ff35583320d3aca00f28f16dc81a7b6d5b71d290bde694860f9` | `2ceb2acb4c9bf0eec3bc8ddc1ecee2422e43d6478a913c026ba70e31db685bb1` |
| B1 | `77` | `08782f57a5bff5d5d3116170767943c28f8e2a4a44b1a59c397ff2e426353be6` | `00eb309e075f0fe88e82e2de5d0f80659f844755995230eda0c423505d376ce8` |
| B2 | `77` | `e483a23afd86f498791f574ab752cedd2537d6f613560885bf2c6a4a4b4f88db` | `01fc67ebd13b481ea3d98f67b3d8eeaa100cfc7e99a72676ccc875e29134999c` |
| A2 | `101` | `f7e80a81391f7c13ef3cd3cd505de791cd96c0f1aa00df8c49f8d5a41ab0b4cf` | `7841e1a47d7166e07a5c36a644c99591cd0bd490d92fa188b40d61052980e5db` |
| B3 | `72` | `dbb4e5d127d21abfd15623d266dea6853593f814a0aa0e4211c439b1022d9d66` | `ef078e71b0b76515527753ed0d06d2d69ef551a2832afbc23dbaa07503550446` |
| A3 | `104` | `a93fd12e1d3a62739a9803609975c83f419e796a3687a81dfb44dda9378ec3f3` | `620d060e5d105ce11fe7fa481c8079ba23464a94232bca4152824f36cc16f2b8` |
| A4 | `98` | `2fdc35a0bc525dc5f9aead07f35ff0f8b974d783562955b78a1eac0590719f5a` | `efe9c3fc75b2e4e63e81307757c6659bf9df57fb6b6b0afc3c24b822cd7bd9aa` |
| B4 | `71` | `e01b8b2cca5315626710afb61493018909a55b64895476e649a0b0ee38bf1a95` | `58e2d6c42b9a551143c94558b5d07c2643ff05764cd7a73125c8e52185aec1ea` |

- [Frozen failed/inconclusive receipt](../results/OPT-0069/result.json),
  `320,189` bytes, SHA-256
  `46d4a308741e60dce68a86bbfad3c1b5f17b2ce367898bdc138758044b451dac`.
- [Raw gates, traces, selected JSONL, and indices](../results/OPT-0069/raw/).
  One stale A1 gate was rejected before timed work and remains preserved; it
  is not one of the eight samples.

The result closes OPT-0069 without production/default selection. It does,
however, isolate complete authentication wall as the causal product metric:
the read/copy increase is an internal redistribution that accompanied an
approximately `22 s` complete-wall reduction. OPT-0071 is a new-ID test of
that explicitly declared objective; it does not alter this record's failed
component gate.

## Authority

- [OPT-0064 direct-request capture](OPT-0064-direct-request-warm-start-load-overlap.md)
- `src/model/acquire.ts` current cache verifier
- `src/model/sha256.ts` current scalar SHA-256 owner
- authenticated production manifests:
  `model/files-reference/manifest.json`,
  `model/files-fp16-dit-rev7-oracle/manifest.json`, and
  `model/files-fp16-vae-experimental/manifest.json`

No implementation, browser timing, GPU work, package mutation, production
selection, or speed claim occurred when this experiment was registered.
