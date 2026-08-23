# OPT-0064 — Direct-request warm-start, load, and upload overlap

## Status

- Evidence: `inconclusive` overall; positive capture-only attribution, failed
  accepted-output identity gate
- Disposition: `benchmark-only`; no upload or overlap optimization authorized
- Date: 2026-08-15
- Risk: exact scheduling, memory residency, authentication, and cancellation
  behavior; no model-math change

## First-principles basis

The remaining product gap cannot be assigned entirely to GPU kernels. One
measured direct M2250 arm attributed `1,428.1 ms` to text, `2,462.2 ms` to the
conditioner, `2,538.4 ms` to DiT load, and about `60 ms` to prepare/release: a
`6,488.4 ms` non-denoise budget. That whole amount is not removable overhead.
The current pipeline serializes conditioner release, DiT phase upload/backend
construction, graph execution, DiT destruction, and then VAE acquisition,
upload, and construction.

The VAE side is smaller: the long profiler measured `711.2 ms` acquisition,
`79.7 ms` phase upload, and `165.5 ms` backend construction. WAV finalization
was `25.2 ms` at 12 seconds and `85.2 ms` at 30 seconds, giving only about
`0.38-0.51 s` of linear 180-second planning context. Output finalization is not
the missing six seconds.

This experiment first measures request-scoped cache read/hash, upload, shader
and pipeline compilation, graph construction, phase release, normalization,
and durable WAV commit. It then screens only dependency-independent overlap
and direct upload mechanisms. The realistic planning range is `1-3 s`; the
measured `6.4884 s` non-denoise wall is an attribution ceiling, not a saving
claim.

## Frozen direction

- Start all candidate work only after the user presses Generate, and include it
  in Generate-to-WAV wall. Benchmark warmup may not hide request work.
- Add capture-only timings for cache open/read/hash, per-file copy/upload/drain
  and cooperative gap, shader/pipeline compilation, bind-group/graph creation,
  execution, release, raw scan/write, normalization, WAV encoding, and durable
  commit. Capture must not change ordering or math.
- Permit overlap only when dependencies and memory ownership prove it safe:
  pipeline-only compilation during conditioner or DiT work; VAE cached-file
  authentication during DiT only if DiT wall does not regress; and reuse only
  on the same device. Never load VAE GPU weights before the DiT has drained and
  been destroyed, and never retain simultaneous model-weight owners.
- Separately screen mapped-at-creation final-buffer upload against the current
  chunked `queue.writeBuffer` path on real package shards. Preserve the existing
  authentication contract: reuse the digest proof only for the exact immutable
  `File` object already authenticated by acquisition, with matching size and
  digest; incrementally hash every ordinary `Blob` while copying. Do not
  redundantly rehash a proven `File` merely because its destination is mapped.
  Publish only after exact size/proof/error-scope success and destroy/remove
  every failed candidate.
- Map and fill only one physical shard at a time. The largest current shard is
  `121,668,608` bytes; the CPU-visible bound is one mapped shard plus one
  at-most-`4 MiB` stream chunk, never a phase-sized mirror and never an early
  buffer for a future heavyweight phase.

## Gates

1. Reconcile every captured stage to complete direct Generate-to-WAV wall and
   prove bounded JS storage, exact package/output hashes, unchanged command and
   drain counts, cancellation, device-loss propagation, and zero live
   resources after success and failure.
2. Use balanced baseline/candidate direct requests with identical authenticated
   inputs and nominal thermal starts. Retain all raw stage and whole-request
   samples. No component may be omitted because it overlaps another interval.
3. Require no DiT, conditioner, or VAE execution regression above `2%`, exact
   latent/waveform/WAV identity, and at least `1,000 ms` complete
   Generate-to-WAV saving in both paired directions before escalation.
4. A positive result may join the final optimized product gate. It does not
   authorize simultaneous model residency, background pre-Generate work, a
   package-authentication shortcut, model-math changes, or an under-one-minute
   claim.

No implementation or GPU work occurred when this experiment was registered.

## Pre-implementation audit clarification

The direct warm-cache path uploads `5,731,837,696` bytes across `102` files:
`1,191,553,024` text bytes, `1,220,575,232` conditioner/constant bytes,
`3,150,917,888` DiT bytes, and `168,791,552` VAE bytes. The current uploader
performs `143` queue drains and `41` explicit queue-empty gaps. Mechanically
scaling OPT-0023's measured `79.7 ms` VAE upload gives about `2.706 s` of
current full-request upload attribution, so a mapped path must remove roughly
`37%` of that projection to save one second.

The same OPT-0023 receipt measured `711.2 ms` to authenticate the VAE files.
Redundantly applying that hash rate to all `5.73 GB` would cost about `24.15 s`
and structurally erase the candidate. This is why the exact immutable-`File`
proof reuse above is a correctness requirement rather than a benchmark
shortcut. Capture-only attribution remains the first implementation
checkpoint; no uploader or scheduling mutation is justified before it
separates OPFS streaming, WebGPU copy calls, drains/gaps, compilation, and
graph construction.

## Capture-only browser result

The explicit-UI, warm-cache Chrome/M3 run completed initialization, the full
unchanged 12-second direct request, output hashing, result release, backend
disposal, and the complete compact capture reconciliation. The persisted
receipt nevertheless has literal `status: "failed"`: its only failure is
`OPT-0064 direct WAV identity changed`. This is not converted into a pass.

The initialization progress reporter reached `ready` in `29,310.0 ms`. The
generation reporter reached `done` in `16,437.100000023842 ms`. These are
observed phase walls, not a balanced performance comparison or a complete
Generate-to-WAV claim. The run recorded `459` capture events, `156` cache
authentication events, `158` exact immutable-`File` proof-reuse events, three
acquisition plans, no diagnostic or fatal-diagnostic codes, and clean result
release and backend disposal.

Initialization is the largest observed phase, but the compact failed receipt
retains only the `156` cache-authentication event count, not the individual
authentication event walls or their sum. The full event list was deliberately
excluded from failure evidence. Consequently the summed cache-authentication
wall is not recoverable from the persisted result or thermal trace, and the
unattributed remainder inside the `29,310.0 ms` initialization wall remains
unresolved. It is not assigned to hashing, OPFS, device setup, or another
mechanism by subtraction. Any decision about hash cost requires a separate
bounded benchmark; this result authorizes no estimate or retry.

### Upload attribution

All `102` files and `5,731,837,696` bytes reconciled to the frozen inventory,
with `143` queue drains and `41` explicit queue-empty gaps. Summed per-file
upload wall was `3,331.100000143051 ms`:

| Component | Captured wall |
| --- | ---: |
| OPFS stream reads | `2,381.3000046014786 ms` |
| `queue.writeBuffer` calls | `389.4999964237213 ms` |
| Queue drains | `149.40000021457672 ms` |
| Queue-empty gaps | `52.80000019073486 ms` |
| Buffer creation | `0.9000002145767212 ms` |
| Error scopes | `8.099999904632568 ms` |

No incremental upload hash or owned bulk copy occurred: acquisition had
already authenticated each exact immutable `File`, and upload reused only that
object-bound proof. The per-phase file-wall sums were `711.2 ms` for text,
`702.0 ms` for conditioner/constants, `1,814.799999833107 ms` for DiT, and
`103.10000026226044 ms` for VAE.

### Construction, execution, and finalization

| Phase | Construction | Execution |
| --- | ---: | ---: |
| Conditioning | `2,356.8000000715256 ms` | `2,266.100000023842 ms` |
| DiT | `141.89999997615814 ms` | `6,724.5 ms` |
| VAE | `112.69999992847443 ms` | `3,722.7000000476837 ms` |

Construction includes the unchanged backend's shader modules, pipelines,
bind groups, buffers, and graph setup. The bounded normalize/encode/durable-WAV
commit took `27.799999952316284 ms`, of which WAV read/scale/write was
`21.200000047683716 ms`. VAE raw streaming is already included in the VAE
execution wall and is not added again.

### Controlling quality failure

The expected `4,608,044`-byte WAV authority remains owner-approved Stage 1
packed-BF16/FP32 Candidate A, SHA-256
`d085b6907c9872667412d6dcecfeee47b76c8038eb2bfbec615931b2d7365477`.
The configured benchmark VAE was the still-unapproved OPT-0028 FP16 profile,
manifest
`94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949`.
It produced the correct byte length but SHA-256
`409b7157ac428910fae17776b1abbd9b42db7509984bcc0aac41871f95152ec2`.
That digest equals the earlier FP16 WAV recorded by OPT-0015, but it does not
inherit the owner-approved FP32 listening authority and does not promote the
current OPT-0028 profile. OPT-0011's fresh instrumental and vocal waveform and
listening gates remain binding.

Therefore the capture-only diagnostic is positive, but Gate 1 and the overall
quality gate failed. No mapped-at-creation path, load overlap, authentication
shortcut, production selector, performance saving, or under-one-minute claim
was attempted or authorized. A future candidate requires a separately valid
current-profile quality authority, followed by the unchanged balanced
performance, execution-regression, lifecycle, and exact-output gates above.

## Persisted evidence

- [Failed overall receipt with positive compact capture](../results/OPT-0064/result.json),
  SHA-256
  `b3be2509a3b81522229cbb69f743aa4d65662f907604f085c5f2269151198498`.
- [Selected through-cleanup thermal trace](../results/OPT-0064/raw/trace.json),
  file SHA-256
  `7de13a65826e4596b73d31ba1a3a617cc4855053107428b849e98eeb721b3144`.
  It contains `137` level-0 observations, zero non-nominal observations, and a
  `918 ms` maximum poll gap.
- [Selected raw thermal JSONL](../results/OPT-0064/raw/selected.jsonl), SHA-256
  `7ecfbbb72a6c51dd193571aea507a8bdae9b47ee5448f4cae44dc8e76a424fcf`;
  the same digest is authenticated inside the parsed trace as
  `rawTraceSha256`.
