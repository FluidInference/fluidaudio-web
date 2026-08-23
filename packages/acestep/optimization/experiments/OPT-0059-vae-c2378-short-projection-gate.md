# OPT-0059 — VAE C2378 short projected-wall gate

## Status

- Evidence: `negative`
- Disposition: `benchmark-only`
- Risk: exact scheduling/window geometry; product waveform identity is still a
  required integration boundary

## First-principles basis

OPT-0035 proved that the stock M3/Chrome runtime can allocate and execute the
two-window C2378 plan within bounded memory, and that its complete stitched
C4500 waveform is raw-bit identical to the C512/overlap64 oracle. It did not
establish performance: the four multi-minute arms drifted so severely that
forward and reverse comparisons changed sign. Its revisit condition explicitly
requires a shorter paired subsystem or interleaved chunk-level screen rather
than another full four-arm C4500 run.

The geometry remains compelling. C512 decodes `5,908` latent frames across
`C448 + 10 * C512 + C340`. A nominal C2378 plan for the same C4500 request
produces two actual C2314 windows and decodes `4,628` frames, or
`0.7833446175x` as much convolution work (`1.276577x` ideal compute speedup).
Under batch64, static planning changes:

| quantity | C512 plan | C2378 plan |
| --- | ---: | ---: |
| graph quanta | `90,675` | `70,994` |
| total command buffers/readbacks | `1,432` | `1,112` |
| requested cooperative idle | `1,431 ms` | `1,111 ms` |
| K7 quanta | `47,204` | `36,982` |
| K1 quanta | `9,450` | `7,374` |
| ConvTranspose quanta | `7,429` | `5,788` |
| Snake quanta | `18,618` | `14,598` |
| Add quanta | `7,974` | `6,252` |

Nearly all quanta retain the same bounded geometry; the plan executes fewer
instances of the same work. The one known shape-specific seam is six final
C128 K1 operations whose C2314 full-shape workgroup counts exceed 65,535 and
therefore use bounded `1 x 256` range dispatches instead of the C512 flat-X
mapping. That seam must be reported separately, but K1 is a small part of the
optimized projection.

OPT-0035 observed `3,758,347,792` peak live bytes with conservative C2378-sized
allocation. Allocating from the actual C2314 maximum is projected at about
`3,661,985,296` bytes, roughly 96 MB lower. Both remain below the frozen 4 GB
gate and the adapter's ordinary advertised per-buffer limits.

## Frozen experiment direction

- Wait for the authenticated revision-7 OPT-0052/OPT-0057 VAE package/profile.
  Compare window geometry only after the intended final K1/K7/ConvTranspose
  owners are fixed; do not benchmark a soon-to-be-replaced family stack.
- Create one package owner and one C2314-capable allocation phase. Build C340,
  C448, C512, and C2314 dispatch sets over the same weights/workspaces. No
  competing backend, duplicate package, browser repack, or simultaneous arm
  residency is allowed.
- Allocate to the largest actual planned window, C2314, rather than the unused
  nominal C2378 ceiling, while preserving the exact two-window C4500 cover:
  latent windows `[0,2314)` and `[2186,4500)`, cores `[0,2250)` and
  `[2250,4500)`, overlap 64.
- Warm each exact shape outside timing. Preserve the selected batch64 FIFO,
  one outstanding command buffer, cancellation/progress bounds, arithmetic,
  overlap, stitch, normalization, output writer, and all kernel/package
  identities.

## Short authoritative gate

After one nominal level-0 thermal check, use balanced paired shape timings
rather than four complete C4500 executions:

1. Main order: `C512, C2314, C2314, C512`.
2. Edge correction: `C340, C448, C448, C340`.
3. Record complete decoder submit-through-drain and readback/map wall, every
   family/mixed bucket, graph/command/drain counts, requested idle, peak live
   bytes, and the six C128 K1 mapping changes.
4. Form the exact plan projections from the paired measurements:
   - control: `10 * C512 + C448 + C340 + 11 ms` between-window idle;
   - candidate: `2 * C2314 + 1 ms` between-window idle.

Require both directional projected comparisons and their aggregate to reach
`1.15x`. Require C2314 decoder wall normalized per decoded frame to be no more
than `1.10x` the C512 normalized wall, and no route-specific family regression
above `1.15x` without a reconciled geometric explanation. Peak live GPU bytes
must stay below `4,000,000,000`; device loss, allocation failure, dispatch
overflow, missing output, non-finite data, topology mismatch, cancellation
failure, or nonzero live resources invalidates the screen.

Correctness must authenticate each warmed/timed shape against its same-profile
oracle and preserve deterministic complete writes, canaries, and seams. A
positive short projection does not replace the already-required sequential
C4500 correctness/lifecycle pass or OPT-0044 listening approval for K4. It does
authorize integrating C2378 into that joint long gate. A negative or mixed
short screen retains C512 without repeating OPT-0035's order-confounded full
sequence.

Planning from the optimized K7/K1/ConvTranspose families suggests about
`6–9 s` of C4500 VAE saving (`1.18–1.27x`), not an observed performance claim.
The mechanism can tolerate roughly 11% worse normalized C2314 throughput and
still clear its wall gate; measurement, not frame-count arithmetic, decides it.

## Result

The literal frozen result is `status: "negative"`, decision
`negative-retain-c512-production-windowing`, and
`performanceGatePassed: false`. The production default did not change and
C2378 was not selected or integrated.

This formal negative result contains strongly positive geometry evidence. The
aggregate exact-plan projection improved from `38,433.09999984503 ms` to
`29,448 ms`, a `1.3051174952405946x` speedup and
`8,985.099999845028 ms` saving. Both directional projections exceeded
`1.30x`, normalized C2314 decoder throughput was about `0.987x` C512, all
correctness and lifecycle checks passed, and maximum live GPU storage was
`3,667,109,696` bytes. The frozen route-family gate nevertheless failed. Its
batch-64 family buckets changed composition sharply between C512 and C2314,
so several bucket ratios did not compare the same share of route work. That
attribution limitation cannot be waived after the fact under this experiment's
declared rule.

### Authority and environment

The canonical receipt is [result.json](../results/OPT-0059/result.json),
`202,701` bytes with trailing newline and SHA-256
`18ea3b278675ab9ac40e77cab005af3cdcb7e4458c169dee9204de56329cdf3a`.
The direct browser download was `202,700` bytes with SHA-256
`bfe708bdded33348993e7a6353f7ddb24bb88d385b153c9d5ac568366f83c36e`;
parsing proved the JSON values identical, with the terminal newline as the only
byte difference. The persisted newline-terminated receipt is authoritative.

- Core and harness commit: `3edc51fe9af454e96c7fb6286d250fc6763380f7`.
- Machine: `Mac15,12`, 10 GPU cores, `17,179,869,184` bytes memory.
- OS/browser: macOS `26.5.2` build `25F84`, Google Chrome
  `151.0.7922.138`, Apple `metal-3`, fixed 32-lane subgroups.
- Runtime: `opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1`, kernel set
  `opt-0066-vae-fp16-fixed32-dual-k4-quality-kernel-set-v1`, precision-map
  SHA-256 `4815ec86311e401a9bf8cec3f4474d479ef3ac28c925f4824fed4c9ae3b8bc60`.
- Package: authenticated converter revision 7, manifest SHA-256
  `36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7`,
  145 tensor records, seven weight files, and `168,791,552` resident bytes.
- Thermal launch: one level-0 `notifyutil` observation after `57,946 ms`,
  followed by a `0 ms` recorded launch delay. This is the declared one-check
  protocol, not a continuous through-cleanup thermal trace.

### Exactness, topology, and lifecycle

One package owner, one C2314-capable guarded allocation, and four exact
C340/C448/C512/C2314 dispatch sets were retained. The C4500 control/candidate
plans were exactly `90,675 / 70,994` graph quanta and `1,432 / 1,112` total
command buffers, with the frozen two-window C2314 cover and all six C128 K1
2D-mapping changes present.

Each shape ran untimed in oracle/profile/profile-repeat order. Per comparison,
all `13,877,760` U32 words (`27,755,520` U16 words) across the four shapes were
exact, with zero mismatches. All eight timed executions then reproduced their
prepared same-profile hash:

| shape | exact output SHA-256 |
| --- | --- |
| C340 | `fcc768dd9b0901c115b5f7b1a77e2e80f547afb00aff0c7d6a4d59558b67f118` |
| C448 | `5a15aa98a8a232b25bddf30e74c034b830957c42a27fe8063da709faa2525c7d` |
| C512 | `84a908b8aa9cc6656b967226de69f51046c109065d38ec07599d8d2165c93564` |
| C2314 | `2a16f0fc4b07661e21628e0b5574c2feeab3882ecef169da52a671c937f36f0c` |

All active writes were complete and finite, inactive output tails remained
untouched, all guarded spans retained canary SHA-256
`3dee710588de5d1031ddeb00a0150cbe5ed5f8851eb9233bb0b04edfbc09d138`,
and no WebGPU runtime event occurred. Cleanup drained the queue and destroyed
all `19 / 19` buffers exactly once, with `40 / 40` map/unmap operations, zero
live or mapped buffers, and an idempotent `84.90000009536743 ms` teardown.

### Projection and the formal family failure

| direction | control projection | candidate projection | saving | speedup | normalized C2314/C512 decoder | literal pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| forward | `38,425.199999690056 ms` | `29,531.400000095367 ms` | `8,893.799999594689 ms` | `1.3011641845481747x` | `0.9865455926985542` | no |
| reverse | `38,441 ms` | `29,364.599999904633 ms` | `9,076.400000095367 ms` | `1.3090932619591225x` | `0.9872662470621996` | no |
| aggregate medians | `38,433.09999984503 ms` | `29,448 ms` | `8,985.099999845028 ms` | `1.3051174952405946x` | `0.9869047401272858` | no |

The projection-speed and normalized-decoder sub-gates passed in both
directions. The complete direction results remained false because the frozen
family rule also required every unexplained normalized route ratio to stay at
or below `1.15`:

| aggregate bucket | C512 batches / quanta | C2314 batches / quanta | normalized ratio | receipt pass |
| --- | ---: | ---: | ---: | --- |
| K7 Conv1D | `52 / 3,328` | `273 / 17,472` | `1.154648180312212` | no |
| K1 Conv1D | `3 / 192` | `45 / 2,880` | `3.7566834580119512` | yes, declared six-route mapping explanation |
| ConvTranspose1D | `6 / 384` | `40 / 2,560` | `1.7448433196682396` | no |
| Snake | `4 / 256` | `83 / 5,312` | `4.396545369917946` | no |
| Add | `3 / 192` | `36 / 2,304` | `2.5324312912364637` | no |
| mixed | `55 / 3,503` | `78 / 4,970` | `0.4020788901264392` | yes |

The batch/quanta counts show the attribution problem directly: pure-family
batches increase at very different rates while the mixed bucket absorbs a
different fraction of each shape. For example, decoded frames scale by about
`4.52x`, but classified pure Snake quanta scale `20.75x` and pure K1 quanta
scale `15x`. These ratios therefore mix route cost with batch-boundary
composition. Forward failed K7, ConvTranspose, Snake, and Add; reverse failed
ConvTranspose, Snake, and Add; aggregate failed K7, ConvTranspose, Snake, and
Add. Only K1 carried the predeclared six-route geometric exception. A future
per-quantum or composition-matched attribution screen would require a new
experiment; it cannot retroactively convert OPT-0059 into a pass.

### Rejected setup and correctness preflights

Three earlier attempts were rejected before the timing phase. They contributed
zero timing samples and are not performance evidence:

1. Package binding authentication incorrectly used the C2314 decoder graph.
   The resolver failed closed with `INVALID_VAE_RUNTIME_PROFILE` because
   package identity is authenticated only against canonical B256; resolved
   bindings are frame-neutral and exact C340/C448/C512/C2314 geometry belongs
   at dispatch-set construction.
2. The first guarded-allocation attempt requested `maxBufferSize` at the
   unguarded logical workspace size. The C340 dispatch-set check rejected
   `workspace 0 violates device storage limits or alignment`: the C2314
   physical workspace is `1,137,377,792` bytes, while its shader-visible
   logical binding is `1,137,377,280` bytes between two 256-byte guards.
3. The proposed OPT-0063 joint K1/Add/successor-Snake path reached C340
   correctness but was deterministic and wholly wrong versus the base:
   `1,305,600 / 1,305,600` U32 mismatches and
   `2,609,625 / 2,611,200` U16 mismatches, first mismatch index `0`.
   Actual slot flow is K1 `S0 -> S1`, Add `S2 + S1 -> S0`, then successor
   Snake `S0 -> S2`. Redirecting the rounded Add store to S1 leaves the live
   S0 next-residual skip stale; writing S0 from the K1 dispatch instead creates
   a cross-workgroup read/write hazard because output-column workgroups still
   read the complete K1 input row. The invalid joint seam was removed and the
   standalone OPT-0063 primitive remains benchmark-only. An Add/Snake-only
   mechanism would require a new experiment ID.

### Decision

OPT-0059 closes negative under its literal frozen gate. The correctness,
lifecycle, memory, and overall projected-wall evidence strongly support the
underlying large-window geometry, but the declared route-family attribution
condition did not pass. Retain C512 production windowing; do not begin the
sequential C4500 integration/listening gate from this result, select C2378, or
change any production default.
