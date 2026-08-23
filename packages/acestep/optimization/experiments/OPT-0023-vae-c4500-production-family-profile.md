# OPT-0023 — current-production C4500 VAE family profiler

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Date: 2026-08-15
- Author/agent: Codex
- Risk class: `exact`, observational/capture-only
- Allocation baseline: pushed `main` commit
  `dc08f76ce44a6a46edbd4b60c9b74e6a7b019363`
- Profiler/harness authority:
  `02230725e460323de7e82ebed00177ec2103ea55`

## Hypothesis and why this is next

One authenticated package-native C4500 production VAE sequence can replace the
current C300-linear extrapolation with exact family and mixed-batch wall
attribution, exact scheduling/readback counts, and a complete-sequence wall.
That measurement should show whether any current VAE family or combined
non-family residual has a concrete mechanism and attainable floor worth at
least `10,000 ms` of absolute saving before another optimization is allocated.

The latest full product took `568.618 s`, of which its then-current VAE took
`431.377 s`. That run already used the C512/64 geometry and FP16 fixed32-K7
path; it predates the integrated OPT-0015 congruent ConvTranspose kernel, not
those earlier changes. It also has no current family attribution. The only
post-OPT-0015 evidence is a 12-second direct C300 profile: complete VAE
`8,054.5 ms`, decoder submit-through-drain `7,265.8 ms`, K7
`3,019.8 ms`, ConvTranspose `2,002.0 ms`, K1 `1,177.3 ms`, mixed
`779.9 ms`, Snake `215.0 ms`, and Add `71.8 ms`. Multiplying that VAE wall by
`5,908 / 300` gives `158.620 s`, but it does not model the actual twelve-window
sequence, ten repeated C512 windows, edge geometries, mapping, OPFS writes,
between-window idles, thermal evolution, or full-sequence residual. It is a
planning estimate, not a measured current long VAE.

OPT-0022 was exact but negative at its primitive stop rule, so no new
ConvTranspose package or production profile exists to measure. OPT-0015 remains
the current owner. Another kernel guess or 180-second product would therefore
precede the missing subsystem evidence. OPT-0023 is localization, not an
optimization candidate.

## Frozen production authority

Fail closed unless the run authenticates all of the following:

- ACE source `6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0` and ACE main-model
  snapshot `19671f406d603126926c1b7e2adc169acbcade22`;
- converter revision `5` FP16-VAE manifest, exactly `714,687` bytes with
  SHA-256
  `5644bcca87678b4f654b9541459355a73ef136c6bb601aa783b6f50fe2f6dba3`;
- exactly `145` unsharded native-layout FP16 VAE tensor records in seven
  authenticated weight files, totaling `84,395,776` elements and
  `168,791,552` resident bytes;
- runtime profile
  `opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1`, kernel set
  `opt-0015-vae-fp16-fixed32-k7-congruent-transpose-kernel-set-v1`, and
  precision-map SHA-256
  `4bd14663b0504e3b890f781e4d01dff62c8dcdc7f87a285a578e35779cd6bc85`;
- integrated OPT-0015 production checkpoint
  `36608b857827b2b1d31ac91bf5cca9639fb0b9ed`, with no OPT-0022 package,
  layout, or selector and no production import or selection of its retained
  benchmark-only kernel; and
- actual browser, OS, Mac model, memory, adapter features and limits, fixed
  subgroup minimum/maximum `32`, `shader-f16`, and `subgroups` in the receipt.

At registration, the reused production sources have these SHA-256 identities:

| Source | SHA-256 |
| --- | --- |
| `src/webgpu/vae-fp16-backend.ts` | `d5440efdc9be32a72613988859dac113daca659023276bc08220edf1135e719d` |
| `src/webgpu/vae-fp16-decoder.ts` | `06f76b31b122c3e794142dc9d4058d31ff4760fbcb6b85465cc33671616d8873` |
| `src/webgpu/vae-fp16-profile.ts` | `eeb063ddada1027f6a3ebd352997500fc5dd1e8bc6d7b61800c8f55b9f33baf9` |
| `src/webgpu/vae-chunks.ts` | `23bbc8e6e7e8b1978075ee64bdd72ee3338058aa0df83d0293a577e0c6dffc22` |
| `src/runtime/webgpu-pipeline.ts` | `2c25b71f55c5e0c2d59e9f4794858cb9615953b07063ffc36690b8465cdbd88a` |

The eventual browser harness, static contract, HTML entry, raw receipt, thermal
trace, and implementation checkpoint must add their own hashes. A source or
identity mismatch is a new experiment, not an OPT-0023 result.

## Deterministic VAE-only input

Use batch `1`, `4,500` latent frames, and `64` channels: exactly `288,000`
FP32 elements / `1,152,000` little-endian bytes. Reuse the exported
`createAceOpt0011LatentFixture(4_500)` authority with generator
`xorshift32-13-17-5-high24-symmetric-f32-v1` and seed `0x00110512`:

1. apply the unsigned xorshift steps `state ^= state << 13`,
   `state ^= state >>> 17`, then `state ^= state << 5`;
2. derive `(state >>> 8) / 8_388_608 - 1`; and
3. store `fround(value)` as little-endian FP32.

The complete latent is finite and nonzero and has SHA-256
`d4e09d07be457583ff8ed4bf420f2ae4a1e822b4f7d6e8a71c300e53123c5971`.
The harness must call the exported fixture generator and authenticate its bytes
before loading the VAE. This is a deterministic OPT-0011 correctness input,
not a DiT output or a musical-quality fixture.

Load only the VAE heavyweight phase. Do not load or run the planner, text
encoder, conditioner, semantic model, DiT, sampler/DCW, or a full product. Stop
after the raw stitched output, receipt, and normal VAE/device cleanup. Do not
normalize, encode a WAV, or listen under this ID.

## Exact C4500 window geometry

Use the current C512/64 production chunker: chunk `512`, overlap `64`, stride
`384`, hop length `1,920`, stereo `48,000 Hz`. It must produce exactly twelve
ordered windows totaling `5,908` decoded latent-window frames:
`C448 + 10 x C512 + C340`.

| Window | Core latent `[start,end)` | Decode latent `[start,end)` | C | Discard latent prefix/suffix | Output audio start | Output audio frames | Decoded audio frames |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `[0,384)` | `[0,448)` | 448 | `0 / 64` | 0 | 737,280 | 860,160 |
| 1 | `[384,768)` | `[320,832)` | 512 | `64 / 64` | 737,280 | 737,280 | 983,040 |
| 2 | `[768,1152)` | `[704,1216)` | 512 | `64 / 64` | 1,474,560 | 737,280 | 983,040 |
| 3 | `[1152,1536)` | `[1088,1600)` | 512 | `64 / 64` | 2,211,840 | 737,280 | 983,040 |
| 4 | `[1536,1920)` | `[1472,1984)` | 512 | `64 / 64` | 2,949,120 | 737,280 | 983,040 |
| 5 | `[1920,2304)` | `[1856,2368)` | 512 | `64 / 64` | 3,686,400 | 737,280 | 983,040 |
| 6 | `[2304,2688)` | `[2240,2752)` | 512 | `64 / 64` | 4,423,680 | 737,280 | 983,040 |
| 7 | `[2688,3072)` | `[2624,3136)` | 512 | `64 / 64` | 5,160,960 | 737,280 | 983,040 |
| 8 | `[3072,3456)` | `[3008,3520)` | 512 | `64 / 64` | 5,898,240 | 737,280 | 983,040 |
| 9 | `[3456,3840)` | `[3392,3904)` | 512 | `64 / 64` | 6,635,520 | 737,280 | 983,040 |
| 10 | `[3840,4224)` | `[3776,4288)` | 512 | `64 / 64` | 7,372,800 | 737,280 | 983,040 |
| 11 | `[4224,4500)` | `[4160,4500)` | 340 | `64 / 0` | 8,110,080 | 529,920 | 652,800 |

Each latent discard frame is exactly `1,920` audio frames. The stitched output
must cover audio frames `[0,8,640,000)` exactly once: `17,280,000` interleaved
FP32 stereo elements and `69,120,000` raw OPFS bytes. The twelve detached full
window readbacks total `90,746,880` bytes before overlap discard:

| Shape | Multiplicity | Detached FP32 elements each | Detached bytes each |
| --- | ---: | ---: | ---: |
| C340 | 1 | 1,305,600 | 5,222,400 |
| C448 | 1 | 1,720,320 | 6,881,280 |
| C512 | 10 | 1,966,080 | 7,864,320 |
| **Total** | **12** | **22,686,720** | **90,746,880** |

## Exact graph, family, and scheduling accounting

Every window uses the same geometry-neutral `88`-operation / `145`-tensor
skeleton. One FP32-to-FP16 ingress quantum precedes each decoder graph. Every
sequence quantum owns one dispatch. Batch eight is unchanged. Each decoder
command buffer is followed by its matching drain and the production
one-millisecond queue-empty interval; each window's final readback command
buffer is separately submitted and drained without a following internal idle.

| Shape | Multiplicity | Operations each | Graph quanta each | Ingress each | Sequence quanta / dispatches each | Decoder CBs each | Readback CBs each | Submissions / drains each |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C340 | 1 | 88 | 5,241 | 1 | 5,242 | 656 | 1 | 657 |
| C448 | 1 | 88 | 6,894 | 1 | 6,895 | 862 | 1 | 863 |
| C512 | 10 | 88 | 7,854 | 1 | 7,855 | 982 | 1 | 983 |
| **C4500 sequence** | **12** | **1,056** | **90,675** | **12** | **90,687** | **11,338** | **12** | **11,350** |

The exact graph-family quantum inventory is:

| Shape / aggregate | K7 Conv1D | K1 Conv1D | ConvTranspose1D | Snake | Add | Total graph quanta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C340 | 2,725 | 546 | 429 | 1,079 | 462 | 5,241 |
| C448 | 3,579 | 714 | 560 | 1,429 | 612 | 6,894 |
| C512 each | 4,090 | 819 | 644 | 1,611 | 690 | 7,854 |
| **C448 + 10xC512 + C340** | **47,204** | **9,450** | **7,429** | **18,618** | **7,974** | **90,675** |

The existing profiler classifies a batch as homogeneous only when all of its
up-to-eight ordered quanta belong to the same family. It keeps ingress and
operation-boundary crossings in an explicit mixed bucket and never prorates
their wall. Freeze the complete batch-eight classification:

| Shape | Pure K7 batches/quanta | Pure K1 batches/quanta | Pure transpose batches/quanta | Pure Snake batches/quanta | Pure Add batches/quanta | Mixed incl. ingress batches/quanta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C340 | `327 / 2,610` | `54 / 432` | `50 / 400` | `106 / 848` | `45 / 360` | `74 / 592` |
| C448 | `436 / 3,487` | `75 / 600` | `66 / 528` | `153 / 1,224` | `63 / 504` | `69 / 552` |
| C512 each | `500 / 3,999` | `89 / 712` | `78 / 624` | `170 / 1,360` | `74 / 592` | `71 / 568` |
| **C4500 sequence** | **`5,763 / 46,087`** | **`1,019 / 8,152`** | **`896 / 7,168`** | **`1,959 / 15,672`** | **`848 / 6,784`** | **`853 / 6,824`** |

The five homogeneous families total `10,485` batches / `83,863` quanta.
Adding mixed gives all `11,338` decoder batches / `90,687` sequence quanta.
There are exactly `11,338 ms` of requested internal queue-empty idle and
`11 ms` of requested between-window idle, for `11,349 ms` requested total.
These counts and requests are topology facts, not claims about observed timer
wall or GPU occupancy.

The full sequence is mandatory. Three separately timed C340/C448/C512 shape
representatives weighted `1/1/10` would miss persistent resource reuse, ten
successive C512 heat/load effects, mapping and detachment order, the eleven
real between-window intervals, overlap scan and discard, OPFS sink behavior,
and sequence residual. Per-window timings remain in the receipt so drift is
visible, but the authoritative VAE wall is the one ordered twelve-window run.

## Capture-only implementation boundary

Do not modify any file under `src/`. Reuse the production
`AceOpt0011Fp16VaeChunkGpuBackend` and its existing `onFamilyProfile` callback.
That callback already records `performance.now()` submit-through-matching-drain
wall for each decoder command buffer, excludes each readback command buffer,
and emits one bounded aggregate after each successfully decoded window.

The browser-only harness may:

- authenticate and acquire the VAE package;
- instantiate the unchanged production backend with the exact OPT-0015 profile
  and a callback that retains exactly twelve bounded family aggregates;
- wrap `decodeWindow` only to timestamp its outer wall, then pass that wrapper
  to the unchanged `streamAceVaeRawChunks` and production raw OPFS sink;
- retain bounded progress, heartbeat, thermal, resource, and aggregate timing
  metadata; and
- hash and serialize only after the attributed interval.

It may not alter WGSL, kernel IDs, arithmetic or precision, weights, package
layout or bindings, profile/default/capability selection, decoder plan,
logical ranges, pass or quantum order, batch size, command-buffer composition,
submission/drain/idle placement, mapping, overlap discard, OPFS writes,
cancellation semantics, or resource lifetime. No per-command logging,
postMessage, serialization, GPU query, or new GPU buffer is allowed inside the
timed decoder sequence. Callback failure or any missing aggregate invalidates
the profile even though production deliberately treats observation as
non-fatal.

Focused contracts must independently derive and freeze every table above,
prove disabled production behavior is unchanged, bind all relevant source and
harness hashes, and show that enabled capture adds observation plumbing only.

## Memory and preparation gates

At timed start, after package upload and temporary preparation resources have
drained and been destroyed, the tracked steady GPU set must be exactly `17`
buffers / `944,808,752` bytes:

| Timed resident allocation | Buffers | Bytes |
| --- | ---: | ---: |
| Seven authenticated VAE weight files | 7 | 168,791,552 |
| FP32 staging input | 1 | 131,072 |
| FP16 decoder input | 1 | 65,536 |
| Three FP16 workspaces | 3 | 754,974,720 |
| FP32 output + MAP_READ readback | 2 | 15,728,640 |
| C340, C448, and C512 dynamic controls | 3 | 5,117,232 |
| **Total** | **17** | **944,808,752** |

The controls must reconcile as C340 `1,341,712`, C448 `1,764,880`, and C512
`2,010,640` bytes. Timed-sequence tracked high-water must remain
`944,808,752` bytes; report any larger preparation/upload high-water
separately. Bounded CPU live storage is at most the `1,152,000`-byte latent
snapshot plus one `7,864,320`-byte maximum returned window, or `9,016,320`
bytes, excluding the file-backed raw sink and small receipt aggregates. No
second full decoded window, full-song waveform array, or duplicate VAE package
may be retained. The raw OPFS file must finish at exactly `69,120,000` bytes.

Compile every selected production pipeline and run exactly one complete untimed
C512 warmup before the thermal gate. Do not run a full-sequence warmup and do
not warm C340 or C448 separately. Use the exact C512 dispatch and its matching
deterministic latent slice; require the `1,966,080`-element / `7,864,320`-byte
output to have its exact length and be finite and nonzero. Do not add a prefill
dispatch or queue write. Record manifest/package acquisition and hashing,
phase upload, one combined backend-create wall, and warmup outside timing. The
public backend `create()` boundary combines shared-buffer allocation,
pipeline/runtime creation, dynamic-control upload, dispatch construction, and
the latent snapshot; OPT-0023 must not estimate or fabricate component walls
inside that combined measurement.
Validate and discard the warmup's family aggregate before arming the collector
that requires exactly twelve timed aggregates. The warmup validates readiness;
it is not an output oracle for the different twelve-window timed sequence.

## Corrected one-run thermal and timing protocol

Use one absolute-cadence external thermal logger at `1,000 ms`, reading
`notifyutil -g com.apple.system.thermalpressurelevel`. Start it before the C512
warmup and retain it continuously through timed output validation and cleanup.
After warmup, require one fresh all-nominal pre-gate with:

- duration at least `30,000 ms`;
- observation count at least `floor(durationMilliseconds / 1,000) + 1`;
- maximum adjacent poll gap at most `1,250 ms`;
- zero non-nominal or missing observations; and
- timed launch no more than `5,000 ms` after the gate-completion timestamp.

The receipt must distinguish the full logger trace-start timestamp/count from
the later fresh post-warmup gate-start timestamp/index/count. Gate duration,
observation consistency, nominality, and launch delay are computed only from
that post-warmup subrange; full-trace coverage is reported separately through
cleanup.

This explicitly closes OPT-0022's drifting-logger observation-count failure.
Preflight rejection before the first timed dispatch is setup evidence, not a
timing sample. Once the first timed window dispatches, retain that one run and
do not retry unchanged work for a cooler trace or better number. Continue to
record and disclose every later thermal transition. Because this is a
single-arm ordered sequence rather than a balanced A/B, later pressure and
per-window drift materially limit absolute comparability and must be part of
the decision, but they do not authorize an unchanged rerun.

Create and open the raw OPFS sink before timing. The authoritative complete VAE
wall begins immediately before the unchanged ordered
`streamAceVaeRawChunks` call and ends only after all twelve windows, eleven
between-window yields, core peak scan/discard, raw writes, and sink finish.
Report separately:

- all twelve existing `onFamilyProfile` aggregates and their sequence totals;
- each outer `decodeWindow` wall and the sum of all decode walls;
- homogeneous and mixed submit-through-drain walls/counts;
- full stream wall, combined outside-decode stream residual, and count
  reconciliation;
- requested internal and between-window idle counts, without relabeling them as
  observed sleep or GPU time; and
- excluded acquisition, upload, combined backend creation, warmup,
  validation/hash/serialization, and cleanup walls.

Do not subtract an estimated readback, idle, encode, map, or OPFS component.
Anything not directly observed stays in the explicit residual. Family
submit-through-drain wall is authoritative for family ranking; complete stream
wall is authoritative for the current VAE subsystem budget.

## Correctness, lifecycle, and receipt gates

Before accepting attribution, require:

- exact package, profile, kernel, precision-map, source, planner, window,
  operation, quantum, dispatch, command-buffer, submission/drain, family,
  requested-idle, memory, and latent reconciliation to this record;
- twelve ordered family callbacks, no missing/duplicate/out-of-order window,
  and exact homogeneous-plus-mixed reconciliation to decoder totals;
- twelve detached outputs with their exact shape-dependent lengths; the
  unchanged stream must validate every retained core length and every one of
  its FP32 values before that core write resolves, with no timed scan or hash
  of discarded overlap;
- exact overlap/core coverage of all `8,640,000` audio frames once, exactly
  `17,280,000` finite interleaved raw FP32 values / `69,120,000` OPFS bytes, a
  finite positive raw peak, and a recorded complete raw-file SHA-256;
- no per-window storage retained after its core write resolves and no mapping
  overlap, map leak, second simultaneous returned window, or duplicate raw
  waveform in JS/WASM memory;
- unchanged progress order, FIFO ownership, one outstanding command buffer,
  drain-before-release, and no later encode/submit/map/write/callback after
  abort or destruction;
- balanced create/destroy and map/unmap counts, idempotent backend destruction,
  explicit VAE weight and device destruction, zero live buffers/bytes, and a
  closed raw file on success and on any actual cancellation or failure; and
- zero validation/uncaptured GPU errors, device loss, promise rejection,
  non-finite value, count mismatch, hash omission, lifecycle fault, or
  source-authentication failure.

Do not add a second target-browser cancellation or lifecycle execution. The
capture changes no scheduler or runtime behavior. Existing production
cancellation contracts plus focused fake-resource tests must prove no later
encode, submit, map, OPFS write, profile callback, or finalization after abort,
and cleanup/idempotent destruction on cancellation and failure. The one browser
execution exercises ordinary success cleanup unless it actually fails or is
cancelled, in which case that path must itself clean up and produces no timing
result.

Listening is not required because capture cannot change arithmetic or selected
production behavior. Any output, topology, or lifecycle change is a failed
exact gate, not a quality-tolerance candidate.

Persist one compact schema-v2 result plus ignored raw browser receipt, complete
thermal trace, and raw output hash/metadata. Record artifact byte lengths and
SHA-256 identities. Do not commit model data, decoded tensors, the raw OPFS
file, or audio.

## Decision rule and required reflection

After the one accepted run, rank K7, K1, ConvTranspose, Snake, Add, and mixed
from their directly measured submit-through-drain walls. Separately report one
combined within-decode non-family residual (encoding, requested idle, readback,
mapping, callback, and other decoder overhead) and one combined outside-decode
stream residual (core scan/write, between-window idle, and other stream
overhead). OPT-0023 does not split or rank the members of either residual. Use
per-window traces to expose order or thermal drift. Reconcile the measured
complete VAE wall with the current OPT-0018 DiT-only wall separately; do not
silently combine measurements into an end-to-end observation.

Register a follow-up mechanism only when both are true:

1. it targets a measured current bucket rather than a C300 extrapolation; and
2. its concrete mechanism and technically credible attainable floor imply at
   least `10,000 ms` of absolute saving over this exact C4500 sequence.

A family wall above ten seconds is not sufficient: the credible saving, after
unaffected work and integration overhead, must itself reach ten seconds. Mixed
wall is never prorated. If thermal drift, mixed work, or residual makes the
next choice ambiguous, close OPT-0023 with that limitation and register only
the smallest capture needed to resolve it; do not improvise a second run under
this ID.

Only if either combined residual is at least `10,000 ms` and large enough to
change the next decision, register the smallest direct measurement that can
separate its relevant members; do not assign or optimize an estimated component
under this ID. If no family has a credible ten-second mechanism, re-rank
cross-family scheduling, readback, storage, and DiT rather than sweeping nearby
kernel spellings. If a
quality-changing bounded-FP16 reduction becomes next, it needs a new declared
risk class, numerical trajectory/waveform gates, and listening authority. Do
not run another three-minute product until an integrated subsystem change makes
it decision-relevant.

## Non-claims

OPT-0023 makes no kernel, package, arithmetic, quality, integration, speedup,
end-to-end, three-minute, release, or under-60-second claim. It does not claim:

- the deterministic OPT-0011 xorshift latent represents real DiT latents or
  audio quality;
- C300 timing scales linearly, C512 repeats are independent, or one long trace
  is thermally stationary;
- submit-through-drain is pure GPU compute, requested idle equals observed
  idle, or residual can be assigned without measurement;
- mixed batches belong fractionally to their member families;
- graph, dispatch, byte, or memory counts imply utilization or throughput;
- package-native VAE-only execution is a full production generation; or
- a ten-second family total, isolated primitive win, or modeled floor proves a
  ten-second product saving.

The result is a current VAE state-of-affairs measurement whose sole purpose is
to choose the next highest-value evidence-backed action.

## Result — positive measurement, benchmark-only

The capture hypothesis passed. The one authoritative target-browser execution
completed the exact `C448 + 10 x C512 + C340` ordered sequence in
`161,392.39999997616 ms` from immediately before
`streamAceVaeRawChunks` through raw-sink finish/flush. It authenticated the
frozen package, deterministic latent, current OPT-0015 production profile,
kernel set, precision map, all 33 bound sources, target device, graph, memory,
window, and scheduling identities. It performed no production integration.

One earlier page preflight was rejected with the exact page error
`OPT-0023 launch did not immediately follow the gate`. Its supplied gate ended
at observation index `92`, epoch `1786788021049`, and the failure was observed
at epoch `1786788031924`. The exact internal click-time delay was not
persisted, so no precise launch delay is claimed. The source-controlled throw
preceded the worker run `postMessage`; the prepared backend was disposed and
there were zero timed windows, dispatches, or timing samples. This was one
rejected setup, not a performance attempt.

The subsequent accepted run is the sole authoritative timed execution. Its
fresh post-warmup gate used observations `168..198`: `31` nominal observations
over `30,001 ms`, maximum gap `1,005 ms`, no missing observation, and no
non-nominal observation. Timed work began `163.599853515625 ms` after the gate
endpoint. The same absolute-cadence trace covered warmup, gate, timed work,
validation, and cleanup with `390` nominal observations over `389,004 ms`,
maximum gap `1,036 ms`, zero transitions, and `29,101 ms` of coverage after
cleanup. There was no unchanged thermal retry.

### Measured attribution

The directly measured decoder-command-buffer submit-through-matching-drain
ranking is:

| Rank | Bucket | Batches | Quanta | Wall (ms) |
| ---: | --- | ---: | ---: | ---: |
| 1 | K7 Conv1D | 5,763 | 46,087 | 59,993.59999811649 |
| 2 | ConvTranspose1D | 896 | 7,168 | 42,401.00000369549 |
| 3 | K1 Conv1D | 1,019 | 8,152 | 25,772.300002217293 |
| 4 | mixed | 853 | 6,824 | 8,943.499999284744 |
| 5 | Snake | 1,959 | 15,672 | 4,301.499997854233 |
| 6 | Add | 848 | 6,784 | 2,041.1999996900558 |

The six measured buckets reconcile to `143,453.1000008583 ms` over all
`11,338` decoder command buffers / `90,687` sequence quanta. The sum of the
twelve outer decode walls is `161,111.30000007153 ms`. Their difference is one
unsplit `17,658.19999921322 ms` within-decode non-family residual. Full stream
wall exceeds the summed decode walls by one unsplit
`281.09999990463257 ms` outside-decode stream residual. No readback, mapping,
requested-idle, encoding, callback, scan, OPFS, or other member is prorated or
estimated from either residual.

The ten successive C512 decode walls ranged from `13,862.0` to
`14,025.799999952316 ms`; their attributed decoder walls ranged from
`12,332.400000333786` to `12,495.200001001358 ms`. The trace remained nominal,
and there is no monotonic late-window slowdown in this single sequence. This
is useful drift disclosure, not a claim that repeated windows are independent
or that the system is thermally stationary.

All exact scheduling counts reconciled: `90,675` graph quanta plus twelve
ingress quanta, `90,687` dispatches, `11,338` decoder plus twelve readback
command buffers, and `11,350` submissions/drains. Requested internal and
between-window idle totaled `11,338 + 11 = 11,349 ms`; those requests are not
observed sleep or pure GPU time.

The timed steady allocation and high-water were exactly `17` buffers /
`944,808,752` bytes, with no timed allocation. Bounded CPU live storage was
`9,016,320` bytes. The stream wrote all `8,640,000` audio frames /
`17,280,000` finite interleaved FP32 samples / `69,120,000` raw bytes, with raw
peak `0.9710559248924255` and SHA-256
`fb8aae85e21a8a93b39baf738d0f2577e18134c627a05562b710341d0d590f7c`.
Hashing was excluded from timing. Cleanup balanced all `17` buffer creations
and destructions and all `13` maps/unmaps, left zero live or mapped buffers,
removed the temporary raw entry, and destroyed the device.

### Big-picture reflection and next-decision boundary

The current measured C4500 VAE is `2,772.4466666428198 ms`
(`1.7478549251724%`)
above the old post-OPT-0015 C300-linear planning estimate of
`158,619.95333333334 ms`. That proximity makes the estimate a useful historical
planning check for this deterministic fixture, but it is not a thermally
matched speedup comparison. The older `431,377 ms` product VAE is likewise not
a current comparator.

Separately adding this VAE wall to OPT-0018's `73,072.6 ms` DiT-only wall gives
`234,464.99999997616 ms`. This arithmetic is not an observed end-to-end run,
but it makes the strategic constraint plain: the current measured VAE alone is
already `101,392.39999997616 ms` above the complete 60-second product budget,
and the separately measured DiT also exceeds that budget. No isolated
micro-optimization or complete elimination of one family can establish the
product target.

K7 is the largest measured family, but its nearby exact and reordered
mechanisms have already produced negative OPT-0014/0016/0017 evidence.
ConvTranspose is second, but OPT-0015 already owns its large exact win and
OPT-0022 closed the neighboring exact subgroup/polyphase geometry. A measured
family total is not a saving claim, so those totals alone do not justify more
nearby spellings. K1 is the largest less-exhausted family at
`25,772.300002217293 ms`; any K1 proposal must still name a mechanism with a
credible floor below `15,772.300002217293 ms` to clear the ten-second rule.
The `17,658.19999921322 ms` within-decode residual is decision-relevant only if
a smaller direct capture can separate a concrete removable member; its
`11,338 ms` requested-idle count must not be equated to observed saving or
changed without responsiveness, cancellation, and scheduling evidence.

OPT-0023 itself allocates or authorizes no follow-up implementation. It closes
as positive benchmark-only evidence and leaves the next ID to a separately
registered mechanism that clears the exact `10,000 ms` credible-saving rule.

### Limits and non-claims

This result does not measure GPU occupancy, utilization, hardware counters, or
pure GPU compute. Submit-through-drain includes host, driver, queue, and fence
effects; it excludes the following requested idle. The xorshift latent is not a
real DiT latent or quality oracle. There was no baseline/candidate A/B, speedup,
arithmetic change, numerical trajectory, waveform comparison, listening,
normalization, WAV encoding, planner, DiT, product, responsiveness, release, or
under-60-second gate. The raw output hash authenticates this capture only; it
does not establish musical quality or bit identity to an upstream waveform.

Artifacts:

- browser receipt: `34,172` bytes, SHA-256
  `6454a37243849ec9838d998abd9ca478d4b3720aa9b3edd4f497d152bda92d5c`;
- complete thermal trace: `59,950` bytes, SHA-256
  `ecd7eded7f17dd9a5a585b8859e30a942a573840e365500a718d3d67e0a64161`;
- compact result: [`../results/OPT-0023/result.json`](../results/OPT-0023/result.json).
