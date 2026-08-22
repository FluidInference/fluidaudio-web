# OPT-0010 — Package-native planner token profiler

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Date: 2026-08-13
- Author/agent: Codex
- Risk class: `exact`; OPT-0010 observes the unchanged planner and does not
  alter its arithmetic, token constraints, sampling, or scheduling

## Hypothesis

Default-planner generation is a separate performance blocker from direct song
generation. The accepted 30-second default-CoT run spent 163.702 seconds in the
semantic planner while generating 260 sampled tokens. Current decode uses
matrix-shaped portable GEMM for M1/M2 work, scores the complete 217,204-row
tied vocabulary, reads every logit shard back, and drains plus yields after
each logical model quantum. A package-native token trace can distinguish dense
execution, tied-head traffic, readback/sampling, and cooperative scheduling
before any one mechanism is selected.

These costs guide later experiments; they are not independent performance
requirements. GEMV, semantic-head restriction, and command-buffer batching
remain separate changes so that a useful contribution is not hidden inside a
combined result.

## Identity

- Allocation baseline: pushed `main` commit
  `6b144d88d44361b80f56b3c0bdc5dceaa1e9dd48`
- Raw-FP16 package manifest SHA-256:
  `c5b547cd08aa5e6d2971b2c9c84940b8af193f2e230ce689258ca81fcd292a3b`
- Planner source revision:
  `148d8ea0225bdab342ee1ae3a354275ccd60ca80`
- Accepted default-CoT receipt SHA-256:
  `554106761fde0a5fab8075324d34fc08cb31b885f044c173cd4ba1ab1facb678`
- Machine: MacBook Air `Mac15,12`, Apple M3, 10 GPU cores, 16 GB unified
  memory; exact browser, OS, adapter, and device identity are retained with the
  result
- Execution profile: existing `raw-fp16` portable planner path; no production
  profile change
- Frozen attribution core/production source commit:
  `00dfd4732aa019bbbb238ae40265fe86cb38f27b`
- Frozen browser harness commit:
  `81e84df955e7cb812a60d9c6decadff3791234e3`

## Change

Add benchmark-only tracing around the existing package-native planner executor.
Do not change its graph, weights, arithmetic, cache, vocabulary, quantum
boundaries, queue-drain policy, idle policy, readback, FSM, or sampler.

The profiler separates:

- M1 CoT and M2 classifier-free semantic-token model execution;
- embedding, each of the 28 layer groups, final normalization/gather, and every
  tied-head shard;
- CPU encode, submit-through-drain, completed cooperative idle, logit copy/map,
  full-vector reconstruction, constraint processing, and sampling; and
- logical weight and activation traffic, padded-row work, physical dispatches,
  command buffers, drains, idles, readback bytes, and output token/cursor state.

Representative short, middle, and long cache positions are used only when they
change attention/cache work or the mechanism choice. One authoritative token
at a cache position is sufficient when the attribution is stable.

## Correctness gate

- Authenticate and load only the required planner phase from the pinned FP16
  package.
- Use accepted planner prompt/token identities and deterministic sampler state.
  Require the unchanged profiler path to reproduce the expected complete logit
  shapes, finite admitted logits, selected token, Philox cursor movement,
  cache append, and repeated-run determinism.
- Reconcile every model quantum, primitive dispatch, command buffer, submit,
  drain, completed real idle, readback shard, and progress event. Preserve raw
  ordered tags rather than only aggregates.
- Prove cancellation after a fully drained quantum and completed idle prevents
  later encoding/submission, then prove idempotent cleanup and no device event.
- No listening is needed for an observational profiler. Any later arithmetic or
  token-selection change receives its own numerical and listening gate.

## Benchmark protocol

1. Compile and upload outside timing, perform one symmetric untimed warmup, then
   begin a continuously nominal external thermal gate and poll through the
   sample and cleanup.
2. Measure one unchanged M1 CoT token and one unchanged M2 semantic token at the
   smallest cache positions that expose distinct work. Add a longer cache point
   only if attention/cache attribution can change the next mechanism.
3. Use completion-fenced wall time and at most one outstanding command buffer.
   Retain encode, submit-through-drain, real idle, readback/map, reconstruction,
   constraint, and sampling intervals without subtracting overlapping clocks.
4. Report current full-head weight bytes and readback bytes, plus the exact
   allowed semantic-code rows as diagnostic opportunity size. Do not time a
   restricted head in this experiment.
5. Close with a ranked, evidence-backed choice among dedicated M1/M2 GEMV,
   semantic-head restriction, quantum batching, or another measured cause. More
   than one may proceed when their contributions are independent.

## Main risks

- A single token can be cache-position-sensitive. Retain the exact cache length
  and add a second position when the trace shows material attention/cache drift.
- Proxy or callback instrumentation can perturb short CPU intervals. Preserve
  the unmodified GPU command topology, record observer overhead, and treat
  completion-fenced wall as authoritative.
- CoT constraints differ from the semantic-code range. A later restricted-head
  implementation must be state-specific and fail closed rather than assuming
  every planner token uses the same vocabulary.
- Faster GEMV can expose cooperative idle as a larger fraction of wall time.
  Scheduling remains independently attributable instead of being silently
  folded into a kernel speedup.

## Evidence and disposition

- Evidence conclusion: `positive`. The frozen observational run completed all
  six M1/M2 cases and made the next mechanisms decision-relevant without
  changing the package-native planner.
- Code disposition: `benchmark-only`. The harness changed no shipped package,
  graph, arithmetic, constraint, sampler, command topology, or production path;
  no production integration occurred.
- Result JSON: [schema-v2 result](../results/OPT-0010/result.json)
- Raw browser receipt:
  `optimization/artifacts/OPT-0010/raw/planner-token-timing.json`, SHA-256
  `244838e6cdad0c5faeeb263356f7c0709820875ca59265e3bfae9ff7e7646bcf`
- Raw external thermal log:
  `optimization/artifacts/OPT-0010/raw/planner-token-timing-thermal.jsonl`,
  SHA-256
  `2cca9cccd1def85c5df4b02496db5b25cd213d72cf20f8e4e9efdc3aeb693438`

### Frozen run identity and preparation

The evidence run used Chrome `151.0.7922.138` on `Mac15,12`, Apple M3 with 10
GPU cores and 16 GiB unified memory, macOS `26.5.2` build `25F84`. The adapter
was Apple Metal 3 with `shader-f16`; the device enabled `shader-f16` and not
subgroups. The authenticated raw-FP16 manifest was 713,747 bytes at converter
revision 4. It supplied 314 planner tensors in 33 weight files, and the planner
resident allocation was 1,325,768,704 bytes.

The run acquired all 36 required files from the new benchmark origin in
46,976 ms, uploaded the planner in `18,349.600000023842` ms, and spent
`11,047.900000035763` ms compiling plus creating the untimed references. Those
preparation intervals are retained but excluded from every token timing. The
result also binds SHA-256 identities for all 19 production planner, tokenizer,
scheduler, arena, and kernel source files used by the attribution core.

### Six exact token observations

Every timed case used a fresh equivalent prefill, at most one outstanding
command buffer, and the unchanged full 217,204-row tied head. `Model` below is
the sum across all 33 model quanta; `layers` covers the 28 layer quanta. Times
are completion-fenced worker `performance.now()` milliseconds.

| Case | Cached/capacity | Timed window | Sampling | Model submit/drain / wall / idle | Layers submit/drain / wall | Head submit/drain | Readback wall | Full-logit U32 words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cot-m1-short` | 120 / 512 | `498.39999997615814` | `279.19999998807907` | `148.69999992847443` / `196.5` / `41.40000003576279` | `133.30000001192093` / `173` | `11.899999976158142` | `22.600000023841858` | 217,204 |
| `cot-m1-mid` | 160 / 1,024 | `410` | `203.30000001192093` | `154.30000030994415` / `203.30000001192093` / `43.69999980926514` | `140.00000017881393` / `182.30000001192093` | `10.900000035762787` | `3.399999976158142` | 217,204 |
| `cot-m1-long` | 212 / 2,048 | `335.5` | `85.89999997615814` | `190.29999995231628` / `244.89999997615814` / `43.700000047683716` | `165.5` / `207.5` | `15.699999988079071` | `4.5` | 217,204 |
| `semantic-m2-short` | 268 / 768 | `406.69999998807907` | `179.39999997615814` | `171.9999998807907` / `221.30000001192093` / `42.59999996423721` | `156.30000001192093` / `199.20000004768372` | `10.999999940395355` | `5.899999976158142` | 434,408 |
| `semantic-m2-mid` | 328 / 1,280 | `385.5` | `182.10000002384186` | `152.80000007152557` / `198` / `40.00000011920929` | `140.90000015497208` / `179.9000000357628` | `9.899999976158142` | `5.300000011920929` | 434,408 |
| `semantic-m2-long` | 401 / 2,048 | `387.39999997615814` | `169` | `167.50000005960464` / `211.69999998807907` / `39.5` | `155.20000004768372` / `192.80000001192093` | `9.699999988079071` | `6.699999988079071` | 434,408 |

All 1,954,836 compared U32 words were bit-identical to their untimed
references. Each case selected the identical token and U32 sampler word and
advanced the same absolute Philox cursor: respectively token/word/draw span
`30915` / `2815301038` / `16..17`, `13572` / `1589390448` / `56..57`,
`151645` / `2307094523` / `108..109`, `192370` / `2004582350` /
`125..126`, `156326` / `503673048` / `185..186`, and `155832` /
`3288166745` / `258..259`.

Each case reconciled exactly 33 model quanta, 624 top-level primitives, 628
physical dispatches, 33 model command buffers, one readback command buffer,
34 submissions/drains/completed real idles, five logit-shard copies plus one
write-status copy, and no more than one outstanding command buffer. M1 read
434,408 raw logit bytes into a 434,688-byte buffer; M2 read 868,816 bytes into
869,120 bytes. The M1 cases scheduled 10,605,297,664 multiply-adds against
662,818,816 logical multiply-adds, while M2 scheduled the same total against
1,325,637,632 logical multiply-adds. Cache-valid attention grew at each pinned
position, but the layer interval remained large at every position.

### Thermal, cancellation, lifecycle, and responsiveness

The browser pre-gate ran from epoch `1786685161546` through `1786685197546`:
37 nominal observations over exactly 36,000 ms with a conservatively reported
maximum gap of 1,006 ms. Timed work began at `1786685208946`, ended at
`1786685219452`, and cleanup completed at `1786685219813`.

The independently retained logger contains 342 consecutive level-zero
observations, sequence `0..341`, from
`2026-08-14T05:21:53.532154+00:00` through
`2026-08-14T05:27:34.546417+00:00`. Its elapsed clock is
`341005.0982500543` ms and its exact largest consecutive monotonic gap is
`1005.0118330400437` ms. It starts before the pre-gate and continues through
the timed interval and cleanup; there are zero non-nominal observations.

The cancellation probe encoded, submitted, drained, and completed exactly one
idle for one quantum, then rejected with `AbortError`; all later encoding and
submission were prevented. Cleanup reconciled 1,338 created buffers with 1,338
unique destroys and 1,338 total destroy calls, left zero live tracked buffers,
was idempotent, observed no runtime/device-loss event, and destroyed the device
after the event check.

The page remained responsive with maximum animation-frame/timer gaps of
`94.09999999999854` / `107.19999998807907` ms. The worker heartbeat's maximum
gap was 2,118.5 ms. That worker gap is retained—not presented as a responsiveness
pass—and is consistent with synchronous CPU sampling or setup blocking in the
worker.

### Decision

The ranked follow-ups are:

1. A state-specific compact semantic head and compact sampler. M2 full-vocabulary
   CPU sampling alone cost 169.0–182.1 ms/token. The exact static row plan keeps
   semantic IDs `151669..215668` across shard intersections 3/4 and handles EOS
   token `151645` separately. It reduces logical head weight traffic and raw
   logits from 444,833,792/868,816 bytes to 131,072,000/256,000 bytes, avoiding
   exactly 70.53461262223531% of each.
2. A dedicated low-row FP16 GEMV with FP32 accumulation. The 28 dense layers
   cost `133.30000001192093..165.5` ms submit-through-drain per token and the
   current matrix-tiled path schedules roughly 16x/8x M1/M2 logical work.
3. Bounded multi-quantum command-buffer batching. The 34 per-token drains/idles
   contributed `39.5..43.700000047683716` ms, a real but smaller standalone
   ceiling than the first two mechanisms.

The semantic restriction was **not executed**. A rough diagnostic applies the
static removable share to measured M2 head, readback, and full-vocabulary
sampling intervals and estimates about 132–140 ms/token saved, or roughly
1.5–1.6x total semantic-token speedup. That is an unmeasured projection, not a
benchmark result, not an additive measured saving, and not a correctness or
production claim. Likewise, scaling the two long-cache observations to about
110 CoT plus 900 semantic tokens gives `385564.9999785423` ms of token decode,
but excludes acquisition, upload, prefills, and every other product stage; it
is only a diagnostic showing that the unchanged planner remains incompatible
with the 180-second-under-60-second product target.

OPT-0010 therefore closes positive as a successful attribution experiment and
remains benchmark-only. Each ranked mechanism requires its own experiment and
correctness/performance gate.
