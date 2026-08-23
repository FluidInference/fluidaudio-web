# OPT-0016 — VAE K7 packed-KIO accumulator microtiles

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Date: 2026-08-14
- Author/agent: Codex
- Risk class: `exact`; every output must retain its increasing-K then
  increasing-Cin FP32 arithmetic and raw output identity
- Evaluated production baseline:
  `36608b857827b2b1d31ac91bf5cca9639fb0b9ed`
- OPT-0014 candidate core:
  `12e128ab323c0024ed683313b4d06c07041213e7`
- OPT-0014 frozen browser gate:
  `3904d212148cf2ecf93f317f8dcce3d59ef232a8`
- Frozen microtile core:
  `997891de0fe449c9b6551e80abc55604256969ad`
- Frozen browser gate:
  `085669d5aec0fc02f3268c8b462385b59fb72ab7`
- Motivating OPT-0014 artifact:
  `optimization/artifacts/OPT-0014/raw/packed-kio-k7-ab.json`,
  SHA-256
  `2445e5e3b07a3d950db8e7badcd74bff6fef687013bbc7fd56389acaedd845c3`

## Motivation

OPT-0014 was exact across 13,854,720 output words and 61,017,600 repack
U16 comparisons, but improved the weighted C300 K7 primitive projection only
from 9,506.499997437 ms to 8,562.849999070168 ms, or
1.1102027944515322x, while retaining an additional 61,017,600-byte packed
weight payload. It is not integrated.

The shipped fixed32 8-row × 128-Cout subgroup and OPT-0014's packed-KIO
16-row × 64-Cout subgroup both retain 32 live FP32 output accumulators per
lane. The packed kernel also doubles row broadcasts from eight to sixteen and
loads FP16 pairs through `u32` plus `unpack2x16float`. The small result despite
materially better weight locality makes register occupancy and invariant serial
FP32 throughput the next bounded hypotheses. Occupancy is inferred rather than
measured because the browser API exposes no hardware register counters.

The current integrated 12-second profile attributes 3,019.8 ms of its
7,265.8 ms decoder submit-through-drain wall to K7, making a material K7 win
decision-relevant.

## Hypothesis

Reducing each lane from 32 live FP32 accumulators to 16 or eight can improve
fixed32 subgroup occupancy enough to overcome the additional workgroups.
Binding the already accepted bit-identical KIO payload directly as typed FP16
can remove unpack instructions without introducing another repack or layout
axis. A 16-row × 32-Cout subgroup is expected to provide the best balance of
register pressure, input reuse, and workgroup count.

## Candidate

Keep 128-thread workgroups containing four fixed-32 subgroups and no
workgroup-memory barriers. Benchmark exactly three new per-subgroup tiles:

| Arm | Rows | Cout | Accumulators/lane | KIO weight view |
| --- | ---: | ---: | ---: | --- |
| `8x64` | 8 | 64 | 16 | `array<vec2<f16>>` |
| `16x32` | 16 | 32 | 16 | `array<f16>` |
| `8x32` | 8 | 32 | 8 | `array<f16>` |

The existing packed `16x64` `array<u32>` plus `unpack2x16float` kernel is the
unchanged control. All new arms bind the same OPT-0014 KIO bytes; direct typed
FP16 access is part of the candidate mechanism, not a separate experimental
axis. No second repack or additional persistent payload is authorized.

Every arm must retain bias initialization, increasing K then increasing Cin
FP32 arithmetic, current output conversion, and current signed-zero behavior.
No FMA/dot reduction, split reduction, model-math change, production selector,
or package-layout change is authorized by this record.

## Decision gate

1. Statically verify dispatch geometry and complete output ownership for all 17
   C300 K7 operations.
2. Before timing, compare raw bits for every arm on representative 1024-, 512-,
   256-, 128-, and final 128-to-2-channel operations spanning dilations 1, 3,
   and 9 plus meaningful first/tail boundaries. Final conv2 comparisons are
   raw U32; other FP16 outputs are raw U16. Require complete qNaN overwrite,
   intact guards/canaries, and deterministic rerun identity.
3. Compile and warm all arms. After one 30-second nominal thermal start, run
   the accepted gate once with no thermal retry.
4. Time the unchanged packed `16x64` control and all three candidates on the
   five representatives with three samples per arm in balanced cyclic order.
   A candidate qualifies only if its weighted representative median is at
   least 1.15x faster than the packed control. If none qualifies, stop.
5. If multiple candidates qualify within 2%, select `16x32`. Compare only the
   selected winner against shipped native fixed32 over the complete exact C300
   K7 sequence: all 17 operations and 2,404 production graph ranges, in
   balanced AB/BA order.
6. Include measured one-time GPU repack time in the candidate total and report
   the unchanged 61,017,600-byte persistent payload. Proceed only if raw output
   bits remain exact, the winner is faster in both paired orders, and its
   aggregate full-sequence speedup is at least 1.25x versus shipped fixed32.
7. Only after that browser result may a separate 12-second production smoke
   route the winner explicitly. It must reproduce the prior WAV exactly, reduce
   K7 family time from 3,019.8 ms to at most 2,415.84 ms, and reduce decoder
   submit-through-drain from 7,265.8 ms to at most 6,765.8 ms before any
   180-second or listening gate.

## Browser result

The frozen Chrome/M3 gate passed the declared exactness screen. Across the 15
first/interior/tail probes and two executions, all three candidates matched the
packed `16x64` control over `7,176,192` raw-U16 and `3,766,272` raw-U32
comparisons. All outputs completely replaced their qNaN prefills, guards and
adjacent canaries remained intact, and every arm's rerun hash was stable. The
five selected KIO repacks also matched their accepted OPT-0014 identities over
`19,500,544` U16 comparisons with intact redzones.

The run waited once for `30,130 ms` at nominal thermal pressure: 31 nominal
observations, a `1,005 ms` maximum poll gap, zero non-nominal observations, and
a `76 ms` launch delay. No unchanged-work thermal retry was performed.

The exact 2,402-range representative score used each tier's median of three
submit-through-drain samples. The two omitted ranges are the disclosed conv1
stratum. Results were:

| Arm | Weighted representative wall | Speedup vs packed `16x64` |
| --- | ---: | ---: |
| packed `16x64` | `3,331.9999829530716 ms` | `1x` |
| `8x64` | `3,407.0999702215195 ms` | `0.9779577975624927x` |
| `16x32` | `3,394.6999900341034 ms` | `0.9815300299687449x` |
| `8x32` | `3,306.499966621399 ms` | `1.007712087279326x` |

No candidate reached the required `1.15x`. The gate therefore stopped at its
declared seam: no winner was selected, none of the remaining 12 operation
weights or full-C300 dispatches was allocated, the complete fixed32 comparison
did not run, and no production selector or integration changed. Cleanup
destroyed all 60 created buffers, left zero live bytes, and destroyed the
device.

The persisted raw receipt is ignored from Git at
`optimization/artifacts/OPT-0016/raw/packed-kio-microtile-ab.json` (31,145
bytes), SHA-256
`3bfbe588d5aa6595b3f49caff670cb62293157157e4a34d0fb12349265266222`.

## Evidence and disposition

- Evidence conclusion: `negative`. Exactness passed, but none of the three
  nearby exact-order accumulator microtiles materially beat the unchanged
  packed control; the best arm reached only `1.007712087279326x` against a
  `1.15x` qualification threshold.
- Code disposition: `abandoned`. Preserve the isolated cores, gate, and exact
  evidence as benchmark history, but do not integrate a microtile or retain
  the 61,017,600-byte KIO payload in production for this mechanism.
- The registered exact-order nearby-microtile stop rule has fired. Another
  unchanged accumulator-tile sweep is not justified.
- Any next K7 mechanism that changes reduction or rounding behavior must use a
  new experiment ID, declare `reordered-rounding` risk, and pass explicit
  tensor, denoise-step, waveform, and listening gates before integration.
- No full-C300 winner comparison, 12-second production run, integrated decoder
  speedup, 180-second generation, listening result, or under-60-second claim
  was produced.
- Canonical result: [result.json](../results/OPT-0016/result.json)

## Stop rule

If no microtile reaches 1.15x over packed `16x64`, or the selected winner does
not reach 1.25x over shipped fixed32 on the complete C300 sequence, record a
negative result and stop this final bounded exact-order K7 accumulator-tile
sweep. Such a result would indicate that invariant serial FP32 throughput or
fixed dispatch cost dominates enough that another nearby exact-order tile shape
is not justified. This stop rule is specific to exact-order microtiles, not a
general prohibition on future K7 work.

Any later K7 experiment that changes rounding behavior through parallel Cin
reductions, subgroup dot or multiple partial accumulators, matrix-style FMA, or
mixed precision must receive a separate never-reused experiment ID and an
explicit `reordered-rounding` risk classification before performance code is
changed. It must be judged by declared tensor, denoise-step, waveform, and
listening gates rather than OPT-0016's raw-bit identity gate.
