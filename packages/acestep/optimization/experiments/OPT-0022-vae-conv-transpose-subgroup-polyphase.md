# OPT-0022 — exact fixed32 subgroup polyphase ConvTranspose1D

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Date: 2026-08-15
- Author/agent: Codex
- Risk class: `exact`; every candidate output must be raw-U16 identical to
  the integrated OPT-0015 congruent kernel
- Allocation baseline: `73a8a85334226a2f2eb888960796ade8875ea6ad`
- Integrated comparison profile: OPT-0015 commit
  `36608b857827b2b1d31ac91bf5cca9639fb0b9ed`
- Frozen OPT-0015 kernel source SHA-256:
  `cbcb9bcd5f856ce1c9e10aabca0ec0f95651c03d2c45b8076de3ba5022c6c3e2`
- Frozen OPT-0015 canonical result: `6,032` bytes, SHA-256
  `06ba77f03c340c612e63320ef06f08ae74bc3f4e3e5aff9abb39a3cdc6ed01f3`
- Fixture/model/VAE manifest SHA-256:
  `cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`,
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`,
  and `5644bcca87678b4f654b9541459355a73ef136c6bb601aa783b6f50fe2f6dba3`

## Why this is next

The latest subsystem pause rejects another exact DiT-panel variation. OPT-0019
through OPT-0021 showed that the best nearby exact dense layout projects only
about eight seconds at M2250, while horizontal dot and vec4-panel rewrites
regressed. DiT batch-eight scheduling has only about `6.245 s` of
evidence-backed stage leverage when OPT-0006 is transferred per eliminated
drain, not the required ten seconds.

The current C300 VAE profile instead measures homogeneous ConvTranspose at
`2,001.9999997615814 ms` for only `88,055,578,624` valid MACs, about
`88 GFLOP/s`. It remains far below the `1.6–2.4 TFLOP/s` demonstrated by
other FP16-heavy shapes on this browser/GPU. OPT-0015 removed the wasted
noncongruent taps, but its surviving exact kernel still stages shared input and
weight panels and executes `28,594,176` workgroup barriers over C300.

The hypothesis is that converter-native output-contiguous polyphase weights
and a fixed32 subgroup-broadcast kernel can remove that utilization bottleneck
without changing model arithmetic. This is materially distinct from
OPT-0015's shared-panel kernel and from OPT-0014 through OPT-0017, which concern
ordinary K7 Conv1D rather than ConvTranspose1D.

## Frozen candidate

Use exactly WG128 with four authenticated 32-lane subgroups. Each subgroup
owns the same 16 phase-aligned output rows and one distinct band of 32 adjacent
output channels. Lane `l` owns one output channel and 16 scalar FP32
accumulators; one workgroup therefore covers `16 x 128` outputs.

WG64 is not an arm because it performs the same subgroup work and logical
traffic with twice as many workgroups. WG256 is not an arm because it merely
halves workgroup count while doubling the logical per-workgroup accumulator
footprint and has no stronger target evidence. There is no geometry, chunk,
vector-width, layout, workgroup-size, or accumulator sweep under this ID.

The candidate kernel identity is
`ace-vae-fp16-fixed32-subgroup-polyphase-conv-transpose1d-v1`. It requires
`shader-f16`, `subgroups`, subgroup minimum and maximum exactly `32`, at least
128 workgroup invocations, and X capacity at least 128. It has no portable or
OPT-0015 fallback under that identity.

### Exact reduction and subgroup schedule

For output time `t`, define `p=t+padding`, `phase=p%stride`, tap kernels
`phase` then `phase+stride`, and source rows `floor(p/stride)` then one less.
The shader must initialize from widened FP16 bias, visit tap zero then tap one,
visit every valid input channel strictly increasing, execute
`sum = sum + f32(input) * f32(weight)`, and store once through the same FP16
conversion as OPT-0015. An invalid source row skips its terms; adding staged
zero is not an exact substitute.

Within each tap, input channels are handled in lexical chunks of eight solely
to coalesce loads; the arithmetic order is unchanged:

1. Every lane preloads eight scalar FP16 weights for its owned output channel.
   For a fixed chunk member, the 32 lanes address adjacent output channels.
2. For phase row `0..15`, lanes `0..7` load the eight adjacent input channels
   for that row. Invalid rows or channel tails produce positive zero for the
   broadcast source, but the accumulator update remains guarded by the
   row/tap validity predicate.
3. Chunk members `0..7` execute in order. All lanes perform the uniform
   `subgroupBroadcast(input_lane, member)` and then update that row's
   accumulator with the corresponding scalar weight.
4. The next chunk begins only after all eight members, so global Cin order is
   identical to the oracle.

There are exactly eight broadcasts per phase row and input-channel chunk.
There is no workgroup memory, workgroup barrier, dot, `fma`, vector reduction,
split reduction, partial buffer, FP16 accumulator, or reordered term.

## Converter-native physical layout

The benchmark may construct both immutable fixture layouts directly, but a
production escalation must replace, not duplicate or runtime-repack, the five
transpose weights. Their logical `[out, 2*stride, in]` values become physical
`[phase=stride, tap=2, in, out]`, where
`packed[phase,tap,cin,cout]` is the exact FP16 word from
`native[cout, phase + tap*stride, cin]`.

Freeze layout identity
`conv-transpose1d-phase-tap-input-output-f16-v1`. A promoted converter must
have a new deterministic revision/package identity, unchanged logical tensor
names and shapes, physical storage shape `[stride,2,inputChannels,outputChannels]`,
full inverse-layout raw-U16 verification, and no native O-K-I duplicate. The
five weights remain exactly `49,610,752` FP16 words / `99,221,504` bytes, so
candidate parameter and resident bytes must not increase. Resolver and runtime
selection fail closed on the old layout under the candidate profile.

## Exact C300 accounting

| Operation | Shape | Ranges | OPT-0015 WGs | Candidate WGs |
| --- | --- | ---: | ---: | ---: |
| block 0 | `300x2048 -> 3000x1024`, S10/K20 | 54 | 69,120 | 4,320 |
| block 1 | `3000x1024 -> 18000x512`, S6/K12 | 81 | 92,544 | 5,784 |
| block 2 | `18000x512 -> 72000x256`, S4/K8 | 81 | 144,000 | 9,000 |
| block 3 | `72000x256 -> 288000x128`, S4/K8 | 81 | 288,000 | 18,000 |
| block 4 | `288000x128 -> 576000x128`, S2/K4 | 81 | 576,000 | 36,000 |
| **Total** | five operations | **378** | **1,169,664** | **73,104** |

The candidate executes 292,416 subgroups, has exactly 16x fewer workgroups,
uses zero workgroup-storage bytes, and removes all 28,594,176 current barrier
events. Scheduled MAC slots remain `117,121,744,896`; exact valid MACs remain
`88,055,578,624`.

Validity-adjusted source-level logical input loads fall from
`22,013,894,656` to `5,503,473,664` bytes, while weight loads remain
`14,640,218,112` bytes. Total logical operands therefore fall from
`36,654,112,768` to `20,143,691,776` bytes (`1.81963x`). The conservative
scheduled upper-bound is `43,920,654,336` versus `21,960,327,168` bytes.
These are source-level request counts, not hardware transactions, bandwidth,
occupancy, or wall-time claims.

## Bounded primitive gate

Arm A is the frozen integrated OPT-0015 congruent owner over native O-K-I
weights. Arm B is the candidate over the bit-preserving polyphase layout.
Before timing:

- authenticate all five shapes, 378 graph ranges, sources, generated shaders,
  fixture/package identities, browser, adapter limits, and exact 32/32
  subgroup bounds;
- inverse-map all `49,610,752` candidate weight words with zero mismatches;
- reuse the 15 exact first/interior/tail probes from OPT-0015 and cover every
  stride phase plus the one-tap boundaries;
- execute balanced A/B and B/A correctness passes and compare at least the
  accepted `8,404,992` raw-U16 output domain with zero mismatches and
  deterministic hashes; and
- require qNaN-prefilled complete finite writes, intact guards/canaries,
  immutable inputs/weights, zero GPU errors, balanced maps, drain-before-
  release, idempotent destruction, device destruction, and zero live bytes.

Compile and warm both arms before one fresh continuous 30-second nominal
thermal gate. Run six balanced timing rounds with alternating AB/BA position;
one sample is one dispatch in one command buffer plus its matching fence.
Compute unrounded stratum medians and weight first/interior/tail strata by all
378 C300 ranges. Do not retry unchanged work for a better trace.

Primitive B qualifies only if every operation aggregate is faster, both timing
positions favor B, the weighted A/B speedup is at least
`1.3398349037268882x`, and every
correctness/lifecycle gate passes. The separately drained projection is only a
screen; it cannot satisfy the production absolute gate.

## Production escalation and stop rules

If the primitive gate fails, record the result and stop without converter,
package, runtime-profile, C300 production, long-window, waveform, listening,
or 180-second work.

If it passes, create one explicit candidate package/kernel-set/runtime profile
with physical replacement of exactly the five weights. After one nominal
thermal gate, run the authenticated 12-second direct C300 production profile.
It must reproduce the accepted raw waveform U32 bits and WAV SHA-256, retain
all cleanup/cancellation invariants, and meet all three ceilings:

- homogeneous ConvTranspose submit-through-drain
  `<= 1,494.213946951832 ms` from the current `2,001.9999997615814 ms`;
- decoder submit-through-drain `<= 6,758.013946308101 ms`; and
- complete VAE wall `<= 7,546.71394719025 ms`.

The transpose ceiling is the formal ten-second long-song projection:
`5,908 / 300 = 19.693333...` decoded-window-frame scaling. A useful, nonbinding
stretch target is `600–1,001 ms`; a marginal threshold pass does not make the
under-one-minute target plausible by itself.

Only after the C300 gate passes may one package-native window profile measure
C448, C512, and C340 and weight them `1/10/1`. Run another 180-second product
only if that measured window projection materially changes the whole-product
budget. Because the mechanism is exact, no listening gate is required if the
complete raw waveform remains bit-identical.

## Non-claims and next reflection

OPT-0022 does not claim subgroup broadcasts are free, request counts are
transactions, primitive timing transfers linearly, post-OPT-0015 long VAE wall
has been measured, or under 60 seconds is currently forecastable. A positive
transpose result is one family win, not a VAE solution. After its integrated
family profile, re-rank K7, K1, mixed, scheduling, and DiT from the new measured
walls before allocating another experiment. If this exact utilization change
misses materially, the next evidence-backed path is a separately registered,
quality-gated bounded-FP16 reduction experiment rather than another exact
layout spelling.

## Result

### Authenticated execution

The target-browser primitive gate completed once and stopped at its declared
boundary. Registration is commit
`88a1bc1f3ffb06a6ca714437ba8616c11ed212f2`; the candidate core, browser
harness, and static contracts are commit
`8345fce46c92afa109c1b38fa45d15df31b94de5`. The raw browser receipt is
68,022 bytes with SHA-256
`6806deefda170661542577a4fe55d978ecb55bc957ced88cb97555c0e2e7bbfa`.
The authoritative continuous thermal trace is 6,328 bytes with SHA-256
`dda409ff5e63bc65b62b7d65f36a7f69f0256a0afc957263d000e34271bee949`.
The canonical result and focused result contract bind both artifacts.

The executable was Google Chrome `151.0.7922.138` (reduced user agent
`151.0.0.0`) on macOS `26.5.2` build `25F84`, MacBook Air `Mac15,12`,
Apple M3 with 10 GPU cores and 16 GiB unified memory. The adapter reported
Apple Metal 3, fixed subgroup size 32, `shader-f16`, WG256, 16,384 bytes of
workgroup storage, a 268,435,456-byte maximum buffer, and a 147,456,000-byte
maximum storage binding.

The source SHA-256 identities were:

- integrated OPT-0015 arm A,
  `cbcb9bcd5f856ce1c9e10aabca0ec0f95651c03d2c45b8076de3ba5022c6c3e2`;
- OPT-0022 arm B,
  `b3a02e29419021d78f669b7ed0333b80c8e5739ed46d9d9645f814d971a9edfa`;
- TypeScript harness and HTML entry,
  `4db0643edc6449087d6b28b5ee6dba3a3b0b9caabc2b4f8eb8d7a44bb27cadb8`
  and
  `b97ca849b7a2613f3e345cef49bfd50456a25931aab32cf8454adcd5a383d1dc`;
  and
- pre-execution and candidate-core contracts,
  `64e3af52b7d50f7b6adff079a7f807382c1501417564f079fcb7287e0ecd3f21`
  and
  `19867f4400308a3ab8c321bdd609ed71e1af17421685532e6a21e514390234f6`.

All five A/B generated-shader pairs matched their frozen identities. No model
package was loaded, and no converter, manifest, persistent package layout,
runtime selector, or production profile changed.

### Exact correctness result

Every exactness and lifecycle gate passed. The inverse-layout gate compared all
`49,610,752` candidate weight words against the native O-K-I fixtures with zero
raw-U16 mismatches. Balanced A/B and B/A execution covered all 15 selected
first/interior/tail probes, all stride phases, and both one-tap boundaries for
every operation. Across 60 output executions and 30 exact comparisons, all
`8,404,992` compared raw-U16 words matched. Deterministic rerun hashes matched,
every qNaN-prefilled selected output was completely written, all guards and
adjacent canaries were intact, all 20 immutable input/native-weight/polyphase-
weight/bias sources remained unchanged across 334,843,904 bytes, and there
were zero uncaptured GPU errors.

Cleanup drained before release, balanced all 100 maps and unmaps, destroyed all
110 created buffers, reached zero live buffers and bytes, destroyed the device,
and remained idempotent on a second call. Logical GPU high-water was
987,542,272 bytes. Because the candidate was bit-identical and did not pass the
performance gate, no numerical-tolerance or listening gate was applicable.

### Negative timing result

After correctness, compilation, and warmup, one fresh continuous nominal gate
qualified the sole authoritative timed run. The gate used trace indexes 24
through 54: 31 nominal observations over 30,001 ms, maximum poll gap 1,003 ms,
zero non-nominal observations, and an exact receipt-recorded 998 ms launch
delay. The full trace retained 111 nominal observations over 110,002 ms with a
maximum 1,004 ms gap and continued 52,710 ms past cleanup.

The one run retained all 180 dispatch samples: six alternating A/B or B/A
rounds for every one of the 15 probes, with one dispatch in one command buffer
and one matching queue-completion fence per sample. Recomputed median-six and
exact graph-range-weighted scores were:

| Operation | Ranges | A (ms) | B (ms) | A / B | B faster |
| --- | ---: | ---: | ---: | ---: | --- |
| `block-0-conv-t1` | 54 | `1379.6500012278557` | `611.7999975085258` | `2.2550670265549146x` | yes |
| `block-1-conv-t1` | 81 | `867.7499999403954` | `798.0499990582466` | `1.0873378873058073x` | yes |
| `block-2-conv-t1` | 81 | `656.5999971032143` | `662.6499999761581` | `0.9908699873641266x` | no |
| `block-3-conv-t1` | 81 | `875.8499980568886` | `867.8499971628189` | `1.009218183926051x` | yes |
| `block-4-conv-t1` | 81 | `848.8500009179115` | `731.5500018596649` | `1.1603444723669738x` | yes |
| **Weighted C300** | **378** | **`4628.699997246265`** | **`3671.8999955654144`** | **`1.2605735458036402x`** | **no** |

B was faster in both timing positions, but it regressed block 2 and its
weighted speedup was only `1.2605735458036402x`, below the frozen
`1.3398349037268882x` threshold. Two of the three mandatory timing conditions
therefore failed, and the primitive stop rule fired.

Exactly three page preflights were rejected before the first timed dispatch,
so none is a benchmark sample, timed run, or unchanged-candidate retry. The
first supplied a stale launch delay of approximately 11 seconds, beyond the
freshness cap. The next two used the original drifting shell logger: 31
observations spanned approximately 31.47 seconds and failed the parser's
observation-count/duration consistency rule because that duration required 32
observations. Two subsequent short logger smoke checks emitted null levels and
were stopped before any page launch; they were diagnostics, not preflights or
benchmark attempts. Absolute-cadence logging corrected the setup before the
sole authoritative run. There was no performance rerun.

### Disposition and lesson

OPT-0022 is negative and abandoned. The fixed32 subgroup/polyphase mechanism
substantially helps the stride-10 operation, but exact arithmetic and reduced
source-level requests do not translate into the required complete-family gain:
the dominant remaining shapes are near parity, and one regresses. This rejects
the frozen mechanism on this browser/GPU; it does not show that subgroup
broadcasts are free, identify hardware transactions, or justify a nearby
geometry, chunk, or layout sweep.

Integrated OPT-0015 remains the production ConvTranspose owner. No converter,
package, runtime, C300 production, long-window, waveform, listening, 180-second
product, or integration escalation was performed or authorized. Before another
kernel experiment or full-song run, the package-native C448 + 10xC512 + C340
long-window VAE profile should replace the current linear extrapolation and
support a fresh subsystem re-rank. Any later bounded-FP16 reduction path needs
a new quality-gated experiment ID.
