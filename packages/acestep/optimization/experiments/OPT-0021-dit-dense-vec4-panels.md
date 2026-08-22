# OPT-0021 — DiT dense K-major vec4 panels

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Date: 2026-08-14
- Author/agent: Codex
- Risk class: `exact`; every output must retain OPT-0019's strictly increasing-K
  FP32 `sum + f32(f16(a)) * f32(f16(b))` sequence and be raw-U32 exact

## Identity and inherited authority

- Allocation baseline and OPT-0020 negative-result authority:
  `bc2484096ff5622bc21d03aad80d08f12719989e`
- Current OPT-0009 dense source SHA-256:
  `a238f67da07c6ba1097da9d9e9e97960ae97d2e1d5c129fcbabf69e962cbb6b3`
- Frozen OPT-0019 exact cooperative source SHA-256:
  `b5dad12724882d3fc942c7df7b10c7b7b89a4bed595125ff11a5905c03152a37`
- OPT-0020 benchmark core and harness commit:
  `1d825a1399fdcc23c4ef3ce18151a2efa2415626`
- OPT-0020 canonical result: 10,076 bytes, SHA-256
  `8086bfc2f7a2524811c0492a37ac23a3e6f6746c56a3bfd098d581b16dd0a4e1`
- OPT-0020 raw receipt: 90,410 bytes, SHA-256
  `8c168677ee630a4ee337e7b3e7e2c0d8c1c2b53493ce6eebdc48765cdb554679`
- OPT-0020 continuous thermal trace: 6,212 bytes, SHA-256
  `cd5461cb760b9e6022a0f8b78105661d1cb17d7e7dd525f415fc7e60298de960`
- Target: MacBook Air `Mac15,12`, Apple M3 with 10 GPU cores and 16 GiB
  unified memory. The exact browser, OS, adapter, fixture, generator, sources,
  and generated shaders must be captured by the result.

## Why this experiment exists

OPT-0019 proved that exact cooperative reuse helps, but its scalar workgroup
representation constructs every FP16 vector from individual array elements.
OPT-0020 then proved that transposing the weight panel and replacing four
ordered scalar-K updates with a horizontal FP32 `dot` is the wrong mechanism:
its fresh complete `4/2/2/1` score regressed from B's
`180.00000029802322 ms` to `302.6499999165535 ms`.

The same OPT-0020 run measured A at `217.99999964237213 ms`. B therefore
saved `37.99999934434891 ms` per layer-evaluation. Reaching ten seconds over
the 192 production layer-evaluations requires `52.0834 ms` per evaluation, so
a candidate sized against that run would need another `14.08340065565109 ms`,
or about 7.8241% of B's score. These values motivate the screen only; all
decisions use fresh balanced A/B/C medians.

## Frozen three-arm screen

The primitive screen has exactly three arms:

- **A — current OPT-0009:** the current WG128 M32xN256xK32 subgroup owner.
- **B — exact OPT-0019:** the frozen WG256 M64xN128xK16 scalar-panel owner.
- **C — OPT-0021:** one WG256 M64xN128xK16 candidate with the exact vector
  panel representation and ordered arithmetic below.

There is no geometry, padding, orientation, unroll, vector-width, accumulator,
subgroup, or workgroup-size sweep. An unpadded panel, K32 panel, transposed
weight panel, scalar-panel hybrid, alternate activation orientation, `dot`,
`fma`, native-FP16 accumulator, subgroup matrix operation, persistent repack,
or lower-precision arm requires a new experiment ID.

## Frozen candidate mechanism

C has a 16x16 invocation grid. Invocation `(x, y)` owns four rows beginning at
`4*y` and eight columns beginning at `8*x`, represented by eight
`vec4<f32>` accumulators. It consumes the existing authenticated
`dit-gemm-n256-k32-tile-major-v1` packed-FP16 payload without changing the
converter, manifest, package, or persistent memory.

For every K16 panel, C performs exactly this sequence:

1. Stage activation panel `A` as
   `array<vec4<f16>, 16 * 17>`, indexed `k * 17 + y`. Each vector contains
   rows `4*y + 0..3` at one K. Every component is independently bounds-guarded
   in the final M tile and explicitly converted from storage FP32 to FP16;
   invalid rows stage positive zero. Specifically, invocation `(x, y)` loads
   K=`x` for rows `4*y + 0..3` and stores `input_panel[x * 17 + y]`.
   Compute invocation `(x, y)` later reads
   `input_panel[k * 17 + local_id.y]`. Element 16 of every K row is padding.
2. Stage weight panel `B` as
   `array<vec4<f16>, 16 * 33>`, indexed `k * 33 + n4`. Invocation `(x, y)`
   reads the same packed eight-weight record as OPT-0019, unpacks its low and
   high halves, and directly stores them at `y * 33 + 2*x` and `+1`.
   Element 32 of every K row is padding.
3. Execute one unconditional workgroup barrier.
4. Execute exactly four lexical K4 groups with bases `0`, `4`, `8`, and `12`;
   there is no scalar-K loop. A group loads four activation vectors and eight
   weight vectors. Within it, every accumulator receives four separate
   statements in K order `base+0`, `base+1`, `base+2`, `base+3`, each of the
   form
   `acc = acc + vec4<f32>(f32(a_component)) * vec4<f32>(b_f16_vector)`.
   Updates across different accumulators may be interleaved, but each one must
   observe that exact increasing-K dependency chain.
5. Execute one unconditional workgroup barrier before either panel is reused.
6. Store the same two output `vec4<f32>` values for each owned row, with the
   existing M2250 tail guard.

The candidate must not call `dot` or `fma`, split an output reduction across
threads, reorder K terms, retain K4 partial sums, change the packed payload,
or silently fall back under its identity. The intended kernel-set identity is
`opt-0021-m64-n128-k16-cooperative-vec4-panels-fp16-fp32-v1`.

## Exact layout and static accounting

WGSL gives `vec4<f16>` an 8-byte alignment, size, and array stride. Flattened
arrays therefore make the allocation explicit:

| Workgroup allocation | Elements | Bytes |
| --- | ---: | ---: |
| K-major activation `[16][17]` | `272 vec4<f16>` | `2,176` |
| K-major weight `[16][33]` | `528 vec4<f16>` | `4,224` |
| Total | `800 vec4<f16>` | `6,400` |

The weight K stride remains exactly `33 * 8 = 264` bytes, identical to
OPT-0019's `132 * 2 = 264` bytes. The activation panel changes orientation but
retains OPT-0019's 2,176-byte allocation. The 6,400-byte total remains below
the target's exposed 16,384-byte per-workgroup limit; no occupancy conclusion
follows from that API limit.

Per full workgroup/K16 panel, the source-level counts are exactly 256
activation `vec4<f16>` stores, 512 weight `vec4<f16>` stores, 4,096 activation
vector loads, 8,192 weight vector loads, 32,768 FP32 vector accumulator
updates, and two barriers. Each invocation performs 16 activation vector
loads, 32 weight vector loads, and 128 ordered vector updates, representing
512 scalar multiply/add terms. These are source-level counts, not emitted
instruction or bank-transaction claims.

Across the four shapes at production multiplicities `4/2/2/1`, B and C have
the same static geometry and traffic:

| Metric | Exact count |
| --- | ---: |
| Workgroups | `6,912` |
| K16 workgroup-panel iterations | `1,032,192` |
| Scheduled scalar multiply/add terms | `135,291,469,824` |
| Valid scalar multiply/add terms | `132,120,576,000` |
| Estimated global activation bytes | `4,128,768,000` |
| Estimated global weight bytes | `4,227,858,432` |
| Estimated global A+B bytes | `8,356,626,432` |
| Barrier events | `2,064,384` |

These are request and dispatch counts, not bandwidth, throughput, occupancy,
or product-speed claims.

## Frozen OPT-0020 fixture, reused by value

The implementation must reproduce OPT-0020's
`opt-0020-finite-fp16-cancellation-v1` generator byte for byte. It must not
import the old browser harness directly: that module auto-installs its page
gate whenever `document` exists. Copying the frozen pure generator into the
new harness is acceptable only with the contracts and hashes below; modifying
the recorded OPT-0020 harness is not.

The activation seeds, in shape order, are `0x31415926`, `0x27182818`,
`0x6a09e667`, and `0xbb67ae85`; weight seeds are `0x3c6ef372`,
`0xa54ff53a`, `0x510e527f`, and `0x9b05688c`. The only unsigned magnitude
patterns are `0x2411`, `0x28b5`, `0x2d53`, `0x31e7`, `0x356b`, and `0x39ad`.
The mix constants are subject `0x9e3779b1`, group `0x85ebca6b`, and offset
`0xc2b2ae35`. `mix32` is frozen as xor-right-16, multiply `0x7feb352d`,
xor-right-15, multiply `0x846ca68b`, xor-right-16, all in U32 arithmetic.

For `group=floor(k/4)` and `offset=k&3`, magnitude selection uses
`cancellationGroup = group % 8 == 7 ? group : group & ~1` and
`symmetricOffset = group % 16 == 0 ? 0 : (offset == 0 || offset == 3 ? 0 : 1)`.
The exact magnitude index is:

`mix32(seed ^ imul(subject+1, 0x9e3779b1) ^ imul(cancellationGroup+1, 0x85ebca6b) ^ imul(symmetricOffset+1, 0xc2b2ae35)) % 6`.

Activation sign is
`(mix32(activationSeed ^ imul(row+1, 0x9e3779b1)) >>> 31) ^ (group & 1)`.
Weight sign is
`(mix32(weightSeed ^ imul(column+1, 0x9e3779b1)) >>> 31) ^ (offset == 1 || offset == 2)`.
The selected magnitude is ORed with `sign << 15`. Activations are the exact
finite normal FP16 values stored in row-major FP32. Weights are written in
N256/K32/K-in-tile/N-in-tile physical order.

The repository fixture-manifest provenance SHA-256 remains
`cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`;
it is not the synthetic generator identity. The frozen OPT-0020 synthetic
fixture-declaration SHA-256 is
`954aff0a07dcc2946ac8191c054e2ecf63473d05ed9ebf81dc7db6d535f80f0c`.
The eight generated-byte identities are:

| Shape | Activation SHA-256 | Packed-weight SHA-256 |
| --- | --- | --- |
| `h-h` | `e66bc914da370971b0a717a3db8e9fa5b26820fe2d0bd5fa156b650f68fc5b99` | `6e985fca3119b135d740c2f8814fe3ddbf538bdaabde96cfca1dd363b62d45eb` |
| `h-1024` | `47b9e4ecef742a678bb263443e783b5ea753dc513af2b986c15d97bbafb315cf` | `3e4db4f1da0770dc2b2ffd6f2b8643e1778b2993495d4a2085bcc0bea7b383f3` |
| `h-6144` | `1469272232904084304f9834d5b9f1cf152ba940c7e25e82e0baac0708838c62` | `898266fc61785f391230299844c4e50e6394daa631783e3a40f95ba436cb088c` |
| `6144-h` | `39604dec14faf7d5f8a2c2400f48ce162cbce268de9b176ac7acf2e84612b1aa` | `e20d4d09edb58957967021c4f4957653be93ab3bcca5828d7a6c32403f2c24da` |

Before browser execution, static tests must regenerate all eight hashes and
prove the finite, non-power-of-two, exponent-band, alternating/seeded-sign,
within/across-K4 cancellation, and explicit K6144 coverage. They must also pin
the candidate source and all four generated shader hashes. A changed fixture,
source, mechanism, or shader requires a new experiment ID.

## Full-output raw-U32 gate

Use exactly M2250/K2048/N2048, M2250/K2048/N1024,
M2250/K2048/N6144, and M2250/K6144/N2048. For each shape execute
`A, B, C, C` over identical immutable fixture buffers. Every execution starts
from a qNaN-prefilled guarded FP32 output. Compare all `25,344,000` outputs,
including row 2249, and require raw-U32 equality for A/B, A/C-first, and
C-first/C-rerun. Thus A, B, and both C executions must share one output hash
per shape; there is no numerical tolerance or listening substitute for a bit
mismatch.

Also require finite complete writes, intact prefix/suffix canaries, no qNaN
prefill remaining, zero uncaptured GPU errors, immutable fixture hashes,
drain-before-release, idempotent cleanup, device destruction, and zero live
resources. The separate candidate owner must fail closed for missing
`shader-f16`, fewer than 256 invocations, X/Y limits below 16, less than 6,400
workgroup-storage bytes, unsupported shapes, invalid bindings, overlap,
destroyed state, or failed compilation. Cache identity, failed-promise
eviction, buffer bounds, and post-destroy rejection must be statically tested.

## One bounded target-browser timing gate

Complete allocation, upload, compilation, correctness, and one symmetric
warmup for every arm/shape before one thermal gate. Wait for at least 30
continuous nominal seconds from
`notifyutil -g com.apple.system.thermalpressurelevel`, sampled nominally every
1,000 ms with no gap above 1,250 ms, and launch timing within 5,000 ms of the
gate completion. Keep external thermal logging continuous through cleanup.
Do not rerun an unchanged candidate for a more favorable trace.

Use OPT-0020's six rounds. Rotate shape order left by `round mod 4`; use arm
orders `ABC`, `ACB`, `BAC`, `BCA`, `CAB`, and `CBA`. Every arm occupies every
position twice and receives six retained samples per shape. One sample is one
dispatch in one command buffer followed by its matching completion fence;
allocation, upload, compilation, warmup, and readback are outside it. The
median is the unrounded mean of the third and fourth sorted samples.

For each arm, define the complete score as

`4*m(h-h) + 2*m(h-1024) + 2*m(h-6144) + m(6144-h)`

and report the feed-forward score `2*m(h-6144) + m(6144-h)` separately.

C qualifies only if all correctness, lifecycle, fixture, thermal, and timing
conditions pass and, using fresh unrounded medians:

- C is faster than B on every shape;
- `B complete / C complete >= 1.075`; and
- `A complete - C complete >= 52.0834 ms`.

The last condition projects at least `52.0834 * 192 = 10,000.0128 ms` of
dense saving. Historical OPT-0019/0020 medians are sizing evidence only. There
is deliberately no inherited 1.55x A/C ratio: this new ID tests the owner's
absolute ten-second objective, while the fresh 1.075 B/C guard requires a
material gain attributable to vector panels. Thresholds may not be rounded,
loosened, or redefined after observation.

## Stop and escalation boundaries

If any primitive condition fails, record the evidence and stop without a
nearby layout, padding, unroll, or geometry variant, package load, integration,
C98/M2250 generation, or listening run under this ID.

A primitive pass authorizes only a separately frozen authenticated
package-native repeated-layer/subgraph gate. It must preserve every output
U32, prove bounded pipeline/dispatch reuse, cancellation, memory, and cleanup,
and use real package bytes. Only that pass may authorize an explicit candidate
production profile and one DiT-only C98/M2250 gate.

The C98 decision requires every retained trajectory tap and the final latent
to be raw-U32 exact, and complete graph submit-through-drain wall at most
`52,148.19999909401 ms`, ten seconds below the frozen OPT-0018 authority of
`62,148.19999909401 ms`. Generation wall is reported but cannot replace the
graph metric. Because the candidate is exact, no listening gate is required
if the package-layer, trajectory, and final-latent U32 gates remain exact. No
VAE decode, waveform, short-song, three-minute song, under-one-minute, or
release claim is authorized by registration or a primitive pass.

## Known implementation risks, not alternate arms

- WGSL alignment is not a blocker: `vec4<f16>` has natural 8-byte alignment
  and stride, and both flattened array sizes are exact. Compilation must still
  be checked on the target browser before any timed sample.
- WGSL does not specify workgroup-memory banks. Odd vector strides 17 and 33
  are deliberate, and the weight panel preserves OPT-0019's byte addresses,
  but neither conflict freedom nor occupancy is claimed. The balanced browser
  gate decides whether vector access helps.
- The K-major activation orientation changes staging access and requires four
  independent tail guards. Tests must prove that no invalid last-tile load is
  relied upon, even if robust buffer access would mask it.
- Exact source order strongly predicts OPT-0019 identity, but compiler
  lowering is not an oracle. Full-output raw-U32 equality is mandatory.
- The frozen OPT-0020 browser module has a top-level `document`-guarded gate
  installer. Directly importing it into the new browser harness would start
  the wrong gate; reproduce only its pure fixture algorithm and verify the
  frozen hashes by value.

## Result

### Authenticated execution

The bounded A/B/C gate ran once in the target browser and stopped at the
primitive boundary. Registration is commit
`7dcbe50c7f04d1e07f6b30657da96372ca8574d1`; the benchmark core and harness
are commit `fa446366fa404e5ce00cf7350c206f7a63ba791b`. The persisted receipt is
72,064 bytes with SHA-256
`607e3173586ed46023d1f55fb7060bed755ab273930701e4010c7e64c15a08b6`;
the continuous thermal trace is 4,322 bytes with SHA-256
`fe482a89246148be4fe76af169f472b4fe4af42f5ee74d7a5925a7f4d12f761f`.
The canonical result and its focused contract bind those raw artifacts.

The executable was Google Chrome `151.0.7922.138` (reduced user agent
`151.0.0.0`) on macOS `26.5.2` build `25F84`, MacBook Air `Mac15,12`, Apple
M3 with 10 GPU cores and 16 GiB unified memory. The adapter reported Apple
Metal 3, fixed subgroup size 32, `shader-f16`, WG256, 16,384 bytes of
workgroup storage, a 268,435,456-byte maximum buffer, and a 134,217,728-byte
maximum storage binding.

The frozen source SHA-256 identities were:

- A, current OPT-0009:
  `a238f67da07c6ba1097da9d9e9e97960ae97d2e1d5c129fcbabf69e962cbb6b3`;
- B, exact OPT-0019:
  `b5dad12724882d3fc942c7df7b10c7b7b89a4bed595125ff11a5905c03152a37`,
  kernel set `opt-0019-m64-n128-k16-cooperative-fp16-fp32-v1`;
- C, OPT-0021 vector panels:
  `2229c55f8b7fe66d3770ef7683de68322632d17749fe2d3085d5d46dcdc22df1`,
  kernel set
  `opt-0021-m64-n128-k16-cooperative-vec4-panels-fp16-fp32-v1`; and
- TypeScript harness, HTML entry, pre-execution contract, and candidate-core
  contract:
  `e0a019be08bc6222594de5e5083cc8cef1395d8395ed003661bc1e71c8cea055`,
  `84fb23733bc30a6d37d0e40418f9fee22a37c0a87c13be3e15e81c75f1690ffa`,
  `f93bfea55377acce21e264e120a45bca5b2a867fa54a75b6a9fe0dd098ac9563`,
  and `61505a4c520b9107a7f529192f58013e48ac6c95317fb98dbd98abfffd0a5806`.

The browser and static contracts authenticated the unchanged synthetic
generator declaration, all eight fixture-byte hashes listed above, and all 12
generated-shader hashes. No production package was loaded; no converter,
manifest, persistent layout, runtime selector, or production profile changed.

### Exact correctness result

All exactness and lifecycle gates passed. Across 16 executions, the gate
compared every output word three ways: A versus B, A versus C-first, and
C-first versus C-rerun. All `76,032,000` raw-U32 comparisons matched. The four
executions for each shape shared one output hash, and all four aggregate hash
manifests were
`73987cdad84f0c37bb206965f3f87f707f95bb0345920fb9830b2d5d86c404b4`.
Every qNaN-prefilled output was finite and completely written, prefix and
suffix canaries and row 2249 were intact, and there were zero uncaptured GPU
errors. Because C was bit-identical, no numerical tolerance or listening gate
was applicable.

Cleanup drained before release, destroyed all 20 created buffers, reached zero
live resources and bytes, destroyed the device, and remained idempotent on a
second call. Logical GPU high-water was 311,749,632 bytes.

### Negative timing result

The single authoritative timing run retained all 72 samples: six for every
arm and shape under the frozen six permutations and rotated shape order. Every
sample was one command buffer, one dispatch, and one matching queue-completion
fence. The recomputed median-six values were:

| Shape | A (ms) | B (ms) | C (ms) | B / C | C faster than B |
| --- | ---: | ---: | ---: | ---: | --- |
| `h-h` | `17.25` | `13.649999976158142` | `14` | `0.9749999982970101x` | no |
| `h-1024` | `6.75` | `7.350000023841858` | `7.149999976158142` | `1.0279720347343526x` | yes |
| `h-6144` | `45.75` | `37.799999952316284` | `38.65000003576279` | `0.9780077598276844x` | no |
| `6144-h` | `53.60000002384186` | `40.09999996423721` | `39.05000001192093` | `1.026888603124091x` | yes |

The complete `4/2/2/1` scores were A `227.60000002384186 ms`, B
`184.99999982118607 ms`, and C `186.6500000357628 ms`. C was
`1.6500002145767212 ms`, or about 0.89%, slower than B. B/C was
`0.991159923845376x` against the frozen `1.075x` minimum. C also beat B on
only two of four shapes, so the per-shape condition failed.

A minus C was `40.94999998807907 ms` against the frozen `52.0834 ms`
minimum. Its 192-layer-evaluation projection is `7,862.399997711182 ms`,
`2,137.612802288819 ms` short of the required `10,000.0128 ms`. The
feed-forward `0/0/2/1` scores were A `145.10000002384186 ms`, B
`115.69999986886978 ms`, and C `116.3500000834465 ms`, for B/C
`0.9944134059810009x`.

The external trace contained 76 consecutive nominal observations over
75,659 ms with a maximum 1,013 ms gap. Its frozen gate interval contained 60
observations over 59,507 ms with the same maximum gap and launched timing 233
ms later. Logging remained nominal through cleanup and continued another
13,922 ms after cleanup. There was no unchanged-candidate retry.

### Disposition and lesson

All three frozen performance conditions failed, so the registered stop rule
fired. No package-native layer gate, runtime integration, C98 trajectory,
M2250 product generation, VAE decode, waveform, listening comparison, or
three-minute run was performed or authorized.

This exact result closes the specific hypothesis that changing OPT-0019's
scalar cooperative panels into the registered padded K-major `vec4<f16>`
representation can supply the missing material gain while retaining the same
geometry, traffic, barriers, and strictly ordered reduction. It was essentially
performance-neutral in the complete score and mixed across shapes. That does
not prove scalar workgroup access is free or identify emitted instructions or
bank transactions; it shows only that this exact source-level vector-panel
mechanism does not move the target. Revisit it only after a material browser
compiler or target-GPU change. A future dense attempt must receive a new ID and
change the mechanism materially rather than sweep a nearby padding, unroll, or
panel-layout variant under OPT-0021.
