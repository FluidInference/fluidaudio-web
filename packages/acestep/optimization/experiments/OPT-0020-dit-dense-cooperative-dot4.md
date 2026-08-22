# OPT-0020 — DiT dense cooperative dot4

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Date: 2026-08-14
- Author/agent: Codex
- Risk class: `reordered-rounding`; FP16 operands are explicitly widened to
  FP32, but each output consumes consecutive increasing-K groups of four with
  `dot(vec4<f32>, vec4<f32>)` instead of OPT-0019's scalar increasing-K FP32
  addition sequence

## Identity and inherited authority

- Allocation baseline and latest recorded-result authority:
  `7f8db6c5cfb17b2c508669e1b136d1bb02bab239`
- Evaluated production and OPT-0018 C98 profiler authority:
  `f92de5a209ebb5f05ba9b37e5f3b7bfc88633d82`
- OPT-0018 compact receipt: 6,070 bytes, SHA-256
  `196d5ce3991a08bb6f065e5c2799ae9a55e1509e5a0849fff206c6775d9d8e83`
- OPT-0018 continuous thermal trace: 32,183 bytes, SHA-256
  `dd37450942d85b71ee05ead7bac78a61e988a72c64aa0705a16f3350fb11d511`
- Frozen OPT-0018 graph submit-through-drain authority:
  `62,148.19999909401 ms`
- OPT-0019 registration commit:
  `83de738b5374699778dcaa373d69118a7fbd6715`
- OPT-0019 benchmark core and harness commit:
  `8900ce670271cbd227142c584fb4a917e5d9cfb9`
- OPT-0019 result commit:
  `7f8db6c5cfb17b2c508669e1b136d1bb02bab239`
- OPT-0019 exact cooperative source SHA-256:
  `b5dad12724882d3fc942c7df7b10c7b7b89a4bed595125ff11a5905c03152a37`
- OPT-0019 canonical result: 6,376 bytes, SHA-256
  `1c38f503503ea621fab7f8b37be3c58854f3c44d05c9cf2241c19195b4ae30ac`
- OPT-0019 raw receipt: 14,126 bytes, SHA-256
  `a2434a16dae3db3936202461a6da6548009148c739da58dd80f458931f55471d`
- OPT-0019 continuous thermal trace: 8,590 bytes, SHA-256
  `ebb0e4000e68338cd29cd5a888e0ee635cb95f54a745eec25612d2bf26e4bf9f`
- Target: MacBook Air `Mac15,12`, Apple M3 with 10 GPU cores and 16 GiB
  unified memory. The browser, OS, adapter, package, fixture, generator, and
  implementation identities must be captured by the result.

OPT-0018 remains decision-useful but formally pending only because its full
authenticated live-browser receipt still requires manual download. This
experiment relies on its persisted compact aggregate, frozen source
classifier, and exact graph authority. It does not claim that the physically
mixed command wall was decomposed into measured dense time.

## Why this experiment exists

OPT-0019 established that cooperative panels are useful but insufficient in
their exact arithmetic form. On one thermally valid target-browser gate, its
WG256 M64xN128xK16 candidate was raw-U32 exact and faster on all four M2250
production shapes. The complete `4/2/2/1` score improved from
`223.20000022649765 ms` to `169.69999998807907 ms`, or
`1.3152622288873117x`, while the feed-forward `0/0/2/1` score improved from
`138.79999989271164 ms` to `109.99999994039536 ms`, or
`1.2618181815265623x`. It missed OPT-0019's frozen `1.55x` complete-score
threshold, so that exact candidate correctly stopped before package or C98
escalation.

The exact kernel still executes 16 serial scalar multiply/add updates for each
output per K16 panel. Transposing the same cooperative weight panel permits
four consecutive K terms to be explicitly widened and submitted as one FP32
`dot` operation. This changes rounding, but it may shorten the dependency and
instruction chain without giving back OPT-0019's global-operand reuse.

A side reference lab observed approximately `1.67x` for its dot4 mechanism
against its exact cooperative counterpart. That result is motivation only: it
did not reproduce this authenticated browser harness, fixture, command
topology, thermal protocol, or production score, and is not transferable as a
speed claim or acceptance threshold. MLX's fast native Metal
SIMD-group-matrix mechanism has no currently available WGSL equivalent here.
ONNX Runtime Web's WGSL tiling is a design reference, not measured proof for
this candidate or target.

## Frozen three-arm screen

The primitive screen has exactly three arms:

- **A — current OPT-0009:** the current WG128 M32xN256xK32 subgroup FP16-input,
  FP32-accumulation owner.
- **B — exact OPT-0019:** the frozen WG256 M64xN128xK16 cooperative panel with
  strictly increasing scalar-K FP32 `sum + a * b` arithmetic.
- **C — OPT-0020 dot4:** the same WG256 M64xN128xK16 ownership, payload,
  dispatch geometry, and global loads as B, with only the workgroup weight
  orientation and reduction primitive changed as declared below.

No other tile, workgroup size, dot width, panel depth, accumulator shape,
subgroup variant, matrix primitive, or native-FP16 accumulation arm belongs to
OPT-0020. There is no geometry sweep.

## Frozen candidate mechanism

C retains B's 16x16 invocation grid. Each thread owns four rows by eight
columns and therefore 32 FP32 output accumulators. For every K16 panel:

1. Load the identical FP16 `A[64][17]` workgroup panel used by B. Activations
   enter as FP32 and undergo the same explicit FP16 conversion before staging.
2. Load exactly the same packed native-FP16 weight records, but transpose them
   into an output-major `B[128][17]` workgroup panel.
3. Execute one uniform workgroup barrier.
4. Visit K4 bases `0`, `4`, `8`, and `12` in that order. Explicitly widen the
   four consecutive activation operands and four consecutive weight operands
   to `vec4<f32>`, then update each scalar output as
   `acc = acc + dot(activation4, weight4)`.
5. Execute one uniform workgroup barrier before either panel is overwritten.

The input panel consumes `64 * 17 * 2 = 2,176` bytes and the transposed weight
panel consumes `128 * 17 * 2 = 4,352` bytes, for exactly `6,528` workgroup
bytes. Both barriers are unconditional, every valid output sees increasing K4
groups, and the M2250 tail store remains guarded. C must not use a split
reduction, cross-thread output reduction, `fma`, native-FP16 accumulation,
clamp, saturation, alternate persistent weight layout, repack, or silent
fallback under an existing kernel identity.

C consumes the authenticated `dit-gemm-n256-k32-tile-major-v1` package layout
without a converter, manifest, package, or persistent-memory change. It must
be a separate benchmark-only owner and fail closed unless `shader-f16`, WG256
with a 16x16 workgroup, and at least 6,528 workgroup-storage bytes are
available. It accepts only M2250 and the four production K/N shapes below.
Cache identity, failed compilation eviction, buffer bounds and disjointness,
destruction, and post-destroy rejection must be pinned by static tests.

## Static resource and traffic model

The screen covers these exact shapes:

- M2250/K2048/N2048;
- M2250/K2048/N1024;
- M2250/K2048/N6144; and
- M2250/K6144/N2048.

B and C have identical workgroups, scheduled work, global operand requests,
and barriers. Their only static resource difference is B's 6,400-byte
K-major panel pair versus C's 6,528-byte transposed-weight panel pair.

| Metric, weighted `4/2/2/1` | B and C |
| --- | ---: |
| Workgroups | `6,912` |
| Scheduled MAC | `135,291,469,824` |
| Valid MAC | `132,120,576,000` |
| Estimated global activation bytes | `4,128,768,000` |
| Estimated global weight bytes | `4,227,858,432` |
| Estimated global A+B bytes | `8,356,626,432` |
| Barrier events | `2,064,384` |

These are static request and dispatch counts, not bandwidth, occupancy,
throughput, stage-wall, or product-speed claims.

## Fixture freeze before browser execution

Before any target-browser execution, the implementation checkpoint must freeze
one deterministic seeded generator for all four shapes. It must generate only
finite FP16 bit patterns for packed weights and FP16-exact activation values
stored in FP32. The generator must include:

- non-power-of-two mantissas;
- alternating and seeded pseudorandom signs;
- multiple bounded finite exponent bands;
- deliberate cancellation within and across consecutive K4 groups; and
- explicit K6144 long-reduction coverage rather than treating the long-K
  production shape as another benign random fixture.

The implementation checkpoint must pin the algorithm, every seed and
constant, and the per-shape activation and packed-weight SHA-256 identities in
source and static-contract tests. It must also prove that the fixtures contain
the declared sign, exponent, non-power-of-two, cancellation, and long-K
classes. Final hashes are intentionally not invented in this registration
record. Neither generator, fixture bytes, arm implementation, metric formula,
threshold, nor timing order may change after the first target-browser
execution. A changed fixture or candidate requires a new experiment ID, not a
retry under OPT-0020.

## Full-output correctness and numerical gate

For each shape, run A once, B once, and C twice over identical immutable input
and weight bytes. Every execution starts from a qNaN-prefilled guarded FP32
output. Compare all outputs, including row 2249, and require complete finite
writes, intact prefix/suffix canaries, zero uncaptured GPU errors, deterministic
C rerun U32 hashes, drain-before-release, idempotent cleanup, and zero live
resources.

A and B must be raw-U32 exact on every output. The two C executions must be
raw-U32 exact to each other. Compare C numerically with A over every output of
every shape: `25,344,000` F32 values per C execution, with no sampling. Report
per-shape and aggregate counts, ranges, hashes, maximum/mean/RMS error, first
and worst locations, a complete sign-aware F32 ULP-distance distribution, and
signed-zero differences.

For a control vector `a`, candidate vector `c`, and error `e = c - a`, define:

- RMS as `sqrt(sum(e^2) / count)`;
- NRMSE as `RMS / max(sqrt(sum(a^2) / count), 1e-12)`;
- SNR as `20 * log10(controlRMS / errorRMS)`, with exact equality reported as
  positive infinity;
- Pearson correlation from the complete vectors using FP64 CPU statistics;
  and
- relative maximum error as
  `max(abs(e)) / max(max(abs(a)), 1e-6)`.

Each shape independently and the aggregate must satisfy all of these frozen
limits:

- NRMSE at most `1e-5`;
- SNR at least `100 dB`;
- Pearson correlation at least `0.999999`;
- relative maximum error at most `1e-3`; and
- maximum absolute error at most `1e-4`.

All five limits are conjunctive. ULP distance and signed-zero differences are
reported diagnostics and have no separate pass threshold. Thresholds may not
be loosened, redefined, or selectively aggregated after output is observed.

## One target-browser timing gate

Complete allocation, upload, shader compilation, all full-output correctness,
and one symmetric warmup of every arm/shape before the thermal gate. Then wait
once for at least 30 continuous nominal seconds and run one timing gate with
continuous external thermal logging through cleanup. Do not repeat an
unchanged screen to obtain a prettier thermal trace. Later thermal pressure is
retained and disclosed rather than erased by a retry.

Use six timing rounds. Let shapes `0..3` be ordered as listed above and rotate
their order left by `round mod 4`. Within each shape, use these exact arm
permutations by round:

1. `A, B, C`;
2. `A, C, B`;
3. `B, A, C`;
4. `B, C, A`;
5. `C, A, B`; and
6. `C, B, A`.

Thus every arm occupies each order position exactly twice and each arm/shape
has six retained samples. One sample is exactly one dispatch in one command
buffer followed by its matching queue-completion fence. No readback,
allocation, upload, compilation, or warmup belongs in the sample. Retain the
raw sample values, round, rotated shape order, arm order, submit/fence
timestamps, thermal coverage, preparation walls, memory high-water, and
cleanup. For an arm/shape median, sort its six samples and average the third
and fourth values without rounding.

For each arm define the complete production-dense score:

`4*m(K2048,N2048) + 2*m(K2048,N1024) + 2*m(K2048,N6144) + m(K6144,N2048)`.

Define the feed-forward projection score:

`2*m(K2048,N6144) + m(K6144,N2048)`.

The historical OPT-0019 samples imply B complete and feed-forward scores of
`169.69999998807907 ms` and `109.99999994039536 ms`; A's historical complete
score was `223.20000022649765 ms`. They are provenance and sizing evidence,
not substitutes for the fresh balanced A/B/C medians.

## Primitive qualification and stop rule

C qualifies only if all correctness, determinism, lifecycle, and numerical
conditions pass and, using the fresh unrounded medians:

- C is faster than B on every one of the four shapes;
- B's complete score divided by C's complete score is at least `1.25`;
- A's complete score divided by C's complete score is at least `1.55`; and
- B's feed-forward score divided by C's feed-forward score is at least `1.25`.

Every condition is required. A result close to a threshold is not rounded into
a pass, and the side-lab result cannot override a miss.

If any condition fails, record the negative evidence and stop. Do not sweep a
nearby geometry, change dot width, alter the generator, rerun for thermal
appearance, load a production package, integrate C, or run C98/M2250 under
this ID.

## Escalation after a primitive pass

A primitive pass authorizes only a separately registered authenticated
package-native repeated-layer/subgraph gate. It does not authorize a production
selector or product profile. Before that gate is implemented, its complete
real-layer numerical tolerances and retained taps must be frozen; it must use
real package bytes and prove bounded compilation, dispatch reuse, memory,
cancellation, and cleanup.

Only a package-layer pass may authorize a separately registered explicit C98
candidate trajectory. That gate must compare every retained denoise-evaluation
tap and the final latent against the current authenticated control under
predeclared trajectory tolerances. It must also preserve finite deterministic
execution, cancellation, and lifecycle behavior and reduce complete graph
submit-through-drain wall from `62,148.19999909401 ms` to at most
`52,148.19999909401 ms`. Generation and stage walls are reported but cannot
replace this graph decision metric.

Only a C98 trajectory and graph pass may proceed to matched control/candidate
raw-waveform validation before ordinary `-1 dBFS` normalization. Because C
changes DiT reduction rounding, numerical evidence alone cannot select it for
production: fresh level-matched instrumental and vocal listening candidates
must receive explicit owner approval. The same VAE and all other product math
must remain fixed across that comparison.

Only after the package layer, C98 trajectory and graph, raw waveform, and fresh
listening gates all pass may an actual 180-second direct product generation be
run. It uses one nominal thermal start with no unchanged-candidate retry and
must report complete wall, stage attribution, graph drains, memory,
responsiveness, cancellation, cleanup, waveform, and listening evidence. No
approximate profile becomes the product default before that complete sequence.

## Non-claims

- OPT-0020 does not claim that WGSL `dot` maps to Metal MMA or any particular
  machine instruction.
- Equal B/C traffic and barriers do not project a wall-time speedup.
- A primitive pass would not establish a real-layer, trajectory, waveform,
  listening, C98 integration, 180-second, or under-one-minute result.
- Registration changes no production kernel, package, selector, profile,
  default, arithmetic, or output.

## Result

### Authenticated execution

The bounded A/B/C gate ran once in the target browser and stopped at the
primitive decision boundary. Registration is commit
`fce77739841572942eca4e96cc6a9f48eb02a971`; the benchmark core and harness
are commit `1d825a1399fdcc23c4ef3ce18151a2efa2415626`. The persisted receipt is
90,410 bytes with SHA-256
`8c168677ee630a4ee337e7b3e7e2c0d8c1c2b53493ce6eebdc48765cdb554679`;
the continuous thermal trace is 6,212 bytes with SHA-256
`cd5461cb760b9e6022a0f8b78105661d1cb17d7e7dd525f415fc7e60298de960`.
The concise canonical result and its focused contract bind those raw
artifacts rather than duplicating their complete sample and ULP payloads.

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
- C, OPT-0020 dot4:
  `466cf7b4c8f860ff55a89b03a5bb2ead99c0d849420918e11d6888e11482d28e`,
  kernel set
  `opt-0020-m64-n128-k16-cooperative-dot4-fp16-fp32-v1`; and
- TypeScript harness, HTML entry, and pre-execution static contract:
  `f2a7f78ef112481b9c4423ca88366a449c02eaf925524121a6f32d782bedda72`,
  `19b226945d8c02329f5ee0e34a39fa1d393f3b8eb20f8e77bf4b8f4992c79b20`,
  and `b2e2970c4f74abffb2c9b72e9e2eb5264cd4d16b6fba855a7d1d45aee61d7205`.

The browser authenticated the fixed generator and all four activation/weight
pairs. In shape order `h-h`, `h-1024`, `h-6144`, `6144-h`, their SHA-256
pairs were respectively:

- `e66bc914da370971b0a717a3db8e9fa5b26820fe2d0bd5fa156b650f68fc5b99` /
  `6e985fca3119b135d740c2f8814fe3ddbf538bdaabde96cfca1dd363b62d45eb`;
- `47b9e4ecef742a678bb263443e783b5ea753dc513af2b986c15d97bbafb315cf` /
  `3e4db4f1da0770dc2b2ffd6f2b8643e1778b2993495d4a2085bcc0bea7b383f3`;
- `1469272232904084304f9834d5b9f1cf152ba940c7e25e82e0baac0708838c62` /
  `898266fc61785f391230299844c4e50e6394daa631783e3a40f95ba436cb088c`;
  and
- `39604dec14faf7d5f8a2c2400f48ce162cbce268de9b176ac7acf2e84612b1aa` /
  `e20d4d09edb58957967021c4f4957653be93ab3bcca5828d7a6c32403f2c24da`.

### Correctness and numerical result

All correctness and numerical conditions passed. Across 16 executions, A and
B were raw-U32 exact over all `25,344,000` output words, and C's first and
second executions were raw-U32 exact over another `25,344,000` words. All
qNaN-prefilled outputs were finite and complete, both canaries and every tail
row were intact, there were no uncaptured GPU errors, and the aggregate raw
output manifests were identical for A/B and independently identical for C's
two runs. C differed numerically from A at `24,304,155` of `25,344,000`
positions, as expected for the declared reordered reduction.

The complete C-versus-A aggregate was NRMSE
`9.336650318270838e-7`, SNR `120.59617812682795 dB`, Pearson correlation
`0.9999999999994078`, relative maximum error
`0.000002798660623218186`, and maximum absolute error
`0.0000858306884765625`. Those values passed the frozen `1e-5`, `100 dB`,
`0.999999`, `1e-3`, and `1e-4` thresholds, respectively, and every shape
passed the same five gates independently. No listening gate was required or
performed because the primitive failed its timing gate and never escalated.

### Negative timing result

The single authoritative run retained all 72 dispatch samples: six samples
for every arm/shape, using the six frozen arm permutations and rotated shape
order. Every sample remained exactly one command buffer, one dispatch, and
one matching queue drain. Median-six values in milliseconds were:

| Shape | A | B | C | B / C |
| --- | ---: | ---: | ---: | ---: |
| `h-h` | `16.949999928474426` | `14.950000047683716` | `21.600000023841858` | `0.6921296310732435x` |
| `h-1024` | `8.349999964237213` | `6.800000011920929` | `11.649999916553497` | `0.5836909923285751x` |
| `h-6144` | `43` | `35.30000001192093` | `64.34999996423721` | `0.548562549052666x` |
| `6144-h` | `47.5` | `36.000000059604645` | `64.25000005960464` | `0.5603112844545913x` |

C was slower than B on every shape. The complete `4/2/2/1` scores were A
`217.99999964237213 ms`, B `180.00000029802322 ms`, and C
`302.6499999165535 ms`. B/C was `0.5947464078891549x` against the frozen
`1.25x` minimum, while A/C was `0.7203039805137254x` against `1.55x`. The
feed-forward `0/0/2/1` scores were A `133.5 ms`, B
`106.6000000834465 ms`, and C `192.94999998807907 ms`; B/C was
`0.5524747348537575x` against `1.25x`. All four timing conditions therefore
failed rather than merely missing one aggregate threshold.

### Thermal, lifecycle, and disposition

Compilation, upload, full-output correctness, and symmetric warmup completed
before one continuous nominal thermal gate. Its 93 observations covered
92,864 ms, had zero non-nominal observations, a 1,031 ms maximum poll gap,
and a 752 ms launch delay. The 109-observation trace remained continuous and
nominal through timing and cleanup. There was no unchanged-candidate retry.

Logical GPU high-water was 311,749,632 bytes. Cleanup drained before release,
destroyed all 20 created buffers, reached zero live buffers and bytes, was
idempotent on its second call, and destroyed the device on the first call.

This is negative evidence and the candidate is abandoned. The frozen stop
rule fired: no package-native layer gate, production integration, C98
trajectory, M2250 product generation, waveform comparison, listening gate, or
product-speed claim was authorized or performed. Revisit only after a
materially changed browser compiler or GPU changes WGSL dot lowering, or as a
newly registered mechanism that materially changes more than the same dot4
reduction under this geometry.
