# OPT-0019 — DiT dense cooperative panels

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Date: 2026-08-14
- Author/agent: Codex
- Risk class: `exact`; the candidate must preserve each output's increasing-K
  FP32 `sum + f32(f16(a)) * f32(f16(b))` sequence

## Identity

- Registration baseline and current production authority:
  `f92de5a209ebb5f05ba9b37e5f3b7bfc88633d82`
- Current production dense owner SHA-256:
  `a238f67da07c6ba1097da9d9e9e97960ae97d2e1d5c129fcbabf69e962cbb6b3`
- OPT-0018 compact receipt: 6,070 bytes, SHA-256
  `196d5ce3991a08bb6f065e5c2799ae9a55e1509e5a0849fff206c6775d9d8e83`
- OPT-0018 continuous thermal trace: 32,183 bytes, SHA-256
  `dd37450942d85b71ee05ead7bac78a61e988a72c64aa0705a16f3350fb11d511`
- Target: MacBook Air `Mac15,12`, Apple M3 with 10 GPU cores and 16 GiB
  unified memory; the exact browser, OS, adapter, package, and fixture
  identities must be captured by the result

## Why this experiment exists

The current C98/M2250 production profile makes dense projection the only
single kernel owner with a credible path to at least ten seconds of DiT saving.
The persisted compact OPT-0018 receipt is 6,070 bytes with SHA-256
`196d5ce3991a08bb6f065e5c2799ae9a55e1509e5a0849fff206c6775d9d8e83`.
It reports 62,148.19999909401 ms across all graph drains:

- pure feed-forward: 26,860.299998641014 ms;
- pure attention projections: 1,625.7000002861023 ms; and
- physically mixed commands: 21,446.5 ms.

An exact descriptor decomposition accounts for all 1,344 mixed command
buffers and 4,992 members. Five of their seven repeated patterns contain the
same OPT-0009 dense owner. Those 960 dense members contribute
8,233,452,306,432 of the mixed commands' 8,606,589,714,432 scheduled MACs
(95.6645%). The mixed wall remains indivisible and is not reported as dense
time.

OPT-0018 is decision-useful but not formally closed: its complete live browser
receipt still requires the owner's manual download because browser security
blocked automated Blob export. This experiment relies only on the persisted
compact aggregate, frozen source classifier, and static descriptor
decomposition. It makes no claim that the mixed wall has already been split.

## Hypothesis

The current WG128 M32xN256xK32 FP16/FP32 kernel assigns four fixed-32
subgroups to four row bands over the same 256 columns. Every subgroup therefore
loads the same packed K/N weight addresses, while each lane retains 64 FP32
accumulators. A cooperative WG256 M64xN128xK16 kernel can instead:

- use a 16x16 thread grid with each thread owning four rows by eight columns;
- retain 32 FP32 accumulators per thread while keeping 8,192 outputs per full
  workgroup;
- stage one A64x17 FP16 panel and one B16x132 FP16 panel in 6,400 bytes of
  padded workgroup storage;
- load each packed weight value once per workgroup rather than once per
  subgroup; and
- consume K in its original increasing order with explicit FP32 add and
  multiply, without `dot`, `fma`, split reductions, or native-FP16
  accumulation.

The authenticated `dit-gemm-n256-k32-tile-major-v1` payload already stores
`[N/256,K/32,K-in-tile,N-in-tile]`. An N128/K16 candidate panel is an exact
quadrant of that layout, so the screen requires no converter, manifest,
package, repack, or persistent-memory change.

Across the registered nine dense projections per layer, static accounting is:

| | Current OPT-0009 | Candidate |
| --- | ---: | ---: |
| Tile / workgroup | M32xN256 / WG128 | M64xN128 / WG256 |
| FP32 accumulators per thread | 64 | 32 |
| Weighted workgroups per layer | 6,816 | 6,912 |
| Weighted scheduled MAC per layer | 133,412,421,632 | 135,291,469,824 |
| Estimated global A+B request bytes per layer | 35.417 GB | 8.357 GB |
| Workgroup storage | 0 | 6,400 B |

The candidate accepts 1.4085% more padded-row MACs and two unconditional
barriers per K16 panel. The global-request estimate is a static hypothesis,
not a bandwidth or wall-time claim.

## Implementation boundary

The first checkpoint adds a separate benchmark-only kernel owner and focused
tests. It must not change the existing OPT-0009 source, package, production
profile, graph packing, scheduler, diagnostics, selector, or default.

The candidate must fail closed unless `shader-f16`, at least 256 total compute
invocations, workgroup X/Y limits of at least 16, and at least 6,400 bytes of
workgroup storage are available. It accepts only M2250 and the four
authenticated production K/N shapes. K and N are exact multiples of the
candidate panels; the final M tile is guarded. Pipeline cache keys must cover
the complete specialized geometry, and failed compilation or destruction must
not publish a dispatch.

## Primitive correctness and timing gate

Compare the current OPT-0009 owner and the candidate on these exact shapes:

- M2250/K2048/N2048;
- M2250/K2048/N1024;
- M2250/K2048/N6144; and
- M2250/K6144/N2048.

Report both the complete production-layer multiplicities `4/2/2/1` and the
three feed-forward projections: two K2048/N6144 gate/up operations plus one
K6144/N2048 down operation.
For each shape, use identical deterministic native-FP16 packed weights and
FP32 activations, qNaN-prefilled guarded FP32 outputs, and sequential resource
reuse. Compare every output U32, include the M tail, require complete writes,
finite outputs, canaries, deterministic reruns, zero uncaptured errors, and
zero live resources. Any mismatch rejects the exact mechanism; it is not
converted into an approximate candidate under this ID.

Compile and warm each arm once before the thermal gate. After correctness,
perform one target-browser timing screen after at least 30 continuous nominal
seconds. Use four timing rounds. Rotate the four shape order left by the round
index; use A/B arm order in rounds zero and two and B/A in rounds one and
three. Each sample is one equivalent command buffer followed by its completion
fence. For each arm and shape, sort its four samples and define the median as
the mean of the middle two. Keep raw samples and order.

Define the complete production-dense score per arm as:

`4*m(K2048,N2048) + 2*m(K2048,N1024) + 2*m(K2048,N6144) + m(K6144,N2048)`.

Define the separately reported feed-forward projection score as
`2*m(K2048,N6144) + m(K6144,N2048)`. Use identical command topology and
continuous thermal logging through cleanup. Do not repeat an unchanged run to
obtain a prettier thermal trace or change the statistic after observing it.

The primitive candidate qualifies only if:

- every shape is faster than the current owner without rounded comparisons;
- the complete `4/2/2/1` production-dense score is at least 1.55x faster and
  saves at least 52.0834 ms per layer-evaluation; and
- preparation, memory, cleanup, and later thermal transitions are reported.

OPT-0009's exact-shape medians give the complete nine-projection score as
150.5 ms per layer-evaluation. Reducing it to at most 98.4166 ms saves more
than 52.0834 ms per layer-evaluation, or ten seconds across the 192 production
layer evaluations. The 1.55x ratio is a small guard above that historical
1.5292x break-even. This is a primitive projection, not an observed stage
saving; mixed-command reach is retained as upside and the later DiT-only gate
must still observe the full ten seconds.

## Escalation and stop rules

If the primitive gate misses any condition, record the result and stop without
production integration, a short song, or another M2250 run.

If it passes, the next boundary is one authenticated package-native repeated
layer/subgraph gate. It must preserve every retained output U32 and demonstrate
that panel compilation and dispatch reuse are bounded. Only then may a new
explicit production profile route all qualifying dense projections to the
candidate.

The final integration decision uses one DiT-only C98/M2250 run with the same
post-DiT checkpoint as OPT-0018. It must reproduce the authenticated final
latent bits, preserve cleanup/cancellation behavior, and reduce graph
submit-through-drain wall from the frozen 62,148.19999909401 ms authority to at
most 52,148.19999909401 ms. Generation and stage walls are reported but cannot
replace that decision metric. No VAE decode, audio, listening, or three-minute
product run is authorized by the primitive result. Because the declared
arithmetic is exact, no listening gate is required if the complete layer and
final-latent U32 gates remain exact.

## Non-claims

- The 4.238x request-byte reduction is not a projected wall speedup.
- The mixed-command aggregate is not decomposed into measured dense time.
- MLX's native SIMD-group matrix throughput is not available through current
  WGSL and is not used as an acceptance threshold.
- OPT-0019 makes no under-one-minute, product-speed, release, thermal,
  responsiveness, or listening claim.

## Result

The isolated target-browser gate passed its exactness contract and failed its
predeclared performance threshold. The canonical receipt is 14,126 bytes with
SHA-256
`a2434a16dae3db3936202461a6da6548009148c739da58dd80f458931f55471d`;
the continuous external thermal trace is 8,590 bytes with SHA-256
`ebb0e4000e68338cd29cd5a888e0ee635cb95f54a745eec25612d2bf26e4bf9f`.
Both are retained under `optimization/artifacts/OPT-0019/raw/`, and the
canonical summary is `optimization/results/OPT-0019/result.json`.

All four production shapes were exact across current first/rerun, candidate
first/rerun, and candidate-versus-current raw-U32 comparisons. In total the
gate performed 101,376,000 U32 comparisons with zero mismatches. Every output
was finite and completely overwrote its qNaN prefill; prefix/suffix canaries,
the M2250 tail, deterministic reruns, and cleanup also passed. Cleanup
destroyed all 20 created buffers, left zero live bytes, was idempotent, and
destroyed the device.

The four-sample median results were:

| M / K / N | Current (ms) | Candidate (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 2250 / 2048 / 2048 | 16.450000047683716 | 11.850000023841858 | 1.3881856552393916x |
| 2250 / 2048 / 1024 | 9.300000071525574 | 6.149999976158142 | 1.5121951394437585x |
| 2250 / 2048 / 6144 | 47.249999940395355 | 36.64999997615814 | 1.2892223730186305x |
| 2250 / 6144 / 2048 | 44.30000001192093 | 36.69999998807907 | 1.2070844693817575x |

Every shape was faster. The complete `4/2/2/1` dense score improved from
223.20000022649765 to 169.69999998807907 ms, a 1.3152622288873117x speedup and
53.50000023841858 ms saving per layer-evaluation. It therefore cleared the
52.0834 ms absolute-saving condition but missed the independently frozen 1.55x
ratio condition. The feed-forward `0/0/2/1` score improved from
138.79999989271164 to 109.99999994039536 ms (1.2618181815265623x).

One earlier launch was rejected by the gate's stale-timestamp preflight before
any timing sample was encoded or submitted. It is not a performance run. The
authoritative attempt was the sole timed run: both kernels had already been
compiled, warmed, and checked for full-output correctness, then a fresh
44.406-second continuous nominal gate completed with 45 observations, a
1,012 ms maximum poll gap, zero non-nominal observations, and a 907 ms launch
delay. No unchanged-candidate performance retry occurred, and the thermal
trace continued through cleanup.

The registered stop rule fired. The cooperative panel remains benchmark-only;
there is no package-native escalation, production integration, C98/M2250 run,
short song, full song, product-speed projection, or listening claim. Revisit
this exact geometry only if a material browser/compiler or target-GPU change
invalidates the screen. A different dense mechanism requires a separately
registered experiment.
