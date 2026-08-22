# OPT-0009 — FP16 GEMM calibration

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Date: 2026-08-13
- Author/agent: Codex
- Risk class: `approximate`; native FP16 accumulation is intentionally a
  different numerical contract from the accepted packed-BF16/FP32 oracle

## Hypothesis

The local M3 exposes fixed 32-lane subgroups and `shader-f16`. Parakeet's
production scalar-subgroup GEMM uses a wider FP16 output tile and native FP16
FMA than ACE's accepted packed-BF16/FP32 GEMM. Measuring the source-authenticated
Parakeet implementation, followed by the same mechanisms on ACE's exact dense
shapes, can establish which FP16 storage and accumulation choices are worth
carrying into the optimized mixed-precision profile.

This is calibration, not a throughput requirement. A remembered or measured
Parakeet peak does not accept or reject an ACE implementation, and missing a
particular TFLOP/s figure does not veto a correct positive production-shape
result.

## Identity

- Allocation baseline: pushed `main` commit
  `303ab8df036df71768a56774c59c75c4cfe30aa9`
- Parakeet source reference:
  `7ee112738262a6f5a0efd2f150748a4087432fbb`
- ACE model manifest SHA-256:
  `18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`
- Reference fixture manifest SHA-256:
  `cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`
- Machine: MacBook Air `Mac15,12`, Apple M3, 10 GPU cores, 16 GB unified
  memory; pinned local Chrome/M3 browser identity is recorded with the result
- Required device contract: `shader-f16`, `subgroups`, and reported subgroup
  minimum and maximum both equal to 32

## Change

Add benchmark-only shader generation, buffer preparation, and browser timing
for the unchanged Parakeet subgroup GEMM sources and narrowly adapted ACE-shape
variants. Production kernels, packages, profiles, selectors, and arithmetic do
not change in this experiment.

The calibration covers:

- Parakeet's native tile-major FP16 and FP32 subgroup kernels at
  `M7520/K1024/N4096`, `M7520/K4096/N1024`, and
  `M7520/K1024/N1024`; and
- ACE's repeated 180-second DiT dense shapes at
  `M2250/K2048/N2048`, `M2250/K2048/N1024`,
  `M2250/K2048/N6144`, and `M2250/K6144/N2048`, comparing the accepted
  packed-BF16/FP32 path, FP16 operands with FP32 accumulation, and native FP16
  accumulation where the geometry supports it.

Packing, allocation, upload, shader compilation, and pipeline creation stay
outside steady-kernel timing but are measured separately. No materialized VAE
im2col or production package conversion belongs in OPT-0009; those are separate
follow-up mechanisms after dense calibration.

## Correctness gate

- Authenticate the exact Parakeet source commit and pin the generated shader
  bundle identities used by the browser harness.
- Compare complete manageable fixtures and selected first/middle/last regions
  of large shapes against an independent FP32 CPU oracle and the corresponding
  GPU baseline. Require complete writes, finite output, guards, deterministic
  repeatability, correct tail handling, and explicit maximum absolute/relative
  and RMS error for every approximate variant.
- Include cancellation-sensitive magnitudes, signed zero, FP16 range edges,
  long-inner accumulation, and a tail case for ACE's `M2250`.
- Calibration itself requires no listening because it is not production
  selected. Any approximate mechanism promoted into the model must later pass
  its layer, subsystem, final-latent or waveform, instrumental, and vocal gates.

## Benchmark protocol

1. Run the unchanged Parakeet FP16 and FP32 tile-major subgroup kernels on the
   three production Parakeet shapes to establish the exact machine/compiler
   calibration.
2. Run the ACE-shape comparison using the same device, buffer identities,
   command topology, and balanced in-page ordering. Keep raw GEMM free of fused
   bias, activation, residual, and other epilogues.
3. Use symmetric warmup, completion-fenced wall time, one outstanding command
   buffer, continuous nominal thermal logging, heartbeat, and post-drain
   cancellation. Retain every sample and its order.
4. Report MAC/s and two-FLOP-per-MAC TFLOP/s, scheduled versus valid work, tile
   and tail utilization, encode/submit/drain time, per-dispatch maxima, logical
   buffer bytes, and compile/preparation cost.
5. Escalate only the mechanisms whose measured ACE-shape behavior and numerical
   error make them useful inputs to the mixed-precision production design. A
   useful result may be shape-specific; no all-shapes condition applies.

## Main risks

- Parakeet's `M7520` provides more occupancy than ACE's `M2250`; its peak is a
  hardware/compiler calibration, not an ACE projection.
- Native FP16 accumulation can overflow or accumulate unacceptable error even
  when every source weight is FP16-representable. Mixed precision may require
  FP32 accumulation or bounded FP32 islands by shape.
- Parakeet's N256 tile is underutilized by narrower ACE operations. The best
  production selector may use more than one geometry.
- Browser/compiler variance can distort isolated samples. Balanced ordering,
  thermal coverage, and complete raw samples are required before attribution.

## Evidence and disposition

- Evidence conclusion and rationale: `positive`. The source-authenticated
  Parakeet native-FP16 calibration reached 2.6069–2.7975 valid logical TFLOP/s
  across its three production shapes. The decision-relevant three-arm ACE gate
  then made FP16 operand/storage with FP32 accumulation the only mechanism worth
  carrying forward: it was exact to its independent CPU model on every output
  of all five adversarial fixtures, produced the same complete finite
  fingerprints as the accepted packed-BF16/FP32 oracle on all four benign exact
  ACE shapes, and was 1.2735–1.3896x faster than that oracle by median fenced
  wall time.
- Code disposition and rationale: `benchmark-only`. Select FP16 operands and
  storage with FP32 accumulation as an input to a later mixed-precision layer
  candidate. Reject native FP16 accumulation for promotion from this
  calibration. No production kernel, runtime selector, graph, package,
  arithmetic profile, or product behavior changed.
- Result JSON: [canonical schema-v2 result](../results/OPT-0009/result.json)
- Interactions: uses the FP16-first direction recorded after `OPT-0008` while
  leaving every integrated exact optimization and the BF16/FP32 oracle intact
- Closed on 2026-08-13 with native harness commit
  `30b3b76c8114d2fa55bdc020d21d57ae53be70f3`, final accumulation harness
  commit `b41108dc1be75da9ba7a72ea64faa98c9dc81ecd`, and the six raw artifact
  identities listed below.

### Source, package, and runtime identity

The native phase authenticated unchanged Parakeet GEMM, runtime-plan, and
capability sources at SHA-256
`35db4fe52a2d096af347ef4f2411159895d563b5df3aecfa19d70a9fb3f47286`,
`30effbabbf3a405f769c90f2ae4f33641d1da366e2578b1a8e3ef5b5213665cf`,
and `5f1f11ad0964ce7e3a373845e7e185a096a7af6b2820b21b4e1d899fa9f9ae15`
at pinned Parakeet commit `7ee112738262a6f5a0efd2f150748a4087432fbb`.
The three-arm phase additionally authenticated the accepted ACE subgroup-GEMM
source at SHA-256
`9ba0c589f975f19b7f7990aae5199581a83f87500a3b97ff15eb6ef5a43311ea`.
Every generated shader identity is retained in the schema-v2 result and the raw
artifacts.

The run used deterministic synthetic calibration fixtures; it did not load or
modify a browser model package. The ACE model-manifest identity remains
`18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6`,
and the reference fixture-manifest identity remains
`cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb`.
Those hashes are provenance, not a claim that a package-native model gate ran.
The browser was Google Chrome 151.0.7922.138 on the pinned MacBook Air
`Mac15,12`, Apple M3 with 10 GPU cores and 16 GiB unified memory, macOS 26.5.2
build 25F84. The adapter reported Metal 3, `shader-f16`, `subgroups`, and fixed
32-lane subgroup minimum and maximum.

### Native Parakeet calibration

All medians below are completion-fenced wall times. TFLOP/s uses two FLOPs per
valid MAC and is the median of the retained per-sample rates.

| Shape | Native FP16 median (ms) | Native FP16 TFLOP/s | Native FP32 median (ms) | Native FP32 TFLOP/s | FP32/FP16 wall ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| M7520/K1024/N4096 | 22.55000001192093 | 2.797456419782355 | 34.69999998807907 | 1.8179346450049414 | 1.5388026594117568x |
| M7520/K4096/N1024 | 23.80000001192093 | 2.6561921780374416 | 36.79999998211861 | 1.7146496178423019 | 1.54621848586917x |
| M7520/K1024/N1024 | 6.049999982118607 | 2.606886001895548 | 9.300000011920929 | 1.6957616150306394 | 1.537190089158352x |

This establishes the remembered roughly 2.7-TFLOP/s local calibration as real,
but it does not project Parakeet's M7520 occupancy or native-FP16 numerical
contract onto ACE.

### Decision-relevant ACE accumulation gate

| ACE shape | Packed-BF16/FP32 median (ms) | FP16/FP32-accum median (ms) | FP16/FP32 TFLOP/s | Oracle/selected wall ratio | Native-FP16 median (ms, rejected) |
| --- | ---: | ---: | ---: | ---: | ---: |
| M2250/K2048/N2048 | 14.5 | 10.950000017881393 | 1.7237225125888544 | 1.3242009110795838x | 8.699999988079071 |
| M2250/K2048/N1024 | 7.450000017881393 | 5.8499999940395355 | 1.613311702319503 | 1.2735042778584735x | 4.649999976158142 |
| M2250/K2048/N6144 | 42.200000047683716 | 30.75 | 1.841406624648034 | 1.3723577251279258x | 23.55000001192093 |
| M2250/K6144/N2048 | 46.55000001192093 | 33.50000002980232 | 1.6904829223325042 | 1.389552237925643x | 23.899999976158142 |

Every arm wrote every output, retained zero qNaN prefill sentinels, was
deterministic, and stayed finite on the four benign exact-shape fixtures. Each
arm produced the same per-shape fingerprint over 25,344,000 output values per
arm and pass. This benign equality is not treated as a general equivalence of
the accumulation contracts.

On all five full-output adversarial fixtures, FP16 operands with FP32
accumulation was bit-exact to the independent FP16-operand/FP32-accumulation CPU
model. Native FP16 accumulation was also exact to its own CPU model, proving
that the following failures are the intended arithmetic contract rather than a
GPU implementation bug:

- signed-zero fixture: 1,409 expected tiny nonzero values collapsed to zero
  (516 positive and 893 negative zeros);
- cancellation fixture: 1,024 values collapsed to zero/class mismatches, with
  maximum absolute error 4.50390625;
- range fixture: 2,112 positive and 2,112 negative infinities replaced finite
  FP32-accumulation results;
- long-K fixture: maximum absolute drift 0.15625 and RMS drift
  0.10170662264158173; and
- benign K2048 fixture: exact, which is retained but does not override the four
  adversarial failures.

With the registered per-layer shape multiplicities `4/2/2/1`, the sum of
medians is 203.85 ms for the accepted oracle, 150.5 ms for selected
FP16/FP32 accumulation, and 115.1 ms for rejected native FP16 accumulation.
Multiplying those microkernel sums by 24 layers and eight evaluations gives the
explicitly diagnostic dense projection 39,139.2 / 28,896 / 22,099.2 ms. It is
not an observed layer, package, denoise, final-latent, song, or end-to-end time.

### Thermal, cancellation, and caveats

The native timing artifact is covered by 103 external nominal observations
over 103,134.382291988 ms with a 1,024.191416974645 ms maximum poll gap. The
three-arm timing artifact is covered by 299 external nominal observations over
298,010.0759580382 ms with a 1,009.9415410077199 ms maximum poll gap. Both logs
span their first timing fence through cleanup, contain no non-nominal reading,
and follow a nominal pre-gate of at least 30 seconds. Packing, allocation,
upload, compilation, and pipeline creation were excluded from steady timing.

Both phases kept one outstanding command buffer and completed a post-drain
cancellation probe: one execution encoded, submitted, and drained, then two
planned executions were prevented. The three-arm timing phase destroyed all
138 created buffers with no live buffer, uncaptured error, or unexpected device
loss.

The three-arm harness did not retain decision-grade heartbeat metrics or a
per-dispatch maximum, so this record makes neither claim. The accepted evidence
is bounded to complete exact-shape GEMM timing, numerical calibration,
cancellation, lifecycle, and thermal validity. It does not integrate a
production path and does not claim a layer, package-native model, denoise-step,
final-latent, listening, waveform, 180-second song, or under-one-minute result.

### Raw artifacts

- native correctness:
  `9533036f898cf772f83e57a7035e1ba71ee0cd85c41ae589f4644b65626a9b56`
- native timing:
  `cc212f5c93ea8dc1a29dfd76c60cf21464caf606461ad233b2ffdfd61bcce4bb`
- native continuous thermal log:
  `d1bb19ad505c13fcf8b16d61d19873bbc4f9f9b4e0f7d9b1d21085b83c276ee5`
- accumulation correctness:
  `3c0a732e03b8cfa19223a7090673415827879158a7fcde773f3098b1ce407ce4`
- accumulation timing:
  `ab89be2eb42f85673ad5a62108777c31ffdf8e58d2af72750c8438542c0dde5b`
- accumulation continuous thermal log:
  `d2d3c0d35000329532ac7ac89974423221dfb934a3730674987bee681eae1b3b`
