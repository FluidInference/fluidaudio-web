# OPT-0075 — DiT width-128 RMSNorm WG128

## Status

- Evidence: `pending`
- Disposition: `benchmark-only`
- Date: 2026-08-21
- Author/agent: Codex
- Risk class: `exact-arithmetic-shape-specialization`
- Allocation baseline: pushed `main` commit
  `d2b0c39978d3d55b66a11d82e66e508a776ad795`

## First-principles basis

The accepted 180-second direct path spends `54,986.2 ms` in DiT denoising.
Current family profiles combine RMSNorm with neighboring work, so no committed
receipt isolates its wall. Static production topology nevertheless exposes a
large hardware-efficiency defect worth one bounded measurement.

`AceCorrectnessRmsNormKernel` always launches WG256, allocates 256 FP32 shared
words, and reduces from stride 128. For width 128, lanes 128–255 load no input,
write zeros to shared memory, participate in every barrier, and store no
output. The first reduction step adds each useful lower-half value to one of
those zeros, then barriers before the same stride-64 tree a WG128 owner could
begin with.

The accepted M2250/C98 path has, per layer and evaluation:

- self-query norm: `1 * 16 * 2,250 = 36,000` width-128 rows;
- self-key norm: `1 * 8 * 2,250 = 18,000` rows; and
- cross-query norm: `1 * 16 * 2,250 = 36,000` rows.

That is `90,000` width-128 workgroups per layer, `2,160,000` per 24-layer
evaluation, and `17,280,000` across the required eight evaluations. The
once-per-request cross cache adds `24 * 1 * 8 * 98 = 18,816` rows, for a total
of `17,298,816` workgroups. The current owner therefore launches
`4,428,496,896` lanes; exactly half, `2,214,248,448`, are the width-128 upper
half that never owns an element. It also performs the same number of shared
zero stores, executes `2,214,248,448` lower-lane additions against those
zeros, allocates 1,024 rather than 512 shared bytes per workgroup, and executes
one extra collective reduction/barrier level in all `17,298,816` groups.

This is not a FLOP optimization. It targets workgroup occupancy, issued lane
work, shared traffic, and barrier depth. Halving empty hardware work has a
credible multi-second ceiling, while a primitive projection below two seconds
is too small to justify graph and product engineering.

No earlier optimization record implements or times a width-128 RMSNorm WG128
owner. Prior WG128 records concern GEMM or attention and do not share this
reduction or selector.

## Frozen implementation direction

1. Add a benchmark-only reference-BF16 RMSNorm owner accepting only
   `width === 128`. It uses WG128 and `array<f32, 128>` shared storage. Current
   WG256 remains the exact oracle and every production selector remains
   untouched during the primitive phase.
2. Preserve the current arithmetic explicitly. Each lane computes the current
   `0 + value * value`, then performs the current stride-128 `+ 0` locally
   before its first shared write. The shared reduction remains stride
   `64,32,16,8,4,2,1` in that order. Preserve the final extra barrier,
   `inverseSqrt`, epsilon placement, output multiplication order, BF16 weight
   loading, and FP32 input/output representation.
3. Do not introduce subgroup reductions, reassociation, vectorized statistics,
   FP16 state, dense row statistics, a new package layout, or any change to
   hidden-width RMSNorm.
4. Primitive code remains absent from the production selector until the
   frozen bit-identity and material paired timing gates pass.

## Primitive correctness gate

1. Compare current WG256 and candidate WG128 over the exact reference-BF16
   production shapes `36,000 x 128`, `18,000 x 128`, and `784 x 128` at epsilon
   `1e-6`. Use deterministic finite F32 inputs, packed BF16-scale fixtures,
   guarded outputs, tail rows, and candidate reruns.
2. Add bounded signed-zero, normal/subnormal-boundary, alternating-magnitude,
   and maximum-finite BF16-scale fixtures. Require complete raw-U32 identity,
   not a numerical envelope, on every output and rerun; require finite
   production outputs, exact non-finite/class identity on adversarial overflow
   edges, intact canaries, and zero uncaptured GPU errors.
3. Prove source generation keeps the explicit second `+ 0`, the exact
   stride-64-through-1 tree, the final extra barrier, and unchanged output
   expression. Prove WG128/shared-128 occurs only at width 128 and malformed or
   unsupported devices fail closed.
4. Reconcile all buffers after repeated destroy and retain exact shader,
   harness, browser, adapter, fixture, and thermal identities.

Any raw-U32 difference closes the specialization without performance timing.

## Primitive performance gate

After correctness and compilation, wait for at least 30 continuous seconds at
thermal level 0. At the click boundary, run one immediate untimed dispatch per
arm and production shape so GPU-clock wakeup is outside measurement. Keep
thermal polling through cleanup.

Time eight balanced paired rounds with both arm orders, both shape orders, one
timestamped compute pass, one command buffer, one submit, and one matching
drain per sample. Encode eight identical RMSNorm dispatches per sample to make
the short primitive large enough to dominate timer quantization; divide only
after retaining raw totals. Weight the two `36,000` routes and one `18,000`
route per layer/evaluation. The `784` cross-cache shape is correctness- and
diagnostic-only because it executes once rather than eight times.

The candidate must satisfy all of the following on both fenced wall and
timestamped GPU time:

- beat current in every paired production-shape and weighted round;
- reach at least `1.25x` by both weighted mean and weighted median; and
- save at least `10.5 ms` by both mean and median for one `2/1` weighted
  per-layer sum `2 * T36000 + T18000` after dividing out the eight harness
  repetitions; this is deliberately not normalized by the multiplicity sum. It
  projects at least `2.016 s` over `24 * 8` without crediting cross-cache work;
  and
- keep both mean and median fenced-wall savings within `0.75x` to `1.25x` of
  the corresponding timestamped GPU saving.

Retain every sample and paired ratio. A reversal, missing thermal provenance,
or material wall/GPU disagreement is inconclusive; do not substitute a fastest
sample, absolute-range post hoc rule, or projection for the paired evidence.

## Escalation and integration gate

A passing primitive may add the exact width-128 selector to an explicit
diagnostic **DiT-only** RMSNorm profile without changing model packages. The
shared kernel is also used by Qwen and the semantic conditioner; neither may
silently inherit this selector. Compare the current and candidate owners on a
complete authenticated M2250/C98 layer and one full denoise evaluation.
Require raw-U32 identity at the layer output and evaluation output snapshot,
deterministic repeats, bounded memory, cancellation, zero live resources, and
a measured evaluation saving consistent with at least `1.5 s` across eight
evaluations. Require all eight evaluations before claiming final-latent
identity.

Production integration then requires an exact 12-second direct product A/B for
output safety: identical final latent, pre-normalization waveform, seams,
normalized WAV hash, and request/sampler identities. Do not use its much
smaller geometry to establish the M2250 materiality claim; use the paired
M2250 evaluation timing, and reserve a 180-second wall comparison for any
remaining ambiguity. Because this is an exact-output source-kernel
specialization, bit identity may satisfy the quality gate without a new
subjective listening round.

## Benchmark identity

- Machine: local `Mac15,12`, Apple M3 10-GPU-core, 16 GiB unified memory.
- Product surface: `reference-bf16` main execution profile with separate
  OPT-0009 dense package, exact M2250/C98 direct request, eight evaluations.
- Browser: record the exact Codex/Chrome surface and user agent; require
  standard `timestamp-query` for the primitive diagnostic.
- Thermal: bind the logger source hash, raw observations, launch-adjacent
  sample, transitions, maximum gap, and post-cleanup observation.
- Authority: fenced wall is primary; timestamps establish that a win occurs
  inside GPU work rather than host queue noise.

## Results

The actual Chrome/M3 primitive run is **inconclusive** under the frozen gate.
Correctness was completely positive: the current and candidate owners had zero
raw-U32 differences across `7,012,352` production outputs and `2,048` bounded
adversarial outputs, candidate reruns were exact, the forced maximum-BF16
fixture reproduced identical finite/non-finite classes, all writes and guards
passed, no GPU errors occurred, and repeated cleanup reconciled `44/44`
buffers with zero live bytes.

WG128 is directionally faster. For the unnormalized per-layer
`2 * T36000 + T18000` mix, candidate/current speedups were:

- mean timestamped GPU `1.826953x`, fenced wall `1.801191x`;
- median timestamped GPU `1.560719x`, fenced wall `1.562977x`.

The absolute mean saving was only `6.96832 ms` GPU / `7.146875 ms` wall and
the median was `5.238784 ms` / `5.531250 ms`, below the required `10.5 ms` on
both authorities. The non-authoritative `24 * 8` arithmetic therefore spans
only `1,005.85–1,372.20 ms`, below the declared `2.016 s` escalation floor.
The candidate also won only `7/8` weighted GPU and wall pairs; Q36000 won
`5/8` and K18000 `7/8`. Mean and median wall/GPU saving agreement passed.

Thermal evidence is valid: `49` launch-gate and `66` through-cleanup level-0
observations, maximum gap `923 ms`, zero non-nominal observations, and trace
coverage extending `10,142 ms` after cleanup. This is a genuine but
engineering-immaterial isolated improvement under the predeclared rules. No
DiT diagnostic profile, evaluation, product, package, selector, listening, or
production integration is authorized. Production remains on WG256.

Evidence is retained in [the compact result](../results/OPT-0075/result.json),
[the exact browser receipt](../results/OPT-0075/browser-receipt.json), and the
[thermal gate](../results/OPT-0075/thermal-gate.json) and
[through-cleanup trace](../results/OPT-0075/thermal-trace.json).

## Authority and interactions

- [Current exact RMSNorm source](../../src/webgpu/kernels/rmsnorm.ts)
- [OPT-0018 DiT family profile](OPT-0018-dit-m2250-production-family-profile.md)
- [OPT-0067 current attention evaluation slice](OPT-0067-dit-quad-query-evaluation-slice-thermal-screen.md)
- [OPT-0073 accepted direct product result](../results/OPT-0073/result.json)

Revisit after a negative result only for a materially different exact
reduction/dataflow, standardized WebGPU matrix/reduction feature, or changed
browser/GPU compiler. Do not relax bit identity or repeat the unchanged WG128
geometry after a clear materiality miss.
