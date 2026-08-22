# OPT-0077 — VAE selected-K7 RFFT16 transform domain

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Date: 2026-08-21
- Author/agent: Codex
- Risk class: `reordered transform-domain rounding with FP16 spectral storage
  and FP16 K4 products`
- Allocation baseline: pushed `main` commit
  `47c4dfed1a2b4826cafda14123282ce72d852cb2`
- Registration commit: `3539a87`
- Candidate/harness commit: `f78d0d24c9b205a9614703e17b7e870793d6107e`

## First-principles basis

The accepted OPT-0073 product takes `94,774.8 ms` Generate-to-WAV on the
local M3. Its two C2314 VAE windows take `32,586.7 ms`, so another narrow
sub-millisecond dispatch optimization is not the right next target.

The accepted OPT-0072 physical VAE profile routes twelve biased K7 operations
through OPT-0051's row-reuse K4 owner: C1024, C512, and two C128 blocks, each
at dilations 1, 3, and 9. The same profile deliberately leaves C256 and the
ingress/final K7 operations on their current native scalar owners. This
experiment changes only the twelve row-reuse routes; it does not retry
OPT-0076's C256 promotion.

OPT-0051's retained planning authority assigns about `16.1252 s` of its
selected-K7 projection to these twelve routes. That number is useful only for
prioritization and the frozen projection arithmetic below; it is not a new
measurement of the current product.

For one output channel and one Cin4 group, ten direct K7 outputs require
`10 * 7 = 70` FP16 dot4 products. A length-16 real FFT of the input and the
zero-padded seven-tap filter produces the same ten valid cross-correlations
with:

- two real endpoint-bin products; and
- seven complex-bin products, each implemented by the ordinary four-real-
  product formula.

That is `2 + 7 * 4 = 30` real dot4 products for ten outputs, a `70 / 30 =
2.333333x` domain-product ceiling before transforms. Unlike an FFT over a long
song, this is a bounded local overlap-save transform. Every tile consumes
sixteen input positions and emits ten output positions; adjacent tiles overlap
by six input positions.

## Rejected pre-implementation direction

The initial audit proposed real Cook-Toom/Winograd F(4,7), whose ten domain
products emit four outputs for a `2.8x` product ceiling. Before allocating
candidate code, the published nine-point-plus-infinity construction at
`[0, -1, 1, 1/2, -1/2, 2, -2, -1/4, 4, infinity]` was generated and checked
symbolically against all four seven-tap correlations.

Its Cook-scaled transforms have approximate two-norm condition numbers
`371.4` for B, `49.47` for G, and `33.87` for A. In deterministic bounded
random screens, FP32 transforms stayed around `1.36e-6` NRMSE, but explicitly
rounding transformed inputs and weights to FP16 produced about `4.87e-3`
NRMSE and maxima around `1.66e-2`. Row and three-way rescaling did not remove
the cancellation. That already conflicts with the established K7 primitive
envelope, so F(4,7) is rejected before implementation and is not an arm of
OPT-0077.

A length-16 orthonormal real DFT has condition number one. The corresponding
static screens were about `1.08e-7` NRMSE with FP32 spectral values, about
`3.18e-4` after FP16 spectral storage, about `3.90e-4` with scalar FP16
products, and `5.15e-4` in an explicit increasing-dot4 partial simulation.
Those figures are design evidence, not a browser correctness result. They
choose the safer transform before code and do not relax any gate below.

## Frozen mathematical mechanism

For each dilation `d`, partition time into residue streams
`x_r[q] = x[r + d*q]`. A tile whose first output is logical stream position
`q0` loads sixteen values `d[i] = x_r[q0 - 3 + i]`, `i = 0..15`, with
out-of-range positions represented by the existing exact K7 padding zeros. It
emits the ten correlations

`y_r[q0 + j] = sum(k = 0..6, g[k] * x_r[q0 - 3 + j + k])`,
`j = 0..9`.

Because `j + k <= 15`, circular wraparound cannot contribute to those ten
positions. The result is therefore the ordinary padded K7 cross-correlation,
not a circular approximation. Mapping back through `r + d*q` makes the same
construction exact for dilations 1, 3, and 9. Four contiguous output times are
not incorrectly treated as one dilation-3 or dilation-9 tile.

Use the orthonormal real length-16 DFT basis. Let `a = 1/4`,
`b = 1/sqrt(8)`, and `theta(t) = 2*pi*t/16`. For a real vector `d`, store:

- `X0 = a * sum(i, d[i])`;
- `X8 = a * sum(i, (-1)^i * d[i])`; and
- `Xc[t] = b * sum(i, d[i] * cos(theta(t)*i))` and
  `Xs[t] = b * sum(i, d[i] * sin(theta(t)*i))`, `t = 1..7`.

Apply the same definition to the zero-padded seven-tap filter to obtain
`G0`, `G8`, `Gc`, and `Gs`. Accumulate over input channels:

- `Z0 = sum(c, X0*G0)` and `Z8 = sum(c, X8*G8)`;
- `R[t] = sum(c, Xc[t]*Gc[t] + Xs[t]*Gs[t])`; and
- `I[t] = sum(c, Xc[t]*Gs[t] - Xs[t]*Gc[t])`.

The ten outputs are

`y[j] = bias + Z0 + (-1)^j*Z8 + sum(t=1..7,
R[t]*cos(theta(t)*j) - I[t]*sin(theta(t)*j))`.

There is no additional factor of 2, 4, or 1/16 in this real-basis formula.
Preserve these correlation signs and use the ordinary four-real-product
formula; do not use Gauss's three-product complex multiply in this ID. The
frozen coordinate order is
`[DC, NYQUIST, COS1, SIN1, ..., COS7, SIN7]`.

For the inverse radix-2 implementation, reconstruct the Hermitian spectrum as
`Q0 = 4*Z0`, `Q8 = 4*Z8`, `Q[t] = 2*(R[t] + i*I[t])`, and
`Q[16-t] = conjugate(Q[t])`. The fixed unitary inverse butterfly carries its
ordinary `1/4` scale and emits the formula above; keep outputs 0 through 9.

The input and the already weight-normalized logical FP16 weights are
transformed with a fixed radix-2 FP32 butterfly schedule. The sixteen real
spectral coordinates are stored as FP16: real bins 0 and 8 plus real/imaginary
pairs for bins 1 through 7. Domain products use native FP16 dot4 operands,
widen each partial once, and keep FP32 running accumulators. The inverse
radix-2 transform, scale, and bias addition are FP32; the existing output
boundary remains an explicit FP16 store.

The radix-2 butterfly order, twiddle constants, normalization, real-coordinate
layout, complex signs, Cin4 order, and explicit casts are part of the kernel
identity. A later Gauss multiply, FP16 butterfly, larger tile, FFT length, or
different normalization requires a new ID.

## Frozen three-stage primitive

Implement an isolated benchmark owner with three ordered dispatch families in
one quantum and no extra queue drain:

1. **Forward transform.** Enumerate globally anchored length-16 tiles for each
   dilation residue stream whose ten-output interval intersects the requested
   range. Load padded NLC FP16 input, execute the fixed FP32 RFFT16 butterflies,
   and store sixteen FP16 real spectral coordinates. Range starts and counts
   may be unaligned to the ten-output tile.
2. **Domain contraction.** Use fixed32 WG128. One workgroup covers sixteen
   tiles and 128 output channels for one real spectral result coordinate. Each
   subgroup owns one 32-channel band; its lanes cooperatively load sixteen
   Cin4 input spectra, use literal-source subgroup broadcasts, and retain
   sixteen named FP32 accumulators per lane. Endpoint results use one dot4 per
   Cin4. Each real or imaginary complex result uses the declared two dot4s per
   Cin4, for thirty real dot4s across the sixteen spectral result coordinates.
   Store results in `[Z0, Z8, R1, I1, ..., R7, I7]` order.
   Dynamic accumulator-array indexing is forbidden.
3. **Inverse transform.** Load the sixteen FP32 spectral contraction values,
   reconstruct `Q`, execute the fixed FP32 unitary inverse RFFT16, add FP32 bias, predicate all
   ten stores against the requested range and logical output extent, and store
   FP16.

The benchmark may pretransform deterministic weights before READY because a
production follow-up would extend the sole converter and package manifest.
The timing excludes that offline package-preparation analogue but reports its
complete persistent payload. The twelve current row-reuse tensors contain
`56,426,496` FP16 bytes. Sixteen real spectral coordinates in place of seven
taps require exactly `128,974,848` bytes, an increase of `72,548,352` bytes.
No native and transformed duplicate may survive in a production package.

The bounded scratch model is `3.2` bytes per logical output for FP16 input
spectra plus `6.4` bytes per logical output for FP32 contraction spectra,
before alignment. Report actual maximum bytes and buffer bindings. Keep every
binding under the authenticated target limits and include all scratch in the
resource peak.

## Static and correctness gate

Before timing:

1. Prove the radix-2 transform against an independent direct DFT over every
   length-16 input basis vector and all seven filter basis vectors. Prove the
   declared correlation signs, exact real-basis normalization, ten valid
   outputs, six-position overlap, and d1/d3/d9 residue mapping. Exhaustively cover
   tile/range intersection and final partial tiles for bounded planner domains.
   The proof must explicitly reject any extra transform-scale factor.
2. Authenticate exactly nine production strata: C1024, C512, and C128 at each
   of dilations 1, 3, and 9. For each stratum exercise first, interior, and tail
   ranges; make at least one interior start and count unaligned to ten and cover
   every residue for d3/d9. Add bounded signed-zero/subnormal, alternating
   cancellation, finite-range/transform-amplification, and partial-tile
   adversarial cases. Add DC, Nyquist-alternating, and sine/cosine basis cases
   for bins 1 through 7 so periodic fixtures cannot hide a bin/sign error.
3. Run the current OPT-0051 row-reuse K4 control once, the candidate twice, and
   an independent scalar-FP32 K7 oracle. Candidate repeats must be raw-U16
   identical. Against the scalar oracle, every probe and the aggregate must
   retain the established OPT-0024 envelope: NRMSE at most `0.001`, SNR at
   least `60 dB`, Pearson at least `0.99999`, and maximum absolute error divided
   by the greater of oracle peak and `1e-6` at most `0.01`.
4. Report maximum, mean, and RMS error; complete ULP distribution; signed-zero
   and finite-class changes; first/worst coordinates; per-stratum and aggregate
   hashes; and candidate-versus-current metrics. Require finite complete qNaN-
   overwriting writes, intact binding/tile/adjacent guards, zero out-of-range
   writes, zero uncaptured GPU errors, and no unexpected device loss.

Any numerical failure closes this exact precision/transform mechanism. Do not
respond by loosening the envelope, changing points, switching FFT length, or
silently retaining FP32 spectral storage under this ID.

## Primitive performance gate

Compile, warm, and complete correctness before READY. Time all nine strata,
not one favorable dilation. Each timed stratum is one complete 512-frame
output range, not a favorable small probe, and therefore includes left/right
padding, interior steady state, every dilation residue, overlap, and one tail.
Use the OPT-0051 production weights divided by their common factor 47: per
dilation C1024 has multiplicity 2, C512 has 3, and C128 has 9. A weighted
sample therefore contains 42 full-range stratum instances. The current control
encodes 42 kernel dispatches and the candidate encodes 126
forward/domain/inverse dispatches. Candidate timing includes all three stages
and their scratch traffic; control timing includes its complete current
dispatch. Both use one pass/command buffer/submit/drain per composite arm.

Collect six balanced forward/reverse rounds after symmetric untimed warmup.
Record timestamped GPU and fenced wall samples. Require:

- every stratum's median timestamped GPU time to be non-slower;
- at least five of six aggregate GPU pairs and five of six fenced-wall pairs
  to win;
- both aggregate mean and median GPU speedup at least `1.35x`;
- both aggregate mean and median fenced-wall speedup at least `1.35x`; and
- applying the lower wall mean/median ratio only to the frozen `16.1252 s`
  scoped planning authority to project at least `4.0 s` saving.

The projection is prioritization arithmetic, not a product-speed claim. A
primitive pass is `positive` benchmark evidence only. It authorizes a new-ID
authenticated package and C512 subsystem gate; it does not select production.

## Thermal and lifecycle protocol

Use a continuous external `notifyutil` thermal trace at about 1 Hz. Correctness,
pipeline compilation, and warmup must finish before timestamped READY. Require
at least 30 seconds of level-0 observations immediately before launch, maximum
observation gap `1,250 ms`, zero missing/non-nominal observations, launch no
later than five seconds after the gate, and trace coverage through cleanup.
There is no unchanged-work thermal retry.

Track transformed persistent bytes, both scratch stages, controls, timestamps,
readbacks, and guards. Drain before release; unmap every mapping; destroy every
created buffer exactly once; end at zero live buffers and bytes; exercise
idempotent owner destruction, post-destroy rejection, construction-failure
cleanup, and rejected-pipeline promise eviction.

## Escalation boundary

If and only if the primitive passes literally, allocate a new experiment for
converter-native replace-not-duplicate weights and an actual C512
candidate/current/current/candidate decoder gate. That gate must preserve the
accepted complete-waveform numerical envelope and attribute selected K7,
decoder, and outer wall in both directions. Only a further C2314/C4500 raw
waveform, seam, bounded-memory, lifecycle, instrumental/vocal listening, and
balanced product gate may select production.

If the primitive fails correctness, per-stratum stability, the `1.35x` speed
gate, or the four-second projection, close this exact RFFT16 mechanism without
an unchanged retry. The safe next target is exact DiT head-layout fusion, not a
smaller transform micro-tune.

## Benchmark identity

- Machine: local `Mac15,12`, Apple M3 10-GPU-core, 16 GiB unified memory.
- Browser: exact stock Chrome/Codex surface with `shader-f16`, fixed 32-lane
  subgroups, and timestamp queries.
- Current product authority: OPT-0073 with public OPT-0072 mapped to physical
  OPT-0066 and two C2314 windows.
- Record the registration/candidate/harness commits; source and generated-WGSL
  hashes; exact transform constants; every raw sample; thermal artifacts; and
  result hashes before closing the ledger row.

## Results

OPT-0077 is negative and abandoned. The isolated three-stage RFFT16 owner and
browser gate were implemented at
`f78d0d24c9b205a9614703e17b7e870793d6107e`. Correctness, bounded-resource,
lifecycle, and thermal gates passed, but the candidate missed every frozen
performance threshold and is not authorized for package or decoder
escalation.

One operator setup preflight was rejected before the authoritative run. The
external trace had started at epoch millisecond `1787309001845`, after the
first page had already reached its internal READY timestamp, so the
`thermal-launch` parser correctly raised `OPT-0077 fresh continuous nominal
thermal launch gate failed`. The entered interval itself contained `98`
level-0 observations from `1787309086853` through `1787309183855`, with a
`1010 ms` maximum gap and no missing or non-nominal observations, but zero
timing dispatches or samples ran. Its cleanup reconciled `717/717` buffers,
`628/628` maps, zero live buffers/bytes, and all declared lifecycle checks.
That page receipt was not persisted, so no hash is claimed; this is disclosed
as operator setup history, not experiment evidence. The authoritative run used
the same continuously running trace and a fresh preparation whose READY
timestamp followed trace start. This was not an unchanged-work timing retry.

The fresh correctness run exercised all nine C1024/C512/C128 by d1/d3/d9
production strata, `27` first/interior/tail probes, and `20` bounded
adversarial/spectral cases (`47` probes total). The radix-2/direct-DFT proof
covered `256` forward coordinates, `112` weight coordinates, `1,120`
correlation outputs, `192` planner domains, `137,280` ranges, and `2,299,440`
selected rows, with exact-once residue/range coverage and no extra scale.
Across `888,448` candidate-versus-scalar U16 values, the candidate repeated
with zero differences. The aggregate candidate/scalar metrics were NRMSE
`0.000078497127375363`, SNR `82.10292472187346 dB`, Pearson
`0.999999996937326`, and relative maximum absolute error
`0.0008223684210526315`; the production-only NRMSE was
`0.00042561128497590247`. All `188` output snapshots were finite and complete,
every qNaN prefill was overwritten, all binding/tile/adjacent guards and
selected tails were intact, scratch was finite and guarded, and there were
zero out-of-range writes, uncaptured GPU errors, or device losses. The compact
result retains the complete aggregate FP16 ULP distribution, finite-class
transitions, first/worst coordinates, and aggregate hashes.

Six balanced 42-current-dispatch versus 126-candidate-dispatch samples, each
covering complete 512-frame ranges for all nine weighted strata in one
pass/command buffer/submit/drain, measured:

- current mean/median GPU `45.09969066666667 / 42.303488 ms` and wall
  `46.13333338499069 / 43.10000002384186 ms`;
- candidate mean/median GPU `45.14338133333333 / 45.449216 ms` and wall
  `46.09999998410543 / 46.30000001192093 ms`; and
- current/candidate mean/median speedups of
  `0.9990321800145174x / 0.9307858687815429x` on GPU and
  `1.0007230672645717x / 0.9308855294329339x` on fenced wall.

Only one of six GPU pairs and one of six wall pairs won, rather than the
required five. C1024-d1, C1024-d9, and C512-d1 had slower candidate median GPU
times, so the all-strata rule also failed. Mean and median GPU and wall
speedups all missed `1.35x`; applying the lower wall ratio to the frozen
`16,125.2 ms` planning authority projects `-1,197.2306213278046 ms`, not the
required positive `4,000 ms`. The first round's cool-start-looking win does
not override the other five paired losses or the frozen balanced summary.

Resource accounting was complete: twelve transformed routes would replace
`56,426,496` native/current bytes with `128,974,848` bytes, an increase of
`72,548,352` bytes. Maximum candidate scratch was `5,308,416` bytes across two
bindings, with the largest single binding `3,538,944` bytes and all
authenticated limits satisfied. The authoritative run peaked at
`313,105,520` live bytes; cleanup drained the queue, balanced `717/717`
created/destroyed buffers and `748/748` maps/unmaps, passed idempotent destroy
and post-destroy rejection, and ended with zero live buffers, bytes, or maps.

The authoritative external trace contained `302` observations, all level 0,
with a `1010 ms` maximum gap and coverage through cleanup. Its SHA-256 is
`1e6c451123330b377fb7d44ba3c3f4e3b7183dc3eb9c9b67a19703d63cf8f21f`.
The uncommitted `898,231`-byte browser receipt is bound by SHA-256
`724b9983e314ff68ec4060254914feaf5b9e8005e3ba2af49a00fe352dd35e4a`.
See the [compact result](../results/OPT-0077/result.json) for exact samples,
gates, hashes, numerics, resources, lifecycle, and thermal facts.

No C512 package experiment, decoder profile, waveform, listening run, package
change, or production integration is authorized. Production remains on the
accepted OPT-0073 path. Do not retry this unchanged RFFT16 overlap-save
mechanism; the declared next target is exact DiT head-layout fusion unless a
materially different transform/dataflow or changed browser/GPU compiler
creates a new mechanism.
