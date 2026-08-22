# OPT-0017 — VAE K7 cooperative dot4 implicit GEMM

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Date: 2026-08-14
- Author/agent: Codex
- Risk class: `reordered-rounding`; FP16 operands are widened to FP32, but
  each output reduces valid terms through increasing-R groups of four using
  `dot(vec4<f32>, vec4<f32>)` rather than the shipped scalar FP32 addition
  sequence
- Evaluated production baseline:
  `36608b857827b2b1d31ac91bf5cca9639fb0b9ed`
- Motivating exact-order core:
  `997891de0fe449c9b6551e80abc55604256969ad`
- Motivating exact-order browser gate:
  `085669d5aec0fc02f3268c8b462385b59fb72ab7`
- Motivating OPT-0016 raw artifact SHA-256:
  `3bfbe588d5aa6595b3f49caff670cb62293157157e4a34d0fb12349265266222`
- Frozen cooperative-dot4 core:
  `b83f4fe94d56787ddb980629ea6f41804543ca69`
- Frozen browser gate:
  `c34efbb67017c679c7932eed1df783254af17631`

## Motivation

OPT-0016 closed the nearby exact-order accumulator-tile space. Its `8x64`,
`16x32`, and `8x32` arms remained raw-bit exact but reached only
`0.9779577975624927x`, `0.9815300299687449x`, and
`1.007712087279326x` versus packed `16x64`; none reached the registered
`1.15x` threshold.

The current integrated 12-second direct profile attributes `3,019.8 ms` of
its `7,265.8 ms` decoder submit-through-drain wall to K7. VAE wall is
`8,054.5 ms`, and total generation wall is `23,018.2 ms`. The current exact
FP16 WAV SHA-256 is
`409b7157ac428910fae17776b1abbd9b42db7509984bcc0aac41871f95152ec2`.

A materially faster K7 path now requires cooperative operand reuse and a
shorter FP32 dependency chain, accepting an explicitly quality-gated rounding
change.

## Candidate

Implement one adaptive 256-thread cooperative implicit-GEMM owner for the 16
biased, FP16-output K7 operations. Leave the final no-bias FP32 `conv2` on the
shipped fixed32 kernel.

Flatten the reduction coordinate as:

`r = kernel * inputChannels + inputChannel`

All production input-channel counts are multiples of 32, so an `R=32` panel
never crosses a kernel-tap boundary.

Use these compile-time workgroup tiles:

| Production shape | Workgroup tile | Output accumulators/thread |
| --- | ---: | ---: |
| `1024 -> 1024` | `32 rows x 128 Cout` | 16 |
| every other FP16-output K7 shape | `64 rows x 64 Cout` | 16 |

Each thread owns a `4 rows x 4 Cout` register tile. For each R32 panel:

1. Cooperatively load dense FP16 input values into a row-major
   `M x 32` workgroup panel.
2. Cooperatively load the existing OPT-0014 KIO FP16 bytes, transposing them
   into an output-major workgroup panel with stride 33.
3. Execute one uniform workgroup barrier.
4. Explicitly widen operands to `vec4<f32>`. For each increasing group of four
   valid reduction terms, add `dot(input4, weight4)` directly to the final FP32
   output accumulator.
5. Execute one uniform workgroup barrier before overwriting either panel.

For a row/kernel pair outside the valid padded input domain, skip its dot
operations entirely. Do not add staged zero terms. Every lane must still reach
both barriers, and output stores remain range- and channel-guarded.

Bias is widened from FP16 to FP32 before accumulation. Internal results retain
the current explicit FP16 output conversion. There is no FP16 accumulation,
split-K buffer, cross-thread output reduction, clamp, saturation, second
persistent weight layout, or fallback under an existing kernel identity.

Reuse OPT-0014's bit-preserving native `[out,kernel,input]` to packed
`[kernel,input,out]` GPU repack. Since final FP32 `conv2` remains native, the
candidate owns 16 packed tensors:

- `30,507,008` unique U16 words;
- `61,014,016` persistent packed bytes; and
- `59,584` repack workgroups.

The packed buffers live from one VAE preparation pass through decoder
destruction. Repack time and the full persistent payload remain part of every
production accounting.

## Rejected compact geometry

A `16 rows x 64 Cout`, WG256, four-accumulator dot4 tile was considered and is
not another candidate arm.

Its dense R32 input panel plus stride-33 transposed weight panel requires
`5,248` workgroup bytes. At C300 it would issue `414,632` workgroups and
`34,299,776` barrier events while reading an estimated `87,801,806,848`
global operand bytes. That is essentially the same global traffic as the
failed packed control, with approximately four times the adaptive candidate's
workgroups and barriers. Another measured nearby-tile sweep is not justified.

## Static resource and traffic model

The adaptive shaders use WG256 and at most `10,496` workgroup bytes, below the
target M3 limits of 256 invocations and 16,384 workgroup bytes.

For the exact C300 candidate scope—16 operations, 2,399 graph ranges, and
`561,787,699,200` logical MACs—the model is:

| Metric | Shipped fixed32 | Packed `16x64` | Cooperative dot4 |
| --- | ---: | ---: | ---: |
| Workgroups | `103,672` | `103,658` | `103,684` |
| Estimated global operand bytes | `149,295,505,408` | `87,801,806,848` | `36,163,764,224` |
| Bytes/MAC | `0.2657508` | `0.1562900` | `0.0643727` |
| Workgroup barrier events | `0` | `0` | `8,579,200` |

The candidate therefore projects `4.1283x` fewer global operand bytes than
shipped fixed32 and `2.4279x` fewer than packed `16x64`, without materially
increasing workgroup count.

Exact candidate totals for the production decoder geometries are:

| Decoder geometry | K7 ranges | Workgroups | Barrier events | Estimated global operand bytes |
| --- | ---: | ---: | ---: | ---: |
| B256 | `2,041` | `88,448` | `7,314,944` | `30,842,814,464` |
| C300 | `2,399` | `103,684` | `8,579,200` | `36,163,764,224` |
| C512 | `4,082` | `176,896` | `14,629,888` | `61,685,628,928` |

These are static traffic estimates, not performance claims.

## Primitive decision gate

1. Pin the generated WGSL, kernel ID, tile selection, R32 divisibility,
   complete output ownership, exact B256/C300/C512 range counts, workgroups,
   workgroup storage, barriers, bindings, cache separation, failure eviction,
   destruction, and post-destroy rejection.
2. Verify the KIO repack bit-for-bit. Verify the four selected representative
   tensors twice and all remaining candidate tensors once, for
   `40,255,488` raw-U16 comparisons. Require deterministic selected-tensor
   hashes, complete qNaN overwrite, intact redzones, and zero mismatch.
3. Before timing, compare the candidate against shipped fixed32 on the existing
   C1024, C512, C256, and C128 first/interior/tail probes. Across two
   executions this covers `2,392,064` candidate-versus-control U16 values.
   Require, per probe and in aggregate:
   - complete finite writes with intact guards and adjacent canaries;
   - deterministic candidate rerun hashes;
   - NRMSE at most `0.001`;
   - SNR at least `60 dB`;
   - Pearson correlation at least `0.99999`; and
   - maximum absolute error divided by the greater of control peak and `1e-6`
     at most `0.01`.
4. Report maximum/mean/RMS error, complete FP16 ULP distribution, signed-zero
   differences, first/worst locations, and output ranges. Raw output identity
   is not required and these thresholds may not be loosened after observing
   candidate output.
5. Compile and warm both arms before timing. After one continuous 30-second
   nominal thermal start, run once with no thermal retry. Use four samples per
   arm per representative tier in balanced alternating order and score each
   tier's median weighted over the represented 2,397 ranges. The two conv1
   ranges and five shipped-final-conv2 ranges remain explicitly outside this
   representative score.
6. Stop before a full allocation unless the candidate is at least `1.75x`
   faster than shipped fixed32 on the weighted representative score.
7. If it qualifies, compare candidate versus shipped fixed32 over all 16
   candidate operations and all 2,399 exact C300 ranges in balanced AB/BA
   order. Include one measured all-16 GPU repack before each candidate run.
   Proceed only if the candidate wins both paired orders and reaches at least
   `2.0x` aggregate speedup including repack.
8. A primitive numerical and timing pass authorizes only an explicit
   production-smoke candidate. It does not authorize a production default,
   quality claim, or listening claim.

## Browser result

The authenticated Chrome/M3 primitive gate passed every registered repack and
numerical requirement. The all-16 KIO repack had zero mismatches across
`40,255,488` raw-U16 comparisons: `30,507,008` unique words, `59,584`
workgroups, four selected tensors rerun deterministically, complete qNaN
overwrite, and intact redzones. The packed payload remained `61,014,016`
bytes.

Across the 12 C1024/C512/C256/C128 first/interior/tail probes and two
executions, the candidate was compared with shipped fixed32 over `2,392,064`
U16 values. Every probe and the aggregate passed the frozen envelope. The
aggregate result was:

- maximum absolute error `0.000030517578125`;
- mean error `-2.667685007446841e-10`;
- RMS error `1.6688909076031648e-7`;
- NRMSE `0.000009066605130144652`;
- SNR `100.85110596605332 dB`;
- Pearson correlation `0.9999999999588972`;
- relative maximum absolute error `0.0005431830526887561`;
- `2,390,380` zero-ULP and `1,684` one-ULP comparisons; and
- zero signed-zero differences.

Both control and candidate numeric ranges were finite, all outputs replaced
their qNaN prefills, guards and adjacent canaries remained intact, and every
candidate rerun hash was stable.

The run then used one `30,198 ms` continuous nominal thermal start: 31
observations, a `1,015 ms` maximum poll gap, zero non-nominal observations,
and an `81 ms` launch delay. There was no unchanged-work thermal retry.

The four-sample median score represented 2,397 exact C300 ranges; the two
conv1 ranges and five shipped-final-conv2 ranges remained explicitly outside
the score. Every measured tier regressed:

| Tier | Shipped fixed32 median | Cooperative dot4 median | Speedup |
| --- | ---: | ---: | ---: |
| C1024 | `8.75 ms` | `19.049999952316284 ms` | `0.45931758645154697x` |
| C512 | `6.049999952316284 ms` | `10.949999988079071 ms` | `0.5525114117719391x` |
| C256 | `3.9999999403953552 ms` | `11.550000011920929 ms` | `0.3463203408023286x` |
| C128 | `5.549999952316284 ms` | `8.899999976158142 ms` | `0.623595501930782x` |

The weighted primitive projection regressed from
`13,761.599894106388 ms` to `26,183.699956297874 ms`: only
`0.5255788875168637x` the fixed32 speed, or about `1.9026639458913086x`
slower. It therefore missed the frozen `1.75x` qualification threshold by a
wide margin.

The declared early stop fired. No candidate was selected, the conditional
full-sequence controls/dispatch allocation was not performed, and the all-16,
2,399-range AB/BA phase did not run. The all-16 weight buffers had already
been allocated and verified before the thermal gate as required; the skipped
phase is specifically the conditional full sequence. Cleanup destroyed all 83
created buffers, left zero live bytes, and destroyed the device.

The persisted raw receipt is ignored from Git at
`optimization/artifacts/OPT-0017/raw/cooperative-dot4-k7-ab.json` (52,616
bytes), SHA-256
`903d810d5c0ea4f0c411587cdeffbc90a462690bb9cd570a1665b619ec7eebb2`.

## Twelve-second production gate

Only after the complete primitive gate passes, integrate the candidate behind a
new explicit kernel-set/profile identity. Existing portable and fixed32
profiles remain unchanged, with no silent fallback or selection under an old
profile ID.

Run the pinned 12-second direct instrumental fixture with the candidate as the
only quality-affecting change.

Require:

- exact U32 identity at all eight DiT denoise-step taps and the complete final
  latent, proving that the changed profile is confined to the VAE;
- the identical immutable final latent as input to control and candidate VAE
  decodes;
- complete comparison of all `1,152,000` raw FP32 stereo waveform samples
  before ordinary `-1 dBFS` normalization;
- finite, nonzero, unclamped output with distinct channels and deterministic
  repeats;
- normalized RMS error at most `0.003` and SNR at least `50 dB`;
- Pearson correlation at least `0.9999` per channel and jointly;
- relative RMS/energy drift at most `0.005`;
- relative peak drift at most `0.01`;
- absolute DC-offset drift at most `0.001` times the greater of control RMS and
  `1e-6`; and
- maximum absolute error divided by the greater of control peak and `1e-6` at
  most `0.02`.

The same run must reduce production K7 submit-through-drain from `3,019.8 ms`
to at most `1,509.9 ms`, decoder submit-through-drain from `7,265.8 ms` to at
most `5,755.9 ms`, and complete VAE wall from `8,054.5 ms` to at most
`6,654.5 ms`. Repack, upload, memory, cleanup, and total generation wall must
be reported separately and may not hide a decode regression.

## Listening and long-run gate

Because this experiment changes VAE rounding, numerical evidence cannot select
it for production.

After the 12-second gate passes:

1. Produce fresh level-matched current-control and candidate instrumental
   outputs from the pinned direct request and obtain explicit owner listening
   approval.
2. Produce fresh current-control and candidate vocal outputs from the pinned
   vocal request, long enough to contain actual vocals—at least 30 seconds—and
   obtain explicit owner listening approval. A 12-second vocal crop does not
   qualify.
3. Validate raw waveforms before normalization, then validate final WAV
   structure, peak, channels, hashes, and any window seams.
4. Run an actual 180-second direct generation only after the combined
   production projection is credible. Use one 30-second nominal start, accept
   the resulting thermal trace without retry, and report complete wall,
   K7/decoder/VAE attribution, repack, memory high-water, cancellation,
   cleanup, waveform, and listening evidence.

No approximate profile becomes the production default before both fresh
listening decisions pass.

## Evidence and disposition

- Evidence conclusion: `negative`. Repack correctness and the complete frozen
  primitive numerical envelope passed, but cooperative dot4 reached only
  `0.5255788875168637x` on the weighted representative score against the
  required `1.75x` threshold.
- Code disposition: `abandoned`. Preserve the isolated core, gate, and result
  as benchmark history, but do not integrate this cooperative-dot4 geometry or
  its `61,014,016`-byte packed payload into a production profile.
- The registered performance stop rule fired before the complete C300 phase.
  Do not repeat this unchanged shared-panel/R32/dot4 tile under another ID.
- No full-C300 timing, 12-second production generation, waveform comparison,
  listening decision, 180-second generation, integrated decoder speedup, or
  under-60-second claim was produced.
- Production integration was not authorized or performed; existing production
  profiles and selectors remain unchanged.
- Canonical result: [result.json](../results/OPT-0017/result.json)

## Stop rule

Stop and record a negative result if:

- repack identity, safety, determinism, or any frozen primitive numerical bound
  fails;
- weighted representative speedup is below `1.75x`;
- full C300 aggregate speedup including repack is below `2.0x` or either paired
  order loses;
- the 12-second denoise/final-latent isolation, waveform envelope, lifecycle, or
  performance thresholds fail; or
- either required owner listening decision rejects the candidate.

Do not tune the numerical envelope after observing output, add the rejected
`16x64` arm, substitute FP16 accumulation, add split-K or cross-thread partial
buffers, bring or change final FP32 `conv2` under the candidate, or conduct
another nearby tile sweep under OPT-0017. Any such mechanism requires a
separately registered experiment.
