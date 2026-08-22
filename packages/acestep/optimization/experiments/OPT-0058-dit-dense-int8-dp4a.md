# OPT-0058 — DiT dense dynamic-int8 DP4a

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Risk: approximate symmetric-int8 activation and weight quantization

## First-principles basis

The current stock Chrome 151 target exposes the standardized WGSL language
feature `packed_4x8_integer_dot_product`. A local page-only capability probe on
the exact M3 machine observed it in `navigator.gpu.wgslLanguageFeatures`
without requesting a device or dispatching GPU work. This is a shipped WGSL
feature, not an experimental browser flag. It enables `dot4I8Packed`, which
consumes four signed 8-bit products per instruction from two packed `u32`
operands.

This is a materially different mechanism from every FP16 dense experiment in
the ledger. OPT-0043 established that current dense wall is about 91% inside
the timestamped GPU pass, so submission-only tuning cannot close the target.
OPT-0032 shortened the FP32 dependency chain with one FP16 K4 dot partial per
four reduction elements, but all-dense K4 missed its final-latent maximum-error
gate. A bounded symmetric-int8 group can instead accumulate several DP4a
results in exact `i32`, convert once per group, apply independently stored
activation/weight scales, and add once to the FP32 running state.

Even the worst supported reduction cannot overflow signed 32-bit local state:
`127 * 127 * 6144 = 99,096,576`. Converter-native weights use one byte per
value rather than two, while group scales remain explicit. The performance bet
is therefore native-or-lowered DP4a throughput, a shorter floating dependency
chain, and lower resident/streamed weight bytes. The risk is quantization error
and the cost of dynamic activation quantization; neither may be hidden by a
prequantized-only score.

Official feature references:

- <https://developer.chrome.com/blog/new-in-webgpu-123>
- <https://www.w3.org/TR/WGSL/#dot4i8packed-builtin>

Capability receipt: [feature-probe.json](../results/OPT-0058/feature-probe.json).

## Frozen primitive direction

- Keep the current M32 × N128 fixed32/WG128 ownership, four adjacent outputs
  per lane, no workgroup storage, no barriers, FP32 bias seed/running state,
  and FP32 output contract used by the bounded K4 path.
- Screen symmetric signed-int8 K32, K64, and K128 groups. K128 is included
  explicitly because scale traffic and FP32 group updates shrink as the group
  widens; it may outperform smaller groups even when its quantization error is
  higher. For each `(row,K-group)`,
  compute one deterministic finite max-absolute activation scale and pack the
  rounded/clamped values to `u32`. Store converter-side weight scales per
  `(output,K-group)` and pack weights in a direct lane/output/packed-K layout.
  Zero groups must use a canonical zero scale and zero payload. NaN/Inf input
  must fail closed rather than silently quantize.
- Within each group, issue `dot4I8Packed` in increasing packed-K order into an
  `i32` local partial. Convert that partial once to FP32, multiply the explicit
  activation and weight scales, and add once to the FP32 running output. Do not
  accumulate the complete long reduction in int8, FP16, or a cross-group
  integer state.
- Measure two boundaries: a prequantized contraction to expose the M3/Chrome
  DP4a ceiling, then the complete dynamic-activation-quantize plus contraction
  pipeline. Offline weight conversion is outside timed inference, but dynamic
  activation quantization is always included and attributed.
- Account for actual graph reuse. One quantized activation may be reused only
  by projections that consume the identical source tensor before it is
  mutated; no speculative reuse or duplicated resident FP16+int8 package is
  allowed in an integration projection.

## Gates

1. Require the actual stock browser language feature, successful WGSL
   compilation without developer flags, exhaustive packed signed-byte
   indexing/inverse proofs, deterministic scale/round/clamp rules, complete
   writes, guards, finite outputs, and clean lifecycle.
2. Cover all four production dense shapes plus signed-zero, zero-group,
   cancellation, finite-range, saturation, and long-K adversarial inputs.
   Against the exact FP16-weight/FP32-accumulator owner, a primitive arm is
   numerically eligible only with NRMSE at most `0.01`, SNR at least `40 dB`,
   Pearson correlation at least `0.999`, zero non-finite outputs, deterministic
   repeats, and no qualitative finite-to-zero collapse above one event per
   million compared values. Report maximum absolute/relative error and every
   saturation count; these metrics do not substitute for a trajectory gate.
3. Run a balanced nominal actual-shape screen. The complete quantize-plus-GEMM
   weighted score, with quantization amortized only by proved production reuse,
   must be at least `1.50x` faster than OPT-0032 K4 and must win every
   production shape. A fast prequantized kernel with a losing complete pipeline
   stops here. Report exact control/K4/DP4a GPU timestamps and fenced wall.
4. A primitive pass authorizes only a new-ID actual-layer and eight-evaluation
   trajectory experiment. That follow-up must compare exact, selective K4, and
   DP4a at every sampler tap, pass the unchanged final-latent numerical gate,
   then pass short instrumental and vocal listening before any production
   package. Weight conversion must be replace-not-duplicate and separately
   authenticated.

This experiment makes no quality, production, product-speed, or under-one-
minute claim. If stock M3 DP4a is emulated too slowly, dynamic quantization
erases the contraction win, or the declared primitive envelope fails, abandon
the mechanism without weakening a gate.

## Result

Authoritative receipt:
[result.json](../results/OPT-0058/result.json), `129,546` bytes, SHA-256
`4dc0a49c5897fe6684bb227c7ca02876dd44ba3958dd5f35f849e1994c8bfbf9`.

The persisted run used stock Chrome 151 with no experimental browser flags on
the exact Apple `metal-3` adapter and its fixed subgroup size of 32. Chrome
reported the standardized WGSL language feature
`packed_4x8_integer_dot_product`, and the packed-dot pipeline compiled
successfully. This proves that the stock target accepts the standardized
language feature; it does not prove that Apple executes `dot4I8Packed` as a
native hardware instruction.

Three earlier setup attempts were rejected before the authoritative run. The
harness first used `enable` where the standardized feature requires
`requires`; qNaN emission then had to be made runtime-dependent; finally, the
correctness-prefill buffers needed the missing `COPY_DST` usage. None of those
attempts entered the timed phase, so none is correctness or performance
evidence.

All three symmetric-int8 groups passed the aggregate production-shape
numerical thresholds over `25,344,000` outputs per arm:

| Arm | NRMSE | SNR (dB) | Pearson | Full aggregate |
| --- | ---: | ---: | ---: | --- |
| G32 | `0.00610125205417712` | `44.29162066233184` | `0.9999848335244867` | pass |
| G64 | `0.00611217089185332` | `44.27609023409446` | `0.9999847936373586` | pass |
| G128 | `0.006112195831234905` | `44.276054793286605` | `0.9999847935468676` | pass |

That aggregate pass is not primitive eligibility. The frozen gate requires
every individual adversarial case to pass and permits at most one
finite-to-zero collapse per million comparisons. Every int8 group failed:

| Arm | Adversarial collapses / `21,504` | Rate | Limit | Eligible |
| --- | ---: | ---: | ---: | --- |
| K4 reference | `13` | `0.0006045386904761905` | `0.000001` | no |
| G32 | `104` | `0.004836309523809524` | `0.000001` | no |
| G64 | `107` | `0.004975818452380952` | `0.000001` | no |
| G128 | `110` | `0.005115327380952381` | `0.000001` | no |

The disqualifying finite-range case alone collapsed `104/4,096`
(`0.025390625`), `107/4,096` (`0.026123046875`), and `110/4,096`
(`0.02685546875`) G32/G64/G128 outputs, respectively. K4 itself collapsed
`2/4,096` (`0.00048828125`) in that case, but the int8 arms were materially
worse; the predeclared gate is not weakened to admit them. The receipt also
records the non-finite-activation fail-closed subgate as failed for every int8
group (`0/1,024` expected qNaN outputs per arm). The collapse result already
independently rejects every group.

Correctness stopped the protocol before timing, so `timing` is `null` and
there is no throughput, native-versus-lowered DP4a, or product-speed result.
The decision is `negative-stop-correctness-before-timing`. Cleanup was exact:
`398/398` buffers destroyed, zero live buffers and bytes, the query set and
device destroyed, and zero uncaptured GPU errors. This dynamic symmetric-int8
direction is abandoned without integration or a follow-up registration.
