# OPT-0013 — N128 FP32 owner with exact K4 source unrolling

## Hypothesis

The converter-native N256 shader combined Parakeet's wide output ownership
with FP32 running state: sixteen `vec4<f32>` accumulators, or 64 scalar FP32
values, per lane. Parakeet uses that footprint for native-f16 accumulation but
deliberately narrows its FP32 owner to N128.

The selected arm reads each physical N256 weight tile as two logical N128
owners, halving accumulator pressure without repacking weights. It also loads
four adjacent f16 activations at once and source-unrolls their four FP32 FMAs.
The FMA sequence remains K0, K1, K2, K3, so this is load/compiler shaping, not
K4 partial accumulation and not approximate arithmetic.

## Interaction result

The mechanisms had to be tested together. On 73,718×384×1,536:

| Arm | GPU TFLOP/s |
| --- | ---: |
| N256/K1 control | 1.582 |
| N256/K4 loads | 1.311 |
| N128/K1 | 1.624 |
| N128/K4 selected | 1.798 |

K4 source unrolling made the high-pressure N256 owner worse, while the same
unrolling made the bounded N128 owner substantially faster. This closes the
tempting but incorrect idea of applying K4 mechanically to every geometry.

## Production-shape evidence

A matched seven-sample Chrome 151 profile measured:

| Boundary | Control | Selected | Gain |
| --- | ---: | ---: | ---: |
| FF1 384→1,536 + GELU | 1.576 | 1.823 TFLOP/s | 15.7% |
| QKV 384→1,536 + fused K rotation | 1.556 | 1.818 TFLOP/s | 16.8% |
| FF2 1,536→384 + residual | 1.626 | 1.858 TFLOP/s | 14.3% |
| attention output 512→384 + residual | 1.626 | 1.835 TFLOP/s | 12.9% |
| adapter 384→384 | 1.684 | 1.896 TFLOP/s | 12.6% |

For the four transformer projections alone, those medians predict about 1.94
seconds less GPU time across 80 blocks. Large packed rows select N128/K4;
tiny row-one mapping projections retain the lower-dispatch original owner.

## Correctness

The raw browser probe compares the physical-N256/N128/K4 owner to the generic
source-order reference and reports zero mismatches. It also validates fused K
rotation for contiguous and strided layouts through attention, again with zero
mismatches across 56,832 f16 words. No native-f16 accumulator or bounded-dot
approximation is used.

Together with OPT-0012, the selected dense owner completed the full supplied
WAV in 39,168.3 ms end-to-end (38,341.1 ms model timing), with deterministic
and four-refinement stages of 9,500.8 and 26,641.5 ms. All final output and
diagnostic statistics remained bit-for-bit unchanged. Sustained measurement
is deferred until the GPU has cooled; consecutive long runs on this machine
have already ranged from 39 to 64 seconds under thermal saturation.

## Disposition

Integrated for packed projections with at least one full 32-row tile. This
brings the production dense family into the 1.80–1.90 exact-FP32 TFLOP/s range
on the tested Chrome/Metal stack.
