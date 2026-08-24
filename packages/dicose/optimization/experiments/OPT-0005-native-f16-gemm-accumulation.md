# OPT-0005 — native-f16 packed GEMM accumulation

## Hypothesis

Parakeet's native-f16 accumulator reaches roughly 2.8 TFLOP/s on the same
hardware class, while the exact f16-operands/f32-accumulator path is materially
slower. Converting the packed GEMM's running `vec4` state and FMA to f16 could
move the dominant 23.23-TFLOP family closer to native throughput.

## Risk and gate

This intentionally changes the numerical contract. ACE-Step rejected native
f16 accumulation after adversarial zero-collapse, overflow, cancellation, and
long-K drift. The candidate was therefore eligible only if it produced a
large full-graph wall improvement and remained healthy at the complete audio
boundary; a marginal timing result could not justify any numerical change.

## Result

The small K32 packed primitive happened to match all 896 reference f16 words,
but the complete model produced different stem and diagnostic statistics,
confirming that long-K contractions changed the graph. The supplied WAV still
passed the broad acceptance envelope with finite outputs and no GPU errors.

Isolated Chrome 151, zero warmups, one cold run:

| Boundary | Exact f32 state (ms) | Native f16 state (ms) | Speedup |
| --- | ---: | ---: | ---: |
| complete model timing | 45,206.9 | 44,723.6 | 1.01× |
| page end-to-end timing | 46,084.7 | 45,555.5 | 1.01× |

The difference is too small to separate from cold-run and thermal variance,
and is nowhere near enough to compensate for the weaker numerical contract.

## Disposition

Negative. Native-f16 accumulation was reverted. The production kernel retains
source-order f32 FMA state.
