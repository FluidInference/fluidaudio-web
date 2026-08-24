# OPT-0012 — Q64 attention with ascending K8 shared tiles

## Hypothesis

The Q32 owner used eight fixed-32 subgroups and four scalar query streams per
subgroup. Every workgroup loaded the same K/V row and crossed two barriers for
each key. Doubling ownership to eight streams keeps the 256-thread workgroup
but halves query workgroups. Loading eight ascending K/V rows into shared
memory then amortizes the barriers across eight numerically unchanged online
softmax updates.

The online update also observes that one of
`exp(old_max - next_max)` and `exp(score - next_max)` is always `exp(0)`.
The selected shader computes the nontrivial exponential and substitutes exact
literal 1 for the other branch while preserving key order and FP32 state.

## Rejected arms

- The one-exponential rewrite by itself was raw-bit exact but moved the
  582.94 ms time kernel only to 571.80 ms in one profile; it was not the main
  mechanism.
- Packing four query streams into `vec4<f32>` and hoisting gate denominators
  was exact but regressed to 583.79/32.90 ms for time/frequency attention.
- Q96 crossed the private-state pressure cliff at 529.66/44.43 ms.
- Q128 was substantially slower and produced an invalid zero timestamp for
  one long-shape sample. It was rejected immediately.
- Q64 without key blocking was positive; K4 was materially better; K8 gave a
  smaller final step. Wider key tiles were not chased after that diminishing
  return.

## Kernel evidence

One matched seven-sample Chrome 151 profile measured:

| Owner | Time 62×1,189 | Frequency 1,189×62 |
| --- | ---: | ---: |
| Q32/K1 control | 575.67 ms | 31.33 ms |
| Q64/K1 | 454.43 ms | 26.35 ms |
| Q64/K4 | 376.50 ms | 21.63 ms |
| Q64/K8 selected | 368.44 ms | 21.69 ms |

The selected owner is 36.0% faster on the dominant time axis and 30.8% faster
on the frequency axis. Across forty blocks of each axis, the isolated medians
predict an 8.67-second GPU-time reduction.

## Numerical contract

Every query retains its own scalar FP32 max, denominator, and two context
states. K tiles are consumed in exactly ascending order. The query8 control
and selected owner have zero mismatches across 56,832 raw f16 words for
contiguous, strided, and producer-rotated-K paths. The complete WAV output
statistics remain unchanged.

The integrated OPT-0012/0013 stack completed a cold supplied-WAV acceptance
run in 39,168.3 ms end-to-end (38,341.1 ms model timing): 9,500.8 ms for the
deterministic stage and 26,641.5 ms for four refinements. Every final stem and
model-diagnostic peak/RMS statistic matched the former Q32/K1 stack exactly.
This cold sample is an acceptance receipt, not a sustained thermal median.

## Disposition

Integrated. The public runtime keeps only the Q32 correctness control and the
selected Q64/K8 owner; Q64/K1, Q64/K4, Q96, Q128, and vectorized-stream arms
were removed rather than left as dormant switches.
