# OPT-0020 — native-K4 M16/N256 owner geometries

## Hypothesis

The best OPT-0019 owner still held 32 FP32 accumulator scalars per lane. Two
materially different geometries tested the remaining ownership tradeoff:

- M16×N128: four rather than eight rows per subgroup, halving accumulator
  state while doubling row workgroups and packed-weight requests.
- M32×N256 on the two N1536 projections: two output vectors per lane, halving
  column workgroups while doubling FP32 accumulator state.

A per-shape selector had to beat native M32×N128 wherever selected, reach at
least 1.22× weighted versus exact, sustain at least 2.55 TFLOP/s on the four
major shapes, and save at least 2.0 seconds.

## Correctness

At M33/K384/N256, M16 exact, M16 K4, and M32×N256 K4 each matched the
corresponding native M32×N128 arithmetic control in all 8,448 f16 words.

## Result

| Projection | Exact | Native M32×N128 | M16×N128 | M32×N256 |
| --- | ---: | ---: | ---: | ---: |
| FF up | 42.4018 ms | 36.2742 ms | 41.2221 ms | 38.3713 ms |
| QKV + rotary | 41.8120 ms | 36.1103 ms | 41.0255 ms | 38.3713 ms |
| FF down | 41.0583 ms | 35.6844 ms | 40.5668 ms | — |
| attention output | 13.7298 ms | 12.0914 ms | 13.5987 ms | — |
| adapter | 10.3547 ms | 8.9457 ms | 10.2236 ms | — |

Every new arm lost to M32×N128. The forced new-arm compound projected
10,820.26 ms, only 1.0603×/651.95 ms better than exact, with 2.13–2.27
TFLOP/s on major shapes. It failed every declared gate.

## Disposition

Negative. M16 duplicated enough scheduling and weight traffic to overwhelm its
register reduction; N256 doubled live FP32 state without reducing physical
weight work. Both arms and the unselected native-layout scaffold were pruned.
