# OPT-0021 — bounded-f16 Flash QK/PV partials

## Hypothesis

Flash attention still sustained substantially less useful arithmetic throughput
than the dense contractions. Six paired arms isolated whether grouping adjacent
products in f16 could reduce instruction pressure while keeping the softmax and
persistent output state in f32:

- current f32 QK and PV accumulation;
- QK K4 partials only;
- PV K4 partials only;
- combined QK K4 and PV K4 partials;
- f16 PV products with f32 accumulation;
- QK K4 plus f16 PV products.

Every candidate first had to keep narrow-shape NRMSE below 0.001 and cosine
above 0.99999. A production candidate then had to beat the retained Flash
kernel on the weighted 40 time-axis plus 40 frequency-axis call budget.

## Correctness

All five approximate arms passed the primitive gate. Across the candidates,
NRMSE ranged from 0.0001995 to 0.0003804, maximum absolute error was at most
0.0009766, and cosine similarity was at least 0.999999928.

## Result

| Arithmetic | Projected GPU time | Speedup | Change vs current |
| --- | ---: | ---: | ---: |
| Current Flash | 8,230.01 ms | 1.000× | — |
| QK K4 | 8,408.27 ms | 0.979× | 178.26 ms slower |
| PV K4 | 8,564.24 ms | 0.961× | 334.23 ms slower |
| QK K4 + PV K4 | 8,800.17 ms | 0.935× | 570.16 ms slower |
| f16 PV products | 8,283.75 ms | 0.994× | 53.74 ms slower |
| QK K4 + f16 PV products | 8,465.94 ms | 0.972× | 235.93 ms slower |

The current time/frequency medians were 194.281/11.469 ms. Every approximate
arm regressed both production geometries rather than merely losing in the
weighted aggregate.

## Disposition

Negative. The extra conversion and grouping instructions cost more than the
reduced f32 arithmetic on this GPU. A full waveform run was not warranted
because no arm passed the performance gate. All selectors, probes, profiler
arms, and shader variants were pruned.
