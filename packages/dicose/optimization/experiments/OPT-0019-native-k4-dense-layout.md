# OPT-0019 — converter-native per-output K4 dense layout

## Hypothesis

OPT-0017's bounded-f16 K4 arm had to rebuild each output column's four K
operands from four K-major `vec4` loads. A physical layout that stores those
four operands together could remove the 4×4 register transpose and approach
the 2.7-TFLOP/s Parakeet path while retaining FP32 running state.

The benchmark-only layout for logical `W[K,N]` was:

```text
[N/128, K/4, output4, lane32, K4]
```

with scalar index:

```text
(((((n/128)*(K/4)+k/4)*4+(n%4))*32+((n%128)/4))*4+(k%4))
```

One direct `vec4<f16>` load therefore supplied K4 for one output, and four
loads produced four adjacent outputs without swizzling. Synthetic buffers had
the same bytes as production; the public converter, 594-MiB package, manifest,
and runtime default were not changed.

## Gate

The layout had to win every production shape, reach at least 1.15× and save at
least 1.5 seconds versus exact, and improve at least 1.10× over OPT-0017's
transposed K4. The last condition prevents a 129.6-MiB package-layout migration
when almost all gain comes from the already-marginal arithmetic arm.

## Correctness

The native layout's exact source-order path matched the current exact owner in
all 4,224 f16 probe words. Its direct K4 path matched the current transposed K4
path in all 4,224 words, proving packing/indexing and identical bounded-dot
association.

## Confirmation timing

Six balanced production-shape rounds reported:

| Projection | Exact | Transposed K4 | Native K4 | Native TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| FF up | 42.2707 ms | 38.8301 ms | 36.2086 ms | 2.402 |
| QKV + rotary | 41.6481 ms | 38.7318 ms | 36.1759 ms | 2.404 |
| FF down | 41.0255 ms | 38.1092 ms | 35.7499 ms | 2.432 |
| attention output | 13.7298 ms | 12.8123 ms | 12.1242 ms | 2.391 |
| adapter | 10.3219 ms | 9.5683 ms | 8.9457 ms | 2.430 |

Weighted by `80/80/80/80/34`, exact was 11,444.88 ms, transposed K4 was
10,603.99 ms, and native K4 was 9,924.84 ms. Native K4 won every shape, reached
1.1532× and saved 1,520.04 ms versus exact, but improved only 1.0684× over
transposed K4. An independent first panel reached the same decision
(1.1566×/1,552.22 ms versus exact and 1.0680× versus transposed K4).

## Disposition

The layout missed its predeclared 1.10× incremental gate. No checkpoint
conversion, package duplication/replacement, waveform A/B, or production
selection occurred. Experimental runtime/layout code was pruned.
