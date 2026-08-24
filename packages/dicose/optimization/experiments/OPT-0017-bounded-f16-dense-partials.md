# OPT-0017 — bounded-f16 dense partials on the current package layout

## First-principles basis

The exact packed owner converts f16 operands to f32 and updates every f32
accumulator once per K value. Parakeet's 2.7-TFLOP/s path instead keeps native
f16 contraction state, which is too numerically fragile over K384–K1536.

This experiment tested two bounded compromises while retaining the existing
WG128/M32×N128 owner and f32 running state:

- K2: two adjacent f16 products are reduced with `dot(vec2<f16>)`, widened,
  and added to f32 state twice per K4 group.
- K4: four adjacent products are reduced with `dot(vec4<f16>)`, widened once,
  and added to f32 state once per K4 group.

The current converter layout is K-major across four adjacent outputs. Both
arms therefore transpose a 4×4 K/output block in registers before each dot.
This isolates arithmetic without changing the package, but it is not the
per-output K4-native layout used by ACE's faster arm.

## Numerical probe

Chrome 151 compared exact, K2, and K4 at M33/K384/N128 using deterministic
signed f16 operands, a partial row tile, identical packed weights, and complete
f16 writes.

| Arm | Changed words / 4,224 | NRMSE | Maximum absolute | Cosine |
| --- | ---: | ---: | ---: | ---: |
| K2 | 1,826 | 0.0003333 | 0.00390625 | 0.999999948 |
| K4 | 2,175 | 0.0004130 | 0.00390625 | 0.999999916 |

Both candidates were finite and deterministic. Reproduce with:

```sh
pnpm test:webgpu
```

## Balanced production-shape panel

The profiler compiled and warmed all arms, then used six balanced exact/K2/K4
orders with one timestamped compute pass and drained submission per sample.

| Projection | Exact TFLOP/s | K2 TFLOP/s | K4 TFLOP/s | Exact→K4 |
| --- | ---: | ---: | ---: | ---: |
| FF1 + GELU | 2.059 | 1.995 | 2.243 | 1.090× |
| QKV + rotary | 2.080 | 2.009 | 2.253 | 1.083× |
| FF2 + residual | 2.118 | 2.045 | 2.288 | 1.080× |
| attention output | 2.126 | 2.015 | 2.262 | 1.064× |
| adapter | 2.093 | 2.017 | 2.257 | 1.078× |

Weighted by the actual `80/80/80/80/34` call counts:

| Arm | Projected GPU time | Speedup | Saving vs exact |
| --- | ---: | ---: | ---: |
| exact | 11,452.35 ms | 1.0000× | — |
| K2 | 11,869.42 ms | 0.9649× | -417.07 ms |
| K4 | 10,582.62 ms | 1.0822× | 869.73 ms |

Every K2 shape regressed. Every K4 shape won with non-overlapping ranges, but
the weighted result missed the predeclared 1.15× and 1.5-second materiality
gate. A full-waveform A/B was deliberately skipped because the primitive did
not earn production escalation.

Reproduce with:

```sh
DICOSE_PROFILE_FOCUS=dense pnpm profile:webgpu
```

## Disposition and next experiment

K2 is rejected. Register-transposed K4 is benchmark-only and not a production
default. The result identifies physical layout—not another loop unroll—as the
next dense lever: pack each output's consecutive K4 operands together so the
native f16 dot consumes direct vector loads. That materially different layout
must beat this arm and exact before any waveform gate or runtime selection.
