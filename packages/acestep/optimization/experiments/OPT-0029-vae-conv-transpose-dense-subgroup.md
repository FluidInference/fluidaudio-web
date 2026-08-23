# OPT-0029 — Dense-tiled VAE ConvTranspose1D

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Date: 2026-08-15
- Risk: exact

## Hypothesis

OPT-0026 is exact and 3.23x faster than production, but its authoritative
long-family projection is still 13.12 seconds for only 3.47 TFLOPs of work.
A dense-style fixed32 tile (32 phase rows x 256 output channels, eight rows and
eight channels per lane) can reuse every input and weight across substantially
more outputs while reading the existing converter-native polyphase layout.
Keeping the loop order tap then increasing Cin preserves OPT-0026 output bits.

## Gate

Compare against OPT-0026 over the same five production shapes in stock Chrome.
Require zero raw-U16 mismatches and at least 1.75x weighted speedup. Escalate
directly into OPT-0028 if it passes; otherwise retain OPT-0026.

## Identity

- Allocation baseline: `72de722`
- Machine: MacBook Air M3, 10 GPU cores, 16 GB
- Browser/API: stock Chrome WebGPU

## Results

- Stock Chrome compiled and dispatched the OPT-0026 control and dense candidate
  for all five C512 production shapes. Preparation compared `141,312,000`
  output U16 words over `49,610,752` shared revision-6 polyphase-weight U16
  words with zero mismatches, zero remaining prefill words, and exact raw-U16
  identity.
- After the required 30-second nominal thermal gate (level 0), the one
  authorized four-round alternating AB/BA run measured:

| Block | OPT-0026 samples (ms) | Dense samples (ms) | Medians (ms) | Speedup |
| --- | --- | --- | ---: | ---: |
| 0, C2048 -> C1024, stride 10 | 84.5, 74.6, 62.6, 66.5 | 59.6, 62.6, 48.5, 52.0 | 70.55 -> 55.80 | 1.264337x |
| 1, C1024 -> C512, stride 6 | 117.7, 106.4, 89.0, 95.5 | 90.9, 92.2, 69.7, 74.5 | 100.95 -> 82.70 | 1.220677x |
| 2, C512 -> C256, stride 4 | 108.3, 84.9, 90.1, 97.4 | 79.6, 67.3, 65.7, 71.5 | 93.75 -> 69.40 | 1.350865x |
| 3, C256 -> C128, stride 4 | 114.7, 90.7, 92.4, 97.1 | 163.1, 130.1, 128.6, 139.9 | 94.75 -> 135.00 | 0.701852x |
| 4, C128 -> C128, stride 2 | 120.9, 93.8, 95.4, 94.6 | 163.5, 129.9, 133.9, 140.9 | 95.00 -> 137.40 | 0.691412x |
| **Sum of medians** |  |  | **455.00 -> 480.30** | **0.947325x** |

- The candidate missed the required `1.75x` aggregate gate and was `25.30 ms`
  slower across the five medians. The receipt's simple C4500 projection is
  `6,825.00 ms` for OPT-0026 versus `7,204.50 ms` for dense, a regression of
  `379.50 ms`.
- Preparation took `4,597.10 ms`; the timed section took `3,842.70 ms`. The
  page destroyed all 30 owned buffers and the device, and Chrome reported no
  warning or error logs.

The canonical receipt is
[`optimization/results/OPT-0029/result.json`](../results/OPT-0029/result.json),
SHA-256 `7c1db94aa4f7258cd61692f9e80775cf939509bc2559a24e00848ef886582b82`.

## Evidence and disposition

Negative evidence. The dense tile improved the three widest-channel blocks,
but substantially regressed both 128-output-channel blocks and made the exact
five-shape aggregate slower than OPT-0026. Retain OPT-0026 and do not integrate
this geometry. Revisit only with materially different ownership that addresses
the low-channel shapes, or after a compiler/backend or GPU-architecture change;
do not repeat the unchanged dense tile.
