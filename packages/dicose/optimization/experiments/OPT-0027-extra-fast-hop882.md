# OPT-0027 — Extra Fast half-rate STFT

## First-principles target

Fast still evaluates about 75.687 logical TFLOP across the 13 fixed-size calls
needed for `trust_nobody.wav`. Even a fictional uniform 2.7 TFLOP/s therefore
spends 28.03 seconds on model arithmetic before STFT, ISTFT, readback, or browser
overhead. Reaching 30 seconds requires fewer transformer rows rather than
another isolated kernel improvement.

The Extra Fast candidate was tested as a third, explicit mode while leaving
Full and Fast unchanged. It kept the 2,048-sample FFT and released deterministic
checkpoint, but changed the analysis/synthesis hop from 441 to 882 samples and
evaluated temporal RoPE at original positions `0, 2, ..., 1100`.

For one 485,100-sample item:

| Family | Fast TFLOP | Extra Fast TFLOP |
| --- | ---: | ---: |
| Transformer dense | 4.3014 | 2.1526 |
| Time attention | 1.2314 | 0.3084 |
| Frequency attention | 0.0693 | 0.0347 |
| Mask estimators | 0.2165 | 0.1084 |
| Band split | 0.0035 | 0.0017 |
| **Total** | **5.8221** | **2.6058** |

The graph retains **44.76%** of Fast's accounted arithmetic. Across 13 chunks,
the projection is 75.687 → 33.875 TFLOP.

## Geometry and correctness boundary

For the same centered window, hop-882 frame `j` is bit-identical to hop-441
frame `2j`. A fixed item therefore becomes 1,101 → 551 frames. The existing
low-overlap schedule remains exactly aligned:

| Quantity | Samples | Hop-882 intervals |
| --- | ---: | ---: |
| Model item | 485,100 | 550 |
| Step | 436,590 | 495 |
| Fade and reflected border | 48,510 | 55 |

The periodic Hann window with hop 882 remains an oversampled, normalized
analysis/synthesis pair. Unit gates cover exact even-frame equivalence for both
odd and even hop-441 frame counts, minimum and arbitrary-length round trips,
exact output length, and overlap alignment. The output-spectrum metadata carries
hop 882 into ISTFT; leaving the released 441 value there would silently zero-fill
roughly half the requested waveform.

Only time-axis QKV rotation and attention use `positionStride=2` with
`positionLimit=(tokens-1)*2`. Frequency attention remains at stride 1. The
existing raw-WebGPU probe reports zero fused-QKV and mapped-attention mismatches
and proves that original-position RoPE changes the result versus compressed
positions.

This arm shares the even-frame transformer predictions of OPT-0022's rejected
stride-2 trunk. Its new behavior is coherent half-rate mask estimation and
hop-882 synthesis rather than an interpolated hidden residual feeding a
full-rate mask head. That removes one mismatch, but it does not establish
perceptual quality. The checkpoint was trained at hop 441, so listening remains
the acceptance gate.

## Supplied-WAV smoke

A fresh isolated Chrome 151 process completed the supplied 11.89-second WAV in
3.11 seconds end to end, including 1.76 seconds of deterministic model time and
0.39 seconds of ISTFT. Every stem had the exact restored 262,144-sample,
22.05-kHz timeline, finite samples, nonzero final-window energy, deterministic
diagnostics, and zero mapping/refinement time.

## Sustained long-track measurement

The source was `/Users/hamza/Desktop/trust_nobody.wav`, SHA-256
`d3c1378d287bbd0bb2b1f294806015cce572cfafe6c495b2f6dbf46432322a1b`. After
resampling it contains 5,608,109 model-rate samples and uses the same 13 chunks
as Fast.

A fresh isolated Chrome 151 process ran one warmup followed by three measured
passes through the same worker and model package:

| Sample | Wall time |
| ---: | ---: |
| 1 | 26.62 s |
| 2 | 28.04 s |
| 3 | 28.82 s |

The sustained median is **28.04 seconds** (range **26.62–28.82 seconds**),
meeting the 30-second target. The median sample reports 21.93 seconds of
deterministic model compute, 1.16 seconds of preparation, and 4.49 seconds of
ISTFT. No approximate K4 accumulation, persistent graph, Wasm SIMD FFT, or
GPU/CPU chunk pipeline is included.

## Disposition

Rejected after listening. Although the numeric gates passed and the candidate
met the timing target, its perceptual degradation was too large relative to
Fast. The public mode and all hop-882/temporal-position-remapping implementation
and probe code were removed. Full remains the released refined default, Fast
remains the full-temporal-resolution deterministic path, and this document is
retained only as benchmark evidence.
