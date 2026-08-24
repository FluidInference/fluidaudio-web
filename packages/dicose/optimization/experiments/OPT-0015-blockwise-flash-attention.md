# OPT-0015 — Q64×K16 blockwise Flash attention

## First-principles diagnosis

The exact Q64/K8 owner still performed one subgroup reduction and one online
softmax update for every query/key pair. Each of the 32 lanes redundantly held
the same max and denominator and executed the same exponential, while large
per-lane query/value state constrained occupancy. It sustained only about
0.56 TFLOP/s even though the exact dense family exceeded 2.1 TFLOP/s.

The selected owner maps a lane to one query/key score for four query rows. It
stages 64 FP32-rotated queries and a K16 f16 key/value tile, evaluates complete
64-wide QK dots as source-ordered FP32 FMAs without subgroup reductions, and
materializes a 64×16 FP32 score tile. One lane per query computes the tile
softmax. The same score tile is immediately consumed by P×V, with every V load
feeding four query accumulators. Max, denominator, and output state merge once
per K16 block.

This changes floating-point association but not the model operation: every
query attends every key; QK, softmax state, and P×V remain FP32; only storage
and final outputs are f16. The shader uses 25,344 bytes of workgroup storage,
so device creation explicitly requests that advertised adapter limit.

## Ownership panel

A matched Chrome 151 timestamp panel used two warmups and seven measured
passes. Compilation, upload, submission, and readback were excluded.

| Owner | Time 62×1,189 | Frequency 1,189×62 | Disposition |
| --- | ---: | ---: | --- |
| exact Q64/K8 control | 319.554 ms | 18.743 ms | Control retained |
| exact state-owner/K8 | 261.489 ms | 15.466 ms | Positive, superseded |
| exact state-owner/K16 | 255.984 ms | 15.401 ms | Positive, superseded |
| Flash Q32/K16 | 232.522 ms | 14.352 ms | Rejected: occupancy did not repay doubled workgroups |
| Flash Q64/K16 selected | 193.659 ms | 11.534 ms | Integrated |

The selected owner is 1.65× faster on the dominant time axis and 1.62× on the
frequency axis, reaching 0.927/0.812 effective TFLOP/s. Across forty blocks of
each geometry, the matched medians predict 5.32 seconds less GPU time.

The exact state-owner arms demonstrated that removing redundant softmax state
was independently useful, but their 1.22–1.25× result was materially below the
blockwise dataflow change. Q32 Flash fit under 16 KiB but lost to Q64 because
duplicated K/V loads and synchronization outweighed the possible occupancy.
Those arms were removed rather than retained as runtime switches.

## Numerical and waveform quality

The narrow q64-versus-Flash probe reports 20 changed words out of 56,832,
NRMSE 9.40e-6, max absolute error 0.00048828125, and cosine
0.9999999999558. This is deliberately a quality comparison, not a claim of
raw-bit equivalence.

The full supplied WAV was then run twice with the same decoded PCM and CD
noise seed: exact Q64 followed by Flash. All 1,048,576 stereo samples per stem
were compared directly.

| Stem | Waveform NRMSE | SNR | Cosine | Worst 4,096-sample window NRMSE |
| --- | ---: | ---: | ---: | ---: |
| drums | 0.000239 | 72.44 dB | 0.999999972 | 0.001352 |
| bass | 0.000356 | 68.98 dB | 0.999999937 | 0.009327 |
| other | 0.000278 | 71.12 dB | 0.999999961 | 0.000831 |
| vocals | 0.000237 | 72.52 dB | 0.999999972 | 0.000314 |

Maximum absolute error was at most 0.063% of the corresponding reference
peak. RMS drift was at most 0.0016% and peak drift at most 0.0151%. The
candidate therefore passed both global and localized waveform gates.

## Integrated acceptance

A fresh-profile cold candidate run completed in 24,157.7 ms end-to-end with
23,438.3 ms model timing: 4,998.0 ms deterministic, 16,460.9 ms refinement,
and 1,360.7 ms ISTFT. Every output was finite and remained inside the recorded
upstream acceptance contract. This is an acceptance sample, not a sustained
thermal median.

## Disposition

Integrated as the production attention owner. Exact Q64 remains as the
explicit waveform-control path; intermediate exact-state, Q32 Flash, and old
Q32 grouped owners are not public runtime choices.
