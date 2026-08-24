# Production arithmetic budget

This is the first-principles budget for the supplied 11.89-second WAV after it
becomes 1,189 STFT frames and 62 learned frequency bands. The transformer sees
`R = 1,189 × 62 = 73,718` rows. Logical FLOPs count a multiply-add as two and
do not claim that every operation maps to identical hardware instructions.

## Where the arithmetic comes from

The deterministic network executes 16 transformer blocks (eight layers over
two axes). Four consistency-distilled stem refinements execute another 64, for
80 blocks total.

| Dense projection per block | Logical GFLOP | Calls | Total TFLOP |
| --- | ---: | ---: | ---: |
| QKV 384→1,536 | 86.961 | 80 | 6.957 |
| gates 384→8 | 0.453 | 80 | 0.036 |
| attention output 512→384 | 28.987 | 80 | 2.319 |
| FF up 384→1,536 | 86.961 | 80 | 6.957 |
| FF down 1,536→384 | 86.961 | 80 | 6.957 |
| **Transformer dense subtotal** | **290.324** | **80** | **23.226** |

Condition adapters, band split, mapping, and mask estimators add about 1.226
TFLOP, giving approximately **24.452 TFLOP of dense work**.

Each of the five network evaluations has eight time-axis and eight
frequency-axis attention calls. Exact all-key attention therefore contributes:

| Attention geometry | Logical GFLOP/call | Calls | Total TFLOP |
| --- | ---: | ---: | ---: |
| 62 sequences × 1,189 tokens | 179.509 | 40 | 7.180 |
| 1,189 sequences × 62 tokens | 9.360 | 40 | 0.374 |
| **Attention subtotal** | | | **7.555** |

The four STFT-adapter convolutions add 0.102 TFLOP. The resulting accounted
total is about **32.11 TFLOP**, before small elementwise and DSP work.

## Measured reconciliation and hard floor

Retained production-shape medians predict roughly 12.65 seconds for the four
large dense projections across 80 blocks and 8.21 seconds for Flash attention.
Those two families alone explain about 20.86 seconds of the 23.44-second cold
model wall, so scheduling and elementwise cleanup cannot produce another order
of magnitude.

Even the impossible best case where all 32.11 TFLOP—including softmax work—ran
continuously at Parakeet's measured 2.7 TFLOP/s has a lower bound of **11.89
seconds**, before ISTFT or browser overhead. Higher utilization can reach
roughly realtime; substantially sub-realtime execution also requires fewer
model operations.

## Priority consequences after the measured branches

1. Do not use whole-graph logical TFLOP/s as a contraction-utilization metric.
   The retained exact dense kernels sustain roughly 2.05–2.13 TFLOP/s; the
   former 0.63 figure mixed dense contractions, online softmax, DSP, dispatch,
   and browser wall time into one denominator.
2. Exact-kernel headroom is no longer large enough to meet a substantially
   sub-realtime goal. Bounded-f16 dense partials saved only a projected 0.87 s,
   native-K4 layout work missed its migration gate, and bounded-f16 Flash
   partials regressed both production geometries. Reopen these families only
   for a materially different algorithm with a multi-second projection.
3. Removing evaluations has the highest demonstrated leverage. Returning the
   already-computed deterministic separator output removes all four CD calls
   and reaches a 5.92-s sustained median. This needs ground-truth and listening
   qualification, not another kernel benchmark.
4. Fast long-track scheduling now overlaps only the existing 10% fade region.
   Relative to applying Full's overlap schedule to Fast, `trust_nobody.wav`
   uses 13 deterministic graph calls instead of 25, a 1.92× chunk-count
   reduction. Its isolated-Chrome sustained median is 79.61 s, including
   69.51 s of deterministic compute in the median sample. This is not evidence
   that seam quality is acceptable, and it does not meet the 30-second target.
5. Reducing only the CD temporal rows previously preserved deterministic
   diagnostics exactly and measured 14.08 s in a controlled pair, with
   1.8–3.5% global waveform NRMSE. That Balanced experiment was not promoted;
   its runtime switch was removed and OPT-0025 retains the historical evidence.
6. If inference-time token reduction fails its listening gate, the reliable
   fallback is a trained smaller graph: distill a lower-token deterministic
   trunk and/or a joint/cheaper refinement policy.
   Untrained all-network stride-2 caused 0.16–0.48 NRMSE, while optimal
   weight-only SVD had no useful error/FLOP crossing. Batch-processing stems,
   skinny projections, normalization, and elementwise fusion do not remove
   enough arithmetic to change this conclusion.
7. OPT-0027 tested a separate Extra Fast path that changed the complete
   STFT/model/mask/ISTFT temporal grid from hop 441 to 882. It reduced a fixed
   item from 1,101 to 551 frames and the 13-call long-track budget from 75.687
   to 33.875 TFLOP, reaching a 28.04-s `trust_nobody.wav` sustained median.
   Listening found the quality loss unacceptable relative to Fast, so the mode
   and its half-rate implementation were rejected and pruned.
