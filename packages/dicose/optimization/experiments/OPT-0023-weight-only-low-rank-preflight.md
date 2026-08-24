# OPT-0023 — weight-only low-rank preflight

## Hypothesis

If the dominant transformer matrices were naturally low-rank, converter-time
truncated SVD could replace one large contraction with two smaller ones and
remove dense arithmetic without retraining.

All 96 shipped QKV, FF-up, and FF-down matrices were decoded directly from the
converter-native f16 package. The audit used optimal singular values, so its
errors are lower bounds before f16 factor quantization and the additional f16
activation boundary required by two WebGPU dispatches.

For these 384↔1,536 matrices, rank `r` has both FLOP and factor-storage ratio
`r / 307.2`.

## Result

| Rank | FLOP ratio | Pooled relative Frobenius residual | Worst residual |
| ---: | ---: | ---: | ---: |
| 64 | 20.83% | 0.6594 | 0.7772 |
| 96 | 31.25% | 0.5679 | 0.6995 |
| 128 | 41.67% | 0.4907 | 0.6260 |
| 192 | 62.50% | 0.3617 | 0.4865 |
| 256 | 83.33% | 0.2519 | 0.3526 |
| 320 | 104.17% | 0.1493 | 0.2169 |

Pooled residual reaches 20% only at rank 289, which retains 94.1% of the
original FLOPs. It reaches 10% only at rank 349, which costs 13.6% more than
the original contraction. Keeping every matrix below 20% requires rank 328
and is also more expensive.

Rank 256 would remove only about 3.48 TFLOP from the complete graph while its
optimal weight residual remains 25.2%. Dispatch and intermediate-activation
traffic would reduce the theoretical saving further. Splitting Q, K, and V
does not create a useful crossing, and only two of 96 individual matrices have
rank-256 residual below 15%.

## Disposition

Negative at preflight. Weight-only SVD does not justify package conversion,
kernel work, or waveform testing. A useful low-rank model would require
retraining, distillation, or activation-aware calibration rather than a free
inference-time factorization.
