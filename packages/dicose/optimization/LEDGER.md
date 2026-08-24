# Optimization ledger

> The speed measurements remain valid, but pre-audit waveform/diagnostic gates
> were local comparisons and did not establish upstream correctness. Quality
> claims must now pass the reference gates described in
> [`CORRECTNESS_AUDIT.md`](CORRECTNESS_AUDIT.md).

| ID | Mechanism | Evidence | Disposition |
| --- | --- | --- | --- |
| OPT-0001 | 128-column subgroup GEMM ownership | Negative: correct but substantially slower on full DiCoSe. | Abandoned / reverted |
| OPT-0002 | Four-query-per-subgroup attention scheduling | Positive: raw f16-equivalent probe, faster cold full run, and release benchmark completed. | Integrated |
| OPT-0003 | Converter-native N128/N256 × K32 subgroup GEMM | Positive: raw-bit exact primitive and 1.64× faster full cold acceptance run. | Integrated |
| OPT-0004 | Persistent GPU RoPE sin/cos table | Positive: raw-bit exact probe and 1.09× faster full cold acceptance run. | Integrated |
| OPT-0005 | Native-f16 packed GEMM accumulation | Negative: changed full output but improved cold wall by only ~1%. | Abandoned / reverted |
| OPT-0006 | Fused attention gating and projection residuals | Positive: raw-bit exact probes and 1.04× faster full cold acceptance run. | Integrated |
| OPT-0007 | M64×N128/WG256 packed GEMM owner | Negative: exact but neutral at the full cold boundary. | Abandoned / reverted |
| OPT-0008 | Strided time attention without layout transposes | Positive: raw-bit exact probe, unchanged full output, and 1.04× faster full cold acceptance run. | Integrated |
| OPT-0009 | Producer-fused one-time K rotation | Positive: raw-bit exact, 1.06× faster than the immediate cold control, and final sustained median 53.67 s. | Integrated |
| OPT-0010 | Fuse CD condition adds into FF2 | Negative: exact only with workgroup staging, then neutral at the full cold boundary. | Abandoned / reverted |
| OPT-0011 | Production-shape GPU timestamp profiler | Positive: separated dense throughput from attention and localized ~24.6 s of baseline attention GPU time. | Integrated |
| OPT-0012 | Q64 attention with ascending K8 shared tiles | Positive: raw-bit exact; time/frequency kernels improved by 36.0%/30.8% against Q32. | Integrated |
| OPT-0013 | N128 FP32 owner with exact K4 source unrolling | Positive: raw-bit exact; production dense shapes reached 1.80–1.90 TFLOP/s. | Integrated |
| OPT-0014 | Subgroup-owned STFT adapter convolutions | Positive: raw-bit exact; entry 3×3 improved 11.5× and each hidden 1×1 improved 30.6×. | Integrated |
| OPT-0015 | Q64×K16 blockwise Flash attention | Positive: 1.65×/1.62× on time/frequency attention; full-waveform NRMSE stayed below 0.00036. | Integrated |
| OPT-0016 | Eight-row subgroup RMSNorm owner | Positive: raw-bit exact across 90 production-width/tail/FiLM cases; 1.52–1.70× at the main C384 shape, projecting ~0.13 s transformer saving. | Integrated |
| OPT-0017 | Bounded-f16 K2/K4 dense partials on the current K-major package layout | Mixed: K2 regressed every shape; K4 reached 2.24–2.29 TFLOP/s but only 1.082× weighted/~0.87 s projected saving. | Benchmark-only; not selected |
| OPT-0018 | Cooperative Flash score exponentials | Negative: raw-bit exact, but two production panels projected only 0.12–0.14 s saving because added synchronization offset SFU concurrency. | Abandoned / pruned |
| OPT-0019 | Converter-native per-output K4 dense layout | Mixed: raw-equivalent to transposed K4 and 1.153×/~1.52 s faster than exact, but only 1.068× over transposed K4, missing the declared layout-migration gate. | Abandoned / pruned |
| OPT-0020 | Native-K4 M16/N256 owner geometries | Negative: both were exact relative to their arithmetic controls but slower than M32×N128 on every eligible shape. | Abandoned / pruned |
| OPT-0021 | Bounded-f16 Flash QK/PV partials | Negative: all five isolated/combined arms passed narrow quality but regressed both production geometries; combined projected 0.57 s slower. | Abandoned / pruned |
| OPT-0022 | Stride-2 full transformer trunk with residual-delta reconstruction | Mixed: primitive geometry/RoPE gates passed and paired end-to-end wall fell 23.85→11.39 s, but waveform NRMSE was 0.16–0.48 with severe local errors. | All-network arm rejected; benchmark evidence retained |
| OPT-0023 | Weight-only truncated-SVD transformer projections | Negative preflight: all 96 dominant matrices are strongly full-rank; rank 256 saves only 16.7% of their FLOPs with a 25.2% optimal residual. | Abandoned before conversion/kernel work |
| OPT-0024 | Deterministic-only output with CD-only setup elided | Positive fast-mode performance: cold total 5.81 s versus 24.73 s refined (4.25×), sustained median 5.92 s; intentionally omits learned refinement and needs ground-truth/listening evidence. | Explicit fast mode; refined remains default |
| OPT-0025 | CD-only stride-2 transformer trunk with full deterministic conditioning | Positive historical evidence: deterministic diagnostics exact, refinement 2.30× and total 1.6486× in the controlled pair, 17.91-s thermally conservative sustained median, stem NRMSE 0.018–0.035. | Removed from the product; experiment evidence retained |
| OPT-0026 | Fast long-track 10% overlap aligned to the existing fade | `trust_nobody.wav` falls from 25 to 13 calls; isolated Chrome 151 sustained median 79.61 s (71.57–113.05 s), with 69.51 s in deterministic compute at the median sample. Listening quality remains unverified. | Integrated for explicit Fast only; Full retains 50% overlap; 30 s target not met |
| OPT-0027 | Extra Fast end-to-end hop-882 analysis, model, masks, and synthesis with original-position temporal RoPE | 1,101→551 frames and 44.76% of Fast arithmetic; supplied-WAV smoke passed, and `trust_nobody.wav` sustained median reached 28.04 s (26.62–28.82 s), but listening found the quality loss too large relative to Fast. | Rejected after listening; mode and implementation pruned; benchmark evidence retained |

See `experiments/` for the reproducible commands and exact measurements.
