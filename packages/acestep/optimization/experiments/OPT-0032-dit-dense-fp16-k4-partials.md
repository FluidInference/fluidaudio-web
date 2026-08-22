# OPT-0032 — DiT dense FP16 K4 partials

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Risk: approximate bounded FP16 partial reduction with FP32 running state

## Hypothesis

Parakeet reaches 2.6–2.8 TFLOP/s mainly through native FP16 accumulation; its
fair FP32 comparator is only about 1.7–1.8 TFLOP/s, matching ACE's isolated
FP16/FP32 kernel. Full FP16 accumulation is already rejected. The positive
OPT-0024 result shows a narrower compromise that this Chrome/M3 compiler
executes efficiently: form each consecutive K4 dot in FP16, widen once, and
retain the running accumulator across K4 groups in FP32.

## Frozen mechanism

- Barrier-free fixed32 WG128, output tile `M32 x N128`.
- Four subgroups; each owns eight rows by 128 columns, and each lane owns
  eight rows by four adjacent columns (32 FP32 accumulators).
- Benchmark-packed FP16 B layout `[N/128,K/4,4,32,4]` so each lane directly
  loads four K4 vectors for its four outputs. A remains FP32 storage and is
  rounded to one broadcast FP16 K4 vector per owned row.
- For increasing K4 groups, compute four FP16 `dot(vec4)` results, widen the
  result vector once, and add once to the FP32 accumulator. Store FP32.
- No shared memory, barriers, split-K, or full-FP16 running accumulator.

This is not OPT-0020: that experiment used FP32 horizontal dot inside a
cooperative K panel and regressed badly. This candidate uses native FP16 K4
partials across four independent outputs and keeps FP32 state between groups.

## Gate

Cover all four M2250 production shapes, complete writes, determinism, finite
outputs, and adversarial signed-zero/cancellation/range/long-K fixtures.
Report max/RMS/relative error, NRMSE, SNR, Pearson, class changes, and nonfinite
counts against OPT-0009. After correctness, use balanced one-shot nominal
timing. Continue only if the weighted `4/2/2/1` score is at least `1.35x` and
the declared numerical envelope passes. Any promotion later requires actual
layer, denoise trajectory, final latent, waveform, instrumental/vocal
listening, and product gates; this primitive result changes no package or
runtime selection.

## Result

Stock Chrome 151/M3 compiled every full and adversarial candidate. The four
complete production shapes compared `25,344,000` FP32 outputs; candidate
reruns were raw-U32 deterministic, all writes completed, all values were
finite, guards remained intact, and the tail row was written.

The aggregate full-shape numerical result versus OPT-0009 was:

- NRMSE `0.0003114215`, SNR `70.1330 dB`, Pearson
  `0.9999999524`;
- RMS error `0.00181183`, mean absolute error `0.00140121`, maximum absolute
  error `0.0144317`; and
- zero non-finite or signed-zero differences. There were `2,484` sign/zero
  class changes among values close to zero.

The signed-zero, K4-cancellation, finite-range, and K6144 adversarial suite
covered another `17,408` values and passed at NRMSE `0.000035894`, SNR
`88.8996 dB`, Pearson `0.9999999994`, and max error `0.0105618`.

The first preparation stopped before timing only because a per-element
relative-RMS veto divides by near-zero reference values: it reported
`0.9464` even while global NRMSE was `0.000311`. That ill-conditioned redundant
veto was removed; relative error remains reported, while finite/class, NRMSE,
SNR, Pearson, max-error, and adversarial gates remain binding.

After one 30-second nominal check, the accepted forward-AB/reverse-BA timing
was:

| Shape | Multiplicity | OPT-0009 mean | K4 mean | Speedup |
| --- | ---: | ---: | ---: | ---: |
| H→H | 4 | 17.3500 ms | 11.1000 ms | 1.56306x |
| H→1024 | 2 | 7.0500 ms | 5.2000 ms | 1.35577x |
| H→6144 | 2 | 37.1000 ms | 28.7500 ms | 1.29043x |
| 6144→H | 1 | 41.9500 ms | 29.8000 ms | 1.40772x |
| **Weighted 4/2/2/1** | | **199.6500 ms** | **142.1000 ms** | **1.404996x** |

Every shape won and the weighted result cleared the `1.35x` primitive gate.
Applying only this ratio to the earlier `~42.85 s` 192-layer dense projection
suggests about `30.50 s`, or `~12.35 s` saved. That is prioritization
arithmetic, not a measured M2250 graph or product result.

Evidence is positive and authorizes converter-native layout work plus actual
layer, denoise-trajectory, final-latent, waveform, and listening gates. It does
not authorize production selection. Cleanup destroyed all 48 buffers and the
device idempotently. The canonical receipt is
[`optimization/results/OPT-0032/result.json`](../results/OPT-0032/result.json),
SHA-256 `64d5aa235c59935b0644362a24a324062598cf607b731c289e7a9dec55b3ab15`.
