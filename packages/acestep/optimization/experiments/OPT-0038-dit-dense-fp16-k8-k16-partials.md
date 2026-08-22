# OPT-0038 — DiT dense FP16 K8/K16 partials

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Risk: approximate bounded K8/K16 FP16 partial reduction with FP32 running state

## Hypothesis

OPT-0032's K4 partials are numerically strong and `1.404996x` faster, but each
output still updates its FP32 dependency chain once per four reduction terms.
Using the same converter-native K4 layout, combine two or four consecutive
FP16 dot4 results in FP16 and widen once per K8 or K16 block. This cuts FP32
updates another 2-4x while bounding low-precision reduction to at most sixteen
terms—far shorter than the already-rejected full K2048/K6144 FP16 accumulator.

## Frozen mechanism and gate

Keep OPT-0032's barrier-free WG128 M32×N128 ownership and weight layout. Add
only K8 and K16 partial variants, preserving increasing block/K4 order and FP32
running/output state. Compare OPT-0009, K4, K8, and K16 across all four M2250
shapes plus the signed-zero, cancellation, finite-range, and long-K adversarial
fixtures. Require finite deterministic complete outputs, the OPT-0032 global
NRMSE/SNR/Pearson/max-error envelope, and no qualitatively new overflow/zero
collapse. Time one balanced gate after the ordinary nominal check. A variant
must improve weighted wall by at least `1.15x` over K4 and remain within the
numerical envelope to supersede it; otherwise retain K4 and stop. No package,
trajectory, listening, or production selection is authorized here.

## Result

Both bounded variants passed the full and adversarial numerical envelopes,
were deterministic and finite, and cleaned up all resources. Full-shape
K4/K8/K16 NRMSE was `0.000311/0.000382/0.000454`, SNR was
`70.13/68.36/66.86 dB`, and maximum absolute error was
`0.01443/0.01733/0.01992`. Finite-to-zero rates were respectively
`0.0395/0.2367/0.6313` events per million full outputs, with one event each in
the adversarial set. An initial preflight used an arbitrary `4x K4` count gate
and stopped despite every declared numerical envelope passing; before timing,
that was replaced with a scale-aware maximum of one event per million plus no
adversarial increase. No timing occurred under the rejected preflight.

One stock-Chrome/M3 timing launch followed a 30.074-second idle and a single
thermal-level-0 observation. Weighted `4/2/2/1` walls were `205.40 ms` for
OPT-0009, `157.35 ms` for K4, `161.45 ms` for K8, and `241.30 ms` for K16.
K8 was `0.974605x` and K16 `0.652093x` versus K4, so neither cleared the
frozen `1.15x` gate. K8 improved H-H (`1.1932x`) and H-1024 (`1.0636x`) but
regressed both high-work shapes; extra bounded FP16 state/unrolling outweighs
fewer FP32 updates there. Retain OPT-0032 K4 as the general integration target.

Receipt: [`../results/OPT-0038/result.json`](../results/OPT-0038/result.json),
SHA-256 `793a509a3e99404725b60c8a1ee81b5605e73218fbf2f3e7a7d05b757c555084`.
External thermal receipt:
[`../results/OPT-0038/thermal.json`](../results/OPT-0038/thermal.json), SHA-256
`0c4ebd5ddc299c4e879a4fed43dcf0d155d3dc0fe469cde024be5fc72587be09`.
