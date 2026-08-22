# OPT-0050 — DiT dense bounded K4 output-vector FMA

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Risk: approximate K4 FP16 partial reduction with FP32 running state

## First-principles basis

OPT-0032 proves that bounded K4 FP16 work with FP32 running state is safe
enough to escalate and reaches `1.8925` valid GPU TFLOP/s, but its inner
primitive is a horizontal `dot(vec4<f16>, vec4<f16>)` for each independent
output. The pinned Parakeet fast path instead uses native
`fma(vec4<f16>)` across four independent output channels, the vector direction
Apple's compiler demonstrably executes well. Its rejected error source is the
unbounded K2048/K6144 FP16 accumulator, not output-vector FMA itself.

Test the missing combination: for each consecutive K4 block, accumulate four
scalar-A × `vec4<f16>`-B operations into a local FP16 output vector using
`fma`, widen that bounded K4 result once, and add it to the existing FP32
running vector. Reorder only the benchmark weight layout to
`[N/128,K/4,K4,lane32,output4]` so each lane's four output weights are one
contiguous vector at fixed K. Keep WG128, M32×N128 ownership, FP32 activation
storage, FP32 running/output storage, increasing K4 order, and no workgroup
memory or barriers.

This is not OPT-0020: reduction remains native FP16 and only four terms long;
the vector lanes are independent outputs, so there is no FP32 horizontal dot.
It is also not Parakeet's full-FP16 accumulator.

## Frozen gate

Compare OPT-0032 dot-K4 against output-vector-FMA K4 on all four exact M2250
production shapes and the same signed-zero, cancellation, finite-range, and
long-K adversarial fixtures. Require exhaustive pack/inverse identity,
complete deterministic finite outputs, guards/tails, and the complete
OPT-0032 numerical envelope. Use a balanced one-submit/one-drain timing screen
after one nominal thermal check. Require every production shape non-slower and
at least `1.15x` weighted speedup over OPT-0032 before converter/package or
trajectory work. Otherwise abandon this layout unchanged.

No production routing, package replacement, trajectory, waveform, listening,
or product-speed claim is authorized here.

## Result

The candidate was raw-U32 identical to OPT-0032 across all `25,344,000`
production outputs and `17,408` adversarial outputs, including signed-zero,
cancellation, finite-range, and K6144 screens. Pack/inverse identity,
deterministic reruns, complete writes, guards, tails, finite classes, cleanup,
and the single level-0 thermal check all passed.

Performance did not. The weighted `4/2/2/1` score was
`168.50 -> 170.55 ms` (`0.987980x`). Only K2048/N6144 was faster
(`1.07760x`); K2048/N2048, K2048/N1024, and K6144/N2048 were
`0.99582x`, `0.72327x`, and `0.92866x`. The candidate therefore failed both
the all-shape rule and the `1.15x` aggregate gate.

This closes the hypothesis that Parakeet's independent-output FP16 vector-FMA
direction is an unclaimed speed mechanism under ACE's bounded K4/FP32-running
contract. Chrome produced identical arithmetic, but no general throughput
gain. Retain OPT-0032 dot-K4 wherever a later trajectory-safe selector permits
K4; do not add this physical layout to the converter.

One earlier prepared page was rejected by the thermal form before any timed
dispatch after tool latency exceeded its check-to-launch bound; all 48 buffers
and the device were destroyed. The persisted result is the subsequent single
actual balanced timing screen, not a performance retry.

Receipt: [result.json](../results/OPT-0050/result.json), SHA-256
`51a61af657046f77300290842b2f81da1eb476f2063c7e5803942ffa4816cd01`.
