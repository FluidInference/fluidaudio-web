# OPT-0051 — VAE K7 K4 row-reuse 16×64 tile

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Risk: approximate K4 FP16 partial reduction with FP32 running state

## First-principles basis

OPT-0024's barrier-free K4 owner is fast enough to prove the arithmetic
mechanism, but each subgroup loads 128 output-channel weight vectors for only
eight rows. Ignoring cache, that is about eight FLOPs per requested weight byte
and predicts the measured sub-TFLOP/s ceiling. Its 32 FP32 accumulators per
lane are already the occupancy budget; merely adding rows would double
register state.

Rebalance the same 32 accumulators to 16 rows × two outputs per lane. A
subgroup then owns 16×64 outputs and reuses every K4 weight vector across twice
as many rows without increasing per-lane FP32 state. Four subgroups cover
16×256 when Cout is at least 256. For Cout 128, map two 64-channel bands across
two 16-row bands so the workgroup remains fully occupied and covers 32×128.
The tile therefore keeps 4,096 outputs/workgroup while roughly halving requested
weight traffic; activation rereads rise, but they are much smaller than K7
weights.

OPT-0014 tested this ownership with exact scalar FP32 arithmetic and gained
only `1.1102x`; OPT-0016 closed nearby exact-order microtiles. Their explicit
revisit condition allows a materially different rounding mechanism. This
experiment combines the same row reuse with OPT-0024's proven native FP16 K4
dot partials and FP32 running state, which neither prior experiment tested.

## Frozen mechanism and gate

Use a benchmark-only bijective layout whose fixed K/tap/Cin4 records make each
lane's two K4 output vectors contiguous. Keep native FP16 inputs/weights,
increasing tap then Cin4 order, one FP16 dot per four Cin values, FP32 bias and
running accumulators, explicit FP16 store, fixed32 WG128, and no workgroup
memory or barriers. Compare against OPT-0024 over all four production channel
tiers and dilation/boundary/adversarial cases. Require exhaustive pack/inverse,
complete deterministic finite outputs, canaries/tails, and the OPT-0024
numerical envelope.

After the ordinary single nominal thermal check, require every tier non-slower
and at least `1.25x` production-weighted speedup before C512 escalation. A pass
still requires the authenticated OPT-0044 trajectory, waveform, and listening
gates before production. A failure closes this K4 16×64 geometry unchanged.

## Result

The sole timed run followed the frozen one-button protocol. Preparation had
already completed before READY; the run then waited `46,002 ms`, observed
exactly one external `notifyutil` thermal level `0`, and launched after
`15,305 ms`, within the harness's frozen `30,000 ms` maximum. No timed retry
was performed.

Correctness passed across all 16 cases: 12 four-tier/dilation production cases
and four signed-zero, cancellation, finite-range, and tail-Cin adversarial
cases. The harness proved `29,707,776` packed U16 words with zero inverse
mismatches and compared `11,870,208` output U16 words with zero mismatches.
Both arms were raw-bit deterministic, finite, complete, and canary-safe. The
candidate core identity was
`59e144c1316d642d362d206222888177cd4e792743b3e23631ca415e923d770a` and
the generated-shader aggregate was
`4418f590a9407f1f2385d4435ee425d78db29442e928897b84dd082b0f92ff0f`.

| Tier | Weight | OPT-0024 median (ms) | 16×64 median (ms) | Speedup | Non-slower |
| --- | ---: | ---: | ---: | ---: | --- |
| C1024 | 282 | 11.4875000045 | 4.924999997 | 2.332487312× | yes |
| C512 | 423 | 2.262500003 | 1.375000007 | 1.645454539× | yes |
| C256 | 423 | 0.4375 | 0.5 | 0.875× | **no** |
| C128 | 1269 | 0.293749996 | 0.174999997 | 1.678571432× | yes |

The production-weighted score improved
`4754.34374685 -> 2404.04999853 ms` (`1.977639296x`), but C256 regressed.
That violates the mandatory every-tier non-slower condition, so the universal
16×64 owner is negative and abandoned unchanged. There was no package,
profile, decoder, OPT-0044, listening, or production escalation. Cleanup
destroyed all `176/176` tracked buffers, left zero live resources, and
destroyed the device.

The heterogeneous evidence is retained rather than averaged away. OPT-0057 is
the separately registered follow-up: use row reuse for C1024/C512/C128 and the
unchanged OPT-0024 owner for C256. Its exactly substituted measured score is
`2377.61249853 ms` versus `4754.34374685 ms` (`1.9996293549892823x`), and its
planning-only long-K7 projection is
`34.966 -> 17.486240593916172 s`. Those values authorize only OPT-0057's new
authenticated layout/profile and declared correctness, numerical, decoder,
waveform, and listening gates.
