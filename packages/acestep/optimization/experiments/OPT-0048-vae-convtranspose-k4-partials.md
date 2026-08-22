# OPT-0048 — VAE ConvTranspose K4 partials

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Risk: bounded FP16 K4 partial reduction with FP32 running state

## Hypothesis

The authoritative C4500 ConvTranspose family contains about `1.734 T` valid
MACs and originally consumed `42.401 s`. OPT-0026 and OPT-0036 remove large
amounts of control, barrier, and reuse overhead, but the best exact
per-operation selector still projects only roughly `0.4` logical TFLOP/s. The
inner operation remains a scalar FP32 Cin reduction even though each output
row has only a small number of congruent taps and is otherwise a dense vector
contraction.

For each valid tap, reduce four adjacent input channels with one native FP16
dot, widen the bounded K4 partial once, and retain the tap-then-Cin4 running
state in FP32. Store weights so every output channel owns a contiguous Cin4
vector and the output-channel vectors requested by a subgroup are contiguous
at fixed phase/tap/Cin4. This is the transpose analogue of the measured
OPT-0024/OPT-0032 mechanism, not another exact reuse-axis variant.

## Frozen mechanism

- Use OPT-0040's static per-shape ownership: channel reuse for blocks 0–2 and
  row reuse for blocks 3–4. Change only the inner Cin reduction.
- Keep the revision-6 polyphase logical mapping, output coverage, congruent-tap
  order, bias seed, FP32 running accumulator, and explicit FP16 output store.
- Benchmark-pack the same FP16 words into
  `[phase, tap, cin4, coutTile, lane, outputWithinLane, cinElement4]` so a lane
  loads one `vec4<f16>` per owned output and the subgroup covers a contiguous
  output-channel band at each Cin4.
- For each increasing valid tap and increasing Cin4, load the corresponding
  input `vec4<f16>`, issue one FP16 dot4 per output, widen each partial once,
  and add once to FP32. Do not accumulate across two taps or across multiple
  Cin4 groups in FP16.
- Do not combine K8/K16 partials, K7 changes, window changes, or a production
  package under this ID. Packing is outside the timed primitive score.

## Gate

Prove exhaustive logical/physical pack and inverse identity, complete output
coverage, deterministic reruns, canaries, tails, finite/class behavior, and
the frozen numerical envelope on all five production shapes. Compare against
the exact OPT-0040 selector in a six-permutation balanced screen after one
level-0 `notifyutil` observation made after at least 30 idle seconds. Require
every shape to be non-slower and the summed production score to improve by at
least `1.50x` before any subsystem escalation.

A passing primitive authorizes a new-ID converter/package and authenticated
C512 decoder quality gate. It does not authorize production routing, a long
projection, or listening approval.

## Result

The candidate passed all correctness and lifecycle gates. The harness proved
the complete `49,610,752`-word pack/inverse bijection and compared all
`141,312,000` production-shape output words twice (`282,624,000` candidate
comparisons). Every candidate word was raw-U16 identical to the exact
OPT-0040 owner, reruns were deterministic, all outputs were finite and
complete, canaries/tails were intact, and cleanup destroyed all 125 buffers
with zero live bytes.

One balanced six-permutation screen ran after a `39.191 s` idle interval and
exactly one thermal-level-0 observation. Packing and correctness remained
outside timing:

| operation | exact median | K4 median | speedup | non-slower |
| --- | ---: | ---: | ---: | --- |
| block 0 | 41.80 ms | 44.85 ms | 0.9320x | no |
| block 1 | 61.10 ms | 48.00 ms | 1.2729x | yes |
| block 2 | 50.90 ms | 28.85 ms | 1.7643x | yes |
| block 3 | 81.85 ms | 33.90 ms | 2.4145x | yes |
| block 4 | 80.55 ms | 36.95 ms | 2.1800x | yes |

The five-shape sum improved `316.20 -> 192.55 ms` (`1.642170863x`) and
cleared the aggregate `1.50x` requirement, but block 0 regressed. Therefore
the universal K4 owner fails the frozen every-shape gate and stops before
subsystem or production escalation.

The result establishes a distinct exact follow-up rather than invalidating the
mechanism: retain OPT-0040 for block 0 and select K4 for blocks 1–4. Directly
substituting those measured medians gives `189.50 ms` (`1.668601582x`) versus
the exact selector. That arithmetic is planning evidence only and must be
registered and gated under a new ID; it does not change this negative result.
One earlier page launch was rejected before timed dispatch after delayed form
entry; it is not a performance sample and its resources were fully cleaned.

Compact receipt: [`../results/OPT-0048/result.json`](../results/OPT-0048/result.json),
SHA-256 `135f3939eb027849f41fa0915e38ee317a940281303196ba00ae58dd739140b2`.
