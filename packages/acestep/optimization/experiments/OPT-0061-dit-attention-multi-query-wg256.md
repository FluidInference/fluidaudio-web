# OPT-0061 — DiT fixed-WG256 multi-query full attention

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Risk: exact FP32 ascending-key online-softmax ownership change

## First-principles basis

OPT-0030 showed that sharing K/V across more queries by widening the workgroup
to WG512/WG1024 loses badly on this M3. OPT-0039 then isolated the actual
mechanism: keep WG256 and let each fixed-32 subgroup carry two independent
query streams. That retained occupancy, halved workgroups and K/V/barrier
events, preserved every output bit, and improved `132.2 -> 97.4 ms`
(`1.3572895x`).

Two streams are not an architectural limit. One subgroup can carry three or
four query streams while the same WG256 still stages one 128-element K/V row.
At the exact B1/Hq16/Hkv8/M2250/D128 shape, query8, dual-query, triple-query,
and quad-query require respectively `4,504`, `2,256`, `1,504`, and `1,128`
workgroups. Triple and quad reduce K/V loads and barrier events by `1.5x` and
`2x` relative to dual-query without adding workgroup storage or changing the
number/order of query-key products. The risk is private-register pressure:
four streams retain 32 query/weighted FP32 values per lane before scalar
online-softmax state, so measurement rather than load-count arithmetic decides
whether occupancy survives.

## Frozen primitive direction

- Keep fixed32 subgroups, WG256, eight subgroups, the existing 1,024-byte K/V
  row staging, two barriers per ascending key, FP32 query/key/value/output, and
  independent online max/denominator/weighted-value state for every query.
- Add isolated three-stream and four-stream candidates only. A subgroup owns
  one query head and query tokens separated by four within a 12- or 16-token
  tile. Tail streams remain guarded without placing barriers in divergent
  control flow.
- Compare current query8, OPT-0039 dual-query, triple-query, and quad-query on
  the exact authenticated production shape. Do not change masks, sliding or
  cross attention, graph routing, package identity, command scheduling, or the
  production default.

## Gates

1. Prove the workgroup/query bijection, exact workgroup/load/barrier counts,
   complete tail coverage, fixed32 capability checks, and disjoint output
   ownership for all four arms.
2. Run every arm twice before timing with qNaN-prefilled guarded output.
   Compare all `4,608,000` raw U32 output words to query8, require deterministic
   repeats, zero mismatches/non-finite values/unwritten words, intact canaries,
   and complete cleanup.
3. After one nominal thermal check, time eight balanced forward/reverse Latin
   rounds with one submit and matching drain per sample. Report every sample,
   medians, workgroups, K/V loads, barrier events, and source-level private
   state. The winning new arm must be at least `1.12x` faster than OPT-0039
   dual-query and `1.50x` faster than query8. A query8 comparison alone cannot
   qualify a candidate that regresses the already-positive dual-query owner.

A primitive pass authorizes only a new-ID production integration replacing
OPT-0045's pending dual-query direction. A miss abandons wider per-subgroup
stream counts on this browser/GPU. Because the arithmetic must be raw-bit
exact, no listening gate is needed if a later layer, every-step, and final
latent identity gate passes. This experiment makes no graph, product, or
under-one-minute claim.

## Result

The authenticated stock-Chrome/M3 primitive gate passed. The authoritative
receipt is
[`optimization/results/OPT-0061/result.json`](../results/OPT-0061/result.json),
SHA-256
`aa94b429d026d8e2093589b8664be24dbd64ffc14f51160ced4682521a3b95e6`.
It retains all 32 timing samples from eight balanced forward/reverse Latin
rounds:

| Arm | Raw submit-through-drain samples (ms) | Median (ms) |
| --- | --- | ---: |
| query8 | `110.89999997615814, 116.10000002384186, 153, 106.29999995231628, 138.19999992847443, 137.20000004768372, 135.70000004768372, 132.19999992847443` | `133.94999998807907` |
| dual-query16 | `114.20000004768372, 102.5, 117.20000004768372, 119.10000002384186, 109.40000009536743, 102.79999995231628, 118.5, 109.60000002384186` | `111.90000003576279` |
| triple-query24 | `77.69999992847443, 103.10000002384186, 99.60000002384186, 115, 92.60000002384186, 71.5, 89.39999997615814, 90.89999997615814` | `91.75` |
| quad-query32 | `70.10000002384186, 99.29999995231628, 81.60000002384186, 69.39999997615814, 77.5, 77, 84.30000007152557, 75.70000004768372` | `77.25` |

Triple-query improved `1.2196185290001393x` over dual-query but only
`1.459945503957265x` over query8, so it failed the declared `1.50x` query8
gate. Quad-query improved `1.448543689783337x` over dual-query and
`1.7339805823699557x` over query8, clearing both frozen gates and becoming the
sole primitive qualifier.

Correctness covered `4,608,000` output elements per execution and
`32,256,000` total raw-U32 comparisons across the eight complete correctness
runs. Every query8, dual, triple, and quad first/repeat comparison had zero
mismatches; all outputs were finite, complete, deterministic, and bit-identical
with SHA-256
`2882495eccd1f1971e998b957dfdd12ee517a2019a6ddccd9d76e38bb81c9892`.
All qNaN-prefill and canary checks passed. Cleanup destroyed all eight created
buffers (`2` before timing and `6` at final teardown) and destroyed the device.

The valid action followed a `31,059 ms` wait and one thermal-level-0
`notifyutil` check, with an `8,554 ms` check-to-launch delay. Compilation and
full correctness/warmup remained outside timing; each timing sample used one
command buffer and one matching queue drain, with output readback outside the
interval. Three earlier attempts are setup evidence only and supplied no timing
sample: one launch-delay rejection stopped before timed work, one level-1
preflight deliberately received no click, and one preflight whose shell capture
was missed likewise received no click.

An earlier structurally valid receipt omitted the required raw timing arrays.
It is retained as
[`provisional-missing-raw-samples.json`](../results/OPT-0061/provisional-missing-raw-samples.json),
SHA-256
`6837ce89aa1b2f9f22c02b4c0b048ea62bccae112503a516ec9bf660d82a5f8e`,
and is explicitly non-authoritative for every performance comparison.

At the frozen eight-evaluation production topology, the same-page median delta
projects only:

`(133.94999998807907 - 77.25) ms * 96 = 5,443.199998855591 ms`.

That is planning arithmetic, not a graph measurement. If the separate
quality-sensitive six-evaluation schedule is later approved, its distinct
72-call arithmetic is:

`(133.94999998807907 - 77.25) ms * 72 = 4,082.399999141693 ms`.

Do not combine or relabel those projections. This result authorizes only a new
production-integration experiment; it does not change the attention profile,
graph, package, sampler, product default, or under-one-minute status.
