# OPT-0014 — VAE K7 packed-KIO fixed32 subgroup Conv1D

## Status

- Evidence: `negative`
- Disposition: `abandoned` (benchmark evidence retained; not integrated)
- Date: 2026-08-14
- Author/agent: Codex
- Risk class: `exact`; the candidate must preserve every operand bit and each
  output's K→Cin FP32 accumulation order
- Allocation production baseline:
  `caf977cb7c173556d482b4fab08beee55fe08328`
- Evaluated production baseline:
  `36608b857827b2b1d31ac91bf5cca9639fb0b9ed`
- Candidate core:
  `12e128ab323c0024ed683313b4d06c07041213e7`
- Frozen browser gate:
  `3904d212148cf2ecf93f317f8dcce3d59ef232a8`

## Hypothesis

GPU-repacking each authenticated K7 weight tensor bit-for-bit from native
O-K-I order into K-I-O order, then using a fixed-32 subgroup tile of 16 rows by
64 output channels, can materially accelerate production VAE decoding.
Adjacent `vec2` weight loads become contiguous and the four subgroups cover
distinct output-channel bands instead of redundantly loading the same weights,
while every output retains increasing K then increasing Cin FP32 arithmetic.

## Baseline

The frozen checkpoint's actual 180-second direct production run took
`568.618 s` total: `125.948 s` DiT and `431.377 s` VAE. It began after one
30-second nominal thermal gate at level `0`; thermal pressure was level `1`
after completion, and there was no thermal retry.

- Raw WAV SHA-256:
  `2a16efb2c06a5318bcded939acb5a2c82ba55a04b36df89528e6a8bdf9112e4c`
- Production receipt SHA-256:
  `1a49db615916f4c38ceed7d0dea7cd14075a614247cb2d7249efa7b1dfe4ed70`

These stage times establish the allocation-time product baseline. The bounded
candidate gate below is a separately drained, exact-C300 weighted primitive
projection; it is not an integrated decoder or song comparison.

## Candidate result

The benchmark candidate owns an explicit packed-KIO K7 path. A GPU preparation
pass must preserve the source FP16 bits exactly; the kernel uses fixed 32-lane
subgroups, assigns two adjacent output channels and 16 row accumulators to each
lane, maps the four subgroups to distinct 64-channel bands, and retains the
source K→Cin FP32 order. No model math, graph geometry, or output normalization
changes are authorized.

The frozen target-browser gate covered all 17 C300 K7 operations and all 2,404
exact production graph ranges through 51 first/interior/tail correctness probes
and 50 weighted timing strata. Two correctness executions compared 13,854,720
raw output words: 12,599,296 U16 words plus 1,255,424 final no-bias U32 words.
There were zero mismatches, every qNaN prefill was overwritten, guards and
adjacent canaries remained intact, and rerun hashes were deterministic.

The GPU repack was also exact twice across all 17 tensors: 61,017,600 U16
comparisons over 30,508,800 unique words, with zero mismatches, complete writes,
intact redzones, and deterministic rerun hashes. The packed payload adds
61,017,600 persistent bytes while the authenticated native weights remain the
fixed32 authority. Its separately measured late repack took 4.5 ms and is not
included in convolution timing.

After one accepted nominal thermal gate of 30,118 ms (31 observations, 1,005 ms
maximum poll gap, no non-nominal observations, 76 ms launch delay, no retry),
the 50 separately drained representatives projected over all 2,404 range
weights as follows:

| Arm | Weighted C300 K7 projection |
| --- | ---: |
| Current fixed32 | 9,506.499997437 ms |
| Packed KIO | 8,562.849999070168 ms |

That is a 943.6499983668327 ms projected saving, or only
1.1102027944515322x. The candidate won 33 strata, lost 15, and tied two; the
shape behavior is heterogeneous rather than a uniformly stronger replacement.
The gate cleaned up all 178 tracked buffers, left zero live bytes, and destroyed
the device idempotently.

## Disposition

Evidence is `negative` for this duplicate packed-KIO, all-17-operation geometry.
The modest weighted primitive gain does not justify another 61,017,600 resident
bytes or production integration, especially with 15 regressing strata. Keep the
candidate source and frozen browser gate as benchmark evidence, but leave the
production selector and package layout unchanged.

This result makes no integrated decoder, 12-second generation, 180-second
generation, listening, product-speed, or under-60-second claim. Listening was
not required because the primitive result was exact and was not integrated.
The negative decision is scoped to this mechanism, not to every possible K7
optimization.

- Artifact: [`packed-kio-k7-ab.json`](../artifacts/OPT-0014/raw/packed-kio-k7-ab.json),
  110,962 bytes, SHA-256
  `2445e5e3b07a3d950db8e7badcd74bff6fef687013bbc7fd56389acaedd845c3`
- Result JSON: [`optimization/results/OPT-0014/result.json`](../results/OPT-0014/result.json)
- Revisit only for a materially different K7 geometry or a layout strategy
  that replaces rather than duplicates weights, later profiling that changes
  K7's decision value, or a target browser/compiler/GPU identity change that
  invalidates these measurements. Do not repeat this unchanged geometry.
