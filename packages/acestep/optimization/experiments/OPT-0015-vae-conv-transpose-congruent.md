# OPT-0015 — VAE FP16 congruent ConvTranspose1D

## Status

- Evidence: `positive`
- Disposition: `integrated`
- Date: 2026-08-14
- Author/agent: Codex
- Risk class: `exact`; every candidate output must retain the current bias,
  increasing valid-tap, then increasing-input-channel FP32 arithmetic order and
  the same FP16 boundary store
- Frozen production baseline:
  `7d4916da0cd480fe03cd5712048cb3f3f4c06310`
- Frozen kernel checkpoint:
  `075ecc0b34b7541cffc0a83412c17ee31bbadab6`
- Frozen browser-gate checkpoint:
  `65603ade17b9f3b9ca92cc0c29be83fe51a6e885`
- Frozen integration checkpoint:
  `36608b857827b2b1d31ac91bf5cca9639fb0b9ed`

## Hypothesis

The current FP16 ConvTranspose1D kernel loops, stages, and barriers across all
`2 * stride` kernel taps even though the exact production geometry admits only
about two taps per output through the stride congruence. A direct-congruent
kernel, or a bit-preserving packed equivalent, should enumerate only those
valid taps in their original order and remove the dominant C300 decoder cost
without changing model math or FP16 output bits.

## Baseline evidence

The frozen 12-second direct C300 production profile has receipt SHA-256
`abd625696efbb3c12f9aa90924531ee780b322e167e69b64842f4b400deb3e76`.
It measured `14,118.300000786781 ms` of VAE decoder submit-to-drain wall time.
Homogeneous ConvTranspose1D batches contributed
`8,016.400000452995 ms` across 43 batches and 344 quanta; mixed batches remain
unattributed, so this is a measured homogeneous lower bound rather than a
complete family total.

OPT-0008 measured `1,924.1999998092651 ms` for homogeneous portable FP32
ConvTranspose1D at B256. That result is useful historical context only: its
window shape, storage precision, kernel, and surrounding profile differ, so it
is not an A/B comparison and supports no regression ratio.

## Candidate scope and gates

The candidate may derive the congruent valid taps directly or prepare an exact
bit-preserving packed view. It must skip only taps that the current bounds and
congruence checks reject, visit retained taps in increasing source-kernel
order, and visit input channels in increasing order. Package identity, graph
geometry, scheduling, normalization, and all other kernels remain unchanged.

Before any production selection:

1. Compare the candidate with the current FP16 kernel over the complete raw
   U16 output domain for the exact production operations, including edge
   geometry, complete writes, guards, and a deterministic rerun. Require zero
   raw-U16 mismatches on both executions.
2. Integrate the passing candidate behind an explicit kernel identity and run
   one short production profile that reports end-to-end and per-family timing.
3. For the performance run, wait once for 30 seconds at nominal thermal
   pressure, then launch and accept the result without a thermal retry.

Primitive correctness or isolated timing makes no product quality or speed
claim. Such claims wait for successful production integration and the short
integrated profile. No candidate code, test, result, or production selection
existed at registration time.

## Primitive gate

The explicit `ace-vae-fp16-congruent-two-tap-conv-transpose1d-v1` kernel
passed the five exact C300 production operations over 15 first/interior/tail
graph ranges. Across two executions of both arms, all `8,404,992` selected
raw-U16 comparisons matched. Complete qNaN-prefilled writes, deterministic
rerun hashes, guards, adjacent canaries, and cleanup also passed.

The bounded browser gate projected the complete 378-range portable sequence at
`13,958.000004947186 ms` and the congruent sequence at
`3,830.9499965310097 ms`, a `3.643482691652564x` speedup. Each representative
range was separately drained and weighted by its exact C300 multiplicity, so
this is primitive decision evidence rather than integrated decoder wall time.

The raw browser receipt remains ignored from Git at
`optimization/artifacts/OPT-0015/raw/conv-transpose-congruent-ab.json` (29,480
bytes), SHA-256
`7dcecd275c93d44a503924eef0ddb4b5a44d542e9ba4bb52d179ee5a6ff5cd61`.

## Production integration and short result

Commit `36608b857827b2b1d31ac91bf5cca9639fb0b9ed` integrated the candidate in
the explicit
`opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1` profile; the existing
portable and OPT-0011 hybrid profiles remain unchanged.

The single integrated 12-second direct-instrumental run waited once for
`30,125 ms` at nominal thermal pressure, spanning 31 nominal observations with
a `1,015 ms` maximum gap. It launched `263 ms` later and was accepted without
an unchanged-work retry. The run completed in `23,018.200000047684 ms`, of
which VAE decode contributed `8,054.5 ms`. Its WAV SHA-256 was
`409b7157ac428910fae17776b1abbd9b42db7509984bcc0aac41871f95152ec2`,
exactly reproducing the pre-candidate FP16 production WAV; the result-receipt
SHA-256 was
`dc653f8f96e59dca39f408ab3e8ece0d0ef23e0d084c075e05e21953120f4ae1`.

The integrated family profile attributed the `7,265.799999117851 ms` decoder
submit-to-drain interval as follows:

| Family | Submit-to-drain wall |
| --- | ---: |
| fixed32 K7 Conv1D | `3,019.800000190735 ms` |
| congruent ConvTranspose1D | `2,001.9999997615814 ms` |
| K1 Conv1D | `1,177.2999993562698 ms` |
| mixed batches | `779.8999999761581 ms` |
| Snake | `215 ms` |
| Add | `71.799999833107 ms` |

Against the frozen family profile, homogeneous ConvTranspose fell from
`8,016.400000452995 ms` to `2,001.9999997615814 ms` (`4.0041958049x`) and
decoder submit-to-drain fell from `14,118.300000786781 ms` to
`7,265.799999117851 ms` (`1.9431170694x`). The earlier comparable 12-second
product checkpoint measured `33,168.1 ms` total and `15,962.4 ms` VAE, so the
new run is directionally `1.4409510735x` faster total and `1.9817989944x`
faster in VAE. That end-to-end comparison is a one-run historical stack
checkpoint, not a contemporaneous isolated-kernel A/B.

## Evidence and disposition

- Evidence conclusion: `positive`. The primitive was raw-U16 exact and showed
  a large timing signal; production integration then retained the exact prior
  FP16 WAV and materially reduced measured transpose and decoder wall time.
- Code disposition: `integrated` behind the explicit OPT-0015 runtime profile.
- No 180-second generation, under-60-second projection, listening change, or
  independent-run variance claim is made.
- Canonical result: [result.json](../results/OPT-0015/result.json)
