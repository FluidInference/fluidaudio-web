# OPT-0013 — DiT full-attention query8 K/V sharing

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Date: 2026-08-14
- Author/agent: Codex
- Risk class: `reordered-rounding`; each key retains the reference FP32
  online-softmax update order, but fixed-32 subgroup dot-product reduction
  changes the within-key addition tree
- Frozen candidate checkpoint: `a88e1b41c7b127d20c3fc4dbdee63acb77612a8c`
- Frozen browser-gate checkpoint: `5b9cc1852e5178c8bbb907dbe4c90eae6004d6b8`

## Hypothesis

At the production 180-second full-attention shape, eight fixed-32 subgroups can
share one 128-dimensional K/V tile across two GQA heads and four adjacent query
tokens. This should remove nearly eightfold redundant global K/V reads and
fivefold barrier work relative to the portable one-query workgroup without
changing FP32 online-softmax state or its per-key update order.

## Frozen scope

The decision gate compares the reference portable kernel against the explicit
`fixed32-subgroup-query8` backend on deterministic F32 Q/K/V at exactly:

- batch `1`, query heads `16`, K/V heads `8`;
- query tokens `2250`, K/V tokens `2250`;
- head dimension `128`; and
- full attention, with every one of the `4,608,000` F32 outputs read back.

The candidate uses 256-lane workgroups. Eight fixed-32 subgroups process two
GQA heads by four adjacent queries while lanes 0–127 load each K and V vector
once into workgroup memory. The benchmark does not select query8 for sliding
attention, load a production package, or change a product runtime default.

## Identity

- Attention source SHA-256:
  `5f64e5148ee60f26023faeb99ac72b46354086f3db48f7778800a9061d2b9ed3`
- Portable WGSL SHA-256:
  `6052f3e86f03ad3f7f3dc80384c63fa3750c0a038b68af0689914c5087929121`
- Query8 WGSL SHA-256:
  `176cd0988c11944fb47cce813e4b6338d867507a00cf416845920c642aec71a4`
- Harness TypeScript / HTML / static-contract SHA-256:
  `791bf74274fe404eccfe4bbf3c1de1bed7ed6fec877f1e16da2e590e268064fe`,
  `61abb352278e336f3e94b7a306691c7de7f04d3e4a02325a832f778227a43489`,
  and `126242d90a15cf123950f3702809f37021016defb511e1e172ce7e7c2e0cc509`
- Deterministic Q / K / V SHA-256:
  `3a8be2a8feda56412ec2e38ea0fbf5ea9dc628523ea8436b9364f5edf9f02440`,
  `806a53b579be00ea2f49675c733f2db60e75c01159c58f20a3e4be5074403c44`,
  and `54c804aada107071e0238a8ae4c2ed3b480ad38ce5b8e8f54b581cb0a0975cba`
- Target: MacBook Air `Mac15,12`, Apple M3, 10 GPU cores, 16 GiB unified
  memory, macOS 26.5.2 build 25F84; raw reduced UA reports Chrome 151.0.0.0
- Gate capability: Apple Metal 3, `subgroups`, subgroup minimum/maximum both
  `32`, maximum 256 compute invocations/workgroup, and 16,384 bytes workgroup
  storage

The reference model and fixture-manifest hashes in the canonical result are
provenance only: this gate uses deterministic synthetic inputs and loads no
model package.

## Correctness gate

The gate requires complete writes, finite outputs, comparison of every output
F32, maximum absolute error at most `1e-4`, and NRMSE at most `1e-5`.

The candidate passed all `4,608,000` comparisons with zero non-finite values in
either arm, maximum absolute error `2.6542693376541138e-8`, mean absolute error
`1.560317666035174e-9`, and NRMSE `3.3672934508159816e-7`. Portable and query8
output hashes are respectively
`df90123ababd0182d7c7b4f6ec1604e4673ab7530cbc61eba41e14c5d0985d39`
and `b252684edb92fcc50beec55d300741e48fe103f43bbb65b03c2f6b958e03afa2`.

This closes only the primitive numerical gate. A production selection still
requires the applicable layer, denoise-trajectory, final-latent, and listening
gates because the dot-product reduction order changed.

## Benchmark protocol and result

Compilation and an exact 1-query/1-key warmup occurred outside timing. The run
then waited once for 30.010055 seconds at nominal thermal pressure, observed 31
nominal polls with a 1010.141 ms maximum gap, and launched 71.437 ms later. It
executed one conservative portable → query8 order with no thermal retry. Each
timed arm includes compute queue drain, output copy/map queue drain, and full
CPU readback.

| Arm | Compute wall | Readback wall | Total wall |
| --- | ---: | ---: | ---: |
| Portable | 1108.8 ms | 8.5 ms | 1117.3 ms |
| Query8 | 136.2 ms | 4.0 ms | 140.2 ms |

The observed speedups are `8.140969x` for compute wall and `7.969330x` for
total drain/readback wall. This is one sample per arm in a fixed conservative
order, so it establishes a large decision-useful signal rather than variance
or a release-quality performance distribution.

Query8 reduced estimated global K/V scalar loads from `20,736,000,000` to
`2,594,304,000` (`7.992895x`), corresponding to `82,944,000,000` →
`10,377,216,000` bytes. It reduced barriers per key from `10` to `2` and
workgroups from `36,000` to `4,504`, split over five ranged dispatches.

## Evidence and disposition

- Evidence conclusion: `positive`. The exact production full-attention shape
  passed its complete F32 numerical gate and showed a large timing signal that
  agrees with the expected K/V traffic and barrier reduction.
- Code disposition: `benchmark-only`. Query8 is an immediate
  production-integration candidate behind authenticated fixed-32 subgroup
  selection, but this result does not claim that production selection occurred.
- No layer, denoise trajectory, final latent, listening, 180-second song, or
  product-speed claim is made.
- Canonical result: [result.json](../results/OPT-0013/result.json)
- Raw browser receipt (ignored from Git):
  `optimization/artifacts/OPT-0013/raw/dit-full-attention-query8-ab.json`,
  SHA-256
  `28a5a368f8b7fc8bf5771e415d8bd90db2ae7adba5961283a2b6010c459fbe28`
