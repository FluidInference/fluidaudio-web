# OPT-0089 — DiT int8 weight fake-quant quality gate (iPhone memory feasibility)

## Status

- Evidence: `pending`
- Disposition: `benchmark-only` (declared up front: this experiment ships no
  kernel and integrates nothing; it produces a quality verdict and a memory
  projection that gate any future int8-weight DiT kernel work)
- Date: 2026-08-24
- Author/agent: Claude agent for Alex-Wengg
- Risk class: `approximate` (weight values change; activations, kernels,
  graph, sampler, and scheduling are untouched)

## Motivation (measured breadcrumb)

A real iPhone 17 Safari tab was OOM-killed during DiT dense-weight upload at
`1,789,925,376 / 3,020,808,192` bytes (layer 14/24). The fp16 rev7 dense DiT
package (`d3fc0020…`, 48 shards, `ACE_OPT_0009_DIT_MIXED_LAYER_BYTES =
3,020,808,192`) cannot fit an iPhone Safari tab alongside the rest of the
phase residency. An int8-weight path would roughly halve DiT resident weight
bytes, but before any int8 kernel is designed the pure quantization damage
must be measured with zero kernel changes.

## Hypothesis

Weight-only symmetric int8 quantization — one fp16 scale per 32-element block
along the input (K) dimension, round-to-nearest, clamp ±127 — applied to all
264 rank-two DiT GEMM/projection tensors in the rev7 package (216 fp16
`dit-gemm-n256-k32-tile-major-v1` + 48 packed-bf16
`dit-gemm-n128-k32-tile-major-v1` cross K/V projections; norms, biases,
scale-shift tables, and constants untouched) preserves end-to-end 30-second
generation quality within a small numerical envelope when the quantized
values are dequantized back to the original storage dtype and run through the
completely unchanged production graph (fake-quant). If true, an int8-resident
DiT (~1.51 GB weights + ~94 MB fp16 scales) is a credible iPhone path and
kernel work is justified; if quality collapses, int8-weight work stops here.

## Relationship to OPT-0058 (int8 DP4a, abandoned)

OPT-0058 quantized activations dynamically AND weights inside a DP4a compute
kernel and was abandoned when adversarial finite-to-zero collapse exceeded
its primitive gate before any timing. This experiment is a different
mechanism, not a revisit of that abandoned kernel: no activation
quantization, no DP4a, no kernel or arithmetic-order change of any kind —
only weight values move, and the damage is measured end-to-end on the real
product graph rather than on primitive adversarial fixtures. Any future int8
kernel remains subject to OPT-0058's recorded lessons.

## Method

1. `scripts/requantize-dit-int8.py` (python3 + numpy, streams shard-by-shard)
   reads the verified local mirror of hosted package
   `v1/dit-revision7/d3fc0020…`, fake-quantizes exactly the 264 GEMM tensors
   in their native tile layout (a 32-K block is physically contiguous as
   `[n_tile, k_block, :, n_in_tile]`), and writes a complete
   content-addressed package tree. The manifest is rewritten textually so
   only the 48 shard SHA-256 values and its own SHA-256 change; the manifest
   byte length (254,357) and every tensor record stay identical, so the
   unchanged runtime loads the package once its two pinned identity constants
   (`DIT_MANIFEST_SHA256`, `ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256`) are
   pointed at the new manifest SHA-256 for the run (temporary local patch,
   never committed — the experiment is reproducible from the script plus the
   documented invocation).
2. Per-tensor quantization error (RMSE, NRMSE, max abs, max relative vs
   tensor amax, SNR, zero-scale blocks) is emitted to a JSON report and
   persisted under `optimization/results/OPT-0089/`.
3. End-to-end: Chrome (puppeteer) against a dev server whose
   `VITE_ACE_MODEL_ORIGIN` serves the local packages; fresh browser profile
   per run; deterministic seeds.

## Gates

1. Environment determinism: a 30 s seed-12345 lo-fi fixture regenerated from
   the UNMODIFIED local packages must reproduce the pinned known-good
   baseline WAV SHA-256
   `095267d7be0317ed9af10c64b8495d573b43207cd86615b6bbe66f27dc17895d`
   before any fake-quant run is interpreted.
2. Quality comparison, fake-quant vs fp16 baseline, on two content-distinct
   prompt/seed pairs (lo-fi hip hop seed 12345; the default Latin-percussion
   product prompt seed 424242), 30 s each: waveform NRMSE, max |diff|,
   per-1-second-segment NRMSE profile, STFT log-magnitude spectral distance,
   and peak/RMS deltas.
3. Verdict recorded in the ledger: numerical envelope for both pairs,
   per-tensor error outliers (with a mixed-precision fp16-retention map if
   any tensor family is disproportionately damaged), and the projected
   iPhone phase-peak residency versus the observed ~1.79 GB kill point.

This experiment makes no listening, timing, or product claim. The standard
listening-gate discipline applies before any int8 path could ever become a
product profile; numerical closeness here authorizes only kernel-design
work under a new ID.

## Results

Pending; recorded here when the requantization and both end-to-end
comparisons complete.
