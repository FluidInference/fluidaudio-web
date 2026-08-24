# OPT-0089 — DiT int8 weight fake-quant quality gate (iPhone memory feasibility)

## Status

- Evidence: `positive` (numerical feasibility only; explicitly no listening
  claim)
- Disposition: `benchmark-only` (this experiment ships no kernel and
  integrates nothing; it produces a quality verdict and a memory projection
  that gate any future int8-weight DiT kernel work)
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

## Identity

- Source package: local verified mirror of hosted `v1/dit-revision7/`
  manifest `d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f`
  (source shard SHA-256s verified against the manifest before quantization)
- Fake-quant package: manifest
  `ef8355b9cffff466b018b51275923982b071234933fe8a32897915eeeb01fa36`,
  manifest byte length preserved at `254,357`; 264 GEMM tensors quantized
  (`3,019,898,880` bytes), 193 passthrough tensors byte-identical
- Reference/VAE packages: unchanged hosted identities (`18f36c64…`,
  `36a54d79…`) served from the same local origin
- Runtime: completely unchanged production graph and profiles
  (`opt-0009-fp16-fp32-dense-v1` dense DiT,
  `opt-0070-fixed32-quad-query32-full-self-production-v1` attention,
  production VAE tuple); temporary local patch of the three pinned
  `d3fc0020…` identity constants only, reverted after the runs
- Machine: Apple Silicon dev machine, stock Chrome via puppeteer, fresh
  browser profile per run, dev server with
  `VITE_ACE_MODEL_ORIGIN=/models/…` local origins
- Output WAV SHA-256s: fp16 lo-fi `095267d7…` (pinned baseline reproduced),
  fp16 latin `bad98ef7ffb0512b4138241c596669ae6f05ee0e7dc6790a9998a5aaf23ed043`,
  int8 lo-fi `d46597d5641fd8cbe0b36ff0dc5ccc65bf5af5d7717a87b3f5cf3757ca2dcc25`,
  int8 latin `221dce8174f58251b4a38bab770d5be39194ac3cda46fcf34be2bbe6650fb88f`

## Results

Authoritative artifacts:
[quant-error.json](../results/OPT-0089/quant-error.json),
[waveform-metrics.json](../results/OPT-0089/waveform-metrics.json).

### Per-tensor quantization damage (weight level)

Uniform and small across all 264 tensors: NRMSE `0.00515–0.00634`
(median `0.00559`), minimum SNR `43.96 dB`, maximum per-element error
`0.394%` of tensor amax, zero zero-scale blocks. Family medians are tightly
clustered (o_proj `0.00535`, k/v/q_proj `0.00550–0.00558`, up/gate
`0.00566–0.00571`, down_proj `0.00574`; worst single tensor
`layers.23.self_attn.v_proj` at `0.00634`). No tensor or family is a
disproportionate outlier, so no mixed-precision fp16-retention map is
required at the weight level.

### Gate 1 — environment determinism

The fp16 run from the unmodified local packages reproduced the pinned
baseline WAV byte-exactly (`095267d7…`). PASS.

### Gate 2 — fake-quant vs fp16, two prompt/seed pairs (30 s each)

- Lo-fi seed 12345: waveform NRMSE `0.0669` (SNR `23.50 dB`), Pearson
  `0.99777`, max |diff| `0.3867`, per-second NRMSE median `0.0484` /
  max `0.3264`, mean log-spectral distance `0.1772` log10 units
  (≈ `3.5 dB`), p95 `0.3640`, peak Δ `+0.0000` (shared −1 dBFS peak
  normalization), RMS Δ `−0.050 dB`.
- Latin-percussion default prompt seed 424242: waveform NRMSE `0.2268`
  (SNR `12.89 dB`), Pearson `0.97460`, max |diff| `0.9170`, per-second
  NRMSE median `0.1173`, mean log-spectral distance `0.2343`
  (≈ `4.7 dB`), p95 `0.4139`, peak Δ `+0.0000`, RMS Δ `+0.089 dB`. The
  headline per-second maximum `5.19` is a small-denominator artifact of the
  ending: at second 27 the baseline is essentially silent (segment RMS
  `0.00061` vs `0.00322`) because the fake-quant realization sustains the
  final decay slightly longer (second 26 RMS `0.0864` vs `0.0382`);
  active-music seconds run `0.08–0.36`.

Interpretation: the damage manifests as trajectory divergence of the
8-evaluation sampler, not noise-like corruption — zero non-finite samples,
exact peak parity, RMS within `0.09 dB`, high global correlation, flat
per-second profiles except the latin ending decay. Waveform NRMSE therefore
overstates perceptual change, but the outputs are not near-bit-identical
realizations; the standard listening gate is mandatory before any product
use of an int8 path.

### Memory projection (iPhone)

Quantized GEMM storage: `1,509,949,440` B int8 + `94,371,840` B fp16
scales. DiT-phase resident weights become `1,735,340,288` B (including
`909,312` B fp16 layer norms/tables and the `130,109,696` B fp16
reference-shared tensors) versus `3,150,917,888` B today. Adding the
measured desktop 30 s DiT-phase non-weight overhead
(`3,277,864,192 − 3,150,917,888 = 126,946,304` B tracked arena/control)
projects an int8 phase peak of ≈ `1.862 GB` tracked GPU bytes. The observed
iPhone 17 kill point was ≈ `1.920 GB` tracked GPU (`1,789,925,376` dense
bytes uploaded + `130,109,696` reference-shared already resident), and the
pipeline destroys the conditioning phase before DiT upload begins, so that
kill already reflects minimal-residency ordering. The projected int8 peak
therefore sits ≈ `58 MB` under the observed kill boundary: plausibly fits,
but marginal, and untracked tab overhead (JS staging, page, wasm) consumes
an unknown share of the real Safari budget. Identified fallback levers:
quantize the `130 MB` reference-shared DiT tensors, and/or int4 for the MLP
family (`1.81 GB` of the fp16 bytes) which this gate's uniform error
profile makes the natural next candidate.

### Verdict

Weight-only int8 per-32-K-block quantization does not collapse ACE-Step
generation: the numerical feasibility gate passes and the projected DiT
phase residency crosses under the observed iPhone OOM boundary. int8-weight
kernel work (int8-resident storage with in-kernel dequant, new experiment
ID) is justified. Per-denoise-step latent comparison and the standard
listening gate are the mandatory next quality checkpoints; this record
makes no listening, timing, or product claim.
