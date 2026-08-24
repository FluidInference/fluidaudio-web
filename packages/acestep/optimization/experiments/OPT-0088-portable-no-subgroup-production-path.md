# OPT-0088 — Portable no-subgroup production path (Safari/Firefox/iOS)

## Status

- Evidence: `pending`
- Disposition: `pending-integration`
- Date: 2026-08-23
- Author/agent: Claude agent for Alex-Wengg
- Risk class: `exact` for dense/K1/K7/ConvTranspose owners,
  `reordered-rounding` for attention (portable oracle reduction tree
  replaces the subgroup-native QK reduction)

## Hypothesis

WebGPU implementations without the `subgroups` feature (Safari, Firefox,
iOS Safari — field report: `FEATURE_UNAVAILABLE: The selected ACE graph
requires subgroups`) can run the unchanged production graph on the
unchanged hosted packages if every subgroup-dependent production owner
gains a workgroup-memory counterpart selected through the existing
execution-profile machinery. This is a compatibility experiment, not a
speed candidate: the expectation is a bounded slowdown versus the
fixed-32 subgroup path on the same machine, reported under the standard
protocol, with fail-closed behavior retained for genuinely missing
requirements (`shader-f16` stays required).

Prior entries do not answer this: OPT-0003/0028 landed portable
counterparts for the reference GEMM and the revision-6 exact VAE owners,
but the revision-7/rev8 production owners (OPT-0032/0037 dense K4,
OPT-0051 K7 row-reuse, OPT-0048 ConvTranspose K4, OPT-0025 K1 subgroup
GEMM host, OPT-0061/0062/0070 attention) were subgroup-only.

## Identity

- Baseline commit: 8df8fdc (main at branch point)
- Candidate commits: c272b2d (dense), c48c050 (K7 row-reuse),
  373e90a (ConvTranspose K4), plus selection/pipeline wiring commits on
  `feat/portable-kernels`
- Model manifests: unchanged hosted production tuple (reference
  `18f36c64…`, DiT rev8 `d3fc0020…`, VAE revision-7 `36a54d79…`);
  consuming hosted layouts unchanged is a hard constraint of this
  experiment — no re-hosting
- Execution profile: portable production (no `subgroups`), `shader-f16`
  required
- Machine: Apple Silicon dev machine, Chrome with subgroups masked via
  the test-only `aceTestDisableSubgroups=1` adapter facade

## Declared numerical relationships

- DiT dense K4 portable (`opt-0088-dense-k4-fp16-portable[-production]-v1`):
  bit-identical to OPT-0032/0037 — `subgroupBroadcast` staging replaced
  by workgroup-memory staging; every arithmetic token identical
  (test-enforced byte equality of the arithmetic WGSL lines).
- VAE K7 row-reuse portable (`opt-0088-vae-conv1d-k4-row-reuse-portable-v1`):
  bit-identical to OPT-0051; identical revision-7 packed index math by
  re-export.
- VAE ConvTranspose K4 portable
  (`opt-0088-vae-conv-transpose1d-{r4c8,r8c4}-k4-portable-v1`):
  bit-identical to OPT-0048; nine regex-extracted arithmetic sections
  byte-equal between generated WGSL sources.
- Attention: the portable path routes to the existing portable oracle
  kernel (`aceCorrectnessAttentionWgsl`), which is the repository's
  declared numerical reference; output is not bit-identical to the
  subgroup query8/quad-query kernels because the QK reduction tree
  differs. End-to-end waveform equivalence versus the subgroup baseline
  is therefore measured (NRMSE + spectral sanity), and the standard
  listening-gate discipline applies before any claim beyond numerical
  equivalence.

## Gates

1. Node suite green including the three new kernel contract tests.
2. End-to-end: Chrome with subgroups masked, 30 s seed-12345 fixture,
   warm cache; waveform NRMSE and spectral comparison versus the
   known-good subgroup-path baseline WAV
   (`095267d7be0317ed9af10c64b8495d573b43207cd86615b6bbe66f27dc17895d`).
3. Timed masked versus unmasked run on the same machine, disclosed
   thermal state, reported as an observed gap (not a tuned claim).

## Results

Pending; recorded here when the end-to-end and timing gates complete.
