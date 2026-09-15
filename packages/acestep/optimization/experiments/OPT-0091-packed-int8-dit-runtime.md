# OPT-0091 — Packed weight-only INT8 DiT runtime

## Status

- Evidence: `pending`
- Disposition: `benchmark-only`
- Date: 2026-09-15
- Risk class: `approximate` model weights, unchanged activations and FP32
  accumulation

## Motivation

The OPT-0090 listening preview stores quantized values after dequantizing them
back into the original FP16/BF16 package. It therefore still downloads the full
5.75 GB cold model set and does not answer the reported download-size problem.
OPT-0089 established only that per-K32 weight quantization is numerically
credible; it explicitly left packed storage and kernels unimplemented.

## Hypothesis

The 216 repeated-layer DiT dense matrices can remain packed as signed INT8 with
one FP16 scale per output and K32 block, then be dequantized at load use inside
fixed32-subgroup and portable WebGPU kernels. Keeping activations at their
current FP16 load boundary and accumulation in increasing-K FP32 order should
match the OPT-0089 fake-quant arithmetic closely while reducing the 3.02 GB
mixed DiT package materially. The 48 cross-attention K/V matrices remain in
their accepted packed-BF16 representation in this first complete runtime.

This is weight-only quantization and does not revisit OPT-0058's rejected
dynamic activation quantization or DP4A contraction.

## Frozen package and arithmetic contract

- Source: authenticated production revision-7 DiT manifest
  `d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f`.
- Quantization: symmetric signed INT8, K block 32, FP16 scale
  `max(abs(weight))/127`, round-to-nearest-even, clamp to `[-127, 127]`, and
  canonical zero-scale blocks.
- Physical tensor layout: for each existing N256/K32 tile, contiguous packed
  signed bytes followed by its 256 FP16 scales. Tensor records remain
  independently aligned and bound.
- Activation values retain the current FP32-to-FP16 load rounding. Each INT8
  value is multiplied by its FP16 block scale, widened to FP32, and accumulated
  in ascending K order into the existing FP32 output.
- Production FP16 manifests, runtime, and default selection remain unchanged.

## Gates

1. Converter: deterministic bytes from the pinned source, complete source and
   tensor accounting, exact declared layout/size, bounded memory, transactional
   output, independent package verification, and rejection of malformed scale,
   offset, dtype, layout, or identity records.
2. Primitive: all four production dense shapes plus boundary rows, zero blocks,
   signed extrema, deterministic repeats, complete writes, finite outputs, and
   cleanup. Compare subgroup and portable owners against CPU fake-quant and the
   existing FP16 owner; record max error, NRMSE, SNR, and correlation.
3. Package-native graph: exact package identity, all 24 layers, every sampler
   tap, final latent, cancellation, cache recovery, resource lifetime, and
   absence of device loss or non-finite values.
4. Product: at least one 30-second default Latin instrumental using a fixed
   seed, valid WAV, waveform metrics against OPT-0089, and human listening
   before describing quality as approved.
5. Delivery: published artifact bytes and manifest re-downloaded and hashed,
   CORS for the GitHub Pages origin verified, UI reports measured download
   bytes, all repository checks pass, and the exact Pages deployment is
   inspected after merge.

## Current result

Registered before implementation. No compressed package, kernel, browser run,
quality approval, download reduction, mobile-support, or production claim is
made yet.
