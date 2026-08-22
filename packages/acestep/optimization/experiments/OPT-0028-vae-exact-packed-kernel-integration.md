# OPT-0028 — Exact packed VAE kernel integration

## Status

- Evidence: `positive`
- Disposition: `integrated`
- Date: 2026-08-15
- Author/agent: Codex
- Risk class: `exact`

## Hypothesis

Replacing the production VAE K1 and ConvTranspose1D paths with the exact
OPT-0025 and OPT-0026 winners, and storing their weights directly in the
converter-native tile-major and polyphase layouts, can realize most of their
primitive gains without browser repacking or duplicate weights. Applied to the
authoritative OPT-0023 family walls, the primitive ratios project K1 from
25.7723 to 2.5190 seconds and ConvTranspose1D from 42.4010 to 13.1165 seconds,
or about 52.54 seconds saved before integration effects.

## Identity

- Allocation baseline: `72de722`
- Source/model revisions remain the repository-pinned revisions.
- Machine: MacBook Air M3, 10 GPU cores, 16 GB
- Browser/API: stock Chrome WebGPU/WASM

## Change

- Replace only K1 weight storage with OPT-0025 tile-major FP16 storage.
- Replace only ConvTranspose1D weight storage with OPT-0026 polyphase FP16
  storage; keep K7 weights native.
- Select the two exact subgroup kernels through a new, truthfully named runtime
  profile and retain packed-layout portable counterparts.
- Do not repack in JavaScript and do not retain duplicate native weights.

## Gates

- Converter output is deterministic, authenticated, transactionally produced,
  and has complete consumed/excluded tensor accounting.
- Native-to-packed-to-native tests are raw-U16 exact for every replaced tensor.
- The integrated C512 output is raw-bit identical to the accepted FP16 oracle,
  including repeats, guards, cancellation, and cleanup.
- Under the 30-second nominal thermal protocol, the combined C512 family wall
  must be materially faster. Escalate to C4500 when the measured projection is
  credible; retain as production only if the long run saves at least 40 seconds
  without a regression elsewhere.
- No listening gate is required for raw-bit-identical arithmetic.

## Results

Pending.

## Closeout — integrated

The converter-native K1 tile-major and ConvTranspose polyphase layouts, their
fixed32 subgroup owners, portable counterparts, authenticated revision-6
package/profile, and fail-closed runtime selection were integrated in
checkpoint `0b27dde`. Subsequent production and diagnostic gates retained
OPT-0028 as the exact revision-6 VAE oracle. The stale prospective “Pending”
section above is preserved as allocation history; the final evidence is
positive and the mechanism is integrated.
