# OPT-0072 — revision-7 dual-K4 VAE and C2378 production promotion

## Status

- Evidence: `positive`
- Disposition: `integrated`
- Risk: small, measured waveform drift from bounded FP16 K4 partials in the
  selected K7 and ConvTranspose owners

## First-principles basis

The revision-7 package changes physical storage only for the selected K7 and
ConvTranspose tensors. The runtime keeps FP32 running state and every FP16
storage boundary, but its bounded K4 partials do not reproduce the scalar
revision-6 reduction bit for bit. This is a quality-affecting change and is
therefore promoted under a new identity rather than relabelling OPT-0054 or
OPT-0066 after the fact.

The evidence now covers the complete three-minute VAE boundary:

- OPT-0066 authenticated all 16 selected physical tensors over `35,880,960`
  U16 words, passed the complete same-arithmetic routing oracle, and measured
  balanced C512 speedups of `1.950901x` K7, `2.000510x` ConvTranspose,
  `1.714172x` decoder, and `1.677501x` outer wall.
- The C4500 long gate compared all `17,280,000` raw FP32 stereo samples.
  Revision 7 reached NRMSE `0.0015226894328326402`, SNR
  `56.34777332344201 dB`, Pearson `0.9999988407064728`, maximum absolute
  error `0.007874935865402222`, relative RMS drift
  `0.0000015989790685799033`, and relative peak drift
  `0.00007150917818767507` versus revision 6.
- The candidate repeat was raw-U32 exact across all `17,280,000` words, with
  identical SHA-256
  `b3a1e4d585f1ed32bba27923e3ed4c09b8af52ad41f48d307c679c7b554072fe`.
  Every sample and seam neighborhood was finite, all four sequential owners
  were destroyed, all `62/62` buffers were destroyed exactly once, and peak
  live GPU bytes were `3,758,347,792`.

The owner explicitly authorized the agent on 2026-08-16 to use its best
judgment to integrate clear speed wins with minuscule audio-domain drift.
This complete-waveform result fits that narrow authorization. It does not
authorize the much larger all-K4 DiT drift, fewer denoising evaluations, or a
general relaxation of quality gates.

## Frozen integration direction

- Add a distinct production VAE identity that maps to the already measured
  OPT-0066 dual-K4 physical profile, revision-7 manifest
  `36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7`,
  and its exact precision-map/kernel-set trust roots. Do not alter the frozen
  OPT-0054/0066 diagnostic identities.
- Pair it only with the explicit OPT-0070 C2378/overlap64 production window
  identity and its `1,168,834,560`-byte device limits. Retain batch64, bounded
  OPFS streaming, normalization, WAV finalization, and one heavy GPU owner.
- Keep the pushed OPT-0070 exact DiT stack, eight-evaluation sampler, DCW,
  conditioning, graph order, cooperative submission, and public request API
  unchanged.
- Remove the revision-6 VAE package from the selected demo/product tuple only
  after protocol, diagnostics, package authentication, backend routing,
  cancellation, and lifecycle tests fail closed on every mixed identity.
- No scalar fallback is permitted after the new production identity is
  selected. Failure must be explicit.

## Gates

1. Statically prove public-production to physical-OPT-0066 mapping, exact
   revision-7 manifest/profile/precision-map identity, C2378-only window
   pairing, full `4,090` K7 plus `644` ConvTranspose quantum reconciliation,
   and unchanged non-selected owners.
2. Preserve the complete C4500 result as the waveform authority; do not rerun
   an unchanged multi-minute VAE comparison merely to obtain timing.
3. Run a short product-path smoke on the final cumulative tuple, requiring
   finite raw/WAV output, stable shape and metadata, no device loss, one heavy
   owner, cancellation at a bounded command-buffer boundary, and zero live
   resources after cleanup.
4. The final requested 180-second generation is the product integration and
   listening artifact. Record cache-ready initialization separately from
   Generate-to-WAV wall, retain all stage timings and thermal observations,
   save the WAV in `~/Downloads`, and make no under-one-minute claim unless
   the measured wall actually clears it.

## Authority

- [OPT-0054 package/layout integration](OPT-0054-vae-revision7-exact-layout-integration.md)
- [OPT-0066 joint quality/timing gate](OPT-0066-vae-revision7-dual-k4-quality-gate.md)
- [C4500 complete-waveform receipt](../results/OPT-0054/long-c2378-result.json)
- [OPT-0070 exact quad/C2378 promotion](OPT-0070-exact-quad-c2378-production-promotion.md)

No production code or default changed when this record was allocated.

## Closeout — integrated

Checkpoint `f45254e` added the distinct OPT-0072 public production identity,
mapped it fail-closed to the already measured OPT-0066 physical owner and
revision-7 package, required the OPT-0070 C2378 tuple, and selected that tuple
in the demo. Protocol, diagnostics, worker, backend, package, cancellation, and
lifecycle tests freeze the identity boundaries and reject mixed tuples; no
scalar fallback was added. Together with the complete C512 and C4500 evidence
above, the code supports positive/integrated status. The cumulative short
smoke completed in `17.0 s`, and the final 180-second generation completed in
`94,774.80000007153 ms`. Its receipt identifies public OPT-0072 mapped to
physical OPT-0066, two C2314 windows, `17,280,000/17,280,000` finite/nonzero
samples, and no device loss. See the
[cumulative result](../results/OPT-0073/result.json).
