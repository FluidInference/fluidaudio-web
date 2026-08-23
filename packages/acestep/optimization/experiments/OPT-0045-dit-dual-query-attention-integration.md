# OPT-0045 — DiT dual-query attention integration

## Status

- Evidence: `inconclusive`
- Disposition: `superseded`
- Risk: exact FP32 ascending-key online-softmax arithmetic

## Hypothesis

OPT-0039 preserved all `4,608,000` M2250 full-attention output words bit for
bit and reduced the primitive median from `132.2 ms` to `97.4 ms`
(`1.3572895x`). The production graph executes the same full-attention shape
once in each of 24 layers across eight evaluations. Applying only the measured
primitive delta projects about `3.34 s` saved across those 96 executions,
before graph-level effects. The other 12 layers per evaluation use sliding
attention and are deliberately outside this experiment.

## Frozen integration direction

- Add a distinct fixed32 DiT profile/kernel-set that differs from the current
  optimized graph only at the exact B1/Hq16/Hkv8/M2250/D128 unmasked full
  self-attention operations screened by OPT-0039.
- Keep WG256, eight fixed32 subgroups, ascending-key FP32 online max,
  denominator and weighted-value updates, all masks/sliding/cross attention,
  dense owners, sampler/DCW, graph ordering, and package identities unchanged.
- Route by exact operation identity and shape; fail closed instead of falling
  back for a mismatched full-attention operation. Individual dispatches must
  report the actual dual-query kernel ID.
- Do not combine the revision-8 K4 package/profile selection under this ID.
  A later combined profile needs its own measured evidence.

## Gates

Prove all 96 expected production routes and zero unintended routes, then
validate actual-layer outputs, every denoise-step latent, and the final M2250
latent against the current query8 profile with raw-U32 identity. Require clean
cancellation/lifecycle and a short balanced graph- or evaluation-slice timing
that demonstrates material realized saving without the thermally confounded
multi-minute fixed-order protocol. Since arithmetic and output bits are exact,
no separate listening gate is required after final-latent identity; no product
or production-default claim follows from OPT-0039 alone.

## Authority

- Primitive record: [OPT-0039](OPT-0039-dit-attention-dual-query-wg256.md)
- Receipt: [result.json](../results/OPT-0039/result.json)

## Closeout — superseded by quad-query integration

No OPT-0045 production gate or result receipt was completed, so the literal
integration evidence is inconclusive. OPT-0061 qualified the stronger exact
quad-query owner, OPT-0062 exercised it at the graph boundary, and OPT-0070
owns its production promotion. The dual-query primitive receipt remains valid
benchmark evidence; this narrower integration direction is superseded.
