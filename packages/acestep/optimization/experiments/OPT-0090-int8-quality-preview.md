# OPT-0090 — INT8 quantization listening preview

## Status

- Evidence: `positive` (artifact and integration gates; external listening pending)
- Disposition: `pending-integration`
- Date: 2026-09-15
- Risk class: `approximate`, opt-in only

## Objective

Expose the authenticated OPT-0089 fake-quant DiT as a public, opt-in listening
comparison while keeping the owner-approved production model as the default.
This preview evaluates audible quantization effects. It is not an int8-resident
runtime and does not reduce download or GPU-resident bytes.

## Trust and product boundaries

- Reproduce the package from the pinned rev7 source with
  `scripts/requantize-dit-int8.py`; require the recorded `ef8355b9…` manifest
  identity and unchanged OPT-0089 quantization metrics.
- Publish it under a distinct immutable path and authenticate every manifest and
  payload through the existing content-addressed loader.
- Retain the production package as the default after reload and in the SDK.
- Label the preview experimental and full-size. Switching after production is
  cached adds a separate 3.02 GB DiT package; shared main and VAE assets remain
  reusable.
- Do not claim packed-int8 size, memory, speed, mobile support, or approved
  listening quality from this preview.

## Gates

1. Reproduction: source and output manifest hashes match OPT-0089; generated
   quantization results match the committed report.
2. Runtime: only the exact preview manifest is accepted with the unchanged
   OPT-0009 layout/runtime profile. Any other identity fails closed.
3. UI: production is the immutable default; model switching is disabled while
   work is active and forces orderly worker/GPU disposal before reuse.
4. Cache/progress: content-addressed assets coexist without deletion or identity
   aliasing; cold-download wording describes the selected variant accurately.
5. Validation: focused protocol/config/page tests, complete ACE package tests,
   integration tests, formatting, and production build pass.
6. Listening remains external: a public user may compare outputs, but promotion
   to a default or packed-int8 runtime requires a separately recorded listening
   decision and implementation gate.

## Rollback

Remove the opt-in UI/configuration and leave the production trust root
unchanged. The unreferenced content-addressed preview package may then be
removed from hosting.

## Result

- Reproduction produced the recorded manifest
  `ef8355b9cffff466b018b51275923982b071234933fe8a32897915eeeb01fa36`;
  the generated quantization report was byte-identical to OPT-0089
  (`7bb7293a…`).
- The full-size package is hosted under the immutable
  `v1/dit-int8-fakequant/<manifest-sha>/` path in Hugging Face commit
  `ac50b5c854fb044ce058acb91d4cd9ab82d99cfa`. The live manifest and one large
  plus one small shard were downloaded again and matched their declared hashes.
- Production remains the default. The Advanced selector is disabled with the
  rest of the form during work, and switching variants disposes the previous
  worker before initializing another.
- Validation passed: 43 web integration tests, 4 unit tests, 2,029 ACE tests,
  package and page typechecks, formatting, and the production build. No fresh
  browser/GPU generation was available in this environment; OPT-0089 retains
  the authenticated browser waveform evidence, while public listening is the
  purpose of this preview.
