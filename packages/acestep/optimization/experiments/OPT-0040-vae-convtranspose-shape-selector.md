# OPT-0040 — VAE ConvTranspose per-shape reuse selector

## Status

- Evidence: `inconclusive`
- Disposition: `superseded`
- Risk: exact tap-then-Cin FP32 accumulation and explicit FP16 store

## Hypothesis

OPT-0036 proved two complementary exact geometries. Channel reuse was fastest
for blocks 0-2, while row reuse was fastest for blocks 3-4. Choosing the
already validated owner by static production operation yields a planning-only
five-shape score of `326.75 ms` versus OPT-0026's `534.0 ms` (`1.634277x`),
and should reduce the remaining exact ConvTranspose family without changing
weights, arithmetic, output bits, or package identity.

## Frozen mechanism

- Route block 0, 1, and 2 `conv_t1` operations to OPT-0036 channel reuse.
- Route block 3 and 4 `conv_t1` operations to OPT-0036 row reuse.
- Keep the authenticated revision-6 polyphase layout, increasing tap-then-Cin
  FP32 reduction order, bias handling, explicit FP16 store, and all other VAE
  owners unchanged.
- Cache the five dispatch owners exactly as the current OPT-0026 owner does;
  add no browser repack, duplicate weights, fallback layout, or approximate
  arithmetic.

## Gates

Prove exact routing/topology statically, then compare the complete authenticated
C512 decoder against the current OPT-0028 exact profile. Require raw-U16 output
identity, deterministic completion, clean cancellation/lifecycle, and at least
`1.10x` ConvTranspose-family improvement with no complete-decoder regression
under one nominal paired timing. If positive, combine it with batch64 in a
separately measured C4500 or C2378 run; this primitive projection alone is not
a long-song or under-60-second claim.

## Closeout — superseded by revision-7 validation

The standalone OPT-0040 gate did not produce a frozen result receipt, so its
literal evidence remains inconclusive. Its exact revision-6 selector remained
an explicit oracle and input to the later OPT-0052/0054/0066 revision-7 work,
which owns the completed package, routing, numerical, and timing evidence.
Close this original integration allocation as superseded, preserving the
OPT-0036 projection as benchmark evidence only.
