# OPT-0063 — VAE exact pointwise epilogue fusion

## Status

- Evidence: `positive`
- Disposition: `benchmark-only`
- Risk: exact FP16 storage-boundary preservation while changing dispatch and
  workspace ownership

## First-principles basis

Revision 7 still executes Snake and Add as standalone kernels. The decoder has
five block-head Snakes, fifteen residual
`Snake -> K7 -> Snake -> K1 -> Add` chains, and a final Snake/K7. Every former
edge is an FP16 storage boundary: producers store binary16, Snake widens that
stored value to FP32 for `exp`/`sin`, and Add widens both stored FP16 operands
before its FP32 sum and final FP16 store.

For the C2378 plan's 4,628 decoded latent frames, the frozen operation counts
contain `15,269,289,984` Snake elements and `6,539,919,360` Add elements.
Standalone pointwise traffic is therefore approximately:

`4 * Snake + 6 * Add = 100,316,676,096 bytes` (`93.43 GiB`).

Fusing only already-adjacent producer epilogues while preserving explicit FP16
rounding can remove an estimated `74,119,086,080 bytes` (`69.03 GiB`) of
intermediate traffic: `43,599,462,400` bytes from fifteen
`K1 -> Add -> successor Snake` chains, `26,159,677,440` bytes from fifteen
`K7 -> second Snake` edges, and `4,359,946,240` bytes from five
`ConvTranspose -> first residual Snake` reads. Arithmetic work, Snake
transcendentals, required externally-live residual values, and final stores
remain.

This is a new mechanism. OPT-0002 explicitly deferred Snake/conv fusion, and no
abandoned experiment tested an explicit-round-preserving fused epilogue on the
authenticated revision-7 stack. It is not a six-second saving claim: old
batch8 pointwise buckets scale to about `3.7 s` at C2378, while batch64 already
amortizes most dispatch cost. A realistic planning range is `1-2.5 s`, with
roughly `4 s` as a ceiling.

## Frozen direction

- Start with the largest complete chain, `K1 -> Add -> successor Snake`.
  Expand to K7 or ConvTranspose epilogues only after that isolated screen wins.
- Preserve each eliminated storage boundary in WGSL with typed temporaries:
  `producerRounded: f16`, then `addRounded: f16`, then Snake beginning from
  `f32(addRounded)`. Do not reassociate, retain FP32 producer accumulation,
  change transcendental spelling, or keep a higher-precision value across a
  former boundary.
- Residual units one and two must emit both the rounded Add value used by the
  next residual and its activated value. Residual unit three may emit only the
  externally required rounded result. Preserve package bytes, revision-7
  layouts, batch64 FIFO ordering, workspace alias lifetimes, and cancellation.
- Add a distinct fail-closed benchmark profile and fused-kernel identities.
  The production default remains unchanged.

## Gates

1. Prove graph-chain eligibility, exact label/shape routing, disjoint output
   ownership, former-boundary liveness, storage limits, and idempotent cleanup.
2. On C512 and representative C2314 ranges, compare every formerly materialized
   intermediate and final output as raw U16 against unchanged revision 7.
   Require zero mismatches, unwritten words, non-finite values, canary changes,
   or deterministic-repeat differences. Explicit casts alone are not proof.
3. After warmup and one nominal thermal check, run balanced C512/C2314 ABBA.
   Retain every raw sample and separately report affected-chain, decoder,
   command/drain, mixed-bucket, and outer wall. Require at least `1.15x` on the
   affected chains, decoder improvement in both directions, and at least
   `750 ms` projected C4500 saving before escalation.
4. A bit mismatch abandons this exact experiment. Any reordered-rounding
   version requires a new ID and the full numerical, waveform, and listening
   gates. A primitive pass authorizes only joint revision-7/C2378 integration
   testing, not production or an under-one-minute claim.

## Result

The bounded isolated `K1 -> explicit-f16 Add -> Snake` screen passed. This is
positive primitive/affected-chain evidence, not decoder or product evidence.

### Exactness

All `13` declared cases were raw-U16 exact at both the former Add boundary and
the final Snake output. The set comprised all five residual-block channel
tiers at C512, all five at C2314, and signed-zero/RNE,
subnormal-cancellation, and transcendental adversarial cases. Per execution,
`10,499,200` words were compared at each boundary; both deterministic
executions had zero former-Add, final-Snake, or rerun mismatches. Every output
was written, all canaries remained intact, and there were zero unwritten-qNaN
or non-finite results.

The authenticated benchmark identities were profile
`opt-0063-isolated-k1-add-snake-exact-screen-v1`, fused kernel
`opt-0063-vae-k1-add-snake-exact-fused-v1`, and fused-core SHA-256
`b50fa754171b243e410d427034867dd06329f5f1ff1d5cb2c372cae5e80ed825`.
The unfused owners remained OPT-0025 K1,
`ace-vae-fp16-portable-add-v1`, and
`ace-vae-fp16-portable-snake-v1`.

### Timing and projection

Compilation, allocation, and upload were excluded. Each case received one
warmup per arm and four retained timing samples per arm, with order
`unfused/fused`, `fused/unfused`, `fused/unfused`, `unfused/fused`. Each
affected-chain arm used one command buffer and one drain.

| Geometry | Projected unfused affected chain | Projected fused affected chain | Speedup |
| --- | ---: | ---: | ---: |
| C512 | `636.0000166296959 ms` | `522.0000019669533 ms` | `1.2183908318643257x` |
| C2314 | `2810.01870338805 ms` | `2332.078115302138 ms` | `1.2049419292389318x` |

The aggregate affected-chain speedup was `1.207401682233929x`, above the
frozen `1.15x` requirement. Applying only the measured C2314 projected delta
to two C2314 windows gives the declared C4500 planning calculation:

`2 * (2810.01870338805 - 2332.078115302138) = 955.881176171824 ms`.

That exceeds the `750 ms` escalation threshold. It remains a two-window
projection, not measured complete-decoder, C4500, waveform, or product saving.

### Exact thermal one-check

The frozen protocol was `wait-30s-then-one-level0-check`. Waiting started at
epoch millisecond `1786849957262`; the single check occurred at
`1786849987794`, after exactly `30532 ms`, and observed level `0`. Timed work
started `65 ms` later at `1786849987859` and completed at
`1786849987977`. This was the declared one-check screen, not a continuous
through-cleanup thermal trace, and supports no sustained-thermal claim.

### Lifecycle and decision

Cleanup destroyed all `260 / 260` created buffers, leaving zero live buffers
and zero live bytes. Peak live storage was `160872608` bytes; device
destruction and idempotent cleanup passed.

The isolated screen is therefore positive and remains benchmark-only. It
authorizes only the next joint revision-7/C2378 decoder screen, including the
previously frozen decoder, ownership, and lifecycle gates. It does not select
a decoder profile, change package bytes or the production default, integrate
the fused owner, or authorize waveform/listening, full-product, or
under-one-minute claims.

Receipt: [result.json](../results/OPT-0063/result.json), SHA-256
`06417bad4337c4a944c1d487ed544803fa2cbea011a325eb48299d7ce7c2f520`.

Registration itself preceded implementation and GPU work; the result above is
the later isolated benchmark screen.
