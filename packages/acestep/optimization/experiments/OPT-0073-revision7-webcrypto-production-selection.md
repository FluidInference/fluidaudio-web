# OPT-0073 — revision-7 warm-cache WebCrypto production selection

## Status

- Evidence: `positive`
- Disposition: `integrated`
- Risk: package trust, bounded host memory, cancellation latency, and warm
  initialization only; no model math, package bytes, GPU work, sampler, VAE,
  normalization, or audio-output change

## Decision basis

OPT-0071 remains a literal failed/inconclusive result. Its frozen `2%`
per-stage rule rejected ratios dominated by tiny device, manifest, tokenizer,
and READY-publication buckets, including a `1.0 ms` device-request delta. It
must not be relabelled after observation.

The same balanced production-seam receipt provides decisive causal evidence:

- the complete untimed trust proof matched `158` logical records, `156` unique
  physical digests, and `7,325,999,133` physical bytes;
- the unchanged ordinary READY boundary authenticated its actual `151/149`
  logical/unique payload inventory and `7,157,207,581` physical bytes;
- all four arms were exact, nominal, cache-only, mutation-free, cancellation
  safe, and lifecycle clean, with zero download bytes;
- sequential WebCrypto retained exactly one file, at most `121,668,608`
  explicit live bytes, and a `365,005,824`-byte conservative transient bound;
- candidate authentication median was `6,290.35 ms` at
  `1,137,807,527.57 B/s`; and
- complete READY savings were `22,724.3 ms` and `22,098.1 ms`. Aggregate
  unrelated wall improved `1.3 ms` in the first direction and regressed only
  `16.8 ms` in the second, so it did not absorb the approximately 22-second
  authentication win.

On 2026-08-16 the owner explicitly delegated best judgment to integrate clear
speed wins. This new ID applies that judgment prospectively to the current
revision-7 production tuple. It does not rewrite OPT-0069 or OPT-0071, relax
package authentication, or authorize a general performance-gate waiver.

## Frozen production selection

1. Select `webcrypto-whole-file` as the sole production warm-cache payload
   authentication owner for the pinned main, revision-7 dense, and revision-7
   VAE packages. Keep exact manifest size/SHA comparison, logical alias proof
   reuse, cache-only failure behavior, progress, acquisition order, and proof
   binding unchanged.
2. Read and digest strictly one physical file at a time. Release its explicit
   bytes before opening the next file. Never parallelize whole-file reads or
   digests, and never skip, sample, truncate, or marker-trust payload bytes.
3. Keep the explicit cancellation boundaries immediately before read, before
   digest, and after the in-flight digest. WebCrypto is not internally
   abortable; at most one sub-128-MiB file may finish before cancellation is
   observed.
4. Do not add a scalar fallback. A missing WebCrypto capability, read failure,
   digest rejection, cancellation, length mismatch, or hash mismatch fails
   explicitly without publishing a proof.
5. Retain the scalar owner only as an explicit benchmark/control seam. Do not
   expose a product A/B selector or silently switch owners at runtime.

## Integration and product checks

1. Focused static tests must prove the production constant selects WebCrypto,
   one-file sequencing and release remain bounded, all manifest comparisons
   remain exact, failure paths publish no proof, and no scalar fallback or
   parallel whole-file path is reachable.
2. Run one short request on the final OPT-0070/0072 product tuple from a warm
   cache. Require ordinary READY, zero cache downloads/mutations, finite and
   structurally valid raw/WAV output, stable metadata, no device loss, and
   clean worker/GPU/payload disposal. Because authentication cannot change
   model arithmetic, any output-identity mismatch is a failure.
3. The final requested 180-second generation is the cumulative production and
   listening check. Record cache-ready initialization separately from
   Generate-to-WAV and combined wall time, retain thermal/lifecycle evidence,
   and save the WAV in `~/Downloads`. This record authorizes no under-one-
   minute claim unless the measured product wall actually clears it.

## Authority

- [OPT-0069 isolated hash screen](OPT-0069-warm-cache-authentication-hash.md)
- [OPT-0069 failed/inconclusive result](../results/OPT-0069/result.json)
- [OPT-0071 production-seam gate](OPT-0071-warm-cache-webcrypto-production-integration.md)
- [OPT-0071 failed/inconclusive result](../results/OPT-0071/result.json)
- [OPT-0070 exact production promotion](OPT-0070-exact-quad-c2378-production-promotion.md)
- [OPT-0072 revision-7 VAE production promotion](OPT-0072-revision7-vae-c2378-production-promotion.md)

No production code/default change, browser/GPU work, cache mutation, or result
was created when this follow-up was registered.

## Closeout — integrated and product-validated

Checkpoint `33feaf6` selected `webcrypto-whole-file` as the sole production
owner. The scalar stream remains an explicit diagnostic control; an injected
WebCrypto read failure proved that production invalidates/rebuilds the cache
entry without invoking a scalar fallback. The focused acquisition and OPT-0069
/0071 suites passed `24/24`, and the combined TypeScript check passed.

The final OPT-0070/0072 tuple then completed a warm-cache 12-second smoke in
`17.0 s` and the requested 180-second production run in
`94,774.80000007153 ms` Generate-to-WAV. The final receipt recorded
`54,986.200000047684 ms` DiT denoising, `32,586.700000047684 ms` VAE decode,
`3,803` cooperative drains, `3,800 ms` requested idle, and
`3,826,885,376` peak tracked GPU bytes. The two actual VAE windows were both
C2314 under the authenticated C2378/overlap64 ceiling.

The downloaded IEEE-F32 stereo WAV is exactly 180 seconds: `8,640,000` frames,
`17,280,000/17,280,000` finite and nonzero samples, peak
`0.8912509083747864`, and SHA-256
`e6a7f69b44639db177ac3f72cb2cba8d7027924795cffd3810daecec1cb87df8`.
The run began after more than 50 seconds continuously at thermal level 0;
level 1 began `72.308 s` after Generate and level 2 began about `2.984 s`
after the measured Generate-to-WAV boundary while receipt/download cleanup was
still active. The stored output was released after download and no device loss
occurred.

Authority:

- [final result](../results/OPT-0073/result.json)
- [browser receipt](../results/OPT-0073/final-180s-receipt.json)
- [thermal trace](../results/OPT-0073/raw/final-180s-thermal.jsonl)

The mechanism is positive/integrated. The measured product wall is `94.8 s`,
so this closeout makes no under-one-minute claim.
