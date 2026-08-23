# OPT-0054 — VAE revision-7 mixed-layout K4 integration

## Status

- Evidence: `inconclusive`
- Disposition: `superseded`
- Risk: exact ConvTranspose layout plus approximate K7 FP16 K4 partials

## Purpose

OPT-0052 and OPT-0057 both require a new authenticated VAE package and their
tensors are disjoint. OPT-0052 is raw-bit exact relative to the current
ConvTranspose arithmetic. OPT-0057 is raw-bit exact only relative to
OPT-0024's K4 arithmetic; that K4 arithmetic is approximate relative to the
current revision-6 scalar-FP32 K7 owner.
Generating two successive 169 MB package identities and running two redundant
full C512/C4500 gates would add latency without isolating any remaining
numerical risk. Integrate them in one revision while retaining separate owner
and timing attribution:

- K1 remains the exact OPT-0025 tile-major layout/owner;
- biased C1024/C512/C128 K7 uses OPT-0057 row-reuse K4, while C256 and all
  ingress/unbiased-final K7 remain on the current native layout/owner;
- ConvTranspose block 0 remains OPT-0040 exact polyphase, while blocks 1–4 use
  OPT-0052's raw-exact K4 layout/owners;
- Snake, Add, ingress, window geometry, and batch64 scheduling remain unchanged.

This revision must replace, not duplicate, each tensor representation. Stable
DiT revision 8 and all source snapshot identities remain untouched.

## Gate

1. Extend only the deterministic converter/manifest/package policy needed for
   the two declared transforms. Prove complete consumed/excluded accounting,
   exhaustive logical/physical bijections, reproducible bytes/hashes, and
   transactional authentication.
2. Add one fail-closed fixed32 production profile with exact label/shape/layout
   routing and per-family dispatch attribution. Keep the current revision-6
   profile as the sequential oracle during the gate, not as a fallback inside
   the new profile.
3. On the accepted C512 fixture, keep all package owners sequential and never
   co-resident. Require the revision-7 output to be raw-bit identical to a
   revision-6 package running the same OPT-0024 K4 K7 arithmetic and exact
   OPT-0040 ConvTranspose selector. Separately compare revision 7 against the
   current revision-6 scalar-FP32 K7 oracle under OPT-0044's frozen numerical
   envelope. Require deterministic reruns, exact topology outside the selected
   owners, cancellation, and zero live resources. Run balanced AB/BA after one
   nominal thermal check. Require both directions to improve K7 and
   ConvTranspose family walls, median K7 at least `1.50x`, median
   ConvTranspose at least `1.30x`, and no decoder/outer-wall regression.
4. Run one C4500 candidate correctness/lifecycle pass against the exact
   revision-6 oracle. Require deterministic complete-waveform and seam metrics,
   bounded peak memory, clean cancellation, and exactly-once destruction. Raw
   waveform identity is not expected because K7 arithmetic changed. A
   thermally stable long timing is desirable but not required; no ratio may be
   inferred from an order-confounded long run.

Package, routing, numerical, and timing passes still do not authorize
production selection. OPT-0044's activation-trajectory, 12-second product
WAV/listening comparison, and explicit owner approval remain mandatory because
the K7 K4 reduction changes arithmetic. A fully approved pass may then replace
the current revision-6 production VAE profile. It still makes no
under-60-second claim until an actual nominal 180-second product run measures
it.

## Closeout — original gate superseded

The original OPT-0054 preparation did not reach READY; its failed setup
evidence remains preserved and the literal experiment is inconclusive. The
corrected same-arithmetic oracle and complete C512 quality/timing gate were
allocated and passed under OPT-0066, while OPT-0072 owns production promotion
after the complete C4500 waveform gate. Close OPT-0054 as superseded rather
than relabelling its original evidence.
