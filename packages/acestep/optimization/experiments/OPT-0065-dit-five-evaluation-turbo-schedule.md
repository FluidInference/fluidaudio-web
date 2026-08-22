# OPT-0065 — Five-evaluation Turbo schedule closure gate

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Risk: quality-affecting sampler trajectory change

## First-principles basis

The exact-kernel program still does not close the product target by itself. A
deliberately optimistic planning stack combines the measured OPT-0061 quad
attention result, OPT-0056's unmeasured nine-second selective-dense saving,
OPT-0055's six-evaluation factor, the revision-7 VAE selector projections, and
the smallest measured load/output context. It remains about `65.98 s`, roughly
`5.98 s` above the target. This is a planning bound, not end-to-end evidence.

After those proposed kernel savings, the evaluation-dependent DiT term is
`47.3572 s` at eight evaluations. Changing six evaluations to five therefore
has a planning-only saving of

`(6 / 8 - 5 / 8) * 47.3572 = 5.91965 s`.

That leaves essentially no margin: the same optimistic arithmetic reaches
about `60.06 s`, so this experiment cannot close the target alone. It becomes
decision-relevant only with a separately positive exact saving such as
OPT-0063 or OPT-0064 and an actual 180-second Generate-to-WAV run.

Five evaluations are derived from the same pinned shift-3 schedule formula,
not fitted timesteps. For `N = 5`, `sigma_i = (N - i) / N`, and `s = 3`:

`t_i = s * sigma_i / (1 + (s - 1) * sigma_i)`

The declared values are exactly:

`[1, 12/13, 9/11, 2/3, 3/7]`

or

`[1, 0.9230769230769231, 0.8181818181818182,
0.6666666666666666, 0.42857142857142855]`.

Round-to-nearest-even BF16 materialization must produce:

`[1, 0.921875, 0.81640625, 0.66796875, 0.427734375]`.

The corresponding BF16 Euler/final-clean update coefficients are:

`[0.078125, 0.10546875, 0.1484375, 0.240234375, 0.427734375]`.

## Frozen experiment direction

- Add a distinct diagnostic sampler identity. Keep the accepted eight-step
  profile and OPT-0055 six-step candidate unchanged.
- Change only evaluation count and the source-derived timestep list. Preserve
  shift-3 Euler, DCW double/Haar behavior, initial noise, conditioning,
  authenticated packages, kernel owners, BF16 materialization, VAE,
  normalization, and WAV output.
- Parameterize the existing bounded graph/control topology; do not fork layer
  math or tune an individual timestep against a listening fixture.
- Compare eight, six, and five evaluations from the same immutable request.
  Timestep `1` and its denoiser prediction must be raw-identical across all
  arms. Later trajectory divergence is expected and is quality evidence, not
  a kernel mismatch.
- Keep a single resident DiT, FIFO ownership, bounded submission,
  cancellation, and complete teardown. Never coexist packages or model-weight
  owners to make one arm appear faster.

## Correctness and quality gates

Validate in this order:

1. Prove the declared fractions, BF16 values, update coefficients, descriptor
   counts, progress totals, DCW scales, and final ping-pong buffer ownership.
2. Capture every five-step latent, the final latent, and raw pre-normalization
   waveform. Require deterministic repeats, finite/class stability, valid
   stereo, clean VAE seams, cancellation, and zero live resources. Report
   latent and waveform NRMSE, SNR, Pearson, maximum error, energy, spectrum,
   transients, and hashes against both eight and six evaluations. These metrics
   are descriptive and cannot approve a sampler change.
3. Present blinded eight/six/five WAVs for the accepted 12-second direct
   instrumental request. Five evaluations proceeds only with explicit owner
   listening approval and no obvious structural, rhythmic, tonal, or artifact
   regression.
4. If the direct fixture is approved, repeat the accepted 30-second
   planner-vocal/default-CoT fixture and require explicit owner approval for
   intelligibility, vocal continuity, arrangement, and artifacts.
5. Only after both short approvals, run the actual nominal 180-second direct
   product request through bounded raw decode, normalization, and durable WAV
   output. Under one minute is established only by that complete run.

Any collapse, lost vocal path, severe artifact, non-finite output,
nondeterminism, seam failure, or owner rejection abandons this exact schedule.
Do not adjust a timestep under OPT-0065; any fitted or custom list requires a
new experiment ID.

## Performance gate

Use the same final optimized package/kernel stack for all arms and retain
balanced raw samples under the repository thermal protocol. Require:

- at least `1.15x` realized DiT-stage speedup over six evaluations in both
  paired directions (the arithmetic ceiling is `6/5 = 1.20x`);
- at least `1.40x` over eight evaluations (ceiling `8/5 = 1.60x`);
- no VAE or non-DiT regression above `2%`; and
- at least `5,500 ms` complete Generate-to-WAV saving versus the approved
  six-evaluation arm before this candidate can be part of the final stack.

This experiment authorizes a listening candidate only. Even a positive short
gate does not authorize a production default or under-one-minute claim.

## Closeout — diagnostic-only, exact eight retained

The joint authenticated 8/6/5 direct-12 diagnostic reached its
listening-ready boundary and proved raw-U32 evaluation-0 denoiser identity
across all schedules. It produced three complete blinded WAVs without formal
timing, then released retained OPFS artifacts after download.

No owner listening attestation or mapping reveal occurred, and the mandatory
vocal, balanced timing, and five-versus-six product gates therefore remain
unmet. Five evaluations are a whole-trajectory quality change, not a
minuscule arithmetic drift that can be selected by numerical proxy. Evidence
is inconclusive/benchmark-only; production remains at the approved eight
evaluations and this exact schedule is not integrated.
