# OPT-0055 — Six-evaluation Turbo schedule quality/performance gate

## Status

- Evidence: `inconclusive`
- Disposition: `benchmark-only`
- Risk: quality-affecting sampler trajectory change

## First-principles basis

Kernel work alone may not close the product target. The pinned direct path runs
the complete 24-layer DiT eight times, so removing two evaluations eliminates
exactly one quarter of the repeated denoiser work: 48 layer evaluations, all
their dense/attention/norm/residual operations, and their scheduling overhead.
This is multiplicative with the revision-8 K4 dense path and any later exact
attention integration. It does not reduce VAE, model-load, conditioning, or
normalization time.

Six evaluations are not an invented solver. Keep the pinned Euler, shift-3,
DCW-double/Haar behavior and derive the ordinary six-point Turbo schedule from
the same source formula that produces the accepted eight-point values. For
`N = 6`, base sigma `sigma_i = (N - i) / N`, and shift `s = 3`:

`t_i = s * sigma_i / (1 + (s - 1) * sigma_i)`

The declared values are therefore exactly:

`[1, 15/16, 6/7, 3/4, 3/5, 3/8]`

or

`[1, 0.9375, 0.8571428571428571, 0.75, 0.6, 0.375]`.

After the pinned source's BF16 schedule materialization, the effective values
must be:

`[1, 0.9375, 0.85546875, 0.75, 0.6015625, 0.375]`.

The corresponding BF16 Euler/final-clean update coefficients are
`[0.0625, 0.08203125, 0.10546875, 0.1484375, 0.2265625, 0.375]`.

## Frozen experiment direction

- Add a distinct diagnostic generation/sampler profile. Keep the accepted
  eight-evaluation profile and production default unchanged.
- Parameterize only the bounded graph/control topology needed to execute six
  evaluations. Do not fork model math or special-case individual layers.
- Hold the initial noise, conditioning/context, cover/DCW inputs, Euler update,
  shift, BF16 materialization, precision profile, weights, kernel owners, VAE,
  overlap policy, normalization, and output writer fixed between arms.
- At evaluation zero, both arms have timestep `1` and the same latent. Require
  the denoiser prediction to match exactly there; after the first Euler update,
  trajectory divergence is expected and must not be mislabeled numerical
  kernel error.
- Keep one FIFO graph owner, bounded command submission, cancellation, and
  complete teardown. Six-evaluation resources must not coexist with a second
  resident DiT package.

## Correctness and quality gates

Validate in this order:

1. Prove the six declared values, BF16 values, update coefficients, graph
   descriptor counts, ping-pong final-buffer ownership, and progress totals.
2. With identical authenticated inputs, require raw identity for the common
   timestep-1 denoiser output, then capture every six-step latent, the final
   latent, and the raw pre-normalization waveform. Require deterministic
   repeats, no non-finite/class changes, valid stereo output, clean seams, and
   complete lifecycle/cancellation evidence. Report final-latent and waveform
   NRMSE, SNR, Pearson, maximum error, energy/spectral summaries, and hashes
   against eight evaluations as descriptive evidence, not a substitute for
   listening.
3. Generate the accepted 12-second direct instrumental request at both eight
   and six evaluations and present the WAVs for owner listening. Because this
   changes sampler behavior, production selection requires explicit owner
   approval even if mechanical metrics look strong.
4. Only after the short direct candidate is approved, repeat the accepted
   30-second planner-vocal/default-CoT listening fixture before a general
   production-default claim. Planner timing remains reported separately.

Any obvious musical collapse, lost vocal path, severe artifact, unstable
repeat, non-finite output, seam failure, or owner rejection abandons the
six-evaluation candidate unchanged. Do not tune the schedule against one
fixture under this ID; any altered/custom timestep list needs a new experiment.

## Performance gate

Use the same optimized kernel/package profile for both arms and measure the
DiT graph/stage separately from decode and full Generate-to-WAV wall. A valid
screen must retain complete balanced samples under the repository thermal
protocol. Require at least `1.20x` realized DiT-stage speedup and no VAE or
non-DiT regression. The arithmetic ceiling from evaluation count alone is
`8/6 = 1.3333x`; a materially smaller gain must be attributed before further
quality escalation.

This experiment authorizes a listening candidate only. It makes no quality,
production-default, or under-60-second claim until the declared gates and an
actual nominal 180-second product run pass.

## Closeout — diagnostic-only, exact eight retained

The authenticated 8/6/5 diagnostic implementation completed the direct
12-second preparation gate: all three schedules shared an exact raw-U32
evaluation-0 denoiser output, produced complete blinded WAV artifacts, and
reached the listening-ready boundary. Formal timing was intentionally absent.

No genuine owner listening attestation was provided, the blind mapping was not
revealed, and neither the required vocal fixture nor a balanced timing gate was
run. The agent did not fabricate auditory approval. Because removing two
complete denoiser evaluations changes the full diffusion trajectory rather
than introducing minuscule arithmetic drift, the owner-delegated VAE judgment
does not transfer. The retained OPFS artifacts were released after the three
blind files were downloaded. Evidence is inconclusive/benchmark-only and the
eight-evaluation production schedule remains selected.
