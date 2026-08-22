# OPT-0044 — VAE K7 dot4 trajectory integration

## Status

- Evidence: `positive`
- Disposition: `superseded`
- Risk: approximate FP16 K4 partial reduction with FP32 running state

## Hypothesis

OPT-0024's authenticated revision-6 C512 subsystem reduced pure K7 wall from
`4,805.70 ms` to `2,232.85 ms` (`2.15227x`), complete decoder wall from
`6,999.55 ms` to `4,327.55 ms`, and outer wall from `8,300.95 ms` to
`5,613.75 ms`. The complete-window waveform remained deterministic and passed
the declared numerical envelope (NRMSE `0.00184056`, SNR `54.701 dB`, Pearson
`0.999998306`). Routing the same 16 biased production K7 operations through
that owner is the largest measured remaining VAE arithmetic opportunity.

## Frozen integration direction

- Add a distinct fixed32 optimized profile/topology that differs from the
  authenticated OPT-0028 exact-packed production profile only at the 16 biased
  K7 Conv1D operations already screened by OPT-0024.
- Preserve native revision-6 K7 weights, K1/ConvTranspose packed layouts,
  operation order, FP16 activation storage, FP32 running state, C512/overlap64
  windowing, batch64 submission, and all non-K7 owners.
- Keep the eight unbiased-final K7 quanta on the exact owner. Fail closed on
  operation identity, bias presence, shape, fixed32 subgroup, or package
  mismatch; add no browser repack, duplicate weights, fallback, native Metal,
  WebNN, or experimental browser flag.
- Do not combine the OPT-0040 ConvTranspose selector or a new K8/K16 reduction
  under this ID. Those mechanisms retain independent evidence.

## Gates

Validate the selected K7 activation trajectory at every decoder block before
checking a complete C512 window. Then require deterministic seam-safe C4500
raw waveform metrics against the accepted exact OPT-0028 oracle, a 12-second
product WAV/listening comparison under the ordinary -1 dBFS normalization, no
new non-finite or qualitative-collapse events, clean cancellation/lifecycle,
and a thermally valid balanced subsystem timing. Production selection requires
explicit owner listening approval; the existing numerical and timing receipt
alone authorizes no default change.

## Authority

- Primitive and C512 result: [OPT-0024](OPT-0024-vae-k7-direct-subgroup-fp16-dot4.md)
- C512 receipt: [c512-subsystem.json](../results/OPT-0024/c512-subsystem.json)
- Exact package/topology oracle: [OPT-0028](OPT-0028-vae-exact-packed-kernel-integration.md)

## Closeout — superseded by the measured revision-7 dual-K4 promotion

This revision-6 K7-only direction was not selected unchanged. OPT-0066 later
screened the complete K7 plus ConvTranspose dual-K4 physical owner, and
OPT-0072 promoted that measured owner with a distinct revision-7 package and
public identity. Its C4500 gate covered all `17,280,000` raw samples at NRMSE
`0.0015226894`, SNR `56.3478 dB`, Pearson `0.999998841`, maximum error
`0.00787494`, deterministic repeat identity, clean seams, and bounded
lifecycle. The final OPT-0073 run then exercised that production identity over
the complete 180-second WAV. Preserve OPT-0044's positive causal K7 evidence,
but mark the proposed integration superseded by OPT-0066/0072 rather than
pending.
