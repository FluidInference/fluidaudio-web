# OPT-0037 — DiT K4 layout and trajectory integration

## Status

- Evidence: `negative`
- Disposition: `abandoned`
- Risk: approximate FP16 K4 partial reduction with FP32 running state

## Hypothesis

OPT-0032 improved the weighted four-shape primitive score from `199.65` to
`142.10 ms` (`1.404996x`) while passing full and adversarial numerical gates.
The repeated 24-layer × eight-evaluation dense projection is about `42.85 s`;
carrying only the measured ratio suggests `~30.50 s`, a credible `~12.35 s`
saving. A converter-native replacement layout can realize that mechanism
without retaining both the current N256/K32 and new K4 copies of several
gigabytes of weights.

## Frozen integration direction

- Extend the sole authenticated converter/package path to replace, not
  duplicate, eligible repeated-layer dense tensors with
  `[N/128,K/4,output4,lane32,K4]` FP16 storage.
- Give the package a new authenticated identity and fail closed on layout,
  fixed32 subgroup, shader-f16, shape, or size mismatch. Do not add a native
  Metal/MPS/WebNN path or browser flag.
- Route only the exact OPT-0032 production shapes/operations through its
  isolated owner. Preserve FP32 inputs, FP32 running accumulators and outputs,
  operation order, sampler/DCW, attention, norms, residuals, and eight
  evaluations.
- Keep a separately selectable exact profile for oracle comparison. Backward
  compatibility with the old optimized package is not required.

## Gates

Authenticate deterministic conversion and complete consumed/excluded tensor
accounting, then validate in order: packed-layout inversion/index probes;
actual source-weight layer outputs; every denoise step; final latent;
12-second raw waveform and listening; and one thermally gated M2250 DiT graph
timing with family attribution. Record NRMSE/SNR/Pearson/max error and class or
nonfinite changes at each numerical level. Production selection requires the
owner's listening approval and a material graph/stage win; the primitive result
alone authorizes none. A later block-K8/K16 experiment must get a new ID.

## Result

The authenticated sequential M2250 gate completed once on the local M3. The
revision-7 control reached and detached its final latent, then completed
pipeline cleanup and backend/device disposal before the revision-8 candidate
was created. No VAE weights were acquired and no audio path ran. Both arms
used the exact same canonical 180-second direct request, conditioning hashes,
main package, eight-step sampler/DCW contract, and 288,000-element final
latent shape.

The candidate remained finite and passed three of the four frozen aggregate
thresholds, but failed the maximum-error threshold:

| Metric | Result | Gate |
| --- | ---: | ---: |
| NRMSE | `0.0165843012` | at most `0.02` |
| SNR | `35.6060565 dB` | at least `34 dB` |
| Pearson | `0.9998616362` | at least `0.999` |
| Maximum absolute error | `0.9955760241` | at most `0.25` |

There were `287,999` differing U32 words, zero non-finite values, and `802`
finite sign-class changes (`412` negative-to-positive and `390`
positive-to-negative). The worst element was index `273,508`: control
`-1.1017516851`, candidate `-2.0973277092`. Control/candidate final-latent
SHA-256 values were respectively
`1812a085f48b7879212633c7193dda08ec2854852a492ce661262c5e6be98f4c`
and
`7da2b7a03fcad61547d56400ba2c499af34fe5459f002a8ab2a998b9214e9943`.

Fixed-order graph submit-through-drain context was
`121,691.70 -> 115,350.20 ms`; graph wall was
`125,601.40 -> 119,277.10 ms`. The receipt deliberately computes no speedup:
the candidate always ran second, no thermal protocol was applied, and this was
a correctness gate. Even as context, the roughly 6.3-second graph delta is
smaller than the primitive-only projection.

The all-216-dense K4 profile therefore stops before waveform/listening and
must not remain the product default. The primitive remains useful evidence,
but its bounded local error accumulated into a trajectory outlier beyond the
frozen integration envelope. A selective-precision follow-up requires a new
ID and must begin with the unique K6144 MLP down-projection reduction rather
than weakening this gate.

Receipt: [result.json](../results/OPT-0037/result.json), SHA-256
`4bb35dcbca8356c92be1bbe9eec72aab03e2646275bec42377e5266c61316bc0`.
