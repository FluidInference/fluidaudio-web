# OPT-0024 — deterministic-only output

## First-principles target

Four consistency-distilled stem evaluations consume 64 of the graph's 80
transformer blocks and about 16.5 seconds of the retained 23.4-second model
wall. The runtime already computes four complete deterministic stem spectra
before refinement. This arm exposes those released deterministic outputs
directly rather than pretending another kernel tweak can match the leverage of
removing four network evaluations.

The per-request mode is explicit and default-off. It returns each deterministic
spectrum through the existing zero-DC conversion and ISTFT. It does not add
noise, scale a CD input, run CD mapping or transformer blocks, apply the
consistency affine, or clamp the output.

Condition capture is also disabled in this mode. The deterministic graph skips
the CD STFT adapter, all 17 condition adapters, the roughly 972-MiB condition
arena, and the roughly 634-MiB convolution intermediates because none of them
feed the deterministic masks.

## Validation

The default remains refined. The output mode is carried explicitly through the
public API and worker result; deterministic results omit CD diagnostics rather
than fabricating them. The panel also requires exact per-stem deterministic
peak/RMS diagnostics between standalone and CD-capturing executions. Type
checking, all 8 unit tests, the WebGPU probe, and diff checks passed before the
waveform panel.

## Supplied-WAV result

A fresh isolated Chrome 151 process initialized one full-temporal/Flash
runtime, ran deterministic-only cold, then ran refined output with the same
decoded PCM and seed `0xd1c05e`.

| Stage | Deterministic only | Refined |
| --- | ---: | ---: |
| deterministic | 5,101.1 ms | 5,168.9 ms |
| mapping | 0 ms | 17.1 ms |
| four refinements | 0 ms | 17,555.6 ms |
| ISTFT | 583.8 ms | 1,360.4 ms |
| **total** | **5,814.6 ms** | **24,727.0 ms** |

This is a **4.25× speedup** and **18,912.4-ms saving**. The supplied-waveform
drift versus the refined result was:

| Stem | NRMSE | SNR | cosine | RMS drift |
| --- | ---: | ---: | ---: | ---: |
| drums | 0.0582 | 24.71 dB | 0.99853 | 1.98% |
| bass | 0.0588 | 24.61 dB | 0.99853 | 2.14% |
| other | 0.0538 | 25.38 dB | 0.99868 | 1.47% |
| vocals | 638.57 | -56.10 dB | 0.00062 | 63,757% |

The vocals ratio is dominated by a near-zero deterministic vocal reference on
this particular fixture: its peak is only about `6.8e-5`, while the refined
vocal peak is about `0.013`. It does not establish which output is closer to a
ground-truth stem.

## Sustained benchmark

The selectable release harness then ran one warmup and three measured
deterministic-only passes in a fresh isolated Chrome 151 profile. End-to-end
samples were 5,936.9, 5,921.2, and 5,840.2 ms, for a **5,921.2-ms median**
(range 5,840.2–5,936.9 ms). The median model total was 5,912.8 ms, including a
5,143.1-ms deterministic pass and 613.2-ms ISTFT. Unlike the longer refined
panels, this short mode did not exhibit an upward thermal slope.

## Disposition

Retain deterministic-only as an explicit fast mode because its 5.81-second
cold boundary and 5.92-second sustained median are the first results near the
product's responsiveness target. Do not replace the refined default from
same-model waveform drift alone. Promotion requires licensed ground-truth
stems, per-stem SDR/SI-SDR and transient gates, and blind listening; the two
model outputs are not ground truth for each other.
