# Accepted baseline

> Correctness reset (2026-08-23): the timing results below remain useful as
> performance measurements, but their old output gates did not compare against
> upstream and allowed shared GELU and CD-time-conditioning defects to pass.
> Do not treat them as quality evidence. See
> [`CORRECTNESS_AUDIT.md`](CORRECTNESS_AUDIT.md) for the repaired reference
> gates and audit findings.

Fixture: `Mixture_audio_1.wav` (SHA-256
`9e487f3a84b974b11b47442d0fd99512ab4826130d04351e8c9625d84e107bb7`),
duplicated to stereo and linearly resampled from 22.05 kHz to 44.1 kHz.

Browser: isolated headless Chrome 151.0.7922.173 on the Apple/Metal WebGPU
adapter, with `shader-f16` and fixed 32-lane `subgroups`.

The pre-optimization generic-kernel acceptance run completed with no device or
page errors and passed deterministic fixture diagnostics:

| Boundary | Time (ms) |
| --- | ---: |
| deterministic BS-RoFormer | 22,003.0 |
| four CD refinements | 80,629.9 |
| complete model timing | 104,924.6 |
| page end-to-end timing | 105,713.2 |

This is a single clean-profile acceptance sample, not a statistical benchmark.
The retained baseline is superseded only after a quality-gated multi-run
benchmark is recorded in `LEDGER.md`.

## Current exact reference

After OPT-0003 through OPT-0016, the unchanged `refined` + `full` graph was
remeasured with the release protocol in isolated Chrome 151: one warmup and
three measured runs. End-to-end samples were 24,665.2, 25,548.8, and 26,230.3
ms, for a **25,548.8-ms median** (range 24,665.2–26,230.3 ms). The corresponding
median model timing was 25,537.2 ms: 5,279.4 ms deterministic, 18,018.8 ms for
the four refinements, and 1,559.8 ms ISTFT.

This is the current exact performance reference. It is **5.34× faster** than
the original 136,560.2-ms sustained median. The rising samples are retained as
evidence of thermal throttling rather than collapsed into the median alone.
