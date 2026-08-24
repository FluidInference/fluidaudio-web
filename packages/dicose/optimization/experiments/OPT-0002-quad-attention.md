# OPT-0002 — quad-query attention

## Hypothesis

The original 256-lane attention workgroup loaded each K/V row once for eight
queries. Four independent query streams per fixed 32-lane subgroup can reuse
that same K/V row for 32 queries, reducing workgroup dispatches and shared
K/V traffic by four without changing each query's arithmetic.

## Correctness gate

`pnpm test:webgpu` runs the prior query-8 schedule alongside the new default
schedule on a two-sequence, 13-token tail shape. All 13,312 raw f16 output
words matched exactly, with no WebGPU validation or uncaptured errors.

The full isolated Chrome fixture run also passed the deterministic f16
acceptance envelope and reported bit-identical stem/diagnostic statistics with
no page, console, or device errors.

## One-run evidence

| Boundary | Generic query-8 (ms) | Quad query (ms) |
| --- | ---: | ---: |
| deterministic BS-RoFormer | 22,003.0 | 18,923.1 |
| four CD refinements | 80,629.9 | 62,486.5 |
| complete model timing | 104,924.6 | 83,621.5 |

That clean-profile acceptance sample is a 20.3% full-model improvement.

## Release benchmark

The automated fresh-profile Chrome 151 benchmark completed one warmup and
three measured full-WAV runs without page/device errors:

| Statistic | End-to-end time (ms) |
| --- | ---: |
| min | 134,853.3 |
| median | 136,560.2 |
| mean | 139,202.6 |
| max | 146,194.2 |

Stage samples (`prepare / deterministic / mapping / refinement / ISTFT / total`
in ms) were:

1. `145.3 / 45,301.1 / 7.0 / 98,899.5 / 1,356.8 / 146,179.4`
2. `126.3 / 33,650.9 / 7.4 / 99,328.2 / 1,290.9 / 134,839.9`
3. `121.9 / 33,144.4 / 6.9 / 101,478.7 / 1,348.0 / 136,548.2`

The multi-run sustained figures are slower than the isolated cold acceptance
sample, consistent with thermal variation across repeated long Metal runs.
They are recorded separately rather than being used to overstate a cold-run
speedup.
