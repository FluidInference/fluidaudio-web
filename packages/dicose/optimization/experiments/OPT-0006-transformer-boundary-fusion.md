# OPT-0006 — transformer boundary fusion

## Hypothesis

Each of the 80 transformer blocks materialized attention context only for a
separate gate kernel to rewrite it, then ran two full-tensor residual-add
kernels after the output and feed-forward projections. Across the supplied
fixture these 240 dispatches update approximately 7.55 billion f16 elements.
Fusing the operations at their producer stores can remove that traffic without
changing model arithmetic.

## Mechanism

- The eight-value gate projection is scheduled before attention. Attention
  rounds its context to f16, applies the same f32 sigmoid division, and rounds
  the gated result back to f16 in its final store.
- Packed output and feed-forward projections accept a residual binding. They
  round the projection to f16 first, add the f16 residual in f32, and round the
  result to f16, preserving the former two-dispatch boundary.
- No activation-bearing projection uses residual fusion; the production fused
  sites have no nonlinear post-op.

## Correctness gates

`pnpm test:webgpu` compares:

- packed projection plus a standalone add against the fused residual owner over
  896 raw f16 words; and
- direct-trig query8 plus standalone gating against table-backed quad attention
  with fused gating over 13,312 raw f16 words.

Both comparisons have zero mismatches. The full WAV also retained every
reported OPT-0004 output and diagnostic statistic exactly, with no GPU or page
errors.

## Full-graph evidence

Isolated Chrome 151, zero warmups, one measured cold run per arm:

| Boundary | OPT-0004 (ms) | Fused (ms) | Speedup |
| --- | ---: | ---: | ---: |
| deterministic | 10,828.5 | 10,627.5 | 1.02× |
| four CD refinements | 32,383.4 | 30,810.7 | 1.05× |
| complete model timing | 45,206.9 | 43,595.6 | 1.04× |
| page end-to-end timing | 46,084.7 | 44,453.4 | 1.04× |

This is cold acceptance evidence rather than a sustained thermal benchmark.

## Disposition

Positive and integrated. The result saves a modest but repeatable structural
floor while preserving raw primitive output and full reported behavior.
