# OPT-0008 — strided time attention without layout transposes

## Hypothesis

The model stores transformer rows physically as `[time, band, dim]`, but each
time-axis block previously copied the complete tensor to `[band, time, dim]`
and back solely to make attention tokens contiguous. RMS normalization, dense
projections, residuals, activations, and adapters are all row-local and do not
require that permutation. Teaching time attention to map logical `(band,
time)` to physical row `time * bands + band` can remove the copies while
leaving all model arithmetic unchanged.

The complete supplied-WAV graph has 40 time-axis blocks: eight deterministic
and eight in each of four CD evaluations. Removing both copies around each
block eliminates 80 dispatches and approximately 9.06 GB of aggregate f16
read/write traffic. It also removes one 56.6 MB workspace tensor.

## Mechanism

- Quad attention accepts an explicit strided-row mode. Contiguous frequency
  attention retains `sequence * tokens + position`; time attention uses
  `position * sequences + sequence` for Q, K, V, fused gates, and output.
- RoPE still uses the logical token position, never the physical row.
- Deterministic time-condition adapters and CD consumers now both retain the
  same `[time, band, dim]` physical order.
- The query8 reference rejects strided mode rather than silently using an
  unsupported layout.

The main risk was locality: successive K/V rows for one band are roughly 190
KiB apart. This experiment therefore required a full-boundary timing rather
than acceptance based on eliminated traffic alone.

## Correctness gates

The browser probe now exercises all four quad query streams and the tail with
3 sequences and 37 tokens. It compares direct-trig query8 plus standalone
gating with both contiguous and physically transposed strided quad attention.
After inverse permutation, each comparison has zero mismatches across 56,832
raw f16 output words.

The full supplied WAV retained every OPT-0006 stem and diagnostic statistic
exactly, with no validation, console, or page errors.

## Full-graph evidence

Isolated Chrome 151, zero warmups, one measured cold run per retained arm:

| Boundary | OPT-0006 (ms) | Strided (ms) | Speedup |
| --- | ---: | ---: | ---: |
| deterministic | 10,627.5 | 10,304.8 | 1.03× |
| four CD refinements | 30,810.7 | 29,587.5 | 1.04× |
| complete model timing | 43,595.6 | 41,861.8 | 1.04× |
| page end-to-end timing | 44,453.4 | 42,639.9 | 1.04× |

This is cold acceptance evidence rather than a sustained thermal benchmark.

## Sustained integrated benchmark

The final integrated OPT-0008 stack was then run under the same release
protocol as the retained baseline: isolated Chrome 151, one warmup, and three
measured runs. End-to-end samples were 46,368.0, 55,192.5, and 61,277.2 ms,
for a **55,192.5 ms median** (range 46,368.0–61,277.2 ms). Against the retained
136,560 ms baseline median (134,850–146,190 ms), this is **2.47× faster** and a
**59.6% wall-time reduction**. The widening thermal range is material and is
reported rather than hidden by the median.

## Disposition

Positive and integrated. The strided reads do not outweigh the removed full
tensor copies on the tested Apple Metal backend, and the change preserves raw
primitive output and every reported full-model statistic.
