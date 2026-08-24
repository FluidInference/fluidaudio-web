# OPT-0009 — producer-fused one-time K rotation

## Hypothesis

Quad attention rotated every K scalar again for every 32-query tile. Across
the supplied 1,189×62 workload and 80 transformer blocks, that is about 60.390
billion scalar K rotations. Rotating K once when the packed QKV projection
stores it requires 3.019 billion, removing 57.370 billion evaluations (95%).

The shader-level savings are roughly 172 GFLOPs, 121 GB of duplicate QKV
reads, and 459 GB of RoPE-table reads. The table is highly cacheable, so those
traffic totals are logical rather than DRAM estimates.

## Numerical contract

The former boundary first stored the raw projection as f16. Attention then
loaded each f16 pair into f32, applied RoPE, and rounded the rotated K value to
f16 workgroup storage. The fused producer preserves that ordering exactly:

1. finish the source-order f32 GEMM and bias;
2. explicitly round the complete QKV projection vector to f16;
3. convert only the rounded K pair back to f32, apply the same table expression,
   and round to f16 again; and
4. store Q and V with only the original projection rounding.

Time-axis rows derive position as `row / sequences`; contiguous frequency rows
use `row % tokens`. Q still rotates in attention, while the pre-rotated-K
attention variant compiles its hot K path down to a direct f16 load.

## Kernel isolation

A first arm put a uniform K-range branch and the rotary helper in the one QKV
shader dispatched over all six N256 tiles. It was exact but regressed the full
cold run to 47,729.4 ms end-to-end; the immediately following unfused control
was 44,811.0 ms. The larger store path penalized the dominant packed GEMM even
for Q and V.

The retained arm dispatches three compile-time-specialized two-tile segments:

- lean packed GEMM for Q;
- packed GEMM plus exact f16-boundary K rotation for K; and
- lean packed GEMM for V.

This adds two dispatches per transformer block (160 total) but keeps rotary
code and register pressure out of four of the six projection tiles.

## Correctness gates

The browser probe puts a real packed N256 QKV producer before attention. It
compares the old attention-side rotation with producer-fused K rotation for
both contiguous and physically strided layouts. Both comparisons have zero
mismatches across 56,832 raw f16 context words. The complete WAV retained every
stem and diagnostic statistic exactly, with no GPU, console, or page errors.

## Full-graph evidence

Isolated Chrome 151, zero warmups, one measured run per adjacent arm:

| Boundary | Unfused control (ms) | Isolated K tiles (ms) | Speedup |
| --- | ---: | ---: | ---: |
| deterministic | 10,227.7 | 10,354.2 | 0.99× |
| four CD refinements | 31,847.9 | 29,112.9 | 1.09× |
| complete model timing | 44,033.6 | 41,620.7 | 1.06× |
| page end-to-end timing | 44,811.0 | 42,364.0 | 1.06× |

A second retained-arm run completed in 41,478.6 ms end-to-end and 40,742.7 ms
total model timing. These are cold acceptance samples, not a new sustained
thermal benchmark.

## Sustained integrated benchmark

The final retained stack used the release protocol: isolated Chrome 151, one
warmup, and three measured runs. End-to-end samples were 43,004.3, 53,665.1,
and 56,903.4 ms, for a **53,665.1 ms median** (range
43,004.3–56,903.4 ms). Against the original 136,560 ms median
(134,850–146,190 ms), the delivered stack is **2.54× faster** with a **60.7%
wall-time reduction**. It is also 2.8% below OPT-0008's 55,192.5 ms sustained
median. The broad range remains a real thermal characteristic of the tested
Apple GPU and is not hidden by the aggregate.

## Disposition

Positive and integrated in its isolated-tile form. The monolithic producer
variant is closed; any future fusion into a dominant GEMM must keep unrelated
tiles on the lean shader path.
