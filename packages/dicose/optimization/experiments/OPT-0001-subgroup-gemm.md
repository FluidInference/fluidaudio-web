# OPT-0001 — 128-column subgroup GEMM

## Hypothesis

Using four 32-lane subgroups to emit a 32×128 output tile directly from
row-major f16 weights would reduce workgroup count and beat the generic 16×16
shared-memory GEMM for the model's 384/512/1536 projections.

## Correctness gate

The isolated Chrome raw-WGSL probe checked a nontrivial 4×128 projection and
passed with no validation or uncaptured-device errors. A full fixture pass
produced bit-identical reported output statistics.

## Result

The full clean-profile run regressed from 104,924.6 ms to 147,497.9 ms model
time. Deterministic inference alone rose from 22,003.0 ms to 58,884.2 ms.
The extra register pressure/subgroup broadcast ownership lost to the original
16×16 tiled kernel on the target Metal adapter.

## Disposition

Negative. The implementation and its special raw probe were removed; generic
linear remains the production path.
