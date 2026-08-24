# OPT-0016 — eight-row subgroup RMSNorm

## First-principles basis

The original RMSNorm assigned a complete WG256 to one row. At the main
73,718-row transformer shape this launched 73,718 workgroups per call even
though each row has only 384 values. Across 165 transformer norms that is over
12 million small workgroups, two workgroup barriers per row, and shared-memory
traffic solely to combine eight fixed-32 subgroup partials.

The candidate assigns one row to each of the eight subgroups in WG256. To keep
the target-browser arithmetic contract, every subgroup emulates the original
eight lane partitions in ascending order: it forms the same per-lane FP32 FMA
chains, performs eight subgroup reductions, and folds those results in the same
slot order. No workgroup storage or barrier remains. One workgroup owns eight
rows, reducing the production C384 dispatch from 73,718 to 9,215 workgroups.

## Correctness gate

Chrome 151 compared the old row owner and the new eight-row owner over all
production widths `C={8,16,48,96,192,384,512,516}`, row tails `1..9` and `17`,
FiLM off/on at C384, dynamic-range-stressing finite f16 inputs, and a forced
small workgroup width that exercised 2-D dispatch flattening. All 133,672
candidate f16 words across 90 comparisons matched bit-for-bit.

Reproduce with:

```sh
pnpm test:webgpu
```

## Production-shape timing

The GPU timestamp profile uses two warmups and seven measured submissions at
73,718 × 384. Compilation, upload, submission, and readback are excluded.

| Shape | Row1 median | Rows8 median | Speedup | Logical bandwidth |
| --- | ---: | ---: | ---: | ---: |
| plain | 2.228224 ms | 1.310720 ms | 1.7000× | 172.78 GB/s |
| FiLM mapped | 2.097152 ms | 1.376256 ms | 1.5238× | 246.82 GB/s |

The 37 plain and 128 mapped transformer calls project about 126 ms of saving.
Band-split norms add a smaller benefit not included in that projection. This
is a valid exact structural cleanup, but the absolute result also proves that
RMSNorm is not the route to multi-second improvement; dense contraction
remains the next priority.

Reproduce with:

```sh
DICOSE_PROFILE_FOCUS=norm pnpm profile:webgpu
```

## Disposition

Integrated. The old row owner remains an explicit profiler/probe control.
