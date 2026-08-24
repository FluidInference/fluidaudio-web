# OPT-0004 — persistent GPU RoPE table

## Hypothesis

The quad-query attention owner still evaluated `pow`, `sin`, and `cos` while
loading every K dimension. On the 1,189-token axis, each key is revisited by
38 query workgroups, across 62 sequences, eight heads, 16 deterministic
attention blocks, and 64 CD attention blocks. The trigonometric result depends
only on `(position, dimension)`, not the layer, stem, sequence, head, or QKV
values. Materializing it once should remove repeated transcendental work with
negligible storage.

## Mechanism

The first attention invocation dispatches a GPU kernel that writes one
`vec2<f32>(cos, sin)` per position/pair. The longest table is about 304 KiB and
persists across deterministic and CD graphs. Quad attention reads the table;
the query8 reference retains direct transcendental evaluation. Creating the
table on the GPU preserves the shader's f32 numerical path and avoids relying
on host trigonometric implementations.

## Correctness

`pnpm test:webgpu` compares the direct-trig query8 reference with table-backed
quad attention over 13,312 raw f16 words: zero mismatches. The full supplied-WAV
acceptance run passed with no browser/GPU errors, and every reported stem and
diagnostic statistic was identical to OPT-0003.

## Full-graph evidence

Isolated Chrome 151, zero warmups, one measured cold run per arm:

| Boundary | OPT-0003 (ms) | RoPE table (ms) | Speedup |
| --- | ---: | ---: | ---: |
| deterministic | 12,013.8 | 10,828.5 | 1.11× |
| four CD refinements | 35,086.8 | 32,383.4 | 1.08× |
| complete model timing | 49,283.0 | 45,206.9 | 1.09× |
| page end-to-end timing | 50,064.7 | 46,084.7 | 1.09× |

This is cold acceptance evidence rather than a sustained thermal benchmark.

## Rejected arm

A second arm rotated every K vector once in-place in the QKV buffer, preserving
the exact f16 K boundary and eliminating repeated rotation arithmetic as well
as transcendental evaluation. It remained bit-identical but added a roughly
75 MB read/write pass to every attention block. Full end-to-end wall regressed
to 48,914.5 ms versus the 46,084.7 ms table arm, so it was removed.

## Disposition

The table-only arm is positive and integrated. The in-place K-hoist is
abandoned; removing arithmetic without accounting for activation traffic was
not a win at the full boundary.
