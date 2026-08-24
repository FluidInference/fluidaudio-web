# OPT-0003 — converter-native packed subgroup GEMM

## First-principles attribution

The supplied 1,189-frame fixture creates 73,718 transformer rows. Static graph
accounting attributes approximately 32.1 logical TFLOPs to one separation:

- 23.23 TFLOPs (72.3%) are transformer dense projections;
- 7.55 TFLOPs (23.5%) are attention; and
- the repeated 384↔1536 projections alone account for about 20.9 TFLOPs.

The prior 16×16 shared-tile GEMM therefore owned the largest credible absolute
saving. The failed OPT-0001 changed subgroup ownership but retained row-major
weights. This experiment instead treats layout and kernel ownership as one
mechanism, following the successful ACE-Step/Parakeet pattern.

## Mechanism

The converter now stores eligible matrices as `[N tile, K tile, K32, N]`,
choosing N256 where possible and N128 for shapes such as N384. The production
kernel uses WG128 with four fixed-32 subgroups. Each subgroup owns eight rows;
each lane owns one N128 or two N256 `vec4` columns. Converter-native weights are
read directly without a workgroup panel or barriers and reused across the
eight rows through subgroup broadcasts.

Operands remain f16. Every contraction visits K in increasing source order,
uses f32 FMA state, adds f16 bias after the contraction, applies the existing
post-op, and rounds once to f16 output. Small or incompatible shapes retain the
generic kernel. The package size is unchanged; 799 of 2,857 logical tensors,
representing 569,180,160 payload bytes, select a packed layout.

## Correctness gates

- `pnpm test:webgpu` compares a 7×32×128 packed GELU projection against the
  generic kernel over all 896 raw f16 output words: zero mismatches.
- `pnpm check`, `pnpm test`, and `pnpm verify:package` pass. The package has
  2,857 logical tensors, 2,829 unique payloads, and 623,246,848 bytes.
- `pnpm test:browser` passes the supplied-WAV f16 acceptance envelope with no
  validation, device, console, or page errors.

## Full-graph evidence

Both samples used isolated Chrome 151 profiles, zero warmups, and one measured
run on the same machine. They are cold acceptance evidence, not a sustained
thermal benchmark.

| Boundary | Pre-change (ms) | Packed GEMM (ms) | Speedup |
| --- | ---: | ---: | ---: |
| deterministic | 19,370.6 | 12,013.8 | 1.61× |
| four CD refinements | 59,820.3 | 35,086.8 | 1.70× |
| complete model timing | 81,286.9 | 49,283.0 | 1.65× |
| page end-to-end timing | 82,260.4 | 50,064.7 | 1.64× |

End-to-end wall fell 39.1%. This result is not compared directly with the
136.56-second sustained OPT-0002 median because thermal cadence differs.

## Disposition

Positive and integrated. Dense math remains the largest counted family, but
the next experiment must target a distinct mechanism rather than nearby tile
geometry. Attention's repeatedly recomputed rotary transcendental work is the
next structural candidate.
