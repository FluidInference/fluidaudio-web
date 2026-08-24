# OPT-0010 — CD condition fusion into FF2

## Hypothesis

Each of four CD evaluations adds one time and one frequency condition after
each of eight layers. Fusing those 64 elementwise passes into FF2's existing
projection/residual store would remove 64 dispatches, bind groups, and uniform
writes. Across 73,718×384 f16 elements, the standalone adds represent 10.870
GB of logical traffic; condition reads remain necessary, so the nominal
eliminated destination round trip is 7.247 GB (6.749 GiB).

## Exactness finding

This boundary contains two semantically observable f16 roundings:

1. `projection16 = f16(GEMM + bias)`
2. `residual16 = f16(f32(projection16) + f32(residual))`
3. `output16 = f16(f32(residual16) + f32(condition))`

A first fused epilogue expressed `residual16` as a local `vec4<f16>`. Its
coarse random probe passed, but the full model changed slightly because the
Metal compiler reassociated the local expression across the nominal f16
boundary. An adversarial probe lane with residual and condition both 0.0006
exposed one raw-word mismatch: the correct double-rounded result was `0xb590`,
while the collapsed expression produced `0xb58f`.

Materializing the intermediate in an 8 KiB `var<workgroup>` f16 stage, crossing
a workgroup barrier, then applying the condition made the adversarial 896-word
probe and the complete WAV bit-identical. Merely spelling a local as f16 is
therefore not a sufficient numerical boundary on this backend.

## Full-graph evidence

The exact staged arm completed cold full-WAV runs in 41,276.2 and 42,224.3 ms
end-to-end. The retained unfused OPT-0009 path completed comparable runs in
41,478.6 and 42,364.0 ms. The distributions overlap: the apparent 0.3–0.5%
advantage is below run-to-run thermal variance and does not justify adding
workgroup storage and a barrier to a dominant FF2 kernel.

## Disposition

Neutral and reverted. The 64 standalone condition adds remain. Revisit only
with an independently useful FF2 owner that can guarantee the intermediate
rounding without reducing occupancy; do not retry local f16 spelling as a
barrier.
