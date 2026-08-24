# OPT-0014 — Subgroup-owned STFT adapter convolutions

## Hypothesis

The STFT adapter applies its convolutions to a 1,025×1,189 grid. The generic
shader assigns one invocation to one output pixel/channel and dispatches the
channel dimension as workgroups, producing 2,460,288 tiny workgroups for both
the 4→128 entry convolution and each 128→128 hidden convolution. This is a
work-ownership failure, not an arithmetic-throughput limit.

The selected owners give each subgroup four adjacent output channels and
eight pixel rows. Four subgroups therefore produce a 32×128 output tile per
workgroup. Weights are staged once per workgroup and activations are broadcast
within each subgroup. The production dispatch falls to 38,086 workgroups,
64.6× fewer, while retaining bias-first FP32 accumulation in source order.

## Production-shape evidence

A matched Chrome 151 timestamp profile used two warmups and seven measured
passes. Compilation, upload, submission, and readback were excluded.

| Boundary | Generic | Selected | Speedup |
| --- | ---: | ---: | ---: |
| 4→128 3×3 entry | 171.442 ms / 0.065 TFLOP/s | 14.942 ms / 0.751 TFLOP/s | 11.5× |
| 128→128 1×1 hidden | 763.167 ms / 0.052 TFLOP/s | 24.969 ms / 1.599 TFLOP/s | 30.6× |

The hidden shape occurs twice. The matched medians predict 1,632.9 ms less
GPU time for the retained entry and hidden owners together. The remaining
128→4 exit 3×3 measured 121.307 ms in the generic shader, so it was left alone
rather than spending effort on a sub-0.13-second ceiling.

## Correctness and integrated acceptance

The raw browser probe compares generic and selected outputs using a 5×7 grid,
which covers all padding edges and corners plus a partial final 32-row tile.
It reports zero mismatches across 4,480 raw f16 output words. The 128×128 1×1
owner likewise reports zero mismatches across 4,736 words.

The complete supplied WAV then passed in 30,458.8 ms end-to-end, with
29,714.2 ms model timing: 6,079.3 ms deterministic, 21,498.3 ms refinement,
and 1,474.7 ms ISTFT. Every final-stem and model-diagnostic peak/RMS statistic
was unchanged. This is a cold acceptance receipt, not a sustained thermal
median.

On the same cool timestamp run, the exact packed dense kernels reached
2.05–2.13 TFLOP/s. This reinforces that the former 0.63 aggregate figure was
whole-graph logical work divided by wall time, not the machine's dense-kernel
ceiling.

## Disposition

Integrated for the exact production 4→128 3×3 and 128→128 1×1 geometries.
The generic convolution remains for the low-impact 128→4 exit and other
shapes.
