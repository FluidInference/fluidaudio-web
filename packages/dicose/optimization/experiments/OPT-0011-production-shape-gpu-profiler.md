# OPT-0011 — production-shape GPU timestamp profiler

## Why the aggregate was misleading

The former 0.63 TFLOP/s figure divided 32.1 trillion logical multiply-add
FLOPs by complete graph wall time. That denominator includes online softmax,
transcendentals, workgroup barriers, elementwise kernels, command encoding,
queue gaps, readback, and CPU DSP. It was therefore not comparable with the
isolated GEMM calibrations reported by Parakeet or ACE-Step.

The dedicated browser profiler requests `timestamp-query` and wraps exactly
one production-shape compute pass per command buffer. Each of seven measured
samples gets its own submission and drain. Pipeline compilation, buffer
allocation/upload, submission, and timestamp readback are outside the GPU
interval. Zero or reversed timestamps fail the run instead of becoming a
plausible-looking result.

Reproduce with:

```sh
pnpm profile:webgpu
```

## Baseline localization

The first Chrome 151 profile, before the retained OPT-0012/0013 kernels,
measured:

| Kernel | GPU median | Effective throughput |
| --- | ---: | ---: |
| plain 73,718×384×1,536 dense | 64.03 ms | 1.358 TFLOP/s |
| FF1 73,718×384×1,536 + GELU | 63.64 ms | 1.367 TFLOP/s |
| FF2 73,718×1,536×384 + residual | 54.39 ms | 1.599 TFLOP/s |
| output 73,718×512×384 + residual | 17.96 ms | 1.614 TFLOP/s |
| time attention, 62×1,189 | 582.94 ms | 0.308 logical TFLOP/s |
| frequency attention, 1,189×62 | 31.85 ms | 0.294 logical TFLOP/s |

The dense kernels were already in ACE's exact-FP32 range rather than running
at 0.63 TFLOP/s. Forty blocks of each attention axis predict 24.59 seconds of
GPU work, which accounts for most of the unexplained end-to-end wall time.
The key fact is not the attention "TFLOP/s" itself—softmax work is absent from
that numerator—but that attention, not GEMM, owned the largest wall slice.

## Disposition

Integrated as `pnpm profile:webgpu`. The profiler retains explicit controls
for the old and selected owners so future compiler/browser changes can be
measured without loading the 623 MB model package.
