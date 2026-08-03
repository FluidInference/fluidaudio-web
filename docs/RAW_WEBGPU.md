# Raw WebGPU compute path

> "Combination of WebGPU and WASM for the best speed. WebGPU for all ML model
> parts, WASM for CPU-heavy things, JS/TS for the rest." — the goal here.

`onnxruntime-web` is the pragmatic baseline for every engine in this repo, but its
WebGPU EP has real ceilings: no fusion (each op is its own dispatch), GPU↔CPU
syncs on unsupported/dynamic ops, and no int8/int4 kernels. For a hot model the
faster path is **hand-written WGSL over GPU-resident tensors** — fused kernels,
nothing leaves the GPU between ops. `src/gpu/compute.js` is that foundation,
started with Kokoro TTS as the target.

## What's here (verified on a real M5 Pro GPU)

`GpuContext` (`src/gpu/compute.js`) — pass a `GPUDevice` (`navigator.gpu` in the
browser, [dawn](https://github.com/kmamal/gpu) in Node). Tensors are GPU-resident
`{ buf, rows, cols }`; only `download()` copies back to CPU.

| Kernel | WGSL | Notes |
|---|---|---|
| `matmul(a, b, {bias, act})` | tiled 16×16, shared-memory | **fused** bias + activation (none/gelu/tanh/relu) in one dispatch |
| `layernorm(x, γ, β)` | 1 workgroup/row, 64-lane reduce | row-wise |
| `softmax(x)` | row-wise, numerically stable | for attention |
| `add` / `mul` | elementwise, row-broadcast | residuals, gating |

Every kernel is parity-checked against a CPU reference **on the real GPU**:

```bash
npm run gpu:verify   # ✓ all kernels: max|gpu-cpu| ≤ tol
npm run gpu:bench    # GEMM GFLOP/s + a GPU-resident FFN block
```

Latest run (M5 Pro, dawn):

```
matmul+bias+gelu  max|gpu-cpu| = 2.7e-5      GEMM 512³        532 GFLOP/s
layernorm         max|gpu-cpu| = 2.4e-7      GEMM 512×512×2048 927 GFLOP/s
softmax           max|gpu-cpu| = 7.5e-9      resident FFN block (200×512, d_ff 2048)
add / mul         exact                        = 1.25 ms/block, 1 readback total
```

The FFN block — `layernorm(x + W2·gelu(W1·x + b1) + b2)`, the ALBERT-encoder core —
chains 4 kernels with **every intermediate resident on the GPU and a single
readback**. That's the fusion+residency win ORT's per-op path can't match; a
6-layer encoder is ~7.5 ms of FFN compute.

## Gotchas already hit

- **Metal `tanh` overflows.** It computes `exp(x)` directly, so a large argument →
  `Inf/Inf = NaN` (CPU `Math.tanh` saturates). The gelu kernel clamps the tanh
  argument to ±20 (already ±1 to f32 precision). Small test matrices hid this;
  K=512 accumulators (~±40) surfaced it.
- **Flag constants aren't global in Node.** `GPUBufferUsage`/`GPUMapMode` are
  ambient in the browser; `scripts/gpu-globals.mjs` registers them on `globalThis`
  so the *same* kernel code runs under dawn. That's what makes headless parity
  testing possible (Chrome has no `navigator.gpu` in the automation env).
- **Naive tiled GEMM ≈ 1 TFLOP/s** (~10% of the M5 Pro's fp32 peak). Fine for
  correctness + the residency demo; a register-blocked / fp16 (`shader-f16` is
  available) kernel is the next perf step.

## Path to a raw-WebGPU Kokoro

Kokoro 82M (StyleTTS2 + iSTFTNet) is ~7 subnets. The transformer kernels above
cover the ALBERT text encoder. Remaining kernels, each verifiable through the same
`gpu:verify` harness before wiring:

1. **embedding gather** (phoneme ids → vectors) — trivial.
2. **conv1d** (+ dilation, groups) — prosody predictor & decoder; the biggest new
   kernel.
3. **LSTM cell** — duration/prosody predictors.
4. **iSTFT** (+ upsampling) — the vocoder tail; FFT kernel.
5. **AdaIN / style modulation** — cheap elementwise once style is resident.

Approach: extract each layer's weights + a reference intermediate from the Kokoro
ONNX (via `onnxruntime-node`), port the layer to WGSL, gate on parity, then chain
GPU-resident. Only when the whole graph is resident + parity-clean do we benchmark
against the current `kokoro-js` (ORT) path — that's the number that decides whether
raw WebGPU is worth shipping for Kokoro.
