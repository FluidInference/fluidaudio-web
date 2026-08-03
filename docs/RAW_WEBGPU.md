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
| `conv1d(x, w, {…})` | one thread / (Cout, Lout) | stride / pad / dilation / groups; covers regular + depthwise (prosody predictor + vocoder) |
| `layernorm(x, γ, β)` | 1 workgroup/row, 64-lane reduce | row-wise |
| `softmax(x)` | row-wise, numerically stable | for attention |
| `add` / `mul` | elementwise, row-broadcast | residuals, gating |
| `transpose` / `sliceCols` / `setCols` | index kernels | multi-head attention plumbing |

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

## Milestone: ALBERT text encoder — end-to-end parity vs ONNX

The first real Kokoro sub-network is ported and **numerically matches the ONNX
model** (`src/gpu/albert.js`, `npm run gpu:albert`). Kokoro's PL-BERT: vocab 178,
embed 128, hidden 768, FFN 2048, **12 weight-shared layers**, 12 heads (head_dim
64), `gelu_new`, LN eps 1e-12. Embeddings (gather + sum + LN) run on CPU; the whole
transformer stack — QKV projections, 12-head scaled-dot-product attention, output
projection, FFN, residual LayerNorms — runs **GPU-resident** on the kernels above.

Verified against the real Kokoro weights + an onnxruntime reference (input_ids →
ALBERT output), on the M5 Pro GPU:

```
ALBERT input  (embeds + 128→768 map): rel 1.5e-7
ALBERT output (12 layers)           : max 1.3e-5   rel 3.4e-6   ← exact, fp32
```

Reproduce: `kokoro-extract-albert.py` (trace ALBERT weights out of the ONNX — the
Linear weights are anonymous `onnx::MatMul_*` initializers, found via the named
bias each feeds) + `kokoro-ref-albert.py` (expose the ALBERT in/out tensors as
graph outputs, run ORT for ground truth) → `npm run gpu:albert`. The attention
scale `1/√64` is folded into the query projection so no extra kernel is needed.

Gotcha: loading the weight `.bin`s in Node — `readFileSync().buffer` is a view into
a shared pool, so slicing it grabs neighbouring garbage → NaN. Copy the exact byte
range (`Uint8Array.from(buf)`).

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

Kokoro 82M (StyleTTS2 + iSTFTNet) is ~7 subnets. Done so far: the **ALBERT text
encoder** (parity above) and **conv1d** (regular + depthwise). Remaining, each
verifiable through the same harness before wiring:

1. ✅ **ALBERT text encoder** — parity vs ONNX (`gpu:albert`).
2. ✅ **conv1d** (+ dilation, groups) — prosody predictor & decoder.
3. **LSTM cell** — duration/prosody predictors.
4. **iSTFT** (+ upsampling) — the vocoder tail; FFT kernel.
5. **AdaIN / style modulation** — cheap elementwise once style is resident.
6. **embedding gather** — currently CPU (a lookup); move to a kernel if it matters.

Next up is the prosody/duration predictor (LSTM + the conv1d now in place), then
the iSTFTNet decoder (the FLOP-heavy tail). Only once the whole graph is resident +
parity-clean do we benchmark against `kokoro-js` — that number decides ship/no-ship.

Approach: extract each layer's weights + a reference intermediate from the Kokoro
ONNX (via `onnxruntime-node`), port the layer to WGSL, gate on parity, then chain
GPU-resident. Only when the whole graph is resident + parity-clean do we benchmark
against the current `kokoro-js` (ORT) path — that's the number that decides whether
raw WebGPU is worth shipping for Kokoro.
