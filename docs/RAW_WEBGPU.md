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
| `conv1d(x, w, {…})` | one thread / (Cout, Lout) | stride / pad / dilation / groups; regular + depthwise |
| `convTranspose1d(x, w, {…})` | one thread / (Cout, Lout), gather form | iSTFTNet upsampler + iSTFT overlap-add; groups |
| `lstm(x, w, r, b, hid)` | 1 workgroup/direction, hidden units as threads | **bidirectional**, ONNX `iofc` gates, timesteps in-kernel (H ≤ 256) |
| `layernorm(x, γ, β)` | 1 workgroup/row, 64-lane reduce | row-wise |
| `adain(x, scale, shift)` | 1 workgroup/channel, 64-lane reduce | instance-norm over time + style affine (StyleTTS2 decoder) |
| `softmax(x)` | row-wise, numerically stable | for attention |
| `add` / `mul` / `leakyRelu` | elementwise | residuals, gating, iSTFTNet activation |
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

## Milestone: LSTM + ConvTranspose1d — parity vs Kokoro ONNX

The two hard primitives for the prosody/duration predictors and the iSTFTNet
decoder, both verified against the **real Kokoro weights** on the M5 Pro:

| op | target node | shape | result |
|---|---|---|---|
| bidirectional LSTM | `predictor/lstm` | inp 640, hid 256, bidir | **rel 4.6e-7** (`gpu:lstm`) |
| ConvTranspose1d | `generator/ups.0` | 512→256, L 94→940, K 20, stride 10 | **rel 4.7e-7** (`gpu:convt`) |

- **LSTM** matches the ONNX op exactly: gate order `iofc`, no peephole, one workgroup
  per direction, hidden units as threads, timesteps looped in-kernel with `h`/`c` in
  workgroup memory. All six Kokoro LSTMs are bidirectional/hidden-256, so this one
  kernel covers them.
- **ConvTranspose1d** is the iSTFTNet upsampler *and* the iSTFT overlap-add. Kokoro's
  iSTFT is a Cos/Sin **DFT matmul + ConvTranspose** — both now verified — so the
  vocoder's spectral tail composes from existing kernels; no separate FFT needed
  (n_fft is small here).

Reproduce: `kokoro-ref-lstm.py` / `kokoro-ref-convt.py` (extract weights + expose
the node's input/output as graph outputs for an ORT reference) → `npm run gpu:lstm`
/ `npm run gpu:convt`.

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
3. ✅ **bidirectional LSTM** — duration/prosody predictors + text encoder (`gpu:lstm`).
4. ✅ **ConvTranspose1d** — iSTFTNet upsampler + iSTFT overlap-add (`gpu:convt`).
5. ✅ **AdaIN** — instance-norm over time + style affine (`adain`).
6. ✅ **LeakyReLU** — iSTFTNet activation (`leakyRelu`).
7. **length regulator** — expand text features by predicted durations (a gather).
8. **harmonic+noise source** for the generator (sine gen + the STFT is matmul/conv).
9. **embedding gather** — currently CPU (a lookup); move to a kernel if it matters.

Every compute-heavy primitive is done and parity-clean; the full op set already
runs (`gpu:kokoro-forward`, 271/274 ops). What's left to hand-wire an end-to-end
audio path is the length regulator (a gather) + source gen — plus the fused-conv /
fp16 optimization pass that would turn the ~10× tie into a win.

## Ship/no-ship: where the Kokoro time actually goes

Before hand-wiring the whole StyleTTS2 graph (a large multi-block build), profile
where the compute is (`kokoro-profile.py`, ORT CPU, 2.05 s audio):

| module | share | | op | share |
|---|--:|---|---|--:|
| **decoder (iSTFTNet)** | **89.7%** | | **Conv** | **64%** |
| bert (ALBERT) | 5.4% | | Sin (source) | 9.7% |
| predictor | 1.4% | | STFT | 5.7% |
| text_encoder | 0.7% | | ConvTranspose | 5.0% |

So the ALBERT encoder we ported perfectly is **~5%** of the work — the whole
question is the **vocoder convs** (~106 GFLOP of conv for 2 s audio, dominated by
resblock convs at `[128, 9841]` K11). `npm run gpu:kokoro-cost` measures raw-WebGPU
conv at that dominant shape on the M5 Pro:

| conv path | throughput | projected conv-only RTFx |
|---|--:|--:|
| direct (`conv1d`) | 252 GFLOP/s | **~4.9×** — loses to kokoro-js |
| **im2col + tiled GEMM** (`conv1dGemm`) | 876 GFLOP/s | **~17×** — beats kokoro-js (~10×) |

The direct `conv1d`/`convTranspose1d` kernels are kept for correctness/parity and
small/grouped convs; the hot vocoder convs route through `conv1dGemm`.

### Register-blocked GEMM (the perf lever)

The GEMM is now register-blocked (64×64 block, 4×4 micro-tile per thread): **927 →
2131 GFLOP/s**, and the dominant conv via `conv1dGemm` **876 → 1759 GFLOP/s** (that
one conv, in isolation, projects to ~34× RTFx).

### Full-forward measurement (the honest end-to-end number)

`npm run gpu:kokoro-forward` replays **all 274 real compute ops** (Conv /
ConvTranspose / MatMul / Gemm / LSTM, actual shapes from an ORT profile) back-to-
back in a **single submit**, timing submit→GPU-finish (excludes CPU alloc/record):

| | RTFx |
|---|--:|
| raw-WebGPU, all compute ops (M5 Pro, dawn) | **~10×** |
| ORT CPU, same ops | ~9× |
| kokoro-js (ORT WebGPU, browser) | ~10× |

**Verdict: with correct, register-blocked kernels raw WebGPU *matches* kokoro-js
(~10×) — it does not yet clearly beat it.** The 34× single-conv microbench doesn't
carry to the whole pipeline: `im2col` trades compute for memory (it materializes a
big patch matrix), and at Kokoro's near-audio-rate lengths (`L ≈ 9841`) the many
convs are memory-bound in aggregate. Beating ORT decisively needs a **fused direct
conv** (no im2col materialization) and/or **fp16** (`shader-f16` is available) — a
real optimization pass, not just correct kernels.

Two hard-won measurement lessons:
- **Denormals cost ~2×.** Replaying with uninitialized (garbage) buffers ran at
  389 ms; zero-initialized, the *same* ops ran at 202 ms. Flush-to-zero / clean
  inputs matter enormously on Metal.
- **Microbench ≠ pipeline.** One hot conv at 34× told a rosier story than the full
  op set at ~10×. Always measure the aggregate.

So the load-bearing question — *is raw-WebGPU Kokoro worth building over
kokoro-js?* — answers: **not for raw speed alone today** (it's a tie); it's worth
it for a smaller/ORT-free bundle, or as the vehicle for the models where ORT is
*blocked* (Nemotron int4, Parakeet int8-collapse). A fused-conv + fp16 pass is what
would turn the tie into a win.

Approach: extract each layer's weights + a reference intermediate from the Kokoro
ONNX (via `onnxruntime-node`), port the layer to WGSL, gate on parity, then chain
GPU-resident. Only when the whole graph is resident + parity-clean do we benchmark
against the current `kokoro-js` (ORT) path — that's the number that decides whether
raw WebGPU is worth shipping for Kokoro.
