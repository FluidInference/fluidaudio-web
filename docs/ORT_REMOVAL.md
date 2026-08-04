# Full ONNX-runtime removal — raw WebGPU/wasm rewrite

Goal: remove `onnxruntime-web`, `@huggingface/transformers` (Whisper), and
`kokoro-js` (Kokoro) entirely, reimplementing every engine's inference as a
hand-written forward pass over the raw-WebGPU kernels in `src/gpu/compute.js`
(device-agnostic: browser `navigator.gpu` + headless `@kmamal/gpu` for parity).

Rationale (measured, see RAW_WEBGPU.md): generic raw kernels *tie* ORT-WebGPU on
thin-GEMM speech models (portable-WGSL ceiling ~16% of peak; `simdgroup` matrix
units unreachable). The wins are (a) capability — int4 (Nemotron) can't run on
ORT-WebGPU at all; (b) bundle size — drop 3 runtimes; (c) per-model fusion, the
only lever that can beat ORT on speed, which a hand-written forward enables and a
generic graph interpreter does not. So: per-model hand-written forwards, NOT a
generic ONNX interpreter.

## Rule: parity before deletion
For each engine, build the raw forward, parity-gate it vs the CURRENT ORT output
(headless, `@kmamal/gpu`) to a documented tolerance, wire it into the engine, and
only THEN delete the ORT path for that engine. Never delete ORT for an engine
that isn't parity-verified.

## Kernel coverage (compute.js, all parity-gated)
Have: matmul(+F16), conv1d/conv1dFast(+F16)/conv1dGemm, im2col, convTranspose1d,
lstm, layernorm, softmax, transpose, sliceCols/setCols, gatherCols, adain,
leakyRelu, ewise(add/mul/...), matmulNBits (int4 dequant).
Gaps to add: SiLU/Swish (ewise: sigmoid(x)*x), LogSoftmax (softmax variant),
CumSum/scan (Kokoro NSF), Resize (Kokoro), ScatterND (Kokoro iSTFT overlap-add).
Reduction helpers (ReduceMean/Pow/Sqrt) — small kernel or host for tiny tensors.
Graph plumbing (Reshape/Unsqueeze/Gather/Shape/Concat/Slice/Cast) is NOT kernels
— it becomes JS control flow in the hand-written forward.

## Model order (easiest → hardest)
1. **Silero VAD** — [IN PROGRESS] proves the full pattern. Arch recovered:
   input[1,512]→reflectPad→STFT(Conv k256 s128, basis[258,1,256])→|mag|(Pow2/Add
   /Sqrt→[1,129,T])→enc0 Conv[128,129,3]p1s1+Relu→enc1[64,128,3]p1s2+Relu→enc2
   [64,64,3]p1s2+Relu→enc3[128,64,3]p1s1+Relu→LSTM(in128 hid128, state[2,1,128])
   →decoder Conv[1,128,1]→Sigmoid. Weights: encoder.{0..3}.reparam_conv.{weight,
   bias}, LSTM W/R[1,512,128] b[1,1024], decoder.decoder.2.{weight[1,128,1],bias},
   stft.forward_basis_buffer[258,1,256]. Source ONNX: /tmp/silero_vad.onnx.
2. **Parakeet v3 encoder** — [IN PROGRESS] NeMo FastConformer, fp32 encoder
   /tmp/pkv3/encoder-model.onnx(+.data 2.4GB). d_model=1024, 24 layers, 8 heads×128,
   d_ff=4096. Input audio_signal[1,128,T] mel → output [1,1024,T/8]. GPU-resident
   (compute.js), unlike Silero. Anchoring for parity: added /pre_encode/out/
   Add_output_0 (conformer-stack input) as an extra output → /tmp/pkv3/
   enc_anchored.onnx (save with load_external_data=False, else it embeds 2.4GB >2GB
   protobuf limit and won't load).
   - **Subsampling (dw_striding 8×)** [DONE, numpy parity 6.6e-3 abs ≈1e-4 rel]:
     mel[1,128,T]→transpose[1,T,128]→unsqueeze[1,1,T,128] → Conv2d(1→256,3×3,s2,
     pad1)+ReLU → dwConv2d(256,3×3,s2,g256)→ptConv2d(256→256,1×1)+ReLU → dwConv2d+
     ptConv2d+ReLU → [1,256,T/8,16] → transpose[0,2,1,3]→flatten[1,T/8,4096] →
     Linear(pre_encode.out W[4096,1024]+bias)→[1,T/8,1024]. Freq 128→64→32→16. The
     Add feeding the conformer input is [bias, matmul] (bias FIRST). NEEDS a conv2d
     kernel in compute.js (only conv1d exists) for the GPU port.
   - **Conformer block ×24** [NEXT — the crux]: each = FF1(½)+relposMHA+conv+FF2(½)
     +norm_out, macaron. FF: LN→Linear1024→4096→SiLU→Linear4096→1024, ×0.5 residual.
     relpos MHA (Transformer-XL style): pos_bias_u/v[8,128], q/k/v/out proj 1024²,
     matrix_ac=（q+u)@kᵀ, matrix_bd=rel_shift((q+v)@pᵀ), scores/√128→softmax→@v. Conv
     module: LN→pointwise_conv1[2048,1024,1]→GLU→depthwise conv→SiLU→pointwise_conv2
     [1024,1024,1], residual. PLAN: lift pos_emb + layer0 output from ORT to parity
     the attention math separately from the sinusoidal pos-enc generation.
   Once done, 3/4/5 reuse the encoder block. Decoder+joint: LSTM+MatMul+Relu.
   Mel: replace onnxMel (nemo128 custom op) with the repo's pure-JS NeMo mel.
   WEIGHTS: fp16 ~1.2GB — CANNOT bundle; must host (revisit the VAD bundle choice).
3. **Nemotron / EOU / Sortformer** — reuse FastConformer. Nemotron encoder is
   int4 (matmulNBits) — the capability win. /tmp/nemo-fp16, /tmp/eou, /tmp/sf.
4. **Kokoro** — most raw already built (src/gpu/kokoro.js frontend done). Finish
   DurationEncoder→length-reg→F0/N→iSTFTNet decoder→generator (NSF: needs CumSum;
   iSTFT: ScatterND). /tmp/kokoro/model.onnx.
5. **Whisper** — hardest: encoder-decoder + autoregressive KV-cache decode +
   tokenizer (currently transformers.js does all of it). onnx-community/whisper-base.

## Status
- [x] 1. Silero VAD — DONE, ORT fully removed from this engine. Raw JS forward
  (src/engines/vad-silero/raw-silero.js), parity vs ORT: single-chunk Δ 2.5e-7,
  187-chunk streaming Δ 5.8e-7; end-to-end 78× RTFx pure JS. Weights BUNDLED
  (silero-weights.bin ~1.2 MB + .manifest.json, extracted 16k-scoped via
  scripts/extract-silero-weights.py). silero.js + index.ts no longer touch
  onnxruntime. scripts/smoke-silero-raw.mjs regression-gates parity. GOTCHA: the
  ONNX inlines 8k+16k branches with SEPARATE weights (not shared) — extract
  strictly from the 16k subgraph. Pure JS (not GPU): tiny model, per-chunk GPU
  dispatch would dominate. NOTE: bundling chosen for weights; does NOT scale to the
  big encoders (GBs) — revisit hosting for models 2–5.
- [ ] 2. Parakeet v3 encoder (+ decoder/joint + JS mel)
- [ ] 3. Nemotron / EOU / Sortformer
- [ ] 4. Kokoro
- [ ] 5. Whisper
- [ ] remove onnxruntime-web / @huggingface/transformers / kokoro-js from package.json
