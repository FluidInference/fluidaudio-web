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
2. **Parakeet v3** — [DONE: ORT-FREE, SHIPPED, 26.4× RTFx in-browser on a 4.7-min
   clip, correct transcript.] Architecture: JS mel → int8 GPU FastConformer encoder
   (raw-encoder.js) → WASM-SIMD CPU decoder (rust/parakeet-decoder → wasm32+simd128,
   raw-decoder-wasm.js). Perf journey 12.4→26.4×: (a) decoder was the wall — GPU
   joint hits the RNNT per-token GPU-sync wall (~101ms/roundtrip in dawn; ~20× in
   browser), so moved it to WASM-SIMD CPU (no GPU sync, 3× over JS); (b) GPU-resident
   encoder (subReshape kernel, no per-window download except the frames handoff);
   (c) preload pos_bias + cache pos-enc (removed ~7300 tiny uploads — NO effect, so
   encoder is FLOP/occupancy-bound not upload-bound); (d) PIPELINE GPU-encode(i+1)
   with CPU-decode(i) → decode hides behind encode (13.5→10.8s). BOTTLENECK now =
   encoder ~9s, small-M (Tsub≈187) GEMM occupancy at the WGSL ceiling (~0.5 TFLOP/s).
   Gap to ORT (47×) = that GEMM occupancy; levers = batched attention heads / small-M
   GEMM kernel (uncertain) or bigger windows (user declined). Weights: HF parakeet/
   encoder-int8.bin (612MB) + decoder-fp32.bin (72MB). MEASUREMENT LESSON: dawn
   headless has a ~101ms GPU-download sync wall → GPU-sync-heavy perf is unmeasurable
   headless; the user's browser is the only real benchmark; WASM/CPU IS measurable.
   [superseded lines below kept for detail]
   [FULL RAW PIPELINE WORKS, ORT-FREE, 13.7× RTFx, correct
   transcript. Remaining: engine wiring (index.ts) + mel frame-count reconcile.]
   Encoder: int8 GPU (raw-encoder.js, 5.3e-7 fp32 / int8 transcript-identical),
   HF parakeet/encoder-int8.bin (612MB). Decoder: JS (raw-decoder.js, embed+2×LSTM
   +joint), HF parakeet/decoder-fp32.bin (72MB). Mel: JS (parakeet-mel.js, reuses
   Nemotron mel + per-feature CMVN + 2^-24 guard) — transcript-correct but frame
   count off-by-one vs nemo128 (800 vs 801), reconcile for robustness. rel_shift
   GPU kernel (relShift) removed the 192-roundtrip bottleneck (20s→0.55s). Weights
   fetched from HF; wire ParakeetV3Engine with GpuContext + windowing (tdt.js) +
   delete ORT encoder/decoder/mel. NeMo FastConformer, fp32 encoder
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
     DECODED layer0 attn wiring (anonymous MatMul weights): q=onnx::MatMul_6400,
     k=6410, v=6411, linear_pos=6412 (all applied to the norm_self_att LN output,
     each reshaped→heads 8×128). pos_emb = the sliced positional table (graph tensor
     "Slice_output_0") feeding linear_pos; q+pos_bias_u / q+pos_bias_v are the Adds.
     Parity anchors to expose: /layers.0/norm_self_att/LayerNormalization_output_0
     (attn input), the pos_emb Slice, /layers.0/norm_out/LayerNormalization_output_0
     (layer0 out). TODO: out-proj weight name + rel_shift indexing + pos-enc gen
     (or lift pos_emb and defer generation).
   - **ALL SUB-MODULES VERIFIED numpy vs ORT (T=80 synthetic mel, layer 0):**
     subsampling ~1e-4 rel; FF (macaron, ×0.5, SiLU, NO bias — W1=MatMul_6398,
     W2=6399; FF2 W=6506/6507) 1.4e-6; rel-pos attn (q6400/k6410/v6411/pos6412/
     out6500, pos_bias_u/v, rel_shift, /√128) 4e-6; conv module (norm_conv LN →
     pointwise_conv1[2048,1024,1]→GLU(a·σ(b))→depthwise Conv6310[1024,1,9] g1024
     sym-pad4 +bias6311→SiLU(σ·mul)→pointwise_conv2[1024,1024,1]) 2.3e-6; FULL
     layer0 (FF1→attn→conv→FF2→norm_out) 3e-6. LayerNorm eps=1e-5. rel_shift:
     pad last dim left 1 → reshape[h,pos+1,t] → drop row0 → reshape[h,t,pos] →
     slice[:,:,:t]. pos-enc gen (NeMo RelPositionalEncoding, positions T-1..-(T-1),
     sin even/cos odd, div=exp(arange(0,d,2)·-ln(10000)/d)) matches lifted pe 4e-7.
     Encoder output = transpose(layer23 norm_out) → [1,1024,T/8].
   - **FULL 24-layer numpy encoder VERIFIED vs ORT final "outputs": 9.8e-7** (errors
     don't compound). Per-layer weights mapped via node.name /layers.L/<role>/ →
     weight input (FF linear1/linear2, self_attn/linear_{q,k,v,pos,out}, conv/
     {pointwise_conv1,depthwise_conv,pointwise_conv2}). No xscale on conformer input
     (pre_encode out used directly). depthwise done as per-channel np.convolve.
   - **GPU primitives added + parity-gated** (compute.js, gpu-verify): conv2d (grouped/
     depthwise, fused bias+relu/silu) 2.4e-7; silu (ACT=4); glu (a·σ(b)) 6e-8.
   - **GPU ENCODER PORT DONE + parity-gated: full 24-layer raw-WebGPU encoder = 5.3e-7
     vs ORT.** src/engines/asr-parakeet/raw-encoder.js (loadParakeetEncoder + async
     parakeetEncode), scripts/extract-parakeet-encoder-weights.py, scripts/smoke-
     parakeet-encoder-raw.mjs. GPU subsampling 4e-6, GPU conformer layer0 3e-6, full
     5.3e-7. Folded into weights at load: q·(1/√128), pos_bias_u/v·(1/√128), FF out
     proj·0.5. rel_shift on host (small index remap). per-head attn = 8× sliceCols/
     matmul/softmax/setCols. Node readFileSync caps at 2GB → chunked reader for the
     2.3GB fp32 bin (browser fetches quantized ~300MB, no issue).
   - **QUANTIZATION = int8, not int4** (user call: int4 too lossy). int8 per-channel
     symmetric VALIDATED WER-neutral: raw encoder-output RMS perturbs ~23% (the 24-layer
     residual stack is chaotically sensitive — matmul-only 23%, conv-only 22.7%, both
     22.9%, block size irrelevant), BUT the TDT decoder is robust → **transcript is
     byte-identical to fp32** on real speech ("The people here to me are the smartest
     people..."). So raw-output RMS is a misleading metric for ASR; the transcript is
     the gate. int8 ≈ 600MB (vs fp32 2.4GB, fp16 1.2GB). granularity (per-channel vs
     per-block 128/64/32) doesn't matter here → use simple per-channel symmetric.
     REMAINING: build int8 artifact (q int8 + per-channel fp32 scales), dequant at load
     (→fp32, existing kernels) OR int8 matmul kernel; upload HF parakeet/; wire raw
     decoder+joint (LSTM+MatMul+Relu) + JS mel (replace nemo128); windowing from tdt.js;
     delete ORT.
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
- [~] 3. Nemotron / EOU / Sortformer
  - [x] **EOU (Parakeet-EOU 120M)** — DONE, ORT-FREE, weights on HF.
    - Encoder: shared raw FastConformer (raw-encoder.js) with EOU streaming config
      {subPad t2/b1/l2/r1, convCausal, attChunk 2, attLeft 70, melBins 128}. fp16
      weights (int8 degrades this 120M RNNT: maxΔ 0.30 drops words; fp16 maxΔ 0.044
      byte-identical transcript). Parity vs ORT 3.2e-5 (fp32) / 4.4e-2 (fp16).
    - Decoder: JS RNNT (raw-decoder-eou.js). GOTCHA: the exported decoder_joint
      prepends a zero SOS timestep EVERY call → the pred LSTM runs TWO steps
      [zeros, embed(token)] from the incoming state. Single-step gave all-blank
      (empty transcript). embed[blank=1026] is a zero padding row.
    - Mel: JsPreprocessor NA (no CMVN). GOTCHA: its `length` = frames-1, so
      parakeetEncode must take the stride from mel.length/melBins, not the passed T.
    - Weights: FluidInference/fluidaudio-web eou/{encoder-fp16,decoder-fp32}.{bin,manifest.json}.
    - Scripts: extract-fastconformer-encoder.py, quantize-encoder-fp16.py, smoke-eou.mjs (ORT ref).
  - [x] **Nemotron 3.5 multilingual (0.6B)** — DONE, ORT-FREE, weights on HF.
    - Run OFFLINE whole-clip (cache-aware streaming ≡ offline-with-limited-context-mask,
      so no cache plumbing). Shared raw FastConformer (24L d1024) with Nemotron config
      {subPad t2/b1/l2/r1, convCausal (dwK 9), attChunk 4, attLeft 56, attRight 3}.
    - **int8** (not int4, per Alex) — 630MB. The 600M model is int8-robust: coherent
      full transcript, unlike the 120M EOU which needed fp16. Offline int8 vs streaming
      encoder-frame reference: per-frame maxΔ 0.03–0.13 in the bulk (final partial chunk
      diverges, expected).
    - **prompt_kernel** = multilingual conditioning MLP applied AFTER the conformer:
      encoded_output = MLP(concat([conformer_out 1024, language_onehot 128]), 1152→2048→1024).
      langId from languages.json promptDictionary (en-US=0). scripts/extract-nemotron-prompt-kernel.py.
    - Decoder: JS 2-layer LSTM RNN-T (raw-decoder-nemotron.js). NO SOS-prepend (unlike
      EOU's fused export — plain single step). joint = enc 1024→640 + pred 640→640 + relu
      + out 640→13087. blank 13087. scripts/extract-nemotron-decoder.py.
    - Weights: FluidInference/fluidaudio-web nemotron/{encoder-int8,decoder-fp32}.{bin,manifest.json}
      + vocab.json + languages.json. Engine src/engines/asr-nemotron/index.ts ORT-free.
    - Transcript (cowen.wav): "The people here to me are the smartest people I've ever met,
      but I think a side result of that is that people here overvalue intelligence and their
      models of the world are built on intelligence mattering much."
  - [ ] Sortformer (18-layer transformer head + spkcache/fifo streaming) — /tmp/sf
- [ ] 4. Kokoro
- [ ] 5. Whisper
- [ ] remove onnxruntime-web / @huggingface/transformers / kokoro-js from package.json
