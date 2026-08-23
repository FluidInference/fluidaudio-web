// VoiceChat-11B audio codec — decode side only (PRVQ dequantize + Latent2Wav).
// Mirrors Speech/nemo/collections/speechlm2/modules/ear_tts_vae_codec.py:
// 31-quantizer PRVQ codebook sum → [T,512] latents → 3 upsampling stages
// (ConvT k=s=9/7/7 + 3 causal ConvNeXt blocks each) → Conv1d→18 → mag/phase →
// 16-point iSTFT (hop 4, hann, constrain_value_range) → 22.05 kHz waveform.
// Offline batch path (no causal caches): 1 token → 1764 samples.
// PyTorch's exact-erf GELU maps to the kernel lib's "gelu_erf" (Abramowitz-
// Stegun erf approx, |err| ≲ 1.5e-7 — inside the waveform NRMSE gate).

const NFFT = 16;
const HOP = 4;
const BINS = NFFT / 2 + 1; // 9
const LOG_MAX_MAG = Math.log(100.0);

export function loadVoicechatCodec(ctx, W) {
  const cn = (i) => ({
    dwW: ctx.upload(W.f32(`D${i}_dw_w`).slice(), 1, W.meta(`D${i}_dw_w`).count), // flat [C,1,7]
    dwB: ctx.upload(W.f32(`D${i}_dw_b`).slice(), 1, W.meta(`D${i}_dw_b`).count),
    lnW: ctx.upload(W.f32(`D${i}_ln_w`).slice(), 1, W.meta(`D${i}_ln_w`).count),
    lnB: ctx.upload(W.f32(`D${i}_ln_b`).slice(), 1, W.meta(`D${i}_ln_b`).count),
    pw1: W.mat(ctx, `D${i}_pw1_w`), // [C, 4C]
    pw1B: ctx.upload(W.f32(`D${i}_pw1_b`).slice(), 1, W.meta(`D${i}_pw1_b`).count),
    pw2: W.mat(ctx, `D${i}_pw2_w`), // [4C, C]
    pw2B: ctx.upload(W.f32(`D${i}_pw2_b`).slice(), 1, W.meta(`D${i}_pw2_b`).count),
  });
  // ConvT weights upload once as flat [1, count] tensors — stable object identity
  // (the wasm backend caches its transposed copy per weight-tensor object).
  const ct = (i) => ({ w: ctx.upload(W.f32(`D${i}_w`).slice(), 1, W.meta(`D${i}_w`).count), cout: W.dims(`D${i}_w`)[1], k: W.dims(`D${i}_w`)[2] });
  return {
    rvq: W.f32("rvq_embs"), // [31*1024*512] CPU (shared with the TTS depthsum/PRVQ side)
    stages: [
      { ct: ct(0), blocks: [cn(1), cn(2), cn(3)] },
      { ct: ct(4), blocks: [cn(5), cn(6), cn(7)] },
      { ct: ct(8), blocks: [cn(9), cn(10), cn(11)] },
    ],
    // final Conv1d(384→18, k=1, no bias) as a [384, 18] matmul in [L,C] layout
    final: (() => {
      const w = W.f32("D12_w"),
        [co, ci] = W.dims("D12_w"); // [18, 384, 1]
      const t = new Float32Array(ci * co);
      for (let a = 0; a < co; a++) for (let b = 0; b < ci; b++) t[b * co + a] = w[a * ci + b];
      return ctx.upload(t, ci, co);
    })(),
  };
}

/** codes [T][31] (0..1023) → latents [T*512] via codebook sums. */
export function dequantize(rvq, codes, latentSize = 512, numQ = 31) {
  const T = codes.length;
  const out = new Float32Array(T * latentSize);
  for (let t = 0; t < T; t++) {
    const row = t * latentSize;
    for (let q = 0; q < numQ; q++) {
      const e = (q * 1024 + codes[t][q]) * latentSize;
      for (let c = 0; c < latentSize; c++) out[row + c] += rvq[e + c];
    }
  }
  return out;
}

/** One ConvNeXt1d block (causal): dw k7 → LN over channels → pw1 gelu → pw2 → +res. */
function convNext(ctx, x, b) {
  const C = x.rows;
  const dw = ctx.conv1d(x, b.dwW, { cout: C, k: 7, groups: C, padLeft: 6, padRight: 0, bias: b.dwB });
  let t = ctx.transpose(dw); // [L, C] — LayerNormNd normalizes over channels per position
  t = ctx.layernorm(t, b.lnW, b.lnB, 1e-6);
  t = ctx.matmul(t, b.pw1, { bias: b.pw1B, act: "gelu_erf" });
  t = ctx.matmul(t, b.pw2, { bias: b.pw2B });
  return ctx.add(x, ctx.transpose(t));
}

const softplus = (x) => (x > 20 ? x : Math.log1p(Math.exp(x)));

/**
 * Latent2Wav offline decode: latents [T*512] → Float32Array waveform [T*1764].
 * @param {*} ctx ComputeContext  @param {*} dec loadVoicechatCodec handle
 */
export async function codecDecode(ctx, dec, latents, T) {
  // [T,512] → [512,T]
  let x = ctx.transpose(ctx.ensureTensor({ data: latents, rows: T, cols: 512 }));
  for (const st of dec.stages) {
    x = ctx.convTranspose1d(x, st.ct.w, { cout: st.ct.cout, k: st.ct.k, stride: st.ct.k });
    for (const b of st.blocks) x = convNext(ctx, x, b);
  }
  const specT = ctx.matmul(ctx.transpose(x), dec.final); // [L, 18]
  const sp = await ctx.download(specT);
  const L = specT.rows; // 441*T spectrogram frames

  // mag/phase → complex spec → iSTFT (spec_to_wav, constrain_value_range=True)
  const win = new Float32Array(NFFT);
  for (let n = 0; n < NFFT; n++) win[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / NFFT));
  const cosT = new Float32Array(BINS * NFFT),
    sinT = new Float32Array(BINS * NFFT);
  for (let k = 0; k < BINS; k++)
    for (let n = 0; n < NFFT; n++) {
      cosT[k * NFFT + n] = Math.cos((2 * Math.PI * k * n) / NFFT);
      sinT[k * NFFT + n] = Math.sin((2 * Math.PI * k * n) / NFFT);
    }
  const outSize = (L - 1) * HOP + NFFT;
  const wav = new Float32Array(outSize),
    env = new Float32Array(outSize);
  const re = new Float32Array(BINS),
    im = new Float32Array(BINS),
    frame = new Float32Array(NFFT);
  for (let t = 0; t < L; t++) {
    const row = t * 18;
    for (let k = 0; k < BINS; k++) {
      const m = 100.0 * Math.exp(-softplus(LOG_MAX_MAG - sp[row + k]));
      const ph = sp[row + BINS + k];
      // DC + Nyquist are real (phase via cos); mid bins full complex
      re[k] = m * Math.cos(ph);
      im[k] = k === 0 || k === BINS - 1 ? 0 : m * Math.sin(ph);
    }
    for (let n = 0; n < NFFT; n++) {
      let acc = re[0] * cosT[n] + re[BINS - 1] * cosT[(BINS - 1) * NFFT + n];
      for (let k = 1; k < BINS - 1; k++) acc += 2 * (re[k] * cosT[k * NFFT + n] - im[k] * sinT[k * NFFT + n]);
      frame[n] = acc / NFFT;
    }
    const off = t * HOP;
    for (let n = 0; n < NFFT; n++) {
      const w = win[n];
      const v = frame[n] >= 0 ? Math.min(frame[n], w) : Math.max(frame[n], -w); // constrain_value_range
      wav[off + n] += v * w;
      env[off + n] += w * w;
    }
  }
  const pad = (NFFT - HOP) >> 1; // 6
  const out = new Float32Array(outSize - 2 * pad);
  for (let i = 0; i < out.length; i++) out[i] = wav[pad + i] / env[pad + i];
  return out;
}
