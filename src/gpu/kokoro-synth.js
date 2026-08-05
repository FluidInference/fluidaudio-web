// Kokoro iSTFTNet vocoder + predictor, hand-ported from the validated numpy
// reference (/tmp/kfinal.py generator == ORT waveform maxΔ 2e-5; /tmp/kdec.py
// decoder body). Runs on the ctx backend (WebGPU or WASM), so no onnxruntime /
// kokoro-js. Heavy ops (conv/convT/adain/snake/matmul) go through ctx; the one-shot
// exotic tail (STFT recombine, iSTFT overlap-add, SineGen) is host-side JS.
//
// Weights: flat fp32 blob + manifest (name→{offset,len,dims}) + roles
// (nodeSuffix→{w,b,r}) from scripts/extract-kokoro.py. R(suffix) mirrors the numpy
// R(): find the role whose node name endsWith(suffix) → its weight/bias tensors.

/** Build a weight accessor over the extracted Kokoro blob. */
export function makeKokoro(ctx, weights, manifest, roles) {
  const cacheT = new Map();
  const raw = (name) => {
    const m = manifest[name];
    if (!m) throw new Error(`kokoro weight missing: ${name}`);
    return weights.subarray(m.offset, m.offset + m.len);
  };
  const dims = (name) => manifest[name].dims;
  // upload a named initializer as a ctx tensor with a chosen [rows,cols] shape.
  const up = (name, rows, cols) => {
    const key = `${name}:${rows}x${cols}`;
    if (cacheT.has(key)) return cacheT.get(key);
    const t = ctx.upload(raw(name).slice(), rows, cols);
    cacheT.set(key, t);
    return t;
  };
  // resolve a node suffix → its weight/bias initializer names.
  const roleKeys = Object.keys(roles);
  const findRole = (suffix) => {
    const k = roleKeys.find((rk) => rk.endsWith(suffix));
    if (!k) throw new Error(`kokoro role missing: ${suffix}`);
    return roles[k];
  };
  return { raw, dims, up, findRole, ctx, manifest };
}

// ── host helpers (one-shot exotic ops) ───────────────────────────────────────
const leakyHost = (data, slope) => { const o = new Float32Array(data.length); for (let i = 0; i < data.length; i++) o[i] = data[i] > 0 ? data[i] : slope * data[i]; return o; };

/**
 * iSTFTNet generator: decode3 [512,T0] + source spec [22,Ts] → waveform.
 * Faithful port of /tmp/kfinal.py. style128 = acoustic style (style[:128]).
 */
export async function generator(K, decode3, sourceSpec, style128) {
  const ctx = K.ctx;
  const conv = (x, suffix, { pad = 0, stride = 1, dil = 1 } = {}) => {
    const r = K.findRole(suffix); const wd = K.dims(r.w); // [Cout, Cin, K]
    const w = K.up(r.w, 1, wd[0] * wd[1] * wd[2]);
    const b = r.b ? K.up(r.b, 1, wd[0]) : null;
    return ctx.conv1d(x, w, { cout: wd[0], k: wd[2], bias: b, stride, pad, dilation: dil });
  };
  const convT = (x, suffix, stride, pad) => {
    const r = K.findRole(suffix); const wd = K.dims(r.w); // [Cin, Cout, K]
    const w = K.up(r.w, 1, wd[0] * wd[1] * wd[2]);
    const b = r.b ? K.up(r.b, 1, wd[1]) : null;
    return ctx.convTranspose1d(x, w, { cout: wd[1], k: wd[2], bias: b, stride, pad });
  };
  // AdaIN: h = fc(style)  → scale = 1+h[:C], shift = h[C:]; instance-norm(x)*scale+shift.
  const adain = async (x, suffix) => {
    const r = K.findRole(`${suffix}/fc/Gemm`); const wd = K.dims(r.w); // Gemm weight [2C, 128]
    const twoC = wd[0], C = twoC / 2;
    const w = K.up(r.w, twoC, 128), b = r.b ? K.up(r.b, 1, twoC) : null;
    const h = await ctx.download(ctx.matmul(w, style128T(ctx, style128))); // [2C,1]
    const bd = b ? await ctx.download(b) : null;
    const scale = new Float32Array(C), shift = new Float32Array(C);
    for (let i = 0; i < C; i++) { scale[i] = 1 + h[i] + (bd ? bd[i] : 0); shift[i] = h[C + i] + (bd ? bd[C + i] : 0); }
    return ctx.adain(x, ctx.upload(scale, 1, C), ctx.upload(shift, 1, C));
  };
  const alpha = (name) => { const d = K.raw(name); return ctx.upload(d.slice(), 1, d.length); };
  // one AdaIN-Snake residual block (convs1 dil[1,3,5] k3 + convs2), pre = e.g. 'resblocks.0'
  const resb = async (x, pre) => {
    const kSize = K.dims(K.findRole(`${pre}/convs1.0/Conv`).w)[2];
    for (let j = 0; j < 3; j++) {
      const d = [1, 3, 5][j];
      let xt = ctx.snake(await adain(x, `${pre}/adain1.${j}`), alpha(`decoder.decoder.generator.${pre}.alpha1.${j}`));
      xt = conv(xt, `${pre}/convs1.${j}/Conv`, { pad: ((kSize - 1) * d) >> 1, dil: d });
      xt = ctx.snake(await adain(xt, `${pre}/adain2.${j}`), alpha(`decoder.decoder.generator.${pre}.alpha2.${j}`));
      xt = conv(xt, `${pre}/convs2.${j}/Conv`, { pad: (kSize - 1) >> 1 });
      x = ctx.add(x, xt);
    }
    return x;
  };
  const group = async (x, a, b, c) => {
    const ra = await resb(x, a), rb = await resb(x, b), rc = await resb(x, c);
    const sum = ctx.add(ctx.add(ra, rb), rc);
    // divide by 3
    const d = await ctx.download(sum); for (let i = 0; i < d.length; i++) d[i] /= 3;
    return ctx.upload(d, sum.rows, sum.cols);
  };
  const leaky = (x, slope) => ctx.leakyRelu(x, slope);
  // reflect-pad ups to noise length (front pad), then add.
  const merge = async (ups, noise) => {
    const u = await ctx.download(ups), n = await ctx.download(noise);
    const C = ups.rows, Lu = ups.cols, Ln = noise.cols, L = Math.min(Lu, Ln);
    const out = new Float32Array(C * L);
    const padF = Ln > Lu ? Ln - Lu : 0; // front reflect pad amount (numpy mode='reflect')
    for (let c = 0; c < C; c++) for (let t = 0; t < L; t++) {
      let uv;
      if (padF && t < padF) uv = u[c * Lu + (padF - t)]; // reflect
      else uv = u[c * Lu + (t - padF)];
      out[c * L + t] = uv + n[c * Ln + t];
    }
    return ctx.upload(out, C, L);
  };

  const spec = ctx.upload(sourceSpec.data.slice(), sourceSpec.rows, sourceSpec.cols);
  let x = ctx.upload(decode3.data.slice(), decode3.rows, decode3.cols);
  x = convT(leaky(x, 0.1), "ups.0/ConvTranspose", 10, 5); // 512→256, stride10 pad5
  x = await merge(x, await resb(conv(spec, "noise_convs.0/Conv", { pad: 3, stride: 6 }), "noise_res.0"));
  x = await group(x, "resblocks.0", "resblocks.1", "resblocks.2");
  x = convT(leaky(x, 0.1), "ups.1/ConvTranspose", 6, 3); // 256→128, stride6 pad3
  x = await merge(x, await resb(conv(spec, "noise_convs.1/Conv", { pad: 0, stride: 1 }), "noise_res.1"));
  x = await group(x, "resblocks.3", "resblocks.4", "resblocks.5");
  x = conv(leaky(x, 0.01), "conv_post/Conv", { pad: 3 }); // [22, T]
  return await istft(K, x);
}

// style as a [128,1] ctx tensor for fc matmul (w[2C,128] @ style[128,1]).
function style128T(ctx, style128) {
  return ctx.upload(style128.slice(), style128.length, 1);
}

/**
 * Full Kokoro synthesis (ORT-free): d_en (from frontend textEncoding) + phoneme ids +
 * style[256] → waveform Float32Array. Chains predictor → SineGen → decoder → generator.
 */
export async function synth(K, dEn, ids, style) {
  const ctx = K.ctx;
  const { xConcat, asr, F0, N } = await predictor(K, dEn, ids, style);
  const source = sineGen(K, F0);                       // [2T*300] source waveform
  const spec = sourceSpec(source);                     // [22, ~frames]
  const decode3T = await decoder(K, xConcat, asr, F0, N, style.slice(0, 128));
  const decode3 = { data: await ctx.download(decode3T), rows: decode3T.rows, cols: decode3T.cols };
  return await generator(K, decode3, spec, style.slice(0, 128));
}

/**
 * Prosody predictor (port of scripts/kokoro-predictor-ref.py, numpy-exact vs ORT).
 * d_en:[seq,512] (bert), ids:Int32Array phoneme ids, style:[256]. Returns the decoder
 * inputs { xConcat[514,T], asr[512,T], F0[1,2T], N[1,2T] } (T = sum(pred_dur)).
 */
export async function predictor(K, dEn, ids, style) {
  const ctx = K.ctx;
  const seq = dEn.rows;
  const sp = style.slice(128, 256); // prosodic style
  const spT = ctx.upload(sp.slice(), 128, 1);
  const spRow = ctx.upload(sp.slice(), 1, 128);
  const lstmB = (x, sfx) => { const r = K.findRole(sfx); return ctx.lstm(x, K.up(r.w, 1, K.manifest[r.w].len), K.up(r.r, 1, K.manifest[r.r].len), K.up(r.b, 1, K.manifest[r.b].len), 256); };
  const gemm = (sfx) => { const r = K.findRole(sfx); const wd = K.dims(r.w); return { w: K.up(r.w, wd[0], wd[1]), b: r.b ? K.up(r.b, 1, wd[0]) : null, wd }; };
  // concat style (broadcast over seq) to x[seq,C] → [seq,C+128]
  const catStyle = async (x) => { const d = await ctx.download(x); const C = x.cols, out = new Float32Array(seq * (C + 128)); for (let i = 0; i < seq; i++) { out.set(d.subarray(i * C, i * C + C), i * (C + 128)); out.set(sp, i * (C + 128) + C); } return ctx.upload(out, seq, C + 128); };
  // AdaLN: x = layernorm(x)*(1+γ)+β, γ,β = fc(sp). fc weight [1024,128].
  const adaLN = async (x, sfx) => {
    const g = gemm(`${sfx}/fc/Gemm`); const h = await ctx.download(ctx.matmul(g.w, spT)); const bd = g.b ? await ctx.download(g.b) : null;
    const C = x.cols, gA = new Float32Array(C), bA = new Float32Array(C);
    for (let i = 0; i < C; i++) { gA[i] = 1 + h[i] + (bd ? bd[i] : 0); bA[i] = h[C + i] + (bd ? bd[C + i] : 0); }
    return ctx.layernorm(x, ctx.upload(gA, 1, C), ctx.upload(bA, 1, C));
  };

  // ── Chain B: DurationEncoder → duration + alignment ──
  let x = dEn;
  for (const [li, ai] of [[0, 1], [2, 3], [4, 5]]) {
    x = lstmB(await catStyle(x), `predictor/text_encoder/lstms.${li}/LSTM`);
    x = await adaLN(x, `predictor/text_encoder/lstms.${ai}`);
  }
  const durEncOut = x; // [seq,512]
  const xp = lstmB(await catStyle(x), "predictor/lstm/LSTM"); // [seq,512]
  const dp = gemm("duration_proj/linear_layer/MatMul");
  const dprojBias = K.manifest["encoder.predictor.duration_proj.linear_layer.bias"];
  const durLogits = await ctx.download(ctx.matmul(xp, dp.w)); // [seq,50]
  const db = dprojBias ? K.raw("encoder.predictor.duration_proj.linear_layer.bias") : null;
  const predDur = new Int32Array(seq);
  for (let i = 0; i < seq; i++) { let s = 0; for (let j = 0; j < 50; j++) { const v = durLogits[i * 50 + j] + (db ? db[j] : 0); s += 1 / (1 + Math.exp(-v)); } predDur[i] = Math.max(1, Math.min(50, Math.round(s))); }
  const T = predDur.reduce((a, b) => a + b, 0);
  // alignment A[seq,T]; en = d^T @ A where d = concat(durEncOut, sp)[seq,640]
  const dData = await ctx.download(durEncOut);
  const d640 = new Float32Array(seq * 640);
  for (let i = 0; i < seq; i++) { d640.set(dData.subarray(i * 512, i * 512 + 512), i * 640); d640.set(sp, i * 640 + 512); }
  const en = new Float32Array(640 * T); // [640,T]
  { let f = 0; for (let i = 0; i < seq; i++) { for (let dd = 0; dd < predDur[i]; dd++) { for (let c = 0; c < 640; c++) en[c * T + (f + dd)] = d640[i * 640 + c]; } f += predDur[i]; } }
  const shared = lstmB(ctx.upload(transposeHost(en, 640, T), T, 640), "shared/LSTM"); // [T,512]
  const prosody = ctx.upload(transposeHost(await ctx.download(shared), T, 512), 512, T); // [512,T]

  // F0/N AdaINResBlocks (prosodic style)
  const f0 = await predResBlocks(K, prosody, "F0", spT, spRow);
  const nn = await predResBlocks(K, prosody, "N", spT, spRow);

  // ── Chain A: text_encoder → asr, aligned by A ──
  const emb = K.raw("encoder.text_encoder.embedding.weight"); // [178,512]
  const embX = new Float32Array(seq * 512);
  for (let i = 0; i < seq; i++) embX.set(emb.subarray(ids[i] * 512, ids[i] * 512 + 512), i * 512);
  let te = ctx.upload(transposeHost(embX, seq, 512), 512, seq); // [512,seq]
  for (const c of ["cnn.0", "cnn.1", "cnn.2"]) {
    const r = K.findRole(`${c}/cnn.${c.slice(-1)}.0/Conv`); const wd = K.dims(r.w);
    te = ctx.conv1d(te, K.up(r.w, 1, wd[0] * wd[1] * wd[2]), { cout: wd[0], k: wd[2], bias: K.up(r.b, 1, wd[0]), pad: 2 });
    te = lnChan(K, te, c);
    te = ctx.leakyRelu(te, 0.2);
  }
  const teL = lstmB(ctx.upload(transposeHost(await ctx.download(te), 512, seq), seq, 512), "encoder/text_encoder/lstm/LSTM"); // [seq,512]
  // asr = teL^T @ A  → [512,T]
  const teLd = await ctx.download(teL); const asr = new Float32Array(512 * T);
  { let f = 0; for (let i = 0; i < seq; i++) { for (let dd = 0; dd < predDur[i]; dd++) { for (let c = 0; c < 512; c++) asr[c * T + (f + dd)] = teLd[i * 512 + c]; } f += predDur[i]; } }
  const asrT = ctx.upload(asr, 512, T);

  // decoder input Concat = [asr; F0_conv(F0)s2; N_conv(N)s2]
  const f0conv = convHost(K, f0, "F0_conv/Conv", 1, 2);
  const nconv = convHost(K, nn, "N_conv/Conv", 1, 2);
  const Tc = Math.min(asr.length / 512, f0conv.cols);
  const xConcat = new Float32Array(514 * Tc);
  for (let c = 0; c < 512; c++) for (let t = 0; t < Tc; t++) xConcat[c * Tc + t] = asr[c * T + t];
  for (let t = 0; t < Tc; t++) { xConcat[512 * Tc + t] = f0conv.data[t]; xConcat[513 * Tc + t] = nconv.data[t]; }
  return { xConcat: { data: xConcat, rows: 514, cols: Tc }, asr: asrT, F0: f0, N: nn, predDur, T };
}

function transposeHost(d, rows, cols) { const o = new Float32Array(rows * cols); for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) o[c * rows + r] = d[r * cols + c]; return o; }

// LayerNorm over channels (x:[C,L], normalize over C per column). gamma/beta named
// encoder.text_encoder.<cnn>.1.gamma/beta. Transpose → ctx.layernorm(over C) → transpose.
function lnChan(K, x, cnn) {
  const ctx = K.ctx, C = x.rows, L = x.cols;
  const g = ctx.upload(K.raw(`encoder.text_encoder.${cnn}.1.gamma`).slice(), 1, C);
  const b = ctx.upload(K.raw(`encoder.text_encoder.${cnn}.1.beta`).slice(), 1, C);
  const xt = ctx.transpose(x);                 // [L,C]
  const yn = ctx.layernorm(xt, g, b);          // LN over C
  return ctx.transpose(yn);                    // [C,L]
}

// F0/N AdaINResBlocks (which='F0'|'N'): blocks .0/.1/.2 (.1 upsamples ×2), then which_proj.
async function predResBlocks(K, prosody, which, spT) {
  const ctx = K.ctx;
  const conv = (x, sfx, { pad = 1, stride = 1 } = {}) => { const r = K.findRole(sfx); const wd = K.dims(r.w); return ctx.conv1d(x, K.up(r.w, 1, wd[0] * wd[1] * wd[2]), { cout: wd[0], k: wd[2], bias: r.b ? K.up(r.b, 1, wd[0]) : null, stride, pad, groups: wd[1] === 1 ? wd[0] : 1 }); };
  const hasRole = (sfx) => { try { K.findRole(sfx); return true; } catch { return false; } };
  const adain = async (x, sfx) => { const r = K.findRole(`${sfx}/fc/Gemm`); const wd = K.dims(r.w); const twoC = wd[0], C = twoC / 2; const h = await ctx.download(ctx.matmul(K.up(r.w, twoC, 128), spT)); const bd = r.b ? await ctx.download(K.up(r.b, 1, twoC)) : null; const sc = new Float32Array(C), sh = new Float32Array(C); for (let i = 0; i < C; i++) { sc[i] = 1 + h[i] + (bd ? bd[i] : 0); sh[i] = h[C + i] + (bd ? bd[C + i] : 0); } return ctx.adain(x, ctx.upload(sc, 1, C), ctx.upload(sh, 1, C)); };
  const upRep = (x) => { const idx = new Uint32Array(x.cols * 2); for (let i = 0; i < x.cols; i++) { idx[2 * i] = i; idx[2 * i + 1] = i; } return ctx.gatherCols(x, idx); };
  const dwcT = (x, sfx) => { const r = K.findRole(sfx); const wd = K.dims(r.w); return ctx.convTranspose1d(x, K.up(r.w, 1, wd[0] * wd[1] * wd[2]), { cout: wd[0], k: wd[2], bias: r.b ? K.up(r.b, 1, wd[0]) : null, stride: 2, pad: 1, groups: wd[0], outputPadding: 1 }); };
  const block = async (x, pre, up) => {
    let res = up ? upRep(x) : x;
    if (hasRole(`${pre}/conv1x1/Conv`)) res = conv(res, `${pre}/conv1x1/Conv`, { pad: 0 });
    x = ctx.leakyRelu(await adain(x, `${pre}/norm1`), 0.2);
    if (up) x = dwcT(x, `${pre}/pool/ConvTranspose`);
    x = conv(x, `${pre}/conv1/Conv`, { pad: 1 });
    x = ctx.leakyRelu(await adain(x, `${pre}/norm2`), 0.2);
    x = conv(x, `${pre}/conv2/Conv`, { pad: 1 });
    const sum = ctx.add(x, res); const d = await ctx.download(sum); const inv = 1 / Math.sqrt(2); for (let i = 0; i < d.length; i++) d[i] *= inv; return ctx.upload(d, sum.rows, sum.cols);
  };
  let x = prosody;
  for (const b of [`${which}.0`, `${which}.1`, `${which}.2`]) x = await block(x, b, b === `${which}.1`);
  return conv(x, `${which}_proj/Conv`, { pad: 0 }); // [1, 2T]
}

// SineGen (m_source) → source waveform (host, deterministic; corr 1.0 vs ORT).
// F0:[1,frames] → per-frame harmonic phase (cumsum×300)×2π → LINEAR upsample ×300 →
// sin×0.1×uv(F0>10) → l_linear[9,1]+bias → tanh. Port of kokoro-predictor-ref.py.
export function sineGen(K, F0) {
  const frames = F0.cols, sr = 24000, up = 300, L = frames * up;
  const lw = K.raw(K.findRole("m_source/l_linear/MatMul").w); // [9,1]
  const lb = K.raw("decoder.decoder.generator.m_source.l_linear.bias")[0];
  const F0d = F0.data;
  // per-harmonic cumulative phase at frame rate, then LINEAR (half_pixel) upsample.
  const src = new Float32Array(L);
  for (let h = 0; h < 9; h++) {
    const ph = new Float32Array(frames); let acc = 0;
    for (let f = 0; f < frames; f++) {
      const rad = (F0d[f] * (h + 1)) / sr; acc += (rad - Math.floor(rad)) * up; ph[f] = acc * 2 * Math.PI;
    }
    for (let j = 0; j < L; j++) {
      const pos = (j + 0.5) / up - 0.5; let lo = Math.floor(pos); const w = pos - lo;
      lo = Math.max(0, Math.min(frames - 1, lo)); const hi = Math.max(0, Math.min(frames - 1, lo + 1));
      src[j] += Math.sin(ph[lo] * (1 - w) + ph[hi] * w) * 0.1 * lw[h];
    }
  }
  for (let j = 0; j < L; j++) {
    const uv = F0d[Math.min(frames - 1, Math.floor(j / up))] > 10 ? 1 : 0;
    src[j] = Math.tanh(src[j] * uv + lb);
  }
  return src;
}
// Forward STFT of the source → [22,frames] spec = [magnitude(11); phase(11)].
// hann-20, frame_len 20, hop 5, no center pad (frames=(L-20)/5+1).
export function sourceSpec(src) {
  const nfft = 20, hop = 5, half = 11, pad = nfft / 2; // center STFT: reflect-pad nfft/2 each side
  const buf = new Float32Array(src.length + 2 * pad);
  buf.set(src, pad);
  for (let i = 0; i < pad; i++) { buf[pad - 1 - i] = src[i + 1]; buf[pad + src.length + i] = src[src.length - 2 - i]; }
  const win = new Float32Array(nfft);
  for (let n = 0; n < nfft; n++) win[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / nfft); // hann periodic
  const frames = Math.floor((buf.length - nfft) / hop) + 1;
  const spec = new Float32Array(22 * frames);
  for (let t = 0; t < frames; t++) {
    for (let k = 0; k < half; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < nfft; n++) { const v = buf[t * hop + n] * win[n]; const a = (-2 * Math.PI * k * n) / nfft; re += v * Math.cos(a); im += v * Math.sin(a); }
      spec[k * frames + t] = Math.sqrt(re * re + im * im);
      spec[(k + half) * frames + t] = Math.atan2(im, re);
    }
  }
  return { data: spec, rows: 22, cols: frames };
}

// Apply a named conv on host (small F0_conv/N_conv, [1,2T]→[1,T]).
function convHost(K, x, sfx, pad, stride) {
  const r = K.findRole(sfx); const w = K.raw(r.w); const wd = K.dims(r.w); // [Cout,Cin,K]
  const Cout = wd[0], Cin = wd[1], k = wd[2], L = x.cols;
  const b = r.b ? K.raw(r.b) : null;
  const Lout = Math.floor((L + 2 * pad - k) / stride) + 1;
  const out = new Float32Array(Cout * Lout);
  for (let co = 0; co < Cout; co++) for (let t = 0; t < Lout; t++) {
    let acc = b ? b[co] : 0;
    for (let ci = 0; ci < Cin; ci++) for (let kk = 0; kk < k; kk++) { const li = t * stride + kk - pad; if (li >= 0 && li < L) acc += x.data[ci * L + li] * w[(co * Cin + ci) * k + kk]; }
    out[co * Lout + t] = acc;
  }
  return { data: out, rows: Cout, cols: Lout };
}

// ── channel concat (stack rows) of CPU-resident tensors ──────────────────────
async function concatRows(ctx, tensors) {
  const cols = tensors[0].cols;
  const datas = await Promise.all(tensors.map((t) => ctx.download(t)));
  const rows = tensors.reduce((s, t) => s + t.rows, 0);
  const out = new Float32Array(rows * cols);
  let r0 = 0;
  for (let ti = 0; ti < tensors.length; ti++) { out.set(datas[ti], r0 * cols); r0 += tensors[ti].rows; }
  return ctx.upload(out, rows, cols);
}

/**
 * iSTFTNet decoder body (encode + decode.0-3), port of /tmp/kdec.py. Inputs are the
 * aligned features. style128 = acoustic style. Returns decode3 [512, T].
 *   xConcat: [514, T]  (asr+F0+N concat, /decoder/decoder/Concat)
 *   asr:[512,T]  F0:[1,2T]  N:[1,2T]
 */
export async function decoder(K, xConcat, asr, F0, N, style128, onStage) {
  const ctx = K.ctx;
  const styT = style128T(ctx, style128);
  const conv = (x, suffix, { pad = 1, stride = 1 } = {}) => {
    const r = K.findRole(suffix); const wd = K.dims(r.w);
    const w = K.up(r.w, 1, wd[0] * wd[1] * wd[2]);
    const b = r.b ? K.up(r.b, 1, wd[0]) : null;
    return ctx.conv1d(x, w, { cout: wd[0], k: wd[2], bias: b, stride, pad });
  };
  const adain = async (x, suffix) => {
    const r = K.findRole(`${suffix}/fc/Gemm`); const wd = K.dims(r.w); const twoC = wd[0], C = twoC / 2;
    const h = await ctx.download(ctx.matmul(K.up(r.w, twoC, 128), styT));
    const bd = r.b ? await ctx.download(K.up(r.b, 1, twoC)) : null;
    const scale = new Float32Array(C), shift = new Float32Array(C);
    for (let i = 0; i < C; i++) { scale[i] = 1 + h[i] + (bd ? bd[i] : 0); shift[i] = h[C + i] + (bd ? bd[C + i] : 0); }
    return ctx.adain(x, ctx.upload(scale, 1, C), ctx.upload(shift, 1, C));
  };
  const lrelu = (x) => ctx.leakyRelu(x, 0.2);
  const upRepeat = (x) => { const idx = new Uint32Array(x.cols * 2); for (let i = 0; i < x.cols; i++) { idx[2 * i] = i; idx[2 * i + 1] = i; } return ctx.gatherCols(x, idx); };
  const dwcT = (x, suffix) => { // depthwise ConvTranspose1d stride2 pad1 opad1
    const r = K.findRole(suffix); const wd = K.dims(r.w); // [Cin,1,K]
    const w = K.up(r.w, 1, wd[0] * wd[1] * wd[2]);
    const b = r.b ? K.up(r.b, 1, wd[0]) : null;
    return ctx.convTranspose1d(x, w, { cout: wd[0], k: wd[2], bias: b, stride: 2, pad: 1, groups: wd[0], outputPadding: 1 });
  };
  const block = async (x, pre, up) => {
    let res = up ? upRepeat(x) : x;
    res = conv(res, `${pre}/conv1x1/Conv`, { pad: 0 });
    x = lrelu(await adain(x, `${pre}/norm1`));
    if (up) x = dwcT(x, `${pre}/pool/ConvTranspose`);
    x = conv(x, `${pre}/conv1/Conv`, { pad: 1 });
    x = lrelu(await adain(x, `${pre}/norm2`));
    x = conv(x, `${pre}/conv2/Conv`, { pad: 1 });
    const sum = ctx.add(x, res);
    const d = await ctx.download(sum); const inv = 1 / Math.sqrt(2); for (let i = 0; i < d.length; i++) d[i] *= inv;
    return ctx.upload(d, sum.rows, sum.cols);
  };

  const asrRes = conv(asr, "asr_res.0/Conv", { pad: 0 });
  const F0d = conv(F0, "F0_conv/Conv", { pad: 1, stride: 2 });
  const Nd = conv(N, "N_conv/Conv", { pad: 1, stride: 2 });
  let x = await block(xConcat, "encode", false);
  if (onStage) await onStage("encode", x, { asrRes, F0d, Nd });
  for (const b of ["decode.0", "decode.1", "decode.2", "decode.3"]) {
    x = await concatRows(ctx, [x, asrRes, F0d, Nd]);
    x = await block(x, b, b === "decode.3");
    if (onStage) await onStage(b, x);
  }
  return x; // decode3 [512, T]
}

// STFT recombine + iSTFT overlap-add (host, one-shot). Port of kfinal.py tail.
async function istft(K, convPostT) {
  const ctx = K.ctx;
  const cp = await ctx.download(convPostT); // [22, T]
  const T = convPostT.cols, half = 11;
  // mag = exp(cp[0:11]); p = sin(cp[11:22]); real = mag*cos(p); imag = mag*sin(p)
  const recomb = new Float32Array(22 * T);
  for (let f = 0; f < half; f++) for (let t = 0; t < T; t++) {
    const mag = Math.exp(cp[f * T + t]), p = Math.sin(cp[(f + half) * T + t]);
    recomb[f * T + t] = mag * Math.cos(p);
    recomb[(f + half) * T + t] = mag * Math.sin(p);
  }
  const ib = K.raw("decoder.decoder.generator.stft.istft.stft.inverse_basis"); // [22,1,20]
  const ws = K.raw("decoder.decoder.generator.stft.istft.stft.window_sum");     // [20]
  const hop = 5, nfft = 20, Lo = (T - 1) * hop + nfft;
  const wav = new Float32Array(Lo), wsum = new Float32Array(Lo);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < nfft; k++) {
      let acc = 0;
      for (let c = 0; c < 22; c++) acc += ib[c * nfft + k] * recomb[c * T + t]; // ib[c,0,k]
      wav[t * hop + k] += acc;
      wsum[t * hop + k] += ws[k];
    }
  }
  const trimmed = new Float32Array(Lo - nfft);
  for (let i = nfft / 2; i < Lo - nfft / 2; i++) trimmed[i - nfft / 2] = 6.0 * wav[i] / (wsum[i] > 1e-9 ? wsum[i] : 1);
  return trimmed;
}

export { leakyHost };
