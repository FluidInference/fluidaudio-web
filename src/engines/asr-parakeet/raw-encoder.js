// FastConformer encoder — hand-written raw-WebGPU (no onnxruntime), on the
// src/gpu/compute.js kernels. Config (d_model, layers, heads, d_ff, depthwise k,
// subsampling channels, mel bins) is INFERRED from the weight manifest, so the same
// code serves Parakeet / Nemotron / EOU / Sortformer — they're all NeMo FastConformers.
// Parity vs ORT on Parakeet: 5.3e-7 (scripts/smoke-parakeet-encoder-raw.mjs).
//
// Arch: mel[1,melBins,T] → dw-striding 8× subsampling (Conv2d ×5) → Linear→D → N×
// conformer blocks (macaron FF ½ · rel-pos MHA · conv module · FF ½ · norm_out).
// Folded into weights at load: q·(1/√HD), pos_bias_u/v·(1/√HD), FF out proj·0.5.

// fp16 → fp32 lookup table (65536 entries), built once and reused. Expanding an
// fp16 weight blob to fp32 at load is a cheap table lookup per value.
let _f16lut = null;
function f16lut() {
  if (_f16lut) return _f16lut;
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) t[h] = s * Math.pow(2, -14) * (f / 1024);
    else if (e === 0x1f) t[h] = f ? NaN : s * Infinity;
    else t[h] = s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return (_f16lut = t);
}

function inferConfig(man) {
  const layers = Object.keys(man).filter((k) => /^L\d+_lnff1_w$/.test(k)).length;
  const ff = man["L0_ff1w1"].dims; // [D, DFF]
  const pb = man["L0_pbu"].dims; // [H, HD]
  const Csub = man["c0w"].dims[0]; // subsampling conv channels
  const Fsub = man["linw"].dims[0] / Csub; // freq bins after 8× reduction
  return { D: ff[0], DFF: ff[1], H: pb[0], HD: pb[1], layers, dwK: man["L0_dw"].dims[2], Csub, Fsub, melBins: Fsub * 8 };
}

/** Upload all encoder weights to GPU once. bin: Float32Array (fp32 manifest) or
 * Uint8Array/ArrayBuffer (int8 manifest, dequantized per-tensor). Returns a handle. */
export function loadParakeetEncoder(ctx, bin, man, cfgOverride = {}) {
  const cfg = { ...inferConfig(man), ...cfgOverride };
  const { HD } = cfg;
  const INV = 1 / Math.sqrt(HD);
  const int8 = Object.values(man).some((m) => m.dtype === "i8");
  const f16 = !int8 && Object.values(man).some((m) => m.dtype === "f16");
  let f32v, i8v, u16v, lut;
  if (int8) { const ab = bin.buffer instanceof ArrayBuffer ? bin.buffer : bin; f32v = new Float32Array(ab); i8v = new Int8Array(ab); }
  if (f16) { u16v = new Uint16Array(bin.buffer, bin.byteOffset, bin.byteLength >> 1); lut = f16lut(); }
  const raw = (k) => {
    const m = man[k];
    if (f16) { const q = u16v.subarray(m.offset, m.offset + m.count), out = new Float32Array(m.count); for (let i = 0; i < m.count; i++) out[i] = lut[q[i]]; return out; }
    if (!int8) return bin.subarray(m.offset, m.offset + m.len);
    if (m.dtype !== "i8") return f32v.subarray(m.offset, m.offset + m.count);
    const q = i8v.subarray(m.i8ByteOffset, m.i8ByteOffset + m.count);
    const sc = f32v.subarray(m.scaleOffset, m.scaleOffset + m.scaleCount);
    const out = new Float32Array(m.count);
    if (m.quant === "col") { const o = m.dims[1]; for (let i = 0; i < m.count; i++) out[i] = q[i] * sc[i % o]; }
    else { const rest = m.count / m.dims[0]; for (let i = 0; i < m.count; i++) out[i] = q[i] * sc[(i / rest) | 0]; }
    return out;
  };
  const scaled = (k, s) => { const a = raw(k).slice(); for (let i = 0; i < a.length; i++) a[i] *= s; return a; };
  const mat = (k) => ctx.upload(raw(k).slice(), man[k].dims[0], man[k].dims[1]);
  const vec = (k) => ctx.upload(raw(k).slice(), 1, man[k].count ?? man[k].len);
  const matScaled = (k, s) => ctx.upload(scaled(k, s), man[k].dims[0], man[k].dims[1]);

  const sub = {
    conv: [0, 1, 2, 3, 4].map((i) => ({ w: vec(`c${i}w`), b: vec(`c${i}b`) })),
    linw: mat("linw"), linb: vec("linb"),
  };
  const layers = [];
  for (let L = 0; L < cfg.layers; L++) {
    const g = (s) => `L${L}_${s}`;
    // pos_bias_u/v uploaded ONCE as per-head GPU tensors [1,HD].
    const pbuS = scaled(g("pbu"), INV), pbvS = scaled(g("pbv"), INV);
    layers.push({
      lnff1: [vec(g("lnff1_w")), vec(g("lnff1_b"))], ff1w1: mat(g("ff1w1")), ff1w2: matScaled(g("ff1w2"), 0.5),
      lnatt: [vec(g("lnatt_w")), vec(g("lnatt_b"))],
      q: matScaled(g("q"), INV), k: mat(g("k")), v: mat(g("v")), pos: mat(g("pos")), out: mat(g("out")),
      pbuT: Array.from({ length: cfg.H }, (_, h) => ctx.upload(pbuS.slice(h * HD, h * HD + HD), 1, HD)),
      pbvT: Array.from({ length: cfg.H }, (_, h) => ctx.upload(pbvS.slice(h * HD, h * HD + HD), 1, HD)),
      lnconv: [vec(g("lnconv_w")), vec(g("lnconv_b"))],
      pw1: vec(g("pw1")), dw: vec(g("dw")), dwb: vec(g("dwb")), pw2: vec(g("pw2")),
      bn: man[g("bnw")] ? [vec(g("bnw")), vec(g("bnb"))] : null, // conv-module norm (EOU)
      lnff2: [vec(g("lnff2_w")), vec(g("lnff2_b"))], ff2w1: mat(g("ff2w1")), ff2w2: matScaled(g("ff2w2"), 0.5),
      lnout: [vec(g("lnout_w")), vec(g("lnout_b"))],
    });
  }
  return { sub, layers, cfg };
}

function posEncoding(Tsub, D) {
  const pe = new Float32Array((2 * Tsub - 1) * D);
  const dv = (i) => Math.exp(i * -(Math.log(10000) / D));
  for (let pi = 0; pi < 2 * Tsub - 1; pi++) {
    const pos = (Tsub - 1) - pi;
    for (let i = 0; i < D; i += 2) { pe[pi * D + i] = Math.sin(pos * dv(i)); pe[pi * D + i + 1] = Math.cos(pos * dv(i)); }
  }
  return pe;
}

/** Run the encoder. mel: Float32Array[melBins*T] (channel-major, mel[c*T+t]).
 * Returns { framesGpu:[Tsub,D], Tsub, dims } (+ data when wantData). */
export async function parakeetEncode(ctx, enc, mel, T, wantData = false) {
  const { D, H, HD, layers: LAYERS, dwK, Csub, melBins } = enc.cfg;
  // subsampling stride-2 conv padding. Parakeet: symmetric 1 each side. EOU (streaming
  // causal): asymmetric, padTotal 3 → out=floor(in/2)+1. Default symmetric.
  const sp = enc.cfg.subPad || { t: 1, b: 1, l: 1, r: 1 };
  const ln = (x, lp) => ctx.layernorm(x, lp[0], lp[1]);
  // The mel buffer stride is its actual frame count, which can exceed the valid
  // `length` (e.g. JsPreprocessor's NA mel reports length = frames-1). Reading
  // mel[c*T+t] with T=length would use the wrong stride and scramble the input.
  const Tfull = mel.length / melBins;
  const x0 = new Float32Array(Tfull * melBins);
  for (let t = 0; t < Tfull; t++) for (let c = 0; c < melBins; c++) x0[t * melBins + c] = mel[c * Tfull + t];
  let s = ctx.upload(x0, 1, Tfull * melBins), Hh = Tfull, Wd = melBins;
  // [cout, cin, k, stride, groups, act, isStride2]
  const conv = [
    [Csub, 1, 3, 2, 1, "relu", true], [Csub, Csub, 3, 2, Csub, "none", true], [Csub, Csub, 1, 1, 1, "relu", false],
    [Csub, Csub, 3, 2, Csub, "none", true], [Csub, Csub, 1, 1, 1, "relu", false],
  ];
  for (let i = 0; i < 5; i++) {
    const [cout, cin, k, st, gr, act, s2] = conv[i];
    const pt = s2 ? sp.t : 0, pb = s2 ? sp.b : 0, pl = s2 ? sp.l : 0, pr = s2 ? sp.r : 0;
    s = ctx.conv2d(s, enc.sub.conv[i].w, { cout, cin, h: Hh, w: Wd, kh: k, kw: k, bias: enc.sub.conv[i].b, strideH: st, strideW: st, padTop: pt, padBottom: pb, padLeft: pl, padRight: pr, groups: gr, act });
    Hh = Math.floor((Hh + pt + pb - k) / st) + 1; Wd = Math.floor((Wd + pl + pr - k) / st) + 1;
  }
  const Tsub = Hh, F = Wd;
  const flat = ctx.subReshape(s, Csub, Tsub, F);
  let x = ctx.matmul(flat, enc.sub.linw, { bias: enc.sub.linb });

  enc._pe = enc._pe || new Map();
  let peT = enc._pe.get(Tsub);
  if (!peT) { peT = ctx.upload(posEncoding(Tsub, D), 2 * Tsub - 1, D); enc._pe.set(Tsub, peT); }
  // Cache-aware streaming attention mask (EOU): chunk-limited on BOTH sides. With
  // chunkStart = chunk*floor(i/chunk), query i attends keys j in
  // [max(0, chunkStart-left), min(Tsub-1, chunkStart+chunk-1)]; -10000 elsewhere
  // (added to scores pre-softmax). left = attLeft (70 frames for EOU).
  let maskT = null;
  if (enc.cfg.attChunk) {
    enc._mask = enc._mask || new Map();
    maskT = enc._mask.get(Tsub);
    if (!maskT) {
      const C = enc.cfg.attChunk, LEFT = enc.cfg.attLeft ?? 70, mk = new Float32Array(Tsub * Tsub);
      for (let i = 0; i < Tsub; i++) {
        const cs = C * Math.floor(i / C);
        const lo = Math.max(0, cs - LEFT), hi = Math.min(Tsub - 1, cs + C - 1);
        for (let j = 0; j < Tsub; j++) mk[i * Tsub + j] = j >= lo && j <= hi ? 0 : -10000;
      }
      maskT = ctx.upload(mk, Tsub, Tsub); enc._mask.set(Tsub, maskT);
    }
  }
  // Depthwise conv module pad: symmetric (Parakeet) or causal (EOU streaming: all
  // pad on the left, none on the right).
  const dwPadL = enc.cfg.convCausal ? dwK - 1 : (dwK - 1) >> 1;
  const dwPadR = enc.cfg.convCausal ? 0 : (dwK - 1) >> 1;
  const ff = (x, lp, w1, w2) => ctx.add(x, ctx.matmul(ctx.matmul(ln(x, lp), w1, { act: "silu" }), w2));

  for (let L = 0; L < LAYERS; L++) {
    const w = enc.layers[L];
    x = ff(x, w.lnff1, w.ff1w1, w.ff1w2);
    const xln = ln(x, w.lnatt);
    const q = ctx.matmul(xln, w.q), k = ctx.matmul(xln, w.k), v = ctx.matmul(xln, w.v);
    const p = ctx.matmul(peT, w.pos);
    const outc = ctx.alloc(Tsub, D);
    for (let h = 0; h < H; h++) {
      const qh = ctx.sliceCols(q, h * HD, HD), kh = ctx.sliceCols(k, h * HD, HD);
      const vh = ctx.sliceCols(v, h * HD, HD), ph = ctx.sliceCols(p, h * HD, HD);
      const qu = ctx.add(qh, w.pbuT[h]);
      const qv = ctx.add(qh, w.pbvT[h]);
      const ac = ctx.matmul(qu, ctx.transpose(kh));
      const bd = ctx.relShift(ctx.matmul(qv, ctx.transpose(ph)));
      let sc = ctx.add(ac, bd);
      if (maskT) sc = ctx.add(sc, maskT);
      const probs = ctx.softmax(sc);
      ctx.setCols(outc, ctx.matmul(probs, vh), h * HD);
    }
    x = ctx.add(x, ctx.matmul(outc, w.out));
    const hT = ctx.transpose(ln(x, w.lnconv));
    const glu = ctx.glu(ctx.conv1d(hT, w.pw1, { cout: 2 * D, k: 1 }));
    let dwo;
    if (w.bn) {
      // EOU: depthwise (no act) → LayerNorm over channels → SiLU. [D,Tsub]→[Tsub,D]→LN→[D,Tsub]
      const d = ctx.conv1d(glu, w.dw, { cout: D, k: dwK, groups: D, padLeft: dwPadL, padRight: dwPadR, bias: w.dwb });
      dwo = ctx.transpose(ctx.silu(ln(ctx.transpose(d), w.bn)));
    } else {
      // Parakeet: BatchNorm folded into depthwise, SiLU fused.
      dwo = ctx.conv1d(glu, w.dw, { cout: D, k: dwK, groups: D, padLeft: dwPadL, padRight: dwPadR, bias: w.dwb, act: "silu" });
    }
    x = ctx.add(x, ctx.transpose(ctx.conv1d(dwo, w.pw2, { cout: D, k: 1 })));
    x = ff(x, w.lnff2, w.ff2w1, w.ff2w2);
    x = ln(x, w.lnout);
  }
  const out = { dims: [1, D, Tsub], framesGpu: x, Tsub };
  if (wantData) out.data = await ctx.download(ctx.transpose(x));
  return out;
}
