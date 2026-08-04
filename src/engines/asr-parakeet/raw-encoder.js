// Parakeet TDT v3 encoder — hand-written raw-WebGPU FastConformer (no onnxruntime).
// Runs on the src/gpu/compute.js kernels. Parity vs ORT: full 24-layer output
// 5.3e-7 (scripts/smoke-parakeet-encoder-raw.mjs).
//
// Arch: mel[1,128,T] → dw-striding 8× subsampling (Conv2d) → Linear→1024 → 24×
// conformer blocks (macaron FF ½ · rel-pos MHA · conv module · FF ½ · norm_out) →
// [1,1024,T/8]. d_model 1024, 8 heads×128, d_ff 4096, LN eps 1e-5.
//
// Constants folded into weights at load: q·(1/√128), pos_bias_u/v·(1/√128),
// FF output projections·0.5. rel_shift is done on host (a small index remap).

const D = 1024, H = 8, HD = 128, DFF = 4096, LAYERS = 24, INV = 1 / Math.sqrt(HD);

/**
 * Upload all encoder weights to GPU once. `bin` is a Float32Array, `man` the
 * manifest {key:{dims,offset,len}} from extract-parakeet-encoder.py.
 * Returns a handle consumed by `encode`.
 */
export function loadParakeetEncoder(ctx, bin, man) {
  // Two manifest formats: fp32 (bin=Float32Array, entries have offset/len) and int8
  // (bin=Uint8Array/ArrayBuffer, entries have dtype "i8" + i8ByteOffset/scaleOffset).
  // int8 is per-channel symmetric, dequantized to fp32 per-tensor at load (existing
  // fp32 kernels). matmul weights: per output col; conv weights: per Cout group.
  const int8 = Object.values(man).some((m) => m.dtype === "i8");
  let f32v, i8v;
  if (int8) { const ab = bin.buffer instanceof ArrayBuffer ? bin.buffer : bin; f32v = new Float32Array(ab); i8v = new Int8Array(ab); }
  const raw = (k) => {
    const m = man[k];
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
  const mat = (k) => ctx.upload(raw(k).slice(), man[k].dims[0], man[k].dims[1]); // [in,out]
  const vec = (k) => ctx.upload(raw(k).slice(), 1, man[k].count ?? man[k].len);
  const matScaled = (k, s) => ctx.upload(scaled(k, s), man[k].dims[0], man[k].dims[1]);

  const sub = {
    conv: [0, 1, 2, 3, 4].map((i) => ({ w: vec(`c${i}w`), b: vec(`c${i}b`) })),
    linw: mat("linw"), linb: vec("linb"),
  };
  const layers = [];
  for (let L = 0; L < LAYERS; L++) {
    const g = (s) => `L${L}_${s}`;
    layers.push({
      lnff1: [vec(g("lnff1_w")), vec(g("lnff1_b"))], ff1w1: mat(g("ff1w1")), ff1w2: matScaled(g("ff1w2"), 0.5),
      lnatt: [vec(g("lnatt_w")), vec(g("lnatt_b"))],
      q: matScaled(g("q"), INV), k: mat(g("k")), v: mat(g("v")), pos: mat(g("pos")), out: mat(g("out")),
      pbu: scaled(g("pbu"), INV), pbv: scaled(g("pbv"), INV), // kept on host, sliced per head
      lnconv: [vec(g("lnconv_w")), vec(g("lnconv_b"))],
      pw1: vec(g("pw1")), dw: vec(g("dw")), dwb: vec(g("dwb")), pw2: vec(g("pw2")),
      lnff2: [vec(g("lnff2_w")), vec(g("lnff2_b"))], ff2w1: mat(g("ff2w1")), ff2w2: matScaled(g("ff2w2"), 0.5),
      lnout: [vec(g("lnout_w")), vec(g("lnout_b"))],
    });
  }
  return { sub, layers };
}

function posEncoding(Tsub) {
  const pe = new Float32Array((2 * Tsub - 1) * D);
  const dv = (i) => Math.exp(i * -(Math.log(10000) / D));
  for (let pi = 0; pi < 2 * Tsub - 1; pi++) {
    const pos = (Tsub - 1) - pi;
    for (let i = 0; i < D; i += 2) { pe[pi * D + i] = Math.sin(pos * dv(i)); pe[pi * D + i + 1] = Math.cos(pos * dv(i)); }
  }
  return pe;
}

/**
 * Run the encoder. `mel`: Float32Array[128*T] (channel-major, mel[c*T+t]).
 * Returns { data: Float32Array, dims: [1,1024,Tsub] } (encoder output, downloaded).
 */
export async function parakeetEncode(ctx, enc, mel, T) {
  const ln = (x, lp) => ctx.layernorm(x, lp[0], lp[1]);
  // --- subsampling: mel[128,T] -> conv2d input [1,T*128] (x[t*128+c]) ---
  const x0 = new Float32Array(T * 128);
  for (let t = 0; t < T; t++) for (let c = 0; c < 128; c++) x0[t * 128 + c] = mel[c * T + t];
  let s = ctx.upload(x0, 1, T * 128), Hh = T, Wd = 128;
  const cfg = [
    [256, 1, 3, 2, 1, 1, "relu"], [256, 256, 3, 2, 1, 256, "none"], [256, 256, 1, 1, 0, 1, "relu"],
    [256, 256, 3, 2, 1, 256, "none"], [256, 256, 1, 1, 0, 1, "relu"],
  ];
  for (let i = 0; i < 5; i++) {
    const [cout, cin, k, st, pad, gr, act] = cfg[i];
    s = ctx.conv2d(s, enc.sub.conv[i].w, { cout, cin, h: Hh, w: Wd, kh: k, kw: k, bias: enc.sub.conv[i].b, strideH: st, strideW: st, padH: pad, padW: pad, groups: gr, act });
    Hh = Math.floor((Hh + 2 * pad - k) / st) + 1; Wd = Math.floor((Wd + 2 * pad - k) / st) + 1;
  }
  const Tsub = Hh, F = Wd;
  const sd = await ctx.download(s); // [256, Tsub*F] -> [Tsub, 256*F]
  const flat = new Float32Array(Tsub * 256 * F);
  for (let c = 0; c < 256; c++) for (let ho = 0; ho < Tsub; ho++) for (let wo = 0; wo < F; wo++)
    flat[ho * (256 * F) + c * F + wo] = sd[c * (Tsub * F) + ho * F + wo];
  let x = ctx.matmul(ctx.upload(flat, Tsub, 256 * F), enc.sub.linw, { bias: enc.sub.linb });

  const peT = ctx.upload(posEncoding(Tsub), 2 * Tsub - 1, D);
  const ff = (x, lp, w1, w2) => ctx.add(x, ctx.matmul(ctx.matmul(ln(x, lp), w1, { act: "silu" }), w2));

  for (let L = 0; L < LAYERS; L++) {
    const w = enc.layers[L];
    x = ff(x, w.lnff1, w.ff1w1, w.ff1w2);
    // rel-pos attention
    const xln = ln(x, w.lnatt);
    const q = ctx.matmul(xln, w.q), k = ctx.matmul(xln, w.k), v = ctx.matmul(xln, w.v);
    const p = ctx.matmul(peT, w.pos);
    const outc = ctx.alloc(Tsub, D);
    for (let h = 0; h < H; h++) {
      const qh = ctx.sliceCols(q, h * HD, HD), kh = ctx.sliceCols(k, h * HD, HD);
      const vh = ctx.sliceCols(v, h * HD, HD), ph = ctx.sliceCols(p, h * HD, HD);
      const qu = ctx.add(qh, ctx.upload(w.pbu.slice(h * HD, h * HD + HD), 1, HD));
      const qv = ctx.add(qh, ctx.upload(w.pbv.slice(h * HD, h * HD + HD), 1, HD));
      const ac = ctx.matmul(qu, ctx.transpose(kh));
      const bd = ctx.relShift(ctx.matmul(qv, ctx.transpose(ph))); // GPU, no roundtrip
      const probs = ctx.softmax(ctx.add(ac, bd));
      ctx.setCols(outc, ctx.matmul(probs, vh), h * HD);
    }
    x = ctx.add(x, ctx.matmul(outc, w.out));
    // conv module
    const hT = ctx.transpose(ln(x, w.lnconv));
    const glu = ctx.glu(ctx.conv1d(hT, w.pw1, { cout: 2 * D, k: 1 }));
    const dwo = ctx.conv1d(glu, w.dw, { cout: D, k: 9, groups: D, pad: 4, bias: w.dwb, act: "silu" });
    x = ctx.add(x, ctx.transpose(ctx.conv1d(dwo, w.pw2, { cout: D, k: 1 })));
    x = ff(x, w.lnff2, w.ff2w1, w.ff2w2);
    x = ln(x, w.lnout);
  }
  // x is [Tsub, D] (frames × d_model) — the GPU decoder consumes it directly.
  // Also return the transposed [1,D,Tsub] download for the ORT-parity path/tests.
  const outT = ctx.transpose(x);
  return { data: await ctx.download(outT), dims: [1, D, Tsub], framesGpu: x, Tsub };
}
