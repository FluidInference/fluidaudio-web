// Whisper encoder — hand-written raw WebGPU (no onnxruntime).
// mel[80,3000] → conv1(80→512,k3,s1)+GELU → conv2(512→512,k3,s2)+GELU → [512,1500]
// → +sinusoidal pos → 6 PRE-LN transformer layers (8-head MHA no-rel-pos scale
// 1/√64, GELU FFN 512→2048→512) → final LN → [1500,512]. Weights: anonymous ONNX
// MatMuls traced by node name at extract time. Linear q/k/v layout is [in,out].

const D = 512, NH = 8, HD = 64, TENC = 1500;
const SCALE = 1 / Math.sqrt(HD);

export function loadWhisperEncoder(ctx, bin, man) {
  const g = (k) => bin.subarray(man[k].offset, man[k].offset + man[k].len);
  const mat = (k) => ctx.upload(g(k).slice(), man[k].dims[0], man[k].dims[1]);
  const matSc = (k, s) => { const a = g(k).slice(); for (let i = 0; i < a.length; i++) a[i] *= s; return ctx.upload(a, man[k].dims[0], man[k].dims[1]); };
  const vec = (k) => ctx.upload(g(k).slice(), 1, man[k].len);
  const vecSc = (k, s) => { const a = g(k).slice(); for (let i = 0; i < a.length; i++) a[i] *= s; return ctx.upload(a, 1, man[k].len); };
  const layers = [];
  const nl = Object.keys(man).filter((k) => /^L\d+_qw$/.test(k)).length;
  for (let L = 0; L < nl; L++) {
    const t = (s) => `L${L}_${s}`;
    layers.push({
      qw: matSc(t("qw"), SCALE), qb: vecSc(t("qb"), SCALE), kw: mat(t("kw")), vw: mat(t("vw")), vb: vec(t("vb")),
      ow: mat(t("ow")), ob: vec(t("ob")), ln1: [vec(t("ln1w")), vec(t("ln1b"))],
      f1w: mat(t("f1w")), f1b: vec(t("f1b")), f2w: mat(t("f2w")), f2b: vec(t("f2b")), ln2: [vec(t("ln2w")), vec(t("ln2b"))],
    });
  }
  return {
    // conv weights kept on CPU (conv1d uploads them via vec); posw uploaded once.
    conv1w: g("conv1w").slice(), conv1b: g("conv1b").slice(), conv2w: g("conv2w").slice(), conv2b: g("conv2b").slice(),
    posw: ctx.upload(g("posw").slice(), TENC, D), lnf: [vec("lnf_w"), vec("lnf_b")], layers,
  };
}

/** mel: Float32Array[80*3000] (channel-major mel[c*3000+t]) → GpuTensor [1500,512]. */
export function whisperEncode(ctx, enc, mel) {
  const ln = (x, lp) => ctx.layernorm(x, lp[0], lp[1]);
  const melG = ctx.upload(mel, 80, 3000);
  const w1 = ctx.upload(enc.conv1w, 512, 80 * 3), b1 = ctx.upload(enc.conv1b, 1, 512);
  const w2 = ctx.upload(enc.conv2w, 512, 512 * 3), b2 = ctx.upload(enc.conv2b, 1, 512);
  let c = ctx.conv1d(melG, w1, { cout: 512, k: 3, pad: 1, bias: b1, act: "gelu_erf" });      // [512,3000]
  c = ctx.conv1d(c, w2, { cout: 512, k: 3, stride: 2, pad: 1, bias: b2, act: "gelu_erf" });   // [512,1500]
  let x = ctx.add(ctx.transpose(c), enc.posw);
  for (const w of enc.layers) {
    const h = ln(x, w.ln1);
    const q = ctx.matmul(h, w.qw, { bias: w.qb }), k = ctx.matmul(h, w.kw), v = ctx.matmul(h, w.vw, { bias: w.vb });
    const outc = ctx.alloc(TENC, D);
    for (let hd = 0; hd < NH; hd++) {
      const qh = ctx.sliceCols(q, hd * HD, HD), kh = ctx.sliceCols(k, hd * HD, HD), vh = ctx.sliceCols(v, hd * HD, HD);
      const probs = ctx.softmax(ctx.matmul(qh, ctx.transpose(kh))); // scale folded into qw
      ctx.setCols(outc, ctx.matmul(probs, vh), hd * HD);
    }
    x = ctx.add(x, ctx.matmul(outc, w.ow, { bias: w.ob }));
    const h2 = ln(x, w.ln2);
    x = ctx.add(x, ctx.matmul(ctx.matmul(h2, w.f1w, { bias: w.f1b, act: "gelu_erf" }), w.f2w, { bias: w.f2b }));
  }
  return ln(x, enc.lnf);
}
