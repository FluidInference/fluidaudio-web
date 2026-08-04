// Sortformer post-encoder head — hand-written raw WebGPU (no onnxruntime).
// FastConformer encoder output [Tsub,512] → encoder_proj (512→192) → 18-layer
// STANDARD transformer (POST-LN, 8-head MHA no rel-pos, ReLU FFN 192→768→192) →
// ReLU → first_hidden_to_hidden (192→192) → ReLU → single_hidden_to_spks (192→4)
// → sigmoid → per-frame 4-speaker probs. Validated exact (maxΔ 0.0) vs ORT.
// Weights are anonymous ONNX MatMuls traced via their named biases at extract time.

const D = 192, NH = 8, HD = 24, SPK = 4;
const SCALE = 1 / Math.sqrt(HD);

export function loadSortformerHead(ctx, bin, man) {
  const g = (k) => bin.subarray(man[k].offset, man[k].offset + man[k].len);
  const mat = (k) => ctx.upload(g(k).slice(), man[k].dims[0], man[k].dims[1]);
  // fold the 1/sqrt(head_dim) attention scale into the query weight (like the encoder).
  const matScaled = (k, s) => { const a = g(k).slice(); for (let i = 0; i < a.length; i++) a[i] *= s; return ctx.upload(a, man[k].dims[0], man[k].dims[1]); };
  const vec = (k) => ctx.upload(g(k).slice(), 1, man[k].len);
  const vecScaled = (k, s) => { const a = g(k).slice(); for (let i = 0; i < a.length; i++) a[i] *= s; return ctx.upload(a, 1, man[k].len); };
  const layers = [];
  const nl = Object.keys(man).filter((k) => /^T\d+_qw$/.test(k)).length;
  for (let L = 0; L < nl; L++) {
    const t = (s) => `T${L}_${s}`;
    layers.push({
      qw: matScaled(t("qw"), SCALE), qb: vecScaled(t("qb"), SCALE), kw: mat(t("kw")), kb: vec(t("kb")), vw: mat(t("vw")), vb: vec(t("vb")),
      ow: mat(t("ow")), ob: vec(t("ob")),
      ln1: [vec(t("ln1w")), vec(t("ln1b"))], ln2: [vec(t("ln2w")), vec(t("ln2b"))],
      dinw: mat(t("dinw")), dinb: vec(t("dinb")), doutw: mat(t("doutw")), doutb: vec(t("doutb")),
    });
  }
  return {
    projw: mat("encoder_proj_w"), projb: vec("encoder_proj_b"),
    fhhw: mat("fhh_w"), fhhb: vec("fhh_b"), spksw: mat("spks_w"), spksb: vec("spks_b"),
    layers,
  };
}

/** framesGpu [Tsub,512] → Float32Array preds [Tsub*4] (per-frame speaker probs). */
export async function sortformerHead(ctx, head, framesGpu, Tsub) {
  const ln = (x, lp) => ctx.layernorm(x, lp[0], lp[1]);
  let x = ctx.matmul(framesGpu, head.projw, { bias: head.projb }); // [Tsub,192]
  for (const w of head.layers) {
    // POST-LN block: sub-layers read x directly (no pre-LN); LN applied to residual+sublayer.
    const q = ctx.matmul(x, w.qw, { bias: w.qb }), k = ctx.matmul(x, w.kw, { bias: w.kb }), v = ctx.matmul(x, w.vw, { bias: w.vb });
    const outc = ctx.alloc(Tsub, D);
    for (let h = 0; h < NH; h++) {
      const qh = ctx.sliceCols(q, h * HD, HD), kh = ctx.sliceCols(k, h * HD, HD), vh = ctx.sliceCols(v, h * HD, HD);
      const probs = ctx.softmax(ctx.matmul(qh, ctx.transpose(kh))); // scale folded into qw
      ctx.setCols(outc, ctx.matmul(probs, vh), h * HD);
    }
    x = ln(ctx.add(x, ctx.matmul(outc, w.ow, { bias: w.ob })), w.ln1);
    const ffn = ctx.matmul(ctx.matmul(x, w.dinw, { bias: w.dinb, act: "relu" }), w.doutw, { bias: w.doutb });
    x = ln(ctx.add(x, ffn), w.ln2);
  }
  const h1 = ctx.matmul(ctx.relu(x), head.fhhw, { bias: head.fhhb, act: "relu" });
  const logits = await ctx.download(ctx.matmul(h1, head.spksw, { bias: head.spksb })); // [Tsub*4]
  const preds = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) preds[i] = 1 / (1 + Math.exp(-logits[i])); // sigmoid
  return preds;
}
