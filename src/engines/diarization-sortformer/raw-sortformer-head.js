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
    // batched over all heads (tiled bmm kernels; scale folded into qw)
    const probs = ctx.softmax(ctx.bmmQK(q, k, null, NH, HD)); // [NH*T, T]
    const outc = ctx.bmmPV(probs, v, NH, HD);                  // [T, NH*HD]
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

/** preds[frames*SPK] → diarization segments (per-speaker threshold, merge gaps, drop shorts). */
export function predsToSegments(preds, frames, frameSec, { threshold = 0.5, minSpeechSec = 0.25, mergeGapSec = 0.25 } = {}) {
  const segments = [];
  for (let s = 0; s < SPK; s++) {
    let start = -1;
    for (let t = 0; t <= frames; t++) {
      const on = t < frames && preds[t * SPK + s] >= threshold;
      if (on && start < 0) start = t;
      if (!on && start >= 0) { segments.push({ speaker: s, start: start * frameSec, end: t * frameSec }); start = -1; }
    }
  }
  const bySpk = new Map();
  for (const seg of segments) (bySpk.get(seg.speaker) ?? bySpk.set(seg.speaker, []).get(seg.speaker)).push(seg);
  const out = [];
  for (const [, list] of bySpk) {
    list.sort((a, b) => a.start - b.start);
    let cur = null;
    for (const seg of list) {
      if (cur && seg.start - cur.end <= mergeGapSec) cur.end = seg.end;
      else { if (cur) out.push(cur); cur = { ...seg }; }
    }
    if (cur) out.push(cur);
  }
  return out.filter((s) => s.end - s.start >= minSpeechSec).sort((a, b) => a.start - b.start);
}
