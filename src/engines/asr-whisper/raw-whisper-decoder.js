// Whisper decoder — hand-written raw WebGPU (no onnxruntime). Autoregressive,
// no KV cache (recomputes the prefix each step — fine for ≤~task-length seqs;
// cross-attn K/V are computed ONCE from the encoder and reused). Each layer:
// PRE-LN causal self-attn → PRE-LN cross-attn(to encoder) → PRE-LN erf-GELU FFN.
// Final LN, logits tied to embed_tokens. 8-head MHA, scale 1/√64.

const D = 512,
  NH = 8,
  HD = 64,
  VOCAB = 51865;
const SCALE = 1 / Math.sqrt(HD);

export function loadWhisperDecoder(ctx, bin, man) {
  const g = (k) => bin.subarray(man[k].offset, man[k].offset + man[k].len);
  const mat = (k) => ctx.upload(g(k).slice(), man[k].dims[0], man[k].dims[1]);
  const matSc = (k, s) => {
    const a = g(k).slice();
    for (let i = 0; i < a.length; i++) a[i] *= s;
    return ctx.upload(a, man[k].dims[0], man[k].dims[1]);
  };
  const vec = (k) => ctx.upload(g(k).slice(), 1, man[k].len);
  const vecSc = (k, s) => {
    const a = g(k).slice();
    for (let i = 0; i < a.length; i++) a[i] *= s;
    return ctx.upload(a, 1, man[k].len);
  };
  const layers = [];
  const nl = Object.keys(man).filter((k) => /^L\d+_sqw$/.test(k)).length;
  for (let L = 0; L < nl; L++) {
    const t = (s) => `L${L}_${s}`;
    layers.push({
      sqw: matSc(t("sqw"), SCALE),
      sqb: vecSc(t("sqb"), SCALE),
      skw: mat(t("skw")),
      svw: mat(t("svw")),
      svb: vec(t("svb")),
      sow: mat(t("sow")),
      sob: vec(t("sob")),
      cqw: matSc(t("cqw"), SCALE),
      cqb: vecSc(t("cqb"), SCALE),
      ckw: mat(t("ckw")),
      cvw: mat(t("cvw")),
      cvb: vec(t("cvb")),
      cow: mat(t("cow")),
      cob: vec(t("cob")),
      ln1: [vec(t("ln1w")), vec(t("ln1b"))],
      ln2: [vec(t("ln2w")), vec(t("ln2b"))],
      ln3: [vec(t("ln3w")), vec(t("ln3b"))],
      f1w: mat(t("f1w")),
      f1b: vec(t("f1b")),
      f2w: mat(t("f2w")),
      f2b: vec(t("f2b")),
    });
  }
  // embed on CPU (row-gather per token); embedT[D,VOCAB] on GPU for the tied vocab proj.
  const embed = g("embed").slice(); // [51865,512]
  const embedT = ctx.transpose(ctx.upload(embed.slice(), VOCAB, D)); // [512,51865]
  return { layers, embed, embedT, pos: g("pos").slice(), lnf: [vec("lnf_w"), vec("lnf_b")] };
}

/** Precompute per-layer cross-attn K/V from the encoder output (once per clip). */
export function whisperCrossKV(ctx, dec, encGpu) {
  return dec.layers.map((w) => ({ k: ctx.matmul(encGpu, w.ckw), v: ctx.matmul(encGpu, w.cvw, { bias: w.cvb }) }));
}

/**
 * KV-cached decode state: per-layer self-attn K/V preallocated [maxLen, D];
 * each step feeds ONE token (the old path recomputed the whole prefix AND the
 * [n,512]@[512,51865] vocab projection over all positions — O(n^2) with a huge
 * constant; cached is O(1) per step).
 */
export function whisperDecodeInit(ctx, dec, maxLen = 448) {
  return {
    n: 0,
    maxLen,
    selfK: dec.layers.map(() => ctx.alloc(maxLen, D)),
    selfV: dec.layers.map(() => ctx.alloc(maxLen, D)),
  };
}

// rows-limited view over a preallocated cache tensor (no copy; both backends).
const rowsView = (t, rows) => (t.buf ? { buf: t.buf, rows, cols: t.cols } : { data: t.data.subarray(0, rows * t.cols), rows, cols: t.cols });

/** Feed one token through the cached decoder; returns logits [VOCAB] for it. */
export async function whisperDecodeNext(ctx, dec, kv, st, token) {
  const ln = (x, lp) => ctx.layernorm(x, lp[0], lp[1]);
  const n = st.n;
  const emb = new Float32Array(D);
  for (let d = 0; d < D; d++) emb[d] = dec.embed[token * D + d] + dec.pos[n * D + d];
  let x = ctx.upload(emb, 1, D);
  if (ctx.beginBatch) ctx.beginBatch(); // one submit for the whole step
  for (let li = 0; li < dec.layers.length; li++) {
    const w = dec.layers[li];
    // causal self-attn against the cache (only past+current exist -> no mask)
    let h = ln(x, w.ln1);
    const q = ctx.matmul(h, w.sqw, { bias: w.sqb });
    ctx.copyRows(st.selfK[li], ctx.matmul(h, w.skw), n);
    ctx.copyRows(st.selfV[li], ctx.matmul(h, w.svw, { bias: w.svb }), n);
    const K = rowsView(st.selfK[li], n + 1),
      V = rowsView(st.selfV[li], n + 1);
    const probs = ctx.softmax(ctx.bmmQK(q, K, null, NH, HD)); // [NH, n+1]
    x = ctx.add(x, ctx.matmul(ctx.bmmPV(probs, V, NH, HD), w.sow, { bias: w.sob }));
    // cross-attn (K/V precomputed once from the encoder)
    h = ln(x, w.ln2);
    const cq = ctx.matmul(h, w.cqw, { bias: w.cqb });
    const cprobs = ctx.softmax(ctx.bmmQK(cq, kv[li].k, null, NH, HD)); // [NH, Tenc]
    x = ctx.add(x, ctx.matmul(ctx.bmmPV(cprobs, kv[li].v, NH, HD), w.cow, { bias: w.cob }));
    // FFN
    h = ln(x, w.ln3);
    x = ctx.add(x, ctx.matmul(ctx.matmul(h, w.f1w, { bias: w.f1b, act: "gelu_erf" }), w.f2w, { bias: w.f2b }));
  }
  st.n = n + 1;
  const logits = ctx.matmul(ln(x, dec.lnf), dec.embedT); // [1, VOCAB]
  if (ctx.endBatch) ctx.endBatch();
  return await ctx.download(logits);
}

/** One decoder forward over tokens[]; returns Float32Array logits for the LAST position. */
export async function whisperDecodeStep(ctx, dec, kv, tokens) {
  const ln = (x, lp) => ctx.layernorm(x, lp[0], lp[1]);
  const n = tokens.length;
  // embed + positional (CPU gather → upload)
  const emb = new Float32Array(n * D);
  for (let i = 0; i < n; i++) {
    const t = tokens[i];
    for (let d = 0; d < D; d++) emb[i * D + d] = dec.embed[t * D + d] + dec.pos[i * D + d];
  }
  let x = ctx.upload(emb, n, D);
  // causal mask [n,n]: j>i → -inf
  const mk = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) mk[i * n + j] = j <= i ? 0 : -1e9;
  const maskT = ctx.upload(mk, n, n);
  const Tenc = kv[0].k.rows;
  for (let li = 0; li < dec.layers.length; li++) {
    const w = dec.layers[li];
    // causal self-attn
    let h = ln(x, w.ln1);
    const q = ctx.matmul(h, w.sqw, { bias: w.sqb }),
      k = ctx.matmul(h, w.skw),
      v = ctx.matmul(h, w.svw, { bias: w.svb });
    let outc = ctx.alloc(n, D);
    for (let hd = 0; hd < NH; hd++) {
      const qh = ctx.sliceCols(q, hd * HD, HD),
        kh = ctx.sliceCols(k, hd * HD, HD),
        vh = ctx.sliceCols(v, hd * HD, HD);
      const probs = ctx.softmax(ctx.add(ctx.matmul(qh, ctx.transpose(kh)), maskT));
      ctx.setCols(outc, ctx.matmul(probs, vh), hd * HD);
    }
    x = ctx.add(x, ctx.matmul(outc, w.sow, { bias: w.sob }));
    // cross-attn (K/V precomputed)
    h = ln(x, w.ln2);
    const cq = ctx.matmul(h, w.cqw, { bias: w.cqb });
    outc = ctx.alloc(n, D);
    for (let hd = 0; hd < NH; hd++) {
      const qh = ctx.sliceCols(cq, hd * HD, HD),
        kh = ctx.sliceCols(kv[li].k, hd * HD, HD),
        vh = ctx.sliceCols(kv[li].v, hd * HD, HD);
      const probs = ctx.softmax(ctx.matmul(qh, ctx.transpose(kh))); // [n,Tenc]
      ctx.setCols(outc, ctx.matmul(probs, vh), hd * HD);
    }
    x = ctx.add(x, ctx.matmul(outc, w.cow, { bias: w.cob }));
    // FFN
    h = ln(x, w.ln3);
    x = ctx.add(x, ctx.matmul(ctx.matmul(h, w.f1w, { bias: w.f1b, act: "gelu_erf" }), w.f2w, { bias: w.f2b }));
  }
  x = ln(x, dec.lnf);
  const all = await ctx.download(ctx.matmul(x, dec.embedT)); // [n, VOCAB]
  return all.slice((n - 1) * VOCAB, n * VOCAB); // last-position logits
}
