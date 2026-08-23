// VoiceChat-11B TTS core — Gemma3 backbone (28L×1152, KV-cached, CFG batch of 2)
// + CAS t5gemma text conditioning + gated audio/text fusion + MoG head with
// 8-iteration PRVQ unmasking. Mirrors Speech/…/modules/ear_tts_model.py and the
// HF Gemma3/T5Gemma eager paths (transformers 5.15.1 — effective config baked
// by scripts/extract-voicechat-tts.py).
//
// Split of labor: all big GEMMs ride the ComputeContext (WASM SIMD / WebGPU);
// the tiny per-row ops (RMSNorm, RoPE, per-head attention over the CPU KV
// cache, MoG component selection, PRVQ nearest-codebook search) are plain-JS
// f32 with f64 accumulation — decision margins (mixture argmax, PRVQ argmin)
// live here, which is why the MoG/PRVQ weights ship f32.
//
// Parity oracle: scripts/ci-smoke-voicechat-tts.mjs (torch goldens from
// scripts/voicechat-tts-reference.py).

const H = 16;
const HD = 72;
const D = 1152;
const HALF = HD / 2;

// ── tiny CPU kernels ─────────────────────────────────────────────────────────
/** Gemma RMSNorm rows in place-free: y = x * rsqrt(mean(x²)+eps) * (1+w). */
function rmsRows(x, rows, cols, w, eps) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const o = r * cols;
    let s = 0;
    for (let j = 0; j < cols; j++) s += x[o + j] * x[o + j];
    const inv = 1 / Math.sqrt(s / cols + eps);
    for (let j = 0; j < cols; j++) out[o + j] = x[o + j] * inv * (1 + w[j]);
  }
  return out;
}

/** Per-head RMSNorm (dim 72) over a [rows, H*HD] projection, in place. */
function headRms(x, rows, w, eps) {
  for (let r = 0; r < rows; r++)
    for (let h = 0; h < H; h++) {
      const o = r * D + h * HD;
      let s = 0;
      for (let j = 0; j < HD; j++) s += x[o + j] * x[o + j];
      const inv = 1 / Math.sqrt(s / HD + eps);
      for (let j = 0; j < HD; j++) x[o + j] = x[o + j] * inv * (1 + w[j]);
    }
}

/** RoPE (rotate-half over [freqs,freqs]) on a [rows, H*HD] projection, in place. */
function rope(x, rows, positions, invFreq, scale) {
  for (let r = 0; r < rows; r++) {
    const p = positions[r];
    for (let h = 0; h < H; h++) {
      const o = r * D + h * HD;
      for (let i = 0; i < HALF; i++) {
        const f = p * invFreq[i],
          c = Math.cos(f),
          s = Math.sin(f);
        const a = x[o + i],
          b = x[o + HALF + i];
        x[o + i] = (a * c - b * s) * scale;
        x[o + HALF + i] = (b * c + a * s) * scale;
      }
    }
  }
}

/**
 * Causal attention of `rows` new positions against a CPU KV cache (per head).
 * q already normed+roped+scaled; kv cache rows 0..pos0+i inclusive are valid.
 * softcap (CAS) applies 50·tanh(s/50) before softmax.
 */
function attendCPU(q, rows, K, V, pos0, softcap, fixedT = 0) {
  const out = new Float32Array(rows * D);
  const scores = new Float32Array(fixedT || pos0 + rows);
  for (let r = 0; r < rows; r++) {
    const T = fixedT || pos0 + r + 1;
    for (let h = 0; h < H; h++) {
      const qo = r * D + h * HD;
      let mx = -Infinity;
      for (let j = 0; j < T; j++) {
        const ko = j * D + h * HD;
        let s = 0;
        for (let d = 0; d < HD; d++) s += q[qo + d] * K[ko + d];
        if (softcap) s = softcap * Math.tanh(s / softcap);
        scores[j] = s;
        if (s > mx) mx = s;
      }
      let sum = 0;
      for (let j = 0; j < T; j++) {
        const e = Math.exp(scores[j] - mx);
        scores[j] = e;
        sum += e;
      }
      const oo = r * D + h * HD;
      for (let j = 0; j < T; j++) {
        const p = scores[j] / sum;
        const vo = j * D + h * HD;
        for (let d = 0; d < HD; d++) out[oo + d] += p * V[vo + d];
      }
    }
  }
  return out;
}

function addInto(dst, src) {
  for (let i = 0; i < dst.length; i++) dst[i] += src[i];
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** Seeded RNG for sampled mode: mulberry32 + Box-Muller + Gumbel. */
export function makeRng(seed) {
  let a = seed >>> 0;
  const u = () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    uniform: u,
    gauss() {
      let x = 0;
      while (x === 0) x = u();
      return Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * u());
    },
    gumbel() {
      return -Math.log(-Math.log(u() + 1e-8) + 1e-8);
    },
  };
}

// ── model assembly ───────────────────────────────────────────────────────────
export function loadVoicechatTtsModel(ctx, W, cfg) {
  const vec = (n) => W.f32(n);
  const layers = [];
  for (let L = 0; L < cfg.layers; L++) {
    layers.push({
      q: W.mat(ctx, `B${L}_q`),
      k: W.mat(ctx, `B${L}_k`),
      v: W.mat(ctx, `B${L}_v`),
      o: W.mat(ctx, `B${L}_o`),
      qn: vec(`B${L}_qn`),
      kn: vec(`B${L}_kn`),
      lnIn: vec(`B${L}_ln_in`),
      lnPostAtt: vec(`B${L}_ln_postatt`),
      lnPreFf: vec(`B${L}_ln_preff`),
      lnPostFf: vec(`B${L}_ln_postff`),
      gate: W.mat(ctx, `B${L}_gate`),
      up: W.mat(ctx, `B${L}_up`),
      down: W.mat(ctx, `B${L}_down`),
      global: cfg.globalLayers.includes(L),
    });
  }
  const invFreqFor = (theta) => {
    const f = new Float64Array(HALF);
    for (let i = 0; i < HALF; i++) f[i] = 1 / Math.pow(theta, (2 * i) / HD);
    return f;
  };
  const mog = { mlps: [], norm: vec("M_norm"), logs: vec("M_logs"), mus: vec("M_mus"), lowmat: vec("M_lowmat") };
  for (let i = 0; i < 3; i++)
    mog.mlps.push({
      pre: vec(`M${i}_pre`),
      post: vec(`M${i}_post`),
      gate: W.mat(ctx, `M${i}_gate`),
      up: W.mat(ctx, `M${i}_up`),
      down: W.mat(ctx, `M${i}_down`),
    });
  return {
    cfg,
    layers,
    bNorm: vec("B_norm"),
    invFreqLocal: invFreqFor(cfg.ropeThetaLocal),
    invFreqGlobal: invFreqFor(cfg.ropeThetaGlobal),
    cas: {
      charEmb: vec("cas_char_emb"),
      q: W.mat(ctx, "C0_q"),
      k: W.mat(ctx, "C0_k"),
      v: W.mat(ctx, "C0_v"),
      o: W.mat(ctx, "C0_o"),
      gate: W.mat(ctx, "C0_gate"),
      up: W.mat(ctx, "C0_up"),
      down: W.mat(ctx, "C0_down"),
      lnPreAtt: vec("C0_ln_preatt"),
      lnPostAtt: vec("C0_ln_postatt"),
      lnPreFf: vec("C0_ln_preff"),
      lnPostFf: vec("C0_ln_postff"),
      norm: vec("cas_norm"),
      proj: W.mat(ctx, "cas_proj"),
      contEmb: vec("cas_cont_emb"),
      specialEmb: vec("cas_special_emb"),
      contFlags: W.u8("cont_flags"),
      specialFlags: W.u8("special_flags"),
      invFreq: invFreqFor(cfg.cas.ropeTheta),
      cache: new Map(),
    },
    bosEmb: vec("bos_emb"),
    nullEmb: vec("null_emb"),
    embedCode: W.mat(ctx, "embed_code"),
    gf: {
      audioW: W.mat(ctx, "gf_audio_w"),
      audioB: ctx.upload(vec("gf_audio_b").slice(), 1, D),
      textW: W.mat(ctx, "gf_text_w"),
      textB: ctx.upload(vec("gf_text_b").slice(), 1, D),
      gate: vec("gf_gate"),
      res: vec("gf_res")[0],
      norm: vec("gf_norm"),
    },
    aria: vec("aria_latent"),
    mog: { ...mog, logits: W.mat(ctx, "M_logits"), else: W.mat(ctx, "M_else") },
  };
}

// ── backbone chunk forward (M positions × 2 CFG streams) ─────────────────────
/**
 * @param {Float32Array} x [(2*M)*D] fused input embeds, stream-major (row s*M+i)
 * @param {number} pos0 absolute position of row i=0
 * @param {Array} caches per-layer {K:[2],V:[2]} Float32Array(maxT*D)
 * @returns {Promise<Float32Array>} hidden after final norm, same layout
 */
async function backboneChunk(ctx, model, x, M, pos0, caches) {
  const cfg = model.cfg;
  const rows = 2 * M;
  const positions = new Int32Array(rows);
  for (let s = 0; s < 2; s++) for (let i = 0; i < M; i++) positions[s * M + i] = pos0 + i;
  const mm = async (data, mat, opts) => await ctx.download(ctx.matmul(ctx.ensureTensor({ data, rows, cols: data.length / rows }), mat, opts));
  for (let L = 0; L < cfg.layers; L++) {
    const w = model.layers[L],
      c = caches[L];
    const invFreq = w.global ? model.invFreqGlobal : model.invFreqLocal;
    const xn = rmsRows(x, rows, D, w.lnIn, cfg.eps);
    const q = await mm(xn, w.q);
    const k = await mm(xn, w.k);
    const v = await mm(xn, w.v);
    headRms(q, rows, w.qn, cfg.eps);
    headRms(k, rows, w.kn, cfg.eps);
    rope(q, rows, positions, invFreq, cfg.attnScale);
    rope(k, rows, positions, invFreq, 1);
    const att = new Float32Array(rows * D);
    for (let s = 0; s < 2; s++) {
      c.K[s].set(k.subarray(s * M * D, (s + 1) * M * D), pos0 * D);
      c.V[s].set(v.subarray(s * M * D, (s + 1) * M * D), pos0 * D);
      att.set(attendCPU(q.subarray(s * M * D, (s + 1) * M * D), M, c.K[s], c.V[s], pos0, 0), s * M * D);
    }
    const o = await mm(att, w.o);
    addInto(x, rmsRows(o, rows, D, w.lnPostAtt, cfg.eps));
    const xn2 = rmsRows(x, rows, D, w.lnPreFf, cfg.eps);
    const g = await mm(xn2, w.gate, { act: "gelu" });
    const u = await mm(xn2, w.up);
    for (let i = 0; i < g.length; i++) g[i] *= u[i];
    const dwn = await mm(g, w.down);
    addInto(x, rmsRows(dwn, rows, D, w.lnPostFf, cfg.eps));
  }
  return rmsRows(x, rows, D, model.bNorm, cfg.eps);
}

// ── CAS conditioning (char-aware subword embedding, cached per token id) ─────
async function casForward(ctx, model, chars) {
  const cas = model.cas,
    cfg = model.cfg,
    n = chars.length;
  const x = new Float32Array(n * D);
  const normzr = cfg.cas.normalizer;
  for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) x[i * D + j] = cas.charEmb[chars[i] * D + j] * normzr;
  const mm = async (data, mat, opts) => await ctx.download(ctx.matmul(ctx.ensureTensor({ data, rows: n, cols: data.length / n }), mat, opts));
  const positions = new Int32Array(n);
  for (let i = 0; i < n; i++) positions[i] = i;
  // one t5gemma encoder layer, bidirectional, softcap 50, no q/k norms
  const xn = rmsRows(x, n, D, cas.lnPreAtt, cfg.eps);
  const q = await mm(xn, cas.q);
  const k = await mm(xn, cas.k);
  const v = await mm(xn, cas.v);
  rope(q, n, positions, cas.invFreq, cfg.cas.attnScale);
  rope(k, n, positions, cas.invFreq, 1);
  const att = attendCPU(q, n, k, v, 0, cfg.cas.softcap, n); // bidirectional: every row sees all n keys
  addInto(x, rmsRows(await mm(att, cas.o), n, D, cas.lnPostAtt, cfg.eps));
  const xn2 = rmsRows(x, n, D, cas.lnPreFf, cfg.eps);
  const g = await mm(xn2, cas.gate, { act: "gelu" });
  const u = await mm(xn2, cas.up);
  for (let i = 0; i < g.length; i++) g[i] *= u[i];
  addInto(x, rmsRows(await mm(g, cas.down), n, D, cas.lnPostFf, cfg.eps));
  const fin = rmsRows(x, n, D, cas.norm, cfg.eps);
  // mean-pool over chars → proj
  const pooled = new Float32Array(D);
  for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) pooled[j] += fin[i * D + j];
  for (let j = 0; j < D; j++) pooled[j] /= n;
  return await ctx.download(ctx.matmul(ctx.ensureTensor({ data: pooled, rows: 1, cols: D }), cas.proj));
}

/** cond vector for one subword id: CAS(chars) [if inCas] + flag embeddings. */
export async function casCond(ctx, model, tokenChars, tokId, inCas) {
  const cas = model.cas;
  let base;
  if (inCas && tokenChars.length > 0) {
    let cached = cas.cache.get(tokId);
    if (!cached) {
      cached = await casForward(ctx, model, tokenChars);
      cas.cache.set(tokId, cached);
    }
    base = cached.slice();
  } else {
    base = new Float32Array(D);
  }
  const cf = cas.contFlags[tokId],
    sf = cas.specialFlags[tokId];
  for (let j = 0; j < D; j++) base[j] += cas.contEmb[cf * D + j] + cas.specialEmb[sf * D + j];
  return base;
}

// ── fusion ───────────────────────────────────────────────────────────────────
/** GatedProjectedSumRMSNorm over R (audio,text) row pairs → fused [R*D]. */
async function gatedFusion(ctx, model, audio, text, R) {
  const gf = model.gf,
    cfg = model.cfg;
  const scaled = new Float32Array(audio.length);
  for (let i = 0; i < audio.length; i++) scaled[i] = audio[i] / cfg.numQuantizers;
  const a = await ctx.download(ctx.matmul(ctx.ensureTensor({ data: scaled, rows: R, cols: D }), gf.audioW, { bias: gf.audioB }));
  const t = await ctx.download(ctx.matmul(ctx.ensureTensor({ data: text, rows: R, cols: D }), gf.textW, { bias: gf.textB }));
  const res = sigmoid(gf.res);
  const h = new Float32Array(R * D);
  for (let r = 0; r < R; r++)
    for (let j = 0; j < D; j++) {
      const g = sigmoid(gf.gate[j]);
      h[r * D + j] = res * (g * a[r * D + j] + (1 - g) * t[r * D + j]);
    }
  return rmsRows(h, R, D, gf.norm, cfg.eps);
}

/** depthsum embedding of one code frame (id==1024 ⇒ zero contribution) → [512]. */
export function depthsum(rvq, code, latent, codebook) {
  const out = new Float32Array(latent);
  for (let q = 0; q < code.length; q++) {
    const id = code[q];
    if (id >= codebook) continue;
    const e = (q * codebook + id) * latent;
    for (let c = 0; c < latent; c++) out[c] += rvq[e + c];
  }
  return out;
}

// ── MoG generate_step (iterative PRVQ unmasking) ─────────────────────────────
async function mogMlpStack(ctx, model, x, rows) {
  const cfg = model.cfg;
  const mm = async (data, mat, opts) => await ctx.download(ctx.matmul(ctx.ensureTensor({ data, rows, cols: data.length / rows }), mat, opts));
  for (const l of model.mog.mlps) {
    const y0 = rmsRows(x, rows, D, l.pre, cfg.eps);
    const g = await mm(y0, l.gate, { act: "gelu" });
    const u = await mm(y0, l.up);
    for (let i = 0; i < g.length; i++) g[i] *= u[i];
    addInto(x, rmsRows(await mm(g, l.down), rows, D, l.post, cfg.eps));
  }
  return rmsRows(x, rows, D, model.mog.norm, cfg.eps);
}

async function generateStep(ctx, model, rvq, rvqNormSq, hC, hU, opts) {
  const cfg = model.cfg,
    mog = model.mog;
  const { latent, codebook, numQuantizers: NQ, numPredictions: NP, lowRank: LR } = cfg;
  const code = new Int32Array(NQ).fill(codebook);
  const codeSum = new Float32Array(latent);
  let cnt = 0;
  for (const k of cfg.unmaskKs) {
    if (k === 0) continue;
    // mog input: embed_code(depthsum(code)) + hidden, CFG batch of 2
    const ce = await ctx.download(ctx.matmul(ctx.ensureTensor({ data: codeSum, rows: 1, cols: latent }), model.embedCode));
    const x = new Float32Array(2 * D);
    for (let j = 0; j < D; j++) {
      x[j] = ce[j] + hC[j];
      x[D + j] = ce[j] + hU[j];
    }
    const xs = await mogMlpStack(ctx, model, x, 2);
    const xg = new Float32Array(D);
    for (let j = 0; j < D; j++) xg[j] = xs[j] + cfg.guidanceScale * (xs[j] - xs[D + j]);
    const logits = await ctx.download(ctx.matmul(ctx.ensureTensor({ data: xg, rows: 1, cols: D }), mog.logits));
    let idx = 0;
    if (opts.deterministic) {
      for (let i = 1; i < NP; i++) if (logits[i] > logits[idx]) idx = i;
    } else {
      idx = sampleTopP(logits, cfg.topP, opts.rng);
    }
    // mu = low_mat[idx] @ (proj_mus[idx] @ x); logs; mu_res
    const muLow = new Float64Array(LR);
    for (let j = 0; j < LR; j++) {
      const ro = (idx * LR + j) * D;
      let s = 0;
      for (let d = 0; d < D; d++) s += mog.mus[ro + d] * xg[d];
      muLow[j] = s;
    }
    const muRes = await ctx.download(ctx.matmul(ctx.ensureTensor({ data: xg, rows: 1, cols: D }), mog.else));
    let logs = 0;
    for (let d = 0; d < D; d++) logs += mog.logs[d] * xg[d];
    if (logs < cfg.minLogStd) logs = cfg.minLogStd;
    const std = Math.exp(logs);
    const z = new Float64Array(latent);
    const lm = idx * latent * LR;
    for (let c = 0; c < latent; c++) {
      let s = 0;
      const ro = lm + c * LR;
      for (let j = 0; j < LR; j++) s += mog.lowmat[ro + j] * muLow[j];
      z[c] = Math.fround(s) * std + muRes[c] + (opts.deterministic ? 0 : std * cfg.noiseScale * opts.rng.gauss());
    }
    // PRVQ greedy encode of quantizers cnt..cnt+k-1 from residual r=z
    for (let qi = cnt; qi < cnt + k; qi++) {
      const base = qi * codebook;
      let best = 0,
        bestD = Infinity;
      for (let j = 0; j < codebook; j++) {
        const e = (base + j) * latent;
        let dot = 0;
        for (let c = 0; c < latent; c++) dot += rvq[e + c] * z[c];
        const dist = rvqNormSq[base + j] - 2 * dot;
        if (dist < bestD) {
          bestD = dist;
          best = j;
        }
      }
      code[qi] = best;
      const e = (base + best) * latent;
      for (let c = 0; c < latent; c++) {
        z[c] -= rvq[e + c];
        codeSum[c] += rvq[e + c];
      }
    }
    cnt += k;
  }
  return code;
}

function sampleTopP(logits, topP, rng) {
  const NP = logits.length;
  // softmax → sort desc → mask beyond top-p (keep ≥1) → gumbel-max over kept log-probs
  let mx = -Infinity;
  for (let i = 0; i < NP; i++) if (logits[i] > mx) mx = logits[i];
  let sum = 0;
  const p = new Float64Array(NP);
  for (let i = 0; i < NP; i++) {
    p[i] = Math.exp(logits[i] - mx);
    sum += p[i];
  }
  const order = Array.from({ length: NP }, (_, i) => i).sort((a, b) => p[b] - p[a]);
  let cum = 0;
  const keep = new Uint8Array(NP);
  for (const i of order) {
    keep[i] = 1;
    cum += p[i] / sum;
    if (cum > topP) break;
  }
  let best = -1,
    bestV = -Infinity;
  for (let i = 0; i < NP; i++) {
    if (!keep[i]) continue;
    const v = Math.log(p[i] / sum) + rng.gumbel();
    if (v > bestV) {
      bestV = v;
      best = i;
    }
  }
  return best;
}

// ── top-level synthesis driver ───────────────────────────────────────────────
/**
 * @param {Int32Array[]|number[][]} charSeqs per-frame conditioning token char ids
 * @param {number[]} frameTokens per-frame subword ids (bos, text…, pad…, eos)
 * @param {Float32Array} rvq flat [31*1024*512]
 * @returns {Promise<{codes:Int32Array[], trace:object}>}
 */
export async function synthesizeCodes(ctx, model, frameTokens, tokenChars, rvq, opts = {}) {
  const cfg = model.cfg;
  const T = frameTokens.length;
  const warmT = cfg.warmFrames;
  const maxT = warmT + T;
  if (maxT > 7000) throw new Error(`voicechat-tts: ${T} frames exceeds the no-sliding-window budget`);
  const caches = model.layers.map(() => ({
    K: [new Float32Array(maxT * D), new Float32Array(maxT * D)],
    V: [new Float32Array(maxT * D), new Float32Array(maxT * D)],
  }));
  const rvqNormSq = opts.rvqNormSq;
  const trace = { stepHidden: [] };
  const runOpts = { deterministic: !!opts.deterministic, rng: makeRng(opts.seed ?? 0) };

  // ── warmup prefill: 37 frames of Aria prompt latent + BOS frame ──
  const audio = new Float32Array(warmT * D);
  for (let i = 0; i < warmT - 1; i++) audio.set(model.aria.subarray(i * D, (i + 1) * D), i * D);
  {
    // last warm position embeds the shifted silence frame + bos_emb
    const ds = depthsum(rvq, cfg.silenceTokens, cfg.latent, cfg.codebook);
    const ce = await ctx.download(ctx.matmul(ctx.ensureTensor({ data: ds, rows: 1, cols: cfg.latent }), model.embedCode));
    const o = (warmT - 1) * D;
    for (let j = 0; j < D; j++) audio[o + j] = ce[j] + model.bosEmb[j];
  }
  const warmCond = new Float32Array(warmT * D);
  for (let i = 0; i < warmT; i++) {
    const tok = cfg.warmSubwordIds[i];
    const cond = await casCond(ctx, model, tokenChars.get(tok) ?? [], tok, cfg.warmSubwordMask[i]);
    warmCond.set(cond, i * D);
  }
  const fusedC = await gatedFusion(ctx, model, audio, warmCond, warmT);
  const nullRows = new Float32Array(warmT * D);
  for (let i = 0; i < warmT; i++) nullRows.set(model.nullEmb, i * D);
  const fusedU = await gatedFusion(ctx, model, audio, nullRows, warmT);
  const wx = new Float32Array(2 * warmT * D);
  wx.set(fusedC, 0);
  wx.set(fusedU, warmT * D);
  const warmHidden = await backboneChunk(ctx, model, wx, warmT, 0, caches);
  if (opts.captureWarm) trace.warmHidden = warmHidden;

  // ── autoregressive frames ──
  const codes = [];
  let prevCode = new Int32Array(cfg.numQuantizers).fill(cfg.speechPad); // init: pad frame (zero embed)
  const nullPair = new Float32Array(D);
  nullPair.set(model.nullEmb);
  const t0 = Date.now();
  let backboneMs = 0,
    mogMs = 0;
  for (let i = 0; i < T; i++) {
    const tok = frameTokens[i];
    const fed = tok === cfg.textEos ? Int32Array.from(cfg.silenceTokens) : prevCode; // force silence on EOS
    const ds = depthsum(rvq, fed, cfg.latent, cfg.codebook);
    const ce = await ctx.download(ctx.matmul(ctx.ensureTensor({ data: ds, rows: 1, cols: cfg.latent }), model.embedCode));
    const cond = await casCond(ctx, model, tokenChars.get(tok) ?? [], tok, true);
    const fC = await gatedFusion(ctx, model, ce, cond, 1);
    const fU = await gatedFusion(ctx, model, ce, nullPair, 1);
    const x = new Float32Array(2 * D);
    x.set(fC, 0);
    x.set(fU, D);
    const tb = Date.now();
    const hidden = await backboneChunk(ctx, model, x, 1, warmT + i, caches);
    backboneMs += Date.now() - tb;
    const hC = hidden.subarray(0, D),
      hU = hidden.subarray(D, 2 * D);
    if (opts.captureSteps && i < opts.captureSteps) trace.stepHidden.push(hidden.slice());
    const tm = Date.now();
    prevCode = await generateStep(ctx, model, rvq, rvqNormSq, hC, hU, runOpts);
    mogMs += Date.now() - tm;
    codes.push(prevCode);
    opts.onProgress?.(i + 1, T);
  }
  trace.msPerStep = (Date.now() - t0) / T;
  trace.backboneMsPerStep = backboneMs / T;
  trace.mogMsPerStep = mogMs / T;
  return { codes, trace };
}

/** Precompute ||e||² per codebook row for the PRVQ argmin. */
export function rvqNorms(rvq, numQ, codebook, latent) {
  const out = new Float64Array(numQ * codebook);
  for (let i = 0; i < numQ * codebook; i++) {
    let s = 0;
    const o = i * latent;
    for (let c = 0; c < latent; c++) s += rvq[o + c] * rvq[o + c];
    out[i] = s;
  }
  return out;
}
