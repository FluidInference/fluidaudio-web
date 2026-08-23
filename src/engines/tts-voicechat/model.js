// VoiceChat-11B TTS core — Gemma3 backbone (28L×1152, KV-cached, CFG batch of 2)
// + CAS t5gemma text conditioning + gated audio/text fusion + MoG head with
// 8-iteration PRVQ unmasking. Mirrors Speech/…/modules/ear_tts_model.py and the
// HF Gemma3/T5Gemma eager paths (transformers 5.15.1 — effective config baked
// by scripts/extract-voicechat-tts.py).
//
// Split of labor: the backbone / CAS / MoG-MLP tensor math runs on ComputeContext
// ops (rmsNorm / headRmsRope / attnCache / matmul) so tensors stay RESIDENT on
// the backend — on WebGPU the whole per-step stack is a handful of batched
// submits with GPU-side KV caches; on WASM the ops are byte-exact ports of the
// original host loops (f64 accumulate, f32 stores), so the deterministic parity
// track is unchanged. The discrete decisions (MoG mixture argmax, PRVQ argmin,
// gumbel/noise) stay host-side in f64 — their margins decide the code matrix,
// which is why the MoG selection weights ship f32 and never leave the CPU.
//
// Parity oracle: scripts/ci-smoke-voicechat-tts.mjs (torch goldens from
// scripts/voicechat-tts-reference.py). WASM is the parity backend (codes
// bit-exact vs torch); WebGPU is gated there on codes-vs-golden as well but
// with accumulation-order drift measured explicitly.

const H = 16;
const HD = 72;
const D = 1152;

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** Gemma RMSNorm on a host array (fusion-side helper; backend ops handle the
 * resident path). y = x * rsqrt(mean(x²)+eps) * (1+w). */
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
  const vec = (n) => W.f32(n); // host array (fusion / MoG-selection side)
  const nrm = (n) => ctx.upload(W.f32(n), 1, W.meta(n).count); // norm weights as backend tensors
  const layers = [];
  for (let L = 0; L < cfg.layers; L++) {
    layers.push({
      q: W.mat(ctx, `B${L}_q`),
      k: W.mat(ctx, `B${L}_k`),
      v: W.mat(ctx, `B${L}_v`),
      o: W.mat(ctx, `B${L}_o`),
      qn: nrm(`B${L}_qn`),
      kn: nrm(`B${L}_kn`),
      lnIn: nrm(`B${L}_ln_in`),
      lnPostAtt: nrm(`B${L}_ln_postatt`),
      lnPreFf: nrm(`B${L}_ln_preff`),
      lnPostFf: nrm(`B${L}_ln_postff`),
      gate: W.mat(ctx, `B${L}_gate`),
      up: W.mat(ctx, `B${L}_up`),
      down: W.mat(ctx, `B${L}_down`),
      global: cfg.globalLayers.includes(L),
    });
  }
  const invFreqFor = (theta) => {
    const f = new Float64Array(HD / 2);
    for (let i = 0; i < HD / 2; i++) f[i] = 1 / Math.pow(theta, (2 * i) / HD);
    return f;
  };
  const mog = { mlps: [], norm: nrm("M_norm"), logs: vec("M_logs"), mus: vec("M_mus"), lowmat: vec("M_lowmat") };
  for (let i = 0; i < 3; i++)
    mog.mlps.push({
      pre: nrm(`M${i}_pre`),
      post: nrm(`M${i}_post`),
      gate: W.mat(ctx, `M${i}_gate`),
      up: W.mat(ctx, `M${i}_up`),
      down: W.mat(ctx, `M${i}_down`),
    });
  return {
    cfg,
    layers,
    bNorm: nrm("B_norm"),
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
      lnPreAtt: nrm("C0_ln_preatt"),
      lnPostAtt: nrm("C0_ln_postatt"),
      lnPreFf: nrm("C0_ln_preff"),
      lnPostFf: nrm("C0_ln_postff"),
      norm: nrm("cas_norm"),
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
    gf: (() => {
      const gate = vec("gf_gate"),
        res = vec("gf_res")[0];
      // GPU fusion path: the sigmoid gate folds into two per-column vectors
      // g1 = res·σ(gate), g2 = res·(1−σ(gate)) so fusion is mul/mul/add/rmsNorm
      // with zero readbacks. The WASM path keeps the original host math (exact).
      const g1 = new Float32Array(D),
        g2 = new Float32Array(D);
      const r = sigmoid(res);
      for (let j = 0; j < D; j++) {
        const g = sigmoid(gate[j]);
        g1[j] = r * g;
        g2[j] = r * (1 - g);
      }
      return {
        audioW: W.mat(ctx, "gf_audio_w"),
        audioB: ctx.upload(vec("gf_audio_b").slice(), 1, D),
        textW: W.mat(ctx, "gf_text_w"),
        textB: ctx.upload(vec("gf_text_b").slice(), 1, D),
        gate,
        res,
        norm: vec("gf_norm"),
        g1T: ctx.upload(g1, 1, D),
        g2T: ctx.upload(g2, 1, D),
        normT: nrm("gf_norm"),
      };
    })(),
    aria: vec("aria_latent"),
    mog: { ...mog, logits: W.mat(ctx, "M_logits"), else: W.mat(ctx, "M_else") },
  };
}

// ── backbone chunk forward (M positions × 2 CFG streams, backend-resident) ───
/**
 * @param {*} x backend tensor [(2*M), D] fused input embeds, stream-major
 * @param {number} pos0 absolute position of row i=0
 * @param {Array} caches per-layer {K, V} backend tensors [2*maxT, D], maxT
 * @returns backend tensor [(2*M), D] hidden after final norm (no readback)
 */
function backboneChunk(ctx, model, x, M, pos0, caches) {
  const cfg = model.cfg;
  for (let L = 0; L < cfg.layers; L++) {
    const w = model.layers[L],
      c = caches[L];
    const invFreq = w.global ? model.invFreqGlobal : model.invFreqLocal;
    const xn = ctx.rmsNorm(x, w.lnIn, cfg.eps);
    const q = ctx.headRmsRope(ctx.matmul(xn, w.q), w.qn, invFreq, { heads: H, headDim: HD, M, pos0, scale: cfg.attnScale, eps: cfg.eps });
    const k = ctx.headRmsRope(ctx.matmul(xn, w.k), w.kn, invFreq, { heads: H, headDim: HD, M, pos0, scale: 1, eps: cfg.eps });
    const v = ctx.matmul(xn, w.v);
    for (let s = 0; s < 2; s++) {
      ctx.copyRows(c.K, ctx.sliceRows(k, s * M, M), s * c.maxT + pos0);
      ctx.copyRows(c.V, ctx.sliceRows(v, s * M, M), s * c.maxT + pos0);
    }
    const att = ctx.attnCache(q, c.K, c.V, { heads: H, headDim: HD, M, pos0, cacheStride: c.maxT });
    x = ctx.rmsNorm(ctx.matmul(att, w.o), w.lnPostAtt, cfg.eps, { add: x });
    const xn2 = ctx.rmsNorm(x, w.lnPreFf, cfg.eps);
    const g = ctx.matmul(xn2, w.gate, { act: "gelu" });
    const u = ctx.matmul(xn2, w.up);
    x = ctx.rmsNorm(ctx.matmul(ctx.mul(g, u), w.down), w.lnPostFf, cfg.eps, { add: x });
  }
  return ctx.rmsNorm(x, model.bNorm, cfg.eps);
}

// ── CAS conditioning (char-aware subword embedding, cached per token id) ─────
async function casForward(ctx, model, chars) {
  const cas = model.cas,
    cfg = model.cfg,
    n = chars.length;
  const xh = new Float32Array(n * D);
  const normzr = cfg.cas.normalizer;
  for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) xh[i * D + j] = cas.charEmb[chars[i] * D + j] * normzr;
  // one t5gemma encoder layer, bidirectional, logit softcap, no q/k norms
  let x = ctx.ensureTensor({ data: xh, rows: n, cols: D });
  const xn = ctx.rmsNorm(x, cas.lnPreAtt, cfg.eps);
  const q = ctx.headRmsRope(ctx.matmul(xn, cas.q), null, cas.invFreq, { heads: H, headDim: HD, M: n, pos0: 0, scale: cfg.cas.attnScale });
  const k = ctx.headRmsRope(ctx.matmul(xn, cas.k), null, cas.invFreq, { heads: H, headDim: HD, M: n, pos0: 0, scale: 1 });
  const v = ctx.matmul(xn, cas.v);
  // bidirectional: every row sees all n keys
  const att = ctx.attnCache(q, k, v, { heads: H, headDim: HD, M: n, causal: false, fixedT: n, softcap: cfg.cas.softcap });
  x = ctx.rmsNorm(ctx.matmul(att, cas.o), cas.lnPostAtt, cfg.eps, { add: x });
  const xn2 = ctx.rmsNorm(x, cas.lnPreFf, cfg.eps);
  const g = ctx.matmul(xn2, cas.gate, { act: "gelu" });
  const u = ctx.matmul(xn2, cas.up);
  x = ctx.rmsNorm(ctx.matmul(ctx.mul(g, u), cas.down), cas.lnPostFf, cfg.eps, { add: x });
  const fin = await ctx.download(ctx.rmsNorm(x, cas.norm, cfg.eps));
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
/** GatedProjectedSumRMSNorm over R (audio,text) row pairs → fused [R*D].
 * WASM/parity path: host math, byte-exact vs the torch goldens. */
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

/** GPU fusion: same math over resident tensors, zero readbacks (record-only).
 * audioT is the RAW code embedding (scaling by 1/numQuantizers happens here). */
function gatedFusionGpu(ctx, model, audioT, textT) {
  const gf = model.gf,
    cfg = model.cfg;
  const a = ctx.matmul(ctx.scale(audioT, 1 / cfg.numQuantizers), gf.audioW, { bias: gf.audioB });
  const t = ctx.matmul(textT, gf.textW, { bias: gf.textB });
  const h = ctx.add(ctx.mul(a, gf.g1T), ctx.mul(t, gf.g2T));
  return ctx.rmsNorm(h, gf.normT, cfg.eps);
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
/** 3-layer GeGLU MLP stack + final norm, backend-resident. x: tensor [rows, D]. */
function mogMlpStack(ctx, model, x) {
  const cfg = model.cfg;
  for (const l of model.mog.mlps) {
    const y0 = ctx.rmsNorm(x, l.pre, cfg.eps);
    const g = ctx.matmul(y0, l.gate, { act: "gelu" });
    const u = ctx.matmul(y0, l.up);
    x = ctx.rmsNorm(ctx.matmul(ctx.mul(g, u), l.down), l.post, cfg.eps, { add: x });
  }
  return ctx.rmsNorm(x, model.mog.norm, cfg.eps);
}

/** @param hidden backend tensor [2, D] (cond row 0, uncond row 1). */
async function generateStep(ctx, model, rvq, rvqNormSq, hidden, opts) {
  const cfg = model.cfg,
    mog = model.mog;
  const { latent, codebook, numQuantizers: NQ, numPredictions: NP, lowRank: LR } = cfg;
  const code = new Int32Array(NQ).fill(codebook);
  const codeSum = new Float32Array(latent);
  let cnt = 0;
  const gpu = ctx.backend === "webgpu";
  for (const k of cfg.unmaskKs) {
    if (k === 0) continue;
    // mog input: embed_code(depthsum(code)) + hidden, CFG batch of 2.
    let xg, logits, muRes;
    if (gpu) {
      // Single batched submit + ONE overlapped readback set per iteration: the
      // CFG combine runs on-GPU as (1+g)·cond − g·uncond (algebraically equal
      // to the host expression; rounding differs at accumulation-order level,
      // which the codes-vs-golden gate measures).
      let staged;
      ctx.withBatchSync(() => {
        const ce = ctx.matmul(ctx.ensureTensor({ data: codeSum.slice(), rows: 1, cols: latent }), model.embedCode);
        const x = ctx.add(hidden, ce); // ce broadcasts over both CFG rows
        const xs = mogMlpStack(ctx, model, x);
        const xgT = ctx.add(ctx.scale(ctx.sliceRows(xs, 0, 1), 1 + cfg.guidanceScale), ctx.scale(ctx.sliceRows(xs, 1, 1), -cfg.guidanceScale));
        staged = [ctx.stageDownload(xgT), ctx.stageDownload(ctx.matmul(xgT, mog.logits)), ctx.stageDownload(ctx.matmul(xgT, mog.else))];
      });
      [xg, logits, muRes] = await Promise.all(staged.map((s) => s.read()));
    } else {
      // WASM/parity path: CFG combine on host in f64, byte-exact vs goldens.
      let xsStaged;
      ctx.withBatchSync(() => {
        const ce = ctx.matmul(ctx.ensureTensor({ data: codeSum, rows: 1, cols: latent }), model.embedCode);
        const x = ctx.add(hidden, ce);
        xsStaged = ctx.stageDownload(mogMlpStack(ctx, model, x));
      });
      const xs = await xsStaged.read();
      xg = new Float32Array(D);
      for (let j = 0; j < D; j++) xg[j] = xs[j] + cfg.guidanceScale * (xs[j] - xs[D + j]);
      const xgT = ctx.ensureTensor({ data: xg, rows: 1, cols: D });
      logits = await ctx.download(ctx.matmul(xgT, mog.logits));
      muRes = await ctx.download(ctx.matmul(xgT, mog.else));
    }
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
 * @param {number[]} frameTokens per-frame subword ids (bos, text…, pad…, eos)
 * @param {Map<number, number[]>} tokenChars per-token conditioning char ids
 * @param {Float32Array} rvq flat [31*1024*512]
 * @returns {Promise<{codes:Int32Array[], trace:object}>}
 */
export async function synthesizeCodes(ctx, model, frameTokens, tokenChars, rvq, opts = {}) {
  const cfg = model.cfg;
  const T = frameTokens.length;
  const warmT = cfg.warmFrames;
  const maxT = warmT + T;
  if (maxT > 7000) throw new Error(`voicechat-tts: ${T} frames exceeds the no-sliding-window budget`);
  // KV caches live on the backend: [2 streams × maxT, D] per layer, pinned
  // (they outlive every per-step arena), returned to the pool at the end.
  const caches = model.layers.map(() => ({
    K: ctx.pin(ctx.alloc(2 * maxT, D)),
    V: ctx.pin(ctx.alloc(2 * maxT, D)),
    maxT,
  }));
  const rvqNormSq = opts.rvqNormSq;
  const trace = { stepHidden: [] };
  const runOpts = { deterministic: !!opts.deterministic, rng: makeRng(opts.seed ?? 0) };
  try {
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
    // cond stream text rows then uncond (null-embedding) rows — fused in ONE
    // gatedFusion call so the result is exactly the stream-major backbone input.
    const text2 = new Float32Array(2 * warmT * D);
    for (let i = 0; i < warmT; i++) {
      const tok = cfg.warmSubwordIds[i];
      const cond = await casCond(ctx, model, tokenChars.get(tok) ?? [], tok, cfg.warmSubwordMask[i]);
      text2.set(cond, i * D);
    }
    for (let i = 0; i < warmT; i++) text2.set(model.nullEmb, (warmT + i) * D);
    const audio2 = new Float32Array(2 * warmT * D);
    audio2.set(audio, 0);
    audio2.set(audio, warmT * D);
    const gpu = ctx.backend === "webgpu";
    const wx = gpu ? null : await gatedFusion(ctx, model, audio2, text2, 2 * warmT);
    const warmArena = ctx.pushArena();
    const warmHidden = ctx.withBatchSync(() => {
      const xT = gpu
        ? gatedFusionGpu(ctx, model, ctx.ensureTensor({ data: audio2, rows: 2 * warmT, cols: D }), ctx.ensureTensor({ data: text2, rows: 2 * warmT, cols: D }))
        : ctx.ensureTensor({ data: wx, rows: 2 * warmT, cols: D });
      return backboneChunk(ctx, model, xT, warmT, 0, caches);
    });
    if (opts.captureWarm) trace.warmHidden = await ctx.download(warmHidden);
    ctx.popArena(warmArena);

    // Prefetch CAS conditioning for every frame token BEFORE the loop so the
    // per-frame path records without a single readback on the GPU backend.
    const condCache = new Map();
    for (const tok of new Set(frameTokens)) condCache.set(tok, await casCond(ctx, model, tokenChars.get(tok) ?? [], tok, true));

    // ── autoregressive frames ──
    const codes = [];
    let prevCode = new Int32Array(cfg.numQuantizers).fill(cfg.speechPad); // init: pad frame (zero embed)
    const t0 = Date.now();
    let backboneMs = 0,
      mogMs = 0;
    for (let i = 0; i < T; i++) {
      const arena = ctx.pushArena();
      const tok = frameTokens[i];
      const fed = tok === cfg.textEos ? Int32Array.from(cfg.silenceTokens) : prevCode; // force silence on EOS
      const ds = depthsum(rvq, fed, cfg.latent, cfg.codebook);
      const cond = condCache.get(tok);
      const textP = new Float32Array(2 * D);
      textP.set(cond, 0);
      textP.set(model.nullEmb, D);
      const tb = Date.now();
      let hidden;
      if (gpu) {
        // Record-only frame input: embed → fusion → backbone in ONE submit.
        hidden = ctx.withBatchSync(() => {
          const ceT = ctx.matmul(ctx.ensureTensor({ data: ds, rows: 1, cols: cfg.latent }), model.embedCode);
          const xT = gatedFusionGpu(ctx, model, ctx.concatRows([ceT, ceT]), ctx.ensureTensor({ data: textP, rows: 2, cols: D }));
          return backboneChunk(ctx, model, xT, 1, warmT + i, caches);
        });
      } else {
        const ce = await ctx.download(ctx.matmul(ctx.ensureTensor({ data: ds, rows: 1, cols: cfg.latent }), model.embedCode));
        const audioP = new Float32Array(2 * D);
        audioP.set(ce, 0);
        audioP.set(ce, D);
        const fused = await gatedFusion(ctx, model, audioP, textP, 2);
        hidden = ctx.withBatchSync(() => backboneChunk(ctx, model, ctx.ensureTensor({ data: fused, rows: 2, cols: D }), 1, warmT + i, caches));
      }
      const captureThis = opts.captureSteps && i < opts.captureSteps;
      // Wall-clock split needs a sync boundary; the batched submit returns
      // before the GPU finishes, so only force one when asked (or capturing).
      if (captureThis || opts.syncTrace) {
        const h = await ctx.download(hidden);
        if (captureThis) trace.stepHidden.push(h);
      }
      backboneMs += Date.now() - tb;
      const tm = Date.now();
      prevCode = await generateStep(ctx, model, rvq, rvqNormSq, hidden, runOpts);
      mogMs += Date.now() - tm;
      codes.push(prevCode);
      ctx.popArena(arena);
      opts.onProgress?.(i + 1, T);
    }
    trace.msPerStep = (Date.now() - t0) / T;
    trace.backboneMsPerStep = backboneMs / T;
    trace.mogMsPerStep = mogMs / T;
    return { codes, trace };
  } finally {
    for (const c of caches) {
      ctx.freeTensor(c.K);
      ctx.freeTensor(c.V);
    }
  }
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
