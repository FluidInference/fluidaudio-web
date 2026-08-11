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
    const s = h & 0x8000 ? -1 : 1,
      e = (h & 0x7c00) >> 10,
      f = h & 0x03ff;
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
  if (int8) {
    const ab = bin.buffer instanceof ArrayBuffer ? bin.buffer : bin;
    f32v = new Float32Array(ab);
    i8v = new Int8Array(ab);
  }
  if (f16) {
    u16v = new Uint16Array(bin.buffer, bin.byteOffset, bin.byteLength >> 1);
    lut = f16lut();
  }
  const raw = (k) => {
    const m = man[k];
    if (f16) {
      const q = u16v.subarray(m.offset, m.offset + m.count),
        out = new Float32Array(m.count);
      for (let i = 0; i < m.count; i++) out[i] = lut[q[i]];
      return out;
    }
    if (!int8) return bin.subarray(m.offset, m.offset + m.len);
    if (m.dtype !== "i8") return f32v.subarray(m.offset, m.offset + m.count);
    const q = i8v.subarray(m.i8ByteOffset, m.i8ByteOffset + m.count);
    const sc = f32v.subarray(m.scaleOffset, m.scaleOffset + m.scaleCount);
    const out = new Float32Array(m.count);
    if (m.quant === "col") {
      const o = m.dims[1];
      for (let i = 0; i < m.count; i++) out[i] = q[i] * sc[i % o];
    } else {
      const rest = m.count / m.dims[0];
      for (let i = 0; i < m.count; i++) out[i] = q[i] * sc[(i / rest) | 0];
    }
    return out;
  };
  const scaled = (k, s) => {
    const a = raw(k).slice();
    for (let i = 0; i < a.length; i++) a[i] *= s;
    return a;
  };
  // Weight matrices upload as f16 storage when the backend supports it: the
  // mixed-precision v4 kernel reads them at half the traffic and half the GPU
  // memory (~2.3GB → 1.17GB for the fp32-dequantized encoder). Activations and
  // biases stay fp32.
  // Tile-major direct-B GEMM weights: DEFAULT ON (browser-verified 282× on the
  // 1hr bench). PR #39 flipped main.ts/README/gates but THIS line's edit
  // silently no-op'd (string-replace mismatch) — and the node gates couldn't
  // catch it because the gate script sets the flag itself. __tmGemm === false
  // (?tm=0 / TM=0) restores the LDS-staged baseline; self-falls-back
  // per-tensor on dims and on devices without probed 32-lane subgroups.
  const upW = (data, r, c) => (globalThis.__tmGemm !== false && ctx.uploadTileMajorF16 ? ctx.uploadTileMajorF16(data, r, c) : ctx.uploadF16(data, r, c)); // context owns the format decision
  const mat = (k) => upW(raw(k).slice(), man[k].dims[0], man[k].dims[1]);
  const vec = (k) => ctx.upload(raw(k).slice(), 1, man[k].count ?? man[k].len);
  const matScaled = (k, s) => upW(scaled(k, s), man[k].dims[0], man[k].dims[1]);

  // NeMo RelPositionalEncoding xscaling (x *= sqrt(d_model) after subsampling): fold
  // sqrt(D) into the pre_encode linear. Some exports (Parakeet) bake it in already;
  // Sortformer leaves it as a runtime Mul → set cfg.xscale to reproduce it.
  const xs = cfg.xscale ? Math.sqrt(cfg.D) : 1;
  const sub = {
    conv: [0, 1, 2, 3, 4].map((i) => ({ w: vec(`c${i}w`), b: vec(`c${i}b`) })),
    linw: xs === 1 ? mat("linw") : matScaled("linw", xs),
    linb: xs === 1 ? vec("linb") : ctx.upload(scaled("linb", xs), 1, man["linb"].count ?? man["linb"].len),
  };
  const layers = [];
  // Optional per-layer linear biases: Sortformer's conformer FF/attn linears have
  // bias=True; Parakeet/EOU/Nemotron don't. Present → vec, absent → null. FF2/FF1
  // linear2 biases fold the 0.5 macaron factor; q bias folds 1/sqrt(HD).
  const vecOpt = (k) => (man[k] ? vec(k) : null);
  const vecScaledOpt = (k, s) => (man[k] ? ctx.upload(scaled(k, s), 1, man[k].count ?? man[k].len) : null);
  // Pointwise conv weights as TRANSPOSED matrices [cin, cout]: the conv module
  // runs them as X@Wt GEMMs (weights on the B side → f16 path, fused bias),
  // valid for any window length. k=1 ⇒ [cout, cin] row-major in the manifest.
  const pwT = (k, cout, cin) => {
    const w = raw(k);
    const t = new Float32Array(cin * cout);
    for (let co = 0; co < cout; co++) for (let ci = 0; ci < cin; ci++) t[ci * cout + co] = w[co * cin + ci];
    return upW(t, cin, cout);
  };
  for (let L = 0; L < cfg.layers; L++) {
    const g = (s) => `L${L}_${s}`;
    // pos_bias_u/v uploaded ONCE as per-head GPU tensors [1,HD].
    const pbuS = scaled(g("pbu"), INV),
      pbvS = scaled(g("pbv"), INV);
    layers.push({
      lnff1: [vec(g("lnff1_w")), vec(g("lnff1_b"))],
      ff1w1: mat(g("ff1w1")),
      ff1w2: matScaled(g("ff1w2"), 0.5),
      ff1b1: vecOpt(g("ff1b1")),
      ff1b2: vecScaledOpt(g("ff1b2"), 0.5),
      lnatt: [vec(g("lnatt_w")), vec(g("lnatt_b"))],
      q: matScaled(g("q"), INV),
      k: mat(g("k")),
      v: mat(g("v")),
      pos: mat(g("pos")),
      out: mat(g("out")),
      qb: vecScaledOpt(g("qb"), INV),
      kb: vecOpt(g("kb")),
      vb: vecOpt(g("vb")),
      outb: vecOpt(g("outb")),
      pbuAll: ctx.upload(pbuS.slice(), 1, cfg.H * HD),
      pbvAll: ctx.upload(pbvS.slice(), 1, cfg.H * HD),
      lnconv: [vec(g("lnconv_w")), vec(g("lnconv_b"))],
      pw1T: pwT(g("pw1"), 2 * cfg.D, cfg.D),
      dw: vec(g("dw")),
      dwb: vec(g("dwb")),
      pw2T: pwT(g("pw2"), cfg.D, cfg.D),
      pw1b: vecOpt(g("pw1b")),
      pw2b: vecOpt(g("pw2b")), // pointwise conv biases (Sortformer)
      bn: man[g("bnw")] ? [vec(g("bnw")), vec(g("bnb"))] : null, // conv-module norm (EOU)
      lnff2: [vec(g("lnff2_w")), vec(g("lnff2_b"))],
      ff2w1: mat(g("ff2w1")),
      ff2w2: matScaled(g("ff2w2"), 0.5),
      ff2b1: vecOpt(g("ff2b1")),
      ff2b2: vecScaledOpt(g("ff2b2"), 0.5),
      lnout: [vec(g("lnout_w")), vec(g("lnout_b"))],
    });
  }
  return { sub, layers, cfg };
}

function posEncoding(Tsub, D) {
  const pe = new Float32Array((2 * Tsub - 1) * D);
  const dv = (i) => Math.exp(i * -(Math.log(10000) / D));
  for (let pi = 0; pi < 2 * Tsub - 1; pi++) {
    const pos = Tsub - 1 - pi;
    for (let i = 0; i < D; i += 2) {
      pe[pi * D + i] = Math.sin(pos * dv(i));
      pe[pi * D + i + 1] = Math.cos(pos * dv(i));
    }
  }
  return pe;
}

// Subsampling + pre-encode linear for ONE window: mel → x [Tsub, D].
function preEncode(ctx, enc, mel) {
  const { Csub, melBins } = enc.cfg;
  const sp = enc.cfg.subPad || { t: 1, b: 1, l: 1, r: 1 };
  const Tfull = mel.length / melBins;
  const x0 = new Float32Array(Tfull * melBins);
  for (let t = 0; t < Tfull; t++) for (let c = 0; c < melBins; c++) x0[t * melBins + c] = mel[c * Tfull + t];
  let s = ctx.upload(x0, 1, Tfull * melBins),
    Hh = Tfull,
    Wd = melBins;
  // [cout, cin, k, stride, groups, act, isStride2]
  const conv = [
    [Csub, 1, 3, 2, 1, "relu", true],
    [Csub, Csub, 3, 2, Csub, "none", true],
    [Csub, Csub, 1, 1, 1, "relu", false],
    [Csub, Csub, 3, 2, Csub, "none", true],
    [Csub, Csub, 1, 1, 1, "relu", false],
  ];
  for (let i = 0; i < 5; i++) {
    const [cout, cin, k, st, gr, act, s2] = conv[i];
    const pt = s2 ? sp.t : 0,
      pb = s2 ? sp.b : 0,
      pl = s2 ? sp.l : 0,
      pr = s2 ? sp.r : 0;
    s = ctx.conv2d(s, enc.sub.conv[i].w, {
      cout,
      cin,
      h: Hh,
      w: Wd,
      kh: k,
      kw: k,
      bias: enc.sub.conv[i].b,
      strideH: st,
      strideW: st,
      padTop: pt,
      padBottom: pb,
      padLeft: pl,
      padRight: pr,
      groups: gr,
      act,
    });
    Hh = Math.floor((Hh + pt + pb - k) / st) + 1;
    Wd = Math.floor((Wd + pl + pr - k) / st) + 1;
  }
  const flat = ctx.subReshape(s, Csub, Hh, Wd);
  return { x: ctx.matmul(flat, enc.sub.linw, { bias: enc.sub.linb }), Tsub: Hh };
}

/** Run the encoder on one window. mel: Float32Array[melBins*T] (channel-major). */
export async function parakeetEncode(ctx, enc, mel, T, wantData = false) {
  const r = await parakeetEncodeBatch(ctx, enc, [mel], wantData);
  return { dims: r.dims, framesGpu: r.framesGpu, Tsub: r.Tsub, data: r.data };
}

/**
 * Run the encoder on W same-length windows CONCATENATED along time-rows: the
 * FF/projection GEMMs see M = W·Tsub (the thin-GEMM occupancy fix); attention and
 * the depthwise conv stay per-window. Returns framesGpu [W·Tsub, D].
 */
export async function parakeetEncodeBatch(ctx, enc, mels, wantData = false, post = null) {
  const { D, H, HD, layers: LAYERS, dwK, Csub, melBins } = enc.cfg;
  const ln = (x, lp) => ctx.layernorm(x, lp[0], lp[1]);
  const W = mels.length;
  const pre = mels.map((m) => preEncode(ctx, enc, m));
  const Tsub = pre[0].Tsub;
  for (const p of pre) if (p.Tsub !== Tsub) throw new Error("parakeetEncodeBatch: windows must have equal Tsub");
  let x = W === 1 ? pre[0].x : ctx.concatRows(pre.map((p) => p.x)); // [W*Tsub, D]

  enc._pe = enc._pe || new Map();
  let peT = enc._pe.get(Tsub);
  if (!peT) {
    peT = ctx.upload(posEncoding(Tsub, D), 2 * Tsub - 1, D);
    enc._pe.set(Tsub, peT);
  }
  // Cache-aware streaming attention mask: chunk-limited on BOTH sides. With
  // chunkStart = chunk*floor(i/chunk), query i attends keys j in
  // [max(0, chunkStart-left), min(Tsub-1, chunkStart+chunk-1+right)]; -10000
  // elsewhere (added to scores pre-softmax). EOU: chunk 2, left 70, right 0.
  // Nemotron: chunk 4, left 56, right 3 (rightContext lookahead).
  let maskT = null;
  if (enc.cfg.attChunk) {
    enc._mask = enc._mask || new Map();
    maskT = enc._mask.get(`${Tsub}|${W}`);
    if (!maskT) {
      const C = enc.cfg.attChunk,
        LEFT = enc.cfg.attLeft ?? 70,
        RIGHT = enc.cfg.attRight ?? 0,
        mk = new Float32Array(Tsub * Tsub);
      for (let i = 0; i < Tsub; i++) {
        const cs = C * Math.floor(i / C);
        const lo = Math.max(0, cs - LEFT),
          hi = Math.min(Tsub - 1, cs + C - 1 + RIGHT);
        for (let j = 0; j < Tsub; j++) mk[i * Tsub + j] = j >= lo && j <= hi ? 0 : -10000;
      }
      // tiled per (window, head) block [W*H*T, T] to match the batched scores layout
      const mkH = new Float32Array(W * H * Tsub * Tsub);
      for (let b = 0; b < W * H; b++) mkH.set(mk, b * Tsub * Tsub);
      maskT = ctx.upload(mkH, W * H * Tsub, Tsub);
      enc._mask.set(`${Tsub}|${W}`, maskT);
    }
  }
  // Depthwise conv module pad: symmetric (Parakeet) or causal (EOU streaming: all
  // pad on the left, none on the right).
  const dwPadL = enc.cfg.convCausal ? dwK - 1 : (dwK - 1) >> 1;
  const dwPadR = enc.cfg.convCausal ? 0 : (dwK - 1) >> 1;
  // b1 (pre-SiLU) and b2 (on linear2) are null unless the model has FF linear biases.
  // Residual adds ride the GEMM epilogue ({add}) — one less full pass over
  // [W*T,D] per add and 4 fewer dispatches per layer.
  const ff = (x, lp, w1, w2, b1, b2) => ctx.matmul(ctx.matmul(ln(x, lp), w1, { bias: b1, act: "silu" }), w2, { bias: b2, add: x });

  // One command-buffer submit for the whole conformer stack (ops are recorded
  // into a single compute pass; per-op submits dominate otherwise). The post
  // hook (joint projection + staging copy) rides the same submit, and
  // withBatchSync guarantees the batch closes even on a mid-stack throw.
  let layerArena = null;
  const staged = ctx.withBatchSync(() => {
    try {
      for (let L = 0; L < LAYERS; L++) {
        // Per-layer arena: a layer's scratch (q/k/v, scores, FF intermediates —
        // the bulk of transient GPU memory) recycles into layer L+2's recording.
        // Same-submit reuse is ordered (earlier ops read before later ops write);
        // only the residual stream x crosses layers → pinned to the group arena.
        layerArena = ctx.pushArena();
        const w = enc.layers[L];
        x = ff(x, w.lnff1, w.ff1w1, w.ff1w2, w.ff1b1, w.ff1b2);
        const xln = ln(x, w.lnatt);
        const q = ctx.matmul(xln, w.q, { bias: w.qb }),
          k = ctx.matmul(xln, w.k, { bias: w.kb }),
          v = ctx.matmul(xln, w.v, { bias: w.vb });
        // pos-emb projection is constant per (layer, Tsub) — cache across window
        // groups (was recomputed 24×/group: ~2.5ms/win on long files). Only batched
        // (W>1) full-window Tsubs are cached: W==1 covers tails, whose Tsub varies
        // per file and would grow the cache without bound on long-lived engines.
        let p;
        if (W > 1) {
          enc._posProj = enc._posProj || new Map();
          const pKey = `${L}|${Tsub}`;
          p = enc._posProj.get(pKey);
          if (!p) {
            p = ctx.matmul(peT, w.pos);
            ctx.pin(p); // cached across groups — exempt from the group arena
            enc._posProj.set(pKey, p);
          }
        } else {
          p = ctx.matmul(peT, w.pos);
        }
        // Batched over windows × heads (pos-emb rows shared across windows).
        // Fused path (opt-in, unmasked models, T ≤ 256, HD 64/128): one
        // dispatch, no [W*H*T, T] score tensor. RECORD CORRECTION: the
        // original HD ≤ 64 guard meant this NEVER engaged on Parakeet
        // (HD=128) — the PR #32-era "+1.6%" was fallback noise and the
        // "Chrome 2× regression" attributed to it was environmental. With
        // HD=128 support it now truly runs — and measures 78.6ms/group vs
        // ~20ms for the multi-pass chain (dawn): the per-lane serial design
        // loses to the tiled bmm kernels. Stays opt-in-off; a competitive
        // version needs a flash-attention-grade tiled kernel (~4% end-to-end
        // upside, task-27 archive has their reference).
        let outc = null;
        if (!maskT && ctx.attnFused && globalThis.__attnFused === true) {
          outc = ctx.attnFused(q, k, v, p, w.pbuAll, w.pbvAll, H, HD, W);
        }
        if (!outc) {
          const ac = ctx.bmmQK(q, k, w.pbuAll, H, HD, W); // [W*H*T, T]
          const bd = ctx.relShiftB(ctx.bmmQK(q, p, w.pbvAll, H, HD, W, true), W * H); // [W*H*T, T]
          let sc = ctx.add(ac, bd);
          if (maskT) sc = ctx.add(sc, maskT);
          const probs = ctx.softmax(sc); // rows = W*H*T
          outc = ctx.bmmPV(probs, v, H, HD, W); // [W*T, H*HD]
        }
        x = ctx.matmul(outc, w.out, { bias: w.outb, add: x });
        // Pointwise convs (k=1) are plain GEMMs; run them X@Wt with weights on the
        // B side (f16 path, fused bias) — shape-valid for any window length.
        const pre1 = ctx.matmul(ln(x, w.lnconv), w.pw1T, { bias: w.pw1b }); // [W*T, 2D]
        const glu = ctx.glu(ctx.transpose(pre1)); // [D, W*T]
        // Depthwise conv is over TIME → per-window (a batched run would leak across
        // window seams). Pointwise convs (k=1) and channel ops act per-timestep.
        const dwConv = (input, opts) => {
          if (W === 1) return ctx.conv1d(input, w.dw, opts);
          const out2 = ctx.alloc(D, W * Tsub);
          for (let wi = 0; wi < W; wi++) {
            ctx.setCols(out2, ctx.conv1d(ctx.sliceCols(input, wi * Tsub, Tsub), w.dw, opts), wi * Tsub);
          }
          return out2;
        };
        let dwo;
        if (w.bn) {
          // EOU: depthwise (no act) → LayerNorm over channels → SiLU. [D,T]→[T,D]→LN→[D,T]
          const d = dwConv(glu, { cout: D, k: dwK, groups: D, padLeft: dwPadL, padRight: dwPadR, bias: w.dwb });
          dwo = ctx.transpose(ctx.silu(ln(ctx.transpose(d), w.bn)));
        } else {
          // Parakeet: BatchNorm folded into depthwise, SiLU fused.
          dwo = dwConv(glu, { cout: D, k: dwK, groups: D, padLeft: dwPadL, padRight: dwPadR, bias: w.dwb, act: "silu" });
        }
        x = ctx.matmul(ctx.transpose(dwo), w.pw2T, { bias: w.pw2b, add: x });
        x = ff(x, w.lnff2, w.ff2w1, w.ff2w2, w.ff2b1, w.ff2b2);
        x = ln(x, w.lnout);
        if (layerArena) {
          ctx.pin(x, true); // x feeds the next layer — promote to the group arena
          ctx.popArena(layerArena);
          layerArena = null;
        }
      }
      return post ? post(x) : null;
    } finally {
      if (layerArena) ctx.popArena(layerArena); // a mid-layer throw must not orphan the scope
    }
  });
  const out = { dims: [1, D, W * Tsub], framesGpu: x, Tsub, W, D, staged };
  if (wantData) out.data = await ctx.download(ctx.transpose(x));
  return out;
}
