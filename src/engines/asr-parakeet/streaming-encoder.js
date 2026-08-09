// True streaming FastConformer encode — per-chunk continuation with GPU-resident
// K/V and depthwise-conv caches (docs/STREAMING.md). Computes the SAME function
// as the offline chunked-causal path: the offline mask already limits queries to
// [chunkStart−left, chunkStart+chunk−1+right], so carrying exactly those
// activations chunk-to-chunk reproduces offline frames (gate:
// scripts/streaming-encode-check.mjs). Requires a streaming config: attChunk +
// convCausal (EOU; Nemotron once right-context buffering lands).
//
// State per stream: per-layer kCache/vCache [≤left, D] (K/V cached directly —
// they're what attention reads), convCache [D, dwK−1] (GLU output tail = the
// depthwise conv's input), a 7-mel-frame subsampling overlap in the mel FIFO,
// and subT (absolute frame count, keeps the chunk grid aligned to offline).

function streamPosEncoding(dMax, dMin, D) {
  const P = dMax - dMin + 1;
  const pe = new Float32Array(P * D);
  const dv = (i) => Math.exp(i * -(Math.log(10000) / D));
  for (let pi = 0; pi < P; pi++) {
    const pos = dMax - pi;
    for (let i = 0; i < D; i += 2) {
      pe[pi * D + i] = Math.sin(pos * dv(i));
      pe[pi * D + i + 1] = Math.cos(pos * dv(i));
    }
  }
  return pe;
}

/** Left receptive field of the 3 stride-2 k=3 convs with padTop=2: subsampled
 * frame s reads mel rows [8s−14, 8s]. */
const SUB_LOOKBACK = 14;

export function createEncodeStream(ctx, enc, { proj = null } = {}) {
  const { D, layers, dwK, attChunk: C } = enc.cfg;
  if (!C || !enc.cfg.convCausal) throw new Error("createEncodeStream: needs a streaming config (attChunk + convCausal)");
  const sp = enc.cfg.subPad || {};
  if (sp.t !== 2) throw new Error("createEncodeStream: subsampling continuation assumes subPad.t === 2");
  const LEFT = enc.cfg.attLeft ?? 70;
  const RIGHT = enc.cfg.attRight ?? 0;
  if (RIGHT !== 0) throw new Error("createEncodeStream: right-context lookahead not implemented yet");
  const dMax = LEFT + C - 1;
  const dMin = -(C - 1 + RIGHT);
  const pe = ctx.upload(streamPosEncoding(dMax, dMin, D), dMax - dMin + 1, D);
  return {
    enc,
    C,
    LEFT,
    RIGHT,
    dMax,
    subT: 0,
    // mel FIFO, frame-major [t·melBins + c] — preEncode's upload layout.
    fifo: new Float32Array(0),
    fifoStart: 0, // global mel-frame index of fifo[0]
    k: new Array(layers).fill(null),
    v: new Array(layers).fill(null),
    cc: new Array(layers).fill(null),
    pe,
    // pos-emb projection per layer is constant for the life of the stream.
    posP: enc.layers.map((w) => ctx.matmul(pe, w.pos)),
    zeroCC: ctx.upload(new Float32Array(D * (dwK - 1)), D, dwK - 1),
    // Optional joint projection {w, b}: frames download as [n, projDim]
    // (one fused GEMM rides the chunk batch — feeds the wasm decoder direct).
    proj,
    flushed: false,
    disposed: false,
  };
}

/** Feed mel frames (mel-major [melBins × count], StreamingMel output). Runs as
 * many chunk passes as the FIFO allows; returns Float32Array [nNew, D] of new
 * encoder frames (row-major), or null. maxChunk bounds per-pass latency. */
export async function encodeStreamPush(ctx, st, mel, count, { maxChunk = 64 } = {}) {
  if (st.flushed) throw new Error("encodeStreamPush: stream already flushed — create a new stream");
  const { melBins } = st.enc.cfg;
  if (count > 0) {
    const old = st.fifo;
    const oldFrames = old.length / melBins;
    const next = new Float32Array((oldFrames + count) * melBins);
    next.set(old);
    // mel-major → frame-major
    for (let t = 0; t < count; t++) for (let c = 0; c < melBins; c++) next[(oldFrames + t) * melBins + c] = mel[c * count + t];
    st.fifo = next;
  }
  const outs = [];
  for (;;) {
    const M = st.fifo.length / melBins;
    let n;
    if (st.subT === 0) {
      n = Math.floor((M + 7) / 8); // first chunk consumes 8n−7 frames (mel [0, 8n−8])
    } else {
      n = Math.floor((M - 7) / 8); // continuation consumes 8n+7 (7-frame overlap tail + 8n new)
    }
    n = Math.min(n, maxChunk) - (Math.min(n, maxChunk) % st.C);
    if (n < st.C) break;
    const m = st.subT === 0 ? 8 * n - 7 : 8 * n + 7;
    outs.push(await runChunk(ctx, st, st.fifo.subarray(0, m * melBins), m, n, false));
    advanceFifo(st, n, melBins);
  }
  return concatOut(outs, st.enc.cfg.D);
}

/** Encode the final partial chunk with the offline bottom pad (signal end).
 * Call once; the stream only accepts dispose() afterwards. */
export async function encodeStreamFlush(ctx, st) {
  if (st.flushed) throw new Error("encodeStreamFlush: stream already flushed");
  st.flushed = true;
  const { melBins } = st.enc.cfg;
  const M = st.fifo.length / melBins;
  if (M === 0 && st.subT === 0) return null; // nothing ever pushed
  // Remaining output length = what the offline conv stack yields for the tail
  // slice: each of the 3 stride-2 k=3 convs pads padTop (first chunk only) and
  // padBottom (flush). Continuation slices start at global mel 8·subT−14, whose
  // first output row is exactly frame subT — so the slice's subsampled length
  // IS the new-frame count.
  const sp = st.enc.cfg.subPad;
  const pt = st.subT === 0 ? sp.t : 0;
  let T = M;
  for (let i = 0; i < 3; i++) T = Math.floor((T + pt + sp.b - 3) / 2) + 1;
  const n = T;
  if (n <= 0) return null;
  const out = await runChunk(ctx, st, st.fifo, M, n, true);
  st.fifo = new Float32Array(0);
  return out;
}

export function disposeEncodeStream(ctx, st) {
  if (st.disposed) return;
  st.disposed = true;
  for (const t of [...st.posP, ...st.k, ...st.v, ...st.cc]) if (t) ctx.freeTensor(t);
  // pe/zeroCC are uploads with alloc-compatible usage — pool them too.
  ctx.freeTensor(st.pe);
  ctx.freeTensor(st.zeroCC);
}

function advanceFifo(st, n, melBins) {
  st.subT += n;
  const keepFrom = 8 * st.subT - SUB_LOOKBACK; // global
  const drop = keepFrom - st.fifoStart;
  st.fifo = st.fifo.slice(drop * melBins);
  st.fifoStart = keepFrom;
}

function concatOut(outs, D) {
  const kept = outs.filter(Boolean);
  if (!kept.length) return null;
  if (kept.length === 1) return kept[0];
  const total = kept.reduce((s, o) => s + o.length, 0);
  const all = new Float32Array(total);
  let off = 0;
  for (const o of kept) {
    all.set(o, off);
    off += o.length;
  }
  return all;
}

// One chunk pass: mel slice (frame-major, m frames) → n new frames [n, D],
// downloaded. Everything records into one batch submit; caches update inside.
async function runChunk(ctx, st, melSlice, m, n, isFlush) {
  const enc = st.enc;
  const { D, H, HD, layers: LAYERS, dwK, Csub, melBins } = enc.cfg;
  const sp = enc.cfg.subPad;
  const isFirst = st.subT === 0;
  const ln = (x, lp) => ctx.layernorm(x, lp[0], lp[1]);
  const ff = (x, lp, w1, w2, b1, b2) => ctx.matmul(ctx.matmul(ln(x, lp), w1, { bias: b1, act: "silu" }), w2, { bias: b2, add: x });
  const arena = ctx.pushArena ? ctx.pushArena() : null;
  const swaps = []; // [slot, layer, newTensor] applied after the batch closes
  let frames = null;
  let melUp = null; // upload() is pool-exempt — return it explicitly below
  try {
    ctx.withBatchSync(() => {
      // Subsampling with continuation pads: time pads only at stream edges
      // (padTop first chunk, padBottom at flush); freq pads always.
      let s = (melUp = ctx.upload(melSlice.slice(0, m * melBins), 1, m * melBins));
      let Hh = m,
        Wd = melBins;
      const conv = [
        [Csub, 1, 3, 2, 1, "relu", true],
        [Csub, Csub, 3, 2, Csub, "none", true],
        [Csub, Csub, 1, 1, 1, "relu", false],
        [Csub, Csub, 3, 2, Csub, "none", true],
        [Csub, Csub, 1, 1, 1, "relu", false],
      ];
      for (let i = 0; i < 5; i++) {
        const [cout, cin, k, stride, gr, act, s2] = conv[i];
        const pt = s2 && isFirst ? sp.t : 0;
        const pb = s2 && isFlush ? sp.b : 0;
        s = ctx.conv2d(s, enc.sub.conv[i].w, {
          cout,
          cin,
          h: Hh,
          w: Wd,
          kh: k,
          kw: k,
          bias: enc.sub.conv[i].b,
          strideH: stride,
          strideW: stride,
          padTop: pt,
          padBottom: pb,
          padLeft: s2 ? sp.l : 0,
          padRight: s2 ? sp.r : 0,
          groups: gr,
          act,
        });
        Hh = Math.floor((Hh + pt + pb - k) / stride) + 1;
        Wd = Math.floor((Wd + (s2 ? sp.l + sp.r : 0) - k) / stride) + 1;
      }
      if (Hh !== n) throw new Error(`streaming subsample: expected ${n} frames, got ${Hh}`);
      let x = ctx.matmul(ctx.subReshape(s, Csub, Hh, Wd), enc.sub.linw, { bias: enc.sub.linb }); // [n, D]

      for (let L = 0; L < LAYERS; L++) {
        const w = enc.layers[L];
        x = ff(x, w.lnff1, w.ff1w1, w.ff1w2, w.ff1b1, w.ff1b2);
        const xln = ln(x, w.lnatt);
        const q = ctx.matmul(xln, w.q, { bias: w.qb });
        const kN = ctx.matmul(xln, w.k, { bias: w.kb });
        const vN = ctx.matmul(xln, w.v, { bias: w.vb });
        const Lc = st.k[L] ? st.k[L].rows : 0;
        const K = Lc ? ctx.concatRows([st.k[L], kN]) : kN;
        const V = Lc ? ctx.concatRows([st.v[L], vN]) : vN;
        const Lk = Lc + n;
        const ac = ctx.bmmQK(q, K, w.pbuAll, H, HD, 1); // [H*n, Lk]
        const bdRaw = ctx.bmmQK(q, st.posP[L], w.pbvAll, H, HD, 1, true); // [H*n, P]
        const bd = ctx.relShiftStream(bdRaw, {
          H,
          n,
          Lk,
          dMax: st.dMax,
          Lc,
          subT: st.subT,
          C: st.C,
          left: st.LEFT,
          right: st.RIGHT,
        });
        const probs = ctx.softmax(ctx.add(ac, bd));
        x = ctx.matmul(ctx.bmmPV(probs, V, H, HD, 1), w.out, { bias: w.outb, add: x });
        const keep = Math.min(st.LEFT, Lk);
        swaps.push(["k", L, ctx.pin(ctx.sliceRows(K, Lk - keep, keep))]);
        swaps.push(["v", L, ctx.pin(ctx.sliceRows(V, Lk - keep, keep))]);

        // Conv module: depthwise input = [convCache ‖ GLU output], causal k=dwK
        // ⇒ emits exactly n frames; cache = last dwK−1 columns.
        const pre1 = ctx.matmul(ln(x, w.lnconv), w.pw1T, { bias: w.pw1b }); // [n, 2D]
        const glu = ctx.glu(ctx.transpose(pre1)); // [D, n]
        const gfull = ctx.alloc(D, dwK - 1 + n);
        ctx.setCols(gfull, st.cc[L] || st.zeroCC, 0);
        ctx.setCols(gfull, glu, dwK - 1);
        swaps.push(["cc", L, ctx.pin(ctx.sliceCols(gfull, n, dwK - 1))]);
        const dopts = { cout: D, k: dwK, groups: D, padLeft: 0, padRight: 0, bias: w.dwb };
        let dwo;
        if (w.bn) {
          const d = ctx.conv1d(gfull, w.dw, dopts);
          dwo = ctx.transpose(ctx.silu(ln(ctx.transpose(d), w.bn)));
        } else {
          dwo = ctx.conv1d(gfull, w.dw, { ...dopts, act: "silu" });
        }
        x = ctx.matmul(ctx.transpose(dwo), w.pw2T, { bias: w.pw2b, add: x });
        x = ff(x, w.lnff2, w.ff2w1, w.ff2w2, w.ff2b1, w.ff2b2);
        x = ln(x, w.lnout);
      }
      if (st.proj) x = ctx.matmul(x, st.proj.w, { bias: st.proj.b });
      frames = ctx.pin ? ctx.pin(x) : x;
    });
  } finally {
    if (arena) ctx.popArena(arena);
  }
  // Batch closed: pool the mel upload, swap caches, pool the replaced generation.
  if (melUp) ctx.freeTensor(melUp);
  for (const [slot, L, t] of swaps) {
    const old = st[slot][L];
    if (old) ctx.freeTensor(old);
    st[slot][L] = t;
  }
  const data = await ctx.download(frames);
  ctx.freeTensor(frames);
  return data;
}
