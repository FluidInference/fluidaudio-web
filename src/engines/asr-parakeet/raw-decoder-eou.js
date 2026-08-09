// Parakeet-EOU decoder+joint — hand-written JS (no onnxruntime). RNNT (not TDT):
// advances one encoder frame per blank, emits tokens otherwise; 1-layer LSTM (hid 640),
// vocab 1027 (blank 1026; EOU 1024 / EOB 1025 are end-of-utterance events, dropped from
// text). Encoder is d512, so the joint's enc projection is 512→640. Small (joint 640→1027,
// 8× smaller than Parakeet) → plain JS is fast enough.

const HID = 640,
  ENC_D = 512,
  LOGITS = 1027,
  BLANK = 1026,
  EOU = 1024,
  EOB = 1025;

export function loadEouDecoder(bin, man) {
  const g = (k) => bin.subarray(man[k].offset, man[k].offset + man[k].len);
  return {
    embed: g("embed"),
    W: g("lstm_W"),
    R: g("lstm_R"),
    B: g("lstm_B"),
    encW: g("encW"),
    encB: g("encB"),
    predW: g("predW"),
    predB: g("predB"),
    outW: g("outW"),
    outB: g("outB"),
  };
}

const sig = (x) => 1 / (1 + Math.exp(-x));

// Single LSTM step (ONNX iofc), h/c updated into nh/nc.
function lstmStep(x, h, c, W, R, B, nh, nc) {
  const H = HID;
  for (let g = 0; g < H; g++) {
    let zi = B[g] + B[4 * H + g],
      zo = B[H + g] + B[5 * H + g],
      zf = B[2 * H + g] + B[6 * H + g],
      zc = B[3 * H + g] + B[7 * H + g];
    const wi = g * H,
      wo = (H + g) * H,
      wf = (2 * H + g) * H,
      wc = (3 * H + g) * H;
    for (let j = 0; j < H; j++) {
      const xj = x[j];
      zi += W[wi + j] * xj;
      zo += W[wo + j] * xj;
      zf += W[wf + j] * xj;
      zc += W[wc + j] * xj;
    }
    for (let j = 0; j < H; j++) {
      const hj = h[j];
      zi += R[wi + j] * hj;
      zo += R[wo + j] * hj;
      zf += R[wf + j] * hj;
      zc += R[wc + j] * hj;
    }
    const cc = sig(zf) * c[g] + sig(zi) * Math.tanh(zc);
    nc[g] = cc;
    nh[g] = sig(zo) * Math.tanh(cc);
  }
}

const ZEROS = new Float32Array(HID);

// Prediction net for `token` from state → decOut + new state (fresh arrays).
// The exported decoder_joint prepends a zero SOS timestep on every call, so the
// LSTM runs TWO steps: [zeros, embed(token)] from the incoming (h,c). Matching
// this exactly is required for byte-parity with the ONNX joint.
function predict(dec, token, h, c) {
  const h1 = new Float32Array(HID),
    c1 = new Float32Array(HID);
  lstmStep(ZEROS, h, c, dec.W, dec.R, dec.B, h1, c1); // SOS prepend
  const nh = new Float32Array(HID),
    nc = new Float32Array(HID);
  lstmStep(dec.embed.subarray(token * HID, token * HID + HID), h1, c1, dec.W, dec.R, dec.B, nh, nc);
  return { decOut: nh, h: nh, c: nc };
}

// joint: enc[512] + decOut[640] → logits[1027] (argmax invariant to the trailing LogSoftmax).
function joint(dec, encFrame, decOut, out) {
  for (let n = 0; n < HID; n++) {
    let e = dec.encB[n],
      p = dec.predB[n];
    for (let k = 0; k < ENC_D; k++) e += encFrame[k] * dec.encW[k * HID + n];
    for (let k = 0; k < HID; k++) p += decOut[k] * dec.predW[k * HID + n];
    const s = e + p;
    out[n] = s > 0 ? s : 0; // relu → reuse `out` as j scratch
  }
  const j = out.slice(0, HID);
  for (let n = 0; n < LOGITS; n++) {
    let s = dec.outB[n];
    for (let k = 0; k < HID; k++) s += j[k] * dec.outW[k * LOGITS + n];
    out[n] = s;
  }
}

/** Fresh decode state for streaming continuation (eouDecodeCont). */
export function createEouStream(dec) {
  return { pred: predict(dec, BLANK, new Float32Array(HID), new Float32Array(HID)) };
}

/**
 * RNNT greedy over frames[Tenc*512], CONTINUING from `st` (LSTM state + last
 * emission persist across calls — a chunk boundary is invisible to the
 * decoder). frame indices in the result are offset by `frameOffset`.
 */
export function eouDecodeCont(dec, st, frames, Tenc, frameOffset = 0, maxSymbols = 10) {
  const ids = [],
    idFrames = [],
    events = [];
  const enc = new Float32Array(ENC_D);
  const out = new Float32Array(LOGITS);
  let t = 0,
    emitted = 0;
  while (t < Tenc) {
    enc.set(frames.subarray(t * ENC_D, t * ENC_D + ENC_D));
    joint(dec, enc, st.pred.decOut, out);
    let maxId = 0,
      maxV = -Infinity;
    for (let i = 0; i < LOGITS; i++)
      if (out[i] > maxV) {
        maxV = out[i];
        maxId = i;
      }
    if (maxId === BLANK || emitted >= maxSymbols) {
      t += 1;
      emitted = 0;
      continue;
    }
    // non-blank emission
    if (maxId === EOU || maxId === EOB) events.push({ type: maxId === EOU ? "eou" : "eob", frame: frameOffset + t });
    else {
      ids.push(maxId);
      idFrames.push(frameOffset + t);
    }
    st.pred = predict(dec, maxId, st.pred.h, st.pred.c);
    emitted++;
  }
  return { ids, idFrames, events };
}

/**
 * RNNT greedy over frames[Tenc*512] (row-major, frames[t*512+d]). Returns
 * { ids, idFrames, events } — ids = text tokens (<1024), events = {type:'eou'|'eob', frame}.
 */
export function eouDecode(dec, frames, Tenc, maxSymbols = 10) {
  return eouDecodeCont(dec, createEouStream(dec), frames, Tenc, 0, maxSymbols);
}

// ── wasm-SIMD decode (rust/parakeet-decoder EOU section) ──────────────────
// The JS loop above recomputes BOTH joint projections every frame — fine for
// clips, the bottleneck at hour scale (~52× decode RTFx). The wasm path takes
// PRE-PROJECTED frames [T,640] (one batched 512→640 GEMM on the backend),
// caches predProj across blank frames, and SIMD-vectorizes the 640→1027 out
// matmul with zero-row skip. Stream state (LSTM h/c + predProj) lives in wasm
// statics: eouWasmDecodeCont continues it, eouWasmReset starts fresh — one
// stream per instance.

/** Instantiate the shared decoder wasm and load the EOU weights. */
export async function loadEouWasmDecoder(wasmBytes, bin, man) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const ex = instance.exports;
  ex.reset_to(ex.__heap_base.value);
  const g = (k) => bin.subarray(man[k].offset, man[k].offset + man[k].len);
  const put = (arr) => {
    const p = ex.alloc(arr.byteLength);
    new Float32Array(ex.memory.buffer, p, arr.length).set(arr);
    return p;
  };
  ex.eou_set_weights(put(g("embed")), put(g("lstm_W")), put(g("lstm_R")), put(g("lstm_B")), put(g("predW")), put(g("predB")), put(g("outW")), put(g("outB")));
  ex.eou_reset();
  return { ex, mark: ex.bump_mark() };
}

export function eouWasmReset(wd) {
  wd.ex.eou_reset();
}

/** RNNT greedy over PRE-PROJECTED frames [Tenc,640] (encB included), CONTINUING
 * the wasm stream state. Same result shape as eouDecodeCont. */
export function eouWasmDecodeCont(wd, framesProj, Tenc, frameOffset = 0, maxSymbols = 10) {
  const ex = wd.ex;
  ex.reset_to(wd.mark);
  // Alloc everything BEFORE writing: alloc may grow memory and detach views.
  const fp = ex.alloc(framesProj.byteLength);
  const cap = Math.max(1, Tenc * maxSymbols);
  const ip = ex.alloc(cap * 4);
  const tp = ex.alloc(cap * 4);
  new Float32Array(ex.memory.buffer, fp, framesProj.length).set(framesProj);
  const n = ex.eou_decode_cont(fp, Tenc, ip, tp, maxSymbols);
  const rawIds = new Int32Array(ex.memory.buffer, ip, n);
  const rawFr = new Int32Array(ex.memory.buffer, tp, n);
  const ids = [],
    idFrames = [],
    events = [];
  for (let i = 0; i < n; i++) {
    const id = rawIds[i],
      fr = frameOffset + rawFr[i];
    if (id === EOU || id === EOB) events.push({ type: id === EOU ? "eou" : "eob", frame: fr });
    else {
      ids.push(id);
      idFrames.push(fr);
    }
  }
  return { ids, idFrames, events };
}
