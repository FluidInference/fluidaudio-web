// Nemotron RNNT decoder+joint — hand-written JS (no onnxruntime). 2-layer LSTM
// prediction net (hidden 640) + joint (enc 1024→640, pred 640→640, relu, out
// 640→13088). Unlike EOU's fused export, the Nemotron decoder does NOT prepend a
// zero SOS timestep — it's a plain single LSTM step per call. blank = 13087.
// ONNX iofc gate order, bias = [Wb(4H) | Rb(4H)].

const HID = 640,
  ENC_D = 1024,
  LOGITS = 13088,
  BLANK = 13087;

export function loadNemotronDecoder(bin, man) {
  const g = (k) => bin.subarray(man[k].offset, man[k].offset + man[k].len);
  return {
    embed: g("embed"),
    l0: { W: g("l0_W"), R: g("l0_R"), B: g("l0_B") },
    l1: { W: g("l1_W"), R: g("l1_R"), B: g("l1_B") },
    encW: g("encW"),
    encB: g("encB"),
    predW: g("predW"),
    predB: g("predB"),
    outW: g("outW"),
    outB: g("outB"),
  };
}

// Multilingual prompt_kernel: applied AFTER the conformer stack. Concats the
// conformer output [1024] with the language one-hot [128], then a 2-layer MLP
// (1152→2048→1024) produces the final encoded_output. Weights live in the encoder
// manifest (pk0w/pk0b/pk2w/pk2b). f32bin = Float32Array over the encoder blob.
export function loadPromptKernel(f32bin, man) {
  const g = (k) => f32bin.subarray(man[k].offset, man[k].offset + (man[k].count ?? man[k].len));
  return { pk0w: g("pk0w"), pk0b: g("pk0b"), pk2w: g("pk2w"), pk2b: g("pk2b") };
}

/** conformer frames[Tsub*1024] → encoded_output[Tsub*1024] for language `langId`. */
export function applyPromptKernel(pk, frames, Tsub, langId = 0) {
  const D = 1024,
    IN = 1152,
    H = 2048;
  const out = new Float32Array(Tsub * D),
    inp = new Float32Array(IN),
    hb = new Float32Array(H);
  for (let t = 0; t < Tsub; t++) {
    for (let d = 0; d < D; d++) inp[d] = frames[t * D + d];
    for (let d = D; d < IN; d++) inp[d] = 0;
    inp[D + langId] = 1;
    for (let h = 0; h < H; h++) {
      let s = pk.pk0b[h];
      for (let k = 0; k < IN; k++) s += inp[k] * pk.pk0w[k * H + h];
      hb[h] = s > 0 ? s : 0;
    }
    for (let d = 0; d < D; d++) {
      let s = pk.pk2b[d];
      for (let k = 0; k < H; k++) s += hb[k] * pk.pk2w[k * D + d];
      out[t * D + d] = s;
    }
  }
  return out;
}

const sig = (x) => 1 / (1 + Math.exp(-x));

// One ONNX-iofc LSTM step: x,h,c[HID] → nh,nc[HID].
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

// Prediction net: embed(token) → LSTM0 → LSTM1. state = {h:[h0,h1], c:[c0,c1]}.
function predict(dec, token, state) {
  const x = dec.embed.subarray(token * HID, token * HID + HID);
  const nh0 = new Float32Array(HID),
    nc0 = new Float32Array(HID);
  lstmStep(x, state.h[0], state.c[0], dec.l0.W, dec.l0.R, dec.l0.B, nh0, nc0);
  const nh1 = new Float32Array(HID),
    nc1 = new Float32Array(HID);
  lstmStep(nh0, state.h[1], state.c[1], dec.l1.W, dec.l1.R, dec.l1.B, nh1, nc1);
  return { decOut: nh1, h: [nh0, nh1], c: [nc0, nc1] };
}

// joint: enc[1024] + decOut[640] → argmax over 13088 (argmax invariant to LogSoftmax).
function jointArgmax(dec, encFrame, decOut) {
  const j = new Float32Array(HID);
  for (let n = 0; n < HID; n++) {
    let e = dec.encB[n],
      p = dec.predB[n];
    for (let k = 0; k < ENC_D; k++) e += encFrame[k] * dec.encW[k * HID + n];
    for (let k = 0; k < HID; k++) p += decOut[k] * dec.predW[k * HID + n];
    const s = e + p;
    j[n] = s > 0 ? s : 0;
  }
  let maxId = 0,
    maxV = -Infinity;
  for (let n = 0; n < LOGITS; n++) {
    let s = dec.outB[n];
    for (let k = 0; k < HID; k++) s += j[k] * dec.outW[k * LOGITS + n];
    if (s > maxV) {
      maxV = s;
      maxId = n;
    }
  }
  return maxId;
}

const zeroState = () => ({ h: [new Float32Array(HID), new Float32Array(HID)], c: [new Float32Array(HID), new Float32Array(HID)] });

/** Fresh decode state for streaming continuation (nemotronDecodeCont). */
export function createNemotronStream(dec) {
  return { pred: predict(dec, BLANK, zeroState()) };
}

/** RNNT greedy over frames[Tenc*1024], CONTINUING from `st` — a chunk boundary
 * is invisible to the decoder. */
export function nemotronDecodeCont(dec, st, frames, Tenc, maxSymbols = 10) {
  const ids = [];
  const idFrames = [];
  const enc = new Float32Array(ENC_D);
  let t = 0,
    emitted = 0;
  while (t < Tenc) {
    enc.set(frames.subarray(t * ENC_D, t * ENC_D + ENC_D));
    const maxId = jointArgmax(dec, enc, st.pred.decOut);
    if (maxId === BLANK || emitted >= maxSymbols) {
      t += 1;
      emitted = 0;
      continue;
    }
    ids.push(maxId);
    idFrames.push(t);
    st.pred = predict(dec, maxId, st.pred);
    emitted++;
  }
  return { ids, idFrames };
}

/** RNNT greedy over frames[Tenc*1024] (row-major). Returns { ids } (text token ids). */
export function nemotronDecode(dec, frames, Tenc, maxSymbols = 10) {
  return nemotronDecodeCont(dec, createNemotronStream(dec), frames, Tenc, maxSymbols);
}

// ── wasm-SIMD decode (rust/parakeet-decoder nemo section) ───────────────────
// The JS jointArgmax recomputes enc-proj + pred-proj per frame with a 13088-
// wide out matmul in scalar JS — minutes per hour of audio. The wasm path
// takes frames that are FULLY processed GPU-side (prompt kernel + 1024→640
// projection), caches predProj across blanks, int8-quantizes the 33MB out
// matrix, and runs the axpy in v128. Stream state lives in the instance.

export async function loadNemoWasmDecoder(wasmBytes, bin, man, { int8 = true } = {}) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const ex = instance.exports;
  ex.reset_to(ex.__heap_base.value);
  const g = (k) => bin.subarray(man[k].offset, man[k].offset + man[k].len);
  const put = (arr) => {
    const ptr = ex.alloc(arr.byteLength);
    new Float32Array(ex.memory.buffer, ptr, arr.length).set(arr);
    return ptr;
  };
  ex.nemo_set_weights(
    put(g("embed")),
    put(g("l0_W")),
    put(g("l0_R")),
    put(g("l0_B")),
    put(g("l1_W")),
    put(g("l1_R")),
    put(g("l1_B")),
    put(g("predW")),
    put(g("predB")),
    put(g("outW")),
    put(g("outB")),
  );
  if (int8) {
    const ow = g("outW"); // [640][13088] row-major
    const H = 640,
      L = ow.length / H;
    const q = new Int8Array(ow.length),
      scales = new Float32Array(H);
    for (let n = 0; n < H; n++) {
      let mx = 0;
      for (let m = 0; m < L; m++) {
        const a = Math.abs(ow[n * L + m]);
        if (a > mx) mx = a;
      }
      const sc = mx / 127 || 1;
      scales[n] = sc;
      for (let m = 0; m < L; m++) q[n * L + m] = Math.max(-127, Math.min(127, Math.round(ow[n * L + m] / sc)));
    }
    const qp = ex.alloc(q.byteLength);
    new Int8Array(ex.memory.buffer, qp, q.length).set(q);
    const sp = ex.alloc(scales.byteLength);
    new Float32Array(ex.memory.buffer, sp, scales.length).set(scales);
    ex.nemo_set_out_q(qp, sp);
  }
  ex.nemo_reset();
  return { ex, mark: ex.bump_mark() };
}

export function nemoWasmReset(wd) {
  wd.ex.nemo_reset();
}

/** RNNT greedy over GPU-processed frames [Tenc,640], continuing wasm state. */
export function nemoWasmDecodeCont(wd, framesProj, Tenc, maxSymbols = 10) {
  const ex = wd.ex;
  ex.reset_to(wd.mark);
  const fp = ex.alloc(framesProj.byteLength);
  const cap = Math.max(1, Tenc * maxSymbols);
  const ip = ex.alloc(cap * 4);
  const tp = ex.alloc(cap * 4);
  new Float32Array(ex.memory.buffer, fp, framesProj.length).set(framesProj);
  const n = ex.nemo_decode_cont(fp, Tenc, ip, tp, maxSymbols);
  return { ids: Array.from(new Int32Array(ex.memory.buffer, ip, n)), idFrames: Array.from(new Int32Array(ex.memory.buffer, tp, n)) };
}
