// Nemotron RNNT decoder+joint — hand-written JS (no onnxruntime). 2-layer LSTM
// prediction net (hidden 640) + joint (enc 1024→640, pred 640→640, relu, out
// 640→13088). Unlike EOU's fused export, the Nemotron decoder does NOT prepend a
// zero SOS timestep — it's a plain single LSTM step per call. blank = 13087.
// ONNX iofc gate order, bias = [Wb(4H) | Rb(4H)].

const HID = 640, ENC_D = 1024, LOGITS = 13088, BLANK = 13087;

export function loadNemotronDecoder(bin, man) {
  const g = (k) => bin.subarray(man[k].offset, man[k].offset + man[k].len);
  return {
    embed: g("embed"),
    l0: { W: g("l0_W"), R: g("l0_R"), B: g("l0_B") },
    l1: { W: g("l1_W"), R: g("l1_R"), B: g("l1_B") },
    encW: g("encW"), encB: g("encB"), predW: g("predW"), predB: g("predB"),
    outW: g("outW"), outB: g("outB"),
  };
}

const sig = (x) => 1 / (1 + Math.exp(-x));

// One ONNX-iofc LSTM step: x,h,c[HID] → nh,nc[HID].
function lstmStep(x, h, c, W, R, B, nh, nc) {
  const H = HID;
  for (let g = 0; g < H; g++) {
    let zi = B[g] + B[4 * H + g], zo = B[H + g] + B[5 * H + g], zf = B[2 * H + g] + B[6 * H + g], zc = B[3 * H + g] + B[7 * H + g];
    const wi = g * H, wo = (H + g) * H, wf = (2 * H + g) * H, wc = (3 * H + g) * H;
    for (let j = 0; j < H; j++) { const xj = x[j]; zi += W[wi + j] * xj; zo += W[wo + j] * xj; zf += W[wf + j] * xj; zc += W[wc + j] * xj; }
    for (let j = 0; j < H; j++) { const hj = h[j]; zi += R[wi + j] * hj; zo += R[wo + j] * hj; zf += R[wf + j] * hj; zc += R[wc + j] * hj; }
    const cc = sig(zf) * c[g] + sig(zi) * Math.tanh(zc);
    nc[g] = cc; nh[g] = sig(zo) * Math.tanh(cc);
  }
}

// Prediction net: embed(token) → LSTM0 → LSTM1. state = {h:[h0,h1], c:[c0,c1]}.
function predict(dec, token, state) {
  const x = dec.embed.subarray(token * HID, token * HID + HID);
  const nh0 = new Float32Array(HID), nc0 = new Float32Array(HID);
  lstmStep(x, state.h[0], state.c[0], dec.l0.W, dec.l0.R, dec.l0.B, nh0, nc0);
  const nh1 = new Float32Array(HID), nc1 = new Float32Array(HID);
  lstmStep(nh0, state.h[1], state.c[1], dec.l1.W, dec.l1.R, dec.l1.B, nh1, nc1);
  return { decOut: nh1, h: [nh0, nh1], c: [nc0, nc1] };
}

// joint: enc[1024] + decOut[640] → argmax over 13088 (argmax invariant to LogSoftmax).
function jointArgmax(dec, encFrame, decOut) {
  const j = new Float32Array(HID);
  for (let n = 0; n < HID; n++) {
    let e = dec.encB[n], p = dec.predB[n];
    for (let k = 0; k < ENC_D; k++) e += encFrame[k] * dec.encW[k * HID + n];
    for (let k = 0; k < HID; k++) p += decOut[k] * dec.predW[k * HID + n];
    const s = e + p; j[n] = s > 0 ? s : 0;
  }
  let maxId = 0, maxV = -Infinity;
  for (let n = 0; n < LOGITS; n++) {
    let s = dec.outB[n];
    for (let k = 0; k < HID; k++) s += j[k] * dec.outW[k * LOGITS + n];
    if (s > maxV) { maxV = s; maxId = n; }
  }
  return maxId;
}

const zeroState = () => ({ h: [new Float32Array(HID), new Float32Array(HID)], c: [new Float32Array(HID), new Float32Array(HID)] });

/** RNNT greedy over frames[Tenc*1024] (row-major). Returns { ids } (text token ids). */
export function nemotronDecode(dec, frames, Tenc, maxSymbols = 10) {
  const ids = [];
  let state = zeroState();
  let pred = predict(dec, BLANK, state);
  const enc = new Float32Array(ENC_D);
  let t = 0, emitted = 0;
  while (t < Tenc) {
    enc.set(frames.subarray(t * ENC_D, t * ENC_D + ENC_D));
    const maxId = jointArgmax(dec, enc, pred.decOut);
    if (maxId === BLANK || emitted >= maxSymbols) { t += 1; emitted = 0; continue; }
    ids.push(maxId);
    pred = predict(dec, maxId, pred);
    emitted++;
  }
  return { ids };
}
