// Parakeet-EOU decoder+joint — hand-written JS (no onnxruntime). RNNT (not TDT):
// advances one encoder frame per blank, emits tokens otherwise; 1-layer LSTM (hid 640),
// vocab 1027 (blank 1026; EOU 1024 / EOB 1025 are end-of-utterance events, dropped from
// text). Encoder is d512, so the joint's enc projection is 512→640. Small (joint 640→1027,
// 8× smaller than Parakeet) → plain JS is fast enough.

const HID = 640, ENC_D = 512, LOGITS = 1027, BLANK = 1026, EOU = 1024, EOB = 1025;

export function loadEouDecoder(bin, man) {
  const g = (k) => bin.subarray(man[k].offset, man[k].offset + man[k].len);
  return {
    embed: g("embed"), W: g("lstm_W"), R: g("lstm_R"), B: g("lstm_B"),
    encW: g("encW"), encB: g("encB"), predW: g("predW"), predB: g("predB"),
    outW: g("outW"), outB: g("outB"),
  };
}

const sig = (x) => 1 / (1 + Math.exp(-x));

// Single LSTM step (ONNX iofc), h/c updated into nh/nc.
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

const ZEROS = new Float32Array(HID);

// Prediction net for `token` from state → decOut + new state (fresh arrays).
// The exported decoder_joint prepends a zero SOS timestep on every call, so the
// LSTM runs TWO steps: [zeros, embed(token)] from the incoming (h,c). Matching
// this exactly is required for byte-parity with the ONNX joint.
function predict(dec, token, h, c) {
  const h1 = new Float32Array(HID), c1 = new Float32Array(HID);
  lstmStep(ZEROS, h, c, dec.W, dec.R, dec.B, h1, c1); // SOS prepend
  const nh = new Float32Array(HID), nc = new Float32Array(HID);
  lstmStep(dec.embed.subarray(token * HID, token * HID + HID), h1, c1, dec.W, dec.R, dec.B, nh, nc);
  return { decOut: nh, h: nh, c: nc };
}

// joint: enc[512] + decOut[640] → logits[1027] (argmax invariant to the trailing LogSoftmax).
function joint(dec, encFrame, decOut, out) {
  for (let n = 0; n < HID; n++) {
    let e = dec.encB[n], p = dec.predB[n];
    for (let k = 0; k < ENC_D; k++) e += encFrame[k] * dec.encW[k * HID + n];
    for (let k = 0; k < HID; k++) p += decOut[k] * dec.predW[k * HID + n];
    const s = e + p; out[n] = s > 0 ? s : 0; // relu → reuse `out` as j scratch
  }
  const j = out.slice(0, HID);
  for (let n = 0; n < LOGITS; n++) { let s = dec.outB[n]; for (let k = 0; k < HID; k++) s += j[k] * dec.outW[k * LOGITS + n]; out[n] = s; }
}

/**
 * RNNT greedy over frames[Tenc*512] (row-major, frames[t*512+d]). Returns
 * { ids, idFrames, events } — ids = text tokens (<1024), events = {type:'eou'|'eob', frame}.
 */
export function eouDecode(dec, frames, Tenc, maxSymbols = 10) {
  const ids = [], idFrames = [], events = [];
  let h = new Float32Array(HID), c = new Float32Array(HID), lastTok = BLANK;
  let pred = predict(dec, lastTok, h, c);
  const enc = new Float32Array(ENC_D);
  const out = new Float32Array(LOGITS);
  let t = 0, emitted = 0;
  while (t < Tenc) {
    enc.set(frames.subarray(t * ENC_D, t * ENC_D + ENC_D));
    joint(dec, enc, pred.decOut, out);
    let maxId = 0, maxV = -Infinity;
    for (let i = 0; i < LOGITS; i++) if (out[i] > maxV) { maxV = out[i]; maxId = i; }
    if (maxId === BLANK || emitted >= maxSymbols) { t += 1; emitted = 0; continue; }
    // non-blank emission
    if (maxId === EOU || maxId === EOB) events.push({ type: maxId === EOU ? "eou" : "eob", frame: t });
    else { ids.push(maxId); idFrames.push(t); }
    lastTok = maxId; pred = predict(dec, lastTok, pred.h, pred.c); emitted++;
  }
  return { ids, idFrames, events };
}
