// Parakeet TDT decoder + joint — hand-written JS (no onnxruntime). The prediction
// network (embedding + 2-layer LSTM, hidden 640) and joint net run per decode step
// (per encoder frame / emitted token) — tiny per step, so JS beats GPU dispatch
// overhead (same reasoning as Silero). Matches the ORT decoder_joint transcript.
//
// Weights (decoder_joint-model.onnx): decoder.prediction.embed.weight[8193,640],
// 2× LSTM (W/R[1,2560,640] iofc, B[1,5120]), joint enc[1024,640]+bias, pred[640,640]
// +bias, out[640,8198]+bias. Output 8198 = 8193 vocab + 5 TDT durations.

const HID = 640, VOCAB = 8193, LOGITS = 8198, LAYERS = 2;

export function loadParakeetDecoder(bin, man, ctx = null) {
  const g = (k) => bin.subarray(man[k].offset, man[k].offset + man[k].len);
  const dec = {
    embed: g("embed"),
    lstm: [
      { W: g("l0_W"), R: g("l0_R"), B: g("l0_B") },
      { W: g("l1_W"), R: g("l1_R"), B: g("l1_B") },
    ],
    encW: g("encW"), encB: g("encB"), predW: g("predW"), predB: g("predB"),
    outW: g("outW"), outB: g("outB"),
    blankId: VOCAB - 1, vocab: VOCAB, logits: LOGITS,
  };
  // GPU joint path: upload the joint projections so the per-frame 640→8198 matmul +
  // argmax run on the GPU (jointArgmax) instead of JS. Embedding + LSTM stay on CPU
  // (per-emission, tiny). encW/predW are [in,out] for ctx.matmul.
  if (ctx) {
    dec.gpu = {
      encW: ctx.upload(dec.encW.slice(), 1024, HID), encB: ctx.upload(dec.encB.slice(), 1, HID),
      predW: ctx.upload(dec.predW.slice(), HID, HID), predB: ctx.upload(dec.predB.slice(), 1, HID),
      outW: ctx.upload(dec.outW.slice(), HID, LOGITS), outB: ctx.upload(dec.outB.slice(), 1, LOGITS),
    };
  }
  return dec;
}

/** predProj [1,640] on GPU from a JS decoder-output vector (matmul + bias). */
function uploadPredProj(ctx, dec, decOut) {
  return ctx.matmul(ctx.upload(decOut.slice ? decOut.slice() : Float32Array.from(decOut), 1, HID), dec.gpu.predW, { bias: dec.gpu.predB });
}

/**
 * TDT greedy decode with the joint on GPU. framesGpu: [Tenc, 640... 1024] GPU tensor
 * (encoder output, rows=frames). LSTM prediction net stays on CPU (per emission).
 * Returns { ids, idFrames }. Requires dec loaded with a ctx (dec.gpu present).
 */
export async function tdtGreedyGpu(ctx, dec, framesGpu, Tenc, maxSymbols = 10, batch = 32) {
  const encProj = ctx.matmul(framesGpu, dec.gpu.encW, { bias: dec.gpu.encB }); // [Tenc,640]
  const ids = [], idFrames = [];
  let state = newDecoderState();
  let lastTok = dec.blankId;
  let pred = predict(dec, lastTok, state);
  let predProj = uploadPredProj(ctx, dec, pred.decOut);
  let t = 0, emitted = 0;
  // predProj is constant until an emission, so joints for a run of frames can be
  // computed in one dispatch. Replay the batch in JS; on emission, predProj changes
  // → the rest of the batch is stale, so break and re-dispatch from the current frame.
  while (t < Tenc) {
    const base = t;
    const count = Math.min(batch, Tenc - t);
    const res = await ctx.download(ctx.jointArgmax(encProj, base, count, predProj, dec.gpu.outW, dec.gpu.outB, HID, dec.vocab, dec.logits));
    while (t - base < count) {
      const b = t - base;
      const maxId = res[b * 4] | 0, step = res[b * 4 + 2] | 0;
      if (maxId !== dec.blankId) {
        state = pred.state; lastTok = maxId; ids.push(maxId); idFrames.push(t); emitted++;
        pred = predict(dec, lastTok, state);
        predProj = uploadPredProj(ctx, dec, pred.decOut);
        if (step > 0) { t += step; emitted = 0; }
        else if (emitted >= maxSymbols) { t += 1; emitted = 0; }
        break; // predProj changed → remaining batch results are stale
      }
      if (step > 0) { t += step; emitted = 0; }
      else { t += 1; emitted = 0; }
    }
  }
  return { ids, idFrames };
}

/** Fresh zero decoder state: h/c for each of the 2 LSTM layers. */
export function newDecoderState() {
  return { h: [new Float32Array(HID), new Float32Array(HID)], c: [new Float32Array(HID), new Float32Array(HID)] };
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// One LSTM step (ONNX iofc), reading h/c, writing into nh/nc. x, h, c: [HID].
function lstmStep(x, h, c, W, R, B, nh, nc) {
  const H = HID;
  for (let g = 0; g < H; g++) {
    let zi = B[g] + B[4 * H + g];
    let zo = B[H + g] + B[5 * H + g];
    let zf = B[2 * H + g] + B[6 * H + g];
    let zc = B[3 * H + g] + B[7 * H + g];
    const wi = g * H, wo = (H + g) * H, wf = (2 * H + g) * H, wc = (3 * H + g) * H;
    for (let j = 0; j < H; j++) { const xj = x[j]; zi += W[wi + j] * xj; zo += W[wo + j] * xj; zf += W[wf + j] * xj; zc += W[wc + j] * xj; }
    for (let j = 0; j < H; j++) { const hj = h[j]; zi += R[wi + j] * hj; zo += R[wo + j] * hj; zf += R[wf + j] * hj; zc += R[wc + j] * hj; }
    const cc = sigmoid(zf) * c[g] + sigmoid(zi) * Math.tanh(zc);
    nc[g] = cc; nh[g] = sigmoid(zo) * Math.tanh(cc);
  }
}

/**
 * Run the prediction net for `token` from `state` → decoder output [640] and the
 * candidate next state (fresh; caller keeps it only when a non-blank is emitted).
 */
export function predict(dec, token, state) {
  const nh = [new Float32Array(HID), new Float32Array(HID)];
  const nc = [new Float32Array(HID), new Float32Array(HID)];
  let inp = dec.embed.subarray(token * HID, token * HID + HID);
  for (let l = 0; l < LAYERS; l++) {
    lstmStep(inp, state.h[l], state.c[l], dec.lstm[l].W, dec.lstm[l].R, dec.lstm[l].B, nh[l], nc[l]);
    inp = nh[l];
  }
  return { decOut: nh[LAYERS - 1], state: { h: nh, c: nc } };
}

/**
 * Joint network: encoder frame [1024] + decoder output [640] → logits [8198].
 * logits[0:8193] = token scores, logits[8193:8198] = TDT duration scores.
 */
export function joint(dec, encFrame, decOut) {
  const H = HID;
  const j = new Float32Array(H);
  for (let n = 0; n < H; n++) {
    let e = dec.encB[n], p = dec.predB[n];
    for (let k = 0; k < 1024; k++) e += encFrame[k] * dec.encW[k * H + n];
    for (let k = 0; k < H; k++) p += decOut[k] * dec.predW[k * H + n];
    const s = e + p; j[n] = s > 0 ? s : 0; // ReLU
  }
  const out = new Float32Array(LOGITS);
  for (let n = 0; n < LOGITS; n++) { let s = dec.outB[n]; for (let k = 0; k < H; k++) s += j[k] * dec.outW[k * LOGITS + n]; out[n] = s; }
  return out;
}

/**
 * TDT greedy decode over encoder frames[Tenc][1024] (frames row-major, frames[t*1024+d]).
 * Returns { ids, idFrames } (idFrames[k] = encoder frame token k was emitted at, for
 * window-seam dedup). Advances by predicted duration; updates the prediction net
 * (LSTM state + last token) only on non-blank emission.
 */
export function tdtGreedy(dec, frames, Tenc, maxSymbols = 10) {
  const D = 1024;
  const ids = [], idFrames = [];
  let state = newDecoderState();
  let lastTok = dec.blankId;
  let t = 0, emitted = 0;
  const enc = new Float32Array(D);
  // The prediction net (LSTM) output depends only on lastTok + state, which change
  // ONLY on a non-blank emission. Cache it so blank-advance frames (the majority)
  // reuse it instead of recomputing the LSTM — the joint still runs every frame.
  let pred = predict(dec, lastTok, state);
  while (t < Tenc) {
    enc.set(frames.subarray(t * D, t * D + D));
    const lg = joint(dec, enc, pred.decOut);
    let maxId = 0, maxV = -Infinity;
    for (let i = 0; i < dec.vocab; i++) if (lg[i] > maxV) { maxV = lg[i]; maxId = i; }
    let step = 0, dV = -Infinity;
    for (let i = dec.vocab; i < dec.logits; i++) if (lg[i] > dV) { dV = lg[i]; step = i - dec.vocab; }
    if (maxId !== dec.blankId) {
      state = pred.state; lastTok = maxId; ids.push(maxId); idFrames.push(t); emitted++;
      pred = predict(dec, lastTok, state); // recompute only after emission
    }
    if (step > 0) { t += step; emitted = 0; }
    else if (maxId === dec.blankId || emitted >= maxSymbols) { t += 1; emitted = 0; }
  }
  return { ids, idFrames };
}
