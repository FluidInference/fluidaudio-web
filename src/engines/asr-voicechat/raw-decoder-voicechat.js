// VoiceChat-11B RNNT decoder+joint — hand-written JS (no onnxruntime). Adapted
// from asr-nemotron/raw-decoder-nemotron.js with the VoiceChat geometry: vocab
// 1024 + blank → LOGITS 1025, BLANK 1024, and NO prompt_kernel (that MLP is
// Nemotron's multilingual conditioning; VoiceChat's RNNT taps asr_emb directly).
// 2-layer LSTM prediction net (hidden 640) + joint (enc 1024→640 — run GPU-side,
// frames arrive pre-projected — pred 640→640, relu, out 640→1025). Like Nemotron,
// no zero-SOS prepend: the stream opens with one plain LSTM step on BLANK.
// ONNX iofc gate order, bias = [Wb(4H) | Rb(4H)] (the extractor permutes
// PyTorch's ifgo). The 1025-wide out matmul is small enough that scalar JS with
// a predProj cache across blanks decodes ~200 frames in single-digit ms — no
// wasm path needed.

const HID = 640,
  LOGITS = 1025,
  BLANK = 1024;

export { BLANK as VC_BLANK, LOGITS as VC_LOGITS };

export function loadVoicechatDecoder(bin, man) {
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

// Joint pred-side projection [640] — constant across blank frames, cached in the
// stream state so a blank-only frame costs one 640×1025 matvec.
function predProj(dec, decOut) {
  const p = new Float32Array(HID);
  for (let n = 0; n < HID; n++) {
    let s = dec.predB[n];
    for (let k = 0; k < HID; k++) s += decOut[k] * dec.predW[k * HID + n];
    p[n] = s;
  }
  return p;
}

const zeroState = () => ({ h: [new Float32Array(HID), new Float32Array(HID)], c: [new Float32Array(HID), new Float32Array(HID)] });

/** Fresh decode state for streaming continuation (voicechatDecodeCont). */
export function createVoicechatStream(dec) {
  const pred = predict(dec, BLANK, zeroState());
  return { pred, pp: predProj(dec, pred.decOut) };
}

/** RNNT greedy over PRE-PROJECTED frames [Tenc, 640] (joint enc proj rides the
 * GPU encode batch), CONTINUING from `st` — a chunk boundary is invisible to the
 * decoder. Returns { ids, idFrames } (frame indices relative to this call). */
export function voicechatDecodeCont(dec, st, framesProj, Tenc, maxSymbols = 10) {
  const ids = [];
  const idFrames = [];
  const j = new Float32Array(HID);
  let t = 0,
    emitted = 0;
  while (t < Tenc) {
    const enc = framesProj.subarray(t * HID, t * HID + HID);
    for (let n = 0; n < HID; n++) {
      const s = enc[n] + st.pp[n];
      j[n] = s > 0 ? s : 0;
    }
    // argmax over out 640→1025 (argmax invariant to LogSoftmax)
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
    if (maxId === BLANK || emitted >= maxSymbols) {
      t += 1;
      emitted = 0;
      continue;
    }
    ids.push(maxId);
    idFrames.push(t);
    st.pred = predict(dec, maxId, st.pred);
    st.pp = predProj(dec, st.pred.decOut);
    emitted++;
  }
  return { ids, idFrames };
}

/** RNNT greedy over pre-projected frames [Tenc, 640] from a fresh state. */
export function voicechatDecode(dec, framesProj, Tenc, maxSymbols = 10) {
  return voicechatDecodeCont(dec, createVoicechatStream(dec), framesProj, Tenc, maxSymbols);
}
