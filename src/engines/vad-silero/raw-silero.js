// Silero VAD v5 forward, hand-written (no onnxruntime). The model is tiny (2.2 MB,
// operates on 512-sample / 32 ms chunks with LSTM state carried across chunks), so
// a plain-JS forward beats GPU here — per-chunk GPU dispatch+readback overhead
// would dominate the trivial compute. Parity-gated vs the ORT output.
//
// Architecture (16 kHz), recovered from silero_vad.onnx:
//   input[512] → reflect-pad 64 at end → STFT (Conv k256 s128, basis[258,1,256])
//   → magnitude (|real+i·imag|, 129 bins) → enc0 Conv[128,129,3] p1 s1 +ReLU
//   → enc1 Conv[64,128,3] p1 s2 +ReLU → enc2 Conv[64,64,3] p1 s2 +ReLU
//   → enc3 Conv[128,64,3] p1 s1 +ReLU → LSTM(in128 hid128, iofc, state[2,1,128])
//   → ReLU → decoder Conv[1,128,1] → Sigmoid → prob.

const HID = 128;

/** Weights: each field is a Float32Array; conv weights are ONNX [Cout,Cin,K]. */
export function makeSileroWeights(bin, manifest) {
  const f = (k) => {
    const m = manifest[k];
    return bin.subarray(m.offset, m.offset + m.len);
  };
  return {
    stftBasis: f("stft_basis"), // [258,1,256]
    enc: [
      { w: f("enc0_w"), b: f("enc0_b"), cout: 128, cin: 129, k: 3, stride: 1 },
      { w: f("enc1_w"), b: f("enc1_b"), cout: 64, cin: 128, k: 3, stride: 2 },
      { w: f("enc2_w"), b: f("enc2_b"), cout: 64, cin: 64, k: 3, stride: 2 },
      { w: f("enc3_w"), b: f("enc3_b"), cout: 128, cin: 64, k: 3, stride: 1 },
    ],
    lstmW: f("lstm_W"), // [1,512,128]  (4H, input)
    lstmR: f("lstm_R"), // [1,512,128]  (4H, hidden)
    lstmB: f("lstm_B"), // [1,1024]     (Wb[512] then Rb[512])
    decW: f("dec_w"), // [1,128,1]
    decB: f("dec_b"), // [1]
  };
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Reflect-pad `n` samples at the END (ONNX reflect: mirror without repeating edge).
function reflectPadEnd(x, n) {
  const out = new Float32Array(x.length + n);
  out.set(x);
  const last = x.length - 1;
  for (let i = 0; i < n; i++) out[x.length + i] = x[last - 1 - i];
  return out;
}

// STFT magnitude: conv `xp` against basis[258,1,256] (stride 128), then combine the
// 129 real + 129 imag channels into 129 magnitude bins. Returns { mag:[129*T], T }.
function stftMag(xp, basis, k = 256, stride = 128) {
  const T = Math.floor((xp.length - k) / stride) + 1;
  const bins = 129;
  const mag = new Float32Array(bins * T);
  for (let t = 0; t < T; t++) {
    const base = t * stride;
    for (let f = 0; f < bins; f++) {
      let re = 0;
      let im = 0;
      const rOff = f * k; // basis row f (real)
      const iOff = (f + bins) * k; // basis row f+129 (imag)
      for (let j = 0; j < k; j++) {
        const s = xp[base + j];
        re += s * basis[rOff + j];
        im += s * basis[iOff + j];
      }
      mag[f * T + t] = Math.sqrt(re * re + im * im);
    }
  }
  return { mag, T };
}

// conv1d: x[Cin*L] row-major (rows=Cin), w ONNX [Cout,Cin,K], bias[Cout].
// pad=1 both sides, given stride. ReLU applied when relu=true. Returns [Cout*Lout].
function conv1d(x, Cin, L, w, b, Cout, k, stride, pad, relu) {
  const Lout = Math.floor((L + 2 * pad - k) / stride) + 1;
  const y = new Float32Array(Cout * Lout);
  for (let co = 0; co < Cout; co++) {
    const wc = co * Cin * k;
    const bias = b[co];
    for (let t = 0; t < Lout; t++) {
      let acc = bias;
      const start = t * stride - pad;
      for (let ci = 0; ci < Cin; ci++) {
        const xr = ci * L;
        const wr = wc + ci * k;
        for (let kk = 0; kk < k; kk++) {
          const idx = start + kk;
          if (idx >= 0 && idx < L) acc += x[xr + idx] * w[wr + kk];
        }
      }
      y[co * Lout + t] = relu && acc < 0 ? 0 : acc;
    }
  }
  return { y, Lout };
}

// Single LSTM step (ONNX iofc gate order), batch 1. h/c are Float32Array[HID],
// updated in place. x = input features[HID].
function lstmStep(x, h, c, W, R, B) {
  const H = HID;
  const nh = new Float32Array(H);
  const nc = new Float32Array(H);
  for (let g = 0; g < H; g++) {
    // gate rows: i=[0,H), o=[H,2H), f=[2H,3H), cell=[3H,4H)
    let zi = B[g] + B[4 * H + g];
    let zo = B[H + g] + B[4 * H + H + g];
    let zf = B[2 * H + g] + B[4 * H + 2 * H + g];
    let zc = B[3 * H + g] + B[4 * H + 3 * H + g];
    const wi = g * H,
      wo = (H + g) * H,
      wf = (2 * H + g) * H,
      wcc = (3 * H + g) * H;
    for (let j = 0; j < H; j++) {
      const xj = x[j];
      zi += W[wi + j] * xj;
      zo += W[wo + j] * xj;
      zf += W[wf + j] * xj;
      zc += W[wcc + j] * xj;
    }
    for (let j = 0; j < H; j++) {
      const hj = h[j];
      zi += R[wi + j] * hj;
      zo += R[wo + j] * hj;
      zf += R[wf + j] * hj;
      zc += R[wcc + j] * hj;
    }
    const it = sigmoid(zi);
    const ot = sigmoid(zo);
    const ft = sigmoid(zf);
    const gt = Math.tanh(zc);
    const cc = ft * c[g] + it * gt;
    nc[g] = cc;
    nh[g] = ot * Math.tanh(cc);
  }
  h.set(nh);
  c.set(nc);
}

/**
 * One VAD step. `x`: Float32Array[512]. `state`: Float32Array[256] (h[0:128], c[128:256]).
 * Returns { prob, state } where state is a fresh Float32Array[256] for the next chunk.
 */
export function sileroForward(x, state, W) {
  const xp = reflectPadEnd(x, 64);
  const { mag, T } = stftMag(xp, W.stftBasis);
  let cur = mag,
    Cin = 129,
    L = T;
  for (const e of W.enc) {
    const r = conv1d(cur, Cin, L, e.w, e.b, e.cout, e.k, e.stride, 1, true);
    cur = r.y;
    L = r.Lout;
    Cin = e.cout;
  }
  // encoder output [128, L]; Silero uses the last time frame into the LSTM.
  const feat = new Float32Array(HID);
  for (let ci = 0; ci < HID; ci++) feat[ci] = cur[ci * L + (L - 1)];

  const h = state.slice(0, HID);
  const c = state.slice(HID, 2 * HID);
  lstmStep(feat, h, c, W.lstmW, W.lstmR, W.lstmB);

  // decoder: ReLU(h) → pointwise conv[1,128,1] → sigmoid
  let acc = W.decB[0];
  for (let j = 0; j < HID; j++) acc += W.decW[j] * (h[j] > 0 ? h[j] : 0);
  const prob = sigmoid(acc);

  const nextState = new Float32Array(2 * HID);
  nextState.set(h, 0);
  nextState.set(c, HID);
  return { prob, state: nextState };
}
