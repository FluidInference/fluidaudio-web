// Verify every WasmContext kernel against a CPU reference — the parity gate for the
// WASM (non-WebGPU) backend. Same references as gpu-verify.mjs. No GPU needed.
//   node scripts/wasm-verify.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createWasmContext } from "../src/gpu/wasm-context.js";

const wasmBytes = readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url)));
const ctx = await createWasmContext(wasmBytes);
const rand = (n) => Float32Array.from({ length: n }, () => Math.random() * 2 - 1);
const maxErr = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
let fails = 0;
const report = (name, err, tol) => { const ok = err <= tol; if (!ok) fails++; console.log(`${ok ? "✓" : "✗"} ${name.padEnd(28)} max|wasm-cpu| = ${err.toExponential(2)}  (tol ${tol.toExponential(0)})`); };
const dl = (t) => t.data;

const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)));
const silu = (x) => x / (1 + Math.exp(-x));

// matmul + bias + gelu
{
  const M = 80, K = 512, N = 300, A = rand(M * K), B = rand(K * N), bias = rand(N);
  const ref = new Float32Array(M * N);
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) { let s = bias[j]; for (let k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j]; ref[i * N + j] = gelu(s); }
  report("matmul+bias+gelu", maxErr(dl(ctx.matmul(ctx.upload(A, M, K), ctx.upload(B, K, N), { bias: ctx.upload(bias, 1, N), act: "gelu" })), ref), 1e-3);
}
// matmul + silu
{
  const M = 32, K = 128, N = 64, A = rand(M * K), B = rand(K * N);
  const ref = new Float32Array(M * N);
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) { let s = 0; for (let k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j]; ref[i * N + j] = silu(s); }
  report("matmul + silu", maxErr(dl(ctx.matmul(ctx.upload(A, M, K), ctx.upload(B, K, N), { act: "silu" })), ref), 1e-3);
}
// conv1d (regular / dilated / depthwise)
function convRefCPU(X, W, bias, Cin, L, Cout, K, stride, pad, dil, groups) {
  const Lout = Math.floor((L + 2 * pad - dil * (K - 1) - 1) / stride) + 1, Y = new Float32Array(Cout * Lout);
  const cinG = Cin / groups, coutG = Cout / groups;
  for (let co = 0; co < Cout; co++) { const g = Math.floor(co / coutG);
    for (let lo = 0; lo < Lout; lo++) { let acc = bias ? bias[co] : 0;
      for (let ci = 0; ci < cinG; ci++) { const rc = g * cinG + ci;
        for (let k = 0; k < K; k++) { const li = lo * stride + k * dil - pad; if (li >= 0 && li < L) acc += X[rc * L + li] * W[(co * cinG + ci) * K + k]; } }
      Y[co * Lout + lo] = acc; } }
  return Y;
}
for (const cfg of [
  { name: "conv1d k3 pad1", Cin: 64, L: 200, Cout: 128, K: 3, stride: 1, pad: 1, dil: 1, groups: 1 },
  { name: "conv1d k5 dil2", Cin: 32, L: 128, Cout: 32, K: 5, stride: 1, pad: 4, dil: 2, groups: 1 },
  { name: "conv1d depthwise", Cin: 80, L: 100, Cout: 80, K: 7, stride: 1, pad: 3, dil: 1, groups: 80 },
  { name: "conv1d stride2", Cin: 16, L: 100, Cout: 24, K: 3, stride: 2, pad: 1, dil: 1, groups: 1 },
]) {
  const { Cin, L, Cout, K, stride, pad, dil, groups } = cfg;
  const X = rand(Cin * L), W = rand(Cout * (Cin / groups) * K), bias = rand(Cout);
  const ref = convRefCPU(X, W, bias, Cin, L, Cout, K, stride, pad, dil, groups);
  report(cfg.name, maxErr(dl(ctx.conv1d(ctx.upload(X, Cin, L), ctx.upload(W, 1, W.length), { cout: Cout, k: K, bias: ctx.upload(bias, 1, Cout), stride, pad, dilation: dil, groups })), ref), 1e-3);
}
// layernorm
{
  const R = 64, C = 512, X = rand(R * C), g = rand(C), b = rand(C), eps = 1e-5, ref = new Float32Array(R * C);
  for (let r = 0; r < R; r++) { let m = 0; for (let j = 0; j < C; j++) m += X[r * C + j]; m /= C; let v = 0; for (let j = 0; j < C; j++) v += (X[r * C + j] - m) ** 2; v /= C; const inv = 1 / Math.sqrt(v + eps); for (let j = 0; j < C; j++) ref[r * C + j] = (X[r * C + j] - m) * inv * g[j] + b[j]; }
  report("layernorm", maxErr(dl(ctx.layernorm(ctx.upload(X, R, C), ctx.upload(g, 1, C), ctx.upload(b, 1, C), eps)), ref), 1e-4);
}
// softmax
{
  const R = 128, C = 200, X = rand(R * C), ref = new Float32Array(R * C);
  for (let r = 0; r < R; r++) { let mx = -Infinity; for (let j = 0; j < C; j++) mx = Math.max(mx, X[r * C + j]); let s = 0; for (let j = 0; j < C; j++) { const e = Math.exp(X[r * C + j] - mx); ref[r * C + j] = e; s += e; } for (let j = 0; j < C; j++) ref[r * C + j] /= s; }
  report("softmax", maxErr(dl(ctx.softmax(ctx.upload(X, R, C))), ref), 1e-6);
}
// add / mul (broadcast)
{
  const R = 40, C = 512, A = rand(R * C), bc = rand(C), refA = new Float32Array(R * C), refM = new Float32Array(R * C);
  for (let i = 0; i < R * C; i++) { refA[i] = A[i] + bc[i % C]; refM[i] = A[i] * bc[i % C]; }
  report("add (broadcast)", maxErr(dl(ctx.add(ctx.upload(A, R, C), ctx.upload(bc, 1, C))), refA), 0);
  report("mul (broadcast)", maxErr(dl(ctx.mul(ctx.upload(A, R, C), ctx.upload(bc, 1, C))), refM), 0);
}
// matmulInt8
{
  const M = 40, K = 256, N = 128, A = rand(M * K), W = rand(K * N);
  const q = new Int8Array(K * N), s = new Float32Array(N);
  for (let n = 0; n < N; n++) { let mx = 0; for (let k = 0; k < K; k++) mx = Math.max(mx, Math.abs(W[k * N + n])); const sc = mx / 127 || 1; s[n] = sc; for (let k = 0; k < K; k++) q[k * N + n] = Math.max(-127, Math.min(127, Math.round(W[k * N + n] / sc))); }
  const ref = new Float32Array(M * N);
  for (let i = 0; i < M; i++) for (let n = 0; n < N; n++) { let acc = 0; for (let k = 0; k < K; k++) acc += A[i * K + k] * q[k * N + n]; ref[i * N + n] = acc * s[n]; }
  report("matmulInt8", maxErr(dl(ctx.matmulInt8(ctx.upload(A, M, K), ctx.uploadBytes(new Uint8Array(q.buffer)), ctx.upload(s, 1, N), N, K)), ref), 1e-3);
}
// matmulNBits (int4, K%32≠0)
{
  const M = 8, N = 64, blk = 32, K = 100, nblk = Math.ceil(K / blk), zpb = Math.ceil(nblk / 2);
  const Bq = new Uint8Array(N * nblk * 16); for (let i = 0; i < Bq.length; i++) Bq[i] = Math.floor(Math.random() * 256);
  const scales = rand(N * nblk).map((v) => Math.abs(v) * 0.1 + 0.02);
  const zpU8 = new Uint8Array(N * zpb); for (let i = 0; i < zpU8.length; i++) zpU8[i] = Math.floor(Math.random() * 256);
  const A = rand(M * K).map((v) => v * 0.1);
  const q = (n, b, jj) => (Bq[(n * nblk + b) * 16 + (jj >> 1)] >> (4 * (jj & 1))) & 0xf;
  const zpAt = (n, b) => (zpU8[n * zpb + (b >> 1)] >> (4 * (b & 1))) & 0xf;
  const ref = new Float32Array(M * N);
  for (let mi = 0; mi < M; mi++) for (let n = 0; n < N; n++) { let acc = 0; for (let b = 0; b < nblk; b++) { const s = scales[n * nblk + b], z = zpAt(n, b); for (let jj = 0; jj < 32; jj++) { const k = b * 32 + jj; if (k >= K) break; acc += A[mi * K + k] * ((q(n, b, jj) - z) * s); } } ref[mi * N + n] = acc; }
  report("matmulNBits (int4)", maxErr(dl(ctx.matmulNBits(ctx.upload(A, M, K), ctx.uploadBytes(Bq), ctx.upload(scales, 1, scales.length), ctx.uploadBytes(zpU8), N)), ref), 1e-3);
}
// conv1dGemm + conv1dFast match direct conv
{
  const Cin = 32, L = 500, Cout = 48, K = 5, pad = 2, X = rand(Cin * L), W = rand(Cout * Cin * K), bias = rand(Cout);
  const ref = convRefCPU(X, W, bias, Cin, L, Cout, K, 1, pad, 1, 1);
  const g = dl(ctx.conv1dGemm(ctx.upload(X, Cin, L), ctx.upload(W, Cout, Cin * K), Cout, K, { pad }));
  for (let co = 0; co < Cout; co++) for (let lo = 0; lo < L; lo++) g[co * L + lo] += bias[co];
  report("conv1dGemm (im2col)", maxErr(g, ref), 1e-3);
  report("conv1dFast (fused)", maxErr(dl(ctx.conv1dFast(ctx.upload(X, Cin, L), ctx.upload(W, Cout, Cin * K), Cout, K, { pad, bias: ctx.upload(bias, 1, Cout) })), ref), 1e-3);
}
// gatherCols
{
  const C = 16, T = 8, X = rand(C * T), idx = Uint32Array.from([0, 0, 0, 1, 2, 2, 3, 4, 4, 4, 4, 7]);
  const out = dl(ctx.gatherCols(ctx.upload(X, C, T), idx));
  let e = 0; for (let r = 0; r < C; r++) for (let f = 0; f < idx.length; f++) e = Math.max(e, Math.abs(out[r * idx.length + f] - X[r * T + idx[f]]));
  report("gatherCols", e, 0);
}
// convTranspose1d (regular + depthwise)
function convTRefCPU(X, W, bias, Cin, L, Cout, K, stride, pad, dil, groups, outPad) {
  const Lout = (L - 1) * stride - 2 * pad + dil * (K - 1) + outPad + 1, Y = new Float32Array(Cout * Lout);
  const cinG = Cin / groups, coutG = Cout / groups;
  for (let co = 0; co < Cout; co++) { const g = Math.floor(co / coutG), coInG = co - g * coutG;
    for (let lo = 0; lo < Lout; lo++) { let acc = bias ? bias[co] : 0;
      for (let ci = 0; ci < cinG; ci++) { const rc = g * cinG + ci;
        for (let k = 0; k < K; k++) { const num = lo + pad - k * dil; if (num >= 0 && num % stride === 0) { const li = num / stride; if (li >= 0 && li < L) acc += X[rc * L + li] * W[rc * (coutG * K) + coInG * K + k]; } } }
      Y[co * Lout + lo] = acc; } }
  return Y;
}
for (const cfg of [
  { name: "convT stride10 (ups)", Cin: 16, L: 20, Cout: 8, K: 20, stride: 10, pad: 5, groups: 1, outPad: 0 },
  { name: "convT depthwise s2", Cin: 32, L: 40, Cout: 32, K: 3, stride: 2, pad: 1, groups: 32, outPad: 1 },
]) {
  const { Cin, L, Cout, K, stride, pad, groups, outPad } = cfg;
  const X = rand(Cin * L), W = rand(Cin * (Cout / groups) * K), bias = rand(Cout);
  const ref = convTRefCPU(X, W, bias, Cin, L, Cout, K, stride, pad, 1, groups, outPad);
  report(cfg.name, maxErr(dl(ctx.convTranspose1d(ctx.upload(X, Cin, L), ctx.upload(W, 1, W.length), { cout: Cout, k: K, bias: ctx.upload(bias, 1, Cout), stride, pad, groups, outputPadding: outPad })), ref), 1e-3);
}
// bidirectional LSTM
{
  const seq = 12, inp = 20, hid = 16, sig = (x) => 1 / (1 + Math.exp(-x));
  const X = rand(seq * inp), W = rand(2 * 4 * hid * inp), R = rand(2 * 4 * hid * hid), B = rand(2 * 8 * hid), ref = new Float32Array(seq * 2 * hid);
  for (let dir = 0; dir < 2; dir++) { const wB = dir * 4 * hid * inp, rB = dir * 4 * hid * hid, bB = dir * 8 * hid, h = new Float32Array(hid), c = new Float32Array(hid);
    for (let s = 0; s < seq; s++) { const t = dir === 1 ? seq - 1 - s : s, hn = new Float32Array(hid);
      for (let u = 0; u < hid; u++) { const gate = (gi) => { let acc = B[bB + gi * hid + u] + B[bB + 4 * hid + gi * hid + u]; for (let k = 0; k < inp; k++) acc += W[wB + (gi * hid + u) * inp + k] * X[t * inp + k]; for (let k = 0; k < hid; k++) acc += R[rB + (gi * hid + u) * hid + k] * h[k]; return acc; };
        const it = sig(gate(0)), ot = sig(gate(1)), ft = sig(gate(2)), ct = Math.tanh(gate(3)), cn = ft * c[u] + it * ct; c[u] = cn; hn[u] = ot * Math.tanh(cn); ref[(t * 2 + dir) * hid + u] = hn[u]; }
      h.set(hn); } }
  report("lstm (bidir, iofc)", maxErr(dl(ctx.lstm(ctx.upload(X, seq, inp), ctx.upload(W, 1, W.length), ctx.upload(R, 1, R.length), ctx.upload(B, 1, B.length), hid)), ref), 1e-4);
}
// adain
{
  const C = 40, L = 300, eps = 1e-5, X = rand(C * L), sc = rand(C), sh = rand(C), ref = new Float32Array(C * L);
  for (let ch = 0; ch < C; ch++) { let m = 0; for (let j = 0; j < L; j++) m += X[ch * L + j]; m /= L; let v = 0; for (let j = 0; j < L; j++) v += (X[ch * L + j] - m) ** 2; v /= L; const inv = 1 / Math.sqrt(v + eps); for (let j = 0; j < L; j++) ref[ch * L + j] = (X[ch * L + j] - m) * inv * sc[ch] + sh[ch]; }
  report("adain", maxErr(dl(ctx.adain(ctx.upload(X, C, L), ctx.upload(sc, 1, C), ctx.upload(sh, 1, C), eps)), ref), 1e-4);
}
// leakyRelu / glu / relShift
{
  const X = rand(4096), slope = 0.2, ref = X.map((v) => (v > 0 ? v : slope * v));
  report("leakyRelu", maxErr(dl(ctx.leakyRelu(ctx.upload(X, 1, 4096), slope)), ref), 0);
}
{
  const C = 32, T = 20, X = rand(2 * C * T), sig = (x) => 1 / (1 + Math.exp(-x)), ref = new Float32Array(C * T);
  for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) ref[c * T + t] = X[c * T + t] * sig(X[(c + C) * T + t]);
  report("glu (channels)", maxErr(dl(ctx.glu(ctx.upload(X, 2 * C, T))), ref), 1e-6);
}
{
  const t = 12, p = 2 * t - 1, X = rand(t * p), xp = new Float32Array(t * 2 * t);
  for (let i = 0; i < t; i++) for (let c = 0; c < p; c++) xp[i * 2 * t + 1 + c] = X[i * p + c];
  const ref = new Float32Array(t * t);
  for (let i = 0; i < t; i++) for (let j = 0; j < t; j++) ref[i * t + j] = xp[t + i * p + j];
  report("relShift", maxErr(dl(ctx.relShift(ctx.upload(X, t, p))), ref), 0);
}
// transpose / sliceCols / setCols
{
  const R = 24, C = 768, W = 64, off = 128, X = rand(R * C);
  const tOut = dl(ctx.transpose(ctx.upload(X, R, C)));
  let tErr = 0; for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) tErr = Math.max(tErr, Math.abs(tOut[c * R + r] - X[r * C + c]));
  report("transpose", tErr, 0);
  const sl = dl(ctx.sliceCols(ctx.upload(X, R, C), off, W));
  let sErr = 0; for (let r = 0; r < R; r++) for (let j = 0; j < W; j++) sErr = Math.max(sErr, Math.abs(sl[r * W + j] - X[r * C + off + j]));
  report("sliceCols", sErr, 0);
  const dst = ctx.upload(new Float32Array(R * C), R, C), src = ctx.upload(rand(R * W), R, W);
  ctx.setCols(dst, src, off);
  let dErr = 0; for (let r = 0; r < R; r++) for (let j = 0; j < W; j++) dErr = Math.max(dErr, Math.abs(dst.data[r * C + off + j] - src.data[r * W + j]));
  report("setCols", dErr, 0);
}
// conv2d (regular + depthwise)
function conv2dRefCPU(X, W, bias, Cin, H, Wd, Cout, Kh, Kw, sH, sW, padH, padW, groups, relu) {
  const Ho = Math.floor((H + 2 * padH - Kh) / sH) + 1, Wo = Math.floor((Wd + 2 * padW - Kw) / sW) + 1, Y = new Float32Array(Cout * Ho * Wo), cinG = Cin / groups, coutG = Cout / groups;
  for (let co = 0; co < Cout; co++) { const g = Math.floor(co / coutG);
    for (let ho = 0; ho < Ho; ho++) for (let wo = 0; wo < Wo; wo++) { let acc = bias ? bias[co] : 0;
      for (let ci = 0; ci < cinG; ci++) { const rc = g * cinG + ci;
        for (let kh = 0; kh < Kh; kh++) { const hi = ho * sH + kh - padH; if (hi < 0 || hi >= H) continue; for (let kw = 0; kw < Kw; kw++) { const wi = wo * sW + kw - padW; if (wi < 0 || wi >= Wd) continue; acc += X[rc * H * Wd + hi * Wd + wi] * W[((co * cinG + ci) * Kh + kh) * Kw + kw]; } } }
      Y[co * Ho * Wo + ho * Wo + wo] = relu ? Math.max(acc, 0) : acc; } }
  return Y;
}
{
  const Cin = 1, H = 20, Wd = 16, Cout = 8, K = 3, X = rand(Cin * H * Wd), W = rand(Cout * Cin * K * K), b = rand(Cout);
  const ref = conv2dRefCPU(X, W, b, Cin, H, Wd, Cout, K, K, 2, 2, 1, 1, 1, true);
  report("conv2d (regular)", maxErr(dl(ctx.conv2d(ctx.upload(X, Cin, H * Wd), ctx.upload(W, 1, W.length), { cout: Cout, cin: Cin, h: H, w: Wd, kh: K, kw: K, bias: ctx.upload(b, 1, Cout), strideH: 2, strideW: 2, padH: 1, padW: 1, groups: 1, act: "relu" })), ref), 1e-3);
}
{
  const C = 16, H = 10, Wd = 8, K = 3, X = rand(C * H * Wd), W = rand(C * K * K), b = rand(C);
  const ref = conv2dRefCPU(X, W, b, C, H, Wd, C, K, K, 2, 2, 1, 1, C, false);
  report("conv2d (depthwise)", maxErr(dl(ctx.conv2d(ctx.upload(X, C, H * Wd), ctx.upload(W, 1, W.length), { cout: C, cin: C, h: H, w: Wd, kh: K, kw: K, bias: ctx.upload(b, 1, C), strideH: 2, strideW: 2, padH: 1, padW: 1, groups: C })), ref), 1e-3);
}
// subReshape / jbatch / argmaxRows / jointArgmax
{
  const C = 8, Tsub = 5, F = 4, X = rand(C * Tsub * F), ref = new Float32Array(Tsub * C * F);
  for (let idx = 0; idx < Tsub * C * F; idx++) { const CF = C * F, ho = Math.floor(idx / CF), rem = idx % CF, c = Math.floor(rem / F), wo = rem % F; ref[idx] = X[c * (Tsub * F) + ho * F + wo]; }
  report("subReshape", maxErr(dl(ctx.subReshape(ctx.upload(X, C, Tsub * F), C, Tsub, F)), ref), 0);
}
{
  const Tenc = 10, hid = 32, B = 4, base = 2, encProj = rand(Tenc * hid), predProj = rand(hid), ref = new Float32Array(B * hid);
  for (let i = 0; i < B; i++) for (let k = 0; k < hid; k++) ref[i * hid + k] = Math.max(0, encProj[(base + i) * hid + k] + predProj[k]);
  report("jbatch", maxErr(dl(ctx.jbatch(ctx.upload(encProj, Tenc, hid), base, B, ctx.upload(predProj, 1, hid), hid)), ref), 1e-6);
}
{
  const B = 6, vocab = 100, logits = 105, X = rand(B * logits), ref = new Float32Array(B * 4);
  for (let r = 0; r < B; r++) { let bt = 0, bv = -1e30, bd = 0, bdv = -1e30; for (let n = 0; n < logits; n++) { const s = X[r * logits + n]; if (n < vocab) { if (s > bv) { bv = s; bt = n; } } else if (s > bdv) { bdv = s; bd = n - vocab; } } ref[r * 4] = bt; ref[r * 4 + 1] = bv; ref[r * 4 + 2] = bd; ref[r * 4 + 3] = bdv; }
  report("argmaxRows", maxErr(dl(ctx.argmaxRows(ctx.upload(X, B, logits), B, vocab, logits)), ref), 0);
}
{
  const Tenc = 5, hidden = 24, vocab = 30, logits = 33, count = 3, frame = 1;
  const encProj = rand(Tenc * hidden), predProj = rand(hidden), outW = rand(hidden * logits), outB = rand(logits), ref = new Float32Array(count * 4);
  for (let w = 0; w < count; w++) { const base = (frame + w) * hidden, j = new Float32Array(hidden); for (let k = 0; k < hidden; k++) j[k] = Math.max(0, encProj[base + k] + predProj[k]); let bt = 0, bv = -1e30, bd = 0, bdv = -1e30; for (let n = 0; n < logits; n++) { let s = outB[n]; for (let k = 0; k < hidden; k++) s += j[k] * outW[k * logits + n]; if (n < vocab) { if (s > bv) { bv = s; bt = n; } } else if (s > bdv) { bdv = s; bd = n - vocab; } } ref[w * 4] = bt; ref[w * 4 + 1] = bv; ref[w * 4 + 2] = bd; ref[w * 4 + 3] = bdv; }
  report("jointArgmax", maxErr(dl(ctx.jointArgmax(ctx.upload(encProj, Tenc, hidden), frame, count, ctx.upload(predProj, 1, hidden), ctx.upload(outW, hidden, logits), ctx.upload(outB, 1, logits), hidden, vocab, logits)), ref), 1e-3);
}

console.log(fails === 0 ? "\nALL WASM KERNELS PARITY OK" : `\n${fails} WASM KERNEL(S) FAILED`);
process.exit(fails === 0 ? 0 : 1);
