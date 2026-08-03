// Verify every WGSL kernel in src/gpu/compute.js against a CPU reference, on a
// real GPU (dawn). This is the parity gate for the raw-WebGPU path — the same
// kernels run in the browser via navigator.gpu.
//   node scripts/gpu-verify.mjs
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";

const dev = await getDevice();
const ctx = new GpuContext(dev);
const rand = (n) => Float32Array.from({ length: n }, () => Math.random() * 2 - 1);
const maxErr = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
let fails = 0;
function report(name, err, tol) {
  const ok = err <= tol;
  if (!ok) fails++;
  console.log(`${ok ? "✓" : "✗"} ${name.padEnd(28)} max|gpu-cpu| = ${err.toExponential(2)}  (tol ${tol.toExponential(0)})`);
}

// ---- matmul (+ fused bias/gelu) ----
{
  const M = 80, K = 512, N = 512;
  const A = rand(M * K), B = rand(K * N), bias = rand(N);
  const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)));
  const ref = new Float32Array(M * N);
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) { let s = bias[j]; for (let k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j]; ref[i * N + j] = gelu(s); }
  const ta = ctx.upload(A, M, K), tb = ctx.upload(B, K, N), tbias = ctx.upload(bias, 1, N);
  const out = await ctx.download(ctx.matmul(ta, tb, { bias: tbias, act: "gelu" }));
  report("matmul+bias+gelu", maxErr(out, ref), 1e-2);
}

// ---- conv1d (regular, padded, dilated, and depthwise) ----
function convRefCPU(X, W, bias, Cin, L, Cout, K, stride, pad, dil, groups) {
  const Lout = Math.floor((L + 2 * pad - dil * (K - 1) - 1) / stride) + 1;
  const Y = new Float32Array(Cout * Lout);
  const cinG = Cin / groups, coutG = Cout / groups;
  for (let co = 0; co < Cout; co++) {
    const g = Math.floor(co / coutG);
    for (let lo = 0; lo < Lout; lo++) {
      let acc = bias ? bias[co] : 0;
      for (let ci = 0; ci < cinG; ci++) {
        const realCi = g * cinG + ci;
        for (let k = 0; k < K; k++) {
          const li = lo * stride + k * dil - pad;
          if (li >= 0 && li < L) acc += X[realCi * L + li] * W[(co * cinG + ci) * K + k];
        }
      }
      Y[co * Lout + lo] = acc;
    }
  }
  return Y;
}
for (const cfg of [
  { name: "conv1d k3 pad1", Cin: 64, L: 200, Cout: 128, K: 3, stride: 1, pad: 1, dil: 1, groups: 1 },
  { name: "conv1d k5 dil2", Cin: 32, L: 128, Cout: 32, K: 5, stride: 1, pad: 4, dil: 2, groups: 1 },
  { name: "conv1d depthwise", Cin: 80, L: 100, Cout: 80, K: 7, stride: 1, pad: 3, dil: 1, groups: 80 },
]) {
  const { Cin, L, Cout, K, stride, pad, dil, groups } = cfg;
  const X = rand(Cin * L), W = rand(Cout * (Cin / groups) * K), bias = rand(Cout);
  const ref = convRefCPU(X, W, bias, Cin, L, Cout, K, stride, pad, dil, groups);
  const tx = ctx.upload(X, Cin, L), tw = ctx.upload(W, 1, W.length), tb = ctx.upload(bias, 1, Cout);
  const out = await ctx.download(ctx.conv1d(tx, tw, { cout: Cout, k: K, bias: tb, stride, pad, dilation: dil, groups }));
  report(cfg.name, maxErr(out, ref), 1e-2);
}

// ---- layernorm ----
{
  const R = 64, C = 512;
  const X = rand(R * C), g = rand(C), b = rand(C), eps = 1e-5;
  const ref = new Float32Array(R * C);
  for (let r = 0; r < R; r++) {
    let mean = 0; for (let j = 0; j < C; j++) mean += X[r * C + j]; mean /= C;
    let v = 0; for (let j = 0; j < C; j++) v += (X[r * C + j] - mean) ** 2; v /= C;
    const inv = 1 / Math.sqrt(v + eps);
    for (let j = 0; j < C; j++) ref[r * C + j] = (X[r * C + j] - mean) * inv * g[j] + b[j];
  }
  const out = await ctx.download(ctx.layernorm(ctx.upload(X, R, C), ctx.upload(g, 1, C), ctx.upload(b, 1, C), eps));
  report("layernorm", maxErr(out, ref), 1e-3);
}

// ---- softmax ----
{
  const R = 128, C = 200;
  const X = rand(R * C);
  const ref = new Float32Array(R * C);
  for (let r = 0; r < R; r++) {
    let mx = -Infinity; for (let j = 0; j < C; j++) mx = Math.max(mx, X[r * C + j]);
    let s = 0; for (let j = 0; j < C; j++) { const e = Math.exp(X[r * C + j] - mx); ref[r * C + j] = e; s += e; }
    for (let j = 0; j < C; j++) ref[r * C + j] /= s;
  }
  const out = await ctx.download(ctx.softmax(ctx.upload(X, R, C)));
  report("softmax", maxErr(out, ref), 1e-5);
}

// ---- elementwise add (broadcast bias) + mul ----
{
  const R = 40, C = 512;
  const A = rand(R * C), bcast = rand(C);
  const refAdd = new Float32Array(R * C), refMul = new Float32Array(R * C);
  for (let i = 0; i < R * C; i++) { refAdd[i] = A[i] + bcast[i % C]; refMul[i] = A[i] * bcast[i % C]; }
  const ta = ctx.upload(A, R, C), tb = ctx.upload(bcast, 1, C);
  report("add (broadcast)", maxErr(await ctx.download(ctx.add(ta, tb)), refAdd), 1e-5);
  report("mul (broadcast)", maxErr(await ctx.download(ctx.mul(ta, tb)), refMul), 1e-5);
}

// ---- f16-storage matmul + fused conv (looser tol: f16 ~3 decimal digits) ----
{
  const M = 128, K = 512, N = 256;
  const A = rand(M * K), B = rand(K * N), bias = rand(N);
  const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)));
  const ref = new Float32Array(M * N);
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) { let s = bias[j]; for (let k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j]; ref[i * N + j] = gelu(s); }
  const out = await ctx.downloadF16(ctx.matmulF16(ctx.uploadF16(A, M, K), ctx.uploadF16(B, K, N), { bias: ctx.uploadF16(bias, 1, N), act: "gelu" }));
  const rel = (() => { let se = 0, sr = 0; for (let i = 0; i < ref.length; i++) { se += (out[i] - ref[i]) ** 2; sr += ref[i] ** 2; } return Math.sqrt(se / sr); })();
  report("matmulF16 (rel)", rel, 5e-3);
}
{
  const Cin = 48, L = 400, Cout = 64, K = 7, pad = 3;
  const X = rand(Cin * L), W = rand(Cout * Cin * K), bias = rand(Cout);
  const ref = convRefCPU(X, W, bias, Cin, L, Cout, K, 1, pad, 1, 1);
  const out = await ctx.downloadF16(ctx.conv1dFastF16(ctx.uploadF16(X, Cin, L), ctx.uploadF16(W, Cout, Cin * K), Cout, K, { pad, bias: ctx.uploadF16(bias, 1, Cout) }));
  const rel = (() => { let se = 0, sr = 0; for (let i = 0; i < ref.length; i++) { se += (out[i] - ref[i]) ** 2; sr += ref[i] ** 2; } return Math.sqrt(se / sr); })();
  report("conv1dFastF16 (rel)", rel, 1e-2);
}

// ---- conv1d via im2col + GEMM (matches the direct conv) ----
{
  const Cin = 32, L = 500, Cout = 48, K = 5, pad = 2;
  const X = rand(Cin * L), W = rand(Cout * Cin * K), bias = rand(Cout);
  const ref = convRefCPU(X, W, bias, Cin, L, Cout, K, 1, pad, 1, 1);
  // conv1dGemm has no bias fold; add bias per-row on CPU for the check.
  const g = await ctx.download(ctx.conv1dGemm(ctx.upload(X, Cin, L), ctx.upload(W, Cout, Cin * K), Cout, K, { pad }));
  const Lout = L; // stride1 pad2 K5
  for (let co = 0; co < Cout; co++) for (let lo = 0; lo < Lout; lo++) g[co * Lout + lo] += bias[co];
  report("conv1dGemm (im2col)", maxErr(g, ref), 1e-2);
}

// ---- conv1dFast (fused implicit GEMM, groups=1) matches direct conv ----
{
  const Cin = 48, L = 400, Cout = 64, K = 7, pad = 3;
  const X = rand(Cin * L), W = rand(Cout * Cin * K), bias = rand(Cout);
  const ref = convRefCPU(X, W, bias, Cin, L, Cout, K, 1, pad, 1, 1);
  const out = await ctx.download(ctx.conv1dFast(ctx.upload(X, Cin, L), ctx.upload(W, Cout, Cin * K), Cout, K, { pad, bias: ctx.upload(bias, 1, Cout) }));
  report("conv1dFast (fused)", maxErr(out, ref), 1e-2);
}

// ---- gatherCols (length regulator) ----
{
  const C = 16, T = 8;
  const X = rand(C * T);
  const idx = Uint32Array.from([0, 0, 0, 1, 2, 2, 3, 4, 4, 4, 4, 7]);
  const out = await ctx.download(ctx.gatherCols(ctx.upload(X, C, T), idx));
  let e = 0; for (let r = 0; r < C; r++) for (let f = 0; f < idx.length; f++) e = Math.max(e, Math.abs(out[r * idx.length + f] - X[r * T + idx[f]]));
  report("gatherCols (len-reg)", e, 0);
}

// ---- convTranspose1d (regular + depthwise) ----
function convTRefCPU(X, W, bias, Cin, L, Cout, K, stride, pad, dil, groups, outPad) {
  const Lout = (L - 1) * stride - 2 * pad + dil * (K - 1) + outPad + 1;
  const Y = new Float32Array(Cout * Lout);
  const cinG = Cin / groups, coutG = Cout / groups;
  for (let co = 0; co < Cout; co++) {
    const g = Math.floor(co / coutG), coInG = co - g * coutG;
    for (let lo = 0; lo < Lout; lo++) {
      let acc = bias ? bias[co] : 0;
      for (let ci = 0; ci < cinG; ci++) {
        const realCi = g * cinG + ci;
        for (let k = 0; k < K; k++) {
          const num = lo + pad - k * dil;
          if (num >= 0 && num % stride === 0) {
            const li = num / stride;
            if (li >= 0 && li < L) acc += X[realCi * L + li] * W[realCi * (coutG * K) + coInG * K + k];
          }
        }
      }
      Y[co * Lout + lo] = acc;
    }
  }
  return Y;
}
for (const cfg of [
  { name: "convT stride10 (ups)", Cin: 16, L: 20, Cout: 8, K: 20, stride: 10, pad: 5, groups: 1, outPad: 0 },
  { name: "convT depthwise s2", Cin: 32, L: 40, Cout: 32, K: 3, stride: 2, pad: 1, groups: 32, outPad: 1 },
]) {
  const { Cin, L, Cout, K, stride, pad, groups, outPad } = cfg;
  const X = rand(Cin * L), W = rand(Cin * (Cout / groups) * K), bias = rand(Cout);
  const ref = convTRefCPU(X, W, bias, Cin, L, Cout, K, stride, pad, 1, groups, outPad);
  const out = await ctx.download(ctx.convTranspose1d(ctx.upload(X, Cin, L), ctx.upload(W, 1, W.length),
    { cout: Cout, k: K, bias: ctx.upload(bias, 1, Cout), stride, pad, groups, outputPadding: outPad }));
  report(cfg.name, maxErr(out, ref), 1e-2);
}

// ---- bidirectional LSTM (ONNX iofc semantics) ----
{
  const seq = 12, inp = 20, hid = 16;
  const sig = (x) => 1 / (1 + Math.exp(-x));
  const X = rand(seq * inp), W = rand(2 * 4 * hid * inp), R = rand(2 * 4 * hid * hid), B = rand(2 * 8 * hid);
  // CPU reference, gate order iofc, output [seq, 2*hid] = [fwd|bwd].
  const ref = new Float32Array(seq * 2 * hid);
  for (let dir = 0; dir < 2; dir++) {
    const wB = dir * 4 * hid * inp, rB = dir * 4 * hid * hid, bB = dir * 8 * hid;
    const h = new Float32Array(hid), c = new Float32Array(hid);
    for (let s = 0; s < seq; s++) {
      const t = dir === 1 ? seq - 1 - s : s;
      const hn = new Float32Array(hid);
      for (let u = 0; u < hid; u++) {
        const gate = (gi) => {
          let acc = B[bB + gi * hid + u] + B[bB + 4 * hid + gi * hid + u];
          for (let k = 0; k < inp; k++) acc += W[wB + (gi * hid + u) * inp + k] * X[t * inp + k];
          for (let k = 0; k < hid; k++) acc += R[rB + (gi * hid + u) * hid + k] * h[k];
          return acc;
        };
        const it = sig(gate(0)), ot = sig(gate(1)), ft = sig(gate(2)), ct = Math.tanh(gate(3));
        const cn = ft * c[u] + it * ct;
        c[u] = cn; hn[u] = ot * Math.tanh(cn);
        ref[(t * 2 + dir) * hid + u] = hn[u];
      }
      h.set(hn);
    }
  }
  const out = await ctx.download(ctx.lstm(ctx.upload(X, seq, inp), ctx.upload(W, 1, W.length), ctx.upload(R, 1, R.length), ctx.upload(B, 1, B.length), hid));
  report("lstm (bidir, iofc)", maxErr(out, ref), 1e-3);
}

// ---- AdaIN (instance-norm over time + per-channel affine) ----
{
  const C = 40, L = 300, eps = 1e-5;
  const X = rand(C * L), sc = rand(C), sh = rand(C);
  const ref = new Float32Array(C * L);
  for (let ch = 0; ch < C; ch++) {
    let mean = 0; for (let j = 0; j < L; j++) mean += X[ch * L + j]; mean /= L;
    let v = 0; for (let j = 0; j < L; j++) v += (X[ch * L + j] - mean) ** 2; v /= L;
    const inv = 1 / Math.sqrt(v + eps);
    for (let j = 0; j < L; j++) ref[ch * L + j] = (X[ch * L + j] - mean) * inv * sc[ch] + sh[ch];
  }
  const out = await ctx.download(ctx.adain(ctx.upload(X, C, L), ctx.upload(sc, 1, C), ctx.upload(sh, 1, C), eps));
  report("adain", maxErr(out, ref), 1e-3);
}

// ---- leaky relu ----
{
  const X = rand(4096), slope = 0.2;
  const ref = X.map((v) => (v > 0 ? v : slope * v));
  const out = await ctx.download(ctx.leakyRelu(ctx.upload(X, 1, 4096), slope));
  report("leakyRelu", maxErr(out, ref), 1e-6);
}

// ---- transpose / sliceCols / setCols (attention plumbing) ----
{
  const R = 24, C = 768, W = 64, off = 128;
  const X = rand(R * C);
  const tOut = await ctx.download(ctx.transpose(ctx.upload(X, R, C)));
  let tErr = 0; for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) tErr = Math.max(tErr, Math.abs(tOut[c * R + r] - X[r * C + c]));
  report("transpose", tErr, 0);

  const sl = await ctx.download(ctx.sliceCols(ctx.upload(X, R, C), off, W));
  let sErr = 0; for (let r = 0; r < R; r++) for (let j = 0; j < W; j++) sErr = Math.max(sErr, Math.abs(sl[r * W + j] - X[r * C + off + j]));
  report("sliceCols", sErr, 0);

  const dst = ctx.upload(new Float32Array(R * C), R, C);
  const src = ctx.upload(rand(R * W), R, W);
  ctx.setCols(dst, src, off);
  const dOut = await ctx.download(dst), sIn = await ctx.download(src);
  let dErr = 0; for (let r = 0; r < R; r++) for (let j = 0; j < W; j++) dErr = Math.max(dErr, Math.abs(dOut[r * C + off + j] - sIn[r * W + j]));
  report("setCols", dErr, 0);
}

console.log(fails === 0 ? "\nALL KERNELS PARITY OK" : `\n${fails} KERNEL(S) FAILED`);
process.exit(fails === 0 ? 0 : 1);
