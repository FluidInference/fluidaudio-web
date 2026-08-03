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
