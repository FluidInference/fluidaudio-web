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

console.log(fails === 0 ? "\nALL KERNELS PARITY OK" : `\n${fails} KERNEL(S) FAILED`);
process.exit(fails === 0 ? 0 : 1);
