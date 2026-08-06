import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
const dev = await getDevice();
const ctx = new GpuContext(dev);
const rand = (n) => Float32Array.from({ length: n }, () => (Math.random() * 2 - 1) * 0.5);
function cpuMM(a, b, M, K, N, bias, act) {
  const o = new Float32Array(M * N);
  for (let i = 0; i < M; i++)
    for (let k = 0; k < K; k++) {
      const av = a[i * K + k];
      for (let j = 0; j < N; j++) o[i * N + j] += av * b[k * N + j];
    }
  const g = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
  for (let i = 0; i < M; i++)
    for (let j = 0; j < N; j++) {
      let x = o[i * N + j] + (bias ? bias[j] : 0);
      o[i * N + j] = act === "gelu" ? g(x) : act === "relu" ? Math.max(0, x) : x;
    }
  return o;
}
let bad = 0;
for (const [M, K, N] of [
  [300, 64, 300],
  [512, 256, 512],
  [136, 64, 132],
]) {
  for (const act of ["none", "gelu", "relu"]) {
    const A = rand(M * K),
      B = rand(K * N),
      bs = rand(N);
    const ga = ctx.upload(A, M, K),
      gb = ctx.upload(B, K, N),
      gbias = ctx.upload(bs, 1, N);
    const gp = await ctx.download(ctx.matmulV4(ga, gb, { bias: gbias, act }));
    const cpu = cpuMM(A, B, M, K, N, bs, act);
    let e = 0;
    for (let i = 0; i < M * N; i++) e = Math.max(e, Math.abs(gp[i] - cpu[i]));
    const ok = e < 2e-4;
    if (!ok) bad++;
    console.log(`v4 ${M}x${K}x${N} act=${act}: maxΔ ${e.toExponential(2)} ${ok ? "OK" : "FAIL"}`);
    // dispatch check: matmul() should route large aligned → v4, match
    if (M >= 256 && N >= 256 && K >= 256 && K % 8 === 0 && N % 4 === 0) {
      const gd = await ctx.download(ctx.matmul(ga, gb, { bias: gbias, act }));
      let ed = 0;
      for (let i = 0; i < M * N; i++) ed = Math.max(ed, Math.abs(gd[i] - cpu[i]));
      console.log(`   matmul()dispatch: maxΔ ${ed.toExponential(2)} ${ed < 2e-4 ? "OK" : "FAIL"}`);
      if (ed >= 2e-4) bad++;
    }
  }
}
console.log(bad ? `${bad} FAILURES` : "ALL PARITY OK");
process.exit(bad ? 1 : 0);
