import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
const dev = await getDevice();
const ctx = new GpuContext(dev);
const rand = (n) => Float32Array.from({ length: n }, () => (Math.random() * 2 - 1) * 0.05);
const done = () => dev.queue.onSubmittedWorkDone();
async function t(fn, M, K, N, iters = 100) {
  const a = ctx.upload(rand(M * K), M, K),
    b = ctx.upload(rand(K * N), K, N);
  for (let w = 0; w < 8; w++) fn(a, b);
  await done();
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn(a, b);
  await done();
  const ms = (performance.now() - t0) / iters;
  return (2 * M * K * N) / (ms / 1e3) / 1e12;
}
// real speech-model shapes (M×K×N)
const shapes = [
  [200, 512, 512],
  [200, 512, 2048],
  [1500, 512, 512],
  [1500, 512, 2048],
  [512, 1024, 1024],
  [512, 1024, 4096],
];
console.log("shape              v1     v3     v4   TFLOP/s");
for (const [M, K, N] of shapes) {
  const v1 = await t((a, b) => ctx.matmul(a, b), M, K, N);
  const v3 = K % 4 === 0 && N % 4 === 0 ? await t((a, b) => ctx.matmulV3(a, b), M, K, N) : NaN;
  const v4 = K % 8 === 0 && N % 4 === 0 ? await t((a, b) => ctx.matmulV4(a, b), M, K, N) : NaN;
  console.log(`${M}x${K}x${N}`.padEnd(18) + `${v1.toFixed(2)}  ${v3.toFixed(2)}  ${v4.toFixed(2)}`);
}
process.exit(0);
