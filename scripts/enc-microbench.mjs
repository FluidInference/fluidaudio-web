// Microbench encoder op classes at real window shapes (T=188, D=1024, H=8, HD=64).
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
const dev = await getDevice();
const ctx = new GpuContext(dev);
const rand = (n) => Float32Array.from({ length: n }, () => Math.random() * 0.1 - 0.05);
const done = () => dev.queue.onSubmittedWorkDone();
const T = 188,
  D = 1024,
  DFF = 4096,
  H = 8,
  HD = 64,
  P = 2 * T - 1;
async function t(name, fn, reps) {
  for (let i = 0; i < 3; i++) fn();
  await done();
  const t0 = performance.now();
  ctx.beginBatch();
  for (let i = 0; i < reps; i++) fn();
  ctx.endBatch();
  await done();
  const ms = (performance.now() - t0) / reps;
  console.log(`${name.padEnd(34)} ${ms.toFixed(3)} ms (batched)`);
  return ms;
}
// int8 FF GEMM  [188,1024]@[1024,4096]
const a = ctx.upload(rand(T * D), T, D);
const q8 = new Int8Array(D * DFF);
for (let i = 0; i < q8.length; i++) q8[i] = (Math.random() * 200 - 100) | 0;
const wq = ctx.uploadBytes(new Uint8Array(q8.buffer));
const sc = ctx.upload(rand(DFF), 1, DFF);
const mFF = await t("int8 GEMM 188x1024x4096 (v2)", () => ctx.matmulInt8(a, wq, sc, DFF, D), 50);
// fp32 GEMM same shape (for reference)
const bf = ctx.upload(rand(D * DFF), D, DFF);
await t("fp32 GEMM 188x1024x4096", () => ctx.matmul(a, bf), 50);
// int8 GEMM  [188,4096]@[4096,1024]
const a2 = ctx.upload(rand(T * DFF), T, DFF);
const q82 = new Int8Array(DFF * D);
for (let i = 0; i < q82.length; i++) q82[i] = (Math.random() * 200 - 100) | 0;
const wq2 = ctx.uploadBytes(new Uint8Array(q82.buffer));
const sc2 = ctx.upload(rand(D), 1, D);
const mFF2 = await t("int8 GEMM 188x4096x1024 (v2)", () => ctx.matmulInt8(a2, wq2, sc2, D, DFF), 50);
// attention batched
const qq = ctx.upload(rand(T * D), T, D),
  kk = ctx.upload(rand(T * D), T, D),
  vv = ctx.upload(rand(T * D), T, D);
const pp = ctx.upload(rand(P * D), P, D),
  qb = ctx.upload(rand(D), 1, D);
const mQK = await t("bmmQK  [H*T,T]", () => ctx.bmmQK(qq, kk, qb, H, HD), 50);
const mBD = await t("bmmQK  [H*T,2T-1]", () => ctx.bmmQK(qq, pp, qb, H, HD), 50);
const probs = ctx.softmax(ctx.bmmQK(qq, kk, qb, H, HD));
const mPV = await t("bmmPV", () => ctx.bmmPV(probs, vv, H, HD), 50);
const mSM = await t("softmax [H*T,T]", () => ctx.softmax(ctx.bmmQK(qq, kk, qb, H, HD)), 30);
// depthwise conv module
const hT = ctx.upload(rand(D * T), D, T);
const dw = ctx.upload(rand(D * 9), 1, D * 9);
const mDW = await t("depthwise conv1d k9", () => ctx.conv1d(hT, dw, { cout: D, k: 9, groups: D, pad: 4 }), 50);
// per-layer + per-window estimate
const layer = 4 * ((mFF + mFF2) / 2) * 2 + mQK + mBD + mPV + 1.5 + mDW; // rough
console.log(`\n≈FF GEMMs/layer ${(2 * (mFF + mFF2)).toFixed(2)}ms  attn ${(mQK + mBD + mPV).toFixed(2)}ms`);
console.log(`≈24-layer window: FF ${(24 * 2 * (mFF + mFF2)).toFixed(0)}ms`);
process.exit(0);
