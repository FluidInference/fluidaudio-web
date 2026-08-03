// Characterize the raw-WebGPU path on the real GPU:
//   1. GEMM throughput at Kokoro-relevant shapes.
//   2. A GPU-RESIDENT transformer FFN block (the ALBERT-encoder core) — every
//      intermediate stays on the GPU, one submit chain, one readback. This is the
//      fusion+residency win over onnxruntime-web's per-op dispatch + CPU syncs.
//   node scripts/gpu-bench.mjs
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";

const dev = await getDevice();
const ctx = new GpuContext(dev);
const rand = (n) => Float32Array.from({ length: n }, () => (Math.random() * 2 - 1) * 0.05);
const done = () => dev.queue.onSubmittedWorkDone();

async function timeGemm(M, K, N, iters) {
  const a = ctx.upload(rand(M * K), M, K), b = ctx.upload(rand(K * N), K, N);
  for (let w = 0; w < 5; w++) ctx.matmul(a, b);
  await done();
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) ctx.matmul(a, b);
  await done();
  const ms = (performance.now() - t0) / iters;
  console.log(`  GEMM ${M}x${K}x${N}: ${ms.toFixed(3)} ms  ${((2 * M * N * K) / (ms / 1000) / 1e9).toFixed(0)} GFLOP/s`);
}

console.log("GEMM throughput:");
await timeGemm(512, 512, 512, 200);
await timeGemm(512, 512, 2048, 200);
await timeGemm(200, 512, 512, 200);

// GPU-resident transformer FFN block: y = layernorm(x + W2·gelu(W1·x + b1) + b2)
// dims: seq tokens × d_model, FFN hidden d_ff. Nothing leaves the GPU until the
// final download.
const SEQ = 200, D = 512, DFF = 2048;
const x = ctx.upload(rand(SEQ * D), SEQ, D);
const W1 = ctx.upload(rand(D * DFF), D, DFF), b1 = ctx.upload(rand(DFF), 1, DFF);
const W2 = ctx.upload(rand(DFF * D), DFF, D), b2 = ctx.upload(rand(D), 1, D);
const g = ctx.upload(new Float32Array(D).fill(1), 1, D), be = ctx.upload(new Float32Array(D).fill(0), 1, D);

function ffnBlock() {
  const h = ctx.matmul(x, W1, { bias: b1, act: "gelu" }); // [SEQ, DFF]  fused matmul+bias+gelu
  const o = ctx.matmul(h, W2, { bias: b2 });              // [SEQ, D]
  const res = ctx.add(o, x);                              // residual
  return ctx.layernorm(res, g, be);                       // [SEQ, D]
}

for (let w = 0; w < 5; w++) ffnBlock();
await done();
const iters = 200;
const t0 = performance.now();
let last;
for (let i = 0; i < iters; i++) last = ffnBlock();
await done();
const ms = (performance.now() - t0) / iters;
await ctx.download(last); // single readback
console.log(`\nGPU-resident FFN block (${SEQ}×${D}, d_ff ${DFF}): ${ms.toFixed(3)} ms/block`);
console.log(`  4 kernels/block, intermediates never leave the GPU — 1 readback total.`);
console.log(`  A 6-layer ALBERT encoder ≈ ${(ms * 6).toFixed(2)} ms of FFN compute.`);
process.exit(0);
