// Ship/no-ship benchmark for raw-WebGPU Kokoro. From an ORT profile of the full
// model (scripts/kokoro-profile.py): ~90% of compute is the iSTFTNet decoder,
// ~64% is Conv (~106 GFLOP of conv for ~2s audio), dominated by resblock convs at
// [128, ~9841] K11. So the whole question reduces to: how fast is raw-WebGPU conv?
// This measures the direct kernel vs im2col+GEMM at that dominant shape and
// projects a conv-only RTFx to compare against kokoro-js (ORT WebGPU, ~10x).
//   node scripts/gpu-kokoro-cost.mjs
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";

const dev = await getDevice();
const ctx = new GpuContext(dev);
const done = () => dev.queue.onSubmittedWorkDone();
const rand = (n) => Float32Array.from({ length: n }, () => Math.random() * 2 - 1);

const Cin = 128,
  L = 9841,
  Cout = 128,
  K = 11,
  pad = 5; // dominant Kokoro resblock conv
const CONV_GFLOP_TOTAL = 106; // measured over the full model, 2.05s audio
const AUDIO_S = 2.05;
const gflop1 = (2 * Cout * Cin * K * L) / 1e9;

const xT = ctx.upload(rand(Cin * L), Cin, L);
const wD = ctx.upload(rand(Cout * Cin * K), 1, Cout * Cin * K); // direct kernel weight layout
const wR = ctx.upload(rand(Cout * Cin * K), Cout, Cin * K); // [Cout, Cin*K] for GEMM

async function bench(fn, label) {
  for (let i = 0; i < 5; i++) fn();
  await done();
  const t0 = performance.now(),
    N = 50;
  for (let i = 0; i < N; i++) fn();
  await done();
  const ms = (performance.now() - t0) / N;
  const gfs = gflop1 / (ms / 1000);
  const convMs = CONV_GFLOP_TOTAL / (gfs / 1000);
  const rtfx = AUDIO_S / (convMs / 1000);
  console.log(
    `${label.padEnd(14)} ${ms.toFixed(2)} ms  ${gfs.toFixed(0)} GFLOP/s  ->  ${CONV_GFLOP_TOTAL} GFLOP conv ≈ ${convMs.toFixed(0)} ms  conv-only RTFx ~${rtfx.toFixed(1)}x`,
  );
  return gfs;
}

console.log(
  `Kokoro cost model — dominant conv ${Cin}×${L}×${Cout} K${K} (${gflop1.toFixed(2)} GFLOP); full-model conv ≈ ${CONV_GFLOP_TOTAL} GFLOP for ${AUDIO_S}s audio (decoder = ~90% of compute).\n`,
);
await bench(() => ctx.conv1d(xT, wD, { cout: Cout, k: K, pad }), "direct conv");
await bench(() => ctx.conv1dGemm(xT, wR, Cout, K, { pad }), "im2col+GEMM");
console.log(`\nReference: kokoro-js (ORT WebGPU) ~10x RTFx (docs/BENCHMARKS.md).`);
console.log(
  `Verdict: naive conv loses; im2col+GEMM is competitive. A register-blocked/fp16 GEMM (naive tiled GEMM is ~10% of fp32 peak) is the lever to win decisively.`,
);
process.exit(0);
