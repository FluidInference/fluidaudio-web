// Measured raw-WebGPU forward cost for Kokoro: replays every compute op
// (Conv/ConvTranspose/MatMul/Gemm/LSTM) at its REAL shape from /tmp/kokoro/ops.json
// (kokoro-ops.py), back-to-back on the GPU with a single sync. Not a hand-wired
// pipeline — a faithful op-for-op cost of the compute that dominates (~76% of ORT
// time; the rest is cheap elementwise/Sin/STFT). Answers "how fast is a raw-WebGPU
// Kokoro forward?" on real hardware.
//   node scripts/gpu-kokoro-forward.mjs
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";

const { audio_s, ort_cpu_total_ms, ops } = JSON.parse(readFileSync("/tmp/kokoro/ops.json", "utf8"));
const dev = await getDevice();
const ctx = new GpuContext(dev);
const done = () => dev.queue.onSubmittedWorkDone();

// Zero-init every allocated buffer so compute runs on clean values (uninitialized
// GPU memory can hold garbage → denormals/NaN that run slow and pollute timing).
const ZEROS = new Float32Array(1 << 22); // 4M
const origAlloc = ctx.alloc.bind(ctx);
ctx.alloc = (r, c) => {
  const t = origAlloc(r, c);
  let off = 0; const n = r * c;
  while (off < n) { const len = Math.min(ZEROS.length, n - off); dev.queue.writeBuffer(t.buf, off * 4, ZEROS, 0, len); off += len; }
  return t;
};
const last2 = (a) => a.slice(-2);
const prod = (a) => a.reduce((x, y) => x * (y || 1), 1);

function runOp(o) {
  try {
    if (o.op === "Conv") {
      const [, Cin, L] = o.in[0]; const w = o.in[1]; const Cout = o.out[0][1]; const K = w[2];
      const groups = Math.max(1, Math.round(Cin / w[1]));
      const x = ctx.alloc(Cin, L);
      if (groups === 1) return ctx.conv1dFast(x, ctx.alloc(Cout, Cin * K), Cout, K, { pad: (K - 1) >> 1 });
      return ctx.conv1d(x, ctx.alloc(1, Cout * (Cin / groups) * K), { cout: Cout, k: K, pad: (K - 1) >> 1, groups });
    }
    if (o.op === "ConvTranspose") {
      const [, Cin, L] = o.in[0]; const Cout = o.out[0][1]; const Lout = o.out[0][2]; const K = o.in[1][2];
      const stride = Math.max(1, Math.round(Lout / L));
      return ctx.convTranspose1d(ctx.alloc(Cin, L), ctx.alloc(1, Cin * Cout * K), { cout: Cout, k: K, stride });
    }
    if (o.op === "MatMul" || o.op === "Gemm") {
      const a = o.in[0]; const out = o.out[0] || [];
      if (!a || a.length < 2) return null;
      const [M0, K] = last2(a); const M = prod(a.slice(0, -2)) * M0;
      const N = out.length ? out[out.length - 1] : (o.in[1] ? o.in[1][o.in[1].length - 1] : 0);
      if (!M || !K || !N) return null;
      return ctx.matmul(ctx.alloc(M, K), ctx.alloc(K, N));
    }
    if (o.op === "LSTM") {
      const [seq, , inp] = o.in[0]; const H = Math.round(o.in[1][1] / 4);
      return ctx.lstm(ctx.alloc(seq, inp), ctx.alloc(1, 2 * 4 * H * inp), ctx.alloc(1, 2 * 4 * H * H), ctx.alloc(1, 2 * 8 * H), H);
    }
  } catch { return null; }
  return null;
}

// warm the pipelines
for (const o of ops.slice(0, 30)) runOp(o);
await done();

// all ops in ONE submit. Buffer alloc + dispatch recording happen in the (untimed)
// loop; the recorded dispatches only EXECUTE at submit — so timing submit→GPU-finish
// isolates pure GPU compute, excluding CPU-side alloc/record overhead.
let ran = 0;
ctx.beginBatch();
for (const o of ops) { if (runOp(o)) ran++; }
await done(); // flush the zero-init writes before timing the dispatches
const t0 = performance.now();
ctx.endBatch();
await done();
const ms = performance.now() - t0;

const byOrt = {}; // o.ms is seconds (dur/1e6)
for (const o of ops) byOrt[o.op] = (byOrt[o.op] || 0) + o.ms * 1000;
const ortComputeMs = Object.values(byOrt).reduce((a, b) => a + b, 0);
console.log(`replayed ${ran}/${ops.length} compute ops (single submit)`);
console.log(`raw-WebGPU compute forward: ${ms.toFixed(1)} ms  ->  RTFx ~${(audio_s / (ms / 1000)).toFixed(1)}x  (audio ${audio_s.toFixed(2)}s)`);
console.log(`ORT CPU, same ops:          ${ortComputeMs.toFixed(0)} ms  ->  RTFx ~${(audio_s / (ortComputeMs / 1000)).toFixed(1)}x`);
console.log("by op (ORT CPU ms):", Object.fromEntries(Object.entries(byOrt).map(([k, v]) => [k, +v.toFixed(0)])));
console.log(`Reference: kokoro-js (ORT WebGPU) ~10x RTFx.`);
console.log(`(compute ops = ~76% of ORT time; remaining Sin/STFT/elementwise are cheap on GPU.)`);
process.exit(0);
