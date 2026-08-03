// Parity: raw-WebGPU ConvTranspose1d vs the Kokoro ONNX iSTFTNet upsampler (ups.0).
// Needs /tmp/kokoro/convt/{W,B,X,Y}.bin (kokoro-ref-convt.py).
//   node scripts/gpu-convt.mjs
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";

const D = "/tmp/kokoro/convt";
const r = JSON.parse(readFileSync(`${D}/ref.json`, "utf8"));
const f32 = (n) => new Float32Array(Uint8Array.from(readFileSync(`${D}/${n}.bin`)).buffer);

const dev = await getDevice();
const ctx = new GpuContext(dev);
const X = ctx.upload(f32("X"), r.Cin, r.L); // [1,Cin,L] flat == [Cin,L]
const W = ctx.upload(f32("W"), 1, r.Cin * (r.Cout / r.groups) * r.K);
const B = r.hasBias ? ctx.upload(f32("B"), 1, r.Cout) : null;
const out = await ctx.download(ctx.convTranspose1d(X, W, {
  cout: r.Cout, k: r.K, bias: B, stride: r.stride, pad: r.pad, groups: r.groups, outputPadding: r.outputPadding,
}));
const refY = f32("Y");

let mx = 0, se = 0, sr = 0;
for (let i = 0; i < out.length; i++) { const d = Math.abs(out[i] - refY[i]); mx = Math.max(mx, d); se += d * d; sr += refY[i] * refY[i]; }
const rel = Math.sqrt(se / sr);
console.log(`ConvTranspose1d (Cin ${r.Cin}→Cout ${r.Cout}, L ${r.L}→${r.Lout}, K ${r.K}, stride ${r.stride}): max ${mx.toExponential(2)}  rel ${rel.toExponential(2)}`);
const ok = rel < 2e-2;
console.log(ok ? "CONVTRANSPOSE PARITY OK (raw WebGPU == ONNX)" : "PARITY FAIL");
process.exit(ok ? 0 : 1);
