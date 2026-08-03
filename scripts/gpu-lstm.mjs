// Parity: raw-WebGPU bidirectional LSTM vs the Kokoro ONNX predictor LSTM.
// Needs /tmp/kokoro/lstm/{W,R,B,X,Y}.bin (kokoro-ref-lstm.py).
//   node scripts/gpu-lstm.mjs
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";

const D = "/tmp/kokoro/lstm";
const ref = JSON.parse(readFileSync(`${D}/ref.json`, "utf8"));
const f32 = (n) => new Float32Array(Uint8Array.from(readFileSync(`${D}/${n}.bin`)).buffer);
const { seq, inp, hid } = ref;

const dev = await getDevice();
const ctx = new GpuContext(dev);
const X = ctx.upload(f32("X"), seq, inp); // [seq,1,inp] flat == [seq,inp]
const W = ctx.upload(f32("W"), 1, 2 * 4 * hid * inp);
const R = ctx.upload(f32("R"), 1, 2 * 4 * hid * hid);
const B = ctx.upload(f32("B"), 1, 2 * 8 * hid);

const out = await ctx.download(ctx.lstm(X, W, R, B, hid)); // [seq, 2*hid]
const refY = f32("Y"); // [seq,2,1,hid] flat == [seq, 2*hid]

let mx = 0, se = 0, sr = 0;
for (let i = 0; i < out.length; i++) { const d = Math.abs(out[i] - refY[i]); mx = Math.max(mx, d); se += d * d; sr += refY[i] * refY[i]; }
const rel = Math.sqrt(se / sr);
console.log(`LSTM (bidir, seq ${seq}, inp ${inp}, hid ${hid}): max ${mx.toExponential(2)}  rel ${rel.toExponential(2)}`);
const ok = rel < 2e-2;
console.log(ok ? "LSTM PARITY OK (raw WebGPU == ONNX)" : "PARITY FAIL");
process.exit(ok ? 0 : 1);
