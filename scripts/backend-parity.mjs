// End-to-end backend parity: run the REAL raw Parakeet encoder through BOTH the
// WebGPU backend (GpuContext/dawn) and the WASM+SIMD backend (WasmContext) on the
// same weights + input, and compare to each other and to the ORT reference. This is
// the capstone gate for "the engines work on webgpu AND wasm, identically".
//   node scripts/backend-parity.mjs [/tmp/pk-raw]
import { readFileSync, openSync, readSync, fstatSync, closeSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { loadParakeetEncoder, parakeetEncode } from "../src/engines/asr-parakeet/raw-encoder.js";

function readBig(path) {
  const fd = openSync(path, "r"), size = fstatSync(fd).size, u8 = new Uint8Array(size);
  let off = 0; const CH = 1 << 30;
  while (off < size) { const n = readSync(fd, u8, off, Math.min(CH, size - off), off); if (n <= 0) break; off += n; }
  closeSync(fd); return u8;
}
const maxErr = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

const dir = process.argv[2] || "/tmp/pk-raw";
const encDir = `${dir}/enc-int8`;
const man = JSON.parse(readFileSync(`${encDir}/manifest.json`, "utf8"));
const weightsU8 = readBig(`${encDir}/weights.bin`);
const isInt8 = Object.values(man).some((m) => m && m.int8);
const weights = isInt8 ? weightsU8 : new Float32Array(weightsU8.buffer, 0, weightsU8.byteLength / 4);
const ref = JSON.parse(readFileSync(`${dir}/ref_final.json`, "utf8"));
const refO = Float32Array.from(ref.out);

// ── WASM backend ──
const wasmBytes = readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url)));
const wctx = await createWasmContext(wasmBytes);
const wenc = loadParakeetEncoder(wctx, weights, man);
let t0 = Date.now();
const wout = (await parakeetEncode(wctx, wenc, Float32Array.from(ref.mel), ref.T, true)).data;
const wms = Date.now() - t0;

// ── WebGPU backend (dawn) ──
const gctx = new GpuContext(await getDevice());
const genc = loadParakeetEncoder(gctx, weights, man);
t0 = Date.now();
const gout = (await parakeetEncode(gctx, genc, Float32Array.from(ref.mel), ref.T, true)).data;
const gms = Date.now() - t0;

const crossErr = maxErr(wout, gout);
const wVsRef = maxErr(wout, refO);
const gVsRef = maxErr(gout, refO);
console.log(`encoder: ${isInt8 ? "int8" : "fp32"}, T=${ref.T}, out ${wout.length}`);
console.log(`  WebGPU: ${gms} ms   WASM: ${wms} ms`);
console.log(`  WebGPU vs WASM  maxΔ = ${crossErr.toExponential(2)}`);
console.log(`  WASM  vs ORTref maxΔ = ${wVsRef.toExponential(2)}`);
console.log(`  WebGPU vs ORTref maxΔ = ${gVsRef.toExponential(2)}`);
// int8 encoder differs from the fp32 ORT ref by quantization; the cross-backend
// number is the real gate (both run identical int8 weights → must agree tightly).
const ok = crossErr < 2e-3;
console.log(ok ? "\nBACKEND PARITY OK (WebGPU == WASM)" : "\nBACKEND PARITY FAIL");
process.exit(ok ? 0 : 1);
