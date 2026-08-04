// Parity proof for the raw-WebGPU Parakeet encoder (raw-encoder.js) vs ORT.
//   1) node scripts/pk-encoder-ref.mjs   (writes /tmp/pk-raw/ref_final.json)
//   2) python3 scripts/extract-parakeet-encoder-weights.py /tmp/pkv3/encoder-model.onnx /tmp/pk-raw/enc
//   3) node scripts/smoke-parakeet-encoder-raw.mjs
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadParakeetEncoder, parakeetEncode } from "../src/engines/asr-parakeet/raw-encoder.js";
import { readFileSync, openSync, readSync, fstatSync, closeSync } from "node:fs";
function readBig(path) { // readFileSync caps at 2GB
  const fd = openSync(path, "r"), size = fstatSync(fd).size, u8 = new Uint8Array(size);
  let off = 0; const CH = 1 << 30;
  while (off < size) { const n = readSync(fd, u8, off, Math.min(CH, size - off), off); if (n <= 0) break; off += n; }
  closeSync(fd); return new Float32Array(u8.buffer, 0, size / 4);
}
const dir = process.argv[2] || "/tmp/pk-raw";
const ctx = new GpuContext(await getDevice());
const man = JSON.parse(readFileSync(`${dir}/enc/manifest.json`, "utf8"));
const enc = loadParakeetEncoder(ctx, readBig(`${dir}/enc/weights.bin`), man);
const ref = JSON.parse(readFileSync(`${dir}/ref_final.json`, "utf8"));
const { data } = await parakeetEncode(ctx, enc, Float32Array.from(ref.mel), ref.T);
const refO = Float32Array.from(ref.out);
let md = 0; for (let i = 0; i < data.length; i++) md = Math.max(md, Math.abs(data[i] - refO[i]));
console.log(`raw Parakeet encoder maxΔ vs ORT = ${md.toExponential(2)}`);
console.log(md < 1e-3 ? "PARITY OK" : "PARITY FAIL");
process.exit(md < 1e-3 ? 0 : 1);
