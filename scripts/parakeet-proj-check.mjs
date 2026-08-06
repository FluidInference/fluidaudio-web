// Gate: GPU-projected decode path (decode_proj) must produce IDENTICAL tokens to
// the classic path on the 120s file, and report timing deltas.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadParakeetEncoder, parakeetEncodeBatch } from "../src/engines/asr-parakeet/raw-encoder.js";
import { loadWasmDecoder, wasmDecode, wasmDecodeProj } from "../src/engines/asr-parakeet/raw-decoder-wasm.js";
import { ParakeetMel } from "../src/engines/asr-parakeet/parakeet-mel.js";
function readWav(p) {
  const b = readFileSync(p);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let o = 12,
    dO = -1,
    dL = 0;
  while (o + 8 <= b.length) {
    const id = String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);
    const sz = dv.getUint32(o + 4, true);
    if (id === "data") {
      dO = o + 8;
      dL = sz;
      break;
    }
    o += 8 + sz + (sz & 1);
  }
  const n = dL / 2,
    out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(dO + i * 2, true) / 32768;
  return out;
}

const wav = readWav("/tmp/pk_120s.wav");
const ctx = new GpuContext(await getDevice());
const rdU8 = (p) => Uint8Array.from(readFileSync(p));
const enc = loadParakeetEncoder(ctx, rdU8("/tmp/pk-raw/enc-int8/weights.bin"), JSON.parse(readFileSync("/tmp/pk-raw/enc-int8/manifest.json")));
const decMan = JSON.parse(readFileSync("/tmp/pk-raw/dec/manifest.json"));
const db = rdU8("/tmp/pk-raw/dec/weights.bin");
const decF32 = new Float32Array(db.buffer, db.byteOffset, db.byteLength / 4);
const dec = await loadWasmDecoder(readFileSync("src/engines/asr-parakeet/parakeet-decoder.wasm"), decF32, decMan);
const g = (k) => decF32.subarray(decMan[k].offset, decMan[k].offset + decMan[k].len);
const encW = ctx.upload(g("encW").slice(), 1024, 640),
  encB = ctx.upload(g("encB").slice(), 1, 640);
const mel = new ParakeetMel(128);
const WIN = 15 * 16000,
  hop = 13 * 16000;
let same = 0,
  diff = 0,
  tOld = 0,
  tNew = 0,
  dlOld = 0,
  dlNew = 0;
for (let s = 0; s + WIN <= wav.length; s += hop) {
  const { features } = mel.process(wav.subarray(s, s + WIN));
  const r = await parakeetEncodeBatch(ctx, enc, [features]);
  let t = performance.now();
  const framesFull = await ctx.download(r.framesGpu);
  dlOld += performance.now() - t;
  t = performance.now();
  const a = wasmDecode(dec, framesFull, r.Tsub).ids;
  tOld += performance.now() - t;
  const proj = ctx.matmul(r.framesGpu, encW, { bias: encB });
  t = performance.now();
  const projCpu = await ctx.download(proj);
  dlNew += performance.now() - t;
  t = performance.now();
  const b = wasmDecodeProj(dec, projCpu, r.Tsub).ids;
  tNew += performance.now() - t;
  if (a.length === b.length && a.every((v, i) => v === b[i])) same++;
  else diff++;
}
console.log(`windows identical: ${same}, differing: ${diff}`);
console.log(`decode  old ${(tOld / same).toFixed(1)}ms/win → proj ${(tNew / same).toFixed(1)}ms/win`);
console.log(`readback old ${(dlOld / same).toFixed(1)}ms/win → proj ${(dlNew / same).toFixed(1)}ms/win`);
process.exit(diff ? 1 : 0);
