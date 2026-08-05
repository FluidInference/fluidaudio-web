// Gate + timing for the shipping windowed pipeline (src/engines/asr-parakeet/
// pipeline.js): serial vs pipelined must be TOKEN-IDENTICAL (same kernels, only
// scheduling differs), and the pipelined wall should approach max(enc, mel+dec).
//   node scripts/parakeet-pipeline-check.mjs [/tmp/pk_120s.wav]
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadParakeetEncoder } from "../src/engines/asr-parakeet/raw-encoder.js";
import { loadWasmDecoder } from "../src/engines/asr-parakeet/raw-decoder-wasm.js";
import { ParakeetMel } from "../src/engines/asr-parakeet/parakeet-mel.js";
import { transcribeWindowed } from "../src/engines/asr-parakeet/pipeline.js";

function readWav(p) {
  const b = readFileSync(p); const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let o = 12, dO = -1, dL = 0;
  while (o + 8 <= b.length) { const id = String.fromCharCode(b[o], b[o+1], b[o+2], b[o+3]); const s = dv.getUint32(o+4, true); if (id === "data") { dO = o+8; dL = s; break; } o += 8 + s + (s & 1); }
  const n = dL/2, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(dO + i*2, true) / 32768;
  return out;
}

const wav = readWav(process.argv[2] || "/tmp/pk_120s.wav");
const durS = wav.length / 16000;
console.log(`audio ${durS.toFixed(1)}s`);
const ctx = new GpuContext(await getDevice());
const rdU8 = (p) => Uint8Array.from(readFileSync(p));
const enc = loadParakeetEncoder(ctx, rdU8("/tmp/pk-raw/enc-int8/weights.bin"), JSON.parse(readFileSync("/tmp/pk-raw/enc-int8/manifest.json")));
const decMan = JSON.parse(readFileSync("/tmp/pk-raw/dec/manifest.json"));
const decU8 = rdU8("/tmp/pk-raw/dec/weights.bin");
const decF32 = new Float32Array(decU8.buffer, decU8.byteOffset, decU8.byteLength / 4);
const dec = await loadWasmDecoder(readFileSync("src/engines/asr-parakeet/parakeet-decoder.wasm"), decF32, decMan);
const g = (k) => decF32.subarray(decMan[k].offset, decMan[k].offset + decMan[k].len);
const projW = ctx.upload(g("encW").slice(), 1024, 640);
const projB = ctx.upload(g("encB").slice(), 1, 640);
const mel = new ParakeetMel(128);

const run = async (pipelined) => {
  const t = performance.now();
  const ids = await transcribeWindowed(ctx, enc, dec, mel, projW, projB, wav, { pipelined, wb: Number(process.env.WB || 6) });
  return { ids, ms: performance.now() - t };
};

await run(false); // warm (shader compile, wasm decoder init)
const serial = await run(false);
const piped = await run(true);
const piped2 = await run(true);

const same = serial.ids.length === piped.ids.length && serial.ids.every((v, i) => v === piped.ids[i])
  && piped2.ids.length === piped.ids.length && piped2.ids.every((v, i) => v === piped.ids[i]);
console.log(`tokens: serial ${serial.ids.length}, pipelined ${piped.ids.length} → ${same ? "IDENTICAL" : "DIVERGED"}`);
const best = Math.min(piped.ms, piped2.ms);
console.log(`serial    ${serial.ms.toFixed(0)}ms  RTFx ${(durS / (serial.ms / 1000)).toFixed(1)}`);
console.log(`pipelined ${best.toFixed(0)}ms  RTFx ${(durS / (best / 1000)).toFixed(1)}  (${(serial.ms / best).toFixed(2)}× vs serial)`);
if (!same) process.exit(1);
process.exit(0);
