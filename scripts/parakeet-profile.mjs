// Unpipelined per-stage profile of the raw Parakeet path on a long wav (dawn).
//   node scripts/parakeet-profile.mjs [/tmp/pk_120s.wav]
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadParakeetEncoder, parakeetEncode, parakeetEncodeBatch } from "../src/engines/asr-parakeet/raw-encoder.js";
import { loadWasmDecoder, wasmDecode } from "../src/engines/asr-parakeet/raw-decoder-wasm.js";
import { ParakeetMel } from "../src/engines/asr-parakeet/parakeet-mel.js";

function readWav(p) {
  const b = readFileSync(p); const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let o = 12, dO = -1, dL = 0;
  while (o + 8 <= b.length) { const id = String.fromCharCode(b[o], b[o+1], b[o+2], b[o+3]); const s = dv.getUint32(o+4, true); if (id === "data") { dO = o+8; dL = s; break; } o += 8 + s + (s & 1); }
  const n = dL/2, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(dO + i*2, true) / 32768;
  return out;
}
const wav = readWav(process.argv[2] || "/tmp/pk_120s.wav");
console.log(`audio ${(wav.length/16000).toFixed(1)}s`);
const ctx = new GpuContext(await getDevice());
const rdU8 = (p) => Uint8Array.from(readFileSync(p));
const encMan = JSON.parse(readFileSync("/tmp/pk-raw/enc-int8/manifest.json"));
const encBin = rdU8("/tmp/pk-raw/enc-int8/weights.bin");
const enc = loadParakeetEncoder(ctx, encBin, encMan);
const decMan = JSON.parse(readFileSync("/tmp/pk-raw/dec/manifest.json"));
const decBinU8 = rdU8("/tmp/pk-raw/dec/weights.bin");
const dec = await loadWasmDecoder(readFileSync("src/engines/asr-parakeet/parakeet-decoder.wasm"), new Float32Array(decBinU8.buffer, decBinU8.byteOffset, decBinU8.byteLength/4), decMan);
const mel = new ParakeetMel(128);
const WIN = 15*16000, OVL = 2*16000, hop = WIN-OVL;
const WB = Number(process.env.WB || 4); // windows per encoder batch
let tMel = 0, tEnc = 0, tDec = 0, tRec = 0, tGpu = 0, nWin = 0, nTok = 0;
const now = () => performance.now();
// mel for all full windows
const melWins = [];
for (let s = 0; s + WIN <= wav.length; s += hop) {
  let t = now();
  const { features } = mel.process(wav.subarray(s, s + WIN));
  tMel += now()-t;
  melWins.push(features);
}
// W=4 vs W=1 equivalence gate on the first 2 windows
{
  const single = [];
  for (const f of melWins.slice(0, 2)) {
    const r = await parakeetEncode(ctx, enc, f, 0);
    single.push(await ctx.download(r.framesGpu));
  }
  const rb = await parakeetEncodeBatch(ctx, enc, melWins.slice(0, 2));
  const both = await ctx.download(rb.framesGpu);
  let md = 0;
  for (let i = 0; i < single[0].length; i++) md = Math.max(md, Math.abs(both[i] - single[0][i]), Math.abs(both[single[0].length + i] - single[1][i]));
  console.log(`batch-vs-single equivalence maxΔ ${md.toExponential(2)} ${md < 1e-4 ? "OK" : "FAIL"}`);
}
for (let b = 0; b < melWins.length; b += WB) {
  const group = melWins.slice(b, b + WB);
  let t = now();
  const r = await parakeetEncodeBatch(ctx, enc, group);
  tRec += now()-t; t = now();
  await ctx.device.queue.onSubmittedWorkDone();
  tGpu += now()-t; t = now();
  const frames = await ctx.download(r.framesGpu);
  tEnc += now()-t;
  t = now();
  for (let wi = 0; wi < group.length; wi++) {
    const { ids } = wasmDecode(dec, frames.subarray(wi * r.Tsub * r.D, (wi + 1) * r.Tsub * r.D), r.Tsub);
    nTok += ids.length; nWin++;
  }
  tDec += now()-t;
}
const tot = tMel+tEnc+tDec+tRec+tGpu;
console.log(`${nWin} windows, ${nTok} tokens`);
console.log(`mel   ${tMel.toFixed(0)}ms (${(100*tMel/tot).toFixed(0)}%)  ${(tMel/nWin).toFixed(0)}ms/win`);
console.log(`enc-record ${tRec.toFixed(0)}ms  ${(tRec/nWin).toFixed(0)}ms/win (CPU dispatch recording)`);
console.log(`enc-gpu    ${tGpu.toFixed(0)}ms  ${(tGpu/nWin).toFixed(0)}ms/win (GPU exec wait)`);
console.log(`enc-dl     ${tEnc.toFixed(0)}ms  ${(tEnc/nWin).toFixed(0)}ms/win (readback)`);
console.log(`dec   ${tDec.toFixed(0)}ms (${(100*tDec/tot).toFixed(0)}%)  ${(tDec/nWin).toFixed(0)}ms/win`);
console.log(`serial total ${tot.toFixed(0)}ms → RTFx ${(wav.length/16000/(tot/1000)).toFixed(1)} (pipelined ≈ max(enc, mel+dec))`);
process.exit(0);
