// Unpipelined per-stage profile of the raw Parakeet path on a long wav (dawn).
//   node scripts/parakeet-profile.mjs [/tmp/pk_120s.wav]
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadParakeetEncoder, parakeetEncode } from "../src/engines/asr-parakeet/raw-encoder.js";
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
let tMel = 0, tEnc = 0, tDec = 0, nWin = 0, nTok = 0;
const now = () => performance.now();
for (let s = 0; s + 16000 < wav.length; s += hop) {
  const slice = wav.subarray(s, Math.min(s+WIN, wav.length));
  let t = now();
  const { features, length } = mel.process(slice);
  tMel += now()-t;
  t = now();
  const r = await parakeetEncode(ctx, enc, features, length);
  const frames = await ctx.download(r.framesGpu);
  tEnc += now()-t;
  t = now();
  const { ids } = wasmDecode(dec, frames, r.Tsub);
  tDec += now()-t;
  nTok += ids.length; nWin++;
}
const tot = tMel+tEnc+tDec;
console.log(`${nWin} windows, ${nTok} tokens`);
console.log(`mel   ${tMel.toFixed(0)}ms (${(100*tMel/tot).toFixed(0)}%)  ${(tMel/nWin).toFixed(0)}ms/win`);
console.log(`enc   ${tEnc.toFixed(0)}ms (${(100*tEnc/tot).toFixed(0)}%)  ${(tEnc/nWin).toFixed(0)}ms/win`);
console.log(`dec   ${tDec.toFixed(0)}ms (${(100*tDec/tot).toFixed(0)}%)  ${(tDec/nWin).toFixed(0)}ms/win`);
console.log(`serial total ${tot.toFixed(0)}ms → RTFx ${(wav.length/16000/(tot/1000)).toFixed(1)} (pipelined ≈ max(enc, mel+dec))`);
process.exit(0);
