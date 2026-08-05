// ids equality: int8 joint vs fp32 joint on the 120s file (all windows).
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadParakeetEncoder, parakeetEncodeBatch } from "../src/engines/asr-parakeet/raw-encoder.js";
import { loadWasmDecoder, wasmDecode } from "../src/engines/asr-parakeet/raw-decoder-wasm.js";
import { ParakeetMel } from "../src/engines/asr-parakeet/parakeet-mel.js";
function readWav(p){const b=readFileSync(p);const dv=new DataView(b.buffer,b.byteOffset,b.byteLength);let o=12,dO=-1,dL=0;while(o+8<=b.length){const id=String.fromCharCode(b[o],b[o+1],b[o+2],b[o+3]);const s=dv.getUint32(o+4,true);if(id==="data"){dO=o+8;dL=s;break;}o+=8+s+(s&1);}const n=dL/2,out=new Float32Array(n);for(let i=0;i<n;i++)out[i]=dv.getInt16(dO+i*2,true)/32768;return out;}
const wav = readWav("/tmp/pk_120s.wav");
const ctx = new GpuContext(await getDevice());
const rdU8=(p)=>Uint8Array.from(readFileSync(p));
const enc = loadParakeetEncoder(ctx, rdU8("/tmp/pk-raw/enc-int8/weights.bin"), JSON.parse(readFileSync("/tmp/pk-raw/enc-int8/manifest.json")));
const decMan = JSON.parse(readFileSync("/tmp/pk-raw/dec/manifest.json"));
const db = rdU8("/tmp/pk-raw/dec/weights.bin");
const decBin = new Float32Array(db.buffer, db.byteOffset, db.byteLength/4);
const wasmB = readFileSync("src/engines/asr-parakeet/parakeet-decoder.wasm");
const dec8 = await loadWasmDecoder(wasmB, decBin, decMan, { int8: true });
const dec32 = await loadWasmDecoder(wasmB, decBin, decMan, { int8: false });
const mel = new ParakeetMel(128);
const WIN=15*16000, hop=13*16000;
let same=0, diff=0, tot8=0, tot32=0;
for (let s=0; s+WIN<=wav.length; s+=hop) {
  const { features } = mel.process(wav.subarray(s, s+WIN));
  const r = await parakeetEncodeBatch(ctx, enc, [features]);
  const frames = await ctx.download(r.framesGpu);
  const a = wasmDecode(dec8, frames, r.Tsub).ids;
  const b = wasmDecode(dec32, frames, r.Tsub).ids;
  tot8+=a.length; tot32+=b.length;
  if (a.length===b.length && a.every((v,i)=>v===b[i])) same++; else diff++;
}
console.log(`windows identical: ${same}, differing: ${diff}  (tokens int8 ${tot8} vs fp32 ${tot32})`);
process.exit(diff ? 1 : 0);
