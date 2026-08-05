import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadParakeetEncoder, parakeetEncode } from "../src/engines/asr-parakeet/raw-encoder.js";
import { loadSortformerHead, sortformerHead } from "../src/engines/diarization-sortformer/raw-sortformer-head.js";
import { ParakeetMel } from "../src/engines/asr-parakeet/parakeet-mel.js";
function readWav(p){const b=readFileSync(p);const dv=new DataView(b.buffer,b.byteOffset,b.byteLength);let o=12,dO=-1,dL=0;while(o+8<=b.length){const id=String.fromCharCode(b[o],b[o+1],b[o+2],b[o+3]);const s=dv.getUint32(o+4,true);if(id==="data"){dO=o+8;dL=s;break;}o+=8+s+(s&1);}const n=dL/2,out=new Float32Array(n);for(let i=0;i<n;i++)out[i]=dv.getInt16(dO+i*2,true)/32768;return out;}
const one = readWav("/tmp/earn40.wav");
const ctx = new GpuContext(await getDevice());
const rdU8=(p)=>Uint8Array.from(readFileSync(p));
const enc = loadParakeetEncoder(ctx, rdU8("/tmp/sf-raw/enc-int8/weights.bin"), JSON.parse(readFileSync("/tmp/sf-raw/enc-int8/manifest.json")), { xscale: true });
const mel = new ParakeetMel(128);
const { features, length } = mel.process(one);
console.log("mel frames", length);
const r = await parakeetEncode(ctx, enc, features, length);
const fr = await ctx.download(r.framesGpu);
console.log("enc out:", r.Tsub, "NaN", fr.some(Number.isNaN), "rms", Math.sqrt(fr.reduce((s,v)=>s+v*v,0)/fr.length).toFixed(4));
// vs ORT ref? compare with /tmp/sf/ref-encout only if same audio (that ref was for a specific clip) — skip; just head:
const hb = rdU8("/tmp/sf-raw/head/weights.bin");
const head = loadSortformerHead(ctx, new Float32Array(hb.buffer,hb.byteOffset,hb.byteLength/4), JSON.parse(readFileSync("/tmp/sf-raw/head/manifest.json")));
const preds = await sortformerHead(ctx, head, r.framesGpu, r.Tsub);
console.log("preds:", preds.length/4, "frames, NaN", Array.from(preds).some(Number.isNaN), "max", Math.max(...preds).toFixed(3));
process.exit(0);
