import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadKokoroBackend } from "../src/engines/tts-kokoro/synth-backend.js";
const K0="/tmp/kokoro";
const rd=(p)=>{const u=Uint8Array.from(readFileSync(p));return new Float32Array(u.buffer,u.byteOffset,u.byteLength/4);};
const vocab=JSON.parse(readFileSync("src/engines/tts-kokoro/vocab.json"));
const inv={};for(const[k,v]of Object.entries(vocab))inv[v]=k;
// local fetchCached mapping HF paths → /tmp files
const map={"kokoro/weights.bin":`${K0}/kw/weights.bin`,"kokoro/manifest.json":`${K0}/kw/manifest.json`,"kokoro/roles.json":`${K0}/kw/roles.json`,"kokoro/be_w.bin":`${K0}/frontend/be_w.bin`,"kokoro/be_b.bin":`${K0}/frontend/be_b.bin`,"kokoro/ref.json":`${K0}/frontend/ref.json`,"kokoro/albert/manifest.json":`${K0}/albert/manifest.json`,"voices/af_heart.bin":`${K0}/af_heart.bin`};
const hfUrl=(repo,path)=>path;
const fetchCached=async(path)=>{ if(map[path])return Uint8Array.from(readFileSync(map[path])); if(path.startsWith("kokoro/albert/"))return Uint8Array.from(readFileSync(`${K0}/albert/${path.split("/").pop()}`)); throw new Error("no map "+path); };
const be=await loadKokoroBackend(fetchCached,hfUrl,vocab);
// reverse hello_ids → phoneme string (drop leading/trailing $=0)
const idb=Uint8Array.from(readFileSync(`${K0}/hello_ids.bin`));const ids=new Int32Array(idb.buffer,idb.byteOffset,idb.byteLength/4);
const phon=Array.from(ids).filter(x=>x!==0).map(x=>inv[x]).join("");
console.log("phonemes:",JSON.stringify(phon),"| reconstructed ids from those:",Array.from(ids).length);
// check voice style vs hello_style
const pack=rd(`${K0}/af_heart.bin`);const si=256*Math.min(Math.max(ids.length-2,0),509);
const hs=rd(`${K0}/hello_style.bin`);let sd=0;for(let i=0;i<256;i++)sd=Math.max(sd,Math.abs(pack[si+i]-hs[i]));
console.log(`voice style af_heart[${ids.length-2}] vs hello_style maxΔ ${sd.toExponential(2)}`);
const wav=await be.synthFromPhonemes(phon,"af_heart");
const ref=rd(`${K0}/real_wav.bin`);const m=Math.min(wav.length,ref.length);let dot=0,a=0,b=0;for(let i=0;i<m;i++){dot+=wav[i]*ref[i];a+=wav[i]**2;b+=ref[i]**2;}
console.log(`backend synth: len ${wav.length} vs ${ref.length}  corr ${(dot/Math.sqrt(a*b)).toFixed(4)}`);
process.exit(0);
