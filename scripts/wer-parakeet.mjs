// Accuracy check for the internalized Parakeet v3 core: WER on FLEURS en_us.
//   node scripts/wer-parakeet.mjs [N] [dir]
import ort from "onnxruntime-node";
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { OnnxMelPreprocessor } from "../src/engines/asr-parakeet/onnxMel.js";
import { ParakeetTokenizer } from "../src/engines/asr-parakeet/tokenizer.js";
import { transcribeTdt } from "../src/engines/asr-parakeet/tdt.js";

const N = parseInt(process.argv[2] || "20", 10);
const dir = process.argv[3] || "/tmp/pkv3";
const FLEURS = "/Users/hanweng/Library/Application Support/FluidAudio/FLEURS-full/en_us";

function readWav(p){const b=readFileSync(p);const dv=new DataView(b.buffer,b.byteOffset,b.byteLength);let o=12,dO=-1,dL=0,sr=16000,bits=16;while(o+8<=b.length){const id=String.fromCharCode(b[o],b[o+1],b[o+2],b[o+3]);const s=dv.getUint32(o+4,true);if(id==="fmt "){sr=dv.getUint32(o+12,true);bits=dv.getUint16(o+22,true);}if(id==="data"){dO=o+8;dL=s;break;}o+=8+s+(s&1);}if(bits!==16)throw new Error("bits="+bits);const n=dL/2,out=new Float32Array(n);for(let i=0;i<n;i++)out[i]=dv.getInt16(dO+i*2,true)/32768;return{samples:out,sampleRate:sr};}
const norm=(t)=>t.toLowerCase().replace(/[^a-z0-9' ]/g," ").split(/\s+/).filter(Boolean);
function wer(ref,hyp){const r=norm(ref),h=norm(hyp),n=r.length,m=h.length;let d=Array.from({length:m+1},(_,i)=>i);for(let i=1;i<=n;i++){let prev=d[0];d[0]=i;for(let j=1;j<=m;j++){const cur=d[j];d[j]=Math.min(d[j]+1,d[j-1]+1,prev+(r[i-1]!==h[j-1]?1:0));prev=cur;}}return{err:d[m],n};}

const refs=new Map();
for(const line of readFileSync(`${FLEURS}/en_us.trans.txt`,"utf8").split(/\r?\n/)){if(!line)continue;const sp=line.indexOf(" ");refs.set(line.slice(0,sp),line.slice(sp+1));}

const encoder=await ort.InferenceSession.create(resolve(dir,"encoder-model.onnx"));
const decoder=await ort.InferenceSession.create(resolve(dir,"decoder_joint-model.int8.onnx"));
const preprocessor=new OnnxMelPreprocessor(ort,await ort.InferenceSession.create(resolve(dir,"nemo128.onnx")),128);
const tokenizer=ParakeetTokenizer.fromVocabText(readFileSync(resolve(dir,"vocab.txt"),"utf8"));

const ids=[...refs.keys()].slice(0,N);
let totErr=0,totN=0,totDur=0,totMs=0;
for(const id of ids){const wav=`${FLEURS}/${id}.wav`;if(!existsSync(wav))continue;const{samples,sampleRate}=readWav(wav);if(sampleRate!==16000)continue;
  const r=await transcribeTdt({ort,encoder,decoder,preprocessor,tokenizer,audio:samples});
  const {err,n}=wer(refs.get(id),r.text);totErr+=err;totN+=n;totDur+=samples.length/sampleRate;totMs+=r.metrics.totalMs;}
console.log(`files: ${ids.length}  audio: ${totDur.toFixed(0)}s  avg RTFx: ${(totDur/(totMs/1000)).toFixed(1)}x`);
console.log(`WER: ${(100*totErr/totN).toFixed(2)}%  (${totErr}/${totN} words)`);
