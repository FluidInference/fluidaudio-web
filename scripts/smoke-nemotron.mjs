// Nemotron 3.5 streaming ASR — headless smoke (ort-node). Needs /tmp/nemo/
// (encoder.onnx+.data, decoder.onnx+.data, joint.onnx+.data, vocab.txt).
//   node scripts/smoke-nemotron.mjs [wav] [langCode]
import ort from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { JsPreprocessor } from "../src/engines/asr-nemotron/nemotron-mel.js";
import { nemotronTranscribe, makeNemotronTokenizer, makeNemotronLangMap } from "../src/engines/asr-nemotron/nemotron-decode.js";

function readWav(p){const b=readFileSync(p);const dv=new DataView(b.buffer,b.byteOffset,b.byteLength);let o=12,dO=-1,dL=0;while(o+8<=b.length){const id=String.fromCharCode(b[o],b[o+1],b[o+2],b[o+3]);const s=dv.getUint32(o+4,true);if(id==="data"){dO=o+8;dL=s;break;}o+=8+s+(s&1);}const n=dL/2,out=new Float32Array(n);for(let i=0;i<n;i++)out[i]=dv.getInt16(dO+i*2,true)/32768;return out;}

const wav = process.argv[2] || "/tmp/pk_intro.wav";
const lang = process.argv[3] || "en-US";
const D = "/tmp/nemo";
const encoder = await ort.InferenceSession.create(`${D}/encoder.onnx`);
const decoder = await ort.InferenceSession.create(`${D}/decoder.onnx`);
const joint = await ort.InferenceSession.create(`${D}/joint.onnx`);
const vocab = readFileSync(`${D}/vocab.txt`, "utf8");
const tokenizer = makeNemotronTokenizer(vocab);
const langId = makeNemotronLangMap(vocab)[lang] ?? 24;
const audio = readWav(wav);
const t0 = Date.now();
const r = await nemotronTranscribe({ ort, encoder, decoder, joint, preprocessor: new JsPreprocessor({ nMels: 128 }), tokenizer, audio, langId });
console.log(`lang=${lang} (id ${langId}) ${((Date.now() - t0) / 1000).toFixed(1)}s ${r.tokenIds.length} tokens`);
console.log("TEXT:", JSON.stringify(r.text));
