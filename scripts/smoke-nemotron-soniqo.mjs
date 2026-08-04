import ort from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { JsPreprocessor } from "../src/engines/asr-nemotron/nemotron-mel.js";
function readWav(p){const b=readFileSync(p);const dv=new DataView(b.buffer,b.byteOffset,b.byteLength);let o=12,dO=-1,dL=0;while(o+8<=b.length){const id=String.fromCharCode(b[o],b[o+1],b[o+2],b[o+3]);const s=dv.getUint32(o+4,true);if(id==="data"){dO=o+8;dL=s;break;}o+=8+s+(s&1);}const n=dL/2,out=new Float32Array(n);for(let i=0;i<n;i++)out[i]=dv.getInt16(dO+i*2,true)/32768;return out;}
const enc=await ort.InferenceSession.create("/tmp/nemo-fp16/encoder.onnx");
const dec=await ort.InferenceSession.create("/tmp/nemo-fp16/decoder.onnx");
const jnt=await ort.InferenceSession.create("/tmp/nemo-fp16/joint.onnx");
const vocab=JSON.parse(readFileSync("/tmp/nemo-fp16/vocab.json","utf8"));
const BLANK=13087;
const audio=readWav(process.argv[2]||"/tmp/pk_intro.wav");
const pp=new JsPreprocessor({nMels:128}); const {features}=pp.process(audio); const T=features.length/128;
const F=32,M=128;
const lm=new Float32Array(128); lm[0]=1;
let pre=new Float32Array(M*9), clc=new Float32Array(24*56*1024), clt=new Float32Array(24*1024*8), clen=0;
// RNNT state
let token=BLANK, h=new Float32Array(2*640), c=new Float32Array(2*640);
const ids=[];
const nChunks=Math.ceil(T/F);
for(let ci=0;ci<nChunks;ci++){
  const chunk=new Float32Array(M*F); const valid=Math.min(F,T-ci*F);
  for(let m=0;m<M;m++)for(let t=0;t<valid;t++)chunk[m*F+t]=features[m*T+ci*F+t];
  const o=await enc.run({
    audio_signal:new ort.Tensor("float32",chunk,[1,128,32]),
    audio_length:new ort.Tensor("int32",Int32Array.from([valid]),[1]),
    language_mask:new ort.Tensor("float32",lm,[1,128]),
    pre_cache:new ort.Tensor("float32",pre,[1,128,9]),
    cache_last_channel:new ort.Tensor("float32",clc,[24,1,56,1024]),
    cache_last_time:new ort.Tensor("float32",clt,[24,1,1024,8]),
    cache_last_channel_len:new ort.Tensor("int32",Int32Array.from([clen]),[1]),
  });
  // carry caches
  pre=o["new_pre_cache"].data.slice(); const nc=o["new_cache_last_channel"]; clc=nc.data.slice();
  clt=o["new_cache_last_time"].data.slice(); clen=Number(o["new_cache_last_channel_len"].data[0]);
  // resize clc/clt if dims grew? they're dynamic. Rebuild with reported dims:
  const nlen=o["encoded_length"]?Number(o["encoded_length"].data[0]):4;
  const eo=o["encoded_output"].data; const nf=o["encoded_output"].dims[1];
  for(let f=0;f<Math.min(nf,nlen||nf);f++){
    const encF=new Float32Array(1024); for(let d=0;d<1024;d++)encF[d]=eo[f*1024+d];
    let emitted=0;
    while(emitted<10){
      const dr=await dec.run({token:new ort.Tensor("int64",BigInt64Array.from([BigInt(token)]),[1,1]),h:new ort.Tensor("float32",h,[2,1,640]),c:new ort.Tensor("float32",c,[2,1,640])});
      const jr=await jnt.run({encoder_output:new ort.Tensor("float32",encF,[1,1,1024]),decoder_output:dr["decoder_output"]});
      const lg=jr["logits"].data; let mi=0,mv=-1e9; for(let i=0;i<13088;i++){if(lg[i]>mv){mv=lg[i];mi=i;}}
      if(mi===BLANK)break;
      ids.push(mi); token=mi; h=dr["h_out"].data.slice(); c=dr["c_out"].data.slice(); emitted++;
    }
  }
}
const text=ids.map(i=>vocab[i]||"").filter(t=>!t.startsWith("<")).join("").replace(/▁/g," ").trim();
console.log("tokens",ids.length,"\nTEXT:",JSON.stringify(text));
process.exit(0);
