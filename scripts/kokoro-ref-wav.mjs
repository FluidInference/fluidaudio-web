import { readFileSync, writeFileSync } from "node:fs";
const rd=(p)=>{const u=Uint8Array.from(readFileSync(p));return new Float32Array(u.buffer,u.byteOffset,u.byteLength/4);};
const wav=rd("/tmp/kokoro/real_wav.bin");
const i16=new Int16Array(wav.length);for(let i=0;i<wav.length;i++)i16[i]=Math.max(-32768,Math.min(32767,Math.round(wav[i]*32767)));
const sr=24000,h=Buffer.alloc(44);h.write("RIFF",0);h.writeUInt32LE(36+i16.byteLength,4);h.write("WAVE",8);h.write("fmt ",12);h.writeUInt32LE(16,16);h.writeUInt16LE(1,20);h.writeUInt16LE(1,22);h.writeUInt32LE(sr,24);h.writeUInt32LE(sr*2,28);h.writeUInt16LE(2,32);h.writeUInt16LE(16,34);h.write("data",36);h.writeUInt32LE(i16.byteLength,40);
writeFileSync("/tmp/kokoro/ort_hello.wav",Buffer.concat([h,Buffer.from(i16.buffer)]));
// numerical sanity on my wav
const my=rd("/tmp/kokoro/js_hello.wav".replace(".wav",".wav")); // placeholder, re-read js wav samples below
console.log("wrote ort_hello.wav");
