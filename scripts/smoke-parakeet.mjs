// End-to-end smoke test: Parakeet TDT v3 via parakeet.js in Node.
//   node scripts/smoke-parakeet.mjs /path/to/16k-mono.wav
import { fromHub } from "parakeet.js";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
// Node-only: ort-web defaults its wasm path to an https CDN URL, which Node's
// ESM loader rejects. Import the *exact* ort instance parakeet.js resolves (its
// nested v1.24.1, `node` export → ort.node.min.mjs) and point it at local wasm.
// Node dedupes by resolved path, so this is the same module parakeet uses.
// (In the browser none of this is needed — the CDN path works.)
const ortEntry = resolve(
  "node_modules/parakeet.js/node_modules/onnxruntime-web/dist/ort.node.min.mjs"
);
const ort = await import(pathToFileURL(ortEntry).href);
ort.env.wasm.wasmPaths = pathToFileURL(
  resolve("node_modules/parakeet.js/node_modules/onnxruntime-web/dist/") + "/"
).href;

function readWavMono16(path) {
  const buf = readFileSync(path);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  // find 'data' chunk
  let off = 12;
  let dataOff = -1, dataLen = 0, sr = 16000, bits = 16;
  while (off + 8 <= buf.length) {
    const id = String.fromCharCode(buf[off], buf[off + 1], buf[off + 2], buf[off + 3]);
    const size = dv.getUint32(off + 4, true);
    if (id === "fmt ") { sr = dv.getUint32(off + 12, true); bits = dv.getUint16(off + 22, true); }
    if (id === "data") { dataOff = off + 8; dataLen = size; break; }
    off += 8 + size + (size & 1);
  }
  if (dataOff < 0 || bits !== 16) throw new Error("expected 16-bit PCM wav");
  const n = dataLen / 2;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(dataOff + i * 2, true) / 32768;
  return { samples: out, sampleRate: sr };
}

const wavPath = process.argv[2] ?? "/tmp/pk_intro.wav";
const { samples, sampleRate } = readWavMono16(wavPath);
console.log(`audio: ${(samples.length / sampleRate).toFixed(1)}s @ ${sampleRate}Hz`);

console.log("loading parakeet-tdt-0.6b-v3 (int8)…");
const model = await fromHub("parakeet-tdt-0.6b-v3", {
  encoderQuant: "int8",
  decoderQuant: "int8",
  backend: "wasm",
});

console.log("transcribing…");
const t0 = Date.now();
const res = await model.transcribe(samples, sampleRate);
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
console.log("TEXT:", res.text ?? res.utterance ?? JSON.stringify(res).slice(0, 300));
