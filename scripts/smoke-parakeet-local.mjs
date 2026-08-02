// Node end-to-end proof for Parakeet v3: bypass parakeet.js's browser Blob-URL
// hub by calling ParakeetModel.fromUrls with local ONNX paths + a data: URL
// tokenizer. Download the files first (see README/commit msg), then:
//   node scripts/smoke-parakeet-local.mjs /tmp/pk_intro.wav /tmp/pkv3
import { ParakeetModel } from "parakeet.js";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

// Set the local wasm path on the exact ort instance parakeet.js uses (Node-only;
// browser uses the CDN). Passing wasmPaths via fromUrls isn't enough — the
// backend registration reads env.wasm.wasmPaths directly.
const ortDist = resolve("node_modules/parakeet.js/node_modules/onnxruntime-web/dist/");
const ort = await import(pathToFileURL(resolve(ortDist, "ort.node.min.mjs")).href);
ort.env.wasm.wasmPaths = pathToFileURL(ortDist + "/").href;

const wavPath = process.argv[2] ?? "/tmp/pk_intro.wav";
const dir = process.argv[3] ?? "/tmp/pkv3";

function readWavMono16(path) {
  const buf = readFileSync(path);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let off = 12, dataOff = -1, dataLen = 0, sr = 16000, bits = 16;
  while (off + 8 <= buf.length) {
    const id = String.fromCharCode(buf[off], buf[off + 1], buf[off + 2], buf[off + 3]);
    const size = dv.getUint32(off + 4, true);
    if (id === "fmt ") { sr = dv.getUint32(off + 12, true); bits = dv.getUint16(off + 22, true); }
    if (id === "data") { dataOff = off + 8; dataLen = size; break; }
    off += 8 + size + (size & 1);
  }
  if (dataOff < 0 || bits !== 16) throw new Error("expected 16-bit PCM wav");
  const n = dataLen / 2, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(dataOff + i * 2, true) / 32768;
  return { samples: out, sampleRate: sr };
}

const { samples, sampleRate } = readWavMono16(wavPath);
console.log(`audio: ${(samples.length / sampleRate).toFixed(1)}s @ ${sampleRate}Hz`);

const vocab = readFileSync(resolve(dir, "vocab.txt"));
const tokenizerUrl = "data:text/plain;base64," + vocab.toString("base64");
const wasmPaths = pathToFileURL(
  resolve("node_modules/parakeet.js/node_modules/onnxruntime-web/dist/") + "/"
).href;

console.log("loading Parakeet v3 int8 (wasm)…");
const model = await ParakeetModel.fromUrls({
  encoderUrl: resolve(dir, "encoder-model.int8.onnx"),
  decoderUrl: resolve(dir, "decoder_joint-model.int8.onnx"),
  tokenizerUrl,
  preprocessorBackend: "js",
  nMels: 128,
  backend: "wasm",
  wasmPaths,
  filenames: { encoder: "encoder-model.int8", decoder: "decoder_joint-model.int8" },
});

console.log("transcribing…");
const t0 = Date.now();
const res = await model.transcribe(samples, sampleRate);
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
console.log("TEXT:", res.text ?? res.utterance ?? JSON.stringify(res).slice(0, 400));
