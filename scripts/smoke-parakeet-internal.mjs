// Headless proof of the INTERNALIZED Parakeet pipeline (no parakeet.js): same
// mel.js + tokenizer.js + tdt.js the browser engine uses, driven by
// onnxruntime-node against local ONNX files.
//
//   node scripts/smoke-parakeet-internal.mjs /tmp/pk_intro.wav /tmp/pkv3 [int8|fp32]
//
// NOTE: the int8 encoder is degenerate on CPU → pass fp32 for a real Node
// transcript (encoder-model.onnx + .data). In the browser the int8 encoder runs
// on WebGPU and is fine.
import ort from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { OnnxMelPreprocessor } from "../src/engines/asr-parakeet/onnxMel.js";
import { ParakeetTokenizer } from "../src/engines/asr-parakeet/tokenizer.js";
import { transcribeTdt } from "../src/engines/asr-parakeet/tdt.js";

function readWav(path) {
  const b = readFileSync(path);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let o = 12,
    dO = -1,
    dL = 0,
    sr = 16000;
  while (o + 8 <= b.length) {
    const id = String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);
    const s = dv.getUint32(o + 4, true);
    if (id === "fmt ") sr = dv.getUint32(o + 12, true);
    if (id === "data") {
      dO = o + 8;
      dL = s;
      break;
    }
    o += 8 + s + (s & 1);
  }
  const n = dL / 2,
    out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(dO + i * 2, true) / 32768;
  return { samples: out, sampleRate: sr };
}

const wav = process.argv[2] || "/tmp/pk_intro.wav";
const dir = process.argv[3] || "/tmp/pkv3";
const quant = process.argv[4] || "fp32";
const encFile = quant === "int8" ? "encoder-model.int8.onnx" : "encoder-model.onnx";

const encoder = await ort.InferenceSession.create(resolve(dir, encFile));
const decoder = await ort.InferenceSession.create(resolve(dir, "decoder_joint-model.int8.onnx"));
const melSession = await ort.InferenceSession.create(resolve(dir, "nemo128.onnx"));
const tokenizer = ParakeetTokenizer.fromVocabText(readFileSync(resolve(dir, "vocab.txt"), "utf8"));
const preprocessor = new OnnxMelPreprocessor(ort, melSession, 128);

const { samples } = readWav(wav);
const t0 = Date.now();
const { text, tokenIds } = await transcribeTdt({ ort, encoder, decoder, preprocessor, tokenizer, audio: samples });
console.log(`encoder=${encFile}  ${((Date.now() - t0) / 1000).toFixed(2)}s  ${tokenIds.length} tokens`);
console.log("TEXT:", JSON.stringify(text));
