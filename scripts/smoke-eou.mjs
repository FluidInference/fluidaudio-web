// Parakeet EOU 120M — headless smoke (ort-node). Needs /tmp/eou/
// (enc.onnx, dj.onnx, vocab.txt).
//   node scripts/smoke-eou.mjs [wav]
// EOU is a NeMo streaming RNNT: it wants UN-normalized (NA) log-mel — the same
// frontend Nemotron uses (no per-feature CMVN). Feeding Parakeet's per_feature
// mel makes the encoder emit content-free frames → all-blank.
import ort from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { JsPreprocessor } from "../src/engines/asr-nemotron/nemotron-mel.js";
import { eouTranscribe, makeEouTokenizer } from "../src/engines/eou-parakeet/eou-decode.js";

function readWav(p) {
  const b = readFileSync(p);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let o = 12,
    dO = -1,
    dL = 0;
  while (o + 8 <= b.length) {
    const id = String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);
    const s = dv.getUint32(o + 4, true);
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
  return out;
}

const wav = process.argv[2] || "/tmp/pk_intro.wav";
const D = "/tmp/eou";
const encoder = await ort.InferenceSession.create(`${D}/enc.onnx`);
const decoder = await ort.InferenceSession.create(`${D}/dj.onnx`);
const tokenizer = makeEouTokenizer(readFileSync(`${D}/vocab.txt`, "utf8"));
const audio = readWav(wav);
const t0 = Date.now();
const r = await eouTranscribe({
  ort,
  encoder,
  decoder,
  preprocessor: new JsPreprocessor({ nMels: 128 }),
  tokenizer,
  audio,
});
console.log(`${((Date.now() - t0) / 1000).toFixed(1)}s  ${r.tokenIds.length} tokens  ${r.frames} frames`);
console.log("TEXT :", JSON.stringify(r.text));
console.log("EVENTS:", JSON.stringify(r.events));
