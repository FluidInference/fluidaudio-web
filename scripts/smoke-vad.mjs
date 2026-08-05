// Silero VAD — headless smoke (ort-node). Needs silero_vad.onnx locally.
//   node scripts/smoke-vad.mjs [wav] [modelPath]
import ort from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { sileroDetect } from "../src/engines/vad-silero/silero.js";

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
const model = process.argv[3] || "/tmp/silero_vad.onnx";
const session = await ort.InferenceSession.create(model);
const audio = readWav(wav);
const t0 = Date.now();
const ranges = await sileroDetect({ ort, session, audio });
const total = ranges.reduce((a, r) => a + (r.end - r.start), 0);
console.log(
  `${((Date.now() - t0) / 1000).toFixed(2)}s  ${ranges.length} speech ranges  (${total.toFixed(2)}s speech of ${(audio.length / 16000).toFixed(2)}s)`,
);
console.log(JSON.stringify(ranges));
