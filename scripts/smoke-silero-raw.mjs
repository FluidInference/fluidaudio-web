// Parity proof for the ORT-free Silero VAD forward (raw-silero.js): runs the raw
// JS forward and the real silero_vad.onnx over a real audio file in 512-sample
// streaming chunks (state threaded across chunks) and reports max |Δ| in the
// per-chunk speech probability. Should be ~1e-6.
//
//   node scripts/smoke-silero-raw.mjs [audio.wav] [silero_vad.onnx] [weightsDir]
// Prereq: weights via  python3 scripts/extract-silero-weights.py <onnx> <weightsDir>
import ort from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { makeSileroWeights, sileroForward } from "../src/engines/vad-silero/raw-silero.js";

const wav = process.argv[2] || "/tmp/cowen.wav";
const onnx = process.argv[3] || "/tmp/silero_vad.onnx";
const dir = process.argv[4] || "/tmp/silero-raw";

function readWav(p, maxSec) {
  const b = readFileSync(p);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let o = 12, dO = -1, dL = 0, sr = 16000;
  while (o + 8 <= b.length) {
    const id = String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);
    const s = dv.getUint32(o + 4, true);
    if (id === "fmt ") sr = dv.getUint32(o + 12, true);
    if (id === "data") { dO = o + 8; dL = s; break; }
    o += 8 + s + (s & 1);
  }
  let n = dL / 2;
  if (maxSec) n = Math.min(n, maxSec * sr);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(dO + i * 2, true) / 32768;
  return out;
}

const man = JSON.parse(readFileSync(`${dir}/manifest.json`, "utf8"));
const buf = readFileSync(`${dir}/weights.bin`);
const W = makeSileroWeights(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4), man);
const audio = readWav(wav, 10);
const s = await ort.InferenceSession.create(onnx);

let ortState = new Float32Array(256);
let rawState = new Float32Array(256);
let maxD = 0, n = 0;
for (let i = 0; i + 512 <= audio.length; i += 512) {
  const chunk = audio.subarray(i, i + 512);
  const oo = await s.run({
    input: new ort.Tensor("float32", Float32Array.from(chunk), [1, 512]),
    state: new ort.Tensor("float32", ortState, [2, 1, 128]),
    sr: new ort.Tensor("int64", BigInt64Array.from([16000n]), []),
  });
  ortState = Float32Array.from(oo.stateN.data);
  const { prob, state } = sileroForward(chunk, rawState, W);
  rawState = state;
  maxD = Math.max(maxD, Math.abs(prob - oo.output.data[0]));
  n++;
}
console.log(`chunks=${n}  max prob |Δ| vs ORT = ${maxD.toExponential(2)}`);
if (maxD > 1e-4) { console.error("PARITY FAIL"); process.exit(1); }
console.log("PARITY OK");
