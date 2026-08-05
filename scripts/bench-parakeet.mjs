// Speed run for the internalized Parakeet v3 core on onnxruntime-node (CPU).
//   node scripts/bench-parakeet.mjs [int8|fp32] [dir] [wav...]
// NOTE: this is a CPU baseline (ort-node). In the browser the encoder runs on
// WebGPU — expect the encode stage to be substantially faster there; mel +
// decoder (WASM) are comparable. int8 encoder is CPU-degenerate (empty), so use
// fp32 for correct-output timing.
import ort from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { OnnxMelPreprocessor } from "../src/engines/asr-parakeet/onnxMel.js";
import { ParakeetTokenizer } from "../src/engines/asr-parakeet/tokenizer.js";
import { transcribeTdt } from "../src/engines/asr-parakeet/tdt.js";

function readWav(p) {
  const b = readFileSync(p);
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

const quant = process.argv[2] || "fp32";
const dir = process.argv[3] || "/tmp/pkv3";
const wavs = process.argv.slice(4);
if (!wavs.length) wavs.push("/tmp/pk_intro.wav", "/tmp/pk_30s.wav", "/tmp/pk_60s.wav", "/tmp/pk_120s.wav");
const encFile = quant === "int8" ? "encoder-model.int8.onnx" : "encoder-model.onnx";

const encoder = await ort.InferenceSession.create(resolve(dir, encFile));
const decoder = await ort.InferenceSession.create(resolve(dir, "decoder_joint-model.int8.onnx"));
const melSession = await ort.InferenceSession.create(resolve(dir, "nemo128.onnx"));
const tokenizer = ParakeetTokenizer.fromVocabText(readFileSync(resolve(dir, "vocab.txt"), "utf8"));
const preprocessor = new OnnxMelPreprocessor(ort, melSession, 128);

console.log(`encoder=${encFile} (ort-node CPU, ${ort.env?.wasm ? "" : ""}threads=${(await import("node:os")).cpus().length})`);
console.log("audio_s   mel_ms  enc_ms  dec_ms  total_ms   RTFx   tokens");
for (const wav of wavs) {
  const { samples, sampleRate } = readWav(wav);
  const dur = samples.length / sampleRate;
  await transcribeTdt({ ort, encoder, decoder, preprocessor, tokenizer, audio: samples }); // warm
  const r = await transcribeTdt({ ort, encoder, decoder, preprocessor, tokenizer, audio: samples });
  const m = r.metrics;
  const rtfx = dur / (m.totalMs / 1000);
  console.log(
    `${dur.toFixed(1).padStart(6)}   ${String(m.melMs).padStart(6)}  ${String(m.encodeMs).padStart(6)}  ${String(m.decodeMs).padStart(6)}  ${String(m.totalMs).padStart(8)}   ${rtfx.toFixed(1).padStart(5)}   ${r.tokenIds.length}`,
  );
}
