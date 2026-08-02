// WER of the browser Parakeet core (fp32 encoder) on LibriSpeech test-clean —
// the canonical FluidAudio ASR benchmark. Decodes flac via ffmpeg.
//   node scripts/wer-librispeech.mjs [N] [pkDir]
import ort from "onnxruntime-node";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { resolve, join } from "node:path";
import { OnnxMelPreprocessor } from "../src/engines/asr-parakeet/onnxMel.js";
import { ParakeetTokenizer } from "../src/engines/asr-parakeet/tokenizer.js";
import { transcribeTdt } from "../src/engines/asr-parakeet/tdt.js";

const N = parseInt(process.argv[2] || "100", 10);
const dir = process.argv[3] || "/tmp/pkv3";
const LS = "/Users/hanweng/Library/Application Support/FluidAudio/Datasets/LibriSpeech/test-clean";

function walk(d, out = []) {
  for (const e of readdirSync(d)) {
    const p = join(d, e);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (e.endsWith(".trans.txt")) out.push(p);
  }
  return out;
}
function flacToF32(path) {
  const buf = execFileSync("ffmpeg", ["-v", "quiet", "-i", path, "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
    { maxBuffer: 1 << 28 });
  const n = buf.length >> 1, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = buf.readInt16LE(i * 2) / 32768;
  return out;
}
const norm = (t) => t.toLowerCase().replace(/[^a-z0-9' ]/g, " ").split(/\s+/).filter(Boolean);
function wer(ref, hyp) {
  const r = norm(ref), h = norm(hyp), n = r.length, m = h.length;
  const d = Array.from({ length: m + 1 }, (_, i) => i);
  for (let i = 1; i <= n; i++) { let prev = d[0]; d[0] = i;
    for (let j = 1; j <= m; j++) { const cur = d[j]; d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] !== h[j - 1] ? 1 : 0)); prev = cur; } }
  return { err: d[m], n };
}

// collect utterances
const utts = [];
for (const t of walk(LS)) {
  const base = t.slice(0, -".trans.txt".length);
  for (const line of readFileSync(t, "utf8").split(/\r?\n/)) {
    if (!line) continue; const sp = line.indexOf(" ");
    utts.push({ flac: `${base.slice(0, base.lastIndexOf("/"))}/${line.slice(0, sp)}.flac`, ref: line.slice(sp + 1) });
  }
}
utts.sort((a, b) => a.flac.localeCompare(b.flac));
const sample = utts.slice(0, N);

const encoder = await ort.InferenceSession.create(resolve(dir, "encoder-model.onnx"));
const decoder = await ort.InferenceSession.create(resolve(dir, "decoder_joint-model.int8.onnx"));
const preprocessor = new OnnxMelPreprocessor(ort, await ort.InferenceSession.create(resolve(dir, "nemo128.onnx")), 128);
const tokenizer = ParakeetTokenizer.fromVocabText(readFileSync(resolve(dir, "vocab.txt"), "utf8"));

let totErr = 0, totN = 0, totDur = 0, totMs = 0, done = 0;
for (const u of sample) {
  let audio; try { audio = flacToF32(u.flac); } catch { continue; }
  const t0 = Date.now();
  const r = await transcribeTdt({ ort, encoder, decoder, preprocessor, tokenizer, audio });
  totMs += Date.now() - t0; totDur += audio.length / 16000;
  const { err, n } = wer(u.ref, r.text); totErr += err; totN += n; done++;
  if (done % 20 === 0) process.stderr.write(`  ${done}/${sample.length} WER ${(100 * totErr / totN).toFixed(2)}%\n`);
}
console.log(`\nLibriSpeech test-clean (${done} utts, ${totDur.toFixed(0)}s audio):`);
console.log(`WER: ${(100 * totErr / totN).toFixed(2)}%  (${totErr}/${totN})  | avg RTFx ${(totDur / (totMs / 1000)).toFixed(1)}x (fp32 CPU)`);
