// Windowed sortformer gate: 120s (earn40 x3) must give 2 CONSISTENT speakers
// across the 90s/15s stitch seam; 40s single-window path unchanged.
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadParakeetEncoder, parakeetEncode } from "../src/engines/asr-parakeet/raw-encoder.js";
import { loadSortformerHead, sortformerHead, predsToSegments, mergeWindowPreds } from "../src/engines/diarization-sortformer/raw-sortformer-head.js";
import { ParakeetMel } from "../src/engines/asr-parakeet/parakeet-mel.js";
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
const one = readWav("/tmp/earn40.wav");
const three = new Float32Array(one.length * 3);
for (let i = 0; i < 3; i++) three.set(one, i * one.length);
const ctx = new GpuContext(await getDevice());
const rdU8 = (p) => Uint8Array.from(readFileSync(p));
const enc = loadParakeetEncoder(ctx, rdU8("/tmp/sf-raw/enc-int8/weights.bin"), JSON.parse(readFileSync("/tmp/sf-raw/enc-int8/manifest.json")), {
  xscale: true,
});
const hb = rdU8("/tmp/sf-raw/head/weights.bin");
const head = loadSortformerHead(ctx, new Float32Array(hb.buffer, hb.byteOffset, hb.byteLength / 4), JSON.parse(readFileSync("/tmp/sf-raw/head/manifest.json")));
const mel = new ParakeetMel(128);
const SPK = 4;
async function runWindow(samples) {
  const { features, length } = mel.process(samples);
  const r = await parakeetEncode(ctx, enc, features, length);
  const preds = await sortformerHead(ctx, head, r.framesGpu, r.Tsub);
  console.log(`   win ${(samples.length / 16000).toFixed(0)}s → T=${r.Tsub} predsMax=${Math.max(...preds).toFixed(3)}`);
  return { preds, frames: preds.length / SPK };
}
async function diarize(samples, sr = 16000) {
  const WIN = 90 * sr,
    OVL = 15 * sr,
    hop = WIN - OVL;
  if (samples.length <= WIN) {
    const w = await runWindow(samples);
    return predsToSegments(w.preds, w.frames, samples.length / sr / w.frames);
  }
  const windows = [],
    ovlF = [];
  for (let s = 0; s < samples.length; s += hop) {
    const end = Math.min(s + WIN, samples.length);
    const w = await runWindow(samples.subarray(s, end));
    const fps = w.frames / ((end - s) / sr);
    windows.push(w);
    ovlF.push(Math.round((OVL / sr) * fps));
    if (end >= samples.length) break;
  }
  const m = mergeWindowPreds(windows, ovlF);
  return predsToSegments(m.preds, m.frames, samples.length / sr / m.frames);
}
const fmt = (segs) => segs.map((s) => `spk${s.speaker}: ${s.start.toFixed(1)}-${s.end.toFixed(1)}`).join("  ");
console.log("40s single-window:");
console.log(" ", fmt(await diarize(one)));
const t0 = performance.now();
const segs = await diarize(three);
console.log(`120s windowed (${(performance.now() - t0).toFixed(0)}ms):`);
console.log(" ", fmt(segs));
const spks = [...new Set(segs.map((s) => s.speaker))];
console.log(`distinct speakers: ${spks.length} ${spks.length === 2 ? "OK" : "CHECK"}`);
process.exit(spks.length === 2 ? 0 : 1);
