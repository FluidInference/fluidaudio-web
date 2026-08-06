// KV-cached vs full-prefix Whisper decode: token-sequence equality + timing.
//   node scripts/whisper-kv-check.mjs
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadWhisperDecoder, whisperCrossKV, whisperDecodeStep, whisperDecodeInit, whisperDecodeNext } from "../src/engines/asr-whisper/raw-whisper-decoder.js";

const ctx = new GpuContext(await getDevice());
const rd = (p) => {
  const u = Uint8Array.from(readFileSync(p));
  return new Float32Array(u.buffer, u.byteOffset, u.byteLength / 4);
};
const dec = loadWhisperDecoder(ctx, rd("/tmp/whisper/dec/weights.bin"), JSON.parse(readFileSync("/tmp/whisper/dec/manifest.json")));
const encG = ctx.upload(rd("/tmp/whisper/ref_enc.bin"), 1500, 512);
const kv = whisperCrossKV(ctx, dec, encG);
const PREFIX = [50258, 50259, 50359, 50363],
  EOT = 50257,
  MAX_NEW = 64;

// old path
let t0 = performance.now();
const oldToks = [...PREFIX];
for (let step = 0; step < MAX_NEW; step++) {
  const logits = await whisperDecodeStep(ctx, dec, kv, oldToks);
  let mi = 0,
    mv = -Infinity;
  for (let i = 0; i < logits.length; i++)
    if (logits[i] > mv) {
      mv = logits[i];
      mi = i;
    }
  if (mi === EOT) break;
  oldToks.push(mi);
}
const oldMs = performance.now() - t0;

// cached path
t0 = performance.now();
const st = whisperDecodeInit(ctx, dec);
const newToks = [...PREFIX];
let logits = null;
for (let i = 0; i < PREFIX.length; i++) logits = await whisperDecodeNext(ctx, dec, kv, st, PREFIX[i]);
for (let step = 0; step < MAX_NEW; step++) {
  let mi = 0,
    mv = -Infinity;
  for (let i = 0; i < logits.length; i++)
    if (logits[i] > mv) {
      mv = logits[i];
      mi = i;
    }
  if (mi === EOT) break;
  newToks.push(mi);
  logits = await whisperDecodeNext(ctx, dec, kv, st, mi);
}
const newMs = performance.now() - t0;

const same = oldToks.length === newToks.length && oldToks.every((v, i) => v === newToks[i]);
console.log(`old: ${oldToks.length - 4} tokens in ${oldMs.toFixed(0)}ms   cached: ${newToks.length - 4} tokens in ${newMs.toFixed(0)}ms`);
console.log(`sequences identical: ${same}`);
console.log("tokens:", JSON.stringify(newToks));
if (!same) {
  console.log("old:", oldToks.slice(4, 20));
  console.log("new:", newToks.slice(4, 20));
}
process.exit(same ? 0 : 1);
