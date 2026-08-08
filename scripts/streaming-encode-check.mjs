// True-streaming-encode gate (docs/STREAMING.md): the offline chunked-causal
// mask means frame i never sees past [chunkStart−left, chunkStart+1], so a
// cache-carrying streaming encode computes the SAME function. Assert it:
//   1. StreamingMel(chunks) == JsPreprocessor.process(full)  (exact)
//   2. streaming encoder frames ≈ offline frames (fp16-kernel tolerance),
//      at BOTH fine (maxChunk=2, every-boundary) and coarse granularity
//   3. streamed decode (state-carried) tokens == offline decode tokens
//   node scripts/streaming-encode-check.mjs
import { fileURLToPath } from "node:url";
import { hfGet, hfJson, hfText, readWav, assert } from "./lib/ci.mjs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadParakeetEncoder, parakeetEncode } from "../src/engines/asr-parakeet/raw-encoder.js";
import { createEncodeStream, encodeStreamPush, encodeStreamFlush, disposeEncodeStream } from "../src/engines/asr-parakeet/streaming-encoder.js";
import { loadEouDecoder, eouDecode, createEouStream, eouDecodeCont } from "../src/engines/asr-parakeet/raw-decoder-eou.js";
import { makeEouTokenizer } from "../src/engines/eou-parakeet/eou-decode.js";
import { JsPreprocessor } from "../src/engines/asr-nemotron/nemotron-mel.js";
import { StreamingMel } from "../src/engines/asr-nemotron/streaming-mel.js";
import { EOU_CFG } from "../src/engines/eou-parakeet/config.js";

const W = "FluidInference/fluidaudio-web";
const ENC_D = 512;
const audio = readWav(fileURLToPath(new URL("../public/sample.wav", import.meta.url)));

// ── 1. mel parity ───────────────────────────────────────────────────────────
const off = new JsPreprocessor({ nMels: 128 }).process(audio);
const offMel = off.features; // mel-major, row stride = array frames (floor(N/160)+1)
// The encoder derives its frame count from the ARRAY (process().length is one
// less; the trailing right-padded frame IS consumed offline) — compare on that.
const T = offMel.length / 128;
assert(Number.isInteger(T), "offline mel array is [128 × T]");

// Awkward chunk sizes on purpose: sub-hop, sub-window, seconds.
const CHUNKS = [7, 153, 160, 401, 1600, 16000, 24000];
function streamMel(samples) {
  const sm = new StreamingMel(128);
  const cols = []; // per-frame Float32Array(128)
  let pos = 0,
    ci = 0;
  const take = (r) => {
    for (let t = 0; t < r.count; t++) {
      const col = new Float32Array(128);
      for (let c = 0; c < 128; c++) col[c] = r.data[c * r.count + t];
      cols.push(col);
    }
  };
  while (pos < samples.length) {
    const len = Math.min(CHUNKS[ci++ % CHUNKS.length], samples.length - pos);
    take(sm.push(samples.subarray(pos, pos + len)));
    pos += len;
  }
  take(sm.flush());
  return cols;
}
const stCols = streamMel(audio);
assert(stCols.length === T, `mel frame count: streaming ${stCols.length} vs offline ${T}`);
let melMax = 0;
for (let t = 0; t < T; t++) for (let c = 0; c < 128; c++) melMax = Math.max(melMax, Math.abs(stCols[t][c] - offMel[c * T + t]));
console.log(`mel parity: ${T} frames, maxΔ ${melMax.toExponential(2)}`);
assert(melMax < 1e-6, `streaming mel exact parity (got ${melMax})`);

// ── 2+3. encoder + decoder parity ───────────────────────────────────────────
const ctx = new GpuContext(await getDevice());
const enc = loadParakeetEncoder(ctx, await hfGet(W, "eou/encoder-fp16.bin"), await hfJson(W, "eou/encoder-fp16.manifest.json"), EOU_CFG);
const decBinU8 = await hfGet(W, "eou/decoder-fp32.bin");
const dec = loadEouDecoder(new Float32Array(decBinU8.buffer, decBinU8.byteOffset, decBinU8.byteLength / 4), await hfJson(W, "eou/decoder-fp32.manifest.json"));
const tokenizer = makeEouTokenizer(await hfText("ysdede/parakeet-realtime-eou-120m-v1-onnx", "vocab.txt"));

const r = await parakeetEncode(ctx, enc, offMel, T);
const offFrames = await ctx.download(r.framesGpu);
const offIds = eouDecode(dec, offFrames, r.Tsub).ids;
console.log(`offline: Tsub ${r.Tsub}, ${offIds.length} tokens`);
ctx.trimPool?.();

async function runStreaming(maxChunk, label) {
  const st = createEncodeStream(ctx, enc);
  const ds = createEouStream(dec);
  const parts = [];
  const ids = [];
  let subT = 0;
  // Feed mel in irregular slices to cross chunk boundaries every way.
  const SLICES = [3, 16, 2, 47, 128, 9, 200];
  let pos = 0,
    si = 0;
  const feed = async (data, count) => {
    const out = await encodeStreamPush(ctx, st, data, count, { maxChunk });
    if (out) {
      parts.push(out);
      const n = out.length / ENC_D;
      ids.push(...eouDecodeCont(dec, ds, out, n, subT).ids);
      subT += n;
    }
  };
  while (pos < T) {
    const cnt = Math.min(SLICES[si++ % SLICES.length], T - pos);
    // reassemble a mel-major slab [128 × cnt] from the streamed columns
    const slab = new Float32Array(128 * cnt);
    for (let t = 0; t < cnt; t++) for (let c = 0; c < 128; c++) slab[c * cnt + t] = stCols[pos + t][c];
    await feed(slab, cnt);
    pos += cnt;
  }
  const tail = await encodeStreamFlush(ctx, st);
  if (tail) {
    parts.push(tail);
    ids.push(...eouDecodeCont(dec, ds, tail, tail.length / ENC_D, subT).ids);
    subT += tail.length / ENC_D;
  }
  disposeEncodeStream(ctx, st);
  ctx.trimPool?.();

  assert(subT === r.Tsub, `${label}: frame count ${subT} vs offline ${r.Tsub}`);
  let maxD = 0;
  let off2 = 0;
  for (const p of parts) {
    for (let i = 0; i < p.length; i++) maxD = Math.max(maxD, Math.abs(p[i] - offFrames[off2 + i]));
    off2 += p.length;
  }
  const tokOk = ids.length === offIds.length && ids.every((v, i) => v === offIds[i]);
  console.log(`${label}: frames maxΔ ${maxD.toExponential(2)}, tokens ${ids.length} vs ${offIds.length} → ${tokOk ? "IDENTICAL" : "DIVERGED"}`);
  assert(maxD < 5e-2, `${label}: encoder frame parity (fp16 tolerance), got ${maxD}`);
  assert(tokOk, `${label}: decoded tokens identical`);
  return ids;
}

const idsFine = await runStreaming(2, "streaming maxChunk=2 (160ms cadence)");
await runStreaming(64, "streaming maxChunk=64");
console.log("text:", JSON.stringify(tokenizer.decode(idsFine)));
assert(/suffrage/i.test(tokenizer.decode(idsFine)), "transcript sanity");
console.log("STREAMING ENCODE GATE OK");
process.exit(0);
