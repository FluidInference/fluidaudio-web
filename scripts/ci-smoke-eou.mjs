// CI smoke: Parakeet EOU 120M on the WASM backend transcribes the bundled sample.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { hfGet, hfJson, hfText, readWav, assert } from "./lib/ci.mjs";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { loadParakeetEncoder, parakeetEncode } from "../src/engines/asr-parakeet/raw-encoder.js";
import {
  loadEouDecoder,
  eouDecode,
  createEouStream,
  eouDecodeCont,
  loadEouWasmDecoder,
  eouWasmDecodeCont,
  eouWasmReset,
} from "../src/engines/asr-parakeet/raw-decoder-eou.js";
import { createEncodeStream, encodeStreamPush, encodeStreamFlush, disposeEncodeStream } from "../src/engines/asr-parakeet/streaming-encoder.js";
import { StreamingMel } from "../src/engines/asr-nemotron/streaming-mel.js";
import { makeEouTokenizer } from "../src/engines/eou-parakeet/eou-decode.js";
import { JsPreprocessor } from "../src/engines/asr-nemotron/nemotron-mel.js";

const W = "FluidInference/fluidaudio-web";
import { EOU_CFG } from "../src/engines/eou-parakeet/config.js";
const ctx = await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url))));
const enc = loadParakeetEncoder(ctx, await hfGet(W, "eou/encoder-fp16.bin"), await hfJson(W, "eou/encoder-fp16.manifest.json"), EOU_CFG);
const decBinU8 = await hfGet(W, "eou/decoder-fp32.bin");
const decF32 = new Float32Array(decBinU8.buffer, decBinU8.byteOffset, decBinU8.byteLength / 4);
const decMan = await hfJson(W, "eou/decoder-fp32.manifest.json");
const dec = loadEouDecoder(decF32, decMan);
const tokenizer = makeEouTokenizer(await hfText("ysdede/parakeet-realtime-eou-120m-v1-onnx", "vocab.txt"));
const audio = readWav(fileURLToPath(new URL("../public/sample.wav", import.meta.url)));
const t0 = Date.now();
const { features, length } = new JsPreprocessor({ nMels: 128 }).process(audio);
const r = await parakeetEncode(ctx, enc, features, length);
const frames = await ctx.download(r.framesGpu);
const { ids } = eouDecode(dec, frames, r.Tsub);
const text = tokenizer.decode(ids);

// wasm-SIMD decoder (the engine's shipping path): pre-projected frames, tokens
// must match the JS reference decoder exactly.
const wdec = await loadEouWasmDecoder(
  readFileSync(fileURLToPath(new URL("../src/engines/asr-parakeet/parakeet-decoder.wasm", import.meta.url))),
  decF32,
  decMan,
);
const projW = ctx.upload(dec.encW.slice(), 512, 640);
const projB = ctx.upload(dec.encB.slice(), 1, 640);
const projFrames = await ctx.download(ctx.matmul(r.framesGpu, projW, { bias: projB }));
eouWasmReset(wdec);
const wIds = eouWasmDecodeCont(wdec, projFrames, r.Tsub).ids;
// NOTE smoke, not proof: rust uses poly expf/tanhf and blocked FP accumulation
// vs JS Math.* — a near-tie argmax on OTHER audio can legitimately differ.
assert(wIds.length === ids.length && wIds.every((v, i) => v === ids[i]), "wasm decoder tokens == JS decoder tokens (this sample)");
console.log(`eou (wasm backend): ${Date.now() - t0}ms →`, JSON.stringify(text));
assert(/suffrage/i.test(text), "transcript contains 'suffrage'");
assert(/classes/i.test(text), "transcript contains 'classes'");

// True-streaming path on the SAME (wasm) backend: cache-carrying chunked
// encode + state-carried decode must reproduce the batch tokens exactly
// (bit-exact on GPU per scripts/streaming-encode-check.mjs; this leg keeps the
// CPU twins honest in CI).
const t1 = Date.now();
const sm = new StreamingMel(128);
const stEnc = createEncodeStream(ctx, enc);
const stDec = createEouStream(dec);
const sIds = [];
let subT = 0;
const consume = (out) => {
  if (!out) return;
  const n = out.length / 512;
  sIds.push(...eouDecodeCont(dec, stDec, out, n, subT).ids);
  subT += n;
};
for (let pos = 0; pos < audio.length; pos += 16000) {
  const { data, count } = sm.push(audio.subarray(pos, pos + 16000));
  if (data) consume(await encodeStreamPush(ctx, stEnc, data, count, { maxChunk: 64 }));
}
const fl = sm.flush();
if (fl.data) consume(await encodeStreamPush(ctx, stEnc, fl.data, fl.count, { maxChunk: 64 }));
consume(await encodeStreamFlush(ctx, stEnc));
disposeEncodeStream(ctx, stEnc);
console.log(`eou streaming (wasm backend): ${Date.now() - t1}ms → ${sIds.length} tokens`);
assert(sIds.length === ids.length && sIds.every((v, i) => v === ids[i]), "streaming tokens == batch tokens");
console.log("EOU SMOKE OK");
process.exit(0);
