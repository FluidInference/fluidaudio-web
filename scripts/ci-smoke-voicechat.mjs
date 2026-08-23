// CI smoke: VoiceChat-11B STT on the WASM backend (no GPU) transcribes NVIDIA's
// sample_general.wav and must contain the reference transcript (mobius
// test_e2e_stt.py CoreML/torch parity: "Hello, do you know what color the sky
// is"). Weights are LOCAL (scripts/extract-voicechat-stt.py →
// public/models/voicechat-stt/, gitignored) — run the extractor first.
//   node scripts/ci-smoke-voicechat.mjs [wavPath]
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";
import { readWav, assert } from "./lib/ci.mjs";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { loadParakeetEncoder } from "../src/engines/asr-parakeet/raw-encoder.js";
import { createEncodeStream, encodeStreamPush, encodeStreamFlush, disposeEncodeStream } from "../src/engines/asr-parakeet/streaming-encoder.js";
import { JsPreprocessor } from "../src/engines/asr-nemotron/nemotron-mel.js";
import { StreamingMel } from "../src/engines/asr-nemotron/streaming-mel.js";
import { loadVoicechatDecoder, createVoicechatStream, voicechatDecodeCont } from "../src/engines/asr-voicechat/raw-decoder-voicechat.js";
import { VOICECHAT_CFG } from "../src/engines/asr-voicechat/config.js";

const W = fileURLToPath(new URL("../public/models/voicechat-stt/", import.meta.url));
const wavPath = process.argv[2] ?? `${homedir()}/Documents/models/voicechat-11b/Speech/examples/speechlm2/sample_audio/sample_general.wav`;
if (!existsSync(`${W}encoder-f16.bin`)) {
  console.error(`missing ${W}encoder-f16.bin — run scripts/extract-voicechat-stt.py first`);
  process.exit(1);
}

const ctx = await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url))));
const enc = loadParakeetEncoder(
  ctx,
  new Uint8Array(readFileSync(`${W}encoder-f16.bin`)),
  JSON.parse(readFileSync(`${W}encoder-f16.manifest.json`, "utf8")),
  VOICECHAT_CFG,
);
const decBinU8 = new Uint8Array(readFileSync(`${W}decoder-fp32.bin`));
const dec = loadVoicechatDecoder(
  new Float32Array(decBinU8.buffer, decBinU8.byteOffset, decBinU8.byteLength / 4),
  JSON.parse(readFileSync(`${W}decoder-fp32.manifest.json`, "utf8")),
);
const vocab = JSON.parse(readFileSync(`${W}vocab.json`, "utf8"));
const audio = readWav(wavPath);
console.log(`audio: ${(audio.length / 16000).toFixed(1)}s, encoder cfg attChunk ${VOICECHAT_CFG.attChunk} left ${VOICECHAT_CFG.attLeft}`);

const idsToText = (ids) =>
  ids
    .map((i) => vocab[i] ?? "")
    .filter((tk) => !tk.startsWith("<"))
    .join("")
    .replace(/▁/g, " ")
    .trim();

// Batch-as-stream, exactly the engine's transcribe() path: streaming NA mel
// (log guard 2^-24 — NeMo default, unlike Nemotron's 1e-10), cache-carrying
// causal encode with the joint enc projection riding each pass, continuous JS
// RNNT decode.
async function run() {
  const t0 = Date.now();
  const mel = new StreamingMel(128);
  mel.pre = new JsPreprocessor({ nMels: 128, logGuard: 2 ** -24 });
  const projW = ctx.upload(dec.encW.slice(), 1024, 640);
  const projB = ctx.upload(dec.encB.slice(), 1, 640);
  const encSt = createEncodeStream(ctx, enc, { proj: { w: projW, b: projB } });
  const st = createVoicechatStream(dec);
  const ids = [];
  let subT = 0;
  let melMs = 0,
    encMs = 0,
    decMs = 0;
  const consume = (out) => {
    if (!out) return;
    const td = Date.now();
    const n = out.length / 640;
    ids.push(...voicechatDecodeCont(dec, st, out, n).ids);
    subT += n;
    decMs += Date.now() - td;
  };
  for (let pos = 0; pos < audio.length; pos += 240 * 16000) {
    const tm = Date.now();
    const { data, count } = mel.push(audio.subarray(pos, pos + 240 * 16000));
    melMs += Date.now() - tm;
    const te = Date.now();
    if (data) consume(await encodeStreamPush(ctx, encSt, data, count, { maxChunk: 768 }));
    encMs += Date.now() - te;
  }
  const fl = mel.flush();
  const te = Date.now();
  if (fl.data) consume(await encodeStreamPush(ctx, encSt, fl.data, fl.count, { maxChunk: 768 }));
  consume(await encodeStreamFlush(ctx, encSt));
  encMs += Date.now() - te;
  disposeEncodeStream(ctx, encSt);
  const text = idsToText(ids);
  console.log(`voicechat (wasm backend): ${Date.now() - t0}ms (mel ${melMs} enc ${encMs} dec ${decMs}) → ${subT} frames,`, JSON.stringify(text));
  return text;
}

const text = await run();
const norm = (s) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
assert(norm(text).includes("do you know what color the sky is"), "transcript contains 'do you know what color the sky is'");
console.log("VOICECHAT SMOKE OK");
process.exit(0);
