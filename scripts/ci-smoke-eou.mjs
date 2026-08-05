// CI smoke: Parakeet EOU 120M on the WASM backend transcribes the bundled sample.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { hfGet, hfJson, hfText, readWav, assert } from "./lib/ci.mjs";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { loadParakeetEncoder, parakeetEncode } from "../src/engines/asr-parakeet/raw-encoder.js";
import { loadEouDecoder, eouDecode } from "../src/engines/asr-parakeet/raw-decoder-eou.js";
import { makeEouTokenizer } from "../src/engines/eou-parakeet/eou-decode.js";
import { JsPreprocessor } from "../src/engines/asr-nemotron/nemotron-mel.js";

const W = "FluidInference/fluidaudio-web";
const EOU_CFG = { melBins: 128, subPad: { t: 2, b: 1, l: 2, r: 1 }, convCausal: true, attChunk: 2, attLeft: 70 };
const ctx = await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url))));
const enc = loadParakeetEncoder(ctx, await hfGet(W, "eou/encoder-fp16.bin"), await hfJson(W, "eou/encoder-fp16.manifest.json"), EOU_CFG);
const decBinU8 = await hfGet(W, "eou/decoder-fp32.bin");
const dec = loadEouDecoder(new Float32Array(decBinU8.buffer, decBinU8.byteOffset, decBinU8.byteLength / 4), await hfJson(W, "eou/decoder-fp32.manifest.json"));
const tokenizer = makeEouTokenizer(await hfText("ysdede/parakeet-realtime-eou-120m-v1-onnx", "vocab.txt"));
const audio = readWav(fileURLToPath(new URL("../public/sample.wav", import.meta.url)));
const t0 = Date.now();
const { features, length } = new JsPreprocessor({ nMels: 128 }).process(audio);
const r = await parakeetEncode(ctx, enc, features, length);
const frames = await ctx.download(r.framesGpu);
const { ids } = eouDecode(dec, frames, r.Tsub);
const text = tokenizer.decode(ids);
console.log(`eou (wasm backend): ${Date.now() - t0}ms →`, JSON.stringify(text));
assert(/suffrage/i.test(text), "transcript contains 'suffrage'");
assert(/classes/i.test(text), "transcript contains 'classes'");
console.log("EOU SMOKE OK");
process.exit(0);
