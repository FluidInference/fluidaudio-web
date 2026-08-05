// CI smoke: Parakeet TDT v3 on the WASM backend (no GPU) transcribes the bundled
// 12s sample and must contain the known transcript words. Hermetic given HF.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { hfGet, hfJson, hfText, readWav, assert } from "./lib/ci.mjs";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { loadParakeetEncoder, parakeetEncodeBatch } from "../src/engines/asr-parakeet/raw-encoder.js";
import { loadWasmDecoder, wasmDecode } from "../src/engines/asr-parakeet/raw-decoder-wasm.js";
import { ParakeetTokenizer } from "../src/engines/asr-parakeet/tokenizer.js";
import { ParakeetMel } from "../src/engines/asr-parakeet/parakeet-mel.js";

const W = "FluidInference/fluidaudio-web";
const ctx = await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url))));
const enc = loadParakeetEncoder(ctx, await hfGet(W, "parakeet/encoder-int8.bin"), await hfJson(W, "parakeet/encoder-int8.manifest.json"));
const decBinU8 = await hfGet(W, "parakeet/decoder-fp32.bin");
const dec = await loadWasmDecoder(
  readFileSync(fileURLToPath(new URL("../src/engines/asr-parakeet/parakeet-decoder.wasm", import.meta.url))),
  new Float32Array(decBinU8.buffer, decBinU8.byteOffset, decBinU8.byteLength / 4),
  await hfJson(W, "parakeet/decoder-fp32.manifest.json"),
);
const tokenizer = ParakeetTokenizer.fromVocabText(await hfText("ysdede/parakeet-tdt-0.6b-v3-onnx", "vocab.txt"));
const audio = readWav(fileURLToPath(new URL("../public/sample.wav", import.meta.url)));
const t0 = Date.now();
const { features, length } = new ParakeetMel(128).process(audio);
const r = await parakeetEncodeBatch(ctx, enc, [features]);
const frames = await ctx.download(r.framesGpu);
const { ids } = wasmDecode(dec, frames, r.Tsub);
const text = tokenizer.decode(ids);
console.log(`parakeet (wasm backend): ${Date.now() - t0}ms →`, JSON.stringify(text));
assert(/menace/i.test(text), "transcript contains 'menace'");
assert(/suffrage/i.test(text), "transcript contains 'suffrage'");
assert(/goodwin/i.test(text), "transcript contains 'Goodwin'");
console.log("PARAKEET SMOKE OK");
process.exit(0);
