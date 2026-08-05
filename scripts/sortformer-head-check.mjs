// Gate the sortformer head (post-batching) vs ORT ref preds, on BOTH backends.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { loadSortformerHead, sortformerHead } from "../src/engines/diarization-sortformer/raw-sortformer-head.js";
const rd = (p) => { const u = Uint8Array.from(readFileSync(p)); return new Float32Array(u.buffer, u.byteOffset, u.byteLength / 4); };
const encOut = rd("/tmp/sf/ref-encout.bin");
const Tsub = encOut.length / 512;
const refPreds = rd("/tmp/sf/ref-preds.bin");
const man = JSON.parse(readFileSync("/tmp/sf-raw/head/manifest.json"));
const bin = rd("/tmp/sf-raw/head/weights.bin");
async function run(ctx, name) {
  const head = loadSortformerHead(ctx, bin, man);
  const frames = ctx.upload(encOut.slice(), Tsub, 512);
  const preds = await sortformerHead(ctx, head, frames, Tsub);
  const p = preds.data ?? preds; // returns Float32Array or tensor per impl
  const arr = p instanceof Float32Array ? p : await ctx.download(preds);
  let md = 0; const m = Math.min(arr.length, refPreds.length);
  for (let i = 0; i < m; i++) md = Math.max(md, Math.abs(arr[i] - refPreds[i]));
  console.log(`${name}: preds (${arr.length} vs ref ${refPreds.length}) maxΔ ${md.toExponential(2)} ${md < 1e-3 ? "OK" : "FAIL"}`);
}
await run(new GpuContext(await getDevice()), "WebGPU");
await run(await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url)))), "WASM  ");
process.exit(0);
