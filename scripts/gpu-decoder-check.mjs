// Gate for the GPU-resident TDT decoder: per-window token identity vs the
// WASM reference (wasmDecodeProj) on real encoder output, plus timing.
//   node scripts/gpu-decoder-check.mjs [/tmp/pk_120s.wav]
import { readFileSync } from "node:fs";
import { readWav } from "./lib/wav.mjs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadParakeetEncoder, parakeetEncodeBatch } from "../src/engines/asr-parakeet/raw-encoder.js";
import { loadWasmDecoder, wasmDecodeProj } from "../src/engines/asr-parakeet/raw-decoder-wasm.js";
import { ParakeetMel } from "../src/engines/asr-parakeet/parakeet-mel.js";
import { loadGpuDecoder, gpuDecodeBatch } from "../src/engines/asr-parakeet/gpu-decoder.js";

const wav = readWav(process.argv[2] || "/tmp/pk_120s.wav");
const ctx = new GpuContext(await getDevice());
await ctx.probeSubgroups?.();
const rdU8 = (p) => Uint8Array.from(readFileSync(p));
const enc = loadParakeetEncoder(ctx, rdU8("/tmp/pk-raw/enc-int8/weights.bin"), JSON.parse(readFileSync("/tmp/pk-raw/enc-int8/manifest.json")));
const decMan = JSON.parse(readFileSync("/tmp/pk-raw/dec/manifest.json"));
const decU8 = rdU8("/tmp/pk-raw/dec/weights.bin");
const decF32 = new Float32Array(decU8.buffer, decU8.byteOffset, decU8.byteLength / 4);
const dec = await loadWasmDecoder(readFileSync("src/engines/asr-parakeet/parakeet-decoder.wasm"), decF32, decMan, { int8: false }); // fp32 reference (GPU path is fp32)
const g = (k) => decF32.subarray(decMan[k].offset, decMan[k].offset + decMan[k].len);
const projW = ctx.upload(g("encW").slice(), 1024, 640);
const projB = ctx.upload(g("encB").slice(), 1, 640);
const gdec = loadGpuDecoder(ctx, decF32, decMan);
const mel = new ParakeetMel(128);

// Encode the first 6 full windows as one batch, project on GPU.
const WIN = 15 * 16000,
  OVL = 2 * 16000,
  hop = WIN - OVL;
const mels = [];
for (let s = 0; s + WIN <= wav.length && mels.length < 6; s += hop) mels.push(mel.process(wav.subarray(s, s + WIN)).features);
const r = await parakeetEncodeBatch(ctx, enc, mels);
const proj = ctx.matmul(r.framesGpu, projW, { bias: projB }); // [W*T, 640]
const W = mels.length,
  T = r.Tsub;
const frames = await ctx.download(proj);

// WASM reference per window.
const ref = [];
let t0 = performance.now();
for (let w = 0; w < W; w++) ref.push(wasmDecodeProj(dec, frames.subarray(w * T * 640, (w + 1) * T * 640), T));
const wasmMs = performance.now() - t0;

// GPU decode, all windows in one dispatch.
t0 = performance.now();
const gpu = await gpuDecodeBatch(ctx, gdec, proj, W, T);
const gpuMs = performance.now() - t0;

let fail = 0;
for (let w = 0; w < W; w++) {
  const a = ref[w],
    b = gpu[w];
  const same = a.ids.length === b.ids.length && a.ids.every((v, i) => v === b.ids[i]) && a.idFrames.every((v, i) => v === b.idFrames[i]);
  if (!same) {
    fail++;
    console.log(`✗ window ${w}: wasm ${a.ids.length} tokens vs gpu ${b.ids.length}`);
    console.log("  wasm:", a.ids.slice(0, 12).join(","), "| gpu:", b.ids.slice(0, 12).join(","));
  } else console.log(`✓ window ${w}: ${a.ids.length} tokens identical (ids + frames)`);
}
console.log(`wasm ${wasmMs.toFixed(0)}ms (serial ${W}) | gpu ${gpuMs.toFixed(0)}ms (one dispatch, incl. readback)`);
console.log(fail === 0 ? "GPU DECODER OK" : `GPU DECODER ${fail} WINDOW(S) DIVERGED`);
process.exit(fail === 0 ? 0 : 1);
