// Gate + timing for the shipping windowed pipeline (src/engines/asr-parakeet/
// pipeline.js): serial vs pipelined must be TOKEN-IDENTICAL (same kernels, only
// scheduling differs), and the pipelined wall should approach max(enc, mel+dec).
//   node scripts/parakeet-pipeline-check.mjs [/tmp/pk_120s.wav]
import { readFileSync } from "node:fs";
import { readWav } from "./lib/wav.mjs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadParakeetEncoder } from "../src/engines/asr-parakeet/raw-encoder.js";
import { loadWasmDecoder } from "../src/engines/asr-parakeet/raw-decoder-wasm.js";
import { ParakeetMel } from "../src/engines/asr-parakeet/parakeet-mel.js";
import { transcribeWindowed } from "../src/engines/asr-parakeet/pipeline.js";

const wav = readWav(process.argv[2] || "/tmp/pk_120s.wav");
const durS = wav.length / 16000;
console.log(`audio ${durS.toFixed(1)}s`);
globalThis.__f16cbk8 = !!process.env.F16CBK8;
globalThis.__sgGemm = !!process.env.SGGEMM; // opt-in subgroup GEMM (see compute.js)
const ctx = new GpuContext(await getDevice());
await ctx.probeSubgroups?.(); // enable subgroup GEMM where hardware qualifies
const rdU8 = (p) => Uint8Array.from(readFileSync(p));
const enc = loadParakeetEncoder(ctx, rdU8("/tmp/pk-raw/enc-int8/weights.bin"), JSON.parse(readFileSync("/tmp/pk-raw/enc-int8/manifest.json")));
const decMan = JSON.parse(readFileSync("/tmp/pk-raw/dec/manifest.json"));
const decU8 = rdU8("/tmp/pk-raw/dec/weights.bin");
const decF32 = new Float32Array(decU8.buffer, decU8.byteOffset, decU8.byteLength / 4);
const dec = await loadWasmDecoder(readFileSync("src/engines/asr-parakeet/parakeet-decoder.wasm"), decF32, decMan);
const g = (k) => decF32.subarray(decMan[k].offset, decMan[k].offset + decMan[k].len);
const projW = ctx.upload(g("encW").slice(), 1024, 640);
const projB = ctx.upload(g("encB").slice(), 1, 640);
const mel = new ParakeetMel(128);

// Optional GPU decoder (GPUDEC=1): whole TDT loop on the GPU, no frame readback.
let gpuDecoder = null;
if (process.env.GPUDEC) {
  const { loadGpuDecoder, gpuDecodeBatch } = await import("../src/engines/asr-parakeet/gpu-decoder.js");
  gpuDecoder = { gdec: loadGpuDecoder(ctx, decF32, decMan), gpuDecodeBatch };
  console.log("gpu decoder: on");
}

// Optional decoder worker pool (same worker file the browser uses), node adapter.
let pool = null;
const POOL = Number(process.env.POOL || 0);
if (POOL > 1) {
  const { Worker } = await import("node:worker_threads");
  const { createDecodePool, nodeWorkerShim, initDecodeWorker } = await import("../src/engines/asr-parakeet/decode-pool.js");
  const wasmBytes = readFileSync("src/engines/asr-parakeet/parakeet-decoder.wasm");
  const decBuf = decU8.buffer.slice(decU8.byteOffset, decU8.byteOffset + decU8.byteLength);
  const workers = await Promise.all(
    Array.from({ length: POOL }, async () => {
      const w = new Worker(new URL("../src/engines/asr-parakeet/decoder-worker.js", import.meta.url), { type: "module" });
      await initDecodeWorker(
        (m) => w.postMessage(m),
        (ok, err) => {
          w.once("message", ok);
          w.once("error", err);
        },
        { wasmBytes: wasmBytes.buffer.slice(wasmBytes.byteOffset, wasmBytes.byteOffset + wasmBytes.byteLength), decBuf, man: decMan },
      );
      return nodeWorkerShim(w);
    }),
  );
  pool = createDecodePool(workers);
  console.log(`decode pool: ${POOL} workers`);
}

const run = async (pipelined, usePool = true) => {
  const t = performance.now();
  const { ids, stats } = await transcribeWindowed(ctx, enc, dec, mel, projW, projB, wav, {
    pipelined,
    wb: Number(process.env.WB || 6),
    decodePool: usePool ? pool : null,
    gpuDecoder,
  });
  return { ids, stats, ms: performance.now() - t };
};

await run(false, false); // warm (shader compile, wasm decoder init)
const serial = await run(false, false);
const piped = await run(true);
const piped2 = await run(true);

const same =
  serial.ids.length === piped.ids.length &&
  serial.ids.every((v, i) => v === piped.ids[i]) &&
  piped2.ids.length === piped.ids.length &&
  piped2.ids.every((v, i) => v === piped.ids[i]);
console.log(`tokens: serial ${serial.ids.length}, pipelined ${piped.ids.length} → ${same ? "IDENTICAL" : "DIVERGED"}`);
const best = Math.min(piped.ms, piped2.ms);
console.log(`serial    ${serial.ms.toFixed(0)}ms  RTFx ${(durS / (serial.ms / 1000)).toFixed(1)}`);
console.log(`pipelined ${best.toFixed(0)}ms  RTFx ${(durS / (best / 1000)).toFixed(1)}  (${(serial.ms / best).toFixed(2)}× vs serial)`);
console.log(`stages: mel ${piped.stats.melMs}ms · encWait ${piped.stats.encWaitMs}ms · decode ${piped.stats.decodeMs}ms`);
pool?.terminate?.();
if (!same) process.exit(1);
process.exit(0);
