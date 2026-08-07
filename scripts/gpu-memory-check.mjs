// GPU memory gate: a 120s transcribe must RECYCLE its intermediates instead of
// allocating a fresh GPUBuffer per tensor. Counts createBuffer calls + bytes
// (ctx.memStats) across the shipping pipeline and asserts pooling holds:
// steady-state groups must hit the pool (reused >> created after warm).
//   node scripts/gpu-memory-check.mjs [/tmp/pk_120s.wav]
import { readFileSync } from "node:fs";
import { readWav } from "./lib/wav.mjs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { loadParakeetEncoder } from "../src/engines/asr-parakeet/raw-encoder.js";
import { loadWasmDecoder } from "../src/engines/asr-parakeet/raw-decoder-wasm.js";
import { ParakeetMel } from "../src/engines/asr-parakeet/parakeet-mel.js";
import { transcribeWindowed } from "../src/engines/asr-parakeet/pipeline.js";

const wav = readWav(process.argv[2] || "/tmp/pk_120s.wav");
const ctx = new GpuContext(await getDevice());
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

const run = () => transcribeWindowed(ctx, enc, dec, mel, projW, projB, wav, {});

// Run 1: pool cold — every intermediate is a fresh buffer.
ctx.memStatsStart();
const r1 = await run();
const cold = ctx.memStats();
// Run 2: pool warm — steady state should be near-zero fresh allocations.
ctx.memStatsStart();
const r2 = await run();
const warm = ctx.memStats();

const same = r1.ids.length === r2.ids.length && r1.ids.every((v, i) => v === r2.ids[i]);
console.log(`tokens: ${r1.ids.length} vs ${r2.ids.length} → ${same ? "IDENTICAL" : "DIVERGED"}`);
console.log(`cold: created ${cold.created} buffers (${(cold.createdBytes / 1e6).toFixed(0)} MB), reused ${cold.reused}`);
console.log(`warm: created ${warm.created} buffers (${(warm.createdBytes / 1e6).toFixed(0)} MB), reused ${warm.reused}`);
const coldRecycleOk = cold.reused > cold.created; // in-run group recycling
// Warm churn = re-creating what the budget evicted; assert it stays a
// fraction of cold (the pool still serves the hot majority).
const warmOk = warm.createdBytes < cold.createdBytes / 3;
const pool = ctx.poolInfo();
const budget = ctx.poolBudgetBytes ?? 1 << 30;
const budgetOk = pool.bytes <= budget;
console.log(`in-run recycling (cold reused > created): ${coldRecycleOk ? "OK" : "FAIL"}`);
console.log(`warm churn bytes < cold/3: ${warmOk ? "OK" : "FAIL"}`);
console.log(`pool retention ${(pool.bytes / 1e6).toFixed(0)} MB ≤ budget ${(budget / 1e6).toFixed(0)} MB: ${budgetOk ? "OK" : "FAIL"}`);
process.exit(same && coldRecycleOk && warmOk && budgetOk ? 0 : 1);
