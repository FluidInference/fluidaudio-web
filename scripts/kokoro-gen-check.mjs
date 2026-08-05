// Gate the JS Kokoro generator (src/gpu/kokoro-synth.js) against ORT's waveform,
// feeding ORT's exact decode3 + source spec (mirrors /tmp/kfinal.py). WasmContext.
//   node scripts/kokoro-gen-check.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { makeKokoro, generator } from "../src/gpu/kokoro-synth.js";

const KW = "/tmp/kokoro/kw";
const rd = (p) => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};
const ctx = await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url))));
const weights = (() => {
  const b = readFileSync(`${KW}/weights.bin`);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
})();
const manifest = JSON.parse(readFileSync(`${KW}/manifest.json`, "utf8"));
const roles = JSON.parse(readFileSync(`${KW}/roles.json`, "utf8"));
const K = makeKokoro(ctx, weights, manifest, roles);

const decode3raw = rd("/tmp/kokoro/real_decode3.bin");
const c3 = rd("/tmp/kokoro/c3.bin");
const style = rd("/tmp/kokoro/hello_style.bin").slice(0, 128);
const Tdec = decode3raw.length / 512,
  Tspec = c3.length / 22;
console.log(`decode3 [512,${Tdec}]  sourceSpec [22,${Tspec}]`);

const wav = await generator(K, { data: decode3raw, rows: 512, cols: Tdec }, { data: c3, rows: 22, cols: Tspec }, style);

const ref = rd("/tmp/kokoro/real_wav.bin");
const m = Math.min(wav.length, ref.length);
let maxD = 0,
  se = 0,
  sr = 0,
  dot = 0,
  na = 0,
  nb = 0;
for (let i = 0; i < m; i++) {
  maxD = Math.max(maxD, Math.abs(wav[i] - ref[i]));
  se += (wav[i] - ref[i]) ** 2;
  dot += wav[i] * ref[i];
  na += wav[i] ** 2;
  nb += ref[i] ** 2;
}
const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
console.log(`len my ${wav.length} ref ${ref.length}`);
console.log(
  `corr ${(dot / Math.sqrt(na * nb)).toFixed(5)}  maxΔ ${maxD.toExponential(2)}  rms my ${rms(wav.subarray(0, m)).toFixed(4)} ref ${rms(ref.subarray(0, m)).toFixed(4)}`,
);
process.exit(0);
