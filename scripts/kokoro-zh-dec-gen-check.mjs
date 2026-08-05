// Gate zh decoder (xConcat→decode3) and generator (decode3+spec→waveform) vs
// same-run ORT anchors. Generator is deterministic GIVEN the spec.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { makeKokoro, decoder, generator } from "../src/gpu/kokoro-synth.js";

const Z = "/tmp/kokoro-zh";
const rd = (p) => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};
const ctx = await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url))));
const K = makeKokoro(ctx, rd(`${Z}/kw/weights.bin`), JSON.parse(readFileSync(`${Z}/kw/manifest.json`)), JSON.parse(readFileSync(`${Z}/kw/roles.json`)));
const style = rd(`${Z}/zh_style.bin`);

const T = 55;
const xConcat = { data: rd(`${Z}/anc_xcat.bin`), rows: 514, cols: T };
const asr = { data: rd(`${Z}/anc_asr.bin`), rows: 512, cols: T };
const F0 = { data: rd(`${Z}/anc_f0p.bin`), rows: 1, cols: 2 * T };
const N = { data: rd(`${Z}/anc_np.bin`), rows: 1, cols: 2 * T };

const d3T = await decoder(K, xConcat, asr, F0, N, style.slice(0, 128));
const d3 = await ctx.download(d3T);
const d3ref = rd(`${Z}/anc_dec3.bin`);
let dm = 0;
for (let i = 0; i < Math.min(d3.length, d3ref.length); i++) dm = Math.max(dm, Math.abs(d3[i] - d3ref[i]));
console.log(`decode3 [${d3T.rows},${d3T.cols}] vs ref ${d3ref.length}  maxΔ ${dm.toExponential(2)}`);

const specRaw = rd(`${Z}/anc_spec.bin`);
const Ts = specRaw.length / 22;
const wav = await generator(K, { data: d3ref, rows: 512, cols: 110 }, { data: specRaw, rows: 22, cols: Ts }, style.slice(0, 128));
const ref = rd(`${Z}/anc_wav.bin`);
const m = Math.min(wav.length, ref.length);
let dot = 0,
  a = 0,
  b = 0,
  mx = 0;
for (let i = 0; i < m; i++) {
  dot += wav[i] * ref[i];
  a += wav[i] ** 2;
  b += ref[i] ** 2;
  mx = Math.max(mx, Math.abs(wav[i] - ref[i]));
}
console.log(`generator: len my ${wav.length} ref ${ref.length}  corr ${(dot / Math.sqrt(a * b)).toFixed(4)}  maxΔ ${mx.toExponential(2)}`);
process.exit(0);
