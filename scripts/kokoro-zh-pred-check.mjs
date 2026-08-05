// Gate the JS Kokoro predictor on zh (v1.1) weights vs same-run ORT anchors.
//   node scripts/kokoro-zh-pred-check.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { makeKokoro, predictor } from "../src/gpu/kokoro-synth.js";

const Z = "/tmp/kokoro-zh";
const rd = (p) => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};
const ctx = await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url))));
const K = makeKokoro(ctx, rd(`${Z}/kw/weights.bin`), JSON.parse(readFileSync(`${Z}/kw/manifest.json`)), JSON.parse(readFileSync(`${Z}/kw/roles.json`)));
const style = rd(`${Z}/zh_style.bin`);
const _ib = readFileSync(`${Z}/zh_ids.bin`);
const ids = new Int32Array(_ib.buffer, _ib.byteOffset, _ib.byteLength / 4);
const dEnRaw = rd(`${Z}/zh_den.bin`);
const seq = dEnRaw.length / 512;

const out = await predictor(K, { data: dEnRaw, rows: seq, cols: 512 }, ids, style);
const refDur = rd(`${Z}/anc_clip.bin`);
const durOk = out.predDur.length === refDur.length && Array.from(out.predDur).every((v, i) => v === Math.round(refDur[i]));
console.log("T =", out.T, "predDur match ORT:", durOk);
const me = async (t, refPath, name) => {
  const o = t.data ? t.data : await ctx.download(t);
  const ref = rd(refPath);
  const m = Math.min(o.length, ref.length);
  let d = 0;
  for (let i = 0; i < m; i++) d = Math.max(d, Math.abs(o[i] - ref[i]));
  console.log(`  ${name}: [${t.rows},${t.cols}] (${o.length}) vs ref ${ref.length}  maxΔ ${d.toExponential(2)}`);
  return d;
};
let bad = 0;
bad += (await me(out.F0, `${Z}/anc_f0p.bin`, "F0")) > 2e-3;
bad += (await me(out.N, `${Z}/anc_np.bin`, "N")) > 2e-3;
bad += (await me(out.asr, `${Z}/anc_asr.bin`, "asr")) > 2e-3;
bad += (await me(out.xConcat, `${Z}/anc_xcat.bin`, "xConcat")) > 2e-3;
if (!durOk) bad++;
console.log(bad ? `\n${bad} STAGE(S) FAIL` : "\nZH PREDICTOR PARITY OK");
process.exit(bad ? 1 : 0);
