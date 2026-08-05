// Gate the JS Kokoro predictor (kokoro-synth.js predictor()) vs ORT anchors (ref run).
//   node scripts/kokoro-pred-check.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { makeKokoro, predictor } from "../src/gpu/kokoro-synth.js";

const KW = "/tmp/kokoro/kw",
  K0 = "/tmp/kokoro";
const rd = (p) => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};
const ctx = await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url))));
const K = makeKokoro(ctx, rd(`${KW}/weights.bin`), JSON.parse(readFileSync(`${KW}/manifest.json`)), JSON.parse(readFileSync(`${KW}/roles.json`)));
const style = rd(`${K0}/ref_style.bin`);
const _ib = readFileSync(`${K0}/ref_ids.bin`);
const ids = new Int32Array(_ib.buffer, _ib.byteOffset, _ib.byteLength / 4);
const dEnRaw = rd(`${K0}/anc__encoder_bert_encoder_Add_output_0.bin`);
const seq = dEnRaw.length / 512;
const dEn = { data: dEnRaw, rows: seq, cols: 512 };

const out = await predictor(K, dEn, ids, style);
const me = async (t, refPath, name) => {
  const o = t.data ? t.data : await ctx.download(t);
  const ref = rd(refPath);
  const m = Math.min(o.length, ref.length);
  let d = 0;
  for (let i = 0; i < m; i++) d = Math.max(d, Math.abs(o[i] - ref[i]));
  console.log(`  ${name}: [${t.rows},${t.cols}] (${o.length}) vs ref ${ref.length}  maxΔ ${d.toExponential(2)}`);
  return d;
};
console.log("T =", out.T, "predDur", Array.from(out.predDur));
let bad = 0;
bad += (await me(out.F0, `${K0}/anc__encoder_F0_proj_Conv_output_0.bin`, "F0")) > 2e-3;
bad += (await me(out.N, `${K0}/anc__encoder_N_proj_Conv_output_0.bin`, "N")) > 2e-3;
bad += (await me(out.asr, `${K0}/anc__encoder_MatMul_1_output_0.bin`, "asr")) > 2e-3;
bad += (await me(out.xConcat, `${K0}/anc__decoder_decoder_Concat_output_0.bin`, "xConcat")) > 2e-3;
console.log(bad ? `\n${bad} PREDICTOR STAGE(S) FAIL` : "\nPREDICTOR PARITY OK");
process.exit(bad ? 1 : 0);
