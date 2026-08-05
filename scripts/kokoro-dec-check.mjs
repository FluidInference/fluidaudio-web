// Gate the JS Kokoro decoder body vs ORT anchors (mirrors /tmp/kdec.py inputs).
//   node scripts/kokoro-dec-check.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { makeKokoro, decoder } from "../src/gpu/kokoro-synth.js";

const KW = "/tmp/kokoro/kw", K0 = "/tmp/kokoro";
const rd = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const ctx = await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url))));
const weights = rd(`${KW}/weights.bin`);
const K = makeKokoro(ctx, weights, JSON.parse(readFileSync(`${KW}/manifest.json`)), JSON.parse(readFileSync(`${KW}/roles.json`)));
const style = rd(`${K0}/ref_style.bin`).slice(0, 128); // acoustic half (anchors are the ref_style run)

const xConcat = { data: rd(`${K0}/anc__decoder_decoder_Concat_output_0.bin`), rows: 514, cols: 15 };
const asr = { data: rd(`${K0}/anc__encoder_MatMul_1_output_0.bin`), rows: 512, cols: 15 };
const F0 = { data: rd(`${K0}/anc__encoder_F0_proj_Conv_output_0.bin`), rows: 1, cols: 30 };
const N = { data: rd(`${K0}/anc__encoder_N_proj_Conv_output_0.bin`), rows: 1, cols: 30 };

const anchorFor = { encode: "dec__decoder_decoder_encode_Div_3_output_0.bin", "decode.0": "dec__decoder_decoder_decode_0_Div_3_output_0.bin", "decode.1": "dec__decoder_decoder_decode_1_Div_3_output_0.bin", "decode.2": "dec__decoder_decoder_decode_2_Div_3_output_0.bin", "decode.3": "dec__decoder_decoder_decode_3_Div_4_output_0.bin" };
const onStage = async (name, t, extra) => {
  if (extra) { for (const [k, v] of Object.entries(extra)) console.log(`  ${k}: [${v.rows},${v.cols}]`); }
  const ref = rd(`${K0}/${anchorFor[name]}`); const o = await ctx.download(t);
  const m = Math.min(o.length, ref.length); let md = 0; for (let i = 0; i < m; i++) md = Math.max(md, Math.abs(o[i] - ref[i]));
  console.log(`  stage ${name}: [${t.rows},${t.cols}] (${o.length}) vs ref ${ref.length}  maxΔ ${md.toExponential(2)}`);
};
const out = await decoder(K, xConcat, asr, F0, N, style, onStage);
const ref = rd(`${K0}/dec__decoder_decoder_decode_3_Div_4_output_0.bin`);
const o = await ctx.download(out);
const m = Math.min(o.length, ref.length);
let md = 0; for (let i = 0; i < m; i++) md = Math.max(md, Math.abs(o[i] - ref[i]));
console.log(`decoder out [${out.rows},${out.cols}] (${o.length}) vs ref ${ref.length}  maxΔ ${md.toExponential(2)}`);
console.log(md < 5e-3 ? "DECODER PARITY OK" : "DECODER PARITY FAIL");
process.exit(0);
