// End-to-end parity for the wired Kokoro frontend: input_ids → ALBERT →
// bert_encoder → d_en, on raw WebGPU vs the ONNX model. Needs /tmp/kokoro/albert/*
// (kokoro-extract-albert.py) and /tmp/kokoro/frontend/* (kokoro-ref-frontend.py).
//   node scripts/gpu-kokoro.mjs
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { textEncoding } from "../src/gpu/kokoro.js";
import { ALBERT_DIMS } from "../src/gpu/albert.js";

const A = "/tmp/kokoro/albert",
  F = "/tmp/kokoro/frontend";
const man = JSON.parse(readFileSync(`${A}/manifest.json`, "utf8"));
const fref = JSON.parse(readFileSync(`${F}/ref.json`, "utf8"));
const rd = (p) => new Float32Array(Uint8Array.from(readFileSync(p)).buffer);
const cpu = {};
for (const k of Object.keys(man)) cpu[k] = rd(`${A}/${k}.bin`);
const ids = new Int32Array(Uint8Array.from(readFileSync(`${F}/input_ids.bin`)).buffer);
const refDen = rd(`${F}/ref_den.bin`);
const seq = fref.seq;

// fold attention scale into query proj (as in gpu-albert.mjs)
const scale = 1 / Math.sqrt(ALBERT_DIMS.HEAD_DIM);
for (let i = 0; i < cpu.q_w.length; i++) cpu.q_w[i] *= scale;
for (let i = 0; i < cpu.q_b.length; i++) cpu.q_b[i] *= scale;

const dev = await getDevice();
const ctx = new GpuContext(dev);
const up2 = (n) => ctx.upload(cpu[n], man[n][0], man[n][1]);
const up1 = (n) => ctx.upload(cpu[n], 1, cpu[n].length);
const albertW = {
  EMBED: ALBERT_DIMS.EMBED,
  map_w: up2("map_w"),
  map_b: up1("map_b"),
  q_w: up2("q_w"),
  q_b: up1("q_b"),
  k_w: up2("k_w"),
  k_b: up1("k_b"),
  v_w: up2("v_w"),
  v_b: up1("v_b"),
  dense_w: up2("dense_w"),
  dense_b: up1("dense_b"),
  ffn_w: up2("ffn_w"),
  ffn_b: up1("ffn_b"),
  ffn_out_w: up2("ffn_out_w"),
  ffn_out_b: up1("ffn_out_b"),
  attn_ln_w: up1("attn_ln_w"),
  attn_ln_b: up1("attn_ln_b"),
  full_ln_w: up1("full_ln_w"),
  full_ln_b: up1("full_ln_b"),
  word_emb: cpu.word_emb,
  pos_emb: cpu.pos_emb,
  tok_emb: cpu.tok_emb,
  emb_ln_w: cpu.emb_ln_w,
  emb_ln_b: cpu.emb_ln_b,
};
const beW = ctx.upload(rd(`${F}/be_w.bin`), fref.be_in, fref.be_out);
const beB = ctx.upload(rd(`${F}/be_b.bin`), 1, fref.be_out);

const den = await ctx.download(textEncoding(ctx, ids, albertW, beW, beB));
let mx = 0,
  se = 0,
  sr = 0;
for (let i = 0; i < den.length; i++) {
  const d = Math.abs(den[i] - refDen[i]);
  mx = Math.max(mx, d);
  se += d * d;
  sr += refDen[i] * refDen[i];
}
const rel = Math.sqrt(se / sr);
console.log(`Kokoro frontend  input_ids → d_en [${seq},${fref.be_out}]: max ${mx.toExponential(2)}  rel ${rel.toExponential(2)}`);
const ok = rel < 2e-2;
console.log(ok ? "FRONTEND PARITY OK (raw WebGPU == ONNX)" : "PARITY FAIL");
process.exit(ok ? 0 : 1);
