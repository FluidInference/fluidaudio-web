// End-to-end parity: Kokoro's ALBERT text encoder on raw WebGPU vs the ONNX model.
// Needs the extracted weights + reference in /tmp/kokoro/albert (see the repo's
// extract/ref python). Runs the full input_ids -> ALBERT-output path on the GPU
// (dawn) and compares to onnxruntime's intermediate.
//   node scripts/gpu-albert.mjs
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { embed, albertForward, ALBERT_DIMS } from "../src/gpu/albert.js";

const D = "/tmp/kokoro/albert";
const manifest = JSON.parse(readFileSync(`${D}/manifest.json`, "utf8"));
const ref = JSON.parse(readFileSync(`${D}/ref.json`, "utf8"));
// Node's Buffer is a view into a shared pool — copy out the exact byte range.
const readBin = (path) => { const b = readFileSync(path); return Uint8Array.from(b); };
const f32 = (name) => new Float32Array(readBin(`${D}/${name}.bin`).buffer);
const cpu = {};
for (const k of Object.keys(manifest)) cpu[k] = f32(k);
const ids = new Int32Array(readBin(`${D}/input_ids.bin`).buffer);
const refIn = f32("ref_albert_in");
const refOut = f32("ref_albert_out");
const seq = ref.seq;

// Fold the attention scale 1/sqrt(head_dim) into the query projection.
const scale = 1 / Math.sqrt(ALBERT_DIMS.HEAD_DIM);
for (let i = 0; i < cpu.q_w.length; i++) cpu.q_w[i] *= scale;
for (let i = 0; i < cpu.q_b.length; i++) cpu.q_b[i] *= scale;

const dev = await getDevice();
const ctx = new GpuContext(dev);

// Upload the weights that run on the GPU. Matmul B tensors are [in,out]; biases
// and LayerNorm params are [1,N].
const rc = (name) => manifest[name];
const up2 = (name) => ctx.upload(cpu[name], rc(name)[0], rc(name)[1]);
const up1 = (name) => ctx.upload(cpu[name], 1, cpu[name].length);
const w = {
  map_w: up2("map_w"), map_b: up1("map_b"),
  q_w: up2("q_w"), q_b: up1("q_b"), k_w: up2("k_w"), k_b: up1("k_b"),
  v_w: up2("v_w"), v_b: up1("v_b"), dense_w: up2("dense_w"), dense_b: up1("dense_b"),
  ffn_w: up2("ffn_w"), ffn_b: up1("ffn_b"), ffn_out_w: up2("ffn_out_w"), ffn_out_b: up1("ffn_out_b"),
  attn_ln_w: up1("attn_ln_w"), attn_ln_b: up1("attn_ln_b"),
  full_ln_w: up1("full_ln_w"), full_ln_b: up1("full_ln_b"),
  // CPU-side embedding tables
  word_emb: cpu.word_emb, pos_emb: cpu.pos_emb, tok_emb: cpu.tok_emb,
  emb_ln_w: cpu.emb_ln_w, emb_ln_b: cpu.emb_ln_b,
};

const stats = (a, b) => {
  let mx = 0, se = 0, sr = 0;
  for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); mx = Math.max(mx, d); se += d * d; sr += b[i] * b[i]; }
  return { max: mx, rms: Math.sqrt(se / a.length), rel: Math.sqrt(se / sr) };
};

// Embeddings (CPU) + 128->768 mapping (GPU) — check against ALBERT input.
const emb = embed(ids, w);                       // [seq,128]
const embT = ctx.upload(emb, seq, ALBERT_DIMS.EMBED);
const mapped = await ctx.download(ctx.matmul(embT, w.map_w, { bias: w.map_b }));
const sIn = stats(mapped, refIn);
console.log(`ALBERT input  (embeds+map): max ${sIn.max.toExponential(2)}  rms ${sIn.rms.toExponential(2)}  rel ${sIn.rel.toExponential(2)}`);

// Full stack -> ALBERT output.
const out = await ctx.download(albertForward(ctx, embT, w, seq));
const sOut = stats(out, refOut);
console.log(`ALBERT output (12 layers) : max ${sOut.max.toExponential(2)}  rms ${sOut.rms.toExponential(2)}  rel ${sOut.rel.toExponential(2)}`);
console.log(`  ref out mean ${(refOut.reduce((a, b) => a + b, 0) / refOut.length).toExponential(2)}  seq ${seq} hidden ${ALBERT_DIMS.HIDDEN}`);

const ok = sOut.rel < 2e-2;
console.log(ok ? "\nALBERT PARITY OK (raw WebGPU == ONNX)" : "\nPARITY FAIL");
process.exit(ok ? 0 : 1);
