// Full ORT-free zh pipeline (ALBERT frontend → predictor → NSF SineGen → decoder →
// generator) vs same-run ORT anchors. Source is stochastic → gate spec MAGNITUDE
// channels + waveform stats; final verify is by-ear.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { textEncoding } from "../src/gpu/kokoro.js";
import { ALBERT_DIMS } from "../src/gpu/albert.js";
import { makeKokoro, synth, predictor, sineGen, sourceSpec } from "../src/gpu/kokoro-synth.js";

const Z = "/tmp/kokoro-zh";
const rd = (p) => {
  const u = Uint8Array.from(readFileSync(p));
  return new Float32Array(u.buffer, u.byteOffset, u.byteLength / 4);
};
const ctx = await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url))));

const man = JSON.parse(readFileSync(`${Z}/albert/manifest.json`));
const cpu = {};
for (const k of Object.keys(man)) cpu[k] = rd(`${Z}/albert/${k}.bin`);
const sc = 1 / Math.sqrt(ALBERT_DIMS.HEAD_DIM);
for (let i = 0; i < cpu.q_w.length; i++) cpu.q_w[i] *= sc;
for (let i = 0; i < cpu.q_b.length; i++) cpu.q_b[i] *= sc;
const up2 = (n) => ctx.upload(cpu[n], man[n][0], man[n][1]),
  up1 = (n) => ctx.upload(cpu[n], 1, cpu[n].length);
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
const fref = JSON.parse(readFileSync(`${Z}/kw/ref.json`));
const beW = ctx.upload(rd(`${Z}/kw/be_w.bin`), fref.be_in, fref.be_out),
  beB = ctx.upload(rd(`${Z}/kw/be_b.bin`), 1, fref.be_out);

const _ib = readFileSync(`${Z}/zh_ids.bin`);
const ids = new Int32Array(_ib.buffer, _ib.byteOffset, _ib.byteLength / 4);
const style = rd(`${Z}/zh_style.bin`);
const dEn = { data: await ctx.download(textEncoding(ctx, ids, albertW, beW, beB)), rows: ids.length, cols: 512 };
const K = makeKokoro(ctx, rd(`${Z}/kw/weights.bin`), JSON.parse(readFileSync(`${Z}/kw/manifest.json`)), JSON.parse(readFileSync(`${Z}/kw/roles.json`)));

// spec magnitude gate (my source vs ORT source; phase random by design)
{
  const { F0 } = await predictor(K, dEn, ids, style);
  const spec = sourceSpec(sineGen(K, F0, { nsfNoise: true, randPhase: true }));
  const ref = rd(`${Z}/anc_spec.bin`);
  const Tr = ref.length / 22,
    Tt = spec.cols;
  let dot = 0,
    a = 0,
    b = 0;
  const Tm = Math.min(Tt, Tr);
  for (let k = 0; k < 11; k++)
    for (let t = 0; t < Tm; t++) {
      const x = spec.data[k * Tt + t],
        y = ref[k * Tr + t];
      dot += x * y;
      a += x * x;
      b += y * y;
    }
  console.log(`spec [22,${Tt}] vs [22,${Tr}]  MAG-corr ${(dot / Math.sqrt(a * b)).toFixed(4)}`);
}
const t0 = Date.now();
const wav = await synth(K, dEn, ids, style);
const ref = rd(`${Z}/anc_wav.bin`);
const rms = (x) => {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] ** 2;
  return Math.sqrt(s / x.length);
};
console.log(`synth ${Date.now() - t0}ms  len my ${wav.length} ref ${ref.length}  rms my ${rms(wav).toFixed(4)} ref ${rms(ref).toFixed(4)}`);
const sr = 24000,
  i16 = new Int16Array(wav.length);
for (let i = 0; i < wav.length; i++) i16[i] = Math.max(-32768, Math.min(32767, Math.round(wav[i] * 32767)));
const h = Buffer.alloc(44);
h.write("RIFF", 0);
h.writeUInt32LE(36 + i16.byteLength, 4);
h.write("WAVE", 8);
h.write("fmt ", 12);
h.writeUInt32LE(16, 16);
h.writeUInt16LE(1, 20);
h.writeUInt16LE(1, 22);
h.writeUInt32LE(sr, 24);
h.writeUInt32LE(sr * 2, 28);
h.writeUInt16LE(2, 32);
h.writeUInt16LE(16, 34);
h.write("data", 36);
h.writeUInt32LE(i16.byteLength, 40);
writeFileSync(`${Z}/js_zh.wav`, Buffer.concat([h, Buffer.from(i16.buffer)]));
// also write ORT ref as wav for A/B
const r16 = new Int16Array(ref.length);
for (let i = 0; i < ref.length; i++) r16[i] = Math.max(-32768, Math.min(32767, Math.round(ref[i] * 32767)));
h.writeUInt32LE(36 + r16.byteLength, 4);
h.writeUInt32LE(r16.byteLength, 40);
writeFileSync(`${Z}/ort_zh.wav`, Buffer.concat([h, Buffer.from(r16.buffer)]));
console.log("wrote js_zh.wav + ort_zh.wav");
process.exit(0);
