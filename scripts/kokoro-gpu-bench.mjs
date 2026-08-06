// Kokoro en full synth on the WebGPU backend (dawn) — timing + correctness.
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { textEncoding } from "../src/gpu/kokoro.js";
import { ALBERT_DIMS } from "../src/gpu/albert.js";
import { makeKokoro, synth } from "../src/gpu/kokoro-synth.js";
const K0 = "/tmp/kokoro",
  KW = "/tmp/kokoro/kw",
  A = "/tmp/kokoro/albert",
  F = "/tmp/kokoro/frontend";
const rd = (p) => {
  const u = Uint8Array.from(readFileSync(p));
  return new Float32Array(u.buffer, u.byteOffset, u.byteLength / 4);
};
const ctx = new GpuContext(await getDevice());
const man = JSON.parse(readFileSync(`${A}/manifest.json`));
const cpu = {};
for (const k of Object.keys(man)) cpu[k] = rd(`${A}/${k}.bin`);
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
const fref = JSON.parse(readFileSync(`${F}/ref.json`));
const beW = ctx.upload(rd(`${F}/be_w.bin`), fref.be_in, fref.be_out),
  beB = ctx.upload(rd(`${F}/be_b.bin`), 1, fref.be_out);
let ids;
if (process.env.LONG) {
  const vocab = JSON.parse(readFileSync("src/engines/tts-kokoro/vocab.json"));
  const phon = "həlˈoʊ wˈɜːld ðɪs ɪz ɐ tˈɛst ɐv ðə lˈɔŋ sˈɛntəns pˈæθ ".repeat(4).trim();
  const a = [0];
  for (const ch of phon) {
    const id = vocab[ch];
    if (id !== undefined) a.push(id);
  }
  a.push(0);
  ids = Int32Array.from(a);
} else {
  const idb = Uint8Array.from(readFileSync(`${K0}/hello_ids.bin`));
  ids = new Int32Array(idb.buffer, idb.byteOffset, idb.byteLength / 4);
}
const style = rd(`${K0}/hello_style.bin`);
const K = makeKokoro(ctx, rd(`${KW}/weights.bin`), JSON.parse(readFileSync(`${KW}/manifest.json`)), JSON.parse(readFileSync(`${KW}/roles.json`)));
const dEn = { data: await ctx.download(textEncoding(ctx, ids, albertW, beW, beB)), rows: ids.length, cols: fref.be_out };
// warm + timed
let wav = await synth(K, dEn, ids, style);
const t0 = performance.now();
wav = await synth(K, dEn, ids, style);
const ms = performance.now() - t0;
if (process.env.LONG) {
  let nan = false,
    pk = 0,
    s2 = 0;
  for (let i = 0; i < wav.length; i++) {
    if (Number.isNaN(wav[i])) nan = true;
    pk = Math.max(pk, Math.abs(wav[i]));
    s2 += wav[i] ** 2;
  }
  console.log(
    `WebGPU LONG synth ${ms.toFixed(0)}ms for ${(wav.length / 24000).toFixed(2)}s audio → RTFx ${(wav.length / 24000 / (ms / 1000)).toFixed(1)}  NaN ${nan}  peak ${pk.toFixed(3)}  rms ${Math.sqrt(s2 / wav.length).toFixed(4)}`,
  );
} else {
  const ref = rd(`${K0}/real_wav.bin`);
  const m = Math.min(wav.length, ref.length);
  let dot = 0,
    a = 0,
    b = 0;
  for (let i = 0; i < m; i++) {
    dot += wav[i] * ref[i];
    a += wav[i] ** 2;
    b += ref[i] ** 2;
  }
  console.log(
    `WebGPU synth ${ms.toFixed(0)}ms for ${(wav.length / 24000).toFixed(2)}s audio → RTFx ${(wav.length / 24000 / (ms / 1000)).toFixed(1)}  corr ${(dot / Math.sqrt(a * b)).toFixed(4)}`,
  );
}
process.exit(0);
