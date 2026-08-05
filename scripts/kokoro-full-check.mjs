// Full ORT-free Kokoro pipeline (frontend ALBERT → predictor → SineGen → decoder →
// generator) on the "hello world…" input, vs ORT's real_wav.bin. WasmContext.
//   node scripts/kokoro-full-check.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { textEncoding } from "../src/gpu/kokoro.js";
import { ALBERT_DIMS } from "../src/gpu/albert.js";
import { makeKokoro, synth } from "../src/gpu/kokoro-synth.js";

const K0 = "/tmp/kokoro", KW = "/tmp/kokoro/kw", A = "/tmp/kokoro/albert", F = "/tmp/kokoro/frontend";
const rd = (p) => { const u = Uint8Array.from(readFileSync(p)); return new Float32Array(u.buffer, u.byteOffset, u.byteLength / 4); };
const ctx = await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url))));

// ── frontend (ALBERT) weights ──
const man = JSON.parse(readFileSync(`${A}/manifest.json`, "utf8"));
const cpu = {}; for (const k of Object.keys(man)) cpu[k] = rd(`${A}/${k}.bin`);
const scale = 1 / Math.sqrt(ALBERT_DIMS.HEAD_DIM);
for (let i = 0; i < cpu.q_w.length; i++) cpu.q_w[i] *= scale;
for (let i = 0; i < cpu.q_b.length; i++) cpu.q_b[i] *= scale;
const up2 = (n) => ctx.upload(cpu[n], man[n][0], man[n][1]);
const up1 = (n) => ctx.upload(cpu[n], 1, cpu[n].length);
const albertW = {
  EMBED: ALBERT_DIMS.EMBED, map_w: up2("map_w"), map_b: up1("map_b"),
  q_w: up2("q_w"), q_b: up1("q_b"), k_w: up2("k_w"), k_b: up1("k_b"), v_w: up2("v_w"), v_b: up1("v_b"),
  dense_w: up2("dense_w"), dense_b: up1("dense_b"), ffn_w: up2("ffn_w"), ffn_b: up1("ffn_b"),
  ffn_out_w: up2("ffn_out_w"), ffn_out_b: up1("ffn_out_b"), attn_ln_w: up1("attn_ln_w"), attn_ln_b: up1("attn_ln_b"),
  full_ln_w: up1("full_ln_w"), full_ln_b: up1("full_ln_b"), word_emb: cpu.word_emb, pos_emb: cpu.pos_emb,
  tok_emb: cpu.tok_emb, emb_ln_w: cpu.emb_ln_w, emb_ln_b: cpu.emb_ln_b,
};
const fref = JSON.parse(readFileSync(`${F}/ref.json`, "utf8"));
const beW = ctx.upload(rd(`${F}/be_w.bin`), fref.be_in, fref.be_out);
const beB = ctx.upload(rd(`${F}/be_b.bin`), 1, fref.be_out);

// ── inputs (hello run) ──
const idb = Uint8Array.from(readFileSync(`${K0}/hello_ids.bin`));
const ids = new Int32Array(idb.buffer, idb.byteOffset, idb.byteLength / 4);
const style = rd(`${K0}/hello_style.bin`);
console.log("hello ids", Array.from(ids));

const dEnT = textEncoding(ctx, ids, albertW, beW, beB);
const dEn = { data: await ctx.download(dEnT), rows: ids.length, cols: fref.be_out };

const K = makeKokoro(ctx, rd(`${KW}/weights.bin`), JSON.parse(readFileSync(`${KW}/manifest.json`)), JSON.parse(readFileSync(`${KW}/roles.json`)));
// ── stage gates for hello (have real_decode3 + c3) ──
import { predictor, sineGen, sourceSpec, decoder } from "../src/gpu/kokoro-synth.js";
{
  const { xConcat, asr, F0, N } = await predictor(K, dEn, ids, style);
  const d3T = await decoder(K, xConcat, asr, F0, N, style.slice(0, 128));
  const d3 = await ctx.download(d3T);
  const rdc = rd(`${K0}/real_decode3.bin`); let dm = 0; for (let i = 0; i < Math.min(d3.length, rdc.length); i++) dm = Math.max(dm, Math.abs(d3[i] - rdc[i]));
  console.log(`decode3 [${d3T.rows},${d3T.cols}] (${d3.length}) vs real_decode3 ${rdc.length}  maxΔ ${dm.toExponential(2)}`);
  const spec = sourceSpec(sineGen(K, F0));
  const c3 = rd(`${K0}/c3.bin`); const Tt = spec.cols;
  const corrOf = (lo, hi) => { let dot = 0, na = 0, nb = 0; for (let ch = lo; ch < hi; ch++) for (let t = 0; t < Tt; t++) { const a = spec.data[ch * Tt + t], b = c3[ch * 10681 + t]; dot += a * b; na += a * a; nb += b * b; } return (dot / Math.sqrt(na * nb)).toFixed(4); };
  // complex corr (wrap-robust): re=mag*cos(phase), im=mag*sin(phase)
  let dot = 0, na = 0, nb = 0;
  for (let k = 0; k < 11; k++) for (let t = 0; t < Tt; t++) {
    const ma = spec.data[k * Tt + t], pa = spec.data[(k + 11) * Tt + t], mb = c3[k * 10681 + t], pb = c3[(k + 11) * 10681 + t];
    const ar = ma * Math.cos(pa), ai = ma * Math.sin(pa), br = mb * Math.cos(pb), bi = mb * Math.sin(pb);
    dot += ar * br + ai * bi; na += ar * ar + ai * ai; nb += br * br + bi * bi;
  }
  console.log(`sourceSpec [22,${spec.cols}] mag-corr ${corrOf(0, 11)}  phase-corr ${corrOf(11, 22)}  complex-corr ${(dot / Math.sqrt(na * nb)).toFixed(4)}`);
  // frame-shift sweep on complex-corr (detect a pure time-offset)
  for (let sh = -4; sh <= 4; sh++) {
    let d = 0, x = 0, y = 0;
    for (let k = 0; k < 11; k++) for (let t = 0; t < Tt; t++) {
      const tc = t + sh; if (tc < 0 || tc >= 10681) continue;
      const ma = spec.data[k * Tt + t], pa = spec.data[(k + 11) * Tt + t], mb = c3[k * 10681 + tc], pb = c3[(k + 11) * 10681 + tc];
      const ar = ma * Math.cos(pa), ai = ma * Math.sin(pa), br = mb * Math.cos(pb), bi = mb * Math.sin(pb);
      d += ar * br + ai * bi; x += ar * ar + ai * ai; y += br * br + bi * bi;
    }
    process.stdout.write(` sh${sh}:${(d / Math.sqrt(x * y)).toFixed(3)}`);
  }
  console.log("");
  // DECISIVE: iSTFT(c3) → ORT source, compare to my sineGen source (time domain)
  const mySrc = sineGen(K, F0);
  const Tc3 = 10681, NF = 20, hop = 5, Lo = (Tc3 - 1) * hop + NF;
  const win = new Float32Array(NF); for (let n = 0; n < NF; n++) win[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / NF);
  const oo = new Float32Array(Lo), ws = new Float32Array(Lo);
  for (let t = 0; t < Tc3; t++) {
    for (let n = 0; n < NF; n++) {
      let v = 0;
      for (let k = 0; k < NF; k++) {
        const kk = k <= 10 ? k : NF - k; const sgn = k <= 10 ? 1 : -1;
        const mag = c3[kk * Tc3 + t], ph = sgn * c3[(kk + 11) * Tc3 + t];
        v += mag * Math.cos(2 * Math.PI * k * n / NF + ph);
      }
      v /= NF; oo[t * hop + n] += v * win[n]; ws[t * hop + n] += win[n] * win[n];
    }
  }
  for (let i = 0; i < Lo; i++) if (ws[i] > 1e-9) oo[i] /= ws[i];
  const ortSrc = oo.subarray(NF / 2, NF / 2 + mySrc.length);
  let sc = 0, mm = 0, oo2 = 0; const mn = Math.min(ortSrc.length, mySrc.length);
  for (let i = 0; i < mn; i++) { sc += mySrc[i] * ortSrc[i]; mm += mySrc[i] ** 2; oo2 += ortSrc[i] ** 2; }
  console.log(`SOURCE time-domain: my-vs-ORT(iSTFT c3) corr ${(sc / Math.sqrt(mm * oo2)).toFixed(4)}  rms my ${Math.sqrt(mm / mn).toFixed(4)} ort ${Math.sqrt(oo2 / mn).toFixed(4)}`);
}
const t0 = Date.now();
const wav = await synth(K, dEn, ids, style);
const ms = Date.now() - t0;
const ref = rd(`${K0}/real_wav.bin`);
const m = Math.min(wav.length, ref.length);
let dot = 0, na = 0, nb = 0;
for (let i = 0; i < m; i++) { dot += wav[i] * ref[i]; na += wav[i] ** 2; nb += ref[i] ** 2; }
const rms = (a, n) => { let s = 0; for (let i = 0; i < n; i++) s += a[i] ** 2; return Math.sqrt(s / n); };
console.log(`synth ${ms}ms  len my ${wav.length} ref ${ref.length}`);
console.log(`corr ${(dot / Math.sqrt(na * nb)).toFixed(4)}  rms my ${rms(wav, m).toFixed(4)} ref ${rms(ref, m).toFixed(4)}`);
// write a WAV to listen to
import { writeFileSync } from "node:fs";
const sr = 24000, i16 = new Int16Array(wav.length);
for (let i = 0; i < wav.length; i++) i16[i] = Math.max(-32768, Math.min(32767, Math.round(wav[i] * 32767)));
const hdr = Buffer.alloc(44); hdr.write("RIFF", 0); hdr.writeUInt32LE(36 + i16.byteLength, 4); hdr.write("WAVE", 8);
hdr.write("fmt ", 12); hdr.writeUInt32LE(16, 16); hdr.writeUInt16LE(1, 20); hdr.writeUInt16LE(1, 22);
hdr.writeUInt32LE(sr, 24); hdr.writeUInt32LE(sr * 2, 28); hdr.writeUInt16LE(2, 32); hdr.writeUInt16LE(16, 34);
hdr.write("data", 36); hdr.writeUInt32LE(i16.byteLength, 40);
writeFileSync(`${K0}/js_hello.wav`, Buffer.concat([hdr, Buffer.from(i16.buffer)]));
console.log(`wrote ${K0}/js_hello.wav`);
const corr = dot / Math.sqrt(na * nb);
process.exit(corr > 0.9 ? 0 : 1);
