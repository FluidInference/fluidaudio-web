// Stage gates (generator, decoder, predictor) on the WebGPU backend.
import { readFileSync } from "node:fs";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { makeKokoro, generator, decoder, predictor } from "../src/gpu/kokoro-synth.js";
const K0 = "/tmp/kokoro", KW = "/tmp/kokoro/kw";
const rd = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const ctx = new GpuContext(await getDevice());
const K = makeKokoro(ctx, rd(`${KW}/weights.bin`), JSON.parse(readFileSync(`${KW}/manifest.json`)), JSON.parse(readFileSync(`${KW}/roles.json`)));
const me = (a, b) => { let m = 0; const n = Math.min(a.length, b.length); for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

// generator gate (hello anchors)
{
  const dec3 = rd(`${K0}/real_decode3.bin`), c3 = rd(`${K0}/c3.bin`), style = rd(`${K0}/hello_style.bin`).slice(0, 128);
  const wav = await generator(K, { data: dec3, rows: 512, cols: 178 }, { data: c3, rows: 22, cols: 10681 }, style);
  const ref = rd(`${K0}/real_wav.bin`); const m = Math.min(wav.length, ref.length);
  let dot = 0, a = 0, b = 0; for (let i = 0; i < m; i++) { dot += wav[i] * ref[i]; a += wav[i] ** 2; b += ref[i] ** 2; }
  console.log(`GPU generator: corr ${(dot / Math.sqrt(a * b)).toFixed(4)}  maxΔ ${me(wav, ref).toExponential(2)}  hasNaN ${wav.some(Number.isNaN)}`);
}
// decoder gate (ref anchors)
{
  const style = rd(`${K0}/ref_style.bin`).slice(0, 128);
  const xC = { data: rd(`${K0}/anc__decoder_decoder_Concat_output_0.bin`), rows: 514, cols: 15 };
  const asr = { data: rd(`${K0}/anc__encoder_MatMul_1_output_0.bin`), rows: 512, cols: 15 };
  const F0 = { data: rd(`${K0}/anc__encoder_F0_proj_Conv_output_0.bin`), rows: 1, cols: 30 };
  const N = { data: rd(`${K0}/anc__encoder_N_proj_Conv_output_0.bin`), rows: 1, cols: 30 };
  const out = await ctx.download(await decoder(K, xC, asr, F0, N, style));
  console.log(`GPU decoder: maxΔ ${me(out, rd(`${K0}/dec__decoder_decoder_decode_3_Div_4_output_0.bin`)).toExponential(2)}  hasNaN ${out.some(Number.isNaN)}`);
}
// predictor gate (ref anchors)
{
  const style = rd(`${K0}/ref_style.bin`);
  const _ib = readFileSync(`${K0}/ref_ids.bin`); const ids = new Int32Array(_ib.buffer, _ib.byteOffset, _ib.byteLength / 4);
  const dEnRaw = rd(`${K0}/anc__encoder_bert_encoder_Add_output_0.bin`);
  const out = await predictor(K, { data: dEnRaw, rows: dEnRaw.length / 512, cols: 512 }, ids, style);
  const f0 = await ctx.download(out.F0), asr = await ctx.download(out.asr);
  console.log(`GPU predictor: F0 maxΔ ${me(f0, rd(`${K0}/anc__encoder_F0_proj_Conv_output_0.bin`)).toExponential(2)}  asr maxΔ ${me(asr, rd(`${K0}/anc__encoder_MatMul_1_output_0.bin`)).toExponential(2)}  xC maxΔ ${me(out.xConcat.data, rd(`${K0}/anc__decoder_decoder_Concat_output_0.bin`)).toExponential(2)}`);
}
process.exit(0);
