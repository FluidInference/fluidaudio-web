// CI smoke + parity gate: VoiceChat-11B TTS (Aria) vs torch goldens.
// Self-skips (exit 0) when models-local/voicechat-tts is absent — the ~3.5 GB
// export is local-only (scripts/extract-voicechat-tts.py); nothing is hosted.
//
// Deterministic parity track (CFG 0.2, top-p 0.95, noise 0, argmax component):
//   • tokenizer ids == reference HF ids
//   • CAS conditioning embeddings vs torch (per-token max |err|)
//   • warmup + first-6-step backbone hiddens (max |err|, cond & uncond)
//   • audio-code matrix [T,31] vs torch — hard gate: EXACT equality
//   • waveform NRMSE vs torch < 1e-2, plus duration/peak/RMS sanity
//
//   node scripts/ci-smoke-voicechat-tts.mjs [--steps N] [--verbose]
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { assert } from "./lib/ci.mjs";
import { createContext } from "../src/gpu/context.js";
import { VoicechatTokenizer } from "../src/engines/tts-voicechat/tokenizer.js";
import { WeightStore } from "../src/engines/tts-voicechat/weights.js";
import { loadVoicechatTtsModel, synthesizeCodes, rvqNorms, casCond } from "../src/engines/tts-voicechat/model.js";
import { loadVoicechatCodec, dequantize, codecDecode } from "../src/engines/tts-voicechat/codec.js";

const dir = fileURLToPath(new URL("../models-local/voicechat-tts", import.meta.url));
if (!existsSync(`${dir}/tts.0.bin`) || !existsSync(`${dir}/golden.0.bin`)) {
  console.log("voicechat-tts smoke: models-local/voicechat-tts absent — skipping (local-only weights)");
  process.exit(0);
}
const stepsArg = process.argv.indexOf("--steps");
const limitSteps = stepsArg > 0 ? parseInt(process.argv[stepsArg + 1], 10) : 0;

const json = (n) => JSON.parse(readFileSync(`${dir}/${n}`, "utf8"));
const shards = (stem) => {
  const man = json(`${stem}.manifest.json`);
  const bins = [];
  for (let i = 0; i < man.shards; i++) bins.push(readFileSync(`${dir}/${stem}.${i}.bin`));
  return new WeightStore(bins, man);
};
const cfg = json("config.json");
const tokJson = json("tokenizer.json");
const ttsW = shards("tts");
const codecW = shards("codec");
const goldenMan = json("golden.manifest.json");
const golden = shards("golden");
const meta = goldenMan.tensors._meta;

const ctx = await createContext({ backend: "wasm" });
const model = loadVoicechatTtsModel(ctx, ttsW, cfg);
const codec = loadVoicechatCodec(ctx, codecW);
const rvqNormSq = rvqNorms(codec.rvq, cfg.numQuantizers, cfg.codebook, cfg.latent);

const maxAbs = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

// ── tokenizer parity ──
const tok = new VoicechatTokenizer(tokJson);
const ids = tok.encode(meta.text);
assert(JSON.stringify(ids) === JSON.stringify(meta.text_ids), `tokenizer ids ${ids} != reference ${meta.text_ids}`);
console.log(`tokenizer: ${ids.length} ids match HF reference`);

// ── CAS conditioning parity ──
const charVocab = new Map(tokJson.chars.map((c, i) => [c, i]));
const tokenChars = (id) => {
  const out = [];
  for (const ch of tokJson.vocab[id] ?? "") if (charVocab.has(ch)) out.push(charVocab.get(ch));
  return out;
};
const casTokens = golden.f32("cas_tokens");
const casEmbs = golden.f32("cas_embs");
let casErr = 0;
for (let i = 0; i < casTokens.length; i++) {
  const id = casTokens[i] | 0;
  const mine = await casCond(ctx, model, tokenChars(id), id, true);
  casErr = Math.max(casErr, maxAbs(mine, casEmbs.subarray(i * 1152, (i + 1) * 1152)));
}
console.log(`CAS embeddings: max |err| ${casErr.toExponential(2)} over ${casTokens.length} tokens`);
assert(casErr < 1e-3, `CAS embedding error ${casErr}`);

// ── full deterministic generation ──
const frameTokens = Array.from(golden.f32("next_subword_ids"), (v) => v | 0);
const T = limitSteps > 0 ? Math.min(limitSteps, frameTokens.length) : frameTokens.length;
const chars = new Map();
for (const t of [...frameTokens, ...cfg.warmSubwordIds]) if (!chars.has(t)) chars.set(t, tokenChars(t));
const t0 = Date.now();
const { codes, trace } = await synthesizeCodes(ctx, model, frameTokens.slice(0, T), chars, codec.rvq, {
  deterministic: true,
  rvqNormSq,
  captureWarm: true,
  captureSteps: 6,
});
console.log(`generation: ${T} frames in ${((Date.now() - t0) / 1000).toFixed(1)}s (${trace.msPerStep.toFixed(0)} ms/frame incl warmup amortized)`);

// warmup + step hiddens (diagnostics — loose gate; the hard gate is the codes)
const wT = cfg.warmFrames;
const wc = golden.f32("warm_hidden_cond"),
  wu = golden.f32("warm_hidden_uncond");
const warmErrC = maxAbs(trace.warmHidden.subarray(0, wT * 1152), wc);
const warmErrU = maxAbs(trace.warmHidden.subarray(wT * 1152), wu);
console.log(`warmup hidden: max |err| cond ${warmErrC.toExponential(2)} / uncond ${warmErrU.toExponential(2)}`);
const sc = golden.f32("step_hidden_cond"),
  su = golden.f32("step_hidden_uncond");
for (let i = 0; i < Math.min(6, trace.stepHidden.length); i++) {
  const eC = maxAbs(trace.stepHidden[i].subarray(0, 1152), sc.subarray(i * 1152, (i + 1) * 1152));
  const eU = maxAbs(trace.stepHidden[i].subarray(1152), su.subarray(i * 1152, (i + 1) * 1152));
  console.log(`  step ${i} hidden: cond ${eC.toExponential(2)} / uncond ${eU.toExponential(2)}`);
}

// ── HARD GATE: code matrix exact ──
const goldCodes = golden.f32("codes_parity");
let mismatched = 0,
  firstBad = -1;
for (let t = 0; t < T; t++)
  for (let q = 0; q < cfg.numQuantizers; q++)
    if (codes[t][q] !== goldCodes[t * cfg.numQuantizers + q]) {
      mismatched++;
      if (firstBad < 0) firstBad = t;
    }
console.log(`codes: ${T * cfg.numQuantizers - mismatched}/${T * cfg.numQuantizers} match${firstBad >= 0 ? `, first mismatch at frame ${firstBad}` : ""}`);
assert(mismatched === 0, `code matrix differs from torch reference (${mismatched} entries)`);
const hash = createHash("sha256")
  .update(Buffer.from(new Int32Array(codes.flatMap((c) => [...c])).buffer))
  .digest("hex")
  .slice(0, 16);
console.log(`code-matrix sha256/16: ${hash}`);

// ── waveform NRMSE + stats (full-length runs only) ──
if (T === frameTokens.length) {
  const latents = dequantize(codec.rvq, codes, cfg.latent, cfg.numQuantizers);
  const latErr = maxAbs(latents, golden.f32("codec_latents_parity"));
  assert(latErr < 1e-5, `dequantized latents err ${latErr}`);
  const tc = Date.now();
  const wav = await codecDecode(ctx, codec, latents, T);
  const codecS = (Date.now() - tc) / 1000;
  console.log(`codec decode: ${T} frames → ${codecS.toFixed(1)}s (RTFx ${((T * 0.08) / codecS).toFixed(2)})`);
  const ref = golden.f32("wav_parity");
  assert(wav.length === ref.length, `wav length ${wav.length} != ${ref.length}`);
  let se = 0,
    ref2 = 0,
    peak = 0,
    s2 = 0,
    nan = false;
  for (let i = 0; i < wav.length; i++) {
    if (Number.isNaN(wav[i])) nan = true;
    se += (wav[i] - ref[i]) ** 2;
    ref2 += ref[i] * ref[i];
    peak = Math.max(peak, Math.abs(wav[i]));
    s2 += wav[i] * wav[i];
  }
  const nrmse = Math.sqrt(se / ref2);
  const rms = Math.sqrt(s2 / wav.length);
  const secs = wav.length / cfg.sampleRate;
  console.log(`waveform: ${secs.toFixed(2)}s, peak ${peak.toFixed(3)}, rms ${rms.toFixed(4)}, NRMSE vs torch ${nrmse.toExponential(2)}`);
  assert(!nan, "no NaN samples");
  assert(nrmse < 1e-2, `waveform NRMSE ${nrmse}`);
  assert(Math.abs(secs - T * 0.08) < 1e-6, "duration == frames × 80 ms");
  assert(peak > 0.05 && peak < 1.0, `peak ${peak} in (0.05, 1.0)`);
  assert(rms > 0.005 && rms < 0.2, `rms ${rms} in (0.005, 0.2)`);
}

// ── sampled-mode sanity: seeded RNG path runs clean and actually samples ──
{
  const N = Math.min(12, frameTokens.length);
  const { codes: smp } = await synthesizeCodes(ctx, model, frameTokens.slice(0, N), chars, codec.rvq, {
    deterministic: false,
    seed: 7,
    rvqNormSq,
  });
  let inRange = true,
    differs = false;
  for (let t = 0; t < N; t++)
    for (let q = 0; q < cfg.numQuantizers; q++) {
      if (smp[t][q] < 0 || smp[t][q] >= cfg.codebook) inRange = false;
      if (smp[t][q] !== goldCodes[t * cfg.numQuantizers + q]) differs = true;
    }
  assert(inRange, "sampled codes in [0, codebook)");
  assert(differs, "sampled codes differ from the deterministic track (noise active)");
  console.log(`sampled mode: ${N} frames OK (seeded, diverges from parity track as expected)`);
}
console.log("VOICECHAT-TTS SMOKE OK");
process.exit(0);
