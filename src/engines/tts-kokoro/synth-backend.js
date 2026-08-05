// ORT-free Kokoro synthesis backend: loads the extracted weights + ALBERT frontend
// and runs the hand-ported pipeline (src/gpu/kokoro-synth.js) on the ctx backend
// (WebGPU or WASM). Replaces kokoro-js's generate_from_ids — no onnxruntime.
//
// Weights: FluidInference/fluidaudio-web/kokoro/{weights.bin,manifest.json,roles.json}
// (predictor+decoder+generator) + kokoro/albert/* + kokoro/be_{w,b}.bin (frontend).
// Tokenizer vocab bundled (vocab.json). Voice packs fetched from the Kokoro repo.

import { createContext } from "../../gpu/context.js";
import { textEncoding } from "../../gpu/kokoro.js";
import { ALBERT_DIMS } from "../../gpu/albert.js";
import { makeKokoro, synth } from "../../gpu/kokoro-synth.js";

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";

/**
 * Load the ORT-free Kokoro backend. `fetchCached(url, onProgress?, name?) → bytes`.
 * modelDir: "kokoro" (en, v1.0) or "kokoro-zh" (v1.1-zh). voiceRepo: the Kokoro HF
 * repo to pull voice packs from.
 */
export async function loadKokoroBackend(fetchCached, hfUrl, vocab, { modelDir = "kokoro", voiceRepo = "onnx-community/Kokoro-82M-v1.0-ONNX", onProgress } = {}) {
  const ctx = await createContext();
  const f32 = (u8) => new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
  const json = async (path) => JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path)));
  const bytes = (path) => fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path);

  // predictor/decoder/generator weights
  const K = makeKokoro(ctx, f32(await bytes(`${modelDir}/weights.bin`)), await json(`${modelDir}/manifest.json`), await json(`${modelDir}/roles.json`));

  // ALBERT frontend (same structure as scripts/gpu-kokoro.mjs)
  const aman = await json(`${modelDir}/albert/manifest.json`);
  const cpu = {};
  for (const k of Object.keys(aman)) cpu[k] = f32(await bytes(`${modelDir}/albert/${k}.bin`));
  const scale = 1 / Math.sqrt(ALBERT_DIMS.HEAD_DIM);
  for (let i = 0; i < cpu.q_w.length; i++) cpu.q_w[i] *= scale;
  for (let i = 0; i < cpu.q_b.length; i++) cpu.q_b[i] *= scale;
  const up2 = (n) => ctx.upload(cpu[n], aman[n][0], aman[n][1]);
  const up1 = (n) => ctx.upload(cpu[n], 1, cpu[n].length);
  const albertW = {
    EMBED: ALBERT_DIMS.EMBED, map_w: up2("map_w"), map_b: up1("map_b"),
    q_w: up2("q_w"), q_b: up1("q_b"), k_w: up2("k_w"), k_b: up1("k_b"), v_w: up2("v_w"), v_b: up1("v_b"),
    dense_w: up2("dense_w"), dense_b: up1("dense_b"), ffn_w: up2("ffn_w"), ffn_b: up1("ffn_b"),
    ffn_out_w: up2("ffn_out_w"), ffn_out_b: up1("ffn_out_b"), attn_ln_w: up1("attn_ln_w"), attn_ln_b: up1("attn_ln_b"),
    full_ln_w: up1("full_ln_w"), full_ln_b: up1("full_ln_b"), word_emb: cpu.word_emb, pos_emb: cpu.pos_emb,
    tok_emb: cpu.tok_emb, emb_ln_w: cpu.emb_ln_w, emb_ln_b: cpu.emb_ln_b,
  };
  const fref = await json(`${modelDir}/ref.json`);
  const beW = ctx.upload(f32(await bytes(`${modelDir}/be_w.bin`)), fref.be_in, fref.be_out);
  const beB = ctx.upload(f32(await bytes(`${modelDir}/be_b.bin`)), 1, fref.be_out);

  const voiceCache = new Map();
  const getVoice = async (name) => {
    if (!voiceCache.has(name)) voiceCache.set(name, f32(await fetchCached(hfUrl(voiceRepo, `voices/${name}.bin`), onProgress, `${name}.bin`)));
    return voiceCache.get(name); // [510*256]
  };

  return {
    backend: ctx.backend ?? "webgpu",
    /** phonemes (IPA string) → 24 kHz Float32Array. */
    async synthFromPhonemes(phonemes, voice = "af_heart") {
      const ids = [0]; // $ BOS
      for (const ch of phonemes) { const id = vocab[ch]; if (id !== undefined) ids.push(id); }
      ids.push(0); // $ EOS
      const idArr = Int32Array.from(ids);
      const pack = await getVoice(voice);
      const si = 256 * Math.min(Math.max(idArr.length - 2, 0), 509);
      const style = pack.slice(si, si + 256);
      const dEnT = textEncoding(ctx, idArr, albertW, beW, beB);
      const dEn = { data: await ctx.download(dEnT), rows: idArr.length, cols: fref.be_out };
      return await synth(K, dEn, idArr, style);
    },
  };
}
