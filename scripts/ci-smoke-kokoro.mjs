// CI smoke: Kokoro EN synthesis on the WASM backend — fixed phoneme input,
// reference-free sanity (duration / level / no NaN). Deterministic (en model).
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { hfGet, hfF32, hfJson, assert } from "./lib/ci.mjs";
import { loadKokoroBackend } from "../src/engines/tts-kokoro/synth-backend.js";

const vocab = JSON.parse(readFileSync(fileURLToPath(new URL("../src/engines/tts-kokoro/vocab.json", import.meta.url))));
// route the backend's fetchCached through the CI cache (repo is encoded per-path)
const fetchCached = async (url) => {
  const m = url.match(/^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/resolve\/main\/(.+)$/);
  if (!m) throw new Error("unexpected url " + url);
  return await hfGet(m[1], m[2]);
};
const hfUrl = (repo, path) => `https://huggingface.co/${repo}/resolve/main/${path}`;
const be = await loadKokoroBackend(fetchCached, hfUrl, vocab);
const t0 = Date.now();
const wav = await be.synthFromPhonemes("həlˈoʊ wˈɜːld ðɪs ɪz ɐ tˈɛst", "af_heart");
const secs = wav.length / 24000;
let nan = false,
  peak = 0,
  s = 0;
for (let i = 0; i < wav.length; i++) {
  if (Number.isNaN(wav[i])) nan = true;
  peak = Math.max(peak, Math.abs(wav[i]));
  s += wav[i] ** 2;
}
const rms = Math.sqrt(s / wav.length);
console.log(`kokoro (wasm backend): ${Date.now() - t0}ms → ${secs.toFixed(2)}s audio, rms ${rms.toFixed(4)}, peak ${peak.toFixed(3)}`);
assert(!nan, "no NaN samples");
assert(secs > 1.5 && secs < 3.5, `duration ${secs.toFixed(2)}s in [1.5, 3.5]`);
assert(rms > 0.02 && rms < 0.2, `rms ${rms.toFixed(4)} in [0.02, 0.2]`);
assert(peak < 1.0, "no clipping");
console.log("KOKORO SMOKE OK");
process.exit(0);
