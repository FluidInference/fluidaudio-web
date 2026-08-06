// Long-utterance stress: big T (LSTM NaN regime), both backends, en engine path.
import { readFileSync, writeFileSync } from "node:fs";
import { loadKokoroBackend } from "../src/engines/tts-kokoro/synth-backend.js";
import { EnglishLexicon } from "../src/engines/tts-kokoro/lexicon.js";
const K0 = "/tmp/kokoro";
const vocab = JSON.parse(readFileSync("src/engines/tts-kokoro/vocab.json"));
const map = {
  "kokoro/weights.bin": `${K0}/kw/weights.bin`,
  "kokoro/manifest.json": `${K0}/kw/manifest.json`,
  "kokoro/roles.json": `${K0}/kw/roles.json`,
  "kokoro/be_w.bin": `${K0}/frontend/be_w.bin`,
  "kokoro/be_b.bin": `${K0}/frontend/be_b.bin`,
  "kokoro/ref.json": `${K0}/frontend/ref.json`,
  "voices/af_heart.bin": `${K0}/af_heart.bin`,
};
const hfUrl = (r, p) => p;
const fetchCached = async (p) => {
  if (map[p]) return Uint8Array.from(readFileSync(map[p]));
  if (p.startsWith("kokoro/albert/")) return Uint8Array.from(readFileSync(`${K0}/albert/${p.split("/").pop()}`));
  throw new Error("no map " + p);
};
const be = await loadKokoroBackend(fetchCached, hfUrl, vocab);
// long sentence via lexicon
const lex = await EnglishLexicon.load(async (url) => {
  throw new Error("no net: " + url);
}).catch(() => null);
// fall back to a hand phoneme string repeated (long IPA)
const base = "həlˈoʊ wˈɜːld ðɪs ɪz ɐ tˈɛst ɐv ðə lˈɔŋ sˈɛntəns pˈæθ ";
const phon = base.repeat(4).trim(); // ~4x → T should be ~350+
console.log("phoneme len", phon.length);
const t0 = Date.now();
const wav = await be.synthFromPhonemes(phon, "af_heart");
const secs = wav.length / 24000;
let nan = false,
  peak = 0,
  s = 0;
for (let i = 0; i < wav.length; i++) {
  if (Number.isNaN(wav[i])) nan = true;
  peak = Math.max(peak, Math.abs(wav[i]));
  s += wav[i] ** 2;
}
console.log(`long synth: ${Date.now() - t0}ms for ${secs.toFixed(2)}s audio  NaN ${nan}  peak ${peak.toFixed(3)}  rms ${Math.sqrt(s / wav.length).toFixed(4)}`);
const i16 = new Int16Array(wav.length);
for (let i = 0; i < wav.length; i++) i16[i] = Math.max(-32768, Math.min(32767, Math.round(wav[i] * 32767)));
const h = Buffer.alloc(44);
h.write("RIFF", 0);
h.writeUInt32LE(36 + i16.byteLength, 4);
h.write("WAVE", 8);
h.write("fmt ", 12);
h.writeUInt32LE(16, 16);
h.writeUInt16LE(1, 20);
h.writeUInt16LE(1, 22);
h.writeUInt32LE(24000, 24);
h.writeUInt32LE(48000, 28);
h.writeUInt16LE(2, 32);
h.writeUInt16LE(16, 34);
h.write("data", 36);
h.writeUInt32LE(i16.byteLength, 40);
writeFileSync(`${K0}/js_long.wav`, Buffer.concat([h, Buffer.from(i16.buffer)]));
console.log("wrote js_long.wav");
process.exit(0);
