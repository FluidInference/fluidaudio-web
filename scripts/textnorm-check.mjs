// Gate for the vendored text-processing-rs wasm (src/vendor/text-processing):
// exercises the exact entry points the engines use — en TN (Kokoro input) and
// en ITN (Parakeet output) — against fixed expectations. Exits non-zero on any
// mismatch.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as m from "../src/vendor/text-processing/text_processing_rs.js";
import { tnBySentence } from "../src/core/textnorm-split.js";

await m.default({ module_or_path: readFileSync(fileURLToPath(new URL("../src/vendor/text-processing/text_processing_rs_bg.wasm", import.meta.url))) });

// TN goes through the SHIPPING sentence-preserving wrapper (textnorm-split.js):
// terminators are kept and each sentence normalizes alone — "$4.50." must keep
// its period AND the "four dollars fifty cents" reading (the raw sentence API
// eats final punctuation and degrades the currency reading).
const tn = (t) => tnBySentence((seg) => m.tnNormalizeSentenceLang(seg, "en"), t);
const TN = [
  ["It costs $4.50 today", "It costs four dollars fifty cents today"],
  ["The year 2026 is here", "The year twenty twenty six is here"],
  ["Call me at 3:30", "Call me at three thirty"],
  ["I have 21 apples", "I have twenty one apples"],
  ["It costs $4.50.", "It costs four dollars fifty cents."],
  ["It costs $4.50. I bought 3 of them.", "It costs four dollars fifty cents. I bought three of them."],
  ["Call me at 3:30!", "Call me at three thirty!"],
];
const ITN = [
  ["i paid four dollars and fifty cents for twenty one apples", "i paid $4.50 for 21 apples"],
  ["it is three thirty", "it is 03:30"],
  ["two hundred people came", "200 people came"],
];

let fail = 0;
for (const [inp, want] of TN) {
  const got = tn(inp);
  const ok = got === want;
  if (!ok) fail++;
  console.log(`${ok ? "✓" : "✗"} TN  ${JSON.stringify(inp)} → ${JSON.stringify(got)}${ok ? "" : `  (want ${JSON.stringify(want)})`}`);
}
for (const [inp, want] of ITN) {
  const got = m.normalizeSentence(inp);
  const ok = got === want;
  if (!ok) fail++;
  console.log(`${ok ? "✓" : "✗"} ITN ${JSON.stringify(inp)} → ${JSON.stringify(got)}${ok ? "" : `  (want ${JSON.stringify(want)})`}`);
}
// Pass-through sanity: TN must never destroy text it can't improve.
const passthrough = "Hello world, nothing to normalize here.";
const pt = tn(passthrough);
const ptOk = pt === passthrough;
if (!ptOk) fail++;
console.log(`${ptOk ? "✓" : "✗"} passthrough ${JSON.stringify(pt)}`);

console.log(fail === 0 ? "TEXTNORM OK" : `TEXTNORM ${fail} FAILURES`);
process.exit(fail === 0 ? 0 : 1);
