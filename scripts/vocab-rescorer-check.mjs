// Gate for the BK-tree custom-vocabulary rescorer (port of FluidAudio's Swift
// text path). Behavior table covers: single-word fuzzy, 2-word compound,
// multi-word phrase, alias, capitalization preservation, and no-false-positive.
import { createVocabularyRescorer, BKTree, levenshtein, normalizeForSimilarity } from "../src/engines/asr-parakeet/vocab-rescorer.js";

let fail = 0;
const eq = (got, want, label) => {
  const ok = got === want;
  if (!ok) fail++;
  console.log(`${ok ? "✓" : "✗"} ${label}: ${JSON.stringify(got)}${ok ? "" : `  (want ${JSON.stringify(want)})`}`);
};

// levenshtein sanity
eq(levenshtein("kitten", "sitting"), 3, "levenshtein kitten/sitting");
eq(levenshtein("", "abc"), 3, "levenshtein empty");

// BK-tree returns the same set as a linear scan
{
  const words = ["nvidia", "newrez", "kokoro", "parakeet", "whisper", "sortformer", "cloudflare", "webgpu"];
  const entries = words.map((w) => ({ term: { text: w }, normalizedText: w }));
  const tree = new BKTree(entries);
  for (const q of ["invidia", "newres", "whispr", "webgu", "zzz"]) {
    for (const maxD of [1, 2, 3]) {
      const got = new Set(tree.search(q, maxD).map((r) => r.normalizedText));
      const want = new Set(words.filter((w) => levenshtein(q, w) <= maxD));
      const same = got.size === want.size && [...got].every((w) => want.has(w));
      if (!same) fail++;
      if (!same) console.log(`✗ BKTree search ${q}/${maxD}: ${[...got]} vs linear ${[...want]}`);
    }
  }
  console.log("✓ BKTree search ≡ linear scan (5 queries × 3 distances)");
}

// rescoring behavior
const r = createVocabularyRescorer(["NVIDIA", "Newrez", "Bank of America", { text: "GPT-4", aliases: ["gpt four"] }]);
eq(r.rescore("we use invidia gpus"), "we use NVIDIA gpus", "single-word fuzzy");
eq(r.rescore("my new res mortgage"), "my Newrez mortgage", "2-word compound");
eq(r.rescore("i bank with bank of amerika now"), "i bank with Bank of America now", "phrase fuzzy");
eq(r.rescore("we asked gpt four about it"), "we asked GPT-4 about it", "alias phrase");
eq(r.rescore("Invidia makes chips"), "NVIDIA makes chips", "casing from canonical");
eq(r.rescore("the quick brown fox jumps"), "the quick brown fox jumps", "no false positive");
eq(r.rescore(""), "", "empty input");
eq(createVocabularyRescorer([]).size, 0, "empty vocab size");

// punctuation preserved around replacements
eq(r.rescore("Is invidia, the chip maker?"), "Is NVIDIA, the chip maker?", "punctuation preserved");

console.log(fail === 0 ? "VOCAB RESCORER OK" : `VOCAB RESCORER ${fail} FAILURES`);
process.exit(fail === 0 ? 0 : 1);
