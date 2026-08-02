// Chinese G2P for Kokoro-zh. hanzi → pinyin (pinyin-pro) → IPA via a table
// PRECOMPUTED from misaki's own pinyin_to_ipa + retone (so the IPA is byte-exact
// to what Kokoro-zh was trained on — no reimplementation of misaki's transcription).
// Polyphone accuracy is currently pinyin-pro's dictionary/context heuristic;
// g2pW ONNX can replace the pinyin stage later for higher accuracy (docs/KOKORO_ZH.md).

import { pinyin } from "pinyin-pro";
import TABLE from "./pinyin-ipa.json" with { type: "json" };

// misaki zh.py map_punctuation
const PUNCT = {
  "、": ", ", "，": ", ", "。": ". ", "．": ". ", "！": "! ", "：": ": ",
  "；": "; ", "？": "? ", "《": " “", "》": "” ", "「": " “",
  "」": "” ", "【": " “", "】": "” ", "（": " (", "）": ") ",
};

/**
 * @param {string} text
 * @returns {{ phonemes: string, coverage: number }} coverage = fraction of
 *   hanzi syllables resolved to IPA (1 = all).
 */
export function chineseToIpa(text) {
  // pinyin-pro segments + resolves polyphones by context; non-Chinese runs are
  // returned verbatim (nonZh: "consecutive").
  const arr = pinyin(text, { toneType: "num", type: "array", nonZh: "consecutive" });
  const parts = [];
  let zh = 0;
  let hit = 0;
  for (const seg of arr) {
    if (PUNCT[seg]) { parts.push(PUNCT[seg]); continue; }
    // A zh syllable from pinyin-pro looks like "ni3" / "men0" (neutral = 0).
    const m = /^([a-zü]+)([0-4])$/.exec(seg);
    if (m) {
      zh++;
      const key = m[1] + (m[2] === "0" ? "5" : m[2]); // neutral 0 → table's 5
      const ipa = TABLE[key] || TABLE[m[1] + "5"];
      if (ipa) { parts.push(ipa); hit++; continue; }
    }
    // Non-Chinese run (Latin, digits, spaces) — pass through for espeak/lexicon
    // handling upstream, or drop pure whitespace.
    if (/\S/.test(seg)) parts.push(seg);
    else parts.push(" ");
  }
  return { phonemes: parts.join(""), coverage: zh ? hit / zh : 0 };
}
