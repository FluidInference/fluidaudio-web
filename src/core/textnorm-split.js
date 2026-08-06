// Sentence-preserving TN wrapper (plain JS so the node gate exercises the
// exact shipping logic). tnNormalizeSentenceLang drops sentence-final
// punctuation and merges sentence boundaries (verified: "It costs $4.50. I
// bought 3." → one run-on with a degraded "four point five zero" reading), so
// the text is split at terminators-followed-by-whitespace (decimals like 4.50
// never match), each sentence normalized alone, and the terminators reattached
// — Kokoro's G2P maps them to pause/intonation tokens.

/** @param {(s: string) => string} normalizeOne @param {string} text */
export function tnBySentence(normalizeOne, text) {
  const out = [];
  const re = /[.!?]+(?=\s|$)/g;
  let last = 0;
  let match;
  const proc = (seg, term) => {
    const lead = (seg.match(/^\s*/) || [""])[0];
    const core = seg.slice(lead.length);
    out.push(lead + (core ? normalizeOne(core) : "") + term);
  };
  while ((match = re.exec(text))) {
    proc(text.slice(last, match.index), match[0]);
    last = match.index + match[0].length;
  }
  if (last < text.length) proc(text.slice(last), "");
  return out.join("");
}
