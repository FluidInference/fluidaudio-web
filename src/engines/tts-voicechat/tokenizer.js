// Nemotron-Nano-9B-v2 tokenizer (Tekken-style byte-level BPE) — encode only.
// Mirrors HF tokenizers: Split(regex, Isolated) → ByteLevel(no prefix space) →
// BPE with ignore_merges (pre-tokens that are whole vocab entries bypass the
// merge loop). Special/added tokens are NOT matched from text — the TTS driver
// injects <s>/</s>/<SPECIAL_12> by id, and synthesis input is plain text.
// Oracle: token ids for the reference text match the HF fast tokenizer
// (scripts/voicechat-tts-reference.py meta.text_ids).

/** GPT-2 byte → unicode char map (the byte-level alphabet the vocab is written in). */
function byteToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const map = new Array(256);
  for (let i = 0; i < bs.length; i++) map[bs[i]] = String.fromCharCode(cs[i]);
  return map;
}

export class VoicechatTokenizer {
  /** @param {{vocab: string[], merges: string[], ignoreMerges: boolean, splitRegex: string}} tok */
  constructor(tok) {
    this.vocab = new Map(tok.vocab.map((t, i) => [t, i]));
    this.ranks = new Map(tok.merges.map((m, i) => [m, i]));
    this.ignoreMerges = tok.ignoreMerges;
    // Rust regex → JS: (?!\S) negative lookahead is supported; \p classes need /u.
    this.split = new RegExp(tok.splitRegex, "gu");
    this.b2u = byteToUnicode();
    this.enc = new TextEncoder();
  }

  /** BPE over one byte-level word (array of alphabet chars) → token strings. */
  bpe(chars) {
    let parts = chars.slice();
    for (;;) {
      let best = -1,
        bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      parts = [...parts.slice(0, best), parts[best] + parts[best + 1], ...parts.slice(best + 2)];
    }
    return parts;
  }

  /** @param {string} text @returns {number[]} token ids (no specials) */
  encode(text) {
    const ids = [];
    for (const m of text.matchAll(this.split)) {
      const bytes = this.enc.encode(m[0]);
      let mapped = "";
      for (const b of bytes) mapped += this.b2u[b];
      if (this.ignoreMerges && this.vocab.has(mapped)) {
        ids.push(this.vocab.get(mapped));
        continue;
      }
      for (const piece of this.bpe([...mapped])) {
        const id = this.vocab.get(piece);
        if (id === undefined) throw new Error(`voicechat tokenizer: piece not in vocab: ${JSON.stringify(piece)}`);
        ids.push(id);
      }
    }
    return ids;
  }
}
