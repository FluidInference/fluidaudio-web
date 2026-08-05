// Whisper BPE detokenizer (GPT-2 byte-level). Maps output ids → token strings →
// bytes (reversing GPT-2's bytes↔unicode table) → UTF-8 text. Text tokens are
// < 50257; specials (EOT/lang/task/timestamps ≥ 50257) are dropped.

const EOT = 50257;

// GPT-2 bytes_to_unicode, inverted (unicode char → byte).
function byteDecoder() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++)
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  const dec = new Map();
  for (let i = 0; i < bs.length; i++) dec.set(String.fromCodePoint(cs[i]), bs[i]);
  return dec;
}

/** @param {Record<string,number>} vocab token-string → id (vocab.json). */
export function makeWhisperTokenizer(vocab) {
  const id2tok = [];
  for (const [tok, id] of Object.entries(vocab)) id2tok[id] = tok;
  const bdec = byteDecoder();
  return {
    /** @param {number[]} ids */
    decode(ids) {
      const bytes = [];
      for (const id of ids) {
        if (id >= EOT) continue; // specials
        const tok = id2tok[id];
        if (tok === undefined) continue;
        for (const ch of tok) {
          const b = bdec.get(ch);
          if (b !== undefined) bytes.push(b);
        }
      }
      return new TextDecoder("utf-8").decode(new Uint8Array(bytes));
    },
  };
}
