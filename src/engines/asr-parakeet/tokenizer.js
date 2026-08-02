// Parakeet SentencePiece tokenizer/decoder. Parses a NeMo `vocab.txt`
// (`<token> <id>` per line), maps the `▁` word-boundary marker to a space, and
// joins. Blank id is the index of `<blk>`.

export class ParakeetTokenizer {
  /** @param {string[]} id2token index=id, value=token */
  constructor(id2token) {
    this.id2token = id2token;
    this.blankId = id2token.findIndex((t) => t === "<blk>");
    if (this.blankId === -1) this.blankId = id2token.length - 1;
    this.sanitized = id2token.map((t) => (t ? t.replace(/▁/g, " ") : t));
  }

  /** @param {string} text contents of vocab.txt */
  static fromVocabText(text) {
    const id2token = [];
    for (const line of text.split(/\r?\n/)) {
      if (!line) continue;
      const [tok, idStr] = line.split(/\s+/);
      const id = parseInt(idStr, 10);
      if (!Number.isNaN(id) && tok) id2token[id] = tok;
    }
    return new ParakeetTokenizer(id2token);
  }

  /** @param {number[]} ids @returns {string} */
  decode(ids) {
    let out = "";
    for (const id of ids) {
      if (id === this.blankId) continue;
      const t = this.sanitized[id];
      if (t !== undefined) out += t;
    }
    return out.trim().replace(/\s+/g, " ");
  }
}
