export class ParakeetTokenizer {
  id2token: string[];
  blankId: number;
  sanitized: string[];
  constructor(id2token: string[]);
  static fromVocabText(text: string): ParakeetTokenizer;
  decode(ids: number[]): string;
}
