export class EnglishLexicon {
  map: Record<string, string>;
  constructor(map: Record<string, string>);
  static load(fetchCached: (url: string, onProgress: undefined, label: string) => Promise<Uint8Array>): Promise<EnglishLexicon>;
  phonemize(text: string): { phonemes: string; coverage: number };
}
