// Misaki-lexicon-first English G2P (matches FluidAudio's KokoroAne frontend).
// Uses the preprocessed us_lexicon_cache.json (word -> IPA token array, already
// in Kokoro's exact phoneme set) from the kokoro HF repo. Lexicon lookup avoids
// espeak for known words; callers fall back to kokoro-js's espeak generate() for
// out-of-vocabulary text.

const LEX_URL = "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main/us_lexicon_cache.json";

export class EnglishLexicon {
  constructor(map) {
    this.map = map; // { word: "joined IPA" }
  }

  static async load(fetchCached) {
    const bytes = await fetchCached(LEX_URL, undefined, "us_lexicon_cache.json");
    const raw = JSON.parse(new TextDecoder().decode(bytes));
    const lower = raw.lower || raw;
    const map = Object.create(null);
    for (const [word, toks] of Object.entries(lower)) {
      map[word] = Array.isArray(toks) ? toks.join("") : toks;
    }
    return new EnglishLexicon(map);
  }

  /** Phonemize via lexicon. Returns { phonemes, coverage } (1 = every word hit). */
  phonemize(text) {
    const words = text.toLowerCase().split(/\s+/).filter(Boolean);
    let hits = 0;
    const parts = [];
    for (const w of words) {
      const key = w.replace(/[^a-z']/g, "");
      if (key && this.map[key]) {
        parts.push(this.map[key]);
        hits++;
      }
    }
    return { phonemes: parts.join(" "), coverage: words.length ? hits / words.length : 0 };
  }
}
