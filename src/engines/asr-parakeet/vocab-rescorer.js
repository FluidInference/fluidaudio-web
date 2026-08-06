// Custom-vocabulary rescoring — JS port of FluidAudio's BK-tree text path
// (Sources/FluidAudio/ASR/Parakeet/SlidingWindow/CustomVocabulary: BKTree.swift,
// VocabularyRescorer+CandidateMatching.swift, +Utilities.swift,
// ContextBiasingConstants.swift). Pure algorithm, no GPU/model work: a user
// word list ("NVIDIA", "Newrez", "Bank of America", …) is indexed in a BK-tree
// and fuzzy-matched against transcript words — including 2-/3-word compounds
// ("new"+"res" → "Newrez") and multi-word phrases — replacing close matches
// with the canonical spelling. The Swift token-level rescorer (TDT logit
// integration) is NOT ported; this is the text-level pass.

// ── string utilities (ports of StringUtils / VocabularyRescorer+Utilities) ──

/** Levenshtein distance, iterative two-row. */
export function levenshtein(a, b) {
  if (a === b) return 0;
  const m = a.length,
    n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  let prev = new Int32Array(n + 1),
    cur = new Int32Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    cur[0] = i;
    const ca = a.charCodeAt(i - 1);
    for (let j = 1; j <= n; j++) {
      const cost = ca === b.charCodeAt(j - 1) ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    [prev, cur] = [cur, prev];
  }
  return prev[n];
}

/** similarity = 1 - dist/maxLen (inputs assumed already lowercased/normalized). */
export function stringSimilarity(a, b) {
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1;
  return 1 - levenshtein(a, b) / maxLen;
}

/** Compound similarity with sqrt length-ratio penalty (prefix/suffix guard). */
export function lengthPenalizedSimilarity(compound, vocabTerm) {
  const base = stringSimilarity(compound, vocabTerm);
  const ratio = Math.min(compound.length, vocabTerm.length) / Math.max(compound.length, vocabTerm.length);
  return base * Math.sqrt(ratio);
}

/** Lowercase, strip punctuation (keep letters/digits/'/-), collapse whitespace. */
export function normalizeForSimilarity(text) {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}'-]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// ── BK-tree (port of BKTree.swift; iterative build to avoid deep recursion) ──

export class BKTree {
  /** entries: [{term, normalizedText}] */
  constructor(entries) {
    this.count = entries.length;
    this.root = null;
    for (const e of entries) this._insert(e);
  }
  _insert(e) {
    if (!this.root) {
      this.root = { ...e, children: new Map() };
      return;
    }
    let node = this.root;
    for (;;) {
      const d = levenshtein(e.normalizedText, node.normalizedText);
      const child = node.children.get(d);
      if (!child) {
        node.children.set(d, { ...e, children: new Map() });
        return;
      }
      node = child;
    }
  }
  /** All entries within maxDistance of query (query pre-normalized). */
  search(query, maxDistance) {
    if (!this.root) return [];
    const results = [];
    const stack = [this.root];
    while (stack.length) {
      const node = stack.pop();
      const distance = levenshtein(query, node.normalizedText);
      if (distance <= maxDistance) results.push({ term: node.term, normalizedText: node.normalizedText, distance });
      const minEdge = Math.max(0, distance - maxDistance),
        maxEdge = distance + maxDistance;
      for (const [edge, child] of node.children) if (edge >= minEdge && edge <= maxEdge) stack.push(child);
    }
    return results;
  }
}

// ── constants (ContextBiasingConstants.swift) ──
const BKTREE_MAX_DISTANCE = 3;
const LARGE_VOCAB = 10;
const EXTRA_LARGE_VOCAB = 100;

/** Swift rescorerConfig(forVocabSize:) — stricter thresholds for bigger lists. */
function defaultMinSimilarity(size) {
  if (size > EXTRA_LARGE_VOCAB) return 0.6;
  if (size > LARGE_VOCAB) return 0.55;
  return 0.5;
}

/** Multi-word spans use a slightly higher floor (Utilities.requiredSimilarity). */
function requiredSimilarity(minSimilarity, spanLength) {
  return spanLength >= 2 ? Math.max(minSimilarity, 0.55) : minSimilarity;
}

function preserveCapitalization(original, replacement) {
  if (!original) return replacement;
  const f = original[0];
  if (f === f.toUpperCase() && f !== f.toLowerCase() && replacement[0] === replacement[0].toLowerCase()) {
    return replacement[0].toUpperCase() + replacement.slice(1);
  }
  return replacement;
}

// ── rescorer ──

/**
 * @param {(string | {text: string, aliases?: string[], minSimilarity?: number})[]} vocabulary
 * @param {{minSimilarity?: number}} [opts]
 */
export function createVocabularyRescorer(vocabulary, opts = {}) {
  const terms = vocabulary.map((t) => (typeof t === "string" ? { text: t } : t)).filter((t) => t.text && normalizeForSimilarity(t.text));
  const minSimilarity = opts.minSimilarity ?? defaultMinSimilarity(terms.length);

  // Index every normalized form (canonical + aliases) — all point at the term.
  const entries = [];
  const exactForms = new Set();
  for (const term of terms) {
    for (const raw of [term.text, ...(term.aliases ?? [])]) {
      const normalized = normalizeForSimilarity(raw);
      if (!normalized) continue;
      entries.push({ term, normalizedText: normalized });
      exactForms.add(normalized);
    }
  }
  const tree = new BKTree(entries);
  // Lowest per-term threshold widens the BK-tree distance bound so per-term
  // overrides aren't pruned before filtering (CandidateMatching searchFloor).
  const searchFloor = Math.min(minSimilarity, ...terms.map((t) => t.minSimilarity ?? minSimilarity));

  const threshold = (term) => term.minSimilarity ?? minSimilarity;

  /** Port of findCandidateTermsForWord (BK-tree path). */
  function findCandidates(normalizedWord, adjacentNormalized) {
    if (!normalizedWord) return [];
    const candidates = [];
    const push = (r, similarity, spanLength) => {
      if (similarity >= threshold(r.term)) candidates.push({ term: r.term, similarity, spanLength });
    };

    // 1. single word
    const maxLen1 = Math.max(normalizedWord.length, 3);
    const maxDist1 = Math.min(BKTREE_MAX_DISTANCE, Math.floor((1 - searchFloor) * maxLen1));
    for (const r of tree.search(normalizedWord, maxDist1)) push(r, stringSimilarity(normalizedWord, r.normalizedText), 1);

    // 2. two-word compound ("new"+"res" → "newres" ≈ "newrez")
    const w2 = adjacentNormalized[0];
    if (w2) {
      const compound2 = normalizedWord + w2;
      const maxDist2 = Math.min(BKTREE_MAX_DISTANCE, Math.floor((1 - searchFloor) * Math.max(compound2.length, 3)));
      // compounds only match SINGLE-word vocab terms (Swift linear-scan scoping)
      for (const r of tree.search(compound2, maxDist2)) if (!r.normalizedText.includes(" ")) push(r, lengthPenalizedSimilarity(compound2, r.normalizedText), 2);
    }

    // 3. three-word compound (len ≥ 6)
    const w3 = adjacentNormalized[1];
    if (w2 && w3) {
      const compound3 = normalizedWord + w2 + w3;
      if (compound3.length >= 6) {
        const maxDist3 = Math.min(BKTREE_MAX_DISTANCE, Math.floor((1 - searchFloor) * compound3.length));
        for (const r of tree.search(compound3, maxDist3))
          if (!r.normalizedText.includes(" ")) push(r, lengthPenalizedSimilarity(compound3, r.normalizedText), 3);
      }
    }

    // 4. multi-word phrases ("bank of america")
    if (adjacentNormalized.length) {
      for (let spanLen = 2; spanLen <= Math.min(4, adjacentNormalized.length + 1); spanLen++) {
        const phrase = [normalizedWord, ...adjacentNormalized.slice(0, spanLen - 1)].join(" ");
        const maxDistP = Math.min(BKTREE_MAX_DISTANCE + 1, Math.floor((1 - searchFloor) * Math.max(phrase.length, 3)));
        // phrases only match MULTI-word vocab terms (Swift linear-scan scoping)
        for (const r of tree.search(phrase, maxDistP)) if (r.normalizedText.includes(" ")) push(r, stringSimilarity(phrase, r.normalizedText), spanLen);
      }
    }

    // similarity desc, then longer spans first
    return candidates.sort((a, b) => b.similarity - a.similarity || b.spanLength - a.spanLength);
  }

  /** Rewrite a transcript: fuzzy vocab matches become canonical spellings. */
  function rescore(text) {
    if (!terms.length || !text) return text;
    // Word tokens with their positions; punctuation/whitespace stay untouched.
    const tokens = [...text.matchAll(/[\p{L}\p{N}'-]+/gu)];
    const words = tokens.map((m) => normalizeForSimilarity(m[0]));
    let out = "";
    let cursor = 0;
    for (let i = 0; i < tokens.length;) {
      const norm = words[i];
      // Exact vocab words pass through (already correct — but recase below via span 1 exact match).
      const adjacent = words.slice(i + 1, i + 4);
      const cands = norm ? findCandidates(norm, adjacent) : [];
      const best = cands.find((c) => c.similarity >= requiredSimilarity(minSimilarity, c.spanLength));
      if (!best || (best.spanLength === 1 && best.similarity < 1 && exactForms.has(norm))) {
        // No match, or the word is itself a vocab form and the "match" is a
        // DIFFERENT term — leave it alone.
        i++;
        continue;
      }
      const startTok = tokens[i];
      const endTok = tokens[i + best.spanLength - 1];
      if (!endTok) {
        i++;
        continue;
      }
      out += text.slice(cursor, startTok.index);
      out += preserveCapitalization(startTok[0], best.term.text);
      cursor = endTok.index + endTok[0].length;
      i += best.spanLength;
    }
    out += text.slice(cursor);
    return out;
  }

  return { rescore, findCandidates, size: terms.length, minSimilarity };
}
