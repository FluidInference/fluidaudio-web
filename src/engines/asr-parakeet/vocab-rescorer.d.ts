export interface VocabTerm {
  text: string;
  aliases?: string[];
  minSimilarity?: number;
}
export interface CandidateMatch {
  term: VocabTerm;
  similarity: number;
  spanLength: number;
}
export interface VocabularyRescorer {
  rescore(text: string): string;
  findCandidates(normalizedWord: string, adjacentNormalized: string[]): CandidateMatch[];
  size: number;
  minSimilarity: number;
}
export function levenshtein(a: string, b: string): number;
export function stringSimilarity(a: string, b: string): number;
export function lengthPenalizedSimilarity(compound: string, vocabTerm: string): number;
export function normalizeForSimilarity(text: string): string;
export class BKTree {
  constructor(entries: Array<{ term: VocabTerm; normalizedText: string }>);
  search(query: string, maxDistance: number): Array<{ term: VocabTerm; normalizedText: string; distance: number }>;
  count: number;
}
export function createVocabularyRescorer(vocabulary: Array<string | VocabTerm>, opts?: { minSimilarity?: number }): VocabularyRescorer;
