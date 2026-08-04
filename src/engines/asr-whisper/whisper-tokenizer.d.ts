export function makeWhisperTokenizer(vocab: Record<string, number>): { decode(ids: number[]): string };
