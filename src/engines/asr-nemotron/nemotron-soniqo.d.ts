export function makeSoniqoTokenizer(vocabJson: Record<string, string>): { decode(ids: number[]): string };
export function soniqoLangPrompt(languages: any, lang: string): number;
export function soniqoTranscribe(o: {
  ort: any; encoder: any; decoder: any; joint: any;
  preprocessor: { nMels: number; process(a: Float32Array): Promise<{ features: Float32Array; length: number }> | { features: Float32Array; length: number } };
  tokenizer: { decode(ids: number[]): string };
  audio: Float32Array; langPrompt?: number;
}): Promise<{ text: string; tokenIds: number[]; metrics: { totalMs: number } }>;
