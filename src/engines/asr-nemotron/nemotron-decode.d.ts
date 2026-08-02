export function nemotronTranscribe(o: {
  ort: any;
  encoder: any;
  decoder: any;
  joint: any;
  preprocessor: { process(audio: Float32Array): { features: Float32Array; length: number } };
  tokenizer: { decode(ids: number[]): string };
  audio: Float32Array;
  langId?: number;
}): Promise<{ text: string; tokenIds: number[]; chunks: number }>;

export function makeNemotronTokenizer(vocabText: string): {
  id2token: string[];
  decode(ids: number[]): string;
};
