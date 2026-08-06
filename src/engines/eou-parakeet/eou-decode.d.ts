export interface EouTokenizer {
  id2token: string[];
  sanitized: string[];
  blankId: number;
  eouId: number;
  eobId: number;
  decode(ids: number[]): string;
}

export function makeEouTokenizer(vocabText: string): EouTokenizer;

export function eouTranscribe(o: {
  ort: any;
  encoder: any;
  decoder: any;
  preprocessor: {
    nMels: number;
    process(audio: Float32Array): Promise<{ features: Float32Array; length: number }> | { features: Float32Array; length: number };
  };
  tokenizer: EouTokenizer;
  audio: Float32Array;
}): Promise<{
  text: string;
  tokenIds: number[];
  events: { type: string; time: number }[];
  frames: number;
  metrics: { melMs: number; encodeMs: number; decodeMs: number; totalMs: number };
}>;
