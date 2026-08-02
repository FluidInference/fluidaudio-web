import type { ParakeetTokenizer } from "./tokenizer";

/** Feature extractor: audio → mel-major log-mel features + valid frame count. */
export interface Preprocessor {
  nMels: number;
  process(audio: Float32Array): Promise<{ features: Float32Array; length: number }>;
}

export function transcribeTdt(o: {
  ort: any;
  encoder: any;
  decoder: any;
  preprocessor: Preprocessor;
  tokenizer: ParakeetTokenizer;
  audio: Float32Array;
}): Promise<{ text: string; tokenIds: number[]; frames: number }>;
