import type { JsPreprocessor } from "./mel";
import type { ParakeetTokenizer } from "./tokenizer";

export function transcribeTdt(o: {
  ort: any;
  encoder: any;
  decoder: any;
  preprocessor: JsPreprocessor;
  tokenizer: ParakeetTokenizer;
  audio: Float32Array;
}): Promise<{ text: string; tokenIds: number[]; frames: number }>;
