export const MEL_CONSTANTS: { HOP_LENGTH: number; SAMPLE_RATE: number; [k: string]: number };
export class JsPreprocessor {
  nMels: number;
  constructor(opts?: { nMels?: number });
  /** audio → { features: [nMels*T] mel-major log-mel (CMVN'd), length: valid T } */
  process(audio: Float32Array): { features: Float32Array; length: number };
}
