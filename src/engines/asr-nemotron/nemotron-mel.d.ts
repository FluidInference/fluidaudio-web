export class JsPreprocessor {
  nMels: number;
  constructor(opts?: { nMels?: number });
  /** NA log-mel (no CMVN); features = mel-major [128*T] raw log-mel. */
  process(audio: Float32Array): { features: Float32Array; length: number };
}
