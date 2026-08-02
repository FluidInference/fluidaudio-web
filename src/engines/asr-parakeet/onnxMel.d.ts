export class OnnxMelPreprocessor {
  nMels: number;
  session: any;
  constructor(ort: any, session: any, nMels?: number);
  process(audio: Float32Array): Promise<{ features: Float32Array; length: number }>;
}
