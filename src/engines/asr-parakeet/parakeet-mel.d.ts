export class ParakeetMel {
  constructor(nMels?: number);
  process(audio: Float32Array): { features: Float32Array; length: number };
}
