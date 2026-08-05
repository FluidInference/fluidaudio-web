export class WhisperMel {
  constructor(melFilters: Float32Array);
  process(audio: Float32Array): { features: Float32Array; length: number };
}
