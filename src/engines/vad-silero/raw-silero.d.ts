export interface SileroWeights {
  stftBasis: Float32Array;
  enc: { w: Float32Array; b: Float32Array; cout: number; cin: number; k: number; stride: number }[];
  lstmW: Float32Array;
  lstmR: Float32Array;
  lstmB: Float32Array;
  decW: Float32Array;
  decB: Float32Array;
}

export interface SileroManifest {
  [key: string]: { dims: number[]; offset: number; len: number };
}

export function makeSileroWeights(bin: Float32Array, manifest: SileroManifest): SileroWeights;

export function sileroForward(x: Float32Array, state: Float32Array, w: SileroWeights): { prob: number; state: Float32Array };
