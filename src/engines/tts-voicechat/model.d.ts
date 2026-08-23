export function loadVoicechatTtsModel(ctx: any, W: any, cfg: any): any;
export function synthesizeCodes(
  ctx: any,
  model: any,
  frameTokens: number[],
  tokenChars: Map<number, number[]>,
  rvq: Float32Array,
  opts?: {
    deterministic?: boolean;
    seed?: number;
    rvqNormSq?: Float64Array;
    captureWarm?: boolean;
    captureSteps?: number;
    syncTrace?: boolean;
    onProgress?: (done: number, total: number) => void;
  },
): Promise<{ codes: Int32Array[]; trace: any }>;
export function depthsum(rvq: Float32Array, code: ArrayLike<number>, latent: number, codebook: number): Float32Array;
export function rvqNorms(rvq: Float32Array, numQ: number, codebook: number, latent: number): Float64Array;
export function casCond(ctx: any, model: any, tokenChars: number[], tokId: number, inCas: boolean): Promise<Float32Array>;
export function makeRng(seed: number): { uniform(): number; gauss(): number; gumbel(): number };
