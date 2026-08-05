import type { GpuContext, GpuTensor } from "./compute";

/** CPU embeddings: word[ids] + pos + tok[0], then LayerNorm(128) -> [seq,128]. */
export function embed(ids: Int32Array | number[], weights: Record<string, Float32Array>): Float32Array;

/** Run the ALBERT transformer stack on the GPU. Returns [seq, 768]. */
export function albertForward(
  ctx: GpuContext,
  embTensor: GpuTensor,
  weights: Record<string, GpuTensor | Float32Array>,
  seq: number,
  layers?: number,
): GpuTensor;

export const ALBERT_DIMS: {
  HIDDEN: number;
  HEADS: number;
  HEAD_DIM: number;
  EMBED: number;
  EPS: number;
  LAYERS: number;
};
