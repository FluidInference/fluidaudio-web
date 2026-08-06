import type { GpuContext, GpuTensor } from "./compute.js";

/** input_ids → ALBERT → bert_encoder (768→512) → d_en [seq,512]. */
export function textEncoding(
  ctx: GpuContext,
  ids: Int32Array | number[],
  albertW: Record<string, GpuTensor | Float32Array | number>,
  beW: GpuTensor,
  beB: GpuTensor,
): GpuTensor;
