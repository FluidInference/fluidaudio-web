import type { GpuContext, GpuTensor } from "../../gpu/compute.js";
export function loadParakeetEncoder(ctx: GpuContext, bin: Float32Array | Uint8Array, man: any): any;
export function parakeetEncode(
  ctx: GpuContext, enc: any, mel: Float32Array, T: number
): Promise<{ data: Float32Array; dims: [number, number, number]; framesGpu: GpuTensor; Tsub: number }>;
