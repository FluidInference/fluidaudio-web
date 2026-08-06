import type { GpuContext, GpuTensor } from "../../gpu/compute.js";
export function loadParakeetEncoder(ctx: GpuContext, bin: Float32Array | Uint8Array, man: any, cfgOverride?: any): any;
export function parakeetEncode(
  ctx: GpuContext,
  enc: any,
  mel: Float32Array,
  T: number,
  wantData?: boolean,
): Promise<{ data?: Float32Array; dims: [number, number, number]; framesGpu: GpuTensor; Tsub: number }>;
export function parakeetEncodeBatch(
  ctx: GpuContext,
  enc: any,
  mels: Float32Array[],
  wantData?: boolean,
  post?: ((x: any) => any) | null,
): Promise<{ framesGpu: any; Tsub: number; W: number; D: number; dims: number[]; data?: Float32Array; staged?: any }>;
