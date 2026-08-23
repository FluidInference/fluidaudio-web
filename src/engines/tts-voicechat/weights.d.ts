export class WeightStore {
  constructor(
    bins: Uint8Array | Uint8Array[],
    man: { shards: number; tensors: Record<string, { dims: number[]; dtype: string; bin: number; byteOffset: number; count: number }> },
  );
  f32(name: string): Float32Array;
  u8(name: string): Uint8Array;
  dims(name: string): number[];
  mat(ctx: any, name: string): any;
}
