export function loadWasmDecoder(
  wasmBytes: ArrayBuffer | Uint8Array, decBin: Float32Array, decMan: any
): Promise<{ ex: any; mark: number; blankId: number; vocab: number }>;
export function wasmDecode(dec: any, frames: Float32Array, Tenc: number): { ids: number[]; idFrames: number[] };
export function wasmDecodeProj(dec: any, proj: Float32Array, Tenc: number): { ids: number[]; idFrames: number[] };
