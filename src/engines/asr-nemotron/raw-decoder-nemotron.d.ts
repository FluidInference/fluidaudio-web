export function loadNemotronDecoder(bin: Float32Array, man: any): any;
export function nemotronDecode(dec: any, frames: Float32Array, Tenc: number, maxSymbols?: number): { ids: number[]; idFrames: number[] };
export function loadPromptKernel(f32bin: Float32Array, man: any): any;
export function applyPromptKernel(pk: any, frames: Float32Array, Tsub: number, langId?: number): Float32Array;
export function createNemotronStream(dec: any): any;
export function nemotronDecodeCont(dec: any, st: any, frames: Float32Array, Tenc: number, maxSymbols?: number): { ids: number[]; idFrames: number[] };
export function loadNemoWasmDecoder(wasmBytes: ArrayBuffer | Uint8Array, bin: Float32Array, man: any, opts?: { int8?: boolean }): Promise<any>;
export function nemoWasmReset(wd: any): void;
export function nemoWasmDecodeCont(wd: any, framesProj: Float32Array, Tenc: number, maxSymbols?: number): { ids: number[]; idFrames: number[] };
