export function loadNemotronDecoder(bin: Float32Array, man: any): any;
export function nemotronDecode(dec: any, frames: Float32Array, Tenc: number, maxSymbols?: number): { ids: number[]; idFrames: number[] };
export function loadPromptKernel(f32bin: Float32Array, man: any): any;
export function applyPromptKernel(pk: any, frames: Float32Array, Tsub: number, langId?: number): Float32Array;
export function createNemotronStream(dec: any): any;
export function nemotronDecodeCont(dec: any, st: any, frames: Float32Array, Tenc: number, maxSymbols?: number): { ids: number[]; idFrames: number[] };
