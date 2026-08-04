export function loadParakeetDecoder(bin: Float32Array, man: any): any;
export function newDecoderState(): any;
export function predict(dec: any, token: number, state: any): { decOut: Float32Array; state: any };
export function joint(dec: any, encFrame: Float32Array, decOut: Float32Array): Float32Array;
export function tdtGreedy(dec: any, frames: Float32Array, Tenc: number, maxSymbols?: number): { ids: number[]; idFrames: number[] };
