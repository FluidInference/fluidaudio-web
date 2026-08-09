export function loadEouDecoder(bin: Float32Array, man: any): any;
export function eouDecode(
  dec: any,
  frames: Float32Array,
  Tenc: number,
  maxSymbols?: number,
): { ids: number[]; idFrames: number[]; events: { type: string; frame: number }[] };
export function createEouStream(dec: any): any;
export function eouDecodeCont(
  dec: any,
  st: any,
  frames: Float32Array,
  Tenc: number,
  frameOffset?: number,
  maxSymbols?: number,
): { ids: number[]; idFrames: number[]; events: { type: string; frame: number }[] };
export function loadEouWasmDecoder(wasmBytes: ArrayBuffer | Uint8Array, bin: Float32Array, man: any): Promise<any>;
export function eouWasmReset(wd: any): void;
export function eouWasmDecodeCont(
  wd: any,
  framesProj: Float32Array,
  Tenc: number,
  frameOffset?: number,
  maxSymbols?: number,
): { ids: number[]; idFrames: number[]; events: { type: string; frame: number }[] };
