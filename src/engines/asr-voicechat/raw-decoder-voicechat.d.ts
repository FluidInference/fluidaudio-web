export const VC_BLANK: number;
export const VC_LOGITS: number;
export function loadVoicechatDecoder(bin: Float32Array, man: any): any;
export function createVoicechatStream(dec: any): any;
export function voicechatDecodeCont(dec: any, st: any, framesProj: Float32Array, Tenc: number, maxSymbols?: number): { ids: number[]; idFrames: number[] };
export function voicechatDecode(dec: any, framesProj: Float32Array, Tenc: number, maxSymbols?: number): { ids: number[]; idFrames: number[] };
