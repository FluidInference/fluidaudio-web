export function loadWhisperDecoder(ctx: any, bin: Float32Array, man: any): any;
export function whisperCrossKV(ctx: any, dec: any, encGpu: any): any[];
export function whisperDecodeStep(ctx: any, dec: any, kv: any[], tokens: number[]): Promise<Float32Array>;
