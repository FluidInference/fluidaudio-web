export function createEncodeStream(
  ctx: any,
  enc: any,
  opts?: { proj?: { w: any; b: any } | null; post?: ((ctx: any, x: any) => any) | null; lookaheadChunks?: number },
): any;
export function encodeStreamPush(ctx: any, st: any, mel: Float32Array, count: number, opts?: { maxChunk?: number }): Promise<Float32Array | null>;
export function encodeStreamFlush(ctx: any, st: any): Promise<Float32Array | null>;
export function disposeEncodeStream(ctx: any, st: any): void;
