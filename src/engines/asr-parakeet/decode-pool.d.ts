export interface DecodePool {
  size: number;
  decode(frames: Float32Array, Tenc: number): Promise<{ ids: number[]; idFrames: number[] }>;
  terminate(): void;
}
export function createDecodePool(
  workers: Array<{ postMessage(m: any, t?: any[]): void; setHandler(f: (m: any) => void): void; terminate(): void }>
): DecodePool;
