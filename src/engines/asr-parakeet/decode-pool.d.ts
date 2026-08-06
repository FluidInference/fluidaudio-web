export interface DecodePool {
  size: number;
  decode(frames: Float32Array, Tenc: number): Promise<{ ids: number[]; idFrames: number[] }>;
  failAll(err: unknown): void;
  terminate(): void;
}
export function createDecodePool(
  workers: Array<{ postMessage(m: any, t?: any[]): void; setHandler(f: (m: any) => void): void; terminate(): void }>,
): DecodePool;

type Shim = { postMessage(m: any, t?: any[]): void; setHandler(f: (m: any) => void): void; terminate(): void };
export function browserWorkerShim(w: Worker): Shim;
export function nodeWorkerShim(w: any): Shim;
export function initDecodeWorker(
  post: (m: any) => void,
  once: (ok: (m: any) => void, err: (e: any) => void) => void,
  payload: Record<string, any>,
): Promise<void>;
