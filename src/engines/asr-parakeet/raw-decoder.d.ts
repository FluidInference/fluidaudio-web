import type { GpuContext } from "../../gpu/compute.js";
export function loadParakeetDecoder(bin: Float32Array, man: any, ctx?: GpuContext | null): any;
export function newDecoderState(): any;
export function predict(dec: any, token: number, state: any): { decOut: Float32Array; state: any };
export function joint(dec: any, encFrame: Float32Array, decOut: Float32Array): Float32Array;
export function tdtGreedy(dec: any, frames: Float32Array, Tenc: number, maxSymbols?: number): { ids: number[]; idFrames: number[] };
export function tdtGreedyGpu(ctx: GpuContext, dec: any, framesGpu: any, Tenc: number, maxSymbols?: number): Promise<{ ids: number[]; idFrames: number[] }>;
