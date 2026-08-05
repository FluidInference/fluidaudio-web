export function loadSortformerHead(ctx: any, bin: Float32Array, man: any): any;
export function sortformerHead(ctx: any, head: any, framesGpu: any, Tsub: number): Promise<Float32Array>;
export function predsToSegments(
  preds: Float32Array, frames: number, frameSec: number,
  opts?: { threshold?: number; minSpeechSec?: number; mergeGapSec?: number }
): { speaker: number; start: number; end: number }[];
export function mergeWindowPreds(windows: { preds: Float32Array; frames: number }[], ovlFrames: number[]): { preds: Float32Array; frames: number };
