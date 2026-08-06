export function transcribeWindowed(
  ctx: any,
  enc: any,
  dec: any,
  mel: any,
  projW: any,
  projB: any,
  samples: Float32Array,
  opts?: { sampleRate?: number; windowSec?: number; overlapSec?: number; wb?: number; pipelined?: boolean }
): Promise<{
  ids: number[];
  stats: { melMs: number; encWaitMs: number; decodeMs: number; windows: number; groups: number };
}>;
