export function transcribeWindowed(
  ctx: any,
  enc: any,
  dec: any,
  mel: any,
  projW: any,
  projB: any,
  samples: Float32Array,
  opts?: {
    sampleRate?: number;
    windowSec?: number;
    overlapSec?: number;
    wb?: number;
    pipelined?: boolean;
    decodePool?: any;
    melPool?: any;
    /** Called after each window is decoded + stitched (in window order) with the window's end sample. */
    onWindowDone?: (processedSamples: number) => void;
  },
): Promise<{
  ids: number[];
  idTimes: number[];
  stats: { melMs: number; encWaitMs: number; decodeMs: number; windows: number; groups: number };
}>;
