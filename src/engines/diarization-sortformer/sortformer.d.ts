export function diarizeSortformer(o: {
  ort: any;
  mel: any;
  sortformer: any;
  audio: Float32Array;
  sampleRate?: number;
  threshold?: number;
  minSpeechSec?: number;
  mergeGapSec?: number;
}): Promise<{ speaker: number; start: number; end: number }[]>;
