import type { SileroWeights } from "./raw-silero.js";

export function sileroDetect(o: {
  weights: SileroWeights;
  audio: Float32Array;
  threshold?: number;
  negThreshold?: number;
  minSpeechMs?: number;
  minSilenceMs?: number;
  speechPadMs?: number;
}): Promise<{ start: number; end: number }[]>;
