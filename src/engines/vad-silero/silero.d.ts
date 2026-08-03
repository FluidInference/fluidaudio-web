export function sileroDetect(o: {
  ort: any;
  session: any;
  audio: Float32Array;
  threshold?: number;
  negThreshold?: number;
  minSpeechMs?: number;
  minSilenceMs?: number;
  speechPadMs?: number;
}): Promise<{ start: number; end: number }[]>;
