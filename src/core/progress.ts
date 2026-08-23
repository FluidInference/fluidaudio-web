// Shared monotonic transcription-progress emitter for the batch ASR engines.
// update() publishes audio-seconds processed at a window boundary; the fraction
// never decreases and never reaches 1 before done() — exactly 1 is reserved for
// the final emit, so a UI can key "finished" off it.

import type { TranscribeProgress } from "./types.js";

/** Cap for window-boundary estimates: exactly 1 is reserved for done(). */
const PRE_DONE_FRACTION = 0.999;

export function makeTranscribeProgress(
  totalSeconds: number,
  onProgress?: (p: TranscribeProgress) => void,
): { update(processedSeconds: number): void; done(): void } | null {
  if (!onProgress) return null;
  let published = 0;
  const emit = (processedSeconds: number, cap: number) => {
    const raw = totalSeconds > 0 ? processedSeconds / totalSeconds : 1;
    published = Math.max(published, Math.min(cap, raw));
    onProgress({ processedSeconds: Math.min(processedSeconds, totalSeconds), totalSeconds, fraction: published });
  };
  return {
    update: (processedSeconds) => emit(processedSeconds, PRE_DONE_FRACTION),
    done: () => emit(totalSeconds, 1),
  };
}
