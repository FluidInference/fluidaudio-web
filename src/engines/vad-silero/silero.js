// Silero VAD v5 core — NO onnxruntime. Drives the hand-written raw-silero.js
// forward (parity-verified vs ORT to ~1e-6; see scripts/smoke-silero-raw.mjs).
// The window emits one probability per 512-sample (32 ms) hop; we threshold with
// hysteresis and merge into speech ranges with min-speech / min-silence guards
// and a symmetric speech pad — the same shape as the upstream JS post-processor.

import { sileroForward } from "./raw-silero.js";

const WINDOW = 512; // required frame size @ 16 kHz
const SR = 16000;

/**
 * @param {{weights:object, audio:Float32Array,
 *          threshold?:number, negThreshold?:number,
 *          minSpeechMs?:number, minSilenceMs?:number, speechPadMs?:number}} o
 * @returns {Promise<{start:number,end:number}[]>}  seconds
 */
export async function sileroDetect({
  weights,
  audio,
  threshold = 0.5,
  negThreshold = threshold - 0.15,
  minSpeechMs = 250,
  minSilenceMs = 100,
  speechPadMs = 30,
}) {
  let state = new Float32Array(2 * 1 * 128); // [h(128) | c(128)]

  const probs = [];
  for (let off = 0; off + WINDOW <= audio.length; off += WINDOW) {
    const frame = audio.subarray(off, off + WINDOW);
    const r = sileroForward(frame, state, weights);
    probs.push(r.prob);
    state = r.state;
  }

  // Hysteresis threshold over the per-window probs, then merge with duration
  // guards. Time base: one window = WINDOW / SR seconds.
  const winSec = WINDOW / SR;
  const minSpeech = minSpeechMs / 1000;
  const minSilence = minSilenceMs / 1000;
  const pad = speechPadMs / 1000;

  const ranges = [];
  let inSpeech = false;
  let speechStart = 0;
  let silenceRun = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    const t = i * winSec;
    if (!inSpeech) {
      if (p >= threshold) { inSpeech = true; speechStart = t; silenceRun = 0; }
    } else if (p < negThreshold) {
      silenceRun += winSec;
      if (silenceRun >= minSilence) {
        const end = t - silenceRun + winSec;
        if (end - speechStart >= minSpeech) ranges.push({ start: speechStart, end });
        inSpeech = false;
      }
    } else {
      silenceRun = 0;
    }
  }
  if (inSpeech) {
    const end = probs.length * winSec;
    if (end - speechStart >= minSpeech) ranges.push({ start: speechStart, end });
  }

  const dur = audio.length / SR;
  return ranges.map((r) => ({
    start: Math.max(0, +(r.start - pad).toFixed(3)),
    end: Math.min(dur, +(r.end + pad).toFixed(3)),
  }));
}
