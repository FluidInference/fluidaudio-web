// Silero VAD v5 core — runtime-agnostic (pass the `ort` module). Bypasses
// @ricky0123/vad-web entirely: that package is CJS and does a dynamic
// `require("onnxruntime-web/wasm")` which Vite can't resolve once ORT is excluded
// from optimizeDeps. Silero's ONNX I/O is trivial, so we drive it directly.
//
// Model I/O (silero_vad.onnx, v5):
//   in : input[1,N] audio, state[2,1,128], sr int64  (N must be 512 @ 16 kHz)
//   out: output[1,1] speech prob, stateN[2,1,128]
//
// The window emits one probability per 512-sample (32 ms) hop; we threshold with
// hysteresis and merge into speech ranges with min-speech / min-silence guards
// and a symmetric speech pad — the same shape as the upstream JS post-processor.

const WINDOW = 512; // required frame size @ 16 kHz
const SR = 16000;

/**
 * @param {{ort:any, session:any, audio:Float32Array,
 *          threshold?:number, negThreshold?:number,
 *          minSpeechMs?:number, minSilenceMs?:number, speechPadMs?:number}} o
 * @returns {Promise<{start:number,end:number}[]>}  seconds
 */
export async function sileroDetect({
  ort,
  session,
  audio,
  threshold = 0.5,
  negThreshold = threshold - 0.15,
  minSpeechMs = 250,
  minSilenceMs = 100,
  speechPadMs = 30,
}) {
  let state = new Float32Array(2 * 1 * 128);
  const srTensor = new ort.Tensor("int64", BigInt64Array.from([BigInt(SR)]), []);

  const probs = [];
  for (let off = 0; off + WINDOW <= audio.length; off += WINDOW) {
    const frame = audio.subarray(off, off + WINDOW);
    const out = await session.run({
      input: new ort.Tensor("float32", frame, [1, WINDOW]),
      state: new ort.Tensor("float32", state, [2, 1, 128]),
      sr: srTensor,
    });
    probs.push(out["output"].data[0]);
    const sn = out["stateN"];
    state = sn.data instanceof Float32Array ? sn.data.slice() : Float32Array.from(sn.data);
    sn.dispose?.();
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
