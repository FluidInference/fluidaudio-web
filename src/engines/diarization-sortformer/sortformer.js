// Sortformer streaming diarization core (NVIDIA diar_streaming_sortformer_4spk-v2.1,
// ONNX fp32). Runtime-agnostic: pass the ort module + a mel session (nemo128,
// 128 mel bins) + the sortformer session.
//
// I/O (verified): chunk[1,T,128] mel + chunk_lengths + spkcache[1,·,512] +
// fifo[1,·,512] (+ *_lengths) -> spkcache_fifo_chunk_preds[1,T',4] (per-frame
// probability for up to 4 speakers) + chunk_pre_encode_embs (state update).
//
// This offline path runs the whole clip as ONE chunk with empty state (verified
// correct: single-speaker -> 1 active, 40s conference -> 2 active). Long-audio
// streaming (threading spkcache/fifo across chunks for cross-window speaker
// consistency) is the follow-up — see the state outputs above.

/**
 * @param {{ort:any, mel:any, sortformer:any, audio:Float32Array, sampleRate?:number,
 *          threshold?:number, minSpeechSec?:number, mergeGapSec?:number}} o
 * @returns {Promise<{speaker:number,start:number,end:number}[]>}
 */
export async function diarizeSortformer({ ort, mel, sortformer, audio, sampleRate = 16000, threshold = 0.5, minSpeechSec = 0.25, mergeGapSec = 0.25 }) {
  if (!audio || audio.length < sampleRate * 0.05) return []; // <50ms: nothing to diarize
  // 1. mel [1,128,T] -> transpose to [1,T,128] (sortformer is T-major)
  const mo = await mel.run({
    waveforms: new ort.Tensor("float32", audio, [1, audio.length]),
    waveforms_lens: new ort.Tensor("int64", BigInt64Array.from([BigInt(audio.length)]), [1]),
  });
  const f = mo["features"];
  const T = f.dims[2];
  const chunk = new Float32Array(T * 128);
  for (let t = 0; t < T; t++) for (let m = 0; m < 128; m++) chunk[t * 128 + m] = f.data[m * T + t];
  f.dispose?.();

  const i64 = (v) => new ort.Tensor("int64", BigInt64Array.from([BigInt(v)]), [1]);
  const empty = (d) => new ort.Tensor("float32", new Float32Array(0), d);
  const out = await sortformer.run({
    chunk: new ort.Tensor("float32", chunk, [1, T, 128]),
    chunk_lengths: i64(T),
    spkcache: empty([1, 0, 512]),
    spkcache_lengths: i64(0),
    fifo: empty([1, 0, 512]),
    fifo_lengths: i64(0),
  });
  const preds = out["spkcache_fifo_chunk_preds"];
  const [, frames, S] = preds.dims;
  const frameSec = audio.length / sampleRate / frames;

  // 2. per-speaker threshold → contiguous segments, merge small gaps, drop shorts
  const segments = [];
  for (let s = 0; s < S; s++) {
    let start = -1;
    for (let t = 0; t <= frames; t++) {
      const on = t < frames && preds.data[t * S + s] >= threshold;
      if (on && start < 0) start = t;
      if (!on && start >= 0) {
        segments.push({ speaker: s, start: start * frameSec, end: t * frameSec });
        start = -1;
      }
    }
  }
  preds.dispose?.();
  return mergeSegments(segments, mergeGapSec, minSpeechSec);
}

function mergeSegments(segs, mergeGapSec, minSpeechSec) {
  const bySpk = new Map();
  for (const s of segs) (bySpk.get(s.speaker) ?? bySpk.set(s.speaker, []).get(s.speaker)).push(s);
  const out = [];
  for (const [, list] of bySpk) {
    list.sort((a, b) => a.start - b.start);
    let cur = null;
    for (const s of list) {
      if (cur && s.start - cur.end <= mergeGapSec) cur.end = s.end;
      else {
        if (cur) out.push(cur);
        cur = { ...s };
      }
    }
    if (cur) out.push(cur);
  }
  return out.filter((s) => s.end - s.start >= minSpeechSec).sort((a, b) => a.start - b.start);
}
