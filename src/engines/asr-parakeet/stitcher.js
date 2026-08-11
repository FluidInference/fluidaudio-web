// Seam dedup for overlapping windows: a frame-estimated overlap skip refined
// by an exact token-match stitch against the tail of the running transcript.
// stitch() MUST be called in window order — it matches against `ids` so far.

/** @returns {{ids: number[], idTimes: number[], stitch: Function}} — ids/idTimes
 *  accumulate across stitch() calls; idTimes are absolute seconds per token
 *  (window start + frame·80ms). */
export function createStitcher({ hop, sampleRate, overlapSamples }) {
  const ids = [];
  const idTimes = [];

  const stitch = (windowIdx, sliceLen, Tenc, wids, idFrames) => {
    const winStartSec = (windowIdx * hop) / sampleRate;
    let skip = 0;
    if (windowIdx > 0 && wids.length) {
      const overlapEnc = Math.round((Tenc * overlapSamples) / sliceLen);
      let frameSkip = 0;
      while (frameSkip < idFrames.length && idFrames[frameSkip] < overlapEnc) frameSkip++;
      const maxL = Math.min(ids.length, wids.length, frameSkip + 8);
      let matched = 0;
      for (let L = maxL; L >= 2; L--) {
        let ok = true;
        for (let i = 0; i < L; i++)
          if (ids[ids.length - L + i] !== wids[i]) {
            ok = false;
            break;
          }
        if (ok) {
          matched = L;
          break;
        }
      }
      skip = Math.max(matched, frameSkip);
    }
    for (let k = skip; k < wids.length; k++) {
      ids.push(wids[k]);
      idTimes.push(winStartSec + idFrames[k] * 0.08);
    }
  };

  return { ids, idTimes, stitch };
}
