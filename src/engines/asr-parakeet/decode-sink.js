// Decode dispatch: one interface over the three decode paths — GPU decoder
// results, worker-pool fan-out, and sync WASM — advancing the global window
// cursor and feeding the stitcher strictly in window order.

import { wasmDecodeProj } from "./raw-decoder-wasm.js";

export function createDecodeSink({ dec, D, decodePool, stitch, stats, now, starts, winSamples, totalSamples }) {
  let w = 0; // global window cursor
  const decJobs = []; // pool mode: in-window-order pending decodes
  const sliceLenAt = (i) => Math.min(starts[i] + winSamples, totalSamples) - starts[i];

  return {
    /** Advance past a group that produced no mels (degenerate tail). */
    skip(n) {
      w += n;
    },

    /** GPU decoder path: tokens arrived directly; frames never left the GPU. */
    consumeGpu(cur, perWindow) {
      const td = now();
      for (let wi = 0; wi < cur.n; wi++, w++) {
        stitch(w, sliceLenAt(w), cur.Tsub, perWindow[wi].ids, perWindow[wi].idFrames);
      }
      stats.decodeMs += now() - td;
      stats.windows += cur.n;
    },

    /** CPU frames for one group: fan out to the worker pool (stitching drains
     * opportunistically in window order so results/buffers don't accumulate
     * unboundedly on multi-hour files), or decode synchronously inline. */
    consumeFrames(cur, frames) {
      const Tenc = cur.Tsub;
      if (decodePool) {
        for (let wi = 0; wi < cur.n; wi++, w++) {
          const win = frames.slice(wi * Tenc * D, (wi + 1) * Tenc * D); // copy: transferred to the worker
          const job = { windowIdx: w, sliceLen: sliceLenAt(w), Tenc, p: decodePool.decode(win, Tenc), r: null };
          job.p = job.p.then((res) => {
            job.r = res;
            return res;
          });
          decJobs.push(job);
        }
        while (decJobs.length && decJobs[0].r) {
          const j = decJobs.shift();
          stitch(j.windowIdx, j.sliceLen, j.Tenc, j.r.ids, j.r.idFrames);
        }
      } else {
        const td = now();
        for (let wi = 0; wi < cur.n; wi++, w++) {
          const win = frames.subarray(wi * Tenc * D, (wi + 1) * Tenc * D);
          const { ids: wids, idFrames } = wasmDecodeProj(dec, win, Tenc);
          stitch(w, sliceLenAt(w), Tenc, wids, idFrames);
        }
        stats.decodeMs += now() - td;
      }
      stats.windows += cur.n;
    },

    /** Await + stitch the pool's remaining jobs (window order). */
    async finish() {
      if (!decJobs.length) return;
      const td = now();
      for (const j of decJobs) {
        const { ids: wids, idFrames } = await j.p;
        stitch(j.windowIdx, j.sliceLen, j.Tenc, wids, idFrames);
      }
      stats.decodeMs += now() - td;
    },
  };
}
