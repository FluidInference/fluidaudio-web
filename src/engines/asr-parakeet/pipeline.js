// Windowed Parakeet transcription with a 3-stage software pipeline:
//
//   GPU:  encode + joint-projection + staging-copy of group g+1 (ONE submit)
//   CPU:  mel extraction for group g+2, then WASM-SIMD decode of group g
//
// The projection and the staging copy ride the encoder's own submit (post hook
// into parakeetEncodeBatch), so after a group's kernels drain the bytes are
// already in a mappable buffer — the GPU never idles waiting for the JS thread
// to submit a readback, and the readback never pays an extra round trip.
// mapAsync for group g is issued BEFORE group g+1 is submitted, so even a
// conservative mapAsync implementation (drain-everything-before-map) cannot
// serialize the pipeline. Used by the browser engine (index.ts) and the node
// gates — same shipping code path in both.

import { parakeetEncodeBatch } from "./raw-encoder.js";
import { wasmDecodeProj } from "./raw-decoder-wasm.js";

/**
 * @returns {Promise<{ids: number[], stats: {melMs: number, encWaitMs: number, decodeMs: number, windows: number, groups: number}}>}
 *   deduped token ids + a stage breakdown. Stages OVERLAP (that is the point of
 *   the pipeline): encWaitMs is only the GPU wait NOT hidden behind CPU work,
 *   so melMs + encWaitMs + decodeMs ≈ wall. A GPU-bound machine shows a large
 *   encWaitMs; a CPU-bound one shows decodeMs dominating.
 */
export async function transcribeWindowed(ctx, enc, dec, mel, projW, projB, samples, opts = {}) {
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
  const stats = { melMs: 0, encWaitMs: 0, decodeMs: 0, windows: 0, groups: 0 };
  // wb=6 measured best on the 120s bench (M5): 130.3x vs 123.8x at wb=4 —
  // bigger GEMM M-dim + fewer group boundaries, while the uneven 6+3 split
  // still overlaps decode of the big group with encode of the small one.
  // decodePool (decode-pool.js): windows decode in parallel on worker threads;
  // browsers where decode dominates the wall (encode 355ms vs decode 1821ms on
  // one measured machine) gain ~pool-size× on the decode term.
  const { sampleRate = 16000, windowSec = 15, overlapSec = 2, wb = 6, pipelined = true, decodePool = null } = opts;
  const winSamples = windowSec * sampleRate;
  const overlapSamples = overlapSec * sampleRate;
  const hop = winSamples - overlapSamples;
  const single = samples.length <= winSamples;
  const D = projW.cols; // joint dim (640)

  const starts = [];
  for (let s = 0; s < samples.length; s += hop) {
    starts.push(s);
    if (single) break;
  }

  // Groups of up to wb equal-length windows (batched through the encoder:
  // bigger GEMMs + one readback per group). A short tail encodes alone.
  const groups = [];
  {
    let cur = [];
    for (let i = 0; i < starts.length; i++) {
      const len = Math.min(starts[i] + winSamples, samples.length) - starts[i];
      if (cur.length && (len !== winSamples || cur.length >= wb)) { groups.push(cur); cur = []; }
      cur.push(i);
      if (len !== winSamples) { groups.push(cur); cur = []; }
    }
    if (cur.length) groups.push(cur);
  }

  const melsFor = (g) => {
    const t0 = now();
    const mels = [];
    for (const i of groups[g]) {
      const slice = single ? samples : samples.subarray(starts[i], Math.min(starts[i] + winSamples, samples.length));
      const { features, length } = mel.process(slice);
      if (length > 0) mels.push(features);
    }
    stats.melMs += now() - t0;
    return mels;
  };

  // Joint encoder projection [W*Tsub,1024]→[W*Tsub,640] + staging copy, recorded
  // inside the encoder's batch. WASM backend has no stageDownload — plain download.
  const post = (x) => {
    const proj = ctx.matmul(x, projW, { bias: projB });
    return ctx.stageDownload ? ctx.stageDownload(proj) : { read: async () => ctx.download(proj) };
  };

  // Record + submit a group; resolves at submit (parakeetEncodeBatch has no
  // internal GPU wait), with the readback's mapAsync already in flight.
  const submit = async (g, mels) => {
    if (!mels.length) return null;
    const r = await parakeetEncodeBatch(ctx, enc, mels, false, post);
    return { framesP: r.staged.read(), Tsub: r.Tsub, n: mels.length };
  };

  const ids = [];
  let w = 0;
  let nextMels = melsFor(0);
  let pending = await submit(0, nextMels);
  nextMels = groups.length > 1 ? melsFor(1) : null;

  // Seam dedup: frame-estimated overlap refined by an exact token-match stitch.
  // MUST run in window order (it matches against the tail of `ids`).
  const stitch = (windowIdx, sliceLen, Tenc, wids, idFrames) => {
    let skip = 0;
    if (windowIdx > 0 && wids.length) {
      const overlapEnc = Math.round((Tenc * overlapSamples) / sliceLen);
      let frameSkip = 0;
      while (frameSkip < idFrames.length && idFrames[frameSkip] < overlapEnc) frameSkip++;
      const maxL = Math.min(ids.length, wids.length, frameSkip + 8);
      let matched = 0;
      for (let L = maxL; L >= 2; L--) {
        let ok = true;
        for (let i = 0; i < L; i++) if (ids[ids.length - L + i] !== wids[i]) { ok = false; break; }
        if (ok) { matched = L; break; }
      }
      skip = Math.max(matched, frameSkip);
    }
    for (let k = skip; k < wids.length; k++) ids.push(wids[k]);
  };

  const decJobs = []; // pool mode: in-window-order pending decodes

  for (let g = 0; g < groups.length; g++) {
    const cur = pending;
    const advance = async () => {
      if (g + 1 < groups.length) {
        pending = await submit(g + 1, nextMels);
        nextMels = g + 2 < groups.length ? melsFor(g + 2) : null;
      } else {
        pending = null;
      }
    };
    // Pipelined: submit g+1 before draining g so the GPU flows group-to-group.
    // Serial (gate baseline): drain g fully first.
    if (pipelined) await advance();
    if (!cur) {
      if (!pipelined) await advance();
      w += groups[g].length;
      continue;
    }
    const tw = now();
    const frames = await cur.framesP;
    stats.encWaitMs += now() - tw;
    stats.groups++;
    if (!pipelined) await advance();

    const Tenc = cur.Tsub;
    if (decodePool) {
      // Fan windows out to the worker pool as soon as their frames land; the
      // GPU keeps encoding and workers decode concurrently. Stitching happens
      // at the end, in window order.
      for (let wi = 0; wi < cur.n; wi++, w++) {
        const win = frames.slice(wi * Tenc * D, (wi + 1) * Tenc * D); // copy: transferred to the worker
        const sliceLen = Math.min(starts[w] + winSamples, samples.length) - starts[w];
        decJobs.push({ windowIdx: w, sliceLen, Tenc, p: decodePool.decode(win, Tenc) });
      }
    } else {
      const td = now();
      for (let wi = 0; wi < cur.n; wi++, w++) {
        const win = frames.subarray(wi * Tenc * D, (wi + 1) * Tenc * D);
        const sliceLen = Math.min(starts[w] + winSamples, samples.length) - starts[w];
        const { ids: wids, idFrames } = wasmDecodeProj(dec, win, Tenc);
        stitch(w, sliceLen, Tenc, wids, idFrames);
      }
      stats.decodeMs += now() - td;
    }
    stats.windows += cur.n;
  }
  if (decJobs.length) {
    const td = now();
    for (const j of decJobs) {
      const { ids: wids, idFrames } = await j.p;
      stitch(j.windowIdx, j.sliceLen, j.Tenc, wids, idFrames);
    }
    stats.decodeMs += now() - td;
  }
  stats.melMs = Math.round(stats.melMs); stats.encWaitMs = Math.round(stats.encWaitMs); stats.decodeMs = Math.round(stats.decodeMs);
  return { ids, stats };
}
