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
//
// This file is the orchestrator; the concerns live in their own modules:
// windowing.js (window/group planning), mel-scheduler.js (mel prefetch),
// decode-sink.js (the three decode paths), stitcher.js (seam dedup).

import { parakeetEncodeBatch } from "./raw-encoder.js";
import { planWindows, groupWindows } from "./windowing.js";
import { createMelScheduler } from "./mel-scheduler.js";
import { createDecodeSink } from "./decode-sink.js";
import { createStitcher } from "./stitcher.js";

/**
 * @returns {Promise<{ids: number[], idTimes: number[], stats: {melMs: number, encWaitMs: number, decodeMs: number, windows: number, groups: number}}>}
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
  const {
    sampleRate = 16000,
    windowSec = 15,
    overlapSec = 2,
    wb = 6,
    pipelined = true,
    decodePool = null,
    gpuDecoder = null,
    melPool = null,
    onWindowDone = null,
  } = opts;
  const winSamples = windowSec * sampleRate;
  const overlapSamples = overlapSec * sampleRate;
  const hop = winSamples - overlapSamples;
  const single = samples.length <= winSamples;
  const D = projW.cols; // joint dim (640)

  const starts = planWindows(samples.length, winSamples, hop);
  const groups = groupWindows(starts, samples.length, winSamples, wb);

  const { melsFor, resolveMels } = createMelScheduler({ mel, melPool, groups, starts, samples, winSamples, single, stats, now });
  const { ids, idTimes, stitch: stitchRaw } = createStitcher({ hop, sampleRate, overlapSamples });
  // Progress rides the stitch: windows stitch strictly in order across all three
  // decode paths, so the end sample of the just-stitched window is monotonic.
  const stitch = onWindowDone
    ? (w, sliceLen, Tsub, wids, idFrames) => {
        stitchRaw(w, sliceLen, Tsub, wids, idFrames);
        onWindowDone(Math.min(starts[w] + winSamples, samples.length));
      }
    : stitchRaw;
  const sink = createDecodeSink({ dec, D, decodePool, stitch, stats, now, starts, winSamples, totalSamples: samples.length });

  // Joint encoder projection [W*Tsub,1024]→[W*Tsub,640] + staging copy, recorded
  // inside the encoder's batch.
  const post = (x) => {
    const proj = ctx.matmul(x, projW, { bias: projB });
    if (gpuDecoder) return { proj }; // GPU decode: no readback at all
    return ctx.stageDownload(proj);
  };

  const openArenas = new Set(); // group scopes, popped on every exit path (incl. throws)

  // Record + submit a group; resolves at submit (parakeetEncodeBatch has no
  // internal GPU wait), with the readback's mapAsync already in flight.
  const submit = async (g, mels) => {
    if (!mels.length) return null;
    // Arena per group: every intermediate the encode allocates returns to the
    // buffer pool the moment this group's frames land on the CPU.
    const arena = ctx.pushArena();
    openArenas.add(arena);
    try {
      const r = await parakeetEncodeBatch(ctx, enc, mels, false, post);
      if (gpuDecoder) {
        // Decode on the GPU immediately (same queue, after the encode): all of
        // this group's windows in one dispatch; only tokens come back.
        const { gpuDecodeBatch } = gpuDecoder;
        return { decP: gpuDecodeBatch(ctx, gpuDecoder.gdec, r.staged.proj, mels.length, r.Tsub), Tsub: r.Tsub, n: mels.length, arena };
      }
      return { framesP: r.staged.read(), Tsub: r.Tsub, n: mels.length, arena };
    } catch (e) {
      openArenas.delete(arena);
      ctx.popArena(arena);
      throw e;
    }
  };
  const closeArena = (arena) => {
    openArenas.delete(arena);
    ctx.popArena(arena);
  };

  if (!groups.length) return { ids, idTimes, stats }; // empty / zero-length input
  let nextMels = melsFor(0);
  let pending = await submit(0, await resolveMels(nextMels));
  nextMels = groups.length > 1 ? melsFor(1) : null;

  try {
    for (let g = 0; g < groups.length; g++) {
      const cur = pending;
      const advance = async () => {
        if (g + 1 < groups.length) {
          pending = await submit(g + 1, await resolveMels(nextMels));
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
        sink.skip(groups[g].length);
        continue;
      }
      const tw = now();
      if (cur.decP) {
        const perWindow = await cur.decP;
        closeArena(cur.arena);
        stats.encWaitMs += now() - tw;
        stats.groups++;
        if (!pipelined) await advance();
        sink.consumeGpu(cur, perWindow);
        continue;
      }
      const frames = await cur.framesP;
      closeArena(cur.arena); // group's GPU work is drained — recycle its buffers
      stats.encWaitMs += now() - tw;
      stats.groups++;
      if (!pipelined) await advance();
      sink.consumeFrames(cur, frames);
    }
    await sink.finish(); // pool mode: stitch the remaining in-flight decodes
    stats.melMs = Math.round(stats.melMs);
    stats.encWaitMs = Math.round(stats.encWaitMs);
    stats.decodeMs = Math.round(stats.decodeMs);
    return { ids, idTimes, stats };
  } finally {
    for (const a of openArenas) ctx.popArena(a); // throws must not orphan group scopes
    ctx.trimPool(); // drained here (final readback resolved) — safe to evict to budget
  }
}
