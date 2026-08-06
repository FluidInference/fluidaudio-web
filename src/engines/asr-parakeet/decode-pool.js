// Round-robin pool over initialized decoder workers. Workers are passed as
// shims ({ postMessage(msg, transfer?), setHandler(fn), terminate() }) so the
// browser Worker and node worker_threads adapters look identical.
// decode() promises REJECT on worker error ({type:"err"} reply or a transport
// failure surfaced via failWorker) — a hung transcribe() is never the failure
// mode; callers fall back or surface the error.
export function createDecodePool(workers) {
  let next = 0;
  let seq = 0;
  const waiting = new Map(); // id -> {resolve, reject}
  for (const w of workers) {
    w.setHandler((m) => {
      const p = waiting.get(m.id);
      if (!p) return;
      waiting.delete(m.id);
      if (m.type === "err") p.reject(new Error(`decode worker: ${m.error}`));
      else p.resolve(m);
    });
  }
  return {
    size: workers.length,
    /** frames is CONSUMED (transferred where supported). */
    decode(frames, Tenc) {
      const id = seq++;
      const w = workers[next++ % workers.length];
      return new Promise((resolve, reject) => {
        waiting.set(id, { resolve, reject });
        try {
          w.postMessage({ type: "decode", id, frames: frames.buffer, Tenc }, [frames.buffer]);
        } catch (e) {
          waiting.delete(id);
          reject(e);
        }
      });
    },
    /** Reject every pending decode (worker died / transport error). */
    failAll(err) {
      for (const [, p] of waiting) p.reject(err instanceof Error ? err : new Error(String(err)));
      waiting.clear();
    },
    terminate() {
      for (const w of workers) w.terminate();
    },
  };
}
