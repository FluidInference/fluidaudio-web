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

// ── worker shims: single-sourced protocol adapters for both runtimes ─────────
// (previously hand-rolled in index.ts AND the node gate, with subtle
// differences — transfer lists, removeAllListeners — that could drift.)

/** Browser Worker → pool shim. */
export function browserWorkerShim(w) {
  return {
    postMessage: (m, t) => w.postMessage(m, t ?? []),
    setHandler: (f) => {
      w.onmessage = (e) => f(e.data);
    },
    terminate: () => w.terminate(),
  };
}

/** node worker_threads Worker → pool shim. */
export function nodeWorkerShim(w) {
  return {
    postMessage: (m, t) => w.postMessage(m, t ?? []),
    setHandler: (f) => {
      w.removeAllListeners("message");
      w.on("message", f);
    },
    terminate: () => w.terminate(),
  };
}

/** Init handshake: send weights, resolve on {type:"ready"}, reject otherwise. */
export function initDecodeWorker(post, once, payload) {
  return new Promise((resolve, reject) => {
    once(
      (m) => (m?.type === "ready" ? resolve() : reject(new Error(String(m?.error ?? "bad init reply")))),
      (err) => reject(err instanceof Error ? err : new Error(String(err))),
    );
    post({ type: "init", ...payload });
  });
}
