// Round-robin pool over initialized decoder workers. Workers are passed as
// shims ({ postMessage(msg, transfer?), setHandler(fn), terminate() }) so the
// browser Worker and node worker_threads adapters look identical.
export function createDecodePool(workers) {
  let next = 0;
  let seq = 0;
  const waiting = new Map();
  for (const w of workers) {
    w.setHandler((m) => {
      const res = waiting.get(m.id);
      if (res) { waiting.delete(m.id); res(m); }
    });
  }
  return {
    size: workers.length,
    /** frames is CONSUMED (transferred where supported). */
    decode(frames, Tenc) {
      const id = seq++;
      const w = workers[next++ % workers.length];
      return new Promise((resolve) => {
        waiting.set(id, resolve);
        w.postMessage({ type: "decode", id, frames: frames.buffer, Tenc }, [frames.buffer]);
      });
    },
    terminate() { for (const w of workers) w.terminate(); },
  };
}
