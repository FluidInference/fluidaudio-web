// GPU buffer pool + arena scopes.
//
// Intermediate tensors used to be one fresh GPUBuffer each, never destroyed —
// GPU memory ballooned until the JS GC lazily collected the wrappers. Now:
// allocRaw() draws from an exact-size free pool, and pushArena()/popArena()
// scope a group/step/synth so its intermediates return to the pool the
// moment the readback lands. Reuse is queue-ordered-safe (later submits
// execute after earlier ones; all kernels fully write their outputs — no
// zero-init assumptions, verified + parity-gated). pin(t) exempts tensors
// that outlive the scope (weight-derived caches, KV caches).
export class BufferPool {
  /** @param {GPUDevice} device */
  constructor(device) {
    this.device = device;
    this._pool = new Map();
    this._pooledBytes = 0;
    this._arenas = [];
    this._memStats = null;
  }

  get(size, usage) {
    const key = `${size}|${usage}`;
    const list = this._pool.get(key);
    if (list && list.length) {
      this._pooledBytes -= size;
      this._memStats && (this._memStats.reused += 1);
      return list.pop();
    }
    if (this._memStats) {
      this._memStats.created += 1;
      this._memStats.createdBytes += size;
    }
    return this.device.createBuffer({ size, usage });
  }

  put(buf, size, usage) {
    // NEVER destroy here: a popped arena's buffers may still be referenced by
    // recorded-but-unsubmitted command buffers (per-layer arenas pop while the
    // batch is still recording) — destroying one drops the whole submit with
    // an async validation error. Eviction happens in trim(), which engines
    // reach at known-drained points (after the run's final readback).
    this._pooledBytes += size;
    const key = `${size}|${usage}`;
    let list = this._pool.get(key);
    if (!list) this._pool.set(key, (list = []));
    list.push(buf);
  }

  /** Pool-or-fresh allocation, tracked by the newest open arena. No open arena
   * → legacy persistent alloc that must NOT draw from the pool: it would never
   * return the buffer (one-way drain — a no-arena caller like the sortformer
   * head would steal the encoder's pooled scratch for good). */
  allocRaw(size, usage) {
    const top = this._arenas.length ? this._arenas[this._arenas.length - 1] : null;
    if (!top) {
      if (this._memStats) {
        this._memStats.created += 1;
        this._memStats.createdBytes += size;
      }
      return this.device.createBuffer({ size, usage });
    }
    const buf = this.get(size, usage);
    top.push({ buf, size, usage });
    return buf;
  }

  /** Destroy pooled buffers down to the byte budget. Call ONLY when the GPU
   * work is drained — pooled buffers are unreferenced by definition then. */
  trim(budget) {
    if (this._pooledBytes <= budget) return;
    // Evict largest size-classes first (fewest destroys to get under budget).
    const keys = [...this._pool.keys()].sort((a, b) => Number(b.split("|")[0]) - Number(a.split("|")[0]));
    for (const key of keys) {
      if (this._pooledBytes <= budget) break;
      const size = Number(key.split("|")[0]);
      const list = this._pool.get(key);
      while (list.length && this._pooledBytes > budget) {
        list.pop().destroy();
        this._pooledBytes -= size;
      }
      if (!list.length) this._pool.delete(key);
    }
  }

  /** Pool occupancy (for gates/telemetry). */
  info() {
    return { bytes: this._pooledBytes };
  }

  /** Open an allocation scope; allocs land in the NEWEST open scope. Returns a
   * handle — scopes may close out of order (the pipelined engines interleave
   * groups), so popArena takes the handle rather than assuming LIFO. */
  pushArena() {
    const arena = [];
    this._arenas.push(arena);
    return arena;
  }

  /** Close a scope: its (unpinned) buffers return to the pool for reuse. */
  popArena(handle) {
    const arena = handle ?? this._arenas[this._arenas.length - 1];
    const i = this._arenas.indexOf(arena);
    if (i < 0) return;
    this._arenas.splice(i, 1);
    // Invariant: each live buf appears in at most one arena entry (alloc pushes
    // once; pin() SPLICES entries out rather than marking them).
    for (const e of arena) this.put(e.buf, e.size, e.usage);
  }

  /** Exempt a tensor from its arena. Default: persistent (never pooled — for
   * caches). toParent: move it to the ENCLOSING arena instead (for tensors
   * that outlive an inner scope but die with the outer one, e.g. the residual
   * stream x outliving a per-layer arena but dying with its group). */
  pin(t, toParent = false) {
    for (let ai = this._arenas.length - 1; ai >= 0; ai--) {
      const arena = this._arenas[ai];
      for (let i = 0; i < arena.length; i++) {
        if (arena[i].buf !== t.buf) continue;
        const e = arena.splice(i, 1)[0];
        if (toParent && ai > 0) this._arenas[ai - 1].push(e);
        // else: persistent — belongs to no arena, never pooled
        return t;
      }
    }
    return t;
  }

  /** Allocation counters for the memory gate (created/createdBytes/reused). */
  statsStart() {
    this._memStats = { created: 0, createdBytes: 0, reused: 0 };
  }

  stats() {
    return this._memStats;
  }
}
