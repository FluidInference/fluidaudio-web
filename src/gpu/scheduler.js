// Command scheduling: batching, dispatch, uniform ring, raw-copy encoding, and
// the timestamp-query profiler. Owns the encoder/pass state that batch mode
// records into; everything that submits GPU work goes through this class.
export class Scheduler {
  /** @param {GPUDevice} device */
  constructor(device) {
    this.device = device;
    this._enc = null;
    this._pass = null;
    this._batchUniforms = 0;
    this._prof = null;
    this._uniRing = new Map();
    this._warnedRing = false;
  }

  /** Batch mode: queue many kernels into one submit. beginBatch()…endBatch(). */
  beginBatch() {
    // Non-reentrant by design: nesting would silently discard the outer
    // encoder's recorded-but-unsubmitted work. Fail loudly instead.
    if (this._enc) throw new Error("beginBatch: a batch is already open");
    this._enc = this.device.createCommandEncoder();
    this._pass = this._enc.beginComputePass();
    this._batchUniforms = 0;
  }

  endBatch() {
    if (!this._pass) throw new Error("endBatch without beginBatch");
    this._pass.end();
    this.device.queue.submit([this._enc.finish()]);
    this._enc = this._pass = null;
  }

  get batchOpen() {
    return !!this._pass;
  }

  /** Submit the open batch (if any) and reopen it — used by readbacks whose
   * staging copy must land on the queue before mapAsync. No-op outside a batch. */
  flush() {
    if (!this._enc || !this._pass) return;
    this._pass.end();
    this.device.queue.submit([this._enc.finish()]);
    this._enc = this._pass = null;
    this.beginBatch();
  }

  /** Record raw encoder commands (buffer copies) ordered with the batch: inside
   * a batch the compute pass is paused/reopened around them so they stay
   * ordered after the dispatches that produce their sources; outside a batch
   * they get their own immediate submit. */
  encodeCopy(record) {
    if (this._enc && this._pass) {
      this._pass.end();
      record(this._enc);
      this._pass = this._enc.beginComputePass();
      return;
    }
    const enc = this.device.createCommandEncoder();
    record(enc);
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Uniform buffers come from a per-size ring (reused via writeBuffer) instead of
   * one new GPUBuffer per dispatch — the dominant buffer churn. Reuse across
   * submits is safe (queue operations are ordered); within ONE batched command
   * buffer every dispatch sees the final write, so the ring must exceed the max
   * dispatches per batch (largest today ~700; warn if a batch wraps the ring).
   */
  uniform(arr) {
    const bytes = arr instanceof ArrayBuffer ? new Uint8Array(arr) : arr;
    const size = bytes.byteLength;
    let ring = this._uniRing.get(size);
    if (!ring) {
      ring = { bufs: [], i: 0 };
      this._uniRing.set(size, ring);
    }
    const CAP = 4096;
    let buf;
    if (ring.bufs.length < CAP) {
      buf = this.device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      ring.bufs.push(buf);
    } else {
      buf = ring.bufs[ring.i % CAP];
    }
    ring.i++;
    if (this._pass) {
      this._batchUniforms += 1;
      if (this._batchUniforms > CAP && !this._warnedRing) {
        this._warnedRing = true;
        console.warn("[gpu] uniform ring wrapped within one batch — split the batch");
      }
    }
    this.device.queue.writeBuffer(buf, 0, bytes);
    return buf;
  }

  run(pipeline, buffers, uniform, groupsX, groupsY = 1, groupsZ = 1) {
    // WebGPU caps each grid dimension at 65535. For flat 1-D kernels (groupsY===1)
    // that exceed it, fold the excess into Y; those kernels linearize the group id
    // via num_workgroups. 2-D callers (GEMM) already pass groupsY and stay in range.
    if (groupsY === 1 && groupsX > 65535) {
      groupsY = Math.ceil(groupsX / 65535);
      groupsX = Math.ceil(groupsX / groupsY);
    }
    const entries = buffers.map((b, i) => ({ binding: i, resource: { buffer: b } }));
    entries.push({ binding: buffers.length, resource: { buffer: uniform } });
    const bg = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const prof = this._prof && this._prof.labels.length < this._prof.max ? this._prof : null;
    const tw = prof ? { querySet: prof.qs, beginningOfPassWriteIndex: prof.labels.length * 2, endOfPassWriteIndex: prof.labels.length * 2 + 1 } : undefined;
    if (prof) prof.labels.push(pipeline.__label || "op");
    if (this._pass) {
      // batched
      if (prof) {
        // own timestamped pass inside the batch encoder
        this._pass.end();
        const p = this._enc.beginComputePass({ timestampWrites: tw });
        p.setPipeline(pipeline);
        p.setBindGroup(0, bg);
        p.dispatchWorkgroups(groupsX, groupsY, groupsZ);
        p.end();
        this._pass = this._enc.beginComputePass();
        return;
      }
      this._pass.setPipeline(pipeline);
      this._pass.setBindGroup(0, bg);
      this._pass.dispatchWorkgroups(groupsX, groupsY, groupsZ);
      return;
    }
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass(prof ? { timestampWrites: tw } : undefined);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(groupsX, groupsY, groupsZ);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Per-dispatch GPU profiling (timestamp-query): between startProfile()/endProfile()
   * every run() gets its own compute pass with begin/end timestamps, labeled by
   * pipeline. endProfile() resolves and returns [{label, ms, count}] sorted by ms.
   * Works inside beginBatch/endBatch (passes share the batch encoder).
   */
  startProfile(maxOps = 2000) {
    // querySet count limit is 4096 → ≤2048 ops
    if (!this.device.features.has("timestamp-query")) throw new Error("timestamp-query not available");
    this._prof = {
      qs: this.device.createQuerySet({ type: "timestamp", count: maxOps * 2 }),
      labels: [],
      max: maxOps,
    };
  }

  async endProfile() {
    const prof = this._prof;
    this._prof = null;
    const n = prof.labels.length;
    const qb = this.device.createBuffer({ size: n * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rb = this.device.createBuffer({ size: n * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.resolveQuerySet(prof.qs, 0, n * 2, qb, 0);
    enc.copyBufferToBuffer(qb, 0, rb, 0, n * 16);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(rb.getMappedRange());
    const agg = new Map();
    for (let i = 0; i < n; i++) {
      const ms = Number(t[2 * i + 1] - t[2 * i]) / 1e6;
      const a2 = agg.get(prof.labels[i]) || { label: prof.labels[i], ms: 0, count: 0 };
      a2.ms += ms;
      a2.count++;
      agg.set(prof.labels[i], a2);
    }
    rb.unmap();
    qb.destroy();
    rb.destroy();
    prof.qs.destroy();
    return [...agg.values()].sort((a2, b) => b.ms - a2.ms);
  }
}
