// Raw-WebGPU compute core — GPU-resident tensors over the hand-written WGSL
// kernels in src/gpu/kernels/. Internals are composed from buffer-pool.js
// (pool + arena scopes), scheduler.js (batching / dispatch / profiler), and
// pipeline-cache.js; this file is the public GpuContext facade + op methods.
//
// This is the "write raw WebGPU (and WASM where needed)" path: instead of handing
// a whole ONNX graph to onnxruntime-web (many un-fused dispatches + GPU↔CPU syncs
// on unsupported/dynamic ops), we keep tensors resident on the GPU and run fused
// kernels. The win is fusion + residency, not a faster single GEMM.
//
// Runtime-agnostic: the caller passes a GPUDevice — `navigator.gpu` in the
// browser, dawn (@kmamal/gpu) in the Node verifier (scripts/gpu-verify.mjs). The
// GPUBufferUsage / GPUMapMode flag constants are ambient globals in the browser;
// the Node harness registers them on globalThis first (scripts/gpu-globals.mjs).
//
// A Tensor is { buf: GPUBuffer, rows, cols } — 2-D row-major f32, GPU-resident.
// Kernels take and return Tensors; only download() copies back to CPU. Verified
// for numerical parity against CPU references on a real M5 Pro GPU.

import { PipelineCache } from "./pipeline-cache.js";
import { BufferPool } from "./buffer-pool.js";
import { Scheduler } from "./scheduler.js";
import { WGSL_ACTF, ACT } from "./kernels/actf.js";
import {
  GEMM_WGSL,
  GEMV_PART_WGSL,
  GEMV_REDUCE_WGSL,
  GEMV_PART4_WGSL,
  GEMV_REDUCE4_WGSL,
  GEMM_V2_WGSL,
  GEMM_V3_WGSL,
  GEMM_V4_WGSL,
  GEMM_F16SG_WGSL,
  GEMM_V4_F16C2_WGSL,
  GEMM_TM_WGSL,
  GEMM_V4_F16C_WGSL,
  GEMM_V4_F16B_WGSL,
  GEMM_F16_WGSL,
  MATMUL_INT8_WGSL,
  MATMUL_INT8_V2_WGSL,
  MATMUL_INT8_V3_WGSL,
  MATMUL_NBITS_WGSL,
} from "./kernels/gemm.js";
import { BMM_QK_WGSL, BMM_PV_WGSL, genAttnFusedWgsl, RELSHIFT_STREAM_WGSL, RELSHIFT_B_WGSL, RELSHIFT_WGSL } from "./kernels/attention.js";
import {
  CONV1D_WGSL,
  SUBRESHAPE_WGSL,
  CONV2D_DW3X3S2_WGSL,
  CONV2D_C1_3X3S2_WGSL,
  CONV2D_WGSL,
  CONVT1D_WGSL,
  CONVT_RESHAPE_WGSL,
  CONVT_WPERM_WGSL,
  IM2COL_WGSL,
  CONV1D_IMPLICIT_WGSL,
  CONV1D_IMPLICIT_F16_WGSL,
} from "./kernels/convolution.js";
import { LAYERNORM_WGSL, SOFTMAX_WGSL, ADAIN_WGSL } from "./kernels/normalization.js";
import { RMSNORM_WGSL, HEADRMS_ROPE_WGSL, ATTN_SCORES_WGSL, ATTN_PV_WGSL } from "./kernels/gemma.js";
import { JBATCH_WGSL, ARGMAX_ROWS_WGSL, TDT_JOINT_WGSL, LSTM_WGSL } from "./kernels/decode.js";
import { GLU_WGSL, GATHERCOLS_WGSL, LEAKY_WGSL, SNAKE_WGSL, EWISE_WGSL, TRANSPOSE_WGSL, SLICECOLS_WGSL, SETCOLS_WGSL } from "./kernels/tensor-ops.js";

/** Request a WebGPU device in the browser (throws if unavailable). */
export async function requestGpuDevice() {
  if (typeof navigator === "undefined" || !navigator.gpu) throw new Error("WebGPU not available");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const lim = adapter.limits;
  // Optional features, taken only when the adapter has them: shader-f16 gates the
  // f16-storage weight path (GpuContext falls back to fp32 without it),
  // timestamp-query enables startProfile/endProfile in the browser.
  const requiredFeatures = ["shader-f16", "timestamp-query", "subgroups"].filter((f) => adapter.features.has(f));
  return adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxBufferSize: lim.maxBufferSize,
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      // GPU TDT decoder keeps LSTM state + reductions in workgroup memory
      // (~29KB) — the 16KB default is too small; Apple/NVIDIA offer 32KB.
      maxComputeWorkgroupStorageSize: lim.maxComputeWorkgroupStorageSize,
    },
  });
}

export class GpuContext {
  /** @param {GPUDevice} device */
  constructor(device) {
    this.device = device;
    this.backend = "webgpu";
    // Composed internals: compiled-pipeline cache, buffer pool + arena scopes,
    // and command scheduling (batching / dispatch / uniforms / profiler). The
    // public GpuContext API delegates; op methods use _pipeline/_uniform/_run.
    this._pipes = new PipelineCache(device);
    this._bufs = new BufferPool(device);
    this._sched = new Scheduler(device);
    // f16 storage needs the device feature AND Float16Array; without either,
    // uploadF16 silently falls back to fp32 (kernels using `enable f16;` would
    // otherwise fail validation asynchronously — no-op dispatches, zero outputs).
    this.hasF16 = !!device.features?.has?.("shader-f16") && typeof Float16Array !== "undefined";
    // Subgroup GEMM backend (adapted from narcotic-sh/parakeet.wgsl, MIT):
    // requires the subgroups feature AND exactly-32-lane subgroups (the kernel
    // geometry assumes lane==row mapping; Apple/NVIDIA report 32/32).
    const info = device.adapterInfo ?? {};
    // Candidate only — probeSubgroups() must CONFIRM (dawn reports the adapter
    // range 4..64 on Apple even though the real size is 32).
    this._sgCandidate = !!device.features?.has?.("subgroups") && (info.subgroupMinSize ?? 32) <= 32 && (info.subgroupMaxSize ?? 32) >= 32;
    this.hasSubgroups32 = false;
    // Surface validation/OOM errors loudly: they are async in WebGPU and would
    // otherwise show up only as silent garbage output.
    device.addEventListener?.("uncapturederror", (e) => console.error("[gpu] uncaptured:", e.error?.message ?? e.error));
  }

  _pipeline(key, code, entry = "main") {
    return this._pipes.get(key, code, entry);
  }

  /** @param {Float32Array} data @returns {{buf:GPUBuffer, rows:number, cols:number}} */
  upload(data, rows, cols) {
    const buf = this.device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.device.queue.writeBuffer(buf, 0, data);
    return { buf, rows, cols };
  }

  /** Normalize a host {data,rows,cols} literal into a backend tensor; backend
   * tensors pass through unchanged. The ONE sanctioned host-vs-tensor check —
   * engine code must not probe .buf/.data itself. */
  ensureTensor(t) {
    return t.buf ? t : this.upload(t.data, t.rows, t.cols);
  }

  // ── buffer pool + arena scopes (BufferPool) ────────────────────────────────
  /** Destroy pooled buffers down to the byte budget. Call ONLY when this
   * context's GPU work is drained (end of a transcribe/synth, after the final
   * readback resolved) — pooled buffers are unreferenced by definition then.
   * Default budget 1GiB — measured knee: below it eviction hits the hot large
   * buffers and warm runs churn hundreds of MB of re-creation; above it is
   * pure idle retention. Consumers can lower ctx.poolBudgetBytes on
   * memory-constrained targets (cost is re-creation time, not correctness). */
  trimPool(budgetBytes) {
    this._bufs.trim(budgetBytes ?? this.poolBudgetBytes ?? 1 << 30);
  }
  /** Pool occupancy (for gates/telemetry). */
  poolInfo() {
    return this._bufs.info();
  }
  pushArena() {
    return this._bufs.pushArena();
  }
  popArena(handle) {
    this._bufs.popArena(handle);
  }
  pin(t, toParent = false) {
    return this._bufs.pin(t, toParent);
  }
  /** One-time async probe: enable the subgroup GEMM only if the device runs
   * 32-lane subgroups with contiguous lane mapping (geometry assumption).
   * Called by createContext(); node gates call it explicitly. */
  async probeSubgroups() {
    if (!this._sgCandidate || this.hasSubgroups32) return this.hasSubgroups32;
    try {
      const mod = this.device.createShaderModule({
        code: `enable subgroups;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) i: u32, @builtin(subgroup_size) ss: u32, @builtin(subgroup_invocation_id) sl: u32) {
  if (i == 0u) { out[0] = ss; }
  if (sl == 0u && ss == 32u) { out[1u + i / 32u] = i; }
}`,
      });
      const pipe = this.device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
      const buf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const stg = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(pipe);
      p.setBindGroup(0, bg);
      p.dispatchWorkgroups(1);
      p.end();
      enc.copyBufferToBuffer(buf, 0, stg, 0, 32);
      this.device.queue.submit([enc.finish()]);
      await stg.mapAsync(GPUMapMode.READ);
      const v = new Uint32Array(stg.getMappedRange().slice(0));
      stg.unmap();
      stg.destroy();
      buf.destroy();
      this.hasSubgroups32 = v[0] === 32 && v[1] === 0 && v[2] === 32 && v[3] === 64 && v[4] === 96;
    } catch {
      this.hasSubgroups32 = false;
    }
    return this.hasSubgroups32;
  }

  /** Allocation counters for the memory gate (created/createdBytes/reused). */
  memStatsStart() {
    this._bufs.statsStart();
  }
  memStats() {
    return this._bufs.stats();
  }

  _allocRaw(size, usage) {
    return this._bufs.allocRaw(size, usage);
  }

  alloc(rows, cols) {
    const size = Math.max(4, rows * cols * 4);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    return { buf: this._allocRaw(size, usage), rows, cols };
  }

  // ── f16 storage ──────────────────────────────────────────────────────────
  /** Upload f32 data as an f16 tensor (half the bytes). Falls back to fp32 when
   * the device lacks shader-f16 (e.g. browsers where the feature is absent). */
  uploadF16(data, rows, cols) {
    if (!this.hasF16) return this.upload(data, rows, cols);
    const u16 = new Uint16Array(new Float16Array(data).buffer);
    const size = Math.max(4, Math.ceil(u16.byteLength / 4) * 4);
    const buf = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.device.queue.writeBuffer(buf, 0, u16);
    return { buf, rows, cols, f16: true };
  }
  /** Upload f32 weights [K,N] PREPACKED tile-major for the direct-B subgroup
   * GEMM (GEMM_TM_WGSL): [N/256 col-tiles][K/32 k-tiles][32 k-rows][32 packs],
   * pack = 8 consecutive f16 columns in a vec4<u32>. Requires K%32==0 &&
   * N%256==0 and 32-lane subgroups (probeSubgroups) — callers fall back to
   * uploadF16 otherwise. Marked {tm:true}; matmul routes on it (opt-in). */
  uploadTileMajorF16(data, K, N) {
    if (!this.hasF16 || K % 32 !== 0 || N % 256 !== 0 || !this.hasSubgroups32) return this.uploadF16(data, K, N);
    const f16 = new Float16Array(data);
    const u16 = new Uint16Array(f16.buffer);
    const packed = new Uint16Array(K * N); // same element count, tile-major order
    let o = 0;
    for (let ct = 0; ct < N / 256; ct++) {
      for (let kt = 0; kt < K / 32; kt++) {
        for (let kr = 0; kr < 32; kr++) {
          const k = kt * 32 + kr;
          const src = k * N + ct * 256;
          packed.set(u16.subarray(src, src + 256), o);
          o += 256;
        }
      }
    }
    const buf = this.device.createBuffer({ size: packed.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.device.queue.writeBuffer(buf, 0, packed);
    return { buf, rows: K, cols: N, f16: true, tm: true };
  }

  /** Direct-B tile-major subgroup GEMM (see uploadTileMajorF16 / GEMM_TM_WGSL). */
  matmulTM(a, b, { bias = null, act = "none", add = null } = {}) {
    const M = a.rows,
      K = a.cols,
      N = b.cols;
    const y = this.alloc(M, N);
    const pipeline = this._pipeline("gemmTM", GEMM_TM_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, add ? 1 : 0, 0, 0]));
    this._run(pipeline, [a.buf, b.buf, bias ? bias.buf : this._dummy(), y.buf, add ? add.buf : this._dummy16()], u, N / 256, Math.ceil(M / 32));
    return y;
  }

  /** Allocate an uninitialized f16 tensor (pooled, arena-scoped like alloc). */
  allocF16(rows, cols) {
    const size = Math.max(4, Math.ceil((rows * cols * 2) / 4) * 4);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    return { buf: this._allocRaw(size, usage), rows, cols, f16: true };
  }
  /** Copy an f16 tensor back to CPU as Float32Array. */
  async downloadF16(t) {
    const n = t.rows * t.cols;
    const size = Math.ceil((n * 2) / 4) * 4;
    const stg = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    // Batch-aware like download(): inside an open batch the copy rides the
    // batch submit (then flush + reopen) so it sees the producing kernels.
    this._sched.encodeCopy((enc) => enc.copyBufferToBuffer(t.buf, 0, stg, 0, size));
    this._sched.flush();
    await stg.mapAsync(GPUMapMode.READ);
    const h = new Float16Array(stg.getMappedRange().slice(0, n * 2));
    const out = Float32Array.from(h);
    stg.unmap();
    stg.destroy();
    return out;
  }

  /** f16 matmul: C = act(A@B + bias). a/b/bias/out all f16 tensors. */
  matmulF16(a, b, { bias = null, act = "none" } = {}) {
    const M = a.rows,
      K = a.cols,
      N = b.cols;
    const c = this.allocF16(M, N);
    const biasBuf = bias ? bias.buf : this.allocF16(1, 1).buf;
    const pipeline = this._pipeline("gemmF16", GEMM_F16_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, 0, 0, 0]));
    this._run(pipeline, [a.buf, b.buf, biasBuf, c.buf], u, Math.ceil(N / 64), Math.ceil(M / 64));
    return c;
  }
  /** Upload raw bytes (packed int4 weights / zero-points) to a storage buffer. */
  uploadBytes(typed) {
    const u32 = typed instanceof Uint32Array ? typed : new Uint32Array(typed.buffer, typed.byteOffset, Math.ceil(typed.byteLength / 4));
    const buf = this.device.createBuffer({ size: Math.max(4, u32.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, u32);
    return { buf };
  }

  /**
   * int4 block-quantized matmul (ONNX MatMulNBits, bits=4, block_size=32) —
   * dequantizes in-shader. a: f32 [M,K]. bq: packed int4 weights [N,nblk,16] (u32
   * buffer). scales: f32 [N*nblk]. zp: packed int4 zero-points [N,zpb] (u32 buffer).
   * Returns f32 [M,N]. Runs on WebGPU where ORT's EP has no int kernel.
   */
  matmulNBits(a, bq, scales, zp, N, blockSize = 32) {
    const M = a.rows,
      K = a.cols;
    const nblk = Math.ceil(K / blockSize);
    const zpb = Math.ceil(nblk / 2);
    const y = this.alloc(M, N);
    const pipeline = this._pipeline("matmulNBits", MATMUL_NBITS_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, nblk, zpb, 0, 0, 0]));
    this._run(pipeline, [a.buf, bq.buf, scales.buf, zp.buf, y.buf], u, Math.ceil((M * N) / 64));
    return y;
  }

  /** f16 fused conv1d (implicit GEMM, groups=1). x/wRows/bias/out all f16. */
  conv1dFastF16(x, wRows, cout, k, { bias = null, stride = 1, pad = 0, dilation = 1, act = "none" } = {}) {
    const Cin = x.rows,
      L = x.cols,
      CinK = Cin * k;
    const Lout = Math.floor((L + 2 * pad - dilation * (k - 1) - 1) / stride) + 1;
    const y = this.allocF16(cout, Lout);
    const biasBuf = bias ? bias.buf : this.allocF16(1, 1).buf;
    const pipeline = this._pipeline("conv1dImplicitF16", CONV1D_IMPLICIT_F16_WGSL);
    const u = this._uniform(new Uint32Array([cout, Lout, CinK, Cin, L, k, stride, pad, dilation, bias ? 1 : 0, ACT[act], 0]));
    this._run(pipeline, [wRows.buf, x.buf, biasBuf, y.buf], u, Math.ceil(Lout / 64), Math.ceil(cout / 64));
    return y;
  }

  _uniform(arr) {
    return this._sched.uniform(arr);
  }

  /** Shared 4-byte dummy buffer for bias-less ops (was a fresh alloc per call). */
  _dummy() {
    if (!this._dummyBuf) this._dummyBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
    return this._dummyBuf;
  }

  /** 16-byte dummy for bindings typed array<vec4<f32>> (min binding size 16). */
  _dummy16() {
    if (!this._dummy16Buf) this._dummy16Buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    return this._dummy16Buf;
  }

  /** Cached per-length zero bias for reroute epilogues (avoids per-call uploads). */
  _zeroBias(n) {
    this._zeroBiasCache = this._zeroBiasCache || new Map();
    let t = this._zeroBiasCache.get(n);
    if (!t) {
      t = this.upload(new Float32Array(n), 1, n);
      this._zeroBiasCache.set(n, t);
    }
    return t;
  }

  // ── command scheduling (Scheduler): batching, dispatch, profiler ──────────
  startProfile(maxOps = 2000) {
    this._sched.startProfile(maxOps);
  }
  async endProfile() {
    return this._sched.endProfile();
  }

  /** Batch mode: queue many kernels into one submit. beginBatch()…endBatch(). */
  beginBatch() {
    this._sched.beginBatch();
  }
  /** Sync batch wrapper for record-only sections (no awaits inside fn). */
  withBatchSync(fn) {
    this.beginBatch();
    try {
      return fn();
    } finally {
      if (this._sched.batchOpen) this.endBatch();
    }
  }

  /** Run fn inside a batch, guaranteed closed on exit (throw included).
   * download() may flush+reopen inside; the finally still sees an open batch. */
  async withBatch(fn) {
    this.beginBatch();
    try {
      return await fn();
    } finally {
      if (this._sched.batchOpen) this.endBatch();
    }
  }

  endBatch() {
    this._sched.endBatch();
  }

  _run(pipeline, buffers, uniform, groupsX, groupsY = 1, groupsZ = 1) {
    this._sched.run(pipeline, buffers, uniform, groupsX, groupsY, groupsZ);
  }

  /** C = act(A@B + bias). a:[M,K] b:[K,N] bias?:[1,N] -> [M,N] */
  matmul(a, b, { bias = null, act = "none", add = null } = {}) {
    const M = a.rows,
      K = a.cols,
      N = b.cols;
    // f16-storage B (weights): f16-COMPUTE v4 (2× ALU rate on Apple; measured
    // 1.46× vs the f32-compute F16B variant, which stays available for A/Bs).
    // `add` (residual [M,N]) fuses into the f16C epilogue; other routes compose
    // with a separate elementwise add so semantics match on every path.
    // Tile-major prepacked weights (uploadTileMajorF16) always route here —
    // the OPT-IN happens at UPLOAD time (weights are only prepacked when
    // __tmGemm === true at load; layout is incompatible with the f16C kernel).
    if (b.tm) {
      return this.matmulTM(a, b, { bias, act, add });
    }
    if (b.f16 && this.hasF16 && K % 4 === 0 && N % 4 === 0) {
      if (!globalThis.__f16bforce) return this.matmulF16C(a, b, { bias, act, add });
      const oB = this.matmulF16B(a, b, { bias, act });
      return add ? this.add(oB, add) : oB;
    }
    // An f16-storage B must never reach the fp32 kernels below (they bind B as
    // array<vec4<f32>> — silent garbage). Fail loudly instead.
    if (b.f16) throw new Error(`matmul: f16 B requires K%4==0 && N%4==0 && supported act (got K=${K}, N=${N}, act=${act})`);
    // Thin-M fp32 GEMV (decode loops): the tile kernels launch only N/64
    // workgroups at M≤4 — occupancy-starved on the pure-bandwidth weight read.
    if (M <= 4 && K >= 64) {
      const y = this.matmulGemv(a, b, { bias, act });
      return add ? this.add(y, add) : y;
    }
    // Large aligned GEMMs benefit from the 128×128/8×8 vec4 kernel (~70% of MLX,
    // vs ~58% for the scalar kernel). Thin/small GEMMs are launch/occupancy-bound —
    // v4 gives no gain there and wastes work padding M/N to 128, so keep v1.
    if (M >= 256 && N >= 256 && K >= 256 && K % 8 === 0 && N % 4 === 0) {
      const o4 = this.matmulV4(a, b, { bias, act });
      return add ? this.add(o4, add) : o4;
    }
    const c = this.alloc(M, N);
    const biasBuf = bias ? bias.buf : this._dummy();
    const pipeline = this._pipeline("gemm", GEMM_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, 0, 0, 0]));
    this._run(pipeline, [a.buf, b.buf, biasBuf, c.buf], u, Math.ceil(N / 64), Math.ceil(M / 64));
    return add ? this.add(c, add) : c;
  }

  /** Split-K GEMV for thin A (M ≤ 4), fp32 B: (N/64)·KS partial workgroups then
   * a reduce pass with fused bias/act. matmul() routes here automatically. */
  matmulGemv(a, b, { bias = null, act = "none" } = {}) {
    const M = a.rows,
      K = a.cols,
      N = b.cols;
    // Slice K down to ~32-step slices (≤64 slices): consecutive dispatches in a
    // pass serialize on barriers, so each GEMV needs enough workgroups on its
    // own to cover the weight-stream latency.
    const KS = Math.max(1, Math.min(Math.floor(K / 32), 64));
    const c = this.alloc(M, N);
    if (N % 4 === 0) {
      const N4 = N / 4;
      const p = this.alloc(KS * M, N);
      const pipe1 = this._pipeline("gemvPart4", GEMV_PART4_WGSL);
      this._run(pipe1, [a.buf, b.buf, p.buf], this._uniform(new Uint32Array([M, N4, K, KS])), Math.ceil(N4 / 64), KS);
      const pipe2 = this._pipeline("gemvReduce4", GEMV_REDUCE4_WGSL);
      const u2 = this._uniform(new Uint32Array([M, N4, KS, ACT[act], bias ? 1 : 0, 0, 0, 0]));
      this._run(pipe2, [p.buf, bias ? bias.buf : this._dummy16(), c.buf], u2, Math.ceil(N4 / 64));
      return c;
    }
    const p = this.alloc(KS * M, N);
    const pipe1 = this._pipeline("gemvPart", GEMV_PART_WGSL);
    this._run(pipe1, [a.buf, b.buf, p.buf], this._uniform(new Uint32Array([M, N, K, KS])), Math.ceil(N / 64), KS);
    const pipe2 = this._pipeline("gemvReduce", GEMV_REDUCE_WGSL);
    const u2 = this._uniform(new Uint32Array([M, N, KS, ACT[act], bias ? 1 : 0, 0, 0, 0]));
    this._run(pipe2, [p.buf, bias ? bias.buf : this._dummy(), c.buf], u2, Math.ceil(N / 64));
    return c;
  }

  matmulV2(a, b, { bias = null, act = "none" } = {}) {
    const M = a.rows,
      K = a.cols,
      N = b.cols;
    const c = this.alloc(M, N);
    const biasBuf = bias ? bias.buf : this._dummy();
    const pipeline = this._pipeline("gemmV2", GEMM_V2_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, 0, 0, 0]));
    this._run(pipeline, [a.buf, b.buf, biasBuf, c.buf], u, Math.ceil(N / 64), Math.ceil(M / 64));
    return c;
  }

  matmulV3(a, b, { bias = null, act = "none" } = {}) {
    const M = a.rows,
      K = a.cols,
      N = b.cols;
    const c = this.alloc(M, N);
    const biasBuf = bias ? bias.buf : this._dummy();
    const pipeline = this._pipeline("gemmV3", GEMM_V3_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, 0, 0, 0]));
    this._run(pipeline, [a.buf, b.buf, biasBuf, c.buf], u, Math.ceil(N / 64), Math.ceil(M / 64));
    return c;
  }

  /** Mixed-precision GEMM: fp32 A × f16-storage B (v4 structure, f32 accumulate). */
  matmulF16B(a, bF16, { bias = null, act = "none" } = {}) {
    const M = a.rows,
      K = a.cols,
      N = bF16.cols;
    const c = this.alloc(M, N);
    const biasBuf = bias ? bias.buf : this._dummy();
    const pipeline = this._pipeline("gemmV4f16b", GEMM_V4_F16B_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, 0, 0, 0]));
    this._run(pipeline, [a.buf, bF16.buf, biasBuf, c.buf], u, Math.ceil(N / 128), Math.ceil(M / 128));
    return c;
  }

  /** f16-COMPUTE v4: f16 tiles + f16 fma per K-tile, f32 accumulate across tiles.
   * `add` fuses a residual [M,N] into the epilogue (post-act), saving a pass. */
  matmulF16C(a, bF16, { bias = null, act = "none", add = null } = {}) {
    const M = a.rows,
      K = a.cols,
      N = bF16.cols;
    const c = this.alloc(M, N);
    const biasBuf = bias ? bias.buf : this._dummy();
    const addBuf = add ? add.buf : this._dummy();
    // Subgroup backend (probe-confirmed 32-lane; design from parakeet.wgsl):
    // barrier-free, no LDS. OPT-IN (globalThis.__sgGemm) — measured 1.25x
    // ISOLATED on M5 but TIED in-context, and COALESCED column ownership
    // (lane*4 split halves, one 128-byte burst per half-tile) did not move it:
    // the constraint is B traffic VOLUME (32-row amortization vs the LDS
    // kernel's 128), which tile-major layout would not change either.
    // Upstream's remaining edge at the GEMM level: f16-STORED A (half our A
    // traffic — our activations are f32) + fixed-shape shaders. Kept opt-in
    // as the basis for the GPU-resident TDT decoder port (task #19), which is
    // the structural piece of their 180x.
    if (this.hasSubgroups32 && globalThis.__sgGemm === true && N % 256 === 0 && K % 16 === 0) {
      const pipeline = this._pipeline("gemmF16sg", GEMM_F16SG_WGSL);
      const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, add ? 1 : 0, 0, 0]));
      this._run(pipeline, [a.buf, bF16.buf, biasBuf, c.buf, addBuf], u, Math.ceil(N / 256), Math.ceil(M / 32));
      return c;
    }
    const bk16 = !globalThis.__f16cbk8 && K % 16 === 0;
    const pipeline = bk16 ? this._pipeline("gemmV4f16c2", GEMM_V4_F16C2_WGSL) : this._pipeline("gemmV4f16c", GEMM_V4_F16C_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, add ? 1 : 0, 0, 0]));
    this._run(pipeline, [a.buf, bF16.buf, biasBuf, c.buf, addBuf], u, Math.ceil(N / 128), Math.ceil(M / 128));
    return c;
  }

  matmulV4(a, b, { bias = null, act = "none" } = {}) {
    const M = a.rows,
      K = a.cols,
      N = b.cols;
    const c = this.alloc(M, N);
    const biasBuf = bias ? bias.buf : this._dummy();
    const pipeline = this._pipeline("gemmV4", GEMM_V4_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, 0, 0, 0]));
    this._run(pipeline, [a.buf, b.buf, biasBuf, c.buf], u, Math.ceil(N / 128), Math.ceil(M / 128));
    return c;
  }

  layernorm(x, gamma, beta, eps = 1e-5) {
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline("layernorm", LAYERNORM_WGSL);
    // Meta: rows,cols (u32), eps (f32), pad — packed into a 16-byte uniform.
    const meta = new ArrayBuffer(16);
    new Uint32Array(meta, 0, 2).set([x.rows, x.cols]);
    new Float32Array(meta, 8, 1)[0] = eps;
    const u = this._uniform(meta);
    this._run(pipeline, [x.buf, gamma.buf, beta.buf, y.buf], u, x.rows);
    return y;
  }

  softmax(x) {
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline("softmax", SOFTMAX_WGSL);
    const u = this._uniform(new Uint32Array([x.rows, x.cols, 0, 0]));
    this._run(pipeline, [x.buf, y.buf], u, x.rows);
    return y;
  }

  /** Gemma RMSNorm rows: y = x·rsqrt(mean(x²)+eps)·(1+w) [+ add]. w:[1,cols]. */
  rmsNorm(x, w, eps = 1e-6, { add = null } = {}) {
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline("rmsnorm", RMSNORM_WGSL);
    const meta = new ArrayBuffer(16);
    new Uint32Array(meta, 0, 3).set([x.rows, x.cols, add ? 1 : 0]);
    new Float32Array(meta, 12, 1)[0] = eps;
    const u = this._uniform(meta);
    this._run(pipeline, [x.buf, w.buf, add ? add.buf : this._dummy(), y.buf], u, x.rows);
    return y;
  }

  /** Fused per-head RMSNorm (skipped when w is null) + rotate-half RoPE on a
   * [rows, heads*headDim] projection. Row r sits at position pos0 + (r % M)
   * (rows are stream-major: CFG/window s occupies rows [s*M, (s+1)*M)).
   * invFreq: HOST Float64Array of headDim/2 inverse frequencies — the WASM
   * backend consumes it at f64 (parity), the GPU caches an f32 copy per array. */
  headRmsRope(x, w, invFreq, { heads, headDim, M, pos0 = 0, scale = 1, eps = 1e-6 }) {
    const y = this.alloc(x.rows, x.cols);
    this._ropeCache = this._ropeCache || new WeakMap();
    let f = this._ropeCache.get(invFreq);
    if (!f) {
      f = this.upload(Float32Array.from(invFreq), 1, invFreq.length);
      this._ropeCache.set(invFreq, f);
    }
    const pipeline = this._pipeline("headrmsrope", HEADRMS_ROPE_WGSL);
    const meta = new ArrayBuffer(32);
    new Uint32Array(meta, 0, 6).set([x.rows, heads, headDim, M, pos0, w ? 1 : 0]);
    new Float32Array(meta, 24, 2).set([scale, eps]);
    const u = this._uniform(meta);
    this._run(pipeline, [x.buf, w ? w.buf : this._dummy(), f.buf, y.buf], u, x.rows * heads);
    return y;
  }

  /**
   * Multi-head attention of M new positions per stream against a stream-strided
   * KV cache: q [W*M, heads*headDim] (W = q.rows/M streams), k/v [W*cacheStride,
   * heads*headDim] with stream w's entries at rows [w*cacheStride, …). Causal:
   * query i attends keys j ≤ pos0+i; fixedT attends a fixed window (bidirectional,
   * e.g. CAS). softcap>0 applies cap·tanh(s/cap) to scores. Returns [W*M, heads*headDim].
   */
  attnCache(q, k, v, { heads, headDim, M, pos0 = 0, cacheStride = 0, causal = true, fixedT = 0, softcap = 0 }) {
    const W = q.rows / M;
    const stride = cacheStride || k.rows / W;
    const Tk = fixedT || pos0 + M;
    const R = W * heads * M;
    const s = this.alloc(R, Tk);
    const pipe = this._pipeline("attnscores", ATTN_SCORES_WGSL);
    const meta = new ArrayBuffer(48);
    new Uint32Array(meta, 0, 8).set([R, Tk, heads, headDim, M, pos0, stride, causal ? 1 : 0]);
    new Float32Array(meta, 32, 1)[0] = softcap;
    this._run(pipe, [q.buf, k.buf, s.buf], this._uniform(meta), Math.ceil((R * Tk) / 64));
    const p = this.softmax(s);
    const y = this.alloc(W * M, heads * headDim);
    const pipe2 = this._pipeline("attnpv", ATTN_PV_WGSL);
    const u2 = this._uniform(new Uint32Array([W * M, Tk, heads, headDim, M, pos0, stride, causal ? 1 : 0]));
    this._run(pipe2, [p.buf, v.buf, y.buf], u2, Math.ceil((W * M * heads * headDim) / 64));
    return y;
  }

  /**
   * 1-D conv. x:[Cin,L] (rows=Cin, cols=L), w GPU tensor of Cout*(Cin/groups)*K
   * f32, bias?:[1,Cout]. Returns [Cout, Lout]. w/bias are passed as GpuTensors
   * (any rows/cols — only .buf is used).
   */
  conv1d(x, w, { cout, k, bias = null, stride = 1, pad = 0, padLeft, padRight, dilation = 1, groups = 1, act = "none" } = {}) {
    // Asymmetric pad supported (padLeft/padRight) for causal convs; default symmetric pad.
    padLeft = padLeft ?? pad;
    padRight = padRight ?? pad;
    // k=1 stride-1 unpadded conv IS a matmul: W[Cout,Cin] @ X[Cin,L] (per-dispatch
    // timestamps showed the pointwise conv-module convs at ~14% of the encoder).
    // Per-row bias/act as an epilogue (matmul's fused bias is per-column).
    if (k === 1 && groups === 1 && stride === 1 && padLeft === 0 && padRight === 0) {
      const out = this.matmul({ buf: w.buf, rows: cout, cols: x.rows }, x);
      if (bias || act !== "none") return this.rowBiasAct(out, bias ?? this._zeroBias(cout), act);
      return out;
    }
    // groups==1 symmetric-pad convs route to the fused implicit-GEMM kernel
    // (~7× the direct kernel on the big vocoder convs; same flat weight layout).
    // Asymmetric-pad and grouped/depthwise convs stay on the direct kernel.
    if (groups === 1 && padLeft === padRight) {
      return this.conv1dFast(x, w, cout, k, { bias, stride, pad: padLeft, dilation, act });
    }
    const Cin = x.rows,
      L = x.cols;
    const Lout = Math.floor((L + padLeft + padRight - dilation * (k - 1) - 1) / stride) + 1;
    const y = this.alloc(cout, Lout);
    const biasBuf = bias ? bias.buf : this._dummy();
    const pipeline = this._pipeline("conv1d", CONV1D_WGSL);
    const u = this._uniform(new Uint32Array([cout, Cin, L, Lout, k, stride, padLeft, dilation, groups, bias ? 1 : 0, ACT[act], 0]));
    this._run(pipeline, [x.buf, w.buf, biasBuf, y.buf], u, Math.ceil((cout * Lout) / 64));
    return y;
  }

  /**
   * Fused TDT joint + argmax for a BATCH of `count` frames starting at `frame`, all
   * sharing predProj (valid until the next emission). encProj:[Tenc,hidden],
   * predProj:[1,hidden], outW:[hidden,logits], outB:[1,logits]. Returns [count,4]:
   * per frame [tokenArgmax, tokenMax, durArgmax, durMax]. hidden must be <= 640.
   * One workgroup per frame → good GPU utilization; one download per batch.
   */
  jointArgmax(encProj, frame, count, predProj, outW, outB, hidden, vocab, logits) {
    const res = this.alloc(count, 4);
    const pipeline = this._pipeline("tdtjoint", TDT_JOINT_WGSL);
    const u = this._uniform(new Uint32Array([frame, hidden, vocab, logits]));
    this._run(pipeline, [encProj.buf, predProj.buf, outW.buf, outB.buf, res.buf], u, count);
    return res;
  }

  /**
   * int8 GEMM: a[M,K] fp32 @ dequant(wq)[K,N] -> [M,N]. wq: GpuTensor over a u32
   * buffer of int8 weights packed 4-per-u32 (row-major [k*N+n]); scale:[1,N] per
   * output column; bias?:[1,N]. Fused act. Weights stay int8 in GPU memory (1/4).
   */
  matmulInt8(a, wq, scale, N, K, { bias = null, act = "none" } = {}) {
    const M = a.rows;
    const y = this.alloc(M, N);
    const biasBuf = bias ? bias.buf : this._dummy();
    const u = this._uniform(new Uint32Array([M, N, K, bias ? 1 : 0, ACT[act], 0, 0, 0]));
    if (M >= 256 && N % 4 === 0 && K % 4 === 0 && globalThis.__i8v2force !== true) {
      const pipeline = this._pipeline("matmulI8v3", MATMUL_INT8_V3_WGSL);
      this._run(pipeline, [a.buf, wq.buf, scale.buf, biasBuf, y.buf], u, Math.ceil(N / 128), Math.ceil(M / 128));
      return y;
    }
    if (N % 4 === 0) {
      const pipeline = this._pipeline("matmulI8v2", MATMUL_INT8_V2_WGSL);
      this._run(pipeline, [a.buf, wq.buf, scale.buf, biasBuf, y.buf], u, Math.ceil(N / 64), Math.ceil(M / 64));
      return y;
    }
    const pipeline = this._pipeline("matmulI8", MATMUL_INT8_WGSL);
    this._run(pipeline, [a.buf, wq.buf, scale.buf, biasBuf, y.buf], u, Math.ceil((M * N) / 64));
    return y;
  }

  /** Subsampling reshape: x[C, Tsub*F] -> [Tsub, C*F] (GPU-resident, no download). */
  subReshape(x, C, Tsub, F) {
    const y = this.alloc(Tsub, C * F);
    const pipeline = this._pipeline("subreshape", SUBRESHAPE_WGSL);
    const u = this._uniform(new Uint32Array([C, Tsub, F, 0]));
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil((Tsub * C * F) / 64));
    return y;
  }

  /** j = relu(encProj[base+i] + predProj) for B frames -> [B, hid]. predProj:[1,hid]. */
  jbatch(encProj, base, B, predProj, hid) {
    const y = this.alloc(B, hid);
    const pipeline = this._pipeline("jbatch", JBATCH_WGSL);
    const u = this._uniform(new Uint32Array([base, B, hid, 0]));
    this._run(pipeline, [encProj.buf, predProj.buf, y.buf], u, Math.ceil((B * hid) / 64));
    return y;
  }

  /** Per-row token+dur argmax of x[B,logits] -> [B,4] (tokenIdx,tokenMax,durIdx,durMax). */
  argmaxRows(x, B, vocab, logits) {
    const res = this.alloc(B, 4);
    const pipeline = this._pipeline("argmaxRows", ARGMAX_ROWS_WGSL);
    const u = this._uniform(new Uint32Array([B, vocab, logits, 0]));
    this._run(pipeline, [x.buf, res.buf], u, B);
    return res;
  }

  /** Batched QK^T / Q·pos^T over all heads: q[T,H*HD], b[Tb,H*HD] → [H*T, Tb]. qb?:[1,H*HD]. */
  bmmQK(q, b, qb, H, HD, W = 1, bShared = false) {
    const T = q.rows / W,
      Tb = bShared ? b.rows : b.rows / W;
    const s = this.alloc(W * H * T, Tb);
    const pipeline = this._pipeline("bmmqk", BMM_QK_WGSL);
    const u = this._uniform(new Uint32Array([T, Tb, H, HD, qb ? 1 : 0, W, bShared ? 1 : 0, 0]));
    this._run(pipeline, [q.buf, b.buf, qb ? qb.buf : this.alloc(1, 1).buf, s.buf], u, Math.ceil(Tb / 64), Math.ceil(T / 64), W * H);
    return s;
  }

  /** Batched probs@V over all heads: p[W*H*Tq, Tk], v[W*Tk, H*HD] → [W*Tq, H*HD]. */
  bmmPV(p, v, H, HD, W = 1) {
    const Tk = v.rows / W,
      Tq = p.rows / (W * H);
    const y = this.alloc(W * Tq, H * HD);
    const pipeline = this._pipeline("bmmpv", BMM_PV_WGSL);
    const u = this._uniform(new Uint32Array([Tq, Tk, H, HD, W, 0, 0, 0]));
    this._run(pipeline, [p.buf, v.buf, y.buf], u, Math.ceil(HD / 64) || 1, Math.ceil(Tq / 64) || 1, W * H);
    return y;
  }

  /** Batched rel_shift: x[H*t, 2t-1] → [H*t, t]. */
  relShiftB(x, H) {
    const t = x.rows / H;
    const y = this.alloc(H * t, t);
    const pipeline = this._pipeline("relshiftb", RELSHIFT_B_WGSL);
    const u = this._uniform(new Uint32Array([t, H, 0, 0]));
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil((H * t * t) / 64));
    return y;
  }

  /**
   * Fused rel-pos attention for FULL (unmasked) windows, T ≤ 256: replaces the
   * bmmQK → relShift → add → softmax → bmmPV chain with one dispatch and no
   * [W·H·T, T] score tensor. q/k/v [W·T, H·HD], pos [2T−1, H·HD] (shared across
   * windows), pbu/pbv [1, H·HD]. Returns [W·T, H·HD].
   */
  attnFused(q, k, v, pos, pbu, pbv, H, HD, W) {
    const T = q.rows / W;
    if (T > 256 || (HD !== 64 && HD !== 128)) return null; // caller falls back to the multi-pass chain
    const QB = globalThis.__attnQb || 4;
    const y = this.alloc(W * T, H * HD);
    const pipeline = this._pipeline(`attnfused${QB}h${HD}`, genAttnFusedWgsl(QB, HD));
    const u = this._uniform(new Uint32Array([T, H, HD, W, pos.rows, QB, 0, 0]));
    this._run(pipeline, [q.buf, k.buf, v.buf, pos.buf, pbu.buf, pbv.buf, y.buf], u, Math.ceil(T / QB), H, W);
    return y;
  }

  /**
   * Streaming rel_shift + chunked-causal mask, rectangular. x[H*n, P] is the
   * bd term of n new queries against a TRUNCATED pos table (row pi ↔ relative
   * distance dMax−pi). Output [H*n, Lk] gathers y[h·n+i, j] = x[h·n+i, dMax−(q−k)]
   * where q = subT+i and k = subT−Lc+j are ABSOLUTE positions — the chunk grid
   * therefore matches the offline mask exactly. Disallowed pairs get −1e4
   * (offline adds −1e4 to scores; both underflow to 0 in softmax).
   */
  relShiftStream(x, { H, n, Lk, dMax, Lc, subT, C, left, right }) {
    const y = this.alloc(H * n, Lk);
    const pipeline = this._pipeline("relshiftstream", RELSHIFT_STREAM_WGSL);
    const u = this._uniform(new Int32Array([H, n, Lk, x.cols, dMax, Lc, subT, C, left, right, 0, 0]));
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil((H * n * Lk) / 64));
    return y;
  }

  /** rel_shift: x = matrix_bd [t, 2t-1] -> [t, t] (relative-position attention). */
  relShift(x) {
    const t = x.rows;
    const y = this.alloc(t, t);
    const pipeline = this._pipeline("relshift", RELSHIFT_WGSL);
    const u = this._uniform(new Uint32Array([t, 0, 0, 0]));
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil((t * t) / 64));
    return y;
  }

  /** SiLU (x*sigmoid(x)) elementwise, same shape. */
  silu(x) {
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline(
      "silu",
      `
      struct Meta { n:u32, _a:u32, _b:u32, _c:u32 };
      @group(0) @binding(0) var<storage, read> X: array<f32>;
      @group(0) @binding(1) var<storage, read_write> Y: array<f32>;
      @group(0) @binding(2) var<uniform> m: Meta;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
        let i = gid.y * (nwg.x * 64u) + gid.x; if (i >= m.n) { return; }
        let v = X[i]; Y[i] = v / (1.0 + exp(-clamp(v, -30.0, 30.0)));
      }`,
    );
    const u = this._uniform(new Uint32Array([x.rows * x.cols, 0, 0, 0]));
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil((x.rows * x.cols) / 64));
    return y;
  }

  /** Elementwise ReLU (standalone; matmul act=relu covers the fused case). */
  relu(x) {
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline(
      "relu",
      `
      struct Meta { n:u32, _a:u32, _b:u32, _c:u32 };
      @group(0) @binding(0) var<storage, read> X: array<f32>;
      @group(0) @binding(1) var<storage, read_write> Y: array<f32>;
      @group(0) @binding(2) var<uniform> m: Meta;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
        let i = gid.y * (nwg.x * 64u) + gid.x; if (i >= m.n) { return; }
        Y[i] = max(X[i], 0.0);
      }`,
    );
    const u = this._uniform(new Uint32Array([x.rows * x.cols, 0, 0, 0]));
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil((x.rows * x.cols) / 64));
    return y;
  }

  /** GLU over channels: x:[2C, T] -> [C, T], y[c,t] = x[c,t]*sigmoid(x[c+C,t]). */
  glu(x) {
    const C = x.rows / 2,
      T = x.cols;
    const y = this.alloc(C, T);
    const pipeline = this._pipeline("glu", GLU_WGSL);
    const u = this._uniform(new Uint32Array([C, T, 0, 0]));
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil((C * T) / 64));
    return y;
  }

  /**
   * 2-D conv (batch 1). x:[Cin, H*W] (rows=Cin), w GPU tensor holding
   * Cout*(Cin/groups)*Kh*Kw f32 (ONNX [Cout,Cin/g,Kh,Kw] flat), bias?:[Cout].
   * Returns [Cout, Ho*Wo]. Supports groups (depthwise) + fused bias/relu/silu.
   */
  conv2d(
    x,
    w,
    {
      cout,
      cin,
      h,
      w: W_,
      kh,
      kw,
      bias = null,
      strideH = 1,
      strideW = 1,
      padH = 0,
      padW = 0,
      padTop,
      padBottom,
      padLeft,
      padRight,
      groups = 1,
      act = "none",
    } = {},
  ) {
    // Asymmetric padding supported (padTop/Bottom/Left/Right); default symmetric padH/padW.
    padTop = padTop ?? padH;
    padBottom = padBottom ?? padH;
    padLeft = padLeft ?? padW;
    padRight = padRight ?? padW;
    // 1×1 stride-1 unpadded conv IS a matmul: W[Cout,Cin] @ X[Cin, H*W]. The naive
    // conv2d kernel was 24% of the encoder window (per-dispatch timestamps); the
    // tiled GEMM does it ~50-100× faster. Per-row bias/act applied as an epilogue.
    if (kh === 1 && kw === 1 && strideH === 1 && strideW === 1 && groups === 1 && padTop === 0 && padBottom === 0 && padLeft === 0 && padRight === 0) {
      const wMat = { buf: w.buf, rows: cout, cols: cin };
      const xMat = { buf: x.buf, rows: cin, cols: h * W_ };
      const out = this.matmul(wMat, xMat);
      if (bias || act !== "none") return this.rowBiasAct(out, bias ?? this._zeroBias(cout), act);
      return out;
    }
    const Ho = Math.floor((h + padTop + padBottom - kh) / strideH) + 1;
    const Wo = Math.floor((W_ + padLeft + padRight - kw) / strideW) + 1;
    const y = this.alloc(cout, Ho * Wo);
    const biasBuf = bias ? bias.buf : this._dummy();
    // Specialized subsampling kernels (9 MACs/output → the generic gather kernel
    // is all index math and launch overhead): depthwise 3×3 s2 keeps the 9
    // weights in registers and computes 4 outputs/thread; cin=1 3×3 s2 computes
    // 4 CHANNELS/thread off one shared 9-value input window.
    if (kh === 3 && kw === 3 && strideH === 2 && strideW === 2) {
      if (groups === cin && cout === cin) {
        const u = this._uniform(new Uint32Array([cout, h, W_, Ho, Wo, padTop, padLeft, bias ? 1 : 0, ACT[act], 0, 0, 0]));
        const WoQ = Math.ceil(Wo / 4);
        this._run(this._pipeline("conv2dDw33s2", CONV2D_DW3X3S2_WGSL), [x.buf, w.buf, biasBuf, y.buf], u, Math.ceil((cout * Ho * WoQ) / 64));
        return y;
      }
      if (cin === 1 && groups === 1 && cout % 4 === 0) {
        const u = this._uniform(new Uint32Array([cout, h, W_, Ho, Wo, padTop, padLeft, bias ? 1 : 0, ACT[act], 0, 0, 0]));
        this._run(this._pipeline("conv2dC133s2", CONV2D_C1_3X3S2_WGSL), [x.buf, w.buf, biasBuf, y.buf], u, Math.ceil(((cout / 4) * Ho * Wo) / 64));
        return y;
      }
    }
    const pipeline = this._pipeline("conv2d", CONV2D_WGSL);
    // kernel Meta padH/padW slots = the "before" (top/left) offset.
    const u = this._uniform(new Uint32Array([cout, cin, h, W_, Ho, Wo, kh, kw, strideH, strideW, padTop, padLeft, groups, bias ? 1 : 0, ACT[act], 0]));
    this._run(pipeline, [x.buf, w.buf, biasBuf, y.buf], u, Math.ceil((cout * Ho * Wo) / 64));
    return y;
  }

  /**
   * 1-D transposed conv. x:[Cin,L], w = Cin*(Cout/groups)*K f32, bias?:[1,Cout].
   * Returns [Cout, Lout]. w/bias passed as GpuTensors (only .buf used).
   */
  convTranspose1d(x, w, { cout, k, bias = null, stride = 1, pad = 0, dilation = 1, groups = 1, outputPadding = 0, act = "none" } = {}) {
    const Cin = x.rows,
      L = x.cols;
    const Lout = (L - 1) * stride - 2 * pad + dilation * (k - 1) + outputPadding + 1;
    // k == stride ⇒ every output position has exactly one kernel tap: route as
    // GEMM Wt[cout·k, Cin] @ x + interleave reshape (~75× the direct gather
    // kernel on the voicechat codec upsamplers). Wt is permuted once per weight
    // tensor on-GPU and cached on it.
    if (stride === k && groups === 1 && dilation === 1 && pad === 0 && outputPadding === 0) {
      let wt = this._ctWtCache?.get(w);
      if (!wt) {
        wt = {
          buf: this._allocRaw(Math.max(4, Cin * cout * k * 4), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST),
          rows: cout * k,
          cols: Cin,
        };
        const pp = this._pipeline("convtWperm", CONVT_WPERM_WGSL);
        this._run(pp, [w.buf, wt.buf], this._uniform(new Uint32Array([Cin, cout, k, 0])), Math.ceil((Cin * cout * k) / 64));
        this._ctWtCache = this._ctWtCache || new WeakMap();
        this._ctWtCache.set(w, wt);
      }
      const cols = this.matmul(wt, x); // [cout*k, L]
      const y = this.alloc(cout, Lout);
      const pipe = this._pipeline("convtReshape", CONVT_RESHAPE_WGSL);
      const u = this._uniform(new Uint32Array([cout, L, k, bias ? 1 : 0, ACT[act], 0, 0, 0]));
      this._run(pipe, [cols.buf, bias ? bias.buf : this._dummy(), y.buf], u, Math.ceil((cout * Lout) / 64));
      return y;
    }
    const y = this.alloc(cout, Lout);
    const biasBuf = bias ? bias.buf : this._dummy();
    const pipeline = this._pipeline("convt1d", CONVT1D_WGSL);
    const u = this._uniform(new Uint32Array([cout, Cin, L, Lout, k, stride, pad, dilation, groups, bias ? 1 : 0, ACT[act], 0]));
    this._run(pipeline, [x.buf, w.buf, biasBuf, y.buf], u, Math.ceil((cout * Lout) / 64));
    return y;
  }

  /** im2col: x[Cin,L] -> [Cin*K, Lout]. */
  im2col(x, k, { stride = 1, pad = 0, dilation = 1 } = {}) {
    const Cin = x.rows,
      L = x.cols;
    const Lout = Math.floor((L + 2 * pad - dilation * (k - 1) - 1) / stride) + 1;
    const cols = this.alloc(Cin * k, Lout);
    const pipeline = this._pipeline("im2col", IM2COL_WGSL);
    const u = this._uniform(new Uint32Array([Cin, L, Lout, k, stride, pad, dilation, 0]));
    this._run(pipeline, [x.buf, cols.buf], u, Math.ceil((Cin * k * Lout) / 64));
    return cols;
  }

  /**
   * conv1d via im2col + tiled GEMM (groups=1). x:[Cin,L], wRows = weight viewed as
   * [Cout, Cin*K]. Returns [Cout, Lout]. Much faster than the direct kernel for the
   * big vocoder convs. (bias handled by the caller / row-add.)
   */
  conv1dGemm(x, wRows, cout, k, { stride = 1, pad = 0, dilation = 1, act = "none" } = {}) {
    const cols = this.im2col(x, k, { stride, pad, dilation }); // [Cin*K, Lout]
    return this.matmul(wRows, cols, { act }); // [Cout, Cin*K] @ [Cin*K, Lout]
  }

  /** AdaIN: instance-norm x[C,L] over time + per-channel affine from style. scale/shift:[C]. */
  adain(x, scale, shift, eps = 1e-5) {
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline("adain", ADAIN_WGSL);
    const meta = new ArrayBuffer(16);
    new Uint32Array(meta, 0, 2).set([x.rows, x.cols]);
    new Float32Array(meta, 8, 1)[0] = eps;
    const u = this._uniform(meta);
    this._run(pipeline, [x.buf, scale.buf, shift.buf, y.buf], u, x.rows);
    return y;
  }

  /**
   * Length regulator: expand x[C, T_text] to [C, T_mel] by repeating each text
   * column per its duration. idxMap is a Uint32Array[T_mel] of source columns
   * (the duration cumsum → frame→token map).
   */
  gatherCols(x, idxMap) {
    const outCols = idxMap.length;
    const y = this.alloc(x.rows, outCols);
    const idxBuf = this.device.createBuffer({ size: Math.max(4, outCols * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(idxBuf, 0, idxMap);
    const pipeline = this._pipeline("gathercols", GATHERCOLS_WGSL);
    const u = this._uniform(new Uint32Array([x.rows, x.cols, outCols, 0]));
    this._run(pipeline, [x.buf, idxBuf, y.buf], u, Math.ceil((x.rows * outCols) / 64));
    return y;
  }

  /** LeakyReLU (elementwise), default slope 0.2 (StyleTTS2 / iSTFTNet). */
  leakyRelu(x, slope = 0.2) {
    const n = x.rows * x.cols;
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline("leaky", LEAKY_WGSL);
    const meta = new ArrayBuffer(16);
    new Uint32Array(meta, 0, 1)[0] = n;
    new Float32Array(meta, 4, 1)[0] = slope;
    const u = this._uniform(meta);
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil(n / 64));
    return y;
  }

  /** Snake activation: y = x + (1/(α+1e-9))·sin(αx)², per-channel α. x:[C,L], alpha:[1,C]. */
  snake(x, alpha) {
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline("snake", SNAKE_WGSL);
    const u = this._uniform(new Uint32Array([x.rows, x.cols, 0, 0]));
    this._run(pipeline, [x.buf, alpha.buf, y.buf], u, Math.ceil((x.rows * x.cols) / 64));
    return y;
  }

  /**
   * Fused conv1d via implicit GEMM (groups=1) — register-blocked, no im2col
   * materialization. x:[Cin,L], wRows = weight as [Cout, Cin*K], bias?:[1,Cout].
   * Returns [Cout, Lout]. The fast path for the big vocaoder convs.
   */
  conv1dFast(x, wRows, cout, k, { bias = null, stride = 1, pad = 0, dilation = 1, act = "none" } = {}) {
    const Cin = x.rows,
      L = x.cols;
    const CinK = Cin * k;
    const Lout = Math.floor((L + 2 * pad - dilation * (k - 1) - 1) / stride) + 1;
    const y = this.alloc(cout, Lout);
    const biasBuf = bias ? bias.buf : this._dummy();
    const pipeline = this._pipeline("conv1dImplicit", CONV1D_IMPLICIT_WGSL);
    const u = this._uniform(new Uint32Array([cout, Lout, CinK, Cin, L, k, stride, pad, dilation, bias ? 1 : 0, ACT[act], 0]));
    this._run(pipeline, [wRows.buf, x.buf, biasBuf, y.buf], u, Math.ceil(Lout / 64), Math.ceil(cout / 64));
    return y;
  }

  /** y = act(x + bias[row]) — per-ROW bias epilogue (for 1×1 convs routed to matmul,
   * whose bias is per output CHANNEL = row, unlike matmul's per-column bias). */
  rowBiasAct(x, bias, act = "none") {
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline(
      "rowbias",
      `
      struct Meta { rows:u32, cols:u32, act:u32, _p:u32 };
      @group(0) @binding(0) var<storage, read> X: array<f32>;
      @group(0) @binding(1) var<storage, read> B: array<f32>;
      @group(0) @binding(2) var<storage, read_write> Y: array<f32>;
      @group(0) @binding(3) var<uniform> m: Meta;
      ${WGSL_ACTF}
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
        let i = gid.y * (nwg.x * 64u) + gid.x;
        if (i >= m.rows * m.cols) { return; }
        Y[i] = actf(X[i] + B[i / m.cols], m.act);
      }`,
    );
    const u = this._uniform(new Uint32Array([x.rows, x.cols, ACT[act], 0]));
    this._run(pipeline, [x.buf, bias.buf, y.buf], u, Math.ceil((x.rows * x.cols) / 64));
    return y;
  }

  /** Multiply by a scalar constant (elementwise). */
  scale(x, s) {
    const n = x.rows * x.cols;
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline(
      "scalek",
      `
      struct Meta { n:u32, s:f32, _a:u32, _b:u32 };
      @group(0) @binding(0) var<storage, read> X: array<f32>;
      @group(0) @binding(1) var<storage, read_write> Y: array<f32>;
      @group(0) @binding(2) var<uniform> m: Meta;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
        let i = gid.y * (nwg.x * 64u) + gid.x; if (i >= m.n) { return; }
        Y[i] = X[i] * m.s;
      }`,
    );
    const meta = new ArrayBuffer(16);
    new Uint32Array(meta, 0, 1)[0] = n;
    new Float32Array(meta, 4, 1)[0] = s;
    const u = this._uniform(meta);
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil(n / 64));
    return y;
  }

  /** Write src's rows into dst starting at rowOffset (contiguous, no readback).
   * Batch-safe: inside beginBatch/endBatch the copy is recorded into the SAME
   * command encoder (pass paused/reopened) so it stays ordered after the
   * dispatches that produce src. */
  copyRows(dst, src, rowOffset) {
    this._sched.encodeCopy((enc) => enc.copyBufferToBuffer(src.buf, 0, dst.buf, rowOffset * dst.cols * 4, src.rows * src.cols * 4));
    return dst;
  }

  /** Rows [row0, row0+count) of x[rows,cols] — contiguous, so a plain buffer
   * copy (batch-safe like copyRows: pass paused/reopened inside a batch). */
  sliceRows(x, row0, count) {
    const out = this.alloc(count, x.cols);
    this._sched.encodeCopy((enc) => enc.copyBufferToBuffer(x.buf, row0 * x.cols * 4, out.buf, 0, count * x.cols * 4));
    return out;
  }

  /** Return a persistent fp32 tensor (pinned cache / upload) to the pool. Only
   * for alloc/upload-usage tensors (STORAGE|COPY_SRC|COPY_DST, fp32). Pool-put
   * never destroys, so recorded-but-unsubmitted readers stay valid — call after
   * the batch that last read the tensor has closed. */
  freeTensor(t) {
    if (!t || !t.buf || t.f16 || t.view) return;
    const size = Math.max(4, t.rows * t.cols * 4);
    this._bufs.put(t.buf, size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  }

  /** Concat along rows (row-major ⇒ contiguous buffer concatenation, no readback). */
  concatRows(tensors) {
    // Batch-safe like copyRows: inside beginBatch/endBatch the copies are recorded
    // into the SAME encoder (pass paused/reopened) so they stay ordered after the
    // dispatches that produce the sources.
    const cols = tensors[0].cols;
    const rows = tensors.reduce((s, t) => s + t.rows, 0);
    const out = this.alloc(rows, cols);
    this._sched.encodeCopy((enc) => {
      let off = 0;
      for (const t of tensors) {
        enc.copyBufferToBuffer(t.buf, 0, out.buf, off * 4, t.rows * t.cols * 4);
        off += t.rows * t.cols;
      }
    });
    return out;
  }

  /** Elementwise. b broadcast over rows if b.rows===1. */
  ewise(a, b, op) {
    const n = a.rows * a.cols;
    const c = this.alloc(a.rows, a.cols);
    const pipeline = this._pipeline("ewise", EWISE_WGSL);
    const u = this._uniform(new Uint32Array([n, a.cols, op === "mul" ? 1 : 0, b.rows]));
    this._run(pipeline, [a.buf, b.buf, c.buf], u, Math.ceil(n / 64));
    return c;
  }
  add(a, b) {
    return this.ewise(a, b, "add");
  }
  mul(a, b) {
    return this.ewise(a, b, "mul");
  }

  /**
   * Bidirectional LSTM (ONNX semantics, iofc gates, batch 1). x:[seq,inp];
   * w/r/b are GPU tensors holding W[2,4H,inp], R[2,4H,H], B[2,8H] flat.
   * Returns Y:[seq, 2*hid] = [fwd | bwd]. H must be <= 256.
   */
  lstm(x, w, r, b, hid) {
    const seq = x.rows,
      inp = x.cols;
    const y = this.alloc(seq, 2 * hid);
    const pipeline = this._pipeline("lstm", LSTM_WGSL);
    const u = this._uniform(new Uint32Array([seq, inp, hid, 0]));
    this._run(pipeline, [x.buf, w.buf, r.buf, b.buf, y.buf], u, 2); // 2 workgroups (fwd/bwd)
    return y;
  }

  /** 2-D transpose: [rows,cols] -> [cols,rows]. */
  transpose(x) {
    const y = this.alloc(x.cols, x.rows);
    const pipeline = this._pipeline("transpose", TRANSPOSE_WGSL);
    const u = this._uniform(new Uint32Array([x.rows, x.cols, 0, 0]));
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil((x.rows * x.cols) / 64));
    return y;
  }

  /** Extract columns [col0, col0+width) from x[rows,cols] -> [rows,width]. */
  sliceCols(x, col0, width) {
    const y = this.alloc(x.rows, width);
    const pipeline = this._pipeline("slicecols", SLICECOLS_WGSL);
    const u = this._uniform(new Uint32Array([x.rows, x.cols, col0, width]));
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil((x.rows * width) / 64));
    return y;
  }

  /** Write src[rows,width] into dst[rows,cols] at column col0 (in place). */
  setCols(dst, src, col0) {
    const pipeline = this._pipeline("setcols", SETCOLS_WGSL);
    const u = this._uniform(new Uint32Array([src.rows, dst.cols, col0, src.cols]));
    this._run(pipeline, [src.buf, dst.buf], u, Math.ceil((src.rows * src.cols) / 64));
    return dst;
  }

  /** Copy a GPU tensor back to CPU. Inside an open batch the staging copy is
   * recorded INTO the batch, which is then flushed (submitted) and reopened —
   * so callers may interleave downloads with batched work freely: one submit
   * per stretch between downloads instead of one per op. */
  async download(t) {
    const staged = this.stageDownload(t);
    this._sched.flush();
    return staged.read();
  }

  /** Record a copy of t into a fresh MAP_READ staging buffer; read() maps it later.
   * Batch-aware: inside beginBatch/endBatch the copy rides the SAME submit as the
   * kernels that produce t (pass paused/reopened like copyRows), so the bytes are
   * mappable the instant that submit drains — no extra submit or GPU round trip.
   * Call read() only after the producing batch has been submitted. */
  stageDownload(t) {
    const size = t.rows * t.cols * 4;
    const stg = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this._sched.encodeCopy((enc) => enc.copyBufferToBuffer(t.buf, 0, stg, 0, size));
    let consumed = false;
    return {
      read: async () => {
        if (consumed) throw new Error("StagedRead.read(): already consumed (single-use)");
        consumed = true;
        await stg.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(stg.getMappedRange().slice(0));
        stg.unmap();
        stg.destroy();
        return out;
      },
    };
  }

  /** Rows-limited view over a preallocated tensor (no copy — aliases the same
   * storage). For growing caches: allocate [maxLen, cols] once, view [n, cols].
   * Spread preserves routing metadata (f16/tm); `view` marks it pool-exempt —
   * freeing a view would pool the parent's live buffer under a wrong size key. */
  rowsView(t, rows) {
    return { ...t, rows, view: true };
  }

  /** Tear down the backend: destroys the GPUDevice (pooled buffers die with it). */
  destroy() {
    this.device.destroy();
  }
}
