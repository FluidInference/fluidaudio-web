// Authoritative types for the compute layer. ONE interface — ComputeContext —
// describes what every backend (GpuContext / WasmContext) implements; the
// GpuContext class adds the WebGPU-only surface (tile-major f16 weights, fused
// attention, profiling, pool stats). Engine code should depend on
// ComputeContext and treat tensor storage as opaque: use download /
// stageDownload / rowsView / ensureTensor instead of touching t.buf or t.data.
// Conformance of both backends to the required members is asserted by
// scripts/interface-conformance.mjs (runs in ci:smoke).

/** 2-D row-major f32 tensor. Storage is backend-owned and opaque to engine
 * code (WebGPU: GPUBuffer, WASM: Float32Array). */
export interface Tensor {
  rows: number;
  cols: number;
}

/** WebGPU-resident tensor (GpuContext). */
export interface GpuTensor extends Tensor {
  buf: GPUBuffer;
}

/** CPU-side tensor literal (weights loaded from disk, host-computed arrays)
 * accepted by ensureTensor(). */
export interface HostTensor {
  data: Float32Array;
  rows: number;
  cols: number;
}

/** Fused activations. Codes shared by every GEMM/conv kernel (kernels/actf.js). */
export type Activation = "none" | "gelu" | "gelu_erf" | "tanh" | "relu" | "silu";

/** Opaque packed-weight handle from uploadBytes (int8 / int4 payloads). */
export type PackedBytes = { buf: GPUBuffer } | { bytes: Uint8Array };

/** Opaque allocation-scope handle from pushArena. */
export type ArenaHandle = unknown;

/** Staged readback: the copy is recorded at stage time (queue-ordered); read()
 * delivers the bytes once the producing submit drains. SINGLE-USE: the staging
 * storage is released on first read(); a second read() throws on every
 * backend. The returned array is a fresh copy the caller may mutate. */
export interface StagedRead {
  read(): Promise<Float32Array>;
}

export interface MatmulOpts<T extends Tensor = Tensor> {
  bias?: T | null;
  act?: Activation;
  /** Fused residual add ([M,N], applied after act). */
  add?: T | null;
}

/** Sliding-window geometry shared by the conv family. */
export interface ConvWindowOpts {
  stride?: number;
  pad?: number;
  dilation?: number;
}

export interface Conv1dOpts<T extends Tensor = Tensor> extends ConvWindowOpts {
  cout: number;
  k: number;
  bias?: T | null;
  padLeft?: number;
  padRight?: number;
  groups?: number;
  act?: Activation;
}

export interface Conv1dFastOpts<T extends Tensor = Tensor> extends ConvWindowOpts {
  bias?: T | null;
  act?: Activation;
}

export interface ConvTranspose1dOpts<T extends Tensor = Tensor> extends Conv1dOpts<T> {
  outputPadding?: number;
}

export interface Conv2dOpts<T extends Tensor = Tensor> {
  cout: number;
  cin: number;
  h: number;
  w: number;
  kh: number;
  kw: number;
  bias?: T | null;
  strideH?: number;
  strideW?: number;
  padH?: number;
  padW?: number;
  padTop?: number;
  padBottom?: number;
  padLeft?: number;
  padRight?: number;
  groups?: number;
  act?: Activation;
}

/**
 * The backend-independent compute interface. Both backends implement every
 * non-optional member; the optional block at the bottom is the WebGPU-only
 * surface, present when the corresponding capability exists (feature-test the
 * member, e.g. `if (ctx.attnFused)`).
 */
export interface ComputeContext<T extends Tensor = Tensor> {
  readonly backend: "webgpu" | "wasm";

  // ── tensors & memory ──────────────────────────────────────────────────────
  upload(data: Float32Array, rows: number, cols: number): T;
  /** f16 storage upload; silently falls back to fp32 where f16 is unavailable. */
  uploadF16(data: Float32Array, rows: number, cols: number): T;
  alloc(rows: number, cols: number): T;
  allocF16(rows: number, cols: number): T;
  /** Normalize a host {data,rows,cols} literal into a backend tensor; backend
   * tensors pass through unchanged. The ONE sanctioned host-vs-tensor check. */
  ensureTensor(t: HostTensor | T): T;
  /** Upload raw packed bytes (int8 / int4 weights, zero-points). */
  uploadBytes(typed: Uint8Array | Uint32Array): PackedBytes;
  /** Return a tensor's storage to the pool immediately (upload() is pool-exempt;
   * rowsView views are ignored). */
  freeTensor(t: T): void;
  /** Exempt a tensor from its arena: persistent by default, or promote to the
   * enclosing arena with toParent (for tensors that outlive an inner scope). */
  pin(t: T, toParent?: boolean): T;
  /** Open an allocation scope; allocs land in the newest open scope. Scopes may
   * close out of order — popArena takes the handle. */
  pushArena(): ArenaHandle;
  popArena(handle?: ArenaHandle): void;
  /** Evict pooled buffers down to the budget. Call only when the queue is drained. */
  trimPool(budgetBytes?: number): void;
  /** Tear down the backend (WebGPU: destroys the device; WASM: no-op). */
  destroy(): void;

  // ── readback ──────────────────────────────────────────────────────────────
  /** Copy a tensor back to the CPU (always a copy; callers may mutate it). */
  download(t: T): Promise<Float32Array>;
  downloadF16(t: T): Promise<Float32Array>;
  /** Record the readback copy now (it rides the current batch's submit); map /
   * read later. Call read() only after the producing batch has been submitted. */
  stageDownload(t: T): StagedRead;
  /** Rows-limited view over a preallocated tensor (no copy — aliases storage;
   * preserves f16/tm routing metadata). For growing caches: allocate
   * [maxLen, cols] once, view [n, cols]. Views are pool-exempt. */
  rowsView(t: T, rows: number): T;

  // ── batching ──────────────────────────────────────────────────────────────
  beginBatch(): void;
  endBatch(): void;
  withBatch<R>(fn: () => Promise<R> | R): Promise<R>;
  withBatchSync<R>(fn: () => R): R;

  // ── linear algebra ────────────────────────────────────────────────────────
  /** C = act(A[M,K] @ B[K,N] + bias[1,N]) [+ add]. */
  matmul(a: T, b: T, opts?: MatmulOpts<T>): T;
  /** Explicit GEMM kernel-variant entry points (matmul() routes automatically;
   * these pin one variant — used by the bench/verify harnesses). */
  matmulV2(a: T, b: T, opts?: Omit<MatmulOpts<T>, "add">): T;
  matmulV3(a: T, b: T, opts?: Omit<MatmulOpts<T>, "add">): T;
  /** Split-K GEMV for thin A (M ≤ 4), fp32 B (matmul() routes here automatically). */
  matmulGemv(a: T, b: T, opts?: Omit<MatmulOpts<T>, "add">): T;
  matmulV4(a: T, b: T, opts?: Omit<MatmulOpts<T>, "add">): T;
  /** f16-compute GEMM (fp32 fallback where f16 is unavailable). */
  matmulF16(a: T, b: T, opts?: Omit<MatmulOpts<T>, "add">): T;
  /** int8 GEMM with in-shader dequant; scale per output column. */
  matmulInt8(a: T, wq: PackedBytes, scale: T, N: number, K: number, opts?: Omit<MatmulOpts<T>, "add">): T;
  /** int4 block-quant matmul (ONNX MatMulNBits, bits=4). */
  matmulNBits(a: T, bq: PackedBytes, scales: T, zp: PackedBytes, N: number, blockSize?: number): T;
  layernorm(x: T, gamma: T, beta: T, eps?: number): T;
  softmax(x: T): T;
  /** Gemma RMSNorm rows: y = x·rsqrt(mean(x²)+eps)·(1+w) [+ add]. w:[1,cols]. */
  rmsNorm(x: T, w: T, eps?: number, opts?: { add?: T | null }): T;

  // ── convolution ───────────────────────────────────────────────────────────
  /** 1-D conv. x:[Cin,L], w = Cout*(Cin/groups)*K f32 -> [Cout,Lout]. */
  conv1d(x: T, w: T, opts: Conv1dOpts<T>): T;
  /** conv1d via im2col + tiled GEMM (groups=1). wRows = [Cout, Cin*K]. */
  conv1dGemm(x: T, wRows: T, cout: number, k: number, opts?: ConvWindowOpts & { act?: Activation }): T;
  /** Fused conv1d via implicit GEMM (groups=1), no im2col materialization. */
  conv1dFast(x: T, wRows: T, cout: number, k: number, opts?: Conv1dFastOpts<T>): T;
  conv1dFastF16(x: T, wRows: T, cout: number, k: number, opts?: Conv1dFastOpts<T>): T;
  /** 2-D conv (batch 1). x:[Cin,H*W] rows=Cin -> [Cout,Ho*Wo]. */
  conv2d(x: T, w: T, opts: Conv2dOpts<T>): T;
  /** 1-D transposed conv. x:[Cin,L], w = Cin*(Cout/groups)*K f32 -> [Cout,Lout]. */
  convTranspose1d(x: T, w: T, opts: ConvTranspose1dOpts<T>): T;
  /** im2col: x[Cin,L] -> [Cin*K, Lout]. */
  im2col(x: T, k: number, opts?: ConvWindowOpts): T;

  // ── recurrent ─────────────────────────────────────────────────────────────
  /** Bidirectional LSTM (ONNX iofc, batch 1). w/r/b flat: W[2,4H,inp] R[2,4H,H] B[2,8H]. -> [seq, 2*hid]. H<=256. */
  lstm(x: T, w: T, r: T, b: T, hid: number): T;

  // ── attention ─────────────────────────────────────────────────────────────
  /** Batched multi-head QK^T over head-strided [T, H*HD] projections -> [(W*H)*Tq, Tb]. */
  bmmQK(q: T, b: T, qb: T | null, H: number, HD: number, W?: number, bShared?: boolean): T;
  /** Batched probs @ values -> [W*Tq, H*HD]. */
  bmmPV(p: T, v: T, H: number, HD: number, W?: number): T;
  /** rel_shift: [t, 2t-1] -> [t, t]. */
  relShift(x: T): T;
  /** Batched rel_shift: [H*t, 2t-1] -> [H*t, t]. */
  relShiftB(x: T, H: number): T;
  /** Streaming rel_shift + chunk mask for cache-aware attention. */
  relShiftStream(x: T, opts: { H: number; n: number; Lk: number; dMax: number; Lc: number; subT: number; C: number; left: number; right: number }): T;
  /** Per-head RMSNorm (skipped when w is null) + rotate-half RoPE on a
   * [rows, heads*headDim] projection; row r sits at position pos0 + (r % M).
   * invFreq is a HOST array (f64 on WASM — parity; f32 upload cached on GPU). */
  headRmsRope(x: T, w: T | null, invFreq: Float64Array, opts: { heads: number; headDim: number; M: number; pos0?: number; scale?: number; eps?: number }): T;
  /** Multi-head attention of M positions per stream against a stream-strided KV
   * cache (stream w's keys/values at rows [w*cacheStride, …)). Causal masks
   * j > pos0+i; fixedT attends a fixed bidirectional window; softcap>0 applies
   * cap·tanh(s/cap) to scores. Returns [W*M, heads*headDim], W = q.rows/M. */
  attnCache(
    q: T,
    k: T,
    v: T,
    opts: { heads: number; headDim: number; M: number; pos0?: number; cacheStride?: number; causal?: boolean; fixedT?: number; softcap?: number },
  ): T;

  // ── elementwise & data movement ───────────────────────────────────────────
  ewise(a: T, b: T, op: "add" | "mul"): T;
  add(a: T, b: T): T;
  mul(a: T, b: T): T;
  scale(x: T, s: number): T;
  silu(x: T): T;
  relu(x: T): T;
  leakyRelu(x: T, slope?: number): T;
  /** Snake activation x + sin^2(alpha x)/alpha, alpha per channel. */
  snake(x: T, alpha: T): T;
  /** GLU over channels: [2C, T] -> [C, T]. */
  glu(x: T): T;
  /** AdaIN: instance-norm x[C,L] over time + per-channel affine. */
  adain(x: T, scale: T, shift: T, eps?: number): T;
  transpose(x: T): T;
  sliceCols(x: T, col0: number, width: number): T;
  /** Write src[rows,width] into dst at column col0 (in place); returns dst. */
  setCols(dst: T, src: T, col0: number): T;
  sliceRows(x: T, row0: number, count: number): T;
  /** Copy src into dst starting at rowOffset (in place); returns dst. */
  copyRows(dst: T, src: T, rowOffset: number): T;
  concatRows(tensors: T[]): T;
  /** Length regulator: expand x[C,T] to [C, idxMap.length] by column gather. */
  gatherCols(x: T, idxMap: Uint32Array): T;

  // ── decoder-side fusions ──────────────────────────────────────────────────
  /** FastConformer subsampling reshape: [C, Tsub*F] -> [Tsub, C*F]. */
  subReshape(x: T, C: number, Tsub: number, F: number): T;
  /** j = relu(encProj[base+i] + predProj) for B frames -> [B, hid]. */
  jbatch(encProj: T, base: number, B: number, predProj: T, hid: number): T;
  /** Per-row token/duration argmax -> [B, 4] = [tokIdx, tokMax, durIdx, durMax]. */
  argmaxRows(x: T, B: number, vocab: number, logits: number): T;
  /** Fused TDT joint + argmax for `count` frames -> [count, 4]. */
  jointArgmax(encProj: T, frame: number, count: number, predProj: T, outW: T, outB: T, hidden: number, vocab: number, logits: number): T;

  // ── WebGPU-only surface (feature-test before use) ─────────────────────────
  /** f16 storage available (shader-f16 + Float16Array). */
  hasF16?: boolean;
  /** Probe-confirmed 32-lane contiguous subgroups (enables the subgroup GEMM). */
  hasSubgroups32?: boolean;
  probeSubgroups?(): Promise<boolean>;
  /** Prepack weights tile-major f16 for the direct-B subgroup GEMM. */
  uploadTileMajorF16?(data: Float32Array, K: number, N: number): T;
  matmulTM?(a: T, b: T, opts?: MatmulOpts<T>): T;
  matmulF16B?(a: T, bF16: T, opts?: Omit<MatmulOpts<T>, "add">): T;
  matmulF16C?(a: T, bF16: T, opts?: MatmulOpts<T>): T;
  /** Fused rel-pos self-attention (QK^T + rel-pos + softmax + PV, one dispatch). */
  attnFused?(q: T, k: T, v: T, pos: T, pbu: T, pbv: T, H: number, HD: number, W: number): T;
  rowBiasAct?(x: T, bias: T, act?: Activation): T;
  startProfile?(maxOps?: number): void;
  endProfile?(): Promise<Array<{ label: string; ms: number }>>;
  memStatsStart?(): void;
  memStats?(): { created: number; createdBytes: number; reused: number } | undefined;
  poolInfo?(): { bytes: number };
}

/** Request a WebGPU device in the browser (throws if unavailable). */
export function requestGpuDevice(): Promise<GPUDevice>;

/**
 * Raw-WebGPU compute core: hand-written WGSL kernels (src/gpu/kernels/) over
 * GPU-resident tensors. Pass a GPUDevice (navigator.gpu in the browser, dawn
 * in Node). Only download/stageDownload copy back to the CPU.
 *
 * The shared surface comes from ComputeContext<GpuTensor> via declaration
 * merging; the class declares only the constructor and the WebGPU-only
 * members (required here, optional on ComputeContext).
 */
export class GpuContext {
  constructor(device: GPUDevice);
  device: GPUDevice;
  hasF16: boolean;
  hasSubgroups32: boolean;
  probeSubgroups(): Promise<boolean>;
  uploadTileMajorF16(data: Float32Array, K: number, N: number): GpuTensor;
  matmulTM(a: GpuTensor, b: GpuTensor, opts?: MatmulOpts<GpuTensor>): GpuTensor;
  matmulF16B(a: GpuTensor, bF16: GpuTensor, opts?: Omit<MatmulOpts<GpuTensor>, "add">): GpuTensor;
  matmulF16C(a: GpuTensor, bF16: GpuTensor, opts?: MatmulOpts<GpuTensor>): GpuTensor;
  attnFused(q: GpuTensor, k: GpuTensor, v: GpuTensor, pos: GpuTensor, pbu: GpuTensor, pbv: GpuTensor, H: number, HD: number, W: number): GpuTensor;
  rowBiasAct(x: GpuTensor, bias: GpuTensor, act?: Activation): GpuTensor;
  startProfile(maxOps?: number): void;
  endProfile(): Promise<Array<{ label: string; ms: number }>>;
  memStatsStart(): void;
  memStats(): { created: number; createdBytes: number; reused: number } | undefined;
  poolInfo(): { bytes: number };
}
export interface GpuContext extends ComputeContext<GpuTensor> {}
