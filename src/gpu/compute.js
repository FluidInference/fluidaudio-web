// Raw-WebGPU compute core — hand-written WGSL kernels, GPU-resident tensors.
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

const TILE = 16;

// C = act(A[MxK] @ B[KxN] + bias[N]).  bias optional (bound as a 1-elem buffer +
// flag), act: 0 none / 1 gelu(tanh approx) / 2 tanh / 3 relu.
const GEMM_WGSL = `
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
const TS = 16u;
var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;
fn gelu(x: f32) -> f32 {
  // Clamp the tanh argument: Metal's tanh computes exp(x) directly, so a large
  // argument overflows to Inf/Inf = NaN (CPU Math.tanh saturates). tanh(±20) is
  // already ±1 to full f32 precision, so clamping is exact here.
  let t = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -20.0, 20.0);
  return 0.5 * x * (1.0 + tanh(t));
}
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let row = w.y * TS + l.y;
  let col = w.x * TS + l.x;
  var acc = 0.0;
  let nT = (m.K + TS - 1u) / TS;
  for (var t = 0u; t < nT; t++) {
    let aC = t * TS + l.x;
    let bR = t * TS + l.y;
    As[l.y * TS + l.x] = select(0.0, A[row * m.K + aC], row < m.M && aC < m.K);
    Bs[l.y * TS + l.x] = select(0.0, B[bR * m.N + col], bR < m.K && col < m.N);
    workgroupBarrier();
    for (var k = 0u; k < TS; k++) { acc += As[l.y * TS + k] * Bs[k * TS + l.x]; }
    workgroupBarrier();
  }
  if (row < m.M && col < m.N) {
    if (m.hasBias == 1u) { acc += bias[col]; }
    if (m.act == 1u) { acc = gelu(acc); }
    else if (m.act == 2u) { acc = tanh(acc); }
    else if (m.act == 3u) { acc = max(acc, 0.0); }
    C[row * m.N + col] = acc;
  }
}`;

// Row-wise LayerNorm over the last dim: y = (x-mean)/sqrt(var+eps) * gamma + beta.
// One workgroup per row; 64 lanes cooperatively reduce via shared memory.
const LAYERNORM_WGSL = `
struct Meta { rows:u32, cols:u32, eps:f32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wg.x;
  if (row >= m.rows) { return; }
  let base = row * m.cols;
  var sum = 0.0;
  for (var j = li; j < m.cols; j += 64u) { sum += X[base + j]; }
  red[li] = sum;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let mean = red[0] / f32(m.cols);
  workgroupBarrier();
  var vs = 0.0;
  for (var j = li; j < m.cols; j += 64u) { let d = X[base + j] - mean; vs += d * d; }
  red[li] = vs;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(m.cols) + m.eps);
  for (var j = li; j < m.cols; j += 64u) {
    Y[base + j] = (X[base + j] - mean) * inv * gamma[j] + beta[j];
  }
}`;

// Row-wise softmax over the last dim (numerically stable).
const SOFTMAX_WGSL = `
struct Meta { rows:u32, cols:u32, _p0:u32, _p1:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let row = g.x;
  if (row >= m.rows) { return; }
  let base = row * m.cols;
  var mx = X[base];
  for (var j = 1u; j < m.cols; j++) { mx = max(mx, X[base + j]); }
  var s = 0.0;
  for (var j = 0u; j < m.cols; j++) { let e = exp(X[base + j] - mx); Y[base + j] = e; s += e; }
  let inv = 1.0 / s;
  for (var j = 0u; j < m.cols; j++) { Y[base + j] = Y[base + j] * inv; }
}`;

// Elementwise C = A (op) B, with B broadcast over rows when B is [1,cols].
// op: 0 add / 1 mul.
const EWISE_WGSL = `
struct Meta { n:u32, cols:u32, op:u32, bRows:u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= m.n) { return; }
  let bIdx = select(i, i % m.cols, m.bRows == 1u);
  let a = A[i]; let b = B[bIdx];
  C[i] = select(a + b, a * b, m.op == 1u);
}`;

const ACT = { none: 0, gelu: 1, tanh: 2, relu: 3 };

export class GpuContext {
  /** @param {GPUDevice} device */
  constructor(device) {
    this.device = device;
    this.pipelines = new Map();
  }

  _pipeline(key, code, entry = "main") {
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: entry } });
      this.pipelines.set(key, p);
    }
    return p;
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

  /** Allocate an uninitialized GPU tensor. */
  alloc(rows, cols) {
    const buf = this.device.createBuffer({
      size: Math.max(4, rows * cols * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    return { buf, rows, cols };
  }

  _uniform(arr) {
    const buf = this.device.createBuffer({ size: arr.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, arr);
    return buf;
  }

  _run(pipeline, buffers, uniform, groupsX, groupsY = 1) {
    const entries = buffers.map((b, i) => ({ binding: i, resource: { buffer: b } }));
    entries.push({ binding: buffers.length, resource: { buffer: uniform } });
    const bg = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(groupsX, groupsY);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** C = act(A@B + bias). a:[M,K] b:[K,N] bias?:[1,N] -> [M,N] */
  matmul(a, b, { bias = null, act = "none" } = {}) {
    const M = a.rows, K = a.cols, N = b.cols;
    const c = this.alloc(M, N);
    const biasBuf = bias ? bias.buf : this.alloc(1, 1).buf;
    const pipeline = this._pipeline("gemm", GEMM_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, 0, 0, 0]));
    this._run(pipeline, [a.buf, b.buf, biasBuf, c.buf], u, Math.ceil(N / TILE), Math.ceil(M / TILE));
    return c;
  }

  layernorm(x, gamma, beta, eps = 1e-5) {
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline("layernorm", LAYERNORM_WGSL);
    // Meta: rows,cols (u32), eps (f32), pad — packed into a 16-byte uniform.
    const meta = new ArrayBuffer(16);
    new Uint32Array(meta, 0, 2).set([x.rows, x.cols]);
    new Float32Array(meta, 8, 1)[0] = eps;
    const u = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(u, 0, meta);
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

  /** Elementwise. b broadcast over rows if b.rows===1. */
  ewise(a, b, op) {
    const n = a.rows * a.cols;
    const c = this.alloc(a.rows, a.cols);
    const pipeline = this._pipeline("ewise", EWISE_WGSL);
    const u = this._uniform(new Uint32Array([n, a.cols, op === "mul" ? 1 : 0, b.rows]));
    this._run(pipeline, [a.buf, b.buf, c.buf], u, Math.ceil(n / 64));
    return c;
  }
  add(a, b) { return this.ewise(a, b, "add"); }
  mul(a, b) { return this.ewise(a, b, "mul"); }

  /** Copy a GPU tensor back to CPU. */
  async download(t) {
    const size = t.rows * t.cols * 4;
    const stg = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(t.buf, 0, stg, 0, size);
    this.device.queue.submit([enc.finish()]);
    await stg.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(stg.getMappedRange().slice(0));
    stg.unmap();
    stg.destroy();
    return out;
  }
}
