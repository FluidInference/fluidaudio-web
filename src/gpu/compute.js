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

// 1-D convolution (batch 1). X:[Cin, L], W:[Cout, Cin/groups, K], bias?:[Cout]
// -> Y:[Cout, Lout], with stride/pad/dilation/groups. One thread per (Cout, Lout).
// Covers regular (groups=1) and depthwise (groups=Cin) convs. act as in GEMM.
const CONV1D_WGSL = `
struct Meta { Cout:u32, Cin:u32, L:u32, Lout:u32, K:u32, stride:u32, pad:u32, dil:u32,
              groups:u32, hasBias:u32, act:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
fn gelu(x: f32) -> f32 {
  let t = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -20.0, 20.0);
  return 0.5 * x * (1.0 + tanh(t));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  if (idx >= m.Cout * m.Lout) { return; }
  let co = idx / m.Lout;
  let lo = idx % m.Lout;
  let cinPerG = m.Cin / m.groups;
  let coutPerG = m.Cout / m.groups;
  let g = co / coutPerG;
  var acc = 0.0;
  for (var ci = 0u; ci < cinPerG; ci++) {
    let realCi = g * cinPerG + ci;
    let wBase = (co * cinPerG + ci) * m.K;
    let xBase = realCi * m.L;
    for (var k = 0u; k < m.K; k++) {
      let li = i32(lo * m.stride + k * m.dil) - i32(m.pad);
      if (li >= 0 && li < i32(m.L)) {
        acc += X[xBase + u32(li)] * W[wBase + k];
      }
    }
  }
  if (m.hasBias == 1u) { acc += bias[co]; }
  if (m.act == 1u) { acc = gelu(acc); }
  else if (m.act == 2u) { acc = tanh(acc); }
  else if (m.act == 3u) { acc = max(acc, 0.0); }
  Y[co * m.Lout + lo] = acc;
}`;

// 1-D transposed convolution (batch 1) — the iSTFTNet upsampler and iSTFT
// overlap-add. X:[Cin,L], W:[Cin, Cout/groups, K], bias?:[Cout] -> Y:[Cout,Lout],
// Lout = (L-1)*stride - 2*pad + dilation*(K-1) + output_padding + 1. Gather form:
// one thread per (Cout, Lout), pulling the input positions that map onto it.
const CONVT1D_WGSL = `
struct Meta { Cout:u32, Cin:u32, L:u32, Lout:u32, K:u32, stride:u32, pad:u32, dil:u32,
              groups:u32, hasBias:u32, act:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  if (idx >= m.Cout * m.Lout) { return; }
  let co = idx / m.Lout;
  let lo = idx % m.Lout;
  let cinPerG = m.Cin / m.groups;
  let coutPerG = m.Cout / m.groups;
  let g = co / coutPerG;
  let coInG = co - g * coutPerG;
  var acc = 0.0;
  for (var ci = 0u; ci < cinPerG; ci++) {
    let realCi = g * cinPerG + ci;
    let wBase = realCi * (coutPerG * m.K) + coInG * m.K;
    let xBase = realCi * m.L;
    for (var k = 0u; k < m.K; k++) {
      let num = i32(lo + m.pad) - i32(k * m.dil);
      if (num >= 0 && (num % i32(m.stride)) == 0) {
        let li = num / i32(m.stride);
        if (li >= 0 && li < i32(m.L)) {
          acc += X[xBase + u32(li)] * W[wBase + k];
        }
      }
    }
  }
  if (m.hasBias == 1u) { acc += bias[co]; }
  if (m.act == 1u) { acc = 0.5 * acc * (1.0 + tanh(clamp(0.7978845608028654 * (acc + 0.044715 * acc * acc * acc), -20.0, 20.0))); }
  else if (m.act == 2u) { acc = tanh(acc); }
  else if (m.act == 3u) { acc = max(acc, 0.0); }
  Y[co * m.Lout + lo] = acc;
}`;

// im2col for conv1d: X[Cin,L] -> Cols[Cin*K, Lout], so a conv becomes a single
// GEMM  W[Cout, Cin*K] @ Cols  — hitting tiled-GEMM throughput instead of the
// direct kernel's memory-bound rate. Row (ci*K+k), col lo.
const IM2COL_WGSL = `
struct Meta { Cin:u32, L:u32, Lout:u32, K:u32, stride:u32, pad:u32, dil:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Cols: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.y * (nwg.x * 64u) + g.x;
  let rows = m.Cin * m.K;
  if (i >= rows * m.Lout) { return; }
  let row = i / m.Lout; let lo = i % m.Lout;
  let ci = row / m.K; let k = row % m.K;
  let li = i32(lo * m.stride + k * m.dil) - i32(m.pad);
  Cols[i] = select(0.0, X[ci * m.L + u32(li)], li >= 0 && li < i32(m.L));
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
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.y * (nwg.x * 64u) + g.x;
  if (i >= m.n) { return; }
  let bIdx = select(i, i % m.cols, m.bRows == 1u);
  let a = A[i]; let b = B[bIdx];
  C[i] = select(a + b, a * b, m.op == 1u);
}`;

// Helpers for multi-head attention: 2-D transpose, and column slice / scatter
// (to split [seq, H*d] into per-head [seq, d] and reassemble).
const TRANSPOSE_WGSL = `
struct Meta { rows:u32, cols:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.y * (nwg.x * 64u) + g.x; if (i >= m.rows * m.cols) { return; }
  let r = i / m.cols; let c = i % m.cols;
  Y[c * m.rows + r] = X[i];
}`;

const SLICECOLS_WGSL = `
struct Meta { rows:u32, C:u32, col0:u32, W:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.y * (nwg.x * 64u) + g.x; if (i >= m.rows * m.W) { return; }
  let r = i / m.W; let j = i % m.W;
  Y[i] = X[r * m.C + m.col0 + j];
}`;

const SETCOLS_WGSL = `
struct Meta { rows:u32, C:u32, col0:u32, W:u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.y * (nwg.x * 64u) + g.x; if (i >= m.rows * m.W) { return; }
  let r = i / m.W; let j = i % m.W;
  dst[r * m.C + m.col0 + j] = src[i];
}`;

// Bidirectional LSTM matching the ONNX LSTM op (batch 1). One workgroup per
// direction, one thread per hidden unit, timesteps looped in-kernel (h/c kept in
// workgroup memory). ONNX gate order is **iofc**; no peephole. Weights:
//   W:[2, 4H, I]  R:[2, 4H, H]  B:[2, 8H] (Wb[4H] then Rb[4H]).
// Output Y:[seq, 2H] = [fwd(H) | bwd(H)] per timestep (ONNX [seq,2,1,H] flattened).
// Requires H <= 256.
const LSTM_WGSL = `
struct Meta { seq:u32, inp:u32, hid:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> R: array<f32>;
@group(0) @binding(3) var<storage, read> Bnd: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> m: Meta;
var<workgroup> hsh: array<f32, 256>;
var<workgroup> csh: array<f32, 256>;
var<workgroup> htmp: array<f32, 256>;
fn sig(x: f32) -> f32 { return 1.0 / (1.0 + exp(-clamp(x, -30.0, 30.0))); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let dir = wg.x;
  let H = m.hid; let I = m.inp;
  let wBase = dir * 4u * H * I;
  let rBase = dir * 4u * H * H;
  let bBase = dir * 8u * H;
  for (var u = tid; u < H; u += 256u) { hsh[u] = 0.0; csh[u] = 0.0; }
  workgroupBarrier();
  for (var s = 0u; s < m.seq; s++) {
    let t = select(s, m.seq - 1u - s, dir == 1u);
    let xBase = t * I;
    for (var u = tid; u < H; u += 256u) {
      var gi = Bnd[bBase + 0u*H + u] + Bnd[bBase + 4u*H + 0u*H + u];
      var go = Bnd[bBase + 1u*H + u] + Bnd[bBase + 4u*H + 1u*H + u];
      var gf = Bnd[bBase + 2u*H + u] + Bnd[bBase + 4u*H + 2u*H + u];
      var gc = Bnd[bBase + 3u*H + u] + Bnd[bBase + 4u*H + 3u*H + u];
      for (var k = 0u; k < I; k++) {
        let xv = X[xBase + k];
        gi += W[wBase + (0u*H + u)*I + k] * xv;
        go += W[wBase + (1u*H + u)*I + k] * xv;
        gf += W[wBase + (2u*H + u)*I + k] * xv;
        gc += W[wBase + (3u*H + u)*I + k] * xv;
      }
      for (var k = 0u; k < H; k++) {
        let hv = hsh[k];
        gi += R[rBase + (0u*H + u)*H + k] * hv;
        go += R[rBase + (1u*H + u)*H + k] * hv;
        gf += R[rBase + (2u*H + u)*H + k] * hv;
        gc += R[rBase + (3u*H + u)*H + k] * hv;
      }
      let cnew = sig(gf) * csh[u] + sig(gi) * tanh(gc);
      csh[u] = cnew;
      let ht = sig(go) * tanh(cnew);
      htmp[u] = ht;
      Y[(t * 2u + dir) * H + u] = ht;
    }
    workgroupBarrier();
    for (var u = tid; u < H; u += 256u) { hsh[u] = htmp[u]; }
    workgroupBarrier();
  }
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

  /**
   * 1-D conv. x:[Cin,L] (rows=Cin, cols=L), w GPU tensor of Cout*(Cin/groups)*K
   * f32, bias?:[1,Cout]. Returns [Cout, Lout]. w/bias are passed as GpuTensors
   * (any rows/cols — only .buf is used).
   */
  conv1d(x, w, { cout, k, bias = null, stride = 1, pad = 0, dilation = 1, groups = 1, act = "none" } = {}) {
    const Cin = x.rows, L = x.cols;
    const Lout = Math.floor((L + 2 * pad - dilation * (k - 1) - 1) / stride) + 1;
    const y = this.alloc(cout, Lout);
    const biasBuf = bias ? bias.buf : this.alloc(1, 1).buf;
    const pipeline = this._pipeline("conv1d", CONV1D_WGSL);
    const u = this._uniform(new Uint32Array([cout, Cin, L, Lout, k, stride, pad, dilation, groups, bias ? 1 : 0, ACT[act], 0]));
    this._run(pipeline, [x.buf, w.buf, biasBuf, y.buf], u, Math.ceil((cout * Lout) / 64));
    return y;
  }

  /**
   * 1-D transposed conv. x:[Cin,L], w = Cin*(Cout/groups)*K f32, bias?:[1,Cout].
   * Returns [Cout, Lout]. w/bias passed as GpuTensors (only .buf used).
   */
  convTranspose1d(x, w, { cout, k, bias = null, stride = 1, pad = 0, dilation = 1, groups = 1, outputPadding = 0, act = "none" } = {}) {
    const Cin = x.rows, L = x.cols;
    const Lout = (L - 1) * stride - 2 * pad + dilation * (k - 1) + outputPadding + 1;
    const y = this.alloc(cout, Lout);
    const biasBuf = bias ? bias.buf : this.alloc(1, 1).buf;
    const pipeline = this._pipeline("convt1d", CONVT1D_WGSL);
    const u = this._uniform(new Uint32Array([cout, Cin, L, Lout, k, stride, pad, dilation, groups, bias ? 1 : 0, ACT[act], 0]));
    this._run(pipeline, [x.buf, w.buf, biasBuf, y.buf], u, Math.ceil((cout * Lout) / 64));
    return y;
  }

  /** im2col: x[Cin,L] -> [Cin*K, Lout]. */
  im2col(x, k, { stride = 1, pad = 0, dilation = 1 } = {}) {
    const Cin = x.rows, L = x.cols;
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

  /**
   * Bidirectional LSTM (ONNX semantics, iofc gates, batch 1). x:[seq,inp];
   * w/r/b are GPU tensors holding W[2,4H,inp], R[2,4H,H], B[2,8H] flat.
   * Returns Y:[seq, 2*hid] = [fwd | bwd]. H must be <= 256.
   */
  lstm(x, w, r, b, hid) {
    const seq = x.rows, inp = x.cols;
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
