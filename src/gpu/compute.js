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

// C = act(A[MxK] @ B[KxN] + bias[N]).  Register-blocked: each 256-thread workgroup
// computes a 64×64 output block; each thread a 4×4 micro-tile from registers, with
// 64×16 / 16×64 shared-memory staging. ~5× the naive tiled kernel by amortizing
// shared-memory reads over 16 MACs each. bias per-N, act: 0 none/1 gelu/2 tanh/3 relu.
const GEMM_WGSL = `
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM = 64u; const BN = 64u; const BK = 16u; const TM = 4u; const TN = 4u;
var<workgroup> As: array<f32, 1024>; // BM*BK
var<workgroup> Bs: array<f32, 1024>; // BK*BN
fn gelu(x: f32) -> f32 {
  let t = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -20.0, 20.0);
  return 0.5 * x * (1.0 + tanh(t));
}
fn actf(x: f32) -> f32 {
  if (m.act == 1u) { return gelu(x); }
  if (m.act == 2u) { return tanh(x); }
  if (m.act == 3u) { return max(x, 0.0); }
  if (m.act == 4u) { return x / (1.0 + exp(-clamp(x, -30.0, 30.0))); }
  return x;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let blockRow = wg.y * BM;
  let blockCol = wg.x * BN;
  let threadRow = (tid / (BN / TN)) * TM; // 0..60 step 4
  let threadCol = (tid % (BN / TN)) * TN;
  var acc: array<f32, 16>; // TM*TN
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  let nT = (m.K + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    // cooperative load: 1024 elems each / 256 threads = 4 per thread
    for (var i = 0u; i < 4u; i++) {
      let idxA = tid + i * 256u;
      let aRow = idxA / BK; let aCol = idxA % BK;
      As[idxA] = select(0.0, A[(blockRow + aRow) * m.K + kk + aCol], blockRow + aRow < m.M && kk + aCol < m.K);
      let idxB = tid + i * 256u;
      let bRow = idxB / BN; let bCol = idxB % BN;
      Bs[idxB] = select(0.0, B[(kk + bRow) * m.N + blockCol + bCol], kk + bRow < m.K && blockCol + bCol < m.N);
    }
    workgroupBarrier();
    for (var k = 0u; k < BK; k++) {
      var aReg: array<f32, 4>;
      var bReg: array<f32, 4>;
      for (var i = 0u; i < TM; i++) { aReg[i] = As[(threadRow + i) * BK + k]; }
      for (var j = 0u; j < TN; j++) { bReg[j] = Bs[k * BN + threadCol + j]; }
      for (var i = 0u; i < TM; i++) {
        for (var j = 0u; j < TN; j++) { acc[i * TN + j] += aReg[i] * bReg[j]; }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < TM; i++) {
    for (var j = 0u; j < TN; j++) {
      let r = blockRow + threadRow + i;
      let c = blockCol + threadCol + j;
      if (r < m.M && c < m.N) {
        var v = acc[i * TN + j];
        if (m.hasBias == 1u) { v += bias[c]; }
        C[r * m.N + c] = actf(v);
      }
    }
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
  else if (m.act == 4u) { acc = acc / (1.0 + exp(-clamp(acc, -30.0, 30.0))); }
  Y[co * m.Lout + lo] = acc;
}`;

// 2-D convolution (batch 1) — FastConformer dw-striding subsampling. X:[Cin,H*W]
// (rows=Cin), W:[Cout,Cin/groups,Kh,Kw] flat, bias?:[Cout] -> Y:[Cout,Ho*Wo]. One
// thread per (Cout, Ho*Wo). Supports groups (depthwise) + fused bias/act.
const CONV2D_WGSL = `
struct Meta { Cout:u32, Cin:u32, H:u32, W:u32, Ho:u32, Wo:u32, Kh:u32, Kw:u32,
              sH:u32, sW:u32, padH:u32, padW:u32, groups:u32, hasBias:u32, act:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  let HW = m.Ho * m.Wo;
  if (idx >= m.Cout * HW) { return; }
  let co = idx / HW;
  let ho = (idx % HW) / m.Wo;
  let wo = (idx % HW) % m.Wo;
  let cinPerG = m.Cin / m.groups;
  let coutPerG = m.Cout / m.groups;
  let g = co / coutPerG;
  var acc = 0.0;
  for (var ci = 0u; ci < cinPerG; ci++) {
    let realCi = g * cinPerG + ci;
    let xC = realCi * m.H * m.W;
    let wC = ((co * cinPerG + ci) * m.Kh) * m.Kw;
    for (var kh = 0u; kh < m.Kh; kh++) {
      let hi = i32(ho * m.sH + kh) - i32(m.padH);
      if (hi < 0 || hi >= i32(m.H)) { continue; }
      for (var kw = 0u; kw < m.Kw; kw++) {
        let wi = i32(wo * m.sW + kw) - i32(m.padW);
        if (wi >= 0 && wi < i32(m.W)) {
          acc += X[xC + u32(hi) * m.W + u32(wi)] * Wt[wC + kh * m.Kw + kw];
        }
      }
    }
  }
  if (m.hasBias == 1u) { acc += bias[co]; }
  if (m.act == 3u) { acc = max(acc, 0.0); }
  else if (m.act == 4u) { acc = acc / (1.0 + exp(-clamp(acc, -30.0, 30.0))); }
  Y[co * HW + ho * m.Wo + wo] = acc;
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
  else if (m.act == 4u) { acc = acc / (1.0 + exp(-clamp(acc, -30.0, 30.0))); }
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

// Fused conv1d as an IMPLICIT GEMM (groups=1): C[Cout,Lout] = W[Cout,Cin*K] @
// cols[Cin*K,Lout], but the cols matrix is never materialized — the B tile reads
// X directly via the conv index map. Same register-blocking as the GEMM (64×64
// block, 4×4 micro-tile), so it hits GEMM throughput WITHOUT im2col's memory
// blow-up (which is what capped the vocoder convs). bias per-Cout (per output row).
const CONV1D_IMPLICIT_WGSL = `
struct Meta { Cout:u32, Lout:u32, CinK:u32, Cin:u32, L:u32, K:u32, stride:u32, pad:u32,
              dil:u32, hasBias:u32, act:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM = 64u; const BN = 64u; const BK = 16u; const TM = 4u; const TN = 4u;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
fn gelu(x: f32) -> f32 {
  let t = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -20.0, 20.0);
  return 0.5 * x * (1.0 + tanh(t));
}
fn actf(x: f32, a: u32) -> f32 {
  if (a == 1u) { return gelu(x); }
  if (a == 2u) { return tanh(x); }
  if (a == 3u) { return max(x, 0.0); }
  return x;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let blockRow = wg.y * BM; // over Cout
  let blockCol = wg.x * BN; // over Lout
  let threadRow = (tid / (BN / TN)) * TM;
  let threadCol = (tid % (BN / TN)) * TN;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  let nT = (m.CinK + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    for (var i = 0u; i < 4u; i++) {
      let idxA = tid + i * 256u;
      let aRow = idxA / BK; let aCol = idxA % BK; // Cout row, contraction col
      As[idxA] = select(0.0, W[(blockRow + aRow) * m.CinK + kk + aCol], blockRow + aRow < m.Cout && kk + aCol < m.CinK);
      let idxB = tid + i * 256u;
      let cr = kk + idxB / BN;          // contraction index = ci*K + kpos
      let lo = blockCol + idxB % BN;    // output position
      var bv = 0.0;
      if (cr < m.CinK && lo < m.Lout) {
        let ci = cr / m.K; let kpos = cr % m.K;
        let li = i32(lo * m.stride + kpos * m.dil) - i32(m.pad);
        if (li >= 0 && li < i32(m.L)) { bv = X[ci * m.L + u32(li)]; }
      }
      Bs[idxB] = bv;
    }
    workgroupBarrier();
    for (var k = 0u; k < BK; k++) {
      var aReg: array<f32, 4>;
      var bReg: array<f32, 4>;
      for (var i = 0u; i < TM; i++) { aReg[i] = As[(threadRow + i) * BK + k]; }
      for (var j = 0u; j < TN; j++) { bReg[j] = Bs[k * BN + threadCol + j]; }
      for (var i = 0u; i < TM; i++) {
        for (var j = 0u; j < TN; j++) { acc[i * TN + j] += aReg[i] * bReg[j]; }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < TM; i++) {
    for (var j = 0u; j < TN; j++) {
      let r = blockRow + threadRow + i;
      let c = blockCol + threadCol + j;
      if (r < m.Cout && c < m.Lout) {
        var v = acc[i * TN + j];
        if (m.hasBias == 1u) { v += bias[r]; }
        Y[r * m.Lout + c] = actf(v, m.act);
      }
    }
  }
}`;

// AdaIN: instance-norm over time (per channel) + style-predicted per-channel affine.
// x:[C,L], scale/shift:[C] -> y = (x-mean_c)/sqrt(var_c+eps)*scale[c] + shift[c].
// One workgroup per channel; 64-lane reduce over L.
const ADAIN_WGSL = `
struct Meta { C:u32, L:u32, eps:f32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> shift: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let ch = wg.x;
  if (ch >= m.C) { return; }
  let base = ch * m.L;
  var sum = 0.0;
  for (var j = li; j < m.L; j += 64u) { sum += X[base + j]; }
  red[li] = sum; workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let mean = red[0] / f32(m.L);
  workgroupBarrier();
  var vs = 0.0;
  for (var j = li; j < m.L; j += 64u) { let d = X[base + j] - mean; vs += d * d; }
  red[li] = vs; workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(m.L) + m.eps);
  let sc = scale[ch]; let sh = shift[ch];
  for (var j = li; j < m.L; j += 64u) { Y[base + j] = (X[base + j] - mean) * inv * sc + sh; }
}`;

// Column gather (length regulator): Y[:, f] = X[:, idx[f]]. Expands text features
// to mel frames by repeating each column per its predicted duration (idx = the
// frame→text-token map, a duration cumsum built on the CPU).
const GATHERCOLS_WGSL = `
struct Meta { rows:u32, inCols:u32, outCols:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.y * (nwg.x * 64u) + g.x;
  if (i >= m.rows * m.outCols) { return; }
  let r = i / m.outCols; let f = i % m.outCols;
  Y[i] = X[r * m.inCols + idx[f]];
}`;

// LeakyReLU (elementwise): y = x>0 ? x : slope*x. Slope in the uniform.
const LEAKY_WGSL = `
struct Meta { n:u32, slope:f32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.y * (nwg.x * 64u) + g.x;
  if (i >= m.n) { return; }
  let v = X[i];
  Y[i] = select(m.slope * v, v, v > 0.0);
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

// ── f16-storage variants ─────────────────────────────────────────────────────
// Same register-blocking as the f32 kernels but with f16 GLOBAL buffers (half the
// memory traffic + Apple's 2× f16 ALU): f16 in/out, f16 multiply, f32 accumulate.
// Measured ~1.3–1.5× over f32, parity vs f32 ≈ rel 3e-4 (fine for TTS).
const GEMM_F16_WGSL = `enable f16;
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> C: array<f16>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM=64u; const BN=64u; const BK=16u; const TM=4u; const TN=4u;
var<workgroup> As: array<f16,1024>; var<workgroup> Bs: array<f16,1024>;
fn gelu(x: f32) -> f32 {
  let t = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -20.0, 20.0);
  return 0.5 * x * (1.0 + tanh(t));
}
fn actf(x: f32, a: u32) -> f32 {
  if (a==1u){return gelu(x);} if (a==2u){return tanh(x);} if (a==3u){return max(x,0.0);} return x;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_index) tid:u32){
  let br=wg.y*BM; let bc=wg.x*BN;
  let tr=(tid/(BN/TN))*TM; let tc=(tid%(BN/TN))*TN;
  var acc: array<f32,16>; for(var i=0u;i<16u;i++){acc[i]=0.0;}
  let nT=(m.K+BK-1u)/BK;
  for(var t=0u;t<nT;t++){ let kk=t*BK;
    for(var i=0u;i<4u;i++){ let ia=tid+i*256u; let ar=ia/BK; let ac=ia%BK;
      As[ia]=select(f16(0.0),A[(br+ar)*m.K+kk+ac],br+ar<m.M&&kk+ac<m.K);
      let bR=ia/BN; let bc2=ia%BN;
      Bs[ia]=select(f16(0.0),B[(kk+bR)*m.N+bc+bc2],kk+bR<m.K&&bc+bc2<m.N);}
    workgroupBarrier();
    for(var k=0u;k<BK;k++){ var a:array<f16,4>; var b:array<f16,4>;
      for(var i=0u;i<TM;i++){a[i]=As[(tr+i)*BK+k];}
      for(var j=0u;j<TN;j++){b[j]=Bs[k*BN+tc+j];}
      for(var i=0u;i<TM;i++){for(var j=0u;j<TN;j++){acc[i*TN+j]+=f32(a[i]*b[j]);}}}
    workgroupBarrier();}
  for(var i=0u;i<TM;i++){for(var j=0u;j<TN;j++){ let r=br+tr+i; let c=bc+tc+j;
    if(r<m.M&&c<m.N){ var v=acc[i*TN+j]; if(m.hasBias==1u){v+=f32(bias[c]);} C[r*m.N+c]=f16(actf(v,m.act)); }}}}`;

// f16 fused conv1d (implicit GEMM, groups=1). W/X/bias/Y all f16.
const CONV1D_IMPLICIT_F16_WGSL = `enable f16;
struct Meta { Cout:u32, Lout:u32, CinK:u32, Cin:u32, L:u32, K:u32, stride:u32, pad:u32,
              dil:u32, hasBias:u32, act:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> W: array<f16>;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f16>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM=64u; const BN=64u; const BK=16u; const TM=4u; const TN=4u;
var<workgroup> As: array<f16,1024>; var<workgroup> Bs: array<f16,1024>;
fn gelu(x: f32) -> f32 {
  let t = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -20.0, 20.0);
  return 0.5 * x * (1.0 + tanh(t));
}
fn actf(x: f32, a: u32) -> f32 {
  if (a==1u){return gelu(x);} if (a==2u){return tanh(x);} if (a==3u){return max(x,0.0);} return x;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_index) tid:u32){
  let br=wg.y*BM; let bc=wg.x*BN;
  let tr=(tid/(BN/TN))*TM; let tc=(tid%(BN/TN))*TN;
  var acc: array<f32,16>; for(var i=0u;i<16u;i++){acc[i]=0.0;}
  let nT=(m.CinK+BK-1u)/BK;
  for(var t=0u;t<nT;t++){ let kk=t*BK;
    for(var i=0u;i<4u;i++){ let ia=tid+i*256u; let ar=ia/BK; let ac=ia%BK;
      As[ia]=select(f16(0.0),W[(br+ar)*m.CinK+kk+ac],br+ar<m.Cout&&kk+ac<m.CinK);
      let cr=kk+ia/BN; let lo=bc+ia%BN; var bv=f16(0.0);
      if(cr<m.CinK && lo<m.Lout){ let ci=cr/m.K; let kp=cr%m.K;
        let li=i32(lo*m.stride+kp*m.dil)-i32(m.pad);
        if(li>=0 && li<i32(m.L)){ bv=X[ci*m.L+u32(li)]; }}
      Bs[ia]=bv;}
    workgroupBarrier();
    for(var k=0u;k<BK;k++){ var a:array<f16,4>; var b:array<f16,4>;
      for(var i=0u;i<TM;i++){a[i]=As[(tr+i)*BK+k];}
      for(var j=0u;j<TN;j++){b[j]=Bs[k*BN+tc+j];}
      for(var i=0u;i<TM;i++){for(var j=0u;j<TN;j++){acc[i*TN+j]+=f32(a[i]*b[j]);}}}
    workgroupBarrier();}
  for(var i=0u;i<TM;i++){for(var j=0u;j<TN;j++){ let r=br+tr+i; let c=bc+tc+j;
    if(r<m.Cout&&c<m.Lout){ var v=acc[i*TN+j]; if(m.hasBias==1u){v+=f32(bias[r]);} Y[r*m.Lout+c]=f16(actf(v,m.act)); }}}}`;

// int4 block-quantized matmul (ONNX MatMulNBits: bits=4, block_size=32). This is
// the one thing ORT's WebGPU EP CAN'T do — it has no int kernels, so int4 models
// (Nemotron) fall back to WASM. Here we read the packed int4 weights + per-block
// scales + int4 zero-points directly and dequantize in-shader: a *capability*
// unlock (runs on the GPU where ORT can't), not a speed play. Y = A @ dequant(B)ᵀ,
// dequant(n,k) = (q(n,k) - zp(n,block)) * scale(n,block), block = k/32.
// Bq: packed uint8 [N, nblk, 16] (2 int4/byte) as u32; zp: packed int4 [N, zpb] as u32.
const MATMUL_NBITS_WGSL = `
struct Meta { M:u32, N:u32, K:u32, nblk:u32, zpb:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage,read> A: array<f32>;
@group(0) @binding(1) var<storage,read> Bq: array<u32>;
@group(0) @binding(2) var<storage,read> scales: array<f32>;
@group(0) @binding(3) var<storage,read> zp: array<u32>;
@group(0) @binding(4) var<storage,read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>, @builtin(num_workgroups) nwg:vec3<u32>){
  let idx = gid.y*(nwg.x*64u)+gid.x; if(idx>=m.M*m.N){return;}
  let mrow=idx/m.N; let n=idx%m.N;
  var acc=0.0;
  for(var b=0u;b<m.nblk;b++){
    let zi=n*m.zpb+(b>>1u); let wz=zp[zi>>2u]; let bz=(wz>>(8u*(zi&3u)))&0xFFu;
    let zpv=f32((bz>>(4u*(b&1u)))&0xFu);
    let s=scales[n*m.nblk+b];
    for(var jj=0u;jj<32u;jj++){
      let k=b*32u+jj;
      if(k>=m.K){break;} // last block is partial when K % 32 != 0
      let bi=(n*m.nblk+b)*16u+(jj>>1u);
      let wq=Bq[bi>>2u]; let bq=(wq>>(8u*(bi&3u)))&0xFFu; let q=f32((bq>>(4u*(jj&1u)))&0xFu);
      acc+=A[mrow*m.K+k]*((q-zpv)*s);
    }
  }
  Y[mrow*m.N+n]=acc;
}`;

const ACT = { none: 0, gelu: 1, tanh: 2, relu: 3, silu: 4 };

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

  // ── f16 storage ──────────────────────────────────────────────────────────
  /** Upload f32 data as an f16 tensor (half the bytes). */
  uploadF16(data, rows, cols) {
    const u16 = new Uint16Array(new Float16Array(data).buffer);
    const size = Math.max(4, Math.ceil((u16.byteLength) / 4) * 4);
    const buf = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.device.queue.writeBuffer(buf, 0, u16);
    return { buf, rows, cols, f16: true };
  }
  /** Allocate an uninitialized f16 tensor. */
  allocF16(rows, cols) {
    const size = Math.max(4, Math.ceil((rows * cols * 2) / 4) * 4);
    const buf = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    return { buf, rows, cols, f16: true };
  }
  /** Copy an f16 tensor back to CPU as Float32Array. */
  async downloadF16(t) {
    const n = t.rows * t.cols;
    const size = Math.ceil((n * 2) / 4) * 4;
    const stg = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(t.buf, 0, stg, 0, size);
    this.device.queue.submit([enc.finish()]);
    await stg.mapAsync(GPUMapMode.READ);
    const h = new Float16Array(stg.getMappedRange().slice(0, n * 2));
    const out = Float32Array.from(h);
    stg.unmap(); stg.destroy();
    return out;
  }
  /** f16 matmul: C = act(A@B + bias). a/b/bias/out all f16 tensors. */
  matmulF16(a, b, { bias = null, act = "none" } = {}) {
    const M = a.rows, K = a.cols, N = b.cols;
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
    const M = a.rows, K = a.cols;
    const nblk = Math.ceil(K / blockSize);
    const zpb = Math.ceil(nblk / 2);
    const y = this.alloc(M, N);
    const pipeline = this._pipeline("matmulNBits", MATMUL_NBITS_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, nblk, zpb, 0, 0, 0]));
    const bg = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [a.buf, bq.buf, scales.buf, zp.buf, y.buf, u].map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    const enc = this.device.createCommandEncoder();
    const pass = this._pass || enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    const tot = Math.ceil((M * N) / 64);
    pass.dispatchWorkgroups(Math.min(tot, 65535), Math.ceil(tot / 65535));
    if (!this._pass) { pass.end(); this.device.queue.submit([enc.finish()]); }
    return y;
  }

  /** f16 fused conv1d (implicit GEMM, groups=1). x/wRows/bias/out all f16. */
  conv1dFastF16(x, wRows, cout, k, { bias = null, stride = 1, pad = 0, dilation = 1, act = "none" } = {}) {
    const Cin = x.rows, L = x.cols, CinK = Cin * k;
    const Lout = Math.floor((L + 2 * pad - dilation * (k - 1) - 1) / stride) + 1;
    const y = this.allocF16(cout, Lout);
    const biasBuf = bias ? bias.buf : this.allocF16(1, 1).buf;
    const pipeline = this._pipeline("conv1dImplicitF16", CONV1D_IMPLICIT_F16_WGSL);
    const u = this._uniform(new Uint32Array([cout, Lout, CinK, Cin, L, k, stride, pad, dilation, bias ? 1 : 0, ACT[act], 0]));
    this._run(pipeline, [wRows.buf, x.buf, biasBuf, y.buf], u, Math.ceil(Lout / 64), Math.ceil(cout / 64));
    return y;
  }

  _uniform(arr) {
    const buf = this.device.createBuffer({ size: arr.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, arr);
    return buf;
  }

  /** Batch mode: queue many kernels into one submit. beginBatch()…endBatch(). */
  beginBatch() {
    this._enc = this.device.createCommandEncoder();
    this._pass = this._enc.beginComputePass();
  }
  endBatch() {
    this._pass.end();
    this.device.queue.submit([this._enc.finish()]);
    this._enc = this._pass = null;
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
    if (this._pass) { // batched
      this._pass.setPipeline(pipeline);
      this._pass.setBindGroup(0, bg);
      this._pass.dispatchWorkgroups(groupsX, groupsY);
      return;
    }
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
    this._run(pipeline, [a.buf, b.buf, biasBuf, c.buf], u, Math.ceil(N / 64), Math.ceil(M / 64));
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
   * 2-D conv (batch 1). x:[Cin, H*W] (rows=Cin), w GPU tensor holding
   * Cout*(Cin/groups)*Kh*Kw f32 (ONNX [Cout,Cin/g,Kh,Kw] flat), bias?:[Cout].
   * Returns [Cout, Ho*Wo]. Supports groups (depthwise) + fused bias/relu/silu.
   */
  conv2d(x, w, { cout, cin, h, w: W_, kh, kw, bias = null, strideH = 1, strideW = 1, padH = 0, padW = 0, groups = 1, act = "none" } = {}) {
    const Ho = Math.floor((h + 2 * padH - kh) / strideH) + 1;
    const Wo = Math.floor((W_ + 2 * padW - kw) / strideW) + 1;
    const y = this.alloc(cout, Ho * Wo);
    const biasBuf = bias ? bias.buf : this.alloc(1, 1).buf;
    const pipeline = this._pipeline("conv2d", CONV2D_WGSL);
    const u = this._uniform(new Uint32Array([cout, cin, h, W_, Ho, Wo, kh, kw, strideH, strideW, padH, padW, groups, bias ? 1 : 0, ACT[act], 0]));
    this._run(pipeline, [x.buf, w.buf, biasBuf, y.buf], u, Math.ceil((cout * Ho * Wo) / 64));
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

  /** AdaIN: instance-norm x[C,L] over time + per-channel affine from style. scale/shift:[C]. */
  adain(x, scale, shift, eps = 1e-5) {
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline("adain", ADAIN_WGSL);
    const meta = new ArrayBuffer(16);
    new Uint32Array(meta, 0, 2).set([x.rows, x.cols]);
    new Float32Array(meta, 8, 1)[0] = eps;
    const u = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(u, 0, meta);
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
    const u = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(u, 0, meta);
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil(n / 64));
    return y;
  }

  /**
   * Fused conv1d via implicit GEMM (groups=1) — register-blocked, no im2col
   * materialization. x:[Cin,L], wRows = weight as [Cout, Cin*K], bias?:[1,Cout].
   * Returns [Cout, Lout]. The fast path for the big vocaoder convs.
   */
  conv1dFast(x, wRows, cout, k, { bias = null, stride = 1, pad = 0, dilation = 1, act = "none" } = {}) {
    const Cin = x.rows, L = x.cols;
    const CinK = Cin * k;
    const Lout = Math.floor((L + 2 * pad - dilation * (k - 1) - 1) / stride) + 1;
    const y = this.alloc(cout, Lout);
    const biasBuf = bias ? bias.buf : this.alloc(1, 1).buf;
    const pipeline = this._pipeline("conv1dImplicit", CONV1D_IMPLICIT_WGSL);
    const u = this._uniform(new Uint32Array([cout, Lout, CinK, Cin, L, k, stride, pad, dilation, bias ? 1 : 0, ACT[act], 0]));
    this._run(pipeline, [wRows.buf, x.buf, biasBuf, y.buf], u, Math.ceil(Lout / 64), Math.ceil(cout / 64));
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
