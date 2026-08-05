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
fn erf_a(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * abs(x));
  let y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*exp(-x*x);
  return select(-y, y, x >= 0.0);
}
fn gelu_erf(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_a(x * 0.70710678118654752)); }

fn actf(x: f32) -> f32 {
  if (m.act == 1u) { return gelu(x); }
  if (m.act == 2u) { return tanh(x); }
  if (m.act == 3u) { return max(x, 0.0); }
  if (m.act == 4u) { return x / (1.0 + exp(-clamp(x, -30.0, 30.0))); }
  if (m.act == 5u) { return gelu_erf(x); }
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

// Vectorized GEMM (siboehm kernel 6): same 64×64 block / 4×4 micro-tile, but As is
// staged TRANSPOSED ([BK][BM]) so each thread's inner-loop A read is 4 contiguous
// floats (bank-conflict-free, vec4-loadable) and the 4×4 MAC is 4 vec4 FMAs. Cuts
// shared-memory instruction count ~4× vs the scalar kernel. Same bias/act semantics.
const GEMM_V2_WGSL = `
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM = 64u; const BN = 64u; const BK = 16u; const TM = 4u; const TN = 4u;
var<workgroup> As: array<f32, 1024>; // TRANSPOSED [BK][BM]
var<workgroup> Bs: array<f32, 1024>; // [BK][BN]
fn gelu(x: f32) -> f32 {
  let t = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -20.0, 20.0);
  return 0.5 * x * (1.0 + tanh(t));
}
fn erf_a(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * abs(x));
  let y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*exp(-x*x);
  return select(-y, y, x >= 0.0);
}
fn gelu_erf(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_a(x * 0.70710678118654752)); }
fn actf(x: f32) -> f32 {
  if (m.act == 1u) { return gelu(x); }
  if (m.act == 2u) { return tanh(x); }
  if (m.act == 3u) { return max(x, 0.0); }
  if (m.act == 4u) { return x / (1.0 + exp(-clamp(x, -30.0, 30.0))); }
  if (m.act == 5u) { return gelu_erf(x); }
  return x;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let blockRow = wg.y * BM;
  let blockCol = wg.x * BN;
  let threadRow = (tid / (BN / TN)) * TM;
  let threadCol = (tid % (BN / TN)) * TN;
  var acc0 = vec4<f32>(0.0); var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0); var acc3 = vec4<f32>(0.0);
  let nT = (m.K + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    for (var i = 0u; i < 4u; i++) {
      let p = tid + i * 256u;
      let aRow = p / BK; let aCol = p % BK;               // source [BM][BK]
      As[aCol * BM + aRow] = select(0.0, A[(blockRow + aRow) * m.K + kk + aCol], blockRow + aRow < m.M && kk + aCol < m.K);
      let bRow = p / BN; let bCol = p % BN;               // [BK][BN]
      Bs[p] = select(0.0, B[(kk + bRow) * m.N + blockCol + bCol], kk + bRow < m.K && blockCol + bCol < m.N);
    }
    workgroupBarrier();
    for (var k = 0u; k < BK; k++) {
      let ab = k * BM + threadRow;
      let aReg = vec4<f32>(As[ab], As[ab + 1u], As[ab + 2u], As[ab + 3u]);
      let bb = k * BN + threadCol;
      let bReg = vec4<f32>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
      acc0 += aReg.x * bReg; acc1 += aReg.y * bReg; acc2 += aReg.z * bReg; acc3 += aReg.w * bReg;
    }
    workgroupBarrier();
  }
  let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
  for (var i = 0u; i < TM; i++) {
    let r = blockRow + threadRow + i;
    if (r >= m.M) { continue; }
    let v = accs[i];
    for (var j = 0u; j < TN; j++) {
      let c = blockCol + threadCol + j;
      if (c < m.N) {
        var x = v[j];
        if (m.hasBias == 1u) { x += bias[c]; }
        C[r * m.N + c] = actf(x);
      }
    }
  }
}`;

// GEMM v3: v2 + 128-bit GMEM loads (A/B bound as vec4<f32>). Tests whether reducing
// global load instructions 4× breaks the plateau. Requires K%4==0 and N%4==0 (bench).
const GEMM_V3_WGSL = `
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM = 64u; const BN = 64u; const BK = 16u; const TM = 4u; const TN = 4u;
var<workgroup> As: array<f32, 1024>; // TRANSPOSED [BK][BM]
var<workgroup> Bs: array<f32, 1024>; // [BK][BN]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let blockRow = wg.y * BM; let blockCol = wg.x * BN;
  let threadRow = (tid / (BN / TN)) * TM;
  let threadCol = (tid % (BN / TN)) * TN;
  let K4 = m.K / 4u; let N4 = m.N / 4u;
  // A load: 64 rows × 4 vec4-cols (BK/4) = 256 vec4 = 1/thread
  let aRow = tid / 4u; let aC = tid % 4u;
  // B load: 16 rows × 16 vec4-cols (BN/4) = 256 vec4 = 1/thread
  let bRow = tid / 16u; let bC = tid % 16u;
  var acc0 = vec4<f32>(0.0); var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0); var acc3 = vec4<f32>(0.0);
  let nT = (m.K + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    let av = A[(blockRow + aRow) * K4 + kk / 4u + aC];
    let aBase = 4u * aC;
    As[(aBase + 0u) * BM + aRow] = av.x; As[(aBase + 1u) * BM + aRow] = av.y;
    As[(aBase + 2u) * BM + aRow] = av.z; As[(aBase + 3u) * BM + aRow] = av.w;
    let bv = B[(kk + bRow) * N4 + blockCol / 4u + bC];
    let bBase = bRow * BN + 4u * bC;
    Bs[bBase] = bv.x; Bs[bBase + 1u] = bv.y; Bs[bBase + 2u] = bv.z; Bs[bBase + 3u] = bv.w;
    workgroupBarrier();
    for (var k = 0u; k < BK; k++) {
      let ab = k * BM + threadRow;
      let aReg = vec4<f32>(As[ab], As[ab + 1u], As[ab + 2u], As[ab + 3u]);
      let bb = k * BN + threadCol;
      let bReg = vec4<f32>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
      acc0 += aReg.x * bReg; acc1 += aReg.y * bReg; acc2 += aReg.z * bReg; acc3 += aReg.w * bReg;
    }
    workgroupBarrier();
  }
  let accs = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
  for (var i = 0u; i < TM; i++) {
    let r = blockRow + threadRow + i; if (r >= m.M) { continue; }
    let v = accs[i];
    for (var j = 0u; j < TN; j++) {
      let c = blockCol + threadCol + j;
      if (c < m.N) { var x = v[j]; if (m.hasBias == 1u) { x += bias[c]; } C[r * m.N + c] = x; }
    }
  }
}`;

// GEMM v4: 128×128 block, 8×8 micro-tile (16 vec4 accumulators/thread), vec4 GMEM.
// 4× the per-thread arithmetic intensity of v3 — fewer global loads per FLOP. 256
// threads, shared As[128×8]/Bs[8×128] (still 1024 each). Requires K%4==0, N%4==0.
const GEMM_V4_WGSL = `
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM = 128u; const BN = 128u; const BK = 8u; const TM = 8u; const TN = 8u;
var<workgroup> As: array<f32, 1024>; // TRANSPOSED [BK][BM]
var<workgroup> Bs: array<f32, 1024>; // [BK][BN]
fn gelu(x: f32) -> f32 {
  let t = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -20.0, 20.0);
  return 0.5 * x * (1.0 + tanh(t));
}
fn erf_a(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * abs(x));
  let y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*exp(-x*x);
  return select(-y, y, x >= 0.0);
}
fn gelu_erf(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_a(x * 0.70710678118654752)); }
fn actf(x: f32) -> f32 {
  if (m.act == 1u) { return gelu(x); }
  if (m.act == 2u) { return tanh(x); }
  if (m.act == 3u) { return max(x, 0.0); }
  if (m.act == 4u) { return x / (1.0 + exp(-clamp(x, -30.0, 30.0))); }
  if (m.act == 5u) { return gelu_erf(x); }
  return x;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let blockRow = wg.y * BM; let blockCol = wg.x * BN;
  let threadRow = (tid / (BN / TN)) * TM; // 16 threads/row-group → 0..120 step 8
  let threadCol = (tid % (BN / TN)) * TN;
  let K4 = m.K / 4u; let N4 = m.N / 4u;
  let aRow = tid / 2u; let aC = tid % 2u;    // A [128][8] = 256 vec4
  let bRow = tid / 32u; let bC = tid % 32u;  // B [8][128] = 256 vec4
  var acc: array<vec4<f32>, 16>; // TM×(TN/4)
  for (var i = 0u; i < 16u; i++) { acc[i] = vec4<f32>(0.0); }
  let nT = (m.K + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    let av = A[(blockRow + aRow) * K4 + kk / 4u + aC];
    let aBase = 4u * aC;
    As[(aBase + 0u) * BM + aRow] = av.x; As[(aBase + 1u) * BM + aRow] = av.y;
    As[(aBase + 2u) * BM + aRow] = av.z; As[(aBase + 3u) * BM + aRow] = av.w;
    let bv = B[(kk + bRow) * N4 + blockCol / 4u + bC];
    let bBase = bRow * BN + 4u * bC;
    Bs[bBase] = bv.x; Bs[bBase + 1u] = bv.y; Bs[bBase + 2u] = bv.z; Bs[bBase + 3u] = bv.w;
    workgroupBarrier();
    for (var k = 0u; k < BK; k++) {
      let ab = k * BM + threadRow;
      var aReg: array<f32, 8>;
      for (var i = 0u; i < 8u; i++) { aReg[i] = As[ab + i]; }
      let bb = k * BN + threadCol;
      let b0 = vec4<f32>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
      let b1 = vec4<f32>(Bs[bb + 4u], Bs[bb + 5u], Bs[bb + 6u], Bs[bb + 7u]);
      for (var i = 0u; i < 8u; i++) { acc[i * 2u] += aReg[i] * b0; acc[i * 2u + 1u] += aReg[i] * b1; }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < TM; i++) {
    let r = blockRow + threadRow + i; if (r >= m.M) { continue; }
    for (var jb = 0u; jb < 2u; jb++) {
      let v = acc[i * 2u + jb];
      for (var jj = 0u; jj < 4u; jj++) {
        let c = blockCol + threadCol + jb * 4u + jj;
        if (c < m.N) { var x = v[jj]; if (m.hasBias == 1u) { x += bias[c]; } C[r * m.N + c] = actf(x); }
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
fn erf_a(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * abs(x));
  let y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t*exp(-x*x);
  return select(-y, y, x >= 0.0);
}
fn gelu_erf(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_a(x * 0.70710678118654752)); }

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
  else if (m.act == 5u) { acc = gelu_erf(acc); }
  Y[co * m.Lout + lo] = acc;
}`;

// FastConformer subsampling reshape: conv output [C, Tsub*F] (rows=C) -> [Tsub, C*F]
// with out[ho, c*F+wo] = in[c, ho*F+wo]. Keeps it GPU-resident (was a download +
// host rearrange + upload per window).
const SUBRESHAPE_WGSL = `
struct Meta { C:u32, Tsub:u32, F:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  let CF = m.C * m.F;
  if (idx >= m.Tsub * CF) { return; }
  let ho = idx / CF;
  let rem = idx % CF;
  let c = rem / m.F;
  let wo = rem % m.F;
  Y[idx] = X[c * (m.Tsub * m.F) + ho * m.F + wo];
}`;

// TDT joint, split for speed: (1) jBatch builds j = relu(encProj[base+i] + predProj)
// for B frames [B,hid]; (2) the fast TILED matmul does [B,hid]@[hid,logits]; (3)
// argmaxRows reduces each row to token/dur argmax. Between emissions predProj is
// constant, so a run of frames is one tiled matmul instead of B tiny 1-workgroup ones.
const JBATCH_WGSL = `
struct Meta { base:u32, B:u32, hid:u32, tenc:u32 };
@group(0) @binding(0) var<storage, read> encProj: array<f32>;
@group(0) @binding(1) var<storage, read> predProj: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  if (idx >= m.B * m.hid) { return; }
  let i = idx / m.hid;
  let k = idx % m.hid;
  let v = encProj[(m.base + i) * m.hid + k] + predProj[k];
  out[idx] = max(0.0, v);
}`;
const ARGMAX_ROWS_WGSL = `
struct Meta { B:u32, vocab:u32, logits:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
var<workgroup> tIdx: array<u32, 256>;
var<workgroup> tVal: array<f32, 256>;
var<workgroup> dIdx: array<u32, 256>;
var<workgroup> dVal: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let L = lid.x;
  let rowBase = wid.x * m.logits;
  var lt = 0u; var lv = -1e30; var ld = 0u; var ldv = -1e30;
  for (var n = L; n < m.logits; n += 256u) {
    let s = X[rowBase + n];
    if (n < m.vocab) { if (s > lv) { lv = s; lt = n; } }
    else { if (s > ldv) { ldv = s; ld = n - m.vocab; } }
  }
  tIdx[L] = lt; tVal[L] = lv; dIdx[L] = ld; dVal[L] = ldv;
  workgroupBarrier();
  if (L == 0u) {
    var bt = 0u; var bv = -1e30; var bd = 0u; var bdv = -1e30;
    for (var i = 0u; i < 256u; i++) {
      if (tVal[i] > bv) { bv = tVal[i]; bt = tIdx[i]; }
      if (dVal[i] > bdv) { bdv = dVal[i]; bd = dIdx[i]; }
    }
    let r = wid.x * 4u;
    result[r] = f32(bt); result[r + 1u] = bv; result[r + 2u] = f32(bd); result[r + 3u] = bdv;
  }
}`;

// int8 GEMM with in-shader dequant: A[M,K] fp32 @ dequant(Wq)[K,N] -> Y[M,N].
// Wq = int8 weights packed 4-per-u32, row-major [k*N+n]; scale[N] per output column
// (symmetric: w = q * scale[n]). Reads weights at 1/4 the bandwidth of fp32 and keeps
// them 1/4 the GPU memory. Fused bias/act. out[m,n] = act(scale[n]*Σ a[m,k]*q[k,n] + b[n]).
const MATMUL_INT8_WGSL = `
struct Meta { M:u32, N:u32, K:u32, hasBias:u32, act:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> Wq: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  if (idx >= m.M * m.N) { return; }
  let row = idx / m.N;
  let col = idx % m.N;
  let aBase = row * m.K;
  var acc = 0.0;
  for (var k = 0u; k < m.K; k++) {
    let li = k * m.N + col;
    let u = Wq[li >> 2u];
    let sh = (li & 3u) * 8u;
    var q = i32((u >> sh) & 255u);
    if (q > 127) { q = q - 256; }
    acc += A[aBase + k] * f32(q);
  }
  acc = acc * scale[col];
  if (m.hasBias == 1u) { acc += bias[col]; }
  if (m.act == 1u) { acc = 0.5 * acc * (1.0 + tanh(clamp(0.7978845608028654 * (acc + 0.044715 * acc * acc * acc), -20.0, 20.0))); }
  else if (m.act == 3u) { acc = max(acc, 0.0); }
  else if (m.act == 4u) { acc = acc / (1.0 + exp(-clamp(acc, -30.0, 30.0))); }
  Y[idx] = acc;
}`;

// ── Batched multi-head attention kernels ────────────────────────────────────
// The per-head attention loop is ~80 dispatches/layer (slice/transpose/matmul/
// softmax per head) and the encoders are LAUNCH-BOUND. These batch all heads into
// single dispatches: head-strided reads over the [T, H*HD] projections, so no
// slicing/transposing/concat dispatches at all (~88 → ~11 dispatches/layer).

// scores[(w*H+h)*T+i, j] = Σ_d (Q[w*T+i, h*HD+d] + qb[h*HD+d]) · B[·, h*HD+d]
// (B = keys → AC term, or projected pos-emb → BD term; qb = pos_bias_u/v.)
// TILED: 64×64 output tile per z-block (w*H+h), register-blocked 4×4 like GEMM —
// the naive 1-thread-per-output version regressed the browser bench ~10%.
const BMM_QK_WGSL = `
struct Meta { T:u32, Tb:u32, H:u32, HD:u32, hasBias:u32, W:u32, bShared:u32, _c:u32 };
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> qb: array<f32>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM = 64u; const BN = 64u; const BK = 16u; const TM = 4u; const TN = 4u;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let b = wg.z;            // window-head block = w*H + h
  let h = b % m.H;
  let w = b / m.H;
  let stride = m.H * m.HD;
  let blockRow = wg.y * BM;   // over T (queries)
  let blockCol = wg.x * BN;   // over Tb (keys / pos rows)
  let threadRow = (tid / (BN / TN)) * TM;
  let threadCol = (tid % (BN / TN)) * TN;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  let nT = (m.HD + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    for (var i = 0u; i < 4u; i++) {
      let idxA = tid + i * 256u;
      let aRow = idxA / BK; let aCol = idxA % BK;
      var av = 0.0;
      if (blockRow + aRow < m.T && kk + aCol < m.HD) {
        av = Q[(w * m.T + blockRow + aRow) * stride + h * m.HD + kk + aCol];
        if (m.hasBias == 1u) { av += qb[h * m.HD + kk + aCol]; }
      }
      As[idxA] = av;
      // Bs[k][j]: B row = key j (per-window or shared), col = h*HD + k
      let bRowT = idxA / BN; let bColT = idxA % BN;
      var bv = 0.0;
      let j = blockCol + bColT;
      if (j < m.Tb && kk + bRowT < m.HD) {
        let brow = select(w * m.T + j, j, m.bShared == 1u);
        bv = B[brow * stride + h * m.HD + kk + bRowT];
      }
      Bs[idxA] = bv;
    }
    workgroupBarrier();
    for (var k = 0u; k < BK; k++) {
      var aReg: array<f32, 4>;
      var bReg: array<f32, 4>;
      for (var i = 0u; i < TM; i++) { aReg[i] = As[(threadRow + i) * BK + k]; }
      for (var j2 = 0u; j2 < TN; j2++) { bReg[j2] = Bs[k * BN + threadCol + j2]; }
      for (var i = 0u; i < TM; i++) {
        for (var j2 = 0u; j2 < TN; j2++) { acc[i * TN + j2] += aReg[i] * bReg[j2]; }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < TM; i++) {
    for (var j2 = 0u; j2 < TN; j2++) {
      let r = blockRow + threadRow + i;
      let c = blockCol + threadCol + j2;
      if (r < m.T && c < m.Tb) {
        S[(b * m.T + r) * m.Tb + c] = acc[i * TN + j2];
      }
    }
  }
}`;

// out[i, h*HD+d] = Σ_j P[h*T+i, j] · V[j, h*HD+d]  (probs @ values, all heads)
const BMM_PV_WGSL = `
struct Meta { T:u32, H:u32, HD:u32, W:u32 };
@group(0) @binding(0) var<storage, read> P: array<f32>;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> m: Meta;
const BM = 64u; const BN = 64u; const BK = 16u; const TM = 4u; const TN = 4u;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let b = wg.z;            // window-head block = w*H + h
  let h = b % m.H;
  let w = b / m.H;
  let stride = m.H * m.HD;
  let blockRow = wg.y * BM;   // over T (queries)
  let blockCol = wg.x * BN;   // over HD (head cols; HD<=64 → one x-block)
  let threadRow = (tid / (BN / TN)) * TM;
  let threadCol = (tid % (BN / TN)) * TN;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  let nT = (m.T + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    for (var i = 0u; i < 4u; i++) {
      let idxA = tid + i * 256u;
      let aRow = idxA / BK; let aCol = idxA % BK;
      As[idxA] = select(0.0, P[(b * m.T + blockRow + aRow) * m.T + kk + aCol], blockRow + aRow < m.T && kk + aCol < m.T);
      let bRowT = idxA / BN; let bColT = idxA % BN;
      var bv = 0.0;
      if (kk + bRowT < m.T && blockCol + bColT < m.HD) {
        bv = V[(w * m.T + kk + bRowT) * stride + h * m.HD + blockCol + bColT];
      }
      Bs[idxA] = bv;
    }
    workgroupBarrier();
    for (var k = 0u; k < BK; k++) {
      var aReg: array<f32, 4>;
      var bReg: array<f32, 4>;
      for (var i = 0u; i < TM; i++) { aReg[i] = As[(threadRow + i) * BK + k]; }
      for (var j2 = 0u; j2 < TN; j2++) { bReg[j2] = Bs[k * BN + threadCol + j2]; }
      for (var i = 0u; i < TM; i++) {
        for (var j2 = 0u; j2 < TN; j2++) { acc[i * TN + j2] += aReg[i] * bReg[j2]; }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < TM; i++) {
    for (var j2 = 0u; j2 < TN; j2++) {
      let r = blockRow + threadRow + i;
      let c = blockCol + threadCol + j2;
      if (r < m.T && c < m.HD) {
        Y[(w * m.T + r) * stride + h * m.HD + c] = acc[i * TN + j2];
      }
    }
  }
}`;

// Batched rel_shift: X [H*t, 2t-1] → Y [H*t, t], each head block shifted independently.
const RELSHIFT_B_WGSL = `
struct Meta { t:u32, H:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  if (idx >= m.H * m.t * m.t) { return; }
  let j = idx % m.t;
  let hi = idx / m.t;
  let i = hi % m.t;
  let h = hi / m.t;
  let p = 2u * m.t - 1u;
  let twoT = 2u * m.t;
  let f = m.t + i * p + j;
  let col = f % twoT;
  if (col == 0u) { Y[idx] = 0.0; }
  else { Y[idx] = X[(h * m.t + f / twoT) * p + (col - 1u)]; }
}`;

// Tiled int8 GEMM (v2): the register-blocked 64×64/4×4 structure of GEMM_WGSL with
// the packed int8 B dequanted during the cooperative tile load (4 int8 per u32 →
// 4 consecutive Bs columns per thread; requires N%4==0, guaranteed for the speech
// encoders). Per-column scale + bias + act applied at write-out. ~3× the naive
// 1-thread-per-output int8 kernel on encoder shapes (A-rows reused from shared).
const MATMUL_INT8_V2_WGSL = `
struct Meta { M:u32, N:u32, K:u32, hasBias:u32, act:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> Wq: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> m: Meta;
const BM = 64u; const BN = 64u; const BK = 16u; const TM = 4u; const TN = 4u;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
fn actf(x: f32, a: u32) -> f32 {
  if (a == 1u) { return 0.5 * x * (1.0 + tanh(clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -20.0, 20.0))); }
  if (a == 3u) { return max(x, 0.0); }
  if (a == 4u) { return x / (1.0 + exp(-clamp(x, -30.0, 30.0))); }
  return x;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let blockRow = wg.y * BM;
  let blockCol = wg.x * BN;
  let threadRow = (tid / (BN / TN)) * TM;
  let threadCol = (tid % (BN / TN)) * TN;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  let nT = (m.K + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    // A tile: 1024 f32 / 256 threads = 4 scalar loads
    for (var i = 0u; i < 4u; i++) {
      let idxA = tid + i * 256u;
      let aRow = idxA / BK; let aCol = idxA % BK;
      As[idxA] = select(0.0, A[(blockRow + aRow) * m.K + kk + aCol], blockRow + aRow < m.M && kk + aCol < m.K);
    }
    // B tile: each thread unpacks ONE u32 → 4 consecutive int8 columns
    {
      let base = tid * 4u;                 // element index in the 1024 tile
      let bRow = base / BN;                // k within tile
      let bCol = base % BN;                // n within tile
      let gk = kk + bRow; let gn = blockCol + bCol;
      if (gk < m.K && gn + 3u < m.N) {
        let u = Wq[(gk * m.N + gn) >> 2u];
        for (var j = 0u; j < 4u; j++) {
          var q = i32((u >> (j * 8u)) & 255u);
          if (q > 127) { q = q - 256; }
          Bs[base + j] = f32(q);
        }
      } else {
        for (var j = 0u; j < 4u; j++) {
          var bv = 0.0;
          if (gk < m.K && gn + j < m.N) {
            let li = gk * m.N + gn + j;
            let u2 = Wq[li >> 2u];
            var q2 = i32((u2 >> ((li & 3u) * 8u)) & 255u);
            if (q2 > 127) { q2 = q2 - 256; }
            bv = f32(q2);
          }
          Bs[base + j] = bv;
        }
      }
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
        var v = acc[i * TN + j] * scale[c];
        if (m.hasBias == 1u) { v += bias[c]; }
        Y[r * m.N + c] = actf(v, m.act);
      }
    }
  }
}`;

// Fused TDT joint + argmax (one dispatch per decode step). Computes
// j = relu(encProj[frame] + predProj) [hidden], out = j @ outW + outB [logits],
// then reduces to token argmax (n<vocab) and duration argmax (n>=vocab). Writes
// [tokenIdx, tokenMax, durIdx, durMax] — 4 floats downloaded per frame instead of
// the full 8198 logits. Kills the JS decoder bottleneck.
const TDT_JOINT_WGSL = `
struct Meta { frame:u32, hidden:u32, vocab:u32, logits:u32 };
@group(0) @binding(0) var<storage, read> encProj: array<f32>;
@group(0) @binding(1) var<storage, read> predProj: array<f32>;
@group(0) @binding(2) var<storage, read> outW: array<f32>;
@group(0) @binding(3) var<storage, read> outB: array<f32>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;
@group(0) @binding(5) var<uniform> m: Meta;
var<workgroup> j: array<f32, 640>;
var<workgroup> tIdx: array<u32, 256>;
var<workgroup> tVal: array<f32, 256>;
var<workgroup> dIdx: array<u32, 256>;
var<workgroup> dVal: array<f32, 256>;
// One workgroup per frame in the batch [m.frame, m.frame+numWorkgroups): each computes
// its own joint+argmax with the SAME predProj (valid until the next emission). result
// is [batch,4]. The caller replays the batch in JS and re-dispatches after an emission.
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let L = lid.x;
  let frame = m.frame + wid.x;
  let base = frame * m.hidden;
  let rbase = wid.x * 4u;
  for (var k = L; k < m.hidden; k += 256u) { j[k] = max(0.0, encProj[base + k] + predProj[k]); }
  workgroupBarrier();
  var lt = 0u; var lv = -1e30; var ld = 0u; var ldv = -1e30;
  for (var n = L; n < m.logits; n += 256u) {
    var s = outB[n];
    for (var k = 0u; k < m.hidden; k++) { s += j[k] * outW[k * m.logits + n]; }
    if (n < m.vocab) { if (s > lv) { lv = s; lt = n; } }
    else { if (s > ldv) { ldv = s; ld = n - m.vocab; } }
  }
  tIdx[L] = lt; tVal[L] = lv; dIdx[L] = ld; dVal[L] = ldv;
  workgroupBarrier();
  if (L == 0u) {
    var bt = 0u; var bv = -1e30; var bd = 0u; var bdv = -1e30;
    for (var i = 0u; i < 256u; i++) {
      if (tVal[i] > bv) { bv = tVal[i]; bt = tIdx[i]; }
      if (dVal[i] > bdv) { bdv = dVal[i]; bd = dIdx[i]; }
    }
    result[rbase] = f32(bt); result[rbase + 1u] = bv; result[rbase + 2u] = f32(bd); result[rbase + 3u] = bdv;
  }
}`;

// rel_shift for relative-position attention: X = matrix_bd [t, 2t-1] -> Y [t, t].
// Closed form of NeMo's pad→reshape→slice: Y[i,j] = xp[f], f = t + i*(2t-1) + j,
// where xp is the left-padded [t,2t] view (col 0 = 0). Avoids the GPU→CPU roundtrip.
const RELSHIFT_WGSL = `
struct Meta { t:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  if (idx >= m.t * m.t) { return; }
  let i = idx / m.t;
  let j = idx % m.t;
  let p = 2u * m.t - 1u;
  let twoT = 2u * m.t;
  let f = m.t + i * p + j;
  let col = f % twoT;
  if (col == 0u) { Y[idx] = 0.0; } else { Y[idx] = X[(f / twoT) * p + (col - 1u)]; }
}`;

// GLU over channels (conformer conv module): X:[2C, T] -> Y:[C, T],
// Y[c,t] = X[c,t] * sigmoid(X[c+C, t]).
const GLU_WGSL = `
struct Meta { C:u32, T:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  if (idx >= m.C * m.T) { return; }
  let c = idx / m.T;
  let t = idx % m.T;
  let a = X[c * m.T + t];
  let b = X[(c + m.C) * m.T + t];
  Y[idx] = a * (1.0 / (1.0 + exp(-clamp(b, -30.0, 30.0))));
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

// Snake activation (StyleTTS2/iSTFTNet): y = x + (1/(α+1e-9)) * sin(αx)² , with a
// per-CHANNEL α (one α per row). alpha:[1,C] (C = x.rows).
const SNAKE_WGSL = `
struct Meta { C:u32, L:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.y * (nwg.x * 64u) + g.x;
  if (i >= m.C * m.L) { return; }
  let a = alpha[i / m.L];
  let s = sin(a * X[i]);
  Y[i] = X[i] + (1.0 / (a + 1e-9)) * s * s;
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
      // clamp tanh args: Metal tanh(x) = exp-based → Inf/Inf = NaN for |x| ≳ 44
      // (cell state drifts past that on long sequences; tanh saturates by ±20).
      let cnew = sig(gf) * csh[u] + sig(gi) * tanh(clamp(gc, -20.0, 20.0));
      csh[u] = cnew;
      let ht = sig(go) * tanh(clamp(cnew, -20.0, 20.0));
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

const ACT = { none: 0, gelu: 1, tanh: 2, relu: 3, silu: 4, gelu_erf: 5 };

/** Request a WebGPU device in the browser (throws if unavailable). */
export async function requestGpuDevice() {
  if (typeof navigator === "undefined" || !navigator.gpu) throw new Error("WebGPU not available");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const lim = adapter.limits;
  return adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: lim.maxBufferSize,
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
    },
  });
}

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

  _run(pipeline, buffers, uniform, groupsX, groupsY = 1, groupsZ = 1) {
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
      this._pass.dispatchWorkgroups(groupsX, groupsY, groupsZ);
      return;
    }
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(groupsX, groupsY, groupsZ);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** C = act(A@B + bias). a:[M,K] b:[K,N] bias?:[1,N] -> [M,N] */
  matmul(a, b, { bias = null, act = "none" } = {}) {
    const M = a.rows, K = a.cols, N = b.cols;
    // Large aligned GEMMs benefit from the 128×128/8×8 vec4 kernel (~70% of MLX,
    // vs ~58% for the scalar kernel). Thin/small GEMMs are launch/occupancy-bound —
    // v4 gives no gain there and wastes work padding M/N to 128, so keep v1.
    if (M >= 256 && N >= 256 && K >= 256 && K % 8 === 0 && N % 4 === 0) {
      return this.matmulV4(a, b, { bias, act });
    }
    const c = this.alloc(M, N);
    const biasBuf = bias ? bias.buf : this.alloc(1, 1).buf;
    const pipeline = this._pipeline("gemm", GEMM_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, 0, 0, 0]));
    this._run(pipeline, [a.buf, b.buf, biasBuf, c.buf], u, Math.ceil(N / 64), Math.ceil(M / 64));
    return c;
  }

  matmulV2(a, b, { bias = null, act = "none" } = {}) {
    const M = a.rows, K = a.cols, N = b.cols;
    const c = this.alloc(M, N);
    const biasBuf = bias ? bias.buf : this.alloc(1, 1).buf;
    const pipeline = this._pipeline("gemmV2", GEMM_V2_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, 0, 0, 0]));
    this._run(pipeline, [a.buf, b.buf, biasBuf, c.buf], u, Math.ceil(N / 64), Math.ceil(M / 64));
    return c;
  }

  matmulV3(a, b, { bias = null, act = "none" } = {}) {
    const M = a.rows, K = a.cols, N = b.cols;
    const c = this.alloc(M, N);
    const biasBuf = bias ? bias.buf : this.alloc(1, 1).buf;
    const pipeline = this._pipeline("gemmV3", GEMM_V3_WGSL);
    const u = this._uniform(new Uint32Array([M, N, K, ACT[act], bias ? 1 : 0, 0, 0, 0]));
    this._run(pipeline, [a.buf, b.buf, biasBuf, c.buf], u, Math.ceil(N / 64), Math.ceil(M / 64));
    return c;
  }

  matmulV4(a, b, { bias = null, act = "none" } = {}) {
    const M = a.rows, K = a.cols, N = b.cols;
    const c = this.alloc(M, N);
    const biasBuf = bias ? bias.buf : this.alloc(1, 1).buf;
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
  conv1d(x, w, { cout, k, bias = null, stride = 1, pad = 0, padLeft, padRight, dilation = 1, groups = 1, act = "none" } = {}) {
    // Asymmetric pad supported (padLeft/padRight) for causal convs; default symmetric pad.
    padLeft = padLeft ?? pad; padRight = padRight ?? pad;
    // groups==1 symmetric-pad convs route to the fused implicit-GEMM kernel
    // (~7× the direct kernel on the big vocoder convs; same flat weight layout).
    // Asymmetric-pad and grouped/depthwise convs stay on the direct kernel.
    if (groups === 1 && padLeft === padRight && (act === "none" || act === "gelu" || act === "tanh" || act === "relu")) {
      return this.conv1dFast(x, w, cout, k, { bias, stride, pad: padLeft, dilation, act });
    }
    const Cin = x.rows, L = x.cols;
    const Lout = Math.floor((L + padLeft + padRight - dilation * (k - 1) - 1) / stride) + 1;
    const y = this.alloc(cout, Lout);
    const biasBuf = bias ? bias.buf : this.alloc(1, 1).buf;
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
    const biasBuf = bias ? bias.buf : this.alloc(1, 1).buf;
    const u = this._uniform(new Uint32Array([M, N, K, bias ? 1 : 0, ACT[act], 0, 0, 0]));
    if (N % 4 === 0) {
      // tiled/register-blocked variant (~3× on encoder shapes)
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
    const T = q.rows / W, Tb = bShared ? b.rows : b.rows / W;
    const s = this.alloc(W * H * T, Tb);
    const pipeline = this._pipeline("bmmqk", BMM_QK_WGSL);
    const u = this._uniform(new Uint32Array([T, Tb, H, HD, qb ? 1 : 0, W, bShared ? 1 : 0, 0]));
    this._run(pipeline, [q.buf, b.buf, qb ? qb.buf : this.alloc(1, 1).buf, s.buf], u, Math.ceil(Tb / 64), Math.ceil(T / 64), W * H);
    return s;
  }

  /** Batched probs@V over all heads: p[H*T,T], v[T,H*HD] → [T, H*HD]. */
  bmmPV(p, v, H, HD, W = 1) {
    const T = v.rows / W;
    const y = this.alloc(W * T, H * HD);
    const pipeline = this._pipeline("bmmpv", BMM_PV_WGSL);
    const u = this._uniform(new Uint32Array([T, H, HD, W]));
    this._run(pipeline, [p.buf, v.buf, y.buf], u, Math.ceil(HD / 64) || 1, Math.ceil(T / 64), W * H);
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
    const pipeline = this._pipeline("silu", `
      struct Meta { n:u32, _a:u32, _b:u32, _c:u32 };
      @group(0) @binding(0) var<storage, read> X: array<f32>;
      @group(0) @binding(1) var<storage, read_write> Y: array<f32>;
      @group(0) @binding(2) var<uniform> m: Meta;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
        let i = gid.y * (nwg.x * 64u) + gid.x; if (i >= m.n) { return; }
        let v = X[i]; Y[i] = v / (1.0 + exp(-clamp(v, -30.0, 30.0)));
      }`);
    const u = this._uniform(new Uint32Array([x.rows * x.cols, 0, 0, 0]));
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil((x.rows * x.cols) / 64));
    return y;
  }

  /** Elementwise ReLU (standalone; matmul act=relu covers the fused case). */
  relu(x) {
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline("relu", `
      struct Meta { n:u32, _a:u32, _b:u32, _c:u32 };
      @group(0) @binding(0) var<storage, read> X: array<f32>;
      @group(0) @binding(1) var<storage, read_write> Y: array<f32>;
      @group(0) @binding(2) var<uniform> m: Meta;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
        let i = gid.y * (nwg.x * 64u) + gid.x; if (i >= m.n) { return; }
        Y[i] = max(X[i], 0.0);
      }`);
    const u = this._uniform(new Uint32Array([x.rows * x.cols, 0, 0, 0]));
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil((x.rows * x.cols) / 64));
    return y;
  }

  /** GLU over channels: x:[2C, T] -> [C, T], y[c,t] = x[c,t]*sigmoid(x[c+C,t]). */
  glu(x) {
    const C = x.rows / 2, T = x.cols;
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
  conv2d(x, w, { cout, cin, h, w: W_, kh, kw, bias = null, strideH = 1, strideW = 1, padH = 0, padW = 0, padTop, padBottom, padLeft, padRight, groups = 1, act = "none" } = {}) {
    // Asymmetric padding supported (padTop/Bottom/Left/Right); default symmetric padH/padW.
    padTop = padTop ?? padH; padBottom = padBottom ?? padH; padLeft = padLeft ?? padW; padRight = padRight ?? padW;
    const Ho = Math.floor((h + padTop + padBottom - kh) / strideH) + 1;
    const Wo = Math.floor((W_ + padLeft + padRight - kw) / strideW) + 1;
    const y = this.alloc(cout, Ho * Wo);
    const biasBuf = bias ? bias.buf : this.alloc(1, 1).buf;
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

  /** Multiply by a scalar constant (elementwise). */
  scale(x, s) {
    const n = x.rows * x.cols;
    const y = this.alloc(x.rows, x.cols);
    const pipeline = this._pipeline("scalek", `
      struct Meta { n:u32, s:f32, _a:u32, _b:u32 };
      @group(0) @binding(0) var<storage, read> X: array<f32>;
      @group(0) @binding(1) var<storage, read_write> Y: array<f32>;
      @group(0) @binding(2) var<uniform> m: Meta;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
        let i = gid.y * (nwg.x * 64u) + gid.x; if (i >= m.n) { return; }
        Y[i] = X[i] * m.s;
      }`);
    const meta = new ArrayBuffer(16);
    new Uint32Array(meta, 0, 1)[0] = n;
    new Float32Array(meta, 4, 1)[0] = s;
    const u = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(u, 0, meta);
    this._run(pipeline, [x.buf, y.buf], u, Math.ceil(n / 64));
    return y;
  }

  /** Concat along rows (row-major ⇒ contiguous buffer concatenation, no readback). */
  concatRows(tensors) {
    const cols = tensors[0].cols;
    const rows = tensors.reduce((s, t) => s + t.rows, 0);
    const out = this.alloc(rows, cols);
    const enc = this.device.createCommandEncoder();
    let off = 0;
    for (const t of tensors) {
      enc.copyBufferToBuffer(t.buf, 0, out.buf, off * 4, t.rows * t.cols * 4);
      off += t.rows * t.cols;
    }
    this.device.queue.submit([enc.finish()]);
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
