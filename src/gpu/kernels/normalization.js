// Row-wise LayerNorm, softmax, and AdaIN (instance norm + affine).

// Row-wise LayerNorm over the last dim: y = (x-mean)/sqrt(var+eps) * gamma + beta.
// One workgroup per row; 64 lanes cooperatively reduce via shared memory.
export const LAYERNORM_WGSL = `
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
export const SOFTMAX_WGSL = `
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

// AdaIN: instance-norm over time (per channel) + style-predicted per-channel affine.
// x:[C,L], scale/shift:[C] -> y = (x-mean_c)/sqrt(var_c+eps)*scale[c] + shift[c].
// One workgroup per channel; 64-lane reduce over L.
export const ADAIN_WGSL = `
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
