// Elementwise / data-movement kernels: GLU, gathers, slicing, transpose, etc.

// GLU over channels (conformer conv module): X:[2C, T] -> Y:[C, T],
// Y[c,t] = X[c,t] * sigmoid(X[c+C, t]).
export const GLU_WGSL = `
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

// Column gather (length regulator): Y[:, f] = X[:, idx[f]]. Expands text features
// to mel frames by repeating each column per its predicted duration (idx = the
// frame→text-token map, a duration cumsum built on the CPU).
export const GATHERCOLS_WGSL = `
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
export const LEAKY_WGSL = `
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
export const SNAKE_WGSL = `
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
export const EWISE_WGSL = `
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
export const TRANSPOSE_WGSL = `
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

export const SLICECOLS_WGSL = `
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

export const SETCOLS_WGSL = `
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
