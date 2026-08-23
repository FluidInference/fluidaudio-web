// Gemma-family decoder kernels: RMSNorm ((1+w) scaling), fused per-head
// RMSNorm + RoPE on q/k projections, and cached causal attention against a
// stream-strided KV cache. Written for the tts-voicechat Gemma3 backbone but
// generic over heads/head_dim; the WASM counterparts in wasm-context.js are the
// byte-exact host references (f64 accumulators) that the parity smoke gates on.

// y = x * rsqrt(mean(x²)+eps) * (1+w) [+ base]. One 64-lane workgroup per row.
export const RMSNORM_WGSL = `
struct Meta { rows:u32, cols:u32, hasAdd:u32, eps:f32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> BASE: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wg.x;
  if (row >= m.rows) { return; }
  let base = row * m.cols;
  var sum = 0.0;
  for (var j = li; j < m.cols; j += 64u) { let v = X[base + j]; sum += v * v; }
  red[li] = sum;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / f32(m.cols) + m.eps);
  for (var j = li; j < m.cols; j += 64u) {
    var v = X[base + j] * inv * (1.0 + W[j]);
    if (m.hasAdd == 1u) { v += BASE[base + j]; }
    Y[base + j] = v;
  }
}`;

// Fused per-head RMSNorm (optional) + rotate-half RoPE on a [rows, H*HD]
// projection. Position of row r is pos0 + (r % M) — rows are stream-major
// (CFG stream s at rows [s*M, (s+1)*M)). One workgroup per (row, head).
export const HEADRMS_ROPE_WGSL = `
struct Meta { rows:u32, heads:u32, hd:u32, M:u32, pos0:u32, doNorm:u32, scale:f32, eps:f32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> F: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wg.x / m.heads;
  let h = wg.x % m.heads;
  if (row >= m.rows) { return; }
  let o = row * m.heads * m.hd + h * m.hd;
  var inv = 1.0;
  if (m.doNorm == 1u) {
    var sum = 0.0;
    for (var j = li; j < m.hd; j += 64u) { let v = X[o + j]; sum += v * v; }
    red[li] = sum;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
    inv = inverseSqrt(red[0] / f32(m.hd) + m.eps);
  }
  let half = m.hd / 2u;
  let p = f32(m.pos0 + (row % m.M));
  for (var i = li; i < half; i += 64u) {
    var a = X[o + i];
    var b = X[o + half + i];
    if (m.doNorm == 1u) {
      a = a * inv * (1.0 + W[i]);
      b = b * inv * (1.0 + W[half + i]);
    }
    let f = p * F[i];
    let c = cos(f);
    let s = sin(f);
    Y[o + i] = (a * c - b * s) * m.scale;
    Y[o + half + i] = (b * c + a * s) * m.scale;
  }
}`;

// Scores for cached attention: q[W*M, H*HD] vs K[W*stride, H*HD] (stream w's
// keys start at row w*stride) → S[W*H*M, Tk]. Causal rows mask j > pos0+i to
// -3e38 (softmax zero); optional logit softcap (CAS): s = cap·tanh(s/cap).
export const ATTN_SCORES_WGSL = `
struct Meta { R:u32, tk:u32, heads:u32, hd:u32, M:u32, pos0:u32, stride:u32, causal:u32, softcap:f32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read_write> S: array<f32>;
@group(0) @binding(3) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let id = gid.y * (nwg.x * 64u) + gid.x;
  if (id >= m.R * m.tk) { return; }
  let r = id / m.tk;
  let j = id % m.tk;
  let w = r / (m.heads * m.M);
  let rem = r % (m.heads * m.M);
  let h = rem / m.M;
  let i = rem % m.M;
  if (m.causal == 1u && j > m.pos0 + i) { S[id] = -3.0e38; return; }
  let d = m.heads * m.hd;
  let qBase = (w * m.M + i) * d + h * m.hd;
  let kBase = (w * m.stride + j) * d + h * m.hd;
  var s = 0.0;
  for (var c = 0u; c < m.hd; c++) { s += Q[qBase + c] * K[kBase + c]; }
  if (m.softcap > 0.0) { s = m.softcap * tanh(s / m.softcap); }
  S[id] = s;
}`;

// probs @ V for cached attention: P[W*H*M, Tk], V[W*stride, H*HD] → [W*M, H*HD].
export const ATTN_PV_WGSL = `
struct Meta { rowsOut:u32, tk:u32, heads:u32, hd:u32, M:u32, pos0:u32, stride:u32, causal:u32 };
@group(0) @binding(0) var<storage, read> P: array<f32>;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let d = m.heads * m.hd;
  let id = gid.y * (nwg.x * 64u) + gid.x;
  if (id >= m.rowsOut * d) { return; }
  let row = id / d;
  let c = id % d;
  let w = row / m.M;
  let i = row % m.M;
  let h = c / m.hd;
  var bound = m.tk;
  if (m.causal == 1u) { bound = m.pos0 + i + 1u; }
  let pBase = ((w * m.heads + h) * m.M + i) * m.tk;
  var acc = 0.0;
  for (var j = 0u; j < bound; j++) { acc += P[pBase + j] * V[(w * m.stride + j) * d + c]; }
  Y[id] = acc;
}`;
