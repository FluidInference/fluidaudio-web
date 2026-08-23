// GEMM kernel family: fp32 tiled variants (v1-v4), f16-weight / f16-compute /
// subgroup / tile-major variants, and the int8 + int4 (NBits) quantized kernels.

import { WGSL_ACTF } from "./actf.js";

// C = act(A[MxK] @ B[KxN] + bias[N]).  Register-blocked: each 256-thread workgroup
// computes a 64×64 output block; each thread a 4×4 micro-tile from registers, with
// 64×16 / 16×64 shared-memory staging. ~5× the naive tiled kernel by amortizing
// shared-memory reads over 16 MACs each. bias per-N, act: 0 none/1 gelu/2 tanh/3 relu.
export const GEMM_WGSL = `
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM = 64u; const BN = 64u; const BK = 16u; const TM = 4u; const TN = 4u;
var<workgroup> As: array<f32, 1024>; // BM*BK
var<workgroup> Bs: array<f32, 1024>; // BK*BN

${WGSL_ACTF}
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
        C[r * m.N + c] = actf(v, m.act);
      }
    }
  }
}`;

// Thin-M GEMV family (M ≤ 4, fp32 B): decode-loop GEMVs are pure weight-bandwidth
// reads, but the 64×64 tile kernels launch only N/64 workgroups at M=2 — far too
// few to cover memory latency (measured ~26 GB/s effective on the voicechat
// backbone). Split-K: pass 1 launches (N-strip)·KS workgroups, each accumulating
// a ~32-step K-slice (B reads stay coalesced across threads, all M rows share
// every B read); pass 2 reduces the KS partials + bias + act. Consecutive
// dispatches in a pass serialize on barriers, so per-dispatch occupancy is the
// lever — hence KS up to 64.
export const GEMV_PART_WGSL = `
struct Meta { M:u32, N:u32, K:u32, KS:u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> P: array<f32>;
@group(0) @binding(3) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  let s = gid.y;
  if (n >= m.N || s >= m.KS) { return; }
  let kPer = (m.K + m.KS - 1u) / m.KS;
  let k0 = s * kPer;
  let k1 = min(m.K, k0 + kPer);
  var acc: array<f32, 4>;
  for (var i = 0u; i < 4u; i++) { acc[i] = 0.0; }
  for (var k = k0; k < k1; k++) {
    let b = B[k * m.N + n];
    for (var r = 0u; r < m.M; r++) { acc[r] += A[r * m.K + k] * b; }
  }
  for (var r = 0u; r < m.M; r++) { P[(s * m.M + r) * m.N + n] = acc[r]; }
}`;

export const GEMV_REDUCE_WGSL = `
struct Meta { M:u32, N:u32, KS:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> P: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> m: Meta;
${WGSL_ACTF}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= m.N) { return; }
  for (var r = 0u; r < m.M; r++) {
    var v = 0.0;
    for (var s = 0u; s < m.KS; s++) { v += P[(s * m.M + r) * m.N + n]; }
    if (m.hasBias == 1u) { v += bias[n]; }
    C[r * m.N + n] = actf(v, m.act);
  }
}`;

// vec4-column variant (N%4==0): each thread owns 4 consecutive columns via one
// vec4 B load per k — 4× the bytes in flight per thread at the same thread
// count, which is what the latency-bound weight stream needs.
export const GEMV_PART4_WGSL = `
struct Meta { M:u32, N4:u32, K:u32, KS:u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> P: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n4 = gid.x;
  let s = gid.y;
  if (n4 >= m.N4 || s >= m.KS) { return; }
  let kPer = (m.K + m.KS - 1u) / m.KS;
  let k0 = s * kPer;
  let k1 = min(m.K, k0 + kPer);
  var acc: array<vec4<f32>, 4>;
  for (var i = 0u; i < 4u; i++) { acc[i] = vec4<f32>(0.0); }
  for (var k = k0; k < k1; k++) {
    let b = B[k * m.N4 + n4];
    for (var r = 0u; r < m.M; r++) { acc[r] += A[r * m.K + k] * b; }
  }
  for (var r = 0u; r < m.M; r++) { P[(s * m.M + r) * m.N4 + n4] = acc[r]; }
}`;

export const GEMV_REDUCE4_WGSL = `
struct Meta { M:u32, N4:u32, KS:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> P: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> m: Meta;
${WGSL_ACTF}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n4 = gid.x;
  if (n4 >= m.N4) { return; }
  for (var r = 0u; r < m.M; r++) {
    var v = vec4<f32>(0.0);
    for (var s = 0u; s < m.KS; s++) { v += P[(s * m.M + r) * m.N4 + n4]; }
    if (m.hasBias == 1u) { v += bias[n4]; }
    C[r * m.N4 + n4] = vec4<f32>(actf(v.x, m.act), actf(v.y, m.act), actf(v.z, m.act), actf(v.w, m.act));
  }
}`;

// Vectorized GEMM (siboehm kernel 6): same 64×64 block / 4×4 micro-tile, but As is
// staged TRANSPOSED ([BK][BM]) so each thread's inner-loop A read is 4 contiguous
// floats (bank-conflict-free, vec4-loadable) and the 4×4 MAC is 4 vec4 FMAs. Cuts
// shared-memory instruction count ~4× vs the scalar kernel. Same bias/act semantics.
export const GEMM_V2_WGSL = `
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM = 64u; const BN = 64u; const BK = 16u; const TM = 4u; const TN = 4u;
var<workgroup> As: array<f32, 1024>; // TRANSPOSED [BK][BM]
var<workgroup> Bs: array<f32, 1024>; // [BK][BN]
${WGSL_ACTF}
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
        C[r * m.N + c] = actf(x, m.act);
      }
    }
  }
}`;

// GEMM v3: v2 + 128-bit GMEM loads (A/B bound as vec4<f32>). Tests whether reducing
// global load instructions 4× breaks the plateau. Requires K%4==0 and N%4==0 (bench).
export const GEMM_V3_WGSL = `
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
export const GEMM_V4_WGSL = `
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM = 128u; const BN = 128u; const BK = 8u; const TM = 8u; const TN = 8u;
var<workgroup> As: array<f32, 1024>; // TRANSPOSED [BK][BM]
var<workgroup> Bs: array<f32, 1024>; // [BK][BN]
${WGSL_ACTF}
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
        if (c < m.N) { var x = v[jj]; if (m.hasBias == 1u) { x += bias[c]; } C[r * m.N + c] = actf(x, m.act); }
      }
    }
  }
}`;

// Mixed-precision GEMM: fp32 A (activations) x f16 B (weights) with the v4
// 128x128/8x8 vec4 structure and f32 accumulation. Halves the dominant weight
// traffic of the encoder GEMMs; activations stay fp32 (no requant error).
// v4 structure, f16 COMPUTE: f16 shared tiles (half the LDS of v4 → better
// occupancy) and f16 fma in the inner loop — 2× ALU rate on Apple GPUs. The f16
// accumulator only ever holds one 8-deep K-tile (BK MACs) before being promoted
// into the f32 accumulator, so rounding never compounds across K. Products are
// f16 either way on the f16-weight path; the extra error vs F16B is A's f16
// rounding (~2^-11 relative) — an order finer than the int8 weight quantization
// the encoder already carries. Gated by token identity, not just maxΔ.
// f16C with BK=16: half the barriers and half the f32-promote overhead per K.
// LDS 2×(16×128) f16 = 8KB — same budget as the fp32 v4 kernel's 8KB.
// Subgroup f16 GEMM — design adapted from narcotic-sh/parakeet.wgsl (MIT):
// no workgroup memory, no barriers. Each 32-lane subgroup computes 8 rows;
// lanes 0..7 load one packed A k-vector each and subgroupBroadcast distributes
// them; every lane owns 8 output columns (two vec4<f16>) read directly from
// row-major f16 B. Full-f16 accumulation (WER-validated upstream at 1.69%
// LibriSpeech; our token gates verify our models). Requires subgroup_size 32
// with contiguous lane mapping — probeSubgroups() verifies both at runtime.
export const GEMM_F16SG_WGSL = `enable f16;
enable subgroups;
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, hasAdd:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> Dt: array<f32>;
@group(0) @binding(5) var<uniform> m: Meta;
${WGSL_ACTF}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(subgroup_invocation_id) lane: u32) {
  let sg = li / 32u;                       // 4 subgroups per workgroup
  let rowBase = wg.y * 32u + sg * 8u;      // 8 rows per subgroup
  // Column ownership split for COALESCING: lane owns cols [lane*4..+3] and
  // [128 + lane*4..+3] — each of the subgroup's two B loads is then 32
  // CONSECUTIVE vec4s (one 128-byte burst per half-tile), instead of lanes
  // striding every other vec4 (lane*8 ownership).
  let colTile = wg.x * 256u;
  let col0 = colTile + lane * 4u;          // first vec4-column group
  let col1 = colTile + 128u + lane * 4u;   // second vec4-column group
  let K4 = m.K / 4u;
  let N4 = m.N / 4u;
  let cva = col0 / 4u;
  let cvb = col1 / 4u;
  var acc: array<vec4<f16>, 16>;           // [row][2 col-vecs], full-f16 accumulation
  for (var i = 0u; i < 16u; i++) { acc[i] = vec4<f16>(0.0h); }
  let colInBounds = col1 + 3u < m.N;
  for (var kv = 0u; kv < K4; kv++) {
    // lanes 0..7 each own one row's A k-vector; broadcast to the subgroup
    var packed_a = vec2<u32>(0u);
    if (lane < 8u && rowBase + lane < m.M) {
      let av = vec4<f16>(A[(rowBase + lane) * K4 + kv]);
      packed_a = bitcast<vec2<u32>>(av);
    }
    var a: array<vec4<f16>, 8>;
    a[0] = bitcast<vec4<f16>>(subgroupBroadcast(packed_a, 0u));
    a[1] = bitcast<vec4<f16>>(subgroupBroadcast(packed_a, 1u));
    a[2] = bitcast<vec4<f16>>(subgroupBroadcast(packed_a, 2u));
    a[3] = bitcast<vec4<f16>>(subgroupBroadcast(packed_a, 3u));
    a[4] = bitcast<vec4<f16>>(subgroupBroadcast(packed_a, 4u));
    a[5] = bitcast<vec4<f16>>(subgroupBroadcast(packed_a, 5u));
    a[6] = bitcast<vec4<f16>>(subgroupBroadcast(packed_a, 6u));
    a[7] = bitcast<vec4<f16>>(subgroupBroadcast(packed_a, 7u));
    if (colInBounds) {
      let k0 = kv * 4u;
      for (var j = 0u; j < 4u; j++) {
        let bRow = (k0 + j) * N4;
        let b0 = B[bRow + cva];
        let b1 = B[bRow + cvb];
        for (var r = 0u; r < 8u; r++) {
          let ar = a[r][j];
          acc[r * 2u] += ar * b0;
          acc[r * 2u + 1u] += ar * b1;
        }
      }
    }
  }
  if (!colInBounds) { return; } // edge columns fall to the guarded kernels via routing
  for (var r = 0u; r < 8u; r++) {
    let row = rowBase + r;
    if (row >= m.M) { continue; }
    for (var cb = 0u; cb < 2u; cb++) {
      let v = vec4<f32>(acc[r * 2u + cb]);
      let cBase = select(col0, col1, cb == 1u);
      for (var jj = 0u; jj < 4u; jj++) {
        let c = cBase + jj;
        var x = f32(v[jj]);
        if (m.hasBias == 1u) { x += bias[c]; }
        x = actf(x, m.act);
        if (m.hasAdd == 1u) { x += Dt[row * m.N + c]; }
        C[row * m.N + c] = x;
      }
    }
  }
}`;

export const GEMM_V4_F16C2_WGSL = `enable f16;
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, hasAdd:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> Dt: array<f32>;
@group(0) @binding(5) var<uniform> m: Meta;
const BM = 128u; const BN = 128u; const BK = 16u; const TM = 8u; const TN = 8u;
var<workgroup> As: array<f16, 2048>; // TRANSPOSED [BK][BM]
var<workgroup> Bs: array<f16, 2048>; // [BK][BN]
${WGSL_ACTF}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let blockRow = wg.y * BM; let blockCol = wg.x * BN;
  let threadRow = (tid / (BN / TN)) * TM;
  let threadCol = (tid % (BN / TN)) * TN;
  let K4 = m.K / 4u; let N4 = m.N / 4u;
  var acc: array<vec4<f32>, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = vec4<f32>(0.0); }
  let nT = (m.K + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    // A tile: [128 rows][16 cols] = 512 vec4 → 2 per thread
    for (var it = 0u; it < 2u; it++) {
      let idx = tid + it * 256u;
      let aRow = idx / 4u; let aC = idx % 4u;
      var av = vec4<f32>(0.0);
      if (blockRow + aRow < m.M && kk + aC * 4u < m.K) { av = A[(blockRow + aRow) * K4 + kk / 4u + aC]; }
      let ah = vec4<f16>(av);
      let aBase = 4u * aC;
      As[(aBase + 0u) * BM + aRow] = ah.x; As[(aBase + 1u) * BM + aRow] = ah.y;
      As[(aBase + 2u) * BM + aRow] = ah.z; As[(aBase + 3u) * BM + aRow] = ah.w;
    }
    // B tile: [16 rows][128 cols] = 512 vec4 → 2 per thread
    for (var it = 0u; it < 2u; it++) {
      let idx = tid + it * 256u;
      let bRow = idx / 32u; let bC = idx % 32u;
      var bv = vec4<f16>(0.0);
      if (kk + bRow < m.K && blockCol + bC * 4u < m.N) { bv = B[(kk + bRow) * N4 + blockCol / 4u + bC]; }
      let bBase = bRow * BN + 4u * bC;
      Bs[bBase] = bv.x; Bs[bBase + 1u] = bv.y; Bs[bBase + 2u] = bv.z; Bs[bBase + 3u] = bv.w;
    }
    workgroupBarrier();
    var acch: array<vec4<f16>, 16>;
    for (var i = 0u; i < 16u; i++) { acch[i] = vec4<f16>(0.0); }
    for (var k = 0u; k < BK; k++) {
      let ab = k * BM + threadRow;
      var aReg: array<f16, 8>;
      for (var i = 0u; i < 8u; i++) { aReg[i] = As[ab + i]; }
      let bb = k * BN + threadCol;
      let b0 = vec4<f16>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
      let b1 = vec4<f16>(Bs[bb + 4u], Bs[bb + 5u], Bs[bb + 6u], Bs[bb + 7u]);
      for (var i = 0u; i < 8u; i++) { acch[i * 2u] += aReg[i] * b0; acch[i * 2u + 1u] += aReg[i] * b1; }
    }
    for (var i = 0u; i < 16u; i++) { acc[i] += vec4<f32>(acch[i]); }
    workgroupBarrier();
  }
  for (var i = 0u; i < TM; i++) {
    let r = blockRow + threadRow + i; if (r >= m.M) { continue; }
    for (var jb = 0u; jb < 2u; jb++) {
      let v = acc[i * 2u + jb];
      for (var jj = 0u; jj < 4u; jj++) {
        let c = blockCol + threadCol + jb * 4u + jj;
        if (c < m.N) {
          var x = v[jj];
          if (m.hasBias == 1u) { x += bias[c]; }
          x = actf(x, m.act);
          if (m.hasAdd == 1u) { x += Dt[r * m.N + c]; }
          C[r * m.N + c] = x;
        }
      }
    }
  }
}`;

// Tile-major direct-B subgroup GEMM — the parakeet.wgsl geometry (task #27).
// Workgroup = 128 lanes = 4 subgroups covering a 32-row × 256-col C tile; each
// subgroup owns 8 rows × 256 cols; each lane owns 8 columns (2 vec4). B is
// PREPACKED tile-major ([colTile][K/32][32][32 packs of 8 f16] — vec4<u32>),
// read directly from global memory: NO workgroup memory, NO barriers. A rows
// are loaded by lanes 0..7 and distributed per-scalar via subgroupBroadcast.
// Accumulators are vec4<f16> end-to-end (f16 FMA = 2× f32 rate on Apple),
// promoted to f32 once at the epilogue (bias → act → residual add, f32 C).
export const GEMM_TM_WGSL = `enable f16;
enable subgroups;
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, hasAdd:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> Dt: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> m: Meta;
${WGSL_ACTF}
@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_index) li: u32,
  @builtin(workgroup_id) wg: vec3<u32>,
) {
  // subgroup_id/subgroup_invocation_id builtins are gated on some runtimes;
  // probeSubgroups() verified 32-lane CONTIGUOUS mapping, so both derive
  // from the local index (this kernel is only routed when the probe passed).
  let lane = li % 32u;
  let sg = li / 32u;
  let K4 = m.K / 4u;
  let N4 = m.N / 4u;
  let rowBase = wg.y * 32u + sg * 8u;
  let kTiles = m.K / 32u;
  var acc: array<vec4<f16>, 16>; // 8 rows × 2 col-vec4
  for (var i = 0u; i < 16u; i++) { acc[i] = vec4<f16>(0.0h); }
  let tileBase0 = wg.x * kTiles * 1024u; // 32 k-rows × 32 packs per K-tile
  for (var kb = 0u; kb < kTiles; kb++) {
    let tileBase = tileBase0 + kb * 1024u;
    for (var kv = 0u; kv < 8u; kv++) {
      var av = vec4<f32>(0.0);
      let aRow = rowBase + lane;
      if (lane < 8u && aRow < m.M) { av = A[aRow * K4 + kb * 8u + kv]; }
      // one vec4 broadcast per row per k_vector (8 total) instead of 32
      // scalar broadcasts; f16-convert once per row.
      let r0 = vec4<f16>(subgroupBroadcast(av, 0u));
      let r1 = vec4<f16>(subgroupBroadcast(av, 1u));
      let r2 = vec4<f16>(subgroupBroadcast(av, 2u));
      let r3 = vec4<f16>(subgroupBroadcast(av, 3u));
      let r4 = vec4<f16>(subgroupBroadcast(av, 4u));
      let r5 = vec4<f16>(subgroupBroadcast(av, 5u));
      let r6 = vec4<f16>(subgroupBroadcast(av, 6u));
      let r7 = vec4<f16>(subgroupBroadcast(av, 7u));
      for (var c = 0u; c < 4u; c++) {
        let pb = B[tileBase + (kv * 4u + c) * 32u + lane];
        let b0 = bitcast<vec4<f16>>(pb.xy);
        let b1 = bitcast<vec4<f16>>(pb.zw);
        let a0 = vec4<f16>(r0[c]);
        acc[0] = fma(a0, b0, acc[0]); acc[1] = fma(a0, b1, acc[1]);
        let a1 = vec4<f16>(r1[c]);
        acc[2] = fma(a1, b0, acc[2]); acc[3] = fma(a1, b1, acc[3]);
        let a2 = vec4<f16>(r2[c]);
        acc[4] = fma(a2, b0, acc[4]); acc[5] = fma(a2, b1, acc[5]);
        let a3 = vec4<f16>(r3[c]);
        acc[6] = fma(a3, b0, acc[6]); acc[7] = fma(a3, b1, acc[7]);
        let a4 = vec4<f16>(r4[c]);
        acc[8] = fma(a4, b0, acc[8]); acc[9] = fma(a4, b1, acc[9]);
        let a5 = vec4<f16>(r5[c]);
        acc[10] = fma(a5, b0, acc[10]); acc[11] = fma(a5, b1, acc[11]);
        let a6 = vec4<f16>(r6[c]);
        acc[12] = fma(a6, b0, acc[12]); acc[13] = fma(a6, b1, acc[13]);
        let a7 = vec4<f16>(r7[c]);
        acc[14] = fma(a7, b0, acc[14]); acc[15] = fma(a7, b1, acc[15]);
      }
    }
  }
  let colVec0 = wg.x * 64u + lane * 2u;
  for (var r = 0u; r < 8u; r++) {
    let row = rowBase + r;
    if (row >= m.M) { continue; }
    for (var cv = 0u; cv < 2u; cv++) {
      let colVec = colVec0 + cv;
      var v = vec4<f32>(acc[r * 2u + cv]);
      if (m.hasBias == 1u) {
        let cb = colVec * 4u;
        v += vec4<f32>(bias[cb], bias[cb + 1u], bias[cb + 2u], bias[cb + 3u]);
      }
      v = vec4<f32>(actf(v.x, m.act), actf(v.y, m.act), actf(v.z, m.act), actf(v.w, m.act));
      if (m.hasAdd == 1u) { v += Dt[row * N4 + colVec]; }
      C[row * N4 + colVec] = v;
    }
  }
}`;

export const GEMM_V4_F16C_WGSL = `enable f16;
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, hasAdd:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> Dt: array<f32>;
@group(0) @binding(5) var<uniform> m: Meta;
const BM = 128u; const BN = 128u; const BK = 8u; const TM = 8u; const TN = 8u;
var<workgroup> As: array<f16, 1024>; // TRANSPOSED [BK][BM]
var<workgroup> Bs: array<f16, 1024>; // [BK][BN]
${WGSL_ACTF}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let blockRow = wg.y * BM; let blockCol = wg.x * BN;
  let threadRow = (tid / (BN / TN)) * TM;
  let threadCol = (tid % (BN / TN)) * TN;
  let K4 = m.K / 4u; let N4 = m.N / 4u;
  let aRow = tid / 2u; let aC = tid % 2u;    // A [128][8] = 256 vec4
  let bRow = tid / 32u; let bC = tid % 32u;  // B [8][128] = 256 vec4
  var acc: array<vec4<f32>, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = vec4<f32>(0.0); }
  let nT = (m.K + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    {
      var av = vec4<f32>(0.0);
      if (blockRow + aRow < m.M && kk + aC * 4u < m.K) { av = A[(blockRow + aRow) * K4 + kk / 4u + aC]; }
      let ah = vec4<f16>(av);
      let aBase = 4u * aC;
      As[(aBase + 0u) * BM + aRow] = ah.x; As[(aBase + 1u) * BM + aRow] = ah.y;
      As[(aBase + 2u) * BM + aRow] = ah.z; As[(aBase + 3u) * BM + aRow] = ah.w;
    }
    {
      var bv = vec4<f16>(0.0);
      if (kk + bRow < m.K && blockCol + bC * 4u < m.N) { bv = B[(kk + bRow) * N4 + blockCol / 4u + bC]; }
      let bBase = bRow * BN + 4u * bC;
      Bs[bBase] = bv.x; Bs[bBase + 1u] = bv.y; Bs[bBase + 2u] = bv.z; Bs[bBase + 3u] = bv.w;
    }
    workgroupBarrier();
    var acch: array<vec4<f16>, 16>;
    for (var i = 0u; i < 16u; i++) { acch[i] = vec4<f16>(0.0); }
    for (var k = 0u; k < BK; k++) {
      let ab = k * BM + threadRow;
      var aReg: array<f16, 8>;
      for (var i = 0u; i < 8u; i++) { aReg[i] = As[ab + i]; }
      let bb = k * BN + threadCol;
      let b0 = vec4<f16>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
      let b1 = vec4<f16>(Bs[bb + 4u], Bs[bb + 5u], Bs[bb + 6u], Bs[bb + 7u]);
      for (var i = 0u; i < 8u; i++) { acch[i * 2u] += aReg[i] * b0; acch[i * 2u + 1u] += aReg[i] * b1; }
    }
    for (var i = 0u; i < 16u; i++) { acc[i] += vec4<f32>(acch[i]); }
    workgroupBarrier();
  }
  for (var i = 0u; i < TM; i++) {
    let r = blockRow + threadRow + i; if (r >= m.M) { continue; }
    for (var jb = 0u; jb < 2u; jb++) {
      let v = acc[i * 2u + jb];
      for (var jj = 0u; jj < 4u; jj++) {
        let c = blockCol + threadCol + jb * 4u + jj;
        if (c < m.N) {
          var x = v[jj];
          if (m.hasBias == 1u) { x += bias[c]; }
          x = actf(x, m.act);
          if (m.hasAdd == 1u) { x += Dt[r * m.N + c]; }
          C[r * m.N + c] = x;
        }
      }
    }
  }
}`;

export const GEMM_V4_F16B_WGSL = `enable f16;
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM = 128u; const BN = 128u; const BK = 8u; const TM = 8u; const TN = 8u;
var<workgroup> As: array<f32, 1024>; // TRANSPOSED [BK][BM]
var<workgroup> Bs: array<f32, 1024>; // [BK][BN]
${WGSL_ACTF}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let blockRow = wg.y * BM; let blockCol = wg.x * BN;
  let threadRow = (tid / (BN / TN)) * TM;
  let threadCol = (tid % (BN / TN)) * TN;
  let K4 = m.K / 4u; let N4 = m.N / 4u;
  let aRow = tid / 2u; let aC = tid % 2u;    // A [128][8] = 256 vec4
  let bRow = tid / 32u; let bC = tid % 32u;  // B [8][128] = 256 vec4
  var acc: array<vec4<f32>, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = vec4<f32>(0.0); }
  let nT = (m.K + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    {
      var av = vec4<f32>(0.0);
      if (blockRow + aRow < m.M && kk + aC * 4u < m.K) { av = A[(blockRow + aRow) * K4 + kk / 4u + aC]; }
      let aBase = 4u * aC;
      As[(aBase + 0u) * BM + aRow] = av.x; As[(aBase + 1u) * BM + aRow] = av.y;
      As[(aBase + 2u) * BM + aRow] = av.z; As[(aBase + 3u) * BM + aRow] = av.w;
    }
    {
      var bv = vec4<f32>(0.0);
      if (kk + bRow < m.K && blockCol + bC * 4u < m.N) { bv = vec4<f32>(B[(kk + bRow) * N4 + blockCol / 4u + bC]); }
      let bBase = bRow * BN + 4u * bC;
      Bs[bBase] = bv.x; Bs[bBase + 1u] = bv.y; Bs[bBase + 2u] = bv.z; Bs[bBase + 3u] = bv.w;
    }
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
        if (c < m.N) {
          var x = v[jj];
          if (m.hasBias == 1u) { x += bias[c]; }
          C[r * m.N + c] = actf(x, m.act);
        }
      }
    }
  }
}`;

// ── f16-storage variants ─────────────────────────────────────────────────────
// Same register-blocking as the f32 kernels but with f16 GLOBAL buffers (half the
// memory traffic + Apple's 2× f16 ALU): f16 in/out, f16 multiply, f32 accumulate.
// Measured ~1.3–1.5× over f32, parity vs f32 ≈ rel 3e-4 (fine for TTS).
export const GEMM_F16_WGSL = `enable f16;
struct Meta { M:u32, N:u32, K:u32, act:u32, hasBias:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> A: array<f16>;
@group(0) @binding(1) var<storage, read> B: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> C: array<f16>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM=64u; const BN=64u; const BK=16u; const TM=4u; const TN=4u;
var<workgroup> As: array<f16,1024>; var<workgroup> Bs: array<f16,1024>;
${WGSL_ACTF}
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

// int8 GEMM with in-shader dequant: A[M,K] fp32 @ dequant(Wq)[K,N] -> Y[M,N].
// Wq = int8 weights packed 4-per-u32, row-major [k*N+n]; scale[N] per output column
// (symmetric: w = q * scale[n]). Reads weights at 1/4 the bandwidth of fp32 and keeps
// them 1/4 the GPU memory. Fused bias/act. out[m,n] = act(scale[n]*Σ a[m,k]*q[k,n] + b[n]).
export const MATMUL_INT8_WGSL = `
struct Meta { M:u32, N:u32, K:u32, hasBias:u32, act:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> Wq: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> m: Meta;
${WGSL_ACTF}
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
  acc = actf(acc, m.act);
  Y[idx] = acc;
}`;

// Tiled int8 GEMM (v2): the register-blocked 64×64/4×4 structure of GEMM_WGSL with
// the packed int8 B dequanted during the cooperative tile load (4 int8 per u32 →
// 4 consecutive Bs columns per thread; requires N%4==0, guaranteed for the speech
// encoders). Per-column scale + bias + act applied at write-out. ~3× the naive
// 1-thread-per-output int8 kernel on encoder shapes (A-rows reused from shared).
export const MATMUL_INT8_V2_WGSL = `
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
${WGSL_ACTF}
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

// int8 GEMM v3: the fp32-v4 structure (128x128 block, 8x8 micro-tile, 16 vec4
// accumulators) with the packed int8 B dequanted in the tile load. Requires
// K%4==0 and N%4==0; routed for M>=256 (window-batched encoder GEMMs), where
// TRUE-GPU timing (timestamp-query) shows the 64-tile int8 kernel ~25-35%
// behind fp32 v4 — this closes most of that gap.
export const MATMUL_INT8_V3_WGSL = `
struct Meta { M:u32, N:u32, K:u32, hasBias:u32, act:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Wq: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<uniform> m: Meta;
const BM = 128u; const BN = 128u; const BK = 8u; const TM = 8u; const TN = 8u;
var<workgroup> As: array<f32, 1024>; // TRANSPOSED [BK][BM]
var<workgroup> Bs: array<f32, 1024>; // [BK][BN]
${WGSL_ACTF}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let blockRow = wg.y * BM; let blockCol = wg.x * BN;
  let threadRow = (tid / (BN / TN)) * TM;
  let threadCol = (tid % (BN / TN)) * TN;
  let K4 = m.K / 4u;
  let aRow = tid / 2u; let aC = tid % 2u;
  var acc: array<vec4<f32>, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = vec4<f32>(0.0); }
  let nT = (m.K + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    {
      var av = vec4<f32>(0.0);
      if (blockRow + aRow < m.M && kk + aC * 4u < m.K) { av = A[(blockRow + aRow) * K4 + kk / 4u + aC]; }
      let aBase = 4u * aC;
      As[(aBase + 0u) * BM + aRow] = av.x; As[(aBase + 1u) * BM + aRow] = av.y;
      As[(aBase + 2u) * BM + aRow] = av.z; As[(aBase + 3u) * BM + aRow] = av.w;
    }
    {
      let base = tid * 4u;
      let bRow = base / BN;
      let bCol = base % BN;
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
        if (c < m.N) {
          var x = v[jj] * scale[c];
          if (m.hasBias == 1u) { x += bias[c]; }
          Y[r * m.N + c] = actf(x, m.act);
        }
      }
    }
  }
}`;

// int4 block-quantized matmul (ONNX MatMulNBits: bits=4, block_size=32). This is
// the one thing ORT's WebGPU EP CAN'T do — it has no int kernels, so int4 models
// (Nemotron) fall back to WASM. Here we read the packed int4 weights + per-block
// scales + int4 zero-points directly and dequantize in-shader: a *capability*
// unlock (runs on the GPU where ORT can't), not a speed play. Y = A @ dequant(B)ᵀ,
// dequant(n,k) = (q(n,k) - zp(n,block)) * scale(n,block), block = k/32.
// Bq: packed uint8 [N, nblk, 16] (2 int4/byte) as u32; zp: packed int4 [N, zpb] as u32.
export const MATMUL_NBITS_WGSL = `
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
