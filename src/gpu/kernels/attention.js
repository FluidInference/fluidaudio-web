// Batched multi-head attention kernels (QK^T, PV, fused rel-pos attention)
// and the rel_shift family for relative-position scores.

// ── Batched multi-head attention kernels ────────────────────────────────────
// The per-head attention loop is ~80 dispatches/layer (slice/transpose/matmul/
// softmax per head) and the encoders are LAUNCH-BOUND. These batch all heads into
// single dispatches: head-strided reads over the [T, H*HD] projections, so no
// slicing/transposing/concat dispatches at all (~88 → ~11 dispatches/layer).

// scores[(w*H+h)*T+i, j] = Σ_d (Q[w*T+i, h*HD+d] + qb[h*HD+d]) · B[·, h*HD+d]
// (B = keys → AC term, or projected pos-emb → BD term; qb = pos_bias_u/v.)
// TILED: 64×64 output tile per z-block (w*H+h), register-blocked 4×4 like GEMM —
// the naive 1-thread-per-output version regressed the browser bench ~10%.
export const BMM_QK_WGSL = `
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
        let brow = select(w * m.Tb + j, j, m.bShared == 1u); // per-window keys stride Tb
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
export const BMM_PV_WGSL = `
struct Meta { Tq:u32, Tk:u32, H:u32, HD:u32, W:u32, _a:u32, _b:u32, _c:u32 };
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
  let blockRow = wg.y * BM;   // over Tq (queries)
  let blockCol = wg.x * BN;   // over HD (head cols; HD<=64 → one x-block)
  let threadRow = (tid / (BN / TN)) * TM;
  let threadCol = (tid % (BN / TN)) * TN;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  let nT = (m.Tk + BK - 1u) / BK;
  for (var t = 0u; t < nT; t++) {
    let kk = t * BK;
    for (var i = 0u; i < 4u; i++) {
      let idxA = tid + i * 256u;
      let aRow = idxA / BK; let aCol = idxA % BK;
      As[idxA] = select(0.0, P[(b * m.Tq + blockRow + aRow) * m.Tk + kk + aCol], blockRow + aRow < m.Tq && kk + aCol < m.Tk);
      let bRowT = idxA / BN; let bColT = idxA % BN;
      var bv = 0.0;
      if (kk + bRowT < m.Tk && blockCol + bColT < m.HD) {
        bv = V[(w * m.Tk + kk + bRowT) * stride + h * m.HD + blockCol + bColT];
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
      if (r < m.Tq && c < m.HD) {
        Y[(w * m.Tq + r) * stride + h * m.HD + c] = acc[i * TN + j2];
      }
    }
  }
}`;

// Batched rel_shift: X [H*t, 2t-1] → Y [H*t, t], each head block shifted independently.

// Fused rel-pos self-attention (full/unmasked, T ≤ 256): QKᵀ + both rel-pos
// terms + softmax + PV in ONE dispatch — the [W·H·T, T] score tensor never
// touches global memory (it was the #1 blocker for larger window batches:
// wb=40 would materialize ~4.5GB of transients on the multi-pass path).
// Workgroup = 64 lanes on an 8-query block of one (window, head): scores live
// in LDS; each K row and each pos-projection row is read exactly once per
// workgroup; V is consumed in 16-row LDS tiles.
export function genAttnFusedWgsl(QB, HD) {
  // Workgroup = HD lanes (64 or 128). LDS: qu/qv 2·QB·HD·4 + sc QB·256·4 +
  // vt VROWS·HD·4 bytes; VROWS shrinks at HD=128 to stay ≤ 12KB (16KB was an
  // occupancy cliff). NOTE the original QB=8/HD≤64 guard meant this kernel
  // NEVER ran on Parakeet (HD=128) — every earlier in-context "fused" number
  // was actually the multi-pass fallback.
  const WGS = HD;
  const VROWS = HD === 128 ? 8 : 16;
  return `enable f16;
struct Meta { T:u32, H:u32, HD:u32, W:u32, P:u32, QB:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> POS: array<f32>;
@group(0) @binding(4) var<storage, read> PBU: array<f32>;
@group(0) @binding(5) var<storage, read> PBV: array<f32>;
@group(0) @binding(6) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(7) var<uniform> m: Meta;
const MAXT = 256u;
const QBLK = ${QB}u;
const HDC = ${HD}u;
const VROWS = ${VROWS}u;
var<workgroup> qu: array<f32, ${QB * HD}>;
var<workgroup> qv: array<f32, ${QB * HD}>;
var<workgroup> sc: array<f32, ${QB * 256}>;
var<workgroup> vt: array<f32, ${VROWS * HD}>;
@compute @workgroup_size(${WGS})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = m.T; let H = m.H; let stride = H * HDC;
  let q0 = wg.x * QBLK;
  let h = wg.y;
  let w = wg.z;
  let ho = h * HDC;
  let base = w * T;
  for (var e = li; e < QBLK * HDC; e += ${WGS}u) {
    let i = e / HDC; let d = e % HDC;
    var qval = 0.0;
    if (q0 + i < T) { qval = Q[(base + q0 + i) * stride + ho + d]; }
    qu[e] = qval + PBU[ho + d];
    qv[e] = qval + PBV[ho + d];
  }
  for (var e = li; e < QBLK * MAXT; e += ${WGS}u) { sc[e] = 0.0; }
  workgroupBarrier();
  for (var j = li; j < T; j += ${WGS}u) {
    let kb = (base + j) * stride + ho;
    for (var i = 0u; i < QBLK; i++) {
      var acc = 0.0;
      for (var d = 0u; d < HDC; d++) { acc += qu[i * HDC + d] * K[kb + d]; }
      sc[i * MAXT + j] = acc;
    }
  }
  workgroupBarrier();
  for (var r = li; r < T + QBLK - 1u; r += ${WGS}u) {
    for (var i = 0u; i < QBLK; i++) {
      let rg = i32(r) + i32(T) - 1i - i32(q0) - i32(QBLK) + 1i;
      let j = rg - (i32(T) - 1i) + i32(q0) + i32(i);
      if (j >= 0i && j < i32(T) && rg >= 0i && rg < i32(m.P)) {
        var acc = 0.0;
        let pg = u32(rg) * stride + ho;
        for (var d = 0u; d < HDC; d++) { acc += qv[i * HDC + d] * POS[pg + d]; }
        sc[i * MAXT + u32(j)] += acc;
      }
    }
  }
  workgroupBarrier();
  if (li < QBLK) {
    let i = li;
    var mx = -3.0e38;
    for (var j = 0u; j < T; j++) { mx = max(mx, sc[i * MAXT + j]); }
    var sum = 0.0;
    for (var j = 0u; j < T; j++) { let e = exp(sc[i * MAXT + j] - mx); sc[i * MAXT + j] = e; sum += e; }
    let inv = 1.0 / sum;
    for (var j = 0u; j < T; j++) { sc[i * MAXT + j] *= inv; }
  }
  workgroupBarrier();
  var acc: array<f32, ${QB}>;
  for (var i = 0u; i < QBLK; i++) { acc[i] = 0.0; }
  var j0 = 0u;
  loop {
    if (j0 >= T) { break; }
    let rows = min(VROWS, T - j0);
    for (var e = li; e < rows * HDC; e += ${WGS}u) {
      let jj = e / HDC; let d = e % HDC;
      vt[jj * HDC + d] = V[(base + j0 + jj) * stride + ho + d];
    }
    workgroupBarrier();
    for (var jj = 0u; jj < rows; jj++) {
      let vv = vt[jj * HDC + li];
      for (var i = 0u; i < QBLK; i++) { acc[i] += sc[i * MAXT + j0 + jj] * vv; }
    }
    workgroupBarrier();
    j0 += VROWS;
  }
  for (var i = 0u; i < QBLK; i++) {
    if (q0 + i < T) { OUT[(base + q0 + i) * stride + ho + li] = acc[i]; }
  }
}`;
}

export const RELSHIFT_STREAM_WGSL = `
struct Meta { H:i32, n:i32, Lk:i32, P:i32, dMax:i32, Lc:i32, subT:i32, C:i32, left:i32, right:i32, _a:i32, _b:i32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = i32(gid.y * (nwg.x * 64u) + gid.x);
  if (idx >= m.H * m.n * m.Lk) { return; }
  let j = idx % m.Lk;
  let r = idx / m.Lk;
  let i = r % m.n;
  let q = m.subT + i;
  let k = m.subT - m.Lc + j;
  let cs = m.C * (q / m.C);
  if (k < cs - m.left || k > cs + m.C - 1 + m.right) { Y[idx] = -10000.0; return; }
  let pi = clamp(m.dMax - (q - k), 0, m.P - 1);
  Y[idx] = X[r * m.P + pi];
}`;

export const RELSHIFT_B_WGSL = `
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

// rel_shift for relative-position attention: X = matrix_bd [t, 2t-1] -> Y [t, t].
// Closed form of NeMo's pad→reshape→slice: Y[i,j] = xp[f], f = t + i*(2t-1) + j,
// where xp is the left-padded [t,2t] view (col 0 = 0). Avoids the GPU→CPU roundtrip.
export const RELSHIFT_WGSL = `
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
