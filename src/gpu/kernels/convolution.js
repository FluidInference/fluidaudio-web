// 1-D / 2-D convolution kernels: direct, depthwise, transposed, im2col and
// implicit-GEMM forms, plus the FastConformer subsampling reshape.

import { WGSL_ACTF } from "./actf.js";

// 1-D convolution (batch 1). X:[Cin, L], W:[Cout, Cin/groups, K], bias?:[Cout]
// -> Y:[Cout, Lout], with stride/pad/dilation/groups. One thread per (Cout, Lout).
// Covers regular (groups=1) and depthwise (groups=Cin) convs. act as in GEMM.
export const CONV1D_WGSL = `
struct Meta { Cout:u32, Cin:u32, L:u32, Lout:u32, K:u32, stride:u32, pad:u32, dil:u32,
              groups:u32, hasBias:u32, act:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;

${WGSL_ACTF}
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
  acc = actf(acc, m.act);
  Y[co * m.Lout + lo] = acc;
}`;

// FastConformer subsampling reshape: conv output [C, Tsub*F] (rows=C) -> [Tsub, C*F]
// with out[ho, c*F+wo] = in[c, ho*F+wo]. Keeps it GPU-resident (was a download +
// host rearrange + upload per window).
export const SUBRESHAPE_WGSL = `
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

// 2-D convolution (batch 1) — FastConformer dw-striding subsampling. X:[Cin,H*W]
// (rows=Cin), W:[Cout,Cin/groups,Kh,Kw] flat, bias?:[Cout] -> Y:[Cout,Ho*Wo]. One
// thread per (Cout, Ho*Wo). Supports groups (depthwise) + fused bias/act.
// Depthwise 3×3 stride-2: one thread computes 4 consecutive outputs of one
// channel row with the 9 weights held in registers. act: 3=relu, 4=silu.
export const CONV2D_DW3X3S2_WGSL = `
struct Meta { C:u32, H:u32, W:u32, Ho:u32, Wo:u32, padH:u32, padW:u32, hasBias:u32, act:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
${WGSL_ACTF}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let WoQ = (m.Wo + 3u) / 4u;
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  if (idx >= m.C * m.Ho * WoQ) { return; }
  let c = idx / (m.Ho * WoQ);
  let ho = (idx / WoQ) % m.Ho;
  let wo0 = (idx % WoQ) * 4u;
  var wgt: array<f32, 9>;
  for (var i = 0u; i < 9u; i++) { wgt[i] = Wt[c * 9u + i]; }
  let xC = c * m.H * m.W;
  var b = 0.0;
  if (m.hasBias == 1u) { b = bias[c]; }
  for (var j = 0u; j < 4u; j++) {
    let wo = wo0 + j;
    if (wo >= m.Wo) { break; }
    var acc = 0.0;
    for (var kh = 0u; kh < 3u; kh++) {
      let hi = i32(ho * 2u + kh) - i32(m.padH);
      if (hi < 0 || hi >= i32(m.H)) { continue; }
      let rowB = xC + u32(hi) * m.W;
      for (var kw = 0u; kw < 3u; kw++) {
        let wi = i32(wo * 2u + kw) - i32(m.padW);
        if (wi >= 0 && wi < i32(m.W)) { acc += X[rowB + u32(wi)] * wgt[kh * 3u + kw]; }
      }
    }
    acc += b;
    acc = actf(acc, m.act);
    Y[c * m.Ho * m.Wo + ho * m.Wo + wo] = acc;
  }
}`;

// cin=1 3×3 stride-2 (the mel-input conv): every output channel reads the SAME
// 3×3 window, so one thread loads it once and computes 4 channels.
export const CONV2D_C1_3X3S2_WGSL = `
struct Meta { Cout:u32, H:u32, W:u32, Ho:u32, Wo:u32, padH:u32, padW:u32, hasBias:u32, act:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
${WGSL_ACTF}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let HW = m.Ho * m.Wo;
  let CQ = m.Cout / 4u;
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  if (idx >= CQ * HW) { return; }
  let co0 = (idx / HW) * 4u;
  let ho = (idx % HW) / m.Wo;
  let wo = (idx % HW) % m.Wo;
  var xv: array<f32, 9>;
  for (var kh = 0u; kh < 3u; kh++) {
    let hi = i32(ho * 2u + kh) - i32(m.padH);
    for (var kw = 0u; kw < 3u; kw++) {
      let wi = i32(wo * 2u + kw) - i32(m.padW);
      var v = 0.0;
      if (hi >= 0 && hi < i32(m.H) && wi >= 0 && wi < i32(m.W)) { v = X[u32(hi) * m.W + u32(wi)]; }
      xv[kh * 3u + kw] = v;
    }
  }
  for (var jc = 0u; jc < 4u; jc++) {
    let co = co0 + jc;
    var acc = 0.0;
    for (var i = 0u; i < 9u; i++) { acc += xv[i] * Wt[co * 9u + i]; }
    if (m.hasBias == 1u) { acc += bias[co]; }
    acc = actf(acc, m.act);
    Y[co * HW + ho * m.Wo + wo] = acc;
  }
}`;

export const CONV2D_WGSL = `
struct Meta { Cout:u32, Cin:u32, H:u32, W:u32, Ho:u32, Wo:u32, Kh:u32, Kw:u32,
              sH:u32, sW:u32, padH:u32, padW:u32, groups:u32, hasBias:u32, act:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
${WGSL_ACTF}
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
  acc = actf(acc, m.act);
  Y[co * HW + ho * m.Wo + wo] = acc;
}`;

// 1-D transposed convolution (batch 1) — the iSTFTNet upsampler and iSTFT
// overlap-add. X:[Cin,L], W:[Cin, Cout/groups, K], bias?:[Cout] -> Y:[Cout,Lout],
// Lout = (L-1)*stride - 2*pad + dilation*(K-1) + output_padding + 1. Gather form:
// one thread per (Cout, Lout), pulling the input positions that map onto it.
export const CONVT1D_WGSL = `
struct Meta { Cout:u32, Cin:u32, L:u32, Lout:u32, K:u32, stride:u32, pad:u32, dil:u32,
              groups:u32, hasBias:u32, act:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> m: Meta;
${WGSL_ACTF}
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
  acc = actf(acc, m.act);
  Y[co * m.Lout + lo] = acc;
}`;

// ConvTranspose1d with k == stride (no output-position overlap, e.g. the
// voicechat codec upsamplers) is a plain GEMM Wt[cout*k, Cin] @ x[Cin, L]
// followed by this interleave: Y[co, t*k + kk] = cols[(co*k + kk)*L + t]
// (+ bias/act). The direct gather kernel walks Cin*K with divmod checks per
// output -- measured 1157 ms vs ~15 ms for GEMM+reshape on the codec stages.
export const CONVT_RESHAPE_WGSL = `
struct Meta { Cout:u32, L:u32, K:u32, hasBias:u32, act:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> cols: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> m: Meta;
${WGSL_ACTF}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  let Lout = m.L * m.K;
  if (idx >= m.Cout * Lout) { return; }
  let co = idx / Lout;
  let j = idx % Lout;
  let t = j / m.K;
  let kk = j % m.K;
  var v = cols[(co * m.K + kk) * m.L + t];
  if (m.hasBias == 1u) { v += bias[co]; }
  Y[idx] = actf(v, m.act);
}`;

// One-time weight permute for the GEMM route: W flat [Cin, Cout, K] ->
// Wt [(co*K + kk), ci] row-major (matmul A operand).
export const CONVT_WPERM_WGSL = `
struct Meta { Cin:u32, Cout:u32, K:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read_write> Wt: array<f32>;
@group(0) @binding(2) var<uniform> m: Meta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.y * (nwg.x * 64u) + gid.x;
  if (idx >= m.Cin * m.Cout * m.K) { return; }
  let ci = idx / (m.Cout * m.K);
  let rest = idx % (m.Cout * m.K);
  Wt[rest * m.Cin + ci] = W[idx];
}`;

// im2col for conv1d: X[Cin,L] -> Cols[Cin*K, Lout], so a conv becomes a single
// GEMM  W[Cout, Cin*K] @ Cols  — hitting tiled-GEMM throughput instead of the
// direct kernel's memory-bound rate. Row (ci*K+k), col lo.
export const IM2COL_WGSL = `
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
export const CONV1D_IMPLICIT_WGSL = `
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
${WGSL_ACTF}
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

// f16 fused conv1d (implicit GEMM, groups=1). W/X/bias/Y all f16.
export const CONV1D_IMPLICIT_F16_WGSL = `enable f16;
struct Meta { Cout:u32, Lout:u32, CinK:u32, Cin:u32, L:u32, K:u32, stride:u32, pad:u32,
              dil:u32, hasBias:u32, act:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> W: array<f16>;
@group(0) @binding(1) var<storage, read> X: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f16>;
@group(0) @binding(4) var<uniform> m: Meta;
const BM=64u; const BN=64u; const BK=16u; const TM=4u; const TN=4u;
var<workgroup> As: array<f16,1024>; var<workgroup> Bs: array<f16,1024>;
${WGSL_ACTF}
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
