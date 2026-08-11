// Decoder-side kernels: TDT joint/argmax family and the bidirectional LSTM.

// TDT joint, split for speed: (1) jBatch builds j = relu(encProj[base+i] + predProj)
// for B frames [B,hid]; (2) the fast TILED matmul does [B,hid]@[hid,logits]; (3)
// argmaxRows reduces each row to token/dur argmax. Between emissions predProj is
// constant, so a run of frames is one tiled matmul instead of B tiny 1-workgroup ones.
export const JBATCH_WGSL = `
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

export const ARGMAX_ROWS_WGSL = `
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

// Fused TDT joint + argmax (one dispatch per decode step). Computes
// j = relu(encProj[frame] + predProj) [hidden], out = j @ outW + outB [logits],
// then reduces to token argmax (n<vocab) and duration argmax (n>=vocab). Writes
// [tokenIdx, tokenMax, durIdx, durMax] — 4 floats downloaded per frame instead of
// the full 8198 logits. Kills the JS decoder bottleneck.
export const TDT_JOINT_WGSL = `
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

// Bidirectional LSTM matching the ONNX LSTM op (batch 1). One workgroup per
// direction, one thread per hidden unit, timesteps looped in-kernel (h/c kept in
// workgroup memory). ONNX gate order is **iofc**; no peephole. Weights:
//   W:[2, 4H, I]  R:[2, 4H, H]  B:[2, 8H] (Wb[4H] then Rb[4H]).
// Output Y:[seq, 2H] = [fwd(H) | bwd(H)] per timestep (ONNX [seq,2,1,H] flattened).
// Requires H <= 256.
export const LSTM_WGSL = `
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
