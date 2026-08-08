// GPU-resident greedy TDT decoder — design adapted from
// narcotic-sh/parakeet.wgsl (MIT, see THIRD_PARTY.md): the ENTIRE
// autoregressive loop runs in one dispatch, one 256-lane workgroup per window,
// tokens read back once. Mirrors rust/parakeet-decoder decode_proj() exactly:
// 2-layer iofc LSTM (candidate-state commit), predproj cache across blank
// frames, OUT = out_b + relu(encProj + predProj) @ out_w[640×8198], TDT
// advance (durations 0..4, MAX_SYMBOLS 10). Gate: token identity vs
// wasmDecodeProj per window (scripts/gpu-decoder-check.mjs).

const HID = 640;
const VOCAB = 8193;
const LOGITS = 8198;
const BLANK = VOCAB - 1;
const MAX_SYMBOLS = 10;
const WG = 256;
const MAX_OUT_PER_WINDOW = 4096; // tokens+frames capacity per window

// Single consolidated weights buffer (device limit: 8 storage buffers/stage).
// OFFSETS is interpolated at load from the manifest lengths.
const wgslFor = (o) => `
struct Meta { T:u32, W:u32, maxOut:u32, _p:u32 };
@group(0) @binding(0) var<storage, read> frames: array<f32>;   // [W*T, 640] pre-projected
@group(0) @binding(1) var<storage, read> wts: array<f32>;      // all decoder weights
@group(0) @binding(2) var<storage, read_write> outIds: array<i32>;    // [W, maxOut]
@group(0) @binding(3) var<storage, read_write> outFrames: array<i32>; // [W, maxOut]
@group(0) @binding(4) var<storage, read_write> outCounts: array<i32>; // [W]
@group(0) @binding(5) var<uniform> m: Meta;
const EMB = ${o.embed}u;
const LW = ${o.lw}u;
const LR = ${o.lr}u;
const LB = ${o.lb}u;
const PW = ${o.predW}u;
const PB = ${o.predB}u;
const OW = ${o.outW}u;
const OB = ${o.outB}u;

const HID = 640u;
const VOCAB = 8193u;
const LOGITS = 8198u;
const BLANK = 8192u;
const WG = 256u;
const OUTPER = 33u;      // ceil(8198/256) logit columns per lane

var<workgroup> h0: array<f32, 640>;
var<workgroup> c0: array<f32, 640>;
var<workgroup> h1: array<f32, 640>;
var<workgroup> c1: array<f32, 640>;
var<workgroup> nh0: array<f32, 640>;
var<workgroup> nc0: array<f32, 640>;
var<workgroup> nh1: array<f32, 640>;
var<workgroup> nc1: array<f32, 640>;
var<workgroup> predProj: array<f32, 640>;
var<workgroup> jvec: array<f32, 640>;
var<workgroup> redV: array<f32, 256>;   // argmax value reduction
var<workgroup> redI: array<u32, 256>;   // argmax index reduction
var<workgroup> durRedV: array<f32, 256>;
var<workgroup> durRedI: array<u32, 256>;
fn sigm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

// Full predictor pass: LSTM l0 (x=embed[tok]) → nh0/nc0, LSTM l1 (x=nh0) →
// nh1/nc1, predProj = predB + nh1 @ predW. Candidate states nh*/nc* are NOT
// committed here (mirrors rust predict()).
fn predictAndProj(tok: u32, li: u32) {
  // layer 0: each lane computes gates for hidden slots g = li, li+WG, ...
  for (var g = li; g < HID; g += WG) {
    var zi = wts[LB + g] + wts[LB + 4u * HID + g];
    var zo = wts[LB + HID + g] + wts[LB + 5u * HID + g];
    var zf = wts[LB + 2u * HID + g] + wts[LB + 6u * HID + g];
    var zc = wts[LB + 3u * HID + g] + wts[LB + 7u * HID + g];
    let ri = g * HID; let ro = (HID + g) * HID; let rf = (2u * HID + g) * HID; let rc = (3u * HID + g) * HID;
    for (var k = 0u; k < HID; k++) {
      let x = wts[EMB + tok * HID + k];
      let hp = h0[k];
      zi += wts[LW + ri + k] * x + wts[LR + ri + k] * hp;
      zo += wts[LW + ro + k] * x + wts[LR + ro + k] * hp;
      zf += wts[LW + rf + k] * x + wts[LR + rf + k] * hp;
      zc += wts[LW + rc + k] * x + wts[LR + rc + k] * hp;
    }
    let cc = sigm(zf) * c0[g] + sigm(zi) * tanh(clamp(zc, -20.0, 20.0));
    nc0[g] = cc;
    nh0[g] = sigm(zo) * tanh(clamp(cc, -20.0, 20.0));
  }
  workgroupBarrier();
  // layer 1: x = nh0
  let L = 2560u * HID;
  for (var g = li; g < HID; g += WG) {
    var zi = wts[LB + 5120u + g] + wts[LB + 5120u + 4u * HID + g];
    var zo = wts[LB + 5120u + HID + g] + wts[LB + 5120u + 5u * HID + g];
    var zf = wts[LB + 5120u + 2u * HID + g] + wts[LB + 5120u + 6u * HID + g];
    var zc = wts[LB + 5120u + 3u * HID + g] + wts[LB + 5120u + 7u * HID + g];
    let ri = L + g * HID; let ro = L + (HID + g) * HID; let rf = L + (2u * HID + g) * HID; let rc = L + (3u * HID + g) * HID;
    for (var k = 0u; k < HID; k++) {
      let x = nh0[k];
      let hp = h1[k];
      zi += wts[LW + ri + k] * x + wts[LR + ri + k] * hp;
      zo += wts[LW + ro + k] * x + wts[LR + ro + k] * hp;
      zf += wts[LW + rf + k] * x + wts[LR + rf + k] * hp;
      zc += wts[LW + rc + k] * x + wts[LR + rc + k] * hp;
    }
    let cc = sigm(zf) * c1[g] + sigm(zi) * tanh(clamp(zc, -20.0, 20.0));
    nc1[g] = cc;
    nh1[g] = sigm(zo) * tanh(clamp(cc, -20.0, 20.0));
  }
  workgroupBarrier();
  // predProj[n] = predB[n] + Σ_k nh1[k] * predW[k*640+n] — lanes own n slots
  for (var n = li; n < HID; n += WG) {
    var acc = wts[PB + n];
    for (var k = 0u; k < HID; k++) {
      acc += nh1[k] * wts[PW + k * HID + n];
    }
    predProj[n] = acc;
  }
  workgroupBarrier();
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let win = wg.x;
  let T = m.T;
  let fBase = win * T * HID;

  // init state
  for (var i = li; i < HID; i += WG) {
    h0[i] = 0.0; c0[i] = 0.0; h1[i] = 0.0; c1[i] = 0.0;
  }
  workgroupBarrier();

  var lastTok = BLANK;
  var nOut = 0u;
  var emitted = 0u;
  var t = 0u;

  // predict(BLANK) + predproj — inline: full predictor pass
  predictAndProj(lastTok, li);

  loop {
    if (t >= T) { break; }
    // j = relu(frames[t] + predProj)
    for (var i = li; i < HID; i += WG) {
      let v = frames[fBase + t * HID + i] + predProj[i];
      jvec[i] = max(v, 0.0);
    }
    workgroupBarrier();

    // OUT columns per lane + local argmax (token range and duration range)
    var bestV = -3.0e38;
    var bestI = 0u;
    var durV = -3.0e38;
    var durI = 0u;
    for (var ci = 0u; ci < OUTPER; ci++) {
      let mcol = li + ci * WG;
      if (mcol >= LOGITS) { break; }
      var o = wts[OB + mcol];
      for (var n = 0u; n < HID; n++) {
        o += jvec[n] * wts[OW + n * LOGITS + mcol];
      }
      if (mcol < VOCAB) {
        if (o > bestV) { bestV = o; bestI = mcol; }
      } else {
        if (o > durV) { durV = o; durI = mcol - VOCAB; }
      }
    }
    redV[li] = bestV; redI[li] = bestI;
    durRedV[li] = durV; durRedI[li] = durI;
    workgroupBarrier();
    // tree-reduce argmax (ties → lower index, matching the sequential scan)
    var stride = WG / 2u;
    loop {
      if (stride == 0u) { break; }
      if (li < stride) {
        let a = redV[li]; let b = redV[li + stride];
        if (b > a || (b == a && redI[li + stride] < redI[li])) { redV[li] = b; redI[li] = redI[li + stride]; }
        let da = durRedV[li]; let db = durRedV[li + stride];
        if (db > da || (db == da && durRedI[li + stride] < durRedI[li])) { durRedV[li] = db; durRedI[li] = durRedI[li + stride]; }
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    // workgroupUniformLoad: the compiler must PROVE these are uniform for the
    // barriers inside the emission branch to be legal.
    let maxId = workgroupUniformLoad(&redI)[0];
    let step = workgroupUniformLoad(&durRedI)[0];

    if (maxId != BLANK) {
      // commit candidate state, record token, predict(next)
      for (var i = li; i < HID; i += WG) {
        h0[i] = nh0[i]; c0[i] = nc0[i]; h1[i] = nh1[i]; c1[i] = nc1[i];
      }
      workgroupBarrier();
      if (li == 0u && nOut < m.maxOut) {
        outIds[win * m.maxOut + nOut] = i32(maxId);
        outFrames[win * m.maxOut + nOut] = i32(t);
      }
      nOut += 1u;
      emitted += 1u;
      lastTok = maxId;
      predictAndProj(lastTok, li);
    }
    if (step > 0u) { t += step; emitted = 0u; }
    else if (maxId == BLANK || emitted >= ${MAX_SYMBOLS}u) { t += 1u; emitted = 0u; }
  }
  if (li == 0u) { outCounts[win] = i32(min(nOut, m.maxOut)); }
}

`;

/** Upload decoder weights once (single consolidated buffer + baked offsets). */
export function loadGpuDecoder(ctx, decF32, decMan) {
  const g = (k) => decF32.subarray(decMan[k].offset, decMan[k].offset + decMan[k].len);
  const parts = [
    ["embed", g("embed")],
    ["lw", g("l0_W"), g("l1_W")],
    ["lr", g("l0_R"), g("l1_R")],
    ["lb", g("l0_B"), g("l1_B")],
    ["predW", g("predW")],
    ["predB", g("predB")],
    ["outW", g("outW")],
    ["outB", g("outB")],
  ];
  const offsets = {};
  let total = 0;
  for (const [name, ...arrs] of parts) {
    offsets[name] = total;
    for (const a of arrs) total += a.length;
  }
  const all = new Float32Array(total);
  let off = 0;
  for (const [, ...arrs] of parts)
    for (const a of arrs) {
      all.set(a, off);
      off += a.length;
    }
  const buf = ctx.device.createBuffer({ size: all.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  ctx.device.queue.writeBuffer(buf, 0, all);
  return { wts: buf, wgsl: wgslFor(offsets) };
}

/** Decode W windows of T pre-projected frames each, entirely on the GPU.
 * framesGpu: [W*T, 640]. Returns per-window { ids, idFrames }. */
export async function gpuDecodeBatch(ctx, gdec, framesGpu, W, T) {
  const dev = ctx.device;
  const maxOut = MAX_OUT_PER_WINDOW;
  const mk = (n) => dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const idsB = mk(W * maxOut);
  const frB = mk(W * maxOut);
  const cntB = mk(W);
  const pipeline = ctx._pipeline("tdtGpuDecoder", gdec.wgsl);
  const u = ctx._uniform(new Uint32Array([T, W, maxOut, 0]));
  ctx._run(pipeline, [framesGpu.buf, gdec.wts, idsB, frB, cntB], u, W);
  const rd = async (buf, n) => {
    const stg = dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, stg, 0, n * 4);
    dev.queue.submit([enc.finish()]);
    await stg.mapAsync(GPUMapMode.READ);
    const out = new Int32Array(stg.getMappedRange().slice(0));
    stg.unmap();
    stg.destroy();
    return out;
  };
  const [ids, frames, counts] = await Promise.all([rd(idsB, W * maxOut), rd(frB, W * maxOut), rd(cntB, W)]);
  idsB.destroy();
  frB.destroy();
  cntB.destroy();
  const perWindow = [];
  for (let w = 0; w < W; w++) {
    const n = counts[w];
    perWindow.push({
      ids: Array.from(ids.subarray(w * maxOut, w * maxOut + n)),
      idFrames: Array.from(frames.subarray(w * maxOut, w * maxOut + n)),
    });
  }
  return perWindow;
}
