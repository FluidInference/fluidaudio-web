// Numeric GPU↔WASM conformance: runs every ComputeContext op with the SAME
// seeded inputs through BOTH backends and compares outputs directly — no CPU
// references, no fixtures, fully hermetic. Complements interface-conformance
// (structure) with behavior: a kernel whose math drifts on one backend fails
// here even if every engine happens to avoid the drifted path.
//
// Completeness is enforced against the d.ts: every required ComputeContext
// method must have a numeric case here or an explicit exemption below —
// adding an op without a conformance case fails this gate.
//   node scripts/backend-conformance.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { getDevice } from "./gpu-globals.mjs";
import { GpuContext } from "../src/gpu/compute.js";
import { createWasmContext } from "../src/gpu/wasm-context.js";
import { computeInterfaceMethods } from "./lib/compute-interface.mjs";

// Plumbing members with no standalone numeric behavior. Everything here is
// still exercised implicitly: upload/download by every case, the arena/batch
// machinery by the withBatch composite case, uploadBytes by the quantized
// GEMMs, the f16 uploads by the f16 cases.
const EXEMPT = new Set([
  "upload",
  "uploadF16",
  "alloc",
  "allocF16",
  "uploadBytes",
  "freeTensor",
  "pin",
  "pushArena",
  "popArena",
  "trimPool",
  "destroy",
  "download",
  "downloadF16",
  "beginBatch",
  "endBatch",
  "withBatchSync",
]);

const g = new GpuContext(await getDevice());
await g.probeSubgroups();
const w = await createWasmContext(readFileSync(fileURLToPath(new URL("../src/gpu/wasm-kernels.wasm", import.meta.url))));

// mulberry32 — deterministic inputs so failures are reproducible.
let seed = 0x5eed;
const rand = () => {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const mkData = (n) => Float32Array.from({ length: n }, () => rand() - 0.5);
const mkBytes = (n) => Uint8Array.from({ length: n }, () => Math.floor(rand() * 256));
/** Same payload uploaded to both backends: [gpuTensor, wasmTensor]. */
const pair = (rows, cols) => {
  const d = mkData(rows * cols);
  return [g.upload(d, rows, cols), w.upload(d.slice(), rows, cols)];
};

const cases = [];
/** method: the ComputeContext member this case covers (for the completeness check). */
const test = (method, label, tol, fn) => cases.push({ method, label, tol, fn });

// ── matmul family: every activation (the gelu_erf drift class) + fused add ──
for (const act of ["none", "gelu", "gelu_erf", "tanh", "relu", "silu"]) {
  test("matmul", `matmul bias+${act} (odd shape)`, 1e-3, async () => {
    const [ag, aw] = pair(37, 53);
    const [bg, bw] = pair(53, 41);
    const [big, biw] = pair(1, 41);
    return [await g.download(g.matmul(ag, bg, { bias: big, act })), await w.download(w.matmul(aw, bw, { bias: biw, act }))];
  });
}
test("matmul", "matmul vec4 shape + fused add", 1e-3, async () => {
  const [ag, aw] = pair(64, 128);
  const [bg, bw] = pair(128, 256);
  const [dg, dw] = pair(64, 256);
  return [await g.download(g.matmul(ag, bg, { act: "relu", add: dg })), await w.download(w.matmul(aw, bw, { act: "relu", add: dw }))];
});
test("matmulV2", "matmulV2", 1e-3, async () => {
  const [ag, aw] = pair(33, 48);
  const [bg, bw] = pair(48, 52);
  return [await g.download(g.matmulV2(ag, bg, { act: "gelu" })), await w.download(w.matmulV2(aw, bw, { act: "gelu" }))];
});
test("matmulV3", "matmulV3 (K%4==0, N%4==0)", 1e-3, async () => {
  const [ag, aw] = pair(33, 64);
  const [bg, bw] = pair(64, 48);
  return [await g.download(g.matmulV3(ag, bg)), await w.download(w.matmulV3(aw, bw))];
});
test("matmulV4", "matmulV4 (K%4==0, N%4==0)", 1e-3, async () => {
  const [ag, aw] = pair(129, 64);
  const [bg, bw] = pair(64, 132);
  const [big, biw] = pair(1, 132);
  return [await g.download(g.matmulV4(ag, bg, { bias: big, act: "silu" })), await w.download(w.matmulV4(aw, bw, { bias: biw, act: "silu" }))];
});
if (g.hasF16) {
  test("matmulF16", "matmulF16 (f16 a/b/c)", 5e-2, async () => {
    const ad = mkData(40 * 64);
    const bd = mkData(64 * 64);
    return [
      await g.downloadF16(g.matmulF16(g.uploadF16(ad, 40, 64), g.uploadF16(bd, 64, 64))),
      await w.download(w.matmulF16(w.upload(ad.slice(), 40, 64), w.upload(bd.slice(), 64, 64))),
    ];
  });
  test("matmul", "matmul f16-B route", 2e-2, async () => {
    const ad = mkData(65 * 128);
    const bd = mkData(128 * 256);
    const [big, biw] = pair(1, 256);
    return [
      await g.download(g.matmul(g.upload(ad, 65, 128), g.uploadF16(bd, 128, 256), { bias: big, act: "silu" })),
      await w.download(w.matmul(w.upload(ad.slice(), 65, 128), w.upload(bd.slice(), 128, 256), { bias: biw, act: "silu" })),
    ];
  });
} else {
  console.log("(f16 unavailable on this device — matmulF16 / f16-B routes skipped)");
  cases.push({ method: "matmulF16", label: "matmulF16 (skipped: no f16)", skip: true });
}
test("matmulInt8", "matmulInt8 bias+relu", 1e-3, async () => {
  const [ag, aw] = pair(9, 20);
  const packed = mkBytes(20 * 16);
  const [sg, sw] = pair(1, 16);
  const [big, biw] = pair(1, 16);
  return [
    await g.download(g.matmulInt8(ag, g.uploadBytes(packed), sg, 16, 20, { bias: big, act: "relu" })),
    await w.download(w.matmulInt8(aw, w.uploadBytes(packed), sw, 16, 20, { bias: biw, act: "relu" })),
  ];
});
test("matmulNBits", "matmulNBits (int4 blocks)", 1e-3, async () => {
  const N = 8;
  const K = 64;
  const nblk = 2;
  const [ag, aw] = pair(7, K);
  const bq = mkBytes(N * nblk * 16);
  const zp = mkBytes(N * Math.ceil(nblk / 2) * 4); // ×4: keep u32-viewable on GPU
  const [sg, sw] = pair(1, N * nblk);
  return [
    await g.download(g.matmulNBits(ag, g.uploadBytes(bq), sg, g.uploadBytes(zp), N)),
    await w.download(w.matmulNBits(aw, w.uploadBytes(bq), sw, w.uploadBytes(zp), N)),
  ];
});

// ── normalization ────────────────────────────────────────────────────────────
test("layernorm", "layernorm", 1e-4, async () => {
  const [xg, xw] = pair(7, 33);
  const [gg, gw] = pair(1, 33);
  const [bg, bw] = pair(1, 33);
  return [await g.download(g.layernorm(xg, gg, bg)), await w.download(w.layernorm(xw, gw, bw))];
});
test("softmax", "softmax", 1e-5, async () => {
  const [xg, xw] = pair(9, 21);
  return [await g.download(g.softmax(xg)), await w.download(w.softmax(xw))];
});
test("adain", "adain", 1e-4, async () => {
  const [xg, xw] = pair(5, 20);
  const [sg, sw] = pair(1, 5);
  const [hg, hw] = pair(1, 5);
  return [await g.download(g.adain(xg, sg, hg)), await w.download(w.adain(xw, sw, hw))];
});

// ── convolution ──────────────────────────────────────────────────────────────
test("conv1d", "conv1d regular stride2 pad3 silu", 1e-3, async () => {
  const [xg, xw] = pair(6, 25);
  const [wg, ww] = pair(1, 8 * 6 * 5);
  const [big, biw] = pair(1, 8);
  const o = { cout: 8, k: 5, stride: 2, pad: 3, act: "silu" };
  return [await g.download(g.conv1d(xg, wg, { ...o, bias: big })), await w.download(w.conv1d(xw, ww, { ...o, bias: biw }))];
});
test("conv1d", "conv1d depthwise asym-pad", 1e-3, async () => {
  const [xg, xw] = pair(6, 19);
  const [wg, ww] = pair(1, 6 * 1 * 3);
  const o = { cout: 6, k: 3, groups: 6, padLeft: 2, padRight: 0 };
  return [await g.download(g.conv1d(xg, wg, o)), await w.download(w.conv1d(xw, ww, o))];
});
test("im2col", "im2col", 0, async () => {
  const [xg, xw] = pair(4, 15);
  const o = { stride: 2, pad: 1 };
  return [await g.download(g.im2col(xg, 3, o)), await w.download(w.im2col(xw, 3, o))];
});
test("conv1dGemm", "conv1dGemm", 1e-3, async () => {
  const [xg, xw] = pair(4, 15);
  const [wg, ww] = pair(6, 4 * 3);
  const o = { stride: 2, pad: 1, act: "relu" };
  return [await g.download(g.conv1dGemm(xg, wg, 6, 3, o)), await w.download(w.conv1dGemm(xw, ww, 6, 3, o))];
});
test("conv1dFast", "conv1dFast bias+gelu", 1e-3, async () => {
  const [xg, xw] = pair(4, 15);
  const [wg, ww] = pair(6, 4 * 3);
  const [big, biw] = pair(1, 6);
  const o = { stride: 1, pad: 1, act: "gelu" };
  return [await g.download(g.conv1dFast(xg, wg, 6, 3, { ...o, bias: big })), await w.download(w.conv1dFast(xw, ww, 6, 3, { ...o, bias: biw }))];
});
if (g.hasF16) {
  test("conv1dFastF16", "conv1dFastF16", 5e-2, async () => {
    const xd = mkData(4 * 15);
    const wd = mkData(6 * 12);
    const o = { stride: 1, pad: 1, act: "silu" };
    return [
      await g.downloadF16(g.conv1dFastF16(g.uploadF16(xd, 4, 15), g.uploadF16(wd, 6, 12), 6, 3, o)),
      await w.download(w.conv1dFastF16(w.upload(xd.slice(), 4, 15), w.upload(wd.slice(), 6, 12), 6, 3, o)),
    ];
  });
} else {
  cases.push({ method: "conv1dFastF16", label: "conv1dFastF16 (skipped: no f16)", skip: true });
}
test("conv2d", "conv2d regular 3x3 s2 relu", 1e-3, async () => {
  const [xg, xw] = pair(3, 9 * 11);
  const [wg, ww] = pair(1, 4 * 3 * 9);
  const [big, biw] = pair(1, 4);
  const o = { cout: 4, cin: 3, h: 9, w: 11, kh: 3, kw: 3, strideH: 2, strideW: 2, padH: 1, padW: 1, act: "relu" };
  return [await g.download(g.conv2d(xg, wg, { ...o, bias: big })), await w.download(w.conv2d(xw, ww, { ...o, bias: biw }))];
});
test("conv2d", "conv2d depthwise 3x3 s2 (dw route)", 1e-3, async () => {
  const [xg, xw] = pair(6, 9 * 11);
  const [wg, ww] = pair(1, 6 * 9);
  const o = { cout: 6, cin: 6, h: 9, w: 11, kh: 3, kw: 3, strideH: 2, strideW: 2, padH: 1, padW: 1, groups: 6, act: "silu" };
  return [await g.download(g.conv2d(xg, wg, o)), await w.download(w.conv2d(xw, ww, o))];
});
test("conv2d", "conv2d cin=1 3x3 s2 (c1 route) asym pad", 1e-3, async () => {
  const [xg, xw] = pair(1, 12 * 10);
  const [wg, ww] = pair(1, 8 * 9);
  const o = { cout: 8, cin: 1, h: 12, w: 10, kh: 3, kw: 3, strideH: 2, strideW: 2, padTop: 1, padBottom: 0, padLeft: 1, padRight: 0 };
  return [await g.download(g.conv2d(xg, wg, o)), await w.download(w.conv2d(xw, ww, o))];
});
test("convTranspose1d", "convTranspose1d s2 groups2", 1e-3, async () => {
  const [xg, xw] = pair(4, 12);
  const [wg, ww] = pair(1, 4 * 3 * 4);
  const [big, biw] = pair(1, 6);
  const o = { cout: 6, k: 4, stride: 2, pad: 1, groups: 2 };
  return [await g.download(g.convTranspose1d(xg, wg, { ...o, bias: big })), await w.download(w.convTranspose1d(xw, ww, { ...o, bias: biw }))];
});

// ── recurrent ────────────────────────────────────────────────────────────────
test("lstm", "bidirectional lstm", 1e-3, async () => {
  const hid = 16;
  const inp = 8;
  const [xg, xw] = pair(6, inp);
  const [wg, ww] = pair(2 * 4 * hid, inp);
  const [rg, rw] = pair(2 * 4 * hid, hid);
  const [bg, bw] = pair(2, 8 * hid);
  return [await g.download(g.lstm(xg, wg, rg, bg, hid)), await w.download(w.lstm(xw, ww, rw, bw, hid))];
});

// ── attention ────────────────────────────────────────────────────────────────
test("bmmQK", "bmmQK W=2 +qb", 1e-4, async () => {
  const H = 3;
  const HD = 16;
  const [qg, qw] = pair(2 * 7, H * HD);
  const [kg, kw] = pair(2 * 9, H * HD);
  const [qbg, qbw] = pair(1, H * HD);
  return [await g.download(g.bmmQK(qg, kg, qbg, H, HD, 2)), await w.download(w.bmmQK(qw, kw, qbw, H, HD, 2))];
});
test("bmmQK", "bmmQK bShared (pos rows)", 1e-4, async () => {
  const H = 3;
  const HD = 16;
  const [qg, qw] = pair(2 * 7, H * HD);
  const [pg, pw] = pair(11, H * HD);
  return [await g.download(g.bmmQK(qg, pg, null, H, HD, 2, true)), await w.download(w.bmmQK(qw, pw, null, H, HD, 2, true))];
});
test("bmmPV", "bmmPV W=2", 1e-4, async () => {
  const H = 3;
  const HD = 16;
  const [pg, pw] = pair(2 * H * 7, 9);
  const [vg, vw] = pair(2 * 9, H * HD);
  return [await g.download(g.bmmPV(pg, vg, H, HD, 2)), await w.download(w.bmmPV(pw, vw, H, HD, 2))];
});
test("relShift", "relShift", 0, async () => {
  const [xg, xw] = pair(9, 17);
  return [await g.download(g.relShift(xg)), await w.download(w.relShift(xw))];
});
test("relShiftB", "relShiftB", 0, async () => {
  const [xg, xw] = pair(3 * 7, 13);
  return [await g.download(g.relShiftB(xg, 3)), await w.download(w.relShiftB(xw, 3))];
});
test("relShiftStream", "relShiftStream", 0, async () => {
  const o = { H: 3, n: 6, Lk: 20, dMax: 16, Lc: 12, subT: 24, C: 8, left: 8, right: 0 };
  const [xg, xw] = pair(3 * 6, 33);
  return [await g.download(g.relShiftStream(xg, o)), await w.download(w.relShiftStream(xw, o))];
});
test("rmsNorm", "rmsNorm (gemma 1+w)", 1e-5, async () => {
  const [xg, xw] = pair(5, 96);
  const [wg, ww] = pair(1, 96);
  return [await g.download(g.rmsNorm(xg, wg, 1e-6)), await w.download(w.rmsNorm(xw, ww, 1e-6))];
});
test("rmsNorm", "rmsNorm + residual add", 1e-5, async () => {
  const [xg, xw] = pair(5, 96);
  const [wg, ww] = pair(1, 96);
  const [ag, aw] = pair(5, 96);
  return [await g.download(g.rmsNorm(xg, wg, 1e-6, { add: ag })), await w.download(w.rmsNorm(xw, ww, 1e-6, { add: aw }))];
});
test("headRmsRope", "headRmsRope (norm + rope, 2 streams)", 1e-4, async () => {
  const o = { heads: 3, headDim: 24, M: 4, pos0: 5, scale: 0.25, eps: 1e-6 };
  const invFreq = Float64Array.from({ length: 12 }, (_, i) => 1 / Math.pow(10000, (2 * i) / 24));
  const [xg, xw] = pair(2 * 4, 3 * 24);
  const [wg, ww] = pair(1, 24);
  return [await g.download(g.headRmsRope(xg, wg, invFreq, o)), await w.download(w.headRmsRope(xw, ww, invFreq, o))];
});
test("headRmsRope", "headRmsRope (rope only, w=null)", 1e-4, async () => {
  const o = { heads: 3, headDim: 24, M: 6, pos0: 0, scale: 1 };
  const invFreq = Float64Array.from({ length: 12 }, (_, i) => 1 / Math.pow(500, (2 * i) / 24));
  const [xg, xw] = pair(6, 3 * 24);
  return [await g.download(g.headRmsRope(xg, null, invFreq, o)), await w.download(w.headRmsRope(xw, null, invFreq, o))];
});
test("attnCache", "attnCache causal, 2 streams, strided cache", 1e-4, async () => {
  const o = { heads: 3, headDim: 16, M: 4, pos0: 6, cacheStride: 12, causal: true };
  const [qg, qw] = pair(2 * 4, 3 * 16);
  const [kg, kw] = pair(2 * 12, 3 * 16);
  const [vg, vw] = pair(2 * 12, 3 * 16);
  return [await g.download(g.attnCache(qg, kg, vg, o)), await w.download(w.attnCache(qw, kw, vw, o))];
});
test("attnCache", "attnCache bidirectional + softcap (CAS)", 1e-4, async () => {
  const o = { heads: 3, headDim: 16, M: 7, causal: false, fixedT: 7, softcap: 50 };
  const [qg, qw] = pair(7, 3 * 16);
  const [kg, kw] = pair(7, 3 * 16);
  const [vg, vw] = pair(7, 3 * 16);
  return [await g.download(g.attnCache(qg, kg, vg, o)), await w.download(w.attnCache(qw, kw, vw, o))];
});

// ── elementwise & data movement ──────────────────────────────────────────────
test("ewise", "ewise mul + row broadcast", 1e-6, async () => {
  const [ag, aw] = pair(7, 13);
  const [bg, bw] = pair(1, 13);
  return [await g.download(g.ewise(ag, bg, "mul")), await w.download(w.ewise(aw, bw, "mul"))];
});
test("add", "add", 1e-6, async () => {
  const [ag, aw] = pair(7, 13);
  const [bg, bw] = pair(7, 13);
  return [await g.download(g.add(ag, bg)), await w.download(w.add(aw, bw))];
});
test("mul", "mul", 1e-6, async () => {
  const [ag, aw] = pair(7, 13);
  const [bg, bw] = pair(7, 13);
  return [await g.download(g.mul(ag, bg)), await w.download(w.mul(aw, bw))];
});
test("scale", "scale", 1e-6, async () => {
  const [xg, xw] = pair(6, 11);
  return [await g.download(g.scale(xg, -1.7)), await w.download(w.scale(xw, -1.7))];
});
test("silu", "silu", 1e-5, async () => {
  const [xg, xw] = pair(6, 11);
  return [await g.download(g.silu(xg)), await w.download(w.silu(xw))];
});
test("relu", "relu", 0, async () => {
  const [xg, xw] = pair(6, 11);
  return [await g.download(g.relu(xg)), await w.download(w.relu(xw))];
});
test("leakyRelu", "leakyRelu slope 0.3", 1e-6, async () => {
  const [xg, xw] = pair(6, 11);
  return [await g.download(g.leakyRelu(xg, 0.3)), await w.download(w.leakyRelu(xw, 0.3))];
});
test("snake", "snake", 1e-5, async () => {
  const [xg, xw] = pair(4, 17);
  const [ag, aw] = pair(1, 4);
  return [await g.download(g.snake(xg, ag)), await w.download(w.snake(xw, aw))];
});
test("glu", "glu", 1e-5, async () => {
  const [xg, xw] = pair(8, 9);
  return [await g.download(g.glu(xg)), await w.download(w.glu(xw))];
});
test("transpose", "transpose", 0, async () => {
  const [xg, xw] = pair(7, 13);
  return [await g.download(g.transpose(xg)), await w.download(w.transpose(xw))];
});
test("sliceCols", "sliceCols", 0, async () => {
  const [xg, xw] = pair(7, 13);
  return [await g.download(g.sliceCols(xg, 3, 6)), await w.download(w.sliceCols(xw, 3, 6))];
});
test("setCols", "setCols", 0, async () => {
  const [dg, dw] = pair(7, 13);
  const [sg, sw] = pair(7, 4);
  return [await g.download(g.setCols(dg, sg, 5)), await w.download(w.setCols(dw, sw, 5))];
});
test("sliceRows", "sliceRows", 0, async () => {
  const [xg, xw] = pair(10, 6);
  return [await g.download(g.sliceRows(xg, 3, 4)), await w.download(w.sliceRows(xw, 3, 4))];
});
test("copyRows", "copyRows", 0, async () => {
  const [dg, dw] = pair(10, 6);
  const [sg, sw] = pair(3, 6);
  return [await g.download(g.copyRows(dg, sg, 4)), await w.download(w.copyRows(dw, sw, 4))];
});
test("concatRows", "concatRows ×3", 0, async () => {
  const [ag, aw] = pair(3, 6);
  const [bg, bw] = pair(2, 6);
  const [cg, cw] = pair(4, 6);
  return [await g.download(g.concatRows([ag, bg, cg])), await w.download(w.concatRows([aw, bw, cw]))];
});
test("gatherCols", "gatherCols", 0, async () => {
  const [xg, xw] = pair(5, 12);
  const idx = Uint32Array.from({ length: 20 }, () => Math.floor(rand() * 12));
  return [await g.download(g.gatherCols(xg, idx)), await w.download(w.gatherCols(xw, idx))];
});

// ── decoder-side fusions ─────────────────────────────────────────────────────
test("subReshape", "subReshape", 0, async () => {
  const [xg, xw] = pair(5, 7 * 4);
  return [await g.download(g.subReshape(xg, 5, 7, 4)), await w.download(w.subReshape(xw, 5, 7, 4))];
});
test("jbatch", "jbatch", 1e-6, async () => {
  const [eg, ew] = pair(10, 32);
  const [pg, pw] = pair(1, 32);
  return [await g.download(g.jbatch(eg, 2, 5, pg, 32)), await w.download(w.jbatch(ew, 2, 5, pw, 32))];
});
test("argmaxRows", "argmaxRows (exact indices)", 1e-3, async () => {
  const [xg, xw] = pair(6, 40);
  return [await g.download(g.argmaxRows(xg, 6, 32, 40)), await w.download(w.argmaxRows(xw, 6, 32, 40))];
});
test("jointArgmax", "jointArgmax (exact indices)", 1e-3, async () => {
  const [eg, ew] = pair(10, 32);
  const [pg, pw] = pair(1, 32);
  const [wg2, ww2] = pair(32, 44);
  const [bg, bw] = pair(1, 44);
  return [await g.download(g.jointArgmax(eg, 1, 4, pg, wg2, bg, 32, 36, 44)), await w.download(w.jointArgmax(ew, 1, 4, pw, ww2, bw, 32, 36, 44))];
});

// ── shared plumbing with observable behavior ─────────────────────────────────
test("ensureTensor", "ensureTensor host literal", 1e-6, async () => {
  const d = mkData(5 * 8);
  const host = () => ({ data: d.slice(), rows: 5, cols: 8 });
  return [await g.download(g.scale(g.ensureTensor(host()), 2)), await w.download(w.scale(w.ensureTensor(host()), 2))];
});
test("rowsView", "rowsView feeds matmul", 1e-3, async () => {
  const [cg, cw] = pair(10, 24);
  const [bg, bw] = pair(24, 12);
  return [await g.download(g.matmul(g.rowsView(cg, 6), bg)), await w.download(w.matmul(w.rowsView(cw, 6), bw))];
});
test("stageDownload", "stageDownload staged read", 1e-3, async () => {
  const [ag, aw] = pair(8, 16);
  const [bg, bw] = pair(16, 8);
  return [await g.stageDownload(g.matmul(ag, bg)).read(), await w.stageDownload(w.matmul(aw, bw)).read()];
});
test("withBatch", "batched pipeline: matmul→copyRows→add→download-in-batch", 1e-3, async () => {
  const run = async (ctx) => {
    const [a, b, cache, d] = [ctx.upload(A, 12, 16), ctx.upload(B, 16, 8), ctx.upload(CACHE, 20, 8), ctx.upload(D, 12, 8)];
    return ctx.withBatch(async () => {
      const y = ctx.matmul(a, b, { act: "gelu" });
      ctx.copyRows(cache, y, 4);
      const mid = await ctx.download(cache); // flush + reopen mid-batch
      const z = ctx.add(y, d);
      const fin = await ctx.download(z);
      const out = new Float32Array(mid.length + fin.length);
      out.set(mid, 0);
      out.set(fin, mid.length);
      return out;
    });
  };
  const A = mkData(12 * 16);
  const B = mkData(16 * 8);
  const CACHE = mkData(20 * 8);
  const D = mkData(12 * 8);
  return [await run(g), await run(w)];
});

// ── run ──────────────────────────────────────────────────────────────────────
let failed = 0;
const tested = new Set();
for (const c of cases) {
  tested.add(c.method);
  if (c.skip) {
    console.log(`- ${c.label}`);
    continue;
  }
  const arena = g.pushArena(); // exercise arena-scoped alloc on every case
  try {
    const [a, b] = await c.fn();
    if (a.length !== b.length) throw new Error(`length mismatch gpu=${a.length} wasm=${b.length}`);
    let max = 0;
    let nan = false;
    for (let i = 0; i < a.length; i++) {
      if (Number.isNaN(a[i]) || Number.isNaN(b[i])) nan = true;
      const d = Math.abs(a[i] - b[i]);
      if (d > max) max = d;
    }
    if (nan) throw new Error("NaN in output");
    if (max > c.tol) throw new Error(`max|gpu-wasm| = ${max.toExponential(2)} > tol ${c.tol}`);
    console.log(`✓ ${c.label.padEnd(46)} max|gpu-wasm| = ${max.toExponential(2)}  (tol ${c.tol})`);
  } catch (e) {
    failed++;
    console.error(`✗ ${c.label}: ${e.message}`);
  } finally {
    g.popArena(arena);
  }
}

// Completeness: every required interface method is tested or exempted.
const { required } = computeInterfaceMethods();
const uncovered = required.filter((m) => !tested.has(m) && !EXEMPT.has(m));
const stale = [...EXEMPT].filter((m) => tested.has(m));
if (uncovered.length) {
  failed++;
  console.error(`✗ required ComputeContext methods with no conformance case: ${uncovered.join(", ")}`);
}
if (stale.length) {
  failed++;
  console.error(`✗ EXEMPT entries that now have cases (remove them): ${stale.join(", ")}`);
}

if (failed) console.error(`\n${failed} FAILURE(S)`);
else console.log(`\nALL ${cases.length} CASES GPU==WASM · ${required.length} required methods covered`);
// dawn's device keeps the event loop alive — exit explicitly (as gpu-verify does).
process.exit(failed ? 1 : 0);
