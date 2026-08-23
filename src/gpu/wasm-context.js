// WasmContext — the CPU/WASM backend for the raw engines, a drop-in for GpuContext
// (src/gpu/compute.js). Same method surface, same math, so a browser WITHOUT WebGPU
// runs the ORT-free engines identically. Hot kernels (matmul / conv1d / int8 / int4)
// call the wasm32+SIMD lib (rust/wasm-kernels); the cheap, memory-bound ops are the
// exact CPU references from scripts/gpu-verify.mjs, in plain JS over Float32Array.
//
// Tensors are CPU-resident: { data: Float32Array, rows, cols }. Byte tensors (packed
// int8/int4 weights): { bytes: Uint8Array }. Bias + activation are applied here in JS
// after the bare-accumulation wasm kernels, matching the WGSL fused semantics exactly.

// ── activations (match compute.js WGSL exactly) ──────────────────────────────
const geluTanh = (x) => {
  const t = Math.max(-20, Math.min(20, 0.7978845608028654 * (x + 0.044715 * x * x * x)));
  return 0.5 * x * (1 + Math.tanh(t));
};
const erfA = (x) => {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
};
const geluErf = (x) => 0.5 * x * (1 + erfA(x * 0.70710678118654752));
const silu1 = (x) => x / (1 + Math.exp(-Math.max(-30, Math.min(30, x))));
function applyAct(v, act) {
  switch (act) {
    case "gelu":
      return geluTanh(v);
    case "tanh":
      return Math.tanh(v);
    case "relu":
      return v > 0 ? v : 0;
    case "silu":
      return silu1(v);
    case "gelu_erf":
      return geluErf(v);
    default:
      return v;
  }
}

/** Instantiate the wasm kernel lib and return a ready WasmContext. */
export async function createWasmContext(wasmBytes) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  return new WasmContext(instance.exports);
}

export class WasmContext {
  constructor(exports) {
    this.ex = exports;
    this.backend = "wasm";
  }

  // ── memory / tensors ──────────────────────────────────────────────────────
  upload(data, rows, cols) {
    return { data: data instanceof Float32Array ? data : Float32Array.from(data), rows, cols };
  }
  alloc(rows, cols) {
    return { data: new Float32Array(rows * cols), rows, cols };
  }
  uploadBytes(typed) {
    const bytes = typed instanceof Uint8Array ? typed : new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
    return { bytes: bytes.slice() };
  }
  // f16 is a WebGPU storage optimization; on CPU everything is f32.
  uploadF16(data, rows, cols) {
    return this.upload(data, rows, cols);
  }
  allocF16(rows, cols) {
    return this.alloc(rows, cols);
  }
  // download is ALWAYS a copy (GpuContext readback is a copy; callers mutate
  // downloaded arrays — e.g. whisper's suppress writes — so aliasing the live
  // tensor storage would silently diverge the two backends).
  async download(t) {
    return t.data.slice();
  }
  async downloadF16(t) {
    return t.data.slice();
  }
  /** Staged readback, same contract as GpuContext: the value is snapshotted at
   * stage time (matching WebGPU queue order — ops recorded after the staging
   * copy don't affect the read), read() delivers it later. Single-use, like the
   * GPU staging buffer — a second read() throws instead of silently sharing a
   * mutable array (callers mutate readback results, e.g. whisper suppression). */
  stageDownload(t) {
    let snap = t.data.slice();
    return {
      read: async () => {
        if (!snap) throw new Error("StagedRead.read(): already consumed (single-use)");
        const out = snap;
        snap = null;
        return out;
      },
    };
  }
  /** Rows-limited view over a preallocated tensor (no copy — aliases storage). */
  rowsView(t, rows) {
    return { ...t, data: t.data.subarray(0, rows * t.cols), rows, view: true };
  }
  /** Normalize a host {data,rows,cols} literal into a backend tensor; backend
   * tensors pass through unchanged (a host f32 literal IS a valid CPU tensor). */
  ensureTensor(t) {
    return t.data instanceof Float32Array ? t : this.upload(t.data, t.rows, t.cols);
  }
  /** Tear down the backend (no device to release on CPU). */
  destroy() {}
  beginBatch() {}
  endBatch() {}
  async withBatch(fn) {
    return fn();
  }
  withBatchSync(fn) {
    return fn();
  }
  // Arenas are a GPU-pool concept; on CPU the handle is inert (but truthy, so
  // callers can pushArena()/popArena(handle) unconditionally on any backend).
  pushArena() {
    return [];
  }
  popArena() {}
  trimPool() {}
  pin(t) {
    return t;
  }

  // ── wasm staging helpers ──────────────────────────────────────────────────
  _f32() {
    return new Float32Array(this.ex.memory.buffer);
  }
  _u8() {
    return new Uint8Array(this.ex.memory.buffer);
  }

  // ── HOT kernels (wasm32 + SIMD) ───────────────────────────────────────────
  /** C = act(A@B + bias) [+ add]. bias:[1,N] per-column; add:[M,N] residual. */
  matmul(a, b, { bias = null, act = "none", add = null } = {}) {
    const M = a.rows,
      K = a.cols,
      N = b.cols;
    const ex = this.ex;
    ex.wasm_reset();
    const aPtr = ex.wasm_alloc(M * K * 4),
      bPtr = ex.wasm_alloc(K * N * 4),
      cPtr = ex.wasm_alloc(M * N * 4);
    let f = this._f32();
    f.set(a.data, aPtr >> 2);
    f.set(b.data, bPtr >> 2);
    ex.matmul_f32(aPtr, bPtr, cPtr, M, K, N);
    f = this._f32();
    const out = f.slice(cPtr >> 2, (cPtr >> 2) + M * N);
    this._biasActColumns(out, M, N, bias, act);
    if (add) {
      const d = add.data;
      for (let i = 0; i < out.length; i++) out[i] += d[i];
    }
    return { data: out, rows: M, cols: N };
  }
  matmulGemv(a, b, o) {
    return this.matmul(a, b, o);
  }
  matmulV2(a, b, o) {
    return this.matmul(a, b, o);
  }
  matmulV3(a, b, o) {
    return this.matmul(a, b, o);
  }
  matmulV4(a, b, o) {
    return this.matmul(a, b, o);
  }
  matmulF16(a, b, o) {
    return this.matmul(a, b, o);
  }

  /** int8 GEMM: a[M,K] f32 @ dequant(wq)[K,N] (per-column scale). */
  matmulInt8(a, wq, scale, N, K, { bias = null, act = "none" } = {}) {
    const M = a.rows;
    const ex = this.ex;
    ex.wasm_reset();
    const aPtr = ex.wasm_alloc(M * K * 4),
      wPtr = ex.wasm_alloc(wq.bytes.length),
      sPtr = ex.wasm_alloc(N * 4),
      yPtr = ex.wasm_alloc(M * N * 4);
    let f = this._f32(),
      u = this._u8();
    f.set(a.data, aPtr >> 2);
    u.set(wq.bytes, wPtr);
    f.set(scale.data, sPtr >> 2);
    ex.matmul_int8(aPtr, wPtr, sPtr, yPtr, M, N, K);
    f = this._f32();
    const out = f.slice(yPtr >> 2, (yPtr >> 2) + M * N);
    this._biasActColumns(out, M, N, bias, act);
    return { data: out, rows: M, cols: N };
  }

  /** int4 block-quant GEMM (ONNX MatMulNBits). No bias/act (matches WGSL). */
  matmulNBits(a, bq, scales, zp, N, blockSize = 32) {
    const M = a.rows,
      K = a.cols;
    const nblk = Math.ceil(K / blockSize),
      zpb = Math.ceil(nblk / 2);
    const ex = this.ex;
    ex.wasm_reset();
    const aPtr = ex.wasm_alloc(M * K * 4),
      bqPtr = ex.wasm_alloc(bq.bytes.length),
      scPtr = ex.wasm_alloc(scales.data.length * 4),
      zpPtr = ex.wasm_alloc(zp.bytes.length),
      yPtr = ex.wasm_alloc(M * N * 4);
    let f = this._f32(),
      u = this._u8();
    f.set(a.data, aPtr >> 2);
    u.set(bq.bytes, bqPtr);
    f.set(scales.data, scPtr >> 2);
    u.set(zp.bytes, zpPtr);
    ex.matmul_nbits(aPtr, bqPtr, scPtr, zpPtr, yPtr, M, N, K, nblk, zpb, blockSize);
    f = this._f32();
    return { data: f.slice(yPtr >> 2, (yPtr >> 2) + M * N), rows: M, cols: N };
  }

  /**
   * 1-D conv. groups==1 goes through im2col + the SIMD matmul (the direct wasm
   * conv is scalar over t with per-sample bounds checks — ~4× slower on the big
   * vocoder convs). Grouped/depthwise convs use the direct kernel.
   */
  conv1d(x, w, { cout, k, bias = null, stride = 1, pad = 0, padLeft, padRight, dilation = 1, groups = 1, act = "none" } = {}) {
    padLeft = padLeft ?? pad;
    padRight = padRight ?? pad;
    const Cin = x.rows,
      L = x.cols;
    const Lout = Math.floor((L + padLeft + padRight - dilation * (k - 1) - 1) / stride) + 1;
    if (groups === 1) {
      // im2col (JS, memory-bound) → SIMD GEMM [Cout, Cin*K] @ [Cin*K, Lout].
      const CinK = Cin * k;
      const cols = new Float32Array(CinK * Lout);
      for (let ci = 0; ci < Cin; ci++) {
        const xrow = ci * L;
        for (let kk = 0; kk < k; kk++) {
          const row = (ci * k + kk) * Lout,
            off = kk * dilation - padLeft;
          if (stride === 1) {
            const lo0 = Math.max(0, -off),
              lo1 = Math.min(Lout, L - off);
            for (let lo = lo0; lo < lo1; lo++) cols[row + lo] = x.data[xrow + lo + off];
          } else {
            for (let lo = 0; lo < Lout; lo++) {
              const li = lo * stride + off;
              if (li >= 0 && li < L) cols[row + lo] = x.data[xrow + li];
            }
          }
        }
      }
      const out = this.matmul({ data: w.data, rows: cout, cols: CinK }, { data: cols, rows: CinK, cols: Lout });
      this._biasActRows(out.data, cout, Lout, bias, act);
      return out;
    }
    const ex = this.ex;
    ex.wasm_reset();
    const xPtr = ex.wasm_alloc(Cin * L * 4),
      wPtr = ex.wasm_alloc(w.data.length * 4),
      yPtr = ex.wasm_alloc(cout * Lout * 4);
    let f = this._f32();
    f.set(x.data, xPtr >> 2);
    f.set(w.data, wPtr >> 2);
    ex.conv1d_f32(xPtr, wPtr, yPtr, cout, Cin, L, Lout, k, stride, padLeft, dilation, groups);
    f = this._f32();
    const out = f.slice(yPtr >> 2, (yPtr >> 2) + cout * Lout);
    this._biasActRows(out, cout, Lout, bias, act);
    return { data: out, rows: cout, cols: Lout };
  }
  conv1dFast(x, wRows, cout, k, opts = {}) {
    return this.conv1d(x, wRows, { ...opts, cout, k });
  }
  conv1dFastF16(x, wRows, cout, k, opts = {}) {
    return this.conv1d(x, wRows, { ...opts, cout, k });
  }
  im2col(x, k, { stride = 1, pad = 0, dilation = 1 } = {}) {
    const Cin = x.rows,
      L = x.cols;
    const Lout = Math.floor((L + 2 * pad - dilation * (k - 1) - 1) / stride) + 1;
    const out = new Float32Array(Cin * k * Lout);
    for (let ci = 0; ci < Cin; ci++)
      for (let kk = 0; kk < k; kk++) {
        const row = (ci * k + kk) * Lout;
        for (let lo = 0; lo < Lout; lo++) {
          const li = lo * stride + kk * dilation - pad;
          out[row + lo] = li >= 0 && li < L ? x.data[ci * L + li] : 0;
        }
      }
    return { data: out, rows: Cin * k, cols: Lout };
  }
  conv1dGemm(x, wRows, cout, k, { stride = 1, pad = 0, dilation = 1, act = "none" } = {}) {
    return this.matmul(wRows, this.im2col(x, k, { stride, pad, dilation }), { act });
  }

  _biasActColumns(out, M, N, bias, act) {
    if (!bias && act === "none") return;
    for (let i = 0; i < M; i++)
      for (let j = 0; j < N; j++) {
        let v = out[i * N + j] + (bias ? bias.data[j] : 0);
        out[i * N + j] = applyAct(v, act);
      }
  }
  _biasActRows(out, R, C, bias, act) {
    if (!bias && act === "none") return;
    for (let r = 0; r < R; r++)
      for (let c = 0; c < C; c++) {
        let v = out[r * C + c] + (bias ? bias.data[r] : 0);
        out[r * C + c] = applyAct(v, act);
      }
  }

  // ── cheap kernels (plain JS, exact CPU refs) ──────────────────────────────
  layernorm(x, gamma, beta, eps = 1e-5) {
    const R = x.rows,
      C = x.cols,
      out = new Float32Array(R * C);
    for (let r = 0; r < R; r++) {
      let mean = 0;
      for (let j = 0; j < C; j++) mean += x.data[r * C + j];
      mean /= C;
      let v = 0;
      for (let j = 0; j < C; j++) {
        const d = x.data[r * C + j] - mean;
        v += d * d;
      }
      v /= C;
      const inv = 1 / Math.sqrt(v + eps);
      for (let j = 0; j < C; j++) out[r * C + j] = (x.data[r * C + j] - mean) * inv * gamma.data[j] + beta.data[j];
    }
    return { data: out, rows: R, cols: C };
  }
  softmax(x) {
    const R = x.rows,
      C = x.cols,
      out = new Float32Array(R * C);
    for (let r = 0; r < R; r++) {
      let mx = -Infinity;
      for (let j = 0; j < C; j++) mx = Math.max(mx, x.data[r * C + j]);
      let s = 0;
      for (let j = 0; j < C; j++) {
        const e = Math.exp(x.data[r * C + j] - mx);
        out[r * C + j] = e;
        s += e;
      }
      for (let j = 0; j < C; j++) out[r * C + j] /= s;
    }
    return { data: out, rows: R, cols: C };
  }
  /** Gemma RMSNorm rows: y = x·rsqrt(mean(x²)+eps)·(1+w) [+ add]. f64 accumulate
   * with per-element f32 store — the parity reference for the GPU kernel. */
  rmsNorm(x, w, eps = 1e-6, { add = null } = {}) {
    const rows = x.rows,
      cols = x.cols;
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      const o = r * cols;
      let s = 0;
      for (let j = 0; j < cols; j++) s += x.data[o + j] * x.data[o + j];
      const inv = 1 / Math.sqrt(s / cols + eps);
      for (let j = 0; j < cols; j++) {
        const v = Math.fround(x.data[o + j] * inv * (1 + w.data[j]));
        out[o + j] = add ? add.data[o + j] + v : v;
      }
    }
    return { data: out, rows, cols };
  }

  /** Per-head RMSNorm (skipped when w is null) + rotate-half RoPE; see the
   * GpuContext doc. invFreq is consumed at f64 here (parity reference). */
  headRmsRope(x, w, invFreq, { heads, headDim, M, pos0 = 0, scale = 1, eps = 1e-6 }) {
    const rows = x.rows,
      D = x.cols,
      half = headDim / 2;
    const out = x.data.slice();
    if (w) {
      for (let r = 0; r < rows; r++)
        for (let h = 0; h < heads; h++) {
          const o = r * D + h * headDim;
          let s = 0;
          for (let j = 0; j < headDim; j++) s += out[o + j] * out[o + j];
          const inv = 1 / Math.sqrt(s / headDim + eps);
          for (let j = 0; j < headDim; j++) out[o + j] = out[o + j] * inv * (1 + w.data[j]);
        }
    }
    for (let r = 0; r < rows; r++) {
      const p = pos0 + (r % M);
      for (let h = 0; h < heads; h++) {
        const o = r * D + h * headDim;
        for (let i = 0; i < half; i++) {
          const f = p * invFreq[i],
            c = Math.cos(f),
            s = Math.sin(f);
          const a = out[o + i],
            b = out[o + half + i];
          out[o + i] = (a * c - b * s) * scale;
          out[o + half + i] = (b * c + a * s) * scale;
        }
      }
    }
    return { data: out, rows, cols: D };
  }

  /** Cached multi-head attention (see GpuContext.attnCache). Scores max/exp/sum
   * and the probs@V accumulation run at f64 per head (parity reference). */
  attnCache(q, k, v, { heads, headDim, M, pos0 = 0, cacheStride = 0, causal = true, fixedT = 0, softcap = 0 }) {
    const D = heads * headDim;
    const W = q.rows / M;
    const stride = cacheStride || k.rows / W;
    const Tk = fixedT || pos0 + M;
    const out = new Float32Array(W * M * D);
    for (let w = 0; w < W; w++) {
      const qo0 = w * M * D,
        co0 = w * stride * D;
      const scores = new Float32Array(Tk);
      for (let r = 0; r < M; r++) {
        const T = causal ? pos0 + r + 1 : Tk;
        for (let h = 0; h < heads; h++) {
          const qo = qo0 + r * D + h * headDim;
          let mx = -Infinity;
          for (let j = 0; j < T; j++) {
            const ko = co0 + j * D + h * headDim;
            let s = 0;
            for (let d = 0; d < headDim; d++) s += q.data[qo + d] * k.data[ko + d];
            if (softcap) s = softcap * Math.tanh(s / softcap);
            scores[j] = s;
            if (s > mx) mx = s;
          }
          let sum = 0;
          for (let j = 0; j < T; j++) {
            const e = Math.exp(scores[j] - mx);
            scores[j] = e;
            sum += e;
          }
          const oo = qo0 + r * D + h * headDim;
          for (let j = 0; j < T; j++) {
            const p = scores[j] / sum;
            const vo = co0 + j * D + h * headDim;
            for (let d = 0; d < headDim; d++) out[oo + d] += p * v.data[vo + d];
          }
        }
      }
    }
    return { data: out, rows: W * M, cols: D };
  }

  adain(x, scale, shift, eps = 1e-5) {
    const C = x.rows,
      L = x.cols,
      out = new Float32Array(C * L);
    for (let ch = 0; ch < C; ch++) {
      let mean = 0;
      for (let j = 0; j < L; j++) mean += x.data[ch * L + j];
      mean /= L;
      let v = 0;
      for (let j = 0; j < L; j++) {
        const d = x.data[ch * L + j] - mean;
        v += d * d;
      }
      v /= L;
      const inv = 1 / Math.sqrt(v + eps);
      for (let j = 0; j < L; j++) out[ch * L + j] = (x.data[ch * L + j] - mean) * inv * scale.data[ch] + shift.data[ch];
    }
    return { data: out, rows: C, cols: L };
  }
  ewise(a, b, op) {
    const R = a.rows,
      C = a.cols,
      out = new Float32Array(R * C),
      bcast = b.rows === 1;
    for (let i = 0; i < R * C; i++) {
      const bv = bcast ? b.data[i % C] : b.data[i];
      out[i] = op === "mul" ? a.data[i] * bv : a.data[i] + bv;
    }
    return { data: out, rows: R, cols: C };
  }
  add(a, b) {
    return this.ewise(a, b, "add");
  }
  mul(a, b) {
    return this.ewise(a, b, "mul");
  }
  scale(x, s) {
    const o = new Float32Array(x.data.length);
    for (let i = 0; i < o.length; i++) o[i] = x.data[i] * s;
    return { data: o, rows: x.rows, cols: x.cols };
  }
  concatRows(tensors) {
    const cols = tensors[0].cols,
      rows = tensors.reduce((s, t) => s + t.rows, 0);
    const out = new Float32Array(rows * cols);
    let off = 0;
    for (const t of tensors) {
      out.set(t.data, off);
      off += t.rows * t.cols;
    }
    return { data: out, rows, cols };
  }
  silu(x) {
    const o = new Float32Array(x.data.length);
    for (let i = 0; i < o.length; i++) o[i] = silu1(x.data[i]);
    return { data: o, rows: x.rows, cols: x.cols };
  }
  relu(x) {
    const o = new Float32Array(x.data.length);
    for (let i = 0; i < o.length; i++) o[i] = Math.max(0, x.data[i]);
    return { data: o, rows: x.rows, cols: x.cols };
  }
  leakyRelu(x, slope = 0.2) {
    const o = new Float32Array(x.data.length);
    for (let i = 0; i < o.length; i++) {
      const v = x.data[i];
      o[i] = v > 0 ? v : slope * v;
    }
    return { data: o, rows: x.rows, cols: x.cols };
  }
  snake(x, alpha) {
    const C = x.rows,
      L = x.cols,
      o = new Float32Array(C * L);
    for (let c = 0; c < C; c++) {
      const a = alpha.data[c],
        inv = 1 / (a + 1e-9);
      for (let j = 0; j < L; j++) {
        const s = Math.sin(a * x.data[c * L + j]);
        o[c * L + j] = x.data[c * L + j] + inv * s * s;
      }
    }
    return { data: o, rows: C, cols: L };
  }
  glu(x) {
    const C = x.rows / 2,
      T = x.cols,
      out = new Float32Array(C * T);
    for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) out[c * T + t] = x.data[c * T + t] * (1 / (1 + Math.exp(-x.data[(c + C) * T + t])));
    return { data: out, rows: C, cols: T };
  }
  transpose(x) {
    const R = x.rows,
      C = x.cols,
      out = new Float32Array(R * C);
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) out[c * R + r] = x.data[r * C + c];
    return { data: out, rows: C, cols: R };
  }
  sliceCols(x, col0, width) {
    const R = x.rows,
      C = x.cols,
      out = new Float32Array(R * width);
    for (let r = 0; r < R; r++) for (let j = 0; j < width; j++) out[r * width + j] = x.data[r * C + col0 + j];
    return { data: out, rows: R, cols: width };
  }
  setCols(dst, src, col0) {
    for (let r = 0; r < src.rows; r++) for (let j = 0; j < src.cols; j++) dst.data[r * dst.cols + col0 + j] = src.data[r * src.cols + j];
    return dst;
  }
  gatherCols(x, idxMap) {
    const C = x.rows,
      T = x.cols,
      outCols = idxMap.length,
      out = new Float32Array(C * outCols);
    for (let r = 0; r < C; r++) for (let f = 0; f < outCols; f++) out[r * outCols + f] = x.data[r * T + idxMap[f]];
    return { data: out, rows: C, cols: outCols };
  }
  /** Batched QK^T / Q·pos^T over all heads: q[T,H*HD], b[Tb,H*HD] → [H*T,Tb]. qb?:[1,H*HD]. */
  bmmQK(q, b, qb, H, HD, W = 1, bShared = false) {
    const T = q.rows / W,
      Tb = bShared ? b.rows : b.rows / W,
      stride = H * HD;
    const out = new Float32Array(W * H * T * Tb);
    for (let w = 0; w < W; w++)
      for (let h = 0; h < H; h++) {
        const ho = h * HD;
        for (let i = 0; i < T; i++) {
          const qBase = (w * T + i) * stride + ho,
            oBase = ((w * H + h) * T + i) * Tb;
          for (let j = 0; j < Tb; j++) {
            const bBase = (bShared ? j : w * Tb + j) * stride + ho; // per-window keys stride Tb
            let acc = 0;
            for (let d = 0; d < HD; d++) acc += (q.data[qBase + d] + (qb ? qb.data[ho + d] : 0)) * b.data[bBase + d];
            out[oBase + j] = acc;
          }
        }
      }
    return { data: out, rows: W * H * T, cols: Tb };
  }
  /** Batched probs@V over all heads: p[H*T,T], v[T,H*HD] → [T,H*HD]. */
  bmmPV(p, v, H, HD, W = 1) {
    const Tk = v.rows / W,
      Tq = p.rows / (W * H),
      stride = H * HD;
    const out = new Float32Array(W * Tq * stride);
    for (let w = 0; w < W; w++)
      for (let i = 0; i < Tq; i++) {
        for (let c = 0; c < stride; c++) {
          const h = (c / HD) | 0,
            pBase = ((w * H + h) * Tq + i) * Tk;
          let acc = 0;
          for (let j = 0; j < Tk; j++) acc += p.data[pBase + j] * v.data[(w * Tk + j) * stride + c];
          out[(w * Tq + i) * stride + c] = acc;
        }
      }
    return { data: out, rows: W * Tq, cols: stride };
  }
  copyRows(dst, src, rowOffset) {
    dst.data.set(src.data, rowOffset * dst.cols);
    return dst;
  }
  sliceRows(x, row0, count) {
    return { data: x.data.slice(row0 * x.cols, (row0 + count) * x.cols), rows: count, cols: x.cols };
  }
  freeTensor() {}
  /** Streaming rel_shift + chunked-causal mask (see compute.js relShiftStream). */
  relShiftStream(x, { H, n, Lk, dMax, Lc, subT, C, left, right }) {
    const P = x.cols;
    const out = new Float32Array(H * n * Lk);
    for (let r = 0; r < H * n; r++) {
      const i = r % n,
        q = subT + i,
        cs = C * Math.floor(q / C);
      for (let j = 0; j < Lk; j++) {
        const k = subT - Lc + j;
        out[r * Lk + j] = k < cs - left || k > cs + C - 1 + right ? -10000 : x.data[r * P + Math.min(P - 1, Math.max(0, dMax - (q - k)))];
      }
    }
    return { data: out, rows: H * n, cols: Lk };
  }
  /** Batched rel_shift: x[H*t, 2t-1] → [H*t, t]. */
  relShiftB(x, H) {
    const t = x.rows / H,
      p = 2 * t - 1,
      twoT = 2 * t;
    const out = new Float32Array(H * t * t);
    for (let h = 0; h < H; h++)
      for (let i = 0; i < t; i++)
        for (let j = 0; j < t; j++) {
          const f = t + i * p + j,
            col = f % twoT;
          out[(h * t + i) * t + j] = col === 0 ? 0 : x.data[(h * t + ((f / twoT) | 0)) * p + (col - 1)];
        }
    return { data: out, rows: H * t, cols: t };
  }
  relShift(x) {
    const t = x.rows,
      p = 2 * t - 1,
      twoT = 2 * t,
      out = new Float32Array(t * t);
    for (let i = 0; i < t; i++)
      for (let j = 0; j < t; j++) {
        const f = t + i * p + j,
          col = f % twoT;
        out[i * t + j] = col === 0 ? 0 : x.data[((f / twoT) | 0) * p + (col - 1)];
      }
    return { data: out, rows: t, cols: t };
  }
  subReshape(x, C, Tsub, F) {
    const CF = C * F,
      out = new Float32Array(Tsub * CF);
    for (let idx = 0; idx < Tsub * CF; idx++) {
      const ho = (idx / CF) | 0,
        rem = idx % CF,
        c = (rem / F) | 0,
        wo = rem % F;
      out[idx] = x.data[c * (Tsub * F) + ho * F + wo];
    }
    return { data: out, rows: Tsub, cols: CF };
  }
  jbatch(encProj, base, B, predProj, hid) {
    const out = new Float32Array(B * hid);
    for (let i = 0; i < B; i++)
      for (let k = 0; k < hid; k++) {
        const v = encProj.data[(base + i) * hid + k] + predProj.data[k];
        out[i * hid + k] = v > 0 ? v : 0;
      }
    return { data: out, rows: B, cols: hid };
  }
  argmaxRows(x, B, vocab, logits) {
    const res = new Float32Array(B * 4);
    for (let r = 0; r < B; r++) {
      let bt = 0,
        bv = -1e30,
        bd = 0,
        bdv = -1e30;
      for (let n = 0; n < logits; n++) {
        const s = x.data[r * logits + n];
        if (n < vocab) {
          if (s > bv) {
            bv = s;
            bt = n;
          }
        } else if (s > bdv) {
          bdv = s;
          bd = n - vocab;
        }
      }
      res[r * 4] = bt;
      res[r * 4 + 1] = bv;
      res[r * 4 + 2] = bd;
      res[r * 4 + 3] = bdv;
    }
    return { data: res, rows: B, cols: 4 };
  }
  jointArgmax(encProj, frame, count, predProj, outW, outB, hidden, vocab, logits) {
    const res = new Float32Array(count * 4);
    const j = new Float32Array(hidden);
    for (let w = 0; w < count; w++) {
      const base = (frame + w) * hidden;
      for (let k = 0; k < hidden; k++) {
        const v = encProj.data[base + k] + predProj.data[k];
        j[k] = v > 0 ? v : 0;
      }
      let bt = 0,
        bv = -1e30,
        bd = 0,
        bdv = -1e30;
      for (let n = 0; n < logits; n++) {
        let s = outB.data[n];
        for (let k = 0; k < hidden; k++) s += j[k] * outW.data[k * logits + n];
        if (n < vocab) {
          if (s > bv) {
            bv = s;
            bt = n;
          }
        } else if (s > bdv) {
          bdv = s;
          bd = n - vocab;
        }
      }
      res[w * 4] = bt;
      res[w * 4 + 1] = bv;
      res[w * 4 + 2] = bd;
      res[w * 4 + 3] = bdv;
    }
    return { data: res, rows: count, cols: 4 };
  }
  conv2d(
    x,
    w,
    {
      cout,
      cin,
      h,
      w: W_,
      kh,
      kw,
      bias = null,
      strideH = 1,
      strideW = 1,
      padH = 0,
      padW = 0,
      padTop,
      padBottom,
      padLeft,
      padRight,
      groups = 1,
      act = "none",
    } = {},
  ) {
    padTop = padTop ?? padH;
    padBottom = padBottom ?? padH;
    padLeft = padLeft ?? padW;
    padRight = padRight ?? padW;
    const Ho = Math.floor((h + padTop + padBottom - kh) / strideH) + 1,
      Wo = Math.floor((W_ + padLeft + padRight - kw) / strideW) + 1;
    const out = new Float32Array(cout * Ho * Wo),
      cinG = cin / groups,
      coutG = cout / groups;
    for (let co = 0; co < cout; co++) {
      const g = (co / coutG) | 0;
      for (let ho = 0; ho < Ho; ho++)
        for (let wo = 0; wo < Wo; wo++) {
          let acc = bias ? bias.data[co] : 0;
          for (let ci = 0; ci < cinG; ci++) {
            const rc = g * cinG + ci;
            for (let khh = 0; khh < kh; khh++) {
              const hi = ho * strideH + khh - padTop;
              if (hi < 0 || hi >= h) continue;
              for (let kww = 0; kww < kw; kww++) {
                const wi = wo * strideW + kww - padLeft;
                if (wi < 0 || wi >= W_) continue;
                acc += x.data[rc * h * W_ + hi * W_ + wi] * w.data[((co * cinG + ci) * kh + khh) * kw + kww];
              }
            }
          }
          out[co * Ho * Wo + ho * Wo + wo] = applyAct(acc, act);
        }
    }
    return { data: out, rows: cout, cols: Ho * Wo };
  }
  convTranspose1d(x, w, { cout, k, bias = null, stride = 1, pad = 0, dilation = 1, groups = 1, outputPadding = 0, act = "none" } = {}) {
    const Cin = x.rows,
      L = x.cols;
    const Lout = (L - 1) * stride - 2 * pad + dilation * (k - 1) + outputPadding + 1;
    if (groups === 1 && dilation === 1) {
      // As GEMM + col2im: cols[(co,k), t] = Wt[(co,k), ci] @ x[ci, t], then
      // scatter-add cols[(co,k), t] into y[co, t*stride + k - pad]. The GEMM is
      // the SIMD matmul; the scatter is memory-bound JS.
      if (!this._ctW) this._ctW = new Map();
      let wt = this._ctW.get(w); // cache transposed weights per-tensor
      if (!wt) {
        wt = new Float32Array(cout * k * Cin); // [(co,k), ci] from w[ci, co, k]
        for (let ci = 0; ci < Cin; ci++)
          for (let co = 0; co < cout; co++) for (let kk = 0; kk < k; kk++) wt[(co * k + kk) * Cin + ci] = w.data[ci * (cout * k) + co * k + kk];
        this._ctW.set(w, wt);
      }
      const cols = this.matmul({ data: wt, rows: cout * k, cols: Cin }, x); // [(co,k), L]
      const out = new Float32Array(cout * Lout);
      if (bias) for (let co = 0; co < cout; co++) out.fill(bias.data[co], co * Lout, (co + 1) * Lout);
      for (let co = 0; co < cout; co++) {
        for (let kk = 0; kk < k; kk++) {
          const crow = (co * k + kk) * L,
            base = kk - pad,
            orow = co * Lout;
          for (let t = 0; t < L; t++) {
            const p = t * stride + base;
            if (p >= 0 && p < Lout) out[orow + p] += cols.data[crow + t];
          }
        }
      }
      if (act !== "none") for (let i = 0; i < out.length; i++) out[i] = applyAct(out[i], act);
      return { data: out, rows: cout, cols: Lout };
    }
    const out = new Float32Array(cout * Lout),
      cinG = Cin / groups,
      coutG = cout / groups;
    for (let co = 0; co < cout; co++) {
      const g = (co / coutG) | 0,
        coInG = co - g * coutG;
      for (let lo = 0; lo < Lout; lo++) {
        let acc = bias ? bias.data[co] : 0;
        for (let ci = 0; ci < cinG; ci++) {
          const rc = g * cinG + ci;
          for (let kk = 0; kk < k; kk++) {
            const num = lo + pad - kk * dilation;
            if (num >= 0 && num % stride === 0) {
              const li = num / stride;
              if (li >= 0 && li < L) acc += x.data[rc * L + li] * w.data[rc * (coutG * k) + coInG * k + kk];
            }
          }
        }
        out[co * Lout + lo] = applyAct(acc, act);
      }
    }
    return { data: out, rows: cout, cols: Lout };
  }
  lstm(x, w, r, b, hid) {
    const seq = x.rows,
      inp = x.cols,
      out = new Float32Array(seq * 2 * hid);
    const sig = (v) => 1 / (1 + Math.exp(-v));
    for (let dir = 0; dir < 2; dir++) {
      const wB = dir * 4 * hid * inp,
        rB = dir * 4 * hid * hid,
        bB = dir * 8 * hid;
      const h = new Float32Array(hid),
        c = new Float32Array(hid);
      for (let s = 0; s < seq; s++) {
        const t = dir === 1 ? seq - 1 - s : s,
          hn = new Float32Array(hid);
        for (let u = 0; u < hid; u++) {
          const gate = (gi) => {
            let acc = b.data[bB + gi * hid + u] + b.data[bB + 4 * hid + gi * hid + u];
            for (let kk = 0; kk < inp; kk++) acc += w.data[wB + (gi * hid + u) * inp + kk] * x.data[t * inp + kk];
            for (let kk = 0; kk < hid; kk++) acc += r.data[rB + (gi * hid + u) * hid + kk] * h[kk];
            return acc;
          };
          const it = sig(gate(0)),
            ot = sig(gate(1)),
            ft = sig(gate(2)),
            ct = Math.tanh(gate(3));
          const cn = ft * c[u] + it * ct;
          c[u] = cn;
          hn[u] = ot * Math.tanh(cn);
          out[(t * 2 + dir) * hid + u] = hn[u];
        }
        h.set(hn);
      }
    }
    return { data: out, rows: seq, cols: 2 * hid };
  }
}
