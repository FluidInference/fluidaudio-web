// WASM-SIMD TDT decoder — CPU decode with NO GPU round-trips (the RNNT autoregressive
// sync wall capped the WebGPU decoder at ~20×). The joint's 640→8198 matmul is
// SIMD-vectorized in rust/parakeet-decoder (compiled to wasm32 +simd128). This is
// the "C++/wasm" half of the stack. Same greedy TDT logic as raw-decoder.js.

/**
 * Instantiate the wasm decoder and copy its weights into wasm memory once.
 * @param {ArrayBuffer|Uint8Array} wasmBytes the compiled parakeet-decoder.wasm
 * @param {Float32Array} decBin decoder weights blob
 * @param {object} decMan manifest {key:{offset,len}}
 */
export async function loadWasmDecoder(wasmBytes, decBin, decMan, { int8 = true } = {}) {
  // WebAssembly.instantiate accepts any BufferSource (TypedArray or ArrayBuffer);
  // pass it as-is — don't use .buffer (a node Buffer views a shared pool).
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const ex = instance.exports;
  ex.reset_to(ex.__heap_base.value); // start the bump allocator past static data + stack
  const g = (k) => decBin.subarray(decMan[k].offset, decMan[k].offset + decMan[k].len);
  // alloc may grow memory (detaching the buffer), so re-read ex.memory.buffer after
  // each alloc before writing.
  const put = (arr) => { const p = ex.alloc(arr.byteLength); new Float32Array(ex.memory.buffer, p, arr.length).set(arr); return p; };
  const keys = ["embed", "l0_W", "l0_R", "l0_B", "l1_W", "l1_R", "l1_B", "encW", "encB", "predW", "predB", "outW", "outB"];
  const ptrs = keys.map((k) => put(g(k)));
  ex.set_weights(...ptrs);
  // int8 the 21MB out matrix (per-row symmetric scales) — 4× less weight traffic.
  if (int8) {
    const ow = g("outW"); // [640][8198] row-major
    const HID = 640, LOGITS = ow.length / HID;
    const q = new Int8Array(ow.length), scales = new Float32Array(HID);
    for (let n = 0; n < HID; n++) {
      let mx = 0;
      for (let m = 0; m < LOGITS; m++) { const a = Math.abs(ow[n * LOGITS + m]); if (a > mx) mx = a; }
      const sc = mx / 127 || 1;
      scales[n] = sc;
      for (let m = 0; m < LOGITS; m++) q[n * LOGITS + m] = Math.max(-127, Math.min(127, Math.round(ow[n * LOGITS + m] / sc)));
    }
    const qp = ex.alloc(q.byteLength);
    new Int8Array(ex.memory.buffer, qp, q.length).set(q);
    const sp = ex.alloc(scales.byteLength);
    new Float32Array(ex.memory.buffer, sp, scales.length).set(scales);
    ex.set_out_q(qp, sp);
  }
  return { ex, mark: ex.bump_mark(), blankId: 8192, vocab: 8193 };
}

/**
 * Decode encoder frames [Tenc*1024] (row-major, frames[t*1024+d]).
 * @returns {{ids:number[], idFrames:number[]}}
 */
/** Decode from PRE-PROJECTED frames [Tenc*640] (encoder projection done on GPU). */
export function wasmDecodeProj(dec, proj, Tenc) {
  const { ex } = dec;
  ex.reset_to(dec.mark);
  const fp = ex.alloc(proj.byteLength);
  new Float32Array(ex.memory.buffer, fp, proj.length).set(proj);
  // TDT can emit up to MAX_SYMBOLS(10) tokens per frame without advancing t —
  // size the output buffers to the true worst case, not Tenc.
  const idp = ex.alloc(Tenc * 10 * 4);
  const frp = ex.alloc(Tenc * 10 * 4);
  const n = ex.decode_proj(fp, Tenc, idp, frp);
  return { ids: Array.from(new Int32Array(ex.memory.buffer, idp, n)), idFrames: Array.from(new Int32Array(ex.memory.buffer, frp, n)) };
}

export function wasmDecode(dec, frames, Tenc) {
  const { ex } = dec;
  ex.reset_to(dec.mark); // reclaim previous window's scratch
  const fp = ex.alloc(frames.byteLength);
  new Float32Array(ex.memory.buffer, fp, frames.length).set(frames);
  const idp = ex.alloc(Tenc * 10 * 4); // MAX_SYMBOLS=10 tokens/frame worst case
  const frp = ex.alloc(Tenc * 10 * 4);
  const n = ex.decode(fp, Tenc, idp, frp);
  const ids = Array.from(new Int32Array(ex.memory.buffer, idp, n));
  const idFrames = Array.from(new Int32Array(ex.memory.buffer, frp, n));
  return { ids, idFrames };
}
