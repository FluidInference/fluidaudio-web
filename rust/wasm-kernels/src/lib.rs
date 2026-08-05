// Raw CPU compute kernels for the WASM (non-WebGPU) backend — the exact same math
// as src/gpu/compute.js's WGSL kernels, so a browser without WebGPU runs the raw
// engines identically. Hot kernels (matmul / conv1d / int8 / int4) live here in
// wasm32 + v128 SIMD; the cheap, memory-bound ops (layernorm, softmax, transpose,
// slice, activations, adain, gather, lstm, convT) stay in JS (WasmContext).
//
// Bias + activation are applied JS-side after these kernels, so the exports below
// are the bare accumulation loops. Tensors live in wasm linear memory; JS copies
// inputs in via a bump arena (wasm_reset / wasm_alloc), calls the kernel with byte
// pointers, then reads the result back. No allocator, no std.
#![no_std]

use core::arch::wasm32::*;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// ── bump arena over linear memory (JS resets it per op) ──────────────────────
extern "C" {
    static __heap_base: u8;
}
static mut BUMP: usize = 0;

#[inline]
fn heap_base() -> usize {
    unsafe { &__heap_base as *const u8 as usize }
}

/// Reset the arena to the start of the heap. Call once before staging an op's inputs.
#[no_mangle]
pub extern "C" fn wasm_reset() {
    unsafe { BUMP = heap_base() }
}

/// Bump-allocate `n` bytes (16-byte aligned), growing linear memory as needed.
#[no_mangle]
pub extern "C" fn wasm_alloc(n: usize) -> *mut u8 {
    unsafe {
        if BUMP == 0 {
            BUMP = heap_base();
        }
        let p = (BUMP + 15) & !15;
        let end = p + n;
        let have = (memory_size(0) as usize) * 65536;
        if end > have {
            let pages = (end - have + 65535) / 65536;
            memory_grow(0, pages);
        }
        BUMP = end;
        p as *mut u8
    }
}

// ── f32 GEMM: C[M,N] = A[M,K] @ B[K,N], row-major. ───────────────────────────
// Broadcast A[i,k], vector-load B[k, j..j+4], FMA into the C row accumulator that
// stays hot across k. Vectorizes over N (contiguous). No bias/act (JS applies).
#[no_mangle]
pub extern "C" fn matmul_f32(a: *const f32, b: *const f32, c: *mut f32, m: usize, k: usize, n: usize) {
    unsafe {
        let n4 = n & !3;
        for i in 0..m {
            let crow = c.add(i * n);
            // zero the accumulator row
            let mut j = 0;
            while j < n4 {
                v128_store(crow.add(j) as *mut v128, f32x4_splat(0.0));
                j += 4;
            }
            while j < n {
                *crow.add(j) = 0.0;
                j += 1;
            }
            let arow = a.add(i * k);
            for kk in 0..k {
                let av = f32x4_splat(*arow.add(kk));
                let aik = *arow.add(kk);
                let brow = b.add(kk * n);
                let mut j = 0;
                while j < n4 {
                    let bv = v128_load(brow.add(j) as *const v128);
                    let cv = v128_load(crow.add(j) as *const v128);
                    v128_store(crow.add(j) as *mut v128, f32x4_add(cv, f32x4_mul(av, bv)));
                    j += 4;
                }
                while j < n {
                    *crow.add(j) += aik * *brow.add(j);
                    j += 1;
                }
            }
        }
    }
}

// ── 1-D conv (general: groups / stride / asymmetric pad / dilation). ──────────
// x[Cin, L] row-major; w = Cout*(Cin/groups)*K flat [co][ci_local][kk];
// y[Cout, Lout]. No bias/act (JS applies). SIMD over the Lout inner run per (co,kk).
#[no_mangle]
pub extern "C" fn conv1d_f32(
    x: *const f32, w: *const f32, y: *mut f32,
    cout: usize, cin: usize, l: usize, lout: usize, k: usize,
    stride: usize, pad_left: usize, dilation: usize, groups: usize,
) {
    unsafe {
        let cin_g = cin / groups;
        let cout_g = cout / groups;
        for co in 0..cout {
            let g = co / cout_g;
            let yrow = y.add(co * lout);
            for t in 0..lout {
                *yrow.add(t) = 0.0;
            }
            for cig in 0..cin_g {
                let ci = g * cin_g + cig;
                let xrow = x.add(ci * l);
                let wbase = (co * cin_g + cig) * k;
                for kk in 0..k {
                    let wv = *w.add(wbase + kk);
                    if wv == 0.0 {
                        continue;
                    }
                    // y[t] += x[t*stride - pad_left + kk*dilation] * wv, valid t only
                    let off = kk * dilation;
                    for t in 0..lout {
                        let src = t * stride + off;
                        if src >= pad_left {
                            let xi = src - pad_left;
                            if xi < l {
                                *yrow.add(t) += *xrow.add(xi) * wv;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── int8 GEMM: a[M,K] f32 @ dequant(wq)[K,N] -> y[M,N]. ───────────────────────
// wq: int8 weights packed 4-per-u32, row-major index [k*N + n] (little-endian
// bytes). scale[N] per output column. No bias/act (JS applies).
#[no_mangle]
pub extern "C" fn matmul_int8(
    a: *const f32, wq: *const u8, scale: *const f32, y: *mut f32,
    m: usize, n: usize, k: usize,
) {
    unsafe {
        for i in 0..m {
            let arow = a.add(i * k);
            let yrow = y.add(i * n);
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let q = *wq.add(kk * n + j) as i8;
                    acc += *arow.add(kk) * (q as f32);
                }
                *yrow.add(j) = acc * *scale.add(j);
            }
        }
    }
}

// ── int4 block-quant GEMM (ONNX MatMulNBits, bits=4, block=blocksize). ────────
// a[M,K] f32 @ dequant(bq) -> y[M,N]. bq: per output row N, nblk blocks × 16 bytes
// (32 int4 packed, low nibble = even index). scales[N*nblk] f32. zp: per row N,
// zpb = ceil(nblk/2) bytes of packed int4 zero-points (asymmetric). Matches the
// WGSL matmulNBits exactly. No bias/act.
#[no_mangle]
pub extern "C" fn matmul_nbits(
    a: *const f32, bq: *const u8, scales: *const f32, zp: *const u8, y: *mut f32,
    m: usize, n: usize, k: usize, nblk: usize, zpb: usize, blocksize: usize,
) {
    unsafe {
        for j in 0..n {
            let bq_row = bq.add(j * nblk * 16);
            let zp_row = zp.add(j * zpb);
            let sc_row = scales.add(j * nblk);
            for i in 0..m {
                let arow = a.add(i * k);
                let mut acc = 0.0f32;
                for blk in 0..nblk {
                    let scale = *sc_row.add(blk);
                    // zero-point: packed int4, even blk = low nibble
                    let zbyte = *zp_row.add(blk / 2);
                    let zpv = if blk & 1 == 0 { zbyte & 0x0f } else { zbyte >> 4 } as f32;
                    let blk_base = blk * blocksize;
                    let bq_blk = bq_row.add(blk * 16);
                    for t in 0..blocksize {
                        let kk = blk_base + t;
                        if kk >= k {
                            break;
                        }
                        let byte = *bq_blk.add(t / 2);
                        let q = if t & 1 == 0 { byte & 0x0f } else { byte >> 4 } as f32;
                        acc += *arow.add(kk) * ((q - zpv) * scale);
                    }
                }
                *y.add(i * n + j) = acc;
            }
        }
    }
}
