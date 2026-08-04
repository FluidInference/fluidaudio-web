// Parakeet TDT decoder + joint in Rust → wasm32 SIMD. CPU decode with NO GPU
// round-trips (the RNNT autoregressive sync wall is why the WebGPU decoder capped
// at ~20×). The joint's 640→8198 matmul autovectorizes to v128 with +simd128.
//
// Layout matches the JS decoder / decoder_joint-model.onnx:
//   embed[8193,640], LSTM ×2 (W/R[1,2560,640] iofc, B[1,5120]),
//   joint enc[1024,640]+bias, pred[640,640]+bias, out[640,8198]+bias.
// Weights live in wasm linear memory (JS copies them once, passes pointers).
#![no_std]

use core::arch::wasm32::*;

const HID: usize = 640;
const ENC_D: usize = 1024;
const VOCAB: usize = 8193;
const LOGITS: usize = 8198;
const BLANK: usize = VOCAB - 1;
const MAX_SYMBOLS: i32 = 10;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

struct W {
    embed: *const f32,
    lw: [*const f32; 2], lr: [*const f32; 2], lb: [*const f32; 2],
    enc_w: *const f32, enc_b: *const f32,
    pred_w: *const f32, pred_b: *const f32,
    out_w: *const f32, out_b: *const f32,
}
static mut WT: W = W {
    embed: core::ptr::null(), lw: [core::ptr::null(); 2], lr: [core::ptr::null(); 2],
    lb: [core::ptr::null(); 2], enc_w: core::ptr::null(), enc_b: core::ptr::null(),
    pred_w: core::ptr::null(), pred_b: core::ptr::null(), out_w: core::ptr::null(), out_b: core::ptr::null(),
};

// Scratch (fixed sizes — no allocator).
static mut J: [f32; HID] = [0.0; HID];
static mut OUT: [f32; LOGITS] = [0.0; LOGITS];
static mut ENCPROJ: [f32; HID] = [0.0; HID];
static mut PREDPROJ: [f32; HID] = [0.0; HID];
static mut H: [[f32; HID]; 2] = [[0.0; HID]; 2];
static mut C: [[f32; HID]; 2] = [[0.0; HID]; 2];
static mut NH: [[f32; HID]; 2] = [[0.0; HID]; 2];
static mut NC: [[f32; HID]; 2] = [[0.0; HID]; 2];
static mut DECOUT: [f32; HID] = [0.0; HID];

#[inline] fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + expf(-x)) }
// minimal expf / tanhf (no libm in no_std). Accurate enough for gates.
#[inline] fn expf(x: f32) -> f32 {
    // clamp then exp via f64-free poly: use core exp through repeated... use a simple approx.
    let xc = if x > 30.0 { 30.0 } else if x < -30.0 { -30.0 } else { x };
    // e^x = 2^(x/ln2); split. Use scalar via libcore's powi-free approach:
    let t = xc * 1.4426950408889634; // x/ln2
    let fl = floorf(t);
    let frac = t - fl;
    // 2^frac ≈ poly (minimax deg4)
    let p = 1.0 + frac * (0.6931472 + frac * (0.2402265 + frac * (0.0555041 + frac * 0.0096181)));
    ldexpf(p, fl as i32)
}
#[inline] fn floorf(x: f32) -> f32 { let i = x as i32 as f32; if i > x { i - 1.0 } else { i } }
#[inline] fn ldexpf(x: f32, e: i32) -> f32 {
    // x * 2^e via bit manipulation of the exponent (e in sane range).
    let mut r = x; let mut n = e;
    while n > 30 { r *= 1073741824.0; n -= 30; }
    while n < -30 { r *= 9.313225746154785e-10; n += 30; }
    r * f32::from_bits(((127 + n) as u32) << 23)
}
#[inline] fn tanhf(x: f32) -> f32 { let e = expf(2.0 * x); (e - 1.0) / (e + 1.0) }

// Simple bump allocator over wasm linear memory (JS allocates weight/frame/output
// buffers, then calls the exported functions with the returned pointers). Starts
// past the crate's static data and grows memory as needed.
static mut BUMP: usize = 1 << 20;
#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    unsafe {
        let ptr = BUMP;
        BUMP += (size + 15) & !15;
        let need_pages = (BUMP + 65535) / 65536;
        let have = core::arch::wasm32::memory_size(0);
        if need_pages > have { core::arch::wasm32::memory_grow(0, need_pages - have); }
        ptr as *mut u8
    }
}
/// Reset the bump pointer to reclaim per-window scratch (weights are allocated
/// first and kept below this mark by the caller).
#[no_mangle]
pub extern "C" fn reset_to(mark: usize) { unsafe { BUMP = mark; } }
#[no_mangle]
pub extern "C" fn bump_mark() -> usize { unsafe { BUMP } }

#[no_mangle]
pub extern "C" fn set_weights(
    embed: *const f32, l0w: *const f32, l0r: *const f32, l0b: *const f32,
    l1w: *const f32, l1r: *const f32, l1b: *const f32,
    enc_w: *const f32, enc_b: *const f32, pred_w: *const f32, pred_b: *const f32,
    out_w: *const f32, out_b: *const f32,
) {
    unsafe {
        WT = W { embed, lw: [l0w, l1w], lr: [l0r, l1r], lb: [l0b, l1b],
            enc_w, enc_b, pred_w, pred_b, out_w, out_b };
    }
}

// One LSTM layer step (ONNX iofc), reading h/c[layer], writing nh/nc[layer].
unsafe fn lstm_step(layer: usize, x: *const f32) {
    let w = WT.lw[layer]; let r = WT.lr[layer]; let b = WT.lb[layer];
    let h = &H[layer]; let c = &C[layer];
    for g in 0..HID {
        let wi = g * HID; let wo = (HID + g) * HID; let wf = (2 * HID + g) * HID; let wc = (3 * HID + g) * HID;
        let mut zi = *b.add(g) + *b.add(4 * HID + g);
        let mut zo = *b.add(HID + g) + *b.add(5 * HID + g);
        let mut zf = *b.add(2 * HID + g) + *b.add(6 * HID + g);
        let mut zc = *b.add(3 * HID + g) + *b.add(7 * HID + g);
        for j in 0..HID {
            let xj = *x.add(j);
            zi += *w.add(wi + j) * xj; zo += *w.add(wo + j) * xj;
            zf += *w.add(wf + j) * xj; zc += *w.add(wc + j) * xj;
        }
        for j in 0..HID {
            let hj = h[j];
            zi += *r.add(wi + j) * hj; zo += *r.add(wo + j) * hj;
            zf += *r.add(wf + j) * hj; zc += *r.add(wc + j) * hj;
        }
        let cc = sigmoid(zf) * c[g] + sigmoid(zi) * tanhf(zc);
        NC[layer][g] = cc; NH[layer][g] = sigmoid(zo) * tanhf(cc);
    }
}

// Prediction net for `token` from current H/C → DECOUT + candidate NH/NC.
unsafe fn predict(token: usize) {
    let emb = WT.embed.add(token * HID);
    lstm_step(0, emb);
    lstm_step(1, NH[0].as_ptr());
    for i in 0..HID { DECOUT[i] = NH[1][i]; }
}
unsafe fn commit_state() { H = NH; C = NC; }

// pred_proj = decOut @ predW + predB (cached across blank frames).
unsafe fn compute_predproj() {
    let w = WT.pred_w; let b = WT.pred_b;
    for n in 0..HID {
        let mut s = *b.add(n);
        for k in 0..HID { s += DECOUT[k] * *w.add(k * HID + n); }
        PREDPROJ[n] = s;
    }
}

// joint for encoder frame `frame_ptr`: enc_proj + pred_proj → relu → OUT[8198].
// The out matmul (640→8198) is the hot loop; SIMD over the logits dimension.
unsafe fn joint(frame: *const f32) {
    let ew = WT.enc_w; let eb = WT.enc_b;
    for n in 0..HID {
        let mut s = *eb.add(n);
        for k in 0..ENC_D { s += *frame.add(k) * *ew.add(k * HID + n); }
        ENCPROJ[n] = s;
    }
    for n in 0..HID { let v = ENCPROJ[n] + PREDPROJ[n]; J[n] = if v > 0.0 { v } else { 0.0 }; }
    // OUT[m] = outB[m] + Σ_n J[n] * outW[n*LOGITS+m] — axpy over m (v128 x4).
    let ow = WT.out_w; let ob = WT.out_b;
    for m in 0..LOGITS { OUT[m] = *ob.add(m); }
    let mm4 = LOGITS & !3;
    for n in 0..HID {
        let jn = J[n];
        if jn == 0.0 { continue; }
        let jv = f32x4_splat(jn);
        let row = ow.add(n * LOGITS);
        let mut m = 0;
        while m < mm4 {
            let acc = v128_load(OUT.as_ptr().add(m) as *const v128);
            let w = v128_load(row.add(m) as *const v128);
            v128_store(OUT.as_mut_ptr().add(m) as *mut v128, f32x4_add(acc, f32x4_mul(jv, w)));
            m += 4;
        }
        while m < LOGITS { OUT[m] += jn * *row.add(m); m += 1; }
    }
}

/// Greedy TDT decode. frames:[Tenc,1024] row-major. Writes token ids to `out`,
/// returns count. Resets state internally.
#[no_mangle]
pub extern "C" fn decode(frames: *const f32, tenc: u32, out_ids: *mut i32, out_frames: *mut i32) -> i32 {
    unsafe {
        H = [[0.0; HID]; 2]; C = [[0.0; HID]; 2];
        let tenc = tenc as usize;
        let mut n_out = 0i32;
        let mut last_tok = BLANK;
        predict(last_tok); compute_predproj();
        let mut t = 0usize;
        let mut emitted = 0i32;
        while t < tenc {
            joint(frames.add(t * ENC_D));
            // argmax token over [0,VOCAB), duration over [VOCAB,LOGITS)
            let mut max_id = 0usize; let mut max_v = f32::NEG_INFINITY;
            for i in 0..VOCAB { if OUT[i] > max_v { max_v = OUT[i]; max_id = i; } }
            let mut step = 0usize; let mut dv = f32::NEG_INFINITY;
            for i in VOCAB..LOGITS { if OUT[i] > dv { dv = OUT[i]; step = i - VOCAB; } }
            if max_id != BLANK {
                commit_state(); last_tok = max_id;
                *out_ids.add(n_out as usize) = max_id as i32;
                *out_frames.add(n_out as usize) = t as i32;
                n_out += 1; emitted += 1;
                predict(last_tok); compute_predproj();
            }
            if step > 0 { t += step; emitted = 0; }
            else if max_id == BLANK || emitted >= MAX_SYMBOLS { t += 1; emitted = 0; }
        }
        n_out
    }
}
