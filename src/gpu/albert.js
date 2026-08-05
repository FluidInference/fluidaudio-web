// Kokoro's PL-BERT (ALBERT) text encoder on raw WebGPU. This is the first real
// Kokoro sub-network ported to the hand-written kernels in compute.js and
// verified for numerical parity against the ONNX model (scripts/gpu-albert.mjs).
//
// Architecture (from the Kokoro ONNX graph): vocab 178, embed 128, hidden 768,
// FFN 2048, 12 weight-shared layers, 12 heads (head_dim 64), gelu_new, LN eps
// 1e-12. Weights are extracted once (scripts/extract via /tmp/kokoro/albert).
//
// Embeddings (gather + sum + LN over 128 dims) run on CPU — a lookup, the "JS for
// the rest" part — then the whole transformer stack runs GPU-resident.

const HIDDEN = 768;
const HEADS = 12;
const HEAD_DIM = HIDDEN / HEADS; // 64
const EMBED = 128;
const EPS = 1e-12;

function cpuLayerNorm(x, rows, cols, g, b, eps) {
  const y = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let mean = 0;
    for (let j = 0; j < cols; j++) mean += x[base + j];
    mean /= cols;
    let v = 0;
    for (let j = 0; j < cols; j++) {
      const d = x[base + j] - mean;
      v += d * d;
    }
    const inv = 1 / Math.sqrt(v / cols + eps);
    for (let j = 0; j < cols; j++) y[base + j] = (x[base + j] - mean) * inv * g[j] + b[j];
  }
  return y;
}

/**
 * CPU embeddings: word[ids] + pos[0..seq) + tok[0], then LayerNorm(128).
 * @returns {Float32Array} [seq, 128]
 */
export function embed(ids, w) {
  const seq = ids.length;
  const e = new Float32Array(seq * EMBED);
  for (let t = 0; t < seq; t++) {
    const wOff = ids[t] * EMBED,
      pOff = t * EMBED;
    for (let j = 0; j < EMBED; j++) {
      e[t * EMBED + j] = w.word_emb[wOff + j] + w.pos_emb[pOff + j] + w.tok_emb[j];
    }
  }
  return cpuLayerNorm(e, seq, EMBED, w.emb_ln_w, w.emb_ln_b, EPS);
}

/**
 * Run the ALBERT transformer stack on the GPU. `w` holds GPU tensors for every
 * weight (uploaded once). `embTensor` is the [seq,128] embeddings on the GPU.
 * Query is pre-scaled by 1/sqrt(head_dim) in the caller (folded into q_w/q_b).
 * @returns {import("./compute").GpuTensor} [seq, 768]
 */
export function albertForward(ctx, embTensor, w, seq, layers = 12) {
  // 128 -> 768 mapping.
  let x = ctx.matmul(embTensor, w.map_w, { bias: w.map_b }); // [seq, 768]

  for (let l = 0; l < layers; l++) {
    // --- self-attention (weights shared across layers) ---
    const Q = ctx.matmul(x, w.q_w, { bias: w.q_b }); // pre-scaled
    const K = ctx.matmul(x, w.k_w, { bias: w.k_b });
    const V = ctx.matmul(x, w.v_w, { bias: w.v_b });
    const ctxHeads = ctx.alloc(seq, HIDDEN);
    for (let h = 0; h < HEADS; h++) {
      const off = h * HEAD_DIM;
      const Qh = ctx.sliceCols(Q, off, HEAD_DIM); // [seq, 64]
      const Kh = ctx.sliceCols(K, off, HEAD_DIM);
      const Vh = ctx.sliceCols(V, off, HEAD_DIM);
      const scores = ctx.matmul(Qh, ctx.transpose(Kh)); // [seq, seq]
      const probs = ctx.softmax(scores);
      const ctxH = ctx.matmul(probs, Vh); // [seq, 64]
      ctx.setCols(ctxHeads, ctxH, off);
    }
    const proj = ctx.matmul(ctxHeads, w.dense_w, { bias: w.dense_b }); // [seq, 768]
    const attnOut = ctx.layernorm(ctx.add(proj, x), w.attn_ln_w, w.attn_ln_b, EPS);

    // --- feed-forward ---
    const h1 = ctx.matmul(attnOut, w.ffn_w, { bias: w.ffn_b, act: "gelu" }); // [seq, 2048]
    const h2 = ctx.matmul(h1, w.ffn_out_w, { bias: w.ffn_out_b }); // [seq, 768]
    x = ctx.layernorm(ctx.add(h2, attnOut), w.full_ln_w, w.full_ln_b, EPS);
  }
  return x;
}

export const ALBERT_DIMS = { HIDDEN, HEADS, HEAD_DIM, EMBED, EPS, LAYERS: 12 };
