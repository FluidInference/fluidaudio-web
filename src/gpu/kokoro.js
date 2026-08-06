// Kokoro raw-WebGPU pipeline — composes the verified kernels into the model graph.
//
// STATUS: the non-vocoder frontend up to `d_en` is wired and parity-verified
// against the ONNX model (scripts/gpu-kokoro.mjs). d_en is the 512-dim text
// encoding the whole back half (prosody predictor + iSTFTNet decoder) consumes.
//
// Remaining to reach audio (documented in docs/RAW_WEBGPU.md): the DurationEncoder
// (alternating bidir-LSTM / AdaLayerNorm-with-style blocks) → durations →
// length-regulate (gatherCols) → F0/N predictor, then the iSTFTNet decoder
// (AdaIN resblocks + ConvTranspose upsampling) and the NSF harmonic source +
// iSTFT generator tail. Every kernel those need exists and is parity-gated; the
// remaining work is faithful topology + weight extraction, not new compute.

import { embed, albertForward } from "./albert.js";

/**
 * input_ids → ALBERT text encoder → bert_encoder (768→512) → d_en.
 * @param {import("./compute").GpuContext} ctx
 * @param {Int32Array|number[]} ids
 * @param {Record<string, any>} albertW  ALBERT weights (GPU tensors + CPU embed tables), as in albert.js
 * @param {import("./compute").GpuTensor} beW  bert_encoder weight [768,512]
 * @param {import("./compute").GpuTensor} beB  bert_encoder bias [1,512]
 * @returns {import("./compute").GpuTensor} d_en [seq, 512]
 */
export function textEncoding(ctx, ids, albertW, beW, beB) {
  const seq = ids.length;
  const emb = embed(ids, albertW); // CPU gather + LN → [seq,128]
  const embT = ctx.upload(emb, seq, albertW.EMBED ?? 128);
  const albertOut = albertForward(ctx, embT, albertW, seq); // [seq,768]
  return ctx.matmul(albertOut, beW, { bias: beB }); // [seq,512]
}
