// VoiceChat-11B STT FastConformer config — shared between the engine and the CI
// smoke so the smoke always exercises the shipped configuration.
//
// NeMo geometry: att_context_size [70, 0], att_context_style chunked_limited →
// chunk grid = att_context[1] + 1 = 1, i.e. FULLY per-frame causal: query i
// attends [i−70, i]. (The model card's "chunk_size 8" streaming option is a push
// granularity for the cache-aware loop, not a mask property — with right
// context 0 any shift-in size computes the identical function.) attChunk 1 +
// attLeft 70 + attRight 0 reproduces that mask exactly in raw-encoder.js.
// Causal dw_striding subsampling pad {t:2,b:1,l:2,r:1} (k−1 / stride−1, same as
// Nemotron/EOU) and causal depthwise conv (k=9, all pad left).
export const VOICECHAT_CFG = { melBins: 128, subPad: { t: 2, b: 1, l: 2, r: 1 }, convCausal: true, attChunk: 1, attLeft: 70, attRight: 0 };
