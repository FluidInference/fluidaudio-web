// Parakeet EOU 120M — RNNT greedy decode with end-of-utterance detection.
// Runtime-agnostic (pass the `ort` module: onnxruntime-web in the browser,
// onnxruntime-node in tests). Same Parakeet family as asr-parakeet, but:
//   • RNNT, not TDT → the joint has no duration bins (advance 1 frame per blank).
//   • fused decoder+joint, single-layer LSTM (pred_layers=1, hidden=640).
//   • no `target_length` input (unlike the TDT decoder_joint).
//   • special tokens: <EOU> (1024) and <EOB> (1025) live IN the vocab; the RNNT
//     blank is id 1026, one past the 1026-entry vocab. Emitting <EOU> is the
//     end-of-utterance signal a voice agent waits for.
//
// Tensor I/O (verified against ysdede/parakeet-realtime-eou-120m-v1-onnx):
//   encoder:       audio_signal[1,128,T], length[1]i64 -> outputs[1,512,Tenc]
//   decoder_joint: encoder_outputs[1,512,1], targets[1,1]i32,
//                  input_states_1/2[1,1,640] -> outputs[..,1027], output_states_1/2

const PRED_HIDDEN = 640;
const PRED_LAYERS = 1;
const MAX_SYMBOLS_PER_STEP = 10;
const BLANK_ID = 1026;
const EOU_ID = 1024;
const EOB_ID = 1025;
// 10 ms mel hop × subsampling_factor 8 = 80 ms per encoder frame.
const FRAME_SEC = 0.08;

/**
 * Tokenizer for the EOU plain-list vocab (one token per line, id = line index).
 * `▁` is the SentencePiece word-boundary marker → space. Ids ≥ 1024 are the
 * <EOU>/<EOB> control tokens and are dropped from the transcript.
 * @param {string} vocabText
 */
export function makeEouTokenizer(vocabText) {
  const id2token = vocabText.split(/\r?\n/).filter((l) => l.length > 0);
  const sanitized = id2token.map((t) => t.replace(/▁/g, " "));
  return {
    id2token,
    sanitized,
    blankId: BLANK_ID,
    eouId: EOU_ID,
    eobId: EOB_ID,
    /** @param {number[]} ids */
    decode(ids) {
      let out = "";
      for (const id of ids) {
        if (id >= EOU_ID) continue; // <EOU>/<EOB>/blank are not text
        const t = sanitized[id];
        if (t !== undefined) out += t;
      }
      return out.trim().replace(/\s+/g, " ");
    },
  };
}

/**
 * Offline RNNT greedy decode over the whole clip, emitting the transcript plus
 * the timestamps at which <EOU>/<EOB> fire.
 * @param {{ort:any, encoder:any, decoder:any, preprocessor:{nMels:number,process:(a:Float32Array)=>Promise<{features:Float32Array,length:number}>}, tokenizer:ReturnType<typeof makeEouTokenizer>, audio:Float32Array}} o
 * @returns {Promise<{text:string, tokenIds:number[], events:{type:string,time:number}[], frames:number, metrics:object}>}
 */
export async function eouTranscribe({ ort, encoder, decoder, preprocessor, tokenizer, audio }) {
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
  const t0 = now();
  const melBins = preprocessor.nMels;
  const { features, length } = await preprocessor.process(audio);
  const T = features.length / melBins;
  if (T === 0) return { text: "", tokenIds: [], events: [], frames: 0, metrics: { melMs: 0, encodeMs: 0, decodeMs: 0, totalMs: 0 } };
  const tMel = now();

  const encOut = await encoder.run({
    audio_signal: new ort.Tensor("float32", features, [1, melBins, T]),
    length: new ort.Tensor("int64", BigInt64Array.from([BigInt(length ?? T)]), [1]),
  });
  const enc = encOut["outputs"] ?? Object.values(encOut)[0];
  const [, D, Tenc] = enc.dims;
  const tEnc = now();

  // [1, D, Tenc] -> row-major [Tenc, D]
  const frames = new Float32Array(Tenc * D);
  const ed = enc.data;
  for (let t = 0; t < Tenc; t++) {
    for (let d = 0; d < D; d++) frames[t * D + d] = ed[d * Tenc + t];
  }
  enc.dispose?.();

  let st1 = new ort.Tensor("float32", new Float32Array(PRED_LAYERS * PRED_HIDDEN), [PRED_LAYERS, 1, PRED_HIDDEN]);
  let st2 = new ort.Tensor("float32", new Float32Array(PRED_LAYERS * PRED_HIDDEN), [PRED_LAYERS, 1, PRED_HIDDEN]);

  const ids = [];
  const events = [];
  const frameBuf = new Float32Array(D);
  const targets = new ort.Tensor("int32", new Int32Array(1), [1, 1]);
  let t = 0;
  let emitted = 0;

  while (t < Tenc) {
    frameBuf.set(frames.subarray(t * D, t * D + D));
    targets.data[0] = ids.length ? ids[ids.length - 1] : BLANK_ID;

    const out = await decoder.run({
      encoder_outputs: new ort.Tensor("float32", frameBuf, [1, D, 1]),
      targets,
      input_states_1: st1,
      input_states_2: st2,
    });
    const data = out["outputs"].data;
    // The joint output carries size-1 time/label dims; the last 1027 values are
    // the per-token logits for this (frame, prev-token) step.
    const V = BLANK_ID + 1;
    const off = data.length - V;
    let maxId = 0;
    let maxVal = -Infinity;
    for (let i = 0; i < V; i++) {
      const v = data[off + i];
      if (v > maxVal) {
        maxVal = v;
        maxId = i;
      }
    }

    if (maxId !== BLANK_ID) {
      st1 = out["output_states_1"] ?? st1;
      st2 = out["output_states_2"] ?? st2;
      ids.push(maxId);
      if (maxId === EOU_ID) events.push({ type: "eou", time: +(t * FRAME_SEC).toFixed(2) });
      else if (maxId === EOB_ID) events.push({ type: "eob", time: +(t * FRAME_SEC).toFixed(2) });
      emitted += 1;
      if (emitted >= MAX_SYMBOLS_PER_STEP) {
        t += 1;
        emitted = 0;
      }
    } else {
      t += 1;
      emitted = 0;
    }
  }

  const tDec = now();
  return {
    text: tokenizer.decode(ids),
    tokenIds: ids,
    events,
    frames: Tenc,
    metrics: {
      melMs: +(tMel - t0).toFixed(1),
      encodeMs: +(tEnc - tMel).toFixed(1),
      decodeMs: +(tDec - tEnc).toFixed(1),
      totalMs: +(tDec - t0).toFixed(1),
    },
  };
}
