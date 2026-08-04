// Parakeet TDT (Token-and-Duration Transducer) offline decode. Runtime-agnostic:
// pass the `ort` module (onnxruntime-web in the browser, onnxruntime-node in
// tests) plus compiled encoder/decoder_joint sessions. Ported from the algorithm
// in parakeet.js (which follows NeMo / onnx-asr): greedy over encoder frames,
// argmax token + argmax duration, advance by the predicted duration, update the
// LSTM state only on non-blank emission.
//
// Tensor I/O (NeMo TDT export):
//   encoder:  audio_signal[1,mel,T], length[1]i64 -> outputs[1,D,Tenc]
//   decoder_joint: encoder_outputs[1,D,1], targets[1,1]i32, target_length[1]i32,
//                  input_states_1/2[2,1,640] -> outputs[vocab+numDur], output_states_1/2

const PRED_LAYERS = 2;
const PRED_HIDDEN = 640;
const MAX_TOKENS_PER_STEP = 10;

// Long audio must be windowed: the encoder output is [1, D, T/8] and a single
// full-clip pass on a 1h file is a ~12 GB tensor WebGPU can't allocate (the
// "[Conv] ... Failed to generate kernel's output [1,181672,64,256]" crash).
// Mirrors native FluidAudio's sliding window: ~15s windows, ~1s overlap, decoder
// state reset per window. Clips at/under WINDOW_SEC stay a single pass (identical
// output to before → headless verifier parity preserved).
const SAMPLE_RATE = 16000;
const WINDOW_SEC = 15;
const OVERLAP_SEC = 2;

/**
 * @param {{ort:any, encoder:any, decoder:any, preprocessor:{nMels:number,process:(a:Float32Array)=>{features:Float32Array,length:number}}, tokenizer:{id2token:string[],blankId:number,decode:(ids:number[])=>string}, audio:Float32Array}} o
 * @returns {Promise<{text:string, tokenIds:number[], frames:number}>}
 */
export async function transcribeTdt({ ort, encoder, decoder, preprocessor, tokenizer, audio }) {
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
  const t0 = now();

  const winSamples = WINDOW_SEC * SAMPLE_RATE;
  const overlapSamples = OVERLAP_SEC * SAMPLE_RATE;
  const hopSamples = winSamples - overlapSamples;
  const single = audio.length <= winSamples;

  const ids = [];
  let framesTotal = 0;
  let melMs = 0;
  let encodeMs = 0;
  let decodeMs = 0;

  for (let start = 0, w = 0; start < audio.length; start += hopSamples, w++) {
    const slice = single ? audio : audio.subarray(start, Math.min(start + winSamples, audio.length));

    const tW0 = now();
    const win = await decodeWindow({ ort, encoder, decoder, preprocessor, tokenizer, audio: slice });
    melMs += win.melMs;
    encodeMs += win.encodeMs;
    decodeMs += win.decodeMs;
    framesTotal += win.frames;

    // Drop tokens the previous window already emitted in the shared overlap. The
    // head of every window after the first re-covers the prior window's tail, so
    // estimate the overlap in tokens from the encoder frames, then prefer an exact
    // token-sequence match (tail of what's emitted == head of this window) for a
    // clean, word-aligned seam; fall back to the frame estimate when they diverge.
    let skip = 0;
    if (w > 0 && win.ids.length) {
      const overlapEnc = Math.round((win.frames * overlapSamples) / slice.length);
      let frameSkip = 0;
      while (frameSkip < win.idFrames.length && win.idFrames[frameSkip] < overlapEnc) frameSkip++;
      const maxL = Math.min(ids.length, win.ids.length, frameSkip + 8);
      let matched = 0;
      for (let L = maxL; L >= 2; L--) {
        let ok = true;
        for (let i = 0; i < L; i++) if (ids[ids.length - L + i] !== win.ids[i]) { ok = false; break; }
        if (ok) { matched = L; break; }
      }
      skip = Math.max(matched, frameSkip);
    }
    for (let k = skip; k < win.ids.length; k++) ids.push(win.ids[k]);

    if (single) break;
  }

  const tEnd = now();
  return {
    text: tokenizer.decode(ids),
    tokenIds: ids,
    frames: framesTotal,
    metrics: {
      melMs: +melMs.toFixed(1),
      encodeMs: +encodeMs.toFixed(1),
      decodeMs: +decodeMs.toFixed(1),
      totalMs: +(tEnd - t0).toFixed(1),
    },
  };
}

/**
 * Encode + TDT-greedy-decode one audio window (fresh decoder state). Returns the
 * emitted token ids plus, per token, the encoder frame it was emitted at
 * (idFrames) so the caller can trim overlap duplicates at window seams.
 * @returns {Promise<{ids:number[], idFrames:number[], frames:number, melMs:number, encodeMs:number, decodeMs:number}>}
 */
async function decodeWindow({ ort, encoder, decoder, preprocessor, tokenizer, audio }) {
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
  const melBins = preprocessor.nMels;
  const t0 = now();
  const { features, length } = await preprocessor.process(audio);
  const T = features.length / melBins;
  if (T === 0) return { ids: [], idFrames: [], frames: 0, melMs: +(now() - t0).toFixed(1), encodeMs: 0, decodeMs: 0 };
  const tMel = now();

  const encOut = await encoder.run({
    audio_signal: new ort.Tensor("float32", features, [1, melBins, T]),
    length: new ort.Tensor("int64", BigInt64Array.from([BigInt(length ?? T)]), [1]),
  });
  const enc = encOut["outputs"] ?? Object.values(encOut)[0];
  const [, D, Tenc] = enc.dims;
  const tEnc = now();

  // [1, D, Tenc] -> [Tenc, D]
  const frames = new Float32Array(Tenc * D);
  const ed = enc.data;
  for (let t = 0; t < Tenc; t++) {
    for (let d = 0; d < D; d++) frames[t * D + d] = ed[d * Tenc + t];
  }
  enc.dispose?.();

  const blankId = tokenizer.blankId;
  const vocab = tokenizer.id2token.length;
  const stateSize = PRED_LAYERS * PRED_HIDDEN;
  let st1 = new ort.Tensor("float32", new Float32Array(stateSize), [PRED_LAYERS, 1, PRED_HIDDEN]);
  let st2 = new ort.Tensor("float32", new Float32Array(stateSize), [PRED_LAYERS, 1, PRED_HIDDEN]);

  const ids = [];
  const idFrames = [];
  const frameBuf = new Float32Array(D);
  const targets = new ort.Tensor("int32", new Int32Array(1), [1, 1]);
  const targetLen = new ort.Tensor("int32", Int32Array.from([1]), [1]);
  let t = 0;
  let emitted = 0;

  while (t < Tenc) {
    frameBuf.set(frames.subarray(t * D, t * D + D));
    targets.data[0] = ids.length ? ids[ids.length - 1] : blankId;

    const out = await decoder.run({
      encoder_outputs: new ort.Tensor("float32", frameBuf, [1, D, 1]),
      targets,
      target_length: targetLen,
      input_states_1: st1,
      input_states_2: st2,
    });
    const data = out["outputs"].data;

    let maxId = 0;
    let maxVal = -Infinity;
    for (let i = 0; i < vocab; i++) if (data[i] > maxVal) { maxVal = data[i]; maxId = i; }
    let step = 0;
    let durMax = -Infinity;
    for (let i = vocab; i < data.length; i++) if (data[i] > durMax) { durMax = data[i]; step = i - vocab; }

    if (maxId !== blankId) {
      st1 = out["output_states_1"] ?? st1;
      st2 = out["output_states_2"] ?? st2;
      ids.push(maxId);
      idFrames.push(t);
      emitted += 1;
    }

    if (step > 0) {
      t += step;
      emitted = 0;
    } else if (maxId === blankId || emitted >= MAX_TOKENS_PER_STEP) {
      t += 1;
      emitted = 0;
    }
  }

  const tDec = now();
  return {
    ids,
    idFrames,
    frames: Tenc,
    melMs: +(tMel - t0).toFixed(1),
    encodeMs: +(tEnc - tMel).toFixed(1),
    decodeMs: +(tDec - tEnc).toFixed(1),
  };
}
