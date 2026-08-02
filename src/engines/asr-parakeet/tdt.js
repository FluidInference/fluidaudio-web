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

/**
 * @param {{ort:any, encoder:any, decoder:any, preprocessor:{nMels:number,process:(a:Float32Array)=>{features:Float32Array,length:number}}, tokenizer:{id2token:string[],blankId:number,decode:(ids:number[])=>string}, audio:Float32Array}} o
 * @returns {Promise<{text:string, tokenIds:number[], frames:number}>}
 */
export async function transcribeTdt({ ort, encoder, decoder, preprocessor, tokenizer, audio }) {
  const melBins = preprocessor.nMels;
  const { features, length } = await preprocessor.process(audio);
  const T = features.length / melBins;

  const encOut = await encoder.run({
    audio_signal: new ort.Tensor("float32", features, [1, melBins, T]),
    length: new ort.Tensor("int64", BigInt64Array.from([BigInt(length ?? T)]), [1]),
  });
  const enc = encOut["outputs"] ?? Object.values(encOut)[0];
  const [, D, Tenc] = enc.dims;

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

  return { text: tokenizer.decode(ids), tokenIds: ids, frames: Tenc };
}
