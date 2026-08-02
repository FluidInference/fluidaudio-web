// Nemotron 3.5 cache-aware streaming RNNT decode. Runtime-agnostic (pass ort +
// encoder/decoder/joint sessions + NA mel preprocessor + tokenizer).
//
// BUILD-ONLY: structurally implemented from the verified I/O + configs. int4
// accuracy needs WebGPU (CPU/WASM degenerate, like Parakeet int8) — the shape
// flow + cache/state threading verify headless; transcript correctness does not.
//
// Per chunk: audio_signal[1,65,128] = 9 pre-encode-cache mel frames + 56 new,
// T-major, NA log-mel; encoder threads cache_last_channel/time; RNNT greedy over
// the 7 encoder frames using decoder(LSTM)+joint, decoder state carried across
// the whole utterance.

const MEL = 128, NEW = 56, PRECACHE = 9, CHUNK_FR = NEW + PRECACHE; // 65
const LAYERS = 24, LEFT = 56, DIM = 1024, TIMEC = 8;
const PRED_LAYERS = 2, PRED_HIDDEN = 640;
const VOCAB = 13088, BLANK = 13087, MAX_SYM = 10;

const i64 = (ort, v) => new ort.Tensor("int64", BigInt64Array.from([BigInt(v)]), [1]);
const zeros = (ort, dims) => new ort.Tensor("float32", new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);

async function runDecoder(ort, decoder, tok, h, c) {
  const out = await decoder.run({
    targets: new ort.Tensor("int64", BigInt64Array.from([BigInt(tok)]), [1, 1]),
    h_in: h,
    c_in: c,
  });
  const dOut = out["decoder_output"]; // [1,640,1]
  const d = new Float32Array(PRED_HIDDEN);
  for (let i = 0; i < PRED_HIDDEN; i++) d[i] = dOut.data[i];
  dOut.dispose?.();
  return { decOutT: new ort.Tensor("float32", d, [1, 1, PRED_HIDDEN]), h: out["h_out"], c: out["c_out"] };
}

/**
 * @param {{ort:any, encoder:any, decoder:any, joint:any,
 *          preprocessor:{process:(a:Float32Array)=>{features:Float32Array,length:number}},
 *          tokenizer:{decode:(ids:number[])=>string}, audio:Float32Array, langId?:number}} o
 * @returns {Promise<{text:string, tokenIds:number[], chunks:number}>}
 */
export async function nemotronTranscribe({ ort, encoder, decoder, joint, preprocessor, tokenizer, audio, langId = 0 }) {
  const { features } = preprocessor.process(audio); // mel-major [128*T], NA
  const T = Math.floor(features.length / MEL);
  const melAt = (t, m) => (t >= 0 && t < T ? features[m * T + t] : 0);

  // encoder streaming caches
  let cc = zeros(ort, [1, LAYERS, LEFT, DIM]);
  let ct = zeros(ort, [1, LAYERS, DIM, TIMEC]);
  let ccl = i64(ort, 0);
  const lang = i64(ort, langId);

  // decoder LSTM state; RNNT predictor primed with BLANK/SOS
  let h = zeros(ort, [PRED_LAYERS, 1, PRED_HIDDEN]);
  let c = zeros(ort, [PRED_LAYERS, 1, PRED_HIDDEN]);
  let dec = await runDecoder(ort, decoder, BLANK, h, c);
  let decOutT = dec.decOutT; h = dec.h; c = dec.c;

  const ids = [];
  const nChunks = Math.max(1, Math.ceil(T / NEW));
  const chunkBuf = new Float32Array(CHUNK_FR * MEL);

  for (let ci = 0; ci < nChunks; ci++) {
    // audio_signal[1,65,128] T-major: [9 prev | 56 new] (zeros before start)
    for (let f = 0; f < CHUNK_FR; f++) {
      const t = ci * NEW - PRECACHE + f;
      for (let m = 0; m < MEL; m++) chunkBuf[f * MEL + m] = melAt(t, m);
    }
    const eo = await encoder.run({
      audio_signal: new ort.Tensor("float32", chunkBuf.slice(), [1, CHUNK_FR, MEL]),
      length: i64(ort, CHUNK_FR),
      cache_last_channel: cc, cache_last_time: ct, cache_last_channel_len: ccl, lang_id: lang,
    });
    const enc = eo["outputs"]; // [1,7,1024]
    cc = eo["cache_last_channel_next"]; ct = eo["cache_last_time_next"]; ccl = eo["cache_last_channel_len_next"];
    const nEnc = enc.dims[1];

    for (let t = 0; t < nEnc; t++) {
      const encFrame = new Float32Array(DIM);
      for (let d = 0; d < DIM; d++) encFrame[d] = enc.data[t * DIM + d];
      const encT = new ort.Tensor("float32", encFrame, [1, 1, DIM]);
      let sym = 0;
      while (sym < MAX_SYM) {
        const jo = await joint.run({ encoder_output: encT, decoder_output: decOutT });
        const logits = jo["joint_output"].data; // [1,1,1,13088]
        let maxId = 0, maxVal = -Infinity;
        for (let i = 0; i < VOCAB; i++) if (logits[i] > maxVal) { maxVal = logits[i]; maxId = i; }
        jo["joint_output"].dispose?.();
        if (maxId === BLANK) break;
        ids.push(maxId);
        dec = await runDecoder(ort, decoder, maxId, h, c);
        decOutT = dec.decOutT; h = dec.h; c = dec.c;
        sym++;
      }
    }
    enc.dispose?.();
  }
  return { text: tokenizer.decode(ids), tokenIds: ids, chunks: nChunks };
}

/** vocab.txt = one token per line, id = line index. */
export function makeNemotronTokenizer(vocabText) {
  const id2token = vocabText.split(/\r?\n/);
  return {
    id2token,
    decode(ids) {
      let out = "";
      for (const id of ids) {
        const t = id2token[id];
        if (!t || (t.startsWith("<") && t.endsWith(">"))) continue; // skip <blank>/<lang>/<unk>
        out += t.replace(/▁/g, " ");
      }
      return out.trim().replace(/\s+/g, " ");
    },
  };
}
