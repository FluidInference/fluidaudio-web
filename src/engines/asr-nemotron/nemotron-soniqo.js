// Nemotron 3.5 streaming RNN-T decode for the soniqo FP16 export
// (soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16). fp16 weights,
// fp32 I/O, purpose-built to run under onnxruntime-web's WebGPU EP (int4 can't).
//
// Streaming contract (config.json): 320 ms chunks = 32 mel frames (mel-major
// [1,128,32]) → 4 encoded frames. Per chunk we thread the encoder caches
// (pre_cache + cache_last_channel/time + len) output→input, and run RNN-T greedy
// over the 4 frames with a 2-layer LSTM prediction net (token + h/c [2,1,640]).
// Verified headless (ort-node): input_ids clip → exact transcript.

const BLANK = 13087;
const MEL = 128;
const CHUNK_FRAMES = 32;

/** vocab.json is { "0":"<unk>", ... }. `▁` → space; drop <…> control/lang tokens. */
export function makeSoniqoTokenizer(vocabJson) {
  const id2tok = vocabJson; // object indexed by string id
  return {
    decode(ids) {
      let out = "";
      for (const id of ids) {
        const t = id2tok[id];
        if (t === undefined || t.startsWith("<")) continue;
        out += t.replace(/▁/g, " ");
      }
      return out.trim().replace(/\s+/g, " ");
    },
  };
}

/** promptDictionary maps a lang code → prompt slot (en-US → 0). */
export function soniqoLangPrompt(languages, lang) {
  const d = languages.promptDictionary || {};
  return d[lang] ?? d["en-US"] ?? 0;
}

/**
 * @param {{ort:any, encoder:any, decoder:any, joint:any,
 *          preprocessor:{nMels:number,process:(a:Float32Array)=>{features:Float32Array,length:number}},
 *          tokenizer:{decode:(ids:number[])=>string}, audio:Float32Array, langPrompt:number}} o
 */
export async function soniqoTranscribe({ ort, encoder, decoder, joint, preprocessor, tokenizer, audio, langPrompt = 0 }) {
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
  const t0 = now();
  const { features } = await preprocessor.process(audio); // mel-major [MEL, T]
  const T = features.length / MEL;
  if (T === 0) return { text: "", tokenIds: [], metrics: { totalMs: +(now() - t0).toFixed(1) } };

  const lm = new Float32Array(128); lm[langPrompt] = 1; // language one-hot
  const langMask = new ort.Tensor("float32", lm, [1, 128]);

  // carried encoder caches (init zero)
  let pre = new Float32Array(MEL * 9);
  let clc = new Float32Array(24 * 56 * 1024);
  let clt = new Float32Array(24 * 1024 * 8);
  let clen = 0;

  // RNN-T prediction state
  let token = BLANK;
  let h = new Float32Array(2 * 640);
  let c = new Float32Array(2 * 640);
  const ids = [];

  const chunkBuf = new Float32Array(MEL * CHUNK_FRAMES);
  const nChunks = Math.ceil(T / CHUNK_FRAMES);
  for (let ci = 0; ci < nChunks; ci++) {
    const valid = Math.min(CHUNK_FRAMES, T - ci * CHUNK_FRAMES);
    chunkBuf.fill(0);
    for (let m = 0; m < MEL; m++) {
      for (let t = 0; t < valid; t++) chunkBuf[m * CHUNK_FRAMES + t] = features[m * T + ci * CHUNK_FRAMES + t];
    }
    const eo = await encoder.run({
      audio_signal: new ort.Tensor("float32", chunkBuf.slice(), [1, 128, 32]),
      audio_length: new ort.Tensor("int32", Int32Array.from([valid]), [1]),
      language_mask: langMask,
      pre_cache: new ort.Tensor("float32", pre, [1, 128, 9]),
      cache_last_channel: new ort.Tensor("float32", clc, [24, 1, 56, 1024]),
      cache_last_time: new ort.Tensor("float32", clt, [24, 1, 1024, 8]),
      cache_last_channel_len: new ort.Tensor("int32", Int32Array.from([clen]), [1]),
    });
    pre = eo["new_pre_cache"].data.slice();
    clc = eo["new_cache_last_channel"].data.slice();
    clt = eo["new_cache_last_time"].data.slice();
    clen = Number(eo["new_cache_last_channel_len"].data[0]);

    const enc = eo["encoded_output"].data;
    const nf = eo["encoded_output"].dims[1];
    const nlen = eo["encoded_length"] ? Number(eo["encoded_length"].data[0]) : nf;
    const encF = new Float32Array(1024);
    for (let f = 0; f < Math.min(nf, nlen || nf); f++) {
      encF.set(enc.subarray(f * 1024, f * 1024 + 1024));
      const encTensor = new ort.Tensor("float32", encF.slice(), [1, 1, 1024]);
      let emitted = 0;
      while (emitted < 10) {
        const dr = await decoder.run({
          token: new ort.Tensor("int64", BigInt64Array.from([BigInt(token)]), [1, 1]),
          h: new ort.Tensor("float32", h, [2, 1, 640]),
          c: new ort.Tensor("float32", c, [2, 1, 640]),
        });
        const jr = await joint.run({ encoder_output: encTensor, decoder_output: dr["decoder_output"] });
        const lg = jr["logits"].data;
        let mi = 0, mv = -Infinity;
        for (let i = 0; i < 13088; i++) if (lg[i] > mv) { mv = lg[i]; mi = i; }
        const blank = mi === BLANK;
        if (!blank) { ids.push(mi); token = mi; h = dr["h_out"].data.slice(); c = dr["c_out"].data.slice(); emitted++; }
        // release per-step ORT outputs (WASM native memory) — h_out/c_out already copied
        dr["decoder_output"].dispose?.(); dr["h_out"].dispose?.(); dr["c_out"].dispose?.(); jr["logits"].dispose?.();
        if (blank) break;
      }
      encTensor.dispose?.();
    }
    for (const k of Object.keys(eo)) eo[k].dispose?.(); // release encoder outputs (cache copies taken above)
  }
  return { text: tokenizer.decode(ids), tokenIds: ids, metrics: { totalMs: +(now() - t0).toFixed(1) } };
}
