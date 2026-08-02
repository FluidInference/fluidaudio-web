// NeMo log-mel preprocessor as an ONNX graph (nemo128.onnx) run on
// onnxruntime-web — so feature extraction is WASM/WebGPU, not JS. Replaces the
// JS FFT path. Same {features, length} contract the TDT decoder expects.
//   waveforms[1,N] + waveforms_lens[1] -> features[1,128,T] + features_lens[1]

export class OnnxMelPreprocessor {
  /** @param {any} ort @param {any} session nemo128 session @param {number} nMels */
  constructor(ort, session, nMels = 128) {
    this.ort = ort;
    this.session = session;
    this.nMels = nMels;
  }

  /** @param {Float32Array} audio @returns {Promise<{features:Float32Array, length:number}>} */
  async process(audio) {
    const waveforms = new this.ort.Tensor("float32", audio, [1, audio.length]);
    const lens = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(audio.length)]), [1]);
    const out = await this.session.run({ waveforms, waveforms_lens: lens });
    waveforms.dispose?.();
    lens.dispose?.();
    const feat = out["features"];
    const T = feat.dims[2];
    const features = feat.data instanceof Float32Array ? feat.data.slice() : Float32Array.from(feat.data);
    const lenOut = out["features_lens"];
    const length = lenOut ? Number(lenOut.data[0]) : T;
    feat.dispose?.();
    lenOut?.dispose?.();
    return { features, length };
  }
}
