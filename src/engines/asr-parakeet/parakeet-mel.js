// Parakeet log-mel frontend — pure JS (no onnxruntime). Replaces onnxMel.js
// (nemo128.onnx). Same NeMo pipeline as the Nemotron mel but with Parakeet's
// per_feature normalization (CMVN) and 2^-24 log guard. Output {features, length}
// matches the ONNX nemo128 preprocessor (feeds the TDT encoder).

import { JsPreprocessor } from "../asr-nemotron/nemotron-mel.js";

export class ParakeetMel {
  constructor(nMels = 128) {
    this.nMels = nMels;
    // Parakeet nemo128: log guard 2^-24 + per-feature CMVN.
    this.pre = new JsPreprocessor({ nMels, logGuard: 2 ** -24 });
  }

  /** @param {Float32Array} audio mono 16kHz -> {features:[nMels*T], length} */
  process(audio) {
    const { rawMel, nFrames, featuresLen } = this.pre.computeRawMel(audio);
    if (featuresLen === 0) return { features: new Float32Array(0), length: 0 };
    const features = this.pre.normalizeFeatures(rawMel, nFrames, featuresLen);
    return { features, length: featuresLen };
  }
}
