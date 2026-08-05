// Whisper log-mel spectrogram (JS, no onnxruntime). 80-bin, n_fft 400, hop 160,
// Hann window (periodic), reflect-pad n_fft/2 each side, power spectrum → mel
// filterbank [201,80] → log10 → clamp(max-8) → (x+4)/4. Matches WhisperFeatureExtractor.
// n_fft=400 is not power-of-2, so a direct 400-point DFT (precomputed cos/sin) is used —
// fine for a one-shot 30s mel (~1s). Audio is padded/trimmed to 30s (480000 samples).

const N_FFT = 400,
  HOP = 160,
  N_MEL = 80,
  N_FREQ = 201,
  N_SAMPLES = 480000,
  N_FRAMES = 3000;

export class WhisperMel {
  /** @param {Float32Array} melFilters [201*80] row-major (freq-major). */
  constructor(melFilters) {
    this.filters = melFilters;
    // Hann window (periodic): w[n] = 0.5 - 0.5*cos(2π n / N)
    this.win = new Float32Array(N_FFT);
    for (let n = 0; n < N_FFT; n++) this.win[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
    // DFT tables: cos/sin[k*N_FFT + n], k in [0,201), n in [0,400)
    this.cos = new Float32Array(N_FREQ * N_FFT);
    this.sin = new Float32Array(N_FREQ * N_FFT);
    for (let k = 0; k < N_FREQ; k++) {
      for (let n = 0; n < N_FFT; n++) {
        const a = (-2 * Math.PI * k * n) / N_FFT;
        this.cos[k * N_FFT + n] = Math.cos(a);
        this.sin[k * N_FFT + n] = Math.sin(a);
      }
    }
  }

  /** audio: Float32Array mono 16k → {features: Float32Array[80*3000], length}. */
  process(audio) {
    // pad/trim to 30s, then reflect-pad N_FFT/2 each side (center STFT).
    const pad = N_FFT >> 1;
    const buf = new Float32Array(N_SAMPLES + 2 * pad);
    const n = Math.min(audio.length, N_SAMPLES);
    for (let i = 0; i < n; i++) buf[pad + i] = audio[i];
    for (let i = 0; i < pad; i++) {
      buf[pad - 1 - i] = buf[pad + 1 + i];
      buf[pad + N_SAMPLES + i] = buf[pad + N_SAMPLES - 2 - i];
    }

    const power = new Float32Array(N_FREQ * N_FRAMES);
    const frame = new Float32Array(N_FFT);
    for (let t = 0; t < N_FRAMES; t++) {
      const off = t * HOP;
      for (let i = 0; i < N_FFT; i++) frame[i] = buf[off + i] * this.win[i];
      for (let k = 0; k < N_FREQ; k++) {
        let re = 0,
          im = 0;
        const cb = k * N_FFT;
        for (let i = 0; i < N_FFT; i++) {
          re += frame[i] * this.cos[cb + i];
          im += frame[i] * this.sin[cb + i];
        }
        power[k * N_FRAMES + t] = re * re + im * im;
      }
    }
    // mel = filters^T @ power → [80,3000]; log10; normalize.
    const mel = new Float32Array(N_MEL * N_FRAMES);
    let maxv = -Infinity;
    for (let m = 0; m < N_MEL; m++) {
      for (let t = 0; t < N_FRAMES; t++) {
        let s = 0;
        for (let f = 0; f < N_FREQ; f++) s += this.filters[f * N_MEL + m] * power[f * N_FRAMES + t];
        let v = Math.log10(Math.max(s, 1e-10));
        mel[m * N_FRAMES + t] = v;
        if (v > maxv) maxv = v;
      }
    }
    const floor = maxv - 8.0;
    for (let i = 0; i < mel.length; i++) mel[i] = (Math.max(mel[i], floor) + 4.0) / 4.0;
    return { features: mel, length: N_FRAMES };
  }
}
