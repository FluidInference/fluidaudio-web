// Streaming NA log-mel with EXACT parity to JsPreprocessor.process() on the
// full signal. A mel frame t reads pre-emphasized samples [t·160−256, t·160+256)
// of the center-padded signal, so frame t is FINAL once t·160+256 real samples
// exist — later audio can't change it. push() emits final frames by recomputing
// over a hop-aligned slice and discarding WARM warmup frames (their windows
// cross the slice's virtual left pad, where the full signal has real samples);
// consumed frames never touch slice sample 0, so the per-slice pre-emphasis
// seam (x[-1]=0) is invisible. flush() emits the right-padded tail, matching
// the offline featuresLen = floor(N/160).
import { JsPreprocessor } from "./nemotron-mel.js";

const HOP = 160;
const HALF = 256; // N_FFT/2 center pad
const WARM = 2; // ceil(HALF/HOP): slice frames whose window crosses the left edge

export class StreamingMel {
  constructor(nMels = 128) {
    this.pre = new JsPreprocessor({ nMels });
    this.nMels = nMels;
    this.buf = new Float32Array(16000);
    this.len = 0; // valid samples in buf
    this.base = 0; // global sample index of buf[0] (hop-aligned)
    this.frames = 0; // mel frames emitted so far (global)
  }

  /** Total samples pushed so far. */
  get samples() {
    return this.base + this.len;
  }

  _append(samples) {
    if (this.len + samples.length > this.buf.length) {
      const nb = new Float32Array(Math.max(this.buf.length * 2, this.len + samples.length));
      nb.set(this.buf.subarray(0, this.len));
      this.buf = nb;
    }
    this.buf.set(samples, this.len);
    this.len += samples.length;
  }

  /** Push samples; returns { data, count } — `count` new FINAL mel frames,
   * mel-major [nMels × count] (data[c·count + t]), or count 0. */
  push(samples) {
    this._append(samples);
    const lastFinal = Math.floor((this.samples - HALF) / HOP); // inclusive
    const n = lastFinal + 1 - this.frames;
    if (n <= 0) return { data: null, count: 0 };
    const out = this._emit(n);
    // Trim: the next emit's slice starts at frame (frames − WARM).
    const keepFrom = Math.max(0, this.frames - WARM) * HOP;
    if (keepFrom > this.base) {
      this.buf.copyWithin(0, keepFrom - this.base, this.len);
      this.len -= keepFrom - this.base;
      this.base = keepFrom;
    }
    return out;
  }

  /** Emit the remaining right-padded tail frames (call once, at stream end).
   * Total emitted = floor(N/160)+1 — matching the ARRAY length process()
   * returns (its `length` field is one less, but the encoder derives the frame
   * count from the array, so the extra right-padded frame is consumed). */
  flush() {
    const n = Math.floor(this.samples / HOP) + 1 - this.frames;
    if (n <= 0) return { data: null, count: 0 };
    return this._emit(n);
  }

  _emit(n) {
    const gf0 = Math.max(0, this.frames - WARM); // slice start, in frames
    const local0 = this.frames - gf0; // first consumed frame within the slice
    const slice = this.buf.subarray(gf0 * HOP - this.base, this.len);
    const { rawMel, nFrames } = this.pre.computeRawMel(slice, local0);
    if (local0 + n > nFrames) throw new Error(`StreamingMel: need ${local0 + n} slice frames, computed ${nFrames}`);
    const out = new Float32Array(this.nMels * n);
    for (let c = 0; c < this.nMels; c++) out.set(rawMel.subarray(c * nFrames + local0, c * nFrames + local0 + n), c * n);
    this.frames += n;
    return { data: out, count: n };
  }
}
