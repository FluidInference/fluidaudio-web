// Microphone capture → growing 16 kHz mono Float32 buffer.
//
// The AudioContext is asked for 16 kHz directly (Chrome resamples in the audio
// stack); if the browser ignores the hint (older Safari), a linear resampler
// downmixes to 16 kHz on the fly. Frames arrive via an AudioWorklet tap —
// ScriptProcessorNode is deprecated and jank-prone.

const TARGET_SR = 16000;

// Worklet source inlined as a Blob URL: bundler-proof (Vite inlines tiny .js
// assets as data: URLs, which audioWorklet.addModule rejects) and SDK-proof
// (no asset path resolution for consumers). Runs on the audio thread.
const WORKLET_SRC = `
class MicTap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (ch && ch.length) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("mic-tap", MicTap);
`;

export class MicCapture {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private chunks: Float32Array[] = [];
  private total = 0;
  private baseIndex = 0; // absolute index of chunks[0][0] (advanced by dropBefore)
  private srcRate = TARGET_SR;
  /** Peak level of the most recent frame (0..1) — for a simple VU indicator. */
  level = 0;

  get running(): boolean {
    return this.ctx !== null;
  }

  /** Captured duration in seconds (at 16 kHz). */
  get seconds(): number {
    return this.total / TARGET_SR;
  }

  async start(): Promise<void> {
    if (this.ctx) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.ctx = new AudioContext({ sampleRate: TARGET_SR });
    this.srcRate = this.ctx.sampleRate; // browsers may ignore the 16 kHz hint
    const workletUrl = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "text/javascript" }));
    try {
      await this.ctx.audioWorklet.addModule(workletUrl);
    } finally {
      URL.revokeObjectURL(workletUrl);
    }
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "mic-tap");
    this.node.port.onmessage = (e: MessageEvent<Float32Array>) => {
      const frame = this.srcRate === TARGET_SR ? e.data : resampleLinear(e.data, this.srcRate, TARGET_SR);
      this.chunks.push(frame);
      this.total += frame.length;
      let peak = 0;
      for (let i = 0; i < frame.length; i++) {
        const a = Math.abs(frame[i]);
        if (a > peak) peak = a;
      }
      this.level = peak;
    };
    source.connect(this.node);
    // No destination connection needed — the worklet is a pure tap.
  }

  /** Last `sec` seconds (or everything, if shorter). Copies into one buffer. */
  tail(sec: number): Float32Array {
    const want = Math.min(this.total, Math.round(sec * TARGET_SR));
    const out = new Float32Array(want);
    let filled = want;
    for (let i = this.chunks.length - 1; i >= 0 && filled > 0; i--) {
      const c = this.chunks[i];
      const take = Math.min(filled, c.length);
      out.set(c.subarray(c.length - take), filled - take);
      filled -= take;
    }
    return out;
  }

  /** Release chunks fully consumed below absolute index `to` — true-streaming
   * consumers never re-read history, so an hours-long session stays bounded.
   * tail()/all() afterwards only cover retained samples (streaming stop paths
   * don't use them). */
  dropBefore(to: number): void {
    while (this.chunks.length && this.baseIndex + this.chunks[0].length <= to) {
      this.baseIndex += this.chunks[0].length;
      this.chunks.shift();
    }
  }

  /** Samples appended since absolute index `from` (must be ≥ any dropBefore
   * watermark); returns them + new total. For incremental consumers
   * (true-streaming engines): poll with the last returned total. */
  since(from: number): { samples: Float32Array; total: number } {
    const want = this.total - from;
    if (want <= 0) return { samples: new Float32Array(0), total: this.total };
    const out = new Float32Array(want);
    let filled = want;
    for (let i = this.chunks.length - 1; i >= 0 && filled > 0; i--) {
      const c = this.chunks[i];
      const take = Math.min(filled, c.length);
      out.set(c.subarray(c.length - take), filled - take);
      filled -= take;
    }
    return { samples: out, total: this.total };
  }

  /** Full capture as one buffer. */
  all(): Float32Array {
    const out = new Float32Array(this.total);
    let off = 0;
    for (const c of this.chunks) {
      out.set(c, off);
      off += c.length;
    }
    return out;
  }

  async stop(): Promise<void> {
    this.node?.disconnect();
    this.node = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    await this.ctx?.close().catch(() => {});
    this.ctx = null;
  }

  clear(): void {
    this.chunks = [];
    this.total = 0;
    this.baseIndex = 0;
    this.level = 0;
  }
}

function resampleLinear(x: Float32Array, from: number, to: number): Float32Array {
  const n = Math.round((x.length * to) / from);
  const out = new Float32Array(n);
  const step = (x.length - 1) / Math.max(1, n - 1);
  for (let i = 0; i < n; i++) {
    const p = i * step;
    const j = Math.floor(p);
    const f = p - j;
    out[i] = x[j] * (1 - f) + (x[Math.min(j + 1, x.length - 1)] ?? 0) * f;
  }
  return out;
}
