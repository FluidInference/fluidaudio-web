// Audio I/O: decode a File/ArrayBuffer to 16 kHz mono float, and capture mic.
// All engines expect 16 kHz mono (the ASR/diar sample rate).

import type { AudioData } from "./types.js";

export const TARGET_SR = 16000;

/** Decode any browser-supported audio file and resample to 16 kHz mono. */
export async function decodeToMono16k(input: ArrayBuffer): Promise<AudioData> {
  // Fast path: canonical 16-bit PCM mono 16 kHz WAV (the benchmark format)
  // parses directly — decodeAudioData on a 110MB 1-hour WAV is slow, allocates
  // multiples of the file size, and can abort outright. Anything else (other
  // rates, stereo, float WAV, compressed) falls through to the browser decoder.
  const fast = tryParseCanonicalWav(input);
  if (fast) return fast;
  // Decode at native rate first (decodeAudioData needs a real AudioContext).
  const tmp = new AudioContext();
  const decoded = await tmp.decodeAudioData(input.slice(0));
  await tmp.close();

  // Downmix to mono.
  const mono = downmix(decoded);

  if (decoded.sampleRate === TARGET_SR) {
    return { samples: mono, sampleRate: TARGET_SR };
  }
  // Resample via OfflineAudioContext.
  const frames = Math.ceil((mono.length * TARGET_SR) / decoded.sampleRate);
  const off = new OfflineAudioContext(1, frames, TARGET_SR);
  const buf = off.createBuffer(1, mono.length, decoded.sampleRate);
  buf.getChannelData(0).set(mono);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return { samples: rendered.getChannelData(0).slice(), sampleRate: TARGET_SR };
}

function tryParseCanonicalWav(input: ArrayBuffer): AudioData | null {
  if (input.byteLength < 44) return null;
  const dv = new DataView(input);
  const tag = (o: number) => String.fromCharCode(dv.getUint8(o), dv.getUint8(o + 1), dv.getUint8(o + 2), dv.getUint8(o + 3));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") return null;
  let o = 12;
  let fmt: { format: number; channels: number; rate: number; bits: number } | null = null;
  while (o + 8 <= input.byteLength) {
    const id = tag(o);
    const size = dv.getUint32(o + 4, true);
    if (id === "fmt ") {
      fmt = { format: dv.getUint16(o + 8, true), channels: dv.getUint16(o + 10, true), rate: dv.getUint32(o + 12, true), bits: dv.getUint16(o + 22, true) };
    } else if (id === "data") {
      if (!fmt || fmt.format !== 1 || fmt.channels !== 1 || fmt.rate !== TARGET_SR || fmt.bits !== 16) return null;
      const n = Math.min(size, input.byteLength - o - 8) >> 1;
      const pcm = new Int16Array(input, o + 8, n);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = pcm[i] / 32768;
      return { samples: out, sampleRate: TARGET_SR };
    }
    o += 8 + size + (size & 1);
  }
  return null;
}

function downmix(buf: AudioBuffer): Float32Array {
  if (buf.numberOfChannels === 1) return buf.getChannelData(0).slice();
  const out = new Float32Array(buf.length);
  for (let ch = 0; ch < buf.numberOfChannels; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < data.length; i++) out[i] += data[i] / buf.numberOfChannels;
  }
  return out;
}

/** Encode Float32 PCM as a 16-bit WAV blob (for TTS playback / download). */
export function pcmToWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}
