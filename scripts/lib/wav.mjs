// Shared 16-bit mono PCM WAV reader for the gate scripts. (There were 14
// hand-copied readWav implementations across scripts/ — new scripts import
// this one; old copies are converted opportunistically as scripts are touched.)
import { readFileSync } from "node:fs";

/** @returns {Float32Array} mono float32 samples in [-1, 1] */
export function readWav(path) {
  const b = readFileSync(path);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let o = 12,
    dO = -1,
    dL = 0;
  while (o + 8 <= b.length) {
    const id = String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);
    const size = dv.getUint32(o + 4, true);
    if (id === "data") {
      dO = o + 8;
      dL = size;
      break;
    }
    o += 8 + size + (size & 1);
  }
  if (dO < 0) throw new Error(`readWav: no data chunk in ${path}`);
  const n = dL / 2;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(dO + i * 2, true) / 32768;
  return out;
}
