// Manifest+bin weight store for tts-voicechat. Unlike the older uniform-dtype
// manifests, tensors carry their own dtype ("f32" | "f16" | "u8"), a shard
// index and BYTE offsets, so precision is a per-tensor extractor decision and
// multi-GB exports stay under the ~2 GiB node/browser single-buffer ceilings
// (see scripts/extract-voicechat-tts.py). f16 payloads expand to f32 through a
// 65536-entry lookup table at load; u8 stays Uint8Array (flag tables).

let _f16lut = null;
function f16lut() {
  if (_f16lut) return _f16lut;
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1,
      e = (h & 0x7c00) >> 10,
      f = h & 0x03ff;
    if (e === 0) t[h] = s * Math.pow(2, -14) * (f / 1024);
    else if (e === 0x1f) t[h] = f ? NaN : s * Infinity;
    else t[h] = s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return (_f16lut = t);
}

export class WeightStore {
  /** @param {Uint8Array[]} bins shard payloads, index-aligned with manifest "bin"
   *  @param {{shards:number, tensors:Record<string,{dims:number[],dtype:string,bin:number,byteOffset:number,count:number}>}} man */
  constructor(bins, man) {
    this.bins = Array.isArray(bins) ? bins : [bins];
    if (this.bins.length !== man.shards) throw new Error(`voicechat weights: got ${this.bins.length} shards, manifest says ${man.shards}`);
    this.man = man.tensors;
  }

  meta(name) {
    const m = this.man[name];
    if (!m) throw new Error(`voicechat weights: missing tensor ${name}`);
    return m;
  }

  /** Tensor payload as Float32Array (f16 expanded via LUT). */
  f32(name) {
    const m = this.meta(name);
    const bin = this.bins[m.bin];
    if (m.dtype === "f32") return new Float32Array(bin.buffer, bin.byteOffset + m.byteOffset, m.count);
    if (m.dtype !== "f16") throw new Error(`voicechat weights: ${name} is ${m.dtype}, not float`);
    const q = new Uint16Array(bin.buffer, bin.byteOffset + m.byteOffset, m.count);
    const lut = f16lut(),
      out = new Float32Array(m.count);
    for (let i = 0; i < m.count; i++) out[i] = lut[q[i]];
    return out;
  }

  u8(name) {
    const m = this.meta(name);
    if (m.dtype !== "u8") throw new Error(`voicechat weights: ${name} is ${m.dtype}, not u8`);
    const bin = this.bins[m.bin];
    return new Uint8Array(bin.buffer, bin.byteOffset + m.byteOffset, m.count);
  }

  dims(name) {
    return this.meta(name).dims;
  }

  /** 2-D matrix as a backend tensor (uploadF16 keeps f16 storage on WebGPU). */
  mat(ctx, name) {
    const m = this.meta(name);
    if (m.dims.length !== 2) throw new Error(`voicechat weights: ${name} dims ${m.dims} not 2-D`);
    const data = this.f32(name);
    return m.dtype === "f16" ? ctx.uploadF16(data.slice(), m.dims[0], m.dims[1]) : ctx.upload(data.slice(), m.dims[0], m.dims[1]);
  }
}
