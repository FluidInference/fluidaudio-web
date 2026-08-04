// Parakeet TDT 0.6B v3 — fully ORT-free. Mel (parakeet-mel.js), FastConformer
// encoder (raw-encoder.js, int8 on raw WebGPU), and TDT decoder+joint
// (raw-decoder-wasm.js, WASM-SIMD on CPU) are all hand-written; no onnxruntime.
// Weights from FluidInference/fluidaudio-web. Long audio windowed (15s / 2s overlap).
//
// Split rationale: the encoder is GPU (big GEMMs); the RNNT decoder is CPU/WASM-SIMD
// because it's autoregressive — one result per token — so a GPU decoder pays a
// round-trip per token (the ~20× wall). WASM-SIMD decodes on CPU with no GPU sync.

import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";
import { GpuContext, requestGpuDevice } from "../../gpu/compute.js";
import { loadParakeetEncoder, parakeetEncode } from "./raw-encoder.js";
import { loadWasmDecoder, wasmDecode } from "./raw-decoder-wasm.js";
import { ParakeetMel } from "./parakeet-mel.js";
import { ParakeetTokenizer } from "./tokenizer.js";
import wasmUrl from "./parakeet-decoder.wasm?url";

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";
const VOCAB_REPO = "ysdede/parakeet-tdt-0.6b-v3-onnx";
const SAMPLE_RATE = 16000;
const WINDOW_SEC = 15;
const OVERLAP_SEC = 2;

export class ParakeetV3Engine implements AsrEngine {
  readonly id = "asr-parakeet";
  readonly label = "Parakeet TDT 0.6B v3";
  private ctx: any = null;
  private enc: any = null;
  private dec: any = null;
  private mel: ParakeetMel | null = null;
  private tokenizer: ParakeetTokenizer | null = null;

  async load(onProgress?: ProgressCb): Promise<void> {
    this.ctx = new GpuContext(await requestGpuDevice());
    const json = async (path: string, repo = WEIGHTS_REPO) =>
      JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(repo, path), onProgress, path)));
    const bytes = (path: string) => fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path);

    const encMan = await json("parakeet/encoder-int8.manifest.json");
    const encBin = await bytes("parakeet/encoder-int8.bin");
    const decMan = await json("parakeet/decoder-fp32.manifest.json");
    const decBin = await bytes("parakeet/decoder-fp32.bin");
    const vocab = new TextDecoder().decode(await fetchCached(hfUrl(VOCAB_REPO, "vocab.txt"), onProgress, "vocab.txt"));
    const wasmBytes = await (await fetch(wasmUrl)).arrayBuffer();

    this.enc = loadParakeetEncoder(this.ctx, encBin, encMan);
    this.dec = await loadWasmDecoder(wasmBytes, new Float32Array(decBin.buffer, decBin.byteOffset, decBin.byteLength / 4), decMan);
    this.mel = new ParakeetMel(128);
    this.tokenizer = ParakeetTokenizer.fromVocabText(vocab);
    onProgress?.({ file: WEIGHTS_REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult> {
    if (!this.enc || !this.dec || !this.mel || !this.tokenizer) throw new Error("ParakeetV3Engine.load() not called");
    const samples = audio.samples;
    const winSamples = WINDOW_SEC * SAMPLE_RATE;
    const overlapSamples = OVERLAP_SEC * SAMPLE_RATE;
    const hop = winSamples - overlapSamples;
    const single = samples.length <= winSamples;
    const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
    let melMs = 0, encodeMs = 0, decodeMs = 0;

    const ids: number[] = [];
    for (let start = 0, w = 0; start < samples.length; start += hop, w++) {
      const slice = single ? samples : samples.subarray(start, Math.min(start + winSamples, samples.length));
      let t0 = now();
      const { features, length } = this.mel.process(slice);
      melMs += now() - t0;
      if (length === 0) { if (single) break; continue; }
      t0 = now();
      const { framesGpu, Tsub: Tenc } = await parakeetEncode(this.ctx, this.enc, features, length);
      // download [Tsub,1024] (frames[t*1024+d]) — also flushes the encoder's GPU work.
      const frames = await this.ctx.download(framesGpu);
      encodeMs += now() - t0;
      t0 = now();
      const { ids: wids, idFrames } = wasmDecode(this.dec, frames, Tenc);
      decodeMs += now() - t0;

      // Seam dedup: frame-estimated overlap refined by an exact token-match stitch.
      let skip = 0;
      if (w > 0 && wids.length) {
        const overlapEnc = Math.round((Tenc * overlapSamples) / slice.length);
        let frameSkip = 0;
        while (frameSkip < idFrames.length && idFrames[frameSkip] < overlapEnc) frameSkip++;
        const maxL = Math.min(ids.length, wids.length, frameSkip + 8);
        let matched = 0;
        for (let L = maxL; L >= 2; L--) {
          let ok = true;
          for (let i = 0; i < L; i++) if (ids[ids.length - L + i] !== wids[i]) { ok = false; break; }
          if (ok) { matched = L; break; }
        }
        skip = Math.max(matched, frameSkip);
      }
      for (let k = skip; k < wids.length; k++) ids.push(wids[k]);
      if (single) break;
    }
    return {
      text: this.tokenizer.decode(ids),
      metrics: { melMs: +melMs.toFixed(0), encodeMs: +encodeMs.toFixed(0), decodeMs: +decodeMs.toFixed(0), totalMs: +(melMs + encodeMs + decodeMs).toFixed(0) },
    };
  }

  async dispose(): Promise<void> {
    this.ctx?.device?.destroy?.();
    this.ctx = this.enc = this.dec = this.mel = this.tokenizer = null;
  }
}
