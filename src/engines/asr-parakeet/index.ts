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
import { createContext } from "../../gpu/context.js";
import { loadParakeetEncoder, parakeetEncodeBatch } from "./raw-encoder.js";
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
    this.ctx = await createContext({ onBackend: (b) => console.info(`[asr-parakeet] backend: ${b}`) });
    const json = async (path: string, repo = WEIGHTS_REPO) => JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(repo, path), onProgress, path)));
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
    const t0 = now();

    // Window start offsets.
    const starts: number[] = [];
    for (let s = 0; s < samples.length; s += hop) {
      starts.push(s);
      if (single) break;
    }

    // Encode windows in GROUPS of up to WB (equal-length windows batched through the
    // encoder: bigger GEMMs + one readback per group instead of per window), and
    // pipeline: encode group g+1 on the GPU while the CPU (WASM) decodes group g.
    const WB = 4;
    const groups: number[][] = [];
    {
      let cur: number[] = [];
      for (let i = 0; i < starts.length; i++) {
        const len = Math.min(starts[i] + winSamples, samples.length) - starts[i];
        if (cur.length && (len !== winSamples || cur.length >= WB)) {
          groups.push(cur);
          cur = [];
        }
        cur.push(i);
        if (len !== winSamples) {
          groups.push(cur);
          cur = [];
        } // short tail encodes alone
      }
      if (cur.length) groups.push(cur);
    }
    const beginGroup = (g: number): Promise<{ frames: Float32Array; Tenc: number; D: number; n: number } | null> => {
      const mels: Float32Array[] = [];
      for (const i of groups[g]) {
        const slice = single ? samples : samples.subarray(starts[i], Math.min(starts[i] + winSamples, samples.length));
        const { features, length } = this.mel!.process(slice);
        if (length > 0) mels.push(features);
      }
      if (!mels.length) return Promise.resolve(null);
      return parakeetEncodeBatch(this.ctx, this.enc, mels).then(async (r: any) => ({
        frames: await this.ctx.download(r.framesGpu),
        Tenc: r.Tsub,
        D: r.D,
        n: mels.length,
      }));
    };

    const ids: number[] = [];
    let w = 0;
    let pendingG = beginGroup(0);
    for (let g = 0; g < groups.length; g++) {
      const grp = await pendingG;
      if (g + 1 < groups.length) pendingG = beginGroup(g + 1); // queue next GPU encode now
      if (!grp) {
        w += groups[g].length;
        continue;
      }
      for (let wi = 0; wi < grp.n; wi++, w++) {
        const Tenc = grp.Tenc;
        const frames = grp.frames.subarray(wi * Tenc * grp.D, (wi + 1) * Tenc * grp.D);
        const sliceLen = Math.min(starts[w] + winSamples, samples.length) - starts[w];
        const { ids: wids, idFrames } = wasmDecode(this.dec, frames, Tenc);

        // Seam dedup: frame-estimated overlap refined by an exact token-match stitch.
        let skip = 0;
        if (w > 0 && wids.length) {
          const overlapEnc = Math.round((Tenc * overlapSamples) / sliceLen);
          let frameSkip = 0;
          while (frameSkip < idFrames.length && idFrames[frameSkip] < overlapEnc) frameSkip++;
          const maxL = Math.min(ids.length, wids.length, frameSkip + 8);
          let matched = 0;
          for (let L = maxL; L >= 2; L--) {
            let ok = true;
            for (let i = 0; i < L; i++)
              if (ids[ids.length - L + i] !== wids[i]) {
                ok = false;
                break;
              }
            if (ok) {
              matched = L;
              break;
            }
          }
          skip = Math.max(matched, frameSkip);
        }
        for (let k = skip; k < wids.length; k++) ids.push(wids[k]);
      }
    }
    // GPU encode and CPU decode are pipelined, so a per-stage split is meaningless;
    // report the wall-clock total.
    return {
      text: this.tokenizer.decode(ids),
      metrics: { melMs: 0, encodeMs: 0, decodeMs: 0, totalMs: +(now() - t0).toFixed(0) },
    };
  }

  async dispose(): Promise<void> {
    this.ctx?.device?.destroy?.();
    this.ctx = this.enc = this.dec = this.mel = this.tokenizer = null;
  }
}
