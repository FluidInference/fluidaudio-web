// Speaker diarization via NVIDIA Sortformer (4-speaker), fully ORT-free.
//
// Run OFFLINE single-chunk (empty spkcache/fifo) on raw WebGPU: the shared
// FastConformer encoder (int8, 17 layers, d512) → the Sortformer head
// (encoder_proj → 18-layer transformer → single_hidden_to_spks → sigmoid) →
// per-frame 4-speaker probs → threshold/merge into segments. mel = per_feature
// (ParakeetMel). Full raw pipeline == ORT preds (maxΔ 1.79e-7; int8 2.3e-3).
// Long-audio streaming (spkcache/fifo threading) is a follow-up.

import { fetchCached, hfUrl } from "../../core/modelCache.js";
import type { AudioData, DiarizationEngine, DiarSegment, ProgressCb } from "../../core/types.js";
import { createContext } from "../../gpu/context.js";
import { loadParakeetEncoder, parakeetEncode } from "../asr-parakeet/raw-encoder.js";
import { loadSortformerHead, sortformerHead, predsToSegments, mergeWindowPreds } from "./raw-sortformer-head.js";
import { ParakeetMel } from "../asr-parakeet/parakeet-mel.js";

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";
// Sortformer FastConformer config: symmetric (Parakeet-style) subsampling + full
// attention offline, but NeMo xscaling (√512) is a runtime Mul in this export.
const SF_CFG = { xscale: true };
const SPK = 4;

export class SortformerDiarizationEngine implements DiarizationEngine {
  readonly id = "diarization-sortformer";
  readonly label = "Diarization (Sortformer)";
  private ctx: any = null;
  private enc: any = null;
  private head: any = null;
  private mel = new ParakeetMel(128);

  async load(onProgress?: ProgressCb): Promise<void> {
    this.ctx = await createContext({ onBackend: (b) => console.info(`[diarization-sortformer] backend: ${b}`) });
    const json = async (path: string) => JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path)));
    const bytes = (path: string) => fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path);

    const encMan = await json("sortformer/encoder-int8.manifest.json");
    const encBin = await bytes("sortformer/encoder-int8.bin");
    const headMan = await json("sortformer/head-fp32.manifest.json");
    const headBin = await bytes("sortformer/head-fp32.bin");

    this.enc = loadParakeetEncoder(this.ctx, encBin, encMan, SF_CFG);
    this.head = loadSortformerHead(this.ctx, new Float32Array(headBin.buffer, headBin.byteOffset, headBin.byteLength / 4), headMan);
    onProgress?.({ file: WEIGHTS_REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async diarize(audio: AudioData): Promise<DiarSegment[]> {
    if (!this.enc || !this.head) throw new Error("SortformerDiarizationEngine.load() not called");
    // Long audio is WINDOWED (90s / 15s overlap) and stitched by overlap-permutation
    // matching: a single chunk over minutes collapses to the dominant speaker (the
    // known single-chunk Sortformer failure mode), and full attention is quadratic.
    const sr = audio.sampleRate;
    // Window size adapts to the device's storage-buffer cap: the subsampling
    // intermediate is 256ch × (melFrames/2) × 64 × 4B ≈ winSec × 3.28 MB (WebGPU
    // spec floor is 128MB → ~35s windows; typical adapters allow the full 90s).
    const capBytes = this.ctx?.device?.limits?.maxStorageBufferBindingSize ?? Infinity;
    const winSec = Math.max(30, Math.min(90, Math.floor((capBytes * 0.85) / (3.28 * 1024 * 1024))));
    const WIN = winSec * sr,
      OVL = Math.min(15, Math.floor(winSec / 4)) * sr,
      hop = WIN - OVL;
    const runWindow = async (samples: Float32Array) => {
      const { features, length } = this.mel.process(samples);
      if (length === 0) return null;
      const r = await parakeetEncode(this.ctx, this.enc, features, length);
      const preds = await sortformerHead(this.ctx, this.head, r.framesGpu, r.Tsub);
      return { preds, frames: preds.length / SPK };
    };
    if (audio.samples.length <= WIN) {
      const w = await runWindow(audio.samples);
      if (!w) return [];
      const frameSec = audio.samples.length / sr / w.frames;
      return predsToSegments(w.preds, w.frames, frameSec) as DiarSegment[];
    }
    const windows: { preds: Float32Array; frames: number }[] = [];
    const ovlFrames: number[] = [];
    for (let s = 0; s < audio.samples.length; s += hop) {
      const end = Math.min(s + WIN, audio.samples.length);
      const w = await runWindow(audio.samples.subarray(s, end));
      if (!w) break;
      const fps = w.frames / ((end - s) / sr);
      windows.push(w);
      ovlFrames.push(Math.round((OVL / sr) * fps));
      if (end >= audio.samples.length) break;
    }
    if (!windows.length) return [];
    const merged = mergeWindowPreds(windows, ovlFrames);
    const frameSec = audio.samples.length / sr / merged.frames;
    return predsToSegments(merged.preds, merged.frames, frameSec) as DiarSegment[];
  }

  async dispose(): Promise<void> {
    this.ctx?.destroy();
    this.ctx = this.enc = this.head = null;
  }
}
