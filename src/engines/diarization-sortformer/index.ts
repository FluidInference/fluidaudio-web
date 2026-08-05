// Speaker diarization via NVIDIA Sortformer (4-speaker), fully ORT-free.
//
// Run OFFLINE single-chunk (empty spkcache/fifo) on raw WebGPU: the shared
// FastConformer encoder (int8, 17 layers, d512) → the Sortformer head
// (encoder_proj → 18-layer transformer → single_hidden_to_spks → sigmoid) →
// per-frame 4-speaker probs → threshold/merge into segments. mel = per_feature
// (ParakeetMel). Full raw pipeline == ORT preds (maxΔ 1.79e-7; int8 2.3e-3).
// Long-audio streaming (spkcache/fifo threading) is a follow-up.

import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AudioData, DiarizationEngine, DiarSegment, ProgressCb } from "../../core/types";
import { createContext } from "../../gpu/context.js";
import { loadParakeetEncoder, parakeetEncode } from "../asr-parakeet/raw-encoder.js";
import { loadSortformerHead, sortformerHead, predsToSegments } from "./raw-sortformer-head.js";
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
    const json = async (path: string) =>
      JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path)));
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
    const { features, length } = this.mel.process(audio.samples);
    if (length === 0) return [];
    const r = await parakeetEncode(this.ctx, this.enc, features, length);
    const preds = await sortformerHead(this.ctx, this.head, r.framesGpu, r.Tsub);
    const frames = preds.length / SPK;
    const frameSec = audio.samples.length / audio.sampleRate / frames;
    return predsToSegments(preds, frames, frameSec) as DiarSegment[];
  }

  async dispose(): Promise<void> {
    this.ctx?.device?.destroy?.();
    this.ctx = this.enc = this.head = null;
  }
}
