// Speaker diarization via NVIDIA Sortformer (streaming, 4-speaker), ONNX fp32.
// Verified headless: single-speaker → 1 active, 40s conference → 2 speakers with
// correct segments. fp32 → runs on WebGPU or WASM (not precision-gated). This is
// the offline single-chunk path; long-audio streaming (spkcache/fifo across
// chunks) is the follow-up (see sortformer.js).

import { createSession, ort } from "../../core/ort";
import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AudioData, DiarizationEngine, DiarSegment, ProgressCb } from "../../core/types";
import { diarizeSortformer } from "./sortformer.js";

const SF_REPO = "cgus/diar_streaming_sortformer_4spk-v2.1-onnx";
const SF_FILE = "diar_streaming_sortformer_4spk-v2.1.onnx";
// nemo128 log-mel (128 bins) — reused from the parakeet ONNX repo.
const MEL_REPO = "ysdede/parakeet-tdt-0.6b-v3-onnx";
const MEL_FILE = "nemo128.onnx";

export class SortformerDiarizationEngine implements DiarizationEngine {
  readonly id = "diarization-sortformer";
  readonly label = "Diarization (Sortformer)";
  private sortformer: any = null;
  private mel: any = null;

  async load(onProgress?: ProgressCb): Promise<void> {
    const sfBytes = await fetchCached(hfUrl(SF_REPO, SF_FILE), onProgress, SF_FILE);
    const melBytes = await fetchCached(hfUrl(MEL_REPO, MEL_FILE), onProgress, MEL_FILE);
    this.sortformer = await createSession(sfBytes, "webgpu");
    this.mel = await createSession(melBytes, "wasm");
    onProgress?.({ file: SF_REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async diarize(audio: AudioData): Promise<DiarSegment[]> {
    if (!this.sortformer || !this.mel) throw new Error("SortformerDiarizationEngine.load() not called");
    return diarizeSortformer({
      ort,
      mel: this.mel,
      sortformer: this.sortformer,
      audio: audio.samples,
      sampleRate: audio.sampleRate,
    });
  }

  async dispose(): Promise<void> {
    await this.sortformer?.release?.();
    await this.mel?.release?.();
    this.sortformer = this.mel = null;
  }
}
