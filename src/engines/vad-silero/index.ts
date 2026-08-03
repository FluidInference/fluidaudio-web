// Silero VAD — reimplemented directly on core/ort.
//
// We do NOT use @ricky0123/vad-web: it's CJS and does a dynamic
// `require("onnxruntime-web/wasm")` that Vite can't resolve once ORT is excluded
// from optimizeDeps (excluding ORT is itself required — see core/ort.ts). The
// Silero ONNX interface is trivial (512-sample windows + a 2×1×128 state), so we
// drive it ourselves and drop the dependency entirely.
//
// Model: silero_vad.onnx (v5) — input[1,512] + state[2,1,128] + sr int64 ->
// output[1,1] speech prob + stateN[2,1,128]. Runs on WASM (tiny, no WebGPU win).

import { createSession, ort } from "../../core/ort";
import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AudioData, ProgressCb, SpeechRange, VadEngine } from "../../core/types";
import { sileroDetect } from "./silero.js";

const REPO = "onnx-community/silero-vad";
const MODEL = "onnx/model.onnx";

export class SileroVadEngine implements VadEngine {
  readonly id = "vad-silero";
  readonly label = "Silero VAD";
  private session: any = null;

  async load(onProgress?: ProgressCb): Promise<void> {
    const bytes = await fetchCached(hfUrl(REPO, MODEL), onProgress, "silero_vad.onnx");
    this.session = await createSession(bytes, "wasm");
    onProgress?.({ file: REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async detect(audio: AudioData): Promise<SpeechRange[]> {
    if (!this.session) throw new Error("SileroVadEngine.load() not called");
    return sileroDetect({ ort, session: this.session, audio: audio.samples });
  }

  async dispose(): Promise<void> {
    await this.session?.release?.();
    this.session = null;
  }
}
