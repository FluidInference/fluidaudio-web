// Silero VAD via @ricky0123/vad-web (onnxruntime-web + WASM). Mature, drop-in.
// Uses the non-real-time path for whole-clip segmentation.

import * as vadWeb from "@ricky0123/vad-web";
import type { AudioData, ProgressCb, SpeechRange, VadEngine } from "../../core/types";

// vad-web is CJS; grab NonRealTimeVAD via namespace so the named-import interop
// can't break at module load.
const NonRealTimeVAD: any = (vadWeb as any).NonRealTimeVAD ?? (vadWeb as any).default?.NonRealTimeVAD;

export class SileroVadEngine implements VadEngine {
  readonly id = "vad-silero";
  readonly label = "Silero VAD";
  private vad: any = null;

  async load(onProgress?: ProgressCb): Promise<void> {
    onProgress?.({ file: "silero-vad", loaded: 0, total: 1, fraction: 0.1 });
    // vad-web bundles the ONNX weights + wasm; it fetches them on first use.
    // Point vad-web at the CDN for its silero ONNX + ort wasm so it resolves
    // under any bundler/host (default paths 404 under Vite).
    const base = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.30/dist/";
    this.vad = await NonRealTimeVAD.new({
      baseAssetPath: base,
      onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/",
    } as any);
    onProgress?.({ file: "silero-vad", loaded: 1, total: 1, fraction: 1 });
  }

  async detect(audio: AudioData): Promise<SpeechRange[]> {
    if (!this.vad) throw new Error("SileroVadEngine.load() not called");
    const ranges: SpeechRange[] = [];
    for await (const { start, end } of this.vad.run(audio.samples, audio.sampleRate)) {
      // vad-web yields start/end in milliseconds.
      ranges.push({ start: start / 1000, end: end / 1000 });
    }
    return ranges;
  }

  async dispose(): Promise<void> {
    this.vad = null;
  }
}
