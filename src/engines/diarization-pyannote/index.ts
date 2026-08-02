// Speaker diarization — the Senko-Web equivalent.
//
// Best path is NOT raw onnxruntime-web but **sherpa-onnx WASM** (k2-fsa, 13.9k★),
// which ships a complete offline diarization pipeline compiled to WebAssembly:
//   pyannote segmentation (sherpa-onnx-pyannote-segmentation-3-0)
//   → speaker embeddings (3D-Speaker / NeMo; FluidAudio uses wespeaker_v2 — swappable)
//   → clustering (num-speakers or threshold)
// exposed via `SherpaOnnxCreateOfflineSpeakerDiarization`.
//
// STATUS: scaffold. sherpa-onnx publishes a WASM bundle + JS glue built with
// `build-wasm-simd-speaker-diarization.sh`; vendor that bundle under
// `public/sherpa/` and load it here. (It's an Emscripten Module, not an npm dep,
// which is why it isn't in package.json.)

import type { AudioData, DiarizationEngine, DiarSegment, ProgressCb } from "../../core/types";

export class PyannoteDiarizationEngine implements DiarizationEngine {
  readonly id = "diarization-pyannote";
  readonly label = "Speaker diarization (pyannote)";
  private sherpa: any = null;

  async load(onProgress?: ProgressCb): Promise<void> {
    onProgress?.({ file: "sherpa-onnx wasm", loaded: 0, total: 1, fraction: 0.1 });
    // TODO: load the vendored sherpa-onnx diarization WASM module, e.g.
    //   const createModule = (await import("/sherpa/sherpa-onnx-wasm-main.js")).default;
    //   this.sherpa = await createModule();
    // then configure SherpaOnnxOfflineSpeakerDiarizationConfig with the
    // segmentation + embedding model paths (preloaded into the module FS).
    throw new Error(
      "PyannoteDiarizationEngine.load: vendor the sherpa-onnx diarization WASM bundle first — see docs/ARCHITECTURE.md"
    );
  }

  async diarize(_audio: AudioData, _opts?: { numSpeakers?: number }): Promise<DiarSegment[]> {
    if (!this.sherpa) throw new Error("load() not called");
    // TODO: call the module's diarization entry point, map the returned
    // (speaker, start, end) triples to DiarSegment[].
    throw new Error("PyannoteDiarizationEngine.diarize: not yet implemented");
  }

  async dispose(): Promise<void> {
    this.sherpa = null;
  }
}
