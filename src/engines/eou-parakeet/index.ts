// Parakeet EOU 120M — streaming end-of-utterance detection.
//
// STATUS: greenfield. No public ONNX export of parakeet-realtime-eou-120m exists,
// so nothing to load yet. The architecture is the same cache-aware streaming
// family as Nemotron (which is proven in-browser), and the model is tiny (120M),
// so once exported it should follow `asr-nemotron`'s pattern closely.
//
// Step 1 (offline, not in this repo): export the streaming EOU encoder + decision
// head from NeMo to ONNX. Step 2: mirror NemotronStreamingEngine here.

import type { ProgressCb, StreamingAsrEngine } from "../../core/types";

export class ParakeetEouEngine implements StreamingAsrEngine {
  readonly id = "eou-parakeet";
  readonly label = "Parakeet EOU 120M";

  async load(_onProgress?: ProgressCb): Promise<void> {
    throw new Error(
      "ParakeetEouEngine: no ONNX export yet. Export parakeet-realtime-eou-120m from NeMo first."
    );
  }
  reset(): void {}
  async push(_chunk: Float32Array): Promise<string> {
    throw new Error("ParakeetEouEngine: not implemented");
  }
  async dispose(): Promise<void> {}
}
