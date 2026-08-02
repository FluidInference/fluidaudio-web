// Nemotron 3.5 streaming ASR (en + multilingual, 40 langs) — cache-aware
// FastConformer-RNNT. The ONNX (INT4) is published by onnx-community and is
// proven to run in-browser (see khawjaahmad/nemotron-asr-webgpu, a 1★ POC).
//
// Cache-aware streaming = the encoder carries state tensors between chunks
// instead of re-processing overlapping audio. That state is the crux of this
// engine: each `push()` feeds one chunk + the previous caches and gets back new
// caches + encoder frames, which the RNNT decoder+joint consume greedily.
//
// STATUS: scaffold. Sessions + registry wired; cache-tensor plumbing + RNNT loop
// TODO. Chunk tiers (80/160/320/560/1120ms) match FluidAudio's variants.

import { createSession, ort } from "../../core/ort";
import { fetchAll } from "../../core/modelCache";
import { REGISTRY } from "../../core/registry";
import type { ProgressCb, StreamingAsrEngine } from "../../core/types";

export class NemotronStreamingEngine implements StreamingAsrEngine {
  readonly id = "asr-nemotron";
  readonly label = "Nemotron 3.5 streaming";
  private encoder: ort.InferenceSession | null = null;
  private decoderJoint: ort.InferenceSession | null = null;
  private caches: Record<string, ort.Tensor> = {};

  async load(onProgress?: ProgressCb): Promise<void> {
    const spec = REGISTRY[this.id];
    const files = await fetchAll(spec.files, onProgress);
    const [encPath, decPath] = spec.files.map((f) => f.path);
    this.encoder = await createSession(files.get(encPath)!, "webgpu");
    this.decoderJoint = await createSession(files.get(decPath)!, "wasm");
    this.reset();
  }

  reset(): void {
    // TODO: initialize cache tensors (cache_last_channel, cache_last_time,
    // cache_last_channel_len) to zeros with the shapes from the encoder's
    // input metadata. Inspect this.encoder.inputNames / .inputMetadata.
    this.caches = {};
  }

  async push(_chunk: Float32Array): Promise<string> {
    if (!this.encoder || !this.decoderJoint) throw new Error("load() not called");
    // TODO(port):
    //   1. mel for this chunk (Nemotron preprocessor / native mel).
    //   2. { encOut, newCaches } = encoder.run({ audio, ...this.caches })
    //   3. this.caches = newCaches
    //   4. RNNT greedy over encOut using decoderJoint; append emitted tokens.
    throw new Error(
      `NemotronStreamingEngine.push: cache plumbing + RNNT loop not yet ported ` +
        `(carrying ${Object.keys(this.caches).length} cache tensors)`
    );
  }

  async dispose(): Promise<void> {
    await this.encoder?.release();
    await this.decoderJoint?.release();
    this.encoder = this.decoderJoint = null;
    this.caches = {};
  }
}
