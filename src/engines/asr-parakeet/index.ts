// Parakeet TDT 0.6B v3 — offline transcription.
//
// Architecture (mirrors FluidAudio's Swift path): mel frontend → FastConformer
// encoder → TDT (token-and-duration transducer) greedy decode over encoder frames
// with a small LSTM decoder + joint. Backend split, per the parakeet.js finding:
//   • encoder → WebGPU (heavy, static-ish shapes; benefits from the GPU)
//   • decoder + joint → WASM (tiny, dynamic, step-by-step; WebGPU is slower here)
//
// STATUS: scaffold. Sessions + registry are wired; the mel extractor and the TDT
// greedy loop are the remaining work. Port them from FluidAudio's
// `TdtDecoderV3` + `AudioMelSpectrogram` (both are pure math, no CoreML).

import { createSession, ort } from "../../core/ort";
import { fetchAll } from "../../core/modelCache";
import { REGISTRY } from "../../core/registry";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";

export class ParakeetV3Engine implements AsrEngine {
  readonly id = "asr-parakeet";
  readonly label = "Parakeet TDT 0.6B v3";
  private encoder: ort.InferenceSession | null = null;
  private decoderJoint: ort.InferenceSession | null = null;

  async load(onProgress?: ProgressCb): Promise<void> {
    const spec = REGISTRY[this.id];
    const files = await fetchAll(spec.files, onProgress);
    const [encPath, decPath] = spec.files.map((f) => f.path);
    this.encoder = await createSession(files.get(encPath)!, "webgpu");
    this.decoderJoint = await createSession(files.get(decPath)!, "wasm");
  }

  async transcribe(_audio: AudioData): Promise<AsrResult> {
    if (!this.encoder || !this.decoderJoint) throw new Error("load() not called");
    // TODO(port from FluidAudio):
    //   1. mel = AudioMelSpectrogram(audio)  — 128-band, NeMo per-feature norm
    //   2. encOut = encoder.run({ audio_signal: mel, length })
    //   3. TDT greedy loop over encOut frames using decoderJoint (blank id, 5
    //      duration bins), emitting (token, timestamp). See TdtDecoderV3.swift.
    //   4. detokenize with the SentencePiece vocab (parakeet_vocab.json).
    throw new Error(
      "ParakeetV3Engine.transcribe: decode loop not yet ported — see TODO / docs/ARCHITECTURE.md"
    );
  }

  async dispose(): Promise<void> {
    await this.encoder?.release();
    await this.decoderJoint?.release();
    this.encoder = this.decoderJoint = null;
  }
}
