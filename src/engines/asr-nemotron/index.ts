// Nemotron 3.5 streaming ASR (en + multilingual, 40 langs) — cache-aware
// FastConformer-RNNT. ONNX: onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4.
//
// The int4 encoder runs on **WASM**, NOT WebGPU. It's numerically healthy on WASM
// (encoder std 0.43 — unlike Parakeet's int8, which collapses there), and ORT-web's
// WebGPU EP mishandles the int4 `MatMulNBits` ops → EMPTY transcript in-browser.
// Verified headless (ort-node WASM): correct output. So force WASM for the encoder;
// WebGPU buys nothing here anyway (thin GEMMs — see docs/RAW_WEBGPU.md).
//
// mel is NA log-mel computed in JS (no ONNX mel ships for Nemotron, and the
// parakeet nemo128 mel bakes per-feature CMVN which is wrong here) — the one JS
// DSP stage; could move to an NA-mel ONNX later.

import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";

const REPO = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4";

// Thin proxy to nemotron.worker.ts. All inference runs in the worker so the int4
// WASM decode (thousands of tiny sequential calls, single-threaded without cross-
// origin isolation) never blocks the main thread / freezes the page. It's still
// slow — the real fast+correct path is the raw-WebGPU int4 kernel (src/gpu) wired
// into the encoder — but the UI stays responsive.
export class NemotronEngine implements AsrEngine {
  readonly id = "asr-nemotron";
  readonly label = "Nemotron 3.5 streaming";
  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();

  /** @param opts.language BCP-47-ish code, e.g. "en-US" / "de" / "zh" (default en-US). */
  constructor(private opts: { language?: string } = {}) {}

  private call(type: string, extra: Record<string, any> = {}, transfer: Transferable[] = []): Promise<any> {
    return new Promise((resolve, reject) => {
      const id = ++this.seq;
      this.pending.set(id, { resolve, reject });
      this.worker!.postMessage({ type, id, ...extra }, transfer);
    });
  }

  async load(onProgress?: ProgressCb): Promise<void> {
    this.worker = new Worker(new URL("./nemotron.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent) => {
      const { id, ok, error, ...rest } = e.data;
      const p = this.pending.get(id);
      if (!p) return;
      this.pending.delete(id);
      ok ? p.resolve(rest) : p.reject(new Error(error));
    };
    await this.call("load"); // worker fetches + compiles the int4 sessions (WASM)
    onProgress?.({ file: REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult> {
    if (!this.worker) throw new Error("NemotronEngine.load() not called");
    const copy = audio.samples.slice(); // copy so we can transfer without touching the caller's buffer
    const r = await this.call("transcribe", { audio: copy.buffer, language: this.opts.language }, [copy.buffer]);
    return { text: r.text };
  }

  async dispose(): Promise<void> {
    this.worker?.terminate();
    this.worker = null;
    this.pending.clear();
  }
}
