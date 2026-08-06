// Parakeet TDT 0.6B v3 — fully ORT-free. Mel (parakeet-mel.js), FastConformer
// encoder (raw-encoder.js, int8 on raw WebGPU), and TDT decoder+joint
// (raw-decoder-wasm.js, WASM-SIMD on CPU) are all hand-written; no onnxruntime.
// Weights from FluidInference/fluidaudio-web. Long audio windowed (15s / 2s overlap).
//
// Split rationale: the encoder is GPU (big GEMMs); the RNNT decoder is CPU/WASM-SIMD
// because it's autoregressive — one result per token — so a GPU decoder pays a
// round-trip per token (the ~20× wall). WASM-SIMD decodes on CPU with no GPU sync.

import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";
import { createContext } from "../../gpu/context.js";
import { loadParakeetEncoder } from "./raw-encoder.js";
import { loadWasmDecoder } from "./raw-decoder-wasm.js";
import { transcribeWindowed } from "./pipeline.js";
import { createDecodePool } from "./decode-pool.js";
import { ParakeetMel } from "./parakeet-mel.js";
import { ParakeetTokenizer } from "./tokenizer.js";
import wasmUrl from "./parakeet-decoder.wasm?url";

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";
const VOCAB_REPO = "ysdede/parakeet-tdt-0.6b-v3-onnx";
const SAMPLE_RATE = 16000;
const WINDOW_SEC = 15;
const OVERLAP_SEC = 2;

export class ParakeetV3Engine implements AsrEngine {
  readonly id = "asr-parakeet";
  readonly label = "Parakeet TDT 0.6B v3";
  private ctx: any = null;
  private enc: any = null;
  private dec: any = null;
  private mel: ParakeetMel | null = null;
  private encProjW: any = null;
  private encProjB: any = null;
  private tokenizer: ParakeetTokenizer | null = null;
  private decodePool: any = null;

  async load(onProgress?: ProgressCb): Promise<void> {
    this.ctx = await createContext({ onBackend: (b) => console.info(`[asr-parakeet] backend: ${b}`) });
    if (this.ctx.device) console.info(`[asr-parakeet] shader-f16: ${this.ctx.hasF16 ? "active" : "ABSENT (fp32 fallback, ~1.5x slower encode)"}`);
    const json = async (path: string, repo = WEIGHTS_REPO) =>
      JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(repo, path), onProgress, path)));
    const bytes = (path: string) => fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path);

    const encMan = await json("parakeet/encoder-int8.manifest.json");
    const encBin = await bytes("parakeet/encoder-int8.bin");
    const decMan = await json("parakeet/decoder-fp32.manifest.json");
    const decBin = await bytes("parakeet/decoder-fp32.bin");
    const vocab = new TextDecoder().decode(await fetchCached(hfUrl(VOCAB_REPO, "vocab.txt"), onProgress, "vocab.txt"));
    const wasmBytes = await (await fetch(wasmUrl)).arrayBuffer();

    this.enc = loadParakeetEncoder(this.ctx, encBin, encMan);
    const decF32 = new Float32Array(decBin.buffer, decBin.byteOffset, decBin.byteLength / 4);
    this.dec = await loadWasmDecoder(wasmBytes, decF32, decMan);
    // The joint's encoder projection (1024→640) runs on the GPU before download:
    // 37% smaller readback and no per-frame GEMV in the wasm decoder.
    {
      const g = (k: string) => decF32.subarray(decMan[k].offset, decMan[k].offset + decMan[k].len);
      this.encProjW = this.ctx.upload(g("encW").slice(), 1024, 640);
      this.encProjB = this.ctx.upload(g("encB").slice(), 1, 640);
    }
    this.mel = new ParakeetMel(128);
    this.tokenizer = ParakeetTokenizer.fromVocabText(vocab);

    // Decoder worker pool: windows decode independently, and on machines where
    // WASM decode dominates the wall (measured: decode 1821ms vs encode 355ms)
    // parallel workers cut the decode term ~pool-size×. Each worker gets its own
    // copy of the decoder weights (no SharedArrayBuffer without COOP/COEP).
    if (typeof Worker !== "undefined") {
      const n = Math.min(4, Math.max(1, ((navigator as any).hardwareConcurrency || 4) - 2));
      if (n > 1) {
        const raw: Worker[] = [];
        try {
          // Avoid an extra 72MB copy when the view already spans the whole buffer
          // (structured clone copies per worker regardless).
          const decBuf = decBin.byteOffset === 0 && decBin.byteLength === decBin.buffer.byteLength
            ? decBin.buffer
            : decBin.buffer.slice(decBin.byteOffset, decBin.byteOffset + decBin.byteLength);
          const workers = await Promise.all(
            Array.from({ length: n }, async () => {
              const w = new Worker(new URL("./decoder-worker.js", import.meta.url), { type: "module" });
              raw.push(w);
              await new Promise<void>((resolve, reject) => {
                // Init must reply {type:"ready"} — an {type:"err"} reply or a
                // Worker error event rejects (never resolve-on-any-message).
                w.onmessage = (e) => (e.data?.type === "ready" ? resolve() : reject(new Error(String(e.data?.error ?? "bad init reply"))));
                w.onerror = (e) => reject(new Error(e.message || "worker error"));
                w.postMessage({ type: "init", wasmBytes, decBuf, man: decMan });
              });
              return {
                postMessage: (m: any, t?: any[]) => w.postMessage(m, t ?? []),
                setHandler: (f: (m: any) => void) => { w.onmessage = (e) => f(e.data); },
                terminate: () => w.terminate(),
              };
            }),
          );
          // Post-init transport errors reject all in-flight decodes instead of hanging.
          const pool = createDecodePool(workers);
          for (const w of raw) w.onerror = (e) => pool.failAll(new Error(e.message || "decode worker died"));
          this.decodePool = pool;
          console.info(`[asr-parakeet] decode pool: ${n} workers`);
        } catch (e) {
          for (const w of raw) w.terminate(); // don't leak partially-spawned workers (each holds a weight copy)
          console.warn("[asr-parakeet] decode pool unavailable, decoding on main thread:", e);
          this.decodePool = null;
        }
      }
    }
    onProgress?.({ file: WEIGHTS_REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult> {
    if (!this.enc || !this.dec || !this.mel || !this.tokenizer) throw new Error("ParakeetV3Engine.load() not called");
    const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
    const t0 = now();
    // Windowed 3-stage pipeline (pipeline.js, shared with the node gates):
    // GPU encodes group g+1 while the CPU runs mel for g+2 and decodes g.
    const { ids, stats } = await transcribeWindowed(this.ctx, this.enc, this.dec, this.mel, this.encProjW, this.encProjB, audio.samples, {
      sampleRate: SAMPLE_RATE,
      windowSec: WINDOW_SEC,
      overlapSec: OVERLAP_SEC,
      decodePool: this.decodePool,
    });
    // Stages overlap (pipelined); encodeMs is the GPU wait NOT hidden behind CPU
    // work, so mel + encode + decode ≈ wall. GPU-bound shows encode dominating.
    return {
      text: this.tokenizer.decode(ids),
      metrics: { melMs: stats.melMs, encodeMs: stats.encWaitMs, decodeMs: stats.decodeMs, totalMs: +(now() - t0).toFixed(0) },
    };
  }

  async dispose(): Promise<void> {
    this.decodePool?.terminate?.();
    this.decodePool = null;
    this.ctx?.device?.destroy?.();
    this.ctx = this.enc = this.dec = this.mel = this.tokenizer = null;
  }
}
