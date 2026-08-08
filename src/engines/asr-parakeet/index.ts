// Parakeet TDT 0.6B v3 — fully ORT-free. Mel (parakeet-mel.js), FastConformer
// encoder (raw-encoder.js, int8 on raw WebGPU), and TDT decoder+joint
// (raw-decoder-wasm.js, WASM-SIMD on CPU) are all hand-written; no onnxruntime.
// Weights from FluidInference/fluidaudio-web. Long audio windowed (15s / 2s overlap).
//
// Split rationale: the encoder is GPU (big GEMMs); the RNNT decoder is CPU/WASM-SIMD
// because it's autoregressive — one result per token — so a GPU decoder pays a
// round-trip per token (the ~20× wall). WASM-SIMD decodes on CPU with no GPU sync.

import { fetchCached, hfUrl } from "../../core/modelCache.js";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types.js";
import { createContext } from "../../gpu/context.js";
import { loadParakeetEncoder } from "./raw-encoder.js";
import { loadWasmDecoder } from "./raw-decoder-wasm.js";
import { transcribeWindowed } from "./pipeline.js";
import { createDecodePool, browserWorkerShim, initDecodeWorker } from "./decode-pool.js";
import { loadTextNorm, itn } from "../../core/textnorm.js";
import { createVocabularyRescorer } from "./vocab-rescorer.js";
import { ParakeetMel } from "./parakeet-mel.js";
import { ParakeetTokenizer } from "./tokenizer.js";
const wasmUrl = new URL("./parakeet-decoder.wasm", import.meta.url); // cross-bundler asset URL

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
  private poolSrc: { wasmBytes: ArrayBuffer; decBin: any; decMan: any } | null = null;
  private itnMod: any | null = null;
  private itnEnabled = false;
  private rescorer: ReturnType<typeof createVocabularyRescorer> | null = null;

  /** OPT-IN inverse text normalization ("twenty one dollars" → "$21").
   * Off by default: on everyday speech it also rewrites words people write out
   * ("no one" → "no 1") and can delete words in non-English transcripts. */
  setItn(enabled: boolean): void {
    this.itnEnabled = enabled;
  }

  /** Custom vocabulary (domain terms, names): fuzzy-matched against the
   * transcript and replaced with canonical spellings ("invidia" → "NVIDIA",
   * "new res" → "Newrez"). Pass [] to clear. */
  setVocabulary(terms: Array<string | { text: string; aliases?: string[]; minSimilarity?: number }>): void {
    this.rescorer = terms.length ? createVocabularyRescorer(terms) : null;
  }

  async load(onProgress?: ProgressCb): Promise<void> {
    this.ctx = await createContext({ onBackend: (b) => console.info(`[asr-parakeet] backend: ${b}`) });
    if (this.ctx.device) console.info(`[asr-parakeet] shader-f16: ${this.ctx.hasF16 ? "active" : "ABSENT (fp32 fallback, ~1.5x slower encode)"}`);
    if (this.ctx.device && typeof navigator !== "undefined" && navigator.gpu) {
      // Capability report for the next optimization tier: cooperative-matrix
      // GEMM needs subgroup-matrix support. Probe the ADAPTER — the device
      // only carries features requested at creation.
      const a = await navigator.gpu.requestAdapter();
      const has = (k: string) => (a?.features.has(k) ? "yes" : "no");
      console.info(`[asr-parakeet] adapter subgroups: ${has("subgroups")} · subgroup-matrix: ${has("chromium-experimental-subgroup-matrix")}`);
    }
    const json = async (path: string, repo = WEIGHTS_REPO) => JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(repo, path), onProgress, path)));
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

    // Decoder worker pool is spawned LAZILY on the first multi-window
    // transcribe (see ensurePool): each worker holds a full copy of the
    // decoder weights (~72MB) — a 3-second clip should not pay ~290MB of RAM
    // and 4 worker spawns for a single-window decode.
    this.poolSrc = { wasmBytes, decBin, decMan };
    // ITN module loaded up front; APPLIED only when setItn(true) (see setItn).
    this.itnMod = await loadTextNorm();
    onProgress?.({ file: WEIGHTS_REPO, loaded: 1, total: 1, fraction: 1 });
  }

  private async ensurePool(): Promise<void> {
    if (this.decodePool || !this.poolSrc || typeof Worker === "undefined") return;
    const { wasmBytes, decBin, decMan } = this.poolSrc;
    this.poolSrc = null; // one attempt; fall back to main-thread decode on failure
    const n = Math.min(4, Math.max(1, ((navigator as any).hardwareConcurrency || 4) - 2));
    if (n <= 1) return;
    const raw: Worker[] = [];
    try {
      const decBuf =
        decBin.byteOffset === 0 && decBin.byteLength === decBin.buffer.byteLength
          ? decBin.buffer
          : decBin.buffer.slice(decBin.byteOffset, decBin.byteOffset + decBin.byteLength);
      const workers = await Promise.all(
        Array.from({ length: n }, async () => {
          const w = new Worker(new URL("./decoder-worker.js", import.meta.url), { type: "module" });
          raw.push(w);
          await initDecodeWorker(
            (m: any) => w.postMessage(m),
            (ok: (m: any) => void, err: (e: any) => void) => {
              w.onmessage = (e) => ok(e.data);
              w.onerror = (e) => err(new Error(e.message || "worker error"));
            },
            { wasmBytes, decBuf, man: decMan },
          );
          return browserWorkerShim(w);
        }),
      );
      const pool = createDecodePool(workers);
      for (const w of raw) w.onerror = (e) => pool.failAll(new Error(e.message || "decode worker died"));
      this.decodePool = pool;
      console.info(`[asr-parakeet] decode pool: ${n} workers`);
    } catch (e) {
      for (const w of raw) w.terminate();
      console.warn("[asr-parakeet] decode pool unavailable, decoding on main thread:", e);
      this.decodePool = null;
    }
  }

  async transcribe(audio: AudioData): Promise<AsrResult> {
    if (!this.enc || !this.dec || !this.mel || !this.tokenizer) throw new Error("ParakeetV3Engine.load() not called");
    const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
    const t0 = now();
    if (audio.samples.length > 2 * WINDOW_SEC * SAMPLE_RATE) await this.ensurePool(); // multi-window → parallel decode pays
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
    // Vocabulary rescoring runs on the RAW spoken-form transcript, BEFORE any
    // ITN — spoken-form aliases ("gpt four") can never match post-ITN text.
    let text = this.tokenizer.decode(ids);
    if (this.rescorer) text = this.rescorer.rescore(text);
    if (this.itnEnabled) text = itn(this.itnMod, text);
    return {
      text,
      metrics: { melMs: stats.melMs, encodeMs: stats.encWaitMs, decodeMs: stats.decodeMs, totalMs: +(now() - t0).toFixed(0) },
    };
  }

  async dispose(): Promise<void> {
    this.poolSrc = null; // release retained decoder bytes (~72MB) if the pool never spawned
    this.decodePool?.terminate?.();
    this.decodePool = null;
    this.ctx?.device?.destroy?.();
    this.ctx = this.enc = this.dec = this.mel = this.tokenizer = null;
  }
}
