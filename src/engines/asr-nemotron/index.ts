// Nemotron 3.5 ASR (multilingual FastConformer-RNNT, 24L d1024), fully ORT-free.
//
// The shipped soniqo model is a cache-aware STREAMING export; here we run the same
// FastConformer OFFLINE (whole clip) on raw WebGPU — cache-aware streaming and
// offline-with-limited-context-mask are equivalent, so we skip the cache plumbing.
// Everything hand-written (no onnxruntime):
//   • Mel: JsPreprocessor NA log-mel (no CMVN).
//   • Encoder: shared raw FastConformer (raw-encoder.js) with Nemotron streaming
//     config — causal subsampling pad, causal depthwise (dwK 9), cache-aware mask
//     (chunk 4, left 56, right 3). int8 weights (600M model is int8-robust, unlike
//     the 120M EOU). Runs on WebGPU.
//   • prompt_kernel: multilingual conditioning MLP (concat conformer_out + language
//     one-hot → 1152→2048→1024) → encoded_output.
//   • Decoder+joint: 2-layer LSTM RNNT (raw-decoder-nemotron.js), plain JS.
// Full offline int8 path == coherent transcript matching the ORT streaming reference.

import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";
import { createContext } from "../../gpu/context.js";
import { loadParakeetEncoder, parakeetEncode } from "../asr-parakeet/raw-encoder.js";
import { loadNemotronDecoder, nemotronDecode, loadPromptKernel, applyPromptKernel } from "./raw-decoder-nemotron.js";
import { JsPreprocessor } from "./nemotron-mel.js";

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";
// Nemotron streaming FastConformer config (see raw-encoder.js): causal subsampling
// pad, causal depthwise conv, cache-aware attention (chunk 4, left 56, right 3).
const NEMO_CFG = { melBins: 128, subPad: { t: 2, b: 1, l: 2, r: 1 }, convCausal: true, attChunk: 4, attLeft: 56, attRight: 3 };

export class NemotronEngine implements AsrEngine {
  readonly id = "asr-nemotron";
  readonly label = "Nemotron 3.5 multilingual";
  private ctx: any = null;
  private enc: any = null;
  private pk: any = null;
  private dec: any = null;
  private mel = new JsPreprocessor({ nMels: 128 });
  private vocab: string[] | null = null;
  private langMap: Record<string, number> = {};

  /** @param opts.language e.g. "en-US" / "de" / "zh-CN" / "ja-JP" (default en-US). */
  constructor(private opts: { language?: string } = {}) {}

  async load(onProgress?: ProgressCb): Promise<void> {
    this.ctx = await createContext({ onBackend: (b) => console.info(`[asr-nemotron] backend: ${b}`) });
    const json = async (path: string) =>
      JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path)));
    const bytes = (path: string) => fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path);

    const encMan = await json("nemotron/encoder-int8.manifest.json");
    const encBin = await bytes("nemotron/encoder-int8.bin");
    const decMan = await json("nemotron/decoder-fp32.manifest.json");
    const decBin = await bytes("nemotron/decoder-fp32.bin");
    this.vocab = await json("nemotron/vocab.json");
    const langs = await json("nemotron/languages.json");
    this.langMap = langs.promptDictionary ?? {};

    this.enc = loadParakeetEncoder(this.ctx, encBin, encMan, NEMO_CFG);
    // prompt_kernel weights are fp32 in the encoder blob's fp32 section.
    this.pk = loadPromptKernel(new Float32Array(encBin.buffer, encBin.byteOffset, encBin.byteLength >> 2), encMan);
    this.dec = loadNemotronDecoder(new Float32Array(decBin.buffer, decBin.byteOffset, decBin.byteLength / 4), decMan);
    onProgress?.({ file: WEIGHTS_REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult> {
    if (!this.enc || !this.dec || !this.pk || !this.vocab) throw new Error("NemotronEngine.load() not called");
    const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
    const t0 = now();
    const { features, length } = this.mel.process(audio.samples);
    if (length === 0) return { text: "", metrics: { melMs: 0, encodeMs: 0, decodeMs: 0, totalMs: 0 } };
    const tMel = now();

    const r = await parakeetEncode(this.ctx, this.enc, features, length);
    const conf = await this.ctx.download(r.framesGpu);
    const langId = this.langMap[this.opts.language ?? "en-US"] ?? 0;
    const enc = applyPromptKernel(this.pk, conf, r.Tsub, langId);
    const tEnc = now();

    const { ids } = nemotronDecode(this.dec, enc, r.Tsub);
    const text = ids.map((i: number) => this.vocab![i] ?? "").filter((tk: string) => !tk.startsWith("<")).join("").replace(/▁/g, " ").trim();
    const tDec = now();

    return {
      text,
      metrics: {
        melMs: +(tMel - t0).toFixed(1),
        encodeMs: +(tEnc - tMel).toFixed(1),
        decodeMs: +(tDec - tEnc).toFixed(1),
        totalMs: +(tDec - t0).toFixed(1),
      },
    };
  }

  async dispose(): Promise<void> {
    this.ctx?.device?.destroy?.();
    this.ctx = this.enc = this.pk = this.dec = this.vocab = null;
  }
}
