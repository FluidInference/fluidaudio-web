// Whisper ASR (whisper-base, 99 languages), fully ORT-free.
//
// Everything hand-written on raw WebGPU + JS (no onnxruntime / transformers.js):
//   • Mel: WhisperMel (80-bin log-mel, direct 400-pt DFT, JS) — 1.4e-5 vs transformers.
//   • Encoder: raw WebGPU conv stem + 6 PRE-LN transformer layers (erf-GELU).
//   • Decoder: raw WebGPU autoregressive greedy (causal self-attn + cross-attn),
//     GPT-2 byte-level BPE detokenizer, forced prefix + suppress tokens.
// Full raw pipeline == the transformers.js transcript. Single 30s window (long-audio
// chunking is a follow-up). Weights from FluidInference/fluidaudio-web (fp32).

import { fetchCached, hfUrl } from "../../core/modelCache";
import type { AsrEngine, AsrResult, AudioData, ProgressCb } from "../../core/types";
import { GpuContext, requestGpuDevice } from "../../gpu/compute.js";
import { loadWhisperEncoder, whisperEncode } from "./raw-whisper-encoder.js";
import { loadWhisperDecoder, whisperCrossKV, whisperDecodeStep } from "./raw-whisper-decoder.js";
import { makeWhisperTokenizer } from "./whisper-tokenizer.js";
import { WhisperMel } from "./whisper-mel.js";
import melFiltersUrl from "./whisper-mel-filters.bin?url";
import suppressTokens from "./whisper-suppress.json";

const WEIGHTS_REPO = "FluidInference/fluidaudio-web";
const VOCAB_REPO = "onnx-community/whisper-base";
// Forced decoder prefix: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>.
const PREFIX = [50258, 50259, 50359, 50363], EOT = 50257, MAX_NEW = 220;

export class WhisperEngine implements AsrEngine {
  readonly id = "asr-whisper";
  readonly label = "Whisper (99 langs)";
  private ctx: any = null;
  private enc: any = null;
  private dec: any = null;
  private mel: WhisperMel | null = null;
  private tokenizer: ReturnType<typeof makeWhisperTokenizer> | null = null;
  private suppress = new Set<number>(suppressTokens as number[]);

  async load(onProgress?: ProgressCb): Promise<void> {
    this.ctx = new GpuContext(await requestGpuDevice());
    const json = async (path: string, repo = WEIGHTS_REPO) =>
      JSON.parse(new TextDecoder().decode(await fetchCached(hfUrl(repo, path), onProgress, path)));
    const bytes = (path: string) => fetchCached(hfUrl(WEIGHTS_REPO, path), onProgress, path);

    const encMan = await json("whisper/encoder-fp32.manifest.json");
    const encBin = await bytes("whisper/encoder-fp32.bin");
    const decMan = await json("whisper/decoder-fp32.manifest.json");
    const decBin = await bytes("whisper/decoder-fp32.bin");
    const vocab = await json("vocab.json", VOCAB_REPO);
    const melFilters = new Float32Array(await (await fetch(melFiltersUrl)).arrayBuffer());

    this.enc = loadWhisperEncoder(this.ctx, new Float32Array(encBin.buffer, encBin.byteOffset, encBin.byteLength / 4), encMan);
    this.dec = loadWhisperDecoder(this.ctx, new Float32Array(decBin.buffer, decBin.byteOffset, decBin.byteLength / 4), decMan);
    this.mel = new WhisperMel(melFilters);
    this.tokenizer = makeWhisperTokenizer(vocab);
    onProgress?.({ file: WEIGHTS_REPO, loaded: 1, total: 1, fraction: 1 });
  }

  async transcribe(audio: AudioData): Promise<AsrResult> {
    if (!this.enc || !this.dec || !this.mel || !this.tokenizer) throw new Error("WhisperEngine.load() not called");
    const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
    const t0 = now();
    const { features } = this.mel.process(audio.samples);
    const encG = whisperEncode(this.ctx, this.enc, features);
    const kv = whisperCrossKV(this.ctx, this.dec, encG);
    const tMel = now();

    const tokens = [...PREFIX];
    for (let step = 0; step < MAX_NEW; step++) {
      const logits = await whisperDecodeStep(this.ctx, this.dec, kv, tokens);
      for (const t of this.suppress) logits[t] = -Infinity;
      if (step === 0) { logits[220] = -Infinity; logits[EOT] = -Infinity; } // begin_suppress
      let maxId = 0, maxV = -Infinity;
      for (let i = 0; i < logits.length; i++) if (logits[i] > maxV) { maxV = logits[i]; maxId = i; }
      if (maxId === EOT) break;
      tokens.push(maxId);
    }
    const text = this.tokenizer.decode(tokens.slice(PREFIX.length)).trim();
    return { text, metrics: { melMs: +(tMel - t0).toFixed(0), encodeMs: 0, decodeMs: +(now() - tMel).toFixed(0), totalMs: +(now() - t0).toFixed(0) } };
  }

  async dispose(): Promise<void> {
    this.ctx?.device?.destroy?.();
    this.ctx = this.enc = this.dec = this.mel = this.tokenizer = null;
  }
}
