# Benchmarks

## Browser verification (bench.html, in-Chrome)

Verified via the auto-benchmark in real Chrome on the 12s bundled sample
(`docs/sample-bench.json`). WebGPU-capable context:

| Engine | ok | run | RTFx | output |
|---|---|--:|--:|---|
| Sortformer diarization | ✅ | 173 ms | 69.4× | 1 speaker, 5 segments (correct) |
| Kokoro TTS (en) | ✅ | 760 ms | 4.44× | 3.38 s audio |

RTFx varies with WebGPU-vs-WASM fallback (Sortformer ranged 15×–69× across runs).
**Parakeet/Nemotron (int8/int4) return empty even when `navigator.gpu` is present
in the automation context** — their quantized encoders need a *fully functional*
WebGPU adapter; verify on a real machine. VAD errors (`vad-web` CJS + Vite).



Reproduce:
```bash
node scripts/bench-parakeet.mjs fp32 /tmp/pkv3           # ASR (needs local ONNX + wavs)
node scripts/bench-kokoro.mjs                            # TTS
```
In the browser, every run prints `⏱ ms · RTFx` in the output panel (WebGPU path).

> **These are `onnxruntime-node` CPU baselines**, not the browser WebGPU numbers.
> They exist to (a) prove correctness headlessly and (b) profile the stages. The
> shipped browser path runs the Parakeet encoder and Kokoro on **WebGPU**, which
> is materially faster for those stages. Measure the real numbers via `npm run
> dev` in Chrome.

## Parakeet TDT v3 — ASR (ort-node CPU, fp32 encoder, M-series, 15 threads)

| audio | mel (WASM) | encode | decode | total | RTFx | tokens |
|--:|--:|--:|--:|--:|--:|--:|
| 12 s | 7.6 ms | 157 ms | 9.7 ms | 175 ms | 68.6× | 41 |
| 30 s | 21 ms | 385 ms | 24 ms | 431 ms | 69.7× | 60 |
| 60 s | 38 ms | 895 ms | 162 ms | 1094 ms | 54.8× | 207 |
| 120 s | 77 ms | 2086 ms | 529 ms | 2691 ms | 44.6× | 434 |

- The **encoder dominates** and RTFx falls with length: the offline encoder uses
  full attention over the whole clip (~quadratic), no chunking. For long files a
  chunked/streaming encoder (or windowed batching, as FluidAudio's Swift path
  does) would flatten this. The encoder is the stage WebGPU accelerates in-browser.
- mel (`nemo128.onnx`) and the decoder run on WASM in both CPU and browser, so
  those columns transfer directly.
- int8 encoder is CPU-degenerate (empty output) — benchmarked in fp32; the browser
  runs int8 on WebGPU.

### Accuracy (correctness of the internalized port)

`node scripts/wer-parakeet.mjs 30` — FLEURS en_us, fp32 encoder:

| files | audio | WER | avg RTFx |
|--:|--:|--:|--:|
| 30 | 289 s | **6.15%** (41/667 words) | 43× |

In line with reference Parakeet TDT v3 on FLEURS (a harder set than LibriSpeech),
confirming the internalized pipeline (ONNX mel + our tokenizer + our TDT decode)
is correct across many files, not just one clip.

### WebGPU numbers — not measurable in this env

Chrome here reports `no navigator.gpu` (headed and headless — the automation
context has no GPU), so real browser WebGPU RTFx must be captured on a normal
machine via `npm run dev` (the UI prints `⏱ ms · RTFx` per run). The CPU numbers
above are the fp32 baseline; in-browser the encoder runs int8 on WebGPU.

## Kokoro TTS (kokoro-js, ort-node CPU, q8)

| chars | gen | audio | RTFx | chars/s |
|--:|--:|--:|--:|--:|
| 12 | 630 ms | 1.65 s | 2.6× | 19 |
| 70 | 1894 ms | 4.75 s | 2.5× | 37 |
| 242 | 6242 ms | 16.9 s | 2.7× | 39 |

- Sustained ~2.5–2.7× RTF, ~39 chars/s on CPU. Upstream reports Kokoro **WebGPU**
  at ~10 s audio per ~1 s (≈10× RTF) — expect a similar jump in-browser.

## Not yet measured
- **Silero VAD** — browser-only (`@ricky0123/vad-web`, AudioWorklet); measure via
  the dev server.
- **Nemotron / diarization / EOU** — scaffolds.
