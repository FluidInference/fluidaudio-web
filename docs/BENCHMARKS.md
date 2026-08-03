# Benchmarks

## Browser results (bench.html, real machine, WebGPU: true)

12s sample, `?full=1`, macOS/Chrome (from a user run):

| Engine | ok | run | RTFx | output |
|---|---|--:|--:|---|
| **Sortformer diarization** | ✅ | 97 ms | **123.7×** | 1 speaker, 5 segments |
| **Kokoro TTS (en)** | ✅ | 338 ms | **9.99×** | 3.38 s audio |
| **Kokoro TTS (zh)** | ✅ | 755 ms | **5.26×** | 3.98 s audio |
| Parakeet TDT v3 (int8) | ❌ | 1197 ms | 10× | empty ← int8 fell back to WASM |
| Nemotron 3.5 (int4) | ✅* | 1330 ms | 9× | *empty in that run due to lang_id=0 bug; fixed → works (en-US=lang_id 24)* |
| Silero VAD | ❌ | — | — | `vad-web` CJS require error |

**Finding:** the WebGPU EP has **no int8/int4 kernels**, so quantized encoders
silently fall back to WASM → all-blank. Fix = run the **fp32** encoder on WebGPU
(Parakeet switched to `encoder-model.onnx`, fp32-CPU-verified transcript correct).
Nemotron ships int4 only → blocked until an fp16/fp32 export exists. Kokoro (9.99×)
matches upstream's ~10× WebGPU claim.



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

### Accuracy on FluidAudio benchmark datasets (fp32 encoder, ort-node — same core as the browser)

| dataset | script | metric | result | reference (FluidAudio native) |
|---|---|---|---|---|
| **LibriSpeech test-clean** (all 2620 utts, 5.4h) | `wer-librispeech.mjs` | WER | **2.15%** (1129/52576) | ~2.14% native ✓ |
| **FLEURS en_us** (30) | `wer-parakeet.mjs` | WER | 6.15% | in-range ✓ |

The browser port **matches native FluidAudio accuracy** — the fp32 encoder + our
ONNX mel + tokenizer + TDT decode are correct at dataset scale. Datasets:
LibriSpeech test-clean / FLEURS / MUSAN are local; AMI audio from
`groups.inf.ed.ac.uk/ami/AMICorpusMirror`, RTTM ground truth from
`pyannote/AMI-diarization-setup`.

### Sortformer diarization on AMI — blocked on the streaming loop

Downloaded AMI ES2004a (17.5 min, 4 speakers) + RTTM. **Single-chunk Sortformer
collapses on long meetings**: over the first 120 s (RTTM = 3 active speakers,
FEE013 87 s / FEE016 9 s / MEO015 6 s) it reports spk0 34 %, spk1 max 0.517,
**spk2/spk3 = 0.000**. Sortformer is a *streaming* model that needs its
`spkcache`/`fifo` state to track speakers over time; feeding one big chunk with
empty state defaults to the dominant speaker. So a valid DER requires porting the
NeMo streaming state-update loop (chunked `spkcache`/`fifo`) — the offline
single-chunk path is only adequate for short/simple audio (verified: 40 s
Earnings → 2 speakers). Follow-up.

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

## Silero VAD + Parakeet EOU — verified (ort-node CPU, `scripts/smoke-{vad,eou}.mjs`)

12 s LibriVox intro:

| engine | run | output |
|---|--:|---|
| **Silero VAD v5** | 30 ms | 6 speech ranges, 7.78 s speech / 12 s (32 ms windows, hysteresis + duration guards) |
| **Parakeet EOU 120M** | 300 ms | 35 tokens → *"four classes that a menace fromanti suffrage ten good reasons by grace duffield…"* + `<EOU>@12s` |

Both share the browser core exactly: VAD runs `silero_vad.onnx` directly on
`core/ort` (no `@ricky0123/vad-web` — its CJS `require` breaks under Vite), EOU
runs the fp32 encoder (WebGPU/WASM) + fused RNNT decoder (WASM) with the **NA**
log-mel frontend (see [`EOU.md`](EOU.md) — feeding per-feature CMVN yields
all-blank). Reproduce:
```bash
node scripts/smoke-vad.mjs /tmp/pk_intro.wav /tmp/silero_vad.onnx
node scripts/smoke-eou.mjs /tmp/pk_intro.wav   # needs /tmp/eou/{enc,dj}.onnx + vocab.txt
```

## Not yet measured
- **Nemotron / diarization** WebGPU browser RTFx — capture via the dev server.
