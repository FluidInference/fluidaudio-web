# Parakeet EOU 120M — port notes

NVIDIA `parakeet_realtime_eou_120m-v1`: a streaming FastConformer **RNNT** with two
extra control tokens — `<EOU>` (end of utterance) and `<EOB>` — so a voice agent
can detect when the user has stopped speaking. ONNX export by the asrjs project:
`ysdede/parakeet-realtime-eou-120m-v1-onnx` (fp32/fp16/int8 encoder + fused
`decoder_joint` + `vocab.txt` + `config.json`).

> The earlier "no public ONNX export" status was stale — several exports now
> exist (`ysdede/`, `CHRV/`, `tteokl/`, `talatapp/`, `soniqo/`, and
> `adityakalro/sherpa-onnx-parakeet-eou-120m`).

## Verified I/O (ort-node)

```
encoder:        audio_signal[1,128,T], length[1]i64 -> outputs[1,512,Tenc]   (mel-major, no cache)
decoder_joint:  encoder_outputs[1,512,1], targets[1,1]i32,
                input_states_1/2[1,1,640]
                -> outputs[1,1,2,1027], output_states_1/2[1,1,640]
```

`config.json`: `mel_bins=128`, `frame_shift=0.01`, `subsampling_factor=8`,
`pred_hidden=640`, `pred_layers=1`, `max_symbols_per_step=10`, `eou_id=1024`,
`eob_id=1025`, `blank_id=1026`. `vocab.txt` is a plain 0-indexed list (id = line
number); `<EOU>`/`<EOB>` are the last two entries, blank (1026) is implicit.

## Decode

Standard RNNT greedy — the same shape as Parakeet TDT but **no duration bins**
(advance one encoder frame per blank) and a **single-layer** LSTM state:

1. NA log-mel (reuse the Nemotron frontend) → encoder → `[1,512,Tenc]`, transpose
   to per-frame `[512]` vectors (encoder frame = 80 ms = 10 ms hop × 8).
2. Per frame: `decoder_joint(frame, targets=[lastToken], input_states)`. The
   output is `[1,1,2,1027]` — the **last 1027 values** are this step's logits.
   argmax; blank = 1026.
3. Non-blank → emit, thread `output_states` back in, keep the same frame (up to
   `max_symbols_per_step`); blank → advance the frame.
4. `<EOU>`/`<EOB>` (ids ≥ 1024) are recorded as timestamped events and dropped
   from the transcript.

## The mel gotcha (this cost the debugging)

`parakeet-realtime-eou` expects **NA (un-normalized) log-mel** — the same frontend
Nemotron uses — **not** Parakeet TDT's per-feature CMVN. asrjs confirms this:
`src/models/nemo-rnnt/config.ts` sets `preprocessorNormalization: 'none'`.

Feeding per-feature-normalized mel (e.g. Parakeet's `nemo128.onnx`) makes the
encoder produce **content-free** frames: healthy magnitude (per-frame RMS ≈ 0.46,
std ≈ 0.04 — nearly flat across the whole clip) but the joint predicts blank
(logprob ≈ 0) on **every** step → empty transcript. It reads like a decode bug
but is entirely the frontend. Switching to the NA mel (`JsPreprocessor` from
`asr-nemotron/nemotron-mel.js`, log guard 1e-10) fixes it immediately.

## Status — ✅ WORKING (ort-node; browser via `scripts/smoke-eou.mjs` parity)

12 s LibriVox intro → `"four classes that a menace fromanti suffrage ten good
reasons by grace duffield…"` (35 tokens, 0.3 s) + `<EOU>@12s`. Reference (larger
Nemotron 600M): "Four classes constitute a menace from anti suffrage ten good
reasons by Grace Duffield" — the 120 M streaming model is a little looser, as
expected; the point is real content + reliable EOU emission.

The fp32 encoder (~460 MB) decodes correctly on **both** WASM and WebGPU (unlike
Parakeet's int8, which collapses on WASM), so WebGPU is preferred but not
required. The fused decoder + NA mel run on WASM.

Follow-ups: true streaming `push()` API (this export has no cache I/O, so a
streaming loop re-encodes the buffer); fp16 encoder to halve the download.
