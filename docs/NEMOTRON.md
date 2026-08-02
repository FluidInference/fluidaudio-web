# Nemotron 3.5 streaming ASR — port notes

Cache-aware FastConformer-RNNT, 40 languages, real-time streaming. ONNX:
`onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4` (encoder 690 MB int4 +
decoder + joint + `silero_vad.onnx` + tokenizer + configs).

## Verified I/O (ort-node; cache threading confirmed across chunks)

```
encoder:
  in : audio_signal[1,65,128]        # 56 new mel frames + 9 pre-encode cache, T-major
       length[1]i64
       cache_last_channel[1,24,56,1024]
       cache_last_time[1,24,1024,8]
       cache_last_channel_len[1]i64
       lang_id[1]i64
  out: outputs[1,7,1024]             # 7 encoder frames per chunk
       cache_last_channel_next / cache_last_time_next / cache_last_channel_len_next
decoder (LSTM): targets[b,t]i32, h_in/c_in[2,b,640] -> decoder_output[b,640,t], h_out/c_out
joint: encoder_output[b,time,1024], decoder_output[b,t,640] -> joint_output[b,time,t,13088]
```

Geometry (`genai_config.json` / `audio_processor_config.json`): `chunk_samples=8960`
(560 ms tier), 24 layers, `left_context=56`, `conv_context=8`, `pre_encode_cache_size=9`.
Mel: `n_fft=512, hop=160, n_mels=128, win=400, hann, preemph=0.97, dither=1e-5,
log add 1e-10, mag_power=2, normalize="NA"` — **no CMVN** (differs from Parakeet's
per-feature norm), and `audio_signal` is **[1,T,mels] T-major** (Parakeet is mel-major).

## Status — BUILD COMPLETE (accuracy pending WebGPU)

- ✅ **Full decode implemented + verified headless** (ort-node, `nemotron-decode.js`):
  NA log-mel (`nemotron-mel.js`, no CMVN, log 1e-10) → 65-frame chunks (9 pre-
  encode cache + 56 new, T-major) → cache-aware encoder (threads `cache_last_channel/
  time`) → RNNT greedy over the 7 enc frames (LSTM `decoder` + `joint`, blank id
  **13087**, max-symbols) → detokenize (`vocab.txt`, ▁→space, skip `<…>`). 2 s clip
  → 4 chunks, pipeline runs clean, shapes + cache/LSTM state thread correctly.
- ⚠️ **0 tokens on CPU** — the expected int4-on-CPU degeneracy (same as Parakeet
  int8; encoder output collapses). **Correct transcripts need WebGPU** — verify by
  ear in the browser. `targets` is int64.
- Engine (`index.ts`, `NemotronEngine`, AsrEngine) wires encoder(WebGPU)+decoder+
  joint(WASM) with **external-data loading** (`externalData: [{path, data}]`, fetches
  `*.onnx.data`) + JS NA-mel + tokenizer. Throws without WebGPU.
- Follow-ups: true streaming `push()` API; move NA-mel to an ONNX/WASM graph;
  `silero_vad.onnx` chunk gating; `lang_id` map for the 40 languages.

## Division of labor (Hamza already has Parakeet v2/v3 + Senko diarization)

Public MIT: `narcotic-sh/senko` (desktop pipeline: pyannote-seg + CAM++ + spectral/
UMAP-HDBSCAN + C++ fbank), `zanshin`, `ffmpeg.wasm`. The **web** Senko (WebGPU) and
his Parakeet web port are **not published** — get them from Hamza directly. So our
net-new focus is **Nemotron** (here) and **Parakeet EOU**; Kokoro + our
internalized Parakeet v3 are done.
