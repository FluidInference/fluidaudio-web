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

## Status

- ✅ **Encoder cache-threading verified**: 3-chunk run, `enc_out[1,7,1024]` each,
  `cache_last_channel_len_next` accumulates 7→14→21, int4 encoder ~30 ms/chunk on
  CPU (dummy mel — plumbing only). The streaming state machine is correct.
- ✅ Engine wires the caches (`reset()` inits zeros; `push()` builds the feeds,
  threads `*_next` back).
- 🚧 **Remaining**: NA-mel (streaming 65-frame chunks with 9-frame overlap,
  normalize=NA, T-major), RNNT greedy decode (decoder LSTM + joint over the 7 enc
  frames, blank/max-symbols), tokenizer (`vocab.txt` / `tokenizer.json`), lang_id
  map. Optionally the bundled `silero_vad.onnx` to gate chunks.
- ⚠️ **Accuracy is browser-only to verify**: int4 needs WebGPU (CPU/WASM likely
  degenerate, like Parakeet int8). Plumbing verifies headless; correctness does
  not. Also: `encoder.onnx` has external data (`.onnx.data`) — ORT-web external-
  data loading (fetch the `.data`, pass via `externalData`) is a TODO for the
  browser engine.

## Division of labor (Hamza already has Parakeet v2/v3 + Senko diarization)

Public MIT: `narcotic-sh/senko` (desktop pipeline: pyannote-seg + CAM++ + spectral/
UMAP-HDBSCAN + C++ fbank), `zanshin`, `ffmpeg.wasm`. The **web** Senko (WebGPU) and
his Parakeet web port are **not published** — get them from Hamza directly. So our
net-new focus is **Nemotron** (here) and **Parakeet EOU**; Kokoro + our
internalized Parakeet v3 are done.
