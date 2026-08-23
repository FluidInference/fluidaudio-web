# VoiceChat-11B in the browser — status and roadmap

[nvidia/NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
is a full-duplex speech-to-speech + tool-calling model (11.1B params, English,
OpenMDW-1.1). This doc records what ships in fluidaudio-web now, what is
blocked, and the evidence behind both. Numbers marked "native" were measured
on an M5 Pro during the CoreML port of the same checkpoint (2026-08-03/04);
they are the budget ceiling a browser port competes against, not browser
results.

## Component map

Per 80 ms frame the full-duplex loop runs:

| Component | Size | Native (M5 Pro) | Browser verdict |
| --- | --- | --- | --- |
| Fast Conformer encoder (24L, d1024, att [70,0] causal, 8× subsample) | 609M | 11–12 ms/frame ANE | **Ships now** — same runtime family as our Nemotron streaming ASR |
| RNNT user-transcript head (2-layer LSTM 640, vocab 1025) | 34 MB | 0.5 ms/step | **Ships now** |
| NemotronH 9B LLM (Mamba2-hybrid, 56 layers) + lm/function heads | 9B / 19 GB fp16 | 43.5 ms/step (CoreML int8 stateful shards, 100% prefill parity); 21.2 ms/tok (MLX 4-bit) | **Blocked** (see below) |
| TTS: Gemma3 backbone 28L×1152 + MoG head | 595M + 159M | ~6 ms/step proxy; CFG 0.2 → 2 evals/step | **Ships now** — `tts-voicechat` (standalone Aria voice, code-matrix bit-exact vs torch; see phase 2) |
| Audio codec decoder (31-quantizer PRVQ, 12.5 Hz → 22.05 kHz) | 763 MB | unmeasured | **Ships now** — part of `tts-voicechat` (waveform NRMSE 1.1e-6; ~24 GPU-ms/s of audio on WebGPU, RTFx 0.33 node/WASM) |

## What ships now: the STT slice (`asr-voicechat`)

The encoder + RNNT chain is geometry-compatible with the FastConformer
WebGPU/WASM runtime already serving `asr-parakeet` / `asr-nemotron`
(manifest-inferred d_model/layers/heads, identical mel frontend: 128 bins,
hop 160, no CMVN). Differences from Nemotron: fully-causal attention
([70,0] look-back, no right-context lookahead), vocab 1025 with blank 1024,
and no multilingual prompt-kernel MLP. The chain was validated natively
token-identical against the torch reference on real audio, which gives the
browser port a hard parity oracle ("Hello, do you know what color the sky
is" on the reference sample).

## Why the full-duplex loop is blocked in the browser

1. **Memory.** The only quantization measured lossless on the fine-tuned 9B
   is int8 (per-block-32 RTN: exactly 100% top-1 agreement, KL 0.006).
   That is ~9.4 GB of LLM weights, ~11–13 GB resident with encoder + TTS +
   codec — beyond practical WebGPU per-origin buffer budgets today (compare:
   iPhone adapters cap single buffers around 1 GB; desktop Chrome is larger
   but not 13 GB of storage buffers).
2. **Quality at 4-bit.** Naive int4 RTN measured **74.3% top-1 — not
   shippable** (int4 + int8 heads: 80.6%). A browser-sized LLM needs a
   calibrated quant (GPTQ/AWQ) or a sensitivity-mixed int8/int4 scheme, and
   that quality work has not been done.
3. **Frame budget.** Native CoreML int8 does 43.5 ms/step — inside the 80 ms
   frame with ~65–72 ms total. Our WebGPU kernels are well behind native
   Metal on decode-shaped GEMV; there is no evidence the 9B step fits 80 ms
   in a browser, and full-duplex has no slack for falling behind real time.

Consequences: no browser full-duplex until (a) a calibrated ≤5-bit quant
proves lossless-enough on this checkpoint and (b) a measured WebGPU decode
step fits the budget. A half-duplex push-to-talk variant (transcribe →
generate → speak, no 80 ms deadline) becomes plausible as soon as (a) exists,
at roughly 5–6 GB of weights.

## Phased plan

1. **Phase 1 (now): `asr-voicechat` STT engine** — encoder + RNNT, batch +
   streaming, parity-gated on the reference transcript. Local weights first;
   HF hosting after the FluidInference repo target is confirmed.
2. **Phase 2: codec + TTS decoders — SHIPPED 2026-08 as `tts-voicechat`.**
   The full speech-decoder slice runs standalone as a TTS voice: Gemma3
   backbone (28L×1152, CFG 0.2 → 2 evals/step, KV-cached) + CAS t5gemma
   char-aware text conditioning + gated audio/text fusion + MoG head with
   8-iteration PRVQ unmasking + the 31-quantizer codec decoder (12.5 Hz →
   22.05 kHz). Parity-gated against a torch reference run of the real
   checkpoint (`scripts/voicechat-tts-reference.py` → goldens;
   `scripts/ci-smoke-voicechat-tts.mjs` → gate): the deterministic track
   (noise 0, argmax-component MoG, CFG kept at 0.2) is **bit-exact on the
   full 50×31 audio-code matrix** and waveform NRMSE 1.3e-6. Two
   reference-as-run findings are baked into the export: the HF sdpa path
   silently drops T5Gemma's attn softcapping (the CAS runs uncapped), and
   backbone GEMMs ship f32 because f16 flips an MoG/PRVQ near-tie around
   frame 22 (`--backbone-dtype f16` remains available, 1147/1550 codes).
   Weights are a local export for now (~3.5 GB via
   `scripts/extract-voicechat-tts.py`; the registry probe hides the engine
   without it). Perf: the decode loop is GPU-RESIDENT on WebGPU (2026-08-23)
   — per 80 ms frame, embed → fusion → 28-layer backbone records as one
   batched submit against GPU KV caches, each of the 5 active MoG unmask
   iterations is one submit + one overlapped readback set {xg, logits,
   mu_res}, and the mixture-argmax / PRVQ-argmin / noise decisions stay
   host-side f64. Timestamp-query on M5 Pro (dawn): 23.7 GPU-ms/frame
   backbone+MoG (was 92.8 before the thin-M split-K GEMV route), ~16 ms
   host decisions, codec ~97 GPU-ms per 4 s utterance (was 1247 — the
   k==stride ConvT stages route as GEMM+reshape). The parity gate holds ON
   THE GPU PATH TOO: 1550/1550 codes exact vs torch, waveform NRMSE 1.1e-6.
   Node wall-clock stays 0.12× because the dawn binding resolves every
   mapAsync on a ~100 ms event poll (5 dependent syncs/frame ≈ 500 ms);
   browsers resolve mapAsync in well under a millisecond, so the in-browser
   estimate is ~48 ms/frame ≈ 1.6× realtime (unverified in a real browser).
   WASM remains the bit-exact parity backend at 0.19× overall.
3. **Phase 3: LLM feasibility gate — RUN 2026-08-23: FAILED.** Calibrated
   sub-8-bit quantization does not recover quality on this checkpoint: best
   point within a 5.0-bit budget was **88.6% top-1 / KL 0.117** (GPTQ+AWQ
   int4 per-block-32 body + per-channel int8 heads, 4.963 eff. bits) against
   a ≥98% pass line; the crossover sits above 6.5 bits (int6 AWQ = 96.4%,
   ~7.1 GB). The damage is distributed hidden-state drift across all 27
   Mamba2 layers rather than a few sensitive ones, so mixed precision cannot
   buy it back under budget. A half-duplex browser VoiceChat is therefore
   shelved; reopening it requires a stronger method class (palettized LUT
   quantization, QuIP#/SpinQuant-style rotation, or QAT on fused
   embeddings), not more PTQ tuning. Full evidence: the calibrated-quant
   harness and results in the mobius voicechat trial.
4. **Full duplex** stays native (FluidAudio Swift/CoreML — encoder, RNNT, and
   the full 9B already converted and real-time viable there).

## License

OpenMDW-1.1's operative grant is permissive ("permission … to deal in the
Model Materials without restriction"), with a patent/copyright retaliation
termination clause and no restrictions on outputs. Redistribution of
converted weights with attribution is consistent with the grant; keep the
license text alongside any hosted artifacts.
