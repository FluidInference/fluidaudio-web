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
| TTS: Gemma3 backbone 28L×1152 + MoG head | 595M + 159M | ~6 ms/step proxy; CFG 0.2 → 2 evals/step | Feasible later; not useful without the LLM |
| Audio codec decoder (31-quantizer PRVQ, 12.5 Hz → 22.05 kHz) | 763 MB | unmeasured | Feasible later (conv stack) |

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
2. **Phase 2: codec + TTS decoders** — both are conv/attention stacks well
   within existing kernel capability; useful standalone as a high-quality
   TTS voice ("Aria" latents ship in the checkpoint) even before the LLM.
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
