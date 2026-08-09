# True streaming encode — implementation spec (in progress)

Goal: `ParakeetEouEngine.push(chunk) / reset()` (the `StreamingAsrEngine`
interface in core/types.ts) with **genuinely incremental** encoding — carry
conformer caches chunk-to-chunk instead of the mic demo's rolling re-decode.
EOU first (fully causal config), Nemotron second (adds right-context 3).

## Why this is provably correct

The offline path already *emulates* streaming: the chunked-causal attention
mask (EOU: chunk 2, left 70) means frame i only ever attends
[chunkStart-70 .. chunkStart+1]. True streaming with activation caches
computes THE SAME function — so the gate is exact:

> **Gate:** encode sample.wav offline vs. chunk-by-chunk with caches →
> subsampled output frames match to ~1e-5; decoded tokens identical.
> (`scripts/streaming-encode-check.mjs`)

Chunk boundaries must align to the offline chunk grid: push multiples of
`attChunk` (EOU: 2) subsampled frames = 160 ms granularity.

## State (per stream, all GPU-resident except melTail)

- `melTail: Float32Array` — raw mel context for the causal subsampling stack.
  Left receptive field of the 3 stride-2 convs (k=3 causal): conv1 needs 2,
  conv2 needs 2·2, conv3 needs 4·2 input frames → tail = 14 mel frames;
  feed [tail ‖ new mel], drop the warmup outputs, keep new subsampled frames.
- Per layer L (×17 for EOU):
  - `kCache[L], vCache[L]: [≤attLeft=70, D]` — cache K/V DIRECTLY (computed
    from xln) rather than layer inputs: 70·1024·4·2·17 ≈ 9.7 MB. Append new
    K/V rows each chunk, keep last 70.
  - `convCache[L]: [D, dwK-1=8]` — tail of the GLU OUTPUT (the depthwise
    conv's input, post-pw1), NOT the layer input.
- `subT: number` — total subsampled frames emitted (chunk-grid alignment).

## Per-chunk flow (one batch submit)

1. CPU: mel of [melTail ‖ chunk], update melTail.
2. Subsample (causal pads only at stream start — `subPad.l` applies to the
   first chunk; later chunks are continuation, no left pad) → x_new [n, D],
   n a multiple of attChunk.
3. Per layer:
   - FF1 (stateless).
   - xln = ln(x); K_new/V_new = xln@Wk/Wv; K = [kCache ‖ K_new] (≤70+n rows).
   - **Attention (rectangular, n×(Lc+n))**: the offline bd-term uses the
     square relShift trick — for streaming write a small dedicated kernel:
     `score[i,j] = (q_i+pbu)·k_j + (q_i+pbv)·posProj[rel(i,j)]` where
     rel(i,j) = (Lc + i) − j ∈ [−n+1 .. Lc], posProj rows precomputed per
     layer (cache like `_posProj`). n ≤ 8 → O(n·78·HD) per head, trivial.
     Mask: chunked-causal evaluated with ABSOLUTE positions
     (subT + i vs subT − Lc + j) so the grid matches offline.
   - Softmax over ≤78 cols; out = probs@V; residual add.
   - Conv module: pw1+GLU → glu_new [D, n]; depthwise over
     [convCache ‖ glu_new] (k=9 causal ⇒ emits exactly n new frames);
     update convCache = last 8 cols; pw2 + residual.
   - FF2, ln_out.
   - Update kCache/vCache (keep last 70 rows).
4. Joint projection of x_new → decoder.

## Decoder state carry

rust decoder resets H/C in `decode_proj()`. Add exported
`decode_cont(frames, tenc, out_ids, out_frames) -> i32` = same loop WITHOUT
the reset and WITHOUT re-predicting from BLANK (keep `LAST_TOK` static across
calls; add `decode_reset()` for stream start). Rebuild wasm32
(`cargo build --target wasm32-unknown-unknown --release -p parakeet-decoder`
+ copy to src/engines/asr-parakeet/parakeet-decoder.wasm). The GPU decoder
alternative: persist h/c buffers per stream instead of zero-init.

## Engine + mic wiring

- `ParakeetEouEngine.push(chunk16k)`: buffer to ≥160 ms, run chunk flow,
  `decode_cont`, return accumulated text (+ EOU/EOB events).
- `reset()`: zero caches, melTail, decoder state, subT.
- main.ts mic loop: `if (typeof engine.push === "function")` → stream chunks
  as they arrive (replace the 1.5 s rolling re-decode for this engine);
  batch engines keep the rolling path.

## Status

- [x] Design + gate defined (this doc)
- [x] `createEncodeStream` / `encodeStreamPush` / `encodeStreamFlush`
      (streaming-encoder.js — separate module, raw-encoder.js untouched)
- [x] Rectangular rel-pos shift + absolute-grid mask (`relShiftStream`,
      compute.js + wasm-context.js twin; bmmQK/bmmPV were already rectangular)
- [x] Causal-continuation subsampling (7-mel-frame FIFO overlap; padTop first
      chunk only, padBottom at flush only)
- [x] Decoder state carry — EOU's decoder is plain JS, so `createEouStream` +
      `eouDecodeCont` (raw-decoder-eou.js); no rust rebuild needed. (Parakeet
      TDT streaming would still need `decode_cont` in rust.)
- [x] streaming-encode-check.mjs parity gate — **bit-exact** (frames maxΔ 0.0,
      tokens identical) at 160 ms and coarse cadence; StreamingMel exact
- [x] Engine push()/finish()/reset() + mic integration (main.ts streams for
      real; batch engines keep the rolling-tail fallback)
- [x] Nemotron variant (right-context 3). DESIGN FINDING: right context
      CASCADES through layers (each layer's exact value needs ~chunk more
      future), so bit-exact streaming is impossible with bounded lookahead.
      Measured decay (truncated-offline floor, int8): B=2 chunks maxΔ 1.4e-2,
      B=3 → 2.1e-3, B=4 → 8.2e-4 ≈ noise floor. Shipped as NeMo-style
      provisional tail: compute [C·k + B] frames per pass, emit C·k, cache
      only emitted K/V, recompute the tail next pass (lookaheadChunks=4 ⇒
      1.28s lookahead, ~1.3s added latency). End-to-end gate: frames
      maxΔ 2.85e-3 incl. cache compounding, TOKENS identical to offline.
