# Local WebGPU primitive checks

The pages in this directory compile and execute Stage 1 correctness kernels on
a real WebGPU adapter:

- source-order linear GEMM with and without bias;
- packed-BF16 weights with FP32 activations and accumulation;
- raw-FP16 weights, activations, source-order accumulation, and an intermediate
  rounding sentinel that differs from FP32 accumulation;
- RMSNorm with FP32 reduction statistics and learned scales; and
- Qwen split-half rotary embeddings across multiple heads and positions; and
- score-buffer-free grouped-query attention with padding and local-window masks;
  and
- one-level FP32 Haar DCW with complementary per-timestep band schedules and
  odd-length zero padding.

Additional standalone pages cover planner left-padding plus physical KV-cache
positions, shard-aware embeddings and transformer tensor transforms, the
fixed mixed-radix FSQ inverse with invalid-code detection, and scoped runtime
allocations with submit-before-drain uniform recycling. The Qwen3 block page
executes an independently-oracled miniature prefill, per-row-position cached
decode, and tied output projection in both profiles.
`text-encoder-correctness.html` executes a nonzero two-layer uncached
Qwen3Model through the production text-graph composer, one cooperative quantum
at a time, then checks the lyric embedding-only path. Its CPU oracle uses
literal Transformers rotary words and independently quantized weights.
`semantic-conditioner-primitives-correctness.html` additionally checks the
packed-BF16 detokenizer special-token expansion and the silence/context layout
used by the direct conditioner.
`dit-plumbing-correctness.html` executes both alternating attention modes and a
nontrivial complete DiT layer against an independent CPU oracle, including the
authenticated BF16-effective timestep contract. `vae-decoder-correctness.html`
executes the FP32 primitive family and a sharded toy decoder while deliberately
using two-output-element quanta; every quantum is queue-drained and followed by
the production one-millisecond idle interval.

`gemm-tiled-correctness.html` exercises the conservative ranged GEMM, checks
full finite/nonzero output coverage plus exact sentinels, and monitors a UI
heartbeat. `planner-model-correctness.html` covers the complete miniature
planner graph. `audio-output-correctness.html` commits a transactional OPFS WAV
inside a worker, transfers the typed Blob, terminates the worker, verifies RIFF
bytes in the page, and releases the stored output.

Run the development server from the repository root:

```sh
pnpm exec vite --host 127.0.0.1
```

Then open any `*.html` page under
`http://127.0.0.1:5173/test/browser/` in a WebGPU browser. Each page sets
`body[data-status="passed"]` only after every supported profile finishes; a
failure sets `body[data-status="failed"]` and prints its stack. Chrome on this
repository's M3 development machine executes both the reference-BF16 and
raw-FP16 numerical cases.

These are correctness checks, not benchmarks. The kernels intentionally read
the converter's source-order layout, and no timing result from this page may be
entered in the Stage 2 optimization ledger.
