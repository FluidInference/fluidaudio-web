# Third-party licenses & acknowledgements

## parakeet.wgsl (MIT)

https://github.com/narcotic-sh/parakeet.wgsl
Copyright (c) 2026 Narcotic Software — MIT License.

Several load-bearing kernel designs in this project are adapted from
parakeet.wgsl, and its author's write-ups guided much of our optimization
work. Specifically:

- **Tile-major direct-B subgroup GEMM** (`GEMM_TM_WGSL` in
  `src/gpu/compute.js`) — the workhorse encoder kernel and our single
  largest speed win (+58% encode): weights prepacked
  `[N/256][K/32][32][32-pack]` and streamed direct from global memory with
  zero workgroup-memory staging and zero barriers, A rows distributed via
  `subgroupBroadcast`, f16 accumulation. Adapted from their
  `src/webgpu/kernels/gemm.ts` geometry.
- **GPU-resident TDT decoder** (`src/engines/asr-parakeet/gpu-decoder.js`,
  opt-in) — the whole greedy TDT loop in one dispatch, following their
  part-2 design.
- The earlier **subgroup-broadcast GEMM** experiments and the batching /
  fused-attention studies were run against their published implementation
  as the reference point.

If this project interests you, check out theirs — it is an excellent,
fast, focused Parakeet implementation for the browser.

## ace-step-1.5.wgsl (MIT)

Upstream repo not yet public (author's live demo: https://acestep.narcotic.sh);
the complete vendored source is in this repository at `packages/acestep`.
Copyright (c) 2026 Narcotic Software (Hamza Qayyum) — MIT License.

The entire music-generation runtime (`packages/acestep`) is a vendored
import of ace-step-1.5.wgsl, and the `/music` page's worker backend
(`src/engines/musicgen-acestep/direct-only-backend.ts`, the generated
semantic-validation tensor metadata, and the download-progress accounting)
is adapted from its companion demo. The complete upstream license and the
ACE-Step / Qwen model-artifact terms are preserved in
`packages/acestep/LICENSE` and `packages/acestep/THIRD_PARTY_LICENSES`.
Model weights retain their upstream ACE-Step and Qwen licenses.

```
MIT License

Copyright (c) 2026 Narcotic Software

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DiCoSe.wgsl (MIT)

Upstream repo currently ships no LICENSE file (its package.json declares
MIT; the license text is pending from the author and will be added at
`packages/dicose/LICENSE` when supplied). The complete vendored source is
in this repository at `packages/dicose`.
Copyright (c) 2026 Hamza Qayyum (Narcotic Software) — MIT License.

The entire stem-separation runtime (`packages/dicose`) is a vendored
import of DiCoSe.wgsl by **Hamza Qayyum**: a raw WebGPU WGSL port of
DiCoSe (BS-RoFormer + one-step consistency-distilled refinement,
arXiv 2412.06965) with its own correctness-audited optimization ledger
(`packages/dicose/optimization/`). The `stem-dicose` engine and the
`/music` page's Split stems feature are thin wrappers over his worker
client. Model weights are converted from the
[karchkha/DiCoSe](https://huggingface.co/karchkha/DiCoSe) checkpoints
(MIT) via `packages/dicose/model/convert.py` and retain their upstream
license.

## parakeet.js / ysdede (MIT)

https://github.com/ysdede/parakeet.js — the NeMo log-mel preprocessor in
`src/engines/asr-nemotron/nemotron-mel.js` is adapted from its mel.js
(MIT), and several model vocabularies are fetched from ysdede's ONNX
export repos at runtime.

## text-processing-rs (Apache-2.0)

Vendored WASM build in `src/vendor/text-processing/` — NeMo-grammar text
normalization / inverse text normalization.

## Model weights

Model weights are downloaded at runtime from their respective Hugging Face
repositories and remain under their original licenses (NVIDIA Parakeet /
Nemotron / Sortformer: NVIDIA Open Model License or CC-BY-4.0 as published;
Kokoro: Apache-2.0; Silero VAD: MIT; Whisper: MIT).
