# DiCoSe WebGPU

A browser stem separator by Hamza Qayyum, using the
[DiCoSe BS-RoFormer and one-step refinement model](https://arxiv.org/abs/2412.06965)
with [official weights](https://huggingface.co/karchkha/DiCoSe).

Inference runs in a dedicated WebGPU worker. Input audio and generated stems
stay in the browser. The runtime returns drums, bass, other, vocals, and a
derived instrumental, restored to the input's sample rate and exact frame count.

## Requirements and modes

Requires HTTPS or localhost, WebGPU with `shader-f16`, fixed 32-wide subgroups,
1 GiB GPU buffers/storage bindings, and 25,344 bytes of workgroup storage.
Use `checkSupport()` to check the device before loading the **623 MB** model.

| Mode | API value       | Behavior                                                                   |
| ---- | --------------- | -------------------------------------------------------------------------- |
| Full | `refined`       | Deterministic separation followed by one-step refinement; package default. |
| Fast | `deterministic` | Skips refinement; default in FluidAudio Web.                               |

Fast trades refinement quality for speed. For long files, Full uses 50% chunk
overlap and Fast uses 10%; both process bounded 11-second model items with
normalized overlap-add. The instrumental is the restored input mixture minus
vocals, not the sum of the other stems.

Measurements and quality evidence are in the
[correctness audit](optimization/CORRECTNESS_AUDIT.md) and
[optimization ledger](optimization/LEDGER.md). Performance depends on the mode,
input length, and hardware.

## Run locally

From the FluidAudio Web repository root:

```bash
npm ci
npm run dicose:check
npm run dicose:test
npm run dicose:build
```

For the integrated site, follow the [root quick start](../../README.md#run-locally).
For the standalone package demo, run these commands from `packages/dicose/`:

```bash
npm run model:prepare
npm run dev
```

Model preparation requires `uv`. It uses the locked Python 3.13 environment,
downloads and verifies about **4.66 GB** of source checkpoints, and writes the
browser package to `public/model/`. Source downloads are cached in
`model/cache/`; these files stay out of Git. See [model preparation](model/README.md).

Open `http://127.0.0.1:5173/`, select a local WAV and Full or Fast, then run
separation. Each stem has playback and WAV download controls.

## API

The public API is exported from [src/index.ts](src/index.ts). Within this workspace:

```ts
import { DiCoSeWorkerClient } from "dicose-wgsl";

const client = new DiCoSeWorkerClient();
try {
  const result = await client.separateAudio(source, {
    outputMode: "deterministic",
  });
  // Four model outputs: result.stems; derived output: result.instrumental.
} finally {
  await client.dispose();
}
```

`source` is a `Blob` or `ArrayBuffer`. Set `manifestUrl` in the constructor to
use another model host. Consumers bundling the prebuilt library may need
`createWorker` to provide a worker that imports `dicose-wgsl/worker`; see the
[FluidAudio integration](../../src/engines/stem-dicose/index.ts).

## Browser checks

From `packages/dicose/`, with model files prepared:

```bash
npm run test:webgpu
npm run test:browser
npm run test:reference-quality
npm run test:refined-reference-quality
npm run test:output-mode-quality
DICOSE_BENCHMARK_OUTPUT_MODE=deterministic npm run benchmark:browser
```

The scripts start Vite and headless Chrome with a temporary profile, then clean
them up. Reference checks compare deterministic and refined outputs and
intermediate tensors against the official implementation. The fixed default
noise seed makes otherwise identical runs reproducible.

For page automation, `?autorun=1` publishes the report to
`window.__DICOSE_BROWSER__.report` and `#result`. `?mode=benchmark` reuses one
worker across warmup and measured runs.
