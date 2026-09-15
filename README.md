# FluidAudio Web

Speech recognition, text to speech, music generation, and audio analysis in
your browser. Inference runs locally using custom WebGPU and WASM kernels;
audio is not uploaded. Model weights download on first use.

This is the browser sibling of the Swift/CoreML
[FluidAudio](https://github.com/FluidInference/FluidAudio) framework.

## Try it

- [Speech to text](https://fluidinference.github.io/fluidaudio-web/)
- [Text to speech](https://fluidinference.github.io/fluidaudio-web/tts.html)
- [Music generation and stem splitting](https://fluidinference.github.io/fluidaudio-web/music.html)
- [Audio analysis](https://fluidinference.github.io/fluidaudio-web/analyze.html)
- [Live captions](https://fluidinference.github.io/fluidaudio-web/live.html)

[Cloudflare mirror](https://fluidaudio-web.hanweng9.workers.dev).

Use a desktop browser with WebGPU for best performance. Speech engines also
support WASM-SIMD. Music generation requires WebGPU with `shader-f16`; stem
splitting additionally requires fixed 32-wide subgroups. These two engines
have no WASM fallback.

Downloads, cached model loading, and shader compilation all affect startup
time. Caching depends on browser storage availability and limits; some large
speech-model files may download again. Performance varies by device, browser,
model, and input. See [benchmarks](docs/BENCHMARKS.md) for measurements.

## Music generation

Open the music page, enter a prompt, and select **Generate song**. Leave lyrics
empty for an instrumental. When generation finishes, use the audio player or
download the stereo 48 kHz WAV. Songs can be 10 seconds to 4 minutes long.

The first generation downloads **5.75 GB** from the FluidInference Hugging Face
mirror and caches it in browser storage (OPFS). Later generations still need
to read and prepare model data. Settings shows cache usage and lets you delete
the downloaded model.

The site uses ACE-Step 1.5 Turbo in direct mode. The optional planner is
available in the underlying runtime but disabled on the public music page.
See the [ACE-Step README](packages/acestep/README.md) for implementation and
validation details.

Choose **Split stems** on a finished song to get drums, bass, other, vocals,
and a derived instrumental, each with playback and download controls. DiCoSe
downloads another **623 MB** on first use. The site defaults to its faster
deterministic mode, which skips refinement. See the
[DiCoSe README](packages/dicose/README.md) for mode differences and requirements.

## Run locally

Requires Node.js `^20.19.0` or `>=22.12.0` and npm. From the repository root:

```bash
npm ci
npm run acestep:build
npm run dicose:build
npm run dev
```

Open `http://localhost:5173/`. Build the workspace libraries before starting
the site: its imports resolve to their generated `dist/` files.

```bash
npm run build          # Build libraries, type-check, and bundle the site
npm run test:unit      # Shared UI progress tests
npm run acestep:test   # ACE-Step tests
npm run dicose:test    # DiCoSe tests
npm run format:check   # Formatting check
```

Model weights are downloaded at runtime and are excluded from the site build.
Set `VITE_ACE_MODEL_ORIGIN` to use another ACE model host or a local package
directory. The default host and package identities are in
[config.ts](src/engines/musicgen-acestep/config.ts).

## SDK

```bash
npm install @fluidinference/fluidaudio-web
```

```ts
import { ParakeetV3Engine } from "@fluidinference/fluidaudio-web/asr-parakeet";
import { decodeToMono16k } from "@fluidinference/fluidaudio-web";

const asr = new ParakeetV3Engine();
try {
  await asr.load((p) => console.log(p.file, p.fraction));
  const audio = await decodeToMono16k(fileArrayBuffer);
  const { text } = await asr.transcribe(audio);
  console.log(text);
} finally {
  await asr.dispose();
}
```

Engine subpaths: `/asr-parakeet`, `/asr-whisper`, `/asr-nemotron`,
`/tts-kokoro` (English or Chinese), `/vad-silero`, `/diarization-sortformer`,
and `/eou-parakeet`. The published package can lag behind this repository;
check its version before using newer APIs.

Use a bundler that supports module workers and `new URL(..., import.meta.url)`
assets, such as Vite or webpack 5. The source SDK supports streaming, caption
exports, and optional Parakeet vocabulary correction and inverse text
normalization. See [streaming](docs/STREAMING.md) and [end-of-utterance detection](docs/EOU.md).

To prepare a release, bump the root `package.json` version and run
`npm run build`, `npm run sdk:test`, and `npm run sdk:pack`. The SDK test validates
the tarball in a clean consumer. The SDK registry includes only packaged engines;
VoiceChat, music generation, and DiCoSe remain site-only.

## Development references

- [Engine catalog](src/engines/registry.ts)
- [Architecture](docs/ARCHITECTURE.md) and [adding a model](docs/PORTING.md)
- [WebGPU implementation](docs/RAW_WEBGPU.md) and [ONNX Runtime removal](docs/ORT_REMOVAL.md)
- [ACE-Step development rules](packages/acestep/AGENTS.md) and [optimization ledger](packages/acestep/optimization/LEDGER.md)
- [DiCoSe correctness audit](packages/dicose/optimization/CORRECTNESS_AUDIT.md)

GitHub Pages deploys `main` only after all [CI jobs](.github/workflows/ci.yml)
pass. Cloudflare Workers uses a separate deployment integration; see the
[Worker configuration](wrangler.jsonc).

## Credits and licenses

Code is MIT licensed; model weights retain their upstream licenses. See
[third-party licenses](THIRD-PARTY-LICENSES.md).

Hamza Qayyum ([Narcotic Software](https://narcotic.sh)) built the ACE-Step
browser port and DiCoSe WebGPU runtime, vendored under `packages/`. FluidInference
integrated them and continues their development. His original ACE-Step demo is
at [acestep.narcotic.sh](https://acestep.narcotic.sh).

The Parakeet encoder GEMM layout and GPU decoder design are adapted from
[parakeet.wgsl](https://github.com/narcotic-sh/parakeet.wgsl). Text normalization
uses the vendored [text-processing-rs](https://github.com/FluidInference/text-processing-rs)
WASM module.
