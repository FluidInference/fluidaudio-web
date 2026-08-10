// Build the publishable SDK into dist-sdk/:
//   1. tsc transpiles the .ts surface (engines/core/index) with declarations.
//   2. The already-ESM .js modules, their .d.ts sidecars, and the wasm/bin/json
//      assets are copied verbatim (same relative paths, so the cross-bundler
//      `new URL("./x.wasm", import.meta.url)` asset references keep working).
//   3. A publish manifest (package.json) with per-engine subpath exports is
//      written — the repo's own package.json stays the private site manifest.
// Output: `cd dist-sdk && npm pack` (or publish).
import { execSync } from "node:child_process";
import { cpSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync, copyFileSync, statSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";

const OUT = "dist-sdk";
// Single version source: the root package.json (private site manifest carries
// the SDK version; bump it there for releases).
const VERSION = JSON.parse(readFileSync("package.json", "utf8")).version;
rmSync(OUT, { recursive: true, force: true });

console.log("── tsc (SDK surface)");
execSync("npx tsc -p tsconfig.sdk.json", { stdio: "inherit" });

console.log("── copy runtime assets (.js/.d.ts/.wasm/.bin/.json)");
const KEEP = /\.(js|d\.ts|wasm|bin|json)$/;
const SKIP_DIRS = new Set(["__pycache__"]);
let copied = 0;
function walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) {
      if (!SKIP_DIRS.has(name)) walk(p);
      continue;
    }
    if (!KEEP.test(name) || name.endsWith(".config.js")) continue;
    const rel = p.slice("src/".length);
    const dst = join(OUT, rel);
    if (existsSync(dst) && (name.endsWith(".js") || name.endsWith(".d.ts"))) continue; // tsc output wins
    mkdirSync(dirname(dst), { recursive: true });
    copyFileSync(p, dst);
    copied++;
  }
}
walk("src");
console.log(`   ${copied} files`);

console.log("── manifest + docs");
const ENGINE_SUBPATHS = ["asr-parakeet", "asr-whisper", "asr-nemotron", "tts-kokoro", "vad-silero", "diarization-sortformer", "eou-parakeet"];
const exports_ = {
  ".": { types: "./index.d.ts", default: "./index.js" },
  "./registry": { types: "./engines/registry.d.ts", default: "./engines/registry.js" },
  "./textnorm": { types: "./core/textnorm.d.ts", default: "./core/textnorm.js" },
  "./vocab-rescorer": { types: "./engines/asr-parakeet/vocab-rescorer.d.ts", default: "./engines/asr-parakeet/vocab-rescorer.js" },
  "./captions": { types: "./core/captions.d.ts", default: "./core/captions.js" },
  "./mic": { types: "./core/mic.d.ts", default: "./core/mic.js" },
};
for (const e of ENGINE_SUBPATHS) {
  exports_[`./${e}`] = { types: `./engines/${e}/index.d.ts`, default: `./engines/${e}/index.js` };
}
writeFileSync(
  join(OUT, "package.json"),
  JSON.stringify(
    {
      name: "@fluidinference/fluidaudio-web",
      version: VERSION,
      description:
        "Local speech AI for the browser — ASR (Parakeet, Whisper, Nemotron), TTS (Kokoro), VAD (Silero), speaker diarization (Sortformer) on hand-written WebGPU + WASM-SIMD kernels. No onnxruntime; model weights stream from Hugging Face and cache locally.",
      license: "MIT",
      repository: { type: "git", url: "git+https://github.com/FluidInference/fluidaudio-web.git" },
      type: "module",
      sideEffects: false,
      // node10/"main" fallbacks for older resolvers (jest, TS<4.7, metro).
      main: "./index.js",
      types: "./index.d.ts",
      exports: { ...exports_, "./package.json": "./package.json" },
      keywords: ["webgpu", "wasm", "speech-to-text", "text-to-speech", "asr", "tts", "vad", "diarization", "whisper", "parakeet", "kokoro", "on-device"],
      dependencies: { "pinyin-pro": "^3.28.2" },
      engines: { node: ">=20" },
    },
    null,
    2,
  ),
);
cpSync("LICENSE", join(OUT, "LICENSE"));
writeFileSync(
  join(OUT, "README.md"),
  `# @fluidinference/fluidaudio-web

Local speech AI in the browser: ASR, TTS, VAD, and speaker diarization on hand-written WebGPU + WASM-SIMD kernels (no onnxruntime). Model weights stream from Hugging Face on first use and cache locally.

\`\`\`ts
import { ParakeetV3Engine } from "@fluidinference/fluidaudio-web/asr-parakeet";
import { decodeToMono16k } from "@fluidinference/fluidaudio-web";

const asr = new ParakeetV3Engine();
await asr.load((p) => console.log(p.file, p.fraction));
asr.setVocabulary(["NVIDIA", "Newrez"]);      // optional fuzzy correction
asr.setItn(true);                              // optional "twenty one" → "21"
const audio = await decodeToMono16k(fileArrayBuffer);
const { text } = await asr.transcribe(audio);
await asr.dispose();
\`\`\`

Engines (one subpath each, tree-shakeable): \`/asr-parakeet\`, \`/asr-whisper\`, \`/asr-nemotron\`, \`/tts-kokoro\` (\`new KokoroTtsEngine({ lang: "en" | "zh" })\`), \`/vad-silero\`, \`/diarization-sortformer\`, \`/eou-parakeet\`. To enumerate dynamically, use \`/registry\` and instantiate via each entry's \`make()\` — registry ids are NOT all valid subpaths (the two Kokoro ids share one subpath).

Requirements: a bundler that supports \`new URL(..., import.meta.url)\` assets, module workers, and JSON imports (Vite and webpack 5 out of the box; Rollup needs @rollup/plugin-json + an import-meta-assets plugin). WebGPU strongly recommended (WASM-SIMD fallback runs everywhere). Weights download from Hugging Face at runtime — no build-time model assets.

Demo/playground (same code): https://fluidaudio-web.hanweng9.workers.dev
`,
);
console.log("── done: dist-sdk/ ready (cd dist-sdk && npm pack)");
