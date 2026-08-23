import { defineConfig, type Plugin } from "vite";
import { createReadStream, existsSync, statSync } from "node:fs";
import { resolve, normalize, join } from "node:path";

// Dev-only: serve gitignored local model exports (models-local/) at /models —
// deliberately OUTSIDE publicDir so `vite build` never copies multi-GB weights
// into dist/ (Cloudflare static assets cap files at 25 MiB). Production serves
// these from a real model host (HF/R2) via each engine's baseUrl.
function serveLocalModels(): Plugin {
  const root = resolve(__dirname, "models-local");
  return {
    name: "serve-local-models",
    configureServer(server) {
      server.middlewares.use("/models", (req, res, next) => {
        const rel = normalize(decodeURIComponent((req.url || "/").split("?")[0])).replace(/^([/\\.])+/, "");
        const file = join(root, rel);
        if (!file.startsWith(root) || !existsSync(file) || !statSync(file).isFile()) return next();
        res.setHeader("content-type", file.endsWith(".json") ? "application/json" : "application/octet-stream");
        res.setHeader("content-length", String(statSync(file).size));
        createReadStream(file).pipe(res);
      });
    },
  };
}

// Fully ORT-free: the engines run on raw WebGPU / WASM-SIMD (src/gpu/*). No
// onnxruntime-web / transformers.js / kokoro-js. The only bundled wasm is our own
// tiny kernel lib (src/gpu/wasm-kernels.wasm) + the parakeet decoder + silero VAD.
//
// NOTE: no cross-origin isolation (COOP/COEP). It gates SharedArrayBuffer, but COEP
// also breaks cross-origin Hugging Face model fetches on Cloudflare/Pages. Every
// RTFx number is measured single-threaded anyway (WebGPU needs no isolation).

export default defineConfig({
  // Base path per deploy target: Cloudflare Pages / local dev serve from root;
  // GitHub Pages project site lives under /<repo>/.
  base: process.env.CF_PAGES ? "/" : process.env.GITHUB_ACTIONS ? "/fluidaudio-web/" : "/",
  plugins: [serveLocalModels()],
  worker: { format: "es" },
  build: {
    target: "es2022",
    // Multi-page: STT + TTS + analysis + live captions + music generation.
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        tts: resolve(__dirname, "tts.html"),
        analyze: resolve(__dirname, "analyze.html"),
        live: resolve(__dirname, "live.html"),
        music: resolve(__dirname, "music.html"),
      },
    },
  },
});
