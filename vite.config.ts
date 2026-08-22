import { defineConfig } from "vite";
import { resolve } from "node:path";

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
  worker: { format: "es" },
  build: {
    target: "es2022",
    // Multi-page: the interactive app + the verify page (run all engines on one file).
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        verify: resolve(__dirname, "verify.html"),
        live: resolve(__dirname, "live.html"),
        bench: resolve(__dirname, "bench.html"),
        music: resolve(__dirname, "music.html"),
      },
    },
  },
});
