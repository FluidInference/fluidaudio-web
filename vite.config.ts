import { defineConfig } from "vite";
import { resolve } from "node:path";
import { readFileSync } from "node:fs";

// onnxruntime-web's threaded+jsep wasm is ~26 MB — over Cloudflare's 25 MB
// per-file asset limit. So ORT loads its wasm from the jsdelivr CDN (see
// core/ort.ts wasmPaths) at the EXACT installed version (no JS/wasm mismatch),
// and postbuild strips the local copies from dist/. Inject the version here.
const ORT_VERSION = JSON.parse(
  readFileSync(resolve(__dirname, "node_modules/onnxruntime-web/package.json"), "utf8"),
).version;

// NOTE: no cross-origin isolation (COOP/COEP). It gates SharedArrayBuffer →
// threaded WASM, but COEP also breaks the cross-origin Hugging Face / jsdelivr
// downloads (both require-corp AND credentialless failed in practice on
// Cloudflare, "No Access-Control-Allow-Origin"). Every RTFx number we ship was
// measured single-threaded anyway (WebGPU needs no isolation), so isolation was
// pure downside here. Left off → model fetches work on every host.

export default defineConfig({
  // Base path per deploy target: Cloudflare Pages / local dev serve from root;
  // GitHub Pages project site lives under /<repo>/.
  base: process.env.CF_PAGES ? "/" : process.env.GITHUB_ACTIONS ? "/fluidaudio-web/" : "/",
  define: { __ORT_VERSION__: JSON.stringify(ORT_VERSION) },
  optimizeDeps: {
    // onnxruntime-web must NOT be pre-bundled: Vite rewrites its dynamic import
    // of `ort-wasm-*.jsep.mjs` into `.vite/deps/…` which 404s. Excluded, ORT
    // loads its own co-located .mjs/.wasm (matching version) correctly.
    exclude: ["onnxruntime-web"],
  },
  worker: { format: "es" },
  build: {
    target: "es2022",
    // Multi-page: the interactive app + the auto-benchmark.
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        bench: resolve(__dirname, "bench.html"),
      },
    },
  },
});
