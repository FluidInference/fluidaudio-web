import { defineConfig } from "vite";
import { resolve } from "node:path";

// onnxruntime-web's multi-threaded WASM backend needs SharedArrayBuffer, which
// browsers only expose under cross-origin isolation. These headers turn it on
// for the dev server and preview; production hosts must send them too.
const coopCoep = (_req: any, res: any, next: any) => {
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  next();
};
const crossOriginIsolation = {
  name: "cross-origin-isolation",
  // Block bodies (return void). An arrow returning `middlewares.use(...)` hands
  // Vite the connect app as a post-hook → "Cannot read properties of undefined
  // (reading 'url')".
  configureServer(s: any) {
    s.middlewares.use(coopCoep);
  },
  configurePreviewServer(s: any) {
    s.middlewares.use(coopCoep);
  },
};

export default defineConfig({
  // Base path per deploy target:
  //  - Cloudflare Pages (CF_PAGES=1): root of the *.pages.dev domain → "/", and its
  //    public/_headers gives COOP/COEP → threaded WASM (SharedArrayBuffer).
  //  - GitHub Pages (GITHUB_ACTIONS, not CF): project site under /<repo>/. No custom
  //    headers there → WebGPU (no COI needed) + single-thread WASM fallback.
  //  - local dev: "/".
  base: process.env.CF_PAGES ? "/" : process.env.GITHUB_ACTIONS ? "/fluidaudio-web/" : "/",
  plugins: [crossOriginIsolation],
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
