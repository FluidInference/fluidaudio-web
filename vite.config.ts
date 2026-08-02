import { defineConfig } from "vite";
import { resolve } from "node:path";

// onnxruntime-web's multi-threaded WASM backend needs SharedArrayBuffer, which
// browsers only expose under cross-origin isolation. These headers turn it on
// for the dev server and preview; production hosts must send them too.
const crossOriginIsolation = {
  name: "cross-origin-isolation",
  configureServer(server: any) {
    server.middlewares.use((_req: any, res: any, next: any) => {
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
      next();
    });
  },
};

export default defineConfig({
  plugins: [crossOriginIsolation],
  // Let Vite pre-bundle CJS deps so named ESM imports interop correctly
  // (excluding them served raw CJS and broke `import { NonRealTimeVAD }`).
  optimizeDeps: {
    include: ["@ricky0123/vad-web", "onnxruntime-web"],
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
