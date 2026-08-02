import { defineConfig } from "vite";

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
  // ORT + kokoro-js ship prebuilt .wasm; don't let Vite try to bundle them.
  optimizeDeps: {
    exclude: ["onnxruntime-web", "kokoro-js", "@ricky0123/vad-web"],
  },
  worker: { format: "es" },
  build: { target: "es2022" },
});
