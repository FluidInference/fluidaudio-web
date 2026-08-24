import { defineConfig } from "vite";

// `vite build` produces the library consumed by the host workspace
// (fluidaudio-web imports "dicose-wgsl" / "dicose-wgsl/worker" from dist/).
// `vite dev` still serves the upstream demo app (index.html + src/demo.ts);
// dev mode ignores build.lib, so the browser test harnesses are unaffected.
export default defineConfig({
  base: "./",
  publicDir: "public",
  worker: {
    format: "es",
  },
  build: {
    target: "esnext",
    outDir: "dist",
    emptyOutDir: true,
    minify: "oxc",
    sourcemap: false,
    copyPublicDir: false,
    lib: {
      entry: {
        index: "src/index.ts",
        worker: "src/worker.ts",
      },
      formats: ["es"],
    },
    rollupOptions: {
      output: {
        preserveModules: false,
      },
    },
  },
  server: {
    host: "127.0.0.1",
  },
});
