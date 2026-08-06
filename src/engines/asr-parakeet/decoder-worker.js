// Decoder worker: runs the WASM-SIMD TDT decoder off the main thread so
// windows decode in parallel (they are independent; the seam stitch happens
// on the main thread, in window order). Dual-runtime: Web Worker (browser)
// or worker_threads (node gates) — same file, same code path.
// Every message gets a reply — errors post {type:"err"} so the pool can
// REJECT the matching promise instead of hanging transcribe() forever.
import { loadWasmDecoder, wasmDecodeProj } from "./raw-decoder-wasm.js";

let dec = null;

async function handle(msg, post) {
  try {
    if (msg.type === "init") {
      dec = await loadWasmDecoder(new Uint8Array(msg.wasmBytes), new Float32Array(msg.decBuf), msg.man);
      post({ type: "ready" });
      return;
    }
    const { ids, idFrames } = wasmDecodeProj(dec, new Float32Array(msg.frames), msg.Tenc);
    post({ type: "res", id: msg.id, ids, idFrames });
  } catch (e) {
    post({ type: "err", id: msg.id, error: String(e && e.stack ? e.stack : e) });
  }
}

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = (e) => handle(e.data, (m) => self.postMessage(m));
} else {
  // node (gates). No top-level await: consumer bundlers compile SDK workers to
  // iife by default, which rejects TLA. worker_threads queues messages until a
  // listener attaches, so the async import loses nothing.
  import("node:worker_threads").then(
    ({ parentPort }) => {
      parentPort.on("message", (d) => handle(d, (m) => parentPort.postMessage(m)));
    },
    (e) => {
      // Deterministic failure instead of a silent no-op worker (e.g. a bundler
      // stubbing node: builtins): surfaces as a worker 'error' → init rejects.
      console.error("[decoder-worker] worker_threads unavailable:", e);
      throw e;
    },
  );
}
