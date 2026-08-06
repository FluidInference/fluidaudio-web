// Decoder worker: runs the WASM-SIMD TDT decoder off the main thread so
// windows decode in parallel (they are independent; the seam stitch happens
// on the main thread, in window order). Dual-runtime: Web Worker (browser)
// or worker_threads (node gates) — same file, same code path.
import { loadWasmDecoder, wasmDecodeProj } from "./raw-decoder-wasm.js";

let dec = null;

async function handle(msg, post) {
  if (msg.type === "init") {
    dec = await loadWasmDecoder(new Uint8Array(msg.wasmBytes), new Float32Array(msg.decBuf), msg.man);
    post({ type: "ready" });
    return;
  }
  const { ids, idFrames } = wasmDecodeProj(dec, new Float32Array(msg.frames), msg.Tenc);
  post({ type: "res", id: msg.id, ids, idFrames });
}

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = (e) => handle(e.data, (m) => self.postMessage(m));
} else {
  const { parentPort } = await import("node:worker_threads");
  parentPort.on("message", (d) => handle(d, (m) => parentPort.postMessage(m)));
}
