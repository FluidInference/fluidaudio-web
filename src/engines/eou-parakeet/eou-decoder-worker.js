// EOU decode worker: the RNNT stream state (LSTM h/c + predProj) lives in the
// wasm instance INSIDE this worker — decode is stateful and sequential, so
// there is exactly ONE worker and the engine chains calls in chunk order. The
// win is overlap (decode chunk k while the GPU encodes k+1) and escaping the
// Chrome main-thread wasm penalty (measured 16.6s worker-less vs 6.4s node on
// the 1-hour bench while the main thread also drives WebGPU).
// Dual-runtime like asr-parakeet/decoder-worker.js; every message is replied
// to — errors post {type:"err"} so the caller rejects instead of hanging.
import { loadEouWasmDecoder, eouWasmDecodeCont, eouWasmReset } from "../asr-parakeet/raw-decoder-eou.js";

let wdec = null;
const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

async function handle(msg, post) {
  try {
    if (msg.type === "init") {
      wdec = await loadEouWasmDecoder(new Uint8Array(msg.wasmBytes), new Float32Array(msg.decBuf), msg.man);
      post({ type: "ready" });
      return;
    }
    if (msg.type === "reset") {
      eouWasmReset(wdec);
      post({ type: "res", id: msg.id });
      return;
    }
    const t0 = now();
    const r = eouWasmDecodeCont(wdec, new Float32Array(msg.frames), msg.n, msg.subT, msg.maxSymbols ?? 10);
    post({ type: "res", id: msg.id, ids: r.ids, idFrames: r.idFrames, events: r.events, ms: now() - t0 });
  } catch (e) {
    post({ type: "err", id: msg.id, error: String(e && e.stack ? e.stack : e) });
  }
}

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = (e) => handle(e.data, (m) => self.postMessage(m));
} else {
  import("node:worker_threads").then(
    ({ parentPort }) => {
      parentPort.on("message", (d) => handle(d, (m) => parentPort.postMessage(m)));
    },
    (e) => {
      console.error("[eou-decoder-worker] worker_threads unavailable:", e);
      throw e;
    },
  );
}
