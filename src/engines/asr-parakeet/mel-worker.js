// Mel worker: ParakeetMel off the main thread. Windows are independent, so a
// small pool computes them in parallel while the main thread drives the GPU —
// mel was ~1.8s of UNHIDDEN wall on the 1-hour browser run (the pipeline
// "overlaps" mel with GPU work, but main-thread JS still delays submits).
// Same dual-runtime + reply-every-message contract as decoder-worker.js.
import { ParakeetMel } from "./parakeet-mel.js";

let mel = null;

function handle(msg, post, transfer) {
  try {
    if (msg.type === "init") {
      mel = new ParakeetMel(msg.nMels ?? 128);
      post({ type: "ready" });
      return;
    }
    const { features, length } = mel.process(new Float32Array(msg.samples));
    post({ type: "res", id: msg.id, features: features.buffer, length }, transfer ? [features.buffer] : undefined);
  } catch (e) {
    post({ type: "err", id: msg.id, error: String(e && e.stack ? e.stack : e) });
  }
}

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = (e) => handle(e.data, (m, t) => self.postMessage(m, t ?? []), true);
} else {
  import("node:worker_threads").then(
    ({ parentPort }) => {
      parentPort.on("message", (d) => handle(d, (m, t) => parentPort.postMessage(m, t ?? []), true));
    },
    (e) => {
      console.error("[mel-worker] worker_threads unavailable:", e);
      throw e;
    },
  );
}
