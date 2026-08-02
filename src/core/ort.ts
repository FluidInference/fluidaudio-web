// onnxruntime-web session factory. Centralizes backend selection so every ASR
// engine gets the same WebGPU→WASM fallback policy and threading config.

import * as ort from "onnxruntime-web";
import type { Backend } from "./types";

let configured = false;

/** One-time global ORT env setup (threads, SIMD, wasm asset paths). */
export function configureOrt(): void {
  if (configured) return;
  const threads = (self.crossOriginIsolated && navigator.hardwareConcurrency) || 1;
  ort.env.wasm.numThreads = Math.min(threads, 8);
  ort.env.wasm.simd = true;
  // Where ORT loads its .wasm/.mjs from. Pinned to the CDN matching the
  // installed version for zero-config; for offline/self-hosted, copy
  // `node_modules/onnxruntime-web/dist/*.{wasm,mjs}` into `public/ort/` and set
  // this to "/ort/".
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  configured = true;
}

export function webgpuAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

/**
 * Create an InferenceSession from raw model bytes, trying the requested backend
 * and falling back to WASM. `preferred: "webgpu"` is best for heavy, static-shape
 * graphs (encoders); dynamic-shape graphs (RNNT/TDT decoders) should ask for
 * "wasm" directly — the WebGPU EP falls back per-op and can be far slower.
 */
export async function createSession(
  model: Uint8Array,
  preferred: Backend
): Promise<ort.InferenceSession> {
  configureOrt();
  const eps: string[] =
    preferred === "webgpu" && webgpuAvailable() ? ["webgpu", "wasm"] : ["wasm"];
  try {
    return await ort.InferenceSession.create(model, {
      executionProviders: eps,
      graphOptimizationLevel: "all",
    });
  } catch (err) {
    if (eps[0] !== "wasm") {
      console.warn(`[ort] ${preferred} session failed, falling back to wasm`, err);
      return ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
    }
    throw err;
  }
}

export { ort };
