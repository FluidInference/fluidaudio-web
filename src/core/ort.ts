// onnxruntime-web session factory. Centralizes backend selection so every ASR
// engine gets the same WebGPU→WASM fallback policy and threading config.

import * as ort from "onnxruntime-web";
import type { Backend } from "./types";

// Injected by Vite from the installed onnxruntime-web version (see vite.config.ts).
declare const __ORT_VERSION__: string;

let configured = false;

/** One-time global ORT env setup (threads, SIMD, wasm asset paths). */
export function configureOrt(): void {
  if (configured) return;
  const threads = (self.crossOriginIsolated && navigator.hardwareConcurrency) || 1;
  ort.env.wasm.numThreads = Math.min(threads, 8);
  ort.env.wasm.simd = true;
  // Load ORT's wasm from jsdelivr at the EXACT installed version. The threaded+jsep
  // wasm is ~26 MB — over Cloudflare's 25 MB per-file limit — so we don't self-host
  // it (postbuild strips the local copies). Pinning to the installed version avoids
  // the JS/wasm mismatch that a stale CDN path caused ("e.getValue is not a
  // function"); jsdelivr sends ACAO:* + CORP:cross-origin, so it also satisfies COEP
  // require-corp on cross-origin-isolated hosts (Cloudflare Pages).
  if (typeof __ORT_VERSION__ === "string") {
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${__ORT_VERSION__}/dist/`;
  }
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
