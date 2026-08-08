// Backend factory: pick WebGPU when available, else the WASM/SIMD CPU backend.
// Both implement the same kernel interface (GpuContext / WasmContext), so the raw
// ORT-free engines run unchanged on either. No onnxruntime anywhere in this path.
import { GpuContext, requestGpuDevice } from "./compute.js";
import { createWasmContext } from "./wasm-context.js";

async function loadWasmBytes() {
  if (typeof window === "undefined" && typeof importScripts === "undefined") {
    // Node (headless tests / smoke scripts): read the file directly.
    const { readFileSync } = await import("node:fs");
    const { fileURLToPath } = await import("node:url");
    return readFileSync(fileURLToPath(new URL("./wasm-kernels.wasm", import.meta.url)));
  }
  // Browser / worker: standard asset-URL pattern — resolved by Vite, webpack 5,
  // Rollup, AND bundler-less browsers (SDK consumers bring their own bundler).
  return (await fetch(new URL("./wasm-kernels.wasm", import.meta.url))).arrayBuffer();
}

/**
 * Create a compute context. Prefers WebGPU; falls back to WASM+SIMD on the CPU.
 * @param {{backend?: "auto"|"webgpu"|"wasm", onBackend?: (b:string)=>void}} [opts]
 */
export async function createContext({ backend = "auto", onBackend } = {}) {
  if (backend !== "wasm" && typeof navigator !== "undefined" && navigator.gpu) {
    try {
      const ctx = new GpuContext(await requestGpuDevice());
      await ctx.probeSubgroups(); // enables the subgroup GEMM on verified-32-lane devices
      onBackend?.("webgpu");
      return ctx;
    } catch {
      /* fall through to WASM */
    }
  }
  const ctx = await createWasmContext(await loadWasmBytes());
  onBackend?.("wasm");
  return ctx;
}
