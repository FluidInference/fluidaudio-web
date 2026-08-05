// WebGPU availability check. Standalone (no onnxruntime) — the raw engines pick
// their backend via src/gpu/context.js; this is only for UI/bench display.
export function webgpuAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}
