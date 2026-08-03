// Register the WebGPU flag constants (GPUBufferUsage, GPUMapMode, …) on
// globalThis so kernel code written for the browser — where these are ambient —
// runs unchanged under dawn (@kmamal/gpu) in Node. Also creates a GPUDevice.
//   import { getDevice } from "./gpu-globals.mjs";
import * as G from "@kmamal/gpu";

const mod = G.default || G;
for (const k of ["GPUBufferUsage", "GPUMapMode", "GPUShaderStage", "GPUTextureUsage"]) {
  if (mod[k] && !globalThis[k]) globalThis[k] = mod[k];
}

export async function getDevice() {
  const gpu = mod.create ? mod.create([]) : mod;
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter (dawn)");
  const feats = adapter.features && adapter.features.has && adapter.features.has("shader-f16") ? ["shader-f16"] : [];
  return adapter.requestDevice({ requiredFeatures: feats });
}
