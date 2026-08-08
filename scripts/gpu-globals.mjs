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
  const feats = [];
  for (const f of process.env.NO_F16 ? ["timestamp-query"] : ["shader-f16", "timestamp-query", "subgroups"]) {
    if (adapter.features && adapter.features.has && adapter.features.has(f)) feats.push(f);
  }
  // lift the 256MB default buffer caps to the adapter's real limits (the browser
  // path in compute.js requestGpuDevice already does this)
  const lim = adapter.limits || {};
  const requiredLimits = {};
  if (lim.maxBufferSize) requiredLimits.maxBufferSize = lim.maxBufferSize;
  if (lim.maxStorageBufferBindingSize) requiredLimits.maxStorageBufferBindingSize = lim.maxStorageBufferBindingSize;
  if (lim.maxComputeWorkgroupStorageSize) requiredLimits.maxComputeWorkgroupStorageSize = lim.maxComputeWorkgroupStorageSize;
  return adapter.requestDevice({ requiredFeatures: feats, requiredLimits });
}
