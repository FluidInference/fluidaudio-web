/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />

// Dedicated inference worker for ACE-Step music generation. The runtime is
// installed with the upstream demo's direct-only boundary: planner/semantic
// material is structurally excluded, matching the direct-only R2 manifest.

import { installAceWorkerRuntime } from "ace-step-1.5.wgsl";

// Workers don't inherit the page's <meta name="referrer" content="no-referrer">,
// and Hugging Face hotlink-blocks requests carrying a *.workers.dev Referer
// (served without CORS headers → surfaces as "Failed to fetch"). Force
// no-referrer on every runtime fetch, same as src/core/modelCache.ts does for
// the page-side engines.
const workerFetch = self.fetch.bind(self);
self.fetch = ((input: RequestInfo | URL, init?: RequestInit) => workerFetch(input, { ...init, referrerPolicy: "no-referrer" })) as typeof fetch;

import { createAceDirectOnlyWebGpuPipelineBackend } from "./direct-only-backend.js";

// TEST-ONLY: the page appends `?testMaxGpuBufferBytes=N` to the worker name
// (driven by the page query parameter `aceTestMaxGpuBufferBytes`) to emulate a
// capped WebGPU adapter — e.g. N=1073741824 reproduces every iOS adapter's
// one-GiB maxBufferSize/maxStorageBufferBindingSize. The facade only ever
// LOWERS limits: adapter snapshots are clamped and requestDevice rejects any
// requiredLimits above the cap, exactly like a real capped adapter would.
function testOnlyGpuBufferCapBytes(): number | undefined {
  const query = self.name.split("?")[1];
  if (query === undefined) return undefined;
  const raw = new URLSearchParams(query).get("testMaxGpuBufferBytes");
  if (raw === null) return undefined;
  const value = Number(raw);
  return Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

function testOnlyClampedGpu(capBytes: number): GPU | undefined {
  const gpu = navigator.gpu;
  if (gpu === undefined) return undefined;
  const clampedLimits = (limits: GPUSupportedLimits): Record<string, number> => {
    const snapshot: Record<string, number> = {};
    for (
      let proto = Object.getPrototypeOf(limits) as object | null;
      proto !== null && proto !== Object.prototype;
      proto = Object.getPrototypeOf(proto) as object | null
    ) {
      for (const propertyName of Object.getOwnPropertyNames(proto)) {
        const value = (limits as unknown as Record<string, unknown>)[propertyName];
        if (typeof value === "number") snapshot[propertyName] = value;
      }
    }
    for (const cappedName of ["maxBufferSize", "maxStorageBufferBindingSize"]) {
      const value = snapshot[cappedName];
      if (value !== undefined) snapshot[cappedName] = Math.min(value, capBytes);
    }
    return snapshot;
  };
  return {
    requestAdapter: async (options?: GPURequestAdapterOptions) => {
      const adapter = await gpu.requestAdapter(options);
      if (adapter === null) return null;
      return {
        features: adapter.features,
        info: adapter.info,
        limits: clampedLimits(adapter.limits),
        requestDevice: async (descriptor?: GPUDeviceDescriptor) => {
          for (const [name, value] of Object.entries(descriptor?.requiredLimits ?? {})) {
            if ((name === "maxBufferSize" || name === "maxStorageBufferBindingSize") && Number(value) > capBytes) {
              throw new DOMException(`TEST-ONLY adapter cap: ${name} ${String(value)} exceeds ${capBytes}`, "OperationError");
            }
          }
          return await adapter.requestDevice(descriptor);
        },
      } as unknown as GPUAdapter;
    },
  } as GPU;
}

const scope = self as unknown as DedicatedWorkerGlobalScope;
const testOnlyCapBytes = testOnlyGpuBufferCapBytes();
const testOnlyGpu = testOnlyCapBytes === undefined ? undefined : testOnlyClampedGpu(testOnlyCapBytes);
installAceWorkerRuntime(scope, createAceDirectOnlyWebGpuPipelineBackend(testOnlyGpu === undefined ? {} : { gpu: testOnlyGpu }));
