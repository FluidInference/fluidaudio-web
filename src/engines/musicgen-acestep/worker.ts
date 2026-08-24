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

// TEST-ONLY: the page threads flags to the worker through its name (driven by
// the page query parameters `aceTestMaxGpuBufferBytes` and
// `aceTestDisableSubgroups`):
// - `testMaxGpuBufferBytes=N` emulates a capped WebGPU adapter — e.g.
//   N=1073741824 reproduces every iOS adapter's one-GiB
//   maxBufferSize/maxStorageBufferBindingSize. The facade only ever LOWERS
//   limits: adapter snapshots are clamped and requestDevice rejects any
//   requiredLimits above the cap, exactly like a real capped adapter would.
// - `testDisableSubgroups=1` emulates Safari/Firefox: the `subgroups` feature
//   is removed from the adapter feature set, subgroupMinSize/subgroupMaxSize
//   disappear from the adapter info, and requestDevice rejects any descriptor
//   that still requires `subgroups`, exactly like a real adapter without the
//   feature would.
interface TestOnlyGpuOverrides {
  readonly capBytes?: number;
  readonly disableSubgroups: boolean;
}

function testOnlyGpuOverrides(): TestOnlyGpuOverrides | undefined {
  const query = self.name.split("?")[1];
  if (query === undefined) return undefined;
  const flags = new URLSearchParams(query);
  const rawCap = flags.get("testMaxGpuBufferBytes");
  const capValue = rawCap === null ? undefined : Number(rawCap);
  const capBytes = capValue !== undefined && Number.isSafeInteger(capValue) && capValue > 0 ? capValue : undefined;
  const disableSubgroups = flags.get("testDisableSubgroups") === "1";
  if (capBytes === undefined && !disableSubgroups) return undefined;
  return { ...(capBytes === undefined ? {} : { capBytes }), disableSubgroups };
}

function testOnlyGpuFacade(overrides: TestOnlyGpuOverrides): GPU | undefined {
  const gpu = navigator.gpu;
  if (gpu === undefined) return undefined;
  const capBytes = overrides.capBytes;
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
    if (capBytes !== undefined) {
      for (const cappedName of ["maxBufferSize", "maxStorageBufferBindingSize"]) {
        const value = snapshot[cappedName];
        if (value !== undefined) snapshot[cappedName] = Math.min(value, capBytes);
      }
    }
    return snapshot;
  };
  const maskedFeatures = (features: GPUSupportedFeatures): GPUSupportedFeatures =>
    overrides.disableSubgroups
      ? (new Set([...features].map(String).filter((feature) => feature !== "subgroups")) as unknown as GPUSupportedFeatures)
      : features;
  const maskedInfo = (info: GPUAdapterInfo): GPUAdapterInfo => {
    if (!overrides.disableSubgroups) return info;
    const source = info as GPUAdapterInfo & {
      readonly isFallbackAdapter?: boolean;
    };
    // Drop subgroupMinSize/subgroupMaxSize like an adapter without the
    // subgroups feature (Safari) would.
    return {
      vendor: source.vendor,
      architecture: source.architecture,
      device: source.device,
      description: source.description,
      ...(source.isFallbackAdapter === undefined ? {} : { isFallbackAdapter: source.isFallbackAdapter }),
    } as GPUAdapterInfo;
  };
  return {
    requestAdapter: async (options?: GPURequestAdapterOptions) => {
      const adapter = await gpu.requestAdapter(options);
      if (adapter === null) return null;
      return {
        features: maskedFeatures(adapter.features),
        info: maskedInfo(adapter.info),
        limits: clampedLimits(adapter.limits),
        requestDevice: async (descriptor?: GPUDeviceDescriptor) => {
          if (overrides.disableSubgroups) {
            for (const feature of descriptor?.requiredFeatures ?? []) {
              if (String(feature) === "subgroups") {
                throw new TypeError("TEST-ONLY adapter mask: required feature subgroups is unavailable");
              }
            }
          }
          if (capBytes !== undefined) {
            for (const [name, value] of Object.entries(descriptor?.requiredLimits ?? {})) {
              if ((name === "maxBufferSize" || name === "maxStorageBufferBindingSize") && Number(value) > capBytes) {
                throw new DOMException(`TEST-ONLY adapter cap: ${name} ${String(value)} exceeds ${capBytes}`, "OperationError");
              }
            }
          }
          return await adapter.requestDevice(descriptor);
        },
      } as unknown as GPUAdapter;
    },
  } as GPU;
}

const scope = self as unknown as DedicatedWorkerGlobalScope;
const testOnlyOverrides = testOnlyGpuOverrides();
const testOnlyGpu = testOnlyOverrides === undefined ? undefined : testOnlyGpuFacade(testOnlyOverrides);
installAceWorkerRuntime(scope, createAceDirectOnlyWebGpuPipelineBackend(testOnlyGpu === undefined ? {} : { gpu: testOnlyGpu }));
