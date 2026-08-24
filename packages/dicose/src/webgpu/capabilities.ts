export interface DiCoSeSupport {
  readonly supported: boolean;
  readonly errors: readonly string[];
  readonly adapter: Readonly<{
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
  }> | null;
}

const REQUIRED_FEATURES = ["shader-f16", "subgroups"] as const;
const REQUIRED_BUFFER_BYTES = 1_024 * 1024 * 1024;
const REQUIRED_WORKGROUP_STORAGE_BYTES = 25_344;

export async function checkSupport(): Promise<DiCoSeSupport> {
  if (!globalThis.isSecureContext) return unsupported("WebGPU requires HTTPS or localhost");
  if (navigator.gpu === undefined) return unsupported("This browser does not expose WebGPU");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) return unsupported("No high-performance WebGPU adapter is available");
  const errors: string[] = [];
  for (const feature of REQUIRED_FEATURES) {
    if (!adapter.features.has(feature)) errors.push(`DiCoSe requires ${feature}`);
  }
  // The raw attention and RMSNorm kernels map one logical 64-wide head over
  // two lanes per member of a fixed eight-subgroup, 256-lane workgroup.
  // Accepting variable subgroup widths would silently produce incorrect audio.
  if (adapter.info.subgroupMinSize !== 32 || adapter.info.subgroupMaxSize !== 32) {
    errors.push("DiCoSe requires fixed 32-wide WebGPU subgroups");
  }
  if (adapter.limits.maxBufferSize < REQUIRED_BUFFER_BYTES) {
    errors.push(`DiCoSe requires a ${REQUIRED_BUFFER_BYTES}-byte GPU buffer`);
  }
  if (adapter.limits.maxStorageBufferBindingSize < REQUIRED_BUFFER_BYTES) {
    errors.push(`DiCoSe requires a ${REQUIRED_BUFFER_BYTES}-byte storage binding`);
  }
  if (adapter.limits.maxComputeWorkgroupStorageSize < REQUIRED_WORKGROUP_STORAGE_BYTES) {
    errors.push(`DiCoSe requires ${REQUIRED_WORKGROUP_STORAGE_BYTES} bytes of workgroup storage`);
  }
  const info = adapter.info;
  return {
    supported: errors.length === 0,
    errors,
    adapter: {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    },
  };
}

export async function requestDiCoSeDevice(): Promise<GPUDevice> {
  const support = await checkSupport();
  if (!support.supported) throw new Error(support.errors.join("; "));
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("WebGPU adapter disappeared");
  // Device limits default to conservative WebGPU minima even when an adapter
  // advertises much larger storage. Request the ceiling explicitly: the f16
  // package itself is ~623 MB and several activation buffers are hundreds of
  // megabytes on the supplied audio.
  return await adapter.requestDevice({
    requiredFeatures: [...REQUIRED_FEATURES],
    requiredLimits: {
      maxBufferSize: REQUIRED_BUFFER_BYTES,
      maxStorageBufferBindingSize: REQUIRED_BUFFER_BYTES,
      maxComputeWorkgroupStorageSize: REQUIRED_WORKGROUP_STORAGE_BYTES,
    },
  });
}

function unsupported(error: string): DiCoSeSupport {
  return { supported: false, errors: [error], adapter: null };
}
