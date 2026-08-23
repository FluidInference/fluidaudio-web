export const ACE_MODEL_PROFILE_IDS = [
  "reference-bf16",
  "raw-fp16",
] as const;

export type AceModelProfileId = (typeof ACE_MODEL_PROFILE_IDS)[number];
export type AceKernelBackend = "portable" | "subgroups";
export type AceSchedulingProfile = "cooperative" | "benchmark";

export type AceExecutionProfileId =
  | "reference-bf16-portable"
  | "reference-bf16-subgroups"
  | "raw-fp16-portable";

export interface AceExecutionProfile {
  readonly id: AceExecutionProfileId;
  readonly modelProfile: AceModelProfileId;
  readonly weightStorage: "packed-bf16-u32" | "float16";
  readonly matrixArithmetic: "float32" | "float16";
  readonly sensitiveReductions: "float32";
  readonly vaeArithmetic: "float32";
  readonly kernelBackend: AceKernelBackend;
  readonly requiredFeatures: readonly AceRequiredWebGpuFeature[];
}

export const ACE_REFERENCE_PORTABLE_PROFILE: Readonly<AceExecutionProfile> =
  Object.freeze({
    id: "reference-bf16-portable",
    modelProfile: "reference-bf16",
    weightStorage: "packed-bf16-u32",
    matrixArithmetic: "float32",
    sensitiveReductions: "float32",
    vaeArithmetic: "float32",
    kernelBackend: "portable",
    requiredFeatures: [] as const,
  });

export const ACE_REFERENCE_SUBGROUP_PROFILE: Readonly<AceExecutionProfile> =
  Object.freeze({
    ...ACE_REFERENCE_PORTABLE_PROFILE,
    id: "reference-bf16-subgroups",
    kernelBackend: "subgroups",
    requiredFeatures: ["subgroups"] as const,
  });

export const ACE_FP16_PORTABLE_PROFILE: Readonly<AceExecutionProfile> =
  Object.freeze({
    id: "raw-fp16-portable",
    modelProfile: "raw-fp16",
    weightStorage: "float16",
    matrixArithmetic: "float16",
    sensitiveReductions: "float32",
    vaeArithmetic: "float32",
    kernelBackend: "portable",
    requiredFeatures: ["shader-f16"] as const,
  });

export const ACE_REQUIRED_SUBGROUP_SIZE = 32;

export const ACE_STOCK_WEBGPU_FEATURES = [
  "shader-f16",
  "subgroups",
  "timestamp-query",
] as const;

export type AceRequiredWebGpuFeature = "shader-f16" | "subgroups";
export type AceOptionalWebGpuFeature = "timestamp-query";
export type AceStockWebGpuFeature =
  (typeof ACE_STOCK_WEBGPU_FEATURES)[number];

export const ACE_WEBGPU_LIMIT_NAMES = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxUniformBufferBindingSize",
  "maxBindGroups",
  "maxBindingsPerBindGroup",
  "maxStorageBuffersPerShaderStage",
  "maxUniformBuffersPerShaderStage",
  "maxDynamicStorageBuffersPerPipelineLayout",
  "maxDynamicUniformBuffersPerPipelineLayout",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
  "maxComputeWorkgroupsPerDimension",
  "minStorageBufferOffsetAlignment",
  "minUniformBufferOffsetAlignment",
] as const;

export type AceWebGpuLimitName =
  (typeof ACE_WEBGPU_LIMIT_NAMES)[number];

export type AceWebGpuLimits = Readonly<
  Record<AceWebGpuLimitName, number>
>;

export const ACE_REQUIRED_WEBGPU_LIMITS: Readonly<
  Partial<Record<AceWebGpuLimitName, number>>
> = Object.freeze({
  // The pinned 256-frame / 64-overlap FP32 VAE correctness graph reaches
  // 62,914,560 activation elements (240 MiB) after its fourth upsample. Ask
  // for the next power-of-two capacity, matching the audited M3 device.
  // A 128 MiB device would require a smaller, separately audited VAE chunk
  // oracle; Stage 1 fails closed instead of silently changing seam geometry.
  maxBufferSize: 256 * 1024 * 1024,
  maxStorageBufferBindingSize: 256 * 1024 * 1024,
  maxStorageBuffersPerShaderStage: 8,
  maxComputeWorkgroupStorageSize: 16 * 1024,
  maxComputeInvocationsPerWorkgroup: 256,
  minStorageBufferOffsetAlignment: 256,
  minUniformBufferOffsetAlignment: 256,
});

const MINIMUM_LIMIT_NAMES = new Set<AceWebGpuLimitName>([
  "minStorageBufferOffsetAlignment",
  "minUniformBufferOffsetAlignment",
]);

export interface AceAdapterInfoSnapshot {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
  readonly isFallbackAdapter?: boolean;
}

export interface AceWebGpuFeatureStatus {
  readonly adapterSupported: boolean;
  readonly deviceEnabled: boolean;
  readonly required: boolean;
  readonly requested: boolean;
}

export type AceWebGpuFeatureReport = Readonly<
  Record<AceStockWebGpuFeature, AceWebGpuFeatureStatus>
>;

/** Immutable diagnostic snapshot created after a GPUDevice is established. */
export interface AceWebGpuCapabilityReport {
  readonly executionProfile: AceExecutionProfile;
  readonly schedulingProfile: AceSchedulingProfile;
  readonly adapterInfo: AceAdapterInfoSnapshot;
  readonly adapterFeatures: readonly string[];
  readonly deviceFeatures: readonly string[];
  readonly stockFeatures: AceWebGpuFeatureReport;
  readonly requiredFeatures: readonly AceRequiredWebGpuFeature[];
  readonly requestedOptionalFeatures: readonly AceOptionalWebGpuFeature[];
  readonly adapterLimits: AceWebGpuLimits;
  readonly deviceLimits: AceWebGpuLimits;
  readonly requestedLimits: Readonly<
    Partial<Record<AceWebGpuLimitName, number>>
  >;
}

export interface AceWebGpuLimitDeficit {
  readonly name: AceWebGpuLimitName;
  readonly requested: number;
  readonly available: number;
  readonly relation: "at-least" | "at-most";
}

export type AceWebGpuUnavailableCode =
  | "INSECURE_CONTEXT"
  | "WEBGPU_UNAVAILABLE"
  | "ADAPTER_UNAVAILABLE"
  | "FEATURE_UNAVAILABLE"
  | "LIMIT_UNAVAILABLE";

export class AceWebGpuUnavailableError extends Error {
  readonly code: AceWebGpuUnavailableCode;
  readonly deficits: readonly AceWebGpuLimitDeficit[];

  constructor(
    code: AceWebGpuUnavailableCode,
    message: string,
    deficits: readonly AceWebGpuLimitDeficit[] = [],
  ) {
    super(message);
    this.name = "AceWebGpuUnavailableError";
    this.code = code;
    this.deficits = deficits;
  }
}

export interface AceSupportReport {
  readonly supported: boolean;
  readonly errors: readonly string[];
  readonly warnings: readonly string[];
  readonly executionProfile: AceExecutionProfile | null;
  readonly adapter: AceAdapterInfoSnapshot | null;
  readonly adapterLimits: AceWebGpuLimits | null;
  readonly limitDeficits: readonly AceWebGpuLimitDeficit[];
  readonly storage: {
    readonly opfsAvailable: boolean;
    readonly persistenceApiAvailable: boolean;
  };
}

/** Runtime guard for JavaScript callers crossing the typed API boundary. */
export function isAceModelProfileId(value: unknown): value is AceModelProfileId {
  return (
    typeof value === "string" &&
    (ACE_MODEL_PROFILE_IDS as readonly string[]).includes(value)
  );
}

/**
 * Select math precision explicitly; only kernel backend is capability-derived.
 * Stage 1 never silently changes the correctness profile to FP16.
 */
export function selectAceExecutionProfile(
  modelProfile: AceModelProfileId,
  availableFeatures: Iterable<string>,
  subgroupMinSize?: number,
  subgroupMaxSize?: number,
): AceExecutionProfile {
  if (!isAceModelProfileId(modelProfile)) {
    throw new TypeError(`Unknown ACE model profile ${String(modelProfile)}`);
  }
  const features = new Set(availableFeatures);
  if (modelProfile === "raw-fp16" && !features.has("shader-f16")) {
    throw new AceWebGpuUnavailableError(
      "FEATURE_UNAVAILABLE",
      "The raw-fp16 profile requires shader-f16",
    );
  }
  const fixedSubgroups =
    features.has("subgroups") &&
    subgroupMinSize === ACE_REQUIRED_SUBGROUP_SIZE &&
    subgroupMaxSize === ACE_REQUIRED_SUBGROUP_SIZE;
  if (modelProfile === "reference-bf16") {
    return fixedSubgroups
      ? ACE_REFERENCE_SUBGROUP_PROFILE
      : ACE_REFERENCE_PORTABLE_PROFILE;
  }
  // The audited fixed-32 subgroup GEMM consumes packed BF16 only. Raw FP16
  // remains on the portable tile-major kernel until it has its own exact A/B.
  return ACE_FP16_PORTABLE_PROFILE;
}

export function snapshotAceWebGpuLimits(
  limits: GPUSupportedLimits | Readonly<Record<string, number>>,
): AceWebGpuLimits {
  const source = limits as unknown as Readonly<Record<string, number>>;
  const snapshot = {} as Record<AceWebGpuLimitName, number>;
  for (const name of ACE_WEBGPU_LIMIT_NAMES) {
    const value = source[name];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new TypeError(`WebGPU limit ${name} is unavailable or non-numeric`);
    }
    snapshot[name] = value;
  }
  return Object.freeze(snapshot);
}

export function findAceWebGpuLimitDeficits(
  availableLimits: GPUSupportedLimits | Readonly<Record<string, number>>,
  requestedLimits: Readonly<Partial<Record<AceWebGpuLimitName, number>>>,
): AceWebGpuLimitDeficit[] {
  const available = availableLimits as unknown as Readonly<Record<string, number>>;
  const deficits: AceWebGpuLimitDeficit[] = [];
  for (const [rawName, requested] of Object.entries(requestedLimits)) {
    if (requested === undefined) continue;
    if (!(ACE_WEBGPU_LIMIT_NAMES as readonly string[]).includes(rawName)) {
      throw new TypeError(`Unknown ACE WebGPU limit ${rawName}`);
    }
    const name = rawName as AceWebGpuLimitName;
    if (!Number.isSafeInteger(requested) || requested < 0) {
      throw new RangeError(`Requested WebGPU limit ${name} must be a non-negative integer`);
    }
    const supported = available[name];
    const relation = MINIMUM_LIMIT_NAMES.has(name) ? "at-most" : "at-least";
    if (typeof supported !== "number" || !Number.isFinite(supported)) {
      deficits.push({ name, requested, available: Number.NaN, relation });
    } else if (
      (relation === "at-most" && supported > requested) ||
      (relation === "at-least" && supported < requested)
    ) {
      deficits.push({ name, requested, available: supported, relation });
    }
  }
  return deficits;
}

export async function inspectWebGpuSupport(
  modelProfile: AceModelProfileId,
): Promise<AceSupportReport> {
  const storage = inspectStorageSupport();
  if (!isAceModelProfileId(modelProfile)) {
    return unsupported(
      `Unknown ACE model profile ${String(modelProfile)}`,
      storage,
    );
  }
  if (globalThis.isSecureContext === false) {
    return unsupported("WebGPU requires HTTPS or localhost", storage);
  }
  const gpu = (globalThis.navigator as Navigator & { readonly gpu?: GPU } | undefined)?.gpu;
  if (gpu === undefined) {
    return unsupported("This browser does not expose WebGPU", storage);
  }

  try {
    const adapter = await gpu.requestAdapter({
      powerPreference: "high-performance",
      forceFallbackAdapter: false,
    });
    if (adapter === null) {
      return unsupported("No WebGPU adapter is available", storage);
    }
    const info = adapter.info as GPUAdapterInfo & {
      readonly subgroupMinSize?: number;
      readonly subgroupMaxSize?: number;
      readonly isFallbackAdapter?: boolean;
    };
    const adapterInfo = snapshotAdapterInfo(info);
    let executionProfile: AceExecutionProfile;
    try {
      executionProfile = selectAceExecutionProfile(
        modelProfile,
        adapter.features,
        info.subgroupMinSize,
        info.subgroupMaxSize,
      );
    } catch (error) {
      return {
        supported: false,
        errors: [error instanceof Error ? error.message : String(error)],
        warnings: [],
        executionProfile: null,
        adapter: adapterInfo,
        adapterLimits: snapshotAceWebGpuLimits(adapter.limits),
        limitDeficits: [],
        storage,
      };
    }
    const limits = snapshotAceWebGpuLimits(adapter.limits);
    const deficits = findAceWebGpuLimitDeficits(
      limits,
      ACE_REQUIRED_WEBGPU_LIMITS,
    );
    return {
      supported: deficits.length === 0,
      errors: deficits.map(
        ({ name, requested, available, relation }) =>
          `${name} must be ${relation} ${requested}; adapter reports ${available}`,
      ),
      warnings: [
        ...(adapterInfo.isFallbackAdapter === true
          ? ["The browser selected a fallback WebGPU adapter."]
          : []),
        ...(!storage.opfsAvailable
          ? ["OPFS is unavailable; the model cache cannot use its intended backend."]
          : []),
      ],
      executionProfile,
      adapter: adapterInfo,
      adapterLimits: limits,
      limitDeficits: deficits,
      storage,
    };
  } catch (error) {
    return unsupported(
      error instanceof Error ? error.message : String(error),
      storage,
    );
  }
}

function snapshotAdapterInfo(
  info: GPUAdapterInfo & {
    readonly subgroupMinSize?: number;
    readonly subgroupMaxSize?: number;
    readonly isFallbackAdapter?: boolean;
  },
): AceAdapterInfoSnapshot {
  return {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
    ...(info.subgroupMinSize === undefined
      ? {}
      : { subgroupMinSize: info.subgroupMinSize }),
    ...(info.subgroupMaxSize === undefined
      ? {}
      : { subgroupMaxSize: info.subgroupMaxSize }),
    ...(info.isFallbackAdapter === undefined
      ? {}
      : { isFallbackAdapter: info.isFallbackAdapter }),
  };
}

function inspectStorageSupport(): AceSupportReport["storage"] {
  const storage = globalThis.navigator?.storage;
  return {
    opfsAvailable:
      storage !== undefined && typeof storage.getDirectory === "function",
    persistenceApiAvailable:
      storage !== undefined && typeof storage.persist === "function",
  };
}

function unsupported(
  error: string,
  storage: AceSupportReport["storage"],
): AceSupportReport {
  return {
    supported: false,
    errors: [error],
    warnings: [],
    executionProfile: null,
    adapter: null,
    adapterLimits: null,
    limitDeficits: [],
    storage,
  };
}
