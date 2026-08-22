import {
  ACE_REQUIRED_WEBGPU_LIMITS,
  ACE_STOCK_WEBGPU_FEATURES,
  ACE_WEBGPU_LIMIT_NAMES,
  AceWebGpuUnavailableError,
  findAceWebGpuLimitDeficits,
  selectAceExecutionProfile,
  snapshotAceWebGpuLimits,
  type AceAdapterInfoSnapshot,
  type AceModelProfileId,
  type AceOptionalWebGpuFeature,
  type AceRequiredWebGpuFeature,
  type AceSchedulingProfile,
  type AceStockWebGpuFeature,
  type AceWebGpuCapabilityReport,
  type AceWebGpuFeatureReport,
  type AceWebGpuLimitName,
} from "./capabilities.js";

export type AceGpuRuntimeEvent =
  | {
      readonly type: "device-lost";
      readonly reason: string;
      readonly message: string;
    }
  | {
      readonly type: "uncaptured-error";
      readonly errorType: string;
      readonly message: string;
    };

export interface AceRequestWebGpuDeviceOptions {
  readonly modelProfile: AceModelProfileId;
  readonly schedulingProfile?: AceSchedulingProfile;
  readonly gpu?: GPU;
  readonly signal?: AbortSignal;
  readonly enableTimestampQueries?: boolean;
  /** Additional graph requirements to union with the selected main profile. */
  readonly requiredFeatures?: readonly AceRequiredWebGpuFeature[];
  /** Additional graph capacities to union with the Stage-1 device contract. */
  readonly requiredLimits?: Readonly<
    Partial<Record<AceWebGpuLimitName, number>>
  >;
  readonly onRuntimeEvent?: (event: AceGpuRuntimeEvent) => void;
}

export class AceWebGpuDeviceContext {
  readonly lost: Promise<AceGpuRuntimeEvent & { readonly type: "device-lost" }>;
  private destroyed = false;
  private readonly uncapturedErrorListener: (event: GPUUncapturedErrorEvent) => void;

  constructor(
    readonly adapter: GPUAdapter,
    readonly device: GPUDevice,
    readonly capabilities: AceWebGpuCapabilityReport,
    onRuntimeEvent?: (event: AceGpuRuntimeEvent) => void,
  ) {
    this.uncapturedErrorListener = (event) => {
      const diagnostic: AceGpuRuntimeEvent = {
        type: "uncaptured-error",
        errorType: event.error.constructor.name,
        message: event.error.message,
      };
      reportRuntimeEvent(onRuntimeEvent, diagnostic);
    };
    device.addEventListener("uncapturederror", this.uncapturedErrorListener);
    this.lost = device.lost.then((info) => {
      const diagnostic = {
        type: "device-lost" as const,
        reason: String(info.reason),
        message: info.message,
      };
      if (!(this.destroyed && info.reason === "destroyed")) {
        reportRuntimeEvent(onRuntimeEvent, diagnostic);
      }
      return diagnostic;
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.device.removeEventListener("uncapturederror", this.uncapturedErrorListener);
    this.device.destroy();
  }
}

function reportRuntimeEvent(
  callback: ((event: AceGpuRuntimeEvent) => void) | undefined,
  event: AceGpuRuntimeEvent,
): void {
  try {
    callback?.(event);
  } catch {
    // Diagnostics are observational. User callback failures must not reject the
    // device-loss promise or escape a browser event listener into graph state.
  }
}

/** Request the exact feature/limit contract used by the selected model profile. */
export async function requestAceWebGpuDevice(
  options: AceRequestWebGpuDeviceOptions,
): Promise<AceWebGpuDeviceContext> {
  options.signal?.throwIfAborted();
  if (globalThis.isSecureContext === false) {
    throw new AceWebGpuUnavailableError(
      "INSECURE_CONTEXT",
      "WebGPU requires HTTPS or localhost",
    );
  }
  const gpu =
    options.gpu ??
    (globalThis.navigator as (Navigator & { readonly gpu?: GPU }) | undefined)?.gpu;
  if (gpu === undefined) {
    throw new AceWebGpuUnavailableError(
      "WEBGPU_UNAVAILABLE",
      "This browser does not expose WebGPU",
    );
  }
  const adapter = await gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  options.signal?.throwIfAborted();
  if (adapter === null) {
    throw new AceWebGpuUnavailableError(
      "ADAPTER_UNAVAILABLE",
      "No WebGPU adapter is available",
    );
  }

  const adapterInfo = snapshotAdapter(adapter.info);
  const executionProfile = selectAceExecutionProfile(
    options.modelProfile,
    adapter.features,
    adapterInfo.subgroupMinSize,
    adapterInfo.subgroupMaxSize,
  );
  const adapterLimits = snapshotAceWebGpuLimits(adapter.limits);
  const requiredLimits = mergeRequiredLimits(
    ACE_REQUIRED_WEBGPU_LIMITS,
    options.requiredLimits ?? {},
  );
  const deficits = findAceWebGpuLimitDeficits(
    adapterLimits,
    requiredLimits,
  );
  if (deficits.length !== 0) {
    throw new AceWebGpuUnavailableError(
      "LIMIT_UNAVAILABLE",
      deficits
        .map(
          ({ name, requested, available, relation }) =>
            `${name} must be ${relation} ${requested}; adapter reports ${available}`,
        )
        .join("; "),
      deficits,
    );
  }

  const requestedOptionalFeatures: AceOptionalWebGpuFeature[] = [];
  if (
    options.enableTimestampQueries === true &&
    adapter.features.has("timestamp-query")
  ) {
    requestedOptionalFeatures.push("timestamp-query");
  }
  const requiredFeatures = Object.freeze([
    ...new Set([
      ...executionProfile.requiredFeatures,
      ...(options.requiredFeatures ?? []),
    ]),
  ]);
  for (const feature of requiredFeatures) {
    if (!adapter.features.has(feature)) {
      throw new AceWebGpuUnavailableError(
        "FEATURE_UNAVAILABLE",
        `The selected ACE graph requires ${feature}`,
      );
    }
  }
  const requestedDeviceFeatures = [
    ...requiredFeatures,
    ...requestedOptionalFeatures,
  ];
  const requestedLimits = deviceRequestedLimits(requiredLimits);
  let device: GPUDevice | undefined;
  try {
    device = await adapter.requestDevice({
      requiredFeatures: requestedDeviceFeatures as GPUFeatureName[],
      requiredLimits: requestedLimits as Record<string, number>,
      defaultQueue: { label: "ace-step-1.5-queue" },
      label: "ace-step-1.5-device",
    });
    options.signal?.throwIfAborted();
    const adapterFeatures = sortedFeatures(adapter.features);
    const deviceFeatures = sortedFeatures(device.features);
    const capabilityReport: AceWebGpuCapabilityReport = Object.freeze({
      executionProfile,
      schedulingProfile: options.schedulingProfile ?? "cooperative",
      adapterInfo,
      adapterFeatures,
      deviceFeatures,
      stockFeatures: stockFeatureReport(
        adapter.features,
        device.features,
        requiredFeatures,
        requestedOptionalFeatures,
      ),
      requiredFeatures,
      requestedOptionalFeatures: Object.freeze(requestedOptionalFeatures),
      adapterLimits,
      deviceLimits: snapshotAceWebGpuLimits(device.limits),
      requestedLimits,
    });
    return new AceWebGpuDeviceContext(
      adapter,
      device,
      capabilityReport,
      options.onRuntimeEvent,
    );
  } catch (error) {
    device?.destroy();
    throw error;
  }
}

function deviceRequestedLimits(
  requiredLimits: Readonly<Partial<Record<AceWebGpuLimitName, number>>>,
): Readonly<
  Partial<Record<AceWebGpuLimitName, number>>
> {
  // Minimum-offset-alignment limits are lower-is-better adapter properties,
  // not capacities to raise in a GPUDeviceDescriptor.
  const requested: Partial<Record<AceWebGpuLimitName, number>> = {};
  for (const [name, value] of Object.entries(requiredLimits)) {
    if (
      value !== undefined &&
      name !== "minStorageBufferOffsetAlignment" &&
      name !== "minUniformBufferOffsetAlignment"
    ) {
      requested[name as AceWebGpuLimitName] = value;
    }
  }
  return Object.freeze(requested);
}

function mergeRequiredLimits(
  base: Readonly<Partial<Record<AceWebGpuLimitName, number>>>,
  additional: Readonly<Partial<Record<AceWebGpuLimitName, number>>>,
): Readonly<Partial<Record<AceWebGpuLimitName, number>>> {
  const merged: Partial<Record<AceWebGpuLimitName, number>> = { ...base };
  for (const [rawName, value] of Object.entries(additional)) {
    if (!(ACE_WEBGPU_LIMIT_NAMES as readonly string[]).includes(rawName)) {
      throw new TypeError(`Unknown ACE WebGPU limit ${rawName}`);
    }
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(
        `Requested WebGPU limit ${rawName} must be a non-negative integer`,
      );
    }
    const name = rawName as AceWebGpuLimitName;
    const current = merged[name];
    merged[name] = current === undefined
      ? value
      : name === "minStorageBufferOffsetAlignment" ||
          name === "minUniformBufferOffsetAlignment"
        ? Math.min(current, value)
        : Math.max(current, value);
  }
  return Object.freeze(merged);
}

function stockFeatureReport(
  adapterFeatures: GPUSupportedFeatures,
  deviceFeatures: GPUSupportedFeatures,
  requiredFeatures: readonly string[],
  optionalFeatures: readonly string[],
): AceWebGpuFeatureReport {
  const report = {} as Record<AceStockWebGpuFeature, {
    adapterSupported: boolean;
    deviceEnabled: boolean;
    required: boolean;
    requested: boolean;
  }>;
  for (const feature of ACE_STOCK_WEBGPU_FEATURES) {
    report[feature] = Object.freeze({
      adapterSupported: adapterFeatures.has(feature),
      deviceEnabled: deviceFeatures.has(feature),
      required: requiredFeatures.includes(feature),
      requested:
        requiredFeatures.includes(feature) || optionalFeatures.includes(feature),
    });
  }
  return Object.freeze(report);
}

function sortedFeatures(features: GPUSupportedFeatures): readonly string[] {
  return Object.freeze([...features].map(String).sort());
}

function snapshotAdapter(
  info: GPUAdapterInfo & {
    readonly subgroupMinSize?: number;
    readonly subgroupMaxSize?: number;
    readonly isFallbackAdapter?: boolean;
  },
): AceAdapterInfoSnapshot {
  return Object.freeze({
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
  });
}
