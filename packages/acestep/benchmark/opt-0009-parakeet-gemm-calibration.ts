/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import parakeetGemmSource from
  "../../parakeet.wgsl/src/webgpu/kernels/gemm.ts?raw";
import parakeetRuntimePlanSource from
  "../../parakeet.wgsl/src/model/runtime-plan.ts?raw";
import parakeetCapabilitiesSource from
  "../../parakeet.wgsl/src/webgpu/capabilities.ts?raw";

import {
  f16SubgroupGemmWgsl,
  f32SubgroupGemmWgsl,
  planF16SubgroupGemm,
  planF32SubgroupGemm,
} from "../../parakeet.wgsl/src/webgpu/kernels/gemm.js";
import {
  RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT,
  RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT,
  RUNTIME_ROW_MAJOR_F16_LAYOUT,
  RUNTIME_ROW_MAJOR_F32_LAYOUT,
  runtimeGemmTileMajorF32ScalarIndex,
  runtimeGemmTileMajorScalarIndex,
} from "../../parakeet.wgsl/src/model/runtime-plan.js";

export const OPT_0009_PARAKEET_COMMIT =
  "7ee112738262a6f5a0efd2f150748a4087432fbb" as const;
export const OPT_0009_PARAKEET_GEMM_SOURCE_SHA256 =
  "35db4fe52a2d096af347ef4f2411159895d563b5df3aecfa19d70a9fb3f47286" as const;
export const OPT_0009_PARAKEET_RUNTIME_PLAN_SOURCE_SHA256 =
  "30effbabbf3a405f769c90f2ae4f33641d1da366e2578b1a8e3ef5b5213665cf" as const;
export const OPT_0009_PARAKEET_CAPABILITIES_SOURCE_SHA256 =
  "5f1f11ad0964ce7e3a373845e7e185a096a7af6b2820b21b4e1d899fa9f9ae15" as const;

export const ACE_OPT_0009_NATIVE_CALIBRATION_SCOPE = Object.freeze({
  comparison: "native-fp16-accumulation-vs-native-fp32-accumulation",
  answersFp16OperandsFp32Accumulation: false,
  closesExperiment: false,
} as const);

export const ACE_OPT_0009_GENERATED_SHADER_SHA256 = Object.freeze({
  "parakeet-m7520-k1024-n4096-fp16":
    "09c3270c6e38ec8149fb727f497f30afa3bab371a807ce5c70a1508777d5e6cf",
  "parakeet-m7520-k1024-n4096-fp32":
    "2d99d53f293ebc747be1f301252dd26ff75817e4236592e94e2569fc8d054a6a",
  "parakeet-m7520-k4096-n1024-fp16":
    "8a8ff0463750b4b411b402c3ada81cb987fd48a0db9d9136fb05c65429b196b6",
  "parakeet-m7520-k4096-n1024-fp32":
    "ed194fffb0425f405a698603dce2346646f817404c09f939d0f3c58e4ff5e8da",
  "parakeet-m7520-k1024-n1024-fp16":
    "e12c4b5b7eb834f510724f2565caaabe1023b6cf8059d00f8005f79c941cbecd",
  "parakeet-m7520-k1024-n1024-fp32":
    "b061e864a587a239a56b49fe979c297e80b9c5bf866c504e2a59905981679420",
  "ace-m2250-k2048-n2048-fp16":
    "15d4bf444a824c59861cd1d97b286f0ccafdf8f095c4da91c6b60ee5e641e93c",
  "ace-m2250-k2048-n2048-fp32":
    "d0a1f43caadb0958c0ba3ff63c4e6c279c906d6a41f6ab712197433ff2335fbc",
  "ace-m2250-k2048-n1024-fp16":
    "0ce4ed915be210333e32b93bd83c4b13ee29aa9c26bf3c287cdde0d7fb84e2e0",
  "ace-m2250-k2048-n1024-fp32":
    "00b320634c5a35dd37e8eef26d670edba30cf601bdc31125f06332ec0ee9183f",
  "ace-m2250-k2048-n6144-fp16":
    "76e7f93f843b7486a734e60a510232a8244d0e86ef95162fd026d193357fe93d",
  "ace-m2250-k2048-n6144-fp32":
    "da1658c1732ab4936b8d052390f8bc8c896e2a2dfd51514f4c8404e08de5547b",
  "ace-m2250-k6144-n2048-fp16":
    "c58f8b1ae6b1d12d07934968f4c69907ff16d392797e6b20d5739c3a73f8af04",
  "ace-m2250-k6144-n2048-fp32":
    "7810e270382ce81406b83ca775d8fb5af3649012a714deba8e3469758d7d429a",
} as const satisfies Readonly<Record<string, string>>);
export const OPT_0009_ALLOCATION_COMMIT =
  "303ab8df036df71768a56774c59c75c4cfe30aa9" as const;

export const OPT_0009_PARAKEET_RAW_SOURCES = Object.freeze({
  gemm: parakeetGemmSource,
  runtimePlan: parakeetRuntimePlanSource,
  capabilities: parakeetCapabilitiesSource,
});

export const ACE_OPT_0009_PAIRED_ORDERS = Object.freeze([
  Object.freeze(["fp16", "fp32"]),
  Object.freeze(["fp32", "fp16"]),
  Object.freeze(["fp32", "fp16"]),
  Object.freeze(["fp16", "fp32"]),
] satisfies readonly (readonly AceOpt0009Precision[])[]);

export interface AceOpt0009ThermalMetadata {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly durationSeconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: 1_000;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
}

export interface AceOpt0009OutputScan {
  readonly elementCount: number;
  readonly finiteCount: number;
  readonly nonFiniteCount: number;
  readonly nonzeroCount: number;
  readonly sentinelCount: number;
  readonly minimum: number | null;
  readonly maximum: number | null;
  readonly rawFnv1a32: string;
  readonly overflowClassification: "not-applicable" | "none" | "observed";
}

export interface AceOpt0009ExecutionCounts {
  readonly plannedExecutions: number;
  readonly encodedCommandBuffers: number;
  readonly submissions: number;
  readonly drains: number;
  readonly dispatches: number;
}

export type AceOpt0009Precision = "fp16" | "fp32";
export type AceOpt0009KernelPath =
  | "parakeet-direct-tile-major"
  | "parakeet-staged-row-major";
export type AceOpt0009Scope =
  | "parakeet-production-calibration"
  | "ace-180s-shape-calibration";

export interface AceOpt0009Shape {
  readonly id: string;
  readonly scope: AceOpt0009Scope;
  readonly rows: number;
  readonly inner: number;
  readonly columns: number;
  readonly kernelPath: AceOpt0009KernelPath;
}

export interface AceOpt0009VariantPlan extends AceOpt0009Shape {
  readonly variantId: string;
  readonly precision: AceOpt0009Precision;
  readonly sourceClassification: "unchanged-parakeet-generator";
  readonly tileRows: 32 | 48;
  readonly tileColumns: 128 | 256;
  readonly workgroups: readonly [number, number, 1];
  readonly scheduledRows: number;
  readonly validMacs: number;
  readonly scheduledMacs: number;
  readonly validFlops: number;
  readonly scheduledFlops: number;
  readonly rowUtilization: number;
  readonly activationBytes: number;
  readonly weightBytes: number;
  readonly outputBytes: number;
}

export const ACE_OPT_0009_SHAPES = Object.freeze([
  shape(
    "parakeet-m7520-k1024-n4096",
    "parakeet-production-calibration",
    7_520,
    1_024,
    4_096,
    "parakeet-direct-tile-major",
  ),
  shape(
    "parakeet-m7520-k4096-n1024",
    "parakeet-production-calibration",
    7_520,
    4_096,
    1_024,
    "parakeet-direct-tile-major",
  ),
  shape(
    "parakeet-m7520-k1024-n1024",
    "parakeet-production-calibration",
    7_520,
    1_024,
    1_024,
    "parakeet-direct-tile-major",
  ),
  shape(
    "ace-m2250-k2048-n2048",
    "ace-180s-shape-calibration",
    2_250,
    2_048,
    2_048,
    "parakeet-staged-row-major",
  ),
  shape(
    "ace-m2250-k2048-n1024",
    "ace-180s-shape-calibration",
    2_250,
    2_048,
    1_024,
    "parakeet-staged-row-major",
  ),
  shape(
    "ace-m2250-k2048-n6144",
    "ace-180s-shape-calibration",
    2_250,
    2_048,
    6_144,
    "parakeet-staged-row-major",
  ),
  shape(
    "ace-m2250-k6144-n2048",
    "ace-180s-shape-calibration",
    2_250,
    6_144,
    2_048,
    "parakeet-staged-row-major",
  ),
] satisfies readonly AceOpt0009Shape[]);

export function planAceOpt0009Variants(): readonly AceOpt0009VariantPlan[] {
  return Object.freeze(ACE_OPT_0009_SHAPES.flatMap((candidate) => [
    planVariant(candidate, "fp16"),
    planVariant(candidate, "fp32"),
  ]));
}

export function aceOpt0009VariantWgsl(
  plan: AceOpt0009VariantPlan,
): string {
  requireCanonicalPlan(plan);
  const specialization = {
    label: `opt-0009-${plan.variantId}`,
    rows: plan.rows,
    inner: plan.inner,
    columns: plan.columns,
    activation: "none" as const,
    hasBias: false,
    residualScale: null,
  };
  if (plan.precision === "fp16") {
    return f16SubgroupGemmWgsl({
      ...specialization,
      weightLayout: plan.kernelPath === "parakeet-direct-tile-major"
        ? RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT
        : RUNTIME_ROW_MAJOR_F16_LAYOUT,
    });
  }
  return f32SubgroupGemmWgsl({
    ...specialization,
    weightLayout: plan.kernelPath === "parakeet-direct-tile-major"
      ? RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT
      : RUNTIME_ROW_MAJOR_F32_LAYOUT,
  });
}

/**
 * Map a logical K-major/output-minor scalar to the physical B scalar index
 * consumed by the pinned Parakeet generator.
 */
export function aceOpt0009WeightScalarIndex(
  plan: Pick<
    AceOpt0009VariantPlan,
    "precision" | "kernelPath" | "inner" | "columns"
  >,
  logicalScalarIndex: number,
): number {
  const scalarCount = checkedProduct(
    plan.inner,
    plan.columns,
    "OPT-0009 weight scalars",
  );
  requireIndex(logicalScalarIndex, scalarCount, "OPT-0009 logical weight");
  if (plan.kernelPath === "parakeet-staged-row-major") {
    return logicalScalarIndex;
  }
  return plan.precision === "fp16"
    ? runtimeGemmTileMajorScalarIndex(
      logicalScalarIndex,
      plan.inner,
      plan.columns,
    )
    : runtimeGemmTileMajorF32ScalarIndex(
      logicalScalarIndex,
      plan.inner,
      plan.columns,
    );
}

export function requireAceOpt0009Fixed32Device(
  features: ReadonlySet<string>,
  subgroupMinSize: number | undefined,
  subgroupMaxSize: number | undefined,
): void {
  if (
    !features.has("shader-f16") ||
    !features.has("subgroups") ||
    subgroupMinSize !== 32 ||
    subgroupMaxSize !== 32
  ) {
    throw new Error(
      "OPT-0009 requires shader-f16 and fixed 32-lane WebGPU subgroups",
    );
  }
}

export function parseAceOpt0009ThermalMetadata(
  parameters: URLSearchParams,
): AceOpt0009ThermalMetadata {
  const source = parameters.get("thermalSource");
  const durationSeconds = requiredQueryNumber(
    parameters,
    "thermalDurationSeconds",
  );
  const observationCount = requiredQueryNumber(
    parameters,
    "thermalObservations",
  );
  const pollMilliseconds = requiredQueryNumber(
    parameters,
    "thermalPollMilliseconds",
  );
  const maximumPollGapMilliseconds = requiredQueryNumber(
    parameters,
    "thermalMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = requiredQueryNumber(
    parameters,
    "thermalNonNominalObservations",
  );
  if (source !== "notifyutil-com.apple.system.thermalpressurelevel") {
    throw new Error("OPT-0009 requires the accepted notifyutil thermal source");
  }
  if (durationSeconds < 30 || observationCount < 31) {
    throw new Error("OPT-0009 requires at least 30 continuous nominal seconds");
  }
  if (pollMilliseconds !== 1_000) {
    throw new Error("OPT-0009 thermal polling must use 1,000 ms intervals");
  }
  if (maximumPollGapMilliseconds > 1_250) {
    throw new Error("OPT-0009 thermal poll gap exceeds 250 ms tolerance");
  }
  if (nonNominalObservationCount !== 0) {
    throw new Error("OPT-0009 thermal pre-gate observed non-nominal pressure");
  }
  return Object.freeze({
    source,
    durationSeconds,
    observationCount,
    pollMilliseconds: 1_000,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0,
  });
}

export function validateAceOpt0009OutputScan(
  precision: AceOpt0009Precision,
  expectedElements: number,
  scan: AceOpt0009OutputScan,
): void {
  if (
    scan.elementCount !== expectedElements ||
    scan.finiteCount + scan.nonFiniteCount !== expectedElements ||
    scan.sentinelCount !== 0 ||
    scan.nonzeroCount > scan.finiteCount
  ) {
    throw new Error("OPT-0009 output scan failed complete-write accounting");
  }
  if (precision === "fp32") {
    if (
      scan.nonFiniteCount !== 0 ||
      scan.overflowClassification !== "not-applicable"
    ) {
      throw new Error("OPT-0009 FP32 output must be completely finite");
    }
    return;
  }
  const expectedClassification = scan.nonFiniteCount === 0
    ? "none"
    : "observed";
  if (scan.overflowClassification !== expectedClassification) {
    throw new Error("OPT-0009 FP16 overflow classification is inconsistent");
  }
}

export function validateAceOpt0009ExecutionCounts(
  counts: AceOpt0009ExecutionCounts,
): void {
  if (
    !Number.isSafeInteger(counts.plannedExecutions) ||
    counts.plannedExecutions <= 0 ||
    counts.encodedCommandBuffers !== counts.plannedExecutions ||
    counts.submissions !== counts.plannedExecutions ||
    counts.drains !== counts.plannedExecutions ||
    counts.dispatches !== counts.plannedExecutions
  ) {
    throw new Error("OPT-0009 execution counts do not reconcile");
  }
}

export async function authenticateAceOpt0009ParakeetSources(
  parameters: URLSearchParams,
): Promise<Readonly<Record<string, string>>> {
  if (parameters.get("parakeetCommit") !== OPT_0009_PARAKEET_COMMIT) {
    throw new Error("OPT-0009 Parakeet commit metadata is missing or stale");
  }
  if (parameters.get("allocationCommit") !== OPT_0009_ALLOCATION_COMMIT) {
    throw new Error("OPT-0009 allocation commit metadata is missing or stale");
  }
  const actual = Object.freeze({
    gemm: await sha256Hex(parakeetGemmSource),
    runtimePlan: await sha256Hex(parakeetRuntimePlanSource),
    capabilities: await sha256Hex(parakeetCapabilitiesSource),
  });
  const expected = {
    gemm: OPT_0009_PARAKEET_GEMM_SOURCE_SHA256,
    runtimePlan: OPT_0009_PARAKEET_RUNTIME_PLAN_SOURCE_SHA256,
    capabilities: OPT_0009_PARAKEET_CAPABILITIES_SOURCE_SHA256,
  } as const;
  for (const key of Object.keys(expected) as (keyof typeof expected)[]) {
    if (actual[key] !== expected[key]) {
      throw new Error(`OPT-0009 rejected unauthenticated Parakeet ${key} source`);
    }
  }
  return actual;
}

function shape(
  id: string,
  scope: AceOpt0009Scope,
  rows: number,
  inner: number,
  columns: number,
  kernelPath: AceOpt0009KernelPath,
): AceOpt0009Shape {
  return Object.freeze({ id, scope, rows, inner, columns, kernelPath });
}

function planVariant(
  candidate: AceOpt0009Shape,
  precision: AceOpt0009Precision,
): AceOpt0009VariantPlan {
  const elementBytes = precision === "fp16" ? 2 : 4;
  const tileRows = candidate.kernelPath === "parakeet-direct-tile-major"
    ? 32
    : 48;
  const tileColumns = precision === "fp16" ? 256 : 128;
  const weightLayout = precision === "fp16"
    ? candidate.kernelPath === "parakeet-direct-tile-major"
      ? RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT
      : RUNTIME_ROW_MAJOR_F16_LAYOUT
    : candidate.kernelPath === "parakeet-direct-tile-major"
      ? RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT
      : RUNTIME_ROW_MAJOR_F32_LAYOUT;
  const workgroups = precision === "fp16"
    ? planF16SubgroupGemm(
      candidate.rows,
      candidate.inner,
      candidate.columns,
      weightLayout as typeof RUNTIME_GEMM_TILE_MAJOR_F16_LAYOUT |
        typeof RUNTIME_ROW_MAJOR_F16_LAYOUT,
    ).workgroups
    : planF32SubgroupGemm(
      candidate.rows,
      candidate.inner,
      candidate.columns,
      weightLayout as typeof RUNTIME_GEMM_TILE_MAJOR_F32_LAYOUT |
        typeof RUNTIME_ROW_MAJOR_F32_LAYOUT,
    ).workgroups;
  const scheduledRows = tileRows * workgroups[1];
  const validMacs = checkedProduct(
    checkedProduct(candidate.rows, candidate.inner, `${candidate.id} MxK`),
    candidate.columns,
    `${candidate.id} valid MACs`,
  );
  const scheduledMacs = checkedProduct(
    checkedProduct(scheduledRows, candidate.inner, `${candidate.id} scheduled MxK`),
    candidate.columns,
    `${candidate.id} scheduled MACs`,
  );
  return Object.freeze({
    ...candidate,
    variantId: `${candidate.id}-${precision}`,
    precision,
    sourceClassification: "unchanged-parakeet-generator",
    tileRows,
    tileColumns,
    workgroups,
    scheduledRows,
    validMacs,
    scheduledMacs,
    validFlops: validMacs * 2,
    scheduledFlops: scheduledMacs * 2,
    rowUtilization: candidate.rows / scheduledRows,
    activationBytes: checkedProduct(
      checkedProduct(candidate.rows, candidate.inner, `${candidate.id} A elements`),
      elementBytes,
      `${candidate.id} A bytes`,
    ),
    weightBytes: checkedProduct(
      checkedProduct(candidate.inner, candidate.columns, `${candidate.id} B elements`),
      elementBytes,
      `${candidate.id} B bytes`,
    ),
    outputBytes: checkedProduct(
      checkedProduct(candidate.rows, candidate.columns, `${candidate.id} C elements`),
      elementBytes,
      `${candidate.id} C bytes`,
    ),
  });
}

function requireCanonicalPlan(plan: AceOpt0009VariantPlan): void {
  const canonical = planAceOpt0009Variants().find(
    (candidate) => candidate.variantId === plan.variantId,
  );
  if (canonical === undefined || JSON.stringify(canonical) !== JSON.stringify(plan)) {
    throw new Error(`OPT-0009 rejected noncanonical variant ${plan.variantId}`);
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (
    !Number.isSafeInteger(left) ||
    left <= 0 ||
    !Number.isSafeInteger(right) ||
    right <= 0 ||
    !Number.isSafeInteger(product)
  ) {
    throw new RangeError(`${label} exceeds safe positive integer arithmetic`);
  }
  return product;
}

function requireIndex(index: number, count: number, label: string): void {
  if (!Number.isSafeInteger(index) || index < 0 || index >= count) {
    throw new RangeError(`${label} index ${index} exceeds ${count}`);
  }
}

type Opt0009Mode = "correctness" | "timing";

interface Opt0009ExecutionTiming {
  readonly roundIndex: number;
  readonly pairedOrder: string;
  readonly orderPosition: number;
  readonly wallMilliseconds: number;
  readonly encodeMilliseconds: number;
  readonly submitMilliseconds: number;
  readonly drainMilliseconds: number;
  readonly validLogicalTflops: number;
  readonly scheduledLogicalTflops: number;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
}

interface Opt0009SampleSummary {
  readonly count: number;
  readonly samples: readonly number[];
  readonly minimum: number;
  readonly median: number;
  readonly maximum: number;
  readonly range: number;
}

interface Opt0009SentinelValue {
  readonly index: number;
  readonly value: number;
}

interface Opt0009OutputReadback extends AceOpt0009OutputScan {
  readonly cpuSentinels: Readonly<Record<string, unknown>>;
  readonly sentinelValues: readonly Opt0009SentinelValue[];
}

interface Opt0009TimingSpanObserver {
  (startedAtEpochMilliseconds: number, completedAtEpochMilliseconds: number): void;
}

interface Opt0009PreparedVariant {
  readonly plan: AceOpt0009VariantPlan;
  readonly activation: GPUBuffer;
  readonly weight: GPUBuffer;
  readonly output: GPUBuffer;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly shaderSha256: string;
  readonly shaderCompilationMessages: readonly Readonly<Record<string, unknown>>[];
  readonly preparation: Readonly<Record<string, unknown>>;
  readonly counts: {
    encodedCommandBuffers: number;
    submissions: number;
    drains: number;
    dispatches: number;
  };
  destroy(): void;
}

class Opt0009BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  created = 0;
  destroyed = 0;
  maximumLive = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    this.live.add(buffer);
    this.created += 1;
    this.maximumLive = Math.max(this.maximumLive, this.live.size);
    return buffer;
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    buffer.destroy();
    this.destroyed += 1;
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number>> {
    return Object.freeze({
      created: this.created,
      destroyed: this.destroyed,
      live: this.live.size,
      maximumLive: this.maximumLive,
    });
  }
}

export function isAceOpt0009NativeCalibrationPage(pathname: string): boolean {
  return pathname.endsWith("/opt-0009-parakeet-gemm-calibration.html");
}

if (
  typeof document !== "undefined" &&
  isAceOpt0009NativeCalibrationPage(window.location.pathname)
) {
  installOpt0009BrowserUi();
}

function installOpt0009BrowserUi(): void {
  const correctness = requireButton("#run-correctness");
  const timing = requireButton("#run-timing");
  const start = (mode: Opt0009Mode): void => {
    correctness.disabled = true;
    timing.disabled = true;
    document.body.dataset.status = "running";
    updateBrowserProgress(`authenticating ${mode} run`);
    void runOpt0009Browser(mode).then(
      (result) => finishBrowser("passed", result),
      (error: unknown) => finishBrowser("failed", {
        schema: "ace-opt-0009-parakeet-gemm-calibration-v1",
        status: "failed",
        experimentId: "OPT-0009",
        mode,
        error: errorReceipt(error),
      }),
    );
  };
  correctness.addEventListener("click", () => start("correctness"), {
    once: true,
  });
  timing.addEventListener("click", () => start("timing"), { once: true });
}

async function runOpt0009Browser(mode: Opt0009Mode): Promise<unknown> {
  const runStartedAtEpochMilliseconds = epochMillisecondsNow();
  const parameters = new URL(window.location.href).searchParams;
  const sourceIdentity = await authenticateAceOpt0009ParakeetSources(parameters);
  const thermal = mode === "timing"
    ? parseAceOpt0009ThermalMetadata(parameters)
    : null;
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAceOpt0009Fixed32Device(
    new Set(adapter.features),
    adapter.info.subgroupMinSize,
    adapter.info.subgroupMaxSize,
  );
  const plans = planAceOpt0009Variants();
  const maximumBindingBytes = Math.max(...plans.flatMap((plan) => [
    plan.activationBytes,
    plan.weightBytes,
    plan.outputBytes,
  ]));
  requireAdapterLimits(adapter, maximumBindingBytes);
  const device = await adapter.requestDevice({
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: {
      maxBufferSize: maximumBindingBytes,
      maxStorageBufferBindingSize: maximumBindingBytes,
      maxComputeInvocationsPerWorkgroup: 192,
      maxComputeWorkgroupSizeX: 192,
      maxComputeWorkgroupStorageSize: 16_384,
    },
    label: "ace-opt-0009-calibration-device",
  });
  const tracker = new Opt0009BufferTracker();
  const uncapturedErrors: Readonly<Record<string, unknown>>[] = [];
  const onUncapturedError = (event: GPUUncapturedErrorEvent): void => {
    uncapturedErrors.push(Object.freeze({
      name: event.error.constructor.name,
      message: event.error.message,
    }));
  };
  device.addEventListener("uncapturederror", onUncapturedError);
  let unexpectedDeviceLoss: Readonly<Record<string, unknown>> | null = null;
  let destroyingDevice = false;
  void device.lost.then((info) => {
    if (!destroyingDevice) {
      unexpectedDeviceLoss = Object.freeze({
        reason: info.reason,
        message: info.message,
      });
    }
  });
  const overallHeartbeat = startHeartbeat();
  const shapeResults: unknown[] = [];
  let cancellation: unknown = null;
  let cleanup: Readonly<Record<string, unknown>>;
  let firstTimedAtEpochMilliseconds: number | null = null;
  let lastTimedAtEpochMilliseconds: number | null = null;
  const observeTimedSpan: Opt0009TimingSpanObserver = (started, completed) => {
    firstTimedAtEpochMilliseconds = firstTimedAtEpochMilliseconds === null
      ? started
      : Math.min(firstTimedAtEpochMilliseconds, started);
    lastTimedAtEpochMilliseconds = lastTimedAtEpochMilliseconds === null
      ? completed
      : Math.max(lastTimedAtEpochMilliseconds, completed);
  };
  try {
    for (const [shapeIndex, candidate] of ACE_OPT_0009_SHAPES.entries()) {
      updateBrowserProgress(
        `${mode} shape ${shapeIndex + 1}/${ACE_OPT_0009_SHAPES.length}: ` +
          candidate.id,
      );
      const pair = plans.filter((plan) => plan.id === candidate.id);
      if (pair.length !== 2) {
        throw new Error(`${candidate.id} did not resolve to one FP16/FP32 pair`);
      }
      const fp16Plan = pair.find((plan) => plan.precision === "fp16")!;
      const fp32Plan = pair.find((plan) => plan.precision === "fp32")!;
      const fp16 = await prepareVariant(device, tracker, fp16Plan);
      let fp32: Opt0009PreparedVariant | null = null;
      try {
        fp32 = await prepareVariant(device, tracker, fp32Plan);
        const result = await runPreparedPair(
          device,
          tracker,
          fp16,
          fp32,
          mode,
          observeTimedSpan,
        );
        if (
          mode === "timing" &&
          shapeIndex === ACE_OPT_0009_SHAPES.length - 1
        ) {
          cancellation = await runPostTimingCancellation(device, fp16);
        }
        shapeResults.push(result);
      } finally {
        fp32?.destroy();
        fp16.destroy();
      }
      if ((tracker.receipt().live as number) !== 0) {
        throw new Error(`${candidate.id} leaked a GPU buffer`);
      }
      await yieldToBrowser();
    }
    if (uncapturedErrors.length !== 0 || unexpectedDeviceLoss !== null) {
      throw new Error("OPT-0009 observed a WebGPU device event");
    }
  } finally {
    const heartbeat = overallHeartbeat.stop();
    tracker.destroyAll();
    device.removeEventListener("uncapturederror", onUncapturedError);
    destroyingDevice = true;
    device.destroy();
    cleanup = Object.freeze({
      ...tracker.receipt(),
      deviceDestroyed: true,
      uncapturedErrors: Object.freeze([...uncapturedErrors]),
      unexpectedDeviceLoss,
      overallHeartbeat: heartbeat,
      cleanupCompletedAtEpochMilliseconds: epochMillisecondsNow(),
    });
  }
  const completedAtEpochMilliseconds = epochMillisecondsNow();
  return Object.freeze({
    schema: "ace-opt-0009-parakeet-gemm-calibration-v1",
    status: "passed",
    experimentId: "OPT-0009",
    classification: "benchmark-only-calibration-no-production-change",
    mode,
    recordedAt: new Date(completedAtEpochMilliseconds).toISOString(),
    runStartedAtEpochMilliseconds,
    firstTimedAtEpochMilliseconds,
    lastTimedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    calibrationScope: ACE_OPT_0009_NATIVE_CALIBRATION_SCOPE,
    fixtureScope: Object.freeze({
      classification: "benign-throughput-and-native-conformance-fixture",
      coversFp16RangeEdges: false,
      coversAdversarialAccumulation: false,
      coversCancellationSensitiveMagnitudes: false,
      coversSignedZero: false,
      canCloseAccumulationChoice: false,
      followUp: "three-arm adversarial FP16-operands/FP32-accumulation fixture",
    }),
    sourceIdentity: Object.freeze({
      parakeetCommit: OPT_0009_PARAKEET_COMMIT,
      allocationCommit: OPT_0009_ALLOCATION_COMMIT,
      sourceSha256: sourceIdentity,
    }),
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      page: window.location.href,
      adapter: adapterReceipt(adapter),
    }),
    protocol: Object.freeze({
      thermal,
      embeddedThermalMetadataRole: "pre-gate-only",
      thermalTimingAuthority:
        "external-continuous-artifact-spanning-timing-through-cleanup",
      postFreezeIdentityJoin:
        "browser-machine-and-harness-commit-metadata-joined-externally",
      pairedOrders: ACE_OPT_0009_PAIRED_ORDERS.map((order) => order.join("-")),
      authoritativeTiming: "performance.now-completion-fenced-wall",
      oneOutstandingCommandBuffer: true,
      oneDispatchPerCommandBuffer: true,
      compilePackUploadExcludedFromTiming: true,
      fmaFlopConvention: 2,
      fp16OverflowPolicy: "classify-and-retain-calibration",
    }),
    shapes: Object.freeze(shapeResults),
    cancellation,
    cleanup,
  });
}

async function prepareVariant(
  device: GPUDevice,
  tracker: Opt0009BufferTracker,
  plan: AceOpt0009VariantPlan,
): Promise<Opt0009PreparedVariant> {
  const started = performance.now();
  const owned: GPUBuffer[] = [];
  let destroyed = false;
  try {
    const activation = tracker.create(device, {
      label: `${plan.variantId}-activation`,
      size: plan.activationBytes,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    owned.push(activation);
    const activationReceipt = fillActivation(activation, plan);
    activation.unmap();
    const weight = tracker.create(device, {
      label: `${plan.variantId}-weight`,
      size: plan.weightBytes,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    owned.push(weight);
    const weightReceipt = fillWeight(weight, plan);
    weight.unmap();
    const output = tracker.create(device, {
      label: `${plan.variantId}-output`,
      size: plan.outputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    owned.push(output);
    new Uint32Array(output.getMappedRange()).fill(
      plan.precision === "fp16" ? 0x7e00_7e00 : 0x7fc0_0000,
    );
    output.unmap();
    const source = aceOpt0009VariantWgsl(plan);
    const shaderSha256 = await sha256Hex(source);
    const expectedShaderSha256 = ACE_OPT_0009_GENERATED_SHADER_SHA256[
      plan.variantId as keyof typeof ACE_OPT_0009_GENERATED_SHADER_SHA256
    ];
    if (
      expectedShaderSha256 === undefined ||
      shaderSha256 !== expectedShaderSha256
    ) {
      throw new Error(
        `${plan.variantId} generated shader failed its pinned SHA-256 identity`,
      );
    }
    const compileStarted = performance.now();
    device.pushErrorScope("validation");
    const module = device.createShaderModule({
      label: `${plan.variantId}-module`,
      code: source,
    });
    const compilation = await module.getCompilationInfo();
    const shaderCompilationMessages = compilation.messages.map((message) =>
      Object.freeze({
        type: message.type,
        message: message.message,
        lineNum: message.lineNum,
        linePos: message.linePos,
      })
    );
    if (compilation.messages.some((message) => message.type === "error")) {
      throw new Error(`${plan.variantId} shader compilation reported an error`);
    }
    const pipeline = await device.createComputePipelineAsync({
      label: `${plan.variantId}-pipeline`,
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const scopedError = await device.popErrorScope();
    if (scopedError !== null) throw scopedError;
    const compileMilliseconds = performance.now() - compileStarted;
    const bindGroup = device.createBindGroup({
      label: `${plan.variantId}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: activation } },
        { binding: 1, resource: { buffer: weight } },
        { binding: 2, resource: { buffer: output } },
      ],
    });
    const counts = {
      encodedCommandBuffers: 0,
      submissions: 0,
      drains: 0,
      dispatches: 0,
    };
    return {
      plan,
      activation,
      weight,
      output,
      pipeline,
      bindGroup,
      shaderSha256,
      shaderCompilationMessages: Object.freeze(shaderCompilationMessages),
      preparation: Object.freeze({
        totalMilliseconds: performance.now() - started,
        compileMilliseconds,
        activationPacking: activationReceipt,
        weightPacking: weightReceipt,
        shaderBytes: new TextEncoder().encode(source).byteLength,
      }),
      counts,
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
        for (const buffer of owned) tracker.destroy(buffer);
      },
    };
  } catch (error) {
    void device.popErrorScope().catch(() => undefined);
    for (const buffer of owned) {
      if (buffer.mapState === "mapped") buffer.unmap();
      tracker.destroy(buffer);
    }
    throw error;
  }
}

async function runPreparedPair(
  device: GPUDevice,
  tracker: Opt0009BufferTracker,
  fp16: Opt0009PreparedVariant,
  fp32: Opt0009PreparedVariant,
  mode: Opt0009Mode,
  observeTimedSpan: Opt0009TimingSpanObserver,
): Promise<unknown> {
  const initialExecutions = new Map<AceOpt0009Precision, Opt0009ExecutionTiming>();
  const initialScans = new Map<AceOpt0009Precision, Opt0009OutputReadback>();
  const rerunScans = new Map<AceOpt0009Precision, Opt0009OutputReadback>();
  const postTimingScans = new Map<AceOpt0009Precision, Opt0009OutputReadback>();
  for (const variant of [fp16, fp32]) {
    initialExecutions.set(
      variant.plan.precision,
      await executeVariant(device, variant, -2, "correctness", 0),
    );
    initialScans.set(
      variant.plan.precision,
      await scanVariantOutput(device, tracker, variant),
    );
  }
  for (const variant of [fp16, fp32]) {
    await executeVariant(
      device,
      variant,
      -1,
      mode === "timing" ? "symmetric-warmup" : "deterministic-rerun",
      0,
    );
    rerunScans.set(
      variant.plan.precision,
      await scanVariantOutput(device, tracker, variant),
    );
  }
  for (const precision of ["fp16", "fp32"] as const) {
    const first = initialScans.get(precision)!;
    const second = rerunScans.get(precision)!;
    if (first.rawFnv1a32 !== second.rawFnv1a32) {
      throw new Error(`${fp16.plan.id} ${precision} rerun was nondeterministic`);
    }
  }
  const samples: Record<AceOpt0009Precision, Opt0009ExecutionTiming[]> = {
    fp16: [],
    fp32: [],
  };
  let timingHeartbeat: ReturnType<typeof startHeartbeat> | null = null;
  let heartbeat: unknown = null;
  if (mode === "timing") {
    timingHeartbeat = startHeartbeat();
    try {
      for (const [roundIndex, order] of ACE_OPT_0009_PAIRED_ORDERS.entries()) {
        for (const [orderPosition, precision] of order.entries()) {
          const variant = precision === "fp16" ? fp16 : fp32;
          const sample = await executeVariant(
            device,
            variant,
            roundIndex,
            order.join("-"),
            orderPosition,
          );
          samples[precision].push(sample);
          observeTimedSpan(
            sample.startedAtEpochMilliseconds,
            sample.completedAtEpochMilliseconds,
          );
          await yieldToBrowser();
        }
      }
    } finally {
      heartbeat = timingHeartbeat.stop();
    }
  }
  if (mode === "timing") {
    for (const variant of [fp16, fp32]) {
      const scan = await scanVariantOutput(device, tracker, variant);
      postTimingScans.set(variant.plan.precision, scan);
      if (
        scan.rawFnv1a32 !==
          initialScans.get(variant.plan.precision)!.rawFnv1a32
      ) {
        throw new Error(
          `${variant.plan.variantId} post-timing output was nondeterministic`,
        );
      }
    }
  }
  const expectedExecutions = mode === "timing"
    ? 2 + ACE_OPT_0009_PAIRED_ORDERS.length
    : 2;
  for (const variant of [fp16, fp32]) {
    validateAceOpt0009ExecutionCounts({
      plannedExecutions: expectedExecutions,
      ...variant.counts,
    });
  }
  return Object.freeze({
    id: fp16.plan.id,
    scope: fp16.plan.scope,
    kernelPath: fp16.plan.kernelPath,
    sourceClassification: fp16.plan.sourceClassification,
    shape: Object.freeze({
      rows: fp16.plan.rows,
      inner: fp16.plan.inner,
      columns: fp16.plan.columns,
    }),
    variants: Object.freeze({
      fp16: variantResult(
        fp16,
        initialExecutions.get("fp16")!,
        initialScans.get("fp16")!,
        rerunScans.get("fp16")!,
        postTimingScans.get("fp16") ?? null,
        samples.fp16,
      ),
      fp32: variantResult(
        fp32,
        initialExecutions.get("fp32")!,
        initialScans.get("fp32")!,
        rerunScans.get("fp32")!,
        postTimingScans.get("fp32") ?? null,
        samples.fp32,
      ),
    }),
    gpuSentinelComparison: Object.freeze({
      initial: compareGpuSentinels(
        initialScans.get("fp16")!,
        initialScans.get("fp32")!,
      ),
      deterministicRerun: compareGpuSentinels(
        rerunScans.get("fp16")!,
        rerunScans.get("fp32")!,
      ),
      postTiming: mode === "timing"
        ? compareGpuSentinels(
          postTimingScans.get("fp16")!,
          postTimingScans.get("fp32")!,
        )
        : null,
      acceptanceThresholdApplied: false,
    }),
    ...(mode === "timing"
      ? {
          pairedDelta: Object.freeze({
            medianFencedWallSpeedup:
              summarizeSamples(samples.fp32.map((sample) => sample.wallMilliseconds)).median /
              summarizeSamples(samples.fp16.map((sample) => sample.wallMilliseconds)).median,
            fp16RoundWins: ACE_OPT_0009_PAIRED_ORDERS.filter((_order, index) =>
              samples.fp16[index]!.wallMilliseconds <
                samples.fp32[index]!.wallMilliseconds
            ).length,
          }),
          timingHeartbeat: heartbeat,
        }
      : {}),
  });
}

function variantResult(
  variant: Opt0009PreparedVariant,
  correctnessExecution: Opt0009ExecutionTiming,
  initialScan: Opt0009OutputReadback,
  rerunScan: Opt0009OutputReadback,
  postTimingScan: Opt0009OutputReadback | null,
  samples: readonly Opt0009ExecutionTiming[],
): Readonly<Record<string, unknown>> {
  const wallSamples = samples.map((sample) => sample.wallMilliseconds);
  const tflopsSamples = samples.map((sample) => sample.validLogicalTflops);
  return Object.freeze({
    plan: variant.plan,
    shaderSha256: variant.shaderSha256,
    shaderCompilationMessages: variant.shaderCompilationMessages,
    preparation: variant.preparation,
    correctnessExecution,
    initialScan,
    deterministicRerunScan: rerunScan,
    postTimingScan,
    postTimingRawFingerprintValidated: postTimingScan === null
      ? null
      : initialScan.rawFnv1a32 === postTimingScan.rawFnv1a32,
    deterministicRawFingerprint: initialScan.rawFnv1a32 === rerunScan.rawFnv1a32,
    samples: Object.freeze([...samples]),
    ...(samples.length === 0
      ? {}
      : {
          summary: Object.freeze({
            fencedWallMilliseconds: summarizeSamples(wallSamples),
            validLogicalTflops: summarizeSamples(tflopsSamples),
          }),
        }),
    executionCountsBeforeCancellation: Object.freeze({ ...variant.counts }),
  });
}

async function executeVariant(
  device: GPUDevice,
  variant: Opt0009PreparedVariant,
  roundIndex: number,
  pairedOrder: string,
  orderPosition: number,
): Promise<Opt0009ExecutionTiming> {
  const startedAtEpochMilliseconds = epochMillisecondsNow();
  const wallStarted = performance.now();
  const encodeStarted = performance.now();
  const encoder = device.createCommandEncoder({
    label: `${variant.plan.variantId}-${pairedOrder}-encoder`,
  });
  const pass = encoder.beginComputePass({
    label: `${variant.plan.variantId}-${pairedOrder}-pass`,
  });
  pass.setPipeline(variant.pipeline);
  pass.setBindGroup(0, variant.bindGroup);
  pass.dispatchWorkgroups(...variant.plan.workgroups);
  variant.counts.dispatches += 1;
  pass.end();
  const command = encoder.finish();
  variant.counts.encodedCommandBuffers += 1;
  const encodeMilliseconds = performance.now() - encodeStarted;
  const submitStarted = performance.now();
  device.queue.submit([command]);
  variant.counts.submissions += 1;
  const submitMilliseconds = performance.now() - submitStarted;
  const drainStarted = performance.now();
  await device.queue.onSubmittedWorkDone();
  variant.counts.drains += 1;
  const drainMilliseconds = performance.now() - drainStarted;
  const wallMilliseconds = performance.now() - wallStarted;
  const completedAtEpochMilliseconds = epochMillisecondsNow();
  return Object.freeze({
    roundIndex,
    pairedOrder,
    orderPosition,
    wallMilliseconds,
    encodeMilliseconds,
    submitMilliseconds,
    drainMilliseconds,
    validLogicalTflops: variant.plan.validFlops / wallMilliseconds / 1e9,
    scheduledLogicalTflops:
      variant.plan.scheduledFlops / wallMilliseconds / 1e9,
    startedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
  });
}

async function scanVariantOutput(
  device: GPUDevice,
  tracker: Opt0009BufferTracker,
  variant: Opt0009PreparedVariant,
): Promise<Opt0009OutputReadback> {
  const readback = tracker.create(device, {
    label: `${variant.plan.variantId}-readback`,
    size: variant.plan.outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder({
      label: `${variant.plan.variantId}-readback-encoder`,
    });
    encoder.copyBufferToBuffer(
      variant.output,
      0,
      readback,
      0,
      variant.plan.outputBytes,
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    const range = readback.getMappedRange();
    const elementCount = variant.plan.rows * variant.plan.columns;
    let finiteCount = 0;
    let nonFiniteCount = 0;
    let nonzeroCount = 0;
    let sentinelCount = 0;
    let minimum = Infinity;
    let maximum = -Infinity;
    let hash = 0x811c_9dc5;
    const sentinels = new Map<number, number>();
    const sentinelIndices = outputSentinelIndices(variant.plan);
    const sentinelSet = new Set(sentinelIndices);
    if (variant.plan.precision === "fp16") {
      const bits = new Uint16Array(range);
      for (let index = 0; index < bits.length; index += 1) {
        const raw = bits[index]!;
        const value = float16BitsToNumber(raw);
        if (raw === 0x7e00) sentinelCount += 1;
        if (Number.isFinite(value)) {
          finiteCount += 1;
          if (value !== 0) nonzeroCount += 1;
          minimum = Math.min(minimum, value);
          maximum = Math.max(maximum, value);
        } else {
          nonFiniteCount += 1;
        }
        hash = Math.imul(hash ^ raw, 0x0100_0193) >>> 0;
        if (sentinelSet.has(index)) sentinels.set(index, value);
      }
    } else {
      const bits = new Uint32Array(range);
      const values = new Float32Array(range);
      for (let index = 0; index < bits.length; index += 1) {
        const raw = bits[index]!;
        const value = values[index]!;
        if (raw === 0x7fc0_0000) sentinelCount += 1;
        if (Number.isFinite(value)) {
          finiteCount += 1;
          if (value !== 0) nonzeroCount += 1;
          minimum = Math.min(minimum, value);
          maximum = Math.max(maximum, value);
        } else {
          nonFiniteCount += 1;
        }
        hash = Math.imul(hash ^ raw, 0x0100_0193) >>> 0;
        if (sentinelSet.has(index)) sentinels.set(index, value);
      }
    }
    const overflowClassification = variant.plan.precision === "fp32"
      ? "not-applicable"
      : nonFiniteCount === 0
        ? "none"
        : "observed";
    const scan: AceOpt0009OutputScan = Object.freeze({
      elementCount,
      finiteCount,
      nonFiniteCount,
      nonzeroCount,
      sentinelCount,
      minimum: finiteCount === 0 ? null : minimum,
      maximum: finiteCount === 0 ? null : maximum,
      rawFnv1a32: hash.toString(16).padStart(8, "0"),
      overflowClassification,
    });
    validateAceOpt0009OutputScan(variant.plan.precision, elementCount, scan);
    const cpuSentinels = validateCpuSentinels(variant.plan, sentinels);
    return Object.freeze({
      ...scan,
      cpuSentinels,
      sentinelValues: Object.freeze(
        [...sentinels.entries()].map(([index, value]) =>
          Object.freeze({ index, value })
        ),
      ),
    });
  } finally {
    if (mapped) readback.unmap();
    tracker.destroy(readback);
  }
}

function validateCpuSentinels(
  plan: AceOpt0009VariantPlan,
  actual: ReadonlyMap<number, number>,
): Readonly<Record<string, unknown>> {
  const records: Readonly<Record<string, unknown>>[] = [];
  let nativeSquaredError = 0;
  let nativeMaximumAbsoluteError = 0;
  let nativeMaximumRelativeError = 0;
  let nativeFiniteComparisonCount = 0;
  let fp32SquaredError = 0;
  let fp32MaximumAbsoluteError = 0;
  let fp32MaximumRelativeError = 0;
  let fp32FiniteComparisonCount = 0;
  for (const index of outputSentinelIndices(plan)) {
    const row = Math.floor(index / plan.columns);
    const column = index % plan.columns;
    const nativeExpected = cpuExpectedOutput(plan, row, column, "native");
    const alwaysFp32Expected = cpuExpectedOutput(
      plan,
      row,
      column,
      "always-fp32",
    );
    const observed = actual.get(index);
    if (observed === undefined) throw new Error("OPT-0009 lost a CPU sentinel");
    const nativeFinite =
      Number.isFinite(observed) && Number.isFinite(nativeExpected);
    const nativeAbsoluteError = nativeFinite
      ? Math.abs(observed - nativeExpected)
      : null;
    const nativeRelativeError = nativeFinite
      ? Math.abs(observed - nativeExpected) /
        Math.max(Math.abs(nativeExpected), 1e-12)
      : null;
    const fp32Finite =
      Number.isFinite(observed) && Number.isFinite(alwaysFp32Expected);
    const alwaysFp32AbsoluteError = fp32Finite
      ? Math.abs(observed - alwaysFp32Expected)
      : null;
    const alwaysFp32RelativeError = fp32Finite
      ? Math.abs(observed - alwaysFp32Expected) /
        Math.max(Math.abs(alwaysFp32Expected), 1e-12)
      : null;
    if (nativeAbsoluteError !== null && nativeRelativeError !== null) {
      nativeFiniteComparisonCount += 1;
      nativeSquaredError += nativeAbsoluteError * nativeAbsoluteError;
      nativeMaximumAbsoluteError = Math.max(
        nativeMaximumAbsoluteError,
        nativeAbsoluteError,
      );
      nativeMaximumRelativeError = Math.max(
        nativeMaximumRelativeError,
        nativeRelativeError,
      );
      const tolerance = plan.precision === "fp16"
        ? Math.max(0.5, Math.abs(nativeExpected) * 0.02)
        : Math.max(0.001, Math.abs(nativeExpected) * 0.00002);
      if (nativeAbsoluteError > tolerance) {
        throw new Error(`${plan.variantId} CPU sentinel ${index} exceeds tolerance`);
      }
    } else if (plan.precision === "fp32") {
      throw new Error(`${plan.variantId} produced a non-finite CPU sentinel`);
    }
    if (alwaysFp32AbsoluteError !== null && alwaysFp32RelativeError !== null) {
      fp32FiniteComparisonCount += 1;
      fp32SquaredError += alwaysFp32AbsoluteError * alwaysFp32AbsoluteError;
      fp32MaximumAbsoluteError = Math.max(
        fp32MaximumAbsoluteError,
        alwaysFp32AbsoluteError,
      );
      fp32MaximumRelativeError = Math.max(
        fp32MaximumRelativeError,
        alwaysFp32RelativeError,
      );
    }
    records.push(Object.freeze({
      index,
      row,
      column,
      nativeExpected,
      alwaysFp32Expected,
      observed,
      nativeAbsoluteError,
      nativeRelativeError,
      alwaysFp32AbsoluteError,
      alwaysFp32RelativeError,
    }));
  }
  return Object.freeze({
    records: Object.freeze(records),
    nativeConformance: Object.freeze({
      finiteComparisonCount: nativeFiniteComparisonCount,
      maximumAbsoluteError: nativeMaximumAbsoluteError,
      maximumRelativeError: nativeMaximumRelativeError,
      rmsError: nativeFiniteComparisonCount === 0
        ? null
        : Math.sqrt(nativeSquaredError / nativeFiniteComparisonCount),
      thresholdApplied: true,
    }),
    alwaysFp32ReferenceError: Object.freeze({
      finiteComparisonCount: fp32FiniteComparisonCount,
      nonFiniteComparisonCount:
        outputSentinelIndices(plan).length - fp32FiniteComparisonCount,
      maximumAbsoluteError: fp32MaximumAbsoluteError,
      maximumRelativeError: fp32MaximumRelativeError,
      rmsError: fp32FiniteComparisonCount === 0
        ? null
        : Math.sqrt(fp32SquaredError / fp32FiniteComparisonCount),
      thresholdApplied: false,
    }),
  });
}

function cpuExpectedOutput(
  plan: AceOpt0009VariantPlan,
  row: number,
  column: number,
  accumulation: "native" | "always-fp32",
): number {
  let accumulator = 0;
  for (let inner = 0; inner < plan.inner; inner += 1) {
    const a = fixtureActivationValue(row * plan.inner + inner);
    const b = fixtureWeightValue(column * plan.inner + inner);
    accumulator = plan.precision === "fp16" && accumulation === "native"
      ? roundFloat16(accumulator + a * b)
      : Math.fround(accumulator + a * b);
  }
  return accumulator;
}

function compareGpuSentinels(
  fp16: Opt0009OutputReadback,
  fp32: Opt0009OutputReadback,
): Readonly<Record<string, unknown>> {
  const fp32ByIndex = new Map(
    fp32.sentinelValues.map((entry) => [entry.index, entry.value]),
  );
  if (
    fp16.sentinelValues.length !== fp32.sentinelValues.length ||
    fp16.sentinelValues.some((entry) => !fp32ByIndex.has(entry.index))
  ) {
    throw new Error("OPT-0009 FP16/FP32 GPU sentinel identities diverged");
  }
  const records: Readonly<Record<string, unknown>>[] = [];
  let finiteComparisonCount = 0;
  let squaredError = 0;
  let maximumAbsoluteError = 0;
  let maximumRelativeError = 0;
  for (const fp16Entry of fp16.sentinelValues) {
    const fp32Value = fp32ByIndex.get(fp16Entry.index)!;
    const finite = Number.isFinite(fp16Entry.value) && Number.isFinite(fp32Value);
    const absoluteError = finite
      ? Math.abs(fp16Entry.value - fp32Value)
      : null;
    const relativeError = finite
      ? Math.abs(fp16Entry.value - fp32Value) /
        Math.max(Math.abs(fp32Value), 1e-12)
      : null;
    if (absoluteError !== null && relativeError !== null) {
      finiteComparisonCount += 1;
      squaredError += absoluteError * absoluteError;
      maximumAbsoluteError = Math.max(maximumAbsoluteError, absoluteError);
      maximumRelativeError = Math.max(maximumRelativeError, relativeError);
    }
    records.push(Object.freeze({
      index: fp16Entry.index,
      fp16Value: fp16Entry.value,
      fp32Value,
      absoluteError,
      relativeError,
    }));
  }
  return Object.freeze({
    comparison: "native-fp16-gpu-vs-native-fp32-gpu",
    records: Object.freeze(records),
    finiteComparisonCount,
    nonFiniteComparisonCount:
      fp16.sentinelValues.length - finiteComparisonCount,
    maximumAbsoluteError,
    maximumRelativeError,
    rmsError: finiteComparisonCount === 0
      ? null
      : Math.sqrt(squaredError / finiteComparisonCount),
    thresholdApplied: false,
  });
}

function outputSentinelIndices(plan: AceOpt0009VariantPlan): readonly number[] {
  const middleRow = Math.floor(plan.rows / 2);
  const lastRow = plan.rows - 1;
  return Object.freeze([...new Set([
    0,
    Math.min(plan.columns - 1, 127),
    Math.min(plan.columns - 1, 128),
    Math.min(plan.columns - 1, 255),
    Math.min(plan.columns - 1, 256),
    plan.columns - 1,
    middleRow * plan.columns,
    middleRow * plan.columns + Math.floor(plan.columns / 2),
    middleRow * plan.columns + plan.columns - 1,
    lastRow * plan.columns,
    lastRow * plan.columns + plan.columns - 1,
  ])].sort((left, right) => left - right));
}

async function runPostTimingCancellation(
  device: GPUDevice,
  variant: Opt0009PreparedVariant,
): Promise<Readonly<Record<string, unknown>>> {
  const controller = new AbortController();
  const plannedDispatches = 3;
  let encoded = 0;
  let submissions = 0;
  let drains = 0;
  let skippedAfterAbort = 0;
  const before = { ...variant.counts };
  for (let index = 0; index < plannedDispatches; index += 1) {
    if (controller.signal.aborted) {
      skippedAfterAbort += 1;
      continue;
    }
    await executeVariant(device, variant, -3, "post-timing-cancellation", index);
    encoded += 1;
    submissions += 1;
    drains += 1;
    if (index === 0) controller.abort("cancel-after-first-drain");
  }
  await yieldToBrowser();
  if (
    !controller.signal.aborted ||
    encoded !== 1 ||
    submissions !== 1 ||
    drains !== 1 ||
    skippedAfterAbort !== plannedDispatches - 1 ||
    variant.counts.encodedCommandBuffers - before.encodedCommandBuffers !== 1 ||
    variant.counts.submissions - before.submissions !== 1 ||
    variant.counts.drains - before.drains !== 1 ||
    variant.counts.dispatches - before.dispatches !== 1
  ) {
    throw new Error("OPT-0009 post-timing cancellation did not stop encoding");
  }
  return Object.freeze({
    variantId: variant.plan.variantId,
    cancellationPoint: "after-first-completion-fence",
    plannedDispatches,
    encodedCommandBuffers: encoded,
    submissions,
    drains,
    skippedAfterAbort,
    noPostAbortEncoding: true,
  });
}

function fillActivation(
  buffer: GPUBuffer,
  plan: AceOpt0009VariantPlan,
): Readonly<Record<string, unknown>> {
  const elements = plan.rows * plan.inner;
  const range = buffer.getMappedRange();
  if (plan.precision === "fp16") {
    const values = new Uint16Array(range);
    for (let index = 0; index < elements; index += 1) {
      values[index] = numberToFloat16Bits(fixtureActivationValue(index));
    }
  } else {
    const values = new Float32Array(range);
    for (let index = 0; index < elements; index += 1) {
      values[index] = fixtureActivationValue(index);
    }
  }
  const probes = packingProbeIndices(elements, plan.inner).map((index) =>
    Object.freeze({
      logicalIndex: index,
      physicalIndex: index,
      raw: rawMappedScalar(range, plan.precision, index),
    })
  );
  return Object.freeze({
    layout: "row-major-m-by-k",
    scalarCount: elements,
    bijective: true,
    probes: Object.freeze(probes),
  });
}

function fillWeight(
  buffer: GPUBuffer,
  plan: AceOpt0009VariantPlan,
): Readonly<Record<string, unknown>> {
  const scalarCount = plan.inner * plan.columns;
  const range = buffer.getMappedRange();
  const fp16 = plan.precision === "fp16" ? new Uint16Array(range) : null;
  const fp32 = plan.precision === "fp32" ? new Float32Array(range) : null;
  for (let inner = 0; inner < plan.inner; inner += 1) {
    for (let column = 0; column < plan.columns; column += 1) {
      const logical = inner * plan.columns + column;
      const physical = aceOpt0009WeightScalarIndex(plan, logical);
      const value = fixtureWeightValue(column * plan.inner + inner);
      if (fp16 !== null) fp16[physical] = numberToFloat16Bits(value);
      else fp32![physical] = value;
    }
  }
  const probes = weightPackingProbeIndices(plan).map((logicalIndex) => {
    const physicalIndex = aceOpt0009WeightScalarIndex(plan, logicalIndex);
    return Object.freeze({
      logicalIndex,
      physicalIndex,
      raw: rawMappedScalar(range, plan.precision, physicalIndex),
    });
  });
  if (new Set(probes.map((probe) => probe.physicalIndex)).size !== probes.length) {
    throw new Error(`${plan.variantId} packing probes were not bijective`);
  }
  return Object.freeze({
    layout: plan.kernelPath === "parakeet-direct-tile-major"
      ? `n${plan.tileColumns}-k32-tile-major`
      : "k-by-output-row-major",
    scalarCount,
    bijectiveBoundaryProbes: true,
    probes: Object.freeze(probes),
  });
}

function packingProbeIndices(count: number, innerStride: number): readonly number[] {
  return [...new Set([
    0,
    Math.min(count - 1, innerStride - 1),
    Math.min(count - 1, innerStride),
    Math.min(count - 1, innerStride * 31),
    Math.min(count - 1, innerStride * 32),
    count - 1,
  ])].sort((left, right) => left - right);
}

function weightPackingProbeIndices(
  plan: AceOpt0009VariantPlan,
): readonly number[] {
  const count = plan.inner * plan.columns;
  const n = plan.tileColumns;
  return [...new Set([
    0,
    Math.min(count - 1, n - 1),
    Math.min(count - 1, n),
    plan.columns - 1,
    Math.min(count - 1, 31 * plan.columns + n - 1),
    Math.min(count - 1, 32 * plan.columns),
    count - 1,
  ])].sort((left, right) => left - right);
}

function rawMappedScalar(
  range: ArrayBuffer,
  precision: AceOpt0009Precision,
  index: number,
): string {
  const raw = precision === "fp16"
    ? new Uint16Array(range)[index]!
    : new Uint32Array(range)[index]!;
  return `0x${raw.toString(16).padStart(precision === "fp16" ? 4 : 8, "0")}`;
}

function fixtureActivationValue(index: number): number {
  return Math.fround((((index * 17 + 3) % 31) - 15) / 32);
}

function fixtureWeightValue(index: number): number {
  return Math.fround((((index * 13 + 7) % 29) - 14) / 64);
}

function summarizeSamples(samples: readonly number[]): Opt0009SampleSummary {
  if (samples.length === 0 || samples.some((value) => !Number.isFinite(value))) {
    throw new Error("OPT-0009 cannot summarize an empty/non-finite sample set");
  }
  const retained = [...samples];
  const sorted = [...retained].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
  return Object.freeze({
    count: retained.length,
    samples: Object.freeze(retained),
    minimum: sorted[0]!,
    median,
    maximum: sorted.at(-1)!,
    range: sorted.at(-1)! - sorted[0]!,
  });
}

function epochMillisecondsNow(): number {
  return performance.timeOrigin + performance.now();
}

function requireAdapterLimits(adapter: GPUAdapter, maximumBindingBytes: number): void {
  const limits = adapter.limits;
  if (
    limits.maxBufferSize < maximumBindingBytes ||
    limits.maxStorageBufferBindingSize < maximumBindingBytes ||
    limits.maxComputeInvocationsPerWorkgroup < 192 ||
    limits.maxComputeWorkgroupSizeX < 192 ||
    limits.maxComputeWorkgroupStorageSize < 16_384
  ) {
    throw new Error("Adapter cannot satisfy the OPT-0009 buffer/kernel contract");
  }
}

function adapterReceipt(adapter: GPUAdapter): Readonly<Record<string, unknown>> {
  return Object.freeze({
    vendor: adapter.info.vendor,
    architecture: adapter.info.architecture,
    device: adapter.info.device,
    description: adapter.info.description,
    isFallbackAdapter: adapter.info.isFallbackAdapter,
    subgroupMinSize: adapter.info.subgroupMinSize,
    subgroupMaxSize: adapter.info.subgroupMaxSize,
    features: Object.freeze([...adapter.features].sort()),
    limits: Object.freeze({
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup:
        adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupStorageSize:
        adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeWorkgroupsPerDimension:
        adapter.limits.maxComputeWorkgroupsPerDimension,
    }),
  });
}

function startHeartbeat(): { stop(): Readonly<Record<string, number>> } {
  let animationFrameCount = 0;
  let timerTickCount = 0;
  let maximumAnimationFrameGapMilliseconds = 0;
  let maximumTimerGapMilliseconds = 0;
  let lastAnimationFrame = performance.now();
  let lastTimer = lastAnimationFrame;
  let stopped = false;
  let frameId = 0;
  const frame = (now: number): void => {
    if (stopped) return;
    maximumAnimationFrameGapMilliseconds = Math.max(
      maximumAnimationFrameGapMilliseconds,
      now - lastAnimationFrame,
    );
    lastAnimationFrame = now;
    animationFrameCount += 1;
    frameId = requestAnimationFrame(frame);
  };
  frameId = requestAnimationFrame(frame);
  const timerId = window.setInterval(() => {
    const now = performance.now();
    maximumTimerGapMilliseconds = Math.max(
      maximumTimerGapMilliseconds,
      now - lastTimer,
    );
    lastTimer = now;
    timerTickCount += 1;
  }, 10);
  return {
    stop(): Readonly<Record<string, number>> {
      if (!stopped) {
        stopped = true;
        cancelAnimationFrame(frameId);
        clearInterval(timerId);
      }
      return Object.freeze({
        animationFrameCount,
        timerTickCount,
        maximumAnimationFrameGapMilliseconds,
        maximumTimerGapMilliseconds,
      });
    },
  };
}

function numberToFloat16Bits(value: number): number {
  FLOAT32_SCRATCH[0] = value;
  const bits = UINT32_SCRATCH[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7f_ffff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const normalized = mantissa | 0x80_0000;
    const shift = 14 - halfExponent;
    const truncated = normalized >>> shift;
    const remainder = normalized & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    return sign | (truncated + (
      remainder > halfway || (remainder === halfway && (truncated & 1) !== 0)
        ? 1
        : 0
    ));
  }
  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1) !== 0)) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      const nextExponent = halfExponent + 1;
      return nextExponent >= 0x1f ? sign | 0x7c00 : sign | (nextExponent << 10);
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

function float16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) {
    return mantissa === 0 ? (sign < 0 ? -0 : 0) : sign * mantissa * 2 ** -24;
  }
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

function roundFloat16(value: number): number {
  return float16BitsToNumber(numberToFloat16Bits(value));
}

const FLOAT32_SCRATCH = new Float32Array(1);
const UINT32_SCRATCH = new Uint32Array(FLOAT32_SCRATCH.buffer);

async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function requiredQueryNumber(parameters: URLSearchParams, name: string): number {
  const raw = parameters.get(name);
  const value = raw === null ? NaN : Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`Missing or invalid OPT-0009 ${name}`);
  }
  return value;
}

function requireButton(selector: string): HTMLButtonElement {
  const button = document.querySelector<HTMLButtonElement>(selector);
  if (button === null) throw new Error(`Missing OPT-0009 button ${selector}`);
  return button;
}

function updateBrowserProgress(message: string): void {
  const progress = document.querySelector<HTMLElement>("#progress");
  if (progress !== null) progress.textContent = message;
}

function finishBrowser(status: "passed" | "failed", result: unknown): void {
  document.body.dataset.status = status;
  updateBrowserProgress(status);
  const output = document.querySelector<HTMLElement>("#result");
  if (output !== null) output.textContent = JSON.stringify(result, null, 2);
}

function errorReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    name: error instanceof Error ? error.name : "Error",
    message: error instanceof Error ? error.message : String(error),
    ...(error instanceof Error && error.stack !== undefined
      ? { stack: error.stack }
      : {}),
  });
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
}
