/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../../src/model/manifest.js";
import {
  AceOpt0009DenseGemmKernel,
  type AceOpt0009DenseGemmDispatch,
} from "../../src/webgpu/kernels/dit-dense-fp16.js";
import {
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
  AceOpt0032DenseK4PartialsKernel,
  packAceOpt0032DenseWeightU16,
  type AceOpt0032DenseK4PartialsDispatch,
} from "../../src/webgpu/kernels/dit-dense-fp16-k4-partials.js";
import {
  ACE_OPT_0058_DENSE_INT8_DP4A_KERNEL_ID,
  ACE_OPT_0058_DENSE_INT8_DP4A_SCALE_LAYOUT,
  ACE_OPT_0058_DENSE_INT8_DP4A_WEIGHT_LAYOUT,
  ACE_OPT_0058_GROUP_SIZES,
  ACE_OPT_0058_PACKED_DOT_LANGUAGE_FEATURE,
  AceOpt0058DenseInt8Dp4aKernel,
  aceOpt0058PackedWeightWordIndex,
  aceOpt0058WeightScaleIndex,
  packAceOpt0058SignedI8x4,
  planAceOpt0058DenseInt8,
  quantizeAndPackAceOpt0058DenseWeight,
  unpackAceOpt0058DenseWeightI8,
  type AceOpt0058DenseInt8Dispatch,
  type AceOpt0058GroupSize,
} from "../../src/webgpu/kernels/dit-dense-int8-dp4a.js";
import type {
  AceGemmBufferBindings,
  AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";

declare global {
  interface Window {
    __ACE_OPT0058_READY__?: Readonly<Record<string, unknown>>;
    __ACE_OPT0058_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

type FixtureKind = "full" | "signed-zero" | "zero-group" |
  "cancellation" | "range" | "saturation" | "long-k";
type SimpleArm = "exact" | "dotK4";
type GroupArm = "g32" | "g64" | "g128";
type TimingArm = SimpleArm |
  "g32Prequantized" | "g32Complete" |
  "g64Prequantized" | "g64Complete" |
  "g128Prequantized" | "g128Complete";

export interface Opt0058CaseSpec {
  readonly id: string;
  readonly shape: AceGemmShape;
  readonly fixtureKind: FixtureKind;
  readonly ordinal: number;
  readonly productionMultiplicity?: 4 | 2 | 1;
}

export interface Opt0058ThermalGate {
  readonly command:
    "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly waitStartedAtEpochMilliseconds: number;
  readonly checkedAtEpochMilliseconds: number;
  readonly waitDurationMilliseconds: number;
  readonly checkCount: 1;
  readonly thermalLevel: 0;
  readonly launchDelayMilliseconds: number;
}

interface CandidateResources {
  readonly groupSize: AceOpt0058GroupSize;
  readonly packedActivation: GPUBuffer;
  readonly packedPrefill: GPUBuffer;
  readonly activationScale: GPUBuffer;
  readonly scalePrefill: GPUBuffer;
  readonly weight: GPUBuffer;
  readonly weightScale: GPUBuffer;
  readonly counters: GPUBuffer;
  readonly counterReadback: GPUBuffer;
  readonly scaleReadback: GPUBuffer;
  readonly packedReadback: GPUBuffer;
  readonly offline: Readonly<{
    packedWords: number;
    scaleElements: number;
    saturationCount: number;
    zeroGroupCount: number;
  }>;
}

interface CaseResources {
  readonly activation: GPUBuffer;
  readonly exactWeight: GPUBuffer;
  readonly dotK4Weight: GPUBuffer;
  readonly candidates: ReadonlyMap<AceOpt0058GroupSize, CandidateResources>;
  readonly buffers: readonly GPUBuffer[];
  readonly packing: Readonly<Record<string, unknown>>;
}

interface GuardedOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly prefill: GPUBuffer;
  readonly readback: GPUBuffer;
  readonly buffers: readonly GPUBuffer[];
  readonly outputElements: number;
  readonly columns: number;
  readonly outputBytes: number;
  readonly totalBytes: number;
}

interface OutputSnapshot {
  readonly words: Uint32Array<ArrayBuffer>;
  readonly nonFiniteCount: number;
  readonly nonzeroCount: number;
  readonly qNaNPrefillCount: number;
  readonly prefixCanaryIntact: boolean;
  readonly suffixCanaryIntact: boolean;
  readonly tailRowWritten: boolean;
}

interface CandidateSnapshot extends OutputSnapshot {
  readonly packedActivationWords: Uint32Array<ArrayBuffer>;
  readonly activationScaleWords: Uint32Array<ArrayBuffer>;
  readonly nonFiniteActivationCount: number;
  readonly activationSaturationCount: number;
  readonly activationZeroGroupCount: number;
  readonly zeroGroupNonzeroPayloadCount: number;
  readonly unwrittenPackedByteCount: number;
}

interface PreparedShape {
  readonly spec: Opt0058CaseSpec;
  readonly resources: CaseResources;
  readonly output: GuardedOutput;
  readonly exact: AceOpt0009DenseGemmDispatch;
  readonly dotK4: AceOpt0032DenseK4PartialsDispatch;
  readonly candidates: ReadonlyMap<
    AceOpt0058GroupSize,
    AceOpt0058DenseInt8Dispatch
  >;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly exactKernel: AceOpt0009DenseGemmKernel;
  readonly dotK4Kernel: AceOpt0032DenseK4PartialsKernel;
  readonly candidateKernel: AceOpt0058DenseInt8Dp4aKernel;
  readonly querySet: GPUQuerySet;
  readonly timestampResolve: GPUBuffer;
  readonly timestampReadback: GPUBuffer;
  readonly shapes: readonly PreparedShape[];
  readonly identity: Readonly<Record<string, unknown>>;
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly preparedAtEpochMilliseconds: number;
  readonly uncapturedErrors: string[];
  destroy(): Readonly<Record<string, unknown>>;
}

interface NumericalAccumulator {
  count: number;
  finiteCount: number;
  controlNonFiniteCount: number;
  candidateNonFiniteCount: number;
  differingU32Count: number;
  signedZeroDifferenceCount: number;
  classChangeCount: number;
  finiteToZeroCollapseCount: number;
  classChanges: Record<string, number>;
  controlSum: number;
  candidateSum: number;
  controlSquareSum: number;
  candidateSquareSum: number;
  crossSum: number;
  errorSum: number;
  absoluteErrorSum: number;
  errorSquareSum: number;
  relativeErrorSquareSum: number;
  maximumAbsoluteControl: number;
  maximumAbsoluteError: number;
  maximumRelativeError: number;
  firstDifference: Readonly<Record<string, unknown>> | null;
  worstDifference: Readonly<Record<string, unknown>> | null;
}

interface TimestampSample {
  readonly arm: TimingArm;
  readonly groupSize?: AceOpt0058GroupSize;
  readonly boundary: "exact" | "dot-k4" | "prequantized-ceiling" |
    "complete-dynamic-quantize-plus-gemm";
  readonly submitAtPerformanceMilliseconds: number;
  readonly fenceAtPerformanceMilliseconds: number;
  readonly fencedWallMilliseconds: number;
  readonly timestampBeginNanoseconds: string;
  readonly timestampEndNanoseconds: string;
  readonly gpuElapsedNanoseconds: number;
  readonly gpuMilliseconds: number;
  readonly commandBufferCount: 1;
  readonly queueDrainCount: 1;
  readonly dynamicQuantizationCount: 0 | 1;
}

const EXPERIMENT_ID = "OPT-0058" as const;
const RECEIPT_SCHEMA = "ace-opt-0058-dense-int8-dp4a-v1";
const ROWS = 2_250;
const FULL_OUTPUT_COUNT = 25_344_000;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_5858;
const TIMESTAMP_QUERY_BYTES = 16;
const REQUIRED_COMPLETE_SPEEDUP = 1.50;
const NRMSE_MAXIMUM = 0.01;
const SNR_DECIBELS_MINIMUM = 40;
const PEARSON_MINIMUM = 0.999;
const FINITE_TO_ZERO_RATE_MAXIMUM = 1 / 1_000_000;
const THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
const MINIMUM_WAIT_MILLISECONDS = 30_000;
const MAXIMUM_WAIT_MILLISECONDS = 60_000;
const MAXIMUM_CHECK_TO_LAUNCH_MILLISECONDS = 5_000;
const FINITE_HALF_MAGNITUDES = Object.freeze([
  0x2411, 0x28b5, 0x2d53, 0x31e7, 0x356b, 0x39ad,
] as const);
const TIMING_ARMS = Object.freeze([
  "exact",
  "dotK4",
  "g32Prequantized",
  "g32Complete",
  "g64Prequantized",
  "g64Complete",
  "g128Prequantized",
  "g128Complete",
] as const satisfies readonly TimingArm[]);
const TIMING_ROUNDS = Object.freeze([
  Object.freeze({
    shapeOrder: Object.freeze([0, 1, 2, 3]),
    armOrder: TIMING_ARMS,
  }),
  Object.freeze({
    shapeOrder: Object.freeze([3, 2, 1, 0]),
    armOrder: Object.freeze(TIMING_ARMS.slice().reverse()),
  }),
]);

if (typeof document !== "undefined" && document.querySelector("#run") !== null) {
  installBrowserGate();
}

export function buildOpt0058Cases(): Readonly<{
  full: readonly Opt0058CaseSpec[];
  adversarial: readonly Opt0058CaseSpec[];
}> {
  return Object.freeze({
    full: Object.freeze([
      fullSpec("h-h", 2_048, 2_048, 4, 0),
      fullSpec("h-1024", 2_048, 1_024, 2, 1),
      fullSpec("h-6144", 2_048, 6_144, 2, 2),
      fullSpec("6144-h", 6_144, 2_048, 1, 3),
    ]),
    adversarial: Object.freeze([
      caseSpec("signed-zero", 1, 2_048, 1_024, "signed-zero", 10),
      caseSpec("zero-group", 2, 2_048, 1_024, "zero-group", 11),
      caseSpec("cancellation", 8, 2_048, 1_024, "cancellation", 12),
      caseSpec("finite-range", 4, 2_048, 1_024, "range", 13),
      caseSpec("saturation", 2, 2_048, 1_024, "saturation", 14),
      caseSpec("long-k6144", 2, 6_144, 2_048, "long-k", 15),
    ]),
  });
}

function installBrowserGate(): void {
  const progress = requireElement<HTMLElement>("#progress");
  const fieldset = requireElement<HTMLFieldSetElement>("#thermal-gate");
  const run = requireElement<HTMLButtonElement>("#run");
  let prepared: PreparedGate | undefined;
  let running: PreparedGate | undefined;
  void prepareGate((message) => progress.textContent = message).then(
    (value) => {
      if (value.correctness["passed"] !== true) {
        const cleanup = value.destroy();
        finishPage("failed", Object.freeze({
          schema: RECEIPT_SCHEMA,
          experimentId: EXPERIMENT_ID,
          passed: false,
          identity: value.identity,
          correctness: value.correctness,
          cleanup,
          decision: "negative-stop-correctness-before-timing",
        }));
        return;
      }
      prepared = value;
      const ready = Object.freeze({
        schema: "ace-opt-0058-ready-v1",
        experimentId: EXPERIMENT_ID,
        status: "ready",
        preparedAtEpochMilliseconds: value.preparedAtEpochMilliseconds,
        identity: value.identity,
        correctness: value.correctness,
        timedWorkCompleted: false,
      });
      window.__ACE_OPT0058_READY__ = ready;
      document.body.dataset.status = "ready";
      progress.textContent =
        "READY — stock packed-dot compilation, four full outputs, six adversarial fixtures, and fail-closed probe passed; begin one fresh 30-second idle wait";
      fieldset.disabled = false;
      run.disabled = false;
    },
    (error: unknown) => finishPage("failed", failureReceipt(error)),
  );
  run.addEventListener("click", () => {
    if (prepared === undefined) return;
    const owned = prepared;
    prepared = undefined;
    running = owned;
    run.disabled = true;
    fieldset.disabled = true;
    const launched = Date.now();
    let thermal: Opt0058ThermalGate;
    try {
      thermal = parseOpt0058ThermalGate(
        fieldParameters("#thermal-gate"),
        owned.preparedAtEpochMilliseconds,
        launched,
      );
    } catch (error) {
      const cleanup = owned.destroy();
      running = undefined;
      finishPage("failed", Object.freeze({ ...failureReceipt(error), cleanup }));
      return;
    }
    document.body.dataset.status = "running";
    void runTiming(owned, thermal, progress).then(
      (receipt) => {
        running = undefined;
        finishPage(receipt["passed"] === true ? "passed" : "failed", receipt);
      },
      (error: unknown) => {
        const cleanup = owned.destroy();
        running = undefined;
        finishPage("failed", Object.freeze({ ...failureReceipt(error), cleanup }));
      },
    );
  }, { once: true });
  window.addEventListener("beforeunload", () => {
    prepared?.destroy();
    running?.destroy();
    prepared = undefined;
    running = undefined;
  });
}

async function prepareGate(
  update: (message: string) => void,
): Promise<PreparedGate> {
  requireLittleEndianHost();
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const wgslLanguageFeatures = getWgslLanguageFeatures();
  if (!wgslLanguageFeatures.includes(ACE_OPT_0058_PACKED_DOT_LANGUAGE_FEATURE)) {
    throw new Error(
      `OPT-0058 missing WGSL language feature ${ACE_OPT_0058_PACKED_DOT_LANGUAGE_FEATURE}`,
    );
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const maximumStorageBytes = maximumStorageBindingBytes();
  const device = await adapter.requestDevice({
    label: "ace-opt-0058-dense-int8-dp4a",
    requiredFeatures: ["shader-f16", "subgroups", "timestamp-query"],
    requiredLimits: {
      maxBufferSize: maximumStorageBytes + 2 * STORAGE_GUARD_BYTES,
      maxStorageBufferBindingSize: maximumStorageBytes,
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 128,
    },
  });
  const tracker = new BufferTracker();
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    errors.push(event.error.message);
  });
  const capability = Object.freeze({
    subgroupMinSize: 32 as const,
    subgroupMaxSize: 32 as const,
  });
  const exactKernel = AceOpt0009DenseGemmKernel.create(device, capability);
  const dotK4Kernel = AceOpt0032DenseK4PartialsKernel.create(device, capability);
  const candidateKernel = AceOpt0058DenseInt8Dp4aKernel.create(device, {
    ...capability,
    packed4x8IntegerDotProduct: true,
  });
  const querySet = device.createQuerySet({
    label: "opt0058-timestamp-query",
    type: "timestamp",
    count: 2,
  });
  const timestampResolve = tracker.create(device, {
    label: "opt0058-timestamp-resolve",
    size: TIMESTAMP_QUERY_BYTES,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const timestampReadback = tracker.create(device, {
    label: "opt0058-timestamp-readback",
    size: TIMESTAMP_QUERY_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let destroyed = false;
  const destroy = (): Readonly<Record<string, unknown>> => {
    if (destroyed) return Object.freeze({ ...tracker.receipt(), repeated: true });
    destroyed = true;
    exactKernel.destroy();
    dotK4Kernel.destroy();
    candidateKernel.destroy();
    querySet.destroy();
    tracker.destroyAll();
    device.destroy();
    return Object.freeze({
      ...tracker.receipt(),
      repeated: false,
      querySetDestroyed: true,
      deviceDestroyed: true,
    });
  };
  try {
    update("proving packed signed-byte index/inverse helpers");
    const layoutProof = buildRuntimeLayoutProof();
    const cases = buildOpt0058Cases();
    const aggregates = createAggregateSet();
    const shapes: PreparedShape[] = [];
    for (const [index, spec] of cases.full.entries()) {
      update(`full exact/K4/G32/G64/G128 ${index + 1}/4: ${spec.id}`);
      const shape = await prepareCase(
        device,
        tracker,
        exactKernel,
        dotK4Kernel,
        candidateKernel,
        spec,
        aggregates.full,
        true,
      );
      shapes.push(shape);
      await browserYield();
    }
    if (aggregates.full.g32.count !== FULL_OUTPUT_COUNT ||
      aggregates.full.g64.count !== FULL_OUTPUT_COUNT ||
      aggregates.full.g128.count !== FULL_OUTPUT_COUNT) {
      throw new Error("OPT-0058 full output count changed");
    }
    const adversarialReceipts: Readonly<Record<string, unknown>>[] = [];
    for (const [index, spec] of cases.adversarial.entries()) {
      update(`adversarial exact/K4/G32/G64/G128 ${index + 1}/6: ${spec.id}`);
      const temporary = await prepareCase(
        device,
        tracker,
        exactKernel,
        dotK4Kernel,
        candidateKernel,
        spec,
        aggregates.adversarial,
        false,
      );
      adversarialReceipts.push(temporary.correctness);
      destroyCase(tracker, temporary);
      await browserYield();
    }
    update("proving NaN/Inf activation fail-closed behavior");
    const failClosed = await runNonFiniteFailClosed(
      device,
      tracker,
      candidateKernel,
    );
    const fullNumerics = finalizeAggregateSet(aggregates.full, "full");
    const adversarialNumerics = finalizeAggregateSet(
      aggregates.adversarial,
      "adversarial",
    );
    const fullReceipts = shapes.map(({ correctness }) => correctness);
    const everyCaseReceipt = [...fullReceipts, ...adversarialReceipts];
    const groupEligibility = Object.freeze(Object.fromEntries(
      ACE_OPT_0058_GROUP_SIZES.map((groupSize) => {
        const arm = groupArm(groupSize);
        const eligible = fullNumerics[arm]["passed"] === true &&
          adversarialNumerics[arm]["passed"] === true &&
          everyCaseReceipt.every((receipt) =>
            caseGroupPassed(receipt, arm)
          );
        return [arm, Object.freeze({
          groupSize,
          everyIndividualCasePassed: everyCaseReceipt.every((receipt) =>
            caseGroupPassed(receipt, arm)
          ),
          eligible,
        })];
      }),
    ));
    const passed = Object.values(groupEligibility).some(
      (entry) => entry.eligible,
    ) && fullReceipts.every((receipt) => receipt["passed"] === true) &&
      adversarialReceipts.every((receipt) => receipt["passed"] === true) &&
      failClosed["passed"] === true;
    if (passed) {
      update("one symmetric untimed warmup per timing arm and shape");
      for (const [shapeIndex, shape] of shapes.entries()) {
        const order = shapeIndex % 2 === 0
          ? TIMING_ARMS
          : Object.freeze(TIMING_ARMS.slice().reverse());
        for (const arm of order) {
          await executeTimingArmAndDrain(device, shape, arm);
        }
      }
    }
    await device.queue.onSubmittedWorkDone();
    await browserYield();
    if (errors.length !== 0) {
      throw new Error(`OPT-0058 uncaptured GPU errors: ${errors.join("; ")}`);
    }
    const identity = buildIdentity(
      adapter,
      device,
      wgslLanguageFeatures,
      layoutProof,
    );
    const correctness = Object.freeze({
      passed,
      completedBeforeReady: true,
      successfulStockWgslCompilationBeforeReady: true,
      fullOutputCountPerArm: FULL_OUTPUT_COUNT,
      fullCases: Object.freeze(fullReceipts),
      fullNumerics,
      adversarialCases: Object.freeze(adversarialReceipts),
      adversarialNumerics,
      groupEligibility,
      nonFiniteActivationFailClosed: failClosed,
      uncapturedGpuErrorCount: errors.length,
    });
    return Object.freeze({
      adapter,
      device,
      tracker,
      exactKernel,
      dotK4Kernel,
      candidateKernel,
      querySet,
      timestampResolve,
      timestampReadback,
      shapes: Object.freeze(shapes),
      identity,
      correctness,
      preparedAtEpochMilliseconds: Date.now(),
      uncapturedErrors: errors,
      destroy,
    });
  } catch (error) {
    await device.queue.onSubmittedWorkDone().catch(() => undefined);
    destroy();
    throw error;
  }
}

async function prepareCase(
  device: GPUDevice,
  tracker: BufferTracker,
  exactKernel: AceOpt0009DenseGemmKernel,
  dotK4Kernel: AceOpt0032DenseK4PartialsKernel,
  candidateKernel: AceOpt0058DenseInt8Dp4aKernel,
  spec: Opt0058CaseSpec,
  aggregates: Readonly<Record<GroupArm | "dotK4", NumericalAccumulator>>,
  full: boolean,
): Promise<PreparedShape> {
  const resources = createCaseResources(device, tracker, spec);
  const output = createGuardedOutput(device, tracker, spec);
  try {
    const dispatches = await createDispatches(
      exactKernel,
      dotK4Kernel,
      candidateKernel,
      spec,
      resources,
      output.binding,
    );
    const correctness = await verifyCase(
      device,
      spec,
      resources,
      output,
      dispatches,
      aggregates,
      full,
    );
    return Object.freeze({
      spec,
      resources,
      output,
      exact: dispatches.exact,
      dotK4: dispatches.dotK4,
      candidates: dispatches.candidates,
      correctness,
    });
  } catch (error) {
    for (const buffer of resources.buffers) tracker.destroy(buffer);
    for (const buffer of output.buffers) tracker.destroy(buffer);
    throw error;
  }
}

function createCaseResources(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0058CaseSpec,
): CaseResources {
  const activationElements = spec.shape.rows * spec.shape.inner;
  const weightElements = spec.shape.inner * spec.shape.columns;
  const buffers: GPUBuffer[] = [];
  const make = (descriptor: GPUBufferDescriptor): GPUBuffer => {
    const buffer = tracker.create(device, descriptor);
    buffers.push(buffer);
    return buffer;
  };
  const activation = make({
    label: `opt0058-${spec.id}-activation`,
    size: activationElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  const exactWeight = make({
    label: `opt0058-${spec.id}-exact-weight`,
    size: weightElements * 2,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const dotK4Weight = make({
    label: `opt0058-${spec.id}-dot-k4-weight`,
    size: weightElements * 2,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    const activationValues = new Float32Array(activation.getMappedRange());
    fillActivation(activationValues, spec);
    activation.unmap();

    const logicalBits = new Uint16Array(weightElements);
    fillLogicalWeight(logicalBits, spec);
    new Uint16Array(exactWeight.getMappedRange()).set(
      packExactWeight(logicalBits, spec.shape.inner, spec.shape.columns),
    );
    exactWeight.unmap();
    new Uint16Array(dotK4Weight.getMappedRange()).set(
      packAceOpt0032DenseWeightU16(
        logicalBits,
        spec.shape.inner,
        spec.shape.columns,
      ),
    );
    dotK4Weight.unmap();

    const logicalValues = Float32Array.from(logicalBits, halfToNumber);
    const candidates = new Map<AceOpt0058GroupSize, CandidateResources>();
    const packingByGroup: Record<string, unknown> = {};
    for (const groupSize of ACE_OPT_0058_GROUP_SIZES) {
      const plan = planAceOpt0058DenseInt8(spec.shape, groupSize);
      const offline = quantizeAndPackAceOpt0058DenseWeight(
        logicalValues,
        spec.shape.inner,
        spec.shape.columns,
        groupSize,
      );
      const packedActivation = make({
        label: `opt0058-${spec.id}-g${groupSize}-packed-activation`,
        size: plan.packedActivationWords * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
      const packedPrefill = make({
        label: `opt0058-${spec.id}-g${groupSize}-packed-prefill`,
        size: plan.packedActivationWords * 4,
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      new Uint32Array(packedPrefill.getMappedRange()).fill(0x8080_8080);
      packedPrefill.unmap();
      const activationScale = make({
        label: `opt0058-${spec.id}-g${groupSize}-activation-scale`,
        size: plan.activationScaleElements * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
      const scalePrefill = make({
        label: `opt0058-${spec.id}-g${groupSize}-scale-prefill`,
        size: plan.activationScaleElements * 4,
        usage: GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      new Uint32Array(scalePrefill.getMappedRange()).fill(0x7fc0_5858);
      scalePrefill.unmap();
      const weight = make({
        label: `opt0058-${spec.id}-g${groupSize}-weight`,
        size: plan.packedWeightWords * 4,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
      });
      new Uint32Array(weight.getMappedRange()).set(offline.packed);
      weight.unmap();
      const weightScale = make({
        label: `opt0058-${spec.id}-g${groupSize}-weight-scale`,
        size: plan.weightScaleElements * 4,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
      });
      new Float32Array(weightScale.getMappedRange()).set(offline.scales);
      weightScale.unmap();
      const counters = make({
        label: `opt0058-${spec.id}-g${groupSize}-counters`,
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
      const counterReadback = make({
        label: `opt0058-${spec.id}-g${groupSize}-counter-readback`,
        size: 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const scaleReadback = make({
        label: `opt0058-${spec.id}-g${groupSize}-scale-readback`,
        size: plan.activationScaleElements * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const packedReadback = make({
        label: `opt0058-${spec.id}-g${groupSize}-packed-readback`,
        size: plan.packedActivationWords * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      candidates.set(groupSize, Object.freeze({
        groupSize,
        packedActivation,
        packedPrefill,
        activationScale,
        scalePrefill,
        weight,
        weightScale,
        counters,
        counterReadback,
        scaleReadback,
        packedReadback,
        offline: Object.freeze({
          packedWords: offline.packed.length,
          scaleElements: offline.scales.length,
          saturationCount: offline.saturationCount,
          zeroGroupCount: offline.zeroGroupCount,
        }),
      }));
      packingByGroup[`g${groupSize}`] = Object.freeze({
        packedWeightWords: offline.packed.length,
        weightScaleElements: offline.scales.length,
        weightSaturationCount: offline.saturationCount,
        weightZeroGroupCount: offline.zeroGroupCount,
        bytesPerLogicalWeight: 1,
        f32ScaleBytes: offline.scales.byteLength,
      });
    }
    return Object.freeze({
      activation,
      exactWeight,
      dotK4Weight,
      candidates,
      buffers: Object.freeze(buffers.slice()),
      packing: Object.freeze({
        logicalWeightElements: logicalBits.length,
        exactFp16WeightBytes: logicalBits.byteLength,
        integrationProjectionRequiresConverterNativeReplaceNotDuplicate: true,
        benchmarkComparisonDuplicatesExactK4AndInt8Arms: true,
        groups: Object.freeze(packingByGroup),
      }),
    });
  } catch (error) {
    for (const buffer of buffers) {
      if (buffer.mapState === "mapped") buffer.unmap();
      tracker.destroy(buffer);
    }
    throw error;
  }
}

async function createDispatches(
  exactKernel: AceOpt0009DenseGemmKernel,
  dotK4Kernel: AceOpt0032DenseK4PartialsKernel,
  candidateKernel: AceOpt0058DenseInt8Dp4aKernel,
  spec: Opt0058CaseSpec,
  resources: CaseResources,
  output: GPUBufferBinding,
): Promise<Readonly<{
  exact: AceOpt0009DenseGemmDispatch;
  dotK4: AceOpt0032DenseK4PartialsDispatch;
  candidates: ReadonlyMap<AceOpt0058GroupSize, AceOpt0058DenseInt8Dispatch>;
}>> {
  const activationBytes = spec.shape.rows * spec.shape.inner * 4;
  const weightBytes = spec.shape.inner * spec.shape.columns * 2;
  const activation = binding(resources.activation, activationBytes);
  const exactBindings: AceGemmBufferBindings = Object.freeze({
    activation,
    weight: binding(resources.exactWeight, weightBytes),
    output,
  });
  const dotK4Bindings: AceGemmBufferBindings = Object.freeze({
    activation,
    weight: binding(resources.dotK4Weight, weightBytes),
    output,
  });
  const exact = await exactKernel.createDispatch(
    `opt0058-${spec.id}-exact`,
    spec.shape,
    exactBindings,
  );
  const dotK4 = await dotK4Kernel.createDispatch(
    `opt0058-${spec.id}-dot-k4`,
    spec.shape,
    dotK4Bindings,
  );
  if (exact.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT ||
    dotK4.weightLayout !== ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT) {
    throw new Error(`OPT-0058 ${spec.id} exact/K4 layouts changed`);
  }
  const candidates = new Map<
    AceOpt0058GroupSize,
    AceOpt0058DenseInt8Dispatch
  >();
  for (const groupSize of ACE_OPT_0058_GROUP_SIZES) {
    const resource = requiredCandidate(resources, groupSize);
    const plan = planAceOpt0058DenseInt8(spec.shape, groupSize);
    const dispatch = await candidateKernel.createDispatch(
      `opt0058-${spec.id}-g${groupSize}`,
      spec.shape,
      groupSize,
      Object.freeze({
        activation,
        packedActivation: binding(
          resource.packedActivation,
          plan.packedActivationWords * 4,
        ),
        activationScale: binding(
          resource.activationScale,
          plan.activationScaleElements * 4,
        ),
        weight: binding(resource.weight, plan.packedWeightWords * 4),
        weightScale: binding(
          resource.weightScale,
          plan.weightScaleElements * 4,
        ),
        quantizationStatus: binding(resource.counters, 4),
        output,
      }),
    );
    if (dispatch.kernelId !== ACE_OPT_0058_DENSE_INT8_DP4A_KERNEL_ID ||
      dispatch.weightLayout !== ACE_OPT_0058_DENSE_INT8_DP4A_WEIGHT_LAYOUT ||
      dispatch.scaleLayout !== ACE_OPT_0058_DENSE_INT8_DP4A_SCALE_LAYOUT) {
      throw new Error(`OPT-0058 ${spec.id} G${groupSize} identity changed`);
    }
    candidates.set(groupSize, dispatch);
  }
  return Object.freeze({ exact, dotK4, candidates });
}

async function verifyCase(
  device: GPUDevice,
  spec: Opt0058CaseSpec,
  resources: CaseResources,
  output: GuardedOutput,
  dispatches: Readonly<{
    exact: AceOpt0009DenseGemmDispatch;
    dotK4: AceOpt0032DenseK4PartialsDispatch;
    candidates: ReadonlyMap<AceOpt0058GroupSize, AceOpt0058DenseInt8Dispatch>;
  }>,
  aggregates: Readonly<Record<GroupArm | "dotK4", NumericalAccumulator>>,
  full: boolean,
): Promise<Readonly<Record<string, unknown>>> {
  const exact = await executeSimpleSnapshot(
    device,
    output,
    dispatches.exact,
    `${spec.id}-exact`,
  );
  requireCompleteSnapshot(exact, `${spec.id} exact`, full);
  const dotK4 = await executeSimpleSnapshot(
    device,
    output,
    dispatches.dotK4,
    `${spec.id}-dot-k4`,
  );
  requireCompleteSnapshot(dotK4, `${spec.id} dot-K4`, full);
  const dotLocal = createAccumulator();
  accumulateNumerics(dotLocal, exact.words, dotK4.words, spec);
  mergeAccumulator(aggregates.dotK4, dotLocal);
  const groupReceipts: Record<string, unknown> = {};
  for (const groupSize of ACE_OPT_0058_GROUP_SIZES) {
    const arm = groupArm(groupSize);
    const resource = requiredCandidate(resources, groupSize);
    const dispatch = requiredDispatch(dispatches.candidates, groupSize);
    const candidate = await executeCandidateSnapshot(
      device,
      output,
      resource,
      dispatch,
      `${spec.id}-g${groupSize}`,
    );
    requireCompleteSnapshot(candidate, `${spec.id} G${groupSize}`, full);
    if (candidate.nonFiniteActivationCount !== 0) {
      throw new Error(`OPT-0058 ${spec.id} unexpectedly rejected finite activation`);
    }
    if (candidate.zeroGroupNonzeroPayloadCount !== 0) {
      throw new Error(`OPT-0058 ${spec.id} G${groupSize} wrote a noncanonical zero group`);
    }
    if (candidate.unwrittenPackedByteCount !== 0) {
      throw new Error(`OPT-0058 ${spec.id} G${groupSize} left packed bytes unwritten`);
    }
    const local = createAccumulator();
    accumulateNumerics(local, exact.words, candidate.words, spec);
    mergeAccumulator(aggregates[arm], local);
    const rerun = await executeCandidateSnapshot(
      device,
      output,
      resource,
      dispatch,
      `${spec.id}-g${groupSize}-rerun`,
    );
    requireCompleteSnapshot(rerun, `${spec.id} G${groupSize} rerun`, full);
    requireExactWords(
      candidate.words,
      rerun.words,
      `${spec.id} G${groupSize} deterministic rerun`,
    );
    requireExactWords(
      candidate.packedActivationWords,
      rerun.packedActivationWords,
      `${spec.id} G${groupSize} packed activation rerun`,
    );
    requireExactWords(
      candidate.activationScaleWords,
      rerun.activationScaleWords,
      `${spec.id} G${groupSize} activation scale rerun`,
    );
    if (
      candidate.activationSaturationCount !== rerun.activationSaturationCount ||
      candidate.activationZeroGroupCount !== rerun.activationZeroGroupCount ||
      rerun.zeroGroupNonzeroPayloadCount !== 0 ||
      rerun.unwrittenPackedByteCount !== 0
    ) {
      throw new Error(`OPT-0058 ${spec.id} G${groupSize} quantizer changed on rerun`);
    }
    groupReceipts[arm] = Object.freeze({
      groupSize,
      outputU32Count: output.outputElements,
      candidateDeterministicRawU32: true,
      dynamicQuantizerDeterministicRawU32: true,
      activationSaturationCount: candidate.activationSaturationCount,
      activationZeroGroupCount: candidate.activationZeroGroupCount,
      zeroGroupNonzeroPayloadCount: candidate.zeroGroupNonzeroPayloadCount,
      unwrittenPackedByteCount: candidate.unwrittenPackedByteCount,
      weightSaturationCount: resource.offline.saturationCount,
      weightZeroGroupCount: resource.offline.zeroGroupCount,
      numerics: finalizeNumerics(local),
      completeWritesFiniteGuardsAndTail: true,
    });
  }
  return Object.freeze({
    id: spec.id,
    fixtureKind: spec.fixtureKind,
    shape: spec.shape,
    outputU32Count: output.outputElements,
    packing: resources.packing,
    executionOrder: Object.freeze([
      "exact", "dotK4",
      "g32Complete", "g32Complete",
      "g64Complete", "g64Complete",
      "g128Complete", "g128Complete",
    ]),
    exactAndDotK4CompleteWritesFiniteGuardsAndTail: true,
    dotK4Numerics: finalizeNumerics(dotLocal),
    groups: Object.freeze(groupReceipts),
    passed: Object.values(groupReceipts).some((entry) =>
      (entry as Readonly<Record<string, unknown>>)["numerics"] !== undefined &&
      ((entry as Readonly<Record<string, unknown>>)["numerics"] as
        Readonly<Record<string, unknown>>)["passed"] === true
    ),
  });
}

async function executeSimpleSnapshot(
  device: GPUDevice,
  output: GuardedOutput,
  dispatch: Pick<
    AceOpt0009DenseGemmDispatch | AceOpt0032DenseK4PartialsDispatch,
    "encode"
  >,
  label: string,
): Promise<OutputSnapshot> {
  const encoder = device.createCommandEncoder({ label: `${label}-correctness` });
  encoder.copyBufferToBuffer(output.prefill, 0, output.buffer, 0, output.totalBytes);
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  encoder.copyBufferToBuffer(output.buffer, 0, output.readback, 0, output.totalBytes);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return await readOutputSnapshot(output);
}

async function executeCandidateSnapshot(
  device: GPUDevice,
  output: GuardedOutput,
  resources: CandidateResources,
  dispatch: AceOpt0058DenseInt8Dispatch,
  label: string,
): Promise<CandidateSnapshot> {
  const scaleBytes = dispatch.plan.activationScaleElements * 4;
  const packedBytes = dispatch.plan.packedActivationWords * 4;
  const encoder = device.createCommandEncoder({ label: `${label}-correctness` });
  encoder.clearBuffer(resources.counters, 0, 4);
  encoder.copyBufferToBuffer(
    resources.packedPrefill,
    0,
    resources.packedActivation,
    0,
    packedBytes,
  );
  encoder.copyBufferToBuffer(
    resources.scalePrefill,
    0,
    resources.activationScale,
    0,
    scaleBytes,
  );
  encoder.copyBufferToBuffer(output.prefill, 0, output.buffer, 0, output.totalBytes);
  const pass = encoder.beginComputePass();
  dispatch.encodeComplete(pass);
  pass.end();
  encoder.copyBufferToBuffer(output.buffer, 0, output.readback, 0, output.totalBytes);
  encoder.copyBufferToBuffer(resources.counters, 0, resources.counterReadback, 0, 4);
  encoder.copyBufferToBuffer(
    resources.activationScale,
    0,
    resources.scaleReadback,
    0,
    scaleBytes,
  );
  encoder.copyBufferToBuffer(
    resources.packedActivation,
    0,
    resources.packedReadback,
    0,
    packedBytes,
  );
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const outputSnapshot = await readOutputSnapshot(output);
  await Promise.all([
    resources.counterReadback.mapAsync(GPUMapMode.READ),
    resources.scaleReadback.mapAsync(GPUMapMode.READ),
    resources.packedReadback.mapAsync(GPUMapMode.READ),
  ]);
  try {
    const counterRange = resources.counterReadback.getMappedRange();
    const scaleRange = resources.scaleReadback.getMappedRange();
    const packedRange = resources.packedReadback.getMappedRange();
    const counters = new Uint32Array(counterRange);
    const scales = new Float32Array(scaleRange);
    const packed = new Uint8Array(packedRange);
    const packedActivationWords = new Uint32Array(packedRange).slice();
    const activationScaleWords = new Uint32Array(scaleRange).slice();
    let activationZeroGroupCount = 0;
    let activationSaturationCount = 0;
    let zeroGroupNonzeroPayloadCount = 0;
    let unwrittenPackedByteCount = 0;
    for (let scaleIndex = 0; scaleIndex < scales.length; scaleIndex += 1) {
      const scale = scales[scaleIndex]!;
      if (scale === 0) activationZeroGroupCount += 1;
      if (!Number.isFinite(scale) || scale < 0) {
        throw new Error(`OPT-0058 ${label} emitted invalid activation scale`);
      }
      const byteBase = scaleIndex * dispatch.groupSize;
      for (let offset = 0; offset < dispatch.groupSize; offset += 1) {
        const byte = packed[byteBase + offset]!;
        const signed = (byte << 24) >> 24;
        if (Math.abs(signed) === 127) activationSaturationCount += 1;
        if (scale === 0 && byte !== 0) zeroGroupNonzeroPayloadCount += 1;
        if (byte === 0x80) unwrittenPackedByteCount += 1;
      }
    }
    return Object.freeze({
      ...outputSnapshot,
      packedActivationWords,
      activationScaleWords,
      nonFiniteActivationCount: counters[0]!,
      activationSaturationCount,
      activationZeroGroupCount,
      zeroGroupNonzeroPayloadCount,
      unwrittenPackedByteCount,
    });
  } finally {
    resources.counterReadback.unmap();
    resources.scaleReadback.unmap();
    resources.packedReadback.unmap();
  }
}

async function readOutputSnapshot(
  output: GuardedOutput,
): Promise<OutputSnapshot> {
  await output.readback.mapAsync(GPUMapMode.READ);
  try {
    const all = new Uint32Array(output.readback.getMappedRange());
    const guardWords = STORAGE_GUARD_BYTES / 4;
    const outputEnd = guardWords + output.outputElements;
    let prefixCanaryIntact = true;
    let suffixCanaryIntact = true;
    for (let index = 0; index < guardWords; index += 1) {
      prefixCanaryIntact &&= all[index] === STORAGE_GUARD_U32;
      suffixCanaryIntact &&= all[outputEnd + index] === STORAGE_GUARD_U32;
    }
    const words = all.slice(guardWords, outputEnd);
    let nonFiniteCount = 0;
    let nonzeroCount = 0;
    let qNaNPrefillCount = 0;
    for (const word of words) {
      if ((word & 0x7f80_0000) === 0x7f80_0000) nonFiniteCount += 1;
      if ((word & 0x7fff_ffff) !== 0) nonzeroCount += 1;
      if (word === OUTPUT_PREFILL_QNAN_U32) qNaNPrefillCount += 1;
    }
    let tailRowWritten = true;
    for (let index = words.length - output.columns; index < words.length; index += 1) {
      if (words[index] === OUTPUT_PREFILL_QNAN_U32) tailRowWritten = false;
    }
    return Object.freeze({
      words,
      nonFiniteCount,
      nonzeroCount,
      qNaNPrefillCount,
      prefixCanaryIntact,
      suffixCanaryIntact,
      tailRowWritten,
    });
  } finally {
    output.readback.unmap();
  }
}

async function runNonFiniteFailClosed(
  device: GPUDevice,
  tracker: BufferTracker,
  kernel: AceOpt0058DenseInt8Dp4aKernel,
): Promise<Readonly<Record<string, unknown>>> {
  const spec = caseSpec("non-finite-fail-closed", 1, 2_048, 1_024, "range", 99);
  const resources = createCaseResources(device, tracker, spec);
  const output = createGuardedOutput(device, tracker, spec);
  try {
    device.queue.writeBuffer(resources.activation, 17 * 4, new Float32Array([
      Number.NaN,
    ]));
    const groupReceipts: Record<string, unknown> = {};
    for (const groupSize of ACE_OPT_0058_GROUP_SIZES) {
      const resource = requiredCandidate(resources, groupSize);
      const plan = planAceOpt0058DenseInt8(spec.shape, groupSize);
      const dispatch = await kernel.createDispatch(
        `opt0058-nonfinite-g${groupSize}`,
        spec.shape,
        groupSize,
        Object.freeze({
          activation: binding(resources.activation, plan.activationElements * 4),
          packedActivation: binding(
            resource.packedActivation,
            plan.packedActivationWords * 4,
          ),
          activationScale: binding(
            resource.activationScale,
            plan.activationScaleElements * 4,
          ),
          weight: binding(resource.weight, plan.packedWeightWords * 4),
          weightScale: binding(
            resource.weightScale,
            plan.weightScaleElements * 4,
          ),
        quantizationStatus: binding(resource.counters, 4),
          output: output.binding,
        }),
      );
      const snapshot = await executeCandidateSnapshot(
        device,
        output,
        resource,
        dispatch,
        `nonfinite-g${groupSize}`,
      );
      let failClosedQNaNCount = 0;
      for (const word of snapshot.words) {
        if (word === 0x7fc0_5858) failClosedQNaNCount += 1;
      }
      groupReceipts[`g${groupSize}`] = Object.freeze({
        groupSize,
        nonFiniteActivationCount: snapshot.nonFiniteActivationCount,
        failClosedQNaNCount,
        expectedOutputCount: output.outputElements,
        passed: snapshot.nonFiniteActivationCount === 1 &&
          failClosedQNaNCount === output.outputElements &&
          snapshot.prefixCanaryIntact && snapshot.suffixCanaryIntact &&
          snapshot.tailRowWritten,
      });
    }
    return Object.freeze({
      injectedNonFiniteCount: 1,
      groups: Object.freeze(groupReceipts),
      passed: Object.values(groupReceipts).every((entry) =>
        (entry as Readonly<Record<string, unknown>>)["passed"] === true
      ),
    });
  } finally {
    for (const buffer of resources.buffers) tracker.destroy(buffer);
    for (const buffer of output.buffers) tracker.destroy(buffer);
  }
}

async function runTiming(
  prepared: PreparedGate,
  thermal: Opt0058ThermalGate,
  progress: HTMLElement,
): Promise<Readonly<Record<string, unknown>>> {
  const samples = new Map<string, Record<TimingArm, TimestampSample[]>>(
    prepared.shapes.map(({ spec }) => [spec.id, emptyTimingSamples()]),
  );
  const rawSamples: Readonly<Record<string, unknown>>[] = [];
  for (const [roundIndex, round] of TIMING_ROUNDS.entries()) {
    for (const shapeIndex of round.shapeOrder) {
      const shape = prepared.shapes[shapeIndex];
      if (shape === undefined) throw new Error("OPT-0058 timing shape changed");
      for (const [armPosition, arm] of round.armOrder.entries()) {
        progress.textContent =
          `timing ${roundIndex + 1}/2 ${shape.spec.id} ${arm}`;
        const sample = await timeArm(prepared, shape, arm);
        samples.get(shape.spec.id)![arm].push(sample);
        rawSamples.push(Object.freeze({
          roundIndex,
          shapeIndex,
          armPosition,
          shapeId: shape.spec.id,
          ...sample,
        }));
      }
      await browserYield();
    }
  }
  await prepared.device.queue.onSubmittedWorkDone();
  await browserYield();
  if (prepared.uncapturedErrors.length !== 0) {
    throw new Error("OPT-0058 observed an uncaptured timing GPU error");
  }
  const timing = summarizeTiming(prepared.shapes, samples);
  const correctnessEligibility = prepared.correctness["groupEligibility"] as
    Readonly<Record<GroupArm, Readonly<{ eligible: boolean }>>>;
  const eligiblePassingGroups = ACE_OPT_0058_GROUP_SIZES.filter((groupSize) => {
    const arm = groupArm(groupSize);
    const summary = timing.groups[arm];
    return correctnessEligibility[arm].eligible && summary.passed;
  });
  const passed = prepared.correctness["passed"] === true &&
    eligiblePassingGroups.length !== 0;
  const environment = environmentReceipt(prepared.adapter, prepared.device);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanupFirst = prepared.destroy();
  const cleanupSecond = prepared.destroy();
  return Object.freeze({
    schema: RECEIPT_SCHEMA,
    experimentId: EXPERIMENT_ID,
    passed,
    disposition: "benchmark-only",
    decision: passed
      ? "positive-primitive-only-follow-up-required"
      : "negative-stop-complete-pipeline-or-numerical-gate",
    kernelId: ACE_OPT_0058_DENSE_INT8_DP4A_KERNEL_ID,
    weightLayout: ACE_OPT_0058_DENSE_INT8_DP4A_WEIGHT_LAYOUT,
    scaleLayout: ACE_OPT_0058_DENSE_INT8_DP4A_SCALE_LAYOUT,
    identity: prepared.identity,
    correctness: prepared.correctness,
    thermal,
    protocol: Object.freeze({
      fullAndAdversarialCorrectnessBeforeReady: true,
      successfulStockCompilationBeforeReady: true,
      noTimedWorkBeforeReady: true,
      oneSymmetricWarmupPerArmPerShape: true,
      balancedPermutations: TIMING_ROUNDS,
      oneSubmitAndOneMatchingDrainPerSample: true,
      outputReadbackInsideTiming: false,
      offlineWeightPackingInsideTiming: false,
      oneDynamicQuantizationPerGemm: true,
      activationReuseAmortization: 1,
      speculativeReuse: false,
      prequantizedCeilingIsNotPrimary: true,
      timestampBoundary: "complete compute pass; counter clear excluded from GPU timestamp but included in fenced submit-to-drain wall",
      weightedProductionMultiplicities: "4/2/2/1",
    }),
    timing,
    eligiblePassingGroups: Object.freeze(eligiblePassingGroups.slice()),
    environment,
    memoryBeforeCleanup,
    cleanup: Object.freeze({
      firstCall: cleanupFirst,
      secondCall: cleanupSecond,
      idempotent: true,
      zeroLiveBuffers: cleanupFirst["liveBufferCount"] === 0 &&
        cleanupSecond["liveBufferCount"] === 0,
    }),
  });
}

async function timeArm(
  prepared: PreparedGate,
  shape: PreparedShape,
  arm: TimingArm,
): Promise<TimestampSample> {
  if (prepared.timestampReadback.mapState !== "unmapped") {
    throw new Error("OPT-0058 timestamp readback must be unmapped");
  }
  const selection = selectTimingArm(shape, arm);
  const encoder = prepared.device.createCommandEncoder({
    label: `opt0058-${shape.spec.id}-${arm}-timestamp-sample`,
  });
  if (selection.complete && selection.resources !== undefined) {
    encoder.clearBuffer(selection.resources.counters, 0, 4);
  }
  const pass = encoder.beginComputePass({
    label: `opt0058-${shape.spec.id}-${arm}-timestamped-compute`,
    timestampWrites: {
      querySet: prepared.querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    },
  });
  selection.encode(pass);
  pass.end();
  encoder.resolveQuerySet(
    prepared.querySet,
    0,
    2,
    prepared.timestampResolve,
    0,
  );
  encoder.copyBufferToBuffer(
    prepared.timestampResolve,
    0,
    prepared.timestampReadback,
    0,
    TIMESTAMP_QUERY_BYTES,
  );
  const command = encoder.finish();
  const submitAtPerformanceMilliseconds = performance.now();
  prepared.device.queue.submit([command]);
  await prepared.device.queue.onSubmittedWorkDone();
  const fenceAtPerformanceMilliseconds = performance.now();
  const fencedWallMilliseconds =
    fenceAtPerformanceMilliseconds - submitAtPerformanceMilliseconds;
  await prepared.timestampReadback.mapAsync(GPUMapMode.READ);
  let timestampBegin: bigint;
  let timestampEnd: bigint;
  try {
    const timestamps = new BigUint64Array(
      prepared.timestampReadback.getMappedRange(),
    );
    timestampBegin = timestamps[0]!;
    timestampEnd = timestamps[1]!;
  } finally {
    prepared.timestampReadback.unmap();
  }
  if (timestampEnd <= timestampBegin) {
    throw new Error(`OPT-0058 ${shape.spec.id} ${arm} timestamp was non-positive`);
  }
  const gpuElapsedNanoseconds = Number(timestampEnd - timestampBegin);
  const gpuMilliseconds = gpuElapsedNanoseconds / 1_000_000;
  if (!Number.isSafeInteger(gpuElapsedNanoseconds) || gpuMilliseconds <= 0 ||
    !Number.isFinite(fencedWallMilliseconds) || fencedWallMilliseconds <= 0) {
    throw new Error(`OPT-0058 ${shape.spec.id} ${arm} timing was invalid`);
  }
  return Object.freeze({
    arm,
    ...(selection.groupSize === undefined
      ? {}
      : { groupSize: selection.groupSize }),
    boundary: selection.boundary,
    submitAtPerformanceMilliseconds,
    fenceAtPerformanceMilliseconds,
    fencedWallMilliseconds,
    timestampBeginNanoseconds: timestampBegin.toString(),
    timestampEndNanoseconds: timestampEnd.toString(),
    gpuElapsedNanoseconds,
    gpuMilliseconds,
    commandBufferCount: 1 as const,
    queueDrainCount: 1 as const,
    dynamicQuantizationCount: selection.complete ? 1 as const : 0 as const,
  });
}

function summarizeTiming(
  shapes: readonly PreparedShape[],
  samples: ReadonlyMap<string, Record<TimingArm, TimestampSample[]>>,
): Readonly<{
  strata: readonly Readonly<Record<string, unknown>>[];
  weighted: Readonly<Record<TimingArm, Readonly<{
    gpuMilliseconds: number;
    fencedWallMilliseconds: number;
  }>>>;
  groups: Readonly<Record<GroupArm, Readonly<{
    groupSize: AceOpt0058GroupSize;
    weightedCompleteSpeedupOverK4: number;
    weightedPrequantizedSpeedupOverK4: number;
    weightedCompleteGpuSpeedupOverK4: number;
    weightedPrequantizedGpuSpeedupOverK4: number;
    everyShapeCompleteFasterThanK4: boolean;
    requiredCompleteSpeedup: number;
    passed: boolean;
  }>>>;
}> {
  const weighted = Object.fromEntries(TIMING_ARMS.map((arm) => [arm, {
    gpuMilliseconds: 0,
    fencedWallMilliseconds: 0,
  }])) as Record<TimingArm, {
    gpuMilliseconds: number;
    fencedWallMilliseconds: number;
  }>;
  const strata = shapes.map((shape) => {
    const shapeSamples = samples.get(shape.spec.id);
    if (shapeSamples === undefined) throw new Error("OPT-0058 samples missing");
    const multiplicity = shape.spec.productionMultiplicity;
    if (multiplicity === undefined) throw new Error("OPT-0058 multiplicity missing");
    const arms = Object.fromEntries(TIMING_ARMS.map((arm) => {
      const armSamples = shapeSamples[arm];
      requireTwoSamples(armSamples, `${shape.spec.id} ${arm}`);
      const summary = Object.freeze({
        samples: Object.freeze(armSamples.slice()),
        gpuMilliseconds: mean2(
          armSamples.map((sample) => sample.gpuMilliseconds),
          `${shape.spec.id} ${arm} GPU`,
        ),
        fencedWallMilliseconds: mean2(
          armSamples.map((sample) => sample.fencedWallMilliseconds),
          `${shape.spec.id} ${arm} wall`,
        ),
      });
      weighted[arm].gpuMilliseconds += summary.gpuMilliseconds * multiplicity;
      weighted[arm].fencedWallMilliseconds +=
        summary.fencedWallMilliseconds * multiplicity;
      return [arm, summary];
    })) as Record<TimingArm, Readonly<{
      samples: readonly TimestampSample[];
      gpuMilliseconds: number;
      fencedWallMilliseconds: number;
    }>>;
    return Object.freeze({
      id: shape.spec.id,
      shape: shape.spec.shape,
      multiplicity,
      arms: Object.freeze(arms),
    });
  });
  const frozenWeighted = Object.freeze(Object.fromEntries(TIMING_ARMS.map(
    (arm) => [arm, Object.freeze({ ...weighted[arm] })],
  ))) as Readonly<Record<TimingArm, Readonly<{
    gpuMilliseconds: number;
    fencedWallMilliseconds: number;
  }>>>;
  const groups = Object.freeze(Object.fromEntries(
    ACE_OPT_0058_GROUP_SIZES.map((groupSize) => {
      const arm = groupArm(groupSize);
      const pre = `${arm}Prequantized` as TimingArm;
      const complete = `${arm}Complete` as TimingArm;
      const weightedCompleteSpeedupOverK4 =
        frozenWeighted.dotK4.fencedWallMilliseconds /
        frozenWeighted[complete].fencedWallMilliseconds;
      const weightedPrequantizedSpeedupOverK4 =
        frozenWeighted.dotK4.fencedWallMilliseconds /
        frozenWeighted[pre].fencedWallMilliseconds;
      const weightedCompleteGpuSpeedupOverK4 =
        frozenWeighted.dotK4.gpuMilliseconds /
        frozenWeighted[complete].gpuMilliseconds;
      const weightedPrequantizedGpuSpeedupOverK4 =
        frozenWeighted.dotK4.gpuMilliseconds /
        frozenWeighted[pre].gpuMilliseconds;
      const everyShapeCompleteFasterThanK4 = strata.every((stratum) => {
        const arms = stratum.arms;
        return arms[complete].fencedWallMilliseconds <
          arms.dotK4.fencedWallMilliseconds;
      });
      const passed = everyShapeCompleteFasterThanK4 &&
        weightedCompleteSpeedupOverK4 >= REQUIRED_COMPLETE_SPEEDUP;
      return [arm, Object.freeze({
        groupSize,
        weightedCompleteSpeedupOverK4,
        weightedPrequantizedSpeedupOverK4,
        weightedCompleteGpuSpeedupOverK4,
        weightedPrequantizedGpuSpeedupOverK4,
        everyShapeCompleteFasterThanK4,
        requiredCompleteSpeedup: REQUIRED_COMPLETE_SPEEDUP,
        passed,
      })];
    }),
  )) as Readonly<Record<GroupArm, Readonly<{
    groupSize: AceOpt0058GroupSize;
    weightedCompleteSpeedupOverK4: number;
    weightedPrequantizedSpeedupOverK4: number;
    weightedCompleteGpuSpeedupOverK4: number;
    weightedPrequantizedGpuSpeedupOverK4: number;
    everyShapeCompleteFasterThanK4: boolean;
    requiredCompleteSpeedup: number;
    passed: boolean;
  }>>>;
  return Object.freeze({
    strata: Object.freeze(strata),
    weighted: frozenWeighted,
    groups,
  });
}

export function parseOpt0058ThermalGate(
  parameters: URLSearchParams,
  preparedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0058ThermalGate {
  const command = requiredParameter(parameters, "thermalCommand");
  const waitStartedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "waitStartedAtEpochMilliseconds",
  );
  const checkedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "checkedAtEpochMilliseconds",
  );
  const checkCount = requiredFiniteParameter(parameters, "checkCount");
  const thermalLevel = requiredFiniteParameter(parameters, "thermalLevel");
  const waitDurationMilliseconds =
    checkedAtEpochMilliseconds - waitStartedAtEpochMilliseconds;
  const launchDelayMilliseconds =
    launchedAtEpochMilliseconds - checkedAtEpochMilliseconds;
  if (
    command !== THERMAL_COMMAND ||
    !Number.isSafeInteger(waitStartedAtEpochMilliseconds) ||
    !Number.isSafeInteger(checkedAtEpochMilliseconds) ||
    waitStartedAtEpochMilliseconds < preparedAtEpochMilliseconds ||
    checkedAtEpochMilliseconds <= waitStartedAtEpochMilliseconds ||
    checkCount !== 1 ||
    thermalLevel !== 0 ||
    waitDurationMilliseconds < MINIMUM_WAIT_MILLISECONDS ||
    waitDurationMilliseconds > MAXIMUM_WAIT_MILLISECONDS ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_CHECK_TO_LAUNCH_MILLISECONDS
  ) {
    throw new Error(
      "OPT-0058 timing requires exactly one level-0 notifyutil check after a fresh 30-60 second idle wait and launch within 5 seconds",
    );
  }
  return Object.freeze({
    command: THERMAL_COMMAND,
    waitStartedAtEpochMilliseconds,
    checkedAtEpochMilliseconds,
    waitDurationMilliseconds,
    checkCount: 1,
    thermalLevel: 0,
    launchDelayMilliseconds,
  });
}

function createAggregateSet(): Readonly<{
  full: Readonly<Record<GroupArm | "dotK4", NumericalAccumulator>>;
  adversarial: Readonly<Record<GroupArm | "dotK4", NumericalAccumulator>>;
}> {
  const set = (): Readonly<Record<GroupArm | "dotK4", NumericalAccumulator>> =>
    Object.freeze({
      dotK4: createAccumulator(),
      g32: createAccumulator(),
      g64: createAccumulator(),
      g128: createAccumulator(),
    });
  return Object.freeze({ full: set(), adversarial: set() });
}

function createAccumulator(): NumericalAccumulator {
  return {
    count: 0,
    finiteCount: 0,
    controlNonFiniteCount: 0,
    candidateNonFiniteCount: 0,
    differingU32Count: 0,
    signedZeroDifferenceCount: 0,
    classChangeCount: 0,
    finiteToZeroCollapseCount: 0,
    classChanges: {},
    controlSum: 0,
    candidateSum: 0,
    controlSquareSum: 0,
    candidateSquareSum: 0,
    crossSum: 0,
    errorSum: 0,
    absoluteErrorSum: 0,
    errorSquareSum: 0,
    relativeErrorSquareSum: 0,
    maximumAbsoluteControl: 0,
    maximumAbsoluteError: 0,
    maximumRelativeError: 0,
    firstDifference: null,
    worstDifference: null,
  };
}

function accumulateNumerics(
  accumulator: NumericalAccumulator,
  controlWords: Uint32Array,
  candidateWords: Uint32Array,
  spec: Opt0058CaseSpec,
): void {
  if (controlWords.length !== candidateWords.length) {
    throw new Error(`OPT-0058 ${spec.id} output lengths changed`);
  }
  const scratch = new ArrayBuffer(4);
  const scratchU32 = new Uint32Array(scratch);
  const scratchF32 = new Float32Array(scratch);
  for (let index = 0; index < controlWords.length; index += 1) {
    const controlWord = controlWords[index]!;
    const candidateWord = candidateWords[index]!;
    scratchU32[0] = controlWord;
    const control = scratchF32[0]!;
    scratchU32[0] = candidateWord;
    const candidate = scratchF32[0]!;
    accumulator.count += 1;
    if (!Number.isFinite(control)) accumulator.controlNonFiniteCount += 1;
    if (!Number.isFinite(candidate)) accumulator.candidateNonFiniteCount += 1;
    const controlClass = f32Class(controlWord);
    const candidateClass = f32Class(candidateWord);
    if (controlClass !== candidateClass) {
      accumulator.classChangeCount += 1;
      const transition = `${controlClass}->${candidateClass}`;
      accumulator.classChanges[transition] =
        (accumulator.classChanges[transition] ?? 0) + 1;
    }
    if (controlWord !== candidateWord) {
      accumulator.differingU32Count += 1;
      accumulator.firstDifference ??= Object.freeze({
        shapeId: spec.id,
        index,
        row: Math.floor(index / spec.shape.columns),
        column: index % spec.shape.columns,
        control,
        candidate,
        controlU32: controlWord,
        candidateU32: candidateWord,
      });
    }
    if ((controlWord & 0x7fff_ffff) === 0 &&
      (candidateWord & 0x7fff_ffff) === 0 && controlWord !== candidateWord) {
      accumulator.signedZeroDifferenceCount += 1;
    }
    if (!Number.isFinite(control) || !Number.isFinite(candidate)) continue;
    if ((controlWord & 0x7fff_ffff) !== 0 &&
      (candidateWord & 0x7fff_ffff) === 0) {
      accumulator.finiteToZeroCollapseCount += 1;
    }
    const error = candidate - control;
    const absoluteError = Math.abs(error);
    const relativeError = absoluteError / Math.max(Math.abs(control), 1e-6);
    accumulator.finiteCount += 1;
    accumulator.controlSum += control;
    accumulator.candidateSum += candidate;
    accumulator.controlSquareSum += control * control;
    accumulator.candidateSquareSum += candidate * candidate;
    accumulator.crossSum += control * candidate;
    accumulator.errorSum += error;
    accumulator.absoluteErrorSum += absoluteError;
    accumulator.errorSquareSum += error * error;
    accumulator.relativeErrorSquareSum += relativeError * relativeError;
    accumulator.maximumAbsoluteControl = Math.max(
      accumulator.maximumAbsoluteControl,
      Math.abs(control),
    );
    accumulator.maximumRelativeError = Math.max(
      accumulator.maximumRelativeError,
      relativeError,
    );
    if (absoluteError > accumulator.maximumAbsoluteError) {
      accumulator.maximumAbsoluteError = absoluteError;
      accumulator.worstDifference = Object.freeze({
        shapeId: spec.id,
        index,
        row: Math.floor(index / spec.shape.columns),
        column: index % spec.shape.columns,
        control,
        candidate,
        error,
        absoluteError,
        relativeError,
      });
    }
  }
}

function mergeAccumulator(
  target: NumericalAccumulator,
  source: NumericalAccumulator,
): void {
  for (const key of [
    "count", "finiteCount", "controlNonFiniteCount",
    "candidateNonFiniteCount", "differingU32Count",
    "signedZeroDifferenceCount", "classChangeCount",
    "finiteToZeroCollapseCount", "controlSum", "candidateSum",
    "controlSquareSum", "candidateSquareSum", "crossSum", "errorSum",
    "absoluteErrorSum", "errorSquareSum", "relativeErrorSquareSum",
  ] as const) target[key] += source[key];
  target.maximumAbsoluteControl = Math.max(
    target.maximumAbsoluteControl,
    source.maximumAbsoluteControl,
  );
  target.maximumRelativeError = Math.max(
    target.maximumRelativeError,
    source.maximumRelativeError,
  );
  if (source.maximumAbsoluteError > target.maximumAbsoluteError) {
    target.maximumAbsoluteError = source.maximumAbsoluteError;
    target.worstDifference = source.worstDifference;
  }
  target.firstDifference ??= source.firstDifference;
  for (const [key, value] of Object.entries(source.classChanges)) {
    target.classChanges[key] = (target.classChanges[key] ?? 0) + value;
  }
}

function finalizeAggregateSet(
  set: Readonly<Record<GroupArm | "dotK4", NumericalAccumulator>>,
  envelope: "full" | "adversarial",
): Readonly<Record<GroupArm | "dotK4", Readonly<Record<string, unknown>>>> {
  return Object.freeze({
    dotK4: Object.freeze({ envelope, ...finalizeNumerics(set.dotK4) }),
    g32: Object.freeze({ envelope, ...finalizeNumerics(set.g32) }),
    g64: Object.freeze({ envelope, ...finalizeNumerics(set.g64) }),
    g128: Object.freeze({ envelope, ...finalizeNumerics(set.g128) }),
  });
}

function finalizeNumerics(
  accumulator: NumericalAccumulator,
): Readonly<Record<string, unknown>> {
  if (accumulator.count === 0 || accumulator.finiteCount === 0) {
    throw new Error("OPT-0058 cannot finalize an empty comparison");
  }
  const count = accumulator.finiteCount;
  const controlRms = Math.sqrt(accumulator.controlSquareSum / count);
  const rmsError = Math.sqrt(accumulator.errorSquareSum / count);
  const relativeRmsError = Math.sqrt(accumulator.relativeErrorSquareSum / count);
  const nrmse = rmsError / Math.max(controlRms, 1e-12);
  const snrDecibels = rmsError === 0
    ? Number.POSITIVE_INFINITY
    : 20 * Math.log10(controlRms / rmsError);
  const covariance = accumulator.crossSum -
    accumulator.controlSum * accumulator.candidateSum / count;
  const controlVariance = accumulator.controlSquareSum -
    accumulator.controlSum ** 2 / count;
  const candidateVariance = accumulator.candidateSquareSum -
    accumulator.candidateSum ** 2 / count;
  const pearsonDenominator = Math.sqrt(
    Math.max(0, controlVariance) * Math.max(0, candidateVariance),
  );
  const pearsonCorrelation = pearsonDenominator === 0
    ? accumulator.errorSquareSum === 0 ? 1 : 0
    : covariance / pearsonDenominator;
  const finiteToZeroCollapseRate =
    accumulator.finiteToZeroCollapseCount / accumulator.count;
  const passed = accumulator.controlNonFiniteCount === 0 &&
    accumulator.candidateNonFiniteCount === 0 &&
    nrmse <= NRMSE_MAXIMUM &&
    snrDecibels >= SNR_DECIBELS_MINIMUM &&
    pearsonCorrelation >= PEARSON_MINIMUM &&
    finiteToZeroCollapseRate <= FINITE_TO_ZERO_RATE_MAXIMUM;
  return Object.freeze({
    count: accumulator.count,
    finiteCount: accumulator.finiteCount,
    differingU32Count: accumulator.differingU32Count,
    signedZeroDifferenceCount: accumulator.signedZeroDifferenceCount,
    classChangeCount: accumulator.classChangeCount,
    classChanges: Object.freeze({ ...accumulator.classChanges }),
    finiteToZeroCollapseCount: accumulator.finiteToZeroCollapseCount,
    finiteToZeroCollapseRate,
    controlNonFiniteCount: accumulator.controlNonFiniteCount,
    candidateNonFiniteCount: accumulator.candidateNonFiniteCount,
    signedMeanError: accumulator.errorSum / count,
    meanAbsoluteError: accumulator.absoluteErrorSum / count,
    rmsError,
    relativeRmsError,
    nrmse,
    snrDecibels: Number.isFinite(snrDecibels)
      ? snrDecibels
      : "positive-infinity",
    pearsonCorrelation,
    maximumAbsoluteControl: accumulator.maximumAbsoluteControl,
    maximumAbsoluteError: accumulator.maximumAbsoluteError,
    maximumRelativeError: accumulator.maximumRelativeError,
    firstDifference: accumulator.firstDifference,
    worstDifference: accumulator.worstDifference,
    thresholds: Object.freeze({
      nrmseMaximum: NRMSE_MAXIMUM,
      snrDecibelsMinimum: SNR_DECIBELS_MINIMUM,
      pearsonMinimum: PEARSON_MINIMUM,
      finiteToZeroCollapseRateMaximum: FINITE_TO_ZERO_RATE_MAXIMUM,
    }),
    passed,
  });
}

function fillActivation(values: Float32Array, spec: Opt0058CaseSpec): void {
  let physical = 0;
  for (let row = 0; row < spec.shape.rows; row += 1) {
    for (let inner = 0; inner < spec.shape.inner; inner += 1) {
      values[physical] = halfToNumber(activationBitsAt(spec, row, inner));
      physical += 1;
    }
  }
}

function fillLogicalWeight(values: Uint16Array, spec: Opt0058CaseSpec): void {
  let physical = 0;
  for (let inner = 0; inner < spec.shape.inner; inner += 1) {
    for (let column = 0; column < spec.shape.columns; column += 1) {
      values[physical] = weightBitsAt(spec, inner, column);
      physical += 1;
    }
  }
}

function activationBitsAt(
  spec: Opt0058CaseSpec,
  row: number,
  inner: number,
): number {
  const group = inner >>> 2;
  const offset = inner & 3;
  const mixed = mix32(
    0x3141_5926 ^ Math.imul(spec.ordinal + 1, 0x9e37_79b1) ^
      Math.imul(row + 1, 0x85eb_ca6b) ^
      Math.imul(group + 1, 0xc2b2_ae35) ^
      Math.imul(offset + 1, 0x27d4_eb2f),
  );
  if (spec.fixtureKind === "signed-zero" ||
    spec.fixtureKind === "zero-group") return (mixed >>> 31) << 15;
  if (spec.fixtureKind === "range") {
    const magnitude = [0x1800, 0x2401, 0x3c00, 0x4400][inner & 3]!;
    return magnitude | ((mixed >>> 31) << 15);
  }
  if (spec.fixtureKind === "saturation") {
    const magnitude = inner % 128 === 0 ? 0x4c00 : 0x1800;
    return magnitude | ((mixed >>> 31) << 15);
  }
  const magnitude = FINITE_HALF_MAGNITUDES[
    mixed % FINITE_HALF_MAGNITUDES.length
  ]!;
  if (spec.fixtureKind === "cancellation") {
    return magnitude | (((row + group) & 1) << 15);
  }
  return magnitude | ((mixed >>> 31) << 15);
}

function weightBitsAt(
  spec: Opt0058CaseSpec,
  inner: number,
  column: number,
): number {
  const group = inner >>> 2;
  const offset = inner & 3;
  const mixed = mix32(
    0x6a09_e667 ^ Math.imul(spec.ordinal + 1, 0x9e37_79b1) ^
      Math.imul(column + 1, 0x85eb_ca6b) ^
      Math.imul(group + 1, 0xc2b2_ae35) ^
      Math.imul(offset + 1, 0x27d4_eb2f),
  );
  if (spec.fixtureKind === "signed-zero") {
    return 0x3c00 | ((mixed >>> 31) << 15);
  }
  if (spec.fixtureKind === "range") {
    const magnitude = [0x4400, 0x1800, 0x2401, 0x3c00][inner & 3]!;
    return magnitude | ((((mixed >>> 31) ^ (offset & 1)) & 1) << 15);
  }
  if (spec.fixtureKind === "saturation") {
    const magnitude = inner % 128 === 0 ? 0x4c00 : 0x1800;
    return magnitude | ((mixed >>> 31) << 15);
  }
  const magnitude = FINITE_HALF_MAGNITUDES[
    mixed % FINITE_HALF_MAGNITUDES.length
  ]!;
  if (spec.fixtureKind === "cancellation") {
    const withinK4Sign = offset === 1 || offset === 2 ? 1 : 0;
    return magnitude | ((((column & 1) ^ withinK4Sign) & 1) << 15);
  }
  return magnitude | ((mixed >>> 31) << 15);
}

function packExactWeight(
  logical: Uint16Array,
  inner: number,
  columns: number,
): Uint16Array<ArrayBuffer> {
  const packed = new Uint16Array(logical.length);
  let physical = 0;
  for (let columnTile = 0; columnTile < columns / 256; columnTile += 1) {
    for (let innerTile = 0; innerTile < inner / 32; innerTile += 1) {
      for (let innerInTile = 0; innerInTile < 32; innerInTile += 1) {
        const k = innerTile * 32 + innerInTile;
        for (let columnInTile = 0; columnInTile < 256; columnInTile += 1) {
          const column = columnTile * 256 + columnInTile;
          packed[physical] = logical[k * columns + column]!;
          physical += 1;
        }
      }
    }
  }
  return packed;
}

function createGuardedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0058CaseSpec,
): GuardedOutput {
  const outputElements = spec.shape.rows * spec.shape.columns;
  const outputBytes = outputElements * 4;
  const totalBytes = outputBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = tracker.create(device, {
    label: `opt0058-${spec.id}-prefill`,
    size: totalBytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  const words = new Uint32Array(prefill.getMappedRange());
  words.fill(STORAGE_GUARD_U32);
  words.fill(
    OUTPUT_PREFILL_QNAN_U32,
    STORAGE_GUARD_BYTES / 4,
    STORAGE_GUARD_BYTES / 4 + outputElements,
  );
  prefill.unmap();
  const buffer = tracker.create(device, {
    label: `opt0058-${spec.id}-output`,
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  const readback = tracker.create(device, {
    label: `opt0058-${spec.id}-readback`,
    size: totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  return Object.freeze({
    buffer,
    binding: Object.freeze({
      buffer,
      offset: STORAGE_GUARD_BYTES,
      size: outputBytes,
    }),
    prefill,
    readback,
    buffers: Object.freeze([prefill, buffer, readback]),
    outputElements,
    columns: spec.shape.columns,
    outputBytes,
    totalBytes,
  });
}

function requireCompleteSnapshot(
  snapshot: OutputSnapshot,
  label: string,
  requireNonzero: boolean,
): void {
  if (
    snapshot.nonFiniteCount !== 0 ||
    (requireNonzero && snapshot.nonzeroCount === 0) ||
    snapshot.qNaNPrefillCount !== 0 ||
    !snapshot.prefixCanaryIntact ||
    !snapshot.suffixCanaryIntact ||
    !snapshot.tailRowWritten
  ) {
    throw new Error(`${label} failed complete-write, finite, tail, or canary gate`);
  }
}

function requireExactWords(
  expected: Uint32Array,
  actual: Uint32Array,
  label: string,
): void {
  if (expected.length !== actual.length) throw new Error(`${label} length changed`);
  for (let index = 0; index < expected.length; index += 1) {
    if (expected[index] !== actual[index]) {
      throw new Error(`${label} raw-U32 mismatch at ${index}`);
    }
  }
}

async function executeTimingArmAndDrain(
  device: GPUDevice,
  shape: PreparedShape,
  arm: TimingArm,
): Promise<void> {
  const selection = selectTimingArm(shape, arm);
  const encoder = device.createCommandEncoder();
  if (selection.complete && selection.resources !== undefined) {
    encoder.clearBuffer(selection.resources.counters, 0, 4);
  }
  const pass = encoder.beginComputePass();
  selection.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

function selectTimingArm(
  shape: PreparedShape,
  arm: TimingArm,
): Readonly<{
  encode(pass: GPUComputePassEncoder): void;
  boundary: TimestampSample["boundary"];
  complete: boolean;
  groupSize?: AceOpt0058GroupSize;
  resources?: CandidateResources;
}> {
  if (arm === "exact") {
    return Object.freeze({
      encode: (pass: GPUComputePassEncoder) => shape.exact.encode(pass),
      boundary: "exact" as const,
      complete: false,
    });
  }
  if (arm === "dotK4") {
    return Object.freeze({
      encode: (pass: GPUComputePassEncoder) => shape.dotK4.encode(pass),
      boundary: "dot-k4" as const,
      complete: false,
    });
  }
  const match = /^g(32|64|128)(Prequantized|Complete)$/.exec(arm);
  if (match === null) throw new Error(`OPT-0058 unknown timing arm ${arm}`);
  const groupSize = Number(match[1]) as AceOpt0058GroupSize;
  const complete = match[2] === "Complete";
  const dispatch = requiredDispatch(shape.candidates, groupSize);
  const resources = requiredCandidate(shape.resources, groupSize);
  return Object.freeze({
    encode: complete
      ? (pass: GPUComputePassEncoder) => dispatch.encodeComplete(pass)
      : (pass: GPUComputePassEncoder) => dispatch.encodePrequantized(pass),
    boundary: complete
      ? "complete-dynamic-quantize-plus-gemm" as const
      : "prequantized-ceiling" as const,
    complete,
    groupSize,
    resources,
  });
}

function buildRuntimeLayoutProof(): Readonly<Record<string, unknown>> {
  const groups: Record<string, unknown> = {};
  for (const groupSize of ACE_OPT_0058_GROUP_SIZES) {
    const inner = groupSize * 2;
    const columns = 128;
    const logical = new Float32Array(inner * columns);
    for (let k = 0; k < inner; k += 1) {
      for (let column = 0; column < columns; column += 1) {
        logical[k * columns + column] =
          ((k * 17 + column * 29) % 253 - 126) / 23;
      }
    }
    const packed = quantizeAndPackAceOpt0058DenseWeight(
      logical,
      inner,
      columns,
      groupSize,
    );
    const inverse = unpackAceOpt0058DenseWeightI8(
      packed.packed,
      inner,
      columns,
      groupSize,
    );
    const visitedWords = new Uint8Array(packed.packed.length);
    const visitedScales = new Uint8Array(packed.scales.length);
    let inverseMismatchCount = 0;
    let repackedWordMismatchCount = 0;
    for (let column = 0; column < columns; column += 1) {
      for (let group = 0; group < inner / groupSize; group += 1) {
        const scaleIndex = aceOpt0058WeightScaleIndex(
          column,
          group,
          inner,
          columns,
          groupSize,
        );
        visitedScales[scaleIndex] = visitedScales[scaleIndex]! + 1;
        const scale = packed.scales[scaleIndex]!;
        for (let packedK = 0; packedK < groupSize / 4; packedK += 1) {
          const wordIndex = aceOpt0058PackedWeightWordIndex(
            group,
            packedK,
            column,
            inner,
            columns,
            groupSize,
          );
          visitedWords[wordIndex] = visitedWords[wordIndex]! + 1;
          const word = packed.packed[wordIndex]!;
          for (let byte = 0; byte < 4; byte += 1) {
            const k = group * groupSize + packedK * 4 + byte;
            const normalized = Math.fround(logical[k * columns + column]! / scale);
            const expected = Math.max(
              -127,
              Math.min(
                127,
                Math.sign(normalized) * Math.floor(Math.abs(normalized) + 0.5),
              ),
            );
            const actual = inverse[k * columns + column]!;
            if (actual !== expected) inverseMismatchCount += 1;
            // Exercise the exact byte pack helper over the full physical domain.
            if (byte === 0) {
              const base = k * columns + column;
              const repacked = packAceOpt0058SignedI8x4(
                inverse[base]!,
                inverse[base + columns]!,
                inverse[base + 2 * columns]!,
                inverse[base + 3 * columns]!,
              );
              if (repacked !== word) repackedWordMismatchCount += 1;
            }
          }
        }
      }
    }
    const everyWordVisitedOnce = visitedWords.every((count) => count === 1);
    const everyScaleVisitedOnce = visitedScales.every((count) => count === 1);
    groups[`g${groupSize}`] = Object.freeze({
      groupSize,
      logicalValues: logical.length,
      packedWords: packed.packed.length,
      scaleElements: packed.scales.length,
      everyWordVisitedOnce,
      everyScaleVisitedOnce,
      inverseMismatchCount,
      repackedWordMismatchCount,
      passed: everyWordVisitedOnce && everyScaleVisitedOnce &&
        inverseMismatchCount === 0 && repackedWordMismatchCount === 0,
    });
  }
  if (!Object.values(groups).every((entry) =>
    (entry as Readonly<Record<string, unknown>>)["passed"] === true
  )) throw new Error("OPT-0058 runtime pack/index/inverse proof failed");
  return Object.freeze({
    signedI8Domain: "[-127,127]",
    deterministicRounding: "f32 divide; round nearest with halves away from zero",
    deterministicClamp: "[-127,127]",
    zeroGroupScaleAndPayloadCanonicalZero: true,
    groups: Object.freeze(groups),
    passed: true,
  });
}

function buildIdentity(
  adapter: GPUAdapter,
  device: GPUDevice,
  wgslLanguageFeatures: readonly string[],
  layoutProof: Readonly<Record<string, unknown>>,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    stockWebGpuOnly: true,
    experimentalBrowserFlags: false,
    userAgent: navigator.userAgent,
    adapter: Object.freeze({
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
      subgroupMinSize: adapter.info.subgroupMinSize,
      subgroupMaxSize: adapter.info.subgroupMaxSize,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    }),
    deviceFeatures: Object.freeze([...device.features].sort()),
    wgslLanguageFeatures: Object.freeze(wgslLanguageFeatures.slice()),
    requiredWgslLanguageFeature:
      ACE_OPT_0058_PACKED_DOT_LANGUAGE_FEATURE,
    packed4x8IntegerDotProductPresent: wgslLanguageFeatures.includes(
      ACE_OPT_0058_PACKED_DOT_LANGUAGE_FEATURE,
    ),
    languageFeaturePresenceDoesNotProveNativeHardwareDp4a: true,
    adapterVendorRecordedForNativeOrLoweredInterpretation: true,
    successfulPackedDotPipelineCompilation: true,
    groupSizes: ACE_OPT_0058_GROUP_SIZES,
    layoutProof,
    fixtureVersion: "opt0032-full-plus-opt0058-adversarial-fp16-v1",
  });
}

function getWgslLanguageFeatures(): string[] {
  const gpu = navigator.gpu as GPU & {
    readonly wgslLanguageFeatures?: ReadonlySet<string>;
  };
  return [...(gpu.wgslLanguageFeatures ?? new Set<string>())].sort();
}

function environmentReceipt(
  adapter: GPUAdapter,
  device: GPUDevice,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    userAgent: navigator.userAgent,
    adapter: Object.freeze({
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
      subgroupMinSize: adapter.info.subgroupMinSize,
      subgroupMaxSize: adapter.info.subgroupMaxSize,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    }),
    deviceFeatures: Object.freeze([...device.features].sort()),
    wgslLanguageFeatures: Object.freeze(getWgslLanguageFeatures()),
    limits: Object.freeze({
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    }),
  });
}

function requireAdapter(adapter: GPUAdapter): void {
  const maximumStorageBytes = maximumStorageBindingBytes();
  if (
    !adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    !adapter.features.has("timestamp-query") ||
    Number(adapter.info.subgroupMinSize) !== 32 ||
    Number(adapter.info.subgroupMaxSize) !== 32 ||
    Number(adapter.limits.maxComputeInvocationsPerWorkgroup) < 128 ||
    Number(adapter.limits.maxComputeWorkgroupSizeX) < 128 ||
    Number(adapter.limits.maxStorageBufferBindingSize) < maximumStorageBytes ||
    Number(adapter.limits.maxBufferSize) <
      maximumStorageBytes + 2 * STORAGE_GUARD_BYTES
  ) {
    throw new Error(
      "OPT-0058 requires stock Chrome timestamps, shader-f16, fixed32 subgroups, WG128, and full-shape storage limits",
    );
  }
}

function maximumStorageBindingBytes(): number {
  const cases = buildOpt0058Cases();
  return Math.max(...[...cases.full, ...cases.adversarial].flatMap(({ shape }) => [
    shape.rows * shape.inner * 4,
    shape.inner * shape.columns * 2,
    shape.rows * shape.columns * 4,
  ]));
}

function destroyCase(tracker: BufferTracker, shape: PreparedShape): void {
  for (const buffer of shape.resources.buffers) tracker.destroy(buffer);
  for (const buffer of shape.output.buffers) tracker.destroy(buffer);
}

function requiredCandidate(
  resources: CaseResources,
  groupSize: AceOpt0058GroupSize,
): CandidateResources {
  const candidate = resources.candidates.get(groupSize);
  if (candidate === undefined) {
    throw new Error(`OPT-0058 missing G${groupSize} resources`);
  }
  return candidate;
}

function requiredDispatch(
  dispatches: ReadonlyMap<AceOpt0058GroupSize, AceOpt0058DenseInt8Dispatch>,
  groupSize: AceOpt0058GroupSize,
): AceOpt0058DenseInt8Dispatch {
  const dispatch = dispatches.get(groupSize);
  if (dispatch === undefined) {
    throw new Error(`OPT-0058 missing G${groupSize} dispatch`);
  }
  return dispatch;
}

function groupArm(groupSize: AceOpt0058GroupSize): GroupArm {
  return `g${groupSize}` as GroupArm;
}

function caseGroupPassed(
  receipt: Readonly<Record<string, unknown>>,
  arm: GroupArm,
): boolean {
  const groups = receipt["groups"] as
    Readonly<Record<GroupArm, Readonly<Record<string, unknown>>>>;
  const numerics = groups[arm]["numerics"] as Readonly<Record<string, unknown>>;
  return numerics["passed"] === true;
}

function emptyTimingSamples(): Record<TimingArm, TimestampSample[]> {
  return {
    exact: [],
    dotK4: [],
    g32Prequantized: [],
    g32Complete: [],
    g64Prequantized: [],
    g64Complete: [],
    g128Prequantized: [],
    g128Complete: [],
  };
}

function requireTwoSamples(
  samples: readonly TimestampSample[],
  label: string,
): void {
  if (samples.length !== 2 || samples.some((sample) =>
    !Number.isFinite(sample.gpuMilliseconds) || sample.gpuMilliseconds <= 0 ||
    !Number.isFinite(sample.fencedWallMilliseconds) ||
    sample.fencedWallMilliseconds <= 0
  )) throw new Error(`OPT-0058 ${label} requires two valid samples`);
}

function mean2(samples: readonly number[], label: string): number {
  if (samples.length !== 2 || samples.some((sample) =>
    !Number.isFinite(sample) || sample <= 0
  )) throw new Error(`OPT-0058 ${label} requires two positive samples`);
  return (samples[0]! + samples[1]!) / 2;
}

function f32Class(word: number): string {
  const absolute = word & 0x7fff_ffff;
  const negative = (word & 0x8000_0000) !== 0;
  if (absolute === 0) return negative ? "negative-zero" : "positive-zero";
  const exponent = word & 0x7f80_0000;
  const mantissa = word & 0x007f_ffff;
  if (exponent === 0x7f80_0000) {
    if (mantissa !== 0) return "nan";
    return negative ? "negative-infinity" : "positive-infinity";
  }
  return negative ? "negative-finite" : "positive-finite";
}

function halfToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) {
    if (mantissa === 0) return sign < 0 ? -0 : 0;
    return sign * 2 ** -14 * mantissa / 1_024;
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1_024);
}

function mix32(value: number): number {
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb_352d);
  value ^= value >>> 15;
  value = Math.imul(value, 0x846c_a68b);
  value ^= value >>> 16;
  return value >>> 0;
}

function fullSpec(
  id: string,
  inner: number,
  columns: number,
  productionMultiplicity: 4 | 2 | 1,
  ordinal: number,
): Opt0058CaseSpec {
  return Object.freeze({
    id,
    shape: Object.freeze({ rows: ROWS, inner, columns }),
    productionMultiplicity,
    fixtureKind: "full" as const,
    ordinal,
  });
}

function caseSpec(
  id: string,
  rows: number,
  inner: number,
  columns: number,
  fixtureKind: Exclude<FixtureKind, "full">,
  ordinal: number,
): Opt0058CaseSpec {
  return Object.freeze({
    id,
    shape: Object.freeze({ rows, inner, columns }),
    fixtureKind,
    ordinal,
  });
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  private readonly sizes = new Map<GPUBuffer, number>();
  private created = 0;
  private destroyed = 0;
  private liveBytes = 0;
  private maximumLiveBytes = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    const size = Number(descriptor.size);
    this.live.add(buffer);
    this.sizes.set(buffer, size);
    this.created += 1;
    this.liveBytes += size;
    this.maximumLiveBytes = Math.max(this.maximumLiveBytes, this.liveBytes);
    return buffer;
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    buffer.destroy();
    this.destroyed += 1;
    this.liveBytes -= this.sizes.get(buffer) ?? 0;
    this.sizes.delete(buffer);
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number>> {
    return Object.freeze({
      createdBufferCount: this.created,
      destroyedBufferCount: this.destroyed,
      liveBufferCount: this.live.size,
      liveBytes: this.liveBytes,
      maximumLiveBytes: this.maximumLiveBytes,
    });
  }
}

function requireLittleEndianHost(): void {
  const words = new Uint16Array([0x0102]);
  if (new Uint8Array(words.buffer)[0] !== 0x02) {
    throw new Error("OPT-0058 fixtures require a little-endian host");
  }
}

function fieldParameters(selector: string): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    `${selector} input[name]`,
  )) parameters.set(input.name, input.value);
  return parameters;
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.length === 0) {
    throw new Error(`OPT-0058 field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) throw new Error(`OPT-0058 field ${name} invalid`);
  return value;
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing OPT-0058 element ${selector}`);
  return element;
}

function finishPage(
  status: "passed" | "failed",
  receipt: Readonly<Record<string, unknown>>,
): void {
  window.__ACE_OPT0058_RESULT__ = receipt;
  document.body.dataset.status = status;
  requireElement<HTMLElement>("#progress").textContent = status;
  requireElement<HTMLElement>("#result").textContent =
    JSON.stringify(receipt, null, 2);
}

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: RECEIPT_SCHEMA,
    experimentId: EXPERIMENT_ID,
    passed: false,
    error: error instanceof Error
      ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
      : String(error),
  });
}

async function browserYield(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}
