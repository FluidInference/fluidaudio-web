/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import currentKernelSource from
  "../../src/webgpu/kernels/dit-dense-fp16.ts?raw";
import candidateKernelSource from
  "../../src/webgpu/kernels/dit-dense-fp16-k4-partials.ts?raw";
import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../../src/model/manifest.js";
import {
  AceOpt0009DenseGemmKernel,
  aceOpt0009DenseGemmWgsl,
  planAceOpt0009DenseGemm,
} from "../../src/webgpu/kernels/dit-dense-fp16.js";
import {
  ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID,
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
  ACE_OPT_0074_DENSE_K2_PARTIALS_KERNEL_ID,
  ACE_OPT_0074_DENSE_K2_PARTIALS_WEIGHT_LAYOUT,
  AceOpt0032DenseK4PartialsKernel,
  AceOpt0074DenseK2PartialsKernel,
  aceOpt0032DenseK4PartialsWgsl,
  aceOpt0074DenseK2PartialsWgsl,
  planAceOpt0032DenseK4Partials,
  type AceOpt0032DenseK4PartialsDispatch,
  type AceOpt0074DenseK2PartialsDispatch,
} from "../../src/webgpu/kernels/dit-dense-fp16-k4-partials.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";

declare global {
  interface Window {
    __ACE_OPT0074_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

type Arm = "exact" | "k2" | "k4";
type DenseDispatch = Pick<AceGemmDispatch, "label" | "encode"> |
  Pick<AceOpt0032DenseK4PartialsDispatch, "label" | "encode"> |
  Pick<AceOpt0074DenseK2PartialsDispatch, "label" | "encode">;
type FixtureKind = "full" | "signed-zero" | "cancellation" | "range" |
  "long-k";

interface CaseSpec {
  readonly id: string;
  readonly shape: AceGemmShape;
  readonly fixtureKind: FixtureKind;
  readonly ordinal: number;
}

interface FullShapeSpec extends CaseSpec {
  readonly productionMultiplicity: 4 | 2 | 1;
  readonly fixtureKind: "full";
}

interface GuardedOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly prefill: GPUBuffer;
  readonly readback: GPUBuffer;
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

interface PreparedShape {
  readonly spec: FullShapeSpec;
  readonly dispatches: Readonly<Record<Arm, DenseDispatch>>;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedHarness {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly exactKernel: AceOpt0009DenseGemmKernel;
  readonly k2Kernel: AceOpt0074DenseK2PartialsKernel;
  readonly k4Kernel: AceOpt0032DenseK4PartialsKernel;
  readonly timestampQuerySet: GPUQuerySet;
  readonly timestampResolve: GPUBuffer;
  readonly timestampReadback: GPUBuffer;
  readonly shapes: readonly PreparedShape[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly identity: Readonly<Record<string, unknown>>;
  readonly uncapturedErrors: readonly string[];
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

interface TimingInput {
  readonly id: FullShapeSpec["id"];
  readonly samples: Readonly<Record<Arm, readonly TimestampSample[]>>;
}

interface TimingFailureEvidence {
  rawSamples: Readonly<Record<string, unknown>>[];
  runStartedAtEpochMilliseconds?: number;
  measurementCompletedAtEpochMilliseconds?: number;
}

interface TimestampSample {
  readonly submitAtPerformanceMilliseconds: number;
  readonly fenceAtPerformanceMilliseconds: number;
  readonly submitAtEpochMilliseconds: number;
  readonly fenceAtEpochMilliseconds: number;
  readonly wallMilliseconds: number;
  readonly timestampBeginNanoseconds: string;
  readonly timestampEndNanoseconds: string;
  readonly gpuElapsedNanoseconds: number;
  readonly gpuMilliseconds: number;
  readonly gpuToWallRatio: number;
  readonly validMultiplyAdds: number;
  readonly scheduledMultiplyAdds: number;
  readonly validGpuTflops: number;
  readonly scheduledGpuTflops: number;
  readonly validWallTflops: number;
  readonly scheduledWallTflops: number;
  readonly commandBufferCount: 1;
  readonly queueDrainCount: 1;
  readonly timestampResolveCount: 1;
  readonly timestampCopyCount: 1;
}

const EXPERIMENT_ID = "OPT-0074" as const;
const RECEIPT_SCHEMA = "ace-opt-0074-dense-k2-partials-v1";
const ROWS = 2_250;
const FULL_OUTPUT_COUNT = 25_344_000;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_3255;
const TIMESTAMP_QUERY_BYTES = 16;
const REQUIRED_WEIGHTED_WALL_SPEEDUP = 1.15;
const REQUIRED_WEIGHTED_WALL_SAVING_MILLISECONDS = 25;
const REQUIRED_WEIGHTED_GPU_SPEEDUP = 1.15;
const REQUIRED_WEIGHTED_GPU_SAVING_MILLISECONDS = 25;
const FULL_NRMSE_MAXIMUM = 0.02;
const FULL_SNR_DECIBELS_MINIMUM = 34;
const FULL_PEARSON_MINIMUM = 0.999;
const FULL_MAXIMUM_ABSOLUTE_ERROR = 0.25;
const ADVERSARIAL_NRMSE_MAXIMUM = 0.05;
const ADVERSARIAL_SNR_DECIBELS_MINIMUM = 26;
const ADVERSARIAL_PEARSON_MINIMUM = 0.995;
const ADVERSARIAL_MAXIMUM_ABSOLUTE_ERROR = 0.5;
const FINITE_HALF_MAGNITUDES = Object.freeze([
  0x2411, 0x28b5, 0x2d53, 0x31e7, 0x356b, 0x39ad,
] as const);
const FULL_SHAPES = Object.freeze([
  fullSpec("h-h", 2_048, 2_048, 4, 0),
  fullSpec("h-1024", 2_048, 1_024, 2, 1),
  fullSpec("h-6144", 2_048, 6_144, 2, 2),
  fullSpec("6144-h", 6_144, 2_048, 1, 3),
] as const);
const ADVERSARIAL_CASES = Object.freeze([
  caseSpec("signed-zero", 1, 2_048, 1_024, "signed-zero", 10),
  caseSpec("k4-cancellation", 8, 2_048, 1_024, "cancellation", 11),
  caseSpec("finite-range", 4, 2_048, 1_024, "range", 12),
  caseSpec("long-k6144", 2, 6_144, 2_048, "long-k", 13),
] as const);
const TIMING_ROUNDS = Object.freeze([
  Object.freeze({
    shapeOrder: Object.freeze([0, 1, 2, 3]),
    armOrder: Object.freeze(["exact", "k2", "k4"] as const),
  }),
  Object.freeze({
    shapeOrder: Object.freeze([3, 2, 1, 0]),
    armOrder: Object.freeze(["k2", "k4", "exact"] as const),
  }),
  Object.freeze({
    shapeOrder: Object.freeze([0, 1, 2, 3]),
    armOrder: Object.freeze(["k4", "exact", "k2"] as const),
  }),
  Object.freeze({
    shapeOrder: Object.freeze([3, 2, 1, 0]),
    armOrder: Object.freeze(["k4", "k2", "exact"] as const),
  }),
  Object.freeze({
    shapeOrder: Object.freeze([0, 1, 2, 3]),
    armOrder: Object.freeze(["k2", "exact", "k4"] as const),
  }),
  Object.freeze({
    shapeOrder: Object.freeze([3, 2, 1, 0]),
    armOrder: Object.freeze(["exact", "k4", "k2"] as const),
  }),
] as const);

const progress = requireElement<HTMLElement>("#progress");
const runButton = requireElement<HTMLButtonElement>("#run");
const result = requireElement<HTMLElement>("#result");
let active: PreparedHarness | undefined;
let running: PreparedHarness | undefined;
let started = false;

void prepareHarness().then(
  (prepared) => {
    if (prepared.correctness["passed"] !== true) {
      const cleanupFirst = prepared.destroy();
      const cleanupSecond = prepared.destroy();
      publish(Object.freeze({
        schema: RECEIPT_SCHEMA,
        experiment: EXPERIMENT_ID,
        status: "correctness-stop",
        passed: false,
        identity: prepared.identity,
        correctness: prepared.correctness,
        cleanup: Object.freeze({ firstCall: cleanupFirst, secondCall: cleanupSecond }),
        decision: "negative-stop-numerical-envelope",
      }), "failed");
      return;
    }
    active = prepared;
    document.body.dataset.status = "ready";
    progress.textContent =
      "READY — exact/K2/K4 full and adversarial gates passed; timing has not run";
    runButton.disabled = false;
  },
  (error: unknown) => fail(error),
);

runButton.addEventListener("click", () => {
  if (started || active === undefined) return;
  started = true;
  runButton.disabled = true;
  document.body.dataset.status = "running";
  const prepared = active;
  active = undefined;
  running = prepared;
  const timingEvidence: TimingFailureEvidence = { rawSamples: [] };
  void runTiming(prepared, timingEvidence).then(
    () => {
      running = undefined;
    },
    (error: unknown) => {
      const memoryBeforeCleanup = prepared.tracker.receipt();
      const cleanupStartedAtEpochMilliseconds = Date.now();
      const cleanupFirst = prepared.destroy();
      const cleanupSecond = prepared.destroy();
      const cleanupCompletedAtEpochMilliseconds = Date.now();
      running = undefined;
      fail(error, Object.freeze({
        stage: "timing",
        identity: prepared.identity,
        correctness: prepared.correctness,
        timing: Object.freeze({
          runStartedAtEpochMilliseconds:
            timingEvidence.runStartedAtEpochMilliseconds,
          measurementCompletedAtEpochMilliseconds:
            timingEvidence.measurementCompletedAtEpochMilliseconds,
          rawSamples: Object.freeze(timingEvidence.rawSamples.slice()),
        }),
        uncapturedGpuErrors: Object.freeze([...prepared.uncapturedErrors]),
        memoryBeforeCleanup,
        cleanup: Object.freeze({
          cleanupStartedAtEpochMilliseconds,
          cleanupCompletedAtEpochMilliseconds,
          firstCall: cleanupFirst,
          secondCall: cleanupSecond,
        }),
      }));
    },
  );
});

window.addEventListener("beforeunload", () => {
  active?.destroy();
  running?.destroy();
  active = undefined;
  running = undefined;
});

async function prepareHarness(): Promise<PreparedHarness> {
  requireLittleEndianHost();
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const maximumStorageBytes = maximumStorageBindingBytes();
  const maximumBufferBytes = maximumStorageBytes + 2 * STORAGE_GUARD_BYTES;
  const device = await adapter.requestDevice({
    label: "ace-opt-0074-webgpu-timestamp-profile-device",
    requiredFeatures: ["shader-f16", "subgroups", "timestamp-query"],
    requiredLimits: {
      maxBufferSize: maximumBufferBytes,
      maxStorageBufferBindingSize: maximumStorageBytes,
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 128,
    },
  });
  const tracker = new BufferTracker();
  const uncapturedErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncapturedErrors.push(event.error.message);
  });
  const capability = Object.freeze({
    subgroupMinSize: 32 as const,
    subgroupMaxSize: 32 as const,
  });
  const exactKernel = AceOpt0009DenseGemmKernel.create(device, capability);
  const k2Kernel = AceOpt0074DenseK2PartialsKernel.create(
    device,
    capability,
  );
  const k4Kernel = AceOpt0032DenseK4PartialsKernel.create(
    device,
    capability,
  );
  const timestampQuerySet = device.createQuerySet({
    label: "opt0074-dense-pass-timestamps",
    type: "timestamp",
    count: 2,
  });
  const timestampResolve = tracker.create(device, {
    label: "opt0074-timestamp-resolve",
    size: TIMESTAMP_QUERY_BYTES,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const timestampReadback = tracker.create(device, {
    label: "opt0074-timestamp-readback",
    size: TIMESTAMP_QUERY_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let destroyed = false;
  const destroy = (): Readonly<Record<string, unknown>> => {
    if (destroyed) {
      return Object.freeze({
        ...tracker.receipt(),
        idempotent: true,
        repeatedCall: true,
      });
    }
    destroyed = true;
    exactKernel.destroy();
    k2Kernel.destroy();
    k4Kernel.destroy();
    timestampQuerySet.destroy();
    tracker.destroyAll();
    device.destroy();
    return Object.freeze({
      ...tracker.receipt(),
      idempotent: true,
      repeatedCall: false,
      deviceDestroyed: true,
    });
  };
  try {
    const identity = await buildIdentity(adapter, device);
    const fullK2Aggregate = createAccumulator();
    const fullK4Aggregate = createAccumulator();
    const shapes: PreparedShape[] = [];
    for (const [index, spec] of FULL_SHAPES.entries()) {
      progress.textContent =
        `full-output exact/K2/K2-rerun/K4 ${index + 1}/4: ${spec.id}`;
      shapes.push(await prepareFullShape(
        device,
        tracker,
        exactKernel,
        k2Kernel,
        k4Kernel,
        spec,
        fullK2Aggregate,
        fullK4Aggregate,
      ));
      await yieldToBrowser();
    }
    if (fullK2Aggregate.count !== FULL_OUTPUT_COUNT ||
      fullK4Aggregate.count !== FULL_OUTPUT_COUNT) {
      throw new Error(
        `OPT-0074 full comparison counts ${fullK2Aggregate.count}/${fullK4Aggregate.count} != ${FULL_OUTPUT_COUNT}`,
      );
    }
    const adversarialK2Aggregate = createAccumulator();
    const adversarialK4Aggregate = createAccumulator();
    const adversarialReceipts: Readonly<Record<string, unknown>>[] = [];
    for (const [index, spec] of ADVERSARIAL_CASES.entries()) {
      progress.textContent =
        `bounded adversarial screen ${index + 1}/4: ${spec.id}`;
      adversarialReceipts.push(await runAdversarialCase(
        device,
        tracker,
        exactKernel,
        k2Kernel,
        k4Kernel,
        spec,
        adversarialK2Aggregate,
        adversarialK4Aggregate,
      ));
      await yieldToBrowser();
    }
    const fullK2Numerics = finalizeNumerics(fullK2Aggregate, "full");
    const fullK4Numerics = finalizeNumerics(fullK4Aggregate, "full");
    const adversarialK2Numerics = finalizeNumerics(
      adversarialK2Aggregate,
      "adversarial",
    );
    const adversarialK4Numerics = finalizeNumerics(
      adversarialK4Aggregate,
      "adversarial",
    );
    const k2NotWorseThanK4 = numericallyNoWorse(
      fullK2Numerics,
      fullK4Numerics,
    ) && numericallyNoWorse(adversarialK2Numerics, adversarialK4Numerics);
    const shapeReceipts = shapes.map((shape) => shape.correctness);
    const correctnessPassed = fullK2Numerics["passed"] === true &&
      adversarialK2Numerics["passed"] === true &&
      fullK4Numerics["passed"] === true &&
      adversarialK4Numerics["passed"] === true &&
      k2NotWorseThanK4 &&
      shapeReceipts.every((receipt) => receipt["passed"] === true) &&
      adversarialReceipts.every((receipt) => receipt["passed"] === true);

    if (correctnessPassed) {
      progress.textContent = "one balanced untimed warmup per arm and full shape";
      for (const [index, shape] of shapes.entries()) {
        const order: readonly Arm[] = index % 2 === 0
          ? ["exact", "k2", "k4"]
          : ["k4", "k2", "exact"];
        for (const arm of order) {
          await executeAndDrain(device, shape.dispatches[arm]);
        }
      }
    }
    await device.queue.onSubmittedWorkDone();
    await settlePostDrainEvents();
    if (uncapturedErrors.length !== 0) {
      throw new Error(
        `OPT-0074 observed ${uncapturedErrors.length} uncaptured GPU errors`,
      );
    }
    const correctness = Object.freeze({
      fullShapeCount: shapes.length,
      fullComparedU32CountPerCandidate: fullK2Aggregate.count,
      expectedFullComparedU32Count: FULL_OUTPUT_COUNT,
      fullK2DeterministicReruns: true,
      fullCompleteWrites: true,
      fullFiniteOutputs: true,
      fullNumerics: Object.freeze({ k2: fullK2Numerics, k4: fullK4Numerics }),
      fullCases: Object.freeze(shapeReceipts),
      adversarialCaseCount: adversarialReceipts.length,
      adversarialComparedU32CountPerCandidate: adversarialK2Aggregate.count,
      adversarialCompleteWrites: true,
      adversarialNumerics: Object.freeze({
        k2: adversarialK2Numerics,
        k4: adversarialK4Numerics,
      }),
      k2NotWorseThanK4,
      adversarialCases: Object.freeze(adversarialReceipts),
      uncapturedGpuErrorCount: uncapturedErrors.length,
      completedBeforeReady: true,
      passed: correctnessPassed,
    });
    return Object.freeze({
      adapter,
      device,
      tracker,
      exactKernel,
      k2Kernel,
      k4Kernel,
      timestampQuerySet,
      timestampResolve,
      timestampReadback,
      shapes: Object.freeze(shapes),
      correctness,
      identity,
      uncapturedErrors,
      destroy,
    });
  } catch (error) {
    await device.queue.onSubmittedWorkDone().catch(() => undefined);
    destroy();
    throw error;
  }
}

async function prepareFullShape(
  device: GPUDevice,
  tracker: BufferTracker,
  exactKernel: AceOpt0009DenseGemmKernel,
  k2Kernel: AceOpt0074DenseK2PartialsKernel,
  k4Kernel: AceOpt0032DenseK4PartialsKernel,
  spec: FullShapeSpec,
  k2Aggregate: NumericalAccumulator,
  k4Aggregate: NumericalAccumulator,
): Promise<PreparedShape> {
  const resources = await createCaseResources(device, tracker, spec);
  const guarded = createGuardedOutput(device, tracker, spec);
  try {
    const dispatches = await createDispatches(
      exactKernel,
      k2Kernel,
      k4Kernel,
      spec,
      resources.activation,
      resources.currentWeight,
      resources.candidateWeight,
      guarded.binding,
    );
    const correctness = await verifyCase(
      device,
      spec,
      guarded,
      dispatches,
      k2Aggregate,
      k4Aggregate,
      true,
      resources.hashes,
    );
    tracker.destroy(guarded.prefill);
    tracker.destroy(guarded.readback);
    return Object.freeze({ spec, dispatches, correctness });
  } catch (error) {
    tracker.destroy(resources.activation);
    tracker.destroy(resources.currentWeight);
    tracker.destroy(resources.candidateWeight);
    tracker.destroy(guarded.buffer);
    tracker.destroy(guarded.prefill);
    tracker.destroy(guarded.readback);
    throw error;
  }
}

async function runAdversarialCase(
  device: GPUDevice,
  tracker: BufferTracker,
  exactKernel: AceOpt0009DenseGemmKernel,
  k2Kernel: AceOpt0074DenseK2PartialsKernel,
  k4Kernel: AceOpt0032DenseK4PartialsKernel,
  spec: CaseSpec,
  k2Aggregate: NumericalAccumulator,
  k4Aggregate: NumericalAccumulator,
): Promise<Readonly<Record<string, unknown>>> {
  const resources = await createCaseResources(device, tracker, spec);
  const guarded = createGuardedOutput(device, tracker, spec);
  try {
    const dispatches = await createDispatches(
      exactKernel,
      k2Kernel,
      k4Kernel,
      spec,
      resources.activation,
      resources.currentWeight,
      resources.candidateWeight,
      guarded.binding,
    );
    return await verifyCase(
      device,
      spec,
      guarded,
      dispatches,
      k2Aggregate,
      k4Aggregate,
      false,
      resources.hashes,
    );
  } finally {
    tracker.destroy(resources.activation);
    tracker.destroy(resources.currentWeight);
    tracker.destroy(resources.candidateWeight);
    tracker.destroy(guarded.buffer);
    tracker.destroy(guarded.prefill);
    tracker.destroy(guarded.readback);
  }
}

async function createCaseResources(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: CaseSpec,
): Promise<Readonly<{
  activation: GPUBuffer;
  currentWeight: GPUBuffer;
  candidateWeight: GPUBuffer;
  hashes: Readonly<Record<string, string>>;
}>> {
  const activation = tracker.create(device, {
    label: `opt0074-${spec.id}-activation`,
    size: spec.shape.rows * spec.shape.inner * 4,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const currentWeight = tracker.create(device, {
    label: `opt0074-${spec.id}-exact-weight`,
    size: spec.shape.inner * spec.shape.columns * 2,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const candidateWeight = tracker.create(device, {
    label: `opt0074-${spec.id}-k2-k4-weight`,
    size: spec.shape.inner * spec.shape.columns * 2,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    const activationRange = activation.getMappedRange();
    fillActivation(new Float32Array(activationRange), spec);
    const activationHash = await sha256Bytes(new Uint8Array(activationRange));
    activation.unmap();
    const currentRange = currentWeight.getMappedRange();
    fillCurrentWeight(new Uint16Array(currentRange), spec);
    const currentWeightHash = await sha256Bytes(new Uint8Array(currentRange));
    currentWeight.unmap();
    const candidateRange = candidateWeight.getMappedRange();
    fillCandidateWeight(new Uint16Array(candidateRange), spec);
    const candidateWeightHash = await sha256Bytes(
      new Uint8Array(candidateRange),
    );
    candidateWeight.unmap();
    return Object.freeze({
      activation,
      currentWeight,
      candidateWeight,
      hashes: Object.freeze({
        activation: activationHash,
        currentWeight: currentWeightHash,
        candidateWeight: candidateWeightHash,
      }),
    });
  } catch (error) {
    if (activation.mapState === "mapped") activation.unmap();
    if (currentWeight.mapState === "mapped") currentWeight.unmap();
    if (candidateWeight.mapState === "mapped") candidateWeight.unmap();
    tracker.destroy(activation);
    tracker.destroy(currentWeight);
    tracker.destroy(candidateWeight);
    throw error;
  }
}

async function createDispatches(
  exactKernel: AceOpt0009DenseGemmKernel,
  k2Kernel: AceOpt0074DenseK2PartialsKernel,
  k4Kernel: AceOpt0032DenseK4PartialsKernel,
  spec: CaseSpec,
  activation: GPUBuffer,
  currentWeight: GPUBuffer,
  candidateWeight: GPUBuffer,
  output: GPUBufferBinding,
): Promise<Readonly<Record<Arm, DenseDispatch>>> {
  const activationBinding = binding(
    activation,
    spec.shape.rows * spec.shape.inner * 4,
  );
  const weightBytes = spec.shape.inner * spec.shape.columns * 2;
  const currentBindings: AceGemmBufferBindings = Object.freeze({
    activation: activationBinding,
    weight: binding(currentWeight, weightBytes),
    output,
  });
  const candidateBindings: AceGemmBufferBindings = Object.freeze({
    activation: activationBinding,
    weight: binding(candidateWeight, weightBytes),
    output,
  });
  const exact = await exactKernel.createDispatch(
    `opt0074-${spec.id}-exact-opt0009`,
    spec.shape,
    currentBindings,
  );
  const k2 = await k2Kernel.createDispatch(
    `opt0074-${spec.id}-candidate-k2`,
    spec.shape,
    candidateBindings,
  );
  const k4 = await k4Kernel.createDispatch(
    `opt0074-${spec.id}-reference-k4`,
    spec.shape,
    candidateBindings,
  );
  if (exact.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT ||
    k2.weightLayout !== ACE_OPT_0074_DENSE_K2_PARTIALS_WEIGHT_LAYOUT ||
    k4.weightLayout !== ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT) {
    throw new Error(`OPT-0074 ${spec.id} exact/K2/K4 weight layout changed`);
  }
  return Object.freeze({ exact, k2, k4 });
}

async function verifyCase(
  device: GPUDevice,
  spec: CaseSpec,
  output: GuardedOutput,
  dispatches: Readonly<Record<Arm, DenseDispatch>>,
  k2Aggregate: NumericalAccumulator,
  k4Aggregate: NumericalAccumulator,
  full: boolean,
  inputHashes: Readonly<Record<string, string>>,
): Promise<Readonly<Record<string, unknown>>> {
  let exact: OutputSnapshot | undefined = await executeCorrectness(
    device,
    output,
    dispatches.exact,
  );
  requireCompleteSnapshot(exact, `${spec.id} exact`, full);
  const exactHash = await sha256U32(exact.words);
  let k2: OutputSnapshot | undefined = await executeCorrectness(
    device,
    output,
    dispatches.k2,
  );
  requireCompleteSnapshot(k2, `${spec.id} K2`, full);
  const k2Hash = await sha256U32(k2.words);
  const localK2 = createAccumulator();
  accumulateNumerics(localK2, exact.words, k2.words, spec);
  mergeAccumulator(k2Aggregate, localK2);
  const k2Numerics = finalizeNumerics(
    localK2,
    full ? "full" : "adversarial",
  );
  const k2Rerun = await executeCorrectness(
    device,
    output,
    dispatches.k2,
  );
  requireCompleteSnapshot(k2Rerun, `${spec.id} K2 rerun`, full);
  requireExactWords(k2.words, k2Rerun.words, `${spec.id} K2 rerun`);
  const k2RerunHash = await sha256U32(k2Rerun.words);
  k2 = undefined;
  let k4: OutputSnapshot | undefined = await executeCorrectness(
    device,
    output,
    dispatches.k4,
  );
  requireCompleteSnapshot(k4, `${spec.id} K4`, full);
  const k4Hash = await sha256U32(k4.words);
  const localK4 = createAccumulator();
  accumulateNumerics(localK4, exact.words, k4.words, spec);
  mergeAccumulator(k4Aggregate, localK4);
  const k4Numerics = finalizeNumerics(
    localK4,
    full ? "full" : "adversarial",
  );
  exact = undefined;
  k4 = undefined;
  return Object.freeze({
    id: spec.id,
    fixtureKind: spec.fixtureKind,
    shape: spec.shape,
    outputU32Count: output.outputElements,
    executionOrder: Object.freeze(["exact", "k2", "k2", "k4"]),
    completeWriteCount: 4,
    k2DeterministicRawU32: true,
    outputHashes: Object.freeze({
      exact: exactHash,
      k2: k2Hash,
      k2Rerun: k2RerunHash,
      k4: k4Hash,
    }),
    inputHashes,
    nonFiniteCount: 0,
    qNaNPrefillCount: 0,
    canariesIntact: true,
    tailRowWritten: true,
    numerics: Object.freeze({ k2: k2Numerics, k4: k4Numerics }),
    k2NotWorseThanK4: numericallyNoWorse(k2Numerics, k4Numerics),
    passed: k2Numerics["passed"] === true &&
      k4Numerics["passed"] === true &&
      numericallyNoWorse(k2Numerics, k4Numerics),
  });
}

async function executeCorrectness(
  device: GPUDevice,
  output: GuardedOutput,
  dispatch: DenseDispatch,
): Promise<OutputSnapshot> {
  const encoder = device.createCommandEncoder({
    label: `${dispatch.label}-correctness`,
  });
  encoder.copyBufferToBuffer(
    output.prefill,
    0,
    output.buffer,
    0,
    output.totalBytes,
  );
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  encoder.copyBufferToBuffer(
    output.buffer,
    0,
    output.readback,
    0,
    output.totalBytes,
  );
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
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
    const tailStart = words.length - output.columns;
    let tailRowWritten = true;
    for (let index = tailStart; index < words.length; index += 1) {
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

async function runTiming(
  prepared: PreparedHarness,
  failureEvidence: TimingFailureEvidence,
): Promise<void> {
  const samples = new Map<string, Record<Arm, TimestampSample[]>>(
    prepared.shapes.map(({ spec }) => [
      spec.id,
      { exact: [], k2: [], k4: [] },
    ]),
  );
  const rawSamples = failureEvidence.rawSamples;
  const runStartedAtEpochMilliseconds = Date.now();
  failureEvidence.runStartedAtEpochMilliseconds =
    runStartedAtEpochMilliseconds;
  for (const [roundIndex, round] of TIMING_ROUNDS.entries()) {
    for (const shapeIndex of round.shapeOrder) {
      const shape = prepared.shapes[shapeIndex];
      if (shape === undefined) throw new Error("OPT-0074 timing shape changed");
      for (const [armPosition, arm] of round.armOrder.entries()) {
        progress.textContent =
          `timing round ${roundIndex + 1}/${TIMING_ROUNDS.length}: ${shape.spec.id} ${arm}`;
        const sample = await timeDispatch(
          prepared.device,
          shape.dispatches[arm],
          prepared.timestampQuerySet,
          prepared.timestampResolve,
          prepared.timestampReadback,
          shape.spec,
          arm,
        );
        samples.get(shape.spec.id)![arm].push(sample);
        rawSamples.push(Object.freeze({
          roundIndex,
          shapeIndex,
          shapeId: shape.spec.id,
          armPosition,
          arm,
          ...sample,
        }));
      }
      await yieldToBrowser();
    }
  }
  await prepared.device.queue.onSubmittedWorkDone();
  await settlePostDrainEvents();
  if (prepared.uncapturedErrors.length !== 0) {
    throw new Error("OPT-0074 observed an uncaptured timing GPU error");
  }
  const timingInputs: TimingInput[] = prepared.shapes.map(({ spec }) => {
    const entry = samples.get(spec.id);
    if (entry === undefined) throw new Error("OPT-0074 timing sample missing");
    return Object.freeze({
      id: spec.id,
      samples: Object.freeze({
        exact: Object.freeze(entry.exact.slice()),
        k2: Object.freeze(entry.k2.slice()),
        k4: Object.freeze(entry.k4.slice()),
      }),
    });
  });
  const measurementCompletedAtEpochMilliseconds = Date.now();
  failureEvidence.measurementCompletedAtEpochMilliseconds =
    measurementCompletedAtEpochMilliseconds;
  const timing = summarizeOpt0074Timing(timingInputs);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanupStartedAtEpochMilliseconds = Date.now();
  const cleanupFirst = prepared.destroy();
  const cleanupSecond = prepared.destroy();
  const cleanupCompletedAtEpochMilliseconds = Date.now();
  const receipt = Object.freeze({
    schema: RECEIPT_SCHEMA,
    experiment: EXPERIMENT_ID,
    status: "completed",
    passed: timing["passed"] === true,
    identity: prepared.identity,
    correctness: prepared.correctness,
    protocol: Object.freeze({
      fullOutputAndAdversarialCorrectnessBeforeReady: true,
      oneUntimedWarmupPerArmPerFullShape: true,
      timingButtonCount: 1,
      rounds: TIMING_ROUNDS.length,
      balancedOrders: TIMING_ROUNDS,
      samplesPerArmPerShape: TIMING_ROUNDS.length,
      oneCommandBufferPerSample: true,
      oneMatchingQueueDrainPerSample: true,
      oneTimestampQueryPairPerSample: true,
      timestampWritesBoundComputePassExactly: true,
      resolveAndCopyAfterComputePass: true,
      timestampReadbackAfterSubmitToDrainWall: true,
      outputReadbackInsideTiming: false,
      weightedProductionMultiplicities: "4/2/2/1",
      arithmeticConvention: "two FLOPs per multiply-add",
      externalAgxUtilizationSampling: Object.freeze({
        capturedByPage: false,
        requiredForFinalDiagnosticInterpretation: true,
        cadence: "coarse external ioreg polling",
        placement: "outside every browser timed interval",
        alignment: "epoch-millisecond bounds in this receipt",
      }),
    }),
    timing: Object.freeze({
      ...timing,
      runStartedAtEpochMilliseconds,
      measurementCompletedAtEpochMilliseconds,
      runCompletedAtEpochMilliseconds: cleanupCompletedAtEpochMilliseconds,
      rawSamples: Object.freeze(rawSamples),
    }),
    decision: Object.freeze({
      disposition: timing["passed"] === true
        ? "positive-in-page-candidate-pending-external-thermal-audit"
        : "negative-stop-primitive-gate",
      primitiveFollowUpAuthorized: false,
      externalThermalGateAuditedByPage: false,
      productionIntegrationAuthorized: false,
      packageChangeAuthorized: false,
      trajectoryOrListeningClaim: false,
    }),
    memoryBeforeCleanup,
    cleanup: Object.freeze({
      cleanupStartedAtEpochMilliseconds,
      cleanupCompletedAtEpochMilliseconds,
      firstCall: cleanupFirst,
      secondCall: cleanupSecond,
      idempotent: true,
      zeroLiveBuffers: cleanupFirst["liveBufferCount"] === 0 &&
        cleanupSecond["liveBufferCount"] === 0,
    }),
  });
  publish(receipt, receipt.passed ? "passed" : "failed");
  progress.textContent = receipt.passed
    ? "completed — K2 cleared in-page gates; external thermal audit pending"
    : "failed — K2 did not clear every primitive timing gate";
}

export function summarizeOpt0074Timing(
  inputs: readonly TimingInput[],
): Readonly<Record<string, unknown>> {
  if (inputs.length !== FULL_SHAPES.length) {
    throw new Error("OPT-0074 timing requires all four production shapes");
  }
  const weightedGpuMilliseconds: Record<Arm, number> = {
    exact: 0,
    k2: 0,
    k4: 0,
  };
  const weightedWallMilliseconds: Record<Arm, number> = {
    exact: 0,
    k2: 0,
    k4: 0,
  };
  const weightedValidMultiplyAdds: Record<Arm, number> = {
    exact: 0,
    k2: 0,
    k4: 0,
  };
  const weightedScheduledMultiplyAdds: Record<Arm, number> = {
    exact: 0,
    k2: 0,
    k4: 0,
  };
  const strata = inputs.map((input, index) => {
    const spec = FULL_SHAPES[index];
    if (spec === undefined || spec.id !== input.id) {
      throw new Error("OPT-0074 timing shape order changed");
    }
    const armSummaries = Object.fromEntries(
      (["exact", "k2", "k4"] as const).map((arm) => {
        const armSamples = input.samples[arm];
        requireTimestampSamples(armSamples, `${spec.id} ${arm}`);
        const gpuMilliseconds = meanSamples(
          armSamples.map((sample) => sample.gpuMilliseconds),
          `${spec.id} ${arm} GPU`,
        );
        const wallMilliseconds = meanSamples(
          armSamples.map((sample) => sample.wallMilliseconds),
          `${spec.id} ${arm} wall`,
        );
        const validMultiplyAdds = armSamples[0]!.validMultiplyAdds;
        const scheduledMultiplyAdds = armSamples[0]!.scheduledMultiplyAdds;
        if (armSamples.some((sample) =>
          sample.validMultiplyAdds !== validMultiplyAdds ||
          sample.scheduledMultiplyAdds !== scheduledMultiplyAdds
        )) {
          throw new Error(`OPT-0074 ${spec.id} ${arm} work changed between rounds`);
        }
        weightedGpuMilliseconds[arm] +=
          gpuMilliseconds * spec.productionMultiplicity;
        weightedWallMilliseconds[arm] +=
          wallMilliseconds * spec.productionMultiplicity;
        weightedValidMultiplyAdds[arm] +=
          validMultiplyAdds * spec.productionMultiplicity;
        weightedScheduledMultiplyAdds[arm] +=
          scheduledMultiplyAdds * spec.productionMultiplicity;
        return [arm, Object.freeze({
          samples: armSamples,
          meanGpuMilliseconds: gpuMilliseconds,
          meanWallMilliseconds: wallMilliseconds,
          medianGpuMilliseconds: medianSamples(
            armSamples.map((sample) => sample.gpuMilliseconds),
            `${spec.id} ${arm} GPU median`,
          ),
          medianWallMilliseconds: medianSamples(
            armSamples.map((sample) => sample.wallMilliseconds),
            `${spec.id} ${arm} wall median`,
          ),
          minimumGpuMilliseconds: Math.min(
            ...armSamples.map((sample) => sample.gpuMilliseconds),
          ),
          maximumGpuMilliseconds: Math.max(
            ...armSamples.map((sample) => sample.gpuMilliseconds),
          ),
          minimumWallMilliseconds: Math.min(
            ...armSamples.map((sample) => sample.wallMilliseconds),
          ),
          maximumWallMilliseconds: Math.max(
            ...armSamples.map((sample) => sample.wallMilliseconds),
          ),
          meanGpuToWallRatio: gpuMilliseconds / wallMilliseconds,
          meanWallMinusGpuMilliseconds: wallMilliseconds - gpuMilliseconds,
          validMultiplyAdds,
          scheduledMultiplyAdds,
          validGpuTflops: tflops(validMultiplyAdds, gpuMilliseconds),
          scheduledGpuTflops: tflops(scheduledMultiplyAdds, gpuMilliseconds),
          validWallTflops: tflops(validMultiplyAdds, wallMilliseconds),
          scheduledWallTflops: tflops(scheduledMultiplyAdds, wallMilliseconds),
        })];
      }),
    );
    return Object.freeze({
      id: spec.id,
      shape: spec.shape,
      multiplicity: spec.productionMultiplicity,
      arms: Object.freeze(armSummaries),
    });
  });
  const weightedArms = Object.fromEntries(
    (["exact", "k2", "k4"] as const).map((arm) => [arm, Object.freeze({
      gpuMilliseconds: weightedGpuMilliseconds[arm],
      wallMilliseconds: weightedWallMilliseconds[arm],
      gpuToWallRatio:
        weightedGpuMilliseconds[arm] / weightedWallMilliseconds[arm],
      wallMinusGpuMilliseconds:
        weightedWallMilliseconds[arm] - weightedGpuMilliseconds[arm],
      validMultiplyAdds: weightedValidMultiplyAdds[arm],
      scheduledMultiplyAdds: weightedScheduledMultiplyAdds[arm],
      validGpuTflops: tflops(
        weightedValidMultiplyAdds[arm],
        weightedGpuMilliseconds[arm],
      ),
      scheduledGpuTflops: tflops(
        weightedScheduledMultiplyAdds[arm],
        weightedGpuMilliseconds[arm],
      ),
      validWallTflops: tflops(
        weightedValidMultiplyAdds[arm],
        weightedWallMilliseconds[arm],
      ),
      scheduledWallTflops: tflops(
        weightedScheduledMultiplyAdds[arm],
        weightedWallMilliseconds[arm],
      ),
    })]),
  );
  const k2GpuSpeedup = weightedGpuMilliseconds.exact /
    weightedGpuMilliseconds.k2;
  const k2WallSpeedup = weightedWallMilliseconds.exact /
    weightedWallMilliseconds.k2;
  const weightedWallSavingMilliseconds = weightedWallMilliseconds.exact -
    weightedWallMilliseconds.k2;
  const shapeConsistency = strata.map((stratum) => {
    const arms = stratum.arms as unknown as Readonly<Record<Arm, Readonly<{
      samples: readonly TimestampSample[];
      meanGpuMilliseconds: number;
      meanWallMilliseconds: number;
      minimumGpuMilliseconds: number;
      maximumGpuMilliseconds: number;
      minimumWallMilliseconds: number;
      maximumWallMilliseconds: number;
    }>>>;
    const pairedRounds = arms.exact.samples.map((exact, roundIndex) => {
      const k2 = arms.k2.samples[roundIndex];
      if (k2 === undefined) {
        throw new Error(`OPT-0074 ${stratum.id} K2 paired round missing`);
      }
      return Object.freeze({
        roundIndex,
        exactWallMilliseconds: exact.wallMilliseconds,
        k2WallMilliseconds: k2.wallMilliseconds,
        wallSpeedup: exact.wallMilliseconds / k2.wallMilliseconds,
        exactGpuMilliseconds: exact.gpuMilliseconds,
        k2GpuMilliseconds: k2.gpuMilliseconds,
        gpuSpeedup: exact.gpuMilliseconds / k2.gpuMilliseconds,
        wallWin: k2.wallMilliseconds < exact.wallMilliseconds,
        gpuWin: k2.gpuMilliseconds < exact.gpuMilliseconds,
      });
    });
    const wallRangesSeparated = arms.k2.maximumWallMilliseconds <
      arms.exact.minimumWallMilliseconds;
    const gpuRangesSeparated = arms.k2.maximumGpuMilliseconds <
      arms.exact.minimumGpuMilliseconds;
    const everyPairedWallWin = pairedRounds.every((round) => round.wallWin);
    const everyPairedGpuWin = pairedRounds.every((round) => round.gpuWin);
    return Object.freeze({
      id: stratum.id,
      pairedRounds: Object.freeze(pairedRounds),
      meanWallWin:
        arms.k2.meanWallMilliseconds < arms.exact.meanWallMilliseconds,
      meanGpuWin:
        arms.k2.meanGpuMilliseconds < arms.exact.meanGpuMilliseconds,
      everyPairedWallWin,
      everyPairedGpuWin,
      wallRangesSeparated,
      gpuRangesSeparated,
      passed: everyPairedWallWin && everyPairedGpuWin &&
        wallRangesSeparated && gpuRangesSeparated,
    });
  });
  const weightedRounds = Array.from(
    { length: TIMING_ROUNDS.length },
    (_, roundIndex) => {
      const arms = Object.fromEntries(
        (["exact", "k2", "k4"] as const).map((arm) => {
          let gpuMilliseconds = 0;
          let wallMilliseconds = 0;
          for (const [shapeIndex, input] of inputs.entries()) {
            const spec = FULL_SHAPES[shapeIndex];
            const sample = input.samples[arm][roundIndex];
            if (spec === undefined || sample === undefined) {
              throw new Error(
                `OPT-0074 weighted round ${roundIndex} ${arm} sample missing`,
              );
            }
            gpuMilliseconds += sample.gpuMilliseconds *
              spec.productionMultiplicity;
            wallMilliseconds += sample.wallMilliseconds *
              spec.productionMultiplicity;
          }
          return [arm, Object.freeze({ gpuMilliseconds, wallMilliseconds })];
        }),
      ) as Readonly<Record<Arm, Readonly<{
        gpuMilliseconds: number;
        wallMilliseconds: number;
      }>>>;
      return Object.freeze({
        roundIndex,
        arms: Object.freeze(arms),
        k2WallSpeedup:
          arms.exact.wallMilliseconds / arms.k2.wallMilliseconds,
        k2GpuSpeedup:
          arms.exact.gpuMilliseconds / arms.k2.gpuMilliseconds,
        k2WallWin: arms.k2.wallMilliseconds < arms.exact.wallMilliseconds,
        k2GpuWin: arms.k2.gpuMilliseconds < arms.exact.gpuMilliseconds,
      });
    },
  );
  const weightedExactWalls = weightedRounds.map((round) =>
    round.arms.exact.wallMilliseconds
  );
  const weightedK2Walls = weightedRounds.map((round) =>
    round.arms.k2.wallMilliseconds
  );
  const weightedExactGpu = weightedRounds.map((round) =>
    round.arms.exact.gpuMilliseconds
  );
  const weightedK2Gpu = weightedRounds.map((round) =>
    round.arms.k2.gpuMilliseconds
  );
  const weightedMedianWallSpeedup = medianSamples(
    weightedExactWalls,
    "weighted exact wall median",
  ) / medianSamples(weightedK2Walls, "weighted K2 wall median");
  const weightedMedianGpuSpeedup = medianSamples(
    weightedExactGpu,
    "weighted exact GPU median",
  ) / medianSamples(weightedK2Gpu, "weighted K2 GPU median");
  const weightedMedianWallSavingMilliseconds = medianSamples(
    weightedExactWalls,
    "weighted exact wall median saving",
  ) - medianSamples(weightedK2Walls, "weighted K2 wall median saving");
  const weightedMedianGpuSavingMilliseconds = medianSamples(
    weightedExactGpu,
    "weighted exact GPU median saving",
  ) - medianSamples(weightedK2Gpu, "weighted K2 GPU median saving");
  const weightedWallRangesSeparated = Math.max(...weightedK2Walls) <
    Math.min(...weightedExactWalls);
  const weightedGpuRangesSeparated = Math.max(...weightedK2Gpu) <
    Math.min(...weightedExactGpu);
  const everyWeightedRoundWallWin = weightedRounds.every((round) =>
    round.k2WallWin
  );
  const everyWeightedRoundGpuWin = weightedRounds.every((round) =>
    round.k2GpuWin
  );
  const weightedGpuSavingMilliseconds = weightedGpuMilliseconds.exact -
    weightedGpuMilliseconds.k2;
  const gates = Object.freeze({
    requiredWeightedWallSpeedup: REQUIRED_WEIGHTED_WALL_SPEEDUP,
    observedWeightedWallSpeedup: k2WallSpeedup,
    weightedWallSpeedupPassed:
      k2WallSpeedup >= REQUIRED_WEIGHTED_WALL_SPEEDUP,
    requiredWeightedWallSavingMilliseconds:
      REQUIRED_WEIGHTED_WALL_SAVING_MILLISECONDS,
    observedWeightedWallSavingMilliseconds: weightedWallSavingMilliseconds,
    weightedWallSavingPassed: weightedWallSavingMilliseconds >=
      REQUIRED_WEIGHTED_WALL_SAVING_MILLISECONDS,
    requiredWeightedGpuSpeedup: REQUIRED_WEIGHTED_GPU_SPEEDUP,
    observedWeightedGpuSpeedup: k2GpuSpeedup,
    weightedGpuSpeedupPassed:
      k2GpuSpeedup >= REQUIRED_WEIGHTED_GPU_SPEEDUP,
    requiredWeightedGpuSavingMilliseconds:
      REQUIRED_WEIGHTED_GPU_SAVING_MILLISECONDS,
    observedWeightedGpuSavingMilliseconds: weightedGpuSavingMilliseconds,
    weightedGpuSavingPassed: weightedGpuSavingMilliseconds >=
      REQUIRED_WEIGHTED_GPU_SAVING_MILLISECONDS,
    observedWeightedMedianWallSpeedup: weightedMedianWallSpeedup,
    weightedMedianWallSpeedupPassed:
      weightedMedianWallSpeedup >= REQUIRED_WEIGHTED_WALL_SPEEDUP,
    observedWeightedMedianGpuSpeedup: weightedMedianGpuSpeedup,
    weightedMedianGpuSpeedupPassed:
      weightedMedianGpuSpeedup >= REQUIRED_WEIGHTED_GPU_SPEEDUP,
    observedWeightedMedianWallSavingMilliseconds:
      weightedMedianWallSavingMilliseconds,
    weightedMedianWallSavingPassed: weightedMedianWallSavingMilliseconds >=
      REQUIRED_WEIGHTED_WALL_SAVING_MILLISECONDS,
    observedWeightedMedianGpuSavingMilliseconds:
      weightedMedianGpuSavingMilliseconds,
    weightedMedianGpuSavingPassed: weightedMedianGpuSavingMilliseconds >=
      REQUIRED_WEIGHTED_GPU_SAVING_MILLISECONDS,
    everyWeightedRoundWallWin,
    everyWeightedRoundGpuWin,
    weightedWallRangesSeparated,
    weightedGpuRangesSeparated,
    everyShapeConsistentAndNonOverlapped:
      shapeConsistency.every((shape) => shape.passed),
  });
  return Object.freeze({
    strata: Object.freeze(strata),
    shapeConsistency: Object.freeze(shapeConsistency),
    weightedArms: Object.freeze(weightedArms),
    weightedRounds: Object.freeze(weightedRounds),
    k2Speedup: Object.freeze({
      gpu: k2GpuSpeedup,
      wall: k2WallSpeedup,
    }),
    k4ReferenceSpeedup: Object.freeze({
      gpu: weightedGpuMilliseconds.exact /
        weightedGpuMilliseconds.k4,
      wall: weightedWallMilliseconds.exact /
        weightedWallMilliseconds.k4,
    }),
    gates,
    passed: gates.weightedWallSpeedupPassed &&
      gates.weightedWallSavingPassed &&
      gates.weightedGpuSpeedupPassed &&
      gates.weightedGpuSavingPassed &&
      gates.weightedMedianWallSpeedupPassed &&
      gates.weightedMedianGpuSpeedupPassed &&
      gates.weightedMedianWallSavingPassed &&
      gates.weightedMedianGpuSavingPassed &&
      gates.everyWeightedRoundWallWin &&
      gates.everyWeightedRoundGpuWin &&
      gates.weightedWallRangesSeparated &&
      gates.weightedGpuRangesSeparated &&
      gates.everyShapeConsistentAndNonOverlapped,
  });
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
  spec: CaseSpec,
): void {
  if (controlWords.length !== candidateWords.length) {
    throw new Error(`OPT-0074 ${spec.id} output length changed`);
  }
  const control = new Float32Array(
    controlWords.buffer,
    controlWords.byteOffset,
    controlWords.length,
  );
  const candidate = new Float32Array(
    candidateWords.buffer,
    candidateWords.byteOffset,
    candidateWords.length,
  );
  for (let index = 0; index < control.length; index += 1) {
    const a = control[index]!;
    const b = candidate[index]!;
    const aWord = controlWords[index]!;
    const bWord = candidateWords[index]!;
    accumulator.count += 1;
    if (!Number.isFinite(a)) accumulator.controlNonFiniteCount += 1;
    if (!Number.isFinite(b)) accumulator.candidateNonFiniteCount += 1;
    const aClass = f32Class(aWord);
    const bClass = f32Class(bWord);
    if (aClass !== bClass) {
      accumulator.classChangeCount += 1;
      const transition = `${aClass}->${bClass}`;
      accumulator.classChanges[transition] =
        (accumulator.classChanges[transition] ?? 0) + 1;
    }
    if (aWord !== bWord) {
      accumulator.differingU32Count += 1;
      accumulator.firstDifference ??= Object.freeze({
        shapeId: spec.id,
        index,
        row: Math.floor(index / spec.shape.columns),
        column: index % spec.shape.columns,
        control: a,
        candidate: b,
        controlU32: aWord,
        candidateU32: bWord,
      });
    }
    if ((aWord & 0x7fff_ffff) === 0 && (bWord & 0x7fff_ffff) === 0 &&
      aWord !== bWord) {
      accumulator.signedZeroDifferenceCount += 1;
    }
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    const error = b - a;
    const absoluteError = Math.abs(error);
    const relativeError = absoluteError / Math.max(Math.abs(a), 1e-6);
    accumulator.finiteCount += 1;
    accumulator.controlSum += a;
    accumulator.candidateSum += b;
    accumulator.controlSquareSum += a * a;
    accumulator.candidateSquareSum += b * b;
    accumulator.crossSum += a * b;
    accumulator.errorSum += error;
    accumulator.absoluteErrorSum += absoluteError;
    accumulator.errorSquareSum += error * error;
    accumulator.relativeErrorSquareSum += relativeError * relativeError;
    accumulator.maximumAbsoluteControl = Math.max(
      accumulator.maximumAbsoluteControl,
      Math.abs(a),
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
        control: a,
        candidate: b,
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
    "count",
    "finiteCount",
    "controlNonFiniteCount",
    "candidateNonFiniteCount",
    "differingU32Count",
    "signedZeroDifferenceCount",
    "classChangeCount",
    "controlSum",
    "candidateSum",
    "controlSquareSum",
    "candidateSquareSum",
    "crossSum",
    "errorSum",
    "absoluteErrorSum",
    "errorSquareSum",
    "relativeErrorSquareSum",
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

function finalizeNumerics(
  accumulator: NumericalAccumulator,
  envelope: "full" | "adversarial",
): Readonly<Record<string, unknown>> {
  if (accumulator.count === 0 || accumulator.finiteCount === 0) {
    throw new Error("OPT-0074 cannot finalize an empty comparison");
  }
  const count = accumulator.finiteCount;
  const controlRms = Math.sqrt(accumulator.controlSquareSum / count);
  const rmsError = Math.sqrt(accumulator.errorSquareSum / count);
  const relativeRmsError = Math.sqrt(
    accumulator.relativeErrorSquareSum / count,
  );
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
  const pearson = pearsonDenominator === 0
    ? accumulator.errorSquareSum === 0 ? 1 : 0
    : covariance / pearsonDenominator;
  const thresholds = envelope === "full"
    ? Object.freeze({
      nrmseMaximum: FULL_NRMSE_MAXIMUM,
      snrDecibelsMinimum: FULL_SNR_DECIBELS_MINIMUM,
      pearsonMinimum: FULL_PEARSON_MINIMUM,
      maximumAbsoluteErrorMaximum: FULL_MAXIMUM_ABSOLUTE_ERROR,
      relativeRmsErrorMaximum: Number.POSITIVE_INFINITY,
    })
    : Object.freeze({
      nrmseMaximum: ADVERSARIAL_NRMSE_MAXIMUM,
      snrDecibelsMinimum: ADVERSARIAL_SNR_DECIBELS_MINIMUM,
      pearsonMinimum: ADVERSARIAL_PEARSON_MINIMUM,
      maximumAbsoluteErrorMaximum: ADVERSARIAL_MAXIMUM_ABSOLUTE_ERROR,
      relativeRmsErrorMaximum: Number.POSITIVE_INFINITY,
    });
  const passed = accumulator.controlNonFiniteCount === 0 &&
    accumulator.candidateNonFiniteCount === 0 &&
    nrmse <= thresholds.nrmseMaximum &&
    snrDecibels >= thresholds.snrDecibelsMinimum &&
    pearson >= thresholds.pearsonMinimum &&
    accumulator.maximumAbsoluteError <=
      thresholds.maximumAbsoluteErrorMaximum;
  return Object.freeze({
    envelope,
    count: accumulator.count,
    finiteCount: accumulator.finiteCount,
    differingU32Count: accumulator.differingU32Count,
    signedZeroDifferenceCount: accumulator.signedZeroDifferenceCount,
    classChangeCount: accumulator.classChangeCount,
    classChanges: Object.freeze({ ...accumulator.classChanges }),
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
    pearsonCorrelation: pearson,
    maximumAbsoluteControl: accumulator.maximumAbsoluteControl,
    maximumAbsoluteError: accumulator.maximumAbsoluteError,
    maximumRelativeError: accumulator.maximumRelativeError,
    firstDifference: accumulator.firstDifference,
    worstDifference: accumulator.worstDifference,
    thresholds,
    passed,
  });
}

function numericallyNoWorse(
  k2: Readonly<Record<string, unknown>>,
  k4: Readonly<Record<string, unknown>>,
): boolean {
  return requireNumericMetric(k2, "nrmse") <=
      requireNumericMetric(k4, "nrmse") &&
    requireNumericMetric(k2, "snrDecibels", true) >=
      requireNumericMetric(k4, "snrDecibels", true) &&
    requireNumericMetric(k2, "pearsonCorrelation") >=
      requireNumericMetric(k4, "pearsonCorrelation") &&
    requireNumericMetric(k2, "maximumAbsoluteError") <=
      requireNumericMetric(k4, "maximumAbsoluteError") &&
    requireNumericMetric(k2, "candidateNonFiniteCount") <=
      requireNumericMetric(k4, "candidateNonFiniteCount") &&
    requireNumericMetric(k2, "classChangeCount") <=
      requireNumericMetric(k4, "classChangeCount") &&
    requireNumericMetric(k2, "signedZeroDifferenceCount") <=
      requireNumericMetric(k4, "signedZeroDifferenceCount");
}

function requireNumericMetric(
  receipt: Readonly<Record<string, unknown>>,
  key: string,
  allowPositiveInfinity = false,
): number {
  const value = receipt[key];
  if (allowPositiveInfinity && value === "positive-infinity") {
    return Number.POSITIVE_INFINITY;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`OPT-0074 numerical receipt has invalid ${key}`);
  }
  return value;
}

function fillActivation(values: Float32Array, spec: CaseSpec): void {
  let physical = 0;
  for (let row = 0; row < spec.shape.rows; row += 1) {
    for (let inner = 0; inner < spec.shape.inner; inner += 1) {
      values[physical] = halfToNumber(activationBitsAt(spec, row, inner));
      physical += 1;
    }
  }
}

function fillCurrentWeight(values: Uint16Array, spec: CaseSpec): void {
  const innerTiles = spec.shape.inner / 32;
  let physical = 0;
  for (let columnTile = 0; columnTile < spec.shape.columns / 256;
    columnTile += 1) {
    for (let innerTile = 0; innerTile < innerTiles; innerTile += 1) {
      for (let innerInTile = 0; innerInTile < 32; innerInTile += 1) {
        const inner = innerTile * 32 + innerInTile;
        for (let columnInTile = 0; columnInTile < 256;
          columnInTile += 1) {
          const column = columnTile * 256 + columnInTile;
          values[physical] = weightBitsAt(spec, inner, column);
          physical += 1;
        }
      }
    }
  }
}

function fillCandidateWeight(values: Uint16Array, spec: CaseSpec): void {
  let physical = 0;
  for (let columnTile = 0; columnTile < spec.shape.columns / 128;
    columnTile += 1) {
    for (let innerK4 = 0; innerK4 < spec.shape.inner / 4; innerK4 += 1) {
      for (let outputInLane = 0; outputInLane < 4; outputInLane += 1) {
        for (let lane = 0; lane < 32; lane += 1) {
          const column = columnTile * 128 + lane * 4 + outputInLane;
          for (let innerInK4 = 0; innerInK4 < 4; innerInK4 += 1) {
            values[physical] = weightBitsAt(
              spec,
              innerK4 * 4 + innerInK4,
              column,
            );
            physical += 1;
          }
        }
      }
    }
  }
}

function activationBitsAt(spec: CaseSpec, row: number, inner: number): number {
  const group = inner >>> 2;
  const offset = inner & 3;
  const mixed = mix32(
    0x3141_5926 ^ Math.imul(spec.ordinal + 1, 0x9e37_79b1) ^
      Math.imul(row + 1, 0x85eb_ca6b) ^ Math.imul(group + 1, 0xc2b2_ae35) ^
      Math.imul(offset + 1, 0x27d4_eb2f),
  );
  if (spec.fixtureKind === "signed-zero") {
    return (mixed >>> 31) << 15;
  }
  if (spec.fixtureKind === "range") {
    const magnitude = [0x1800, 0x2401, 0x3c00, 0x4400][inner & 3]!;
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

function weightBitsAt(spec: CaseSpec, inner: number, column: number): number {
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
  const magnitude = FINITE_HALF_MAGNITUDES[
    mixed % FINITE_HALF_MAGNITUDES.length
  ]!;
  if (spec.fixtureKind === "cancellation") {
    const withinK4Sign = offset === 1 || offset === 2 ? 1 : 0;
    return magnitude | ((((column & 1) ^ withinK4Sign) & 1) << 15);
  }
  return magnitude | ((mixed >>> 31) << 15);
}

function createGuardedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: CaseSpec,
): GuardedOutput {
  const outputElements = spec.shape.rows * spec.shape.columns;
  const outputBytes = outputElements * 4;
  const totalBytes = outputBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = tracker.create(device, {
    label: `opt0074-${spec.id}-prefill`,
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
    label: `opt0074-${spec.id}-output`,
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  const readback = tracker.create(device, {
    label: `opt0074-${spec.id}-readback`,
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
    outputElements,
    columns: spec.shape.columns,
    outputBytes,
    totalBytes,
  });
}

async function buildIdentity(
  adapter: GPUAdapter,
  device: GPUDevice,
): Promise<Readonly<Record<string, unknown>>> {
  const requestedMaximumStorageBytes = maximumStorageBindingBytes();
  const requestedMaximumBufferBytes = requestedMaximumStorageBytes +
    2 * STORAGE_GUARD_BYTES;
  const generatedShaders = [];
  for (const spec of [...FULL_SHAPES, ...ADVERSARIAL_CASES]) {
    const exactPlan = planAceOpt0009DenseGemm(spec.shape);
    const partialPlan = planAceOpt0032DenseK4Partials(spec.shape);
    generatedShaders.push(Object.freeze({
      id: spec.id,
      exactWgslSha256: await sha256Text(
        aceOpt0009DenseGemmWgsl(spec.shape),
      ),
      k2WgslSha256: await sha256Text(
        aceOpt0074DenseK2PartialsWgsl(spec.shape),
      ),
      k4WgslSha256: await sha256Text(
        aceOpt0032DenseK4PartialsWgsl(spec.shape),
      ),
      plans: Object.freeze({
        exact: Object.freeze({
          tile: Object.freeze([
            exactPlan.tileRows,
            exactPlan.tileColumns,
            exactPlan.tileInner,
          ]),
          workgroupSize: exactPlan.workgroupSize,
          workgroupCount: exactPlan.workgroupCount,
          packedWeightStorageShape: exactPlan.packedWeightStorageShape,
        }),
        k2AndK4: Object.freeze({
          tile: Object.freeze([
            partialPlan.tileRows,
            partialPlan.tileColumns,
            partialPlan.tileInner,
          ]),
          workgroupSize: partialPlan.workgroupSize,
          workgroupCount: partialPlan.workgroupCount,
          packedWeightStorageShape: partialPlan.packedWeightStorageShape,
        }),
      }),
    }));
  }
  return Object.freeze({
    exactKernel: "OPT-0009",
    exactWeightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
    k2Kernel: ACE_OPT_0074_DENSE_K2_PARTIALS_KERNEL_ID,
    k2WeightLayout: ACE_OPT_0074_DENSE_K2_PARTIALS_WEIGHT_LAYOUT,
    k4Kernel: ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID,
    k4WeightLayout: ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
    exactKernelSourceSha256: await sha256Text(currentKernelSource),
    k2AndK4KernelSourceSha256: await sha256Text(candidateKernelSource),
    generatedShaders: Object.freeze(generatedShaders),
    fixtureVersion: "opt0074-opt0032-full-and-adversarial-fp16-v1",
    browserUserAgent: navigator.userAgent,
    browserLanguage: navigator.language,
    browserHardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated,
    requestedDeviceDescriptor: Object.freeze({
      features: Object.freeze([
        "shader-f16",
        "subgroups",
        "timestamp-query",
      ]),
      limits: Object.freeze({
        maxBufferSize: requestedMaximumBufferBytes,
        maxStorageBufferBindingSize: requestedMaximumStorageBytes,
        maxComputeInvocationsPerWorkgroup: 128,
        maxComputeWorkgroupSizeX: 128,
      }),
    }),
    adapter: Object.freeze({
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
      subgroupMinSize: adapter.info.subgroupMinSize,
      subgroupMaxSize: adapter.info.subgroupMaxSize,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
      features: Object.freeze([...adapter.features].sort()),
    }),
    deviceFeatures: Object.freeze([...device.features].sort()),
    deviceLimits: Object.freeze({
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
  if (!adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    !adapter.features.has("timestamp-query") ||
    adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32 ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 128 ||
    adapter.limits.maxComputeWorkgroupSizeX < 128 ||
    adapter.limits.maxStorageBufferBindingSize < maximumStorageBytes ||
    adapter.limits.maxBufferSize <
      maximumStorageBytes + 2 * STORAGE_GUARD_BYTES) {
    throw new Error(
      "OPT-0074 requires stock Chrome timestamp-query, shader-f16, fixed32 subgroups, WG128, and full-shape storage limits",
    );
  }
}

function maximumStorageBindingBytes(): number {
  return Math.max(...[...FULL_SHAPES, ...ADVERSARIAL_CASES].flatMap(({ shape }) => [
    shape.rows * shape.inner * 4,
    shape.inner * shape.columns * 2,
    shape.rows * shape.columns * 4,
  ]));
}

function requireCompleteSnapshot(
  snapshot: OutputSnapshot,
  label: string,
  requireNonzero: boolean,
): void {
  if (snapshot.nonFiniteCount !== 0 ||
    (requireNonzero && snapshot.nonzeroCount === 0) ||
    snapshot.qNaNPrefillCount !== 0 ||
    !snapshot.prefixCanaryIntact ||
    !snapshot.suffixCanaryIntact ||
    !snapshot.tailRowWritten) {
    throw new Error(`${label} failed complete-write, finite, tail, or canary gate`);
  }
}

function requireExactWords(
  expected: Uint32Array,
  actual: Uint32Array,
  label: string,
): void {
  if (expected.length !== actual.length) {
    throw new Error(`${label} length changed`);
  }
  for (let index = 0; index < expected.length; index += 1) {
    if (expected[index] !== actual[index]) {
      throw new Error(`${label} raw-U32 mismatch at ${index}`);
    }
  }
}

async function executeAndDrain(
  device: GPUDevice,
  dispatch: DenseDispatch,
): Promise<void> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function timeDispatch(
  device: GPUDevice,
  dispatch: DenseDispatch,
  timestampQuerySet: GPUQuerySet,
  timestampResolve: GPUBuffer,
  timestampReadback: GPUBuffer,
  spec: FullShapeSpec,
  arm: Arm,
): Promise<TimestampSample> {
  if (timestampReadback.mapState !== "unmapped") {
    throw new Error("OPT-0074 timestamp readback must be unmapped before timing");
  }
  const encoder = device.createCommandEncoder({
    label: `opt0074-${spec.id}-${arm}-timestamp-sample`,
  });
  const pass = encoder.beginComputePass({
    label: `opt0074-${spec.id}-${arm}-timestamped-compute`,
    timestampWrites: {
      querySet: timestampQuerySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    },
  });
  dispatch.encode(pass);
  pass.end();
  encoder.resolveQuerySet(
    timestampQuerySet,
    0,
    2,
    timestampResolve,
    0,
  );
  encoder.copyBufferToBuffer(
    timestampResolve,
    0,
    timestampReadback,
    0,
    TIMESTAMP_QUERY_BYTES,
  );
  const command = encoder.finish();
  const submitAtPerformanceMilliseconds = performance.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  const fenceAtPerformanceMilliseconds = performance.now();
  const wallMilliseconds =
    fenceAtPerformanceMilliseconds - submitAtPerformanceMilliseconds;
  await timestampReadback.mapAsync(GPUMapMode.READ);
  let timestampBegin: bigint;
  let timestampEnd: bigint;
  try {
    const timestamps = new BigUint64Array(
      timestampReadback.getMappedRange(),
    );
    timestampBegin = timestamps[0]!;
    timestampEnd = timestamps[1]!;
  } finally {
    timestampReadback.unmap();
  }
  if (timestampEnd <= timestampBegin) {
    throw new Error(
      `OPT-0074 ${spec.id} ${arm} returned a non-positive timestamp interval`,
    );
  }
  const gpuElapsedNanoseconds = Number(timestampEnd - timestampBegin);
  const gpuMilliseconds = gpuElapsedNanoseconds / 1_000_000;
  if (!Number.isSafeInteger(gpuElapsedNanoseconds) ||
    !Number.isFinite(gpuMilliseconds) || gpuMilliseconds <= 0 ||
    !Number.isFinite(wallMilliseconds) || wallMilliseconds <= 0) {
    throw new Error(`OPT-0074 ${spec.id} ${arm} returned invalid timing`);
  }
  const plan = arm === "exact"
    ? planAceOpt0009DenseGemm(spec.shape)
    : planAceOpt0032DenseK4Partials(spec.shape);
  const scheduledMultiplyAdds = plan.outputRanges.reduce(
    (sum, range) => sum + range.multiplyAdds,
    0,
  );
  const validMultiplyAdds =
    spec.shape.rows * spec.shape.inner * spec.shape.columns;
  return Object.freeze({
    submitAtPerformanceMilliseconds,
    fenceAtPerformanceMilliseconds,
    submitAtEpochMilliseconds:
      performance.timeOrigin + submitAtPerformanceMilliseconds,
    fenceAtEpochMilliseconds:
      performance.timeOrigin + fenceAtPerformanceMilliseconds,
    wallMilliseconds,
    timestampBeginNanoseconds: timestampBegin.toString(),
    timestampEndNanoseconds: timestampEnd.toString(),
    gpuElapsedNanoseconds,
    gpuMilliseconds,
    gpuToWallRatio: gpuMilliseconds / wallMilliseconds,
    validMultiplyAdds,
    scheduledMultiplyAdds,
    validGpuTflops: tflops(validMultiplyAdds, gpuMilliseconds),
    scheduledGpuTflops: tflops(scheduledMultiplyAdds, gpuMilliseconds),
    validWallTflops: tflops(validMultiplyAdds, wallMilliseconds),
    scheduledWallTflops: tflops(scheduledMultiplyAdds, wallMilliseconds),
    commandBufferCount: 1 as const,
    queueDrainCount: 1 as const,
    timestampResolveCount: 1 as const,
    timestampCopyCount: 1 as const,
  });
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
    return sign * 2 ** -14 * (mantissa / 1_024);
  }
  if (exponent === 0x1f) {
    return mantissa === 0
      ? sign * Number.POSITIVE_INFINITY
      : Number.NaN;
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
): FullShapeSpec {
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
): CaseSpec {
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

function meanSamples(samples: readonly number[], label: string): number {
  if (samples.length !== TIMING_ROUNDS.length ||
    samples.some((sample) => !Number.isFinite(sample) || sample <= 0)) {
    throw new Error(
      `OPT-0074 ${label} requires ${TIMING_ROUNDS.length} finite positive samples`,
    );
  }
  return samples.reduce((sum, sample) => sum + sample, 0) / samples.length;
}

function medianSamples(samples: readonly number[], label: string): number {
  if (samples.length !== TIMING_ROUNDS.length ||
    samples.some((sample) => !Number.isFinite(sample) || sample <= 0)) {
    throw new Error(
      `OPT-0074 ${label} requires ${TIMING_ROUNDS.length} finite positive samples`,
    );
  }
  const sorted = [...samples].sort((left, right) => left - right);
  const middle = sorted.length / 2;
  return (sorted[middle - 1]! + sorted[middle]!) / 2;
}

function requireTimestampSamples(
  samples: readonly TimestampSample[],
  label: string,
): void {
  if (samples.length !== TIMING_ROUNDS.length || samples.some((sample) =>
    !Number.isFinite(sample.gpuMilliseconds) || sample.gpuMilliseconds <= 0 ||
    !Number.isFinite(sample.wallMilliseconds) || sample.wallMilliseconds <= 0 ||
    !Number.isFinite(sample.gpuToWallRatio) || sample.gpuToWallRatio <= 0 ||
    !Number.isSafeInteger(sample.gpuElapsedNanoseconds) ||
    sample.gpuElapsedNanoseconds <= 0
  )) {
    throw new Error(
      `OPT-0074 ${label} requires ${TIMING_ROUNDS.length} finite positive timestamp samples`,
    );
  }
}

function tflops(multiplyAdds: number, milliseconds: number): number {
  if (!Number.isSafeInteger(multiplyAdds) || multiplyAdds <= 0 ||
    !Number.isFinite(milliseconds) || milliseconds <= 0) {
    throw new Error("OPT-0074 TFLOP/s inputs must be finite and positive");
  }
  return 2 * multiplyAdds / (milliseconds * 1_000_000_000);
}

function requireLittleEndianHost(): void {
  const words = new Uint16Array([0x0102]);
  if (new Uint8Array(words.buffer)[0] !== 0x02) {
    throw new Error("OPT-0074 fixtures require a little-endian host");
  }
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing ${selector}`);
  return element;
}

async function sha256Text(value: string): Promise<string> {
  return await sha256Bytes(new TextEncoder().encode(value));
}

async function sha256U32(value: Uint32Array): Promise<string> {
  return await sha256Bytes(
    new Uint8Array(value.buffer, value.byteOffset, value.byteLength),
  );
}

async function sha256Bytes(value: Uint8Array): Promise<string> {
  const copy = new Uint8Array(new ArrayBuffer(value.byteLength));
  copy.set(value);
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", copy.buffer),
  );
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function settlePostDrainEvents(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function publish(
  receipt: Readonly<Record<string, unknown>>,
  status: "passed" | "failed",
): void {
  window.__ACE_OPT0074_RESULT__ = receipt;
  result.textContent = JSON.stringify(receipt);
  document.body.dataset.status = status;
}

function fail(
  error: unknown,
  evidence: Readonly<Record<string, unknown>> = Object.freeze({}),
): void {
  const receipt = Object.freeze({
    schema: RECEIPT_SCHEMA,
    experiment: EXPERIMENT_ID,
    status: "failed",
    passed: false,
    ...evidence,
    error: error instanceof Error
      ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
      : String(error),
  });
  publish(receipt, "failed");
  progress.textContent = error instanceof Error ? error.message : String(error);
  runButton.disabled = true;
}
