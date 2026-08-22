/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import currentKernelSource from
  "../../src/webgpu/kernels/dit-dense-fp16.ts?raw";
import k4KernelSource from
  "../../src/webgpu/kernels/dit-dense-fp16-k4-partials.ts?raw";
import candidateKernelSource from
  "../../src/webgpu/kernels/dit-dense-fp16-k8-k16-partials.ts?raw";
import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../../src/model/manifest.js";
import {
  AceOpt0009DenseGemmKernel,
  aceOpt0009DenseGemmWgsl,
  planAceOpt0009DenseGemm,
} from "../../src/webgpu/kernels/dit-dense-fp16.js";
import {
  ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID,
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
  AceOpt0032DenseK4PartialsKernel,
  aceOpt0032DenseK4PartialsWgsl,
  planAceOpt0032DenseK4Partials,
  type AceOpt0032DenseK4PartialsDispatch,
} from "../../src/webgpu/kernels/dit-dense-fp16-k4-partials.js";
import {
  ACE_OPT_0038_DENSE_K8_PARTIALS_KERNEL_ID,
  ACE_OPT_0038_DENSE_K16_PARTIALS_KERNEL_ID,
  ACE_OPT_0038_DENSE_WEIGHT_LAYOUT,
  AceOpt0038DenseBoundedPartialsKernel,
  aceOpt0038DenseBoundedPartialsWgsl,
  planAceOpt0038DenseBoundedPartials,
  type AceOpt0038DenseBoundedPartialsDispatch,
} from "../../src/webgpu/kernels/dit-dense-fp16-k8-k16-partials.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";

declare global {
  interface Window {
    __ACE_OPT0038_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

type Arm = "control" | "k4" | "k8" | "k16";
type ApproximateArm = Exclude<Arm, "control">;
type ExperimentalArm = Exclude<ApproximateArm, "k4">;
type DenseDispatch = Pick<AceGemmDispatch, "label" | "encode"> |
  Pick<AceOpt0032DenseK4PartialsDispatch, "label" | "encode"> |
  Pick<AceOpt0038DenseBoundedPartialsDispatch, "label" | "encode">;
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
  readonly currentKernel: AceOpt0009DenseGemmKernel;
  readonly k4Kernel: AceOpt0032DenseK4PartialsKernel;
  readonly k8Kernel: AceOpt0038DenseBoundedPartialsKernel;
  readonly k16Kernel: AceOpt0038DenseBoundedPartialsKernel;
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
  finiteToZeroCount: number;
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
  readonly samples: Readonly<Record<Arm, readonly number[]>>;
}

const EXPERIMENT_ID = "OPT-0038" as const;
const ROWS = 2_250;
const FULL_OUTPUT_COUNT = 25_344_000;
const ADVERSARIAL_OUTPUT_COUNT = 17_408;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_3255;
const REQUIRED_SPEEDUP_OVER_K4 = 1.15;
const FULL_NRMSE_MAXIMUM = 0.02;
const FULL_SNR_DECIBELS_MINIMUM = 34;
const FULL_PEARSON_MINIMUM = 0.999;
const FULL_MAXIMUM_ABSOLUTE_ERROR = 0.25;
const ADVERSARIAL_NRMSE_MAXIMUM = 0.05;
const ADVERSARIAL_SNR_DECIBELS_MINIMUM = 26;
const ADVERSARIAL_PEARSON_MINIMUM = 0.995;
const ADVERSARIAL_MAXIMUM_ABSOLUTE_ERROR = 0.5;
const FULL_FINITE_TO_ZERO_RATE_MAXIMUM = 1 / 1_000_000;
const ADVERSARIAL_FINITE_TO_ZERO_EVENT_FLOOR = 4;
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
    armOrder: Object.freeze(["control", "k4", "k8", "k16"] as const),
  }),
  Object.freeze({
    shapeOrder: Object.freeze([3, 2, 1, 0]),
    armOrder: Object.freeze(["k16", "k8", "k4", "control"] as const),
  }),
] as const);

const progress = requireElement<HTMLElement>("#progress");
const runButton = requireElement<HTMLButtonElement>("#run");
const result = requireElement<HTMLElement>("#result");
let active: PreparedHarness | undefined;
let started = false;

void prepareHarness().then(
  (prepared) => {
    if (prepared.correctness["passed"] !== true) {
      const cleanup = prepared.destroy();
      publish(Object.freeze({
        schema: "ace-opt-0038-dense-k8-k16-partials-v1",
        experiment: EXPERIMENT_ID,
        status: "correctness-stop",
        passed: false,
        identity: prepared.identity,
        correctness: prepared.correctness,
        cleanup,
        decision: "negative-stop-no-numerically-eligible-variant",
      }), "failed");
      return;
    }
    active = prepared;
    document.body.dataset.status = "ready";
    progress.textContent =
      "READY — four-arm full outputs and adversarial screens completed; timing has not run";
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
  void runTiming(prepared).catch((error: unknown) => {
    prepared.destroy();
    fail(error);
  });
});

window.addEventListener("beforeunload", () => {
  active?.destroy();
  active = undefined;
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
    label: "ace-opt-0038-dense-k8-k16-partials-device",
    requiredFeatures: ["shader-f16", "subgroups"],
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
  const currentKernel = AceOpt0009DenseGemmKernel.create(device, capability);
  const k4Kernel = AceOpt0032DenseK4PartialsKernel.create(
    device,
    capability,
  );
  const k8Kernel = AceOpt0038DenseBoundedPartialsKernel.create(
    device,
    capability,
    "k8",
  );
  const k16Kernel = AceOpt0038DenseBoundedPartialsKernel.create(
    device,
    capability,
    "k16",
  );
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
    currentKernel.destroy();
    k4Kernel.destroy();
    k8Kernel.destroy();
    k16Kernel.destroy();
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
    const fullAggregates = createApproximateAccumulators();
    const shapes: PreparedShape[] = [];
    for (const [index, spec] of FULL_SHAPES.entries()) {
      progress.textContent =
        `four-arm full-output and deterministic reruns ${index + 1}/4: ${spec.id}`;
      shapes.push(await prepareFullShape(
        device,
        tracker,
        currentKernel,
        k4Kernel,
        k8Kernel,
        k16Kernel,
        spec,
        fullAggregates,
      ));
      await yieldToBrowser();
    }
    for (const arm of ["k4", "k8", "k16"] as const) {
      if (fullAggregates[arm].count !== FULL_OUTPUT_COUNT) {
        throw new Error(
          `OPT-0038 ${arm} full comparison count ${fullAggregates[arm].count} != ${FULL_OUTPUT_COUNT}`,
        );
      }
    }
    const adversarialAggregates = createApproximateAccumulators();
    const adversarialReceipts: Readonly<Record<string, unknown>>[] = [];
    for (const [index, spec] of ADVERSARIAL_CASES.entries()) {
      progress.textContent =
        `bounded adversarial screen ${index + 1}/4: ${spec.id}`;
      adversarialReceipts.push(await runAdversarialCase(
        device,
        tracker,
        currentKernel,
        k4Kernel,
        k8Kernel,
        k16Kernel,
        spec,
        adversarialAggregates,
      ));
      await yieldToBrowser();
    }
    for (const arm of ["k4", "k8", "k16"] as const) {
      if (adversarialAggregates[arm].count !== ADVERSARIAL_OUTPUT_COUNT) {
        throw new Error(
          `OPT-0038 ${arm} adversarial comparison count ${adversarialAggregates[arm].count} != ${ADVERSARIAL_OUTPUT_COUNT}`,
        );
      }
    }
    const fullNumerics = finalizeApproximateNumerics(fullAggregates, "full");
    const adversarialNumerics = finalizeApproximateNumerics(
      adversarialAggregates,
      "adversarial",
    );
    const shapeReceipts = shapes.map((shape) => shape.correctness);
    const k4NumericsPassed = armNumericsPassed(
      fullNumerics.k4,
      adversarialNumerics.k4,
    );
    const numericalEligibility = Object.freeze({
      k8: armNumericsPassed(fullNumerics.k8, adversarialNumerics.k8) &&
        qualitativeZeroCollapsePassed(
          fullAggregates.k8,
          adversarialAggregates.k4,
          adversarialAggregates.k8,
        ),
      k16: armNumericsPassed(fullNumerics.k16, adversarialNumerics.k16) &&
        qualitativeZeroCollapsePassed(
          fullAggregates.k16,
          adversarialAggregates.k4,
          adversarialAggregates.k16,
        ),
    });
    const correctnessPassed = k4NumericsPassed &&
      (numericalEligibility.k8 || numericalEligibility.k16) &&
      shapeReceipts.every((receipt) => receipt["passed"] === true) &&
      adversarialReceipts.every((receipt) => receipt["passed"] === true);

    if (correctnessPassed) {
      progress.textContent = "one symmetric untimed warmup per arm and full shape";
      for (const [index, shape] of shapes.entries()) {
        const order: readonly Arm[] = index % 2 === 0
          ? ["control", "k4", "k8", "k16"]
          : ["k16", "k8", "k4", "control"];
        for (const arm of order) {
          await executeAndDrain(device, shape.dispatches[arm]);
        }
      }
    }
    await device.queue.onSubmittedWorkDone();
    await settlePostDrainEvents();
    if (uncapturedErrors.length !== 0) {
      throw new Error(
        `OPT-0038 observed ${uncapturedErrors.length} uncaptured GPU errors`,
      );
    }
    const correctness = Object.freeze({
      fullShapeCount: shapes.length,
      fullComparedU32CountPerApproximateArm: FULL_OUTPUT_COUNT,
      expectedFullComparedU32CountPerApproximateArm: FULL_OUTPUT_COUNT,
      allApproximateArmsDeterministicRawU32: true,
      fullCompleteWrites: true,
      fullFiniteOutputs: true,
      fullNumerics,
      fullCases: Object.freeze(shapeReceipts),
      adversarialCaseCount: adversarialReceipts.length,
      adversarialComparedU32CountPerApproximateArm: ADVERSARIAL_OUTPUT_COUNT,
      adversarialCompleteWrites: true,
      adversarialNumerics,
      adversarialCases: Object.freeze(adversarialReceipts),
      k4NumericsPassed,
      numericalEligibility,
      qualitativeZeroCollapseGate: Object.freeze({
        rule: "full finite-to-zero rate <= 1 per million; adversarial count <= max(4, K4 count)",
        fullFiniteToZero: Object.freeze({
          k4: finiteToZeroSummary(fullAggregates.k4),
          k8: finiteToZeroSummary(fullAggregates.k8),
          k16: finiteToZeroSummary(fullAggregates.k16),
        }),
        adversarialFiniteToZero: Object.freeze({
          k4: finiteToZeroSummary(adversarialAggregates.k4),
          k8: finiteToZeroSummary(adversarialAggregates.k8),
          k16: finiteToZeroSummary(adversarialAggregates.k16),
        }),
      }),
      uncapturedGpuErrorCount: uncapturedErrors.length,
      completedBeforeReady: true,
      passed: correctnessPassed,
    });
    return Object.freeze({
      adapter,
      device,
      tracker,
      currentKernel,
      k4Kernel,
      k8Kernel,
      k16Kernel,
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
  currentKernel: AceOpt0009DenseGemmKernel,
  k4Kernel: AceOpt0032DenseK4PartialsKernel,
  k8Kernel: AceOpt0038DenseBoundedPartialsKernel,
  k16Kernel: AceOpt0038DenseBoundedPartialsKernel,
  spec: FullShapeSpec,
  aggregates: Record<ApproximateArm, NumericalAccumulator>,
): Promise<PreparedShape> {
  const resources = await createCaseResources(device, tracker, spec);
  const guarded = createGuardedOutput(device, tracker, spec);
  try {
    const dispatches = await createDispatches(
      currentKernel,
      k4Kernel,
      k8Kernel,
      k16Kernel,
      spec,
      resources.activation,
      resources.currentWeight,
      resources.k4Weight,
      guarded.binding,
    );
    const correctness = await verifyCase(
      device,
      spec,
      guarded,
      dispatches,
      aggregates,
      true,
      resources.hashes,
    );
    tracker.destroy(guarded.prefill);
    tracker.destroy(guarded.readback);
    return Object.freeze({ spec, dispatches, correctness });
  } catch (error) {
    tracker.destroy(resources.activation);
    tracker.destroy(resources.currentWeight);
    tracker.destroy(resources.k4Weight);
    tracker.destroy(guarded.buffer);
    tracker.destroy(guarded.prefill);
    tracker.destroy(guarded.readback);
    throw error;
  }
}

async function runAdversarialCase(
  device: GPUDevice,
  tracker: BufferTracker,
  currentKernel: AceOpt0009DenseGemmKernel,
  k4Kernel: AceOpt0032DenseK4PartialsKernel,
  k8Kernel: AceOpt0038DenseBoundedPartialsKernel,
  k16Kernel: AceOpt0038DenseBoundedPartialsKernel,
  spec: CaseSpec,
  aggregates: Record<ApproximateArm, NumericalAccumulator>,
): Promise<Readonly<Record<string, unknown>>> {
  const resources = await createCaseResources(device, tracker, spec);
  const guarded = createGuardedOutput(device, tracker, spec);
  try {
    const dispatches = await createDispatches(
      currentKernel,
      k4Kernel,
      k8Kernel,
      k16Kernel,
      spec,
      resources.activation,
      resources.currentWeight,
      resources.k4Weight,
      guarded.binding,
    );
    return await verifyCase(
      device,
      spec,
      guarded,
      dispatches,
      aggregates,
      false,
      resources.hashes,
    );
  } finally {
    tracker.destroy(resources.activation);
    tracker.destroy(resources.currentWeight);
    tracker.destroy(resources.k4Weight);
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
  k4Weight: GPUBuffer;
  hashes: Readonly<Record<string, string>>;
}>> {
  const activation = tracker.create(device, {
    label: `opt0038-${spec.id}-activation`,
    size: spec.shape.rows * spec.shape.inner * 4,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const currentWeight = tracker.create(device, {
    label: `opt0038-${spec.id}-current-weight`,
    size: spec.shape.inner * spec.shape.columns * 2,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const k4Weight = tracker.create(device, {
    label: `opt0038-${spec.id}-shared-k4-weight`,
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
    const k4Range = k4Weight.getMappedRange();
    fillCandidateWeight(new Uint16Array(k4Range), spec);
    const k4WeightHash = await sha256Bytes(
      new Uint8Array(k4Range),
    );
    k4Weight.unmap();
    return Object.freeze({
      activation,
      currentWeight,
      k4Weight,
      hashes: Object.freeze({
        activation: activationHash,
        currentWeight: currentWeightHash,
        sharedK4Weight: k4WeightHash,
      }),
    });
  } catch (error) {
    if (activation.mapState === "mapped") activation.unmap();
    if (currentWeight.mapState === "mapped") currentWeight.unmap();
    if (k4Weight.mapState === "mapped") k4Weight.unmap();
    tracker.destroy(activation);
    tracker.destroy(currentWeight);
    tracker.destroy(k4Weight);
    throw error;
  }
}

async function createDispatches(
  currentKernel: AceOpt0009DenseGemmKernel,
  k4Kernel: AceOpt0032DenseK4PartialsKernel,
  k8Kernel: AceOpt0038DenseBoundedPartialsKernel,
  k16Kernel: AceOpt0038DenseBoundedPartialsKernel,
  spec: CaseSpec,
  activation: GPUBuffer,
  currentWeight: GPUBuffer,
  k4Weight: GPUBuffer,
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
  const k4Bindings: AceGemmBufferBindings = Object.freeze({
    activation: activationBinding,
    weight: binding(k4Weight, weightBytes),
    output,
  });
  const control = await currentKernel.createDispatch(
    `opt0038-${spec.id}-control-opt0009`,
    spec.shape,
    currentBindings,
  );
  const k4 = await k4Kernel.createDispatch(
    `opt0038-${spec.id}-k4`,
    spec.shape,
    k4Bindings,
  );
  const k8 = await k8Kernel.createDispatch(
    `opt0038-${spec.id}-k8`,
    spec.shape,
    k4Bindings,
  );
  const k16 = await k16Kernel.createDispatch(
    `opt0038-${spec.id}-k16`,
    spec.shape,
    k4Bindings,
  );
  if (control.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT ||
    k4.weightLayout !== ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT ||
    k8.weightLayout !== ACE_OPT_0038_DENSE_WEIGHT_LAYOUT ||
    k16.weightLayout !== ACE_OPT_0038_DENSE_WEIGHT_LAYOUT) {
    throw new Error(`OPT-0038 ${spec.id} four-arm weight layout changed`);
  }
  return Object.freeze({ control, k4, k8, k16 });
}

async function verifyCase(
  device: GPUDevice,
  spec: CaseSpec,
  output: GuardedOutput,
  dispatches: Readonly<Record<Arm, DenseDispatch>>,
  aggregates: Record<ApproximateArm, NumericalAccumulator>,
  full: boolean,
  inputHashes: Readonly<Record<string, string>>,
): Promise<Readonly<Record<string, unknown>>> {
  const control = await executeCorrectness(
    device,
    output,
    dispatches.control,
  );
  requireCompleteSnapshot(control, `${spec.id} control`, full);
  const controlHash = await sha256U32(control.words);
  const outputHashes: Record<string, unknown> = { control: controlHash };
  const numerics: Record<string, unknown> = {};
  for (const arm of ["k4", "k8", "k16"] as const) {
    const candidate = await executeCorrectness(device, output, dispatches[arm]);
    requireCompleteSnapshot(candidate, `${spec.id} ${arm}`, full);
    const candidateHash = await sha256U32(candidate.words);
    const local = createAccumulator();
    accumulateNumerics(local, control.words, candidate.words, spec);
    mergeAccumulator(aggregates[arm], local);
    numerics[arm] = finalizeNumerics(
      local,
      full ? "full" : "adversarial",
    );
    const rerun = await executeCorrectness(device, output, dispatches[arm]);
    requireCompleteSnapshot(rerun, `${spec.id} ${arm} rerun`, full);
    requireExactWords(candidate.words, rerun.words, `${spec.id} ${arm} rerun`);
    outputHashes[arm] = Object.freeze({
      first: candidateHash,
      deterministicRerun: await sha256U32(rerun.words),
    });
  }
  return Object.freeze({
    id: spec.id,
    fixtureKind: spec.fixtureKind,
    shape: spec.shape,
    outputU32Count: output.outputElements,
    executionOrder: Object.freeze([
      "control", "k4", "k4", "k8", "k8", "k16", "k16",
    ]),
    completeWriteCount: 7,
    deterministicRawU32: Object.freeze({ k4: true, k8: true, k16: true }),
    outputHashes: Object.freeze(outputHashes),
    inputHashes,
    nonFiniteCount: 0,
    qNaNPrefillCount: 0,
    canariesIntact: true,
    tailRowWritten: true,
    numerics: Object.freeze(numerics),
    passed: true,
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

async function runTiming(prepared: PreparedHarness): Promise<void> {
  const samples = new Map<string, Record<Arm, number[]>>(
    prepared.shapes.map(({ spec }) => [
      spec.id,
      { control: [], k4: [], k8: [], k16: [] },
    ]),
  );
  const rawSamples: Readonly<Record<string, unknown>>[] = [];
  for (const [roundIndex, round] of TIMING_ROUNDS.entries()) {
    for (const shapeIndex of round.shapeOrder) {
      const shape = prepared.shapes[shapeIndex];
      if (shape === undefined) throw new Error("OPT-0038 timing shape changed");
      for (const [armPosition, arm] of round.armOrder.entries()) {
        progress.textContent =
          `timing round ${roundIndex + 1}/2: ${shape.spec.id} ${arm}`;
        const sample = await timeDispatch(
          prepared.device,
          shape.dispatches[arm],
        );
        samples.get(shape.spec.id)![arm].push(sample.wallMilliseconds);
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
    throw new Error("OPT-0038 observed an uncaptured timing GPU error");
  }
  const timingInputs: TimingInput[] = prepared.shapes.map(({ spec }) => {
    const entry = samples.get(spec.id);
    if (entry === undefined) throw new Error("OPT-0038 timing sample missing");
    return Object.freeze({
      id: spec.id,
      samples: Object.freeze({
        control: Object.freeze(entry.control.slice()),
        k4: Object.freeze(entry.k4.slice()),
        k8: Object.freeze(entry.k8.slice()),
        k16: Object.freeze(entry.k16.slice()),
      }),
    });
  });
  const numericalEligibility = requireNumericalEligibility(
    prepared.correctness["numericalEligibility"],
  );
  const timing = summarizeOpt0038Timing(timingInputs, numericalEligibility);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanupFirst = prepared.destroy();
  const cleanupSecond = prepared.destroy();
  const receipt = Object.freeze({
    schema: "ace-opt-0038-dense-k8-k16-partials-v1",
    experiment: EXPERIMENT_ID,
    status: "completed",
    passed: timing["passed"] === true,
    identity: prepared.identity,
    correctness: prepared.correctness,
    protocol: Object.freeze({
      fullOutputAndAdversarialCorrectnessBeforeReady: true,
      oneUntimedWarmupPerArmPerFullShape: true,
      timingButtonCount: 1,
      rounds: 2,
      balancedOrders: TIMING_ROUNDS,
      samplesPerArmPerShape: 2,
      oneCommandBufferPerSample: true,
      oneMatchingQueueDrainPerSample: true,
      outputReadbackInsideTiming: false,
      weightedProductionMultiplicities: "4/2/2/1",
      performanceGate: "at least 1.15x weighted speedup over K4",
    }),
    timing: Object.freeze({
      ...timing,
      rawSamples: Object.freeze(rawSamples),
    }),
    decision: Object.freeze({
      disposition: timing["passed"] === true
        ? "positive-bounded-partial-supersedes-k4-primitive"
        : "negative-retain-opt0032-k4",
      supersedingVariant: timing["winner"],
      productionIntegrationAuthorized: false,
      packageChangeAuthorized: false,
      trajectoryOrListeningClaim: false,
    }),
    memoryBeforeCleanup,
    cleanup: Object.freeze({
      firstCall: cleanupFirst,
      secondCall: cleanupSecond,
      idempotent: true,
      zeroLiveBuffers: cleanupFirst["liveBufferCount"] === 0 &&
        cleanupSecond["liveBufferCount"] === 0,
    }),
  });
  publish(receipt, receipt.passed ? "passed" : "failed");
  progress.textContent = receipt.passed
    ? `passed — ${String(timing["winner"])} supersedes K4`
    : "completed — neither bounded variant cleared 1.15× over K4";
}

export function summarizeOpt0038Timing(
  inputs: readonly TimingInput[],
  numericalEligibility: Readonly<Record<ExperimentalArm, boolean>>,
): Readonly<Record<string, unknown>> {
  if (inputs.length !== FULL_SHAPES.length) {
    throw new Error("OPT-0038 timing requires all four production shapes");
  }
  const weighted: Record<Arm, number> = {
    control: 0,
    k4: 0,
    k8: 0,
    k16: 0,
  };
  const strata = inputs.map((input, index) => {
    const spec = FULL_SHAPES[index];
    if (spec === undefined || spec.id !== input.id) {
      throw new Error("OPT-0038 timing shape order changed");
    }
    const means: Record<Arm, number> = {
      control: mean2(input.samples.control, `${spec.id} control`),
      k4: mean2(input.samples.k4, `${spec.id} k4`),
      k8: mean2(input.samples.k8, `${spec.id} k8`),
      k16: mean2(input.samples.k16, `${spec.id} k16`),
    };
    for (const arm of ["control", "k4", "k8", "k16"] as const) {
      weighted[arm] += means[arm] * spec.productionMultiplicity;
    }
    return Object.freeze({
      id: spec.id,
      shape: spec.shape,
      multiplicity: spec.productionMultiplicity,
      samplesMilliseconds: input.samples,
      meansMilliseconds: Object.freeze(means),
      speedupsOverK4: Object.freeze({
        k8: means.k4 / means.k8,
        k16: means.k4 / means.k16,
      }),
      speedupsOverControl: Object.freeze({
        k4: means.control / means.k4,
        k8: means.control / means.k8,
        k16: means.control / means.k16,
      }),
    });
  });
  const weightedSpeedupsOverK4 = Object.freeze({
    k8: weighted.k4 / weighted.k8,
    k16: weighted.k4 / weighted.k16,
  });
  const qualified = Object.freeze({
    k8: numericalEligibility.k8 &&
      weightedSpeedupsOverK4.k8 >= REQUIRED_SPEEDUP_OVER_K4,
    k16: numericalEligibility.k16 &&
      weightedSpeedupsOverK4.k16 >= REQUIRED_SPEEDUP_OVER_K4,
  });
  const eligibleWinners = (["k8", "k16"] as const).filter(
    (arm) => qualified[arm],
  );
  const winner = eligibleWinners.length === 0
    ? null
    : eligibleWinners.reduce((best, arm) =>
      weighted[arm] < weighted[best] ? arm : best
    );
  return Object.freeze({
    strata: Object.freeze(strata),
    weightedMilliseconds: Object.freeze(weighted),
    weightedSpeedupsOverControl: Object.freeze({
      k4: weighted.control / weighted.k4,
      k8: weighted.control / weighted.k8,
      k16: weighted.control / weighted.k16,
    }),
    weightedSpeedupsOverK4,
    numericalEligibility,
    requiredSpeedupOverK4: REQUIRED_SPEEDUP_OVER_K4,
    qualified,
    winner,
    passed: winner !== null,
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
    finiteToZeroCount: 0,
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

function createApproximateAccumulators(): Record<
  ApproximateArm,
  NumericalAccumulator
> {
  return {
    k4: createAccumulator(),
    k8: createAccumulator(),
    k16: createAccumulator(),
  };
}

function finalizeApproximateNumerics(
  accumulators: Record<ApproximateArm, NumericalAccumulator>,
  envelope: "full" | "adversarial",
): Record<ApproximateArm, Readonly<Record<string, unknown>>> {
  return {
    k4: finalizeNumerics(accumulators.k4, envelope),
    k8: finalizeNumerics(accumulators.k8, envelope),
    k16: finalizeNumerics(accumulators.k16, envelope),
  };
}

function armNumericsPassed(
  full: Readonly<Record<string, unknown>>,
  adversarial: Readonly<Record<string, unknown>>,
): boolean {
  return full["passed"] === true && adversarial["passed"] === true;
}

function qualitativeZeroCollapsePassed(
  fullCandidate: NumericalAccumulator,
  adversarialK4: NumericalAccumulator,
  adversarialCandidate: NumericalAccumulator,
): boolean {
  return finiteToZeroRate(fullCandidate) <=
      FULL_FINITE_TO_ZERO_RATE_MAXIMUM &&
    adversarialCandidate.finiteToZeroCount <=
      Math.max(
        ADVERSARIAL_FINITE_TO_ZERO_EVENT_FLOOR,
        adversarialK4.finiteToZeroCount,
      );
}

function finiteToZeroRate(accumulator: NumericalAccumulator): number {
  return accumulator.count === 0
    ? Number.POSITIVE_INFINITY
    : accumulator.finiteToZeroCount / accumulator.count;
}

function finiteToZeroSummary(
  accumulator: NumericalAccumulator,
): Readonly<Record<string, number>> {
  const rate = finiteToZeroRate(accumulator);
  return Object.freeze({
    count: accumulator.finiteToZeroCount,
    comparedCount: accumulator.count,
    rate,
    eventsPerMillion: rate * 1_000_000,
  });
}

function requireNumericalEligibility(
  value: unknown,
): Readonly<Record<ExperimentalArm, boolean>> {
  if (typeof value !== "object" || value === null ||
    typeof (value as Record<string, unknown>)["k8"] !== "boolean" ||
    typeof (value as Record<string, unknown>)["k16"] !== "boolean") {
    throw new Error("OPT-0038 numerical eligibility receipt changed");
  }
  return Object.freeze({
    k8: (value as Record<string, boolean>)["k8"]!,
    k16: (value as Record<string, boolean>)["k16"]!,
  });
}

function accumulateNumerics(
  accumulator: NumericalAccumulator,
  controlWords: Uint32Array,
  candidateWords: Uint32Array,
  spec: CaseSpec,
): void {
  if (controlWords.length !== candidateWords.length) {
    throw new Error(`OPT-0038 ${spec.id} output length changed`);
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
    if (Number.isFinite(a) && (aWord & 0x7fff_ffff) !== 0 &&
      (bWord & 0x7fff_ffff) === 0) {
      accumulator.finiteToZeroCount += 1;
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
    "finiteToZeroCount",
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
    throw new Error("OPT-0038 cannot finalize an empty comparison");
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
    finiteToZeroCount: accumulator.finiteToZeroCount,
    finiteToZeroRate: finiteToZeroRate(accumulator),
    finiteToZeroEventsPerMillion:
      finiteToZeroRate(accumulator) * 1_000_000,
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
    label: `opt0038-${spec.id}-prefill`,
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
    label: `opt0038-${spec.id}-output`,
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  const readback = tracker.create(device, {
    label: `opt0038-${spec.id}-readback`,
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
  const generatedShaders = [];
  for (const spec of [...FULL_SHAPES, ...ADVERSARIAL_CASES]) {
    const controlPlan = planAceOpt0009DenseGemm(spec.shape);
    const k4Plan = planAceOpt0032DenseK4Partials(spec.shape);
    const k8Plan = planAceOpt0038DenseBoundedPartials(spec.shape, "k8");
    const k16Plan = planAceOpt0038DenseBoundedPartials(spec.shape, "k16");
    generatedShaders.push(Object.freeze({
      id: spec.id,
      controlWgslSha256: await sha256Text(
        aceOpt0009DenseGemmWgsl(spec.shape),
      ),
      k4WgslSha256: await sha256Text(
        aceOpt0032DenseK4PartialsWgsl(spec.shape),
      ),
      k8WgslSha256: await sha256Text(
        aceOpt0038DenseBoundedPartialsWgsl(spec.shape, "k8"),
      ),
      k16WgslSha256: await sha256Text(
        aceOpt0038DenseBoundedPartialsWgsl(spec.shape, "k16"),
      ),
      plans: Object.freeze({
        control: Object.freeze({
          tile: Object.freeze([
            controlPlan.tileRows,
            controlPlan.tileColumns,
            controlPlan.tileInner,
          ]),
          workgroupSize: controlPlan.workgroupSize,
          workgroupCount: controlPlan.workgroupCount,
          packedWeightStorageShape: controlPlan.packedWeightStorageShape,
        }),
        k4: Object.freeze({
          tile: Object.freeze([
            k4Plan.tileRows,
            k4Plan.tileColumns,
            k4Plan.tileInner,
          ]),
          workgroupSize: k4Plan.workgroupSize,
          workgroupCount: k4Plan.workgroupCount,
          packedWeightStorageShape: k4Plan.packedWeightStorageShape,
        }),
        k8: Object.freeze({
          partialInner: k8Plan.partialInner,
          k4GroupsPerPartial: k8Plan.k4GroupsPerPartial,
          innerPartialBlocks: k8Plan.innerPartialBlocks,
          workgroupSize: k8Plan.workgroupSize,
          workgroupCount: k8Plan.workgroupCount,
          packedWeightStorageShape: k8Plan.packedWeightStorageShape,
        }),
        k16: Object.freeze({
          partialInner: k16Plan.partialInner,
          k4GroupsPerPartial: k16Plan.k4GroupsPerPartial,
          innerPartialBlocks: k16Plan.innerPartialBlocks,
          workgroupSize: k16Plan.workgroupSize,
          workgroupCount: k16Plan.workgroupCount,
          packedWeightStorageShape: k16Plan.packedWeightStorageShape,
        }),
      }),
    }));
  }
  return Object.freeze({
    currentKernel: "OPT-0009",
    currentWeightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
    k4Kernel: ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID,
    k8Kernel: ACE_OPT_0038_DENSE_K8_PARTIALS_KERNEL_ID,
    k16Kernel: ACE_OPT_0038_DENSE_K16_PARTIALS_KERNEL_ID,
    sharedApproximateWeightLayout: ACE_OPT_0038_DENSE_WEIGHT_LAYOUT,
    currentKernelSourceSha256: await sha256Text(currentKernelSource),
    k4KernelSourceSha256: await sha256Text(k4KernelSource),
    candidateKernelSourceSha256: await sha256Text(candidateKernelSource),
    generatedShaders: Object.freeze(generatedShaders),
    fixtureVersion: "opt0038-four-arm-full-and-adversarial-fp16-v1",
    browserUserAgent: navigator.userAgent,
    adapter: Object.freeze({
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
      subgroupMinSize: adapter.info.subgroupMinSize,
      subgroupMaxSize: adapter.info.subgroupMaxSize,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    }),
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
    adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32 ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 128 ||
    adapter.limits.maxComputeWorkgroupSizeX < 128 ||
    adapter.limits.maxStorageBufferBindingSize < maximumStorageBytes ||
    adapter.limits.maxBufferSize <
      maximumStorageBytes + 2 * STORAGE_GUARD_BYTES) {
    throw new Error(
      "OPT-0038 requires stock Chrome shader-f16, fixed32 subgroups, WG128, and full-shape storage limits",
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
): Promise<Readonly<{
  submitAtPerformanceMilliseconds: number;
  fenceAtPerformanceMilliseconds: number;
  wallMilliseconds: number;
  commandBufferCount: 1;
  queueDrainCount: 1;
}>> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  const command = encoder.finish();
  const submitAtPerformanceMilliseconds = performance.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  const fenceAtPerformanceMilliseconds = performance.now();
  return Object.freeze({
    submitAtPerformanceMilliseconds,
    fenceAtPerformanceMilliseconds,
    wallMilliseconds:
      fenceAtPerformanceMilliseconds - submitAtPerformanceMilliseconds,
    commandBufferCount: 1 as const,
    queueDrainCount: 1 as const,
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

function mean2(samples: readonly number[], label: string): number {
  if (samples.length !== 2 ||
    samples.some((sample) => !Number.isFinite(sample) || sample <= 0)) {
    throw new Error(`OPT-0038 ${label} requires two finite positive samples`);
  }
  return (samples[0]! + samples[1]!) / 2;
}

function requireLittleEndianHost(): void {
  const words = new Uint16Array([0x0102]);
  if (new Uint8Array(words.buffer)[0] !== 0x02) {
    throw new Error("OPT-0038 fixtures require a little-endian host");
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
  window.__ACE_OPT0038_RESULT__ = receipt;
  result.textContent = JSON.stringify(receipt);
  document.body.dataset.status = status;
}

function fail(error: unknown): void {
  const receipt = Object.freeze({
    schema: "ace-opt-0038-dense-k8-k16-partials-v1",
    experiment: EXPERIMENT_ID,
    status: "failed",
    passed: false,
    error: error instanceof Error
      ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
      : String(error),
  });
  publish(receipt, "failed");
  progress.textContent = error instanceof Error ? error.message : String(error);
  runButton.disabled = true;
}
