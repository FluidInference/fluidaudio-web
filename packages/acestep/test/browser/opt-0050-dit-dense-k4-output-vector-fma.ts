/// <reference types="@webgpu/types" />

import {
  ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID,
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
  AceOpt0032DenseK4PartialsKernel,
  packAceOpt0032DenseWeightU16,
} from "../../src/webgpu/kernels/dit-dense-fp16-k4-partials.js";
import {
  ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_KERNEL_ID,
  ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_WEIGHT_LAYOUT,
  AceOpt0050DenseK4OutputVectorFmaKernel,
  packAceOpt0050DenseWeightU16,
  unpackAceOpt0050DenseWeightU16,
} from
  "../../src/webgpu/kernels/dit-dense-fp16-k4-output-vector-fma.js";
import type {
  AceOpt0032DenseK4PartialsDispatch,
} from "../../src/webgpu/kernels/dit-dense-fp16-k4-partials.js";
import type {
  AceOpt0050DenseK4OutputVectorFmaDispatch,
} from
  "../../src/webgpu/kernels/dit-dense-fp16-k4-output-vector-fma.js";
import type { AceGemmBufferBindings, AceGemmShape } from
  "../../src/webgpu/kernels/gemm.js";

declare global {
  interface Window {
    __ACE_OPT0050_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

type Arm = "dotK4" | "outputVectorFma";
type FixtureKind = "full" | "signed-zero" | "cancellation" | "range" |
  "long-k";
type DenseDispatch = Pick<
  AceOpt0032DenseK4PartialsDispatch,
  "label" | "encode"
> | Pick<AceOpt0050DenseK4OutputVectorFmaDispatch, "label" | "encode">;

export interface Opt0050CaseSpec {
  readonly id: string;
  readonly shape: AceGemmShape;
  readonly fixtureKind: FixtureKind;
  readonly ordinal: number;
  readonly productionMultiplicity?: 4 | 2 | 1;
}

export interface Opt0050ThermalGate {
  readonly command:
    "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly waitStartedAtEpochMilliseconds: number;
  readonly checkedAtEpochMilliseconds: number;
  readonly waitDurationMilliseconds: number;
  readonly checkCount: 1;
  readonly thermalLevel: 0;
  readonly launchDelayMilliseconds: number;
}

interface CaseResources {
  readonly activation: GPUBuffer;
  readonly dotWeight: GPUBuffer;
  readonly fmaWeight: GPUBuffer;
  readonly packProof: Readonly<Record<string, number | boolean>>;
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
  readonly spec: Opt0050CaseSpec;
  readonly dispatches: Readonly<Record<Arm, DenseDispatch>>;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly dotKernel: AceOpt0032DenseK4PartialsKernel;
  readonly fmaKernel: AceOpt0050DenseK4OutputVectorFmaKernel;
  readonly shapes: readonly PreparedShape[];
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

const EXPERIMENT_ID = "OPT-0050" as const;
const ROWS = 2_250;
const FULL_OUTPUT_COUNT = 25_344_000;
const ADVERSARIAL_OUTPUT_COUNT = 17_408;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_5050;
const REQUIRED_WEIGHTED_SPEEDUP = 1.15;
const FULL_NRMSE_MAXIMUM = 0.02;
const FULL_SNR_DECIBELS_MINIMUM = 34;
const FULL_PEARSON_MINIMUM = 0.999;
const FULL_MAXIMUM_ABSOLUTE_ERROR = 0.25;
const ADVERSARIAL_NRMSE_MAXIMUM = 0.05;
const ADVERSARIAL_SNR_DECIBELS_MINIMUM = 26;
const ADVERSARIAL_PEARSON_MINIMUM = 0.995;
const ADVERSARIAL_MAXIMUM_ABSOLUTE_ERROR = 0.5;
const THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
const MINIMUM_WAIT_MILLISECONDS = 30_000;
const MAXIMUM_WAIT_MILLISECONDS = 60_000;
const MAXIMUM_CHECK_TO_LAUNCH_MILLISECONDS = 5_000;
const FINITE_HALF_MAGNITUDES = Object.freeze([
  0x2411, 0x28b5, 0x2d53, 0x31e7, 0x356b, 0x39ad,
] as const);
const TIMING_ROUNDS = Object.freeze([
  Object.freeze({
    shapeOrder: Object.freeze([0, 1, 2, 3]),
    armOrder: Object.freeze(["dotK4", "outputVectorFma"] as const),
  }),
  Object.freeze({
    shapeOrder: Object.freeze([3, 2, 1, 0]),
    armOrder: Object.freeze(["outputVectorFma", "dotK4"] as const),
  }),
]);

if (typeof document !== "undefined" && document.querySelector("#run") !== null) {
  installBrowserGate();
}

export function buildOpt0050Cases(): Readonly<{
  full: readonly Opt0050CaseSpec[];
  adversarial: readonly Opt0050CaseSpec[];
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
      caseSpec("k4-cancellation", 8, 2_048, 1_024, "cancellation", 11),
      caseSpec("finite-range", 4, 2_048, 1_024, "range", 12),
      caseSpec("long-k6144", 2, 6_144, 2_048, "long-k", 13),
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
          schema: "ace-opt-0050-dense-k4-output-vector-fma-v1",
          experimentId: EXPERIMENT_ID,
          passed: false,
          correctness: value.correctness,
          cleanup,
        }));
        return;
      }
      prepared = value;
      document.body.dataset.status = "ready";
      progress.textContent =
        "READY — four full M2250 outputs and all OPT-0032 adversarial fixtures passed; begin one fresh 30-second idle wait";
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
    let thermal: Opt0050ThermalGate;
    try {
      thermal = parseOpt0050ThermalGate(
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
        finishPage(receipt.passed === true ? "passed" : "failed", receipt);
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
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const maximumStorageBytes = maximumStorageBindingBytes();
  const device = await adapter.requestDevice({
    label: "ace-opt-0050-dense-k4-output-vector-fma",
    requiredFeatures: ["shader-f16", "subgroups"],
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
  const dotKernel = AceOpt0032DenseK4PartialsKernel.create(device, capability);
  const fmaKernel = AceOpt0050DenseK4OutputVectorFmaKernel.create(
    device,
    capability,
  );
  let destroyed = false;
  const destroy = (): Readonly<Record<string, unknown>> => {
    if (destroyed) return Object.freeze({ ...tracker.receipt(), repeated: true });
    destroyed = true;
    dotKernel.destroy();
    fmaKernel.destroy();
    tracker.destroyAll();
    device.destroy();
    return Object.freeze({
      ...tracker.receipt(),
      repeated: false,
      deviceDestroyed: true,
    });
  };
  try {
    const cases = buildOpt0050Cases();
    const fullAggregate = createAccumulator();
    const shapes: PreparedShape[] = [];
    for (const [index, spec] of cases.full.entries()) {
      update(`full output ${index + 1}/4: ${spec.id}`);
      shapes.push(await prepareFullShape(
        device,
        tracker,
        dotKernel,
        fmaKernel,
        spec,
        fullAggregate,
      ));
      await browserYield();
    }
    const adversarialAggregate = createAccumulator();
    const adversarial: Readonly<Record<string, unknown>>[] = [];
    for (const [index, spec] of cases.adversarial.entries()) {
      update(`adversarial ${index + 1}/4: ${spec.id}`);
      adversarial.push(await runAdversarialCase(
        device,
        tracker,
        dotKernel,
        fmaKernel,
        spec,
        adversarialAggregate,
      ));
      await browserYield();
    }
    if (fullAggregate.count !== FULL_OUTPUT_COUNT ||
      adversarialAggregate.count !== ADVERSARIAL_OUTPUT_COUNT) {
      throw new Error("OPT-0050 fixture output count changed");
    }
    const fullNumerics = finalizeNumerics(fullAggregate, "full");
    const adversarialNumerics = finalizeNumerics(
      adversarialAggregate,
      "adversarial",
    );
    const fullReceipts = shapes.map(({ correctness }) => correctness);
    const passed = fullNumerics["passed"] === true &&
      adversarialNumerics["passed"] === true &&
      fullReceipts.every((receipt) => receipt["passed"] === true) &&
      adversarial.every((receipt) => receipt["passed"] === true);
    if (passed) {
      for (const [index, shape] of shapes.entries()) {
        const order: readonly Arm[] = index % 2 === 0
          ? ["dotK4", "outputVectorFma"]
          : ["outputVectorFma", "dotK4"];
        for (const arm of order) {
          await executeAndDrain(device, shape.dispatches[arm]);
        }
      }
    }
    await device.queue.onSubmittedWorkDone();
    await browserYield();
    if (errors.length !== 0) {
      throw new Error(`OPT-0050 uncaptured GPU errors: ${errors.join("; ")}`);
    }
    return Object.freeze({
      adapter,
      device,
      tracker,
      dotKernel,
      fmaKernel,
      shapes: Object.freeze(shapes),
      correctness: Object.freeze({
        passed,
        completedBeforeReady: true,
        fullOutputCount: fullAggregate.count,
        fullCandidateDeterministicRawU32: true,
        fullCompleteWritesFiniteGuardsAndTails: true,
        fullNumerics,
        fullCases: Object.freeze(fullReceipts),
        adversarialOutputCount: adversarialAggregate.count,
        adversarialCompleteWritesFiniteGuardsAndTails: true,
        adversarialNumerics,
        adversarialCases: Object.freeze(adversarial),
        uncapturedGpuErrorCount: errors.length,
      }),
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

async function prepareFullShape(
  device: GPUDevice,
  tracker: BufferTracker,
  dotKernel: AceOpt0032DenseK4PartialsKernel,
  fmaKernel: AceOpt0050DenseK4OutputVectorFmaKernel,
  spec: Opt0050CaseSpec,
  aggregate: NumericalAccumulator,
): Promise<PreparedShape> {
  const resources = createCaseResources(device, tracker, spec);
  const output = createGuardedOutput(device, tracker, spec);
  try {
    const dispatches = await createDispatches(
      dotKernel,
      fmaKernel,
      spec,
      resources,
      output.binding,
    );
    const correctness = await verifyCase(
      device,
      spec,
      resources.packProof,
      output,
      dispatches,
      aggregate,
      true,
    );
    tracker.destroy(output.prefill);
    tracker.destroy(output.readback);
    return Object.freeze({ spec, dispatches, correctness });
  } catch (error) {
    destroyResources(tracker, resources);
    tracker.destroy(output.buffer);
    tracker.destroy(output.prefill);
    tracker.destroy(output.readback);
    throw error;
  }
}

async function runAdversarialCase(
  device: GPUDevice,
  tracker: BufferTracker,
  dotKernel: AceOpt0032DenseK4PartialsKernel,
  fmaKernel: AceOpt0050DenseK4OutputVectorFmaKernel,
  spec: Opt0050CaseSpec,
  aggregate: NumericalAccumulator,
): Promise<Readonly<Record<string, unknown>>> {
  const resources = createCaseResources(device, tracker, spec);
  const output = createGuardedOutput(device, tracker, spec);
  try {
    const dispatches = await createDispatches(
      dotKernel,
      fmaKernel,
      spec,
      resources,
      output.binding,
    );
    return await verifyCase(
      device,
      spec,
      resources.packProof,
      output,
      dispatches,
      aggregate,
      false,
    );
  } finally {
    destroyResources(tracker, resources);
    tracker.destroy(output.buffer);
    tracker.destroy(output.prefill);
    tracker.destroy(output.readback);
  }
}

function createCaseResources(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0050CaseSpec,
): CaseResources {
  const activationElements = spec.shape.rows * spec.shape.inner;
  const weightElements = spec.shape.inner * spec.shape.columns;
  const activation = tracker.create(device, {
    label: `opt0050-${spec.id}-activation`,
    size: activationElements * 4,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const dotWeight = tracker.create(device, {
    label: `opt0050-${spec.id}-dot-k4-weight`,
    size: weightElements * 2,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const fmaWeight = tracker.create(device, {
    label: `opt0050-${spec.id}-output-fma-weight`,
    size: weightElements * 2,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    const activationValues = new Float32Array(activation.getMappedRange());
    fillActivation(activationValues, spec);
    activation.unmap();

    const logical = new Uint16Array(weightElements);
    fillLogicalWeight(logical, spec);
    const dotPacked = packAceOpt0032DenseWeightU16(
      logical,
      spec.shape.inner,
      spec.shape.columns,
    );
    const fmaPacked = packAceOpt0050DenseWeightU16(
      logical,
      spec.shape.inner,
      spec.shape.columns,
    );
    const inverse = unpackAceOpt0050DenseWeightU16(
      fmaPacked,
      spec.shape.inner,
      spec.shape.columns,
    );
    let inverseMismatchCount = 0;
    for (let index = 0; index < logical.length; index += 1) {
      if (logical[index] !== inverse[index]) inverseMismatchCount += 1;
    }
    new Uint16Array(dotWeight.getMappedRange()).set(dotPacked);
    dotWeight.unmap();
    new Uint16Array(fmaWeight.getMappedRange()).set(fmaPacked);
    fmaWeight.unmap();
    return Object.freeze({
      activation,
      dotWeight,
      fmaWeight,
      packProof: Object.freeze({
        logicalWords: logical.length,
        packedWords: fmaPacked.length,
        inverseWordsCompared: inverse.length,
        inverseMismatchCount,
        exhaustivePackInverseIdentity: inverseMismatchCount === 0 &&
          logical.length === fmaPacked.length,
      }),
    });
  } catch (error) {
    if (activation.mapState === "mapped") activation.unmap();
    if (dotWeight.mapState === "mapped") dotWeight.unmap();
    if (fmaWeight.mapState === "mapped") fmaWeight.unmap();
    tracker.destroy(activation);
    tracker.destroy(dotWeight);
    tracker.destroy(fmaWeight);
    throw error;
  }
}

function destroyResources(tracker: BufferTracker, resources: CaseResources): void {
  tracker.destroy(resources.activation);
  tracker.destroy(resources.dotWeight);
  tracker.destroy(resources.fmaWeight);
}

async function createDispatches(
  dotKernel: AceOpt0032DenseK4PartialsKernel,
  fmaKernel: AceOpt0050DenseK4OutputVectorFmaKernel,
  spec: Opt0050CaseSpec,
  resources: CaseResources,
  output: GPUBufferBinding,
): Promise<Readonly<Record<Arm, DenseDispatch>>> {
  const activationBytes = spec.shape.rows * spec.shape.inner * 4;
  const weightBytes = spec.shape.inner * spec.shape.columns * 2;
  const common = Object.freeze({
    activation: binding(resources.activation, activationBytes),
    output,
  });
  const dotBindings: AceGemmBufferBindings = Object.freeze({
    ...common,
    weight: binding(resources.dotWeight, weightBytes),
  });
  const fmaBindings: AceGemmBufferBindings = Object.freeze({
    ...common,
    weight: binding(resources.fmaWeight, weightBytes),
  });
  const dotK4 = await dotKernel.createDispatch(
    `opt0050-${spec.id}-dot-k4-control`,
    spec.shape,
    dotBindings,
  );
  const outputVectorFma = await fmaKernel.createDispatch(
    `opt0050-${spec.id}-output-vector-fma`,
    spec.shape,
    fmaBindings,
  );
  if (
    dotK4.weightLayout !== ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT ||
    outputVectorFma.weightLayout !==
      ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_WEIGHT_LAYOUT
  ) {
    throw new Error(`OPT-0050 ${spec.id} weight layout changed`);
  }
  return Object.freeze({ dotK4, outputVectorFma });
}

async function verifyCase(
  device: GPUDevice,
  spec: Opt0050CaseSpec,
  packProof: Readonly<Record<string, number | boolean>>,
  output: GuardedOutput,
  dispatches: Readonly<Record<Arm, DenseDispatch>>,
  aggregate: NumericalAccumulator,
  full: boolean,
): Promise<Readonly<Record<string, unknown>>> {
  let control: OutputSnapshot | undefined = await executeSnapshot(
    device,
    output,
    dispatches.dotK4,
  );
  requireCompleteSnapshot(control, `${spec.id} dot-K4`, full);
  let candidate: OutputSnapshot | undefined = await executeSnapshot(
    device,
    output,
    dispatches.outputVectorFma,
  );
  requireCompleteSnapshot(candidate, `${spec.id} output-vector-FMA`, full);
  const local = createAccumulator();
  accumulateNumerics(local, control.words, candidate.words, spec);
  mergeAccumulator(aggregate, local);
  const numerics = finalizeNumerics(local, full ? "full" : "adversarial");
  control = undefined;
  const rerun = await executeSnapshot(
    device,
    output,
    dispatches.outputVectorFma,
  );
  requireCompleteSnapshot(rerun, `${spec.id} FMA rerun`, full);
  requireExactWords(candidate.words, rerun.words, `${spec.id} FMA rerun`);
  candidate = undefined;
  const packingPassed = packProof["exhaustivePackInverseIdentity"] === true;
  return Object.freeze({
    id: spec.id,
    fixtureKind: spec.fixtureKind,
    shape: spec.shape,
    outputU32Count: output.outputElements,
    executionOrder: Object.freeze(["dotK4", "outputVectorFma", "outputVectorFma"]),
    packProof,
    candidateDeterministicRawU32: true,
    completeWritesFiniteGuardsAndTail: true,
    numerics,
    passed: packingPassed && numerics["passed"] === true,
  });
}

async function executeSnapshot(
  device: GPUDevice,
  output: GuardedOutput,
  dispatch: DenseDispatch,
): Promise<OutputSnapshot> {
  const encoder = device.createCommandEncoder({
    label: `${dispatch.label}-correctness`,
  });
  encoder.copyBufferToBuffer(output.prefill, 0, output.buffer, 0, output.totalBytes);
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  encoder.copyBufferToBuffer(output.buffer, 0, output.readback, 0, output.totalBytes);
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

async function runTiming(
  prepared: PreparedGate,
  thermal: Opt0050ThermalGate,
  progress: HTMLElement,
): Promise<Readonly<Record<string, unknown>>> {
  const samples = new Map<string, Record<Arm, number[]>>(
    prepared.shapes.map(({ spec }) => [
      spec.id,
      { dotK4: [], outputVectorFma: [] },
    ]),
  );
  const rawSamples: Readonly<Record<string, unknown>>[] = [];
  for (const [roundIndex, round] of TIMING_ROUNDS.entries()) {
    for (const shapeIndex of round.shapeOrder) {
      const shape = prepared.shapes[shapeIndex];
      if (shape === undefined) throw new Error("OPT-0050 timing shape changed");
      for (const [armPosition, arm] of round.armOrder.entries()) {
        progress.textContent =
          `timing ${roundIndex + 1}/2 ${shape.spec.id} ${arm}`;
        const sample = await timeDispatch(prepared.device, shape.dispatches[arm]);
        samples.get(shape.spec.id)![arm].push(sample.wallMilliseconds);
        rawSamples.push(Object.freeze({
          roundIndex,
          shapeIndex,
          armPosition,
          arm,
          ...sample,
        }));
      }
      await browserYield();
    }
  }
  await prepared.device.queue.onSubmittedWorkDone();
  await browserYield();
  if (prepared.uncapturedErrors.length !== 0) {
    throw new Error("OPT-0050 observed an uncaptured timing GPU error");
  }
  let weightedDot = 0;
  let weightedFma = 0;
  const strata = prepared.shapes.map(({ spec }) => {
    const values = samples.get(spec.id)!;
    const dotK4 = mean2(values.dotK4, `${spec.id} dot-K4`);
    const outputVectorFma = mean2(
      values.outputVectorFma,
      `${spec.id} output-vector-FMA`,
    );
    const multiplicity = spec.productionMultiplicity;
    if (multiplicity === undefined) throw new Error("OPT-0050 multiplicity missing");
    weightedDot += dotK4 * multiplicity;
    weightedFma += outputVectorFma * multiplicity;
    return Object.freeze({
      id: spec.id,
      shape: spec.shape,
      multiplicity,
      samplesMilliseconds: Object.freeze({
        dotK4: Object.freeze(values.dotK4.slice()),
        outputVectorFma: Object.freeze(values.outputVectorFma.slice()),
      }),
      meansMilliseconds: Object.freeze({ dotK4, outputVectorFma }),
      speedup: dotK4 / outputVectorFma,
      nonSlower: outputVectorFma <= dotK4,
    });
  });
  const weightedSpeedup = weightedDot / weightedFma;
  const everyProductionShapeNonSlower = strata.every(({ nonSlower }) => nonSlower);
  const timingPassed = everyProductionShapeNonSlower &&
    weightedSpeedup >= REQUIRED_WEIGHTED_SPEEDUP;
  const environment = environmentReceipt(prepared.adapter, prepared.device);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanupFirst = prepared.destroy();
  const cleanupSecond = prepared.destroy();
  return Object.freeze({
    schema: "ace-opt-0050-dense-k4-output-vector-fma-v1",
    experimentId: EXPERIMENT_ID,
    passed: prepared.correctness["passed"] === true && timingPassed,
    disposition: "benchmark-only",
    controlKernelId: ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID,
    controlWeightLayout: ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
    candidateKernelId: ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_KERNEL_ID,
    candidateWeightLayout:
      ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_WEIGHT_LAYOUT,
    packingOutsideTimedRegion: true,
    correctness: prepared.correctness,
    thermal,
    protocol: Object.freeze({
      fullAndAdversarialCorrectnessBeforeReady: true,
      oneSymmetricWarmupPerArmPerShape: true,
      balancedOrders: TIMING_ROUNDS,
      oneSubmitAndOneMatchingDrainPerSample: true,
      outputReadbackInsideTiming: false,
      weightedProductionMultiplicities: "4/2/2/1",
    }),
    timing: Object.freeze({
      strata: Object.freeze(strata),
      weightedDotK4Milliseconds: weightedDot,
      weightedOutputVectorFmaMilliseconds: weightedFma,
      weightedSpeedup,
      requiredWeightedSpeedup: REQUIRED_WEIGHTED_SPEEDUP,
      everyProductionShapeNonSlower,
      rawSamples: Object.freeze(rawSamples),
      passed: timingPassed,
    }),
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

export function parseOpt0050ThermalGate(
  parameters: URLSearchParams,
  preparedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0050ThermalGate {
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
  const waitDurationMilliseconds = checkedAtEpochMilliseconds -
    waitStartedAtEpochMilliseconds;
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    checkedAtEpochMilliseconds;
  if (
    command !== THERMAL_COMMAND || checkCount !== 1 || thermalLevel !== 0 ||
    waitStartedAtEpochMilliseconds < preparedAtEpochMilliseconds ||
    waitDurationMilliseconds < MINIMUM_WAIT_MILLISECONDS ||
    waitDurationMilliseconds > MAXIMUM_WAIT_MILLISECONDS ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_CHECK_TO_LAUNCH_MILLISECONDS
  ) {
    throw new Error(
      "OPT-0050 requires exactly one level-0 notifyutil check after one fresh 30-second idle wait",
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
  spec: Opt0050CaseSpec,
): void {
  if (controlWords.length !== candidateWords.length) {
    throw new Error(`OPT-0050 ${spec.id} output length changed`);
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
      aWord !== bWord) accumulator.signedZeroDifferenceCount += 1;
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
    "count", "finiteCount", "controlNonFiniteCount",
    "candidateNonFiniteCount", "differingU32Count",
    "signedZeroDifferenceCount", "classChangeCount", "controlSum",
    "candidateSum", "controlSquareSum", "candidateSquareSum", "crossSum",
    "errorSum", "absoluteErrorSum", "errorSquareSum", "relativeErrorSquareSum",
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
    throw new Error("OPT-0050 cannot finalize an empty comparison");
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
  const pearson = pearsonDenominator === 0
    ? accumulator.errorSquareSum === 0 ? 1 : 0
    : covariance / pearsonDenominator;
  const thresholds = envelope === "full"
    ? Object.freeze({
      nrmseMaximum: FULL_NRMSE_MAXIMUM,
      snrDecibelsMinimum: FULL_SNR_DECIBELS_MINIMUM,
      pearsonMinimum: FULL_PEARSON_MINIMUM,
      maximumAbsoluteErrorMaximum: FULL_MAXIMUM_ABSOLUTE_ERROR,
    })
    : Object.freeze({
      nrmseMaximum: ADVERSARIAL_NRMSE_MAXIMUM,
      snrDecibelsMinimum: ADVERSARIAL_SNR_DECIBELS_MINIMUM,
      pearsonMinimum: ADVERSARIAL_PEARSON_MINIMUM,
      maximumAbsoluteErrorMaximum: ADVERSARIAL_MAXIMUM_ABSOLUTE_ERROR,
    });
  const passed = accumulator.controlNonFiniteCount === 0 &&
    accumulator.candidateNonFiniteCount === 0 &&
    nrmse <= thresholds.nrmseMaximum &&
    snrDecibels >= thresholds.snrDecibelsMinimum &&
    pearson >= thresholds.pearsonMinimum &&
    accumulator.maximumAbsoluteError <= thresholds.maximumAbsoluteErrorMaximum;
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

function fillActivation(values: Float32Array, spec: Opt0050CaseSpec): void {
  let physical = 0;
  for (let row = 0; row < spec.shape.rows; row += 1) {
    for (let inner = 0; inner < spec.shape.inner; inner += 1) {
      values[physical] = halfToNumber(activationBitsAt(spec, row, inner));
      physical += 1;
    }
  }
}

function fillLogicalWeight(values: Uint16Array, spec: Opt0050CaseSpec): void {
  let physical = 0;
  for (let inner = 0; inner < spec.shape.inner; inner += 1) {
    for (let column = 0; column < spec.shape.columns; column += 1) {
      values[physical] = weightBitsAt(spec, inner, column);
      physical += 1;
    }
  }
}

function activationBitsAt(
  spec: Opt0050CaseSpec,
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
  if (spec.fixtureKind === "signed-zero") return (mixed >>> 31) << 15;
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

function weightBitsAt(
  spec: Opt0050CaseSpec,
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
  spec: Opt0050CaseSpec,
): GuardedOutput {
  const outputElements = spec.shape.rows * spec.shape.columns;
  const outputBytes = outputElements * 4;
  const totalBytes = outputBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = tracker.create(device, {
    label: `opt0050-${spec.id}-prefill`,
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
    label: `opt0050-${spec.id}-output`,
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  const readback = tracker.create(device, {
    label: `opt0050-${spec.id}-readback`,
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
    commandBufferCount: 1,
    queueDrainCount: 1,
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

function requireAdapter(adapter: GPUAdapter): void {
  const maximumStorageBytes = maximumStorageBindingBytes();
  if (
    !adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    Number(adapter.info.subgroupMinSize) !== 32 ||
    Number(adapter.info.subgroupMaxSize) !== 32 ||
    Number(adapter.limits.maxComputeInvocationsPerWorkgroup) < 128 ||
    Number(adapter.limits.maxComputeWorkgroupSizeX) < 128 ||
    Number(adapter.limits.maxStorageBufferBindingSize) < maximumStorageBytes ||
    Number(adapter.limits.maxBufferSize) <
      maximumStorageBytes + 2 * STORAGE_GUARD_BYTES
  ) {
    throw new Error(
      "OPT-0050 requires stock Chrome shader-f16, fixed32 subgroups, WG128, and full-shape storage limits",
    );
  }
}

function maximumStorageBindingBytes(): number {
  const cases = buildOpt0050Cases();
  return Math.max(...[...cases.full, ...cases.adversarial].flatMap(({ shape }) => [
    shape.rows * shape.inner * 4,
    shape.inner * shape.columns * 2,
    shape.rows * shape.columns * 4,
  ]));
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
    features: Object.freeze([...device.features].sort()),
    limits: Object.freeze({
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    }),
    fixtureVersion: "opt0032-full-and-adversarial-fp16-v1",
  });
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
): Opt0050CaseSpec {
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
): Opt0050CaseSpec {
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
  if (
    samples.length !== 2 ||
    samples.some((sample) => !Number.isFinite(sample) || sample <= 0)
  ) {
    throw new Error(`OPT-0050 ${label} requires two finite positive samples`);
  }
  return (samples[0]! + samples[1]!) / 2;
}

function requireLittleEndianHost(): void {
  const words = new Uint16Array([0x0102]);
  if (new Uint8Array(words.buffer)[0] !== 0x02) {
    throw new Error("OPT-0050 fixtures require a little-endian host");
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
    throw new Error(`OPT-0050 field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0050 field ${name} is invalid`);
  }
  return value;
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing OPT-0050 element ${selector}`);
  return element;
}

function finishPage(
  status: "passed" | "failed",
  receipt: Readonly<Record<string, unknown>>,
): void {
  window.__ACE_OPT0050_RESULT__ = receipt;
  document.body.dataset.status = status;
  requireElement<HTMLElement>("#progress").textContent = status;
  requireElement<HTMLElement>("#result").textContent =
    JSON.stringify(receipt, null, 2);
}

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0050-dense-k4-output-vector-fma-v1",
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
