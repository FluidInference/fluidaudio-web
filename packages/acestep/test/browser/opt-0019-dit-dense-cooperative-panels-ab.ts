/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import currentCoreSource from
  "../../src/webgpu/kernels/dit-dense-fp16.ts?raw";
import candidateCoreSource from
  "../../src/webgpu/kernels/dit-dense-fp16-cooperative-panels.ts?raw";
import {
  AceOpt0009DenseGemmKernel,
  aceOpt0009DenseGemmWgsl,
  planAceOpt0009DenseGemm,
} from "../../src/webgpu/kernels/dit-dense-fp16.js";
import {
  ACE_OPT_0019_DENSE_COOPERATIVE_KERNEL_SET_ID,
  AceOpt0019DenseCooperativePanelsKernel,
  aceOpt0019DenseCooperativePanelsWgsl,
  planAceOpt0019DenseCooperativePanels,
} from "../../src/webgpu/kernels/dit-dense-fp16-cooperative-panels.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";

export type Opt0019Arm = "current" | "candidate";

export interface Opt0019ShapeSpec {
  readonly id: "h-h" | "h-1024" | "h-6144" | "6144-h";
  readonly shape: AceGemmShape;
  readonly productionMultiplicity: 4 | 2 | 1;
  readonly feedForwardMultiplicity: 0 | 2 | 1;
}

export interface Opt0019TimingInput {
  readonly id: Opt0019ShapeSpec["id"];
  readonly samples: Readonly<Record<Opt0019Arm, readonly number[]>>;
}

export interface Opt0019ThermalGate {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: 1_000;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly launchDelayMilliseconds: number;
}

interface GuardedOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly prefill: GPUBuffer;
  readonly readback: GPUBuffer;
  readonly outputElements: number;
  readonly outputBytes: number;
  readonly totalBytes: number;
}

interface PreparedShape {
  readonly spec: Opt0019ShapeSpec;
  readonly activation: GPUBuffer;
  readonly weight: GPUBuffer;
  readonly output: GPUBuffer;
  readonly dispatches: Readonly<Record<Opt0019Arm, AceGemmDispatch>>;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly currentKernel: AceOpt0009DenseGemmKernel;
  readonly candidateKernel: AceOpt0019DenseCooperativePanelsKernel;
  readonly shapes: readonly PreparedShape[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly sourceAuthority: Readonly<Record<string, unknown>>;
  readonly uncapturedErrors: readonly string[];
  readonly preparedCompletedAtEpochMilliseconds: number;
  readonly updateProgress: (message: string) => void;
  destroy(): Readonly<Record<string, unknown>>;
}

interface ReadbackSnapshot {
  readonly words: Uint32Array;
  readonly nonFiniteCount: number;
  readonly nonzeroCount: number;
  readonly qNaNPrefillCount: number;
  readonly prefixCanaryIntact: boolean;
  readonly suffixCanaryIntact: boolean;
  readonly tailRowWritten: boolean;
}

const EXPERIMENT_ID = "OPT-0019" as const;
const REGISTRATION_COMMIT =
  "83de738b5374699778dcaa373d69118a7fbd6715" as const;
const CURRENT_CORE_SOURCE_SHA256 =
  "a238f67da07c6ba1097da9d9e9e97960ae97d2e1d5c129fcbabf69e962cbb6b3" as const;
const CANDIDATE_CORE_SOURCE_SHA256 =
  "b5dad12724882d3fc942c7df7b10c7b7b89a4bed595125ff11a5905c03152a37" as const;
const ROWS = 2_250;
const HIDDEN = 2_048;
const EXPANDED = 6_144;
const CURRENT_TILE_ROWS = 32;
const CURRENT_TILE_COLUMNS = 256;
const CURRENT_TILE_INNER = 32;
const CANDIDATE_TILE_ROWS = 64;
const CANDIDATE_TILE_COLUMNS = 128;
const CANDIDATE_TILE_INNER = 16;
const TIMING_ROUNDS = 4;
const COMPLETE_DENSE_SPEEDUP_THRESHOLD = 1.55;
const COMPLETE_DENSE_SAVING_THRESHOLD_MS = 52.0834;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_1955;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const THERMAL_POLL_MILLISECONDS = 1_000;
const MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS = 1_250;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 5_000;
const ARMS = Object.freeze(["current", "candidate"] as const);
const ACTIVATION_PATTERN = new Float32Array([
  -0.25, -0.1875, -0.125, -0.09375,
  -0.0625, -0.046875, -0.03125, -0.015625,
  0.015625, 0.03125, 0.046875, 0.0625,
  0.09375, 0.125, 0.1875, 0.25,
]);
const WEIGHT_PATTERN = new Uint16Array([
  0xb400, 0xb200, 0xb000, 0xae00,
  0xac00, 0xaa00, 0xa800, 0xa400,
  0x2400, 0x2800, 0x2a00, 0x2c00,
  0x2e00, 0x3000, 0x3200, 0x3400,
]);

const SHAPE_SPECS = Object.freeze([
  shapeSpec("h-h", HIDDEN, HIDDEN, 4, 0),
  shapeSpec("h-1024", HIDDEN, 1_024, 2, 0),
  shapeSpec("h-6144", HIDDEN, EXPANDED, 2, 2),
  shapeSpec("6144-h", EXPANDED, HIDDEN, 1, 1),
]);

export function buildOpt0019ShapeSpecs(): readonly Opt0019ShapeSpec[] {
  return SHAPE_SPECS;
}

export function buildOpt0019TimingOrders(): readonly Readonly<{
  roundIndex: number;
  shapeIndex: number;
  order: readonly Opt0019Arm[];
}>[] {
  return Object.freeze(Array.from({ length: TIMING_ROUNDS }, (_, roundIndex) => {
    const shapeOrder = SHAPE_SPECS.map((_, shapeIndex) => shapeIndex);
    const rotatedShapeOrder = [
      ...shapeOrder.slice(roundIndex),
      ...shapeOrder.slice(0, roundIndex),
    ];
    const armRotation = roundIndex % ARMS.length;
    const order = Object.freeze([
      ...ARMS.slice(armRotation),
      ...ARMS.slice(0, armRotation),
    ]);
    return rotatedShapeOrder.map((shapeIndex) => {
      return Object.freeze({
        roundIndex,
        shapeIndex,
        order,
      });
    });
  }).flat());
}

export function summarizeOpt0019Timing(
  inputs: readonly Opt0019TimingInput[],
): Readonly<Record<string, unknown>> {
  if (inputs.length !== SHAPE_SPECS.length) {
    throw new Error("OPT-0019 requires all four exact-shape timings");
  }
  let completeCurrent = 0;
  let completeCandidate = 0;
  let feedForwardCurrent = 0;
  let feedForwardCandidate = 0;
  const strata = inputs.map((input, index) => {
    const spec = SHAPE_SPECS[index];
    if (spec === undefined || input.id !== spec.id) {
      throw new Error("OPT-0019 timing shape order changed");
    }
    const currentMedian = median4(input.samples.current);
    const candidateMedian = median4(input.samples.candidate);
    completeCurrent += currentMedian * spec.productionMultiplicity;
    completeCandidate += candidateMedian * spec.productionMultiplicity;
    feedForwardCurrent += currentMedian * spec.feedForwardMultiplicity;
    feedForwardCandidate += candidateMedian * spec.feedForwardMultiplicity;
    return Object.freeze({
      id: spec.id,
      shape: spec.shape,
      productionMultiplicity: spec.productionMultiplicity,
      feedForwardMultiplicity: spec.feedForwardMultiplicity,
      samples: input.samples,
      medians: Object.freeze({
        current: currentMedian,
        candidate: candidateMedian,
      }),
      candidateFaster: candidateMedian < currentMedian,
      speedup: currentMedian / candidateMedian,
    });
  });
  const completeDenseSpeedup = completeCurrent / completeCandidate;
  const completeDenseSavingMilliseconds = completeCurrent - completeCandidate;
  const everyShapeFaster = strata.every((stratum) => stratum.candidateFaster);
  const passed = everyShapeFaster &&
    completeDenseSpeedup >= COMPLETE_DENSE_SPEEDUP_THRESHOLD &&
    completeDenseSavingMilliseconds >= COMPLETE_DENSE_SAVING_THRESHOLD_MS;
  return Object.freeze({
    samplesPerArmPerShape: TIMING_ROUNDS,
    completeDense: Object.freeze({
      multiplicities: "4/2/2/1",
      currentMilliseconds: completeCurrent,
      candidateMilliseconds: completeCandidate,
      savingMilliseconds: completeDenseSavingMilliseconds,
      speedup: completeDenseSpeedup,
      speedupThreshold: COMPLETE_DENSE_SPEEDUP_THRESHOLD,
      savingThresholdMilliseconds: COMPLETE_DENSE_SAVING_THRESHOLD_MS,
    }),
    feedForward: Object.freeze({
      multiplicities: "0/0/2/1",
      currentMilliseconds: feedForwardCurrent,
      candidateMilliseconds: feedForwardCandidate,
      savingMilliseconds: feedForwardCurrent - feedForwardCandidate,
      speedup: feedForwardCurrent / feedForwardCandidate,
    }),
    everyShapeFaster,
    strata: Object.freeze(strata),
    passed,
    decision: passed
      ? "positive-primitive-qualifier"
      : "negative-stop-primitive-gate",
  });
}

export function parseOpt0019ThermalGate(
  parameters: URLSearchParams,
  preparedCompletedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0019ThermalGate {
  const source = requiredParameter(parameters, "thermalSource");
  const startedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalStartedAtEpochMilliseconds",
  );
  const completedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalCompletedAtEpochMilliseconds",
  );
  const observationCount = requiredFiniteParameter(
    parameters,
    "thermalObservations",
  );
  const pollMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalPollMilliseconds",
  );
  const maximumPollGapMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = requiredFiniteParameter(
    parameters,
    "thermalNonNominalObservations",
  );
  const durationMilliseconds = completedAtEpochMilliseconds -
    startedAtEpochMilliseconds;
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    completedAtEpochMilliseconds;
  if (source !== THERMAL_SOURCE ||
    !Number.isSafeInteger(observationCount) || observationCount < 31 ||
    pollMilliseconds !== THERMAL_POLL_MILLISECONDS ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS ||
    nonNominalObservationCount !== 0 ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    startedAtEpochMilliseconds < preparedCompletedAtEpochMilliseconds ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS) {
    throw new Error("OPT-0019 thermal gate is incomplete, stale, or non-nominal");
  }
  return Object.freeze({
    source: THERMAL_SOURCE,
    startedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount,
    pollMilliseconds: THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0,
    launchDelayMilliseconds,
  });
}

if (typeof document !== "undefined") installBrowserGate();

function installBrowserGate(): void {
  const progress = requireElement<HTMLElement>("#progress");
  const thermalGate = requireElement<HTMLFieldSetElement>("#thermal-gate");
  const run = requireElement<HTMLButtonElement>("#run");
  let prepared: PreparedGate | undefined;
  void prepareGate((message) => {
    progress.textContent = message;
  }).then(
    (value) => {
      prepared = value;
      document.body.dataset.status = "ready";
      progress.textContent = "ready: collect one 30-second nominal interval";
      thermalGate.disabled = false;
      run.disabled = false;
    },
    (error: unknown) => finishPage("failed", failureReceipt(error)),
  );
  run.addEventListener("click", () => {
    if (prepared === undefined) return;
    run.disabled = true;
    thermalGate.disabled = true;
    document.body.dataset.status = "running";
    progress.textContent = "running four rotated AB/BA timing rounds";
    const owned = prepared;
    prepared = undefined;
    void runTimedGate(owned).then(
      (receipt) => finishPage("passed", receipt),
      (error: unknown) => {
        owned.destroy();
        finishPage("failed", failureReceipt(error));
      },
    );
  }, { once: true });
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  private readonly sizes = new Map<GPUBuffer, number>();
  created = 0;
  destroyed = 0;
  liveBytes = 0;
  maximumLiveBytes = 0;

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

async function prepareGate(
  updateProgress: (message: string) => void,
): Promise<PreparedGate> {
  const sourceAuthority = await buildSourceAuthority();
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const largestBindingBytes = maximumBindingBytes();
  const device = await adapter.requestDevice({
    label: "ace-opt-0019-dense-cooperative-panels-ab-device",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: {
      maxBufferSize: largestBindingBytes + 2 * STORAGE_GUARD_BYTES,
      maxStorageBufferBindingSize: largestBindingBytes,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 16,
      maxComputeWorkgroupStorageSize: 6_400,
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
  const candidateKernel = AceOpt0019DenseCooperativePanelsKernel.create(device);
  const shapes: PreparedShape[] = [];
  let destroyed = false;
  const destroy = (): Readonly<Record<string, unknown>> => {
    if (destroyed) {
      return Object.freeze({ ...tracker.receipt(), idempotent: true });
    }
    destroyed = true;
    currentKernel.destroy();
    candidateKernel.destroy();
    tracker.destroyAll();
    tracker.destroyAll();
    device.destroy();
    return Object.freeze({
      ...tracker.receipt(),
      idempotent: true,
      deviceDestroyed: true,
    });
  };
  try {
    for (const [index, spec] of SHAPE_SPECS.entries()) {
      updateProgress(`full-output correctness ${index + 1}/4: ${spec.id}`);
      shapes.push(await prepareShape(
        device,
        tracker,
        currentKernel,
        candidateKernel,
        spec,
        index,
      ));
      await yieldToBrowser();
    }
    updateProgress("warming both exact kernels on all four shapes");
    for (const shape of shapes) {
      for (const arm of ARMS) {
        await executeTimedDispatch(device, shape.dispatches[arm]);
      }
      await yieldToBrowser();
    }
    await settlePostDrainEvents();
    if (uncapturedErrors.length !== 0) {
      throw new Error(`OPT-0019 observed ${uncapturedErrors.length} GPU errors`);
    }
    const correctnessCases = shapes.map((shape) => shape.correctness);
    return Object.freeze({
      adapter,
      device,
      tracker,
      currentKernel,
      candidateKernel,
      shapes: Object.freeze(shapes),
      correctness: Object.freeze({
        shapeCount: shapes.length,
        executionCount: shapes.length * ARMS.length * 2,
        comparisonsPerOutputWord: 4,
        comparedU32Count: correctnessCases.reduce((sum, item) =>
          sum + Number(item["comparedU32Count"]), 0),
        mismatchCount: 0,
        qNaNPrefillCompleteWrites: true,
        canariesUntouched: true,
        finiteOutputs: true,
        deterministicReruns: true,
        cases: Object.freeze(correctnessCases),
      }),
      sourceAuthority,
      uncapturedErrors,
      preparedCompletedAtEpochMilliseconds: Date.now(),
      updateProgress,
      destroy,
    });
  } catch (error) {
    destroy();
    throw error;
  }
}

async function prepareShape(
  device: GPUDevice,
  tracker: BufferTracker,
  currentKernel: AceOpt0009DenseGemmKernel,
  candidateKernel: AceOpt0019DenseCooperativePanelsKernel,
  spec: Opt0019ShapeSpec,
  ordinal: number,
): Promise<PreparedShape> {
  const currentPlan = planAceOpt0009DenseGemm(spec.shape);
  const candidatePlan = planAceOpt0019DenseCooperativePanels(spec.shape);
  if (currentPlan.tileRows !== CURRENT_TILE_ROWS ||
    currentPlan.tileColumns !== CURRENT_TILE_COLUMNS ||
    currentPlan.tileInner !== CURRENT_TILE_INNER ||
    candidatePlan.tileRows !== CANDIDATE_TILE_ROWS ||
    candidatePlan.tileColumns !== CANDIDATE_TILE_COLUMNS ||
    candidatePlan.tileInner !== CANDIDATE_TILE_INNER) {
    throw new Error(`OPT-0019 ${spec.id} tile authority changed`);
  }
  const activation = createActivationBuffer(device, tracker, spec, ordinal);
  const weight = createWeightBuffer(device, tracker, spec, ordinal);
  const guarded = createGuardedOutput(device, tracker, spec);
  const bindings: AceGemmBufferBindings = Object.freeze({
    activation: binding(activation, currentPlan.activationElements * 4),
    weight: binding(weight, currentPlan.weightElements * 2),
    output: guarded.binding,
  });
  const current = await currentKernel.createDispatch(
    `opt-0019-${spec.id}-current`,
    spec.shape,
    bindings,
  );
  const candidate = await candidateKernel.createDispatch(
    `opt-0019-${spec.id}-candidate`,
    spec.shape,
    bindings,
  );
  const correctness = await verifyCompleteShape(
    device,
    spec,
    guarded,
    Object.freeze({ current, candidate }),
  );
  tracker.destroy(guarded.prefill);
  tracker.destroy(guarded.readback);
  return Object.freeze({
    spec,
    activation,
    weight,
    output: guarded.buffer,
    dispatches: Object.freeze({ current, candidate }),
    correctness: Object.freeze({
      ...correctness,
      currentPlan: compactPlan(currentPlan),
      candidatePlan: compactPlan(candidatePlan),
    }),
  });
}

async function verifyCompleteShape(
  device: GPUDevice,
  spec: Opt0019ShapeSpec,
  output: GuardedOutput,
  dispatches: Readonly<Record<Opt0019Arm, AceGemmDispatch>>,
): Promise<Readonly<Record<string, unknown>>> {
  const currentFirst = await executeCorrectnessDispatch(
    device,
    output,
    dispatches.current,
  );
  requireCompleteSnapshot(currentFirst, `${spec.id} current first`);
  const currentRerun = await executeCorrectnessDispatch(
    device,
    output,
    dispatches.current,
  );
  requireCompleteSnapshot(currentRerun, `${spec.id} current rerun`);
  requireExactComparison(
    currentFirst.words,
    currentRerun.words,
    `${spec.id} current rerun`,
  );
  const candidateFirst = await executeCorrectnessDispatch(
    device,
    output,
    dispatches.candidate,
  );
  requireCompleteSnapshot(candidateFirst, `${spec.id} candidate first`);
  requireExactComparison(
    currentFirst.words,
    candidateFirst.words,
    `${spec.id} candidate first`,
  );
  const candidateRerun = await executeCorrectnessDispatch(
    device,
    output,
    dispatches.candidate,
  );
  requireCompleteSnapshot(candidateRerun, `${spec.id} candidate rerun`);
  requireExactComparison(
    currentFirst.words,
    candidateRerun.words,
    `${spec.id} candidate rerun`,
  );
  requireExactComparison(
    candidateFirst.words,
    candidateRerun.words,
    `${spec.id} candidate self-rerun`,
  );
  const sha256 = await sha256U32(currentFirst.words);
  return Object.freeze({
    id: spec.id,
    shape: spec.shape,
    outputU32Count: output.outputElements,
    executionCount: 4,
    comparisonCount: 4,
    comparedU32Count: output.outputElements * 4,
    mismatchCount: 0,
    outputSha256: sha256,
    nonFiniteCount: 0,
    qNaNPrefillCount: 0,
    nonzeroCount: currentFirst.nonzeroCount,
    currentFirstRerunExact: true,
    candidateFirstRerunExact: true,
    candidateVersusCurrentExact: true,
    prefixCanaryIntact: true,
    suffixCanaryIntact: true,
    tailRowWritten: true,
  });
}

async function executeCorrectnessDispatch(
  device: GPUDevice,
  output: GuardedOutput,
  dispatch: AceGemmDispatch,
): Promise<ReadbackSnapshot> {
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(output.prefill, 0, output.buffer, 0,
    output.totalBytes);
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  encoder.copyBufferToBuffer(output.buffer, 0, output.readback, 0,
    output.totalBytes);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await output.readback.mapAsync(GPUMapMode.READ);
  try {
    const all = new Uint32Array(output.readback.getMappedRange());
    const guardWords = STORAGE_GUARD_BYTES / 4;
    const firstOutput = guardWords;
    const lastOutput = firstOutput + output.outputElements;
    let prefixCanaryIntact = true;
    let suffixCanaryIntact = true;
    for (let index = 0; index < guardWords; index += 1) {
      prefixCanaryIntact &&= all[index] === STORAGE_GUARD_U32;
      suffixCanaryIntact &&=
        all[lastOutput + index] === STORAGE_GUARD_U32;
    }
    const words = all.slice(firstOutput, lastOutput);
    let nonFiniteCount = 0;
    let nonzeroCount = 0;
    let qNaNPrefillCount = 0;
    for (const word of words) {
      if ((word & 0x7f80_0000) === 0x7f80_0000) nonFiniteCount += 1;
      if ((word & 0x7fff_ffff) !== 0) nonzeroCount += 1;
      if (word === OUTPUT_PREFILL_QNAN_U32) qNaNPrefillCount += 1;
    }
    const tailStart = words.length - output.outputElements / ROWS;
    let tailRowWritten = true;
    for (let index = tailStart; index < words.length; index += 1) {
      if (words[index] === OUTPUT_PREFILL_QNAN_U32) {
        tailRowWritten = false;
        break;
      }
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

async function runTimedGate(
  prepared: PreparedGate,
): Promise<Readonly<Record<string, unknown>>> {
  const launchedAtEpochMilliseconds = Date.now();
  const thermal = parseOpt0019ThermalGate(
    collectThermalParameters(),
    prepared.preparedCompletedAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const timingInputs = prepared.shapes.map(({ spec }) => ({
    id: spec.id,
    samples: { current: [], candidate: [] } as Record<Opt0019Arm, number[]>,
  }));
  const timingStartedAtEpochMilliseconds = Date.now();
  for (const entry of buildOpt0019TimingOrders()) {
    const shape = prepared.shapes[entry.shapeIndex];
    const input = timingInputs[entry.shapeIndex];
    if (shape === undefined || input === undefined) {
      throw new Error("OPT-0019 timing topology changed");
    }
    prepared.updateProgress(
      `timing round ${entry.roundIndex + 1}/4, shape ${entry.shapeIndex + 1}/4`,
    );
    for (const arm of entry.order) {
      input.samples[arm].push(await executeTimedDispatch(
        prepared.device,
        shape.dispatches[arm],
      ));
    }
    await yieldToBrowser();
  }
  const timingCompletedAtEpochMilliseconds = Date.now();
  await settlePostDrainEvents();
  if (prepared.uncapturedErrors.length !== 0) {
    throw new Error("OPT-0019 observed an uncaptured GPU error");
  }
  const timing = summarizeOpt0019Timing(timingInputs.map((input) =>
    Object.freeze({
      id: input.id,
      samples: Object.freeze({
        current: Object.freeze([...input.samples.current]),
        candidate: Object.freeze([...input.samples.candidate]),
      }),
    })
  ));
  const environment = environmentReceipt(prepared.adapter, prepared.device);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanup = prepared.destroy();
  return Object.freeze({
    schema: "ace-opt-0019-dit-dense-cooperative-panels-ab-v1",
    status: "passed",
    experimentId: EXPERIMENT_ID,
    classification: "bounded-primitive-decision-gate-not-integrated",
    recordedAt: new Date().toISOString(),
    identity: prepared.sourceAuthority,
    environment,
    protocol: Object.freeze({
      thermal,
      fullOutputCorrectnessCompletedBeforeThermalGate: true,
      bothKernelsCompiledAndWarmedBeforeThermalGate: true,
      timingOrder: "shape order left-rotated by round; AB in rounds 1/3 and BA in rounds 2/4",
      authoritativeTiming: "performance.now-immediately-before-submit-through-matching-queue-drain",
      continuousExternalThermalTraceRequiredThroughCleanup: true,
      oneThirtySecondNominalGate: true,
      unchangedThermalRetryPerformed: false,
    }),
    correctness: prepared.correctness,
    timing: Object.freeze({
      ...timing,
      timingStartedAtEpochMilliseconds,
      timingCompletedAtEpochMilliseconds,
      caveat: "Primitive dense projection only; the later C98 DiT-only gate remains the observed ten-second authority.",
    }),
    decision: Object.freeze({
      disposition: timing["decision"],
      packageNativeEscalationAuthorized: timing["passed"],
      productionIntegrationAuthorized: false,
      m2250IntegrationRunAuthorized: false,
    }),
    memory: memoryBeforeCleanup,
    cleanup,
  });
}

async function executeTimedDispatch(
  device: GPUDevice,
  dispatch: AceGemmDispatch,
): Promise<number> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  const command = encoder.finish();
  const started = performance.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - started;
}

function createActivationBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0019ShapeSpec,
  ordinal: number,
): GPUBuffer {
  const elements = spec.shape.rows * spec.shape.inner;
  const buffer = tracker.create(device, {
    label: `opt-0019-${spec.id}-activation`,
    size: elements * 4,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const values = new Float32Array(buffer.getMappedRange());
  let state = (0x9e37_79b9 ^ (ordinal * 0x45d9_f3b)) >>> 0;
  for (let index = 0; index < values.length; index += 1) {
    state = xorshift32((state + index + 1) >>> 0);
    values[index] = ACTIVATION_PATTERN[state & 15]!;
  }
  buffer.unmap();
  return buffer;
}

function createWeightBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0019ShapeSpec,
  ordinal: number,
): GPUBuffer {
  const elements = spec.shape.inner * spec.shape.columns;
  const buffer = tracker.create(device, {
    label: `opt-0019-${spec.id}-packed-weight`,
    size: elements * 2,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const values = new Uint16Array(buffer.getMappedRange());
  let state = (0x243f_6a88 ^ (ordinal * 0x27d4_eb2d)) >>> 0;
  for (let index = 0; index < values.length; index += 1) {
    state = xorshift32((state + index + 1) >>> 0);
    values[index] = WEIGHT_PATTERN[state & 15]!;
  }
  buffer.unmap();
  return buffer;
}

function createGuardedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0019ShapeSpec,
): GuardedOutput {
  const outputElements = spec.shape.rows * spec.shape.columns;
  const outputBytes = outputElements * 4;
  const totalBytes = outputBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = tracker.create(device, {
    label: `opt-0019-${spec.id}-output-prefill`,
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
    label: `opt-0019-${spec.id}-guarded-output`,
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  const readback = tracker.create(device, {
    label: `opt-0019-${spec.id}-readback`,
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
    outputBytes,
    totalBytes,
  });
}

async function buildSourceAuthority(): Promise<Readonly<Record<string, unknown>>> {
  const currentSourceSha256 = await sha256Text(currentCoreSource);
  const candidateSourceSha256 = await sha256Text(candidateCoreSource);
  if (currentSourceSha256 !== CURRENT_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0019 rejected unauthenticated current dense core");
  }
  if (candidateSourceSha256 !== CANDIDATE_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0019 rejected unauthenticated candidate dense core");
  }
  const generatedShaders = [];
  for (const spec of SHAPE_SPECS) {
    generatedShaders.push(Object.freeze({
      id: spec.id,
      currentSha256: await sha256Text(aceOpt0009DenseGemmWgsl(spec.shape)),
      candidateSha256: await sha256Text(
        aceOpt0019DenseCooperativePanelsWgsl(spec.shape),
      ),
    }));
  }
  return Object.freeze({
    registrationCommit: REGISTRATION_COMMIT,
    currentCoreSourceSha256: currentSourceSha256,
    candidateCoreSourceSha256: candidateSourceSha256,
    candidateKernelSetId: ACE_OPT_0019_DENSE_COOPERATIVE_KERNEL_SET_ID,
    generatedShaders: Object.freeze(generatedShaders),
  });
}

function requireAdapter(adapter: GPUAdapter): void {
  const info = adapter.info;
  const largestBindingBytes = maximumBindingBytes();
  if (!adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    info.subgroupMinSize !== 32 || info.subgroupMaxSize !== 32 ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 256 ||
    adapter.limits.maxComputeWorkgroupSizeX < 256 ||
    adapter.limits.maxComputeWorkgroupSizeY < 16 ||
    adapter.limits.maxComputeWorkgroupStorageSize < 6_400 ||
    adapter.limits.maxStorageBufferBindingSize < largestBindingBytes ||
    adapter.limits.maxBufferSize < largestBindingBytes + 2 * STORAGE_GUARD_BYTES) {
    throw new Error("OPT-0019 requires the authenticated fixed32 WG256 device contract");
  }
}

function environmentReceipt(
  adapter: GPUAdapter,
  device: GPUDevice,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    userAgent: navigator.userAgent,
    page: window.location.href,
    adapterInfo: Object.freeze({
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
      maxComputeWorkgroupSizeY: device.limits.maxComputeWorkgroupSizeY,
      maxComputeWorkgroupStorageSize:
        device.limits.maxComputeWorkgroupStorageSize,
    }),
  });
}

function compactPlan(plan: Readonly<{
  rows: number;
  inner: number;
  columns: number;
  tileRows: number;
  tileColumns: number;
  tileInner: number;
  workgroupSize: number;
  rowTiles: number;
  columnTiles: number;
  innerTiles: number;
  workgroupCount: number;
  outputRanges: readonly Readonly<{ multiplyAdds: number }>[];
}>): Readonly<Record<string, unknown>> {
  return Object.freeze({
    rows: plan.rows,
    inner: plan.inner,
    columns: plan.columns,
    tileRows: plan.tileRows,
    tileColumns: plan.tileColumns,
    tileInner: plan.tileInner,
    workgroupSize: plan.workgroupSize,
    rowTiles: plan.rowTiles,
    columnTiles: plan.columnTiles,
    innerTiles: plan.innerTiles,
    workgroupCount: plan.workgroupCount,
    scheduledMultiplyAdds: plan.outputRanges.reduce(
      (sum, range) => sum + range.multiplyAdds,
      0,
    ),
  });
}

function requireCompleteSnapshot(snapshot: ReadbackSnapshot, label: string): void {
  if (snapshot.nonFiniteCount !== 0 || snapshot.nonzeroCount === 0 ||
    snapshot.qNaNPrefillCount !== 0 || !snapshot.prefixCanaryIntact ||
    !snapshot.suffixCanaryIntact || !snapshot.tailRowWritten) {
    throw new Error(`${label} failed complete-write, finite, tail, or canary gate`);
  }
}

function requireExactComparison(
  expected: Uint32Array,
  actual: Uint32Array,
  label: string,
): void {
  if (expected.length !== actual.length) {
    throw new Error(`${label} output length changed`);
  }
  for (let index = 0; index < expected.length; index += 1) {
    if (expected[index] !== actual[index]) {
      throw new Error(
        `${label} raw-U32 mismatch at ${index}: ` +
          `${expected[index]?.toString(16)} != ${actual[index]?.toString(16)}`,
      );
    }
  }
}

function shapeSpec(
  id: Opt0019ShapeSpec["id"],
  inner: number,
  columns: number,
  productionMultiplicity: Opt0019ShapeSpec["productionMultiplicity"],
  feedForwardMultiplicity: Opt0019ShapeSpec["feedForwardMultiplicity"],
): Opt0019ShapeSpec {
  return Object.freeze({
    id,
    shape: Object.freeze({ rows: ROWS, inner, columns }),
    productionMultiplicity,
    feedForwardMultiplicity,
  });
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function maximumBindingBytes(): number {
  return Math.max(...SHAPE_SPECS.flatMap(({ shape }) => [
    shape.rows * shape.inner * 4,
    shape.inner * shape.columns * 2,
    shape.rows * shape.columns * 4,
  ]));
}

function xorshift32(value: number): number {
  value ^= value << 13;
  value ^= value >>> 17;
  value ^= value << 5;
  return value >>> 0;
}

function median4(samples: readonly number[]): number {
  if (samples.length !== TIMING_ROUNDS ||
    samples.some((sample) => !Number.isFinite(sample) || sample <= 0)) {
    throw new Error("OPT-0019 requires four finite positive samples per arm");
  }
  const sorted = [...samples].sort((left, right) => left - right);
  return (sorted[1]! + sorted[2]!) / 2;
}

function collectThermalParameters(): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    "#thermal-gate input[name]",
  )) parameters.set(input.name, input.value);
  return parameters;
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.length === 0) {
    throw new Error(`OPT-0019 thermal field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0019 thermal field ${name} is not finite`);
  }
  return value;
}

async function sha256Text(value: string): Promise<string> {
  return sha256Bytes(new TextEncoder().encode(value));
}

async function sha256U32(values: Uint32Array): Promise<string> {
  return sha256Bytes(new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  ));
}

async function sha256Bytes(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", Uint8Array.from(bytes));
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function requireElement<ElementType extends Element>(selector: string): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing browser element ${selector}`);
  return element;
}

function finishPage(
  status: "passed" | "failed",
  receipt: Readonly<Record<string, unknown>>,
): void {
  document.body.dataset.status = status;
  requireElement<HTMLElement>("#progress").textContent = status;
  requireElement<HTMLElement>("#result").textContent = JSON.stringify(
    receipt,
    null,
    2,
  );
  (window as typeof window & {
    __ACE_OPT0019_RESULT__?: Readonly<Record<string, unknown>>;
  }).__ACE_OPT0019_RESULT__ = receipt;
}

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0019-dit-dense-cooperative-panels-ab-v1",
    status: "failed",
    experimentId: EXPERIMENT_ID,
    recordedAt: new Date().toISOString(),
    error: error instanceof Error
      ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
      : String(error),
  });
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

async function settlePostDrainEvents(): Promise<void> {
  await yieldToBrowser();
  await yieldToBrowser();
}
