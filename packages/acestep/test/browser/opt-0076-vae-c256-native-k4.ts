/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import currentSelectorSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.ts?raw";
import directK4Source from
  "../../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.ts?raw";
import promotedSelectorSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-c256-k4-selector.ts?raw";
import experimentSource from
  "../../optimization/experiments/OPT-0076-vae-c256-native-k4-promotion.md?raw";
import htmlSource from "./opt-0076-vae-c256-native-k4.html?raw";
import harnessSource from "./opt-0076-vae-c256-native-k4.ts?raw";
import {
  ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
  AceOpt0057VaeK7ShapeSelectorKernel,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import {
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
  AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
  aceOpt0024VaeConv1dDirectDot4SubgroupWgsl,
  planAceOpt0024VaeConv1dDirectDot4SubgroupRange,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.js";
import {
  ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID,
  AceOpt0076VaeC256K4SelectorKernel,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-c256-k4-selector.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dBindings,
  type AceFp16VaeConv1dPlan,
} from "../../src/webgpu/kernels/vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../../src/webgpu/kernels/vae-primitives.js";
import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
  aceFp16VaeConv1dSubgroupWgsl,
  planAceFp16VaeConv1dSubgroupRange,
} from "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";

declare global {
  interface Window {
    __ACE_OPT0076_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

type Arm = "current" | "candidate";
type ExecutionArm = Arm | "oracle";
type FixtureKind = "production-local" | "bounded-adversarial";
type ProbeId = "first" | "interior" | "tail";

interface ProbeSpec {
  readonly id: ProbeId;
  readonly base: number;
  readonly count: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
}

interface CaseSpec {
  readonly id: string;
  readonly operationLabel: string;
  readonly dilation: 1 | 3 | 9;
  readonly fixtureKind: FixtureKind;
  readonly shape: AceVaeConv1dShape;
  readonly probes: readonly ProbeSpec[];
}

interface Encodable {
  readonly label: string;
  readonly kernelId: string;
  readonly selectorKernelId?: string | undefined;
  readonly literalSelectorKernelId?: string | undefined;
  readonly operationLabel?: string | undefined;
  readonly owner?: string | undefined;
  readonly plan?: AceFp16VaeConv1dPlan | undefined;
  readonly outputRange?: Readonly<{
    readonly base: number;
    readonly count: number;
  }> | undefined;
  encode(pass: GPUComputePassEncoder): void;
}

interface GuardedOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly prefill: GPUBuffer;
  readonly logicalBytes: number;
  readonly totalBytes: number;
}

interface PreparedCase {
  readonly spec: CaseSpec;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly output: GuardedOutput;
  readonly dispatches: Readonly<Record<ProbeId,
    Readonly<Record<ExecutionArm, Encodable>>>>;
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly ownedBuffers: readonly GPUBuffer[];
}

interface PreparedHarness {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly currentKernel: AceOpt0057VaeK7ShapeSelectorKernel;
  readonly candidateKernel: AceOpt0076VaeC256K4SelectorKernel;
  readonly oracleKernel: AceOpt0024VaeConv1dDirectDot4SubgroupKernel;
  readonly querySet: GPUQuerySet;
  readonly queryResolve: GPUBuffer;
  readonly queryReadback: GPUBuffer;
  readonly productionCases: readonly PreparedCase[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly identity: Readonly<Record<string, unknown>>;
  readonly uncapturedErrors: readonly string[];
  readonly deviceLosses: readonly string[];
  destroy(): Readonly<Record<string, unknown>>;
}

interface OutputSnapshot {
  readonly words: Uint16Array<ArrayBuffer>;
  readonly sha256: string;
  readonly nonFiniteCount: number;
  readonly qNaNPrefillCount: number;
  readonly prefixGuardIntact: boolean;
  readonly suffixGuardIntact: boolean;
  readonly adjacentBeforeIntact: boolean;
  readonly adjacentAfterIntact: boolean;
  readonly outOfRangeWriteCount: number;
  readonly tailWritten: boolean;
}

interface TimestampSample {
  readonly submitAtPerformanceMilliseconds: number;
  readonly fenceAtPerformanceMilliseconds: number;
  readonly submitAtEpochMilliseconds: number;
  readonly fenceAtEpochMilliseconds: number;
  readonly wallMilliseconds: number;
  readonly gpuMilliseconds: number;
  readonly timestampBeginNanoseconds: string;
  readonly timestampEndNanoseconds: string;
  readonly gpuElapsedNanoseconds: number;
  readonly gpuToWallRatio: number;
  readonly dispatchCount: 9;
  readonly commandBufferCount: 1;
  readonly queueDrainCount: 1;
}

interface TimingFailureEvidence {
  rawSamples: Readonly<Record<string, unknown>>[];
  readyAtEpochMilliseconds: number;
  runStartedAtEpochMilliseconds?: number;
  measurementCompletedAtEpochMilliseconds?: number;
}

interface NumericalAccumulator {
  count: number;
  rawU16MismatchCount: number;
  signedZeroDifferenceCount: number;
  sumError: number;
  sumAbsoluteError: number;
  sumSquaredError: number;
  sumControlSquared: number;
  sumControl: number;
  sumCandidate: number;
  sumCandidateSquared: number;
  sumProduct: number;
  maximumAbsoluteError: number;
  controlPeak: number;
  controlMinimum: number;
  controlMaximum: number;
  candidateMinimum: number;
  candidateMaximum: number;
  ulpDistribution: Map<number, number>;
  firstDifference: Readonly<Record<string, unknown>> | null;
  worstDifference: Readonly<Record<string, unknown>> | null;
}

const EXPERIMENT_ID = "OPT-0076" as const;
const RECEIPT_SCHEMA = "ace-opt-0076-vae-c256-native-k4-v1";
const INPUT_FRAMES = 2_400;
const CHANNELS = 256;
const OUTPUT_RANGE_ROWS = 64;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_F16 = 0x7e55;
const TIMESTAMP_QUERY_BYTES = 16;
const TIMING_ROUNDS = Object.freeze([
  Object.freeze(["current", "candidate"] as const),
  Object.freeze(["candidate", "current"] as const),
  Object.freeze(["candidate", "current"] as const),
  Object.freeze(["current", "candidate"] as const),
  Object.freeze(["current", "candidate"] as const),
  Object.freeze(["candidate", "current"] as const),
]);
const REQUIRED_SPEEDUP = 2;
const NRMSE_MAXIMUM = 0.001;
const SNR_MINIMUM_DB = 60;
const PEARSON_MINIMUM = 0.99999;
const RELATIVE_MAXIMUM_ABSOLUTE_ERROR_MAXIMUM = 0.01;
const INPUT_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x2400, 0xa400, 0x2c00, 0xac00, 0x3000, 0xb000,
  0x3400, 0xb400, 0x3555, 0xb555, 0x1800, 0x9800, 0x0001, 0x8001,
]);
const BIAS_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x2000, 0xa000, 0x2400, 0xa400, 0x2800, 0xa800,
]);
const ADVERSARIAL_INPUT_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
  0x1000, 0x9000, 0x2400, 0xa400, 0x3555, 0xb555, 0x3800, 0xb800,
]);

const PROBES = Object.freeze([
  probe("first", 0, OUTPUT_RANGE_ROWS),
  probe("interior", 1_152, OUTPUT_RANGE_ROWS),
  probe("tail", INPUT_FRAMES - 32, 32),
]);
const PRODUCTION_CASES = Object.freeze([
  caseSpec(1, "production-local"),
  caseSpec(3, "production-local"),
  caseSpec(9, "production-local"),
]);
const ADVERSARIAL_CASES = Object.freeze([
  caseSpec(1, "bounded-adversarial"),
  caseSpec(3, "bounded-adversarial"),
  caseSpec(9, "bounded-adversarial"),
]);

const progress = requireElement<HTMLElement>("#progress");
const runButton = requireElement<HTMLButtonElement>("#run");
const result = requireElement<HTMLElement>("#result");
let active: PreparedHarness | undefined;
let running: PreparedHarness | undefined;
let started = false;
let readyAtEpochMilliseconds: number | null = null;
let preparationFailureEvidence: Readonly<Record<string, unknown>> =
  Object.freeze({ stage: "before-device", readyAtEpochMilliseconds: null });

void prepareHarness().then(onPrepared, (error: unknown) => {
  fail(error, preparationFailureEvidence);
});

runButton.addEventListener("click", () => {
  if (started || active === undefined) return;
  started = true;
  runButton.disabled = true;
  document.body.dataset.status = "running";
  const prepared = active;
  active = undefined;
  running = prepared;
  if (readyAtEpochMilliseconds === null) {
    const cleanupFirst = prepared.destroy();
    const cleanupSecond = prepared.destroy();
    running = undefined;
    fail(new Error("OPT-0076 timing cannot start before READY"),
      Object.freeze({ readyAtEpochMilliseconds: null,
        cleanup: Object.freeze({ firstCall: cleanupFirst,
          secondCall: cleanupSecond }) }));
    return;
  }
  const failureEvidence: TimingFailureEvidence = {
    rawSamples: [],
    readyAtEpochMilliseconds,
  };
  void runTiming(prepared, failureEvidence).then(
    () => { running = undefined; },
    (error: unknown) => {
      const memoryBeforeCleanup = prepared.tracker.receipt();
      const cleanupStartedAtEpochMilliseconds = Date.now();
      const cleanupFirst = prepared.destroy();
      const cleanupSecond = prepared.destroy();
      const cleanupCompletedAtEpochMilliseconds = Date.now();
      running = undefined;
      fail(error, Object.freeze({
        stage: "timing",
        readyAtEpochMilliseconds: failureEvidence.readyAtEpochMilliseconds,
        identity: prepared.identity,
        correctness: prepared.correctness,
        timing: Object.freeze({ ...failureEvidence,
          rawSamples: Object.freeze(failureEvidence.rawSamples.slice()) }),
        uncapturedGpuErrors: Object.freeze([...prepared.uncapturedErrors]),
        deviceLosses: Object.freeze([...prepared.deviceLosses]),
        memoryBeforeCleanup,
        cleanup: Object.freeze({ cleanupStartedAtEpochMilliseconds,
          cleanupCompletedAtEpochMilliseconds, firstCall: cleanupFirst,
          secondCall: cleanupSecond }),
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

function onPrepared(prepared: PreparedHarness): void {
  if (prepared.correctness["passed"] !== true) {
    const cleanupFirst = prepared.destroy();
    const cleanupSecond = prepared.destroy();
    publish(Object.freeze({
      schema: RECEIPT_SCHEMA,
      experiment: EXPERIMENT_ID,
      status: "correctness-stop",
      passed: false,
      readyAtEpochMilliseconds: null,
      identity: prepared.identity,
      correctness: prepared.correctness,
      cleanup: Object.freeze({ firstCall: cleanupFirst, secondCall: cleanupSecond }),
      decision: "negative-stop-selector-or-primitive-gate",
    }), "failed");
    return;
  }
  active = prepared;
  readyAtEpochMilliseconds = Date.now();
  document.body.dataset.status = "ready";
  progress.textContent =
    "READY — selector/oracle identity and numerical gates passed; timing has not run";
  runButton.disabled = false;
}

async function prepareHarness(): Promise<PreparedHarness> {
  requireLittleEndianHost();
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const requiredStorageBytes = maximumStorageBindingBytes();
  const device = await adapter.requestDevice({
    label: "ace-opt-0076-c256-native-k4-device",
    requiredFeatures: ["shader-f16", "subgroups", "timestamp-query"],
    requiredLimits: {
      maxBufferSize: requiredStorageBytes + 2 * STORAGE_GUARD_BYTES,
      maxStorageBufferBindingSize: requiredStorageBytes,
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 128,
    },
  });
  const tracker = new BufferTracker();
  const uncapturedErrors: string[] = [];
  const deviceLosses: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncapturedErrors.push(event.error.message);
  });
  void device.lost.then((info) => {
    if (info.reason !== "destroyed") {
      deviceLosses.push(`${info.reason}: ${info.message}`);
    }
  });
  const capability = Object.freeze({
    subgroupMinSize: 32 as const,
    subgroupMaxSize: 32 as const,
  });
  let currentKernel: AceOpt0057VaeK7ShapeSelectorKernel | undefined;
  let candidateKernel: AceOpt0076VaeC256K4SelectorKernel | undefined;
  let oracleKernel: AceOpt0024VaeConv1dDirectDot4SubgroupKernel | undefined;
  let querySet: GPUQuerySet | undefined;
  let queryResolve: GPUBuffer | undefined;
  let queryReadback: GPUBuffer | undefined;
  let destroyed = false;
  const destroy = (): Readonly<Record<string, unknown>> => {
    if (destroyed) {
      return Object.freeze({ ...tracker.receipt(), idempotent: true,
        repeatedCall: true });
    }
    destroyed = true;
    currentKernel?.destroy();
    candidateKernel?.destroy();
    oracleKernel?.destroy();
    querySet?.destroy();
    tracker.destroyAll();
    device.destroy();
    return Object.freeze({ ...tracker.receipt(), idempotent: true,
      repeatedCall: false, deviceDestroyed: true });
  };
  try {
    const identity = await buildIdentity(adapter, device);
    currentKernel = AceOpt0057VaeK7ShapeSelectorKernel.create(
      device,
      capability,
    );
    candidateKernel = AceOpt0076VaeC256K4SelectorKernel.create(
      device,
      capability,
    );
    oracleKernel = AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      device,
      capability,
    );
    querySet = device.createQuerySet({
      label: "opt0076-composite-timestamps",
      type: "timestamp",
      count: 2,
    });
    queryResolve = tracker.create(device, {
      label: "opt0076-timestamp-resolve",
      size: TIMESTAMP_QUERY_BYTES,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    queryReadback = tracker.create(device, {
      label: "opt0076-timestamp-readback",
      size: TIMESTAMP_QUERY_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const allNumerics = createNumericalAccumulator();
    const productionNumerics = createNumericalAccumulator();
    const adversarialNumerics = createNumericalAccumulator();
    const productionCases: PreparedCase[] = [];
    const allCorrectness: Readonly<Record<string, unknown>>[] = [];
    let exactOracleU16Count = 0;
    let deterministicCandidateU16Count = 0;
    let deterministicOracleU16Count = 0;
    const specs = [...PRODUCTION_CASES, ...ADVERSARIAL_CASES];
    for (const [index, spec] of specs.entries()) {
      progress.textContent =
        `correctness ${index + 1}/${specs.length}: ${spec.id}`;
      const prepared = await prepareCase(
        device,
        tracker,
        currentKernel,
        candidateKernel,
        oracleKernel,
        spec,
        allNumerics,
        spec.fixtureKind === "production-local"
          ? productionNumerics
          : adversarialNumerics,
      );
      allCorrectness.push(prepared.correctness);
      exactOracleU16Count += Number(
        prepared.correctness["candidateOracleComparedU16Count"],
      );
      deterministicCandidateU16Count += Number(
        prepared.correctness["candidateRerunComparedU16Count"],
      );
      deterministicOracleU16Count += Number(
        prepared.correctness["oracleRerunComparedU16Count"],
      );
      tracker.destroy(prepared.output.prefill);
      if (spec.fixtureKind === "production-local") {
        productionCases.push(prepared);
      } else {
        for (const buffer of prepared.ownedBuffers) tracker.destroy(buffer);
      }
      await yieldToBrowser();
    }
    await device.queue.onSubmittedWorkDone();
    await settlePostDrainEvents();
    const aggregateNumerics = summarizeNumerics(allNumerics);
    const productionNumericalMetrics = summarizeNumerics(productionNumerics);
    const adversarialNumericalMetrics = summarizeNumerics(adversarialNumerics);
    const numericalEnvelopePassed = numericalPassed(aggregateNumerics);
    const expectedComparedU16Count = 6 * PROBES.reduce(
      (sum, probeSpec) => sum + probeSpec.count,
      0,
    );
    const correctnessPassed =
      exactOracleU16Count === expectedComparedU16Count &&
      deterministicCandidateU16Count === expectedComparedU16Count &&
      deterministicOracleU16Count === expectedComparedU16Count &&
      allCorrectness.every((entry) => entry["passed"] === true) &&
      numericalEnvelopePassed && uncapturedErrors.length === 0 &&
      deviceLosses.length === 0;
    const correctness = Object.freeze({
      caseCount: allCorrectness.length,
      productionCaseCount: PRODUCTION_CASES.length,
      adversarialCaseCount: ADVERSARIAL_CASES.length,
      probesPerCase: PROBES.length,
      expectedComparedU16Count,
      candidateOracleComparedU16Count: exactOracleU16Count,
      candidateRerunComparedU16Count: deterministicCandidateU16Count,
      oracleRerunComparedU16Count: deterministicOracleU16Count,
      exactSelectorOracleRawU16: allCorrectness.every((entry) =>
        entry["exactSelectorOracleRawU16"] === true),
      deterministicCandidateAndOracleReruns: allCorrectness.every((entry) =>
        entry["candidateAndOracleRerunsDeterministic"] === true),
      allOutputsFiniteAndComplete: allCorrectness.every((entry) =>
        entry["allOutputsFiniteAndComplete"] === true),
      guardsAndTailsIntact: allCorrectness.every((entry) =>
        entry["guardsAndTailsIntact"] === true),
      numericalEnvelope: Object.freeze({
        thresholds: Object.freeze({ nrmseMaximum: NRMSE_MAXIMUM,
          snrMinimumDb: SNR_MINIMUM_DB, pearsonMinimum: PEARSON_MINIMUM,
          relativeMaximumAbsoluteErrorMaximum:
            RELATIVE_MAXIMUM_ABSOLUTE_ERROR_MAXIMUM }),
        aggregate: aggregateNumerics,
        productionLocal: productionNumericalMetrics,
        boundedAdversarial: adversarialNumericalMetrics,
        passed: numericalEnvelopePassed,
      }),
      cases: Object.freeze(allCorrectness),
      uncapturedGpuErrorCount: uncapturedErrors.length,
      deviceLossCount: deviceLosses.length,
      completedBeforeReady: true,
      passed: correctnessPassed,
    });
    return Object.freeze({ adapter, device, tracker, currentKernel,
      candidateKernel, oracleKernel, querySet, queryResolve, queryReadback,
      productionCases: Object.freeze(productionCases), correctness, identity,
      uncapturedErrors, deviceLosses, destroy });
  } catch (error) {
    await device.queue.onSubmittedWorkDone().catch(() => undefined);
    const memoryBeforeCleanup = tracker.receipt();
    const cleanupFirst = destroy();
    const cleanupSecond = destroy();
    preparationFailureEvidence = Object.freeze({ stage: "preparation",
      readyAtEpochMilliseconds: null,
      memoryBeforeCleanup, cleanup: Object.freeze({ firstCall: cleanupFirst,
        secondCall: cleanupSecond }) });
    throw error;
  }
}

async function prepareCase(
  device: GPUDevice,
  tracker: BufferTracker,
  currentKernel: AceOpt0057VaeK7ShapeSelectorKernel,
  candidateKernel: AceOpt0076VaeC256K4SelectorKernel,
  oracleKernel: AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
  spec: CaseSpec,
  allNumerics: NumericalAccumulator,
  tierNumerics: NumericalAccumulator,
): Promise<PreparedCase> {
  const plan = planAceFp16VaeConv1d(spec.shape, "float16");
  const input = createFilledBuffer(device, tracker, `${spec.id}-input`,
    plan.inputBindingBytes, spec.fixtureKind === "production-local"
      ? INPUT_PATTERN : ADVERSARIAL_INPUT_PATTERN);
  const weight = createWeightBuffer(device, tracker, spec, plan);
  const bias = createFilledBuffer(device, tracker, `${spec.id}-bias`,
    plan.biasBindingBytes, BIAS_PATTERN);
  const output = createGuardedOutput(device, tracker, spec, plan);
  const controls = PROBES.map((probeSpec) => createRangeControl(
    device,
    tracker,
    spec,
    probeSpec,
  ));
  const ownedBuffers = Object.freeze([
    input,
    weight,
    bias,
    output.buffer,
    output.prefill,
    ...controls,
  ]);
  const bindings: AceFp16VaeConv1dBindings = Object.freeze({
    input: binding(input, plan.inputBindingBytes),
    weight: binding(weight, plan.weightBindingBytes),
    bias: binding(bias, plan.biasBindingBytes),
    output: output.binding,
  });
  const dispatchEntries = await Promise.all(PROBES.map(async (
    probeSpec,
    probeIndex,
  ) => {
    const range = rangeBinding(controls[probeIndex]!, probeSpec);
    const currentDispatch = await currentKernel.createDispatch(
      `${spec.id}-${probeSpec.id}-current`, spec.operationLabel, spec.shape,
      bindings, "float16", range,
    );
    const candidateDispatch = await candidateKernel.createDispatch(
      `${spec.id}-${probeSpec.id}-candidate`, spec.operationLabel, spec.shape,
      bindings, "float16", range,
    );
    const oracleDispatch = await oracleKernel.createDispatch(
      `${spec.id}-${probeSpec.id}-oracle`, spec.shape, bindings, "float16",
      range,
    );
    const currentRange = planAceFp16VaeConv1dSubgroupRange(
      currentDispatch.plan,
      probeSpec,
    );
    const candidateRange =
      planAceOpt0024VaeConv1dDirectDot4SubgroupRange(
        candidateDispatch.plan,
        probeSpec,
      );
    const oracleRange =
      planAceOpt0024VaeConv1dDirectDot4SubgroupRange(
        oracleDispatch.plan,
        probeSpec,
      );
    if (oracleDispatch.outputRange.base !== oracleRange.base ||
      oracleDispatch.outputRange.count !== oracleRange.count) {
      throw new Error(`OPT-0076 ${spec.id} ${probeSpec.id} oracle range changed`);
    }
    const current = withRangeMetadata(currentDispatch, currentRange);
    const candidate = withRangeMetadata(candidateDispatch, candidateRange);
    const oracle = withRangeMetadata(oracleDispatch, oracleRange);
    requireDispatchIdentity(spec, probeSpec, current, candidate, oracle);
    return [probeSpec.id, Object.freeze({ current, candidate, oracle })] as const;
  }));
  const dispatches = Object.freeze(Object.fromEntries(dispatchEntries)) as unknown as
    Readonly<Record<ProbeId, Readonly<Record<ExecutionArm, Encodable>>>>;
  const inputHashes = Object.freeze({
    input: await hashBufferUploadPattern(
      plan.inputBindingBytes,
      spec.fixtureKind === "production-local"
        ? INPUT_PATTERN
        : ADVERSARIAL_INPUT_PATTERN,
    ),
    weight: await hashWeightFixture(spec, plan),
    bias: await hashBufferUploadPattern(plan.biasBindingBytes, BIAS_PATTERN),
  });
  const correctness = await verifyCase(device, tracker, spec, output,
    dispatches, allNumerics, tierNumerics, inputHashes);
  return Object.freeze({ spec, plan, output, dispatches, correctness,
    ownedBuffers });
}

function createFilledBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  bytes: number,
  pattern: Uint16Array,
): GPUBuffer {
  const buffer = tracker.create(device, {
    label,
    size: bytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  fillPeriodic(new Uint16Array(buffer.getMappedRange()), pattern);
  buffer.unmap();
  return buffer;
}

function createWeightBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: CaseSpec,
  plan: AceFp16VaeConv1dPlan,
): GPUBuffer {
  const buffer = tracker.create(device, {
    label: `${spec.id}-native-o-k-i-weight`,
    size: plan.weightBindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const words = new Uint16Array(buffer.getMappedRange());
  for (let index = 0; index < words.length; index += 1) {
    words[index] = deterministicWeightBits(spec, index);
  }
  buffer.unmap();
  return buffer;
}

function createGuardedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: CaseSpec,
  plan: AceFp16VaeConv1dPlan,
): GuardedOutput {
  const logicalBytes = plan.outputBindingBytes;
  const totalBytes = logicalBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = tracker.create(device, {
    label: `${spec.id}-output-prefill`,
    size: totalBytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  const words = new Uint32Array(prefill.getMappedRange());
  words.fill(STORAGE_GUARD_U32);
  new Uint16Array(
    words.buffer,
    STORAGE_GUARD_BYTES,
    plan.outputElements,
  ).fill(OUTPUT_PREFILL_QNAN_F16);
  prefill.unmap();
  const buffer = tracker.create(device, {
    label: `${spec.id}-guarded-output`,
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  return Object.freeze({
    buffer,
    binding: Object.freeze({ buffer, offset: STORAGE_GUARD_BYTES,
      size: logicalBytes }),
    prefill,
    logicalBytes,
    totalBytes,
  });
}

function createRangeControl(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: CaseSpec,
  probeSpec: ProbeSpec,
): GPUBuffer {
  const buffer = tracker.create(device, {
    label: `${spec.id}-${probeSpec.id}-range`,
    size: 16,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set([
    probeSpec.base,
    probeSpec.count,
    0,
    0,
  ]);
  buffer.unmap();
  return buffer;
}

function withRangeMetadata(
  dispatch: Encodable,
  outputRange: Readonly<{ readonly base: number; readonly count: number }>,
): Encodable {
  return Object.freeze({
    label: dispatch.label,
    kernelId: dispatch.kernelId,
    selectorKernelId: dispatch.selectorKernelId,
    literalSelectorKernelId: dispatch.literalSelectorKernelId,
    operationLabel: dispatch.operationLabel,
    owner: dispatch.owner,
    plan: dispatch.plan,
    outputRange: Object.freeze({ base: outputRange.base, count: outputRange.count }),
    encode(pass: GPUComputePassEncoder): void {
      dispatch.encode(pass);
    },
  });
}

function requireDispatchIdentity(
  spec: CaseSpec,
  probeSpec: ProbeSpec,
  current: Encodable,
  candidate: Encodable,
  oracle: Encodable,
): void {
  if (
    current.selectorKernelId !== ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID ||
    current.kernelId !== ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID ||
    current.owner !== "native-scalar-fp32" ||
    candidate.selectorKernelId !==
      ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID ||
    candidate.literalSelectorKernelId !==
      ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID ||
    candidate.owner !== "native-k4" ||
    current.operationLabel !== spec.operationLabel ||
    candidate.operationLabel !== spec.operationLabel ||
    candidate.kernelId !==
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID ||
    oracle.kernelId !==
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID ||
    current.plan?.dilation !== spec.dilation ||
    candidate.plan?.dilation !== spec.dilation ||
    oracle.plan?.dilation !== spec.dilation ||
    current.plan?.outputElements !== INPUT_FRAMES * CHANNELS ||
    candidate.plan?.outputElements !== INPUT_FRAMES * CHANNELS ||
    oracle.plan?.outputElements !== INPUT_FRAMES * CHANNELS ||
    current.outputRange?.base !== probeSpec.base ||
    current.outputRange?.count !== probeSpec.count ||
    candidate.outputRange?.base !== probeSpec.base ||
    candidate.outputRange?.count !== probeSpec.count ||
    oracle.outputRange?.base !== probeSpec.base ||
    oracle.outputRange?.count !== probeSpec.count
  ) {
    throw new Error(
      `OPT-0076 ${spec.id} ${probeSpec.id} selected an unexpected owner`,
    );
  }
}

async function verifyCase(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: CaseSpec,
  output: GuardedOutput,
  dispatches: Readonly<Record<ProbeId,
    Readonly<Record<ExecutionArm, Encodable>>>>,
  allNumerics: NumericalAccumulator,
  tierNumerics: NumericalAccumulator,
  inputHashes: Readonly<Record<string, string>>,
): Promise<Readonly<Record<string, unknown>>> {
  const probeReceipts: Readonly<Record<string, unknown>>[] = [];
  let candidateOracleComparedU16Count = 0;
  let candidateRerunComparedU16Count = 0;
  let oracleRerunComparedU16Count = 0;
  for (const probeSpec of spec.probes) {
    const set = dispatches[probeSpec.id];
    const current = await executeCorrectness(
      device, tracker, output, set.current, probeSpec,
    );
    const oracle = await executeCorrectness(
      device, tracker, output, set.oracle, probeSpec,
    );
    const candidate = await executeCorrectness(
      device, tracker, output, set.candidate, probeSpec,
    );
    const candidateRerun = await executeCorrectness(
      device, tracker, output, set.candidate, probeSpec,
    );
    const oracleRerun = await executeCorrectness(
      device, tracker, output, set.oracle, probeSpec,
    );
    const candidateOracle = compareRawWords(
      oracle.words,
      candidate.words,
      spec,
      probeSpec,
      "candidate-vs-oracle",
    );
    const candidateRepeat = compareRawWords(
      candidate.words,
      candidateRerun.words,
      spec,
      probeSpec,
      "candidate-rerun",
    );
    const oracleRepeat = compareRawWords(
      oracle.words,
      oracleRerun.words,
      spec,
      probeSpec,
      "oracle-rerun",
    );
    const numerical = compareNumerics(
      current.words,
      oracle.words,
      spec,
      probeSpec,
    );
    mergeNumericalAccumulator(allNumerics, numerical);
    mergeNumericalAccumulator(tierNumerics, numerical);
    candidateOracleComparedU16Count += candidateOracle.count;
    candidateRerunComparedU16Count += candidateRepeat.count;
    oracleRerunComparedU16Count += oracleRepeat.count;
    const snapshots = [current, oracle, candidate, candidateRerun, oracleRerun];
    const complete = snapshots.every(snapshotPassed);
    const candidateAndOracleRerunsDeterministic =
      candidateRepeat.differingU16Count === 0 &&
      oracleRepeat.differingU16Count === 0;
    const guardsAndTailsIntact = snapshots.every((snapshot) =>
      snapshot.prefixGuardIntact && snapshot.suffixGuardIntact &&
      snapshot.adjacentBeforeIntact && snapshot.adjacentAfterIntact &&
      snapshot.outOfRangeWriteCount === 0 && snapshot.tailWritten
    );
    const numericalMetrics = summarizeNumerics(numerical);
    const numericalEnvelopePassed = numericalPassed(numericalMetrics);
    const passed = complete && candidateOracle.differingU16Count === 0 &&
      candidateRepeat.differingU16Count === 0 &&
      oracleRepeat.differingU16Count === 0 && numericalEnvelopePassed;
    probeReceipts.push(Object.freeze({
      id: probeSpec.id,
      base: probeSpec.base,
      count: probeSpec.count,
      firstOutputRow: probeSpec.firstOutputRow,
      outputRowCount: probeSpec.outputRowCount,
      executionOrder: Object.freeze([
        "current", "oracle", "candidate", "candidate", "oracle",
      ]),
      selectorAndKernelIds: Object.freeze({
        currentSelector: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
        currentKernel: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
        candidateSelector: ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID,
        candidateLiteralSelector:
          ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
        candidateAndOracleKernel:
          ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
      }),
      dispatchRangeMetadata: Object.freeze({
        current: set.current.outputRange,
        candidate: set.candidate.outputRange,
        oracle: set.oracle.outputRange,
        matchesProbe: ([set.current, set.candidate, set.oracle]).every(
          (dispatch) => dispatch.outputRange?.base === probeSpec.base &&
            dispatch.outputRange.count === probeSpec.count,
        ),
      }),
      hashes: Object.freeze({ current: current.sha256, oracle: oracle.sha256,
        candidate: candidate.sha256, candidateRerun: candidateRerun.sha256,
        oracleRerun: oracleRerun.sha256 }),
      candidateOracle,
      candidateRepeat,
      oracleRepeat,
      snapshotChecks: Object.freeze(snapshots.map(snapshotReceipt)),
      candidateAndOracleRerunsDeterministic,
      guardsAndTailsIntact,
      allOutputsFiniteAndComplete: complete,
      currentVersusOracle: numericalMetrics,
      numericalEnvelopePassed,
      passed,
    }));
  }
  const allOutputsFiniteAndComplete = probeReceipts.every((entry) =>
    entry["allOutputsFiniteAndComplete"] === true);
  const candidateAndOracleRerunsDeterministic = probeReceipts.every((entry) =>
    entry["candidateAndOracleRerunsDeterministic"] === true);
  const guardsAndTailsIntact = probeReceipts.every((entry) =>
    entry["guardsAndTailsIntact"] === true);
  return Object.freeze({
    id: spec.id,
    fixtureKind: spec.fixtureKind,
    operationLabel: spec.operationLabel,
    dilation: spec.dilation,
    shape: spec.shape,
    inputHashes,
    candidateOracleComparedU16Count,
    candidateRerunComparedU16Count,
    oracleRerunComparedU16Count,
    exactSelectorOracleRawU16: probeReceipts.every((entry) => {
      const comparison = entry["candidateOracle"] as
        Readonly<Record<string, unknown>>;
      return comparison["differingU16Count"] === 0;
    }),
    candidateAndOracleRerunsDeterministic,
    allOutputsFiniteAndComplete,
    guardsAndTailsIntact,
    probes: Object.freeze(probeReceipts),
    passed: probeReceipts.every((entry) => entry["passed"] === true),
  });
}

async function executeCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  output: GuardedOutput,
  dispatch: Encodable,
  probeSpec: ProbeSpec,
): Promise<OutputSnapshot> {
  const readback = tracker.create(device, {
    label: `${dispatch.label}-readback`,
    size: output.totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({
      label: `${dispatch.label}-correctness`,
    });
    encoder.copyBufferToBuffer(
      output.prefill, 0, output.buffer, 0, output.totalBytes,
    );
    const pass = encoder.beginComputePass();
    dispatch.encode(pass);
    pass.end();
    encoder.copyBufferToBuffer(
      output.buffer, 0, readback, 0, output.totalBytes,
    );
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    try {
      const mapped = readback.getMappedRange();
      const allU32 = new Uint32Array(mapped);
      const guardU32 = STORAGE_GUARD_BYTES / 4;
      const suffixU32 = guardU32 + output.logicalBytes / 4;
      let prefixGuardIntact = true;
      let suffixGuardIntact = true;
      for (let index = 0; index < guardU32; index += 1) {
        prefixGuardIntact &&= allU32[index] === STORAGE_GUARD_U32;
        suffixGuardIntact &&=
          allU32[suffixU32 + index] === STORAGE_GUARD_U32;
      }
      const allU16 = new Uint16Array(mapped);
      const logical = allU16.subarray(
        STORAGE_GUARD_BYTES / 2,
        STORAGE_GUARD_BYTES / 2 + output.logicalBytes / 2,
      );
      const words = logical.slice(
        probeSpec.base,
        probeSpec.base + probeSpec.count,
      );
      let nonFiniteCount = 0;
      let qNaNPrefillCount = 0;
      let outOfRangeWriteCount = 0;
      for (const word of words) {
        if ((word & 0x7c00) === 0x7c00) nonFiniteCount += 1;
        if (word === OUTPUT_PREFILL_QNAN_F16) qNaNPrefillCount += 1;
      }
      for (let index = 0; index < logical.length; index += 1) {
        const selected = index >= probeSpec.base &&
          index < probeSpec.base + probeSpec.count;
        if (!selected && logical[index] !== OUTPUT_PREFILL_QNAN_F16) {
          outOfRangeWriteCount += 1;
        }
      }
      const adjacentWords = STORAGE_GUARD_BYTES / 2;
      const before = logical.subarray(
        Math.max(0, probeSpec.base - adjacentWords),
        probeSpec.base,
      );
      const after = logical.subarray(
        probeSpec.base + probeSpec.count,
        Math.min(
          logical.length,
          probeSpec.base + probeSpec.count + adjacentWords,
        ),
      );
      const adjacentBeforeIntact = !before.some((word) =>
        word !== OUTPUT_PREFILL_QNAN_F16);
      const adjacentAfterIntact = !after.some((word) =>
        word !== OUTPUT_PREFILL_QNAN_F16);
      const tail = words.subarray(words.length - CHANNELS);
      const tailWritten = !tail.some((word) =>
        word === OUTPUT_PREFILL_QNAN_F16);
      return Object.freeze({ words, sha256: await sha256U16(words),
        nonFiniteCount, qNaNPrefillCount, prefixGuardIntact,
        suffixGuardIntact, adjacentBeforeIntact, adjacentAfterIntact,
        outOfRangeWriteCount, tailWritten });
    } finally {
      readback.unmap();
    }
  } finally {
    tracker.destroy(readback);
  }
}

function compareRawWords(
  expected: Uint16Array,
  actual: Uint16Array,
  spec: CaseSpec,
  probeSpec: ProbeSpec,
  comparison: string,
): Readonly<Record<string, unknown>> & {
  readonly count: number;
  readonly differingU16Count: number;
} {
  if (expected.length !== actual.length || expected.length !== probeSpec.count) {
    throw new Error(`OPT-0076 ${spec.id} ${probeSpec.id} length changed`);
  }
  let differingU16Count = 0;
  let signedZeroDifferenceCount = 0;
  let classDifferenceCount = 0;
  let firstDifference: Readonly<Record<string, unknown>> | null = null;
  for (let index = 0; index < expected.length; index += 1) {
    const expectedWord = expected[index]!;
    const actualWord = actual[index]!;
    if (expectedWord !== actualWord) {
      differingU16Count += 1;
      firstDifference ??= Object.freeze({ index,
        outputIndex: probeSpec.base + index,
        outputRow: Math.floor((probeSpec.base + index) / CHANNELS),
        outputChannel: (probeSpec.base + index) % CHANNELS,
        expectedU16: expectedWord, actualU16: actualWord,
        expected: fp16ToNumber(expectedWord), actual: fp16ToNumber(actualWord) });
    }
    if ((expectedWord === 0 && actualWord === 0x8000) ||
      (expectedWord === 0x8000 && actualWord === 0)) {
      signedZeroDifferenceCount += 1;
    }
    if (fp16Class(expectedWord) !== fp16Class(actualWord)) {
      classDifferenceCount += 1;
    }
  }
  return Object.freeze({ comparison, count: expected.length,
    differingU16Count, signedZeroDifferenceCount, classDifferenceCount,
    firstDifference, passed: differingU16Count === 0 });
}

function createNumericalAccumulator(): NumericalAccumulator {
  return { count: 0, rawU16MismatchCount: 0,
    signedZeroDifferenceCount: 0, sumError: 0, sumAbsoluteError: 0,
    sumSquaredError: 0, sumControlSquared: 0, sumControl: 0,
    sumCandidate: 0, sumCandidateSquared: 0, sumProduct: 0,
    maximumAbsoluteError: 0, controlPeak: 0, firstDifference: null,
    controlMinimum: Number.POSITIVE_INFINITY,
    controlMaximum: Number.NEGATIVE_INFINITY,
    candidateMinimum: Number.POSITIVE_INFINITY,
    candidateMaximum: Number.NEGATIVE_INFINITY,
    ulpDistribution: new Map(), worstDifference: null };
}

function compareNumerics(
  currentWords: Uint16Array,
  oracleWords: Uint16Array,
  spec: CaseSpec,
  probeSpec: ProbeSpec,
): NumericalAccumulator {
  if (currentWords.length !== oracleWords.length) {
    throw new Error("OPT-0076 numerical output lengths changed");
  }
  const result = createNumericalAccumulator();
  for (let index = 0; index < currentWords.length; index += 1) {
    const currentBits = currentWords[index]!;
    const oracleBits = oracleWords[index]!;
    const control = fp16ToNumber(currentBits);
    const candidate = fp16ToNumber(oracleBits);
    if (!Number.isFinite(control) || !Number.isFinite(candidate)) {
      throw new Error(`OPT-0076 ${spec.id} ${probeSpec.id} non-finite output`);
    }
    const error = candidate - control;
    const absoluteError = Math.abs(error);
    const location = Object.freeze({ caseId: spec.id, probeId: probeSpec.id,
      index, outputIndex: probeSpec.base + index,
      currentU16: currentBits, oracleU16: oracleBits, current: control,
      oracle: candidate, absoluteError });
    result.count += 1;
    result.sumError += error;
    result.sumAbsoluteError += absoluteError;
    result.sumSquaredError += error * error;
    result.sumControlSquared += control * control;
    result.sumControl += control;
    result.sumCandidate += candidate;
    result.sumCandidateSquared += candidate * candidate;
    result.sumProduct += control * candidate;
    result.controlPeak = Math.max(result.controlPeak, Math.abs(control));
    result.controlMinimum = Math.min(result.controlMinimum, control);
    result.controlMaximum = Math.max(result.controlMaximum, control);
    result.candidateMinimum = Math.min(result.candidateMinimum, candidate);
    result.candidateMaximum = Math.max(result.candidateMaximum, candidate);
    if (absoluteError > result.maximumAbsoluteError) {
      result.maximumAbsoluteError = absoluteError;
      result.worstDifference = location;
    }
    if (currentBits !== oracleBits) {
      result.rawU16MismatchCount += 1;
      result.firstDifference ??= location;
    }
    if ((currentBits === 0 && oracleBits === 0x8000) ||
      (currentBits === 0x8000 && oracleBits === 0)) {
      result.signedZeroDifferenceCount += 1;
    }
    const ulp = Math.abs(orderedFp16(currentBits) - orderedFp16(oracleBits));
    result.ulpDistribution.set(
      ulp,
      (result.ulpDistribution.get(ulp) ?? 0) + 1,
    );
  }
  return result;
}

function mergeNumericalAccumulator(
  target: NumericalAccumulator,
  source: NumericalAccumulator,
): void {
  target.count += source.count;
  target.rawU16MismatchCount += source.rawU16MismatchCount;
  target.signedZeroDifferenceCount += source.signedZeroDifferenceCount;
  target.sumError += source.sumError;
  target.sumAbsoluteError += source.sumAbsoluteError;
  target.sumSquaredError += source.sumSquaredError;
  target.sumControlSquared += source.sumControlSquared;
  target.sumControl += source.sumControl;
  target.sumCandidate += source.sumCandidate;
  target.sumCandidateSquared += source.sumCandidateSquared;
  target.sumProduct += source.sumProduct;
  target.controlPeak = Math.max(target.controlPeak, source.controlPeak);
  target.controlMinimum = Math.min(target.controlMinimum, source.controlMinimum);
  target.controlMaximum = Math.max(target.controlMaximum, source.controlMaximum);
  target.candidateMinimum = Math.min(
    target.candidateMinimum,
    source.candidateMinimum,
  );
  target.candidateMaximum = Math.max(
    target.candidateMaximum,
    source.candidateMaximum,
  );
  target.firstDifference ??= source.firstDifference;
  if (source.maximumAbsoluteError > target.maximumAbsoluteError) {
    target.maximumAbsoluteError = source.maximumAbsoluteError;
    target.worstDifference = source.worstDifference;
  }
  for (const [ulp, count] of source.ulpDistribution) {
    target.ulpDistribution.set(
      ulp,
      (target.ulpDistribution.get(ulp) ?? 0) + count,
    );
  }
}

function summarizeNumerics(
  value: NumericalAccumulator,
): Readonly<Record<string, unknown>> {
  if (value.count < 1) throw new Error("OPT-0076 empty numerical comparison");
  const nrmse = Math.sqrt(
    value.sumSquaredError / Math.max(value.sumControlSquared, 1e-24),
  );
  const snr = value.sumSquaredError === 0
    ? Number.POSITIVE_INFINITY
    : 10 * Math.log10(value.sumControlSquared / value.sumSquaredError);
  const covariance = value.count * value.sumProduct -
    value.sumControl * value.sumCandidate;
  const controlVariance = value.count * value.sumControlSquared -
    value.sumControl * value.sumControl;
  const candidateVariance = value.count * value.sumCandidateSquared -
    value.sumCandidate * value.sumCandidate;
  const denominator = Math.sqrt(Math.max(0,
    controlVariance * candidateVariance));
  const pearson = denominator === 0
    ? (value.sumSquaredError === 0 ? 1 : 0)
    : Math.max(-1, Math.min(1, covariance / denominator));
  const fp16UlpDistribution = Object.freeze(Object.fromEntries(
    [...value.ulpDistribution.entries()]
      .sort(([left], [right]) => left - right)
      .map(([ulp, count]) => [String(ulp), count]),
  ));
  return Object.freeze({ comparedValueCount: value.count,
    rawU16MismatchCount: value.rawU16MismatchCount,
    signedZeroDifferenceCount: value.signedZeroDifferenceCount,
    maximumAbsoluteError: value.maximumAbsoluteError,
    relativeMaximumAbsoluteError: value.maximumAbsoluteError /
      Math.max(value.controlPeak, 1e-6),
    meanAbsoluteError: value.sumAbsoluteError / value.count,
    meanError: value.sumError / value.count,
    rmsError: Math.sqrt(value.sumSquaredError / value.count),
    nrmse, snrDb: Number.isFinite(snr) ? snr : "Infinity", pearson,
    controlPeak: value.controlPeak,
    numericOutputRanges: Object.freeze({
      control: Object.freeze({ minimum: value.controlMinimum,
        maximum: value.controlMaximum }),
      candidate: Object.freeze({ minimum: value.candidateMinimum,
        maximum: value.candidateMaximum }),
    }),
    fp16UlpDistribution,
    fp16UlpDistributionCount: Object.values(fp16UlpDistribution).reduce(
      (sum, count) => sum + count,
      0,
    ),
    firstDifference: value.firstDifference,
    worstDifference: value.worstDifference });
}

function numericalPassed(metrics: Readonly<Record<string, unknown>>): boolean {
  const snr = metrics["snrDb"] === "Infinity"
    ? Number.POSITIVE_INFINITY
    : Number(metrics["snrDb"]);
  return Number(metrics["nrmse"]) <= NRMSE_MAXIMUM &&
    snr >= SNR_MINIMUM_DB &&
    Number(metrics["pearson"]) >= PEARSON_MINIMUM &&
    Number(metrics["relativeMaximumAbsoluteError"]) <=
      RELATIVE_MAXIMUM_ABSOLUTE_ERROR_MAXIMUM;
}

async function runTiming(
  prepared: PreparedHarness,
  failureEvidence: TimingFailureEvidence,
): Promise<void> {
  const samples: Record<Arm, TimestampSample[]> = {
    current: [],
    candidate: [],
  };
  const runStartedAtEpochMilliseconds = Date.now();
  failureEvidence.runStartedAtEpochMilliseconds =
    runStartedAtEpochMilliseconds;
  progress.textContent = "click-boundary untimed current/C256-K4 warmup";
  const warmupStartedAtEpochMilliseconds = Date.now();
  await executeComposite(prepared.device, prepared.productionCases, "current");
  await executeComposite(prepared.device, prepared.productionCases, "candidate");
  const warmupCompletedAtEpochMilliseconds = Date.now();
  const measurementStartedAtEpochMilliseconds = Date.now();
  for (const [roundIndex, order] of TIMING_ROUNDS.entries()) {
    for (const [armPosition, arm] of order.entries()) {
      progress.textContent =
        `timing round ${roundIndex + 1}/${TIMING_ROUNDS.length}: ${arm}`;
      const sample = await timeComposite(prepared, arm);
      samples[arm].push(sample);
      failureEvidence.rawSamples.push(Object.freeze({ roundIndex, armPosition,
        arm, composite: "d1/d3/d9 x first/interior/tail", ...sample }));
    }
    await yieldToBrowser();
  }
  await prepared.device.queue.onSubmittedWorkDone();
  await settlePostDrainEvents();
  if (prepared.uncapturedErrors.length !== 0 ||
    prepared.deviceLosses.length !== 0) {
    throw new Error("OPT-0076 observed a timing GPU error or device loss");
  }
  const measurementCompletedAtEpochMilliseconds = Date.now();
  failureEvidence.measurementCompletedAtEpochMilliseconds =
    measurementCompletedAtEpochMilliseconds;
  const timing = summarizeOpt0076Timing(Object.freeze({
    current: Object.freeze(samples.current.slice()),
    candidate: Object.freeze(samples.candidate.slice()),
  }));
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanupStartedAtEpochMilliseconds = Date.now();
  const cleanupFirst = prepared.destroy();
  const cleanupSecond = prepared.destroy();
  const cleanupCompletedAtEpochMilliseconds = Date.now();
  const cleanupPassed = cleanupFirst["liveBufferCount"] === 0 &&
    cleanupSecond["liveBufferCount"] === 0 &&
    cleanupFirst["liveBytes"] === 0 && cleanupSecond["liveBytes"] === 0 &&
    cleanupFirst["createdBufferCount"] ===
      cleanupFirst["destroyedBufferCount"] &&
    cleanupSecond["createdBufferCount"] ===
      cleanupSecond["destroyedBufferCount"];
  const overallPassed = timing["passed"] === true &&
    prepared.correctness["passed"] === true && cleanupPassed &&
    prepared.uncapturedErrors.length === 0 &&
    prepared.deviceLosses.length === 0;
  const disposition = overallPassed
    ? "positive-in-page-selector-primitive-pending-external-thermal-audit"
    : cleanupPassed
    ? "negative-stop-selector-primitive-gate"
    : "invalid-stop-resource-reconciliation-failed";
  const receipt = Object.freeze({
    schema: RECEIPT_SCHEMA,
    experiment: EXPERIMENT_ID,
    status: "completed",
    passed: overallPassed,
    disposition,
    readyAtEpochMilliseconds: failureEvidence.readyAtEpochMilliseconds,
    identity: prepared.identity,
    correctness: prepared.correctness,
    protocol: Object.freeze({
      correctnessBeforeReady: true,
      readyEpochRecordedBeforeButtonEnabled: true,
      clickBoundaryWarmupPerArm: 1,
      rounds: TIMING_ROUNDS.length,
      balancedArmOrders: TIMING_ROUNDS,
      compositeDispatchesPerSample: 9,
      compositeCoverage: Object.freeze({ dilations: [1, 3, 9],
        ranges: ["first", "interior", "tail"] }),
      oneTimestampPairOnePassOneCommandBufferOneSubmitOneDrainPerSample: true,
      outputReadbackInsideTiming: false,
      externalContinuousThermalTraceCapturedByPage: false,
    }),
    timing: Object.freeze({ ...timing,
      readyAtEpochMilliseconds: failureEvidence.readyAtEpochMilliseconds,
      runStartedAtEpochMilliseconds,
      warmupStartedAtEpochMilliseconds, warmupCompletedAtEpochMilliseconds,
      measurementStartedAtEpochMilliseconds,
      measurementCompletedAtEpochMilliseconds,
      runCompletedAtEpochMilliseconds: cleanupCompletedAtEpochMilliseconds,
      rawSamples: Object.freeze(failureEvidence.rawSamples.slice()) }),
    decision: Object.freeze({
      disposition,
      externalThermalGateAuditedByPage: false,
      diagnosticProfileEscalationAuthorized: false,
      productionIntegrationAuthorized: false,
      qualityOrListeningClaim: false,
    }),
    uncapturedGpuErrorCount: prepared.uncapturedErrors.length,
    deviceLossCount: prepared.deviceLosses.length,
    memoryBeforeCleanup,
    cleanup: Object.freeze({ cleanupStartedAtEpochMilliseconds,
      cleanupCompletedAtEpochMilliseconds, firstCall: cleanupFirst,
      secondCall: cleanupSecond, idempotent: true,
      zeroLiveBuffers: cleanupFirst["liveBufferCount"] === 0 &&
        cleanupSecond["liveBufferCount"] === 0,
      zeroLiveBytes: cleanupFirst["liveBytes"] === 0 &&
        cleanupSecond["liveBytes"] === 0,
      createdEqualsDestroyed:
        cleanupFirst["createdBufferCount"] ===
          cleanupFirst["destroyedBufferCount"] &&
        cleanupSecond["createdBufferCount"] ===
          cleanupSecond["destroyedBufferCount"],
      passed: cleanupPassed }),
  });
  publish(receipt, receipt.passed ? "passed" : "failed");
  progress.textContent = receipt.passed
    ? "completed — in-page C256 gate passed; external thermal audit pending"
    : "failed — C256 selector did not clear every paired timing gate";
}

async function executeComposite(
  device: GPUDevice,
  cases: readonly PreparedCase[],
  arm: Arm,
): Promise<void> {
  const encoder = device.createCommandEncoder({
    label: `opt0076-${arm}-warmup-composite`,
  });
  const pass = encoder.beginComputePass();
  encodeComposite(pass, cases, arm);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

function encodeComposite(
  pass: GPUComputePassEncoder,
  cases: readonly PreparedCase[],
  arm: Arm,
): void {
  for (const preparedCase of cases) {
    for (const probeSpec of preparedCase.spec.probes) {
      preparedCase.dispatches[probeSpec.id][arm].encode(pass);
    }
  }
}

async function timeComposite(
  prepared: PreparedHarness,
  arm: Arm,
): Promise<TimestampSample> {
  if (prepared.queryReadback.mapState !== "unmapped") {
    throw new Error("OPT-0076 timestamp readback was still mapped");
  }
  const encoder = prepared.device.createCommandEncoder({
    label: `opt0076-${arm}-timestamp-sample`,
  });
  const pass = encoder.beginComputePass({
    label: `opt0076-${arm}-timestamped-composite`,
    timestampWrites: { querySet: prepared.querySet,
      beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
  });
  encodeComposite(pass, prepared.productionCases, arm);
  pass.end();
  encoder.resolveQuerySet(prepared.querySet, 0, 2, prepared.queryResolve, 0);
  encoder.copyBufferToBuffer(prepared.queryResolve, 0,
    prepared.queryReadback, 0, TIMESTAMP_QUERY_BYTES);
  const command = encoder.finish();
  const submitAtPerformanceMilliseconds = performance.now();
  prepared.device.queue.submit([command]);
  await prepared.device.queue.onSubmittedWorkDone();
  const fenceAtPerformanceMilliseconds = performance.now();
  const wallMilliseconds = fenceAtPerformanceMilliseconds -
    submitAtPerformanceMilliseconds;
  await prepared.queryReadback.mapAsync(GPUMapMode.READ);
  let begin: bigint;
  let end: bigint;
  try {
    const timestamps = new BigUint64Array(
      prepared.queryReadback.getMappedRange(),
    );
    begin = timestamps[0]!;
    end = timestamps[1]!;
  } finally {
    prepared.queryReadback.unmap();
  }
  if (end <= begin) throw new Error("OPT-0076 timestamp interval was empty");
  const gpuElapsedNanoseconds = Number(end - begin);
  const gpuMilliseconds = gpuElapsedNanoseconds / 1_000_000;
  if (!Number.isSafeInteger(gpuElapsedNanoseconds) ||
    !Number.isFinite(gpuMilliseconds) || gpuMilliseconds <= 0 ||
    !Number.isFinite(wallMilliseconds) || wallMilliseconds <= 0) {
    throw new Error("OPT-0076 timing sample was invalid");
  }
  return Object.freeze({ submitAtPerformanceMilliseconds,
    fenceAtPerformanceMilliseconds,
    submitAtEpochMilliseconds: performance.timeOrigin +
      submitAtPerformanceMilliseconds,
    fenceAtEpochMilliseconds: performance.timeOrigin +
      fenceAtPerformanceMilliseconds,
    wallMilliseconds, gpuMilliseconds,
    timestampBeginNanoseconds: begin.toString(),
    timestampEndNanoseconds: end.toString(), gpuElapsedNanoseconds,
    gpuToWallRatio: gpuMilliseconds / wallMilliseconds,
    dispatchCount: 9 as const, commandBufferCount: 1 as const,
    queueDrainCount: 1 as const });
}

export function summarizeOpt0076Timing(
  samples: Readonly<Record<Arm, readonly TimestampSample[]>>,
): Readonly<Record<string, unknown>> {
  requireTimingSamples(samples.current, "current");
  requireTimingSamples(samples.candidate, "candidate");
  const arms = Object.freeze(Object.fromEntries(
    (["current", "candidate"] as const).map((arm) => {
      const values = samples[arm];
      const gpu = values.map((sample) => sample.gpuMilliseconds);
      const wall = values.map((sample) => sample.wallMilliseconds);
      return [arm, Object.freeze({ samples: values,
        meanGpuMilliseconds: mean(gpu), medianGpuMilliseconds: median(gpu),
        minimumGpuMilliseconds: Math.min(...gpu),
        maximumGpuMilliseconds: Math.max(...gpu),
        meanWallMilliseconds: mean(wall),
        medianWallMilliseconds: median(wall),
        minimumWallMilliseconds: Math.min(...wall),
        maximumWallMilliseconds: Math.max(...wall),
        meanGpuToWallRatio: mean(gpu) / mean(wall) })];
    }),
  )) as unknown as Readonly<Record<Arm, Readonly<{
    samples: readonly TimestampSample[];
    meanGpuMilliseconds: number;
    medianGpuMilliseconds: number;
    meanWallMilliseconds: number;
    medianWallMilliseconds: number;
  }>>>;
  const pairedRounds = samples.current.map((current, roundIndex) => {
    const candidate = samples.candidate[roundIndex]!;
    return Object.freeze({ roundIndex,
      currentWallMilliseconds: current.wallMilliseconds,
      candidateWallMilliseconds: candidate.wallMilliseconds,
      wallSpeedup: current.wallMilliseconds / candidate.wallMilliseconds,
      wallWin: candidate.wallMilliseconds < current.wallMilliseconds,
      currentGpuMilliseconds: current.gpuMilliseconds,
      candidateGpuMilliseconds: candidate.gpuMilliseconds,
      gpuSpeedup: current.gpuMilliseconds / candidate.gpuMilliseconds,
      gpuWin: candidate.gpuMilliseconds < current.gpuMilliseconds });
  });
  const meanWallSpeedup = arms.current.meanWallMilliseconds /
    arms.candidate.meanWallMilliseconds;
  const medianWallSpeedup = arms.current.medianWallMilliseconds /
    arms.candidate.medianWallMilliseconds;
  const meanGpuSpeedup = arms.current.meanGpuMilliseconds /
    arms.candidate.meanGpuMilliseconds;
  const medianGpuSpeedup = arms.current.medianGpuMilliseconds /
    arms.candidate.medianGpuMilliseconds;
  const gates = Object.freeze({ requiredMeanAndMedianSpeedup: REQUIRED_SPEEDUP,
    everyPairedAggregateWallWin: pairedRounds.every((round) => round.wallWin),
    everyPairedAggregateGpuWin: pairedRounds.every((round) => round.gpuWin),
    observedMeanWallSpeedup: meanWallSpeedup,
    meanWallSpeedupPassed: meanWallSpeedup >= REQUIRED_SPEEDUP,
    observedMedianWallSpeedup: medianWallSpeedup,
    medianWallSpeedupPassed: medianWallSpeedup >= REQUIRED_SPEEDUP,
    observedMeanGpuSpeedup: meanGpuSpeedup,
    meanGpuSpeedupPassed: meanGpuSpeedup >= REQUIRED_SPEEDUP,
    observedMedianGpuSpeedup: medianGpuSpeedup,
    medianGpuSpeedupPassed: medianGpuSpeedup >= REQUIRED_SPEEDUP });
  return Object.freeze({
    sampleCountPerArm: TIMING_ROUNDS.length,
    aggregateDispatchesPerSample: 9,
    arms,
    pairedRounds: Object.freeze(pairedRounds),
    speedup: Object.freeze({ meanWall: meanWallSpeedup,
      medianWall: medianWallSpeedup, meanGpu: meanGpuSpeedup,
      medianGpu: medianGpuSpeedup }),
    gates,
    passed: gates.everyPairedAggregateWallWin &&
      gates.everyPairedAggregateGpuWin && gates.meanWallSpeedupPassed &&
      gates.medianWallSpeedupPassed && gates.meanGpuSpeedupPassed &&
      gates.medianGpuSpeedupPassed,
  });
}

async function buildIdentity(
  adapter: GPUAdapter,
  device: GPUDevice,
): Promise<Readonly<Record<string, unknown>>> {
  const generatedShaders = await Promise.all(PRODUCTION_CASES.map(async (
    spec,
  ) => Object.freeze({ id: spec.id, dilation: spec.dilation,
    currentScalarWgslSha256: await sha256Text(
      aceFp16VaeConv1dSubgroupWgsl(spec.shape, true, "float16"),
    ),
    candidateAndOracleK4WgslSha256: await sha256Text(
      aceOpt0024VaeConv1dDirectDot4SubgroupWgsl(
        spec.shape,
        true,
        "float16",
      ),
    ) })));
  const storageBytes = maximumStorageBindingBytes();
  return Object.freeze({
    currentSelectorKernelId: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
    currentScalarKernelId: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
    candidateSelectorKernelId: ACE_OPT_0076_VAE_C256_K4_SELECTOR_KERNEL_ID,
    candidateLiteralSelectorKernelId:
      ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
    candidateAndOracleKernelId:
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
    sourceSha256: Object.freeze({
      currentSelector: await sha256Text(currentSelectorSource),
      candidateSelector: await sha256Text(promotedSelectorSource),
      directK4: await sha256Text(directK4Source),
      harness: await sha256Text(harnessSource),
      html: await sha256Text(htmlSource),
      experimentRecord: await sha256Text(experimentSource),
    }),
    generatedShaders: Object.freeze(generatedShaders),
    fixtureVersion: "opt0076-c256-c2400-local-plus-bounded-v1",
    productionLocalGeometry: Object.freeze({ inputFrames: INPUT_FRAMES,
      channels: CHANNELS, dilations: [1, 3, 9], probes: PROBES,
      source: "C300-derived C256 batch-64 local topology; 2400 mod 64 = 32" }),
    browserUserAgent: navigator.userAgent,
    browserLanguage: navigator.language,
    browserHardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated,
    requestedDeviceDescriptor: Object.freeze({
      features: Object.freeze(["shader-f16", "subgroups", "timestamp-query"]),
      limits: Object.freeze({ maxBufferSize: storageBytes +
        2 * STORAGE_GUARD_BYTES, maxStorageBufferBindingSize: storageBytes,
        maxComputeInvocationsPerWorkgroup: 128,
        maxComputeWorkgroupSizeX: 128 }),
    }),
    adapter: Object.freeze({ vendor: adapter.info.vendor,
      architecture: adapter.info.architecture, device: adapter.info.device,
      description: adapter.info.description,
      subgroupMinSize: adapter.info.subgroupMinSize,
      subgroupMaxSize: adapter.info.subgroupMaxSize,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
      features: Object.freeze([...adapter.features].sort()) }),
    deviceFeatures: Object.freeze([...device.features].sort()),
    deviceLimits: Object.freeze({ maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX }),
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
    if (buffer.mapState === "mapped") buffer.unmap();
    buffer.destroy();
    this.destroyed += 1;
    this.liveBytes -= this.sizes.get(buffer) ?? 0;
    this.sizes.delete(buffer);
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number>> {
    return Object.freeze({ createdBufferCount: this.created,
      destroyedBufferCount: this.destroyed, liveBufferCount: this.live.size,
      liveBytes: this.liveBytes, maximumLiveBytes: this.maximumLiveBytes });
  }
}

function caseSpec(
  dilation: 1 | 3 | 9,
  fixtureKind: FixtureKind,
): CaseSpec {
  const operationLabel = dilation === 1
    ? "block-2-res-1-conv1"
    : dilation === 3
    ? "block-2-res-2-conv1"
    : "block-2-res-3-conv1";
  return Object.freeze({
    id: `d${dilation}-${fixtureKind}`,
    operationLabel,
    dilation,
    fixtureKind,
    shape: Object.freeze({ batch: 1, inputFrames: INPUT_FRAMES,
      inputChannels: CHANNELS, outputChannels: CHANNELS, kernelSize: 7,
      stride: 1, dilation, padding: 3 * dilation }),
    probes: PROBES,
  });
}

function probe(
  id: ProbeId,
  firstOutputRow: number,
  outputRowCount: number,
): ProbeSpec {
  return Object.freeze({ id, base: firstOutputRow * CHANNELS,
    count: outputRowCount * CHANNELS, firstOutputRow, outputRowCount });
}

function rangeBinding(
  control: GPUBuffer,
  probeSpec: ProbeSpec,
): AceVaeOutputRangeBinding {
  return Object.freeze({ base: probeSpec.base, count: probeSpec.count,
    control: Object.freeze({ buffer: control, offset: 0, size: 16 }) });
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function snapshotPassed(snapshot: OutputSnapshot): boolean {
  return snapshot.nonFiniteCount === 0 && snapshot.qNaNPrefillCount === 0 &&
    snapshot.prefixGuardIntact && snapshot.suffixGuardIntact &&
    snapshot.adjacentBeforeIntact && snapshot.adjacentAfterIntact &&
    snapshot.outOfRangeWriteCount === 0 && snapshot.tailWritten;
}

function snapshotReceipt(
  snapshot: OutputSnapshot,
): Readonly<Record<string, unknown>> {
  return Object.freeze({ sha256: snapshot.sha256,
    outputU16Count: snapshot.words.length,
    nonFiniteCount: snapshot.nonFiniteCount,
    qNaNPrefillCount: snapshot.qNaNPrefillCount,
    prefixGuardIntact: snapshot.prefixGuardIntact,
    suffixGuardIntact: snapshot.suffixGuardIntact,
    adjacentBeforeIntact: snapshot.adjacentBeforeIntact,
    adjacentAfterIntact: snapshot.adjacentAfterIntact,
    outOfRangeWriteCount: snapshot.outOfRangeWriteCount,
    tailWritten: snapshot.tailWritten,
    passed: snapshotPassed(snapshot) });
}

function deterministicWeightBits(spec: CaseSpec, nativeIndex: number): number {
  const special = nativeIndex & 0x0fff;
  if (special === 0) return 0x0000;
  if (special === 1) return 0x8000;
  if (special === 2) return 0x0001;
  if (special === 3) return 0x8001;
  if (spec.fixtureKind === "bounded-adversarial" && special < 16) {
    return ADVERSARIAL_INPUT_PATTERN[special]!;
  }
  const mixed = mix32(nativeIndex ^ Math.imul(spec.dilation, 0x9e37_79b9));
  return ((mixed >>> 16) & 0x8000) | 0x1000 | (mixed & 0x03ff);
}

function mix32(value: number): number {
  let mixed = value >>> 0;
  mixed ^= mixed >>> 16;
  mixed = Math.imul(mixed, 0x7feb_352d);
  mixed ^= mixed >>> 15;
  mixed = Math.imul(mixed, 0x846c_a68b);
  mixed ^= mixed >>> 16;
  return mixed >>> 0;
}

function fillPeriodic(destination: Uint16Array, pattern: Uint16Array): void {
  if (destination.length < 1 || pattern.length < 1) {
    throw new RangeError("OPT-0076 periodic fixture is empty");
  }
  const initial = Math.min(destination.length, pattern.length);
  destination.set(pattern.subarray(0, initial));
  let filled = initial;
  while (filled < destination.length) {
    const count = Math.min(filled, destination.length - filled);
    destination.copyWithin(filled, 0, count);
    filled += count;
  }
}

async function hashBufferUploadPattern(
  bytes: number,
  pattern: Uint16Array,
): Promise<string> {
  const words = new Uint16Array(bytes / 2);
  fillPeriodic(words, pattern);
  return await sha256Bytes(new Uint8Array(words.buffer));
}

async function hashWeightFixture(
  spec: CaseSpec,
  plan: AceFp16VaeConv1dPlan,
): Promise<string> {
  const words = new Uint16Array(plan.weightBindingBytes / 2);
  for (let index = 0; index < words.length; index += 1) {
    words[index] = deterministicWeightBits(spec, index);
  }
  return await sha256Bytes(new Uint8Array(words.buffer));
}

function fp16ToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) {
    if (mantissa === 0) return sign < 0 ? -0 : 0;
    return sign * 2 ** -14 * (mantissa / 1_024);
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1_024);
}

function fp16Class(bits: number): string {
  const absolute = bits & 0x7fff;
  const negative = (bits & 0x8000) !== 0;
  if (absolute === 0) return negative ? "negative-zero" : "positive-zero";
  const exponent = bits & 0x7c00;
  const mantissa = bits & 0x03ff;
  if (exponent === 0x7c00) {
    if (mantissa !== 0) return "nan";
    return negative ? "negative-infinity" : "positive-infinity";
  }
  if (exponent === 0) {
    return negative ? "negative-subnormal" : "positive-subnormal";
  }
  return negative ? "negative-normal" : "positive-normal";
}

function orderedFp16(bits: number): number {
  return (bits & 0x8000) !== 0
    ? 0x8000 - (bits & 0x7fff)
    : 0x8000 + bits;
}

function requireAdapter(adapter: GPUAdapter): void {
  const storageBytes = maximumStorageBindingBytes();
  if (!adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    !adapter.features.has("timestamp-query") ||
    adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32 ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 128 ||
    adapter.limits.maxComputeWorkgroupSizeX < 128 ||
    adapter.limits.maxStorageBufferBindingSize < storageBytes ||
    adapter.limits.maxBufferSize < storageBytes + 2 * STORAGE_GUARD_BYTES) {
    throw new Error(
      "OPT-0076 requires timestamp-query, shader-f16, fixed32 subgroups, WG128, and C2400 storage limits",
    );
  }
}

function maximumStorageBindingBytes(): number {
  return Math.max(...[...PRODUCTION_CASES, ...ADVERSARIAL_CASES].flatMap(
    (spec) => {
      const plan = planAceFp16VaeConv1d(spec.shape, "float16");
      return [plan.inputBindingBytes, plan.weightBindingBytes,
        plan.biasBindingBytes, plan.outputBindingBytes];
    },
  ));
}

function requireTimingSamples(
  samples: readonly TimestampSample[],
  label: string,
): void {
  if (samples.length !== TIMING_ROUNDS.length || samples.some((sample) =>
    !Number.isFinite(sample.gpuMilliseconds) || sample.gpuMilliseconds <= 0 ||
    !Number.isFinite(sample.wallMilliseconds) || sample.wallMilliseconds <= 0 ||
    !Number.isSafeInteger(sample.gpuElapsedNanoseconds) ||
    sample.gpuElapsedNanoseconds <= 0 || sample.dispatchCount !== 9
  )) {
    throw new Error(
      `OPT-0076 ${label} requires ${TIMING_ROUNDS.length} valid samples`,
    );
  }
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = sorted.length / 2;
  return (sorted[middle - 1]! + sorted[middle]!) / 2;
}

function requireLittleEndianHost(): void {
  const words = new Uint16Array([0x0102]);
  if (new Uint8Array(words.buffer)[0] !== 0x02) {
    throw new Error("OPT-0076 fixtures require a little-endian host");
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

async function sha256U16(value: Uint16Array): Promise<string> {
  return await sha256Bytes(new Uint8Array(
    value.buffer,
    value.byteOffset,
    value.byteLength,
  ));
}

async function sha256Bytes(value: Uint8Array): Promise<string> {
  const copy = new Uint8Array(new ArrayBuffer(value.byteLength));
  copy.set(value);
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", copy.buffer),
  );
  return [...digest].map((byte) =>
    byte.toString(16).padStart(2, "0")).join("");
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
  window.__ACE_OPT0076_RESULT__ = receipt;
  result.textContent = JSON.stringify(receipt);
  document.body.dataset.status = status;
}

function fail(
  error: unknown,
  evidence: Readonly<Record<string, unknown>> = Object.freeze({}),
): void {
  const receipt = Object.freeze({ schema: RECEIPT_SCHEMA,
    experiment: EXPERIMENT_ID, status: "failed", passed: false,
    readyAtEpochMilliseconds, ...evidence,
    error: error instanceof Error
      ? Object.freeze({ name: error.name, message: error.message,
        stack: error.stack })
      : String(error) });
  publish(receipt, "failed");
  progress.textContent = error instanceof Error ? error.message : String(error);
  runButton.disabled = true;
}
