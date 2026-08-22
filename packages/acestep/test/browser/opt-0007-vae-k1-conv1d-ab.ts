import {
  AceOpt0007VaeK1Conv1dKernel,
  aceOpt0007VaeK1Conv1dWgsl,
  planAceOpt0007VaeK1Conv1d,
  type AceOpt0007VaeK1Conv1dDispatch,
  type AceOpt0007VaeK1Conv1dOutputRange,
} from "../../benchmark/opt-0007-vae-k1-conv1d.js";
import {
  AceCorrectnessVaePrimitiveKernel,
  type AceVaeConv1dShape,
  type AceVaeConvBindings,
  type AceVaeDispatch,
  type AceVaeConvPlan,
} from "../../src/webgpu/kernels/vae-primitives.js";
import {
  runAceOpt0006QuantumBatches,
  type AceOpt0006BatchProgress,
  type AceOpt0006EncodableQuantum,
} from "../../benchmark/opt-0006-vae-command-buffer-coalescing.js";

export type Opt0007RunMode =
  | "correctness"
  | "screen-c1024"
  | "screen-c128"
  | "sequence-c128";
type Opt0007EndpointMode = "screen-c1024" | "screen-c128";
export type Opt0007CorrectnessCaseId =
  | "bias-batch-c65-tails"
  | "no-bias-batch-c64"
  | "bias-c63-range-tail"
  | "arithmetic-discriminants";
type KernelId = "scalar" | "candidate";

interface RangeDispatch {
  readonly label: string;
  readonly rangeCount: number;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
}

interface OwnedDispatches {
  readonly scalar: RangeDispatch;
  readonly candidate: AceOpt0007VaeK1Conv1dDispatch;
  readonly scalarKernel: AceCorrectnessVaePrimitiveKernel;
  readonly candidateKernel: AceOpt0007VaeK1Conv1dKernel;
  readonly controls: readonly GPUBuffer[];
  destroy(): void;
}

interface HeartbeatResult {
  readonly animationFrameCount: number;
  readonly timerTickCount: number;
  readonly maximumAnimationFrameGapMilliseconds: number;
  readonly maximumTimerGapMilliseconds: number;
}

interface HeartbeatController {
  snapshot(): HeartbeatResult;
  stop(): HeartbeatResult;
}

interface RangeTiming {
  readonly rangeIndex: number;
  readonly encodeMilliseconds: number;
  readonly submitMilliseconds: number;
  readonly drainMilliseconds: number;
}

interface ExecutionTiming {
  readonly roundIndex: number;
  readonly pairedOrder: string;
  readonly orderPosition: number;
  readonly wallMilliseconds: number;
  readonly encodeMilliseconds: number;
  readonly submitMilliseconds: number;
  readonly drainMilliseconds: number;
  readonly commandBufferCount: number;
  readonly maximumSingleDrainMilliseconds: number;
  readonly logicalGmacPerSecond: number;
  readonly ranges: readonly RangeTiming[];
}

interface SampleSummary {
  readonly count: number;
  readonly samples: readonly number[];
  readonly minimum: number;
  readonly median: number;
  readonly maximum: number;
  readonly range: number;
}

interface OutputFingerprint {
  readonly elementCount: number;
  readonly finiteCount: number;
  readonly nonzeroCount: number;
  readonly positiveZeroCount: number;
  readonly negativeZeroCount: number;
  readonly fnv1a32: string;
}

interface IdentityResult {
  readonly comparedElementCount: number;
  readonly fullRangeBitIdentical: true;
  readonly bitMismatchCount: 0;
  readonly scalarFingerprint: OutputFingerprint;
  readonly candidateFingerprint: OutputFingerprint;
}

interface ExecutionSummary {
  readonly wallMilliseconds: SampleSummary;
  readonly encodeMilliseconds: SampleSummary;
  readonly submitMilliseconds: SampleSummary;
  readonly drainMilliseconds: SampleSummary;
  readonly maximumSingleDrainMilliseconds: SampleSummary;
  readonly logicalGmacPerSecond: SampleSummary;
}

interface SequenceRangeTiming {
  readonly localRangeIndex: number;
  readonly plannedRangeIndex: number;
  readonly passIndex: number;
}

interface SequenceExecutionCounters {
  commandEncoderCount: number;
  passCount: number;
  dispatchCount: number;
}

interface SequenceBatchTiming {
  readonly batchIndex: number;
  readonly firstLocalRangeIndex: number;
  readonly rangeCount: 8;
  readonly passCount: 8;
  readonly dispatchCount: 8;
  readonly commandBufferCount: 1;
  readonly queueDrains: 1;
  readonly encodeMilliseconds: number;
  readonly submitMilliseconds: number;
  readonly drainMilliseconds: number;
  readonly explicitIdleMilliseconds: number;
  readonly ranges: readonly SequenceRangeTiming[];
  readonly heartbeat: HeartbeatResult;
}

interface SequenceExecutionTiming {
  readonly roundIndex: number;
  readonly pairedOrder: string;
  readonly orderPosition: number;
  readonly wallMilliseconds: number;
  readonly encodeMilliseconds: number;
  readonly submitMilliseconds: number;
  readonly drainMilliseconds: number;
  readonly explicitIdleMilliseconds: number;
  readonly rangeCount: 16;
  readonly passCount: 16;
  readonly dispatchCount: 16;
  readonly progressEventCount: 16;
  readonly commandBufferCount: 2;
  readonly queueDrains: 2;
  readonly explicitIdleCount: 2;
  readonly maximumOutstandingCommandBuffers: 1;
  readonly maximumSingleDrainMilliseconds: number;
  readonly logicalGmacPerSecond: number;
  readonly batches: readonly SequenceBatchTiming[];
  readonly progress: readonly AceOpt0006BatchProgress[];
}

interface SequenceExecutionSummary {
  readonly wallMilliseconds: SampleSummary;
  readonly encodeMilliseconds: SampleSummary;
  readonly submitMilliseconds: SampleSummary;
  readonly drainMilliseconds: SampleSummary;
  readonly explicitIdleMilliseconds: SampleSummary;
  readonly maximumSingleDrainMilliseconds: SampleSummary;
  readonly logicalGmacPerSecond: SampleSummary;
}

export interface Opt0007ThermalGateMetadata {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly durationSeconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
}

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const OUTPUT_SENTINEL_BITS = 0x7fc0_0000;
const THERMAL_POLL_TOLERANCE_MILLISECONDS = 250;
const EXPLICIT_IDLE_MILLISECONDS = 1;
const SCRATCH_FLOAT = new Float32Array(1);
const SCRATCH_BITS = new Uint32Array(SCRATCH_FLOAT.buffer);

export const OPT_0007_SEQUENCE_RANGE_COUNT = 16;
export const OPT_0007_SEQUENCE_RANGES_PER_COMMAND_BUFFER = 8;

export const OPT_0007_SCREEN_C1024_SHAPE = Object.freeze({
  batch: 1,
  inputFrames: 2_560,
  inputChannels: 1_024,
  outputChannels: 1_024,
  kernelSize: 1,
  stride: 1,
  dilation: 1,
  padding: 0,
} satisfies AceVaeConv1dShape);

export const OPT_0007_SCREEN_C128_SHAPE = Object.freeze({
  batch: 1,
  inputFrames: 245_760,
  inputChannels: 128,
  outputChannels: 128,
  kernelSize: 1,
  stride: 1,
  dilation: 1,
  padding: 0,
} satisfies AceVaeConv1dShape);

export const OPT_0007_PAIRED_ORDERS = Object.freeze([
  Object.freeze(["scalar", "candidate"]),
  Object.freeze(["candidate", "scalar"]),
  Object.freeze(["candidate", "scalar"]),
  Object.freeze(["scalar", "candidate"]),
] satisfies readonly (readonly KernelId[])[]);

export const OPT_0007_CORRECTNESS_CASES = Object.freeze([
  Object.freeze({
    id: "bias-batch-c65-tails",
    hasBias: true,
    shape: Object.freeze({
      batch: 2, inputFrames: 35, inputChannels: 65, outputChannels: 13,
      kernelSize: 1, stride: 1, dilation: 1, padding: 0,
    }),
  }),
  Object.freeze({
    id: "no-bias-batch-c64",
    hasBias: false,
    shape: Object.freeze({
      batch: 3, inputFrames: 19, inputChannels: 64, outputChannels: 11,
      kernelSize: 1, stride: 1, dilation: 1, padding: 0,
    }),
  }),
  Object.freeze({
    id: "bias-c63-range-tail",
    hasBias: true,
    shape: Object.freeze({
      batch: 2, inputFrames: 37, inputChannels: 63, outputChannels: 9,
      kernelSize: 1, stride: 1, dilation: 1, padding: 0,
    }),
  }),
  Object.freeze({
    id: "arithmetic-discriminants",
    hasBias: false,
    shape: Object.freeze({
      batch: 1, inputFrames: 19, inputChannels: 65, outputChannels: 9,
      kernelSize: 1, stride: 1, dilation: 1, padding: 0,
    }),
  }),
] satisfies readonly {
  readonly id: Opt0007CorrectnessCaseId;
  readonly hasBias: boolean;
  readonly shape: AceVaeConv1dShape;
}[]);

export const OPT_0007_ARITHMETIC_SENTINELS = Object.freeze({
  contractedOutputIndex: 5 * 9,
  cancellationOutputIndex: 10 * 9 + 1,
});

export function parseOpt0007RunMode(parameters: URLSearchParams): Opt0007RunMode {
  const mode = parameters.get("runMode");
  if (
    mode === "correctness" || mode === "screen-c1024" ||
    mode === "screen-c128" || mode === "sequence-c128"
  ) {
    return mode;
  }
  throw new Error(
    "OPT-0007 requires explicit " +
      "runMode=correctness|screen-c1024|screen-c128|sequence-c128",
  );
}

export function sequenceC128RangeIndices(rangeCount: number): readonly number[] {
  if (
    !Number.isSafeInteger(rangeCount) ||
    rangeCount < OPT_0007_SEQUENCE_RANGE_COUNT
  ) {
    throw new RangeError(
      `OPT-0007 sequence requires at least ${OPT_0007_SEQUENCE_RANGE_COUNT} ranges`,
    );
  }
  const first = Math.floor(
    (rangeCount - OPT_0007_SEQUENCE_RANGE_COUNT) / 2,
  );
  return Object.freeze(Array.from(
    { length: OPT_0007_SEQUENCE_RANGE_COUNT },
    (_, index) => first + index,
  ));
}

export function expectedOpt0007SequenceCounts(): Readonly<{
  rangeCount: 16;
  passCount: 16;
  dispatchCount: 16;
  progressEventCount: 16;
  commandBufferCount: 2;
  queueDrains: 2;
  explicitIdleCount: 2;
  rangesPerCommandBuffer: 8;
}> {
  return Object.freeze({
    rangeCount: 16,
    passCount: 16,
    dispatchCount: 16,
    progressEventCount: 16,
    commandBufferCount: 2,
    queueDrains: 2,
    explicitIdleCount: 2,
    rangesPerCommandBuffer: 8,
  });
}

export function parseOpt0007ThermalGateMetadata(
  parameters: URLSearchParams,
): Opt0007ThermalGateMetadata | undefined {
  if (parseOpt0007RunMode(parameters) === "correctness") return undefined;
  const source = parameters.get("thermalSource");
  const durationSeconds = requiredNumber(parameters, "thermalDurationSeconds");
  const observationCount = requiredNumber(parameters, "thermalObservations");
  const pollMilliseconds = requiredNumber(parameters, "thermalPollMilliseconds");
  const maximumPollGapMilliseconds = requiredNumber(
    parameters,
    "thermalMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = requiredNumber(
    parameters,
    "thermalNonNominalObservations",
  );
  if (source !== "notifyutil-com.apple.system.thermalpressurelevel") {
    throw new Error("OPT-0007 requires the accepted notifyutil thermal source");
  }
  if (durationSeconds < 30 || observationCount < 31) {
    throw new Error("OPT-0007 requires 30 continuous nominal seconds");
  }
  if (pollMilliseconds !== 1_000) {
    throw new Error("OPT-0007 thermal polling must use 1,000 ms intervals");
  }
  if (maximumPollGapMilliseconds > pollMilliseconds + THERMAL_POLL_TOLERANCE_MILLISECONDS) {
    throw new Error("OPT-0007 thermal poll gap exceeds tolerance");
  }
  if (nonNominalObservationCount !== 0) {
    throw new Error("OPT-0007 thermal gate observed non-nominal pressure");
  }
  return Object.freeze({
    source,
    durationSeconds,
    observationCount,
    pollMilliseconds,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0,
  });
}

export function summarizeOpt0007Samples(samples: readonly number[]): SampleSummary {
  if (samples.length === 0 || samples.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new RangeError("OPT-0007 samples must be finite and non-negative");
  }
  const retained = [...samples];
  const sorted = [...retained].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const minimum = sorted[0]!;
  const maximum = sorted.at(-1)!;
  return Object.freeze({
    count: retained.length,
    samples: Object.freeze(retained),
    minimum,
    median: sorted.length % 2 === 0
      ? (sorted[middle - 1]! + sorted[middle]!) / 2
      : sorted[middle]!,
    maximum,
    range: maximum - minimum,
  });
}

export function opt0007InputValue(
  caseId: Opt0007CorrectnessCaseId,
  index: number,
): number {
  if (caseId === "arithmetic-discriminants") {
    const channel = index % 65;
    const time = Math.floor(index / 65) % 19;
    if (time === 5) {
      if (channel === 63) return Math.fround(-68.06752014160156);
      if (channel === 64) return Math.fround(12.192401885986328);
    }
    if (time === 10) {
      if (channel === 0) return 16_777_216;
      if (channel === 63) return 1;
      if (channel === 64) return -16_777_216;
    }
    return channel % 2 === 0 ? 0 : -0;
  }
  if (index % 37 === 0) return -0;
  if (index % 41 === 0) return 0;
  return Math.fround((((index * 17 + 3) % 31) - 15) / 32);
}

export function opt0007WeightValue(
  caseId: Opt0007CorrectnessCaseId,
  index: number,
): number {
  const fixture = OPT_0007_CORRECTNESS_CASES.find(({ id }) => id === caseId);
  if (fixture === undefined) throw new Error(`unknown OPT-0007 case ${caseId}`);
  const { inputChannels } = fixture.shape;
  if (caseId === "arithmetic-discriminants") {
    const channel = index % inputChannels;
    const outputChannel = Math.floor(index / inputChannels);
    if (outputChannel === 0) {
      if (channel === 63) return 1;
      if (channel === 64) return 5.5625;
      return channel % 2 === 0 ? 0 : -0;
    }
    if (outputChannel === 1) {
      if (channel === 0 || channel === 63 || channel === 64) return 1;
      return 0;
    }
    return outputChannel % 2 === 0 ? 0 : -0;
  }
  if (index % 43 === 0) return -0;
  return Math.fround((((index * 13 + 7) % 29) - 14) / 64);
}

export function opt0007BiasValue(
  caseId: Opt0007CorrectnessCaseId,
  outputChannel: number,
): number {
  if (caseId === "bias-batch-c65-tails" && outputChannel === 0) return -0;
  return Math.fround(((outputChannel % 7) - 3) / 64);
}

export function opt0007ExpectedOutputValue(
  caseId: Opt0007CorrectnessCaseId,
  outputIndex: number,
): number {
  const fixture = OPT_0007_CORRECTNESS_CASES.find(({ id }) => id === caseId);
  if (fixture === undefined) throw new Error(`unknown OPT-0007 case ${caseId}`);
  const { shape } = fixture;
  if (outputIndex < 0 || outputIndex >= shape.batch * shape.inputFrames * shape.outputChannels) {
    throw new RangeError("OPT-0007 expected output is outside its shape");
  }
  const outputChannel = outputIndex % shape.outputChannels;
  const row = Math.floor(outputIndex / shape.outputChannels);
  let sum = fixture.hasBias ? opt0007BiasValue(caseId, outputChannel) : 0;
  const inputBase = row * shape.inputChannels;
  const weightBase = outputChannel * shape.inputChannels;
  for (let channel = 0; channel < shape.inputChannels; channel += 1) {
    sum = Math.fround(
      sum + opt0007InputValue(caseId, inputBase + channel) *
        opt0007WeightValue(caseId, weightBase + channel),
    );
  }
  return sum;
}

if (typeof document !== "undefined") installStartHandler();

function installStartHandler(): void {
  const start = document.querySelector<HTMLButtonElement>("#start");
  if (start === null) throw new Error("Missing start button");
  start.addEventListener("click", () => {
    start.disabled = true;
    document.body.dataset.status = "running";
    updateProgress("starting");
    void run().then(
      (result) => finish("passed", result),
      (error: unknown) => finish("failed", {
        schema: "ace-opt-0007-vae-k1-conv1d-ab-v1",
        status: "failed",
        experimentId: "OPT-0007",
        error: {
          name: error instanceof Error ? error.name : "Error",
          message: error instanceof Error ? error.message : String(error),
          ...(error instanceof Error && error.stack !== undefined ? { stack: error.stack } : {}),
        },
      }),
    );
  }, { once: true });
}

async function run(): Promise<unknown> {
  const parameters = new URL(window.location.href).searchParams;
  const runMode = parseOpt0007RunMode(parameters);
  const thermalGate = parseOpt0007ThermalGateMetadata(parameters);
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("No WebGPU adapter");
  const shapes = runMode === "correctness"
    ? OPT_0007_CORRECTNESS_CASES.map(({ shape }) => shape)
    : [productionShapeFor(runMode)];
  const largestBinding = Math.max(...shapes.map(largestStorageBindingBytes));
  const workgroupStorageBytes = Math.max(
    ...shapes.map((shape) => planAceOpt0007VaeK1Conv1d(shape).workgroupStorageBytes),
  );
  assertAdapter(adapter, largestBinding, workgroupStorageBytes);
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: largestBinding,
      maxStorageBufferBindingSize: largestBinding,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 8,
      maxComputeWorkgroupStorageSize: workgroupStorageBytes,
    },
  });
  try {
    const common = {
      schema: "ace-opt-0007-vae-k1-conv1d-ab-v1",
      status: "passed",
      experimentId: "OPT-0007",
      runMode,
      recordedAt: new Date().toISOString(),
      browser: { userAgent: navigator.userAgent, page: window.location.href },
      adapter: adapterIdentity(adapter),
      fullWindowOrSong: false,
    };
    if (runMode === "correctness") {
      updateProgress("running complete manageable exactness graph");
      return Object.freeze({
        ...common,
        classification: "benchmark-only-complete-manageable-exactness",
        correctness: await runCorrectnessGraph(device),
      });
    }
    if (runMode === "sequence-c128") {
      updateProgress("preparing independent C128 16-range sequence screen");
      const sequence = await runC128SequenceScreen(device);
      return Object.freeze({
        ...common,
        completedAt: new Date().toISOString(),
        classification: "benchmark-only-bounded-c128-production-sequence-screen",
        protocol: {
          thermalGate,
          baselineKernel: "AceCorrectnessVaePrimitiveKernel",
          candidateKernel: "AceOpt0007VaeK1Conv1dKernel",
          compileAllocationUploadExcludedFromTiming: true,
          independentScalarAndCandidateOutputs: true,
          outputPrefill: "quiet-NaN-u32-sentinel",
          warmupExecutionsPerKernel: 1,
          symmetricFullSequenceWarmups: true,
          pairedOrders: OPT_0007_PAIRED_ORDERS.map((order) => order.join("-")),
          samplesPerKernel: OPT_0007_PAIRED_ORDERS.length,
          balancedOrderPattern: "forward-reverse-reverse-forward",
          oneCommandBufferOutstanding: true,
          queueDrainAfterEveryCommandBuffer: true,
          rangesPerCommandBuffer:
            OPT_0007_SEQUENCE_RANGES_PER_COMMAND_BUFFER,
          realQueueEmptyIdleMillisecondsRequested: EXPLICIT_IDLE_MILLISECONDS,
          idleAfterFinalPhysicalBatch: true,
          finalIdleReason: "subsequent-decoder-work-exists",
          authoritativeTiming: "performance.now-fenced-wall-clock",
          continuousExternalThermalMetadata: true,
          noPerformanceAcceptanceThreshold: true,
        },
        sequence,
      });
    }
    updateProgress(`preparing independent ${runMode} production-range screen`);
    return Object.freeze({
      ...common,
      classification: "benchmark-only-independent-production-range-screen",
      protocol: {
        thermalGate,
        baselineKernel: "AceCorrectnessVaePrimitiveKernel",
        candidateKernel: "AceOpt0007VaeK1Conv1dKernel",
        compileAllocationUploadExcludedFromTiming: true,
        independentScalarAndCandidateOutputs: true,
        outputPrefill: "quiet-NaN-u32-sentinel",
        warmupExecutionsPerKernel: 1,
        pairedOrders: OPT_0007_PAIRED_ORDERS.map((order) => order.join("-")),
        oneCommandBufferOutstanding: true,
        queueDrainAfterEveryCommandBuffer: true,
        authoritativeTiming: "performance.now-fenced-wall-clock",
        continuousExternalThermalMetadata: true,
        noPerformanceAcceptanceThreshold: true,
      },
      screen: await runProductionScreen(device, runMode),
    });
  } finally {
    device.destroy();
  }
}

async function runCorrectnessGraph(device: GPUDevice): Promise<unknown> {
  const cases: unknown[] = [];
  for (const fixture of OPT_0007_CORRECTNESS_CASES) {
    const plan = planAceOpt0007VaeK1Conv1d(fixture.shape);
    let input: GPUBuffer | undefined;
    let weight: GPUBuffer | undefined;
    let bias: GPUBuffer | undefined;
    let scalarOutput: GPUBuffer | undefined;
    let candidateOutput: GPUBuffer | undefined;
    let scalarPrepared: OwnedDispatches | undefined;
    let candidatePrepared: OwnedDispatches | undefined;
    try {
      input = mappedGeneratedStorage(
        device,
        `${fixture.id}-input`,
        plan.inputElements,
        (index) => opt0007InputValue(fixture.id, index),
      );
      weight = mappedGeneratedStorage(
        device,
        `${fixture.id}-weight`,
        plan.weightElements,
        (index) => opt0007WeightValue(fixture.id, index),
      );
      bias = fixture.hasBias
        ? mappedGeneratedStorage(
            device,
            `${fixture.id}-bias`,
            plan.outputChannels,
            (channel) => opt0007BiasValue(fixture.id, channel),
          )
        : undefined;
      scalarOutput = mappedSentinelOutput(
        device,
        `${fixture.id}-scalar`,
        plan.outputElements,
      );
      candidateOutput = mappedSentinelOutput(
        device,
        `${fixture.id}-candidate`,
        plan.outputElements,
      );
      scalarPrepared = await prepareDispatches(device, fixture.shape, {
        input: binding(input), weight: binding(weight),
        ...(bias === undefined ? {} : { bias: binding(bias) }),
        output: binding(scalarOutput),
      });
      candidatePrepared = await prepareDispatches(device, fixture.shape, {
        input: binding(input), weight: binding(weight),
        ...(bias === undefined ? {} : { bias: binding(bias) }),
        output: binding(candidateOutput),
      });
      await executeUntimed(device, scalarPrepared.scalar);
      await executeUntimed(device, candidatePrepared.candidate);
      const scalar = await readOutput(device, scalarOutput, 0, plan.outputElements,
        `${fixture.id}-scalar-full`);
      const candidate = await readOutput(device, candidateOutput, 0, plan.outputElements,
        `${fixture.id}-candidate-full`);
      const identity = validateFullIdentity(scalar, candidate, plan.outputElements, fixture.id);
      const sentinelIndices = [...new Set([
        0, fixture.shape.outputChannels - 1, fixture.shape.outputChannels,
        Math.floor(plan.outputElements / 2), plan.outputElements - 1,
        ...plan.outputRanges.flatMap((range) => [
          range.firstOutput,
          range.firstOutput + range.outputCount - 1,
        ]),
        ...(fixture.id === "arithmetic-discriminants"
          ? Object.values(OPT_0007_ARITHMETIC_SENTINELS)
          : []),
      ])];
      const scalarBits = bitsOf(scalar);
      const candidateBits = bitsOf(candidate);
      const sentinels = sentinelIndices.map((index) => {
        const expected = opt0007ExpectedOutputValue(fixture.id, index);
        const expectedBits = float32Bits(expected);
        if (scalarBits[index] !== expectedBits || candidateBits[index] !== expectedBits) {
          throw new Error(`${fixture.id} arithmetic sentinel ${index} mismatch`);
        }
        return Object.freeze({ index, expectedBits: hex32(expectedBits), scalarBitExact: true,
          candidateBitExact: true });
      });
      cases.push(Object.freeze({
        id: fixture.id,
        shape: fixture.shape,
        hasBias: fixture.hasBias,
        outputPrefill: "quiet-NaN-u32-sentinel",
        inputChannelChunkBoundaryOrTail: fixture.shape.inputChannels,
        frameTail: plan.outputFrames % plan.tileFrames,
        outputChannelTail: plan.outputChannels % plan.tileChannels,
        rangeCount: plan.outputRangeCount,
        rangeBatches: plan.outputRanges.map(({ batch }) => batch),
        ...identity,
        cpuArithmeticSentinelsBitExact: true,
        sentinels: Object.freeze(sentinels),
      }));
    } finally {
      scalarPrepared?.destroy();
      candidatePrepared?.destroy();
      input?.destroy();
      weight?.destroy();
      bias?.destroy();
      scalarOutput?.destroy();
      candidateOutput?.destroy();
    }
  }
  return Object.freeze({
    classification: "complete-manageable-full-u32-scalar-versus-candidate",
    cases: Object.freeze(cases),
  });
}

async function runProductionScreen(
  device: GPUDevice,
  mode: Opt0007EndpointMode,
): Promise<unknown> {
  const shape = productionShapeFor(mode);
  const plan = planAceOpt0007VaeK1Conv1d(shape);
  const rangeIndex = middleRangeIndex(plan.outputRangeCount);
  const range = plan.outputRanges[rangeIndex]!;
  let input: GPUBuffer | undefined;
  let weight: GPUBuffer | undefined;
  let bias: GPUBuffer | undefined;
  let scalarOutput: GPUBuffer | undefined;
  let candidateOutput: GPUBuffer | undefined;
  let scalarPrepared: OwnedDispatches | undefined;
  let candidatePrepared: OwnedDispatches | undefined;
  try {
    input = mappedGeneratedStorage(
      device,
      `${mode}-input`,
      plan.inputElements,
      productionInputValue,
    );
    weight = mappedGeneratedStorage(
      device,
      `${mode}-weight`,
      plan.weightElements,
      productionWeightValue,
    );
    bias = mappedGeneratedStorage(
      device,
      `${mode}-bias`,
      plan.outputChannels,
      productionBiasValue,
    );
    scalarOutput = outputBuffer(
      device,
      `${mode}-scalar-output`,
      plan.outputElements,
    );
    candidateOutput = outputBuffer(
      device,
      `${mode}-candidate-output`,
      plan.outputElements,
    );
    scalarPrepared = await prepareDispatches(device, shape, {
      input: binding(input), weight: binding(weight), bias: binding(bias),
      output: binding(scalarOutput),
    });
    candidatePrepared = await prepareDispatches(device, shape, {
      input: binding(input), weight: binding(weight), bias: binding(bias),
      output: binding(candidateOutput),
    });
    const scalar = singleRangeDispatch(scalarPrepared.scalar, rangeIndex);
    const candidate = singleRangeDispatch(candidatePrepared.candidate, rangeIndex);
    const sentinel = new Uint32Array(range.outputCount).fill(OUTPUT_SENTINEL_BITS);
    device.queue.writeBuffer(scalarOutput, range.firstOutput * FLOAT32_BYTES, sentinel);
    device.queue.writeBuffer(candidateOutput, range.firstOutput * FLOAT32_BYTES, sentinel);
    const guardIndices = outputGuardIndices(range, plan.outputElements);
    const guardSentinel = new Uint32Array([OUTPUT_SENTINEL_BITS]);
    for (const index of guardIndices) {
      device.queue.writeBuffer(scalarOutput, index * FLOAT32_BYTES, guardSentinel);
      device.queue.writeBuffer(candidateOutput, index * FLOAT32_BYTES, guardSentinel);
    }
    await device.queue.onSubmittedWorkDone();
    updateProgress(`${mode} symmetric scalar warmup`);
    const scalarWarmup = await executeDispatch(device, scalar, range.multiplyAdds, -1,
      "warmup-scalar-candidate", 0);
    updateProgress(`${mode} symmetric candidate warmup`);
    const candidateWarmup = await executeDispatch(device, candidate, range.multiplyAdds, -1,
      "warmup-scalar-candidate", 1);
    const scalarValues = await readOutput(device, scalarOutput, range.firstOutput,
      range.outputCount, `${mode}-scalar-range`);
    const candidateValues = await readOutput(device, candidateOutput, range.firstOutput,
      range.outputCount, `${mode}-candidate-range`);
    const correctness = validateFullIdentity(scalarValues, candidateValues,
      range.outputCount, mode);
    const guards = await validateGuards(
      device,
      scalarOutput,
      candidateOutput,
      guardIndices,
      mode,
    );
    const samples: Record<KernelId, ExecutionTiming[]> = { scalar: [], candidate: [] };
    const heartbeat = startHeartbeat();
    let responsiveness: HeartbeatResult;
    try {
      for (const [roundIndex, order] of OPT_0007_PAIRED_ORDERS.entries()) {
        for (const [orderPosition, kernelId] of order.entries()) {
          updateProgress(`${mode} round ${roundIndex + 1} ${kernelId}`);
          samples[kernelId].push(await executeDispatch(
            device,
            kernelId === "scalar" ? scalar : candidate,
            range.multiplyAdds,
            roundIndex,
            order.join("-"),
            orderPosition,
          ));
          await yieldToPage();
        }
      }
    } finally {
      responsiveness = heartbeat.stop();
    }
    const scalarPost = await readOutput(device, scalarOutput, range.firstOutput,
      range.outputCount, `${mode}-scalar-post`);
    const candidatePost = await readOutput(device, candidateOutput, range.firstOutput,
      range.outputCount, `${mode}-candidate-post`);
    validateFullIdentity(scalarPost, candidatePost, range.outputCount, `${mode}-post`);
    const postTimingGuards = await validateGuards(
      device,
      scalarOutput,
      candidateOutput,
      guardIndices,
      `${mode}-post-timing`,
    );
    const cancellation = await runCancellationProof(
      device,
      candidatePrepared.candidate,
    );
    const scalarSummary = summarizeExecutions(samples.scalar);
    const candidateSummary = summarizeExecutions(samples.candidate);
    return Object.freeze({
      endpoint: mode,
      shape,
      scope: {
        fullPlanRangeCount: plan.outputRangeCount,
        selectedRangeIndex: rangeIndex,
        firstOutput: range.firstOutput,
        outputCount: range.outputCount,
        outputRowCount: range.outputRowCount,
        validMultiplyAdds: range.multiplyAdds,
        measuredCommandBuffersPerExecution: 1,
        fullOperationExecuted: false,
      },
      memory: {
        inputBytes: plan.inputElements * FLOAT32_BYTES,
        weightBytes: plan.weightElements * FLOAT32_BYTES,
        biasBytes: plan.outputChannels * FLOAT32_BYTES,
        outputBytesPerIndependentBuffer: plan.outputElements * FLOAT32_BYTES,
        simultaneousOutputCopies: 2,
        candidateWorkgroupStorageBytes: plan.workgroupStorageBytes,
      },
      sourceMarkers: candidateSourceMarkers(shape),
      warmup: { scalar: scalarWarmup, candidate: candidateWarmup },
      correctness: {
        ...correctness,
        guards,
        postTimingBitIdentical: true,
        postTimingGuards,
      },
      cancellation,
      scalar: { rawSamples: Object.freeze(samples.scalar), summary: scalarSummary },
      candidate: { rawSamples: Object.freeze(samples.candidate), summary: candidateSummary },
      pairedRounds: OPT_0007_PAIRED_ORDERS.map((order, roundIndex) => Object.freeze({
        roundIndex,
        order: order.join("-"),
        scalarWallMilliseconds: samples.scalar[roundIndex]!.wallMilliseconds,
        candidateWallMilliseconds: samples.candidate[roundIndex]!.wallMilliseconds,
      })),
      delta: {
        medianWallSpeedup: scalarSummary.wallMilliseconds.median /
          candidateSummary.wallMilliseconds.median,
        candidateRoundWins: samples.candidate.filter((sample, index) =>
          sample.wallMilliseconds < samples.scalar[index]!.wallMilliseconds).length,
      },
      responsiveness,
    });
  } finally {
    scalarPrepared?.destroy();
    candidatePrepared?.destroy();
    input?.destroy();
    weight?.destroy();
    bias?.destroy();
    scalarOutput?.destroy();
    candidateOutput?.destroy();
  }
}

async function runC128SequenceScreen(device: GPUDevice): Promise<unknown> {
  const mode = "sequence-c128";
  const shape = OPT_0007_SCREEN_C128_SHAPE;
  const plan = planAceOpt0007VaeK1Conv1d(shape);
  const selectedRangeIndices = sequenceC128RangeIndices(plan.outputRangeCount);
  const selectedRanges = selectedRangeIndices.map(
    (rangeIndex) => plan.outputRanges[rangeIndex]!,
  );
  validateConsecutiveSequenceRanges(selectedRanges);
  const firstOutput = selectedRanges[0]!.firstOutput;
  const outputCount = selectedRanges.reduce(
    (total, range) => total + range.outputCount,
    0,
  );
  const validMultiplyAdds = selectedRanges.reduce(
    (total, range) => total + range.multiplyAdds,
    0,
  );
  const lastOutput = firstOutput + outputCount;
  let input: GPUBuffer | undefined;
  let weight: GPUBuffer | undefined;
  let bias: GPUBuffer | undefined;
  let scalarOutput: GPUBuffer | undefined;
  let candidateOutput: GPUBuffer | undefined;
  let scalarPrepared: OwnedDispatches | undefined;
  let candidatePrepared: OwnedDispatches | undefined;
  try {
    input = mappedGeneratedStorage(
      device,
      `${mode}-input`,
      plan.inputElements,
      productionInputValue,
    );
    weight = mappedGeneratedStorage(
      device,
      `${mode}-weight`,
      plan.weightElements,
      productionWeightValue,
    );
    bias = mappedGeneratedStorage(
      device,
      `${mode}-bias`,
      plan.outputChannels,
      productionBiasValue,
    );
    scalarOutput = outputBuffer(
      device,
      `${mode}-scalar-output`,
      plan.outputElements,
    );
    candidateOutput = outputBuffer(
      device,
      `${mode}-candidate-output`,
      plan.outputElements,
    );
    scalarPrepared = await prepareDispatches(device, shape, {
      input: binding(input),
      weight: binding(weight),
      bias: binding(bias),
      output: binding(scalarOutput),
    });
    candidatePrepared = await prepareDispatches(device, shape, {
      input: binding(input),
      weight: binding(weight),
      bias: binding(bias),
      output: binding(candidateOutput),
    });
    const scalar = selectedRangesDispatch(
      scalarPrepared.scalar,
      selectedRangeIndices,
    );
    const candidate = selectedRangesDispatch(
      candidatePrepared.candidate,
      selectedRangeIndices,
    );
    const sentinel = new Uint32Array(outputCount).fill(OUTPUT_SENTINEL_BITS);
    device.queue.writeBuffer(
      scalarOutput,
      firstOutput * FLOAT32_BYTES,
      sentinel,
    );
    device.queue.writeBuffer(
      candidateOutput,
      firstOutput * FLOAT32_BYTES,
      sentinel,
    );
    const guardIndices = Object.freeze([
      firstOutput - 1,
      lastOutput,
    ]);
    if (guardIndices[0]! < 0 || guardIndices[1]! >= plan.outputElements) {
      throw new Error("OPT-0007 C128 sequence lacks both boundary guards");
    }
    const guardSentinel = new Uint32Array([OUTPUT_SENTINEL_BITS]);
    for (const index of guardIndices) {
      device.queue.writeBuffer(
        scalarOutput,
        index * FLOAT32_BYTES,
        guardSentinel,
      );
      device.queue.writeBuffer(
        candidateOutput,
        index * FLOAT32_BYTES,
        guardSentinel,
      );
    }
    await device.queue.onSubmittedWorkDone();

    updateProgress("sequence-c128 symmetric scalar full-sequence warmup");
    const scalarWarmup = await executeSequenceDispatch(
      device,
      scalar,
      selectedRangeIndices,
      validMultiplyAdds,
      -1,
      "warmup-scalar-candidate",
      0,
    );
    updateProgress("sequence-c128 symmetric candidate full-sequence warmup");
    const candidateWarmup = await executeSequenceDispatch(
      device,
      candidate,
      selectedRangeIndices,
      validMultiplyAdds,
      -1,
      "warmup-scalar-candidate",
      1,
    );
    const scalarValues = await readOutput(
      device,
      scalarOutput,
      firstOutput,
      outputCount,
      `${mode}-scalar-union`,
    );
    const candidateValues = await readOutput(
      device,
      candidateOutput,
      firstOutput,
      outputCount,
      `${mode}-candidate-union`,
    );
    const correctness = validateFullIdentity(
      scalarValues,
      candidateValues,
      outputCount,
      mode,
    );
    const guards = await validateGuards(
      device,
      scalarOutput,
      candidateOutput,
      guardIndices,
      mode,
    );

    const samples: Record<KernelId, SequenceExecutionTiming[]> = {
      scalar: [],
      candidate: [],
    };
    const heartbeat = startHeartbeat();
    let responsiveness: HeartbeatResult;
    try {
      for (const [roundIndex, order] of OPT_0007_PAIRED_ORDERS.entries()) {
        for (const [orderPosition, kernelId] of order.entries()) {
          updateProgress(
            `sequence-c128 round ${roundIndex + 1} ${kernelId}`,
          );
          samples[kernelId].push(await executeSequenceDispatch(
            device,
            kernelId === "scalar" ? scalar : candidate,
            selectedRangeIndices,
            validMultiplyAdds,
            roundIndex,
            order.join("-"),
            orderPosition,
          ));
          await yieldToPage();
        }
      }
    } finally {
      responsiveness = heartbeat.stop();
    }
    const scalarPost = await readOutput(
      device,
      scalarOutput,
      firstOutput,
      outputCount,
      `${mode}-scalar-post`,
    );
    const candidatePost = await readOutput(
      device,
      candidateOutput,
      firstOutput,
      outputCount,
      `${mode}-candidate-post`,
    );
    validateFullIdentity(
      scalarPost,
      candidatePost,
      outputCount,
      `${mode}-post`,
    );
    const postTimingGuards = await validateGuards(
      device,
      scalarOutput,
      candidateOutput,
      guardIndices,
      `${mode}-post-timing`,
    );
    const cancellation = await runSequenceCancellationProof(
      device,
      candidate,
      selectedRangeIndices,
    );
    const scalarSummary = summarizeSequenceExecutions(samples.scalar);
    const candidateSummary = summarizeSequenceExecutions(samples.candidate);
    return Object.freeze({
      endpoint: mode,
      shape,
      scope: {
        fullPlanRangeCount: plan.outputRangeCount,
        selectedRangeIndices,
        selectedRangeCount: selectedRangeIndices.length,
        ranges: Object.freeze(selectedRanges.map((range, localRangeIndex) =>
          Object.freeze({
            localRangeIndex,
            plannedRangeIndex: selectedRangeIndices[localRangeIndex]!,
            firstOutput: range.firstOutput,
            outputCount: range.outputCount,
            outputRowCount: range.outputRowCount,
            validMultiplyAdds: range.multiplyAdds,
          })
        )),
        firstOutput,
        outputCount,
        outputBytes: outputCount * FLOAT32_BYTES,
        outputRows: selectedRanges.reduce(
          (total, range) => total + range.outputRowCount,
          0,
        ),
        validMultiplyAdds,
        countsPerExecution: expectedOpt0007SequenceCounts(),
        fullOperationExecuted: false,
        fullWindowOrSong: false,
      },
      memory: {
        inputBytes: plan.inputElements * FLOAT32_BYTES,
        weightBytes: plan.weightElements * FLOAT32_BYTES,
        biasBytes: plan.outputChannels * FLOAT32_BYTES,
        outputBytesPerIndependentBuffer: plan.outputElements * FLOAT32_BYTES,
        simultaneousOutputCopies: 2,
        unionReadbackBytes: outputCount * FLOAT32_BYTES,
        candidateWorkgroupStorageBytes: plan.workgroupStorageBytes,
      },
      sourceMarkers: candidateSourceMarkers(shape),
      warmup: { scalar: scalarWarmup, candidate: candidateWarmup },
      correctness: {
        ...correctness,
        classification: "full-contiguous-16-range-union-u32-identity",
        outputPrefill: "quiet-NaN-u32-sentinel",
        guards,
        postTimingBitIdentical: true,
        postTimingGuards,
      },
      cancellation,
      scalar: {
        rawSamples: Object.freeze(samples.scalar),
        summary: scalarSummary,
      },
      candidate: {
        rawSamples: Object.freeze(samples.candidate),
        summary: candidateSummary,
      },
      pairedRounds: OPT_0007_PAIRED_ORDERS.map(
        (order, roundIndex) => Object.freeze({
          roundIndex,
          order: order.join("-"),
          scalarWallMilliseconds:
            samples.scalar[roundIndex]!.wallMilliseconds,
          candidateWallMilliseconds:
            samples.candidate[roundIndex]!.wallMilliseconds,
        }),
      ),
      delta: {
        medianWallSpeedup:
          scalarSummary.wallMilliseconds.median /
          candidateSummary.wallMilliseconds.median,
        candidateRoundWins: samples.candidate.filter((sample, index) =>
          sample.wallMilliseconds < samples.scalar[index]!.wallMilliseconds
        ).length,
      },
      responsiveness,
    });
  } finally {
    scalarPrepared?.destroy();
    candidatePrepared?.destroy();
    input?.destroy();
    weight?.destroy();
    bias?.destroy();
    scalarOutput?.destroy();
    candidateOutput?.destroy();
  }
}

async function prepareDispatches(
  device: GPUDevice,
  shape: AceVaeConv1dShape,
  bindings: AceVaeConvBindings,
): Promise<OwnedDispatches> {
  const scalarKernel = AceCorrectnessVaePrimitiveKernel.create(device);
  const candidateKernel = AceOpt0007VaeK1Conv1dKernel.create(device);
  const controls: GPUBuffer[] = [];
  try {
    const candidate = await candidateKernel.createDispatch("opt-0007-candidate", shape, bindings);
    const scalarDispatches: AceVaeDispatch<AceVaeConvPlan>[] = [];
    for (const [index, range] of candidate.plan.outputRanges.entries()) {
      const control = rangeControlBuffer(device, `opt-0007-scalar-control-${index}`,
        range.firstOutput, range.outputCount);
      controls.push(control);
      scalarDispatches.push(await scalarKernel.createConv1dDispatch(
        `opt-0007-scalar-${index}`,
        shape,
        bindings,
        { base: range.firstOutput, count: range.outputCount,
          control: { buffer: control, offset: 0, size: 16 } },
      ));
    }
    const scalar: RangeDispatch = Object.freeze({
      label: "opt-0007-scalar",
      rangeCount: scalarDispatches.length,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        const dispatch = scalarDispatches[rangeIndex];
        if (dispatch === undefined) throw new RangeError("invalid OPT-0007 scalar range");
        dispatch.encode(pass);
      },
    });
    return Object.freeze({
      scalar,
      candidate,
      scalarKernel,
      candidateKernel,
      controls: Object.freeze(controls),
      destroy(): void {
        scalarKernel.destroy();
        candidateKernel.destroy();
        for (const control of controls) control.destroy();
      },
    });
  } catch (error) {
    scalarKernel.destroy();
    candidateKernel.destroy();
    for (const control of controls) control.destroy();
    throw error;
  }
}

function singleRangeDispatch(dispatch: RangeDispatch, selected: number): RangeDispatch {
  if (!Number.isSafeInteger(selected) || selected < 0 || selected >= dispatch.rangeCount) {
    throw new RangeError("invalid OPT-0007 selected range");
  }
  return Object.freeze({
    label: `${dispatch.label}-selected-${selected}`,
    rangeCount: 1,
    encodeRange(pass: GPUComputePassEncoder, local: number): void {
      if (local !== 0) throw new RangeError("single-range dispatch accepts only zero");
      dispatch.encodeRange(pass, selected);
    },
  });
}

function selectedRangesDispatch(
  dispatch: RangeDispatch,
  selectedRangeIndices: readonly number[],
): RangeDispatch {
  if (
    selectedRangeIndices.length === 0 ||
    selectedRangeIndices.some((rangeIndex) =>
      !Number.isSafeInteger(rangeIndex) || rangeIndex < 0 ||
      rangeIndex >= dispatch.rangeCount
    )
  ) {
    throw new RangeError("invalid OPT-0007 selected range sequence");
  }
  return Object.freeze({
    label: `${dispatch.label}-selected-sequence`,
    rangeCount: selectedRangeIndices.length,
    encodeRange(pass: GPUComputePassEncoder, localRangeIndex: number): void {
      const plannedRangeIndex = selectedRangeIndices[localRangeIndex];
      if (plannedRangeIndex === undefined) {
        throw new RangeError("invalid OPT-0007 local sequence range");
      }
      dispatch.encodeRange(pass, plannedRangeIndex);
    },
  });
}

function selectedSequenceQuanta(
  dispatch: RangeDispatch,
  selectedRangeIndices: readonly number[],
): readonly AceOpt0006EncodableQuantum[] {
  if (
    dispatch.rangeCount !== OPT_0007_SEQUENCE_RANGE_COUNT ||
    selectedRangeIndices.length !== OPT_0007_SEQUENCE_RANGE_COUNT
  ) {
    throw new Error("OPT-0007 sequence quantum topology changed");
  }
  return Object.freeze(Array.from(
    { length: OPT_0007_SEQUENCE_RANGE_COUNT },
    (_, localRangeIndex) => Object.freeze({
      id: `range-${selectedRangeIndices[localRangeIndex]}`,
      encode(pass: GPUComputePassEncoder): void {
        dispatch.encodeRange(pass, localRangeIndex);
      },
    }),
  ));
}

function instrumentSequenceDevice(
  device: GPUDevice,
  counters: SequenceExecutionCounters,
): Pick<GPUDevice, "createCommandEncoder"> {
  return {
    createCommandEncoder(
      descriptor?: GPUCommandEncoderDescriptor,
    ): GPUCommandEncoder {
      counters.commandEncoderCount += 1;
      const encoder = device.createCommandEncoder(descriptor);
      return new Proxy(encoder, {
        get(target, property): unknown {
          if (property === "beginComputePass") {
            return (passDescriptor?: GPUComputePassDescriptor) => {
              counters.passCount += 1;
              const pass = target.beginComputePass(passDescriptor);
              return new Proxy(pass, {
                get(passTarget, passProperty): unknown {
                  if (passProperty === "dispatchWorkgroups") {
                    return (
                      x: number,
                      y?: number,
                      z?: number,
                    ): void => {
                      counters.dispatchCount += 1;
                      passTarget.dispatchWorkgroups(x, y, z);
                    };
                  }
                  const value = Reflect.get(passTarget, passProperty);
                  return typeof value === "function"
                    ? value.bind(passTarget)
                    : value;
                },
              });
            };
          }
          const value = Reflect.get(target, property);
          return typeof value === "function" ? value.bind(target) : value;
        },
      });
    },
  };
}

function validateSequenceProgress(
  progress: readonly AceOpt0006BatchProgress[],
): void {
  const expected = expectedOpt0007SequenceCounts();
  if (progress.length !== expected.progressEventCount) {
    throw new Error("OPT-0007 sequence progress count changed");
  }
  for (let quantumIndex = 0; quantumIndex < progress.length; quantumIndex += 1) {
    const batchIndex = Math.floor(
      quantumIndex / OPT_0007_SEQUENCE_RANGES_PER_COMMAND_BUFFER,
    );
    const event = progress[quantumIndex]!;
    if (
      event.completedQuanta !== quantumIndex + 1 ||
      event.totalQuanta !== expected.rangeCount ||
      event.commandBuffersSubmitted !== batchIndex + 1 ||
      event.queueDrains !== batchIndex + 1 ||
      event.cooperativeIdleMs !== batchIndex + 1 ||
      event.completedBatchIndex !== batchIndex ||
      event.totalBatches !== expected.commandBufferCount
    ) {
      throw new Error(`OPT-0007 sequence progress ${quantumIndex} changed`);
    }
  }
}

async function executeSequenceDispatch(
  device: GPUDevice,
  dispatch: RangeDispatch,
  selectedRangeIndices: readonly number[],
  logicalMultiplyAdds: number,
  roundIndex: number,
  pairedOrder: string,
  orderPosition: number,
): Promise<SequenceExecutionTiming> {
  const expected = expectedOpt0007SequenceCounts();
  if (
    dispatch.rangeCount !== expected.rangeCount ||
    selectedRangeIndices.length !== expected.rangeCount
  ) {
    throw new Error("OPT-0007 sequence execution topology changed");
  }
  const counters: SequenceExecutionCounters = {
    commandEncoderCount: 0,
    passCount: 0,
    dispatchCount: 0,
  };
  const batches: SequenceBatchTiming[] = [];
  let encodeMilliseconds = 0;
  let submitMilliseconds = 0;
  let drainMilliseconds = 0;
  let explicitIdleMilliseconds = 0;
  const progress: AceOpt0006BatchProgress[] = [];
  let explicitIdleCount = 0;
  let outstanding = 0;
  let maximumOutstanding = 0;
  const heartbeat = startHeartbeat();
  let activeBatchIndex = 0;
  let activeBatchEncodeStarted = 0;
  let activeBatchEncoded = 0;
  let activeBatchSubmitted = 0;
  let activeBatchDrainStarted = 0;
  let activeBatchDrained = 0;
  let activeIdleStarted = 0;
  const quanta = selectedSequenceQuanta(
    dispatch,
    selectedRangeIndices,
  );
  const instrumentedDevice = {
    createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder {
      counters.commandEncoderCount += 1;
      activeBatchIndex = counters.commandEncoderCount - 1;
      activeBatchEncodeStarted = performance.now();
      const encoder = device.createCommandEncoder(descriptor);
      return new Proxy(encoder, {
        get(target, property): unknown {
          if (property === "beginComputePass") {
            return (passDescriptor?: GPUComputePassDescriptor) => {
              counters.passCount += 1;
              const pass = target.beginComputePass(passDescriptor);
              return new Proxy(pass, {
                get(passTarget, passProperty): unknown {
                  if (passProperty === "dispatchWorkgroups") {
                    return (...args: Parameters<GPUComputePassEncoder["dispatchWorkgroups"]>) => {
                      counters.dispatchCount += 1;
                      return passTarget.dispatchWorkgroups(...args);
                    };
                  }
                  const value = Reflect.get(passTarget, passProperty);
                  return typeof value === "function" ? value.bind(passTarget) : value;
                },
              });
            };
          }
          if (property === "finish") {
            return (finishDescriptor?: GPUCommandBufferDescriptor) => {
              const command = target.finish(finishDescriptor);
              activeBatchEncoded = performance.now() - activeBatchEncodeStarted;
              return command;
            };
          }
          const value = Reflect.get(target, property);
          return typeof value === "function" ? value.bind(target) : value;
        },
      });
    },
  } satisfies Pick<GPUDevice, "createCommandEncoder">;
  const instrumentedQueue = {
    submit(commandBuffers: Iterable<GPUCommandBuffer>): undefined {
      const retained = [...commandBuffers];
      if (retained.length !== 1 || outstanding !== 0) {
        throw new Error("OPT-0007 sequence submit violated singleton FIFO");
      }
      const submitStarted = performance.now();
      const result = device.queue.submit(retained);
      activeBatchSubmitted = performance.now() - submitStarted;
      outstanding = 1;
      maximumOutstanding = Math.max(maximumOutstanding, outstanding);
      activeBatchDrainStarted = performance.now();
      return result;
    },
    async onSubmittedWorkDone(): Promise<undefined> {
      await device.queue.onSubmittedWorkDone();
      activeBatchDrained = performance.now() - activeBatchDrainStarted;
      outstanding = 0;
      return undefined;
    },
  } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
  const started = performance.now();
  let result;
  try {
    result = await runAceOpt0006QuantumBatches({
      device: instrumentedDevice,
      queue: instrumentedQueue,
      quanta,
      maximumQuantaPerCommandBuffer:
        OPT_0007_SEQUENCE_RANGES_PER_COMMAND_BUFFER,
      signal: new AbortController().signal,
      finalCommandBufferRemains: true,
      label: `${dispatch.label}-production-batch8-sequence`,
      yieldQueueIdle: async () => {
        activeIdleStarted = performance.now();
        await realQueueEmptyIdle();
        const idle = performance.now() - activeIdleStarted;
        if (idle < EXPLICIT_IDLE_MILLISECONDS) {
          throw new Error("OPT-0007 sequence real idle collapsed");
        }
        explicitIdleCount += 1;
        explicitIdleMilliseconds += idle;
        encodeMilliseconds += activeBatchEncoded;
        submitMilliseconds += activeBatchSubmitted;
        drainMilliseconds += activeBatchDrained;
        const firstLocalRangeIndex =
          activeBatchIndex * OPT_0007_SEQUENCE_RANGES_PER_COMMAND_BUFFER;
        batches.push(Object.freeze({
          batchIndex: activeBatchIndex,
          firstLocalRangeIndex,
          rangeCount: 8,
          passCount: 8,
          dispatchCount: 8,
          commandBufferCount: 1,
          queueDrains: 1,
          encodeMilliseconds: activeBatchEncoded,
          submitMilliseconds: activeBatchSubmitted,
          drainMilliseconds: activeBatchDrained,
          explicitIdleMilliseconds: idle,
          ranges: Object.freeze(Array.from({ length: 8 }, (_, offset) => {
            const localRangeIndex = firstLocalRangeIndex + offset;
            return Object.freeze({
              localRangeIndex,
              plannedRangeIndex: selectedRangeIndices[localRangeIndex]!,
              passIndex: localRangeIndex,
            });
          })),
          heartbeat: heartbeat.snapshot(),
        }));
      },
      onProgress: (event) => progress.push(event),
    });
    validateSequenceProgress(progress);
  } finally {
    heartbeat.stop();
  }
  const wallMilliseconds = performance.now() - started;
  if (
    counters.passCount !== expected.passCount ||
    counters.dispatchCount !== expected.dispatchCount ||
    counters.commandEncoderCount !== expected.commandBufferCount ||
    result.commandBuffersSubmitted !== expected.commandBufferCount ||
    result.queueDrains !== expected.queueDrains ||
    progress.length !== expected.progressEventCount ||
    result.completedQuanta !== expected.rangeCount ||
    result.batchCount !== expected.commandBufferCount ||
    result.cooperativeIdleMs !== expected.explicitIdleCount ||
    explicitIdleCount !== expected.explicitIdleCount ||
    maximumOutstanding !== 1 || outstanding !== 0
  ) {
    throw new Error("OPT-0007 sequence execution accounting changed");
  }
  return Object.freeze({
    roundIndex,
    pairedOrder,
    orderPosition,
    wallMilliseconds,
    encodeMilliseconds,
    submitMilliseconds,
    drainMilliseconds,
    explicitIdleMilliseconds,
    rangeCount: 16,
    passCount: 16,
    dispatchCount: 16,
    progressEventCount: 16,
    commandBufferCount: 2,
    queueDrains: 2,
    explicitIdleCount: 2,
    maximumOutstandingCommandBuffers: 1,
    maximumSingleDrainMilliseconds: Math.max(
      ...batches.map((batch) => batch.drainMilliseconds),
    ),
    logicalGmacPerSecond: logicalMultiplyAdds / wallMilliseconds / 1e6,
    batches: Object.freeze(batches),
    progress: Object.freeze(progress),
  });
}

async function runSequenceCancellationProof(
  device: GPUDevice,
  dispatch: RangeDispatch,
  selectedRangeIndices: readonly number[],
): Promise<unknown> {
  const controller = new AbortController();
  const counters: SequenceExecutionCounters = {
    commandEncoderCount: 0,
    passCount: 0,
    dispatchCount: 0,
  };
  let encodedPasses = 0;
  let submittedCommandBuffers = 0;
  let queueDrains = 0;
  let explicitIdleCount = 0;
  let triggerDeliveredAt: number | undefined;
  let firstDrainCompletedAt: number | undefined;
  const heartbeat = startHeartbeat();
  const started = performance.now();
  let responsiveness: HeartbeatResult;
  let explicitIdleMilliseconds = 0;
  try {
    const quanta = selectedSequenceQuanta(
      dispatch,
      selectedRangeIndices,
    );
    let rejection: unknown;
    try {
      await runAceOpt0006QuantumBatches({
        device: instrumentSequenceDevice(device, counters),
        queue: {
          submit(commandBuffers): undefined {
            device.queue.submit(commandBuffers);
            submittedCommandBuffers += 1;
            return undefined;
          },
          async onSubmittedWorkDone(): Promise<undefined> {
            await device.queue.onSubmittedWorkDone();
            queueDrains += 1;
            firstDrainCompletedAt = performance.now();
            return undefined;
          },
        },
        quanta,
        maximumQuantaPerCommandBuffer:
          OPT_0007_SEQUENCE_RANGES_PER_COMMAND_BUFFER,
        signal: controller.signal,
        finalCommandBufferRemains: true,
        label: "opt-0007-sequence-cancellation",
        yieldQueueIdle: async () => {
          const idleStarted = performance.now();
          await realQueueEmptyIdle();
          explicitIdleCount += 1;
          explicitIdleMilliseconds += performance.now() - idleStarted;
        },
        onProgress: (event) => {
          if (event.completedQuanta === 8 && !controller.signal.aborted) {
            triggerDeliveredAt = performance.now();
            controller.abort(new DOMException(
              "OPT-0007 sequence cancellation",
              "AbortError",
            ));
          }
        },
      });
    } catch (error) {
      rejection = error;
    }
    encodedPasses = counters.passCount;
    if (
      !controller.signal.aborted ||
      encodedPasses !== 8 ||
      counters.dispatchCount !== 8 ||
      counters.commandEncoderCount !== 1 ||
      submittedCommandBuffers !== 1 ||
      queueDrains !== 1 ||
      explicitIdleCount !== 1 ||
      explicitIdleMilliseconds < EXPLICIT_IDLE_MILLISECONDS ||
      triggerDeliveredAt === undefined ||
      firstDrainCompletedAt === undefined ||
      selectedRangeIndices.length !== 16 ||
      !(rejection instanceof DOMException && rejection.name === "AbortError")
    ) {
      throw new Error("OPT-0007 sequence cancellation proof failed");
    }
    return Object.freeze({
      classification: "abort-after-first-drained-eight-pass-batch",
      plannedRangeIndices: selectedRangeIndices,
      plannedBatchCount: 2,
      encodedPasses,
      physicalDispatches: counters.dispatchCount,
      commandEncoders: counters.commandEncoderCount,
      submittedCommandBuffers,
      queueDrains,
      explicitIdleCount,
      explicitIdleMilliseconds,
      firstBatchFullyDrained: true,
      firstBatchPassesAlreadySubmitted: 8,
      signalAborted: true,
      laterBatchEncodingPrevented: true,
      laterBatchSubmissionPrevented: true,
      triggerDeliveryMilliseconds: triggerDeliveredAt - started,
      firstDrainMilliseconds: firstDrainCompletedAt - started,
      cancellationObservationMilliseconds:
        triggerDeliveredAt - firstDrainCompletedAt,
      abortWasDeliveredAfterDrainCompleted:
        triggerDeliveredAt >= firstDrainCompletedAt,
      responsiveness: heartbeat.snapshot(),
    });
  } finally {
    responsiveness = heartbeat.stop();
    void responsiveness;
  }
}

async function executeUntimed(device: GPUDevice, dispatch: RangeDispatch): Promise<void> {
  for (let rangeIndex = 0; rangeIndex < dispatch.rangeCount; rangeIndex += 1) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatch.encodeRange(pass, rangeIndex);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
}

async function executeDispatch(
  device: GPUDevice,
  dispatch: RangeDispatch,
  logicalMultiplyAdds: number,
  roundIndex: number,
  pairedOrder: string,
  orderPosition: number,
): Promise<ExecutionTiming> {
  const ranges: RangeTiming[] = [];
  let encodeMilliseconds = 0;
  let submitMilliseconds = 0;
  let drainMilliseconds = 0;
  const started = performance.now();
  for (let rangeIndex = 0; rangeIndex < dispatch.rangeCount; rangeIndex += 1) {
    const encodeStarted = performance.now();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatch.encodeRange(pass, rangeIndex);
    pass.end();
    const command = encoder.finish();
    const encoded = performance.now() - encodeStarted;
    const submitStarted = performance.now();
    device.queue.submit([command]);
    const submitted = performance.now() - submitStarted;
    const drainStarted = performance.now();
    await device.queue.onSubmittedWorkDone();
    const drained = performance.now() - drainStarted;
    encodeMilliseconds += encoded;
    submitMilliseconds += submitted;
    drainMilliseconds += drained;
    ranges.push(Object.freeze({ rangeIndex, encodeMilliseconds: encoded,
      submitMilliseconds: submitted, drainMilliseconds: drained }));
  }
  const wallMilliseconds = performance.now() - started;
  return Object.freeze({
    roundIndex, pairedOrder, orderPosition, wallMilliseconds,
    encodeMilliseconds, submitMilliseconds, drainMilliseconds,
    commandBufferCount: dispatch.rangeCount,
    maximumSingleDrainMilliseconds: Math.max(...ranges.map((range) => range.drainMilliseconds)),
    logicalGmacPerSecond: logicalMultiplyAdds / wallMilliseconds / 1e6,
    ranges: Object.freeze(ranges),
  });
}

async function runCancellationProof(device: GPUDevice, dispatch: RangeDispatch): Promise<unknown> {
  if (dispatch.rangeCount < 2) {
    throw new Error("OPT-0007 cancellation proof requires at least two real ranges");
  }
  const controller = new AbortController();
  let encodedRanges = 0;
  let submittedCommandBuffers = 0;
  let queueDrains = 0;
  let triggerDeliveredAt: number | undefined;
  let firstDrainCompletedAt: number | undefined;
  let triggerHandle: number | undefined;
  let resolveTrigger: (() => void) | undefined;
  const trigger = new Promise<void>((resolve) => {
    resolveTrigger = resolve;
  });
  const started = performance.now();
  try {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    dispatch.encodeRange(pass, 0);
    encodedRanges += 1;
    pass.end();
    device.queue.submit([encoder.finish()]);
    submittedCommandBuffers += 1;
    triggerHandle = window.setTimeout(() => {
      triggerDeliveredAt = performance.now();
      controller.abort("opt-0007-cancel");
      resolveTrigger!();
    }, 0);
    await device.queue.onSubmittedWorkDone();
    queueDrains += 1;
    firstDrainCompletedAt = performance.now();
    await trigger;
    for (let rangeIndex = 1; rangeIndex < dispatch.rangeCount; rangeIndex += 1) {
      if (controller.signal.aborted) break;
      const laterEncoder = device.createCommandEncoder();
      const laterPass = laterEncoder.beginComputePass();
      dispatch.encodeRange(laterPass, rangeIndex);
      encodedRanges += 1;
      laterPass.end();
      device.queue.submit([laterEncoder.finish()]);
      submittedCommandBuffers += 1;
      await device.queue.onSubmittedWorkDone();
      queueDrains += 1;
    }
  } finally {
    if (triggerHandle !== undefined) window.clearTimeout(triggerHandle);
  }
  if (
    !controller.signal.aborted ||
    encodedRanges !== 1 ||
    submittedCommandBuffers !== 1 ||
    queueDrains !== 1 ||
    triggerDeliveredAt === undefined ||
    firstDrainCompletedAt === undefined
  ) {
    throw new Error("OPT-0007 cancellation proof failed");
  }
  return Object.freeze({
    classification: "abort-delivered-while-or-after-first-range-drain",
    plannedRangeCount: dispatch.rangeCount,
    encodedRanges,
    submittedCommandBuffers,
    queueDrains,
    firstRangeFullyDrained: true,
    signalAborted: true,
    laterRangeEncodingPrevented: true,
    laterRangeSubmissionPrevented: true,
    triggerDeliveryMilliseconds: triggerDeliveredAt - started,
    firstDrainMilliseconds: firstDrainCompletedAt - started,
    abortWasDeliveredBeforeDrainCompleted:
      triggerDeliveredAt <= firstDrainCompletedAt,
  });
}

async function validateGuards(
  device: GPUDevice,
  scalarOutput: GPUBuffer,
  candidateOutput: GPUBuffer,
  indices: readonly number[],
  label: string,
): Promise<unknown> {
  const scalar = await readSparseBits(device, scalarOutput, indices, `${label}-scalar-guards`);
  const candidate = await readSparseBits(device, candidateOutput, indices, `${label}-candidate-guards`);
  if ([...scalar, ...candidate].some((bits) => bits !== OUTPUT_SENTINEL_BITS)) {
    throw new Error(`${label} guard sentinel changed`);
  }
  return Object.freeze({ indices: Object.freeze(indices), exactOnceRangeBoundary: true,
    scalarBits: [...scalar].map(hex32), candidateBits: [...candidate].map(hex32) });
}

function validateFullIdentity(
  scalar: Float32Array,
  candidate: Float32Array,
  expectedElements: number,
  label: string,
): IdentityResult {
  if (scalar.length !== expectedElements || candidate.length !== expectedElements) {
    throw new Error(`${label} readback length changed`);
  }
  const scalarBits = bitsOf(scalar);
  const candidateBits = bitsOf(candidate);
  let bitMismatchCount = 0;
  for (let index = 0; index < expectedElements; index += 1) {
    if (scalarBits[index] !== candidateBits[index]) bitMismatchCount += 1;
  }
  const scalarFingerprint = fingerprint(scalar, scalarBits);
  const candidateFingerprint = fingerprint(candidate, candidateBits);
  if (bitMismatchCount !== 0 || scalarFingerprint.finiteCount !== expectedElements ||
    candidateFingerprint.finiteCount !== expectedElements ||
    scalarFingerprint.nonzeroCount === 0 || candidateFingerprint.nonzeroCount === 0) {
    throw new Error(`${label} full U32 identity failed (${bitMismatchCount} mismatches)`);
  }
  return Object.freeze({ comparedElementCount: expectedElements, fullRangeBitIdentical: true,
    bitMismatchCount: 0, scalarFingerprint, candidateFingerprint });
}

function summarizeExecutions(samples: readonly ExecutionTiming[]): ExecutionSummary {
  const select = (fn: (sample: ExecutionTiming) => number) =>
    summarizeOpt0007Samples(samples.map(fn));
  return Object.freeze({
    wallMilliseconds: select((sample) => sample.wallMilliseconds),
    encodeMilliseconds: select((sample) => sample.encodeMilliseconds),
    submitMilliseconds: select((sample) => sample.submitMilliseconds),
    drainMilliseconds: select((sample) => sample.drainMilliseconds),
    maximumSingleDrainMilliseconds: select((sample) => sample.maximumSingleDrainMilliseconds),
    logicalGmacPerSecond: select((sample) => sample.logicalGmacPerSecond),
  });
}

function summarizeSequenceExecutions(
  samples: readonly SequenceExecutionTiming[],
): SequenceExecutionSummary {
  if (samples.length !== OPT_0007_PAIRED_ORDERS.length) {
    throw new Error("OPT-0007 sequence sample count changed");
  }
  const select = (fn: (sample: SequenceExecutionTiming) => number) =>
    summarizeOpt0007Samples(samples.map(fn));
  return Object.freeze({
    wallMilliseconds: select((sample) => sample.wallMilliseconds),
    encodeMilliseconds: select((sample) => sample.encodeMilliseconds),
    submitMilliseconds: select((sample) => sample.submitMilliseconds),
    drainMilliseconds: select((sample) => sample.drainMilliseconds),
    explicitIdleMilliseconds: select((sample) =>
      sample.explicitIdleMilliseconds
    ),
    maximumSingleDrainMilliseconds: select((sample) =>
      sample.maximumSingleDrainMilliseconds
    ),
    logicalGmacPerSecond: select((sample) => sample.logicalGmacPerSecond),
  });
}

function validateConsecutiveSequenceRanges(
  ranges: readonly AceOpt0007VaeK1Conv1dOutputRange[],
): void {
  if (ranges.length !== OPT_0007_SEQUENCE_RANGE_COUNT) {
    throw new Error("OPT-0007 sequence range count changed");
  }
  for (let index = 1; index < ranges.length; index += 1) {
    const previous = ranges[index - 1]!;
    const current = ranges[index]!;
    if (
      current.firstOutput !== previous.firstOutput + previous.outputCount ||
      current.batch !== previous.batch
    ) {
      throw new Error("OPT-0007 sequence ranges are not contiguous and batch-local");
    }
  }
}

function outputGuardIndices(
  range: AceOpt0007VaeK1Conv1dOutputRange,
  outputElements: number,
): readonly number[] {
  return Object.freeze([...new Set([
    Math.max(0, range.firstOutput - 1),
    Math.min(outputElements - 1, range.firstOutput + range.outputCount),
  ])].filter((index) =>
    index < range.firstOutput || index >= range.firstOutput + range.outputCount
  ));
}

function productionShapeFor(
  mode: Exclude<Opt0007RunMode, "correctness">,
): AceVaeConv1dShape {
  return mode === "screen-c1024" ? OPT_0007_SCREEN_C1024_SHAPE : OPT_0007_SCREEN_C128_SHAPE;
}

export function middleRangeIndex(rangeCount: number): number {
  if (!Number.isSafeInteger(rangeCount) || rangeCount <= 0) {
    throw new RangeError("OPT-0007 range count must be positive");
  }
  return Math.floor(rangeCount / 2);
}

function productionInputValue(index: number): number {
  return Math.fround((((index * 17 + 3) % 31) - 15) / 32);
}

function productionWeightValue(index: number): number {
  return Math.fround((((index * 13 + 7) % 29) - 14) / 64);
}

function productionBiasValue(channel: number): number {
  return Math.fround(((channel % 11) - 5) / 128);
}

function candidateSourceMarkers(shape: AceVaeConv1dShape): unknown {
  const source = aceOpt0007VaeK1Conv1dWgsl(shape, true);
  const markers = Object.freeze({
    workgroup16x8: source.includes("@workgroup_size") && source.includes("16") && source.includes("8"),
    sharedInput: source.includes("var<workgroup>") && source.includes("input"),
    sharedWeight: source.includes("var<workgroup>") && source.includes("weight"),
    twoUniformBarriersPerChunk: source.match(/workgroupBarrier\(\)/g)?.length === 2,
    noExplicitFma: !source.includes("fma("),
    fp32Only: !source.includes("enable f16"),
  });
  if (Object.values(markers).some((value) => !value)) {
    throw new Error("OPT-0007 candidate source markers changed");
  }
  return markers;
}

function mappedGeneratedStorage(
  device: GPUDevice,
  label: string,
  elements: number,
  value: (index: number) => number,
): GPUBuffer {
  const buffer = device.createBuffer({ label, size: elements * FLOAT32_BYTES,
    usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  try {
    const mapped = new Float32Array(buffer.getMappedRange());
    for (let index = 0; index < elements; index += 1) mapped[index] = value(index);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    buffer.destroy();
    throw error;
  }
}

function mappedSentinelOutput(device: GPUDevice, label: string, elements: number): GPUBuffer {
  const buffer = device.createBuffer({ label, size: elements * FLOAT32_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
  try {
    new Uint32Array(buffer.getMappedRange()).fill(OUTPUT_SENTINEL_BITS);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    buffer.destroy();
    throw error;
  }
}

function outputBuffer(device: GPUDevice, label: string, elements: number): GPUBuffer {
  return device.createBuffer({ label, size: elements * FLOAT32_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
}

function rangeControlBuffer(device: GPUDevice, label: string, base: number, count: number): GPUBuffer {
  const buffer = device.createBuffer({ label, size: 16, usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true });
  try {
    new Uint32Array(buffer.getMappedRange()).set([base, count, 0, 0]);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    buffer.destroy();
    throw error;
  }
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size: buffer.size });
}

async function readOutput(
  device: GPUDevice,
  source: GPUBuffer,
  firstElement: number,
  elementCount: number,
  label: string,
): Promise<Float32Array<ArrayBuffer>> {
  const bytes = elementCount * FLOAT32_BYTES;
  const readback = device.createBuffer({ label, size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, firstElement * FLOAT32_BYTES, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    const result = new Float32Array(elementCount);
    result.set(new Float32Array(readback.getMappedRange()));
    return result;
  } finally {
    if (mapped) readback.unmap();
    readback.destroy();
  }
}

async function readSparseBits(
  device: GPUDevice,
  source: GPUBuffer,
  indices: readonly number[],
  label: string,
): Promise<Uint32Array> {
  const result = new Uint32Array(indices.length);
  for (const [position, index] of indices.entries()) {
    const value = await readOutput(device, source, index, 1, `${label}-${position}`);
    result[position] = bitsOf(value)[0]!;
  }
  return result;
}

function bitsOf(values: Float32Array): Uint32Array {
  return new Uint32Array(values.buffer, values.byteOffset, values.length);
}

function fingerprint(values: Float32Array, bits: Uint32Array): OutputFingerprint {
  let finiteCount = 0;
  let nonzeroCount = 0;
  let positiveZeroCount = 0;
  let negativeZeroCount = 0;
  let hash = 0x811c_9dc5;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (Number.isFinite(value)) finiteCount += 1;
    if (value !== 0) nonzeroCount += 1;
    else if (Object.is(value, -0)) negativeZeroCount += 1;
    else positiveZeroCount += 1;
    hash = Math.imul(hash ^ bits[index]!, 0x0100_0193) >>> 0;
  }
  return Object.freeze({ elementCount: values.length, finiteCount, nonzeroCount,
    positiveZeroCount, negativeZeroCount, fnv1a32: hash.toString(16).padStart(8, "0") });
}

function float32Bits(value: number): number {
  SCRATCH_FLOAT[0] = value;
  return SCRATCH_BITS[0]!;
}

function hex32(value: number): string {
  return `0x${value.toString(16).padStart(8, "0")}`;
}

function largestStorageBindingBytes(shape: AceVaeConv1dShape): number {
  const plan = planAceOpt0007VaeK1Conv1d(shape);
  return Math.max(plan.inputElements, plan.weightElements, plan.outputElements,
    plan.outputChannels) * FLOAT32_BYTES;
}

function assertAdapter(adapter: GPUAdapter, largestBinding: number, workgroupStorage: number): void {
  if (adapter.limits.maxBufferSize < largestBinding ||
    adapter.limits.maxStorageBufferBindingSize < largestBinding ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 256 ||
    adapter.limits.maxComputeWorkgroupSizeX < 256 ||
    adapter.limits.maxComputeWorkgroupSizeY < 8 ||
    adapter.limits.maxComputeWorkgroupStorageSize < workgroupStorage) {
    throw new Error("Adapter cannot satisfy the OPT-0007 paired contract");
  }
}

function adapterIdentity(adapter: GPUAdapter): unknown {
  return Object.freeze({
    vendor: adapter.info.vendor,
    architecture: adapter.info.architecture,
    device: adapter.info.device,
    description: adapter.info.description,
    isFallbackAdapter: adapter.info.isFallbackAdapter,
    features: Object.freeze([...adapter.features].sort()),
    limits: Object.freeze({
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
    }),
  });
}

function startHeartbeat(): HeartbeatController {
  let animationFrameCount = 0;
  let timerTickCount = 0;
  let maximumAnimationFrameGapMilliseconds = 0;
  let maximumTimerGapMilliseconds = 0;
  let stopped = false;
  let lastAnimationFrame = performance.now();
  let lastTimer = performance.now();
  let frameHandle = 0;
  const frame = (now: number): void => {
    if (stopped) return;
    animationFrameCount += 1;
    maximumAnimationFrameGapMilliseconds = Math.max(maximumAnimationFrameGapMilliseconds,
      now - lastAnimationFrame);
    lastAnimationFrame = now;
    frameHandle = requestAnimationFrame(frame);
  };
  frameHandle = requestAnimationFrame(frame);
  const timerHandle = window.setInterval(() => {
    const now = performance.now();
    timerTickCount += 1;
    maximumTimerGapMilliseconds = Math.max(maximumTimerGapMilliseconds, now - lastTimer);
    lastTimer = now;
  }, 50);
  const snapshot = (): HeartbeatResult => Object.freeze({
    animationFrameCount,
    timerTickCount,
    maximumAnimationFrameGapMilliseconds,
    maximumTimerGapMilliseconds,
  });
  return {
    snapshot,
    stop(): HeartbeatResult {
      stopped = true;
      cancelAnimationFrame(frameHandle);
      window.clearInterval(timerHandle);
      return snapshot();
    },
  };
}

async function realQueueEmptyIdle(): Promise<void> {
  await new Promise<void>((resolve) => {
    window.setTimeout(resolve, EXPLICIT_IDLE_MILLISECONDS);
  });
}

async function yieldToPage(): Promise<void> {
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
}

function requiredNumber(parameters: URLSearchParams, key: string): number {
  const value = Number(parameters.get(key));
  if (!Number.isFinite(value)) throw new Error(`OPT-0007 ${key} is required`);
  return value;
}

function updateProgress(message: string): void {
  const progress = document.querySelector<HTMLElement>("#progress");
  if (progress !== null) progress.textContent = message;
}

function finish(status: "passed" | "failed", result: unknown): void {
  document.body.dataset.status = status;
  const output = document.querySelector<HTMLElement>("#result");
  if (output !== null) output.textContent = JSON.stringify(result, null, 2);
  updateProgress(status);
}
