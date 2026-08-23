import {
  AceOpt0005VaeConv1dChannelChunksKernel,
  aceOpt0005VaeConv1dChannelChunksWgsl,
  planAceOpt0005VaeConv1dChannelChunks,
  type AceOpt0005VaeConv1dDispatch,
  type AceOpt0005VaeConv1dOutputRange,
} from "../../benchmark/opt-0005-vae-conv1d.js";
import {
  AceOpt0004VaeConv1dKernel,
  planAceOpt0004VaeConv1d,
  type AceOpt0004VaeConv1dDispatch,
} from "../../benchmark/opt-0004-vae-conv1d.js";
import {
  AceCorrectnessVaePrimitiveKernel,
  planAceVaeConv1d,
  type AceVaeConv1dShape,
  type AceVaeConvBindings,
  type AceVaeDispatch,
  type AceVaeConvPlan,
} from "../../src/webgpu/kernels/vae-primitives.js";

type PairedKernelId = "scalar" | "chunked";
type OverlapKernelId = "scalar" | "opt0004" | "chunked";
export type Opt0005CorrectnessCaseId =
  | "dilation1-channel-tail-bias"
  | "dilation3-batch-padding-no-bias"
  | "dilation9-range-tail-bias"
  | "arithmetic-discriminants";

export type Opt0005RunMode =
  | "screen-d1"
  | "block0-d1"
  | "screen-d9"
  | "overlap-d1";

interface RangeDispatch {
  readonly label: string;
  readonly rangeCount: number;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
}

export function effectiveOpt0005RangeCount(
  mode: Opt0005RunMode,
  plannedRangeCount: number,
): number {
  return screenRangeIndex(mode, plannedRangeCount) === undefined
    ? plannedRangeCount
    : 1;
}

function singleRangeDispatch(
  dispatch: RangeDispatch,
  rangeIndex: number,
): RangeDispatch {
  if (
    !Number.isSafeInteger(rangeIndex) || rangeIndex < 0 ||
    rangeIndex >= dispatch.rangeCount
  ) {
    throw new RangeError(
      `${dispatch.label} selected range ${rangeIndex} is invalid`,
    );
  }
  return Object.freeze({
    label: `${dispatch.label}-selected-range-${rangeIndex}`,
    rangeCount: 1,
    encodeRange(pass: GPUComputePassEncoder, localRangeIndex: number): void {
      if (localRangeIndex !== 0) {
        throw new RangeError("OPT-0005 single-range dispatch only accepts 0");
      }
      dispatch.encodeRange(pass, rangeIndex);
    },
  });
}

interface OwnedPreparedDispatches {
  readonly scalar: RangeDispatch;
  readonly chunked: AceOpt0005VaeConv1dDispatch;
  readonly scalarKernel: AceCorrectnessVaePrimitiveKernel;
  readonly chunkedKernel: AceOpt0005VaeConv1dChannelChunksKernel;
  readonly controls: readonly GPUBuffer[];
  destroy(): void;
}

interface OwnedOverlapDispatches extends OwnedPreparedDispatches {
  readonly opt0004: AceOpt0004VaeConv1dDispatch;
  readonly opt0004Kernel: AceOpt0004VaeConv1dKernel;
}

interface RangeTiming {
  readonly rangeIndex: number;
  readonly encodeMilliseconds: number;
  readonly submitMilliseconds: number;
  readonly drainMilliseconds: number;
  readonly explicitIdleMilliseconds?: number;
}

interface ExecutionTiming {
  readonly roundIndex: number;
  readonly pairedOrder: string;
  readonly orderPosition: number;
  readonly wallMilliseconds: number;
  readonly activeWallMilliseconds: number;
  readonly encodeMilliseconds: number;
  readonly submitMilliseconds: number;
  readonly drainMilliseconds: number;
  readonly explicitIdleMilliseconds: number;
  readonly commandBufferCount: number;
  readonly maximumSingleDrainMilliseconds: number;
  readonly logicalGmacPerSecond: number;
  readonly ranges: readonly RangeTiming[];
}

interface HeartbeatResult {
  readonly animationFrameCount: number;
  readonly timerTickCount: number;
  readonly maximumAnimationFrameGapMilliseconds: number;
  readonly maximumTimerGapMilliseconds: number;
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

export interface Opt0005ThermalGateMetadata {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly durationSeconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
}

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const OUTPUT_SENTINEL_BITS = 0x7fc0_0000;
const EXPLICIT_IDLE_MILLISECONDS = 1;
const THERMAL_POLL_TOLERANCE_MILLISECONDS = 250;
const SCRATCH_FLOAT = new Float32Array(1);
const SCRATCH_BITS = new Uint32Array(SCRATCH_FLOAT.buffer);

export const OPT_0005_BLOCK0_D1_SHAPE = Object.freeze({
  batch: 1,
  inputFrames: 2_560,
  inputChannels: 1_024,
  outputChannels: 1_024,
  kernelSize: 7,
  stride: 1,
  dilation: 1,
  padding: 3,
} satisfies AceVaeConv1dShape);

export const OPT_0005_SCREEN_D1_SHAPE = Object.freeze({
  ...OPT_0005_BLOCK0_D1_SHAPE,
} satisfies AceVaeConv1dShape);

export const OPT_0005_SCREEN_D9_SHAPE = Object.freeze({
  ...OPT_0005_BLOCK0_D1_SHAPE,
  dilation: 9,
  padding: 27,
} satisfies AceVaeConv1dShape);

export const OPT_0005_OVERLAP_D1_SHAPE = Object.freeze({
  batch: 1,
  inputFrames: 2_048,
  inputChannels: 128,
  outputChannels: 128,
  kernelSize: 7,
  stride: 1,
  dilation: 1,
  padding: 3,
} satisfies AceVaeConv1dShape);

export const OPT_0005_PAIRED_ORDERS = Object.freeze([
  Object.freeze(["scalar", "chunked"]),
  Object.freeze(["chunked", "scalar"]),
  Object.freeze(["chunked", "scalar"]),
  Object.freeze(["scalar", "chunked"]),
] satisfies readonly (readonly PairedKernelId[])[]);

export const OPT_0005_OVERLAP_ORDERS = Object.freeze([
  Object.freeze(["scalar", "opt0004", "chunked"]),
  Object.freeze(["chunked", "opt0004", "scalar"]),
  Object.freeze(["chunked", "opt0004", "scalar"]),
  Object.freeze(["scalar", "opt0004", "chunked"]),
] satisfies readonly (readonly OverlapKernelId[])[]);

export const OPT_0005_CORRECTNESS_CASES = Object.freeze([
  Object.freeze({
    id: "dilation1-channel-tail-bias",
    hasBias: true,
    shape: Object.freeze({
      batch: 2,
      inputFrames: 35,
      inputChannels: 65,
      outputChannels: 13,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 3,
    }),
  }),
  Object.freeze({
    id: "dilation3-batch-padding-no-bias",
    hasBias: false,
    shape: Object.freeze({
      batch: 3,
      inputFrames: 51,
      inputChannels: 64,
      outputChannels: 11,
      kernelSize: 7,
      stride: 1,
      dilation: 3,
      padding: 9,
    }),
  }),
  Object.freeze({
    id: "dilation9-range-tail-bias",
    hasBias: true,
    shape: Object.freeze({
      batch: 2,
      inputFrames: 67,
      inputChannels: 63,
      outputChannels: 9,
      kernelSize: 7,
      stride: 1,
      dilation: 9,
      padding: 27,
    }),
  }),
  Object.freeze({
    id: "arithmetic-discriminants",
    hasBias: false,
    shape: Object.freeze({
      batch: 1,
      inputFrames: 19,
      inputChannels: 65,
      outputChannels: 9,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 3,
    }),
  }),
] satisfies readonly {
  readonly id: Opt0005CorrectnessCaseId;
  readonly hasBias: boolean;
  readonly shape: AceVaeConv1dShape;
}[]);

export const OPT_0005_ARITHMETIC_SENTINELS = Object.freeze({
  contractedOutputIndex: 8 * 9,
  cancellationOutputIndex: 13 * 9 + 1,
});

export function summarizeOpt0005Samples(
  samples: readonly number[],
): SampleSummary {
  if (samples.length === 0) throw new RangeError("samples must not be empty");
  if (samples.some((sample) => !Number.isFinite(sample) || sample < 0)) {
    throw new RangeError("samples must be finite non-negative numbers");
  }
  const retained = [...samples];
  const sorted = [...retained].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
  const minimum = sorted[0]!;
  const maximum = sorted.at(-1)!;
  return Object.freeze({
    count: retained.length,
    samples: Object.freeze(retained),
    minimum,
    median,
    maximum,
    range: maximum - minimum,
  });
}

export function parseOpt0005ThermalGateMetadata(
  parameters: URLSearchParams,
): Opt0005ThermalGateMetadata {
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
    throw new Error("OPT-0005 requires the accepted notifyutil thermal source");
  }
  if (durationSeconds < 30 || observationCount < 31) {
    throw new Error("OPT-0005 requires 30 continuous nominal seconds");
  }
  if (pollMilliseconds !== 1_000) {
    throw new Error("OPT-0005 thermal polling must use 1,000 ms intervals");
  }
  if (
    maximumPollGapMilliseconds >
      pollMilliseconds + THERMAL_POLL_TOLERANCE_MILLISECONDS
  ) {
    throw new Error("OPT-0005 thermal poll gap exceeds tolerance");
  }
  if (nonNominalObservationCount !== 0) {
    throw new Error("OPT-0005 thermal gate observed non-nominal pressure");
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

export function parseOpt0005ProductionMode(
  parameters: URLSearchParams,
): Opt0005RunMode {
  const raw = parameters.get("runMode");
  if (
    raw === "screen-d1" || raw === "block0-d1" || raw === "screen-d9" ||
    raw === "overlap-d1"
  ) return raw;
  throw new Error(
    "OPT-0005 requires explicit runMode=screen-d1|block0-d1|screen-d9|overlap-d1",
  );
}

export function opt0005InputValue(
  caseId: Opt0005CorrectnessCaseId,
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
    if (time === 11 && channel === 0) return 0.5;
    return channel % 2 === 0 ? 0 : -0;
  }
  if (index % 37 === 0) return -0;
  if (index % 41 === 0) return 0;
  return Math.fround((((index * 17 + 3) % 31) - 15) / 32);
}

export function opt0005WeightValue(
  caseId: Opt0005CorrectnessCaseId,
  index: number,
): number {
  const fixture = OPT_0005_CORRECTNESS_CASES.find(({ id }) => id === caseId);
  if (fixture === undefined) throw new Error(`Unknown OPT-0005 case ${caseId}`);
  const { inputChannels, kernelSize } = fixture.shape;
  if (caseId === "arithmetic-discriminants") {
    const channel = index % inputChannels;
    const kernel = Math.floor(index / inputChannels) % kernelSize;
    const outputChannel = Math.floor(index / (inputChannels * kernelSize));
    if (outputChannel === 0 && kernel === 0) {
      if (channel === 63) return 1;
      if (channel === 64) return 5.5625;
      return channel % 2 === 0 ? 0 : -0;
    }
    if (outputChannel === 1) {
      if (kernel === 0 && channel === 0) return 1;
      if (kernel === 0 && channel === 63) return 1;
      if (kernel === 0 && channel === 64) return 1;
      if (kernel === 1 && channel === 0) return 1;
      return 0;
    }
    return outputChannel % 2 === 0 ? 0 : -0;
  }
  if (index % 43 === 0) return -0;
  return Math.fround((((index * 13 + 7) % 29) - 14) / 64);
}

export function opt0005BiasValue(
  caseId: Opt0005CorrectnessCaseId,
  outputChannel: number,
): number {
  if (caseId === "dilation1-channel-tail-bias" && outputChannel === 0) {
    return -0;
  }
  return Math.fround(((outputChannel % 7) - 3) / 64);
}

export function opt0005OutputSentinelIndices(
  shape: AceVaeConv1dShape,
): readonly number[] {
  const plan = planAceOpt0005VaeConv1dChannelChunks(shape);
  const candidates = [
    0,
    Math.min(plan.outputElements - 1, plan.outputChannels - 1),
    Math.min(plan.outputElements - 1, plan.outputChannels),
    Math.min(plan.outputElements - 1, 15 * plan.outputChannels),
    Math.min(plan.outputElements - 1, 16 * plan.outputChannels),
    Math.floor(plan.outputElements / 2),
    plan.outputElements - 1,
  ];
  for (const range of plan.outputRanges) {
    candidates.push(
      range.firstOutput,
      Math.min(plan.outputElements - 1, range.firstOutput + range.outputCount - 1),
    );
  }
  return Object.freeze([...new Set(candidates)]);
}

export function opt0005ExpectedOutputValue(
  caseId: Opt0005CorrectnessCaseId,
  outputIndex: number,
): number {
  const fixture = OPT_0005_CORRECTNESS_CASES.find(({ id }) => id === caseId);
  if (fixture === undefined) throw new Error(`Unknown OPT-0005 case ${caseId}`);
  return expectedValue(
    fixture.shape,
    outputIndex,
    (index) => opt0005InputValue(caseId, index),
    (index) => opt0005WeightValue(caseId, index),
    fixture.hasBias
      ? (channel) => opt0005BiasValue(caseId, channel)
      : undefined,
  );
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
        schema: "ace-opt-0005-vae-conv1d-paired-ab-v1",
        status: "failed",
        experimentId: "OPT-0005",
        error: {
          name: error instanceof Error ? error.name : "Error",
          message: error instanceof Error ? error.message : String(error),
          ...(error instanceof Error && error.stack !== undefined
            ? { stack: error.stack }
            : {}),
        },
      }),
    );
  }, { once: true });
}

async function run(): Promise<unknown> {
  const parameters = new URL(window.location.href).searchParams;
  const thermalGate = parseOpt0005ThermalGateMetadata(parameters);
  const runMode = parseOpt0005ProductionMode(parameters);
  const measuredShape = measuredShapeFor(runMode);
  const measuredPlan = planAceOpt0005VaeConv1dChannelChunks(measuredShape);
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  assertAdapter(adapter, measuredPlan.workgroupStorageBytes, measuredShape);
  const largestBinding = largestStorageBindingBytes(measuredShape);
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: largestBinding,
      maxStorageBufferBindingSize: largestBinding,
      // The candidate uses WG(16,8), but the scalar authority uses WG(256,1).
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 8,
      maxComputeWorkgroupStorageSize: measuredPlan.workgroupStorageBytes,
    },
  });
  try {
    updateProgress("running complete bounded correctness graph");
    const correctnessPreflight = await runCorrectnessPreflight(device);
    updateProgress("running cancellation scheduling proof");
    const cancellation = await runCancellationProof(device);
    updateProgress(`preparing ${runMode} bounded measurement`);
    const production = await runMeasuredProductionOperation(
      device,
      measuredShape,
      runMode,
    );
    return Object.freeze({
      schema: "ace-opt-0005-vae-conv1d-paired-ab-v1",
      status: "passed",
      experimentId: "OPT-0005",
      classification: "benchmark-local-candidate-no-production-vae-change",
      recordedAt: new Date().toISOString(),
      browser: {
        userAgent: navigator.userAgent,
        page: window.location.href,
      },
      adapter: adapterIdentity(adapter),
      protocol: {
        thermalGate,
        runMode,
        baselineKernel: "AceCorrectnessVaePrimitiveKernel",
        candidateKernel: "AceOpt0005VaeConv1dChannelChunksKernel",
        modelProfile: "reference-bf16-subgroups-vae-fp32",
        compileAllocationAndUploadExcludedFromTiming: true,
        warmupExecutionsPerKernel: 1,
        samplesPerKernel: runMode === "overlap-d1"
          ? OPT_0005_OVERLAP_ORDERS.length
          : OPT_0005_PAIRED_ORDERS.length,
        pairedOrders: (runMode === "overlap-d1"
          ? OPT_0005_OVERLAP_ORDERS
          : OPT_0005_PAIRED_ORDERS).map((order) => order.join("-")),
        oneCommandBufferOutstanding: true,
        queueDrainAfterEveryCommandBuffer: true,
        queueEmptyIdleMillisecondsRequested: EXPLICIT_IDLE_MILLISECONDS,
        idleAfterFinalComputeRange: false,
        authoritativeTiming: "performance.now-fenced-wall-clock",
        candidateSourceMarkers: sourceMarkers(measuredShape),
      },
      memory: {
        inputBytes: measuredPlan.inputElements * FLOAT32_BYTES,
        weightBytes: measuredPlan.weightElements * FLOAT32_BYTES,
        biasBytes: measuredPlan.outputChannels * FLOAT32_BYTES,
        reusableOutputBytes: measuredPlan.outputElements * FLOAT32_BYTES,
        simultaneousFullOutputCopies: 1,
        candidateWorkgroupStorageBytes: measuredPlan.workgroupStorageBytes,
        candidateInputTileBytes: measuredPlan.inputTileBytes,
        candidateWeightTileBytes: measuredPlan.weightTileBytes,
      },
      measurementScope: {
        fullPlanRangeCount: measuredPlan.outputRangeCount,
        selectedRangeIndex: screenRangeIndex(
          runMode,
          measuredPlan.outputRangeCount,
        ) ?? null,
        effectiveCommandBufferCount: effectiveOpt0005RangeCount(
          runMode,
          measuredPlan.outputRangeCount,
        ),
      },
      correctnessPreflight,
      cancellation,
      production,
    });
  } finally {
    device.destroy();
  }
}

async function runCorrectnessPreflight(device: GPUDevice): Promise<unknown> {
  const results: unknown[] = [];
  for (const fixture of OPT_0005_CORRECTNESS_CASES) {
    const plan = planAceOpt0005VaeConv1dChannelChunks(fixture.shape);
    const inputValues = Float32Array.from(
      { length: plan.inputElements },
      (_, index) => opt0005InputValue(fixture.id, index),
    );
    const weightValues = Float32Array.from(
      { length: plan.weightElements },
      (_, index) => opt0005WeightValue(fixture.id, index),
    );
    const biasValues = fixture.hasBias
      ? Float32Array.from(
          { length: plan.outputChannels },
          (_, channel) => opt0005BiasValue(fixture.id, channel),
        )
      : undefined;
    const input = mappedStorage(device, `${fixture.id}-input`, inputValues);
    const weight = mappedStorage(device, `${fixture.id}-weight`, weightValues);
    const bias = biasValues === undefined
      ? undefined
      : mappedStorage(device, `${fixture.id}-bias`, biasValues);
    const scalarOutput = mappedSentinelOutput(
      device,
      `${fixture.id}-scalar-output`,
      plan.outputElements,
    );
    const chunkedOutput = mappedSentinelOutput(
      device,
      `${fixture.id}-chunked-output`,
      plan.outputElements,
    );
    let scalarPrepared: OwnedPreparedDispatches | undefined;
    let chunkedPrepared: OwnedPreparedDispatches | undefined;
    try {
      scalarPrepared = await prepareDispatches(device, fixture.shape, {
        input: binding(input),
        weight: binding(weight),
        ...(bias === undefined ? {} : { bias: binding(bias) }),
        output: binding(scalarOutput),
      });
      chunkedPrepared = await prepareDispatches(device, fixture.shape, {
        input: binding(input),
        weight: binding(weight),
        ...(bias === undefined ? {} : { bias: binding(bias) }),
        output: binding(chunkedOutput),
      });
      await executeUntimed(device, scalarPrepared.scalar);
      await executeUntimed(device, chunkedPrepared.chunked);
      const scalarValues = await readOutput(
        device,
        scalarOutput,
        0,
        plan.outputElements,
        `${fixture.id}-scalar`,
      );
      const chunkedValues = await readOutput(
        device,
        chunkedOutput,
        0,
        plan.outputElements,
        `${fixture.id}-chunked`,
      );
      const scalarBits = bitsOf(scalarValues);
      const chunkedBits = bitsOf(chunkedValues);
      let bitMismatchCount = 0;
      for (let index = 0; index < plan.outputElements; index += 1) {
        if (scalarBits[index] !== chunkedBits[index]) bitMismatchCount += 1;
      }
      const scalarFingerprint = fingerprint(scalarValues, scalarBits);
      const chunkedFingerprint = fingerprint(chunkedValues, chunkedBits);
      if (
        scalarFingerprint.finiteCount !== plan.outputElements ||
        chunkedFingerprint.finiteCount !== plan.outputElements ||
        bitMismatchCount !== 0
      ) {
        throw new Error(
          `${fixture.id} correctness failed: finite ` +
            `${scalarFingerprint.finiteCount}/${chunkedFingerprint.finiteCount}, ` +
            `A/B mismatches ${bitMismatchCount}`,
        );
      }
      const sentinelIndices = fixture.id === "arithmetic-discriminants"
        ? Object.freeze([...new Set([
            ...opt0005OutputSentinelIndices(fixture.shape),
            OPT_0005_ARITHMETIC_SENTINELS.contractedOutputIndex,
            OPT_0005_ARITHMETIC_SENTINELS.cancellationOutputIndex,
          ])])
        : opt0005OutputSentinelIndices(fixture.shape);
      const sentinels = sentinelIndices.map(
        (index) => {
          const expected = opt0005ExpectedOutputValue(fixture.id, index);
          const expectedBits = float32Bits(expected);
          const scalarBitExact = scalarBits[index] === expectedBits;
          const chunkedBitExact = chunkedBits[index] === expectedBits;
          if (!scalarBitExact || !chunkedBitExact) {
            throw new Error(
              `${fixture.id} CPU sentinel ${index} mismatch: ` +
                `${hex32(scalarBits[index]!)} / ${hex32(chunkedBits[index]!)} ` +
                `!= ${hex32(expectedBits)}`,
            );
          }
          return Object.freeze({
            index,
            expected,
            expectedBits: hex32(expectedBits),
            scalar: scalarValues[index],
            chunked: chunkedValues[index],
            scalarBitExact,
            chunkedBitExact,
          });
        },
      );
      results.push(Object.freeze({
        id: fixture.id,
        shape: fixture.shape,
        hasBias: fixture.hasBias,
        outputRangeCount: plan.outputRangeCount,
        rangeBatches: plan.outputRanges.map(({ batch }) => batch),
        frameTail: plan.outputFrames % plan.tileFrames,
        channelTail: plan.outputChannels % plan.tileChannels,
        outputPrefill: "quiet-NaN-u32-sentinel",
        fullDomainFinite: true,
        fullDomainBitIdentical: true,
        fullDomainBitMismatchCount: bitMismatchCount,
        scalarFingerprint,
        chunkedFingerprint,
        cpuSentinelsBitExact: true,
        sentinels: Object.freeze(sentinels),
      }));
    } finally {
      scalarPrepared?.destroy();
      chunkedPrepared?.destroy();
      input.destroy();
      weight.destroy();
      bias?.destroy();
      scalarOutput.destroy();
      chunkedOutput.destroy();
    }
  }
  return Object.freeze({
    classification: "complete-manageable-multirange-actual-gpu-correctness",
    cases: Object.freeze(results),
  });
}

async function runCancellationProof(device: GPUDevice): Promise<unknown> {
  const fixture = OPT_0005_CORRECTNESS_CASES[0]!;
  const plan = planAceOpt0005VaeConv1dChannelChunks(fixture.shape);
  if (plan.outputRangeCount < 2) {
    throw new Error("OPT-0005 cancellation fixture must have multiple ranges");
  }
  const input = mappedGeneratedStorage(
    device,
    "opt-0005-cancel-input",
    plan.inputElements,
    (index) => opt0005InputValue(fixture.id, index),
  );
  const weight = mappedGeneratedStorage(
    device,
    "opt-0005-cancel-weight",
    plan.weightElements,
    (index) => opt0005WeightValue(fixture.id, index),
  );
  const bias = mappedGeneratedStorage(
    device,
    "opt-0005-cancel-bias",
    plan.outputChannels,
    (channel) => opt0005BiasValue(fixture.id, channel),
  );
  const output = outputBuffer(
    device,
    "opt-0005-cancel-output",
    plan.outputElements,
  );
  const prepared = await prepareDispatches(device, fixture.shape, {
    input: binding(input),
    weight: binding(weight),
    bias: binding(bias),
    output: binding(output),
  });
  try {
    const controller = new AbortController();
    let rangesSubmitted = 0;
    let triggerDelivered = false;
    let activeRangeDrained = false;
    let explicitIdleMilliseconds = 0;
    const trigger = new Promise<void>((resolve) => {
      window.setTimeout(() => {
        triggerDelivered = true;
        controller.abort("opt-0005-cancellation-trigger");
        resolve();
      }, 0);
    });
    for (let rangeIndex = 0; rangeIndex < prepared.chunked.rangeCount; rangeIndex += 1) {
      if (controller.signal.aborted) break;
      const encoder = device.createCommandEncoder({
        label: `opt-0005-cancel-range-${rangeIndex}`,
      });
      const pass = encoder.beginComputePass();
      prepared.chunked.encodeRange(pass, rangeIndex);
      pass.end();
      device.queue.submit([encoder.finish()]);
      rangesSubmitted += 1;
      await device.queue.onSubmittedWorkDone();
      activeRangeDrained = true;
      explicitIdleMilliseconds = await queueEmptyIdle();
      await trigger;
    }
    if (
      !triggerDelivered || !controller.signal.aborted ||
      !activeRangeDrained || rangesSubmitted !== 1
    ) {
      throw new Error("OPT-0005 cancellation scheduler submitted after abort");
    }
    return Object.freeze({
      classification: "benchmark-scheduler-trigger-between-bounded-ranges",
      plannedRangeCount: prepared.chunked.rangeCount,
      rangesSubmitted,
      activeRangeDrained,
      explicitIdleMilliseconds,
      realQueueEmptyIntervalObserved: explicitIdleMilliseconds >= 1,
      triggerDelivered,
      signalAborted: controller.signal.aborted,
      laterRangeSubmissionPrevented: true,
    });
  } finally {
    prepared.destroy();
    input.destroy();
    weight.destroy();
    bias.destroy();
    output.destroy();
  }
}

async function runMeasuredProductionOperation(
  device: GPUDevice,
  shape: AceVaeConv1dShape,
  runMode: Opt0005RunMode,
): Promise<unknown> {
  const plan = planAceOpt0005VaeConv1dChannelChunks(shape);
  const selectedRangeIndex = screenRangeIndex(runMode, plan.outputRangeCount);
  const preparationStarted = performance.now();
  const input = mappedGeneratedStorage(
    device,
    `opt-0005-${runMode}-input`,
    plan.inputElements,
    opt0005ProductionInputValue,
  );
  const weight = mappedGeneratedStorage(
    device,
    `opt-0005-${runMode}-weight`,
    plan.weightElements,
    opt0005ProductionWeightValue,
  );
  const bias = mappedGeneratedStorage(
    device,
    `opt-0005-${runMode}-bias`,
    plan.outputChannels,
    opt0005ProductionBiasValue,
  );
  const output = outputBuffer(
    device,
    `opt-0005-${runMode}-reusable-output`,
    plan.outputElements,
  );
  const bindings: AceVaeConvBindings = {
    input: binding(input),
    weight: binding(weight),
    bias: binding(bias),
    output: binding(output),
  };
  const prepared = runMode === "overlap-d1"
    ? await prepareOverlapDispatches(device, shape, bindings)
    : await prepareDispatches(device, shape, bindings);
  const scalarDispatch = selectedRangeIndex === undefined
    ? prepared.scalar
    : singleRangeDispatch(prepared.scalar, selectedRangeIndex);
  const chunkedDispatch = selectedRangeIndex === undefined
    ? prepared.chunked
    : singleRangeDispatch(prepared.chunked, selectedRangeIndex);
  const preparationMilliseconds = performance.now() - preparationStarted;
  try {
    const selectedRanges = selectedRangeIndex === undefined
      ? selectedProductionRanges(plan.outputRanges)
      : Object.freeze([plan.outputRanges[selectedRangeIndex]!]);
    const validMultiplyAdds = selectedRangeIndex === undefined
      ? opt0005ValidMultiplyAdds(shape)
      : opt0005RangeValidMultiplyAdds(
          shape,
          plan.outputRanges[selectedRangeIndex]!,
        );
    updateProgress(`${runMode} scalar warmup and correctness slices`);
    const scalarWarmup = await executeDispatch(
      device,
      scalarDispatch,
      validMultiplyAdds,
      -1,
      "warmup",
      0,
    );
    await queueEmptyIdle();
    const scalarSlices = await readSelectedRanges(
      device,
      output,
      selectedRanges,
      "scalar",
    );
    prefillSelectedRanges(device, output, selectedRanges);
    await device.queue.onSubmittedWorkDone();
    updateProgress(`${runMode} chunked warmup and correctness slices`);
    const chunkedWarmup = await executeDispatch(
      device,
      chunkedDispatch,
      validMultiplyAdds,
      -1,
      "warmup",
      1,
    );
    await queueEmptyIdle();
    const chunkedSlices = await readSelectedRanges(
      device,
      output,
      selectedRanges,
      "chunked",
    );
    const correctness = validateProductionSlices(
      shape,
      selectedRanges,
      scalarSlices,
      chunkedSlices,
    );
    await queueEmptyIdle();
    await yieldToPage();

    const pairedSamples: Record<PairedKernelId, ExecutionTiming[]> = {
      scalar: [],
      chunked: [],
    };
    const overlapSamples: Record<OverlapKernelId, ExecutionTiming[]> = {
      scalar: [],
      opt0004: [],
      chunked: [],
    };
    let opt0004Warmup: ExecutionTiming | undefined;
    let opt0004Correctness: unknown;
    if (runMode === "overlap-d1") {
      const overlap = prepared as OwnedOverlapDispatches;
      prefillSelectedRanges(device, output, selectedRanges);
      await device.queue.onSubmittedWorkDone();
      updateProgress(`${runMode} OPT-0004 warmup and correctness slice`);
      opt0004Warmup = await executeDispatch(
        device,
        overlap.opt0004,
        validMultiplyAdds,
        -1,
        "warmup",
        2,
      );
      await queueEmptyIdle();
      const opt0004Slices = await readSelectedRanges(
        device,
        output,
        selectedRanges,
        "opt0004",
      );
      opt0004Correctness = validateProductionSlices(
        shape,
        selectedRanges,
        chunkedSlices,
        opt0004Slices,
      );
      await queueEmptyIdle();
    }
    const heartbeat = startHeartbeat();
    let measuredResponsiveness: HeartbeatResult;
    try {
      const orders = runMode === "overlap-d1"
        ? OPT_0005_OVERLAP_ORDERS
        : OPT_0005_PAIRED_ORDERS;
      for (const [roundIndex, order] of orders.entries()) {
        for (const [orderPosition, kernelId] of order.entries()) {
          updateProgress(
            `${runMode} round ${roundIndex + 1} ${kernelId} ` +
              `(${orderPosition + 1}/${order.length})`,
          );
          const dispatch = kernelId === "scalar"
            ? scalarDispatch
            : kernelId === "chunked"
            ? chunkedDispatch
            : (prepared as OwnedOverlapDispatches).opt0004;
          const sample = await executeDispatch(
            device,
            dispatch,
            validMultiplyAdds,
            roundIndex,
            order.join("-"),
            orderPosition,
          );
          if (runMode === "overlap-d1") {
            overlapSamples[kernelId as OverlapKernelId].push(sample);
          } else {
            pairedSamples[kernelId as PairedKernelId].push(sample);
          }
          await yieldToPage();
        }
      }
    } finally {
      measuredResponsiveness = heartbeat.stop();
    }
    const samples = runMode === "overlap-d1" ? overlapSamples : pairedSamples;
    const scalarSummary = summarizeExecutions(samples.scalar);
    const chunkedSummary = summarizeExecutions(samples.chunked);
    const rounds = (runMode === "overlap-d1"
      ? OPT_0005_OVERLAP_ORDERS
      : OPT_0005_PAIRED_ORDERS).map((_order, roundIndex) => {
      const scalar = samples.scalar[roundIndex]!;
      const chunked = samples.chunked[roundIndex]!;
      return Object.freeze({
        roundIndex,
        scalarActiveWallMilliseconds: scalar.activeWallMilliseconds,
        chunkedActiveWallMilliseconds: chunked.activeWallMilliseconds,
        chunkedWon: chunked.activeWallMilliseconds < scalar.activeWallMilliseconds,
      });
    });
    return Object.freeze({
      shape,
      plan: {
        outputFrames: plan.outputFrames,
        outputElements: plan.outputElements,
        outputRangeCount: plan.outputRangeCount,
        selectedRangeIndex: selectedRangeIndex ?? null,
        measuredCommandBufferCount: scalarDispatch.rangeCount,
        outputRowsPerRange: plan.outputRanges.map((range) => range.outputRowCount),
        validMultiplyAdds,
        conservativePlannedMultiplyAdds: plan.outputRanges.reduce(
          (total, range) => total + range.multiplyAdds,
          0,
        ),
        workgroupStorageBytes: plan.workgroupStorageBytes,
      },
      preparationMilliseconds,
      warmup: {
        scalar: scalarWarmup,
        chunked: chunkedWarmup,
        ...(opt0004Warmup === undefined ? {} : { opt0004: opt0004Warmup }),
      },
      correctness,
      ...(opt0004Correctness === undefined
        ? {}
        : { opt0004Correctness }),
      scalar: {
        samples: Object.freeze(samples.scalar),
        summary: scalarSummary,
      },
      chunked: {
        samples: Object.freeze(samples.chunked),
        summary: chunkedSummary,
      },
      ...(runMode === "overlap-d1"
        ? {
            opt0004: {
              samples: Object.freeze(overlapSamples.opt0004),
              summary: summarizeExecutions(overlapSamples.opt0004),
            },
          }
        : {}),
      rounds: Object.freeze(rounds),
      delta: {
        medianActiveWallSpeedup:
          scalarSummary.activeWallMilliseconds.median /
          chunkedSummary.activeWallMilliseconds.median,
        medianLogicalGmacPerSecondRatio:
          chunkedSummary.logicalGmacPerSecond.median /
          scalarSummary.logicalGmacPerSecond.median,
        chunkedRoundWins: rounds.filter((round) => round.chunkedWon).length,
      },
      measuredResponsiveness,
    });
  } finally {
    prepared.destroy();
    input.destroy();
    weight.destroy();
    bias.destroy();
    output.destroy();
  }
}

async function prepareDispatches(
  device: GPUDevice,
  shape: AceVaeConv1dShape,
  bindings: AceVaeConvBindings,
): Promise<OwnedPreparedDispatches> {
  const scalarKernel = AceCorrectnessVaePrimitiveKernel.create(device);
  const chunkedKernel = AceOpt0005VaeConv1dChannelChunksKernel.create(device);
  const controls: GPUBuffer[] = [];
  try {
    const chunked = await chunkedKernel.createDispatch(
      "opt-0005-chunked",
      shape,
      bindings,
    );
    const scalarDispatches: AceVaeDispatch<AceVaeConvPlan>[] = [];
    for (const [index, range] of chunked.plan.outputRanges.entries()) {
      const control = rangeControlBuffer(
        device,
        `opt-0005-scalar-range-${index}`,
        range.firstOutput,
        range.outputCount,
      );
      controls.push(control);
      scalarDispatches.push(await scalarKernel.createConv1dDispatch(
        `opt-0005-scalar-range-${index}`,
        shape,
        bindings,
        {
          base: range.firstOutput,
          count: range.outputCount,
          control: { buffer: control, offset: 0, size: 16 },
        },
      ));
    }
    const scalar: RangeDispatch = Object.freeze({
      label: "opt-0005-scalar",
      rangeCount: scalarDispatches.length,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        const dispatch = scalarDispatches[rangeIndex];
        if (dispatch === undefined) {
          throw new RangeError(`OPT-0005 scalar range ${rangeIndex} is invalid`);
        }
        dispatch.encode(pass);
      },
    });
    return Object.freeze({
      scalar,
      chunked,
      scalarKernel,
      chunkedKernel,
      controls: Object.freeze(controls),
      destroy(): void {
        scalarKernel.destroy();
        chunkedKernel.destroy();
        for (const control of controls) control.destroy();
      },
    });
  } catch (error) {
    scalarKernel.destroy();
    chunkedKernel.destroy();
    for (const control of controls) control.destroy();
    throw error;
  }
}

async function prepareOverlapDispatches(
  device: GPUDevice,
  shape: AceVaeConv1dShape,
  bindings: AceVaeConvBindings,
): Promise<OwnedOverlapDispatches> {
  planAceOpt0004VaeConv1d(shape);
  const base = await prepareDispatches(device, shape, bindings);
  const opt0004Kernel = AceOpt0004VaeConv1dKernel.create(device);
  try {
    const opt0004 = await opt0004Kernel.createDispatch(
      "opt-0005-overlap-opt0004",
      shape,
      bindings,
    );
    if (opt0004.rangeCount !== base.chunked.rangeCount) {
      throw new Error("OPT-0005 overlap range topology differs from OPT-0004");
    }
    return Object.freeze({
      ...base,
      opt0004,
      opt0004Kernel,
      destroy(): void {
        base.destroy();
        opt0004Kernel.destroy();
      },
    });
  } catch (error) {
    base.destroy();
    opt0004Kernel.destroy();
    throw error;
  }
}

async function executeUntimed(
  device: GPUDevice,
  dispatch: RangeDispatch,
): Promise<void> {
  for (let rangeIndex = 0; rangeIndex < dispatch.rangeCount; rangeIndex += 1) {
    const encoder = device.createCommandEncoder({
      label: `${dispatch.label}-untimed-${rangeIndex}`,
    });
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
  let explicitIdleMilliseconds = 0;
  const wallStarted = performance.now();
  for (let rangeIndex = 0; rangeIndex < dispatch.rangeCount; rangeIndex += 1) {
    const encodeStarted = performance.now();
    const encoder = device.createCommandEncoder({
      label: `${dispatch.label}-range-${rangeIndex}-encoder`,
    });
    const pass = encoder.beginComputePass({
      label: `${dispatch.label}-range-${rangeIndex}-pass`,
    });
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
    let idle: number | undefined;
    if (rangeIndex + 1 < dispatch.rangeCount) {
      idle = await queueEmptyIdle();
      explicitIdleMilliseconds += idle;
    }
    encodeMilliseconds += encoded;
    submitMilliseconds += submitted;
    drainMilliseconds += drained;
    ranges.push(Object.freeze({
      rangeIndex,
      encodeMilliseconds: encoded,
      submitMilliseconds: submitted,
      drainMilliseconds: drained,
      ...(idle === undefined ? {} : { explicitIdleMilliseconds: idle }),
    }));
  }
  const wallMilliseconds = performance.now() - wallStarted;
  const activeWallMilliseconds = Math.max(
    Number.EPSILON,
    wallMilliseconds - explicitIdleMilliseconds,
  );
  return Object.freeze({
    roundIndex,
    pairedOrder,
    orderPosition,
    wallMilliseconds,
    activeWallMilliseconds,
    encodeMilliseconds,
    submitMilliseconds,
    drainMilliseconds,
    explicitIdleMilliseconds,
    commandBufferCount: dispatch.rangeCount,
    maximumSingleDrainMilliseconds: Math.max(
      0,
      ...ranges.map((range) => range.drainMilliseconds),
    ),
    logicalGmacPerSecond: logicalMultiplyAdds / activeWallMilliseconds / 1e6,
    ranges: Object.freeze(ranges),
  });
}

function summarizeExecutions(samples: readonly ExecutionTiming[]): {
  readonly activeWallMilliseconds: SampleSummary;
  readonly wallMilliseconds: SampleSummary;
  readonly encodeMilliseconds: SampleSummary;
  readonly submitMilliseconds: SampleSummary;
  readonly drainMilliseconds: SampleSummary;
  readonly explicitIdleMilliseconds: SampleSummary;
  readonly maximumSingleDrainMilliseconds: SampleSummary;
  readonly logicalGmacPerSecond: SampleSummary;
} {
  if (samples.length === 0) throw new RangeError("execution samples missing");
  const summarize = (select: (sample: ExecutionTiming) => number) =>
    summarizeOpt0005Samples(samples.map(select));
  return Object.freeze({
    activeWallMilliseconds: summarize((sample) => sample.activeWallMilliseconds),
    wallMilliseconds: summarize((sample) => sample.wallMilliseconds),
    encodeMilliseconds: summarize((sample) => sample.encodeMilliseconds),
    submitMilliseconds: summarize((sample) => sample.submitMilliseconds),
    drainMilliseconds: summarize((sample) => sample.drainMilliseconds),
    explicitIdleMilliseconds: summarize((sample) =>
      sample.explicitIdleMilliseconds
    ),
    maximumSingleDrainMilliseconds: summarize((sample) =>
      sample.maximumSingleDrainMilliseconds
    ),
    logicalGmacPerSecond: summarize((sample) => sample.logicalGmacPerSecond),
  });
}

function selectedProductionRanges(
  ranges: readonly AceOpt0005VaeConv1dOutputRange[],
): readonly AceOpt0005VaeConv1dOutputRange[] {
  if (ranges.length === 0) throw new Error("OPT-0005 production ranges missing");
  return Object.freeze([...new Set([
    ranges[0]!,
    ranges[Math.floor(ranges.length / 2)]!,
    ranges.at(-1)!,
  ])]);
}

async function readSelectedRanges(
  device: GPUDevice,
  output: GPUBuffer,
  ranges: readonly AceOpt0005VaeConv1dOutputRange[],
  kernelId: OverlapKernelId,
): Promise<readonly Float32Array[]> {
  const slices: Float32Array[] = [];
  for (const [index, range] of ranges.entries()) {
    slices.push(await readOutput(
      device,
      output,
      range.firstOutput,
      range.outputCount,
      `opt-0005-${kernelId}-slice-${index}`,
    ));
  }
  return Object.freeze(slices);
}

function validateProductionSlices(
  shape: AceVaeConv1dShape,
  ranges: readonly AceOpt0005VaeConv1dOutputRange[],
  scalarSlices: readonly Float32Array[],
  candidateSlices: readonly Float32Array[],
): unknown {
  let bitMismatchCount = 0;
  let comparedElementCount = 0;
  const sliceResults = ranges.map((range, sliceIndex) => {
    const scalar = scalarSlices[sliceIndex]!;
    const candidate = candidateSlices[sliceIndex]!;
    const scalarBits = bitsOf(scalar);
    const candidateBits = bitsOf(candidate);
    let sliceMismatches = 0;
    for (let index = 0; index < range.outputCount; index += 1) {
      if (scalarBits[index] !== candidateBits[index]) sliceMismatches += 1;
    }
    bitMismatchCount += sliceMismatches;
    comparedElementCount += range.outputCount;
    const scalarFingerprint = fingerprint(scalar, scalarBits);
    const candidateFingerprint = fingerprint(candidate, candidateBits);
    if (
      scalarFingerprint.finiteCount !== range.outputCount ||
      candidateFingerprint.finiteCount !== range.outputCount ||
      scalarFingerprint.nonzeroCount === 0 ||
      candidateFingerprint.nonzeroCount === 0
    ) {
      throw new Error(
        `OPT-0005 production slice ${sliceIndex} was not completely ` +
          "overwritten with finite, nondegenerate output",
      );
    }
    const sentinelLocalIndices = Object.freeze([...new Set([
      0,
      Math.min(range.outputCount - 1, shape.outputChannels - 1),
      Math.floor(range.outputCount / 2),
      range.outputCount - 1,
    ])]);
    const sentinels = sentinelLocalIndices.map((localIndex) => {
      const outputIndex = range.firstOutput + localIndex;
      const expected = expectedValue(
        shape,
        outputIndex,
        opt0005ProductionInputValue,
        opt0005ProductionWeightValue,
        opt0005ProductionBiasValue,
      );
      const expectedBits = float32Bits(expected);
      if (
        scalarBits[localIndex] !== expectedBits ||
        candidateBits[localIndex] !== expectedBits
      ) {
        throw new Error(
          `OPT-0005 production CPU sentinel ${outputIndex} mismatch`,
        );
      }
      return Object.freeze({
        outputIndex,
        expected,
        expectedBits: hex32(expectedBits),
        scalarBitExact: true,
        candidateBitExact: true,
      });
    });
    return Object.freeze({
      firstOutput: range.firstOutput,
      outputCount: range.outputCount,
      bitMismatchCount: sliceMismatches,
      scalarFingerprint,
      candidateFingerprint,
      sentinels: Object.freeze(sentinels),
    });
  });
  if (bitMismatchCount !== 0) {
    throw new Error(
      `OPT-0005 production slices have ${bitMismatchCount} bit mismatches`,
    );
  }
  return Object.freeze({
    classification: "first-middle-last-complete-range-slices",
    comparedElementCount,
    bitIdentical: true,
    bitMismatchCount,
    cpuSentinelsBitExact: true,
    slices: Object.freeze(sliceResults),
  });
}

export function opt0005ProductionInputValue(index: number): number {
  return Math.fround((((index * 17 + 3) % 31) - 15) / 32);
}

export function opt0005ProductionWeightValue(index: number): number {
  return Math.fround((((index * 13 + 7) % 29) - 14) / 64);
}

export function opt0005ProductionBiasValue(outputChannel: number): number {
  return Math.fround(((outputChannel % 11) - 5) / 128);
}

export function opt0005ValidMultiplyAdds(shape: AceVaeConv1dShape): number {
  const plan = planAceVaeConv1d(shape);
  let validKernelVisits = 0;
  for (let outputTime = 0; outputTime < plan.outputFrames; outputTime += 1) {
    for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
      const paddedTime = outputTime * shape.stride + kernel * shape.dilation;
      if (paddedTime < shape.padding) continue;
      const inputTime = paddedTime - shape.padding;
      if (inputTime < shape.inputFrames) validKernelVisits += 1;
    }
  }
  return validKernelVisits * shape.batch * shape.inputChannels *
    shape.outputChannels;
}

export function opt0005RangeValidMultiplyAdds(
  shape: AceVaeConv1dShape,
  range: Pick<
    AceOpt0005VaeConv1dOutputRange,
    "batch" | "firstOutputTime" | "outputRowCount"
  >,
): number {
  const plan = planAceVaeConv1d(shape);
  if (
    range.batch < 0 || range.batch >= shape.batch ||
    range.firstOutputTime < 0 || range.outputRowCount <= 0 ||
    range.firstOutputTime + range.outputRowCount > plan.outputFrames
  ) {
    throw new RangeError("OPT-0005 valid-MAC range is outside the shape");
  }
  let validKernelVisits = 0;
  const end = range.firstOutputTime + range.outputRowCount;
  for (let outputTime = range.firstOutputTime; outputTime < end; outputTime += 1) {
    for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
      const paddedTime = outputTime * shape.stride + kernel * shape.dilation;
      if (paddedTime < shape.padding) continue;
      const inputTime = paddedTime - shape.padding;
      if (inputTime < shape.inputFrames) validKernelVisits += 1;
    }
  }
  return validKernelVisits * shape.inputChannels * shape.outputChannels;
}

function expectedValue(
  shape: AceVaeConv1dShape,
  outputIndex: number,
  inputValue: (index: number) => number,
  weightValue: (index: number) => number,
  biasValue?: (outputChannel: number) => number,
): number {
  const plan = planAceVaeConv1d(shape);
  if (
    !Number.isSafeInteger(outputIndex) || outputIndex < 0 ||
    outputIndex >= plan.outputElements
  ) {
    throw new RangeError("OPT-0005 expected output index is outside the plan");
  }
  const outputChannel = outputIndex % shape.outputChannels;
  const row = Math.floor(outputIndex / shape.outputChannels);
  const outputTime = row % plan.outputFrames;
  const batch = Math.floor(row / plan.outputFrames);
  let sum = biasValue === undefined ? 0 : biasValue(outputChannel);
  for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
    const paddedTime = outputTime * shape.stride + kernel * shape.dilation;
    if (paddedTime < shape.padding) continue;
    const inputTime = paddedTime - shape.padding;
    if (inputTime >= shape.inputFrames) continue;
    const inputBase = (batch * shape.inputFrames + inputTime) *
      shape.inputChannels;
    const weightBase = (outputChannel * shape.kernelSize + kernel) *
      shape.inputChannels;
    for (let inputChannel = 0; inputChannel < shape.inputChannels; inputChannel += 1) {
      sum = Math.fround(
        sum + inputValue(inputBase + inputChannel) *
          weightValue(weightBase + inputChannel),
      );
    }
  }
  return sum;
}

function measuredShapeFor(
  mode: Opt0005RunMode,
): AceVaeConv1dShape {
  switch (mode) {
    case "screen-d1": return OPT_0005_SCREEN_D1_SHAPE;
    case "block0-d1": return OPT_0005_BLOCK0_D1_SHAPE;
    case "screen-d9": return OPT_0005_SCREEN_D9_SHAPE;
    case "overlap-d1": return OPT_0005_OVERLAP_D1_SHAPE;
  }
}

export function screenRangeIndex(
  mode: Opt0005RunMode,
  rangeCount: number,
): number | undefined {
  if (!Number.isSafeInteger(rangeCount) || rangeCount <= 0) {
    throw new RangeError("OPT-0005 range count must be a positive integer");
  }
  return mode === "screen-d1" || mode === "screen-d9"
    ? Math.floor(rangeCount / 2)
    : undefined;
}

function mappedStorage(
  device: GPUDevice,
  label: string,
  values: Float32Array,
): GPUBuffer {
  return mappedGeneratedStorage(
    device,
    label,
    values.length,
    (index) => values[index]!,
  );
}

function mappedGeneratedStorage(
  device: GPUDevice,
  label: string,
  elements: number,
  value: (index: number) => number,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: elements * FLOAT32_BYTES,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    const mapped = new Float32Array(buffer.getMappedRange());
    for (let index = 0; index < elements; index += 1) {
      mapped[index] = value(index);
    }
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    buffer.destroy();
    throw error;
  }
}

function mappedSentinelOutput(
  device: GPUDevice,
  label: string,
  elements: number,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: elements * FLOAT32_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
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

function outputBuffer(
  device: GPUDevice,
  label: string,
  elements: number,
): GPUBuffer {
  return device.createBuffer({
    label,
    size: elements * FLOAT32_BYTES,
    usage:
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

function prefillSelectedRanges(
  device: GPUDevice,
  output: GPUBuffer,
  ranges: readonly AceOpt0005VaeConv1dOutputRange[],
): void {
  const maximumElements = Math.max(...ranges.map((range) => range.outputCount));
  const sentinel = new Uint32Array(maximumElements);
  sentinel.fill(OUTPUT_SENTINEL_BITS);
  for (const range of ranges) {
    device.queue.writeBuffer(
      output,
      range.firstOutput * FLOAT32_BYTES,
      sentinel,
      0,
      range.outputCount,
    );
  }
}

function rangeControlBuffer(
  device: GPUDevice,
  label: string,
  base: number,
  count: number,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: 16,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
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

async function readOutput(
  device: GPUDevice,
  output: GPUBuffer,
  firstElement: number,
  elementCount: number,
  label: string,
): Promise<Float32Array<ArrayBuffer>> {
  const bytes = elementCount * FLOAT32_BYTES;
  const readback = device.createBuffer({
    label: `${label}-readback`,
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder({ label: `${label}-copy` });
    encoder.copyBufferToBuffer(
      output,
      firstElement * FLOAT32_BYTES,
      readback,
      0,
      bytes,
    );
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

function bitsOf(values: Float32Array): Uint32Array {
  return new Uint32Array(values.buffer, values.byteOffset, values.length);
}

function fingerprint(
  values: Float32Array,
  bits: Uint32Array,
): OutputFingerprint {
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
  return Object.freeze({
    elementCount: values.length,
    finiteCount,
    nonzeroCount,
    positiveZeroCount,
    negativeZeroCount,
    fnv1a32: hash.toString(16).padStart(8, "0"),
  });
}

function sourceMarkers(shape: AceVaeConv1dShape): unknown {
  const source = aceOpt0005VaeConv1dChannelChunksWgsl(shape, true);
  const markers = Object.freeze({
    workgroup16x8: source.includes("@workgroup_size(\n  16,\n  8,\n  1,"),
    sharedInput: source.includes("var<workgroup> input_tile"),
    sharedWeight: source.includes("var<workgroup> weight_tile"),
    inputChannelChunk64: source.includes(
      "input_channel_chunk * 64u",
    ),
    paddedInputStride: source.includes("chunk_channel * 17u"),
    paddedWeightStride: source.includes(
      "tile_output_channel * 65u",
    ),
    realChannelTail: source.includes("let chunk_channel_count = min("),
    kernelBeforeChunk: source.indexOf("var kernel = 0u") <
      source.indexOf("var input_channel_chunk = 0u"),
    twoUniformBarriersPerChunk:
      source.match(/workgroupBarrier\(\)/g)?.length === 2,
    noExplicitFma: !source.includes("fma("),
    fp32Only: source.includes("array<f32>") && !source.includes("enable f16"),
  });
  if (Object.values(markers).some((value) => !value)) {
    throw new Error("OPT-0005 candidate shader source markers changed");
  }
  return markers;
}

function assertAdapter(
  adapter: GPUAdapter,
  workgroupStorageBytes: number,
  shape: AceVaeConv1dShape,
): void {
  const largestBinding = largestStorageBindingBytes(shape);
  if (
    adapter.limits.maxBufferSize < largestBinding ||
    adapter.limits.maxStorageBufferBindingSize < largestBinding ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 256 ||
    adapter.limits.maxComputeWorkgroupSizeX < 256 ||
    adapter.limits.maxComputeWorkgroupSizeY < 8 ||
    adapter.limits.maxComputeWorkgroupStorageSize < workgroupStorageBytes
  ) {
    throw new Error("Adapter cannot satisfy the OPT-0005 paired Conv1D contract");
  }
}

function largestStorageBindingBytes(shape: AceVaeConv1dShape): number {
  const plan = planAceOpt0005VaeConv1dChannelChunks(shape);
  return Math.max(
    plan.inputElements * FLOAT32_BYTES,
    plan.weightElements * FLOAT32_BYTES,
    plan.outputElements * FLOAT32_BYTES,
    plan.outputChannels * FLOAT32_BYTES,
  );
}

function adapterIdentity(adapter: GPUAdapter): unknown {
  const { info, limits } = adapter;
  return Object.freeze({
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
    isFallbackAdapter: info.isFallbackAdapter,
    features: Object.freeze([...adapter.features].sort()),
    limits: Object.freeze({
      maxBufferSize: limits.maxBufferSize,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup:
        limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
      maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
    }),
  });
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function requiredNumber(parameters: URLSearchParams, name: string): number {
  const raw = parameters.get(name);
  if (raw === null || raw.length === 0) {
    throw new Error(`Missing OPT-0005 ${name}`);
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`Invalid OPT-0005 ${name}`);
  }
  return value;
}

function float32Bits(value: number): number {
  SCRATCH_FLOAT[0] = value;
  return SCRATCH_BITS[0]!;
}

function hex32(value: number): string {
  return `0x${(value >>> 0).toString(16).padStart(8, "0")}`;
}

function startHeartbeat(): { stop(): HeartbeatResult } {
  const animationGaps: number[] = [];
  const timerGaps: number[] = [];
  let stopped = false;
  let animationFrameCount = 0;
  let timerTickCount = 0;
  let lastAnimation = performance.now();
  let lastTimer = lastAnimation;
  let frameHandle = 0;
  const frame = (now: number): void => {
    if (stopped) return;
    animationGaps.push(now - lastAnimation);
    lastAnimation = now;
    animationFrameCount += 1;
    frameHandle = requestAnimationFrame(frame);
  };
  frameHandle = requestAnimationFrame(frame);
  const timerHandle = window.setInterval(() => {
    const now = performance.now();
    timerGaps.push(now - lastTimer);
    lastTimer = now;
    timerTickCount += 1;
  }, 10);
  return {
    stop(): HeartbeatResult {
      if (!stopped) {
        stopped = true;
        cancelAnimationFrame(frameHandle);
        clearInterval(timerHandle);
      }
      return Object.freeze({
        animationFrameCount,
        timerTickCount,
        maximumAnimationFrameGapMilliseconds: Math.max(0, ...animationGaps),
        maximumTimerGapMilliseconds: Math.max(0, ...timerGaps),
      });
    },
  };
}

async function queueEmptyIdle(): Promise<number> {
  const started = performance.now();
  await new Promise<void>((resolve) => {
    window.setTimeout(resolve, EXPLICIT_IDLE_MILLISECONDS);
  });
  return performance.now() - started;
}

async function yieldToPage(): Promise<void> {
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
}

function updateProgress(message: string): void {
  const node = document.querySelector<HTMLElement>("#progress");
  if (node === null) throw new Error("Missing progress node");
  node.textContent = message;
}

function finish(status: "passed" | "failed", result: unknown): void {
  document.body.dataset.status = status;
  updateProgress(status);
  const node = document.querySelector<HTMLPreElement>("#result");
  if (node === null) throw new Error("Missing result node");
  node.textContent = JSON.stringify(result);
}
