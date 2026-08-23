import {
  ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES,
  planAceOpt0006QuantumBatches,
  runAceOpt0006QuantumBatches,
  type AceOpt0006BatchProgress,
  type AceOpt0006EncodableQuantum,
} from "../../benchmark/opt-0006-vae-command-buffer-coalescing.js";
import {
  AceChannelChunkedVaeConv1dKernel,
  type AceChannelChunkedVaeConv1dDispatch,
} from "../../src/webgpu/kernels/vae-conv1d-channel-chunks.js";
import { planAceOpt0005VaeConv1dChannelChunks } from
  "../../benchmark/opt-0005-vae-conv1d.js";
import type { AceVaeConv1dShape } from
  "../../src/webgpu/kernels/vae-primitives.js";

type BatchSize =
  typeof ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES[number];

interface HeartbeatSnapshot {
  readonly animationFrameCount: number;
  readonly timerTickCount: number;
  readonly maximumAnimationFrameGapMilliseconds: number;
  readonly maximumTimerGapMilliseconds: number;
}

interface ExecutionCounters {
  passCount: number;
  dispatchCount: number;
  commandEncoderCount: number;
}

interface BatchDrain {
  readonly batchIndex: number;
  readonly drainMilliseconds: number;
  readonly heartbeat: HeartbeatSnapshot;
}

interface ExecutionResult {
  readonly batchSize: BatchSize;
  readonly wallMilliseconds: number;
  readonly sumBatchDrainMilliseconds: number;
  readonly completedQuanta: number;
  readonly passCount: number;
  readonly dispatchCount: number;
  readonly commandBufferCount: number;
  readonly queueDrains: number;
  readonly progressEventCount: number;
  readonly explicitIdleCount: number;
  readonly explicitIdleMilliseconds: number;
  readonly maximumOutstandingCommandBuffers: number;
  readonly batchDrains: readonly BatchDrain[];
  readonly progress: readonly AceOpt0006BatchProgress[];
  readonly heartbeat: HeartbeatSnapshot;
}

type PreparedRangeDispatches = readonly AceChannelChunkedVaeConv1dDispatch[];

interface SampleSummary {
  readonly count: number;
  readonly samples: readonly number[];
  readonly minimum: number;
  readonly median: number;
  readonly maximum: number;
}

export interface Opt0006ThermalGateMetadata {
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

export const OPT_0006_PRODUCTION_RANGE_SHAPE = Object.freeze({
  batch: 1,
  inputFrames: 2_560,
  inputChannels: 1_024,
  outputChannels: 1_024,
  kernelSize: 7,
  stride: 1,
  dilation: 1,
  padding: 3,
} satisfies AceVaeConv1dShape);

export const OPT_0006_PRODUCTION_RANGE_INDICES = Object.freeze(
  Array.from({ length: 16 }, (_, index) => index + 32),
);

const FORWARD_BATCH_ORDER = ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES;
const REVERSE_BATCH_ORDER = Object.freeze([...FORWARD_BATCH_ORDER].reverse());

export const OPT_0006_BALANCED_BATCH_ORDERS = Object.freeze([
  FORWARD_BATCH_ORDER,
  REVERSE_BATCH_ORDER,
  REVERSE_BATCH_ORDER,
  FORWARD_BATCH_ORDER,
] satisfies readonly (readonly BatchSize[])[]);

export function expectedOpt0006SchedulingCounts(batchSize: number): Readonly<{
  quantumCount: 16;
  passCount: 16;
  dispatchCount: 16;
  commandBufferCount: number;
  queueDrains: number;
  progressEventCount: 16;
  explicitIdleCount: number;
}> {
  const commandBufferCount = planAceOpt0006QuantumBatches(
    OPT_0006_PRODUCTION_RANGE_INDICES.length,
    batchSize,
  ).length;
  return Object.freeze({
    quantumCount: 16,
    passCount: 16,
    dispatchCount: 16,
    commandBufferCount,
    queueDrains: commandBufferCount,
    progressEventCount: 16,
    // A readback remains after decoder compute, including the final batch.
    explicitIdleCount: commandBufferCount,
  });
}

export function summarizeOpt0006Samples(
  samples: readonly number[],
): SampleSummary {
  if (
    samples.length === 0 ||
    samples.some((sample) => !Number.isFinite(sample) || sample < 0)
  ) {
    throw new RangeError("OPT-0006 samples must be finite and non-negative");
  }
  const retained = [...samples];
  const sorted = [...retained].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return Object.freeze({
    count: retained.length,
    samples: Object.freeze(retained),
    minimum: sorted[0]!,
    median: sorted.length % 2 === 0
      ? (sorted[middle - 1]! + sorted[middle]!) / 2
      : sorted[middle]!,
    maximum: sorted.at(-1)!,
  });
}

export function parseOpt0006ThermalGateMetadata(
  parameters: URLSearchParams,
): Opt0006ThermalGateMetadata {
  if (parameters.get("runMode") !== "production-ranges") {
    throw new Error("OPT-0006 requires explicit runMode=production-ranges");
  }
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
    throw new Error("OPT-0006 requires the accepted notifyutil thermal source");
  }
  if (durationSeconds < 30 || observationCount < 31) {
    throw new Error("OPT-0006 requires 30 continuous nominal seconds");
  }
  if (pollMilliseconds !== 1_000) {
    throw new Error("OPT-0006 thermal polling must use 1,000 ms intervals");
  }
  if (
    maximumPollGapMilliseconds >
      pollMilliseconds + THERMAL_POLL_TOLERANCE_MILLISECONDS
  ) {
    throw new Error("OPT-0006 thermal poll gap exceeds tolerance");
  }
  if (nonNominalObservationCount !== 0) {
    throw new Error("OPT-0006 thermal gate observed non-nominal pressure");
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
        schema: "ace-opt-0006-vae-command-buffer-coalescing-production-ab-v1",
        status: "failed",
        experimentId: "OPT-0006",
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
  const thermalGate = parseOpt0006ThermalGateMetadata(
    new URL(window.location.href).searchParams,
  );
  const plan = planAceOpt0005VaeConv1dChannelChunks(
    OPT_0006_PRODUCTION_RANGE_SHAPE,
  );
  validateSelectedRanges(plan);
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  const largestBinding = Math.max(
    plan.inputElements,
    plan.weightElements,
    plan.outputElements,
  ) * FLOAT32_BYTES;
  assertAdapter(adapter, largestBinding, plan.workgroupStorageBytes);
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: largestBinding,
      maxStorageBufferBindingSize: largestBinding,
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 16,
      maxComputeWorkgroupSizeY: 8,
      maxComputeWorkgroupStorageSize: plan.workgroupStorageBytes,
    },
  });
  try {
    updateProgress("allocating, uploading, and compiling outside timing");
    return await runPreparedProbe(device, adapter, thermalGate);
  } finally {
    device.destroy();
  }
}

async function runPreparedProbe(
  device: GPUDevice,
  adapter: GPUAdapter,
  thermalGate: Opt0006ThermalGateMetadata,
): Promise<unknown> {
  const plan = planAceOpt0005VaeConv1dChannelChunks(
    OPT_0006_PRODUCTION_RANGE_SHAPE,
  );
  let input: GPUBuffer | undefined;
  let weight: GPUBuffer | undefined;
  let bias: GPUBuffer | undefined;
  let rangeControl: GPUBuffer | undefined;
  let kernel: AceChannelChunkedVaeConv1dKernel | undefined;
  const outputs = new Map<BatchSize, GPUBuffer>();
  const dispatches = new Map<BatchSize, PreparedRangeDispatches>();
  try {
    const stableInput = mappedGeneratedStorage(
      device,
      "opt-0006-production-input",
      plan.inputElements,
      productionInputValue,
    );
    input = stableInput;
    const stableWeight = mappedGeneratedStorage(
      device,
      "opt-0006-production-weight",
      plan.weightElements,
      productionWeightValue,
    );
    weight = stableWeight;
    const stableBias = mappedGeneratedStorage(
      device,
      "opt-0006-production-bias",
      plan.outputChannels,
      productionBiasValue,
    );
    bias = stableBias;
    const rangeControlStride = device.limits.minUniformBufferOffsetAlignment;
    const stableRangeControl = mappedRangeControls(
      device,
      plan,
      rangeControlStride,
    );
    rangeControl = stableRangeControl;
    const stableKernel = AceChannelChunkedVaeConv1dKernel.create(device);
    kernel = stableKernel;
    for (const batchSize of ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES) {
      const output = mappedSentinelOutput(
        device,
        `opt-0006-output-batch-${batchSize}`,
        plan.outputElements,
      );
      outputs.set(batchSize, output);
      const rangeDispatches: AceChannelChunkedVaeConv1dDispatch[] = [];
      for (const [localIndex, rangeIndex] of
        OPT_0006_PRODUCTION_RANGE_INDICES.entries()) {
        const range = plan.outputRanges[rangeIndex]!;
        rangeDispatches.push(await stableKernel.createDispatch(
          `opt-0006-production-batch-${batchSize}-range-${rangeIndex}`,
          OPT_0006_PRODUCTION_RANGE_SHAPE,
          {
            input: binding(stableInput),
            weight: binding(stableWeight),
            bias: binding(stableBias),
            output: binding(output),
          },
          {
            base: range.firstOutput,
            count: range.outputCount,
            control: {
              buffer: stableRangeControl,
              offset: localIndex * rangeControlStride,
              size: 16,
            },
          },
        ));
      }
      dispatches.set(batchSize, Object.freeze(rangeDispatches));
    }

    updateProgress("running qNaN-prefilled full-union correctness");
    const correctnessRuns = new Map<BatchSize, ExecutionResult>();
    const outputsByBatch = new Map<BatchSize, Float32Array>();
    for (const batchSize of ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES) {
      correctnessRuns.set(batchSize, await executePrepared(
        device,
        dispatches.get(batchSize)!,
        batchSize,
      ));
      outputsByBatch.set(batchSize, await readSelectedUnion(
        device,
        outputs.get(batchSize)!,
        plan,
        `opt-0006-correctness-batch-${batchSize}`,
      ));
    }
    const correctness = validateFullUnionIdentity(outputsByBatch);

    updateProgress("running direct post-drain cancellation proof");
    const cancellation = await runCancellationProof(
      device,
      dispatches.get(4)!,
    );

    updateProgress("running thermally gated balanced timing rounds");
    const samples = new Map<BatchSize, ExecutionResult[]>(
      ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES.map((batchSize) =>
        [batchSize, []]
      ),
    );
    for (const [roundIndex, order] of OPT_0006_BALANCED_BATCH_ORDERS.entries()) {
      for (const [orderPosition, batchSize] of order.entries()) {
        updateProgress(
          `round ${roundIndex + 1}/${OPT_0006_BALANCED_BATCH_ORDERS.length} ` +
            `batch ${batchSize} (${orderPosition + 1}/${order.length})`,
        );
        samples.get(batchSize)!.push(await executePrepared(
          device,
          dispatches.get(batchSize)!,
          batchSize,
        ));
        await yieldToPage();
      }
    }
    const postTiming = new Map<BatchSize, Float32Array>();
    for (const batchSize of ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES) {
      postTiming.set(batchSize, await readSelectedUnion(
        device,
        outputs.get(batchSize)!,
        plan,
        `opt-0006-post-timing-batch-${batchSize}`,
      ));
    }
    validateFullUnionIdentity(postTiming);

    const summaries = Object.fromEntries(
      ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES.map((batchSize) => {
        const retained = samples.get(batchSize)!;
        return [String(batchSize), Object.freeze({
          samples: Object.freeze(retained),
          wallMilliseconds: summarizeOpt0006Samples(
            retained.map((sample) => sample.wallMilliseconds),
          ),
          sumBatchDrainMilliseconds: summarizeOpt0006Samples(
            retained.map((sample) => sample.sumBatchDrainMilliseconds),
          ),
          maximumBatchDrainMilliseconds: summarizeOpt0006Samples(
            retained.map((sample) => Math.max(
              ...sample.batchDrains.map((batch) => batch.drainMilliseconds),
            )),
          ),
        })];
      }),
    );
    const baselineMedian = summaries["1"]!.wallMilliseconds.median;
    return Object.freeze({
      schema: "ace-opt-0006-vae-command-buffer-coalescing-production-ab-v1",
      status: "passed",
      experimentId: "OPT-0006",
      classification: "benchmark-only-production-range-scheduling-screen",
      recordedAt: new Date().toISOString(),
      browser: { userAgent: navigator.userAgent, page: window.location.href },
      adapter: adapterIdentity(adapter),
      protocol: {
        thermalGate,
        kernel: "AceChannelChunkedVaeConv1dKernel",
        compilationAllocationUploadExcludedFromTiming: true,
        independentOutputPerBatchSize: true,
        outputPrefill: "quiet-NaN-u32-sentinel",
        candidateBatchSizes: ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES,
        balancedOrders: OPT_0006_BALANCED_BATCH_ORDERS,
        finalCommandBufferRemains: true,
        realQueueEmptyIdleMillisecondsRequested: EXPLICIT_IDLE_MILLISECONDS,
        oneCommandBufferOutstanding: true,
      },
      scope: {
        shape: OPT_0006_PRODUCTION_RANGE_SHAPE,
        selectedRangeIndices: OPT_0006_PRODUCTION_RANGE_INDICES,
        selectedRangeCount: OPT_0006_PRODUCTION_RANGE_INDICES.length,
        firstOutput: plan.outputRanges[32]!.firstOutput,
        outputElements: 524_288,
        outputElementsPerQuantum: 32_768,
        multiplyAddsPerQuantum: 234_881_024,
        totalMultiplyAdds: 3_758_096_384,
        fullWindowOrSong: false,
      },
      correctness: {
        ...correctness,
        schedulingRuns: Object.fromEntries(correctnessRuns),
        postTimingIdentity: true,
      },
      cancellation,
      timing: {
        summaries,
        medianWallSpeedupVersusBatch1: Object.fromEntries(
          ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES.map((batchSize) =>
            [String(batchSize), baselineMedian /
              summaries[String(batchSize)]!.wallMilliseconds.median]
          ),
        ),
      },
    });
  } finally {
    kernel?.destroy();
    rangeControl?.destroy();
    input?.destroy();
    weight?.destroy();
    bias?.destroy();
    for (const output of outputs.values()) output.destroy();
  }
}

async function executePrepared(
  device: GPUDevice,
  dispatches: PreparedRangeDispatches,
  batchSize: BatchSize,
): Promise<ExecutionResult> {
  const expected = expectedOpt0006SchedulingCounts(batchSize);
  const counters: ExecutionCounters = {
    passCount: 0,
    dispatchCount: 0,
    commandEncoderCount: 0,
  };
  const heartbeat = startHeartbeat();
  const batchDrains: BatchDrain[] = [];
  const idleDurations: number[] = [];
  const progress: AceOpt0006BatchProgress[] = [];
  let outstanding = 0;
  let maximumOutstanding = 0;
  const queue = {
    submit(commandBuffers: Iterable<GPUCommandBuffer>): undefined {
      const retained = [...commandBuffers];
      if (retained.length !== 1 || outstanding !== 0) {
        throw new Error("OPT-0006 violated one-command-buffer-outstanding");
      }
      outstanding = 1;
      maximumOutstanding = Math.max(maximumOutstanding, outstanding);
      return device.queue.submit(retained);
    },
    async onSubmittedWorkDone(): Promise<undefined> {
      const started = performance.now();
      await device.queue.onSubmittedWorkDone();
      outstanding = 0;
      batchDrains.push(Object.freeze({
        batchIndex: batchDrains.length,
        drainMilliseconds: performance.now() - started,
        heartbeat: heartbeat.snapshot(),
      }));
      return undefined;
    },
  } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
  const quanta = selectedQuanta(dispatches);
  const started = performance.now();
  let result;
  let heartbeatResult: HeartbeatSnapshot | undefined;
  try {
    result = await runAceOpt0006QuantumBatches({
      device: instrumentedDevice(device, counters),
      queue,
      quanta,
      maximumQuantaPerCommandBuffer: batchSize,
      signal: new AbortController().signal,
      finalCommandBufferRemains: true,
      label: `opt-0006-production-batch-${batchSize}`,
      yieldQueueIdle: async () => {
        const idleStarted = performance.now();
        await realQueueEmptyIdle();
        idleDurations.push(performance.now() - idleStarted);
      },
      onProgress: (event) => progress.push(event),
    });
  } finally {
    heartbeatResult = heartbeat.stop();
  }
  const wallMilliseconds = performance.now() - started;
  const explicitIdleMilliseconds = idleDurations.reduce(
    (total, duration) => total + duration,
    0,
  );
  const actual = {
    quantumCount: result.completedQuanta,
    passCount: counters.passCount,
    dispatchCount: counters.dispatchCount,
    commandBufferCount: result.commandBuffersSubmitted,
    queueDrains: result.queueDrains,
    progressEventCount: progress.length,
    explicitIdleCount: idleDurations.length,
  };
  for (const key of Object.keys(expected) as Array<keyof typeof expected>) {
    if (actual[key] !== expected[key]) {
      throw new Error(
        `OPT-0006 batch ${batchSize} ${key} ${actual[key]} != ${expected[key]}`,
      );
    }
  }
  if (
    counters.commandEncoderCount !== expected.commandBufferCount ||
    result.batchCount !== expected.commandBufferCount ||
    result.cooperativeIdleMs !== expected.explicitIdleCount ||
    maximumOutstanding !== 1 || outstanding !== 0
  ) {
    throw new Error(`OPT-0006 batch ${batchSize} scheduler telemetry changed`);
  }
  validateProgress(progress, batchSize);
  return Object.freeze({
    batchSize,
    wallMilliseconds,
    sumBatchDrainMilliseconds: batchDrains.reduce(
      (total, batch) => total + batch.drainMilliseconds,
      0,
    ),
    completedQuanta: result.completedQuanta,
    passCount: counters.passCount,
    dispatchCount: counters.dispatchCount,
    commandBufferCount: result.commandBuffersSubmitted,
    queueDrains: result.queueDrains,
    progressEventCount: progress.length,
    explicitIdleCount: idleDurations.length,
    explicitIdleMilliseconds,
    maximumOutstandingCommandBuffers: maximumOutstanding,
    batchDrains: Object.freeze(batchDrains),
    progress: Object.freeze(progress),
    heartbeat: heartbeatResult!,
  });
}

async function runCancellationProof(
  device: GPUDevice,
  dispatches: PreparedRangeDispatches,
): Promise<unknown> {
  const controller = new AbortController();
  const counters: ExecutionCounters = {
    passCount: 0,
    dispatchCount: 0,
    commandEncoderCount: 0,
  };
  let submissions = 0;
  let drains = 0;
  let progressEvents = 0;
  let idleCount = 0;
  let idleMilliseconds = 0;
  let abortedAt = 0;
  const queue = {
    submit(commandBuffers: Iterable<GPUCommandBuffer>): undefined {
      const retained = [...commandBuffers];
      if (retained.length !== 1) throw new Error("cancellation submit not singleton");
      submissions += 1;
      return device.queue.submit(retained);
    },
    async onSubmittedWorkDone(): Promise<undefined> {
      await device.queue.onSubmittedWorkDone();
      drains += 1;
      return undefined;
    },
  } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
  let rejection: unknown;
  try {
    await runAceOpt0006QuantumBatches({
      device: instrumentedDevice(device, counters),
      queue,
      quanta: selectedQuanta(dispatches),
      maximumQuantaPerCommandBuffer: 4,
      signal: controller.signal,
      finalCommandBufferRemains: true,
      label: "opt-0006-cancellation",
      yieldQueueIdle: async () => {
        const started = performance.now();
        await realQueueEmptyIdle();
        idleMilliseconds += performance.now() - started;
        idleCount += 1;
      },
      onProgress: (event) => {
        progressEvents += 1;
        if (event.completedQuanta === 4) {
          abortedAt = performance.now();
          controller.abort(
            new DOMException("OPT-0006 cancellation", "AbortError"),
          );
        }
      },
    });
  } catch (error) {
    rejection = error;
  }
  const deliveredAt = performance.now();
  if (
    !(rejection instanceof DOMException) || rejection.name !== "AbortError" ||
    submissions !== 1 || drains !== 1 || counters.commandEncoderCount !== 1 ||
    counters.passCount !== 4 || counters.dispatchCount !== 4 ||
    progressEvents !== 4 || idleCount !== 1 || idleMilliseconds < 1
  ) {
    throw new Error("OPT-0006 cancellation did not stop after its first drain");
  }
  return Object.freeze({
    batchSize: 4,
    plannedBatchCount: 4,
    submittedCommandBuffers: submissions,
    queueDrains: drains,
    encodedPasses: counters.passCount,
    physicalDispatches: counters.dispatchCount,
    progressEventCount: progressEvents,
    completedIdleCount: idleCount,
    completedIdleMilliseconds: idleMilliseconds,
    abortName: rejection.name,
    deliveryLatencyMilliseconds: deliveredAt - abortedAt,
    laterEncodingPrevented: true,
    laterSubmissionPrevented: true,
  });
}

function selectedQuanta(
  dispatches: PreparedRangeDispatches,
): readonly AceOpt0006EncodableQuantum[] {
  if (dispatches.length !== OPT_0006_PRODUCTION_RANGE_INDICES.length) {
    throw new Error("OPT-0006 prepared dispatch count changed");
  }
  return Object.freeze(OPT_0006_PRODUCTION_RANGE_INDICES.map((rangeIndex, index) =>
    Object.freeze({
      id: `production-range-${rangeIndex}`,
      encode(pass: GPUComputePassEncoder): void {
        dispatches[index]!.encode(pass);
      },
    })
  ));
}

function instrumentedDevice(
  device: GPUDevice,
  counters: ExecutionCounters,
): Pick<GPUDevice, "createCommandEncoder"> {
  return {
    createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor) {
      counters.commandEncoderCount += 1;
      const encoder = device.createCommandEncoder(descriptor);
      return new Proxy(encoder, {
        get(target, property) {
          if (property === "beginComputePass") {
            return (passDescriptor?: GPUComputePassDescriptor) => {
              counters.passCount += 1;
              const pass = target.beginComputePass(passDescriptor);
              return new Proxy(pass, {
                get(passTarget, passProperty) {
                  if (passProperty === "dispatchWorkgroups") {
                    return (x: number, y?: number, z?: number) => {
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

function validateProgress(
  progress: readonly AceOpt0006BatchProgress[],
  batchSize: BatchSize,
): void {
  const batches = planAceOpt0006QuantumBatches(16, batchSize);
  if (progress.length !== 16) throw new Error("progress count changed");
  for (let quantumIndex = 0; quantumIndex < 16; quantumIndex += 1) {
    const batchIndex = Math.floor(quantumIndex / batchSize);
    const event = progress[quantumIndex]!;
    if (
      event.completedQuanta !== quantumIndex + 1 || event.totalQuanta !== 16 ||
      event.commandBuffersSubmitted !== batchIndex + 1 ||
      event.queueDrains !== batchIndex + 1 ||
      event.cooperativeIdleMs !== batchIndex + 1 ||
      event.completedBatchIndex !== batchIndex ||
      event.totalBatches !== batches.length
    ) {
      throw new Error(
        `OPT-0006 batch ${batchSize} progress ${quantumIndex} changed`,
      );
    }
  }
}

function validateSelectedRanges(
  plan: ReturnType<typeof planAceOpt0005VaeConv1dChannelChunks>,
): void {
  if (plan.outputRangeCount !== 80) {
    throw new Error("OPT-0006 production operation no longer has 80 ranges");
  }
  let expectedFirstOutput = plan.outputRanges[32]!.firstOutput;
  for (const rangeIndex of OPT_0006_PRODUCTION_RANGE_INDICES) {
    const range = plan.outputRanges[rangeIndex]!;
    if (
      range.firstOutput !== expectedFirstOutput ||
      range.outputRowCount !== 32 || range.outputCount !== 32_768 ||
      range.multiplyAdds !== 234_881_024
    ) throw new Error(`OPT-0006 production range ${rangeIndex} changed`);
    expectedFirstOutput += range.outputCount;
  }
}

function validateFullUnionIdentity(
  outputs: ReadonlyMap<BatchSize, Float32Array>,
): Readonly<Record<string, unknown>> {
  const authority = outputs.get(1);
  if (authority === undefined || authority.length !== 524_288) {
    throw new Error("OPT-0006 batch-1 authority is incomplete");
  }
  const authorityBits = bitsOf(authority);
  const authorityFingerprint = fingerprint(authority, authorityBits);
  if (
    authorityFingerprint.finiteCount !== authority.length ||
    authorityFingerprint.nonzeroCount === 0
  ) throw new Error("OPT-0006 authority retained qNaNs or degenerate output");
  const comparisons = ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES.map(
    (batchSize) => {
      const output = outputs.get(batchSize)!;
      const bits = bitsOf(output);
      let bitMismatchCount = 0;
      for (let index = 0; index < bits.length; index += 1) {
        if (bits[index] !== authorityBits[index]) bitMismatchCount += 1;
      }
      if (bitMismatchCount !== 0) {
        throw new Error(
          `OPT-0006 batch ${batchSize} has ${bitMismatchCount} bit mismatches`,
        );
      }
      return Object.freeze({
        batchSize,
        comparedElements: bits.length,
        bitMismatchCount,
        fingerprint: fingerprint(output, bits),
      });
    },
  );
  return Object.freeze({
    authorityBatchSize: 1,
    comparedElementsPerCandidate: authority.length,
    bitIdentical: true,
    authorityFingerprint,
    comparisons: Object.freeze(comparisons),
  });
}

async function readSelectedUnion(
  device: GPUDevice,
  output: GPUBuffer,
  plan: ReturnType<typeof planAceOpt0005VaeConv1dChannelChunks>,
  label: string,
): Promise<Float32Array> {
  const firstOutput = plan.outputRanges[32]!.firstOutput;
  const elementCount = 524_288;
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
      firstOutput * FLOAT32_BYTES,
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

function mappedRangeControls(
  device: GPUDevice,
  plan: ReturnType<typeof planAceOpt0005VaeConv1dChannelChunks>,
  stride: number,
): GPUBuffer {
  if (!Number.isSafeInteger(stride) || stride < 16 || stride % 4 !== 0) {
    throw new RangeError("OPT-0006 range-control stride is invalid");
  }
  const buffer = device.createBuffer({
    label: "opt-0006-production-range-controls",
    size: stride * OPT_0006_PRODUCTION_RANGE_INDICES.length,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  try {
    const words = new Uint32Array(buffer.getMappedRange());
    for (const [localIndex, rangeIndex] of
      OPT_0006_PRODUCTION_RANGE_INDICES.entries()) {
      const range = plan.outputRanges[rangeIndex]!;
      words.set(
        [range.firstOutput, range.outputCount, 0, 0],
        localIndex * stride / Uint32Array.BYTES_PER_ELEMENT,
      );
    }
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    buffer.destroy();
    throw error;
  }
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
    for (let index = 0; index < elements; index += 1) mapped[index] = value(index);
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

function productionInputValue(index: number): number {
  return Math.fround((((index * 17 + 3) % 31) - 15) / 32);
}

function productionWeightValue(index: number): number {
  return Math.fround((((index * 13 + 7) % 29) - 14) / 64);
}

function productionBiasValue(outputChannel: number): number {
  return Math.fround(((outputChannel % 11) - 5) / 128);
}

function bitsOf(values: Float32Array): Uint32Array {
  return new Uint32Array(values.buffer, values.byteOffset, values.length);
}

function fingerprint(values: Float32Array, bits: Uint32Array): Readonly<{
  elementCount: number;
  finiteCount: number;
  nonzeroCount: number;
  fnv1a32: string;
}> {
  let finiteCount = 0;
  let nonzeroCount = 0;
  let hash = 0x811c_9dc5;
  for (let index = 0; index < values.length; index += 1) {
    if (Number.isFinite(values[index]!)) finiteCount += 1;
    if (values[index] !== 0) nonzeroCount += 1;
    hash = Math.imul(hash ^ bits[index]!, 0x0100_0193) >>> 0;
  }
  return Object.freeze({
    elementCount: values.length,
    finiteCount,
    nonzeroCount,
    fnv1a32: hash.toString(16).padStart(8, "0"),
  });
}

function startHeartbeat(): Readonly<{
  snapshot(): HeartbeatSnapshot;
  stop(): HeartbeatSnapshot;
}> {
  const animationGaps: number[] = [];
  const timerGaps: number[] = [];
  let stopped = false;
  let animationFrameCount = 0;
  let timerTickCount = 0;
  let lastAnimation = performance.now();
  let lastTimer = lastAnimation;
  let frameHandle = 0;
  const snapshot = (): HeartbeatSnapshot => Object.freeze({
    animationFrameCount,
    timerTickCount,
    maximumAnimationFrameGapMilliseconds: Math.max(0, ...animationGaps),
    maximumTimerGapMilliseconds: Math.max(0, ...timerGaps),
  });
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
  return Object.freeze({
    snapshot,
    stop(): HeartbeatSnapshot {
      if (!stopped) {
        stopped = true;
        cancelAnimationFrame(frameHandle);
        clearInterval(timerHandle);
      }
      return snapshot();
    },
  });
}

async function realQueueEmptyIdle(): Promise<void> {
  await new Promise<void>((resolve) => {
    window.setTimeout(resolve, EXPLICIT_IDLE_MILLISECONDS);
  });
}

async function yieldToPage(): Promise<void> {
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
}

function assertAdapter(
  adapter: GPUAdapter,
  largestBinding: number,
  workgroupStorageBytes: number,
): void {
  const limits = adapter.limits;
  if (
    limits.maxBufferSize < largestBinding ||
    limits.maxStorageBufferBindingSize < largestBinding ||
    limits.maxComputeInvocationsPerWorkgroup < 128 ||
    limits.maxComputeWorkgroupSizeX < 16 ||
    limits.maxComputeWorkgroupSizeY < 8 ||
    limits.maxComputeWorkgroupStorageSize < workgroupStorageBytes
  ) throw new Error("Adapter cannot satisfy the OPT-0006 production screen");
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
      maxComputeWorkgroupStorageSize:
        adapter.limits.maxComputeWorkgroupStorageSize,
    }),
  });
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function requiredNumber(parameters: URLSearchParams, name: string): number {
  const raw = parameters.get(name);
  if (raw === null || raw.length === 0) {
    throw new Error(`Missing OPT-0006 ${name}`);
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`Invalid OPT-0006 ${name}`);
  }
  return value;
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
