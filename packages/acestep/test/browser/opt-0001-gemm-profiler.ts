import {
  ACE_TILED_GEMM_WORKGROUP_BYTES,
  AceCorrectnessGemmKernel,
  type AceGemmDispatch,
  type AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";

type ShapeId = "h-to-h" | "h-to-1024" | "h-to-6144" | "6144-to-h";

export interface Opt0001GemmShape {
  readonly id: ShapeId;
  readonly shape: AceGemmShape;
}

interface MillisecondSummary {
  readonly count: number;
  readonly total: number;
  readonly minimum: number;
  readonly median: number;
  readonly maximum: number;
}

interface RangeTiming {
  readonly rangeIndex: number;
  readonly encodeMilliseconds: number;
  readonly submitMilliseconds: number;
  readonly drainMilliseconds: number;
  readonly explicitIdleMilliseconds?: number;
}

interface HeartbeatResult {
  readonly animationFrames: number;
  readonly timerTicks: number;
  readonly maximumAnimationFrameGapMilliseconds: number;
  readonly maximumTimerGapMilliseconds: number;
}

interface OutputChecksum {
  readonly elementCount: number;
  readonly finiteCount: number;
  readonly nonzeroCount: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly sum: number;
  readonly absoluteSum: number;
  readonly weightedSum: number;
  readonly fnv1a32: string;
}

interface OutputSentinel {
  readonly index: number;
  readonly actual: number;
  readonly expected: number;
  readonly absoluteError: number;
}

interface ReadbackTiming {
  readonly allocationMilliseconds: number;
  readonly encodeMilliseconds: number;
  readonly submitMilliseconds: number;
  readonly drainMilliseconds: number;
  readonly mapMilliseconds: number;
  readonly checksumAndSentinelsMilliseconds: number;
}

interface ShapeResult {
  readonly id: ShapeId;
  readonly shape: AceGemmShape;
  readonly logical: {
    readonly activationElements: number;
    readonly weightElements: number;
    readonly outputElements: number;
    readonly logicalMultiplyAdds: number;
    readonly scheduledMultiplyAdds: number;
    readonly logicalFlops: number;
    readonly scheduledFlops: number;
  };
  readonly logicalGpuBytes: {
    readonly activation: number;
    readonly packedBf16Weight: number;
    readonly output: number;
    readonly rangeParameters: number;
    readonly executionWorkingSet: number;
    readonly validationPeak: number;
  };
  readonly commands: {
    readonly ranges: number;
    readonly computePasses: number;
    readonly computeDispatches: number;
    readonly computeCommandBuffers: number;
    readonly readbackCommandBuffers: number;
    readonly queueSubmissions: number;
    readonly queueDrains: number;
    readonly explicitIdleIntervals: number;
  };
  readonly ranges: readonly {
    readonly index: number;
    readonly firstOutput: number;
    readonly outputCount: number;
    readonly firstWorkgroup: number;
    readonly workgroupCount: number;
    readonly multiplyAdds: number;
  }[];
  readonly timings: {
    readonly bufferPreparationMilliseconds: number;
    readonly compileAndBindGroupMilliseconds: number;
    readonly warmupCooperativeWallMilliseconds: number;
    readonly warmupOutputResetMilliseconds: number;
    readonly cooperativeComputeWallMilliseconds: number;
    readonly activeComputeWallMilliseconds: number;
    readonly encode: MillisecondSummary;
    readonly submit: MillisecondSummary;
    readonly drain: MillisecondSummary;
    readonly explicitIdle: MillisecondSummary;
    readonly rangeSamples: readonly RangeTiming[];
    readonly readback: ReadbackTiming;
  };
  readonly throughput: {
    readonly logicalActiveTflops: number;
    readonly logicalCooperativeTflops: number;
    readonly scheduledActiveTflops: number;
    readonly scheduledCooperativeTflops: number;
  };
  readonly responsiveness: HeartbeatResult;
  readonly correctness: {
    readonly outputPrefill: "u32-zero-after-warmup";
    readonly fullDomainFinite: true;
    readonly maximumSentinelAbsoluteError: number;
    readonly checksum: OutputChecksum;
    readonly sentinels: readonly OutputSentinel[];
  };
}

interface ProfilerResult {
  readonly schema: "ace-opt-0001-gemm-profiler-v1";
  readonly status: "passed";
  readonly experimentId: "OPT-0001";
  readonly classification: "benchmark-only-no-production-math-change";
  readonly recordedAt: string;
  readonly browser: {
    readonly userAgent: string;
    readonly page: string;
  };
  readonly adapter: {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
    readonly isFallbackAdapter: boolean;
    readonly subgroupMinSize?: number;
    readonly subgroupMaxSize?: number;
    readonly features: readonly string[];
    readonly limits: {
      readonly maxBufferSize: number;
      readonly maxStorageBufferBindingSize: number;
      readonly maxComputeWorkgroupStorageSize: number;
      readonly maxComputeInvocationsPerWorkgroup: number;
      readonly maxComputeWorkgroupSizeX: number;
      readonly maxComputeWorkgroupsPerDimension: number;
    };
  };
  readonly protocol: {
    readonly modelProfile: "reference-bf16";
    readonly kernel: "AceCorrectnessGemmKernel";
    readonly usesSubgroups: false;
    readonly warmupExecutions: 1;
    readonly measuredExecutionsPerShape: 1;
    readonly oneCommandBufferOutstanding: true;
    readonly queueDrainAfterEveryCommandBuffer: true;
    readonly queueEmptyIdleMillisecondsRequested: 1;
    readonly idleAfterFinalComputeRange: false;
    readonly authoritativeTiming: "performance.now-wall-clock";
    readonly fixture: {
      readonly id: "opt-0001-deterministic-nondegenerate-v1";
      readonly activation: "(((index*17+3)%31)-15)/32";
      readonly weight: "(((index*13+7)%29)-14)/64-packed-bf16";
    };
  };
  readonly shapes: readonly ShapeResult[];
  readonly summary: {
    readonly shapeCount: number;
    readonly logicalFlops: number;
    readonly scheduledFlops: number;
    readonly compileAndBindGroupMilliseconds: number;
    readonly cooperativeComputeWallMilliseconds: number;
    readonly activeComputeWallMilliseconds: number;
    readonly explicitIdleMilliseconds: number;
    readonly logicalActiveTflops: number;
    readonly logicalCooperativeTflops: number;
    readonly scheduledActiveTflops: number;
    readonly scheduledCooperativeTflops: number;
    readonly computeCommandBuffers: number;
    readonly queueSubmissionsIncludingReadback: number;
    readonly maximumAnimationFrameGapMilliseconds: number;
    readonly maximumTimerGapMilliseconds: number;
    readonly maximumSentinelAbsoluteError: number;
  };
}

interface OwnedBuffers {
  readonly activation: GPUBuffer;
  readonly weight: GPUBuffer;
  readonly output: GPUBuffer;
}

const HIDDEN_SIZE = 2_048;
const ROWS = 2_250;
const EXPLICIT_IDLE_MILLISECONDS = 1;
const OUTPUT_SENTINEL_BITS = 0x7fc0_0000;
const SENTINEL_TOLERANCE = 1e-5;
const BYTES_PER_F32 = Float32Array.BYTES_PER_ELEMENT;
const BF16_FLOAT_SCRATCH = new Float32Array(1);
const BF16_BIT_SCRATCH = new Uint32Array(BF16_FLOAT_SCRATCH.buffer);

export const OPT_0001_GEMM_SHAPES: readonly Opt0001GemmShape[] = Object.freeze([
  Object.freeze({
    id: "h-to-h",
    shape: Object.freeze({ rows: ROWS, inner: HIDDEN_SIZE, columns: HIDDEN_SIZE }),
  }),
  Object.freeze({
    id: "h-to-1024",
    shape: Object.freeze({ rows: ROWS, inner: HIDDEN_SIZE, columns: 1_024 }),
  }),
  Object.freeze({
    id: "h-to-6144",
    shape: Object.freeze({ rows: ROWS, inner: HIDDEN_SIZE, columns: 6_144 }),
  }),
  Object.freeze({
    id: "6144-to-h",
    shape: Object.freeze({ rows: ROWS, inner: 6_144, columns: HIDDEN_SIZE }),
  }),
]);

export function opt0001ActivationValue(index: number): number {
  return Math.fround((((index * 17 + 3) % 31) - 15) / 32);
}

export function opt0001WeightValue(index: number): number {
  return Math.fround((((index * 13 + 7) % 29) - 14) / 64);
}

if (typeof document !== "undefined") {
  const start = requireStartButton();
  start.addEventListener("click", () => {
    start.disabled = true;
    document.body.dataset.status = "running";
    updateProgress("starting");
    const resultNode = document.querySelector<HTMLPreElement>("#result");
    if (resultNode === null) throw new Error("Missing result element");
    resultNode.textContent = "running";
    void run().then(
      (result) => finish("passed", result),
      (error: unknown) => finish("failed", {
        schema: "ace-opt-0001-gemm-profiler-v1",
        status: "failed",
        experimentId: "OPT-0001",
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

async function run(): Promise<ProfilerResult> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  assertAdapterLimits(adapter);
  const largestStorageBinding = largestStorageBindingBytes();
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: largestStorageBinding,
      maxStorageBufferBindingSize: largestStorageBinding,
      maxComputeWorkgroupStorageSize: ACE_TILED_GEMM_WORKGROUP_BYTES,
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 128,
    },
  });
  let deviceLost = false;
  void device.lost.then(() => {
    deviceLost = true;
  });
  try {
    const shapes: ShapeResult[] = [];
    for (let index = 0; index < OPT_0001_GEMM_SHAPES.length; index += 1) {
      const fixture = OPT_0001_GEMM_SHAPES[index]!;
      updateProgress(`running ${index + 1}/${OPT_0001_GEMM_SHAPES.length}: ${fixture.id}`);
      shapes.push(await runShape(device, fixture));
      if (deviceLost) throw new Error("WebGPU device was lost during the profiler run");
      await yieldToPage();
    }
    const result: ProfilerResult = {
      schema: "ace-opt-0001-gemm-profiler-v1",
      status: "passed",
      experimentId: "OPT-0001",
      classification: "benchmark-only-no-production-math-change",
      recordedAt: new Date().toISOString(),
      browser: {
        userAgent: navigator.userAgent,
        page: window.location.href,
      },
      adapter: adapterIdentity(adapter),
      protocol: {
        modelProfile: "reference-bf16",
        kernel: "AceCorrectnessGemmKernel",
        usesSubgroups: false,
        warmupExecutions: 1,
        measuredExecutionsPerShape: 1,
        oneCommandBufferOutstanding: true,
        queueDrainAfterEveryCommandBuffer: true,
        queueEmptyIdleMillisecondsRequested: EXPLICIT_IDLE_MILLISECONDS,
        idleAfterFinalComputeRange: false,
        authoritativeTiming: "performance.now-wall-clock",
        fixture: {
          id: "opt-0001-deterministic-nondegenerate-v1",
          activation: "(((index*17+3)%31)-15)/32",
          weight: "(((index*13+7)%29)-14)/64-packed-bf16",
        },
      },
      shapes,
      summary: summarizeShapes(shapes),
    };
    return result;
  } finally {
    device.destroy();
  }
}

async function runShape(
  device: GPUDevice,
  fixture: Opt0001GemmShape,
): Promise<ShapeResult> {
  const preparationStarted = performance.now();
  const owned = createOwnedBuffers(device, fixture);
  const bufferPreparationMilliseconds = performance.now() - preparationStarted;
  const kernel = AceCorrectnessGemmKernel.create(device, "reference-bf16");
  try {
    const compileStarted = performance.now();
    const dispatch = await kernel.createDispatch(
      `opt-0001-${fixture.id}`,
      fixture.shape,
      {
        activation: binding(owned.activation),
        weight: binding(owned.weight),
        output: binding(owned.output),
      },
    );
    const compileAndBindGroupMilliseconds = performance.now() - compileStarted;
    const warmup = await executeCooperatively(device, dispatch);
    const warmupOutputResetMilliseconds = await resetOutputAfterWarmup(
      device,
      fixture,
      owned.output,
    );
    await queueEmptyIdle();
    await yieldToPage();
    const execution = await executeCooperatively(device, dispatch);
    const validation = await readAndValidateOutput(
      device,
      fixture,
      dispatch,
      owned.output,
    );
    const logicalMultiplyAdds =
      fixture.shape.rows * fixture.shape.inner * fixture.shape.columns;
    const scheduledMultiplyAdds = dispatch.plan.outputRanges.reduce(
      (total, range) => total + range.multiplyAdds,
      0,
    );
    const logicalFlops = logicalMultiplyAdds * 2;
    const scheduledFlops = scheduledMultiplyAdds * 2;
    const activationBytes = dispatch.plan.activationElements * BYTES_PER_F32;
    const weightBytes = Math.ceil(dispatch.plan.weightElements / 2) * BYTES_PER_F32;
    const outputBytes = dispatch.plan.outputElements * BYTES_PER_F32;
    const rangeParameterBytes = Math.max(256, dispatch.rangeCount * 256);
    const executionWorkingSet =
      activationBytes + weightBytes + outputBytes + rangeParameterBytes;
    const activeComputeWallMilliseconds = Math.max(
      Number.EPSILON,
      execution.cooperativeComputeWallMilliseconds -
        execution.explicitIdle.total,
    );
    return {
      id: fixture.id,
      shape: fixture.shape,
      logical: {
        activationElements: dispatch.plan.activationElements,
        weightElements: dispatch.plan.weightElements,
        outputElements: dispatch.plan.outputElements,
        logicalMultiplyAdds,
        scheduledMultiplyAdds,
        logicalFlops,
        scheduledFlops,
      },
      logicalGpuBytes: {
        activation: activationBytes,
        packedBf16Weight: weightBytes,
        output: outputBytes,
        rangeParameters: rangeParameterBytes,
        executionWorkingSet,
        validationPeak: executionWorkingSet + outputBytes,
      },
      commands: {
        ranges: dispatch.rangeCount,
        computePasses: dispatch.rangeCount,
        computeDispatches: dispatch.rangeCount,
        computeCommandBuffers: dispatch.rangeCount,
        readbackCommandBuffers: 1,
        queueSubmissions: dispatch.rangeCount + 1,
        queueDrains: dispatch.rangeCount + 1,
        explicitIdleIntervals: Math.max(0, dispatch.rangeCount - 1),
      },
      ranges: dispatch.plan.outputRanges.map((range, index) => ({
        index,
        firstOutput: range.firstOutput,
        outputCount: range.outputCount,
        firstWorkgroup: range.firstWorkgroup,
        workgroupCount: range.workgroupCount,
        multiplyAdds: range.multiplyAdds,
      })),
      timings: {
        bufferPreparationMilliseconds: roundTiming(bufferPreparationMilliseconds),
        compileAndBindGroupMilliseconds: roundTiming(
          compileAndBindGroupMilliseconds,
        ),
        warmupCooperativeWallMilliseconds: roundTiming(
          warmup.cooperativeComputeWallMilliseconds,
        ),
        warmupOutputResetMilliseconds: roundTiming(
          warmupOutputResetMilliseconds,
        ),
        cooperativeComputeWallMilliseconds: roundTiming(
          execution.cooperativeComputeWallMilliseconds,
        ),
        activeComputeWallMilliseconds: roundTiming(activeComputeWallMilliseconds),
        encode: execution.encode,
        submit: execution.submit,
        drain: execution.drain,
        explicitIdle: execution.explicitIdle,
        rangeSamples: execution.rangeSamples,
        readback: validation.timing,
      },
      throughput: {
        logicalActiveTflops: tflops(logicalFlops, activeComputeWallMilliseconds),
        logicalCooperativeTflops: tflops(
          logicalFlops,
          execution.cooperativeComputeWallMilliseconds,
        ),
        scheduledActiveTflops: tflops(
          scheduledFlops,
          activeComputeWallMilliseconds,
        ),
        scheduledCooperativeTflops: tflops(
          scheduledFlops,
          execution.cooperativeComputeWallMilliseconds,
        ),
      },
      responsiveness: execution.heartbeat,
      correctness: {
        outputPrefill: "u32-zero-after-warmup",
        fullDomainFinite: true,
        maximumSentinelAbsoluteError: validation.maximumSentinelAbsoluteError,
        checksum: validation.checksum,
        sentinels: validation.sentinels,
      },
    };
  } finally {
    kernel.destroy();
    owned.activation.destroy();
    owned.weight.destroy();
    owned.output.destroy();
  }
}

async function resetOutputAfterWarmup(
  device: GPUDevice,
  fixture: Opt0001GemmShape,
  output: GPUBuffer,
): Promise<number> {
  const started = performance.now();
  const encoder = device.createCommandEncoder({
    label: `opt-0001-${fixture.id}-warmup-reset-encoder`,
  });
  encoder.clearBuffer(output);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - started;
}

async function executeCooperatively(
  device: GPUDevice,
  dispatch: AceGemmDispatch,
): Promise<{
  readonly cooperativeComputeWallMilliseconds: number;
  readonly encode: MillisecondSummary;
  readonly submit: MillisecondSummary;
  readonly drain: MillisecondSummary;
  readonly explicitIdle: MillisecondSummary;
  readonly rangeSamples: readonly RangeTiming[];
  readonly heartbeat: HeartbeatResult;
}> {
  const heartbeat = startHeartbeat();
  const rangeSamples: RangeTiming[] = [];
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
    const encodeMilliseconds = performance.now() - encodeStarted;

    const submitStarted = performance.now();
    device.queue.submit([command]);
    const submitMilliseconds = performance.now() - submitStarted;

    const drainStarted = performance.now();
    await device.queue.onSubmittedWorkDone();
    const drainMilliseconds = performance.now() - drainStarted;
    if (rangeIndex + 1 < dispatch.rangeCount) {
      const explicitIdleMilliseconds = await queueEmptyIdle();
      rangeSamples.push({
        rangeIndex,
        encodeMilliseconds: roundTiming(encodeMilliseconds),
        submitMilliseconds: roundTiming(submitMilliseconds),
        drainMilliseconds: roundTiming(drainMilliseconds),
        explicitIdleMilliseconds: roundTiming(explicitIdleMilliseconds),
      });
    } else {
      rangeSamples.push({
        rangeIndex,
        encodeMilliseconds: roundTiming(encodeMilliseconds),
        submitMilliseconds: roundTiming(submitMilliseconds),
        drainMilliseconds: roundTiming(drainMilliseconds),
      });
    }
  }
  const cooperativeComputeWallMilliseconds = performance.now() - wallStarted;
  const heartbeatResult = heartbeat.stop();
  if (
    cooperativeComputeWallMilliseconds >= 50 &&
    heartbeatResult.animationFrames === 0 &&
    heartbeatResult.timerTicks === 0
  ) {
    throw new Error(
      `${dispatch.label} starved both page heartbeat mechanisms for ` +
        `${cooperativeComputeWallMilliseconds} ms`,
    );
  }
  return {
    cooperativeComputeWallMilliseconds,
    encode: summarizeMilliseconds(rangeSamples.map((sample) =>
      sample.encodeMilliseconds
    )),
    submit: summarizeMilliseconds(rangeSamples.map((sample) =>
      sample.submitMilliseconds
    )),
    drain: summarizeMilliseconds(rangeSamples.map((sample) =>
      sample.drainMilliseconds
    )),
    explicitIdle: summarizeMilliseconds(rangeSamples.flatMap((sample) =>
      sample.explicitIdleMilliseconds === undefined
        ? []
        : [sample.explicitIdleMilliseconds]
    )),
    rangeSamples,
    heartbeat: heartbeatResult,
  };
}

async function readAndValidateOutput(
  device: GPUDevice,
  fixture: Opt0001GemmShape,
  dispatch: AceGemmDispatch,
  output: GPUBuffer,
): Promise<{
  readonly timing: ReadbackTiming;
  readonly checksum: OutputChecksum;
  readonly sentinels: readonly OutputSentinel[];
  readonly maximumSentinelAbsoluteError: number;
}> {
  const outputBytes = fixture.shape.rows * fixture.shape.columns * BYTES_PER_F32;
  const allocationStarted = performance.now();
  const readback = device.createBuffer({
    label: `opt-0001-${fixture.id}-readback`,
    size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const allocationMilliseconds = performance.now() - allocationStarted;
  try {
    const encodeStarted = performance.now();
    const encoder = device.createCommandEncoder({
      label: `opt-0001-${fixture.id}-readback-encoder`,
    });
    encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
    const command = encoder.finish();
    const encodeMilliseconds = performance.now() - encodeStarted;

    const submitStarted = performance.now();
    device.queue.submit([command]);
    const submitMilliseconds = performance.now() - submitStarted;
    const drainStarted = performance.now();
    await device.queue.onSubmittedWorkDone();
    const drainMilliseconds = performance.now() - drainStarted;
    const mapStarted = performance.now();
    await readback.mapAsync(GPUMapMode.READ);
    const mapMilliseconds = performance.now() - mapStarted;

    const validationStarted = performance.now();
    const mapped = readback.getMappedRange();
    const values = new Float32Array(mapped);
    const words = new Uint32Array(mapped);
    const checksum = checksumOutput(values, words);
    const sentinelIndices = outputSentinelIndices(fixture.shape, dispatch);
    const sentinels: OutputSentinel[] = [];
    let maximumSentinelAbsoluteError = 0;
    for (const index of sentinelIndices) {
      const actual = values[index]!;
      const expected = expectedOutputValue(fixture.shape, index);
      const absoluteError = Math.abs(actual - expected);
      if (!Number.isFinite(actual) || absoluteError > SENTINEL_TOLERANCE) {
        throw new Error(
          `${fixture.id} sentinel ${index} mismatch: ${actual} != ${expected} ` +
            `(absolute error ${absoluteError})`,
        );
      }
      maximumSentinelAbsoluteError = Math.max(
        maximumSentinelAbsoluteError,
        absoluteError,
      );
      sentinels.push({ index, actual, expected, absoluteError });
    }
    if (checksum.finiteCount !== values.length) {
      throw new Error(
        `${fixture.id} output has ${values.length - checksum.finiteCount} ` +
          "non-finite or unwritten values",
      );
    }
    if (
      checksum.nonzeroCount !== values.length ||
      checksum.minimum === checksum.maximum ||
      checksum.absoluteSum === 0
    ) {
      throw new Error(`${fixture.id} output checksum is degenerate`);
    }
    const checksumAndSentinelsMilliseconds = performance.now() - validationStarted;
    readback.unmap();
    return {
      timing: {
        allocationMilliseconds: roundTiming(allocationMilliseconds),
        encodeMilliseconds: roundTiming(encodeMilliseconds),
        submitMilliseconds: roundTiming(submitMilliseconds),
        drainMilliseconds: roundTiming(drainMilliseconds),
        mapMilliseconds: roundTiming(mapMilliseconds),
        checksumAndSentinelsMilliseconds: roundTiming(
          checksumAndSentinelsMilliseconds,
        ),
      },
      checksum,
      sentinels,
      maximumSentinelAbsoluteError,
    };
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

function createOwnedBuffers(
  device: GPUDevice,
  fixture: Opt0001GemmShape,
): OwnedBuffers {
  const { shape } = fixture;
  const activationElements = shape.rows * shape.inner;
  const weightElements = shape.columns * shape.inner;
  const outputElements = shape.rows * shape.columns;
  const activation = device.createBuffer({
    label: `opt-0001-${fixture.id}-activation`,
    size: activationElements * BYTES_PER_F32,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const weight = device.createBuffer({
    label: `opt-0001-${fixture.id}-weight`,
    size: Math.ceil(weightElements / 2) * BYTES_PER_F32,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const output = device.createBuffer({
    label: `opt-0001-${fixture.id}-output`,
    size: outputElements * BYTES_PER_F32,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  try {
    const activationValues = new Float32Array(activation.getMappedRange());
    for (let index = 0; index < activationValues.length; index += 1) {
      activationValues[index] = opt0001ActivationValue(index);
    }
    const weightWords = new Uint32Array(weight.getMappedRange());
    for (let word = 0; word < weightWords.length; word += 1) {
      const lowIndex = word * 2;
      const low = toBf16Bits(opt0001WeightValue(lowIndex));
      const high = lowIndex + 1 < weightElements
        ? toBf16Bits(opt0001WeightValue(lowIndex + 1))
        : 0;
      weightWords[word] = low | (high << 16);
    }
    new Uint32Array(output.getMappedRange()).fill(OUTPUT_SENTINEL_BITS);
    activation.unmap();
    weight.unmap();
    output.unmap();
    return { activation, weight, output };
  } catch (error) {
    if (activation.mapState === "mapped") activation.unmap();
    if (weight.mapState === "mapped") weight.unmap();
    if (output.mapState === "mapped") output.unmap();
    activation.destroy();
    weight.destroy();
    output.destroy();
    throw error;
  }
}

function checksumOutput(
  values: Float32Array,
  words: Uint32Array,
): OutputChecksum {
  let finiteCount = 0;
  let nonzeroCount = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  let sum = 0;
  let absoluteSum = 0;
  let weightedSum = 0;
  let fnv1a32 = 0x811c_9dc5;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (Number.isFinite(value)) finiteCount += 1;
    if (value !== 0) nonzeroCount += 1;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    sum += value;
    absoluteSum += Math.abs(value);
    weightedSum += value * ((index % 251) + 1);
    fnv1a32 = Math.imul(fnv1a32 ^ words[index]!, 0x0100_0193) >>> 0;
  }
  if (
    !Number.isFinite(sum) ||
    !Number.isFinite(absoluteSum) ||
    !Number.isFinite(weightedSum)
  ) {
    throw new Error("Output checksum overflowed or became non-finite");
  }
  return {
    elementCount: values.length,
    finiteCount,
    nonzeroCount,
    minimum,
    maximum,
    sum,
    absoluteSum,
    weightedSum,
    fnv1a32: fnv1a32.toString(16).padStart(8, "0"),
  };
}

function outputSentinelIndices(
  shape: AceGemmShape,
  dispatch: AceGemmDispatch,
): readonly number[] {
  const candidates = [
    0,
    Math.min(shape.columns - 1, 127),
    Math.min(shape.columns - 1, 128),
    shape.columns - 1,
    Math.floor(shape.rows / 2) * shape.columns,
    Math.floor(shape.rows / 2) * shape.columns + shape.columns - 1,
    (shape.rows - 1) * shape.columns,
    shape.rows * shape.columns - 1,
  ];
  const columnTiles = Math.ceil(shape.columns / dispatch.plan.tileColumns);
  for (const range of dispatch.plan.outputRanges) {
    for (const workgroup of [
      range.firstWorkgroup,
      range.firstWorkgroup + range.workgroupCount - 1,
    ]) {
      const rowTile = Math.floor(workgroup / columnTiles);
      const columnTile = workgroup % columnTiles;
      const row = Math.min(shape.rows - 1, rowTile * dispatch.plan.tileRows);
      const column = Math.min(
        shape.columns - 1,
        columnTile * dispatch.plan.tileColumns,
      );
      candidates.push(row * shape.columns + column);
    }
  }
  return Object.freeze([...new Set(candidates)]);
}

function expectedOutputValue(shape: AceGemmShape, outputIndex: number): number {
  const row = Math.floor(outputIndex / shape.columns);
  const column = outputIndex % shape.columns;
  let sum = 0;
  for (let inner = 0; inner < shape.inner; inner += 1) {
    const activation = opt0001ActivationValue(row * shape.inner + inner);
    const weight = opt0001WeightValue(column * shape.inner + inner);
    sum = Math.fround(sum + Math.fround(activation * weight));
  }
  return sum;
}

function summarizeShapes(shapes: readonly ShapeResult[]): ProfilerResult["summary"] {
  const logicalFlops = sum(shapes.map((shape) => shape.logical.logicalFlops));
  const scheduledFlops = sum(shapes.map((shape) => shape.logical.scheduledFlops));
  const compileAndBindGroupMilliseconds = sum(shapes.map((shape) =>
    shape.timings.compileAndBindGroupMilliseconds
  ));
  const cooperativeComputeWallMilliseconds = sum(shapes.map((shape) =>
    shape.timings.cooperativeComputeWallMilliseconds
  ));
  const activeComputeWallMilliseconds = sum(shapes.map((shape) =>
    shape.timings.activeComputeWallMilliseconds
  ));
  const explicitIdleMilliseconds = sum(shapes.map((shape) =>
    shape.timings.explicitIdle.total
  ));
  return {
    shapeCount: shapes.length,
    logicalFlops,
    scheduledFlops,
    compileAndBindGroupMilliseconds: roundTiming(compileAndBindGroupMilliseconds),
    cooperativeComputeWallMilliseconds: roundTiming(
      cooperativeComputeWallMilliseconds,
    ),
    activeComputeWallMilliseconds: roundTiming(activeComputeWallMilliseconds),
    explicitIdleMilliseconds: roundTiming(explicitIdleMilliseconds),
    logicalActiveTflops: tflops(logicalFlops, activeComputeWallMilliseconds),
    logicalCooperativeTflops: tflops(
      logicalFlops,
      cooperativeComputeWallMilliseconds,
    ),
    scheduledActiveTflops: tflops(scheduledFlops, activeComputeWallMilliseconds),
    scheduledCooperativeTflops: tflops(
      scheduledFlops,
      cooperativeComputeWallMilliseconds,
    ),
    computeCommandBuffers: sum(shapes.map((shape) =>
      shape.commands.computeCommandBuffers
    )),
    queueSubmissionsIncludingReadback: sum(shapes.map((shape) =>
      shape.commands.queueSubmissions
    )),
    maximumAnimationFrameGapMilliseconds: Math.max(...shapes.map((shape) =>
      shape.responsiveness.maximumAnimationFrameGapMilliseconds
    )),
    maximumTimerGapMilliseconds: Math.max(...shapes.map((shape) =>
      shape.responsiveness.maximumTimerGapMilliseconds
    )),
    maximumSentinelAbsoluteError: Math.max(...shapes.map((shape) =>
      shape.correctness.maximumSentinelAbsoluteError
    )),
  };
}

function summarizeMilliseconds(values: readonly number[]): MillisecondSummary {
  if (values.length === 0) {
    return { count: 0, total: 0, minimum: 0, median: 0, maximum: 0 };
  }
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  const median = ordered.length % 2 === 0
    ? (ordered[middle - 1]! + ordered[middle]!) / 2
    : ordered[middle]!;
  return {
    count: values.length,
    total: roundTiming(sum(values)),
    minimum: roundTiming(ordered[0]!),
    median: roundTiming(median),
    maximum: roundTiming(ordered.at(-1)!),
  };
}

function startHeartbeat(): { stop(): HeartbeatResult } {
  let active = true;
  let animationFrames = 0;
  let timerTicks = 0;
  let maximumAnimationFrameGapMilliseconds = 0;
  let maximumTimerGapMilliseconds = 0;
  let lastAnimationFrame = performance.now();
  let lastTimer = lastAnimationFrame;
  let animationFrameId = 0;
  const onAnimationFrame = (now: number): void => {
    maximumAnimationFrameGapMilliseconds = Math.max(
      maximumAnimationFrameGapMilliseconds,
      now - lastAnimationFrame,
    );
    lastAnimationFrame = now;
    animationFrames += 1;
    if (active) animationFrameId = requestAnimationFrame(onAnimationFrame);
  };
  animationFrameId = requestAnimationFrame(onAnimationFrame);
  const timerId = window.setInterval(() => {
    const now = performance.now();
    maximumTimerGapMilliseconds = Math.max(
      maximumTimerGapMilliseconds,
      now - lastTimer,
    );
    lastTimer = now;
    timerTicks += 1;
  }, 16);
  return {
    stop(): HeartbeatResult {
      active = false;
      cancelAnimationFrame(animationFrameId);
      clearInterval(timerId);
      const now = performance.now();
      maximumAnimationFrameGapMilliseconds = Math.max(
        maximumAnimationFrameGapMilliseconds,
        now - lastAnimationFrame,
      );
      maximumTimerGapMilliseconds = Math.max(
        maximumTimerGapMilliseconds,
        now - lastTimer,
      );
      return {
        animationFrames,
        timerTicks,
        maximumAnimationFrameGapMilliseconds: roundTiming(
          maximumAnimationFrameGapMilliseconds,
        ),
        maximumTimerGapMilliseconds: roundTiming(maximumTimerGapMilliseconds),
      };
    },
  };
}

async function queueEmptyIdle(): Promise<number> {
  const started = performance.now();
  await new Promise<void>((resolve) => {
    window.setTimeout(resolve, EXPLICIT_IDLE_MILLISECONDS);
  });
  const elapsed = performance.now() - started;
  if (elapsed < 0.9) {
    throw new Error(`Queue-empty idle interval was only ${elapsed} ms`);
  }
  return elapsed;
}

async function yieldToPage(): Promise<void> {
  await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
}

function adapterIdentity(adapter: GPUAdapter): ProfilerResult["adapter"] {
  const { info, limits } = adapter;
  return {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
    isFallbackAdapter: info.isFallbackAdapter,
    ...(info.subgroupMinSize === undefined
      ? {}
      : { subgroupMinSize: info.subgroupMinSize }),
    ...(info.subgroupMaxSize === undefined
      ? {}
      : { subgroupMaxSize: info.subgroupMaxSize }),
    features: Object.freeze([...adapter.features].sort()),
    limits: {
      maxBufferSize: limits.maxBufferSize,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
    },
  };
}

function assertAdapterLimits(adapter: GPUAdapter): void {
  const largest = largestStorageBindingBytes();
  if (
    adapter.limits.maxBufferSize < largest ||
    adapter.limits.maxStorageBufferBindingSize < largest
  ) {
    throw new Error(
      `Adapter cannot bind the largest OPT-0001 GEMM buffer (${largest} bytes)`,
    );
  }
  if (
    adapter.limits.maxComputeWorkgroupStorageSize <
      ACE_TILED_GEMM_WORKGROUP_BYTES ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 128 ||
    adapter.limits.maxComputeWorkgroupSizeX < 128
  ) {
    throw new Error("Adapter cannot run AceCorrectnessGemmKernel");
  }
}

function largestStorageBindingBytes(): number {
  return Math.max(...OPT_0001_GEMM_SHAPES.flatMap(({ shape }) => [
    shape.rows * shape.inner * BYTES_PER_F32,
    Math.ceil(shape.columns * shape.inner / 2) * BYTES_PER_F32,
    shape.rows * shape.columns * BYTES_PER_F32,
  ]));
}

function toBf16Bits(value: number): number {
  BF16_FLOAT_SCRATCH[0] = value;
  const source = BF16_BIT_SCRATCH[0]!;
  return ((source + 0x7fff + ((source >>> 16) & 1)) >>> 16) & 0xffff;
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function tflops(flops: number, milliseconds: number): number {
  return roundMetric(flops / milliseconds / 1e9);
}

function roundTiming(value: number): number {
  return Math.round(value * 1_000) / 1_000;
}

function roundMetric(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function updateProgress(message: string): void {
  const progress = document.querySelector<HTMLElement>("#progress");
  if (progress !== null) progress.textContent = message;
}

function requireStartButton(): HTMLButtonElement {
  const start = document.querySelector<HTMLButtonElement>("#start");
  if (start === null) throw new Error("Missing start button");
  return start;
}

function finish(status: "passed" | "failed", result: object): void {
  document.body.dataset.status = status;
  const resultNode = document.querySelector<HTMLPreElement>("#result");
  if (resultNode === null) throw new Error("Missing result element");
  resultNode.textContent = JSON.stringify(result);
  updateProgress(status);
  document.title = `ACE OPT-0001 GEMM profiler ${status}`;
}
