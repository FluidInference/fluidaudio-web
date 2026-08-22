import {
  ACE_TILED_GEMM_WORKGROUP_BYTES,
  AceCorrectnessGemmKernel,
  type AceGemmShape,
  planAceTiledGemm,
} from "../../src/webgpu/kernels/gemm.js";
import {
  ACE_OPT_0003_SUBGROUP_GEMM_WORKGROUP_BYTES,
  AceOpt0003SubgroupGemmKernel,
  packAceOpt0003SubgroupBf16Weight,
  planAceOpt0003SubgroupGemm,
} from "../../benchmark/opt-0003-subgroup-gemm.js";

type ShapeId = "h-to-h" | "h-to-1024" | "h-to-6144" | "6144-to-h";
type KernelId = "portable" | "subgroup";

export interface Opt0003GemmShape {
  readonly id: ShapeId;
  readonly shape: AceGemmShape;
}

export interface Opt0003SampleSummary {
  readonly count: number;
  readonly samples: readonly number[];
  readonly minimum: number;
  readonly median: number;
  readonly maximum: number;
  readonly range: number;
}

export interface Opt0003ThermalGateMetadata {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly durationSeconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
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
  readonly logicalTflops: number;
  readonly ranges: readonly RangeTiming[];
}

interface HeartbeatResult {
  readonly animationFrameCount: number;
  readonly timerTickCount: number;
  readonly maximumAnimationFrameGapMilliseconds: number;
  readonly maximumTimerGapMilliseconds: number;
}

interface OutputFingerprint {
  readonly elementCount: number;
  readonly finiteCount: number;
  readonly nonzeroCount: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly fnv1a32: string;
}

interface OutputSentinel {
  readonly index: number;
  readonly expected: number;
  readonly portable: number;
  readonly subgroup: number;
  readonly portableBitExact: boolean;
  readonly subgroupBitExact: boolean;
}

interface PairedDispatch {
  readonly label: string;
  readonly rangeCount: number;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
}

interface OwnedShapeResources {
  readonly activation: GPUBuffer;
  readonly portableWeight: GPUBuffer;
  readonly subgroupWeight: GPUBuffer;
  readonly portableOutput: GPUBuffer;
  readonly subgroupOutput: GPUBuffer;
  readonly sentinel: GPUBuffer;
  destroy(): void;
}

const HIDDEN_SIZE = 2_048;
const ROWS = 2_250;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const OUTPUT_SENTINEL_BITS = 0x7fc0_0000;
const EXPLICIT_IDLE_MILLISECONDS = 1;
const THERMAL_POLL_TOLERANCE_MILLISECONDS = 250;
const SCRATCH_FLOAT = new Float32Array(1);
const SCRATCH_BITS = new Uint32Array(SCRATCH_FLOAT.buffer);

export const OPT_0003_GEMM_SHAPES = Object.freeze([
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
] satisfies readonly Opt0003GemmShape[]);

export const OPT_0003_PAIRED_ORDERS = Object.freeze([
  Object.freeze(["portable", "subgroup"]),
  Object.freeze(["subgroup", "portable"]),
  Object.freeze(["subgroup", "portable"]),
  Object.freeze(["portable", "subgroup"]),
] satisfies readonly (readonly KernelId[])[]);

export function opt0003ActivationValue(index: number): number {
  return Math.fround((((index * 17 + 3) % 31) - 15) / 32);
}

export function opt0003WeightValue(index: number): number {
  return Math.fround((((index * 13 + 7) % 29) - 14) / 64);
}

export function summarizeOpt0003Samples(
  samples: readonly number[],
): Opt0003SampleSummary {
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

export function parseOpt0003ThermalGateMetadata(
  parameters: URLSearchParams,
): Opt0003ThermalGateMetadata {
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
    throw new Error("OPT-0003 requires the accepted notifyutil thermal source");
  }
  if (durationSeconds < 30 || observationCount < 31) {
    throw new Error("OPT-0003 requires 30 continuous nominal seconds");
  }
  if (pollMilliseconds !== 1_000) {
    throw new Error("OPT-0003 thermal polling must use 1,000 ms intervals");
  }
  if (
    maximumPollGapMilliseconds >
      pollMilliseconds + THERMAL_POLL_TOLERANCE_MILLISECONDS
  ) {
    throw new Error("OPT-0003 thermal poll gap exceeds tolerance");
  }
  if (nonNominalObservationCount !== 0) {
    throw new Error("OPT-0003 thermal gate observed non-nominal pressure");
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
        schema: "ace-opt-0003-subgroup-gemm-paired-ab-v1",
        status: "failed",
        experimentId: "OPT-0003",
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
  const thermalGate = parseOpt0003ThermalGateMetadata(
    new URL(window.location.href).searchParams,
  );
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  assertAdapter(adapter);
  const largestBinding = largestStorageBindingBytes();
  const device = await adapter.requestDevice({
    requiredFeatures: ["subgroups"],
    requiredLimits: {
      maxBufferSize: largestBinding,
      maxStorageBufferBindingSize: largestBinding,
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 128,
      maxComputeWorkgroupStorageSize: Math.max(
        ACE_TILED_GEMM_WORKGROUP_BYTES,
        ACE_OPT_0003_SUBGROUP_GEMM_WORKGROUP_BYTES,
      ),
    },
  });
  try {
    updateProgress("running bounded subgroup correctness preflight");
    const correctnessPreflight = await runCorrectnessPreflight(device);
    const shapes: unknown[] = [];
    for (const [index, fixture] of OPT_0003_GEMM_SHAPES.entries()) {
      updateProgress(
        `shape ${index + 1}/${OPT_0003_GEMM_SHAPES.length}: ${fixture.id}`,
      );
      shapes.push(await runShape(device, fixture, index));
      await yieldToPage();
    }
    const aggregate = summarizeAggregate(shapes as readonly ShapeResult[]);
    return {
      schema: "ace-opt-0003-subgroup-gemm-paired-ab-v1",
      status: "passed",
      experimentId: "OPT-0003",
      classification: "benchmark-local-candidate-no-production-gemm-change",
      recordedAt: new Date().toISOString(),
      browser: {
        userAgent: navigator.userAgent,
        page: window.location.href,
      },
      adapter: adapterIdentity(adapter),
      protocol: {
        thermalGate,
        modelProfile: "reference-bf16-subgroups",
        baselineKernel: "AceCorrectnessGemmKernel",
        candidateKernel: "AceOpt0003SubgroupGemmKernel",
        deterministicFixture: {
          id: "opt-0003-deterministic-nondegenerate-v1",
          activation: "(((index*17+3)%31)-15)/32-fp32",
          logicalWeight: "(((index*13+7)%29)-14)/64-bf16",
        },
        candidatePrepackExcludedFromSteadyTiming: true,
        warmupExecutionsPerKernelPerShape: 1,
        samplesPerKernelPerShape: OPT_0003_PAIRED_ORDERS.length,
        pairedOrders: OPT_0003_PAIRED_ORDERS.map((order) => order.join("-")),
        oneCommandBufferOutstanding: true,
        queueDrainAfterEveryCommandBuffer: true,
        queueEmptyIdleMillisecondsRequested: EXPLICIT_IDLE_MILLISECONDS,
        idleAfterFinalComputeRange: false,
        authoritativeTiming: "performance.now-wall-clock",
        heartbeatScope: "warmed-paired-samples-per-shape",
        heartbeatExcludes: Object.freeze([
          "host-fixture-generation",
          "candidate-prepack",
          "gpu-buffer-allocation-and-upload",
          "pipeline-compilation",
          "warmup",
          "readback-and-validation",
          "json-serialization",
        ]),
      },
      memory: {
        portableWorkgroupStorageBytes: ACE_TILED_GEMM_WORKGROUP_BYTES,
        subgroupWorkgroupStorageBytes:
          ACE_OPT_0003_SUBGROUP_GEMM_WORKGROUP_BYTES,
        candidatePersistentHighWaterDeltaBytesAfterLayoutReplacement: 0,
        pairedHarnessRetainsBothWeightLayouts: true,
      },
      correctnessPreflight,
      shapes,
      aggregate,
    };
  } finally {
    device.destroy();
  }
}

interface ShapeResult {
  readonly id: ShapeId;
  readonly shape: AceGemmShape;
  readonly logicalFlops: number;
  readonly portable: {
    readonly commandBufferCountPerExecution: number;
    readonly samples: readonly ExecutionTiming[];
    readonly summary: ReturnType<typeof summarizeExecutions>;
  };
  readonly subgroup: {
    readonly commandBufferCountPerExecution: number;
    readonly samples: readonly ExecutionTiming[];
    readonly summary: ReturnType<typeof summarizeExecutions>;
  };
  readonly measuredResponsiveness: HeartbeatResult;
}

async function runShape(
  device: GPUDevice,
  fixture: Opt0003GemmShape,
  shapeIndex: number,
): Promise<ShapeResult & Readonly<Record<string, unknown>>> {
  const preparation = createShapeResources(device, fixture);
  const portableKernel = AceCorrectnessGemmKernel.create(
    device,
    "reference-bf16",
  );
  const subgroupKernel = AceOpt0003SubgroupGemmKernel.create(device, {
    subgroupMinSize: 32,
    subgroupMaxSize: 32,
  });
  try {
    const portableCompileStart = performance.now();
    const portableDispatch = await portableKernel.createDispatch(
      `opt-0003-${fixture.id}-portable`,
      fixture.shape,
      {
        activation: binding(preparation.resources.activation),
        weight: binding(preparation.resources.portableWeight),
        output: binding(preparation.resources.portableOutput),
      },
    );
    const portableCompileMilliseconds = performance.now() - portableCompileStart;
    const subgroupCompileStart = performance.now();
    const subgroupDispatch = await subgroupKernel.createDispatch(
      `opt-0003-${fixture.id}-subgroup`,
      fixture.shape,
      {
        activation: binding(preparation.resources.activation),
        weight: binding(preparation.resources.subgroupWeight),
        output: binding(preparation.resources.subgroupOutput),
      },
    );
    const subgroupCompileMilliseconds = performance.now() - subgroupCompileStart;
    const logicalFlops =
      fixture.shape.rows * fixture.shape.inner * fixture.shape.columns * 2;

    updateProgress(`shape ${shapeIndex + 1}: ${fixture.id} portable warmup`);
    const portableWarmup = await executeDispatch(
      device,
      portableDispatch,
      logicalFlops,
      -1,
      "portable-subgroup",
      0,
    );
    await resetOutputs(device, preparation.resources);
    updateProgress(`shape ${shapeIndex + 1}: ${fixture.id} subgroup warmup`);
    const subgroupWarmup = await executeDispatch(
      device,
      subgroupDispatch,
      logicalFlops,
      -1,
      "portable-subgroup",
      1,
    );
    await resetOutputs(device, preparation.resources);
    await queueEmptyIdle();
    await yieldToPage();

    const samples: Record<KernelId, ExecutionTiming[]> = {
      portable: [],
      subgroup: [],
    };
    const heartbeat = startHeartbeat();
    let measuredResponsiveness: HeartbeatResult;
    try {
      for (const [roundIndex, order] of OPT_0003_PAIRED_ORDERS.entries()) {
        for (const [orderPosition, kernelId] of order.entries()) {
          updateProgress(
            `shape ${shapeIndex + 1}: ${fixture.id} round ${roundIndex + 1} ` +
              `${kernelId} (${orderPosition + 1}/2)`,
          );
          samples[kernelId].push(await executeDispatch(
            device,
            kernelId === "portable" ? portableDispatch : subgroupDispatch,
            logicalFlops,
            roundIndex,
            order.join("-"),
            orderPosition,
          ));
          await yieldToPage();
        }
      }
    } finally {
      measuredResponsiveness = heartbeat.stop();
    }

    updateProgress(`shape ${shapeIndex + 1}: ${fixture.id} exact validation`);
    await resetOutputs(device, preparation.resources);
    await executeDispatch(
      device,
      portableDispatch,
      logicalFlops,
      -2,
      "validation",
      0,
    );
    await executeDispatch(
      device,
      subgroupDispatch,
      logicalFlops,
      -2,
      "validation",
      1,
    );
    const portableOutput = await readOutput(
      device,
      fixture,
      preparation.resources.portableOutput,
      "portable",
    );
    const subgroupOutput = await readOutput(
      device,
      fixture,
      preparation.resources.subgroupOutput,
      "subgroup",
    );
    const correctness = validateOutputs(
      fixture,
      portableOutput,
      subgroupOutput,
    );
    const portableSummary = summarizeExecutions(samples.portable);
    const subgroupSummary = summarizeExecutions(samples.subgroup);
    return {
      id: fixture.id,
      shape: fixture.shape,
      logicalFlops,
      logicalBytes: {
        activation: fixture.shape.rows * fixture.shape.inner * FLOAT32_BYTES,
        packedBf16Weight:
          Math.ceil(fixture.shape.columns * fixture.shape.inner / 2) *
            FLOAT32_BYTES,
        output: fixture.shape.rows * fixture.shape.columns * FLOAT32_BYTES,
      },
      preparation: {
        activationAndLogicalWeightMilliseconds:
          preparation.activationAndLogicalWeightMilliseconds,
        candidateHostPrepackMilliseconds:
          preparation.candidateHostPrepackMilliseconds,
        gpuBufferUploadMilliseconds: preparation.gpuBufferUploadMilliseconds,
        candidatePrepackInputBytes: preparation.packedWeightBytes,
        candidatePrepackOutputBytes: preparation.packedWeightBytes,
        candidateTransientHostBytes: preparation.packedWeightBytes * 2,
      },
      compilation: {
        portableMilliseconds: portableCompileMilliseconds,
        subgroupMilliseconds: subgroupCompileMilliseconds,
      },
      warmup: {
        portable: portableWarmup,
        subgroup: subgroupWarmup,
      },
      portable: {
        commandBufferCountPerExecution: portableDispatch.rangeCount,
        samples: Object.freeze(samples.portable),
        summary: portableSummary,
      },
      subgroup: {
        commandBufferCountPerExecution: subgroupDispatch.rangeCount,
        samples: Object.freeze(samples.subgroup),
        summary: subgroupSummary,
      },
      delta: {
        medianActiveWallSpeedup:
          portableSummary.activeWallMilliseconds.median /
          subgroupSummary.activeWallMilliseconds.median,
        medianLogicalTflopsRatio:
          subgroupSummary.logicalTflops.median /
          portableSummary.logicalTflops.median,
        subgroupRoundWins: OPT_0003_PAIRED_ORDERS.filter((_order, roundIndex) =>
          samples.subgroup[roundIndex]!.activeWallMilliseconds <
            samples.portable[roundIndex]!.activeWallMilliseconds
        ).length,
      },
      measuredResponsiveness,
      correctness,
    };
  } finally {
    portableKernel.destroy();
    subgroupKernel.destroy();
    preparation.resources.destroy();
  }
}

async function runCorrectnessPreflight(device: GPUDevice): Promise<unknown> {
  const cases = [
    Object.freeze({
      id: "row-tail-with-bias",
      shape: Object.freeze({ rows: 33, inner: 32, columns: 128 }),
      activation(index: number): number {
        return opt0003ActivationValue(index);
      },
      weight(index: number): number {
        return opt0003WeightValue(index);
      },
      bias(column: number): number {
        return Math.fround(((column % 7) - 3) / 64);
      },
    }),
    Object.freeze({
      id: "adversarial-cancellation",
      shape: Object.freeze({ rows: 33, inner: 32, columns: 128 }),
      activation(index: number): number {
        const row = Math.floor(index / 32);
        const inner = index % 32;
        if (row % 2 === 0) {
          if (inner === 0) return Math.fround(-68.06752014160156);
          if (inner === 1) return Math.fround(12.192401885986328);
          return 0;
        }
        const sequence = [16_777_216, 1, -16_777_216, 0.5] as const;
        return inner < sequence.length ? sequence[inner]! : 0;
      },
      weight(index: number): number {
        const column = Math.floor(index / 32);
        const inner = index % 32;
        if (column % 2 !== 0) return 1;
        if (inner === 0) return 1;
        if (inner === 1) return 5.5625;
        return 0;
      },
      bias: undefined,
    }),
  ] as const;
  const results = [];
  for (const fixture of cases) {
    const { shape } = fixture;
    const activationValues = Float32Array.from(
      { length: shape.rows * shape.inner },
      (_, index) => fixture.activation(index),
    );
    const weightValues = Float32Array.from(
      { length: shape.columns * shape.inner },
      (_, index) => fixture.weight(index),
    );
    const portableWeightWords = packBf16Values(weightValues);
    const subgroupWeightWords = packAceOpt0003SubgroupBf16Weight(
      portableWeightWords,
      shape,
    );
    const biasValues = fixture.bias === undefined
      ? undefined
      : Float32Array.from(
          { length: shape.columns },
          (_, column) => fixture.bias(column),
        );
    const biasWords = biasValues === undefined
      ? undefined
      : packBf16Values(biasValues);
    const outputElements = shape.rows * shape.columns;
    const activation = mappedStorageBuffer(
      device,
      `opt-0003-preflight-${fixture.id}-activation`,
      activationValues,
    );
    const portableWeight = mappedStorageBuffer(
      device,
      `opt-0003-preflight-${fixture.id}-portable-weight`,
      portableWeightWords,
    );
    const subgroupWeight = mappedStorageBuffer(
      device,
      `opt-0003-preflight-${fixture.id}-subgroup-weight`,
      subgroupWeightWords,
    );
    const bias = biasWords === undefined
      ? undefined
      : mappedStorageBuffer(
          device,
          `opt-0003-preflight-${fixture.id}-bias`,
          biasWords,
        );
    const portableOutput = mappedSentinelOutput(
      device,
      `opt-0003-preflight-${fixture.id}-portable-output`,
      outputElements,
    );
    const subgroupOutput = mappedSentinelOutput(
      device,
      `opt-0003-preflight-${fixture.id}-subgroup-output`,
      outputElements,
    );
    const portableKernel = AceCorrectnessGemmKernel.create(
      device,
      "reference-bf16",
    );
    const subgroupKernel = AceOpt0003SubgroupGemmKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    try {
      const portableDispatch = await portableKernel.createDispatch(
        `opt-0003-preflight-${fixture.id}-portable`,
        shape,
        {
          activation: binding(activation),
          weight: binding(portableWeight),
          output: binding(portableOutput),
          ...(bias === undefined ? {} : { bias: binding(bias) }),
        },
      );
      const subgroupDispatch = await subgroupKernel.createDispatch(
        `opt-0003-preflight-${fixture.id}-subgroup`,
        shape,
        {
          activation: binding(activation),
          weight: binding(subgroupWeight),
          output: binding(subgroupOutput),
          ...(bias === undefined ? {} : { bias: binding(bias) }),
        },
      );
      const logicalFlops = shape.rows * shape.inner * shape.columns * 2;
      const portableExecution = await executeDispatch(
        device,
        portableDispatch,
        logicalFlops,
        -3,
        "correctness-preflight",
        0,
      );
      const subgroupExecution = await executeDispatch(
        device,
        subgroupDispatch,
        logicalFlops,
        -3,
        "correctness-preflight",
        1,
      );
      const browserFixture = { id: fixture.id, shape };
      const portableValues = await readOutput(
        device,
        browserFixture,
        portableOutput,
        "portable",
      );
      const subgroupValues = await readOutput(
        device,
        browserFixture,
        subgroupOutput,
        "subgroup",
      );
      const portableBits = new Uint32Array(portableValues.buffer);
      const subgroupBits = new Uint32Array(subgroupValues.buffer);
      let bitMismatchCount = 0;
      let contractedCpuMismatchCount = 0;
      let separatelyRoundedCpuMismatchCount = 0;
      for (let index = 0; index < outputElements; index += 1) {
        const row = Math.floor(index / shape.columns);
        const column = index % shape.columns;
        let contractedExpected = 0;
        let separatelyRoundedExpected = 0;
        for (let inner = 0; inner < shape.inner; inner += 1) {
          const activationValue =
            activationValues[row * shape.inner + inner]!;
          const weightValue = weightValues[column * shape.inner + inner]!;
          contractedExpected = Math.fround(
            contractedExpected + activationValue * weightValue,
          );
          separatelyRoundedExpected = Math.fround(
            separatelyRoundedExpected +
              Math.fround(activationValue * weightValue),
          );
        }
        if (biasValues !== undefined) {
          contractedExpected = Math.fround(
            contractedExpected + biasValues[column]!,
          );
          separatelyRoundedExpected = Math.fround(
            separatelyRoundedExpected + biasValues[column]!,
          );
        }
        const contractedExpectedBits = float32Bits(contractedExpected);
        const separatelyRoundedExpectedBits = float32Bits(
          separatelyRoundedExpected,
        );
        if (portableBits[index] !== subgroupBits[index]) bitMismatchCount += 1;
        if (
          portableBits[index] !== contractedExpectedBits ||
          subgroupBits[index] !== contractedExpectedBits
        ) {
          contractedCpuMismatchCount += 1;
        }
        if (
          portableBits[index] !== separatelyRoundedExpectedBits ||
          subgroupBits[index] !== separatelyRoundedExpectedBits
        ) {
          separatelyRoundedCpuMismatchCount += 1;
        }
      }
      if (bitMismatchCount !== 0 || contractedCpuMismatchCount !== 0) {
        throw new Error(
          `${fixture.id} preflight mismatches: A/B=${bitMismatchCount}, ` +
            `contracted CPU=${contractedCpuMismatchCount}`,
        );
      }
      let adversarialDiagnostics: Readonly<Record<string, unknown>> | undefined;
      if (fixture.id === "adversarial-cancellation") {
        if (separatelyRoundedCpuMismatchCount === 0) {
          throw new Error(
            "adversarial preflight did not discriminate contraction rounding",
          );
        }
        const fusedIndex = 0;
        const cancellationIndex = shape.columns + 1;
        const fusedContractedBits = 0xbe7d_3830;
        const fusedSeparatelyRoundedBits = 0xbe7d_3800;
        const sourceOrderCancellationBits = float32Bits(0.5);
        const reassociatedCancellationBits = float32Bits(1.5);
        if (
          portableBits[fusedIndex] !== fusedContractedBits ||
          subgroupBits[fusedIndex] !== fusedContractedBits ||
          portableBits[cancellationIndex] !== sourceOrderCancellationBits ||
          subgroupBits[cancellationIndex] !== sourceOrderCancellationBits ||
          sourceOrderCancellationBits === reassociatedCancellationBits
        ) {
          throw new Error(
            "adversarial preflight did not preserve contracted source order",
          );
        }
        adversarialDiagnostics = Object.freeze({
          fusedDiscriminantOutputIndex: fusedIndex,
          fusedContractedBits: hex32(fusedContractedBits),
          fusedSeparatelyRoundedBits: hex32(fusedSeparatelyRoundedBits),
          cancellationOutputIndex: cancellationIndex,
          sourceOrderCancellation: 0.5,
          sourceOrderCancellationBits: hex32(sourceOrderCancellationBits),
          reassociatedCancellation: 1.5,
          reassociatedCancellationBits: hex32(reassociatedCancellationBits),
        });
      }
      const portableFingerprint = fingerprint(portableValues, portableBits);
      const subgroupFingerprint = fingerprint(subgroupValues, subgroupBits);
      if (
        portableFingerprint.finiteCount !== outputElements ||
        subgroupFingerprint.finiteCount !== outputElements
      ) {
        throw new Error(`${fixture.id} preflight left an unwritten sentinel`);
      }
      results.push(Object.freeze({
        id: fixture.id,
        shape,
        hasBias: bias !== undefined,
        rowTail: shape.rows % 32,
        fixtureClass: fixture.id === "adversarial-cancellation"
          ? "large-opposite-terms-with-small-residual"
          : "deterministic-nondegenerate-with-bf16-bias",
        fullDomainFinite: true,
        bitIdentical: true,
        bitMismatchCount,
        independentContractedCpuBitExact: true,
        contractedCpuMismatchCount,
        separatelyRoundedCpuMismatchCount,
        ...(adversarialDiagnostics === undefined
          ? {}
          : { adversarialDiagnostics }),
        portableFingerprint,
        subgroupFingerprint,
        portableExecution,
        subgroupExecution,
      }));
    } finally {
      portableKernel.destroy();
      subgroupKernel.destroy();
      activation.destroy();
      portableWeight.destroy();
      subgroupWeight.destroy();
      bias?.destroy();
      portableOutput.destroy();
      subgroupOutput.destroy();
    }
  }
  return Object.freeze({
    classification: "bounded-actual-gpu-correctness-outside-timing",
    cases: Object.freeze(results),
  });
}

function createShapeResources(
  device: GPUDevice,
  fixture: Opt0003GemmShape,
): {
  readonly resources: OwnedShapeResources;
  readonly activationAndLogicalWeightMilliseconds: number;
  readonly candidateHostPrepackMilliseconds: number;
  readonly gpuBufferUploadMilliseconds: number;
  readonly packedWeightBytes: number;
} {
  const { shape } = fixture;
  const activationElements = shape.rows * shape.inner;
  const weightElements = shape.columns * shape.inner;
  const outputElements = shape.rows * shape.columns;
  const logicalStarted = performance.now();
  const activationValues = new Float32Array(activationElements);
  for (let index = 0; index < activationValues.length; index += 1) {
    activationValues[index] = opt0003ActivationValue(index);
  }
  const portableWeightWords = new Uint32Array(Math.ceil(weightElements / 2));
  for (let word = 0; word < portableWeightWords.length; word += 1) {
    const lowIndex = word * 2;
    const low = toBf16Bits(opt0003WeightValue(lowIndex));
    const high = lowIndex + 1 < weightElements
      ? toBf16Bits(opt0003WeightValue(lowIndex + 1))
      : 0;
    portableWeightWords[word] = low | (high << 16);
  }
  const activationAndLogicalWeightMilliseconds = performance.now() - logicalStarted;
  const prepackStarted = performance.now();
  const subgroupWeightWords = packAceOpt0003SubgroupBf16Weight(
    portableWeightWords,
    shape,
  );
  const candidateHostPrepackMilliseconds = performance.now() - prepackStarted;
  if (subgroupWeightWords.byteLength !== portableWeightWords.byteLength) {
    throw new Error(`${fixture.id} subgroup prepack changed BF16 payload bytes`);
  }

  const owned: GPUBuffer[] = [];
  const makeMapped = (
    label: string,
    bytes: number,
    usage: GPUBufferUsageFlags,
  ): GPUBuffer => {
    const buffer = device.createBuffer({
      label,
      size: bytes,
      usage,
      mappedAtCreation: true,
    });
    owned.push(buffer);
    return buffer;
  };
  const uploadStarted = performance.now();
  try {
    const activation = makeMapped(
      `opt-0003-${fixture.id}-activation`,
      activationValues.byteLength,
      GPUBufferUsage.STORAGE,
    );
    new Float32Array(activation.getMappedRange()).set(activationValues);
    activation.unmap();
    const portableWeight = makeMapped(
      `opt-0003-${fixture.id}-portable-weight`,
      portableWeightWords.byteLength,
      GPUBufferUsage.STORAGE,
    );
    new Uint32Array(portableWeight.getMappedRange()).set(portableWeightWords);
    portableWeight.unmap();
    const subgroupWeight = makeMapped(
      `opt-0003-${fixture.id}-subgroup-weight`,
      subgroupWeightWords.byteLength,
      GPUBufferUsage.STORAGE,
    );
    new Uint32Array(subgroupWeight.getMappedRange()).set(subgroupWeightWords);
    subgroupWeight.unmap();
    const outputBytes = outputElements * FLOAT32_BYTES;
    const portableOutput = makeMapped(
      `opt-0003-${fixture.id}-portable-output`,
      outputBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    new Uint32Array(portableOutput.getMappedRange()).fill(OUTPUT_SENTINEL_BITS);
    portableOutput.unmap();
    const subgroupOutput = makeMapped(
      `opt-0003-${fixture.id}-subgroup-output`,
      outputBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    new Uint32Array(subgroupOutput.getMappedRange()).fill(OUTPUT_SENTINEL_BITS);
    subgroupOutput.unmap();
    const sentinel = makeMapped(
      `opt-0003-${fixture.id}-sentinel`,
      outputBytes,
      GPUBufferUsage.COPY_SRC,
    );
    new Uint32Array(sentinel.getMappedRange()).fill(OUTPUT_SENTINEL_BITS);
    sentinel.unmap();
    const gpuBufferUploadMilliseconds = performance.now() - uploadStarted;
    return {
      resources: Object.freeze({
        activation,
        portableWeight,
        subgroupWeight,
        portableOutput,
        subgroupOutput,
        sentinel,
        destroy(): void {
          for (const buffer of owned) buffer.destroy();
        },
      }),
      activationAndLogicalWeightMilliseconds,
      candidateHostPrepackMilliseconds,
      gpuBufferUploadMilliseconds,
      packedWeightBytes: portableWeightWords.byteLength,
    };
  } catch (error) {
    for (const buffer of owned) {
      if (buffer.mapState === "mapped") buffer.unmap();
      buffer.destroy();
    }
    throw error;
  }
}

function mappedStorageBuffer(
  device: GPUDevice,
  label: string,
  values: Float32Array | Uint32Array,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: values.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    new Uint8Array(buffer.getMappedRange()).set(
      new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
    );
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

function packBf16Values(values: Float32Array): Uint32Array<ArrayBuffer> {
  const packed = new Uint32Array(Math.ceil(values.length / 2));
  for (let word = 0; word < packed.length; word += 1) {
    const lowIndex = word * 2;
    const low = toBf16Bits(values[lowIndex]!);
    const high = lowIndex + 1 < values.length
      ? toBf16Bits(values[lowIndex + 1]!)
      : 0;
    packed[word] = low | (high << 16);
  }
  return packed;
}

async function resetOutputs(
  device: GPUDevice,
  resources: OwnedShapeResources,
): Promise<void> {
  const encoder = device.createCommandEncoder({ label: "opt-0003-output-reset" });
  encoder.copyBufferToBuffer(
    resources.sentinel,
    0,
    resources.portableOutput,
    0,
    resources.portableOutput.size,
  );
  encoder.copyBufferToBuffer(
    resources.sentinel,
    0,
    resources.subgroupOutput,
    0,
    resources.subgroupOutput.size,
  );
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function executeDispatch(
  device: GPUDevice,
  dispatch: PairedDispatch,
  logicalFlops: number,
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
    logicalTflops: logicalFlops / activeWallMilliseconds / 1e9,
    ranges: Object.freeze(ranges),
  });
}

async function readOutput(
  device: GPUDevice,
  fixture: Readonly<{ readonly id: string; readonly shape: AceGemmShape }>,
  output: GPUBuffer,
  kernelId: KernelId,
): Promise<Float32Array<ArrayBuffer>> {
  const bytes = fixture.shape.rows * fixture.shape.columns * FLOAT32_BYTES;
  const readback = device.createBuffer({
    label: `opt-0003-${fixture.id}-${kernelId}-readback`,
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder({
      label: `opt-0003-${fixture.id}-${kernelId}-readback-encoder`,
    });
    encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    const result = new Float32Array(bytes / FLOAT32_BYTES);
    result.set(new Float32Array(readback.getMappedRange()));
    return result;
  } finally {
    if (mapped) readback.unmap();
    readback.destroy();
  }
}

function validateOutputs(
  fixture: Opt0003GemmShape,
  portable: Float32Array,
  subgroup: Float32Array,
): Readonly<Record<string, unknown>> {
  if (portable.length !== subgroup.length) {
    throw new Error(`${fixture.id} output lengths differ`);
  }
  const portableBits = new Uint32Array(
    portable.buffer,
    portable.byteOffset,
    portable.length,
  );
  const subgroupBits = new Uint32Array(
    subgroup.buffer,
    subgroup.byteOffset,
    subgroup.length,
  );
  let bitMismatchCount = 0;
  for (let index = 0; index < portable.length; index += 1) {
    if (!Number.isFinite(portable[index]!) || !Number.isFinite(subgroup[index]!)) {
      throw new Error(`${fixture.id} left a non-finite output at ${index}`);
    }
    if (portable[index] === 0 || subgroup[index] === 0) {
      throw new Error(`${fixture.id} produced a degenerate zero at ${index}`);
    }
    if (portableBits[index] !== subgroupBits[index]) bitMismatchCount += 1;
  }
  if (bitMismatchCount !== 0) {
    throw new Error(`${fixture.id} has ${bitMismatchCount} full-domain bit mismatches`);
  }
  const sentinels: OutputSentinel[] = [];
  for (const index of opt0003OutputSentinelIndices(fixture.shape)) {
    const expected = opt0003ExpectedOutputValue(fixture.shape, index);
    const expectedBits = float32Bits(expected);
    const portableBitExact = portableBits[index] === expectedBits;
    const subgroupBitExact = subgroupBits[index] === expectedBits;
    if (!portableBitExact || !subgroupBitExact) {
      throw new Error(
        `${fixture.id} CPU sentinel ${index} is not bit exact: ` +
          `${portable[index]} / ${subgroup[index]} != ${expected}`,
      );
    }
    sentinels.push(Object.freeze({
      index,
      expected,
      portable: portable[index]!,
      subgroup: subgroup[index]!,
      portableBitExact,
      subgroupBitExact,
    }));
  }
  return Object.freeze({
    outputPrefill: "quiet-NaN-u32-sentinel",
    fullDomainFinite: true,
    fullDomainNonzero: true,
    bitIdentical: true,
    bitMismatchCount,
    portableFingerprint: fingerprint(portable, portableBits),
    subgroupFingerprint: fingerprint(subgroup, subgroupBits),
    cpuSentinelsBitExact: true,
    sentinels: Object.freeze(sentinels),
  });
}

function fingerprint(
  values: Float32Array,
  bits: Uint32Array,
): OutputFingerprint {
  let finiteCount = 0;
  let nonzeroCount = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  let hash = 0x811c_9dc5;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (Number.isFinite(value)) finiteCount += 1;
    if (value !== 0) nonzeroCount += 1;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    hash = Math.imul(hash ^ bits[index]!, 0x0100_0193) >>> 0;
  }
  return Object.freeze({
    elementCount: values.length,
    finiteCount,
    nonzeroCount,
    minimum,
    maximum,
    fnv1a32: hash.toString(16).padStart(8, "0"),
  });
}

export function opt0003OutputSentinelIndices(
  shape: AceGemmShape,
): readonly number[] {
  const plan = planAceTiledGemm(shape);
  const subgroupPlan = planAceOpt0003SubgroupGemm(shape);
  const middleRow = Math.floor(shape.rows / 2);
  const candidates = [
    0,
    Math.min(shape.columns - 1, 127),
    Math.min(shape.columns - 1, 128),
    shape.columns - 1,
    middleRow * shape.columns,
    middleRow * shape.columns + Math.floor(shape.columns / 2),
    middleRow * shape.columns + shape.columns - 1,
    (shape.rows - 1) * shape.columns,
    shape.rows * shape.columns - 1,
    (1_337 * shape.columns + 811) % (shape.rows * shape.columns),
  ];
  for (const range of plan.outputRanges) {
    candidates.push(
      range.firstOutput,
      Math.min(
        shape.rows * shape.columns - 1,
        range.firstOutput + range.outputCount - 1,
      ),
    );
  }
  for (const range of subgroupPlan.outputRanges) {
    candidates.push(
      range.firstOutput,
      Math.min(
        shape.rows * shape.columns - 1,
        range.firstOutput + range.outputCount - 1,
      ),
    );
  }
  return Object.freeze([...new Set(candidates)]);
}

export function opt0003ExpectedOutputValue(
  shape: AceGemmShape,
  outputIndex: number,
): number {
  const row = Math.floor(outputIndex / shape.columns);
  const column = outputIndex % shape.columns;
  let sum = 0;
  for (let inner = 0; inner < shape.inner; inner += 1) {
    const activation = opt0003ActivationValue(row * shape.inner + inner);
    const weight = opt0003WeightValue(column * shape.inner + inner);
    sum = Math.fround(sum + Math.fround(activation * weight));
  }
  return sum;
}

function summarizeExecutions(samples: readonly ExecutionTiming[]): {
  readonly wallMilliseconds: Opt0003SampleSummary;
  readonly activeWallMilliseconds: Opt0003SampleSummary;
  readonly encodeMilliseconds: Opt0003SampleSummary;
  readonly submitMilliseconds: Opt0003SampleSummary;
  readonly drainMilliseconds: Opt0003SampleSummary;
  readonly explicitIdleMilliseconds: Opt0003SampleSummary;
  readonly maximumSingleDrainMilliseconds: Opt0003SampleSummary;
  readonly logicalTflops: Opt0003SampleSummary;
} {
  if (samples.length === 0) throw new RangeError("execution samples missing");
  const summarize = (select: (sample: ExecutionTiming) => number) =>
    summarizeOpt0003Samples(samples.map(select));
  return Object.freeze({
    wallMilliseconds: summarize((sample) => sample.wallMilliseconds),
    activeWallMilliseconds: summarize((sample) => sample.activeWallMilliseconds),
    encodeMilliseconds: summarize((sample) => sample.encodeMilliseconds),
    submitMilliseconds: summarize((sample) => sample.submitMilliseconds),
    drainMilliseconds: summarize((sample) => sample.drainMilliseconds),
    explicitIdleMilliseconds: summarize((sample) =>
      sample.explicitIdleMilliseconds
    ),
    maximumSingleDrainMilliseconds: summarize((sample) =>
      sample.maximumSingleDrainMilliseconds
    ),
    logicalTflops: summarize((sample) => sample.logicalTflops),
  });
}

function summarizeAggregate(shapes: readonly ShapeResult[]): unknown {
  const logicalFlops = shapes.reduce(
    (total, shape) => total + shape.logicalFlops,
    0,
  );
  const rounds = OPT_0003_PAIRED_ORDERS.map((_order, roundIndex) => {
    const portableActiveMilliseconds = shapes.reduce((total, shape) =>
      total + shape.portable.samples.find((sample) =>
        sample.roundIndex === roundIndex
      )!.activeWallMilliseconds, 0);
    const subgroupActiveMilliseconds = shapes.reduce((total, shape) =>
      total + shape.subgroup.samples.find((sample) =>
        sample.roundIndex === roundIndex
      )!.activeWallMilliseconds, 0);
    const portableLogicalTflops =
      logicalFlops / portableActiveMilliseconds / 1e9;
    const subgroupLogicalTflops =
      logicalFlops / subgroupActiveMilliseconds / 1e9;
    return Object.freeze({
      roundIndex,
      portableActiveMilliseconds,
      subgroupActiveMilliseconds,
      portableLogicalTflops,
      subgroupLogicalTflops,
      subgroupWon: subgroupActiveMilliseconds < portableActiveMilliseconds,
    });
  });
  const portableActive = summarizeOpt0003Samples(
    rounds.map((round) => round.portableActiveMilliseconds),
  );
  const subgroupActive = summarizeOpt0003Samples(
    rounds.map((round) => round.subgroupActiveMilliseconds),
  );
  const portableTflops = summarizeOpt0003Samples(
    rounds.map((round) => round.portableLogicalTflops),
  );
  const subgroupTflops = summarizeOpt0003Samples(
    rounds.map((round) => round.subgroupLogicalTflops),
  );
  return Object.freeze({
    shapeWeighting: "sum-logical-flops-over-sum-active-wall",
    logicalFlopsPerAggregateRound: logicalFlops,
    rounds: Object.freeze(rounds),
    portableActiveMilliseconds: portableActive,
    subgroupActiveMilliseconds: subgroupActive,
    portableLogicalTflops: portableTflops,
    subgroupLogicalTflops: subgroupTflops,
    medianActiveWallSpeedup:
      portableActive.median / subgroupActive.median,
    medianLogicalTflopsRatio: subgroupTflops.median / portableTflops.median,
    subgroupRoundWins: rounds.filter((round) => round.subgroupWon).length,
    maximumCandidateDrainMilliseconds: Math.max(...shapes.map((shape) =>
      shape.subgroup.summary.maximumSingleDrainMilliseconds.maximum
    )),
    maximumAnimationFrameGapMilliseconds: Math.max(...shapes.map((shape) =>
      shape.measuredResponsiveness.maximumAnimationFrameGapMilliseconds
    )),
    maximumTimerGapMilliseconds: Math.max(...shapes.map((shape) =>
      shape.measuredResponsiveness.maximumTimerGapMilliseconds
    )),
  });
}

function adapterIdentity(adapter: GPUAdapter): Readonly<Record<string, unknown>> {
  const { info, limits } = adapter;
  return Object.freeze({
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
    limits: Object.freeze({
      maxBufferSize: limits.maxBufferSize,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup:
        limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
    }),
  });
}

function assertAdapter(adapter: GPUAdapter): void {
  const largest = largestStorageBindingBytes();
  if (!adapter.features.has("subgroups")) {
    throw new Error("OPT-0003 requires the WebGPU subgroups feature");
  }
  if (
    adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32
  ) {
    throw new Error(
      `OPT-0003 requires fixed subgroup size 32, observed ` +
        `${String(adapter.info.subgroupMinSize)}..` +
        `${String(adapter.info.subgroupMaxSize)}`,
    );
  }
  if (
    adapter.limits.maxBufferSize < largest ||
    adapter.limits.maxStorageBufferBindingSize < largest ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 128 ||
    adapter.limits.maxComputeWorkgroupSizeX < 128 ||
    adapter.limits.maxComputeWorkgroupStorageSize <
      ACE_TILED_GEMM_WORKGROUP_BYTES
  ) {
    throw new Error("Adapter cannot satisfy the OPT-0003 paired GEMM contract");
  }
}

function largestStorageBindingBytes(): number {
  return Math.max(...OPT_0003_GEMM_SHAPES.flatMap(({ shape }) => [
    shape.rows * shape.inner * FLOAT32_BYTES,
    Math.ceil(shape.columns * shape.inner / 2) * FLOAT32_BYTES,
    shape.rows * shape.columns * FLOAT32_BYTES,
  ]));
}

function requiredNumber(parameters: URLSearchParams, name: string): number {
  const raw = parameters.get(name);
  if (raw === null || raw.length === 0) {
    throw new Error(`Missing OPT-0003 ${name}`);
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`Invalid OPT-0003 ${name}`);
  }
  return value;
}

function toBf16Bits(value: number): number {
  SCRATCH_FLOAT[0] = value;
  const source = SCRATCH_BITS[0]!;
  return ((source + 0x7fff + ((source >>> 16) & 1)) >>> 16) & 0xffff;
}

function float32Bits(value: number): number {
  SCRATCH_FLOAT[0] = value;
  return SCRATCH_BITS[0]!;
}

function hex32(value: number): string {
  return `0x${(value >>> 0).toString(16).padStart(8, "0")}`;
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
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
  const elapsed = performance.now() - started;
  // The production invariant is the real one-millisecond timer request.
  // Retain the observed wall time, but do not reject a 0.9 ms quantized clock
  // reading (Chrome exposed 0.899999976 ms for one accepted timer interval).
  return elapsed;
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
