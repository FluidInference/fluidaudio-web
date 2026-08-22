/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import gemmKernelSource from "../../src/webgpu/kernels/gemm.ts?raw";
import opt0083KernelSource from
  "../../src/webgpu/kernels/planner-low-row-bf16-gemv.ts?raw";
import {
  AceCorrectnessGemmKernel,
  aceCorrectnessGemmWgsl,
  type AceGemmDispatch,
  type AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";
import {
  ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID,
  ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID,
  AceOpt0083PlannerDirectLowRowBf16GemvKernel,
  AceOpt0083PlannerLowRowBf16GemvKernel,
  aceOpt0083PlannerDirectLowRowBf16GemvWgsl,
  aceOpt0083PlannerLowRowBf16GemvWgsl,
  planAceOpt0083PlannerDirectLowRowBf16Gemv,
  planAceOpt0083PlannerLowRowBf16Gemv,
} from "../../src/webgpu/kernels/planner-low-row-bf16-gemv.js";

type Arm = "A" | "B" | "C";
type RowCount = 1 | 2;

interface ThermalLaunch {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly command: "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly traceStartedAtEpochMilliseconds: number;
  readonly gateStartedAtEpochMilliseconds: number;
  readonly gateCompletedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: 1_000;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly missingObservationCount: 0;
  readonly readyToGateDelayMilliseconds: number;
  readonly launchDelayMilliseconds: number;
}

interface RoleSpec {
  readonly id: "query" | "key" | "value" | "attention-output" |
    "gate" | "up" | "down";
  readonly ordinal: number;
  readonly inner: 1_024 | 2_048 | 3_072;
  readonly columns: 1_024 | 2_048 | 3_072;
  readonly weightBytes: number;
  readonly weightOffset: number;
}

interface DiagnosticSpec {
  readonly id: string;
  readonly inner: number;
  readonly columns: number;
  readonly weightBytes: number;
  readonly source: "layer" | "tied-head-tail";
  readonly roleIndex: number | null;
}

interface GuardedOutput {
  readonly label: string;
  readonly rows: RowCount;
  readonly columns: number;
  readonly outputElements: number;
  readonly outputBytes: number;
  readonly totalBytes: number;
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly prefill: GPUBuffer;
  readonly readback: GPUBuffer;
}

interface OutputSnapshot {
  readonly words: Uint32Array<ArrayBuffer>;
  readonly sha256: string;
  readonly nonFiniteCount: number;
  readonly qNaNPrefillCount: number;
  readonly prefixGuardIntact: boolean;
  readonly suffixGuardIntact: boolean;
  readonly firstOutputWritten: boolean;
  readonly lastOutputWritten: boolean;
}

interface WeightSet {
  readonly index: number;
  readonly buffer: GPUBuffer;
  readonly bindings: readonly GPUBufferBinding[];
  readonly signature: Readonly<Record<string, unknown>>;
}

interface DispatchSet {
  readonly weightSetIndex: number;
  readonly rows: RowCount;
  readonly arms: Readonly<Record<Arm, readonly AceGemmDispatch[]>>;
}

interface TailDispatchSet {
  readonly rows: RowCount;
  readonly arms: Readonly<Record<Arm, AceGemmDispatch>>;
}

interface TimestampSample {
  readonly roundIndex: number | null;
  readonly weightSetIndex: number | null;
  readonly rows: RowCount;
  readonly arm: Arm;
  readonly scope: "complete-layer" | "shape-diagnostic";
  readonly shapeId: string | null;
  readonly submitAtPerformanceMilliseconds: number;
  readonly fenceAtPerformanceMilliseconds: number;
  readonly wallMilliseconds: number;
  readonly timestampBeginNanoseconds: string;
  readonly timestampEndNanoseconds: string;
  readonly timestampValid: boolean;
  readonly timestampUnavailableReason: string | null;
  readonly gpuElapsedNanoseconds: number | null;
  readonly gpuMilliseconds: number | null;
  readonly gpuToWallRatio: number | null;
  readonly packedWeightBytes: number;
  readonly effectivePackedWeightBandwidthGbPerSecond: number | null;
  readonly effectivePackedWeightWallBandwidthGbPerSecond: number;
  readonly validMultiplyAdds: number;
  readonly scheduledMultiplyAdds: number;
  readonly validGpuTflops: number | null;
  readonly scheduledGpuTflops: number | null;
  readonly commandBufferCount: 1;
  readonly computePassCount: 1;
  readonly queueDrainCount: 1;
  readonly timestampResolveCount: 1;
  readonly timestampCopyCount: 1;
}

interface PreparedHarness {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly kernels: Readonly<{
    readonly A: AceCorrectnessGemmKernel;
    readonly B: AceOpt0083PlannerDirectLowRowBf16GemvKernel;
    readonly C: AceOpt0083PlannerLowRowBf16GemvKernel;
  }>;
  readonly weightSets: readonly WeightSet[];
  readonly tailWeight: GPUBuffer;
  readonly activations: Readonly<Record<string, GPUBuffer>>;
  readonly outputs: Readonly<Record<string, readonly GuardedOutput[]>>;
  readonly tailOutputs: Readonly<Record<string, GuardedOutput>>;
  readonly dispatchSets: readonly DispatchSet[];
  readonly tailDispatchSets: Readonly<Record<string, TailDispatchSet>>;
  readonly querySet: GPUQuerySet;
  readonly queryResolve: GPUBuffer;
  readonly queryReadback: GPUBuffer;
  readonly identity: Readonly<Record<string, unknown>>;
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly rejectionContracts: Readonly<Record<string, unknown>>;
  readonly warmup: Readonly<Record<string, unknown>>;
  readonly readyAtEpochMilliseconds: number;
  readonly uncapturedErrors: string[];
  readonly deviceLosses: string[];
  cleanup(): Promise<Readonly<Record<string, unknown>>>;
}

type IncomingMessage =
  | Readonly<{ readonly type: "prepare" }>
  | Readonly<{ readonly type: "run"; readonly thermalLaunch: ThermalLaunch }>
  | Readonly<{ readonly type: "cancel" }>;

const EXPERIMENT_ID = "OPT-0083" as const;
const RECEIPT_SCHEMA =
  "ace-opt-0083-planner-low-row-bf16-gemv-primitive-v1";
const ALLOCATION_BASELINE_COMMIT =
  "552e977be6b1b5c8b01c346d4aeaa7f63c0edbf2";
const WEIGHT_SET_COUNT = 8;
const TIED_HEAD_TAIL_ROWS = 20_596;
const STORAGE_ALIGNMENT = 256;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_8355;
const TIMESTAMP_QUERY_BYTES = 16;
const TIMING_ROUND_COUNT = 16;
const REQUIRED_PAIR_WINS = 14;
const REQUIRED_SPEEDUP = 1.75;
const REQUIRED_BANDWIDTH_GBPS = 20;

const ROLE_BASES = Object.freeze([
  Object.freeze({ id: "query", inner: 1_024, columns: 2_048 }),
  Object.freeze({ id: "key", inner: 1_024, columns: 1_024 }),
  Object.freeze({ id: "value", inner: 1_024, columns: 1_024 }),
  Object.freeze({ id: "attention-output", inner: 2_048, columns: 1_024 }),
  Object.freeze({ id: "gate", inner: 1_024, columns: 3_072 }),
  Object.freeze({ id: "up", inner: 1_024, columns: 3_072 }),
  Object.freeze({ id: "down", inner: 3_072, columns: 1_024 }),
] as const);

const ROLE_SPECS: readonly RoleSpec[] = (() => {
  let offset = 0;
  return Object.freeze(ROLE_BASES.map((base, ordinal) => {
    offset = align(offset, STORAGE_ALIGNMENT);
    const weightBytes = base.inner * base.columns * 2;
    const spec = Object.freeze({ ...base, ordinal, weightBytes,
      weightOffset: offset }) as RoleSpec;
    offset += weightBytes;
    return spec;
  }));
})();

const LAYER_WEIGHT_BYTES = align(ROLE_SPECS.at(-1)!.weightOffset +
  ROLE_SPECS.at(-1)!.weightBytes, STORAGE_ALIGNMENT);
const TAIL_WEIGHT_BYTES = 1_024 * TIED_HEAD_TAIL_ROWS * 2;
const LOGICAL_LAYER_MULTIPLY_ADDS_M1 = ROLE_SPECS.reduce(
  (sum, role) => sum + role.inner * role.columns,
  0,
);

const DIAGNOSTIC_SPECS: readonly DiagnosticSpec[] = Object.freeze([
  Object.freeze({ id: "query-1024x2048", inner: 1_024, columns: 2_048,
    weightBytes: 1_024 * 2_048 * 2, source: "layer", roleIndex: 0 }),
  Object.freeze({ id: "key-value-1024x1024", inner: 1_024, columns: 1_024,
    weightBytes: 1_024 * 1_024 * 2, source: "layer", roleIndex: 1 }),
  Object.freeze({ id: "attention-output-2048x1024", inner: 2_048,
    columns: 1_024, weightBytes: 2_048 * 1_024 * 2,
    source: "layer", roleIndex: 3 }),
  Object.freeze({ id: "gate-up-1024x3072", inner: 1_024, columns: 3_072,
    weightBytes: 1_024 * 3_072 * 2, source: "layer", roleIndex: 4 }),
  Object.freeze({ id: "down-3072x1024", inner: 3_072, columns: 1_024,
    weightBytes: 3_072 * 1_024 * 2, source: "layer", roleIndex: 6 }),
  Object.freeze({ id: "tied-head-tail-1024x20596", inner: 1_024,
    columns: TIED_HEAD_TAIL_ROWS, weightBytes: TAIL_WEIGHT_BYTES,
    source: "tied-head-tail", roleIndex: null }),
]);

const M1_ORDERS = Object.freeze([
  "ABC", "BAC", "CAB", "ABC", "BAC", "CAB", "ABC", "CAB",
]);
const M2_ORDERS = Object.freeze([
  "ACB", "BCA", "CBA", "ACB", "BCA", "CBA", "BCA", "CBA",
]);

let lifecycle: "idle" | "preparing" | "ready" | "running" | "settled" =
  "idle";
let abortController: AbortController | undefined;
let prepared: PreparedHarness | undefined;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  if (message.type === "cancel") {
    abortController?.abort(new DOMException("OPT-0083 cancelled", "AbortError"));
    return;
  }
  if (message.type === "prepare") {
    if (lifecycle !== "idle") return;
    lifecycle = "preparing";
    abortController = new AbortController();
    void prepareHarness(abortController.signal).then(
      async (value) => {
        prepared = value;
        lifecycle = "ready";
        try {
          self.postMessage({ type: "ready",
            readyAtEpochMilliseconds: value.readyAtEpochMilliseconds,
            preparation: Object.freeze({ identity: value.identity,
              correctness: value.correctness,
              rejectionContracts: value.rejectionContracts,
              warmup: value.warmup }) });
        } catch (error) {
          await failAndCleanup("ready-publication", error);
        }
      },
      (error) => failAndCleanup("preparation", error),
    );
    return;
  }
  if (message.type === "run" && lifecycle === "ready" &&
    prepared !== undefined) {
    lifecycle = "running";
    void runTiming(prepared, message.thermalLaunch,
      abortController!.signal).then(
      (evidence) => {
        lifecycle = "settled";
        self.postMessage({ type: "measurement-complete", evidence });
      },
      (error) => failAndCleanup("timing", error),
    );
  }
});

async function failAndCleanup(phase: string, error: unknown): Promise<void> {
  const active = prepared;
  prepared = undefined;
  const cleanup = active === undefined ? undefined : await active.cleanup()
    .catch((cleanupError) => Object.freeze({ cleanupError:
      errorValue(cleanupError) }));
  lifecycle = "settled";
  self.postMessage({ type: "failed", phase, error: errorValue(error),
    ...(cleanup === undefined ? {} : { evidence: Object.freeze({ cleanup }) }) });
}

async function prepareHarness(signal: AbortSignal): Promise<PreparedHarness> {
  checkpoint(signal);
  requireLittleEndianHost();
  postProgress("requesting the target WebGPU adapter and timestamp-query device");
  if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter was returned");
  requireAdapter(adapter);
  const device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query"],
    requiredLimits: {
      maxComputeWorkgroupStorageSize: 17_024,
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 128,
      maxStorageBufferBindingSize: Math.max(
        LAYER_WEIGHT_BYTES,
        TAIL_WEIGHT_BYTES,
      ),
      maxBufferSize: Math.max(LAYER_WEIGHT_BYTES, TAIL_WEIGHT_BYTES) +
        2 * STORAGE_GUARD_BYTES,
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
      deviceLosses.push(`${info.reason}:${info.message}`);
    }
  });

  let kernelA: AceCorrectnessGemmKernel | undefined;
  let kernelB: AceOpt0083PlannerDirectLowRowBf16GemvKernel | undefined;
  let kernelC: AceOpt0083PlannerLowRowBf16GemvKernel | undefined;
  let querySet: GPUQuerySet | undefined;
  let cleanupPromise: Promise<Readonly<Record<string, unknown>>> | undefined;
  let firstDispatchContext: Readonly<{
    readonly shape: AceGemmShape;
    readonly activation: GPUBufferBinding;
    readonly weight: GPUBufferBinding;
    readonly output: GPUBufferBinding;
  }> | undefined;

  const cleanup = (): Promise<Readonly<Record<string, unknown>>> => {
    if (cleanupPromise !== undefined) {
      return cleanupPromise.then((value) => Object.freeze({ ...value,
        repeatedCall: true, idempotent: true }));
    }
    cleanupPromise = (async () => {
      const cleanupStartedAtEpochMilliseconds = Date.now();
      await device.queue.onSubmittedWorkDone().catch(() => undefined);
      let postDestroyBRejected = firstDispatchContext === undefined;
      let postDestroyCRejected = firstDispatchContext === undefined;
      kernelB?.destroy();
      kernelC?.destroy();
      if (firstDispatchContext !== undefined) {
        const bindings = Object.freeze({ activation:
          firstDispatchContext.activation, weight: firstDispatchContext.weight,
          output: firstDispatchContext.output });
        postDestroyBRejected = await rejects(() => kernelB!.createDispatch(
          "opt0083-post-destroy-B", firstDispatchContext!.shape, bindings,
        ));
        postDestroyCRejected = await rejects(() => kernelC!.createDispatch(
          "opt0083-post-destroy-C", firstDispatchContext!.shape, bindings,
        ));
      }
      kernelA?.destroy();
      querySet?.destroy();
      tracker.destroyAll();
      device.destroy();
      await settleEvents();
      const resources = tracker.receipt();
      const cleanupCompletedAtEpochMilliseconds = Date.now();
      return Object.freeze({ cleanupStartedAtEpochMilliseconds,
        cleanupCompletedAtEpochMilliseconds, ...resources,
        createdEqualsDestroyed: resources.createdBufferCount ===
          resources.destroyedBufferCount,
        zeroLiveBuffers: resources.liveBufferCount === 0,
        zeroLiveBytes: resources.liveBytes === 0,
        mapsBalanced: resources.mapCount === resources.unmapCount &&
          resources.activeMapCount === 0,
        postDestroyBRejected, postDestroyCRejected,
        deviceDestroyed: true, repeatedCall: false, idempotent: true });
    })();
    return cleanupPromise;
  };

  try {
    kernelA = AceCorrectnessGemmKernel.create(
      device, "reference-bf16", "source-row-major",
    );
    kernelB = AceOpt0083PlannerDirectLowRowBf16GemvKernel.create(
      device, "reference-bf16", "source-row-major",
    );
    kernelC = AceOpt0083PlannerLowRowBf16GemvKernel.create(
      device, "reference-bf16", "source-row-major",
    );
    querySet = device.createQuerySet({ label: "opt0083-timestamps",
      type: "timestamp", count: 2 });
    const queryResolve = tracker.create(device, {
      label: "opt0083-timestamp-resolve", size: TIMESTAMP_QUERY_BYTES,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const queryReadback = tracker.create(device, {
      label: "opt0083-timestamp-readback", size: TIMESTAMP_QUERY_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    postProgress("creating deterministic paired-K activations");
    const activations = createActivations(device, tracker);
    postProgress("creating eight distinct rotating 30 MiB layer weight sets");
    const weightSets: WeightSet[] = [];
    for (let index = 0; index < WEIGHT_SET_COUNT; index += 1) {
      checkpoint(signal);
      weightSets.push(await createWeightSet(device, tracker, index));
      postProgress(`created rotating weight set ${index + 1}/${WEIGHT_SET_COUNT}`);
      await yieldToWorker();
    }
    checkpoint(signal);
    postProgress("creating the exact 20,596-column tied-head tail weight");
    const tailWeight = await createTailWeight(device, tracker);

    const outputs: Record<string, readonly GuardedOutput[]> = {};
    const tailOutputs: Record<string, GuardedOutput> = {};
    for (const rows of [1, 2] as const) {
      outputs[String(rows)] = Object.freeze(ROLE_SPECS.map((role) =>
        createGuardedOutput(device, tracker,
          `opt0083-M${rows}-${role.id}`, rows, role.columns)));
      tailOutputs[String(rows)] = createGuardedOutput(device, tracker,
        `opt0083-M${rows}-tied-head-tail`, rows, TIED_HEAD_TAIL_ROWS);
    }

    postProgress("compiling A/B/C and binding all eight rotating sets");
    const dispatchSets: DispatchSet[] = [];
    for (const weightSet of weightSets) {
      for (const rows of [1, 2] as const) {
        checkpoint(signal);
        const rowOutputs = outputs[String(rows)]!;
        const arms: Record<Arm, AceGemmDispatch[]> = { A: [], B: [], C: [] };
        for (const [roleIndex, role] of ROLE_SPECS.entries()) {
          const shape = Object.freeze({ rows, inner: role.inner,
            columns: role.columns });
          const bindings = Object.freeze({
            activation: activationBinding(activations, rows, role.inner),
            weight: weightSet.bindings[roleIndex]!,
            output: rowOutputs[roleIndex]!.binding,
          });
          arms.A.push(await kernelA.createDispatch(
            `opt0083-set${weightSet.index}-M${rows}-${role.id}-A`,
            shape, bindings,
          ));
          arms.B.push(await kernelB.createDispatch(
            `opt0083-set${weightSet.index}-M${rows}-${role.id}-B`,
            shape, bindings,
          ));
          arms.C.push(await kernelC.createDispatch(
            `opt0083-set${weightSet.index}-M${rows}-${role.id}-C`,
            shape, bindings,
          ));
          firstDispatchContext ??= Object.freeze({ shape,
            activation: bindings.activation, weight: bindings.weight,
            output: bindings.output });
        }
        dispatchSets.push(Object.freeze({ weightSetIndex: weightSet.index,
          rows, arms: Object.freeze({ A: Object.freeze(arms.A),
            B: Object.freeze(arms.B), C: Object.freeze(arms.C) }) }));
      }
      await yieldToWorker();
    }

    const tailDispatchSets: Record<string, TailDispatchSet> = {};
    for (const rows of [1, 2] as const) {
      const shape = Object.freeze({ rows, inner: 1_024,
        columns: TIED_HEAD_TAIL_ROWS });
      const bindings = Object.freeze({
        activation: activationBinding(activations, rows, 1_024),
        weight: binding(tailWeight, 0, TAIL_WEIGHT_BYTES),
        output: tailOutputs[String(rows)]!.binding,
      });
      tailDispatchSets[String(rows)] = Object.freeze({ rows,
        arms: Object.freeze({
          A: await kernelA.createDispatch(`opt0083-M${rows}-tail-A`,
            shape, bindings),
          B: await kernelB.createDispatch(`opt0083-M${rows}-tail-B`,
            shape, bindings),
          C: await kernelC.createDispatch(`opt0083-M${rows}-tail-C`,
            shape, bindings),
        }) });
    }

    checkpoint(signal);
    const identity = await buildIdentity(adapter, device, weightSets);
    postProgress("running production-like, cancellation, and tail raw-U32 gates");
    const correctness = await runCorrectness(device, tracker,
      dispatchSets, tailDispatchSets, outputs, tailOutputs, signal);
    if (correctness["passed"] !== true) {
      throw new Error("OPT-0083 raw-U32 correctness gate failed");
    }
    postProgress("running candidate rejection and bounded-cancellation contracts");
    const rejectionContracts = await runRejectionContracts(device,
      kernelB, kernelC, weightSets[0]!, activations, outputs, signal);
    if (rejectionContracts["passed"] !== true) {
      throw new Error("OPT-0083 rejection-contract gate failed");
    }

    postProgress("warming every A/B/C M1/M2 complete-layer arm");
    const warmupStartedAtEpochMilliseconds = Date.now();
    const warmupSet = dispatchSets.filter(({ weightSetIndex }) =>
      weightSetIndex === WEIGHT_SET_COUNT - 1);
    let warmupExecutions = 0;
    let warmupDispatchExecutions = 0;
    for (const entry of warmupSet) {
      const order: readonly Arm[] = entry.rows === 1
        ? ["A", "B", "C"] : ["C", "B", "A"];
      for (const arm of order) {
        checkpoint(signal);
        for (const [roleIndex, dispatch] of entry.arms[arm].entries()) {
          postProgress(
            `warming complete layer M${entry.rows} arm ${arm} ` +
            `role ${ROLE_SPECS[roleIndex]!.id} ` +
            `(${warmupDispatchExecutions + 1}/42)`,
          );
          await executeAndDrain(device, Object.freeze([dispatch]));
          warmupDispatchExecutions += 1;
        }
        warmupExecutions += 1;
      }
    }
    await settleEvents();
    if (uncapturedErrors.length !== 0 || deviceLosses.length !== 0) {
      throw new Error("OPT-0083 observed a GPU error during preparation");
    }
    const warmupCompletedAtEpochMilliseconds = Date.now();
    const readyAtEpochMilliseconds = Date.now();
    const value: PreparedHarness = Object.freeze({ adapter, device, tracker,
      kernels: Object.freeze({ A: kernelA, B: kernelB, C: kernelC }),
      weightSets: Object.freeze(weightSets), tailWeight,
      activations: Object.freeze(activations), outputs: Object.freeze(outputs),
      tailOutputs: Object.freeze(tailOutputs),
      dispatchSets: Object.freeze(dispatchSets),
      tailDispatchSets: Object.freeze(tailDispatchSets), querySet,
      queryResolve, queryReadback, identity, correctness, rejectionContracts,
      warmup: Object.freeze({ warmupStartedAtEpochMilliseconds,
        warmupCompletedAtEpochMilliseconds, warmupExecutions,
        warmupDispatchExecutions,
        oneCompleteLayerWarmupPerArmPerRowCount: true,
        completedBeforeReady: true }),
      readyAtEpochMilliseconds, uncapturedErrors, deviceLosses, cleanup });
    return value;
  } catch (error) {
    await cleanup();
    throw error;
  }
}

function createActivations(
  device: GPUDevice,
  tracker: BufferTracker,
): Record<string, GPUBuffer> {
  const result: Record<string, GPUBuffer> = {};
  for (const inner of [1_024, 2_048, 3_072] as const) {
    const buffer = tracker.create(device, {
      label: `opt0083-M2-K${inner}-activation`,
      size: 2 * inner * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    const values = new Float32Array(buffer.getMappedRange());
    for (let row = 0; row < 2; row += 1) {
      for (let pair = 0; pair < inner / 2; pair += 1) {
        const mixed = mix32(Math.imul(inner, 0x9e37_79b1) ^
          Math.imul(row + 1, 0x85eb_ca6b) ^
          Math.imul(pair + 1, 0xc2b2_ae35));
        const magnitude = [0.125, 0.25, 0.5, 1][mixed & 3]!;
        const value = (mixed & 4) === 0 ? magnitude : -magnitude;
        const offset = row * inner + pair * 2;
        // Equal adjacent values make the alternating-sign long-K fixture
        // cancel pair-by-pair while remaining finite and production-like.
        values[offset] = value;
        values[offset + 1] = value;
      }
    }
    tracker.unmap(buffer);
    result[String(inner)] = buffer;
  }
  return result;
}

async function createWeightSet(
  device: GPUDevice,
  tracker: BufferTracker,
  setIndex: number,
): Promise<WeightSet> {
  const buffer = tracker.create(device, {
    label: `opt0083-layer-weight-set-${setIndex}`,
    size: LAYER_WEIGHT_BYTES,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const words = new Uint32Array(buffer.getMappedRange());
  let firstWord = 0;
  let lastWord = 0;
  let rollingSignature = 0x811c_9dc5;
  for (const role of ROLE_SPECS) {
    const first = role.weightOffset / Uint32Array.BYTES_PER_ELEMENT;
    const pairsPerColumn = role.inner / 2;
    for (let column = 0; column < role.columns; column += 1) {
      for (let pair = 0; pair < pairsPerColumn; pair += 1) {
        const mixed = mix32(Math.imul(setIndex + 1, 0x9e37_79b1) ^
          Math.imul(role.ordinal + 1, 0x85eb_ca6b) ^
          Math.imul(column + 1, 0xc2b2_ae35) ^
          Math.imul(pair + 1, 0x27d4_eb2f));
        const word = setIndex === 1
          ? cancellationBf16Pair(mixed)
          : productionBf16Pair(mixed);
        words[first + column * pairsPerColumn + pair] = word;
        if (role.ordinal === 0 && column === 0 && pair === 0) firstWord = word;
        if (role.ordinal === ROLE_SPECS.length - 1 &&
          column === role.columns - 1 && pair === pairsPerColumn - 1) {
          lastWord = word;
        }
        if ((pair & 255) === 0) {
          rollingSignature = Math.imul(rollingSignature ^ word, 0x0100_0193);
        }
      }
    }
  }
  tracker.unmap(buffer);
  const bindings = Object.freeze(ROLE_SPECS.map((role) =>
    binding(buffer, role.weightOffset, role.weightBytes)));
  return Object.freeze({ index: setIndex, buffer, bindings,
    signature: Object.freeze({ setIndex, byteLength: LAYER_WEIGHT_BYTES,
      fixture: setIndex === 1
        ? "pairwise-alternating-sign-exact-cancellation"
        : "finite-powers-of-two-production-like",
      firstWord: hex32(firstWord), lastWord: hex32(lastWord),
      rollingSignature: hex32(rollingSignature >>> 0) }) });
}

async function createTailWeight(
  device: GPUDevice,
  tracker: BufferTracker,
): Promise<GPUBuffer> {
  const buffer = tracker.create(device, {
    label: "opt0083-tied-head-tail-weight",
    size: TAIL_WEIGHT_BYTES,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const words = new Uint32Array(buffer.getMappedRange());
  const pairsPerColumn = 1_024 / 2;
  for (let column = 0; column < TIED_HEAD_TAIL_ROWS; column += 1) {
    for (let pair = 0; pair < pairsPerColumn; pair += 1) {
      words[column * pairsPerColumn + pair] = productionBf16Pair(mix32(
        0x5a17_2059 ^ Math.imul(column + 1, 0x9e37_79b1) ^
        Math.imul(pair + 1, 0x85eb_ca6b),
      ));
    }
  }
  tracker.unmap(buffer);
  return buffer;
}

function createGuardedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  rows: RowCount,
  columns: number,
): GuardedOutput {
  const outputElements = rows * columns;
  const outputBytes = outputElements * Float32Array.BYTES_PER_ELEMENT;
  const totalBytes = outputBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = tracker.create(device, {
    label: `${label}-prefill`, size: totalBytes,
    usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true,
  });
  const prefillWords = new Uint32Array(prefill.getMappedRange());
  prefillWords.fill(STORAGE_GUARD_U32);
  prefillWords.fill(OUTPUT_PREFILL_QNAN_U32,
    STORAGE_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT,
    STORAGE_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT + outputElements);
  tracker.unmap(prefill);
  const buffer = tracker.create(device, {
    label: `${label}-guarded-output`, size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  const readback = tracker.create(device, {
    label: `${label}-readback`, size: totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  return Object.freeze({ label, rows, columns, outputElements, outputBytes,
    totalBytes, buffer, binding: binding(buffer, STORAGE_GUARD_BYTES,
      outputBytes), prefill, readback });
}

async function runCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  dispatchSets: readonly DispatchSet[],
  tailDispatchSets: Readonly<Record<string, TailDispatchSet>>,
  outputs: Readonly<Record<string, readonly GuardedOutput[]>>,
  tailOutputs: Readonly<Record<string, GuardedOutput>>,
  signal: AbortSignal,
): Promise<Readonly<Record<string, unknown>>> {
  const cases: Readonly<Record<string, unknown>>[] = [];
  for (const rows of [1, 2] as const) {
    checkpoint(signal);
    const production = requireDispatchSet(dispatchSets, 0, rows);
    cases.push(await verifyScenario(device, tracker,
      `production-like-M${rows}-complete-seven`, rows,
      outputs[String(rows)]!, production.arms, signal));
    postProgress(`raw-U32 production-like complete layer M${rows} passed`);

    const cancellation = requireDispatchSet(dispatchSets, 1, rows);
    const downIndex = ROLE_SPECS.findIndex(({ id }) => id === "down");
    const downOutputs = Object.freeze([outputs[String(rows)]![downIndex]!]);
    const downArms = Object.freeze({
      A: Object.freeze([cancellation.arms.A[downIndex]!] as const),
      B: Object.freeze([cancellation.arms.B[downIndex]!] as const),
      C: Object.freeze([cancellation.arms.C[downIndex]!] as const),
    });
    cases.push(await verifyScenario(device, tracker,
      `adversarial-pairwise-cancellation-K3072-M${rows}`, rows,
      downOutputs, downArms, signal));
    postProgress(`raw-U32 adversarial K3072 cancellation M${rows} passed`);

    const tail = tailDispatchSets[String(rows)]!;
    const oneTailOutput = Object.freeze([tailOutputs[String(rows)]!]);
    const tailArms = Object.freeze({
      A: Object.freeze([tail.arms.A] as const),
      B: Object.freeze([tail.arms.B] as const),
      C: Object.freeze([tail.arms.C] as const),
    });
    cases.push(await verifyScenario(device, tracker,
      `tied-head-tail-N20596-M${rows}`, rows,
      oneTailOutput, tailArms, signal));
    postProgress(`raw-U32 tied-head N20596 tail M${rows} passed`);
  }
  const comparedOutputWordsPerSixRuns = cases.reduce((sum, value) =>
    sum + Number(value["outputWordCount"]), 0);
  const passed = cases.every((value) => value["passed"] === true);
  return Object.freeze({ fixtureVersion:
    "opt0083-production-cancellation-tail-raw-u32-v1",
    cases: Object.freeze(cases), caseCount: cases.length,
    comparisonsPerCase: 6,
    comparedOutputWordsPerSixRuns,
    comparedOutputWordVisits: comparedOutputWordsPerSixRuns * 6,
    everyProductionRoleCovered: true,
    bothM1AndM2Covered: true,
    productionLikeFiniteDataCovered: true,
    adversarialLongKCancellationCovered: true,
    tiedHeadPhysicalTailColumns: TIED_HEAD_TAIL_ROWS,
    deterministicRerunsCovered: true,
    completeWritesAndGuardsCovered: true,
    allRawU32Exact: passed,
    passed });
}

async function verifyScenario(
  device: GPUDevice,
  tracker: BufferTracker,
  id: string,
  rows: RowCount,
  outputs: readonly GuardedOutput[],
  arms: Readonly<Record<Arm, readonly AceGemmDispatch[]>>,
  signal: AbortSignal,
): Promise<Readonly<Record<string, unknown>>> {
  const snapshots: Record<Arm, [OutputSnapshot[], OutputSnapshot[]]> = {
    A: [[], []], B: [[], []], C: [[], []],
  };
  for (const arm of ["A", "A", "B", "B", "C", "C"] as const) {
    checkpoint(signal);
    const runIndex = snapshots[arm][0].length === 0 ? 0 : 1;
    snapshots[arm][runIndex] = await executeCorrectnessGroup(device, tracker,
      outputs, arms[arm]);
    await yieldToWorker();
  }
  const comparisons = Object.freeze({
    A_rerun: compareSnapshotGroups(snapshots.A[0], snapshots.A[1]),
    A_B: compareSnapshotGroups(snapshots.A[0], snapshots.B[0]),
    B_rerun: compareSnapshotGroups(snapshots.B[0], snapshots.B[1]),
    A_C: compareSnapshotGroups(snapshots.A[0], snapshots.C[0]),
    C_rerun: compareSnapshotGroups(snapshots.C[0], snapshots.C[1]),
    B_C: compareSnapshotGroups(snapshots.B[0], snapshots.C[0]),
  });
  const complete = (Object.values(snapshots) as [OutputSnapshot[],
    OutputSnapshot[]][]).every((runs) => runs.every((run) =>
      run.every(completeSnapshot)));
  const exact = Object.values(comparisons).every((value) =>
    value.mismatchCount === 0);
  const hashes = Object.freeze(Object.fromEntries(
    (Object.entries(snapshots) as [Arm, [OutputSnapshot[],
      OutputSnapshot[]]][]).map(([arm, runs]) => [arm,
        Object.freeze(runs.map((run) => Object.freeze(run.map((snapshot) =>
          snapshot.sha256))))]),
  ));
  const outputWordCount = outputs.reduce((sum, output) =>
    sum + output.outputElements, 0);
  return Object.freeze({ id, rows, outputCount: outputs.length,
    outputWordCount, executionOrder: "A,A,B,B,C,C", hashes, comparisons,
    allOutputsFiniteCompleteAndGuarded: complete,
    allRawU32Exact: exact, passed: complete && exact });
}

async function executeCorrectnessGroup(
  device: GPUDevice,
  tracker: BufferTracker,
  outputs: readonly GuardedOutput[],
  dispatches: readonly AceGemmDispatch[],
): Promise<OutputSnapshot[]> {
  if (outputs.length !== dispatches.length || outputs.length === 0) {
    throw new Error("OPT-0083 correctness group topology changed");
  }
  if (outputs.some(({ readback }) => readback.mapState !== "unmapped")) {
    throw new Error("OPT-0083 correctness readback must be unmapped");
  }
  const encoder = device.createCommandEncoder({
    label: `opt0083-${dispatches[0]!.label}-correctness`,
  });
  for (const output of outputs) {
    encoder.copyBufferToBuffer(output.prefill, 0, output.buffer, 0,
      output.totalBytes);
  }
  const pass = encoder.beginComputePass({
    label: `opt0083-${dispatches[0]!.label}-correctness-pass`,
  });
  for (const dispatch of dispatches) dispatch.encode(pass);
  pass.end();
  for (const output of outputs) {
    encoder.copyBufferToBuffer(output.buffer, 0, output.readback, 0,
      output.totalBytes);
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await Promise.all(outputs.map(({ readback }) => tracker.mapRead(readback)));
  try {
    return await Promise.all(outputs.map(snapshotOutput));
  } finally {
    for (const { readback } of outputs) tracker.unmap(readback);
  }
}

async function snapshotOutput(output: GuardedOutput): Promise<OutputSnapshot> {
  const all = new Uint32Array(output.readback.getMappedRange());
  const guardWords = STORAGE_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT;
  const end = guardWords + output.outputElements;
  let prefixGuardIntact = true;
  let suffixGuardIntact = true;
  for (let index = 0; index < guardWords; index += 1) {
    prefixGuardIntact &&= all[index] === STORAGE_GUARD_U32;
    suffixGuardIntact &&= all[end + index] === STORAGE_GUARD_U32;
  }
  const words = all.slice(guardWords, end);
  let nonFiniteCount = 0;
  let qNaNPrefillCount = 0;
  for (const word of words) {
    if ((word & 0x7f80_0000) === 0x7f80_0000) nonFiniteCount += 1;
    if (word === OUTPUT_PREFILL_QNAN_U32) qNaNPrefillCount += 1;
  }
  return Object.freeze({ words, sha256: await sha256Bytes(new Uint8Array(
    words.buffer, words.byteOffset, words.byteLength,
  )), nonFiniteCount, qNaNPrefillCount, prefixGuardIntact,
  suffixGuardIntact,
  firstOutputWritten: words[0] !== OUTPUT_PREFILL_QNAN_U32,
  lastOutputWritten: words.at(-1) !== OUTPUT_PREFILL_QNAN_U32 });
}

function compareSnapshotGroups(
  expected: readonly OutputSnapshot[],
  actual: readonly OutputSnapshot[],
): Readonly<{ readonly comparedWordCount: number;
  readonly mismatchCount: number; readonly firstMismatch: unknown }> {
  if (expected.length !== actual.length) {
    throw new Error("OPT-0083 snapshot group count changed");
  }
  let comparedWordCount = 0;
  let mismatchCount = 0;
  let firstMismatch: unknown = null;
  for (let outputIndex = 0; outputIndex < expected.length; outputIndex += 1) {
    const left = expected[outputIndex]!.words;
    const right = actual[outputIndex]!.words;
    if (left.length !== right.length) {
      throw new Error("OPT-0083 snapshot output length changed");
    }
    comparedWordCount += left.length;
    for (let index = 0; index < left.length; index += 1) {
      if (left[index] === right[index]) continue;
      mismatchCount += 1;
      firstMismatch ??= Object.freeze({ outputIndex, index,
        expectedU32: hex32(left[index]!), actualU32: hex32(right[index]!) });
    }
  }
  return Object.freeze({ comparedWordCount, mismatchCount, firstMismatch });
}

function completeSnapshot(snapshot: OutputSnapshot): boolean {
  return snapshot.nonFiniteCount === 0 && snapshot.qNaNPrefillCount === 0 &&
    snapshot.prefixGuardIntact && snapshot.suffixGuardIntact &&
    snapshot.firstOutputWritten && snapshot.lastOutputWritten;
}

async function runRejectionContracts(
  device: GPUDevice,
  kernelB: AceOpt0083PlannerDirectLowRowBf16GemvKernel,
  kernelC: AceOpt0083PlannerLowRowBf16GemvKernel,
  weightSet: WeightSet,
  activations: Readonly<Record<string, GPUBuffer>>,
  outputs: Readonly<Record<string, readonly GuardedOutput[]>>,
  signal: AbortSignal,
): Promise<Readonly<Record<string, unknown>>> {
  checkpoint(signal);
  const shape = Object.freeze({ rows: 1, inner: 1_024, columns: 1_024 });
  const activation = activationBinding(activations, 1, 1_024);
  const weight = weightSet.bindings[1]!;
  const output = outputs["1"]![1]!.binding;
  const cases: Readonly<Record<string, unknown>>[] = [];
  const owners = Object.freeze({ B: kernelB, C: kernelC });
  for (const [arm, owner] of Object.entries(owners) as ["B" | "C",
    typeof kernelB | typeof kernelC][]) {
    cases.push(await expectRejection(`${arm}-short-weight-binding`, () =>
      owner.createDispatch(`opt0083-${arm}-short`, shape, Object.freeze({
        activation,
        weight: binding(weight.buffer, weight.offset ?? 0,
          Number(weight.size) - 4), output,
      }))));
    cases.push(await expectRejection(`${arm}-alias-output-activation`, () =>
      owner.createDispatch(`opt0083-${arm}-alias`, shape, Object.freeze({
        activation, weight, output: activation,
      }))));
    cases.push(await expectRejection(`${arm}-bias-rejected`, () =>
      owner.createDispatch(`opt0083-${arm}-bias`, shape, Object.freeze({
        activation, weight, output, bias: binding(
          activations["1024"]!, 0, 4,
        ),
      }))));
  }
  cases.push(await expectRejection("B-wrong-profile", async () => {
    AceOpt0083PlannerDirectLowRowBf16GemvKernel.create(
      device, "raw-fp16", "source-row-major",
    );
  }));
  cases.push(await expectRejection("C-wrong-profile", async () => {
    AceOpt0083PlannerLowRowBf16GemvKernel.create(
      device, "raw-fp16", "source-row-major",
    );
  }));
  cases.push(await expectRejection("B-row-count-3", async () => {
    planAceOpt0083PlannerDirectLowRowBf16Gemv({ rows: 3, inner: 1_024,
      columns: 1_024 });
  }));
  cases.push(await expectRejection("C-row-count-3", async () => {
    planAceOpt0083PlannerLowRowBf16Gemv({ rows: 3, inner: 1_024,
      columns: 1_024 });
  }));
  cases.push(await expectRejection("B-dispatch-limit", async () => {
    planAceOpt0083PlannerDirectLowRowBf16Gemv({ rows: 1, inner: 2,
      columns: (Number(device.limits.maxComputeWorkgroupsPerDimension) + 1) *
        128 });
  }));
  cases.push(await expectRejection("C-dispatch-limit", async () => {
    planAceOpt0083PlannerLowRowBf16Gemv({ rows: 1, inner: 2,
      columns: (Number(device.limits.maxComputeWorkgroupsPerDimension) + 1) *
        128 });
  }));
  const cancellationProbe = new AbortController();
  cancellationProbe.abort(new DOMException(
    "OPT-0083 cancellation preflight", "AbortError",
  ));
  let cancellationCheckpointRejected = false;
  try {
    checkpoint(cancellationProbe.signal);
  } catch (error) {
    cancellationCheckpointRejected = error instanceof DOMException &&
      error.name === "AbortError";
  }
  const passed = cases.every((value) => value["rejected"] === true) &&
    cancellationCheckpointRejected;
  return Object.freeze({ cases: Object.freeze(cases),
    expectedRejectionCount: 12, observedRejectionCount:
      cases.filter((value) => value["rejected"] === true).length,
    shortBindingCovered: true, aliasCovered: true, wrongProfileCovered: true,
    biasCovered: true, rowCountCovered: true, deviceLimitCovered: true,
    cancellationCheckpointRejected,
    cancellationMechanism: "worker AbortSignal checked between bounded fills, drained executions, maps, and timing samples",
    maximumUninterruptibleGpuUnit: "one complete-layer command buffer and matching drain",
    passed });
}

async function expectRejection(
  id: string,
  operation: () => Promise<unknown>,
): Promise<Readonly<Record<string, unknown>>> {
  try {
    await operation();
    return Object.freeze({ id, rejected: false, error: null });
  } catch (error) {
    return Object.freeze({ id, rejected: true, error: errorValue(error) });
  }
}

async function runTiming(
  prepared: PreparedHarness,
  thermalLaunch: ThermalLaunch,
  signal: AbortSignal,
): Promise<Readonly<Record<string, unknown>>> {
  validateThermalLaunch(thermalLaunch, prepared.readyAtEpochMilliseconds);
  const measurementStartedAtEpochMilliseconds = Date.now();
  const primarySamples: TimestampSample[] = [];
  const rounds = buildTimingRounds();
  for (const round of rounds) {
    checkpoint(signal);
    const dispatchSet = requireDispatchSet(prepared.dispatchSets,
      round.weightSetIndex, round.rows);
    for (const [armPosition, arm] of round.armOrder.entries()) {
      postProgress(
        `primary round ${round.roundIndex + 1}/${TIMING_ROUND_COUNT} ` +
        `M${round.rows} set${round.weightSetIndex} ${arm} position${armPosition + 1}`,
      );
      checkpoint(signal);
      primarySamples.push(await timeDispatches(prepared, dispatchSet.arms[arm],
        Object.freeze({ roundIndex: round.roundIndex,
          weightSetIndex: round.weightSetIndex, rows: round.rows, arm,
          scope: "complete-layer", shapeId: null,
          packedWeightBytes: LAYER_WEIGHT_BYTES })));
    }
    await yieldToWorker();
  }

  postProgress("primary timing complete; collecting non-gating per-shape diagnostics");
  const diagnosticSamples: TimestampSample[] = [];
  const diagnosticWeightSet = prepared.weightSets[7]!;
  for (const [specIndex, spec] of DIAGNOSTIC_SPECS.entries()) {
    for (const rows of [1, 2] as const) {
      checkpoint(signal);
      const dispatches = diagnosticDispatches(prepared, diagnosticWeightSet,
        spec, rows);
      const order = parseArmOrder((specIndex + rows) % 2 === 0
        ? "ABC" : "CBA");
      for (const arm of order) {
        diagnosticSamples.push(await timeDispatches(prepared,
          Object.freeze([dispatches[arm]]), Object.freeze({
            roundIndex: null,
            weightSetIndex: spec.source === "layer" ? 7 : null,
            rows, arm, scope: "shape-diagnostic", shapeId: spec.id,
            packedWeightBytes: spec.weightBytes,
          })));
      }
    }
    await yieldToWorker();
  }

  await prepared.device.queue.onSubmittedWorkDone();
  await settleEvents();
  const measurementCompletedAtEpochMilliseconds = Date.now();
  if (prepared.uncapturedErrors.length !== 0 ||
    prepared.deviceLosses.length !== 0) {
    throw new Error("OPT-0083 observed a timing GPU error or device loss");
  }
  const timestampDiagnostics = timestampDiagnosticsReceipt(
    primarySamples, diagnosticSamples,
  );
  const timing = summarizeTiming(primarySamples, diagnosticSamples);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanupFirst = await prepared.cleanup();
  const cleanupSecond = await prepared.cleanup();
  const cleanupCompletedAtEpochMilliseconds = Number(
    cleanupFirst["cleanupCompletedAtEpochMilliseconds"],
  );
  const cleanupPassed = cleanupFirst["zeroLiveBuffers"] === true &&
    cleanupFirst["zeroLiveBytes"] === true &&
    cleanupFirst["createdEqualsDestroyed"] === true &&
    cleanupFirst["mapsBalanced"] === true &&
    cleanupFirst["postDestroyBRejected"] === true &&
    cleanupFirst["postDestroyCRejected"] === true &&
    cleanupSecond["repeatedCall"] === true &&
    prepared.uncapturedErrors.length === 0 &&
    prepared.deviceLosses.length === 0;
  const inPagePassed = prepared.correctness["passed"] === true &&
    prepared.rejectionContracts["passed"] === true &&
    timing["passed"] === true && cleanupPassed;
  return Object.freeze({ schema: RECEIPT_SCHEMA, experiment: EXPERIMENT_ID,
    status: "awaiting-external-thermal-completion", passed: false,
    inPagePassed, thermalLaunch, cleanupCompletedAtEpochMilliseconds,
    identity: prepared.identity, correctness: prepared.correctness,
    rejectionContracts: prepared.rejectionContracts,
    warmup: prepared.warmup,
    protocol: Object.freeze({
      allArmsCompiledBeforeTiming: true,
      timingRoundCount: TIMING_ROUND_COUNT,
      pairedRoundsPerRowCount: 8,
      distinctRotatingWeightSetCount: WEIGHT_SET_COUNT,
      balancedInterleavedArmOrders: true,
      maximumArmPositionImbalance: 1,
      completeSevenOperationLayerPerPrimarySample: true,
      oneComputePassCommandBufferSubmitDrainPerSample: true,
      oneTimestampPairPerSample: true,
      fencedWallIsAuthoritativeForAllPassGatesAndArmSelection: true,
      gpuTimestampMetricsAreDiagnostic: true,
      gpuTimestampDiagnosticsMayBeUnavailableWithoutInvalidatingWallTiming:
        true,
      gpuTimestampDiagnostics: timestampDiagnostics,
      outputReadbackInsideTiming: false,
      allocationUploadCompilationInsideTiming: false,
      perShapeTimingDiagnosticOnly: true,
      tiedHeadTailShapeIncluded: true,
      unchangedTimingRetryPerformed: false,
    }),
    timing: Object.freeze({ ...timing,
      measurementStartedAtEpochMilliseconds,
      measurementCompletedAtEpochMilliseconds }),
    uncapturedGpuErrors: Object.freeze([...prepared.uncapturedErrors]),
    deviceLosses: Object.freeze([...prepared.deviceLosses]),
    memoryBeforeCleanup,
    cleanup: Object.freeze({ firstCall: cleanupFirst,
      secondCall: cleanupSecond, passed: cleanupPassed }) });
}

function buildTimingRounds(): readonly Readonly<{
  readonly roundIndex: number;
  readonly weightSetIndex: number;
  readonly rows: RowCount;
  readonly armOrder: readonly Arm[];
}>[] {
  const rounds: Readonly<{ readonly roundIndex: number;
    readonly weightSetIndex: number; readonly rows: RowCount;
    readonly armOrder: readonly Arm[] }>[] = [];
  for (let set = 0; set < WEIGHT_SET_COUNT; set += 1) {
    rounds.push(Object.freeze({ roundIndex: rounds.length,
      weightSetIndex: set, rows: 1,
      armOrder: parseArmOrder(M1_ORDERS[set]!) }));
    rounds.push(Object.freeze({ roundIndex: rounds.length,
      weightSetIndex: set, rows: 2,
      armOrder: parseArmOrder(M2_ORDERS[set]!) }));
  }
  if (rounds.length !== TIMING_ROUND_COUNT) {
    throw new Error("OPT-0083 timing round count changed");
  }
  for (const arm of ["A", "B", "C"] as const) {
    const positions = [0, 1, 2].map((position) => rounds.filter((round) =>
      round.armOrder[position] === arm).length);
    if (Math.max(...positions) - Math.min(...positions) > 1) {
      throw new Error(`OPT-0083 ${arm} arm-position balance changed`);
    }
  }
  for (const candidate of ["B", "C"] as const) {
    const beforeControl = rounds.filter((round) =>
      round.armOrder.indexOf(candidate) < round.armOrder.indexOf("A")).length;
    if (beforeControl < 7 || beforeControl > 9) {
      throw new Error(`OPT-0083 ${candidate}/A relative balance changed`);
    }
  }
  return Object.freeze(rounds);
}

async function timeDispatches(
  prepared: PreparedHarness,
  dispatches: readonly AceGemmDispatch[],
  metadata: Readonly<{
    readonly roundIndex: number | null;
    readonly weightSetIndex: number | null;
    readonly rows: RowCount;
    readonly arm: Arm;
    readonly scope: "complete-layer" | "shape-diagnostic";
    readonly shapeId: string | null;
    readonly packedWeightBytes: number;
  }>,
): Promise<TimestampSample> {
  if (dispatches.length === 0 || prepared.queryReadback.mapState !== "unmapped") {
    throw new Error("OPT-0083 timestamp topology changed");
  }
  const encoder = prepared.device.createCommandEncoder({
    label: `opt0083-${metadata.scope}-${metadata.rows}-${metadata.arm}`,
  });
  const pass = encoder.beginComputePass({
    label: `opt0083-${metadata.scope}-${metadata.rows}-${metadata.arm}-pass`,
    timestampWrites: { querySet: prepared.querySet,
      beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
  });
  for (const dispatch of dispatches) dispatch.encode(pass);
  pass.end();
  encoder.resolveQuerySet(prepared.querySet, 0, 2,
    prepared.queryResolve, 0);
  encoder.copyBufferToBuffer(prepared.queryResolve, 0,
    prepared.queryReadback, 0, TIMESTAMP_QUERY_BYTES);
  const command = encoder.finish();
  const submitAtPerformanceMilliseconds = performance.now();
  prepared.device.queue.submit([command]);
  await prepared.device.queue.onSubmittedWorkDone();
  const fenceAtPerformanceMilliseconds = performance.now();
  const wallMilliseconds = fenceAtPerformanceMilliseconds -
    submitAtPerformanceMilliseconds;
  await prepared.tracker.mapRead(prepared.queryReadback);
  let begin: bigint;
  let end: bigint;
  try {
    const timestamps = new BigUint64Array(
      prepared.queryReadback.getMappedRange(),
    );
    begin = timestamps[0]!;
    end = timestamps[1]!;
  } finally {
    prepared.tracker.unmap(prepared.queryReadback);
  }
  if (!Number.isFinite(wallMilliseconds) || wallMilliseconds <= 0) {
    throw new Error("OPT-0083 authoritative fenced-wall sample was invalid");
  }
  const timestampDelta = end - begin;
  const timestampCandidateNanoseconds = Number(timestampDelta);
  const timestampCandidateMilliseconds =
    timestampCandidateNanoseconds / 1_000_000;
  const timestampValid = end > begin &&
    Number.isSafeInteger(timestampCandidateNanoseconds) &&
    Number.isFinite(timestampCandidateMilliseconds) &&
    timestampCandidateMilliseconds > 0;
  const timestampUnavailableReason = timestampValid ? null : end <= begin
    ? "webgpu-timestamp-end-not-greater-than-begin"
    : !Number.isSafeInteger(timestampCandidateNanoseconds)
      ? "webgpu-timestamp-delta-not-a-safe-integer"
      : "webgpu-timestamp-duration-not-finite-positive";
  const gpuElapsedNanoseconds = timestampValid
    ? timestampCandidateNanoseconds : null;
  const gpuMilliseconds = timestampValid
    ? timestampCandidateMilliseconds : null;
  const validMultiplyAdds = dispatches.reduce((sum, dispatch) =>
    sum + dispatch.plan.rows * dispatch.plan.inner * dispatch.plan.columns, 0);
  const scheduledMultiplyAdds = dispatches.reduce((sum, dispatch) =>
    sum + dispatch.plan.outputRanges.reduce((innerSum, range) =>
      innerSum + range.multiplyAdds, 0), 0);
  return Object.freeze({ ...metadata, submitAtPerformanceMilliseconds,
    fenceAtPerformanceMilliseconds, wallMilliseconds,
    timestampBeginNanoseconds: begin.toString(),
    timestampEndNanoseconds: end.toString(), timestampValid,
    timestampUnavailableReason, gpuElapsedNanoseconds, gpuMilliseconds,
    gpuToWallRatio: gpuMilliseconds === null
      ? null : gpuMilliseconds / wallMilliseconds,
    effectivePackedWeightBandwidthGbPerSecond:
      gpuMilliseconds === null ? null :
        metadata.packedWeightBytes / (gpuMilliseconds * 1_000_000),
    effectivePackedWeightWallBandwidthGbPerSecond:
      metadata.packedWeightBytes / (wallMilliseconds * 1_000_000),
    validMultiplyAdds, scheduledMultiplyAdds,
    validGpuTflops: gpuMilliseconds === null
      ? null : tflops(validMultiplyAdds, gpuMilliseconds),
    scheduledGpuTflops: gpuMilliseconds === null
      ? null : tflops(scheduledMultiplyAdds, gpuMilliseconds),
    commandBufferCount: 1, computePassCount: 1, queueDrainCount: 1,
    timestampResolveCount: 1, timestampCopyCount: 1 });
}

function summarizeTiming(
  primarySamples: readonly TimestampSample[],
  diagnosticSamples: readonly TimestampSample[],
): Readonly<Record<string, unknown>> {
  if (primarySamples.length !== TIMING_ROUND_COUNT * 3) {
    throw new Error("OPT-0083 primary sample count changed");
  }
  const rows = ([1, 2] as const).map((rowCount) => {
    const arms = Object.freeze(Object.fromEntries((["A", "B", "C"] as const)
      .map((arm) => {
        const samples = primarySamples.filter((sample) =>
          sample.rows === rowCount && sample.arm === arm);
        if (samples.length !== 8) {
          throw new Error(`OPT-0083 M${rowCount} ${arm} sample count changed`);
        }
        const validTimestampSamples = samples.filter((sample) =>
          sample.timestampValid);
        const gpu = validTimestampSamples.length === samples.length
          ? validTimestampSamples.map((sample) => sample.gpuMilliseconds!)
          : null;
        const wall = samples.map((sample) => sample.wallMilliseconds);
        const bandwidth = validTimestampSamples.length === samples.length
          ? validTimestampSamples.map((sample) =>
              sample.effectivePackedWeightBandwidthGbPerSecond!)
          : null;
        const wallBandwidth = samples.map((sample) =>
          sample.effectivePackedWeightWallBandwidthGbPerSecond);
        return [arm, Object.freeze({ sampleCount: samples.length,
          validGpuTimestampSampleCount: validTimestampSamples.length,
          gpuTimestampDiagnosticsAvailable:
            validTimestampSamples.length === samples.length,
          meanGpuMilliseconds: gpu === null ? null : mean(gpu),
          medianGpuMilliseconds: gpu === null ? null : median(gpu),
          meanWallMilliseconds: mean(wall),
          medianWallMilliseconds: median(wall),
          meanEffectivePackedWeightBandwidthGbPerSecond:
            bandwidth === null ? null : mean(bandwidth),
          medianEffectivePackedWeightBandwidthGbPerSecond:
            bandwidth === null ? null : median(bandwidth),
          minimumEffectivePackedWeightBandwidthGbPerSecond:
            bandwidth === null ? null : Math.min(...bandwidth),
          maximumEffectivePackedWeightBandwidthGbPerSecond:
            bandwidth === null ? null : Math.max(...bandwidth),
          meanEffectivePackedWeightWallBandwidthGbPerSecond:
            mean(wallBandwidth),
          medianEffectivePackedWeightWallBandwidthGbPerSecond:
            median(wallBandwidth),
          minimumEffectivePackedWeightWallBandwidthGbPerSecond:
            Math.min(...wallBandwidth),
          maximumEffectivePackedWeightWallBandwidthGbPerSecond:
            Math.max(...wallBandwidth) })];
      })) as unknown as Record<Arm, Readonly<Record<string, unknown>>>);
    return Object.freeze({ rows: rowCount, arms });
  });

  const candidateSummaries = (["B", "C"] as const).map((arm) => {
    const control = primarySamples.filter((sample) => sample.arm === "A")
      .sort((left, right) => left.roundIndex! - right.roundIndex!);
    const candidate = primarySamples.filter((sample) => sample.arm === arm)
      .sort((left, right) => left.roundIndex! - right.roundIndex!);
    if (control.length !== TIMING_ROUND_COUNT ||
      candidate.length !== TIMING_ROUND_COUNT ||
      control.some((value, index) => value.roundIndex !==
        candidate[index]!.roundIndex)) {
      throw new Error(`OPT-0083 ${arm} paired sample topology changed`);
    }
    const paired = control.map((value, index) => {
      const other = candidate[index]!;
      const gpuComparable = value.gpuMilliseconds !== null &&
        other.gpuMilliseconds !== null;
      return Object.freeze({ roundIndex: value.roundIndex, rows: value.rows,
        weightSetIndex: value.weightSetIndex,
        controlGpuMilliseconds: value.gpuMilliseconds,
        candidateGpuMilliseconds: other.gpuMilliseconds,
        gpuTimestampComparable: gpuComparable,
        gpuSpeedup: gpuComparable
          ? value.gpuMilliseconds! / other.gpuMilliseconds! : null,
        gpuWin: gpuComparable
          ? other.gpuMilliseconds! < value.gpuMilliseconds! : null,
        controlWallMilliseconds: value.wallMilliseconds,
        candidateWallMilliseconds: other.wallMilliseconds,
        wallSpeedup: value.wallMilliseconds / other.wallMilliseconds,
        wallWin: other.wallMilliseconds < value.wallMilliseconds });
    });
    const gpuPairComparableCount = paired.filter(
      ({ gpuTimestampComparable }) => gpuTimestampComparable,
    ).length;
    const gpuPairWinsAmongComparable = paired.filter(({ gpuWin }) =>
      gpuWin === true).length;
    const gpuPairWins = gpuPairComparableCount === TIMING_ROUND_COUNT
      ? gpuPairWinsAmongComparable : null;
    const wallPairWins = paired.filter(({ wallWin }) => wallWin).length;
    const aggregateGpuDiagnosticsAvailable = control.every((value) =>
      value.gpuMilliseconds !== null) && candidate.every((value) =>
        value.gpuMilliseconds !== null);
    const controlGpuTotal = aggregateGpuDiagnosticsAvailable
      ? control.reduce((sum, value) => sum + value.gpuMilliseconds!, 0)
      : null;
    const candidateGpuTotal = aggregateGpuDiagnosticsAvailable
      ? candidate.reduce((sum, value) => sum + value.gpuMilliseconds!, 0)
      : null;
    const controlWallTotal = control.reduce((sum, value) =>
      sum + value.wallMilliseconds, 0);
    const candidateWallTotal = candidate.reduce((sum, value) =>
      sum + value.wallMilliseconds, 0);
    const aggregateGpuSpeedup = controlGpuTotal === null ||
      candidateGpuTotal === null ? null : controlGpuTotal / candidateGpuTotal;
    const aggregateGpuBandwidth = candidateGpuTotal === null ? null :
      LAYER_WEIGHT_BYTES * candidate.length /
        (candidateGpuTotal * 1_000_000);
    const aggregateWallSpeedup = controlWallTotal / candidateWallTotal;
    const aggregateWallBandwidth = LAYER_WEIGHT_BYTES * candidate.length /
      (candidateWallTotal * 1_000_000);
    const rowGates = rows.map((row) => {
      const controlRow = row.arms.A;
      const candidateRow = row.arms[arm];
      const controlMedian = nullableNumber(
        controlRow["medianGpuMilliseconds"],
      );
      const candidateMedian = nullableNumber(
        candidateRow["medianGpuMilliseconds"],
      );
      const gpuBandwidth = nullableNumber(candidateRow[
        "medianEffectivePackedWeightBandwidthGbPerSecond"]);
      const controlWallMedian = Number(controlRow["medianWallMilliseconds"]);
      const candidateWallMedian = Number(candidateRow[
        "medianWallMilliseconds"]);
      const wallBandwidth = Number(candidateRow[
        "medianEffectivePackedWeightWallBandwidthGbPerSecond"]);
      return Object.freeze({ rows: row.rows, controlMedianGpuMilliseconds:
        controlMedian, candidateMedianGpuMilliseconds: candidateMedian,
        medianGpuSpeedup: controlMedian === null || candidateMedian === null
          ? null : controlMedian / candidateMedian,
        candidateMedianGpuBandwidthGbPerSecond: gpuBandwidth,
        candidateGpuMedianFaster:
          controlMedian === null || candidateMedian === null
            ? null : candidateMedian < controlMedian,
        candidateGpuBandwidthPassed: gpuBandwidth === null
          ? null : gpuBandwidth >= REQUIRED_BANDWIDTH_GBPS,
        controlMedianWallMilliseconds: controlWallMedian,
        candidateMedianWallMilliseconds: candidateWallMedian,
        medianWallSpeedup: controlWallMedian / candidateWallMedian,
        candidateMedianWallBandwidthGbPerSecond: wallBandwidth,
        candidateWallMedianFaster: candidateWallMedian < controlWallMedian,
        candidateWallBandwidthPassed:
          wallBandwidth >= REQUIRED_BANDWIDTH_GBPS });
    });
    const gates = Object.freeze({
      requiredPairWins: REQUIRED_PAIR_WINS,
      gpuPairComparableCount, gpuPairWinsAmongComparable, gpuPairWins,
      wallPairWins,
      gpuPairWinsDiagnosticPassed:
        gpuPairComparableCount === TIMING_ROUND_COUNT
          ? gpuPairWinsAmongComparable >= REQUIRED_PAIR_WINS : null,
      authoritativeWallPairWinsPassed: wallPairWins >= REQUIRED_PAIR_WINS,
      everyM1M2GpuMedianFasterDiagnostic: rowGates.every((gate) =>
        gate.candidateGpuMedianFaster !== null)
        ? rowGates.every((gate) => gate.candidateGpuMedianFaster === true)
        : null,
      everyM1M2WallMedianFaster: rowGates.every((gate) =>
        gate.candidateWallMedianFaster),
      requiredAggregateLayerMixSpeedup: REQUIRED_SPEEDUP,
      aggregateGpuSpeedup,
      aggregateGpuSpeedupDiagnosticPassed: aggregateGpuSpeedup === null
        ? null : aggregateGpuSpeedup >= REQUIRED_SPEEDUP,
      aggregateWallSpeedup,
      authoritativeAggregateWallSpeedupPassed:
        aggregateWallSpeedup >= REQUIRED_SPEEDUP,
      requiredEffectivePackedWeightBandwidthGbPerSecond:
        REQUIRED_BANDWIDTH_GBPS,
      aggregateEffectivePackedWeightGpuBandwidthGbPerSecond:
        aggregateGpuBandwidth,
      aggregateGpuBandwidthDiagnosticPassed: aggregateGpuBandwidth === null
        ? null : aggregateGpuBandwidth >= REQUIRED_BANDWIDTH_GBPS,
      aggregateEffectivePackedWeightWallBandwidthGbPerSecond:
        aggregateWallBandwidth,
      authoritativeAggregateWallBandwidthPassed:
        aggregateWallBandwidth >= REQUIRED_BANDWIDTH_GBPS,
      everyM1M2GpuMedianBandwidthDiagnosticPassed: rowGates.every((gate) =>
        gate.candidateGpuBandwidthPassed !== null)
        ? rowGates.every((gate) => gate.candidateGpuBandwidthPassed === true)
        : null,
      everyM1M2WallMedianBandwidthPassed: rowGates.every((gate) =>
        gate.candidateWallBandwidthPassed),
    });
    const passed = gates.authoritativeWallPairWinsPassed &&
      gates.everyM1M2WallMedianFaster &&
      gates.authoritativeAggregateWallSpeedupPassed &&
      gates.authoritativeAggregateWallBandwidthPassed &&
      gates.everyM1M2WallMedianBandwidthPassed;
    return Object.freeze({ arm, rowGates: Object.freeze(rowGates),
      pairedRounds: Object.freeze(paired), gates, passed });
  });
  const passing = candidateSummaries.filter(({ passed }) => passed)
    .sort((left, right) =>
      Number(right.gates.aggregateWallSpeedup) -
      Number(left.gates.aggregateWallSpeedup));
  const selectedArm = passing[0]?.arm ?? null;
  const diagnostics = DIAGNOSTIC_SPECS.flatMap((spec) =>
    ([1, 2] as const).map((rowCount) => {
      const arms = Object.freeze(Object.fromEntries((["A", "B", "C"] as const)
        .map((arm) => {
          const sample = diagnosticSamples.find((entry) =>
            entry.shapeId === spec.id && entry.rows === rowCount &&
            entry.arm === arm);
          if (sample === undefined) {
            throw new Error("OPT-0083 diagnostic sample missing");
          }
          return [arm, Object.freeze({ gpuMilliseconds: sample.gpuMilliseconds,
            wallMilliseconds: sample.wallMilliseconds,
            effectivePackedWeightBandwidthGbPerSecond:
              sample.effectivePackedWeightBandwidthGbPerSecond,
            effectivePackedWeightWallBandwidthGbPerSecond:
              sample.effectivePackedWeightWallBandwidthGbPerSecond,
            validMultiplyAdds: sample.validMultiplyAdds,
            scheduledMultiplyAdds: sample.scheduledMultiplyAdds })];
        })));
      return Object.freeze({ id: spec.id, rows: rowCount,
        shape: Object.freeze({ rows: rowCount, inner: spec.inner,
          columns: spec.columns }), source: spec.source, arms });
    }));
  const positionCounts = Object.freeze(Object.fromEntries(
    (["A", "B", "C"] as const).map((arm) => [arm, Object.freeze([0, 1, 2]
      .map((position) => buildTimingRounds().filter((round) =>
        round.armOrder[position] === arm).length))]),
  ));
  return Object.freeze({ primarySampleCount: primarySamples.length,
    diagnosticSampleCount: diagnosticSamples.length,
    completeLayerPackedWeightBytes: LAYER_WEIGHT_BYTES,
    logicalLayerMultiplyAddsM1: LOGICAL_LAYER_MULTIPLY_ADDS_M1,
    rowStrata: Object.freeze(rows),
    candidateSummaries: Object.freeze(candidateSummaries), selectedArm,
    armPositionCounts: positionCounts,
    diagnostics: Object.freeze(diagnostics),
    rawPrimarySamples: Object.freeze(primarySamples),
    rawDiagnosticSamples: Object.freeze(diagnosticSamples),
    passed: selectedArm !== null });
}

function timestampDiagnosticsReceipt(
  primarySamples: readonly TimestampSample[],
  diagnosticSamples: readonly TimestampSample[],
): Readonly<Record<string, unknown>> {
  const summarize = (samples: readonly TimestampSample[]) => {
    const validSampleCount = samples.filter(({ timestampValid }) =>
      timestampValid).length;
    const reasonCounts: Record<string, number> = {};
    for (const sample of samples) {
      if (sample.timestampValid) continue;
      const reason = sample.timestampUnavailableReason ??
        "unspecified-webgpu-timestamp-unavailability";
      reasonCounts[reason] = (reasonCounts[reason] ?? 0) + 1;
    }
    return Object.freeze({ sampleCount: samples.length, validSampleCount,
      unavailableSampleCount: samples.length - validSampleCount,
      unavailableReasonCounts: Object.freeze(reasonCounts),
      everyTimestampValid: validSampleCount === samples.length });
  };
  const all = Object.freeze([...primarySamples, ...diagnosticSamples]);
  return Object.freeze({
    policy: "nullable-diagnostic-never-substituted-never-authoritative",
    primary: summarize(primarySamples),
    diagnostic: summarize(diagnosticSamples),
    complete: summarize(all),
  });
}

function diagnosticDispatches(
  prepared: PreparedHarness,
  weightSet: WeightSet,
  spec: DiagnosticSpec,
  rows: RowCount,
): Readonly<Record<Arm, AceGemmDispatch>> {
  if (spec.source === "tied-head-tail") {
    return prepared.tailDispatchSets[String(rows)]!.arms;
  }
  const roleIndex = spec.roleIndex!;
  const entry = requireDispatchSet(prepared.dispatchSets,
    weightSet.index, rows);
  return Object.freeze({ A: entry.arms.A[roleIndex]!,
    B: entry.arms.B[roleIndex]!, C: entry.arms.C[roleIndex]! });
}

async function buildIdentity(
  adapter: GPUAdapter,
  device: GPUDevice,
  weightSets: readonly WeightSet[],
): Promise<Readonly<Record<string, unknown>>> {
  const shapeIdentities: Readonly<Record<string, unknown>>[] = [];
  for (const spec of DIAGNOSTIC_SPECS) {
    for (const rows of [1, 2] as const) {
      const shape = Object.freeze({ rows, inner: spec.inner,
        columns: spec.columns });
      const direct = planAceOpt0083PlannerDirectLowRowBf16Gemv(shape);
      const panel = planAceOpt0083PlannerLowRowBf16Gemv(shape);
      shapeIdentities.push(Object.freeze({ id: spec.id, rows,
        shape, generatedWgslSha256: Object.freeze({
          A: await sha256Text(aceCorrectnessGemmWgsl(
            "reference-bf16", shape, false, "source-row-major",
          )),
          B: await sha256Text(
            aceOpt0083PlannerDirectLowRowBf16GemvWgsl(shape),
          ),
          C: await sha256Text(aceOpt0083PlannerLowRowBf16GemvWgsl(shape)),
        }),
        plans: Object.freeze({ B: planReceipt(direct),
          C: planReceipt(panel) }) }));
    }
  }
  return Object.freeze({ allocationBaselineCommit: ALLOCATION_BASELINE_COMMIT,
    controlKernel: "AceCorrectnessGemmKernel-reference-bf16-source-row-major",
    armBKernelId:
      ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID,
    armCKernelId: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID,
    sourceSha256: Object.freeze({ A: await sha256Text(gemmKernelSource),
      BC: await sha256Text(opt0083KernelSource) }),
    generatedShapes: Object.freeze(shapeIdentities),
    weightLayout: "source-row-major-bf16-pairs-lsb-u32",
    weightSetCount: WEIGHT_SET_COUNT,
    weightSetBytes: LAYER_WEIGHT_BYTES,
    weightSetSignatures: Object.freeze(weightSets.map(({ signature }) =>
      signature)),
    tiedHeadTailColumns: TIED_HEAD_TAIL_ROWS,
    tiedHeadTailWeightBytes: TAIL_WEIGHT_BYTES,
    adapterInfo: Object.freeze({
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
      subgroupMinSize: adapter.info.subgroupMinSize,
      subgroupMaxSize: adapter.info.subgroupMaxSize,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    }),
    adapterFeatures: Object.freeze([...adapter.features].sort()),
    adapterLimits: limitsReceipt(adapter.limits),
    explicitlyRequestedFeatures: Object.freeze(["timestamp-query"]),
    explicitlyRequestedLimits: Object.freeze({
      maxComputeWorkgroupStorageSize: 17_024,
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 128,
      maxStorageBufferBindingSize: Math.max(
        LAYER_WEIGHT_BYTES, TAIL_WEIGHT_BYTES,
      ),
      maxBufferSize: Math.max(LAYER_WEIGHT_BYTES, TAIL_WEIGHT_BYTES) +
        2 * STORAGE_GUARD_BYTES,
    }),
    deviceFeatures: Object.freeze([...device.features].sort()),
    deviceLimits: limitsReceipt(device.limits),
    armCRequiresRaisedProductionWorkgroupStorageLimit: true,
    ordinaryProductionRequestedWorkgroupStorageBytesAtAllocation: 16_384,
    armCRequiredWorkgroupStorageBytes: 17_024,
    browserUserAgent: navigator.userAgent,
    browserLanguage: navigator.language,
    browserHardwareConcurrency: navigator.hardwareConcurrency,
    stockWebGpuOnly: true,
    experimentalBrowserFlags: false });
}

function planReceipt(
  plan: ReturnType<typeof planAceOpt0083PlannerDirectLowRowBf16Gemv> |
    ReturnType<typeof planAceOpt0083PlannerLowRowBf16Gemv>,
): Readonly<Record<string, unknown>> {
  return Object.freeze({ kernelId: plan.kernelId, rows: plan.rows,
    inner: plan.inner, columns: plan.columns, tileRows: plan.tileRows,
    tileColumns: plan.tileColumns, tileInner: plan.tileInner,
    workgroupSize: plan.workgroupSize, workgroupCount: plan.workgroupCount,
    outputRangeCount: plan.outputRangeCount,
    scheduledMultiplyAdds: plan.scheduledMultiplyAdds,
    validMultiplyAdds: plan.validMultiplyAdds,
    workgroupStorageBytes: plan.workgroupStorageBytes,
    barriersPerWorkgroup: plan.barriersPerWorkgroup,
    estimatedGlobalWeightBytes: plan.estimatedGlobalWeightBytes,
    estimatedGlobalActivationBytes: plan.estimatedGlobalActivationBytes });
}

function limitsReceipt(limits: GPUSupportedLimits): Readonly<Record<string, number>> {
  return Object.freeze({ maxBufferSize: Number(limits.maxBufferSize),
    maxStorageBufferBindingSize: Number(limits.maxStorageBufferBindingSize),
    maxComputeInvocationsPerWorkgroup:
      Number(limits.maxComputeInvocationsPerWorkgroup),
    maxComputeWorkgroupSizeX: Number(limits.maxComputeWorkgroupSizeX),
    maxComputeWorkgroupStorageSize:
      Number(limits.maxComputeWorkgroupStorageSize),
    maxComputeWorkgroupsPerDimension:
      Number(limits.maxComputeWorkgroupsPerDimension) });
}

function requireDispatchSet(
  sets: readonly DispatchSet[],
  weightSetIndex: number,
  rows: RowCount,
): DispatchSet {
  const value = sets.find((entry) => entry.weightSetIndex === weightSetIndex &&
    entry.rows === rows);
  if (value === undefined) {
    throw new Error(`OPT-0083 missing set${weightSetIndex} M${rows}`);
  }
  return value;
}

function activationBinding(
  activations: Readonly<Record<string, GPUBuffer>>,
  rows: RowCount,
  inner: number,
): GPUBufferBinding {
  const buffer = activations[String(inner)];
  if (buffer === undefined) {
    throw new Error(`OPT-0083 activation K${inner} is missing`);
  }
  return binding(buffer, 0, rows * inner * Float32Array.BYTES_PER_ELEMENT);
}

function binding(
  buffer: GPUBuffer,
  offset: number,
  size: number,
): GPUBufferBinding {
  return Object.freeze({ buffer, offset, size });
}

function parseArmOrder(value: string): readonly Arm[] {
  const arms = [...value] as Arm[];
  if (arms.length !== 3 || new Set(arms).size !== 3 ||
    arms.some((arm) => arm !== "A" && arm !== "B" && arm !== "C")) {
    throw new Error("OPT-0083 arm order changed");
  }
  return Object.freeze(arms);
}

function productionBf16Pair(mixed: number): number {
  const magnitudes = [0x3d80, 0x3e00, 0x3e80, 0x3f00] as const;
  const low = magnitudes[mixed & 3]! | ((mixed >>> 4) & 0x8000);
  const highMixed = mix32(mixed ^ 0xa5a5_5a5a);
  const high = magnitudes[highMixed & 3]! |
    ((highMixed >>> 4) & 0x8000);
  return ((high << 16) | low) >>> 0;
}

function cancellationBf16Pair(mixed: number): number {
  const magnitudes = [0x3d80, 0x3e00, 0x3e80, 0x3f00] as const;
  const low = magnitudes[mixed & 3]!;
  const high = low | 0x8000;
  return ((high << 16) | low) >>> 0;
}

async function executeAndDrain(
  device: GPUDevice,
  dispatches: readonly AceGemmDispatch[],
): Promise<void> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (const dispatch of dispatches) dispatch.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

function validateThermalLaunch(
  launch: ThermalLaunch,
  readyAtEpochMilliseconds: number,
): void {
  if (launch.source !==
      "notifyutil-com.apple.system.thermalpressurelevel" ||
    launch.command !==
      "notifyutil -g com.apple.system.thermalpressurelevel" ||
    launch.pollMilliseconds !== 1_000 ||
    launch.nonNominalObservationCount !== 0 ||
    launch.missingObservationCount !== 0 ||
    launch.traceStartedAtEpochMilliseconds > readyAtEpochMilliseconds ||
    launch.gateStartedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    launch.gateCompletedAtEpochMilliseconds -
      launch.gateStartedAtEpochMilliseconds < 30_000 ||
    launch.maximumPollGapMilliseconds > 1_250 ||
    Date.now() - launch.gateCompletedAtEpochMilliseconds > 5_500) {
    throw new Error("OPT-0083 worker rejected the thermal launch identity");
  }
}

function requireAdapter(adapter: GPUAdapter): void {
  if (!adapter.features.has("timestamp-query") ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 128 ||
    adapter.limits.maxComputeWorkgroupSizeX < 128 ||
    adapter.limits.maxComputeWorkgroupStorageSize < 17_024 ||
    adapter.limits.maxStorageBufferBindingSize <
      Math.max(LAYER_WEIGHT_BYTES, TAIL_WEIGHT_BYTES) ||
    adapter.limits.maxBufferSize <
      Math.max(LAYER_WEIGHT_BYTES, TAIL_WEIGHT_BYTES) +
        2 * STORAGE_GUARD_BYTES) {
    throw new Error(
      "OPT-0083 requires timestamp-query, WG128, 17,024-byte workgroup storage, and full planner binding limits",
    );
  }
}

function checkpoint(signal: AbortSignal): void {
  if (signal.aborted) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new DOMException("OPT-0083 cancelled", "AbortError");
  }
}

async function rejects(operation: () => Promise<unknown>): Promise<boolean> {
  try {
    await operation();
    return false;
  } catch {
    return true;
  }
}

function mean(values: readonly number[]): number {
  if (values.length === 0 || values.some((value) =>
    !Number.isFinite(value) || value <= 0)) {
    throw new Error("OPT-0083 mean requires finite positive values");
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function nullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value : null;
}

function median(values: readonly number[]): number {
  if (values.length === 0 || values.some((value) =>
    !Number.isFinite(value) || value <= 0)) {
    throw new Error("OPT-0083 median requires finite positive values");
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function tflops(multiplyAdds: number, milliseconds: number): number {
  return 2 * multiplyAdds / (milliseconds * 1_000_000_000);
}

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function mix32(value: number): number {
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb_352d);
  value ^= value >>> 15;
  value = Math.imul(value, 0x846c_a68b);
  value ^= value >>> 16;
  return value >>> 0;
}

function hex32(value: number): string {
  return `0x${(value >>> 0).toString(16).padStart(8, "0")}`;
}

function requireLittleEndianHost(): void {
  const word = new Uint16Array([0x0102]);
  if (new Uint8Array(word.buffer)[0] !== 0x02) {
    throw new Error("OPT-0083 fixtures require a little-endian host");
  }
}

async function sha256Text(value: string): Promise<string> {
  return sha256Bytes(new TextEncoder().encode(value));
}

async function sha256Bytes(value: Uint8Array): Promise<string> {
  const copy = value.slice();
  const digest = new Uint8Array(await crypto.subtle.digest(
    "SHA-256", copy.buffer,
  ));
  return [...digest].map((byte) =>
    byte.toString(16).padStart(2, "0")).join("");
}

function errorValue(error: unknown): Readonly<Record<string, unknown>> {
  return error instanceof Error
    ? Object.freeze({ name: error.name, message: error.message,
        stack: error.stack ?? null })
    : Object.freeze({ name: typeof error, message: String(error), stack: null });
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

async function yieldToWorker(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

async function settleEvents(): Promise<void> {
  await Promise.resolve();
  await yieldToWorker();
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  private readonly sizes = new Map<GPUBuffer, number>();
  private created = 0;
  private destroyed = 0;
  private maps = 0;
  private unmaps = 0;
  private activeMaps = 0;
  private liveBytesValue = 0;
  private maximumLiveBytesValue = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    const size = Number(descriptor.size);
    this.live.add(buffer);
    this.sizes.set(buffer, size);
    this.created += 1;
    this.liveBytesValue += size;
    this.maximumLiveBytesValue = Math.max(this.maximumLiveBytesValue,
      this.liveBytesValue);
    if (descriptor.mappedAtCreation === true) {
      this.maps += 1;
      this.activeMaps += 1;
    }
    return buffer;
  }

  async mapRead(buffer: GPUBuffer): Promise<void> {
    await buffer.mapAsync(GPUMapMode.READ);
    this.maps += 1;
    this.activeMaps += 1;
  }

  unmap(buffer: GPUBuffer): void {
    if (buffer.mapState !== "mapped") {
      throw new Error("OPT-0083 attempted to unmap an unmapped buffer");
    }
    buffer.unmap();
    this.unmaps += 1;
    this.activeMaps -= 1;
    if (this.activeMaps < 0) {
      throw new Error("OPT-0083 map accounting became negative");
    }
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    buffer.destroy();
    this.destroyed += 1;
    this.liveBytesValue -= this.sizes.get(buffer) ?? 0;
    this.sizes.delete(buffer);
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number>> {
    return Object.freeze({ createdBufferCount: this.created,
      destroyedBufferCount: this.destroyed, liveBufferCount: this.live.size,
      liveBytes: this.liveBytesValue,
      maximumLiveBytes: this.maximumLiveBytesValue,
      mapCount: this.maps, unmapCount: this.unmaps,
      activeMapCount: this.activeMaps });
  }
}
