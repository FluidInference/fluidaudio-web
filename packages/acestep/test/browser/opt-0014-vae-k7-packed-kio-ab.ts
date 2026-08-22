/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import fixed32CoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.ts?raw";
import packedKioCoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-subgroup.ts?raw";
import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
  AceFp16VaeConv1dSubgroupKernel,
  aceFp16VaeConv1dSubgroupWgsl,
  planAceFp16VaeConv1dSubgroupRange,
  type AceFp16VaeConv1dSubgroupDispatch,
} from "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import {
  ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_KERNEL_ID,
  ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_REPACK_KERNEL_ID,
  AceOpt0014VaeConv1dPackedKioSubgroupKernel,
  aceOpt0014VaeConv1dPackedKioRepackWgsl,
  aceOpt0014VaeConv1dPackedKioWgsl,
  planAceOpt0014VaeConv1dPackedKioRange,
  planAceOpt0014VaeConv1dPackedKioWeight,
  type AceOpt0014VaeConv1dPackedKioDispatch,
  type AceOpt0014VaeConv1dRepackDispatch,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-subgroup.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dOutputStorage,
  type AceFp16VaeConv1dPlan,
} from "../../src/webgpu/kernels/vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderConvOperation,
} from "../../src/webgpu/vae-decoder.js";

type KernelArm = "fixed32" | "packedKio";
type TimingStratum = "first" | "interior" | "tail";

interface ExactRange {
  readonly rangeIndex: number;
  readonly base: number;
  readonly count: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
}

export interface Opt0014ProbeRange {
  readonly id: TimingStratum;
  readonly stratum: TimingStratum;
  readonly source: "graph" | "synthetic-centered";
  readonly rangeIndex: number | null;
  readonly base: number;
  readonly count: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
  readonly weight: number | null;
}

export interface Opt0014C300Operation {
  readonly operationIndex: number;
  readonly label: string;
  readonly shape: AceVaeConv1dShape;
  readonly outputStorage: AceFp16VaeConv1dOutputStorage;
  readonly outputElements: number;
  readonly ranges: readonly ExactRange[];
  readonly correctnessProbes: readonly Opt0014ProbeRange[];
  readonly timingStrata: readonly Opt0014ProbeRange[];
}

export interface Opt0014ThermalGate {
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

export interface Opt0014WeightedStratumInput {
  readonly operationLabel: string;
  readonly stratum: TimingStratum;
  readonly weight: number;
  readonly fixed32Samples: readonly number[];
  readonly packedKioSamples: readonly number[];
}

interface EncodableDispatch {
  readonly kernelId: string;
  readonly outputRange: Readonly<{
    readonly base: number;
    readonly count: number;
    readonly workgroupsX: number;
    readonly workgroupsY: number;
  }>;
  encode(pass: GPUComputePassEncoder): void;
}

interface GuardedBinding {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
}

interface PreparedWeight {
  readonly native: GPUBuffer;
  readonly packed: GuardedBinding;
  readonly repack: AceOpt0014VaeConv1dRepackDispatch;
  readonly uniqueU16Words: number;
  readonly verification: Readonly<Record<string, unknown>>;
}

interface PreparedOperation {
  readonly topology: Opt0014C300Operation;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly weight: PreparedWeight;
  readonly dispatches: Readonly<
    Record<KernelArm, Readonly<Record<TimingStratum, EncodableDispatch>>>
  >;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly topology: readonly Opt0014C300Operation[];
  readonly operations: readonly PreparedOperation[];
  readonly tracker: BufferTracker;
  readonly fixed32Kernel: AceFp16VaeConv1dSubgroupKernel;
  readonly packedKioKernel: AceOpt0014VaeConv1dPackedKioSubgroupKernel;
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly repackCorrectness: Readonly<Record<string, unknown>>;
  readonly sourceAuthority: Readonly<Record<string, unknown>>;
  readonly preparedCompletedAtEpochMilliseconds: number;
  destroy(): Readonly<Record<string, unknown>>;
}

interface ReadbackArm {
  readonly sha256: string;
  readonly nonFiniteCount: number;
  readonly prefillQNaNCount: number;
  readonly guardsAndCanariesUntouched: boolean;
  readonly words: Uint16Array | Uint32Array;
}

const EXPERIMENT_ID = "OPT-0014" as const;
const CORE_COMMIT =
  "12e128ab323c0024ed683313b4d06c07041213e7" as const;
const PACKED_KIO_CORE_SOURCE_SHA256 =
  "802cb0ad1d2c57c0cc51cbd4a7c88632e00d543b526f2ed0b94e9fc393a3d8d8" as const;
const FIXED32_CORE_SOURCE_SHA256 =
  "7d218516d6b2c8d6e3332a53101be5fdeae1142096c442433915bfa58941ce32" as const;
const C300_INPUT_FRAMES = 300;
const C300_OPERATION_COUNT = 17;
const C300_EXACT_RANGE_COUNT = 2_404;
const CORRECTNESS_PROBE_COUNT = 51;
const TIMING_STRATUM_COUNT = 50;
const CORRECTNESS_EXECUTIONS = 2;
const EXPECTED_COMPARED_OUTPUT_WORDS = 13_854_720;
const EXPECTED_UNIQUE_REPACK_U16_WORDS = 30_508_800;
const EXPECTED_REPACK_COMPARED_U16_WORDS = 61_017_600;
const EXPECTED_PACKED_WEIGHT_BYTES = 61_017_600;
const RANGE_CONTROL_BYTES = 16;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const ADJACENT_CANARY_U32 = 0x5aa5_3cc3;
const OUTPUT_PREFILL_QNAN_F16 = 0x7e55;
const OUTPUT_PREFILL_QNAN_F32 = 0x7fc5_5555;
const REPACK_PREFILL_QNAN_F16 = 0x7e6d;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const THERMAL_POLL_MILLISECONDS = 1_000;
const MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS = 1_250;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 5_000;

const INPUT_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x2400, 0xa400,
  0x2c00, 0xac00, 0x3000, 0xb000,
  0x3400, 0xb400, 0x3555, 0xb555,
  0x1800, 0x9800, 0x0001, 0x8001,
]);
const BIAS_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x2000, 0xa000,
  0x2400, 0xa400, 0x2800, 0xa800,
]);

export const OPT_0014_C300_EXPECTED_TOPOLOGY = Object.freeze([
  expected("conv1", 300, 64, 2_048, 1, 3, 2,
    524_288, null, 290_816, 32_768, null, 524_288, 90_112),
  expected("block-0-res-1-conv1", 3_000, 1_024, 1_024, 1, 3, 94,
    32_768, 47, 1_540_096, 32_768, 92, 3_047_424, 24_576),
  expected("block-0-res-2-conv1", 3_000, 1_024, 1_024, 3, 9, 94,
    32_768, 47, 1_540_096, 32_768, 92, 3_047_424, 24_576),
  expected("block-0-res-3-conv1", 3_000, 1_024, 1_024, 9, 27, 94,
    32_768, 47, 1_540_096, 32_768, 92, 3_047_424, 24_576),
  expected("block-1-res-1-conv1", 18_000, 512, 512, 1, 3, 141,
    65_536, 70, 4_587_520, 65_536, 139, 9_175_040, 40_960),
  expected("block-1-res-2-conv1", 18_000, 512, 512, 3, 9, 141,
    65_536, 70, 4_587_520, 65_536, 139, 9_175_040, 40_960),
  expected("block-1-res-3-conv1", 18_000, 512, 512, 9, 27, 141,
    65_536, 70, 4_587_520, 65_536, 139, 9_175_040, 40_960),
  expected("block-2-res-1-conv1", 72_000, 256, 256, 1, 3, 141,
    131_072, 70, 9_175_040, 131_072, 139, 18_350_080, 81_920),
  expected("block-2-res-2-conv1", 72_000, 256, 256, 3, 9, 141,
    131_072, 70, 9_175_040, 131_072, 139, 18_350_080, 81_920),
  expected("block-2-res-3-conv1", 72_000, 256, 256, 9, 27, 141,
    131_072, 70, 9_175_040, 131_072, 139, 18_350_080, 81_920),
  expected("block-3-res-1-conv1", 288_000, 128, 128, 1, 3, 141,
    262_144, 70, 18_350_080, 262_144, 139, 36_700_160, 163_840),
  expected("block-3-res-2-conv1", 288_000, 128, 128, 3, 9, 141,
    262_144, 70, 18_350_080, 262_144, 139, 36_700_160, 163_840),
  expected("block-3-res-3-conv1", 288_000, 128, 128, 9, 27, 141,
    262_144, 70, 18_350_080, 262_144, 139, 36_700_160, 163_840),
  expected("block-4-res-1-conv1", 576_000, 128, 128, 1, 3, 282,
    262_144, 141, 36_962_304, 262_144, 280, 73_662_464, 65_536),
  expected("block-4-res-2-conv1", 576_000, 128, 128, 3, 9, 282,
    262_144, 141, 36_962_304, 262_144, 280, 73_662_464, 65_536),
  expected("block-4-res-3-conv1", 576_000, 128, 128, 9, 27, 282,
    262_144, 141, 36_962_304, 262_144, 280, 73_662_464, 65_536),
  expected("conv2", 576_000, 128, 2, 1, 3, 5,
    262_144, 2, 524_288, 262_144, 3, 1_048_576, 103_424),
]);

export function buildOpt0014C300Topology(): readonly Opt0014C300Operation[] {
  const graph = planAceVaeDecoder(C300_INPUT_FRAMES);
  const cooperative = planAceVaeDecoderQuanta(graph);
  const operations = graph.operations
    .map((operation, operationIndex) => ({ operation, operationIndex }))
    .filter((entry): entry is Readonly<{
      operation: AceVaeDecoderConvOperation;
      operationIndex: number;
    }> => entry.operation.kind === "conv1d" &&
      entry.operation.shape.kernelSize === 7)
    .map(({ operation, operationIndex }, ordinal) => {
      const outputStorage = operation.bias === undefined
        ? "float32" as const
        : "float16" as const;
      const plan = planAceFp16VaeConv1d(operation.shape, outputStorage);
      const ranges = cooperative.quanta
        .filter((quantum) => quantum.operationIndex === operationIndex)
        .map((quantum, rangeIndex): ExactRange => {
          if (quantum.operationKind !== "conv1d" ||
            quantum.primitives.length !== 1) {
            throw new Error(`${operation.label} C300 quantum topology changed`);
          }
          const primitive = quantum.primitives[0]!;
          if (primitive.firstOutputChannel !== 0 ||
            primitive.outputChannels !== plan.outputChannels ||
            primitive.outputBase !== quantum.logicalOutputBase ||
            primitive.outputCount !== quantum.logicalOutputCount ||
            primitive.outputBase % plan.outputChannels !== 0 ||
            primitive.outputCount % plan.outputChannels !== 0) {
            throw new Error(`${operation.label} C300 primitive topology changed`);
          }
          const range = Object.freeze({
            rangeIndex,
            base: primitive.outputBase,
            count: primitive.outputCount,
            firstOutputRow: primitive.outputBase / plan.outputChannels,
            outputRowCount: primitive.outputCount / plan.outputChannels,
          });
          planAceFp16VaeConv1dSubgroupRange(plan, range);
          planAceOpt0014VaeConv1dPackedKioRange(plan, range);
          return range;
        });
      if (ranges.length < 2) {
        throw new Error(`${operation.label} lacks first/tail C300 ranges`);
      }
      const first = probeFromGraph("first", ranges[0]!, 1);
      const tail = probeFromGraph("tail", ranges.at(-1)!, 1);
      const interior = ranges.length === 2
        ? centeredConv1Probe(plan)
        : probeFromGraph(
          "interior",
          ranges[Math.floor(ranges.length / 2)]!,
          ranges.length - 2,
        );
      const correctnessProbes = Object.freeze([first, interior, tail]);
      const timingStrata = Object.freeze(
        correctnessProbes.filter((probe) => probe.source === "graph"),
      );
      const operationTopology = Object.freeze({
        operationIndex,
        label: operation.label,
        shape: operation.shape,
        outputStorage,
        outputElements: plan.outputElements,
        ranges: Object.freeze(ranges),
        correctnessProbes,
        timingStrata,
      });
      assertExpectedTopology(operationTopology, ordinal);
      return operationTopology;
    });
  const exactRangeCount = operations.reduce(
    (sum, operation) => sum + operation.ranges.length,
    0,
  );
  const correctnessProbeCount = operations.reduce(
    (sum, operation) => sum + operation.correctnessProbes.length,
    0,
  );
  const timingStratumCount = operations.reduce(
    (sum, operation) => sum + operation.timingStrata.length,
    0,
  );
  const timingWeight = operations.flatMap((operation) => operation.timingStrata)
    .reduce((sum, stratum) => sum + stratum.weight!, 0);
  const comparedWords = operations.flatMap((operation) =>
    operation.correctnessProbes
  ).reduce((sum, probe) => sum + probe.count, 0) * CORRECTNESS_EXECUTIONS;
  if (operations.length !== C300_OPERATION_COUNT ||
    exactRangeCount !== C300_EXACT_RANGE_COUNT ||
    correctnessProbeCount !== CORRECTNESS_PROBE_COUNT ||
    timingStratumCount !== TIMING_STRATUM_COUNT ||
    timingWeight !== C300_EXACT_RANGE_COUNT ||
    comparedWords !== EXPECTED_COMPARED_OUTPUT_WORDS) {
    throw new Error("OPT-0014 exact C300 topology totals changed");
  }
  return Object.freeze(operations);
}

export function parseOpt0014ThermalGate(
  parameters: URLSearchParams,
  preparedCompletedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0014ThermalGate {
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
  const durationMilliseconds =
    completedAtEpochMilliseconds - startedAtEpochMilliseconds;
  const launchDelayMilliseconds =
    launchedAtEpochMilliseconds - completedAtEpochMilliseconds;
  if (source !== THERMAL_SOURCE) {
    throw new Error("OPT-0014 requires the accepted notifyutil thermal source");
  }
  if (startedAtEpochMilliseconds < preparedCompletedAtEpochMilliseconds ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    completedAtEpochMilliseconds > launchedAtEpochMilliseconds ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS) {
    throw new Error(
      "OPT-0014 thermal interval is stale, short, or predates preparation",
    );
  }
  if (!Number.isSafeInteger(observationCount) ||
    observationCount < Math.floor(durationMilliseconds / pollMilliseconds) + 1 ||
    pollMilliseconds !== THERMAL_POLL_MILLISECONDS ||
    maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS ||
    nonNominalObservationCount !== 0) {
    throw new Error(
      "OPT-0014 thermal observations are incomplete or non-nominal",
    );
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

export function summarizeOpt0014WeightedTiming(
  strata: readonly Opt0014WeightedStratumInput[],
): Readonly<Record<string, unknown>> {
  if (strata.length !== TIMING_STRATUM_COUNT) {
    throw new RangeError("OPT-0014 requires exactly 50 timing strata");
  }
  let exactRangeCount = 0;
  let fixed32ProjectedMilliseconds = 0;
  let packedKioProjectedMilliseconds = 0;
  const summaries = strata.map((stratum) => {
    if (!Number.isSafeInteger(stratum.weight) || stratum.weight < 1) {
      throw new RangeError("OPT-0014 timing weight must be positive");
    }
    const fixed32Median = median(stratum.fixed32Samples);
    const packedKioMedian = median(stratum.packedKioSamples);
    exactRangeCount += stratum.weight;
    fixed32ProjectedMilliseconds += fixed32Median * stratum.weight;
    packedKioProjectedMilliseconds += packedKioMedian * stratum.weight;
    return Object.freeze({
      operationLabel: stratum.operationLabel,
      stratum: stratum.stratum,
      weight: stratum.weight,
      fixed32Samples: Object.freeze([...stratum.fixed32Samples]),
      packedKioSamples: Object.freeze([...stratum.packedKioSamples]),
      fixed32MedianMilliseconds: fixed32Median,
      packedKioMedianMilliseconds: packedKioMedian,
      medianSpeedup: fixed32Median / packedKioMedian,
    });
  });
  if (exactRangeCount !== C300_EXACT_RANGE_COUNT) {
    throw new Error("OPT-0014 weighted timing did not cover 2,404 ranges");
  }
  return Object.freeze({
    exactRangeCount,
    fixed32ProjectedMilliseconds,
    packedKioProjectedMilliseconds,
    projectedSpeedup:
      fixed32ProjectedMilliseconds / packedKioProjectedMilliseconds,
    strata: Object.freeze(summaries),
  });
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
    const bytes = Number(descriptor.size);
    this.live.add(buffer);
    this.sizes.set(buffer, bytes);
    this.created += 1;
    this.liveBytes += bytes;
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

interface SharedResources {
  readonly input: GPUBuffer;
  readonly bias: GPUBuffer;
  readonly outputs: Readonly<Record<KernelArm, GPUBuffer>>;
  readonly f16Prefill: GPUBuffer;
  readonly f32Prefill: GPUBuffer;
  readonly repackPrefill: GPUBuffer;
  readonly controls: GPUBuffer;
  readonly controlOffsets: ReadonlyMap<string, number>;
  readonly maximumOutputBindingBytes: number;
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
    progress.textContent = "running balanced weighted C300 timing";
    const owned = prepared;
    prepared = undefined;
    void runTimedGate(owned).then(
      (result) => finishPage("passed", result),
      (error: unknown) => {
        owned.destroy();
        finishPage("failed", failureReceipt(error));
      },
    );
  }, { once: true });
}

async function prepareGate(
  updateProgress: (message: string) => void,
): Promise<PreparedGate> {
  const topology = buildOpt0014C300Topology();
  const sourceAuthority = await buildSourceAuthority(topology);
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter, topology);
  const device = await adapter.requestDevice({
    label: "ace-opt-0014-packed-kio-k7-ab-device",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: requiredDeviceLimits(adapter, topology),
  });
  const fixedCapability = Object.freeze({
    subgroupMinSize: 32 as const,
    subgroupMaxSize: 32 as const,
  });
  const tracker = new BufferTracker();
  const fixed32Kernel = AceFp16VaeConv1dSubgroupKernel.create(
    device,
    fixedCapability,
  );
  const packedKioKernel =
    AceOpt0014VaeConv1dPackedKioSubgroupKernel.create(
      device,
      fixedCapability,
    );
  let destroyed = false;
  const destroy = (): Readonly<Record<string, unknown>> => {
    if (destroyed) {
      return Object.freeze({ ...tracker.receipt(), idempotent: true });
    }
    destroyed = true;
    fixed32Kernel.destroy();
    packedKioKernel.destroy();
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
    updateProgress("allocating bounded shared C300 resources");
    const shared = createSharedResources(device, tracker, topology);
    const operations: PreparedOperation[] = [];
    const correctnessCases: unknown[] = [];
    const repackCases: unknown[] = [];
    let comparedOutputWordCount = 0;
    let comparedOutputU16WordCount = 0;
    let comparedOutputU32WordCount = 0;
    let uniqueRepackedU16WordCount = 0;
    let comparedRepackedU16WordCount = 0;
    for (const [index, operation] of topology.entries()) {
      updateProgress(
        `repack/correctness ${index + 1}/${topology.length}: ${operation.label}`,
      );
      const prepared = await prepareOperation(
        device,
        tracker,
        fixed32Kernel,
        packedKioKernel,
        shared,
        operation,
      );
      operations.push(prepared);
      correctnessCases.push(prepared.correctness);
      repackCases.push(prepared.weight.verification);
      comparedOutputWordCount += Number(
        prepared.correctness["comparedWordCount"],
      );
      if (operation.outputStorage === "float16") {
        comparedOutputU16WordCount += Number(
          prepared.correctness["comparedWordCount"],
        );
      } else {
        comparedOutputU32WordCount += Number(
          prepared.correctness["comparedWordCount"],
        );
      }
      uniqueRepackedU16WordCount += prepared.weight.uniqueU16Words;
      comparedRepackedU16WordCount += prepared.weight.uniqueU16Words * 2;
      await yieldToBrowser();
    }
    await device.queue.onSubmittedWorkDone();
    if (comparedOutputWordCount !== EXPECTED_COMPARED_OUTPUT_WORDS ||
      comparedOutputU16WordCount !== 12_599_296 ||
      comparedOutputU32WordCount !== 1_255_424 ||
      uniqueRepackedU16WordCount !== EXPECTED_UNIQUE_REPACK_U16_WORDS ||
      comparedRepackedU16WordCount !== EXPECTED_REPACK_COMPARED_U16_WORDS) {
      throw new Error("OPT-0014 correctness coverage totals changed");
    }
    const correctness = Object.freeze({
      operationCount: operations.length,
      probeCount: CORRECTNESS_PROBE_COUNT,
      executionsPerProbe: CORRECTNESS_EXECUTIONS,
      comparedWordCount: comparedOutputWordCount,
      comparedU16WordCount: comparedOutputU16WordCount,
      comparedU32WordCount: comparedOutputU32WordCount,
      rawU16ExceptFinalRawU32: true,
      mismatchCount: 0,
      qNaNPrefillCompleteWrites: true,
      guardsAndAdjacentCanariesUntouched: true,
      deterministicRerunHashes: true,
      cases: Object.freeze(correctnessCases),
    });
    const repackCorrectness = Object.freeze({
      operationCount: operations.length,
      executionsPerOperation: 2,
      uniqueU16WordCount: uniqueRepackedU16WordCount,
      comparedU16WordCount: comparedRepackedU16WordCount,
      packedPayloadBytes: EXPECTED_PACKED_WEIGHT_BYTES,
      mismatchCount: 0,
      qNaNPrefillCompleteWrites: true,
      redzonesUntouched: true,
      deterministicRerunHashes: true,
      cases: Object.freeze(repackCases),
    });
    return Object.freeze({
      adapter,
      device,
      topology,
      operations: Object.freeze(operations),
      tracker,
      fixed32Kernel,
      packedKioKernel,
      correctness,
      repackCorrectness,
      sourceAuthority,
      preparedCompletedAtEpochMilliseconds: Date.now(),
      destroy,
    });
  } catch (error) {
    destroy();
    throw error;
  }
}

function createSharedResources(
  device: GPUDevice,
  tracker: BufferTracker,
  topology: readonly Opt0014C300Operation[],
): SharedResources {
  const plans = topology.map((operation) => planAceFp16VaeConv1d(
    operation.shape,
    operation.outputStorage,
  ));
  const maximumInputBindingBytes = Math.max(
    ...plans.map((plan) => plan.inputBindingBytes),
  );
  const maximumOutputBindingBytes = Math.max(
    ...plans.map((plan) => plan.outputBindingBytes),
  );
  const maximumBiasBindingBytes = Math.max(
    ...plans.map((plan) => plan.biasBindingBytes),
  );
  const maximumF16ProbeBytes = Math.max(...topology.flatMap((operation) =>
    operation.outputStorage === "float16"
      ? operation.correctnessProbes.map((probe) => probe.count * 2)
      : [4]
  ));
  const maximumF32ProbeBytes = Math.max(...topology.flatMap((operation) =>
    operation.outputStorage === "float32"
      ? operation.correctnessProbes.map((probe) => probe.count * 4)
      : [4]
  ));
  const maximumWeightBytes = Math.max(...topology.map((operation) =>
    planAceOpt0014VaeConv1dPackedKioWeight(operation.shape).packedBindingBytes
  ));
  const input = createPeriodicU16Buffer(
    device,
    tracker,
    "opt-0014-shared-input",
    maximumInputBindingBytes,
    INPUT_PATTERN,
  );
  const bias = createPeriodicU16Buffer(
    device,
    tracker,
    "opt-0014-shared-bias",
    maximumBiasBindingBytes,
    BIAS_PATTERN,
  );
  const outputs = Object.freeze({
    fixed32: createUninitializedBuffer(
      device,
      tracker,
      "opt-0014-fixed32-output",
      STORAGE_GUARD_BYTES + maximumOutputBindingBytes + STORAGE_GUARD_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    ),
    packedKio: createUninitializedBuffer(
      device,
      tracker,
      "opt-0014-packed-kio-output",
      STORAGE_GUARD_BYTES + maximumOutputBindingBytes + STORAGE_GUARD_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    ),
  });
  const f16Prefill = createRepeatedU32Buffer(
    device,
    tracker,
    "opt-0014-f16-qnan-prefill",
    maximumF16ProbeBytes,
    OUTPUT_PREFILL_QNAN_F16 | (OUTPUT_PREFILL_QNAN_F16 << 16),
  );
  const f32Prefill = createRepeatedU32Buffer(
    device,
    tracker,
    "opt-0014-f32-qnan-prefill",
    maximumF32ProbeBytes,
    OUTPUT_PREFILL_QNAN_F32,
  );
  const repackPrefill = createRepeatedU32Buffer(
    device,
    tracker,
    "opt-0014-repack-qnan-prefill",
    maximumWeightBytes,
    REPACK_PREFILL_QNAN_F16 | (REPACK_PREFILL_QNAN_F16 << 16),
  );
  const controls = createRangeControls(device, tracker, topology);
  return Object.freeze({
    input,
    bias,
    outputs,
    f16Prefill,
    f32Prefill,
    repackPrefill,
    controls: controls.buffer,
    controlOffsets: controls.offsets,
    maximumOutputBindingBytes,
  });
}

async function prepareOperation(
  device: GPUDevice,
  tracker: BufferTracker,
  fixed32Kernel: AceFp16VaeConv1dSubgroupKernel,
  packedKioKernel: AceOpt0014VaeConv1dPackedKioSubgroupKernel,
  shared: SharedResources,
  topology: Opt0014C300Operation,
): Promise<PreparedOperation> {
  const plan = planAceFp16VaeConv1d(
    topology.shape,
    topology.outputStorage,
  );
  const weight = await prepareWeight(
    device,
    tracker,
    packedKioKernel,
    shared.repackPrefill,
    topology,
  );
  const fixed32Dispatches = {} as Record<
    TimingStratum,
    AceFp16VaeConv1dSubgroupDispatch
  >;
  const packedKioDispatches = {} as Record<
    TimingStratum,
    AceOpt0014VaeConv1dPackedKioDispatch
  >;
  for (const probe of topology.correctnessProbes) {
    const controlOffset = shared.controlOffsets.get(
      probeKey(topology.label, probe.id),
    );
    if (controlOffset === undefined) {
      throw new Error(`${topology.label} ${probe.id} control missing`);
    }
    const range: AceVaeOutputRangeBinding = Object.freeze({
      base: probe.base,
      count: probe.count,
      control: Object.freeze({
        buffer: shared.controls,
        offset: controlOffset,
        size: RANGE_CONTROL_BYTES,
      }),
    });
    const common = Object.freeze({
      input: binding(shared.input, plan.inputBindingBytes),
      ...(topology.outputStorage === "float16"
        ? { bias: binding(shared.bias, plan.biasBindingBytes) }
        : {}),
    });
    const fixed32 = await fixed32Kernel.createDispatch(
      `${topology.label}-${probe.id}-fixed32`,
      topology.shape,
      Object.freeze({
        ...common,
        weight: binding(weight.native, plan.weightBindingBytes),
        output: outputBinding(shared.outputs.fixed32, plan),
      }),
      topology.outputStorage,
      range,
    );
    const packedKio = await packedKioKernel.createDispatch(
      `${topology.label}-${probe.id}-packed-kio`,
      topology.shape,
      Object.freeze({
        ...common,
        packedWeight: weight.packed.binding,
        output: outputBinding(shared.outputs.packedKio, plan),
      }),
      topology.outputStorage,
      range,
    );
    assertDispatches(topology, plan, probe, fixed32, packedKio);
    fixed32Dispatches[probe.id] = fixed32;
    packedKioDispatches[probe.id] = packedKio;
  }
  const dispatches = Object.freeze({
    fixed32: Object.freeze(fixed32Dispatches),
    packedKio: Object.freeze(packedKioDispatches),
  });
  const correctness = await runOperationCorrectness(
    device,
    tracker,
    shared,
    topology,
    plan,
    dispatches,
  );
  return Object.freeze({ topology, plan, weight, dispatches, correctness });
}

async function prepareWeight(
  device: GPUDevice,
  tracker: BufferTracker,
  packedKioKernel: AceOpt0014VaeConv1dPackedKioSubgroupKernel,
  repackPrefill: GPUBuffer,
  topology: Opt0014C300Operation,
): Promise<PreparedWeight> {
  const plan = planAceOpt0014VaeConv1dPackedKioWeight(topology.shape);
  const native = tracker.create(device, {
    label: `${topology.label}-native-weight`,
    size: plan.nativeBindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const nativeWords = new Uint16Array(native.getMappedRange());
  for (let index = 0; index < plan.inputChannels * 7 * plan.outputChannels;
    index += 1) {
    nativeWords[index] = deterministicWeightBits(topology.operationIndex, index);
  }
  native.unmap();
  const packedBuffer = createUninitializedBuffer(
    device,
    tracker,
    `${topology.label}-packed-weight`,
    STORAGE_GUARD_BYTES + plan.packedBindingBytes + STORAGE_GUARD_BYTES,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  );
  const packed = Object.freeze({
    buffer: packedBuffer,
    binding: Object.freeze({
      buffer: packedBuffer,
      offset: STORAGE_GUARD_BYTES,
      size: plan.packedBindingBytes,
    }),
  });
  const repack = await packedKioKernel.createRepackDispatch(
    `${topology.label}-repack`,
    topology.shape,
    Object.freeze({
      nativeWeight: binding(native, plan.nativeBindingBytes),
      packedWeight: packed.binding,
    }),
  );
  const executions = [];
  for (let execution = 0; execution < 2; execution += 1) {
    executions.push(await executeAndVerifyRepack(
      device,
      tracker,
      repackPrefill,
      topology,
      packed,
      repack,
      execution,
    ));
  }
  if (executions[0]!["sha256"] !== executions[1]!["sha256"]) {
    throw new Error(`${topology.label} repack rerun changed`);
  }
  return Object.freeze({
    native,
    packed,
    repack,
    uniqueU16Words: plan.packedWordCount * 2,
    verification: Object.freeze({
      label: topology.label,
      inputChannels: plan.inputChannels,
      outputChannels: plan.outputChannels,
      uniqueU16WordCount: plan.packedWordCount * 2,
      comparedU16WordCount: plan.packedWordCount * 4,
      packedBytes: plan.packedBindingBytes,
      repackWorkgroups: plan.repackWorkgroups,
      sha256: executions[0]!["sha256"],
      mismatchCount: 0,
      qNaNPrefillCount: 0,
      redzonesUntouched: true,
      deterministicRerun: true,
    }),
  });
}

async function executeAndVerifyRepack(
  device: GPUDevice,
  tracker: BufferTracker,
  repackPrefill: GPUBuffer,
  topology: Opt0014C300Operation,
  packed: GuardedBinding,
  repack: AceOpt0014VaeConv1dRepackDispatch,
  execution: number,
): Promise<Readonly<Record<string, unknown>>> {
  const plan = planAceOpt0014VaeConv1dPackedKioWeight(topology.shape);
  const guardedBytes = STORAGE_GUARD_BYTES + plan.packedBindingBytes +
    STORAGE_GUARD_BYTES;
  writeU32Pattern(
    device,
    packed.buffer,
    0,
    STORAGE_GUARD_BYTES,
    STORAGE_GUARD_U32,
  );
  writeU32Pattern(
    device,
    packed.buffer,
    STORAGE_GUARD_BYTES + plan.packedBindingBytes,
    STORAGE_GUARD_BYTES,
    STORAGE_GUARD_U32,
  );
  const readback = tracker.create(device, {
    label: `${topology.label}-repack-readback-${execution}`,
    size: guardedBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder({
      label: `${topology.label}-repack-verify-${execution}`,
    });
    encoder.copyBufferToBuffer(
      repackPrefill,
      0,
      packed.buffer,
      STORAGE_GUARD_BYTES,
      plan.packedBindingBytes,
    );
    const pass = encoder.beginComputePass();
    repack.encode(pass);
    pass.end();
    encoder.copyBufferToBuffer(packed.buffer, 0, readback, 0, guardedBytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    const range = readback.getMappedRange();
    const prefix = new Uint32Array(
      range,
      0,
      STORAGE_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT,
    );
    const payload = new Uint16Array(
      range,
      STORAGE_GUARD_BYTES,
      plan.packedBindingBytes / Uint16Array.BYTES_PER_ELEMENT,
    );
    const suffix = new Uint32Array(
      range,
      STORAGE_GUARD_BYTES + plan.packedBindingBytes,
      STORAGE_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT,
    );
    if (!everyU32(prefix, STORAGE_GUARD_U32) ||
      !everyU32(suffix, STORAGE_GUARD_U32)) {
      throw new Error(`${topology.label} repack redzone changed`);
    }
    let prefillQNaNCount = 0;
    let mismatchCount = 0;
    let firstMismatchIndex: number | null = null;
    for (let packedIndex = 0; packedIndex < payload.length; packedIndex += 1) {
      const actual = payload[packedIndex]!;
      if (actual === REPACK_PREFILL_QNAN_F16) prefillQNaNCount += 1;
      const outputChannel = packedIndex % plan.outputChannels;
      const kernelInput = Math.floor(packedIndex / plan.outputChannels);
      const inputChannel = kernelInput % plan.inputChannels;
      const kernel = Math.floor(kernelInput / plan.inputChannels);
      const nativeIndex =
        (outputChannel * 7 + kernel) * plan.inputChannels + inputChannel;
      const expectedBits = deterministicWeightBits(
        topology.operationIndex,
        nativeIndex,
      );
      if (actual === expectedBits) continue;
      mismatchCount += 1;
      firstMismatchIndex ??= packedIndex;
    }
    if (prefillQNaNCount !== 0 || mismatchCount !== 0) {
      throw new Error(
        `${topology.label} repack failed: qnan=${prefillQNaNCount}, ` +
          `mismatch=${mismatchCount}@${String(firstMismatchIndex)}`,
      );
    }
    const detached = payload.slice();
    return Object.freeze({
      sha256: await sha256Hex(new Uint8Array(detached.buffer)),
      comparedU16WordCount: detached.length,
      mismatchCount,
      firstMismatchIndex,
      prefillQNaNCount,
      redzonesUntouched: true,
    });
  } finally {
    if (mapped) readback.unmap();
    tracker.destroy(readback);
  }
}

async function runOperationCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  shared: SharedResources,
  topology: Opt0014C300Operation,
  plan: AceFp16VaeConv1dPlan,
  dispatches: Readonly<
    Record<KernelArm, Readonly<Record<TimingStratum, EncodableDispatch>>>
  >,
): Promise<Readonly<Record<string, unknown>>> {
  const probeReceipts = [];
  let comparedWordCount = 0;
  for (const probe of topology.correctnessProbes) {
    const executions = [];
    for (const order of [
      ["fixed32", "packedKio"],
      ["packedKio", "fixed32"],
    ] as const) {
      const result = await executeCorrectnessPair(
        device,
        tracker,
        shared,
        topology,
        plan,
        probe,
        dispatches,
        order,
      );
      executions.push(result);
      comparedWordCount += probe.count;
    }
    const first = executions[0]!;
    const rerun = executions[1]!;
    if (first["fixed32Sha256"] !== rerun["fixed32Sha256"] ||
      first["packedKioSha256"] !== rerun["packedKioSha256"]) {
      throw new Error(`${topology.label} ${probe.id} rerun changed`);
    }
    probeReceipts.push(Object.freeze({
      stratum: probe.stratum,
      source: probe.source,
      rangeIndex: probe.rangeIndex,
      base: probe.base,
      count: probe.count,
      outputWordType: topology.outputStorage === "float16" ? "u16" : "u32",
      fixed32Sha256: first["fixed32Sha256"],
      packedKioSha256: first["packedKioSha256"],
      mismatchCount: 0,
      deterministicRerun: true,
    }));
  }
  return Object.freeze({
    label: topology.label,
    shape: topology.shape,
    outputStorage: topology.outputStorage,
    exactGraphRangeCount: topology.ranges.length,
    probeCount: topology.correctnessProbes.length,
    comparedWordCount,
    mismatchCount: 0,
    completeWrites: true,
    guardsAndAdjacentCanariesUntouched: true,
    deterministicRerun: true,
    probes: Object.freeze(probeReceipts),
  });
}

async function executeCorrectnessPair(
  device: GPUDevice,
  tracker: BufferTracker,
  shared: SharedResources,
  topology: Opt0014C300Operation,
  plan: AceFp16VaeConv1dPlan,
  probe: Opt0014ProbeRange,
  dispatches: Readonly<
    Record<KernelArm, Readonly<Record<TimingStratum, EncodableDispatch>>>
  >,
  order: readonly KernelArm[],
): Promise<Readonly<Record<string, unknown>>> {
  const elementBytes = topology.outputStorage === "float16" ? 2 : 4;
  const selectedBytes = probe.count * elementBytes;
  const selectedStart = probe.base * elementBytes;
  const beforeBytes = Math.min(STORAGE_GUARD_BYTES, selectedStart);
  const afterBytes = Math.min(
    STORAGE_GUARD_BYTES,
    plan.outputBindingBytes - selectedStart - selectedBytes,
  );
  const armBytes = STORAGE_GUARD_BYTES * 2 + selectedBytes + beforeBytes +
    afterBytes;
  const readback = tracker.create(device, {
    label: `${topology.label}-${probe.id}-output-readback`,
    size: armBytes * 2,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    for (const arm of order) {
      primeOutputProbe(
        device,
        shared.outputs[arm],
        plan,
        selectedStart,
        selectedBytes,
        beforeBytes,
        afterBytes,
      );
    }
    const encoder = device.createCommandEncoder({
      label: `${topology.label}-${probe.id}-correctness`,
    });
    for (const arm of order) {
      const output = shared.outputs[arm];
      const prefill = topology.outputStorage === "float16"
        ? shared.f16Prefill
        : shared.f32Prefill;
      encoder.copyBufferToBuffer(
        prefill,
        0,
        output,
        STORAGE_GUARD_BYTES + selectedStart,
        selectedBytes,
      );
      const pass = encoder.beginComputePass();
      dispatches[arm][probe.id].encode(pass);
      pass.end();
      const readbackBase = arm === "fixed32" ? 0 : armBytes;
      encodeProbeReadback(
        encoder,
        output,
        readback,
        readbackBase,
        plan,
        selectedStart,
        selectedBytes,
        beforeBytes,
        afterBytes,
      );
    }
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    const mappedRange = readback.getMappedRange();
    const arms = {} as Record<KernelArm, ReadbackArm>;
    for (const arm of ["fixed32", "packedKio"] as const) {
      arms[arm] = await scanOutputReadback(
        mappedRange,
        arm === "fixed32" ? 0 : armBytes,
        topology.outputStorage,
        probe.count,
        selectedBytes,
        beforeBytes,
        afterBytes,
      );
    }
    const fixedWords = arms.fixed32.words;
    const packedWords = arms.packedKio.words;
    let mismatchCount = 0;
    let firstMismatchIndex: number | null = null;
    for (let index = 0; index < fixedWords.length; index += 1) {
      if (fixedWords[index] === packedWords[index]) continue;
      mismatchCount += 1;
      firstMismatchIndex ??= index;
    }
    if (mismatchCount !== 0) {
      throw new Error(
        `${topology.label} ${probe.id} raw-word mismatch ` +
          `${mismatchCount}@${String(firstMismatchIndex)}`,
      );
    }
    return Object.freeze({
      fixed32Sha256: arms.fixed32.sha256,
      packedKioSha256: arms.packedKio.sha256,
      comparedWordCount: fixedWords.length,
      mismatchCount,
      firstMismatchIndex,
      completeWrites: true,
      guardsAndAdjacentCanariesUntouched: true,
    });
  } finally {
    if (mapped) readback.unmap();
    tracker.destroy(readback);
  }
}

function primeOutputProbe(
  device: GPUDevice,
  output: GPUBuffer,
  plan: AceFp16VaeConv1dPlan,
  selectedStart: number,
  selectedBytes: number,
  beforeBytes: number,
  afterBytes: number,
): void {
  writeU32Pattern(
    device,
    output,
    0,
    STORAGE_GUARD_BYTES,
    STORAGE_GUARD_U32,
  );
  writeU32Pattern(
    device,
    output,
    STORAGE_GUARD_BYTES + plan.outputBindingBytes,
    STORAGE_GUARD_BYTES,
    STORAGE_GUARD_U32,
  );
  if (beforeBytes > 0) {
    writeU32Pattern(
      device,
      output,
      STORAGE_GUARD_BYTES + selectedStart - beforeBytes,
      beforeBytes,
      ADJACENT_CANARY_U32,
    );
  }
  if (afterBytes > 0) {
    writeU32Pattern(
      device,
      output,
      STORAGE_GUARD_BYTES + selectedStart + selectedBytes,
      afterBytes,
      ADJACENT_CANARY_U32,
    );
  }
}

function encodeProbeReadback(
  encoder: GPUCommandEncoder,
  output: GPUBuffer,
  readback: GPUBuffer,
  base: number,
  plan: AceFp16VaeConv1dPlan,
  selectedStart: number,
  selectedBytes: number,
  beforeBytes: number,
  afterBytes: number,
): void {
  let target = base;
  encoder.copyBufferToBuffer(
    output,
    0,
    readback,
    target,
    STORAGE_GUARD_BYTES,
  );
  target += STORAGE_GUARD_BYTES;
  encoder.copyBufferToBuffer(
    output,
    STORAGE_GUARD_BYTES + plan.outputBindingBytes,
    readback,
    target,
    STORAGE_GUARD_BYTES,
  );
  target += STORAGE_GUARD_BYTES;
  encoder.copyBufferToBuffer(
    output,
    STORAGE_GUARD_BYTES + selectedStart,
    readback,
    target,
    selectedBytes,
  );
  target += selectedBytes;
  if (beforeBytes > 0) {
    encoder.copyBufferToBuffer(
      output,
      STORAGE_GUARD_BYTES + selectedStart - beforeBytes,
      readback,
      target,
      beforeBytes,
    );
    target += beforeBytes;
  }
  if (afterBytes > 0) {
    encoder.copyBufferToBuffer(
      output,
      STORAGE_GUARD_BYTES + selectedStart + selectedBytes,
      readback,
      target,
      afterBytes,
    );
  }
}

async function scanOutputReadback(
  mappedRange: ArrayBuffer,
  base: number,
  outputStorage: AceFp16VaeConv1dOutputStorage,
  wordCount: number,
  selectedBytes: number,
  beforeBytes: number,
  afterBytes: number,
): Promise<ReadbackArm> {
  const prefix = new Uint32Array(
    mappedRange,
    base,
    STORAGE_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT,
  );
  const suffix = new Uint32Array(
    mappedRange,
    base + STORAGE_GUARD_BYTES,
    STORAGE_GUARD_BYTES / Uint32Array.BYTES_PER_ELEMENT,
  );
  const selectedOffset = base + STORAGE_GUARD_BYTES * 2;
  const words = outputStorage === "float16"
    ? new Uint16Array(mappedRange, selectedOffset, wordCount).slice()
    : new Uint32Array(mappedRange, selectedOffset, wordCount).slice();
  let adjacentOffset = selectedOffset + selectedBytes;
  const before = new Uint32Array(
    mappedRange,
    adjacentOffset,
    beforeBytes / Uint32Array.BYTES_PER_ELEMENT,
  );
  adjacentOffset += beforeBytes;
  const after = new Uint32Array(
    mappedRange,
    adjacentOffset,
    afterBytes / Uint32Array.BYTES_PER_ELEMENT,
  );
  let nonFiniteCount = 0;
  let prefillQNaNCount = 0;
  for (const word of words) {
    if (outputStorage === "float16") {
      if ((word & 0x7c00) === 0x7c00) nonFiniteCount += 1;
      if (word === OUTPUT_PREFILL_QNAN_F16) prefillQNaNCount += 1;
    } else {
      if ((word & 0x7f80_0000) === 0x7f80_0000) nonFiniteCount += 1;
      if (word === OUTPUT_PREFILL_QNAN_F32) prefillQNaNCount += 1;
    }
  }
  const guardsAndCanariesUntouched =
    everyU32(prefix, STORAGE_GUARD_U32) &&
    everyU32(suffix, STORAGE_GUARD_U32) &&
    everyU32(before, ADJACENT_CANARY_U32) &&
    everyU32(after, ADJACENT_CANARY_U32);
  if (nonFiniteCount !== 0 || prefillQNaNCount !== 0 ||
    !guardsAndCanariesUntouched) {
    throw new Error(
      `OPT-0014 output write/guard scan failed: finite=${nonFiniteCount}, ` +
        `qnan=${prefillQNaNCount}, guards=${guardsAndCanariesUntouched}`,
    );
  }
  return Object.freeze({
    sha256: await sha256Hex(new Uint8Array(
      words.buffer,
      words.byteOffset,
      words.byteLength,
    )),
    nonFiniteCount,
    prefillQNaNCount,
    guardsAndCanariesUntouched,
    words,
  });
}

async function runTimedGate(
  prepared: PreparedGate,
): Promise<Readonly<Record<string, unknown>>> {
  const launchedAtEpochMilliseconds = Date.now();
  const thermal = parseOpt0014ThermalGate(
    collectThermalParameters(),
    prepared.preparedCompletedAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const strata: Opt0014WeightedStratumInput[] = [];
  const timingStartedAtEpochMilliseconds = Date.now();
  for (const operation of prepared.operations) {
    for (const stratum of operation.topology.timingStrata) {
      const samples: Record<KernelArm, number[]> = {
        fixed32: [],
        packedKio: [],
      };
      for (const order of [
        ["fixed32", "packedKio"],
        ["packedKio", "fixed32"],
      ] as const) {
        for (const arm of order) {
          samples[arm].push(await executeTimedDispatch(
            prepared.device,
            operation.dispatches[arm][stratum.id],
          ));
        }
      }
      strata.push(Object.freeze({
        operationLabel: operation.topology.label,
        stratum: stratum.stratum,
        weight: stratum.weight!,
        fixed32Samples: Object.freeze(samples.fixed32),
        packedKioSamples: Object.freeze(samples.packedKio),
      }));
    }
    await yieldToBrowser();
  }
  const timingCompletedAtEpochMilliseconds = Date.now();
  const timing = summarizeOpt0014WeightedTiming(strata);
  const repackTimingStartedAtEpochMilliseconds = Date.now();
  const repackTotalMilliseconds = await executeTimedRepackBatch(
    prepared.device,
    prepared.operations.map((operation) => operation.weight.repack),
  );
  const repackTimingCompletedAtEpochMilliseconds = Date.now();
  const capability = capabilityReceipt(prepared.adapter, prepared.device);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanup = prepared.destroy();
  return Object.freeze({
    schema: "ace-opt-0014-vae-fixed32-packed-kio-k7-ab-v1",
    status: "passed",
    experimentId: EXPERIMENT_ID,
    classification:
      "primitive-weighted-decision-gate-not-integrated-decoder-wall",
    recordedAt: new Date().toISOString(),
    identity: Object.freeze({ sourceAuthority: prepared.sourceAuthority }),
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      page: window.location.href,
      ...capability,
    }),
    protocol: Object.freeze({
      thermal,
      correctnessAndRepackVerificationCompletedBeforeThermalGate: true,
      allPipelinesCompiledAndWarmedBeforeTiming: true,
      correctnessArmOrders: Object.freeze([
        "fixed32-packedKio",
        "packedKio-fixed32",
      ]),
      timingArmOrdersPerStratum: Object.freeze([
        "fixed32-packedKio",
        "packedKio-fixed32",
      ]),
      samplesPerArmPerStratum: 2,
      authoritativeTiming: "performance.now-submit-through-queue-drain",
      compileAllocationUploadReadbackCleanupAndRepackExcludedFromConvTiming:
        true,
      exactC300GraphRangeWeights: true,
      unchangedThermalRetryPerformed: false,
      integratedDecoderWallClaim: null,
      qualityClaim: null,
      listeningClaim: null,
      productionSelectorClaim: null,
    }),
    topology: Object.freeze({
      decoderInputFrames: C300_INPUT_FRAMES,
      operationCount: prepared.topology.length,
      exactRangeCount: C300_EXACT_RANGE_COUNT,
      correctnessProbeCount: CORRECTNESS_PROBE_COUNT,
      timingStratumCount: TIMING_STRATUM_COUNT,
      operations: Object.freeze(prepared.topology.map((operation) =>
        Object.freeze({
          label: operation.label,
          shape: operation.shape,
          outputStorage: operation.outputStorage,
          outputElements: operation.outputElements,
          exactRangeCount: operation.ranges.length,
          correctnessProbes: operation.correctnessProbes,
          timingStrata: operation.timingStrata,
        })
      )),
    }),
    correctness: prepared.correctness,
    convTiming: Object.freeze({
      ...timing,
      timingStartedAtEpochMilliseconds,
      timingCompletedAtEpochMilliseconds,
      caveat:
        "The 50 separately drained representatives are weighted by all 2,404 exact C300 graph ranges; this is a primitive decision projection, not an integrated scheduler or decoder wall.",
    }),
    repack: Object.freeze({
      correctness: prepared.repackCorrectness,
      timing: Object.freeze({
        measuredAfterAllConvTiming: true,
        includedInConvProjection: false,
        commandBufferCount: 1,
        computePassCount: 1,
        submitAndDrainCount: 1,
        dispatchCount: C300_OPERATION_COUNT,
        workgroupCount: 59_588,
        sampleCount: 1,
        totalMilliseconds: repackTotalMilliseconds,
        timingStartedAtEpochMilliseconds: repackTimingStartedAtEpochMilliseconds,
        timingCompletedAtEpochMilliseconds:
          repackTimingCompletedAtEpochMilliseconds,
      }),
      memory: Object.freeze({
        uniquePackedPayloadBytes: EXPECTED_PACKED_WEIGHT_BYTES,
        guardedPackedAllocationBytes:
          EXPECTED_PACKED_WEIGHT_BYTES +
          C300_OPERATION_COUNT * STORAGE_GUARD_BYTES * 2,
        verifierPrefillBytes: Math.max(...prepared.operations.map((operation) =>
          Number(operation.weight.verification["packedBytes"])
        )),
        lifetime:
          "packed weights remain resident for the bounded gate; native package weights remain the fixed32 authority",
      }),
    }),
    memory: memoryBeforeCleanup,
    cleanup,
  });
}

async function executeTimedDispatch(
  device: GPUDevice,
  dispatch: EncodableDispatch,
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

async function executeTimedRepackBatch(
  device: GPUDevice,
  dispatches: readonly AceOpt0014VaeConv1dRepackDispatch[],
): Promise<number> {
  if (dispatches.length !== C300_OPERATION_COUNT) {
    throw new Error("OPT-0014 repack timing dispatch count changed");
  }
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (const dispatch of dispatches) dispatch.encode(pass);
  pass.end();
  const command = encoder.finish();
  const started = performance.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - started;
}

async function buildSourceAuthority(
  topology: readonly Opt0014C300Operation[],
): Promise<Readonly<Record<string, unknown>>> {
  const fixed32SourceSha256 = await sha256Text(fixed32CoreSource);
  const packedKioSourceSha256 = await sha256Text(packedKioCoreSource);
  if (fixed32SourceSha256 !== FIXED32_CORE_SOURCE_SHA256 ||
    packedKioSourceSha256 !== PACKED_KIO_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0014 rejected unauthenticated kernel core source");
  }
  const fixed32Shaders: string[] = [];
  const packedKioShaders: string[] = [];
  const repackShaders: string[] = [];
  for (const operation of topology) {
    const hasBias = operation.outputStorage === "float16";
    fixed32Shaders.push(
      `${operation.label}\n${aceFp16VaeConv1dSubgroupWgsl(
        operation.shape,
        hasBias,
        operation.outputStorage,
      )}`,
    );
    packedKioShaders.push(
      `${operation.label}\n${aceOpt0014VaeConv1dPackedKioWgsl(
        operation.shape,
        hasBias,
        operation.outputStorage,
      )}`,
    );
    repackShaders.push(
      `${operation.label}\n${aceOpt0014VaeConv1dPackedKioRepackWgsl(
        operation.shape,
      )}`,
    );
  }
  return Object.freeze({
    coreCommit: CORE_COMMIT,
    fixed32CoreSourceSha256: fixed32SourceSha256,
    packedKioCoreSourceSha256: packedKioSourceSha256,
    fixed32KernelId: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
    packedKioKernelId: ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_KERNEL_ID,
    repackKernelId: ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_REPACK_KERNEL_ID,
    generatedShaderCountPerArm: topology.length,
    fixed32GeneratedAggregateSha256: await sha256Text(
      fixed32Shaders.join("\n\u0000\n"),
    ),
    packedKioGeneratedAggregateSha256: await sha256Text(
      packedKioShaders.join("\n\u0000\n"),
    ),
    repackGeneratedAggregateSha256: await sha256Text(
      repackShaders.join("\n\u0000\n"),
    ),
  });
}

function expected(
  label: string,
  inputFrames: number,
  inputChannels: number,
  outputChannels: number,
  dilation: number,
  padding: number,
  rangeCount: number,
  firstCount: number,
  interiorRangeIndex: number | null,
  interiorBase: number,
  interiorCount: number,
  interiorWeight: number | null,
  tailBase: number,
  tailCount: number,
) {
  return Object.freeze({
    label,
    inputFrames,
    inputChannels,
    outputChannels,
    dilation,
    padding,
    outputStorage: label === "conv2" ? "float32" : "float16",
    rangeCount,
    firstCount,
    interiorRangeIndex,
    interiorBase,
    interiorCount,
    interiorWeight,
    tailBase,
    tailCount,
  });
}

function probeFromGraph(
  stratum: TimingStratum,
  range: ExactRange,
  weight: number,
): Opt0014ProbeRange {
  return Object.freeze({
    id: stratum,
    stratum,
    source: "graph",
    rangeIndex: range.rangeIndex,
    base: range.base,
    count: range.count,
    firstOutputRow: range.firstOutputRow,
    outputRowCount: range.outputRowCount,
    weight,
  });
}

function centeredConv1Probe(
  plan: AceFp16VaeConv1dPlan,
): Opt0014ProbeRange {
  const outputRowCount = 16;
  const firstOutputRow = Math.floor((plan.outputFrames - outputRowCount) / 2);
  const probe = Object.freeze({
    id: "interior" as const,
    stratum: "interior" as const,
    source: "synthetic-centered" as const,
    rangeIndex: null,
    base: firstOutputRow * plan.outputChannels,
    count: outputRowCount * plan.outputChannels,
    firstOutputRow,
    outputRowCount,
    weight: null,
  });
  if (plan.outputFrames !== 300 || plan.outputChannels !== 2_048 ||
    probe.base !== 290_816 || probe.count !== 32_768) {
    throw new Error("OPT-0014 conv1 centered correctness slice changed");
  }
  planAceFp16VaeConv1dSubgroupRange(plan, probe);
  planAceOpt0014VaeConv1dPackedKioRange(plan, probe);
  return probe;
}

function assertExpectedTopology(
  operation: Opt0014C300Operation,
  ordinal: number,
): void {
  const expectedValue = OPT_0014_C300_EXPECTED_TOPOLOGY[ordinal];
  const [first, interior, tail] = operation.correctnessProbes;
  if (expectedValue === undefined || first === undefined ||
    interior === undefined || tail === undefined ||
    operation.label !== expectedValue.label ||
    operation.shape.batch !== 1 || operation.shape.stride !== 1 ||
    operation.shape.kernelSize !== 7 ||
    operation.shape.inputFrames !== expectedValue.inputFrames ||
    operation.shape.inputChannels !== expectedValue.inputChannels ||
    operation.shape.outputChannels !== expectedValue.outputChannels ||
    operation.shape.dilation !== expectedValue.dilation ||
    operation.shape.padding !== expectedValue.padding ||
    operation.outputStorage !== expectedValue.outputStorage ||
    operation.ranges.length !== expectedValue.rangeCount ||
    first.base !== 0 || first.count !== expectedValue.firstCount ||
    interior.rangeIndex !== expectedValue.interiorRangeIndex ||
    interior.base !== expectedValue.interiorBase ||
    interior.count !== expectedValue.interiorCount ||
    interior.weight !== expectedValue.interiorWeight ||
    tail.base !== expectedValue.tailBase ||
    tail.count !== expectedValue.tailCount) {
    throw new Error(`${operation.label} rejected unexpected C300 geometry`);
  }
}

function createPeriodicU16Buffer(
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
  try {
    const destination = new Uint16Array(buffer.getMappedRange());
    fillPeriodic(destination, pattern);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function createRepeatedU32Buffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  bytes: number,
  value: number,
): GPUBuffer {
  if (bytes < 4 || bytes % 4 !== 0) {
    throw new RangeError(`${label} must be a non-empty U32 buffer`);
  }
  const buffer = tracker.create(device, {
    label,
    size: bytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).fill(value);
  buffer.unmap();
  return buffer;
}

function createUninitializedBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  return tracker.create(device, { label, size, usage });
}

function fillPeriodic(
  destination: Uint16Array,
  pattern: Uint16Array,
): void {
  if (destination.length < 1 || pattern.length < 1) {
    throw new RangeError("OPT-0014 periodic upload geometry changed");
  }
  const initial = Math.min(pattern.length, destination.length);
  destination.set(pattern.subarray(0, initial));
  let filled = initial;
  while (filled < destination.length) {
    const count = Math.min(filled, destination.length - filled);
    destination.copyWithin(filled, 0, count);
    filled += count;
  }
}

function createRangeControls(
  device: GPUDevice,
  tracker: BufferTracker,
  topology: readonly Opt0014C300Operation[],
): Readonly<{ buffer: GPUBuffer; offsets: ReadonlyMap<string, number> }> {
  const alignment = Number(device.limits.minUniformBufferOffsetAlignment);
  if (!Number.isSafeInteger(alignment) || alignment < RANGE_CONTROL_BYTES ||
    !Number.isInteger(Math.log2(alignment))) {
    throw new RangeError("OPT-0014 uniform alignment is invalid");
  }
  const probes = topology.flatMap((operation) =>
    operation.correctnessProbes.map((probe) => ({ operation, probe }))
  );
  if (probes.length !== CORRECTNESS_PROBE_COUNT) {
    throw new Error("OPT-0014 range-control probe count changed");
  }
  const buffer = tracker.create(device, {
    label: "opt-0014-range-controls",
    size: probes.length * alignment,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  const words = new Uint32Array(buffer.getMappedRange());
  const offsets = new Map<string, number>();
  for (const [index, { operation, probe }] of probes.entries()) {
    const byteOffset = index * alignment;
    const wordOffset = byteOffset / Uint32Array.BYTES_PER_ELEMENT;
    words[wordOffset] = probe.base;
    words[wordOffset + 1] = probe.count;
    offsets.set(probeKey(operation.label, probe.id), byteOffset);
  }
  buffer.unmap();
  return Object.freeze({ buffer, offsets });
}

function assertDispatches(
  operation: Opt0014C300Operation,
  plan: AceFp16VaeConv1dPlan,
  probe: Opt0014ProbeRange,
  fixed32: AceFp16VaeConv1dSubgroupDispatch,
  packedKio: AceOpt0014VaeConv1dPackedKioDispatch,
): void {
  const expectedFixed32 = planAceFp16VaeConv1dSubgroupRange(plan, probe);
  const expectedPackedKio = planAceOpt0014VaeConv1dPackedKioRange(plan, probe);
  if (fixed32.kernelId !== ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID ||
    packedKio.kernelId !== ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_KERNEL_ID ||
    !sameDispatchRange(fixed32.outputRange, expectedFixed32) ||
    !sameDispatchRange(packedKio.outputRange, expectedPackedKio)) {
    throw new Error(`${operation.label} ${probe.id} dispatch changed`);
  }
}

function sameDispatchRange(
  left: Readonly<{
    base: number;
    count: number;
    workgroupsX: number;
    workgroupsY: number;
  }>,
  right: Readonly<{
    base: number;
    count: number;
    workgroupsX: number;
    workgroupsY: number;
  }>,
): boolean {
  return left.base === right.base && left.count === right.count &&
    left.workgroupsX === right.workgroupsX &&
    left.workgroupsY === right.workgroupsY;
}

function requiredDeviceLimits(
  adapter: GPUAdapter,
  topology: readonly Opt0014C300Operation[],
): Record<string, number> {
  let maximumBuffer = 4;
  let maximumStorageBinding = 4;
  let maximumDispatch = 1;
  let maximumProbeF16Bytes = 4;
  let maximumProbeF32Bytes = 4;
  for (const operation of topology) {
    const plan = planAceFp16VaeConv1d(
      operation.shape,
      operation.outputStorage,
    );
    const packed = planAceOpt0014VaeConv1dPackedKioWeight(operation.shape);
    maximumStorageBinding = Math.max(
      maximumStorageBinding,
      plan.inputBindingBytes,
      plan.weightBindingBytes,
      plan.biasBindingBytes,
      plan.outputBindingBytes,
      packed.packedBindingBytes,
    );
    maximumBuffer = Math.max(
      maximumBuffer,
      plan.inputBindingBytes,
      plan.weightBindingBytes,
      plan.biasBindingBytes,
      STORAGE_GUARD_BYTES + plan.outputBindingBytes + STORAGE_GUARD_BYTES,
      STORAGE_GUARD_BYTES + packed.packedBindingBytes + STORAGE_GUARD_BYTES,
    );
    maximumDispatch = Math.max(maximumDispatch, packed.repackWorkgroups);
    for (const probe of operation.correctnessProbes) {
      const fixed32 = planAceFp16VaeConv1dSubgroupRange(plan, probe);
      const packedKio = planAceOpt0014VaeConv1dPackedKioRange(plan, probe);
      maximumDispatch = Math.max(
        maximumDispatch,
        fixed32.workgroupsX,
        fixed32.workgroupsY,
        packedKio.workgroupsX,
        packedKio.workgroupsY,
      );
      if (operation.outputStorage === "float16") {
        maximumProbeF16Bytes = Math.max(maximumProbeF16Bytes, probe.count * 2);
      } else {
        maximumProbeF32Bytes = Math.max(maximumProbeF32Bytes, probe.count * 4);
      }
    }
  }
  maximumBuffer = Math.max(
    maximumBuffer,
    maximumProbeF16Bytes,
    maximumProbeF32Bytes,
  );
  const requested = {
    maxBufferSize: maximumBuffer,
    maxStorageBufferBindingSize: maximumStorageBinding,
    maxUniformBufferBindingSize: RANGE_CONTROL_BYTES,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupsPerDimension: maximumDispatch,
  };
  for (const [name, minimum] of Object.entries(requested)) {
    const actual = Number(adapter.limits[name as keyof GPUSupportedLimits]);
    if (!Number.isFinite(actual) || actual < minimum) {
      throw new RangeError(
        `OPT-0014 adapter ${name}=${actual} is below ${minimum}`,
      );
    }
  }
  return requested;
}

function requireAdapter(
  adapter: GPUAdapter,
  topology: readonly Opt0014C300Operation[],
): void {
  if (!adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups")) {
    throw new Error("OPT-0014 requires adapter shader-f16 and subgroups");
  }
  if (adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32) {
    throw new Error("OPT-0014 requires authenticated fixed 32-lane subgroups");
  }
  if (adapter.limits.minStorageBufferOffsetAlignment > STORAGE_GUARD_BYTES) {
    throw new Error("OPT-0014 storage redzone is below adapter alignment");
  }
  requiredDeviceLimits(adapter, topology);
}

function capabilityReceipt(adapter: GPUAdapter, device: GPUDevice) {
  return Object.freeze({
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
    deviceFeatures: Object.freeze([...device.features].sort()),
    deviceLimits: Object.freeze({
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupsPerDimension:
        device.limits.maxComputeWorkgroupsPerDimension,
      minStorageBufferOffsetAlignment:
        device.limits.minStorageBufferOffsetAlignment,
      minUniformBufferOffsetAlignment:
        device.limits.minUniformBufferOffsetAlignment,
    }),
  });
}

function deterministicWeightBits(
  operationIndex: number,
  nativeIndex: number,
): number {
  const special = nativeIndex & 0x0fff;
  if (special === 0) return 0x0000;
  if (special === 1) return 0x8000;
  if (special === 2) return 0x0001;
  if (special === 3) return 0x8001;
  const mixed = mix32(
    nativeIndex ^ Math.imul(operationIndex + 1, 0x9e37_79b9),
  );
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

function writeU32Pattern(
  device: GPUDevice,
  buffer: GPUBuffer,
  offset: number,
  bytes: number,
  value: number,
): void {
  if (offset % 4 !== 0 || bytes % 4 !== 0) {
    throw new RangeError("OPT-0014 U32 pattern write is unaligned");
  }
  const words = new Uint32Array(bytes / Uint32Array.BYTES_PER_ELEMENT);
  words.fill(value);
  device.queue.writeBuffer(buffer, offset, words);
}

function everyU32(words: Uint32Array, expectedValue: number): boolean {
  for (const value of words) if (value !== expectedValue) return false;
  return true;
}

function outputBinding(
  buffer: GPUBuffer,
  plan: AceFp16VaeConv1dPlan,
): GPUBufferBinding {
  return Object.freeze({
    buffer,
    offset: STORAGE_GUARD_BYTES,
    size: plan.outputBindingBytes,
  });
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function probeKey(label: string, id: TimingStratum): string {
  return `${label}:${id}`;
}

function collectThermalParameters(): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    "#thermal-gate input[name]",
  )) parameters.set(input.name, input.value);
  return parameters;
}

function median(samples: readonly number[]): number {
  if (samples.length !== 2 || samples.some((sample) =>
    !Number.isFinite(sample) || sample <= 0
  )) {
    throw new RangeError(
      "OPT-0014 timing requires two finite positive samples per arm",
    );
  }
  return (samples[0]! + samples[1]!) / 2;
}

async function sha256Text(value: string): Promise<string> {
  return sha256Hex(new TextEncoder().encode(value));
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", Uint8Array.from(bytes));
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.length === 0) {
    throw new Error(`OPT-0014 thermal field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0014 thermal field ${name} is not finite`);
  }
  return value;
}

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, 0));
}

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0014-vae-fixed32-packed-kio-k7-ab-v1",
    status: "failed",
    experimentId: EXPERIMENT_ID,
    error: Object.freeze({
      name: error instanceof Error ? error.name : "Error",
      message: error instanceof Error ? error.message : String(error),
      ...(error instanceof Error && error.stack !== undefined
        ? { stack: error.stack }
        : {}),
    }),
  });
}

function finishPage(
  status: "passed" | "failed",
  result: Readonly<Record<string, unknown>>,
): void {
  document.body.dataset.status = status;
  requireElement<HTMLElement>("#progress").textContent = status;
  requireElement<HTMLElement>("#result").textContent = JSON.stringify(
    result,
    null,
    2,
  );
}

function requireElement<ElementType extends Element>(
  selector: string,
): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing OPT-0014 element ${selector}`);
  return element;
}
