/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import kernelCoreSource from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.ts?raw";
import {
  ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
  AceFp16VaeConvTranspose1dKernel,
  aceFp16VaeCongruentConvTranspose1dWgsl,
  aceFp16VaeConvTranspose1dWgsl,
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dCongruentRange,
  planAceFp16VaeConvTranspose1dRange,
  type AceFp16VaeConvTranspose1dDispatch,
  type AceFp16VaeConvTranspose1dPlan,
} from "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import type { AceVaeConvTranspose1dShape } from
  "../../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderConvTransposeOperation,
} from "../../src/webgpu/vae-decoder.js";

type KernelArm = "portable" | "congruent";
type TimingStratum = "first" | "interior" | "tail";

interface ExactRange {
  readonly rangeIndex: number;
  readonly base: number;
  readonly count: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
}

export interface Opt0015SelectedRange extends ExactRange {
  readonly stratum: TimingStratum;
  readonly weight: number;
}

export interface Opt0015C300Operation {
  readonly operationIndex: number;
  readonly label: string;
  readonly shape: AceVaeConvTranspose1dShape;
  readonly outputFrames: number;
  readonly outputElements: number;
  readonly ranges: readonly ExactRange[];
  readonly selectedRanges: readonly Opt0015SelectedRange[];
}

export interface Opt0015ThermalGate {
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

export interface Opt0015WeightedStratumInput {
  readonly operationLabel: string;
  readonly stratum: TimingStratum;
  readonly weight: number;
  readonly portableSamples: readonly number[];
  readonly congruentSamples: readonly number[];
}

interface EncodableDispatch {
  readonly kernelId: string;
  readonly outputRange: Readonly<{
    readonly base: number;
    readonly count: number;
    readonly workgroupsX: number;
    readonly workgroupsY: number;
    readonly workgroupsZ?: number;
  }>;
  encode(pass: GPUComputePassEncoder): void;
}

interface GuardedOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
}

interface PreparedOperation {
  readonly topology: Opt0015C300Operation;
  readonly plan: AceFp16VaeConvTranspose1dPlan;
  readonly outputs: Readonly<Record<KernelArm, GuardedOutput>>;
  readonly prefill: GPUBuffer;
  readonly dispatches: Readonly<Record<KernelArm, readonly EncodableDispatch[]>>;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly topology: readonly Opt0015C300Operation[];
  readonly operations: readonly PreparedOperation[];
  readonly tracker: BufferTracker;
  readonly portableKernel: AceFp16VaeConvTranspose1dKernel;
  readonly congruentKernel: AceFp16VaeConvTranspose1dKernel<
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
  >;
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly sourceAuthority: Readonly<Record<string, unknown>>;
  readonly preparedCompletedAtEpochMilliseconds: number;
  destroy(): Readonly<Record<string, unknown>>;
}

interface ReadbackArm {
  readonly sha256: string;
  readonly nonFiniteCount: number;
  readonly prefillQNaNCount: number;
  readonly guardsAndCanariesUntouched: boolean;
  readonly bits: Uint16Array;
}

const EXPERIMENT_ID = "OPT-0015" as const;
const BASELINE_COMMIT =
  "7d4916da0cd480fe03cd5712048cb3f3f4c06310" as const;
const CORE_COMMIT =
  "075ecc0b34b7541cffc0a83412c17ee31bbadab6" as const;
const CORE_SOURCE_SHA256 =
  "cbcb9bcd5f856ce1c9e10aabca0ec0f95651c03d2c45b8076de3ba5022c6c3e2" as const;
const C300_INPUT_FRAMES = 300;
const FLOAT16_BYTES = 2;
const RANGE_CONTROL_BYTES = 16;
const OUTPUT_GUARD_BYTES = 256;
const OUTPUT_GUARD_F16 = 0x7e33;
const OUTPUT_CANARY_F16 = 0x7e11;
const OUTPUT_PREFILL_QNAN_F16 = 0x7e55;
const SOURCE_PADDING_F16 = 0x7e77;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const THERMAL_POLL_MILLISECONDS = 1_000;
const MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS = 1_250;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 5_000;

const INPUT_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x0001, 0x8001,
  0x1000, 0x9000, 0x2000, 0xa000,
  0x2800, 0xa800, 0x3000, 0xb000,
  0x3400, 0xb400, 0x3800, 0xb800,
]);
const WEIGHT_PATTERN = new Uint16Array([
  0x3c00, 0xbc00, 0x3800, 0xb800,
  0x3400, 0xb400, 0x3000, 0xb000,
  0x2c00, 0xac00, 0x2400, 0xa400,
  0x0000, 0x8000,
]);
const BIAS_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x1000, 0x9000,
  0x2000, 0xa000, 0x2800, 0xa800,
]);

export const OPT_0015_C300_EXPECTED_TOPOLOGY = Object.freeze([
  Object.freeze({
    label: "block-0-conv-t1",
    shape: Object.freeze({
      batch: 1,
      inputFrames: 300,
      inputChannels: 2_048,
      outputChannels: 1_024,
      kernelSize: 20,
      stride: 10,
      dilation: 1,
      padding: 5,
      outputPadding: 0,
    }),
    outputFrames: 3_000,
    rangeCount: 54,
    fullRangeRows: 56,
    tailRangeRows: 32,
  }),
  Object.freeze({
    label: "block-1-conv-t1",
    shape: Object.freeze({
      batch: 1,
      inputFrames: 3_000,
      inputChannels: 1_024,
      outputChannels: 512,
      kernelSize: 12,
      stride: 6,
      dilation: 1,
      padding: 3,
      outputPadding: 0,
    }),
    outputFrames: 18_000,
    rangeCount: 81,
    fullRangeRows: 224,
    tailRangeRows: 80,
  }),
  Object.freeze({
    label: "block-2-conv-t1",
    shape: Object.freeze({
      batch: 1,
      inputFrames: 18_000,
      inputChannels: 512,
      outputChannels: 256,
      kernelSize: 8,
      stride: 4,
      dilation: 1,
      padding: 2,
      outputPadding: 0,
    }),
    outputFrames: 72_000,
    rangeCount: 81,
    fullRangeRows: 896,
    tailRangeRows: 320,
  }),
  Object.freeze({
    label: "block-3-conv-t1",
    shape: Object.freeze({
      batch: 1,
      inputFrames: 72_000,
      inputChannels: 256,
      outputChannels: 128,
      kernelSize: 8,
      stride: 4,
      dilation: 1,
      padding: 2,
      outputPadding: 0,
    }),
    outputFrames: 288_000,
    rangeCount: 81,
    fullRangeRows: 3_584,
    tailRangeRows: 1_280,
  }),
  Object.freeze({
    label: "block-4-conv-t1",
    shape: Object.freeze({
      batch: 1,
      inputFrames: 288_000,
      inputChannels: 128,
      outputChannels: 128,
      kernelSize: 4,
      stride: 2,
      dilation: 1,
      padding: 1,
      outputPadding: 0,
    }),
    outputFrames: 576_000,
    rangeCount: 81,
    fullRangeRows: 7_168,
    tailRangeRows: 2_560,
  }),
] satisfies readonly Readonly<{
  readonly label: string;
  readonly shape: AceVaeConvTranspose1dShape;
  readonly outputFrames: number;
  readonly rangeCount: number;
  readonly fullRangeRows: number;
  readonly tailRangeRows: number;
}>[]);

export function buildOpt0015C300Topology():
  readonly Opt0015C300Operation[] {
  const graph = planAceVaeDecoder(C300_INPUT_FRAMES);
  const cooperative = planAceVaeDecoderQuanta(graph);
  const operations = graph.operations
    .map((operation, operationIndex) => ({ operation, operationIndex }))
    .filter((entry): entry is Readonly<{
      operation: AceVaeDecoderConvTransposeOperation;
      operationIndex: number;
    }> => entry.operation.kind === "conv-transpose1d")
    .map(({ operation, operationIndex }, ordinal) => {
      const plan = planAceFp16VaeConvTranspose1d(operation.shape);
      const ranges = cooperative.quanta
        .filter((quantum) => quantum.operationIndex === operationIndex)
        .map((quantum, rangeIndex): ExactRange => {
          if (
            quantum.operationKind !== "conv-transpose1d" ||
            quantum.primitives.length !== 1
          ) {
            throw new Error(`${operation.label} C300 quantum topology changed`);
          }
          const primitive = quantum.primitives[0]!;
          if (
            primitive.physicalPartIndex !== 0 ||
            primitive.firstOutputChannel !== 0 ||
            primitive.outputChannels !== operation.shape.outputChannels ||
            primitive.outputBase !== quantum.logicalOutputBase ||
            primitive.outputCount !== quantum.logicalOutputCount ||
            primitive.outputBase % operation.shape.outputChannels !== 0 ||
            primitive.outputCount % operation.shape.outputChannels !== 0
          ) {
            throw new Error(`${operation.label} C300 primitive topology changed`);
          }
          planAceFp16VaeConvTranspose1dRange(plan, {
            base: primitive.outputBase,
            count: primitive.outputCount,
          });
          return Object.freeze({
            rangeIndex,
            base: primitive.outputBase,
            count: primitive.outputCount,
            firstOutputRow:
              primitive.outputBase / operation.shape.outputChannels,
            outputRowCount:
              primitive.outputCount / operation.shape.outputChannels,
          });
        });
      if (ranges.length < 3) {
        throw new Error(`${operation.label} needs first/interior/tail ranges`);
      }
      const expected = OPT_0015_C300_EXPECTED_TOPOLOGY[ordinal];
      if (
        expected === undefined || operation.label !== expected.label ||
        !sameShape(operation.shape, expected.shape) ||
        plan.outputFrames !== expected.outputFrames ||
        ranges.length !== expected.rangeCount ||
        ranges[0]!.outputRowCount !== expected.fullRangeRows ||
        ranges.at(-1)!.outputRowCount !== expected.tailRangeRows
      ) {
        throw new Error(`${operation.label} rejected unexpected C300 geometry`);
      }
      const middleIndex = Math.floor(ranges.length / 2);
      const selectedRanges = Object.freeze([
        Object.freeze({ ...ranges[0]!, stratum: "first" as const, weight: 1 }),
        Object.freeze({
          ...ranges[middleIndex]!,
          stratum: "interior" as const,
          weight: ranges.length - 2,
        }),
        Object.freeze({
          ...ranges.at(-1)!,
          stratum: "tail" as const,
          weight: 1,
        }),
      ]);
      return Object.freeze({
        operationIndex,
        label: operation.label,
        shape: operation.shape,
        outputFrames: plan.outputFrames,
        outputElements: plan.outputElements,
        ranges: Object.freeze(ranges),
        selectedRanges,
      });
    });
  if (
    operations.length !== OPT_0015_C300_EXPECTED_TOPOLOGY.length ||
    operations.reduce((sum, operation) => sum + operation.ranges.length, 0) !==
      378
  ) {
    throw new Error("OPT-0015 exact C300 transpose topology changed");
  }
  return Object.freeze(operations);
}

export function parseOpt0015ThermalGate(
  parameters: URLSearchParams,
  preparedCompletedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0015ThermalGate {
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
    throw new Error("OPT-0015 requires the accepted notifyutil thermal source");
  }
  if (
    startedAtEpochMilliseconds < preparedCompletedAtEpochMilliseconds ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    completedAtEpochMilliseconds > launchedAtEpochMilliseconds ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS
  ) {
    throw new Error("OPT-0015 thermal interval is stale, short, or predates preparation");
  }
  if (
    !Number.isSafeInteger(observationCount) ||
    observationCount < Math.floor(durationMilliseconds / pollMilliseconds) + 1 ||
    pollMilliseconds !== THERMAL_POLL_MILLISECONDS ||
    maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS ||
    nonNominalObservationCount !== 0
  ) {
    throw new Error("OPT-0015 thermal observations are incomplete or non-nominal");
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

export function summarizeOpt0015WeightedTiming(
  strata: readonly Opt0015WeightedStratumInput[],
): Readonly<Record<string, unknown>> {
  if (strata.length === 0) throw new RangeError("OPT-0015 timing strata missing");
  let exactRangeCount = 0;
  let portableProjectedMilliseconds = 0;
  let congruentProjectedMilliseconds = 0;
  const summaries = strata.map((stratum) => {
    if (!Number.isSafeInteger(stratum.weight) || stratum.weight < 1) {
      throw new RangeError("OPT-0015 timing weight must be positive");
    }
    const portableMedian = median(stratum.portableSamples);
    const congruentMedian = median(stratum.congruentSamples);
    exactRangeCount += stratum.weight;
    portableProjectedMilliseconds += portableMedian * stratum.weight;
    congruentProjectedMilliseconds += congruentMedian * stratum.weight;
    return Object.freeze({
      operationLabel: stratum.operationLabel,
      stratum: stratum.stratum,
      weight: stratum.weight,
      portableSamples: Object.freeze([...stratum.portableSamples]),
      congruentSamples: Object.freeze([...stratum.congruentSamples]),
      portableMedianMilliseconds: portableMedian,
      congruentMedianMilliseconds: congruentMedian,
      medianSpeedup: portableMedian / congruentMedian,
    });
  });
  return Object.freeze({
    exactRangeCount,
    portableProjectedMilliseconds,
    congruentProjectedMilliseconds,
    projectedSpeedup:
      portableProjectedMilliseconds / congruentProjectedMilliseconds,
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
  const topology = buildOpt0015C300Topology();
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter, topology);
  const limits = requiredDeviceLimits(adapter, topology);
  const device = await adapter.requestDevice({
    label: "ace-opt-0015-conv-transpose1d-ab-device",
    requiredFeatures: ["shader-f16"],
    requiredLimits: limits,
  });
  const tracker = new BufferTracker();
  const portableKernel = AceFp16VaeConvTranspose1dKernel.create(device);
  const congruentKernel =
    AceFp16VaeConvTranspose1dKernel.createCongruent(device);
  const preparedOperations: PreparedOperation[] = [];
  let destroyed = false;
  const destroy = (): Readonly<Record<string, unknown>> => {
    if (destroyed) return Object.freeze({ ...tracker.receipt(), idempotent: true });
    destroyed = true;
    portableKernel.destroy();
    congruentKernel.destroy();
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
    const correctnessCases: unknown[] = [];
    let comparedU16Count = 0;
    for (const [index, operation] of topology.entries()) {
      updateProgress(`correctness ${index + 1}/5: ${operation.label}`);
      const prepared = await prepareOperation(
        device,
        tracker,
        portableKernel,
        congruentKernel,
        operation,
      );
      preparedOperations.push(prepared);
      correctnessCases.push(prepared.correctness);
      comparedU16Count += Number(prepared.correctness["comparedU16Count"]);
      await yieldToBrowser();
    }
    await device.queue.onSubmittedWorkDone();
    const sourceAuthority = await buildSourceAuthority(topology);
    const preparedCompletedAtEpochMilliseconds = Date.now();
    const correctness = Object.freeze({
      caseCount: correctnessCases.length,
      selectedRangeCount: topology.reduce(
        (sum, operation) => sum + operation.selectedRanges.length,
        0,
      ),
      executionsPerArmPerRange: 2,
      comparedU16Count,
      mismatchCount: 0,
      completeSelectedRangeRawU16Comparison: true,
      deterministicRerunHashes: true,
      qNaNPrefillCompleteWrites: true,
      guardsAndAdjacentCanariesUntouched: true,
      cases: Object.freeze(correctnessCases),
    });
    return Object.freeze({
      adapter,
      device,
      topology,
      operations: Object.freeze(preparedOperations),
      tracker,
      portableKernel,
      congruentKernel,
      correctness,
      sourceAuthority,
      preparedCompletedAtEpochMilliseconds,
      destroy,
    });
  } catch (error) {
    destroy();
    throw error;
  }
}

async function prepareOperation(
  device: GPUDevice,
  tracker: BufferTracker,
  portableKernel: AceFp16VaeConvTranspose1dKernel,
  congruentKernel: AceFp16VaeConvTranspose1dKernel<
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
  >,
  topology: Opt0015C300Operation,
): Promise<PreparedOperation> {
  const plan = planAceFp16VaeConvTranspose1d(topology.shape);
  const input = createPeriodicStorage(
    device,
    tracker,
    `${topology.label}-input`,
    plan.inputElements,
    plan.inputBindingBytes,
    INPUT_PATTERN,
    topology.operationIndex * 7,
  );
  const weight = createPeriodicStorage(
    device,
    tracker,
    `${topology.label}-weight`,
    plan.weightElements,
    plan.weightBindingBytes,
    WEIGHT_PATTERN,
    topology.operationIndex * 11,
  );
  const bias = createPeriodicStorage(
    device,
    tracker,
    `${topology.label}-bias`,
    plan.outputChannels,
    plan.biasBindingBytes,
    BIAS_PATTERN,
    topology.operationIndex * 5,
  );
  const outputs = Object.freeze({
    portable: createGuardedOutput(
      device,
      tracker,
      `${topology.label}-portable-output`,
      plan,
      topology.selectedRanges,
    ),
    congruent: createGuardedOutput(
      device,
      tracker,
      `${topology.label}-congruent-output`,
      plan,
      topology.selectedRanges,
    ),
  });
  const maximumRangeBytes = Math.max(
    ...topology.selectedRanges.map((range) => range.count * FLOAT16_BYTES),
  );
  const prefill = tracker.create(device, {
    label: `${topology.label}-qnan-prefill`,
    size: maximumRangeBytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint16Array(prefill.getMappedRange()).fill(OUTPUT_PREFILL_QNAN_F16);
  prefill.unmap();
  const controlAlignment = Number(device.limits.minUniformBufferOffsetAlignment);
  const controls = createRangeControls(
    device,
    tracker,
    `${topology.label}-controls`,
    topology.selectedRanges,
    controlAlignment,
  );
  const common = Object.freeze({
    input: binding(input, plan.inputBindingBytes),
    weight: binding(weight, plan.weightBindingBytes),
    bias: binding(bias, plan.biasBindingBytes),
  });
  const portableDispatches: AceFp16VaeConvTranspose1dDispatch[] = [];
  const congruentDispatches: EncodableDispatch[] = [];
  for (const [index, range] of topology.selectedRanges.entries()) {
    const rangeBinding = Object.freeze({
      base: range.base,
      count: range.count,
      control: Object.freeze({
        buffer: controls,
        offset: index * controlAlignment,
        size: RANGE_CONTROL_BYTES,
      }),
    });
    const portable = await portableKernel.createDispatch(
      `${topology.label}-${range.stratum}-portable`,
      topology.shape,
      Object.freeze({ ...common, output: outputs.portable.binding }),
      rangeBinding,
    );
    const congruent = await congruentKernel.createDispatch(
      `${topology.label}-${range.stratum}-congruent`,
      topology.shape,
      Object.freeze({ ...common, output: outputs.congruent.binding }),
      rangeBinding,
    );
    assertDispatches(topology, range, portable, congruent);
    portableDispatches.push(portable);
    congruentDispatches.push(congruent);
  }
  const dispatches = Object.freeze({
    portable: Object.freeze(portableDispatches),
    congruent: Object.freeze(congruentDispatches),
  });
  const correctness = await runOperationCorrectness(
    device,
    tracker,
    topology,
    plan,
    outputs,
    prefill,
    dispatches,
  );
  return Object.freeze({
    topology,
    plan,
    outputs,
    prefill,
    dispatches,
    correctness,
  });
}

async function runOperationCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  topology: Opt0015C300Operation,
  plan: AceFp16VaeConvTranspose1dPlan,
  outputs: Readonly<Record<KernelArm, GuardedOutput>>,
  prefill: GPUBuffer,
  dispatches: Readonly<Record<KernelArm, readonly EncodableDispatch[]>>,
): Promise<Readonly<Record<string, unknown>>> {
  const ranges: unknown[] = [];
  let comparedU16Count = 0;
  for (const [rangeIndex, range] of topology.selectedRanges.entries()) {
    const executions = [];
    for (const [roundIndex, order] of [
      ["portable", "congruent"],
      ["congruent", "portable"],
    ].entries()) {
      const result = await executeCorrectnessPair(
        device,
        tracker,
        topology.label,
        plan,
        range,
        outputs,
        prefill,
        dispatches,
        order as readonly KernelArm[],
        rangeIndex,
        roundIndex,
      );
      executions.push(result);
      comparedU16Count += range.count;
    }
    const first = executions[0]!;
    const rerun = executions[1]!;
    if (
      first["portableSha256"] !== rerun["portableSha256"] ||
      first["congruentSha256"] !== rerun["congruentSha256"]
    ) {
      throw new Error(`${topology.label} ${range.stratum} rerun changed`);
    }
    ranges.push(Object.freeze({
      stratum: range.stratum,
      rangeIndex: range.rangeIndex,
      base: range.base,
      count: range.count,
      firstOutputRow: range.firstOutputRow,
      outputRowCount: range.outputRowCount,
      weight: range.weight,
      portableSha256: first["portableSha256"],
      congruentSha256: first["congruentSha256"],
      rawU16MismatchCount: 0,
      deterministicRerun: true,
    }));
  }
  return Object.freeze({
    label: topology.label,
    shape: topology.shape,
    outputFrames: plan.outputFrames,
    exactGraphRangeCount: topology.ranges.length,
    selectedRangeCount: topology.selectedRanges.length,
    comparedU16Count,
    rawU16MismatchCount: 0,
    completeSelectedWrites: true,
    deterministicRerun: true,
    ranges: Object.freeze(ranges),
  });
}

async function executeCorrectnessPair(
  device: GPUDevice,
  tracker: BufferTracker,
  operationLabel: string,
  plan: AceFp16VaeConvTranspose1dPlan,
  range: Opt0015SelectedRange,
  outputs: Readonly<Record<KernelArm, GuardedOutput>>,
  prefill: GPUBuffer,
  dispatches: Readonly<Record<KernelArm, readonly EncodableDispatch[]>>,
  order: readonly KernelArm[],
  selectedRangeIndex: number,
  roundIndex: number,
): Promise<Readonly<Record<string, unknown>>> {
  const selectedBytes = range.count * FLOAT16_BYTES;
  const encoder = device.createCommandEncoder({
    label: `${operationLabel}-${range.stratum}-correctness-${roundIndex}`,
  });
  for (const arm of ["portable", "congruent"] as const) {
    encoder.copyBufferToBuffer(
      prefill,
      0,
      outputs[arm].buffer,
      OUTPUT_GUARD_BYTES + range.base * FLOAT16_BYTES,
      selectedBytes,
    );
  }
  const pass = encoder.beginComputePass();
  for (const arm of order) dispatches[arm][selectedRangeIndex]!.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const readback = await readAndCompareRange(
    device,
    tracker,
    operationLabel,
    plan,
    range,
    outputs,
    roundIndex,
  );
  return Object.freeze({
    roundIndex,
    order: Object.freeze([...order]),
    comparedU16Count: range.count,
    rawU16MismatchCount: 0,
    portableSha256: readback.portable.sha256,
    congruentSha256: readback.congruent.sha256,
  });
}

async function readAndCompareRange(
  device: GPUDevice,
  tracker: BufferTracker,
  operationLabel: string,
  plan: AceFp16VaeConvTranspose1dPlan,
  range: Opt0015SelectedRange,
  outputs: Readonly<Record<KernelArm, GuardedOutput>>,
  roundIndex: number,
): Promise<Readonly<Record<KernelArm, ReadbackArm>>> {
  const selectedBytes = range.count * FLOAT16_BYTES;
  const payloadOffset = range.base * FLOAT16_BYTES;
  const beforeBytes = Math.min(OUTPUT_GUARD_BYTES, payloadOffset);
  const afterBytes = Math.min(
    OUTPUT_GUARD_BYTES,
    plan.outputBindingBytes - payloadOffset - selectedBytes,
  );
  const armBytes = OUTPUT_GUARD_BYTES * 2 + selectedBytes +
    beforeBytes + afterBytes;
  if ([selectedBytes, beforeBytes, afterBytes, armBytes].some((value) =>
    value % 4 !== 0
  )) {
    throw new Error(`${operationLabel} selected readback lost U32 alignment`);
  }
  const readback = tracker.create(device, {
    label: `${operationLabel}-${range.stratum}-${roundIndex}-readback`,
    size: armBytes * 2,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder();
    for (const [armIndex, arm] of (["portable", "congruent"] as const)
      .entries()) {
      const target = outputs[arm].buffer;
      const base = armIndex * armBytes;
      const selectedOffset = base + OUTPUT_GUARD_BYTES * 2;
      const beforeOffset = selectedOffset + selectedBytes;
      const afterOffset = beforeOffset + beforeBytes;
      encoder.copyBufferToBuffer(
        target,
        0,
        readback,
        base,
        OUTPUT_GUARD_BYTES,
      );
      encoder.copyBufferToBuffer(
        target,
        OUTPUT_GUARD_BYTES + plan.outputBindingBytes,
        readback,
        base + OUTPUT_GUARD_BYTES,
        OUTPUT_GUARD_BYTES,
      );
      encoder.copyBufferToBuffer(
        target,
        OUTPUT_GUARD_BYTES + payloadOffset,
        readback,
        selectedOffset,
        selectedBytes,
      );
      if (beforeBytes > 0) {
        encoder.copyBufferToBuffer(
          target,
          OUTPUT_GUARD_BYTES + payloadOffset - beforeBytes,
          readback,
          beforeOffset,
          beforeBytes,
        );
      }
      if (afterBytes > 0) {
        encoder.copyBufferToBuffer(
          target,
          OUTPUT_GUARD_BYTES + payloadOffset + selectedBytes,
          readback,
          afterOffset,
          afterBytes,
        );
      }
    }
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    const mappedRange = readback.getMappedRange();
    const arms = {} as Record<KernelArm, ReadbackArm>;
    for (const [armIndex, arm] of (["portable", "congruent"] as const)
      .entries()) {
      const base = armIndex * armBytes;
      const selectedOffset = base + OUTPUT_GUARD_BYTES * 2;
      const beforeOffset = selectedOffset + selectedBytes;
      const afterOffset = beforeOffset + beforeBytes;
      const prefix = new Uint16Array(
        mappedRange,
        base,
        OUTPUT_GUARD_BYTES / FLOAT16_BYTES,
      );
      const suffix = new Uint16Array(
        mappedRange,
        base + OUTPUT_GUARD_BYTES,
        OUTPUT_GUARD_BYTES / FLOAT16_BYTES,
      );
      const selected = new Uint16Array(
        mappedRange,
        selectedOffset,
        range.count,
      );
      const before = new Uint16Array(
        mappedRange,
        beforeOffset,
        beforeBytes / FLOAT16_BYTES,
      );
      const after = new Uint16Array(
        mappedRange,
        afterOffset,
        afterBytes / FLOAT16_BYTES,
      );
      let nonFiniteCount = 0;
      let prefillQNaNCount = 0;
      for (const bits of selected) {
        if ((bits & 0x7c00) === 0x7c00) nonFiniteCount += 1;
        if (bits === OUTPUT_PREFILL_QNAN_F16) prefillQNaNCount += 1;
      }
      const guardsAndCanariesUntouched =
        everyBit(prefix, OUTPUT_GUARD_F16) &&
        everyBit(suffix, OUTPUT_GUARD_F16) &&
        everyBit(before, OUTPUT_CANARY_F16) &&
        everyBit(after, OUTPUT_CANARY_F16);
      if (
        nonFiniteCount !== 0 || prefillQNaNCount !== 0 ||
        !guardsAndCanariesUntouched
      ) {
        throw new Error(
          `${operationLabel} ${range.stratum} ${arm} write/guard scan failed`,
        );
      }
      const detached = selected.slice();
      arms[arm] = Object.freeze({
        sha256: await sha256Hex(new Uint8Array(detached.buffer)),
        nonFiniteCount,
        prefillQNaNCount,
        guardsAndCanariesUntouched,
        bits: detached,
      });
    }
    let mismatchCount = 0;
    let firstMismatchIndex: number | null = null;
    const portableBits = arms.portable.bits;
    const congruentBits = arms.congruent.bits;
    for (let index = 0; index < portableBits.length; index += 1) {
      if (portableBits[index] === congruentBits[index]) continue;
      mismatchCount += 1;
      firstMismatchIndex ??= index;
    }
    if (mismatchCount !== 0) {
      throw new Error(
        `${operationLabel} ${range.stratum} raw U16 mismatch ` +
          `${mismatchCount}@${String(firstMismatchIndex)}`,
      );
    }
    return Object.freeze(arms);
  } finally {
    if (mapped) readback.unmap();
    tracker.destroy(readback);
  }
}

async function runTimedGate(
  prepared: PreparedGate,
): Promise<Readonly<Record<string, unknown>>> {
  const launchedAtEpochMilliseconds = Date.now();
  const thermal = parseOpt0015ThermalGate(
    collectThermalParameters(),
    prepared.preparedCompletedAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const strata: Opt0015WeightedStratumInput[] = [];
  const timingStartedAtEpochMilliseconds = Date.now();
  for (const operation of prepared.operations) {
    for (const [selectedRangeIndex, range] of operation.topology.selectedRanges
      .entries()) {
      const samples: Record<KernelArm, number[]> = {
        portable: [],
        congruent: [],
      };
      for (const order of [
        ["portable", "congruent"],
        ["congruent", "portable"],
      ] as const) {
        for (const arm of order) {
          samples[arm].push(await executeTimedDispatch(
            prepared.device,
            operation.dispatches[arm][selectedRangeIndex]!,
          ));
        }
      }
      strata.push(Object.freeze({
        operationLabel: operation.topology.label,
        stratum: range.stratum,
        weight: range.weight,
        portableSamples: Object.freeze(samples.portable),
        congruentSamples: Object.freeze(samples.congruent),
      }));
    }
    await yieldToBrowser();
  }
  const timingCompletedAtEpochMilliseconds = Date.now();
  const timing = summarizeOpt0015WeightedTiming(strata);
  if (timing["exactRangeCount"] !== 378) {
    throw new Error("OPT-0015 weighted timing did not cover 378 C300 ranges");
  }
  const capability = capabilityReceipt(prepared.adapter, prepared.device);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanup = prepared.destroy();
  return Object.freeze({
    schema: "ace-opt-0015-vae-fp16-conv-transpose1d-ab-v1",
    status: "passed",
    experimentId: EXPERIMENT_ID,
    classification:
      "primitive-weighted-decision-gate-not-integrated-decoder-wall",
    recordedAt: new Date().toISOString(),
    identity: Object.freeze({
      baselineCommit: BASELINE_COMMIT,
      sourceAuthority: prepared.sourceAuthority,
    }),
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      page: window.location.href,
      ...capability,
    }),
    protocol: Object.freeze({
      thermal,
      correctnessCompletedBeforeThermalGate: true,
      allPipelinesCompiledAndWarmedByCorrectnessBeforeTiming: true,
      armOrdersPerStratum: Object.freeze([
        "portable-congruent",
        "congruent-portable",
      ]),
      samplesPerArmPerStratum: 2,
      authoritativeTiming: "performance.now-submit-through-queue-drain",
      compileAllocationUploadReadbackAndCleanupExcludedFromTiming: true,
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
      exactRangeCount: 378,
      operations: Object.freeze(prepared.topology.map((operation) =>
        Object.freeze({
          label: operation.label,
          shape: operation.shape,
          outputFrames: operation.outputFrames,
          outputElements: operation.outputElements,
          exactRangeCount: operation.ranges.length,
          selectedRanges: operation.selectedRanges,
        })
      )),
    }),
    correctness: prepared.correctness,
    timing: Object.freeze({
      ...timing,
      timingStartedAtEpochMilliseconds,
      timingCompletedAtEpochMilliseconds,
      caveat:
        "Each exact first/interior/tail range representative is separately drained and weighted by its C300 graph multiplicity; this is a primitive decision projection, not the integrated scheduler or decoder wall.",
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

async function buildSourceAuthority(
  topology: readonly Opt0015C300Operation[],
): Promise<Readonly<Record<string, unknown>>> {
  const kernelCoreSourceSha256 = await sha256Text(kernelCoreSource);
  if (kernelCoreSourceSha256 !== CORE_SOURCE_SHA256) {
    throw new Error("OPT-0015 rejected unauthenticated kernel core source");
  }
  const generatedShaders = [];
  for (const operation of topology) {
    generatedShaders.push(Object.freeze({
      label: operation.label,
      portableSha256: await sha256Text(
        aceFp16VaeConvTranspose1dWgsl(operation.shape),
      ),
      congruentSha256: await sha256Text(
        aceFp16VaeCongruentConvTranspose1dWgsl(operation.shape),
      ),
    }));
  }
  return Object.freeze({
    coreCommit: CORE_COMMIT,
    kernelCoreSourceSha256,
    portableKernelId: ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
    congruentKernelId: ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID,
    generatedShaders: Object.freeze(generatedShaders),
  });
}

function createPeriodicStorage(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  elements: number,
  bindingBytes: number,
  pattern: Uint16Array,
  shift: number,
): GPUBuffer {
  const buffer = tracker.create(device, {
    label,
    size: bindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    const destination = new Uint16Array(buffer.getMappedRange());
    destination.fill(SOURCE_PADDING_F16);
    const shifted = new Uint16Array(pattern.length);
    for (let index = 0; index < pattern.length; index += 1) {
      shifted[index] = pattern[(index + shift) % pattern.length]!;
    }
    fillPeriodicPrefix(destination, shifted, elements);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function fillPeriodicPrefix(
  destination: Uint16Array,
  pattern: Uint16Array,
  elements: number,
): void {
  if (elements < 1 || elements > destination.length || pattern.length < 1) {
    throw new RangeError("OPT-0015 periodic upload geometry changed");
  }
  const initial = Math.min(pattern.length, elements);
  destination.set(pattern.subarray(0, initial));
  let filled = initial;
  while (filled < elements) {
    const count = Math.min(filled, elements - filled);
    destination.copyWithin(filled, 0, count);
    filled += count;
  }
}

function createGuardedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  plan: AceFp16VaeConvTranspose1dPlan,
  ranges: readonly Opt0015SelectedRange[],
): GuardedOutput {
  const buffer = tracker.create(device, {
    label,
    size: OUTPUT_GUARD_BYTES + plan.outputBindingBytes + OUTPUT_GUARD_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  const guard = new Uint16Array(OUTPUT_GUARD_BYTES / FLOAT16_BYTES);
  guard.fill(OUTPUT_GUARD_F16);
  device.queue.writeBuffer(buffer, 0, guard);
  device.queue.writeBuffer(
    buffer,
    OUTPUT_GUARD_BYTES + plan.outputBindingBytes,
    guard,
  );
  const canary = new Uint16Array(OUTPUT_GUARD_BYTES / FLOAT16_BYTES);
  canary.fill(OUTPUT_CANARY_F16);
  for (const range of ranges) {
    const start = range.base * FLOAT16_BYTES;
    const end = start + range.count * FLOAT16_BYTES;
    const before = Math.min(OUTPUT_GUARD_BYTES, start);
    const after = Math.min(
      OUTPUT_GUARD_BYTES,
      plan.outputBindingBytes - end,
    );
    if (before > 0) {
      device.queue.writeBuffer(
        buffer,
        OUTPUT_GUARD_BYTES + start - before,
        canary,
        0,
        before / FLOAT16_BYTES,
      );
    }
    if (after > 0) {
      device.queue.writeBuffer(
        buffer,
        OUTPUT_GUARD_BYTES + end,
        canary,
        0,
        after / FLOAT16_BYTES,
      );
    }
  }
  return Object.freeze({
    buffer,
    binding: Object.freeze({
      buffer,
      offset: OUTPUT_GUARD_BYTES,
      size: plan.outputBindingBytes,
    }),
  });
}

function createRangeControls(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  ranges: readonly Opt0015SelectedRange[],
  alignment: number,
): GPUBuffer {
  if (!Number.isSafeInteger(alignment) || alignment < RANGE_CONTROL_BYTES) {
    throw new RangeError("OPT-0015 uniform alignment is invalid");
  }
  const buffer = tracker.create(device, {
    label,
    size: ranges.length * alignment,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  const words = new Uint32Array(buffer.getMappedRange());
  for (const [index, range] of ranges.entries()) {
    const offset = index * alignment / Uint32Array.BYTES_PER_ELEMENT;
    words[offset] = range.base;
    words[offset + 1] = range.count;
  }
  buffer.unmap();
  return buffer;
}

function assertDispatches(
  operation: Opt0015C300Operation,
  range: Opt0015SelectedRange,
  portable: AceFp16VaeConvTranspose1dDispatch,
  congruent: EncodableDispatch,
): void {
  const plan = planAceFp16VaeConvTranspose1d(operation.shape);
  const expectedPortable = planAceFp16VaeConvTranspose1dRange(plan, range);
  const expectedCongruent =
    planAceFp16VaeConvTranspose1dCongruentRange(plan, range);
  if (
    portable.kernelId !== ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID ||
    congruent.kernelId !== ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID ||
    portable.outputRange.base !== expectedPortable.base ||
    portable.outputRange.count !== expectedPortable.count ||
    congruent.outputRange.base !== expectedCongruent.base ||
    congruent.outputRange.count !== expectedCongruent.count ||
    congruent.outputRange.workgroupsZ !== operation.shape.stride
  ) {
    throw new Error(`${operation.label} ${range.stratum} dispatch changed`);
  }
}

function requiredDeviceLimits(
  adapter: GPUAdapter,
  topology: readonly Opt0015C300Operation[],
): Record<string, number> {
  let maximumBuffer = 1;
  let maximumStorageBinding = 1;
  let maximumWorkgroupStorage = 1;
  let maximumDispatch = 1;
  for (const operation of topology) {
    const plan = planAceFp16VaeConvTranspose1d(operation.shape);
    maximumStorageBinding = Math.max(
      maximumStorageBinding,
      plan.inputBindingBytes,
      plan.weightBindingBytes,
      plan.biasBindingBytes,
      plan.outputBindingBytes,
    );
    maximumBuffer = Math.max(
      maximumBuffer,
      plan.inputBindingBytes,
      plan.weightBindingBytes,
      OUTPUT_GUARD_BYTES + plan.outputBindingBytes + OUTPUT_GUARD_BYTES,
    );
    maximumWorkgroupStorage = Math.max(
      maximumWorkgroupStorage,
      plan.workgroupStorageBytes,
    );
    for (const range of operation.selectedRanges) {
      const portable = planAceFp16VaeConvTranspose1dRange(plan, range);
      const congruent = planAceFp16VaeConvTranspose1dCongruentRange(plan, range);
      maximumDispatch = Math.max(
        maximumDispatch,
        portable.workgroupsX,
        portable.workgroupsY,
        congruent.workgroupsX,
        congruent.workgroupsY,
        congruent.workgroupsZ,
      );
    }
  }
  const requested = {
    maxBufferSize: maximumBuffer,
    maxStorageBufferBindingSize: maximumStorageBinding,
    maxUniformBufferBindingSize: RANGE_CONTROL_BYTES,
    maxComputeWorkgroupStorageSize: maximumWorkgroupStorage,
    maxComputeInvocationsPerWorkgroup: 128,
    maxComputeWorkgroupSizeX: 16,
    maxComputeWorkgroupSizeY: 8,
    maxComputeWorkgroupsPerDimension: maximumDispatch,
  };
  for (const [name, minimum] of Object.entries(requested)) {
    const actual = Number(adapter.limits[name as keyof GPUSupportedLimits]);
    if (!Number.isFinite(actual) || actual < minimum) {
      throw new RangeError(
        `OPT-0015 adapter ${name}=${actual} is below ${minimum}`,
      );
    }
  }
  return requested;
}

function requireAdapter(
  adapter: GPUAdapter,
  topology: readonly Opt0015C300Operation[],
): void {
  if (!adapter.features.has("shader-f16")) {
    throw new Error("OPT-0015 requires adapter shader-f16");
  }
  if (adapter.limits.minStorageBufferOffsetAlignment > OUTPUT_GUARD_BYTES) {
    throw new Error("OPT-0015 output guard is below storage alignment");
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
      maxComputeWorkgroupStorageSize:
        device.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupsPerDimension:
        device.limits.maxComputeWorkgroupsPerDimension,
      minStorageBufferOffsetAlignment:
        device.limits.minStorageBufferOffsetAlignment,
      minUniformBufferOffsetAlignment:
        device.limits.minUniformBufferOffsetAlignment,
    }),
  });
}

function collectThermalParameters(): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    "#thermal-gate input[name]",
  )) parameters.set(input.name, input.value);
  return parameters;
}

function median(samples: readonly number[]): number {
  if (
    samples.length === 0 || samples.some((sample) =>
      !Number.isFinite(sample) || sample <= 0
    )
  ) {
    throw new RangeError("OPT-0015 timing samples must be finite and positive");
  }
  const sorted = [...samples].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function sameShape(
  left: AceVaeConvTranspose1dShape,
  right: AceVaeConvTranspose1dShape,
): boolean {
  return left.batch === right.batch &&
    left.inputFrames === right.inputFrames &&
    left.inputChannels === right.inputChannels &&
    left.outputChannels === right.outputChannels &&
    left.kernelSize === right.kernelSize && left.stride === right.stride &&
    left.dilation === right.dilation && left.padding === right.padding &&
    left.outputPadding === right.outputPadding;
}

function everyBit(bits: Uint16Array, expected: number): boolean {
  for (const value of bits) if (value !== expected) return false;
  return true;
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

async function sha256Text(value: string): Promise<string> {
  return sha256Hex(new TextEncoder().encode(value));
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    Uint8Array.from(bytes),
  );
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.length === 0) {
    throw new Error(`OPT-0015 thermal field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0015 thermal field ${name} is not finite`);
  }
  return value;
}

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, 0));
}

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0015-vae-fp16-conv-transpose1d-ab-v1",
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

function requireElement<ElementType extends Element>(selector: string): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing OPT-0015 element ${selector}`);
  return element;
}
