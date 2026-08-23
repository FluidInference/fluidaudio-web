/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import currentCoreSource from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.ts?raw";
import candidateCoreSource from
  "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-subgroup.ts?raw";
import {
  ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID,
  AceFp16VaeConvTranspose1dKernel,
  aceFp16VaeCongruentConvTranspose1dWgsl,
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dCongruentRange,
  type AceFp16VaeConvTranspose1dPlan,
} from "../../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import {
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
  ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID,
  AceOpt0022VaeConvTranspose1dSubgroupKernel,
  aceOpt0022VaeConvTranspose1dSubgroupWgsl,
  planAceOpt0022VaeConvTranspose1d,
  planAceOpt0022VaeConvTranspose1dRange,
} from "../../src/webgpu/kernels/vae-conv-transpose1d-fp16-subgroup.js";
import type { AceVaeConvTranspose1dShape } from
  "../../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderConvTransposeOperation,
} from "../../src/webgpu/vae-decoder.js";

export type Opt0022Arm = "A" | "B";
export type Opt0022TimingStratum = "first" | "interior" | "tail";

interface ExactRange {
  readonly rangeIndex: number;
  readonly base: number;
  readonly count: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
}

export interface Opt0022SelectedRange extends ExactRange {
  readonly stratum: Opt0022TimingStratum;
  readonly weight: number;
  readonly phaseMask: number;
  readonly oneTapOutputRowCount: number;
  readonly twoTapOutputRowCount: number;
  readonly containsFirstOutputRow: boolean;
  readonly containsLastOutputRow: boolean;
}

export interface Opt0022C300Operation {
  readonly operationIndex: number;
  readonly ordinal: number;
  readonly label: string;
  readonly shape: AceVaeConvTranspose1dShape;
  readonly outputFrames: number;
  readonly outputElements: number;
  readonly ranges: readonly ExactRange[];
  readonly selectedRanges: readonly Opt0022SelectedRange[];
}

export interface Opt0022ThermalGate {
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

export interface Opt0022TimingRoundInput {
  readonly roundIndex: number;
  readonly order: "AB" | "BA";
  readonly aMilliseconds: number;
  readonly bMilliseconds: number;
}

export interface Opt0022ProbeTimingInput {
  readonly operationLabel: string;
  readonly stratum: Opt0022TimingStratum;
  readonly weight: number;
  readonly rounds: readonly Opt0022TimingRoundInput[];
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

interface CandidateKernelOwner {
  createDispatch(
    label: string,
    shape: AceVaeConvTranspose1dShape,
    bindings: Readonly<{
      input: GPUBufferBinding;
      polyphaseWeight: GPUBufferBinding;
      bias: GPUBufferBinding;
      output: GPUBufferBinding;
    }>,
    range: Readonly<{
      base: number;
      count: number;
      control: GPUBufferBinding;
    }>,
  ): Promise<EncodableDispatch>;
  destroy(): void;
}

interface GuardedOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
}

interface ImmutableSource {
  readonly label: string;
  readonly buffer: GPUBuffer;
  readonly bindingBytes: number;
  readonly initialSha256: string;
}

interface PreparedOperation {
  readonly topology: Opt0022C300Operation;
  readonly plan: AceFp16VaeConvTranspose1dPlan;
  readonly sources: readonly ImmutableSource[];
  readonly outputs: Readonly<Record<Opt0022Arm, GuardedOutput>>;
  readonly prefill: GPUBuffer;
  readonly dispatches: Readonly<Record<Opt0022Arm, readonly EncodableDispatch[]>>;
  readonly layout: Readonly<Record<string, unknown>>;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly currentKernel: AceFp16VaeConvTranspose1dKernel<
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
  >;
  readonly candidateKernel: CandidateKernelOwner;
  readonly topology: readonly Opt0022C300Operation[];
  readonly operations: readonly PreparedOperation[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly sourceAuthority: Readonly<Record<string, unknown>>;
  readonly preparation: Readonly<Record<string, unknown>>;
  readonly uncapturedErrors: readonly string[];
  readonly preparedCompletedAtEpochMilliseconds: number;
  readonly updateProgress: (message: string) => void;
  destroy(): Readonly<Record<string, unknown>>;
}

interface ReadbackArm {
  readonly sha256: string;
  readonly bits: Uint16Array;
  readonly nonFiniteCount: number;
  readonly qNaNPrefillCount: number;
  readonly prefixGuardIntact: boolean;
  readonly suffixGuardIntact: boolean;
  readonly adjacentCanariesIntact: boolean;
}

const EXPERIMENT_ID = "OPT-0022" as const;
const REGISTRATION_COMMIT =
  "88a1bc1f3ffb06a6ca714437ba8616c11ed212f2" as const;
const INTEGRATED_CURRENT_COMMIT =
  "36608b857827b2b1d31ac91bf5cca9639fb0b9ed" as const;
const CURRENT_CORE_SOURCE_SHA256 =
  "cbcb9bcd5f856ce1c9e10aabca0ec0f95651c03d2c45b8076de3ba5022c6c3e2" as const;
const CANDIDATE_CORE_SOURCE_SHA256 =
  "b3a02e29419021d78f669b7ed0333b80c8e5739ed46d9d9645f814d971a9edfa" as const;
const FIXTURE_MANIFEST_SHA256 =
  "cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb" as const;
const MODEL_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6" as const;
const VAE_MANIFEST_SHA256 =
  "5644bcca87678b4f654b9541459355a73ef136c6bb601aa783b6f50fe2f6dba3" as const;
const C300_INPUT_FRAMES = 300;
const C300_EXACT_RANGE_COUNT = 378;
const SELECTED_PROBE_COUNT = 15;
const TOTAL_WEIGHT_U16_COUNT = 49_610_752;
const CORRECTNESS_COMPARISON_U16_COUNT = 8_404_992;
const TIMING_ROUNDS = 6;
export const OPT_0022_WEIGHTED_SPEEDUP_THRESHOLD =
  1.3398349037268882 as const;
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
  0x3555,
]);
const WEIGHT_PATTERN = new Uint16Array([
  0x3c00, 0xbc00, 0x3800, 0xb800,
  0x3400, 0xb400, 0x3000, 0xb000,
  0x2c00, 0xac00, 0x2400, 0xa400,
  0x0000, 0x8000,
  0x3555, 0xb555, 0x2aaa, 0xaaaa, 0x1c01,
]);
const BIAS_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x1000, 0x9000,
  0x2000, 0xa000, 0x2800, 0xa800,
]);

const EXPECTED_GENERATED_SHADER_SHA256 = Object.freeze([
  Object.freeze({
    label: "block-0-conv-t1",
    A: "891cf509d3a1753676ed4e352ff72b24d2dc3a8cab923af04109b979737455bf",
    B: "812c99bbbdb26a27cf06ca3f90247322a0b91de7254688c5d5fb4b155a8b963a",
  }),
  Object.freeze({
    label: "block-1-conv-t1",
    A: "5484dfeadd387c80aeffb5f696924642663e417b002dd8bf328bec247e34fcc6",
    B: "5997931ef29ad39982c98e45ef8620d57b3c5db25b1589e639ae4a0a6b1dfdc0",
  }),
  Object.freeze({
    label: "block-2-conv-t1",
    A: "a8d547c77e94c426d69b0d590f5d8fc35f8613e56da50df189d4556433f7bb7c",
    B: "529f81b8cc9a2a4a09849e024c2c543fba39dc514864e4c0294fdfeb17fc45ab",
  }),
  Object.freeze({
    label: "block-3-conv-t1",
    A: "91908682c7c1e65bea607463b66616329b8f5ac72c20c4cf093ffec5b5a6a549",
    B: "4f8e5b4d4e203d394f1fa39b2a3127d8fe3f05fb889c92c9d873fb5c7b83d732",
  }),
  Object.freeze({
    label: "block-4-conv-t1",
    A: "f560adea6e1369a438c30e5272861a194c486a70799c15b73eb26ca31ddcb2c7",
    B: "92921ab4159f36788fbf076fb1a2c1f10ad52adddb24f8a5689a9e045101f5f8",
  }),
]);

export function opt0022InputFixtureBitsAt(
  operationOrdinal: number,
  inputTime: number,
  inputChannel: number,
  inputChannels: number,
): number {
  requireCoordinate(operationOrdinal, 5, "operation ordinal");
  if (!Number.isSafeInteger(inputTime) || inputTime < 0 ||
    !Number.isSafeInteger(inputChannels) || inputChannels < 1) {
    throw new RangeError("OPT-0022 input fixture geometry changed");
  }
  requireCoordinate(inputChannel, inputChannels, "input channel");
  const linearIndex = inputTime * inputChannels + inputChannel;
  if (!Number.isSafeInteger(linearIndex)) {
    throw new RangeError("OPT-0022 input fixture index is not safe");
  }
  return INPUT_PATTERN[
    (linearIndex + operationOrdinal * 7) % INPUT_PATTERN.length
  ]!;
}

export function opt0022NativeWeightFixtureBitsAt(
  operationOrdinal: number,
  outputChannel: number,
  kernel: number,
  inputChannel: number,
  shape: AceVaeConvTranspose1dShape,
): number {
  requireCoordinate(operationOrdinal, 5, "operation ordinal");
  const nativeIndex = opt0022NativeWeightIndex(
    shape,
    outputChannel,
    kernel,
    inputChannel,
  );
  return WEIGHT_PATTERN[
    (nativeIndex + operationOrdinal * 11) % WEIGHT_PATTERN.length
  ]!;
}

export const OPT_0022_C300_EXPECTED_TOPOLOGY = Object.freeze([
  expectedOperation("block-0-conv-t1", 300, 2_048, 1_024, 20, 10, 5,
    3_000, 54, 56, 32),
  expectedOperation("block-1-conv-t1", 3_000, 1_024, 512, 12, 6, 3,
    18_000, 81, 224, 80),
  expectedOperation("block-2-conv-t1", 18_000, 512, 256, 8, 4, 2,
    72_000, 81, 896, 320),
  expectedOperation("block-3-conv-t1", 72_000, 256, 128, 8, 4, 2,
    288_000, 81, 3_584, 1_280),
  expectedOperation("block-4-conv-t1", 288_000, 128, 128, 4, 2, 1,
    576_000, 81, 7_168, 2_560),
]);

function expectedOperation(
  label: string,
  inputFrames: number,
  inputChannels: number,
  outputChannels: number,
  kernelSize: number,
  stride: number,
  padding: number,
  outputFrames: number,
  rangeCount: number,
  fullRangeRows: number,
  tailRangeRows: number,
) {
  return Object.freeze({
    label,
    shape: Object.freeze({
      batch: 1,
      inputFrames,
      inputChannels,
      outputChannels,
      kernelSize,
      stride,
      dilation: 1,
      padding,
      outputPadding: 0,
    }),
    outputFrames,
    rangeCount,
    fullRangeRows,
    tailRangeRows,
  });
}

export function buildOpt0022C300Topology(): readonly Opt0022C300Operation[] {
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
      const candidatePlan = planAceOpt0022VaeConvTranspose1d(operation.shape);
      const ranges = cooperative.quanta
        .filter((quantum) => quantum.operationIndex === operationIndex)
        .map((quantum, rangeIndex): ExactRange => {
          if (quantum.operationKind !== "conv-transpose1d" ||
            quantum.primitives.length !== 1) {
            throw new Error(`${operation.label} C300 quantum topology changed`);
          }
          const primitive = quantum.primitives[0]!;
          if (primitive.physicalPartIndex !== 0 ||
            primitive.firstOutputChannel !== 0 ||
            primitive.outputChannels !== operation.shape.outputChannels ||
            primitive.outputBase !== quantum.logicalOutputBase ||
            primitive.outputCount !== quantum.logicalOutputCount ||
            primitive.outputBase % operation.shape.outputChannels !== 0 ||
            primitive.outputCount % operation.shape.outputChannels !== 0) {
            throw new Error(`${operation.label} C300 primitive topology changed`);
          }
          planAceFp16VaeConvTranspose1dCongruentRange(plan, {
            base: primitive.outputBase,
            count: primitive.outputCount,
          });
          planAceOpt0022VaeConvTranspose1dRange(candidatePlan, {
            base: primitive.outputBase,
            count: primitive.outputCount,
          });
          return Object.freeze({
            rangeIndex,
            base: primitive.outputBase,
            count: primitive.outputCount,
            firstOutputRow: primitive.outputBase /
              operation.shape.outputChannels,
            outputRowCount: primitive.outputCount /
              operation.shape.outputChannels,
          });
        });
      if (ranges.length < 3) {
        throw new Error(`${operation.label} needs first/interior/tail ranges`);
      }
      const expected = OPT_0022_C300_EXPECTED_TOPOLOGY[ordinal];
      if (expected === undefined || operation.label !== expected.label ||
        !sameShape(operation.shape, expected.shape) ||
        plan.outputFrames !== expected.outputFrames ||
        ranges.length !== expected.rangeCount ||
        ranges[0]!.outputRowCount !== expected.fullRangeRows ||
        ranges.at(-1)!.outputRowCount !== expected.tailRangeRows) {
        throw new Error(`${operation.label} rejected unexpected C300 geometry`);
      }
      const middleIndex = Math.floor(ranges.length / 2);
      const selections = [
        { range: ranges[0]!, stratum: "first" as const, weight: 1 },
        {
          range: ranges[middleIndex]!,
          stratum: "interior" as const,
          weight: ranges.length - 2,
        },
        { range: ranges.at(-1)!, stratum: "tail" as const, weight: 1 },
      ];
      const selectedRanges = Object.freeze(selections.map((selection) =>
        Object.freeze({
          ...selection.range,
          stratum: selection.stratum,
          weight: selection.weight,
          ...rangeCoverage(operation.shape, plan.outputFrames, selection.range),
        })
      ));
      const completePhaseMask = (1 << operation.shape.stride) - 1;
      if (selectedRanges.some((range) =>
        range.phaseMask !== completePhaseMask ||
        range.oneTapOutputRowCount + range.twoTapOutputRowCount !==
          range.outputRowCount
      ) || !selectedRanges[0]!.containsFirstOutputRow ||
        !selectedRanges[2]!.containsLastOutputRow ||
        selectedRanges[0]!.oneTapOutputRowCount < 1 ||
        selectedRanges[2]!.oneTapOutputRowCount < 1) {
        throw new Error(`${operation.label} lost exact phase/edge coverage`);
      }
      return Object.freeze({
        operationIndex,
        ordinal,
        label: operation.label,
        shape: operation.shape,
        outputFrames: plan.outputFrames,
        outputElements: plan.outputElements,
        ranges: Object.freeze(ranges),
        selectedRanges,
      });
    });
  if (operations.length !== 5 ||
    operations.reduce((sum, operation) => sum + operation.ranges.length, 0) !==
      C300_EXACT_RANGE_COUNT ||
    operations.reduce((sum, operation) =>
      sum + operation.selectedRanges.length, 0) !== SELECTED_PROBE_COUNT ||
    operations.reduce((sum, operation) =>
      sum + planAceFp16VaeConvTranspose1d(operation.shape).weightElements, 0) !==
      TOTAL_WEIGHT_U16_COUNT ||
    operations.some((operation) =>
      operation.shape.inputChannels % INPUT_PATTERN.length === 0
    ) ||
    new Set(WEIGHT_PATTERN).size !== WEIGHT_PATTERN.length ||
    operations.some((operation) => {
      const { inputChannels, kernelSize, stride } = operation.shape;
      return gcd(kernelSize * inputChannels, WEIGHT_PATTERN.length) !== 1 ||
        gcd(inputChannels, WEIGHT_PATTERN.length) !== 1 ||
        gcd(stride * inputChannels, WEIGHT_PATTERN.length) !== 1;
    }) ||
    2 * operations.reduce((sum, operation) =>
      sum + operation.selectedRanges.reduce((rangeSum, range) =>
        rangeSum + range.count, 0), 0) !== CORRECTNESS_COMPARISON_U16_COUNT) {
    throw new Error("OPT-0022 exact C300 accounting changed");
  }
  return Object.freeze(operations);
}

function rangeCoverage(
  shape: AceVaeConvTranspose1dShape,
  outputFrames: number,
  range: ExactRange,
): Readonly<{
  phaseMask: number;
  oneTapOutputRowCount: number;
  twoTapOutputRowCount: number;
  containsFirstOutputRow: boolean;
  containsLastOutputRow: boolean;
}> {
  let phaseMask = 0;
  let oneTapOutputRowCount = 0;
  let twoTapOutputRowCount = 0;
  for (let rowOffset = 0; rowOffset < range.outputRowCount; rowOffset += 1) {
    const outputTime = range.firstOutputRow + rowOffset;
    const padded = outputTime + shape.padding;
    const phase = padded % shape.stride;
    phaseMask |= 1 << phase;
    let validTaps = 0;
    for (let tap = 0; tap < 2; tap += 1) {
      const kernel = phase + tap * shape.stride;
      const inputTime = (padded - kernel) / shape.stride;
      if (inputTime >= 0 && inputTime < shape.inputFrames) validTaps += 1;
    }
    if (validTaps === 1) oneTapOutputRowCount += 1;
    else if (validTaps === 2) twoTapOutputRowCount += 1;
    else throw new Error("OPT-0022 selected output row has zero valid taps");
  }
  return Object.freeze({
    phaseMask,
    oneTapOutputRowCount,
    twoTapOutputRowCount,
    containsFirstOutputRow: range.firstOutputRow === 0,
    containsLastOutputRow:
      range.firstOutputRow + range.outputRowCount === outputFrames,
  });
}

export function opt0022NativeWeightIndex(
  shape: AceVaeConvTranspose1dShape,
  outputChannel: number,
  kernel: number,
  inputChannel: number,
): number {
  requireCoordinate(outputChannel, shape.outputChannels, "output channel");
  requireCoordinate(kernel, shape.kernelSize, "kernel");
  requireCoordinate(inputChannel, shape.inputChannels, "input channel");
  return (outputChannel * shape.kernelSize + kernel) * shape.inputChannels +
    inputChannel;
}

export function opt0022PolyphaseWeightIndex(
  shape: AceVaeConvTranspose1dShape,
  phase: number,
  tap: number,
  inputChannel: number,
  outputChannel: number,
): number {
  requireCoordinate(phase, shape.stride, "phase");
  requireCoordinate(tap, 2, "tap");
  requireCoordinate(inputChannel, shape.inputChannels, "input channel");
  requireCoordinate(outputChannel, shape.outputChannels, "output channel");
  return ((phase * 2 + tap) * shape.inputChannels + inputChannel) *
    shape.outputChannels + outputChannel;
}

export function compareOpt0022InverseWeightLayouts(
  native: Uint16Array,
  polyphase: Uint16Array,
  shape: AceVaeConvTranspose1dShape,
): Readonly<Record<string, unknown>> {
  const expectedWords = shape.outputChannels * shape.kernelSize *
    shape.inputChannels;
  if (native.length !== expectedWords || polyphase.length !== expectedWords ||
    shape.kernelSize !== 2 * shape.stride) {
    throw new RangeError("OPT-0022 inverse-layout geometry changed");
  }
  let mismatchCount = 0;
  let firstMismatch: Readonly<Record<string, number>> | null = null;
  for (let outputChannel = 0; outputChannel < shape.outputChannels;
    outputChannel += 1) {
    for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
      const phase = kernel % shape.stride;
      const tap = Math.floor(kernel / shape.stride);
      for (let inputChannel = 0; inputChannel < shape.inputChannels;
        inputChannel += 1) {
        const nativeIndex = opt0022NativeWeightIndex(
          shape,
          outputChannel,
          kernel,
          inputChannel,
        );
        const polyphaseIndex = opt0022PolyphaseWeightIndex(
          shape,
          phase,
          tap,
          inputChannel,
          outputChannel,
        );
        const expected = native[nativeIndex]!;
        const actual = polyphase[polyphaseIndex]!;
        if (expected === actual) continue;
        mismatchCount += 1;
        firstMismatch ??= Object.freeze({
          outputChannel,
          kernel,
          inputChannel,
          nativeIndex,
          polyphaseIndex,
          expected,
          actual,
        });
      }
    }
  }
  return Object.freeze({
    comparedU16Count: expectedWords,
    mismatchCount,
    firstMismatch,
    rawU16Exact: mismatchCount === 0,
  });
}

export function buildOpt0022TimingOrders(): readonly Readonly<{
  roundIndex: number;
  probeOrdinal: number;
  rotatedProbeOrder: readonly number[];
  order: readonly Opt0022Arm[];
}>[] {
  const probeOrdinals = Array.from(
    { length: SELECTED_PROBE_COUNT },
    (_, index) => index,
  );
  return Object.freeze(Array.from({ length: TIMING_ROUNDS }, (_, roundIndex) => {
    const rotation = roundIndex % SELECTED_PROBE_COUNT;
    const rotatedProbeOrder = Object.freeze([
      ...probeOrdinals.slice(rotation),
      ...probeOrdinals.slice(0, rotation),
    ]);
    const order = Object.freeze(
      roundIndex % 2 === 0 ? ["A", "B"] as const : ["B", "A"] as const,
    );
    return rotatedProbeOrder.map((probeOrdinal) => Object.freeze({
      roundIndex,
      probeOrdinal,
      rotatedProbeOrder,
      order,
    }));
  }).flat());
}

export function parseOpt0022ThermalGate(
  parameters: URLSearchParams,
  preparedCompletedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0022ThermalGate {
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
    !Number.isFinite(preparedCompletedAtEpochMilliseconds) ||
    preparedCompletedAtEpochMilliseconds <= 0 ||
    !Number.isFinite(launchedAtEpochMilliseconds) ||
    launchedAtEpochMilliseconds <= 0 ||
    startedAtEpochMilliseconds <= 0 || completedAtEpochMilliseconds <= 0 ||
    !Number.isSafeInteger(observationCount) ||
    observationCount < Math.floor(
      durationMilliseconds / THERMAL_POLL_MILLISECONDS,
    ) + 1 ||
    pollMilliseconds !== THERMAL_POLL_MILLISECONDS ||
    maximumPollGapMilliseconds <= 0 ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS ||
    nonNominalObservationCount !== 0 ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    startedAtEpochMilliseconds < preparedCompletedAtEpochMilliseconds ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS) {
    throw new Error("OPT-0022 thermal gate is incomplete, stale, or non-nominal");
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

export function summarizeOpt0022Timing(
  inputs: readonly Opt0022ProbeTimingInput[],
): Readonly<Record<string, unknown>> {
  const expected = buildOpt0022C300Topology().flatMap((operation) =>
    operation.selectedRanges.map((range) => Object.freeze({
      operationLabel: operation.label,
      stratum: range.stratum,
      weight: range.weight,
      exactRangeCount: operation.ranges.length,
    }))
  );
  if (inputs.length !== expected.length) {
    throw new Error("OPT-0022 requires all 15 exact timing probes");
  }
  const operationTotals = new Map<string, {
    exactRangeCount: number;
    aMilliseconds: number;
    bMilliseconds: number;
  }>();
  const positionTotals = [
    { aMilliseconds: 0, bMilliseconds: 0 },
    { aMilliseconds: 0, bMilliseconds: 0 },
  ];
  let exactRangeCount = 0;
  let aWeightedMilliseconds = 0;
  let bWeightedMilliseconds = 0;
  const strata = inputs.map((input, index) => {
    const frozen = expected[index]!;
    if (input.operationLabel !== frozen.operationLabel ||
      input.stratum !== frozen.stratum || input.weight !== frozen.weight ||
      input.rounds.length !== TIMING_ROUNDS) {
      throw new Error(`OPT-0022 timing probe ${index} changed`);
    }
    const aSamples: number[] = [];
    const bSamples: number[] = [];
    const positionSamples = [
      { A: [] as number[], B: [] as number[] },
      { A: [] as number[], B: [] as number[] },
    ];
    for (let roundIndex = 0; roundIndex < TIMING_ROUNDS; roundIndex += 1) {
      const round = input.rounds[roundIndex]!;
      const expectedOrder = roundIndex % 2 === 0 ? "AB" : "BA";
      if (round.roundIndex !== roundIndex || round.order !== expectedOrder ||
        !finitePositive(round.aMilliseconds) ||
        !finitePositive(round.bMilliseconds)) {
        throw new Error(`OPT-0022 timing round ${roundIndex} changed`);
      }
      aSamples.push(round.aMilliseconds);
      bSamples.push(round.bMilliseconds);
      const aPosition = round.order === "AB" ? 0 : 1;
      const bPosition = 1 - aPosition;
      positionSamples[aPosition]!.A.push(round.aMilliseconds);
      positionSamples[bPosition]!.B.push(round.bMilliseconds);
    }
    const aMedian = medianExact(aSamples, 6);
    const bMedian = medianExact(bSamples, 6);
    const positionMedians = positionSamples.map((samples) => Object.freeze({
      A: medianExact(samples.A, 3),
      B: medianExact(samples.B, 3),
    }));
    exactRangeCount += input.weight;
    aWeightedMilliseconds += aMedian * input.weight;
    bWeightedMilliseconds += bMedian * input.weight;
    for (const position of [0, 1] as const) {
      positionTotals[position]!.aMilliseconds +=
        positionMedians[position]!.A * input.weight;
      positionTotals[position]!.bMilliseconds +=
        positionMedians[position]!.B * input.weight;
    }
    let operation = operationTotals.get(input.operationLabel);
    if (operation === undefined) {
      operation = {
        exactRangeCount: frozen.exactRangeCount,
        aMilliseconds: 0,
        bMilliseconds: 0,
      };
      operationTotals.set(input.operationLabel, operation);
    }
    operation.aMilliseconds += aMedian * input.weight;
    operation.bMilliseconds += bMedian * input.weight;
    return Object.freeze({
      operationLabel: input.operationLabel,
      stratum: input.stratum,
      weight: input.weight,
      rounds: Object.freeze(input.rounds.map((round) => Object.freeze({
        roundIndex: round.roundIndex,
        order: round.order,
        A: round.aMilliseconds,
        B: round.bMilliseconds,
      }))),
      medians: Object.freeze({ A: aMedian, B: bMedian }),
      positionMedians: Object.freeze({
        first: positionMedians[0],
        second: positionMedians[1],
      }),
      bFaster: bMedian < aMedian,
      speedup: aMedian / bMedian,
    });
  });
  if (exactRangeCount !== C300_EXACT_RANGE_COUNT ||
    operationTotals.size !== OPT_0022_C300_EXPECTED_TOPOLOGY.length) {
    throw new Error("OPT-0022 weighted range accounting changed");
  }
  const operations = [...operationTotals].map(([label, values]) =>
    Object.freeze({
      label,
      exactRangeCount: values.exactRangeCount,
      A: values.aMilliseconds,
      B: values.bMilliseconds,
      bFaster: values.bMilliseconds < values.aMilliseconds,
      speedup: values.aMilliseconds / values.bMilliseconds,
    })
  );
  const positions = positionTotals.map((values, position) => Object.freeze({
    position,
    A: values.aMilliseconds,
    B: values.bMilliseconds,
    bFaster: values.bMilliseconds < values.aMilliseconds,
    speedup: values.aMilliseconds / values.bMilliseconds,
  }));
  const weightedSpeedup = aWeightedMilliseconds / bWeightedMilliseconds;
  const everyOperationBFaster = operations.every((operation) =>
    operation.bFaster
  );
  const bothTimingPositionsBFaster = positions.every((position) =>
    position.bFaster
  );
  const passed = everyOperationBFaster && bothTimingPositionsBFaster &&
    weightedSpeedup >= OPT_0022_WEIGHTED_SPEEDUP_THRESHOLD;
  return Object.freeze({
    timingRounds: TIMING_ROUNDS,
    samplesPerArmPerProbe: TIMING_ROUNDS,
    exactRangeCount,
    weighted: Object.freeze({
      A: aWeightedMilliseconds,
      B: bWeightedMilliseconds,
      speedup: weightedSpeedup,
      threshold: OPT_0022_WEIGHTED_SPEEDUP_THRESHOLD,
    }),
    operations: Object.freeze(operations),
    positions: Object.freeze(positions),
    everyOperationBFaster,
    bothTimingPositionsBFaster,
    strata: Object.freeze(strata),
    passed,
    decision: passed
      ? "positive-package-layer-gate-authorized"
      : "negative-stop-primitive-gate",
  });
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  private readonly sizes = new Map<GPUBuffer, number>();
  createdBufferCount = 0;
  destroyedBufferCount = 0;
  liveBytes = 0;
  maximumLiveBytes = 0;
  mapCount = 0;
  unmapCount = 0;
  activeMapCount = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    const size = Number(descriptor.size);
    this.live.add(buffer);
    this.sizes.set(buffer, size);
    this.createdBufferCount += 1;
    this.liveBytes += size;
    this.maximumLiveBytes = Math.max(this.maximumLiveBytes, this.liveBytes);
    if (descriptor.mappedAtCreation === true) {
      this.mapCount += 1;
      this.activeMapCount += 1;
    }
    return buffer;
  }

  async mapRead(buffer: GPUBuffer): Promise<void> {
    await buffer.mapAsync(GPUMapMode.READ);
    this.mapCount += 1;
    this.activeMapCount += 1;
  }

  unmap(buffer: GPUBuffer): void {
    if (buffer.mapState !== "mapped") {
      throw new Error("OPT-0022 rejected an unbalanced unmap");
    }
    buffer.unmap();
    this.unmapCount += 1;
    this.activeMapCount -= 1;
    if (this.activeMapCount < 0) {
      throw new Error("OPT-0022 map accounting underflow");
    }
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    if (buffer.mapState === "mapped") this.unmap(buffer);
    buffer.destroy();
    this.destroyedBufferCount += 1;
    this.liveBytes -= this.sizes.get(buffer) ?? 0;
    this.sizes.delete(buffer);
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number | boolean>> {
    return Object.freeze({
      createdBufferCount: this.createdBufferCount,
      destroyedBufferCount: this.destroyedBufferCount,
      liveBufferCount: this.live.size,
      liveBytes: this.liveBytes,
      maximumLiveBytes: this.maximumLiveBytes,
      mapCount: this.mapCount,
      unmapCount: this.unmapCount,
      activeMapCount: this.activeMapCount,
      mapsBalanced: this.mapCount === this.unmapCount &&
        this.activeMapCount === 0,
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
    progress.textContent = "running six alternating A/B timing rounds";
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

async function prepareGate(
  updateProgress: (message: string) => void,
): Promise<PreparedGate> {
  const preparationStartedAtEpochMilliseconds = Date.now();
  const preparationStarted = performance.now();
  const topology = buildOpt0022C300Topology();
  updateProgress("authenticating current and candidate kernel sources");
  const sourceAuthority = await buildSourceAuthority(topology);
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter, topology);
  const requiredLimits = requiredDeviceLimits(adapter, topology);
  const device = await adapter.requestDevice({
    label: "ace-opt-0022-subgroup-polyphase-ab-device",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits,
  });
  const uncapturedErrors: string[] = [];
  const onUncapturedError = (event: GPUUncapturedErrorEvent): void => {
    uncapturedErrors.push(`${event.error.constructor.name}: ${event.error.message}`);
  };
  device.addEventListener("uncapturederror", onUncapturedError);
  const tracker = new BufferTracker();
  const currentKernel = AceFp16VaeConvTranspose1dKernel.createCongruent(device);
  const candidateKernel = AceOpt0022VaeConvTranspose1dSubgroupKernel.create(
    device,
    Object.freeze({ subgroupMinSize: 32, subgroupMaxSize: 32 }),
  );
  const operations: PreparedOperation[] = [];
  let queueDrained = false;
  let destroyed = false;
  const destroy = (): Readonly<Record<string, unknown>> => {
    if (destroyed) {
      return Object.freeze({
        ...tracker.receipt(),
        idempotent: true,
        repeatedCall: true,
        drainBeforeRelease: queueDrained,
      });
    }
    destroyed = true;
    device.removeEventListener("uncapturederror", onUncapturedError);
    currentKernel.destroy();
    candidateKernel.destroy();
    tracker.destroyAll();
    tracker.destroyAll();
    device.destroy();
    return Object.freeze({
      ...tracker.receipt(),
      idempotent: true,
      repeatedCall: false,
      drainBeforeRelease: queueDrained,
      deviceDestroyed: true,
    });
  };
  try {
    const correctnessStarted = performance.now();
    let inverseLayoutComparedU16Count = 0;
    let outputComparedU16Count = 0;
    for (const [index, operation] of topology.entries()) {
      updateProgress(`layout/exactness ${index + 1}/5: ${operation.label}`);
      const prepared = await prepareOperation(
        device,
        tracker,
        currentKernel,
        candidateKernel,
        operation,
      );
      operations.push(prepared);
      inverseLayoutComparedU16Count += Number(
        prepared.layout["comparedU16Count"],
      );
      outputComparedU16Count += Number(
        prepared.correctness["comparedU16Count"],
      );
      await yieldToBrowser();
    }
    await device.queue.onSubmittedWorkDone();
    queueDrained = true;
    await settlePostDrainEvents();
    updateProgress("verifying all source buffers immutable after correctness");
    const immutableAfterCorrectness = await verifyAllSourcesImmutable(
      device,
      tracker,
      operations,
      updateProgress,
      "post-correctness",
    );
    await settlePostDrainEvents();
    if (uncapturedErrors.length !== 0) {
      throw new Error("OPT-0022 observed an uncaptured GPU error in preparation");
    }
    if (inverseLayoutComparedU16Count !== TOTAL_WEIGHT_U16_COUNT ||
      outputComparedU16Count !== CORRECTNESS_COMPARISON_U16_COUNT ||
      immutableAfterCorrectness["passed"] !== true) {
      throw new Error("OPT-0022 aggregate correctness accounting changed");
    }
    const cases = operations.map((operation) => operation.correctness);
    const preparedCompletedAtEpochMilliseconds = Date.now();
    const correctness = Object.freeze({
      operationCount: operations.length,
      selectedProbeCount: SELECTED_PROBE_COUNT,
      exactGraphRangeCount: C300_EXACT_RANGE_COUNT,
      inverseLayoutComparedU16Count,
      inverseLayoutMismatchCount: 0,
      outputExecutionCount: SELECTED_PROBE_COUNT * 2 * 2,
      outputExactComparisonCount: SELECTED_PROBE_COUNT * 2,
      outputComparedU16Count,
      outputMismatchCount: 0,
      balancedCorrectnessOrders: Object.freeze(["AB", "BA"]),
      allPhasesCoveredInEveryProbe: true,
      bothOneTapBoundariesCoveredPerOperation: true,
      qNaNPrefillCompleteWrites: true,
      guardsAndAdjacentCanariesUntouched: true,
      deterministicRerunHashes: true,
      immutableAfterCorrectness,
      uncapturedGpuErrorCount: uncapturedErrors.length,
      cases: Object.freeze(cases),
      passed: true,
    });
    const preparation = Object.freeze({
      startedAtEpochMilliseconds: preparationStartedAtEpochMilliseconds,
      completedAtEpochMilliseconds: preparedCompletedAtEpochMilliseconds,
      totalMilliseconds: performance.now() - preparationStarted,
      layoutOutputAndSourceCorrectnessMilliseconds:
        performance.now() - correctnessStarted,
      memoryHighWaterBytes: tracker.maximumLiveBytes,
      allPipelinesCompiledAndWarmedByCorrectness: true,
    });
    return Object.freeze({
      adapter,
      device,
      tracker,
      currentKernel,
      candidateKernel,
      topology,
      operations: Object.freeze(operations),
      correctness,
      sourceAuthority,
      preparation,
      uncapturedErrors,
      preparedCompletedAtEpochMilliseconds,
      updateProgress,
      destroy,
    });
  } catch (error) {
    await device.queue.onSubmittedWorkDone().catch(() => undefined);
    queueDrained = true;
    destroy();
    throw error;
  }
}

async function prepareOperation(
  device: GPUDevice,
  tracker: BufferTracker,
  currentKernel: AceFp16VaeConvTranspose1dKernel<
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
  >,
  candidateKernel: CandidateKernelOwner,
  topology: Opt0022C300Operation,
): Promise<PreparedOperation> {
  const plan = planAceFp16VaeConvTranspose1d(topology.shape);
  const input = await createPeriodicSource(
    device,
    tracker,
    `${topology.label}-input`,
    plan.inputElements,
    plan.inputBindingBytes,
    INPUT_PATTERN,
    topology.ordinal * 7,
  );
  const weights = await createWeightSources(
    device,
    tracker,
    topology,
    plan,
  );
  const bias = await createPeriodicSource(
    device,
    tracker,
    `${topology.label}-bias`,
    plan.outputChannels,
    plan.biasBindingBytes,
    BIAS_PATTERN,
    topology.ordinal * 5,
  );
  const outputs = Object.freeze({
    A: createGuardedOutput(
      device,
      tracker,
      `${topology.label}-A-output`,
      plan,
      topology.selectedRanges,
    ),
    B: createGuardedOutput(
      device,
      tracker,
      `${topology.label}-B-output`,
      plan,
      topology.selectedRanges,
    ),
  });
  const maximumRangeBytes = Math.max(...topology.selectedRanges.map((range) =>
    range.count * FLOAT16_BYTES
  ));
  const prefill = tracker.create(device, {
    label: `${topology.label}-qnan-prefill`,
    size: maximumRangeBytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint16Array(prefill.getMappedRange()).fill(OUTPUT_PREFILL_QNAN_F16);
  tracker.unmap(prefill);
  const controlAlignment = Number(device.limits.minUniformBufferOffsetAlignment);
  const controls = createRangeControls(
    device,
    tracker,
    `${topology.label}-controls`,
    topology.selectedRanges,
    controlAlignment,
  );
  const currentDispatches: EncodableDispatch[] = [];
  const candidateDispatches: EncodableDispatch[] = [];
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
    const current = await currentKernel.createDispatch(
      `${topology.label}-${range.stratum}-A-current`,
      topology.shape,
      Object.freeze({
        input: binding(input.buffer, plan.inputBindingBytes),
        weight: binding(weights.native.buffer, plan.weightBindingBytes),
        bias: binding(bias.buffer, plan.biasBindingBytes),
        output: outputs.A.binding,
      }),
      rangeBinding,
    );
    const candidate = await candidateKernel.createDispatch(
      `${topology.label}-${range.stratum}-B-subgroup`,
      topology.shape,
      Object.freeze({
        input: binding(input.buffer, plan.inputBindingBytes),
        polyphaseWeight: binding(
          weights.polyphase.buffer,
          plan.weightBindingBytes,
        ),
        bias: binding(bias.buffer, plan.biasBindingBytes),
        output: outputs.B.binding,
      }),
      rangeBinding,
    );
    assertDispatches(topology, range, current, candidate);
    currentDispatches.push(current);
    candidateDispatches.push(candidate);
  }
  const dispatches = Object.freeze({
    A: Object.freeze(currentDispatches),
    B: Object.freeze(candidateDispatches),
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
    sources: Object.freeze([
      input,
      weights.native,
      weights.polyphase,
      bias,
    ]),
    outputs,
    prefill,
    dispatches,
    layout: weights.layout,
    correctness,
  });
}

async function createWeightSources(
  device: GPUDevice,
  tracker: BufferTracker,
  topology: Opt0022C300Operation,
  plan: AceFp16VaeConvTranspose1dPlan,
): Promise<Readonly<{
  native: ImmutableSource;
  polyphase: ImmutableSource;
  layout: Readonly<Record<string, unknown>>;
}>> {
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const nativeBuffer = tracker.create(device, {
    label: `${topology.label}-native-o-k-i-weight`,
    size: plan.weightBindingBytes,
    usage,
    mappedAtCreation: true,
  });
  const polyphaseBuffer = tracker.create(device, {
    label: `${topology.label}-polyphase-p-t-i-o-weight`,
    size: plan.weightBindingBytes,
    usage,
    mappedAtCreation: true,
  });
  let nativeUnmapped = false;
  let polyphaseUnmapped = false;
  try {
    const native = new Uint16Array(nativeBuffer.getMappedRange());
    const polyphase = new Uint16Array(polyphaseBuffer.getMappedRange());
    native.fill(SOURCE_PADDING_F16);
    polyphase.fill(SOURCE_PADDING_F16);
    const shifted = shiftedPattern(WEIGHT_PATTERN, topology.ordinal * 11);
    fillPeriodicPrefix(native, shifted, plan.weightElements);
    const { shape } = topology;
    let physical = 0;
    for (let phase = 0; phase < shape.stride; phase += 1) {
      for (let tap = 0; tap < 2; tap += 1) {
        const kernel = phase + tap * shape.stride;
        for (let inputChannel = 0; inputChannel < shape.inputChannels;
          inputChannel += 1) {
          for (let outputChannel = 0; outputChannel < shape.outputChannels;
            outputChannel += 1) {
            const nativeIndex =
              (outputChannel * shape.kernelSize + kernel) *
                shape.inputChannels + inputChannel;
            polyphase[physical] = native[nativeIndex]!;
            physical += 1;
          }
        }
      }
      await yieldToBrowser();
    }
    if (physical !== plan.weightElements) {
      throw new Error(`${topology.label} polyphase write count changed`);
    }
    let mismatchCount = 0;
    let firstMismatch: Readonly<Record<string, number>> | null = null;
    let comparedU16Count = 0;
    for (let outputChannel = 0; outputChannel < shape.outputChannels;
      outputChannel += 1) {
      for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
        const phase = kernel % shape.stride;
        const tap = Math.floor(kernel / shape.stride);
        for (let inputChannel = 0; inputChannel < shape.inputChannels;
          inputChannel += 1) {
          const nativeIndex =
            (outputChannel * shape.kernelSize + kernel) * shape.inputChannels +
            inputChannel;
          const polyphaseIndex =
            ((phase * 2 + tap) * shape.inputChannels + inputChannel) *
              shape.outputChannels + outputChannel;
          const expected = native[nativeIndex]!;
          const actual = polyphase[polyphaseIndex]!;
          comparedU16Count += 1;
          if (expected === actual) continue;
          mismatchCount += 1;
          firstMismatch ??= Object.freeze({
            outputChannel,
            kernel,
            inputChannel,
            nativeIndex,
            polyphaseIndex,
            expected,
            actual,
          });
        }
      }
      if (outputChannel % 32 === 31) await yieldToBrowser();
    }
    if (comparedU16Count !== plan.weightElements || mismatchCount !== 0) {
      throw new Error(
        `${topology.label} inverse layout mismatch ${mismatchCount} ` +
          `at ${JSON.stringify(firstMismatch)}`,
      );
    }
    const nativeSha256 = await sha256Bytes(new Uint8Array(
      native.buffer,
      native.byteOffset,
      plan.weightBindingBytes,
    ));
    const polyphaseSha256 = await sha256Bytes(new Uint8Array(
      polyphase.buffer,
      polyphase.byteOffset,
      plan.weightBindingBytes,
    ));
    tracker.unmap(nativeBuffer);
    nativeUnmapped = true;
    tracker.unmap(polyphaseBuffer);
    polyphaseUnmapped = true;
    return Object.freeze({
      native: Object.freeze({
        label: `${topology.label}:native-weight`,
        buffer: nativeBuffer,
        bindingBytes: plan.weightBindingBytes,
        initialSha256: nativeSha256,
      }),
      polyphase: Object.freeze({
        label: `${topology.label}:polyphase-weight`,
        buffer: polyphaseBuffer,
        bindingBytes: plan.weightBindingBytes,
        initialSha256: polyphaseSha256,
      }),
      layout: Object.freeze({
        layoutId: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
        nativeLogicalShape: Object.freeze([
          shape.outputChannels,
          shape.kernelSize,
          shape.inputChannels,
        ]),
        polyphasePhysicalShape: Object.freeze([
          shape.stride,
          2,
          shape.inputChannels,
          shape.outputChannels,
        ]),
        comparedU16Count,
        mismatchCount,
        rawU16Exact: true,
        nativeSha256,
        polyphaseSha256,
        nativeBytes: plan.weightStorageBytes,
        polyphaseBytes: plan.weightStorageBytes,
        residentByteIncrease: 0,
      }),
    });
  } catch (error) {
    if (!nativeUnmapped && nativeBuffer.mapState === "mapped") {
      tracker.unmap(nativeBuffer);
    }
    if (!polyphaseUnmapped && polyphaseBuffer.mapState === "mapped") {
      tracker.unmap(polyphaseBuffer);
    }
    tracker.destroy(nativeBuffer);
    tracker.destroy(polyphaseBuffer);
    throw error;
  }
}

async function createPeriodicSource(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  elements: number,
  bindingBytes: number,
  pattern: Uint16Array,
  shift: number,
): Promise<ImmutableSource> {
  const buffer = tracker.create(device, {
    label,
    size: bindingBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  let unmapped = false;
  try {
    const destination = new Uint16Array(buffer.getMappedRange());
    destination.fill(SOURCE_PADDING_F16);
    fillPeriodicPrefix(destination, shiftedPattern(pattern, shift), elements);
    const initialSha256 = await sha256Bytes(new Uint8Array(
      destination.buffer,
      destination.byteOffset,
      bindingBytes,
    ));
    tracker.unmap(buffer);
    unmapped = true;
    return Object.freeze({ label, buffer, bindingBytes, initialSha256 });
  } catch (error) {
    if (!unmapped && buffer.mapState === "mapped") tracker.unmap(buffer);
    tracker.destroy(buffer);
    throw error;
  }
}

function shiftedPattern(pattern: Uint16Array, shift: number): Uint16Array {
  if (pattern.length === 0) throw new RangeError("OPT-0022 pattern is empty");
  const normalized = ((shift % pattern.length) + pattern.length) %
    pattern.length;
  const shifted = new Uint16Array(pattern.length);
  for (let index = 0; index < pattern.length; index += 1) {
    shifted[index] = pattern[(index + normalized) % pattern.length]!;
  }
  return shifted;
}

function fillPeriodicPrefix(
  destination: Uint16Array,
  pattern: Uint16Array,
  elements: number,
): void {
  if (!Number.isSafeInteger(elements) || elements < 1 ||
    elements > destination.length || pattern.length < 1) {
    throw new RangeError("OPT-0022 periodic upload geometry changed");
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
  ranges: readonly Opt0022SelectedRange[],
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
  ranges: readonly Opt0022SelectedRange[],
  alignment: number,
): GPUBuffer {
  if (!Number.isSafeInteger(alignment) || alignment < RANGE_CONTROL_BYTES ||
    alignment % 4 !== 0) {
    throw new RangeError("OPT-0022 uniform alignment is invalid");
  }
  const buffer = tracker.create(device, {
    label,
    size: ranges.length * alignment,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  const words = new Uint32Array(buffer.getMappedRange());
  for (const [index, range] of ranges.entries()) {
    const offset = index * alignment / Uint32Array.BYTES_PER_ELEMENT;
    words[offset] = range.base;
    words[offset + 1] = range.count;
  }
  tracker.unmap(buffer);
  return buffer;
}

function assertDispatches(
  operation: Opt0022C300Operation,
  range: Opt0022SelectedRange,
  current: EncodableDispatch,
  candidate: EncodableDispatch,
): void {
  const currentPlan = planAceFp16VaeConvTranspose1d(operation.shape);
  const expectedCurrent = planAceFp16VaeConvTranspose1dCongruentRange(
    currentPlan,
    range,
  );
  const candidatePlan = planAceOpt0022VaeConvTranspose1d(operation.shape);
  const expectedCandidate = planAceOpt0022VaeConvTranspose1dRange(
    candidatePlan,
    range,
  );
  if (current.kernelId !==
      ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID ||
    candidate.kernelId !==
      ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID ||
    current.outputRange.base !== expectedCurrent.base ||
    current.outputRange.count !== expectedCurrent.count ||
    current.outputRange.workgroupsX !== expectedCurrent.workgroupsX ||
    current.outputRange.workgroupsY !== expectedCurrent.workgroupsY ||
    current.outputRange.workgroupsZ !== expectedCurrent.workgroupsZ ||
    candidate.outputRange.base !== expectedCandidate.base ||
    candidate.outputRange.count !== expectedCandidate.count ||
    candidate.outputRange.workgroupsX !== expectedCandidate.workgroupsX ||
    candidate.outputRange.workgroupsY !== expectedCandidate.workgroupsY ||
    candidate.outputRange.workgroupsZ !== expectedCandidate.workgroupsZ ||
    current.outputRange.workgroupsX !== candidate.outputRange.workgroupsX ||
    current.outputRange.workgroupsZ !== candidate.outputRange.workgroupsZ ||
    current.outputRange.workgroupsY !==
      candidate.outputRange.workgroupsY * 16) {
    throw new Error(`${operation.label} ${range.stratum} dispatch changed`);
  }
}

async function runOperationCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  topology: Opt0022C300Operation,
  plan: AceFp16VaeConvTranspose1dPlan,
  outputs: Readonly<Record<Opt0022Arm, GuardedOutput>>,
  prefill: GPUBuffer,
  dispatches: Readonly<Record<Opt0022Arm, readonly EncodableDispatch[]>>,
): Promise<Readonly<Record<string, unknown>>> {
  const ranges: Readonly<Record<string, unknown>>[] = [];
  let comparedU16Count = 0;
  for (const [selectedRangeIndex, range] of topology.selectedRanges.entries()) {
    const executions: Readonly<Record<string, unknown>>[] = [];
    for (const [roundIndex, order] of [
      Object.freeze(["A", "B"] as const),
      Object.freeze(["B", "A"] as const),
    ].entries()) {
      const execution = await executeCorrectnessPair(
        device,
        tracker,
        topology.label,
        plan,
        range,
        outputs,
        prefill,
        dispatches,
        order,
        selectedRangeIndex,
        roundIndex,
      );
      executions.push(execution);
      comparedU16Count += range.count;
    }
    const first = executions[0]!;
    const rerun = executions[1]!;
    if (first["aSha256"] !== rerun["aSha256"] ||
      first["bSha256"] !== rerun["bSha256"] ||
      first["aSha256"] !== first["bSha256"] ||
      rerun["aSha256"] !== rerun["bSha256"]) {
      throw new Error(`${topology.label} ${range.stratum} rerun hash changed`);
    }
    ranges.push(Object.freeze({
      stratum: range.stratum,
      rangeIndex: range.rangeIndex,
      base: range.base,
      count: range.count,
      firstOutputRow: range.firstOutputRow,
      outputRowCount: range.outputRowCount,
      weight: range.weight,
      phaseMask: range.phaseMask,
      allPhasesCovered: range.phaseMask ===
        (1 << topology.shape.stride) - 1,
      oneTapOutputRowCount: range.oneTapOutputRowCount,
      twoTapOutputRowCount: range.twoTapOutputRowCount,
      containsFirstOutputRow: range.containsFirstOutputRow,
      containsLastOutputRow: range.containsLastOutputRow,
      outputSha256: first["aSha256"],
      rawU16MismatchCount: 0,
      deterministicRerun: true,
    }));
  }
  return Object.freeze({
    label: topology.label,
    shape: topology.shape,
    outputFrames: plan.outputFrames,
    exactGraphRangeCount: topology.ranges.length,
    selectedProbeCount: topology.selectedRanges.length,
    correctnessExecutionsPerArmPerProbe: 2,
    comparedU16Count,
    rawU16MismatchCount: 0,
    completeSelectedWrites: true,
    qNaNPrefillCompleteWrites: true,
    guardsAndAdjacentCanariesUntouched: true,
    deterministicRerun: true,
    allPhasesCoveredInEveryProbe: true,
    firstAndLastOneTapBoundariesCovered: true,
    ranges: Object.freeze(ranges),
  });
}

async function executeCorrectnessPair(
  device: GPUDevice,
  tracker: BufferTracker,
  operationLabel: string,
  plan: AceFp16VaeConvTranspose1dPlan,
  range: Opt0022SelectedRange,
  outputs: Readonly<Record<Opt0022Arm, GuardedOutput>>,
  prefill: GPUBuffer,
  dispatches: Readonly<Record<Opt0022Arm, readonly EncodableDispatch[]>>,
  order: readonly Opt0022Arm[],
  selectedRangeIndex: number,
  roundIndex: number,
): Promise<Readonly<Record<string, unknown>>> {
  const selectedBytes = range.count * FLOAT16_BYTES;
  const encoder = device.createCommandEncoder({
    label: `${operationLabel}-${range.stratum}-exactness-${roundIndex}`,
  });
  for (const arm of ["A", "B"] as const) {
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
    order: order.join(""),
    commandBufferCount: 1,
    dispatchCount: 2,
    matchingQueueCompletionFenceCount: 1,
    comparedU16Count: range.count,
    rawU16MismatchCount: 0,
    aSha256: readback.A.sha256,
    bSha256: readback.B.sha256,
  });
}

async function readAndCompareRange(
  device: GPUDevice,
  tracker: BufferTracker,
  operationLabel: string,
  plan: AceFp16VaeConvTranspose1dPlan,
  range: Opt0022SelectedRange,
  outputs: Readonly<Record<Opt0022Arm, GuardedOutput>>,
  roundIndex: number,
): Promise<Readonly<Record<Opt0022Arm, ReadbackArm>>> {
  const selectedBytes = range.count * FLOAT16_BYTES;
  const payloadOffset = range.base * FLOAT16_BYTES;
  const beforeBytes = Math.min(OUTPUT_GUARD_BYTES, payloadOffset);
  const afterBytes = Math.min(
    OUTPUT_GUARD_BYTES,
    plan.outputBindingBytes - payloadOffset - selectedBytes,
  );
  const armBytes = OUTPUT_GUARD_BYTES * 2 + selectedBytes + beforeBytes +
    afterBytes;
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
    for (const [armIndex, arm] of (["A", "B"] as const).entries()) {
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
    await tracker.mapRead(readback);
    mapped = true;
    const mappedRange = readback.getMappedRange();
    const arms = {} as Record<Opt0022Arm, ReadbackArm>;
    for (const [armIndex, arm] of (["A", "B"] as const).entries()) {
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
      let qNaNPrefillCount = 0;
      for (const bits of selected) {
        if ((bits & 0x7c00) === 0x7c00) nonFiniteCount += 1;
        if (bits === OUTPUT_PREFILL_QNAN_F16) qNaNPrefillCount += 1;
      }
      const prefixGuardIntact = everyBit(prefix, OUTPUT_GUARD_F16);
      const suffixGuardIntact = everyBit(suffix, OUTPUT_GUARD_F16);
      const adjacentCanariesIntact = everyBit(before, OUTPUT_CANARY_F16) &&
        everyBit(after, OUTPUT_CANARY_F16);
      if (nonFiniteCount !== 0 || qNaNPrefillCount !== 0 ||
        !prefixGuardIntact || !suffixGuardIntact ||
        !adjacentCanariesIntact) {
        throw new Error(
          `${operationLabel} ${range.stratum} ${arm} write/guard scan failed`,
        );
      }
      const detached = selected.slice();
      arms[arm] = Object.freeze({
        sha256: await sha256Bytes(new Uint8Array(detached.buffer)),
        bits: detached,
        nonFiniteCount,
        qNaNPrefillCount,
        prefixGuardIntact,
        suffixGuardIntact,
        adjacentCanariesIntact,
      });
    }
    let mismatchCount = 0;
    let firstMismatchIndex: number | null = null;
    for (let index = 0; index < arms.A.bits.length; index += 1) {
      if (arms.A.bits[index] === arms.B.bits[index]) continue;
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
    if (mapped) tracker.unmap(readback);
    tracker.destroy(readback);
  }
}

async function verifyAllSourcesImmutable(
  device: GPUDevice,
  tracker: BufferTracker,
  operations: readonly PreparedOperation[],
  updateProgress: (message: string) => void,
  phase: "post-correctness" | "post-timing",
): Promise<Readonly<Record<string, unknown>>> {
  const cases: Readonly<Record<string, unknown>>[] = [];
  let sourceCount = 0;
  let comparedBytes = 0;
  for (const [operationIndex, operation] of operations.entries()) {
    updateProgress(
      `${phase} source integrity ${operationIndex + 1}/${operations.length}`,
    );
    const sources: Readonly<Record<string, unknown>>[] = [];
    for (const source of operation.sources) {
      const actualSha256 = await hashGpuBuffer(
        device,
        tracker,
        source.buffer,
        source.bindingBytes,
        `${source.label}-${phase}`,
      );
      if (actualSha256 !== source.initialSha256) {
        throw new Error(`${source.label} mutated during ${phase}`);
      }
      sourceCount += 1;
      comparedBytes += source.bindingBytes;
      sources.push(Object.freeze({
        label: source.label,
        bindingBytes: source.bindingBytes,
        sha256: actualSha256,
        unchanged: true,
      }));
    }
    cases.push(Object.freeze({
      label: operation.topology.label,
      sources: Object.freeze(sources),
      passed: true,
    }));
    await yieldToBrowser();
  }
  return Object.freeze({
    phase,
    sourceCount,
    comparedBytes,
    allInputNativePolyphaseWeightAndBiasSourcesUnchanged: true,
    cases: Object.freeze(cases),
    passed: true,
  });
}

async function hashGpuBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  source: GPUBuffer,
  bytes: number,
  label: string,
): Promise<string> {
  if (!Number.isSafeInteger(bytes) || bytes < 4 || bytes % 4 !== 0) {
    throw new RangeError("OPT-0022 immutable-source byte count changed");
  }
  const readback = tracker.create(device, {
    label: `${label}-readback`,
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await tracker.mapRead(readback);
    mapped = true;
    return await sha256Bytes(new Uint8Array(readback.getMappedRange()));
  } finally {
    if (mapped) tracker.unmap(readback);
    tracker.destroy(readback);
  }
}

async function runTimedGate(
  prepared: PreparedGate,
): Promise<Readonly<Record<string, unknown>>> {
  const launchedAtEpochMilliseconds = Date.now();
  const thermal = parseOpt0022ThermalGate(
    collectThermalParameters(),
    prepared.preparedCompletedAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const probes = prepared.operations.flatMap((operation, operationIndex) =>
    operation.topology.selectedRanges.map((range, selectedRangeIndex) => ({
      operation,
      operationIndex,
      range,
      selectedRangeIndex,
    }))
  );
  if (probes.length !== SELECTED_PROBE_COUNT) {
    throw new Error("OPT-0022 timing probe count changed");
  }
  const inputs = probes.map(({ operation, range }) => ({
    operationLabel: operation.topology.label,
    stratum: range.stratum,
    weight: range.weight,
    rounds: [] as Opt0022TimingRoundInput[],
  }));
  let dispatchSampleCount = 0;
  const timingStartedAtEpochMilliseconds = Date.now();
  for (const entry of buildOpt0022TimingOrders()) {
    const probe = probes[entry.probeOrdinal];
    const input = inputs[entry.probeOrdinal];
    if (probe === undefined || input === undefined) {
      throw new Error("OPT-0022 rotated timing topology changed");
    }
    prepared.updateProgress(
      `timing round ${entry.roundIndex + 1}/6, probe ` +
        `${entry.probeOrdinal + 1}/15`,
    );
    let aMilliseconds = 0;
    let bMilliseconds = 0;
    for (const arm of entry.order) {
      const sample = await executeTimedDispatch(
        prepared.device,
        probe.operation.dispatches[arm][probe.selectedRangeIndex]!,
      );
      dispatchSampleCount += 1;
      if (arm === "A") aMilliseconds = sample.wallMilliseconds;
      else bMilliseconds = sample.wallMilliseconds;
    }
    input.rounds.push(Object.freeze({
      roundIndex: entry.roundIndex,
      order: entry.order.join("") as "AB" | "BA",
      aMilliseconds,
      bMilliseconds,
    }));
    await yieldToBrowser();
  }
  const timingCompletedAtEpochMilliseconds = Date.now();
  await prepared.device.queue.onSubmittedWorkDone();
  await settlePostDrainEvents();
  prepared.updateProgress("verifying all source buffers immutable after timing");
  const immutableAfterTiming = await verifyAllSourcesImmutable(
    prepared.device,
    prepared.tracker,
    prepared.operations,
    prepared.updateProgress,
    "post-timing",
  );
  await prepared.device.queue.onSubmittedWorkDone();
  await settlePostDrainEvents();
  if (prepared.uncapturedErrors.length !== 0 ||
    immutableAfterTiming["passed"] !== true) {
    throw new Error("OPT-0022 timing integrity gate failed");
  }
  const timing = summarizeOpt0022Timing(inputs.map((input) => Object.freeze({
    operationLabel: input.operationLabel,
    stratum: input.stratum,
    weight: input.weight,
    rounds: Object.freeze(input.rounds),
  })));
  const environment = environmentReceipt(prepared.adapter, prepared.device);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanupStartedAtEpochMilliseconds = Date.now();
  const firstCleanup = prepared.destroy();
  const secondCleanup = prepared.destroy();
  const cleanupCompletedAtEpochMilliseconds = Date.now();
  const cleanup = Object.freeze({
    startedAtEpochMilliseconds: cleanupStartedAtEpochMilliseconds,
    completedAtEpochMilliseconds: cleanupCompletedAtEpochMilliseconds,
    firstCall: firstCleanup,
    secondCall: secondCleanup,
    idempotent: true,
    drainBeforeRelease: firstCleanup["drainBeforeRelease"] === true,
    balancedMaps: firstCleanup["mapsBalanced"] === true &&
      secondCleanup["mapsBalanced"] === true,
    zeroLiveResources: firstCleanup["liveBufferCount"] === 0 &&
      firstCleanup["liveBytes"] === 0 &&
      secondCleanup["liveBufferCount"] === 0 &&
      secondCleanup["liveBytes"] === 0,
    deviceDestroyed: firstCleanup["deviceDestroyed"] === true,
  });
  if (!cleanup.drainBeforeRelease || !cleanup.balancedMaps ||
    !cleanup.zeroLiveResources || !cleanup.deviceDestroyed) {
    throw new Error("OPT-0022 cleanup gate failed");
  }
  return Object.freeze({
    schema: "ace-opt-0022-vae-conv-transpose-subgroup-polyphase-ab-v1",
    status: "passed",
    experimentId: EXPERIMENT_ID,
    classification: "bounded-exact-primitive-decision-gate-not-integrated",
    recordedAt: new Date().toISOString(),
    identity: prepared.sourceAuthority,
    environment,
    preparation: prepared.preparation,
    protocol: Object.freeze({
      thermal,
      fullLayoutAndOutputCorrectnessCompletedBeforeThermalGate: true,
      bothKernelsCompiledAndWarmedBeforeThermalGate: true,
      correctnessOrders: Object.freeze(["AB", "BA"]),
      timingRounds: TIMING_ROUNDS,
      timingOrder:
        "six alternating AB/BA rounds; 15-probe order rotated left by round",
      sampleTopology:
        "one dispatch in one command buffer followed by its matching queue-completion fence",
      authoritativeTiming:
        "performance.now immediately before submit through matching queue drain",
      continuousExternalThermalTraceRequiredThroughCleanup: true,
      thermalTraceRequiredThroughEpochMilliseconds:
        cleanupCompletedAtEpochMilliseconds,
      oneThirtySecondNominalGate: true,
      unchangedThermalRetryPerformed: false,
    }),
    topology: compactTopology(prepared.topology),
    layout: Object.freeze({
      layoutId: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
      operationCount: prepared.operations.length,
      comparedU16Count: TOTAL_WEIGHT_U16_COUNT,
      mismatchCount: 0,
      residentByteIncrease: 0,
      cases: Object.freeze(prepared.operations.map((operation) =>
        operation.layout
      )),
    }),
    correctness: prepared.correctness,
    immutableAfterTiming,
    timing: Object.freeze({
      ...timing,
      timingStartedAtEpochMilliseconds,
      timingCompletedAtEpochMilliseconds,
      dispatchSampleCount,
      caveat:
        "Weighted separately drained primitive ranges only; no converter, package, decoder, waveform, listening, or product claim.",
    }),
    decision: Object.freeze({
      disposition: timing["decision"],
      packageNativeLayerGateAuthorized: timing["passed"] === true,
      converterAndPackageWorkAuthorized: timing["passed"] === true,
      productionIntegrationAuthorized: false,
      c300ProductionRunAuthorized: false,
      longWindowRunAuthorized: false,
      waveformOrListeningRunAuthorized: false,
      m2250ProductRunAuthorized: false,
    }),
    memory: memoryBeforeCleanup,
    cleanup,
  });
}

async function executeTimedDispatch(
  device: GPUDevice,
  dispatch: EncodableDispatch,
): Promise<Readonly<{
  wallMilliseconds: number;
  commandBufferCount: 1;
  dispatchCount: 1;
  matchingQueueCompletionFenceCount: 1;
}>> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  const command = encoder.finish();
  const started = performance.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  const wallMilliseconds = performance.now() - started;
  return Object.freeze({
    wallMilliseconds,
    commandBufferCount: 1,
    dispatchCount: 1,
    matchingQueueCompletionFenceCount: 1,
  });
}

async function buildSourceAuthority(
  topology: readonly Opt0022C300Operation[],
): Promise<Readonly<Record<string, unknown>>> {
  const currentCoreSourceSha256 = await sha256Text(currentCoreSource);
  const candidateCoreSourceSha256 = await sha256Text(candidateCoreSource);
  if (currentCoreSourceSha256 !== CURRENT_CORE_SOURCE_SHA256 ||
    candidateCoreSourceSha256 !== CANDIDATE_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0022 rejected unauthenticated kernel source bytes");
  }
  const generatedShaders = [];
  for (const operation of topology) {
    generatedShaders.push(Object.freeze({
      label: operation.label,
      A: await sha256Text(
        aceFp16VaeCongruentConvTranspose1dWgsl(operation.shape),
      ),
      B: await sha256Text(
        aceOpt0022VaeConvTranspose1dSubgroupWgsl(operation.shape),
      ),
    }));
  }
  if (JSON.stringify(generatedShaders) !==
    JSON.stringify(EXPECTED_GENERATED_SHADER_SHA256)) {
    throw new Error("OPT-0022 rejected unauthenticated generated shaders");
  }
  return Object.freeze({
    registrationCommit: REGISTRATION_COMMIT,
    integratedCurrentCommit: INTEGRATED_CURRENT_COMMIT,
    currentCore: Object.freeze({
      bytes: currentCoreSource.length,
      sha256: currentCoreSourceSha256,
      kernelId: ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID,
      weightLayout: "native-output-kernel-input-f16",
    }),
    candidateCore: Object.freeze({
      bytes: candidateCoreSource.length,
      sha256: candidateCoreSourceSha256,
      kernelId: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID,
      weightLayout: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
    }),
    fixtureAndPackageProvenance: Object.freeze({
      fixtureManifestSha256: FIXTURE_MANIFEST_SHA256,
      modelManifestSha256: MODEL_MANIFEST_SHA256,
      vaeManifestSha256: VAE_MANIFEST_SHA256,
      fixtureClassification:
        "deterministic finite synthetic benchmark fixture derived from the authenticated C300 topology; no package weights loaded",
    }),
    generatedShaders: Object.freeze(generatedShaders),
  });
}

function requiredDeviceLimits(
  adapter: GPUAdapter,
  topology: readonly Opt0022C300Operation[],
): Record<string, number> {
  let maximumBuffer = 1;
  let maximumStorageBinding = 1;
  let maximumWorkgroupStorage = 1;
  let maximumDispatch = 1;
  for (const operation of topology) {
    const currentPlan = planAceFp16VaeConvTranspose1d(operation.shape);
    const candidatePlan = planAceOpt0022VaeConvTranspose1d(operation.shape);
    maximumStorageBinding = Math.max(
      maximumStorageBinding,
      currentPlan.inputBindingBytes,
      currentPlan.weightBindingBytes,
      currentPlan.biasBindingBytes,
      currentPlan.outputBindingBytes,
      candidatePlan.inputBindingBytes,
      candidatePlan.polyphaseWeightBindingBytes,
      candidatePlan.biasBindingBytes,
      candidatePlan.outputBindingBytes,
    );
    maximumBuffer = Math.max(
      maximumBuffer,
      currentPlan.inputBindingBytes,
      currentPlan.weightBindingBytes,
      OUTPUT_GUARD_BYTES + currentPlan.outputBindingBytes +
        OUTPUT_GUARD_BYTES,
    );
    maximumWorkgroupStorage = Math.max(
      maximumWorkgroupStorage,
      currentPlan.workgroupStorageBytes,
      candidatePlan.workgroupStorageBytes,
    );
    for (const range of operation.selectedRanges) {
      const A = planAceFp16VaeConvTranspose1dCongruentRange(
        currentPlan,
        range,
      );
      const B = planAceOpt0022VaeConvTranspose1dRange(candidatePlan, range);
      maximumDispatch = Math.max(
        maximumDispatch,
        A.workgroupsX,
        A.workgroupsY,
        A.workgroupsZ,
        B.workgroupsX,
        B.workgroupsY,
        B.workgroupsZ,
      );
    }
  }
  const requested = {
    maxBufferSize: maximumBuffer,
    maxStorageBufferBindingSize: maximumStorageBinding,
    maxUniformBufferBindingSize: RANGE_CONTROL_BYTES,
    maxComputeWorkgroupStorageSize: maximumWorkgroupStorage,
    maxComputeInvocationsPerWorkgroup: 128,
    maxComputeWorkgroupSizeX: 128,
    maxComputeWorkgroupSizeY: 8,
    maxComputeWorkgroupsPerDimension: maximumDispatch,
  };
  for (const [name, minimum] of Object.entries(requested)) {
    const actual = Number(adapter.limits[name as keyof GPUSupportedLimits]);
    if (!Number.isFinite(actual) || actual < minimum) {
      throw new RangeError(
        `OPT-0022 adapter ${name}=${actual} is below ${minimum}`,
      );
    }
  }
  return requested;
}

function requireAdapter(
  adapter: GPUAdapter,
  topology: readonly Opt0022C300Operation[],
): void {
  if (!adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups")) {
    throw new Error("OPT-0022 requires shader-f16 and subgroups");
  }
  if (adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32) {
    throw new Error(
      "OPT-0022 requires authenticated subgroup min/max exactly 32/32",
    );
  }
  if (adapter.limits.minStorageBufferOffsetAlignment > OUTPUT_GUARD_BYTES) {
    throw new Error("OPT-0022 output guard is below storage alignment");
  }
  requiredDeviceLimits(adapter, topology);
}

function environmentReceipt(adapter: GPUAdapter, device: GPUDevice) {
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
    adapterFeatures: Object.freeze([...adapter.features].sort()),
    deviceFeatures: Object.freeze([...device.features].sort()),
    deviceLimits: Object.freeze({
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize:
        device.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: device.limits.maxComputeWorkgroupSizeY,
      maxComputeWorkgroupsPerDimension:
        device.limits.maxComputeWorkgroupsPerDimension,
      minStorageBufferOffsetAlignment:
        device.limits.minStorageBufferOffsetAlignment,
      minUniformBufferOffsetAlignment:
        device.limits.minUniformBufferOffsetAlignment,
    }),
  });
}

function compactTopology(
  topology: readonly Opt0022C300Operation[],
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    decoderInputFrames: C300_INPUT_FRAMES,
    operationCount: topology.length,
    exactRangeCount: C300_EXACT_RANGE_COUNT,
    selectedProbeCount: SELECTED_PROBE_COUNT,
    operations: Object.freeze(topology.map((operation) => Object.freeze({
      label: operation.label,
      shape: operation.shape,
      outputFrames: operation.outputFrames,
      outputElements: operation.outputElements,
      exactRangeCount: operation.ranges.length,
      selectedRanges: Object.freeze(operation.selectedRanges.map((range) =>
        Object.freeze({
          stratum: range.stratum,
          rangeIndex: range.rangeIndex,
          base: range.base,
          count: range.count,
          weight: range.weight,
          phaseMask: range.phaseMask,
          oneTapOutputRowCount: range.oneTapOutputRowCount,
          twoTapOutputRowCount: range.twoTapOutputRowCount,
          containsFirstOutputRow: range.containsFirstOutputRow,
          containsLastOutputRow: range.containsLastOutputRow,
        })
      )),
    }))),
  });
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

function requireCoordinate(value: number, extent: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= extent) {
    throw new RangeError(`OPT-0022 ${label} ${value} is out of bounds`);
  }
}

function gcd(left: number, right: number): number {
  if (!Number.isSafeInteger(left) || left < 1 ||
    !Number.isSafeInteger(right) || right < 1) {
    throw new RangeError("OPT-0022 gcd inputs must be positive safe integers");
  }
  let a = left;
  let b = right;
  while (b !== 0) {
    const remainder = a % b;
    a = b;
    b = remainder;
  }
  return a;
}

function everyBit(values: Uint16Array, expected: number): boolean {
  for (const value of values) if (value !== expected) return false;
  return true;
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function finitePositive(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

function medianExact(samples: readonly number[], expectedLength: number): number {
  if (samples.length !== expectedLength || !samples.every(finitePositive)) {
    throw new Error(
      `OPT-0022 requires ${expectedLength} finite positive samples`,
    );
  }
  const sorted = [...samples].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
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
    throw new Error(`OPT-0022 thermal field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0022 thermal field ${name} is not finite`);
  }
  return value;
}

async function sha256Text(value: string): Promise<string> {
  return sha256Bytes(new TextEncoder().encode(value));
}

async function sha256Bytes(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", Uint8Array.from(bytes));
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function requireElement<ElementType extends Element>(selector: string): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing OPT-0022 element ${selector}`);
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
    __ACE_OPT0022_RESULT__?: Readonly<Record<string, unknown>>;
  }).__ACE_OPT0022_RESULT__ = receipt;
}

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0022-vae-conv-transpose-subgroup-polyphase-ab-v1",
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
