/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import fixed32CoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.ts?raw";
import candidateCoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.ts?raw";
import conv1dCoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16.ts?raw";
import decoderCoreSource from "../../src/webgpu/vae-decoder.ts?raw";
import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
  AceFp16VaeConv1dSubgroupKernel,
  aceFp16VaeConv1dSubgroupWgsl,
  planAceFp16VaeConv1dSubgroupRange,
} from "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import {
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
  AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
  aceOpt0024VaeConv1dDirectDot4SubgroupWgsl,
  planAceOpt0024VaeConv1dDirectDot4SubgroupRange,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.js";
import {
  planAceFp16VaeConv1d,
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

export type Opt0024Arm = "shipped" | "candidate";
export type Opt0024ProbeId = "first" | "interior" | "tail";
export type Opt0024TierId = "c1024" | "c512" | "c256" | "c128";

interface ExactRange {
  readonly rangeIndex: number;
  readonly base: number;
  readonly count: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
}

export interface Opt0024Probe {
  readonly id: Opt0024ProbeId;
  readonly source: "graph" | "synthetic-interior";
  readonly rangeIndex: number | null;
  readonly base: number;
  readonly count: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
}

export interface Opt0024Operation {
  readonly ordinal: number;
  readonly operationIndex: number;
  readonly label: string;
  readonly shape: AceVaeConv1dShape;
  readonly outputElements: number;
  readonly ranges: readonly ExactRange[];
  readonly probes: readonly Opt0024Probe[];
}

export interface Opt0024Representative {
  readonly id: Opt0024TierId;
  readonly operation: Opt0024Operation;
  readonly weight: number;
  readonly probes: readonly Opt0024Probe[];
}

export interface Opt0024TimingInput {
  readonly id: Opt0024TierId;
  readonly operationLabel: string;
  readonly weight: number;
  readonly samples: Readonly<Record<Opt0024Arm, readonly number[]>>;
}

export interface Opt0024ThermalGate {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly command: "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly traceStartedAtEpochMilliseconds: number;
  readonly traceStartObservationIndex: number;
  readonly gateStartedAtEpochMilliseconds: number;
  readonly gateStartObservationIndex: number;
  readonly gateCompletedAtEpochMilliseconds: number;
  readonly gateCompletedObservationIndex: number;
  readonly observationCount: number;
  readonly pollMilliseconds: 1_000;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly missingObservationCount: 0;
  readonly launchDelayMilliseconds: number;
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

interface PreparedOperation {
  readonly topology: Opt0024Operation;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly weight: GPUBuffer;
  readonly dispatches: Readonly<Record<
    Opt0024ProbeId,
    Readonly<Record<Opt0024Arm, EncodableDispatch>>
  >>;
}

interface SharedResources {
  readonly input: GPUBuffer;
  readonly bias: GPUBuffer;
  readonly outputs: Readonly<Record<Opt0024Arm, GPUBuffer>>;
  readonly prefill: GPUBuffer;
  readonly controls: GPUBuffer;
  readonly controlOffsets: ReadonlyMap<string, number>;
  readonly maximumOutputBindingBytes: number;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly shippedKernel: AceFp16VaeConv1dSubgroupKernel;
  readonly candidateKernel: AceOpt0024VaeConv1dDirectDot4SubgroupKernel;
  readonly topology: readonly Opt0024Operation[];
  readonly operations: readonly PreparedOperation[];
  readonly representatives: readonly Readonly<{
    representative: Opt0024Representative;
    prepared: PreparedOperation;
  }>[];
  readonly shared: SharedResources;
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly sourceAuthority: Readonly<Record<string, unknown>>;
  readonly preparation: Readonly<Record<string, unknown>>;
  readonly uncapturedErrors: readonly string[];
  readonly deviceLosses: readonly string[];
  readonly preparedCompletedAtEpochMilliseconds: number;
  readonly updateProgress: (message: string) => void;
  cleanup(): Promise<Readonly<Record<string, unknown>>>;
}

interface NumericalLocation {
  readonly executionIndex: number;
  readonly localIndex: number;
  readonly outputIndex: number;
  readonly outputRow: number;
  readonly outputChannel: number;
  readonly controlBits: number;
  readonly candidateBits: number;
  readonly control: number;
  readonly candidate: number;
  readonly absoluteError: number;
}

interface NumericalAccumulator {
  count: number;
  rawU16MismatchCount: number;
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
  signedZeroDifferenceCount: number;
  ulpDistribution: Map<number, number>;
  firstDifference: NumericalLocation | null;
  worst: NumericalLocation | null;
}

interface ReadbackArm {
  readonly words: Uint16Array;
  readonly sha256: string;
  readonly nonFiniteCount: number;
  readonly prefillQNaNCount: number;
  readonly guardsUntouched: boolean;
}

class Opt0024GateFailure extends Error {
  constructor(
    message: string,
    readonly gateReceipt: Readonly<Record<string, unknown>>,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "Opt0024GateFailure";
  }
}

const EXPERIMENT_ID = "OPT-0024" as const;
const REGISTRATION_COMMIT =
  "51ced73b7f2078ade30d409afcd7fdba7082f171";
const REGISTRATION_RECORD_SHA256 =
  "64ef45495d2f217aa220d0dde5d06497b828fd33a619817e35180b48813c2d5c";
const FIXED32_CORE_SOURCE_SHA256 =
  "7d218516d6b2c8d6e3332a53101be5fdeae1142096c442433915bfa58941ce32";
const CANDIDATE_CORE_SOURCE_SHA256 =
  "fe3bf8110cef1a3bb791006e9d376fe549e9f00fe30e4738d7429cb0daf65841";
const CONV1D_CORE_SOURCE_SHA256 =
  "fd14f625e3efeba3277bd9c4e8aa052af92a2b44c078108303173c9bb42a4310";
const DECODER_CORE_SOURCE_SHA256 =
  "07f294e2aadd615c0a8b840884f43205bc00c146362f54048a39a85440da1d3e";
const C300_INPUT_FRAMES = 300;
const C300_OPERATION_COUNT = 16;
const C300_EXACT_RANGE_COUNT = 2_399;
const C300_PROBE_COUNT = 48;
const C300_UNIQUE_PROBE_U16_COUNT = 6_299_648;
const C300_CANDIDATE_CONTROL_COMPARISON_COUNT = 12_599_296;
const TIMING_WEIGHT_TOTAL = 2_397;
const TIMING_SAMPLES_PER_ARM_TIER = 4;
const TIMED_COMPOSITE_DISPATCHES = 3;
const TIMED_SUBMIT_DRAIN_EVENTS = 32;
const TIMED_DISPATCH_COUNT = 96;
const C128_SPEEDUP_THRESHOLD = 1.20;
const WEIGHTED_SPEEDUP_THRESHOLD = 1.25;
const NRMSE_LIMIT = 0.001;
const SNR_MINIMUM_DB = 60;
const PEARSON_MINIMUM = 0.99999;
const RELATIVE_MAXIMUM_ABSOLUTE_ERROR_LIMIT = 0.01;
const RANGE_CONTROL_BYTES = 16;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const ADJACENT_CANARY_U32 = 0x5aa5_3cc3;
const OUTPUT_PREFILL_QNAN_F16 = 0x7e55;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
const THERMAL_TRACE_SCHEMA =
  "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1" as const;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const THERMAL_POLL_MILLISECONDS = 1_000;
const MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS = 1_250;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 5_000;

const ARMS = Object.freeze(["shipped", "candidate"] as const);
const CORRECTNESS_ORDERS = Object.freeze([
  Object.freeze(["shipped", "candidate"] as const),
  Object.freeze(["candidate", "shipped"] as const),
]);
const TIMING_PAIR_ORDERS = Object.freeze([
  Object.freeze(["shipped", "candidate"] as const),
  Object.freeze(["candidate", "shipped"] as const),
  Object.freeze(["candidate", "shipped"] as const),
  Object.freeze(["shipped", "candidate"] as const),
]);
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
const REPRESENTATIVE_SPECS = Object.freeze([
  Object.freeze({
    id: "c1024" as const,
    label: "block-0-res-1-conv1",
    weight: 282,
  }),
  Object.freeze({
    id: "c512" as const,
    label: "block-1-res-2-conv1",
    weight: 423,
  }),
  Object.freeze({
    id: "c256" as const,
    label: "block-2-res-1-conv1",
    weight: 423,
  }),
  Object.freeze({
    id: "c128" as const,
    label: "block-4-res-3-conv1",
    weight: 1_269,
  }),
]);

export const OPT_0024_EXPECTED_GENERATED_SHADER_SHA256 = Object.freeze([
  Object.freeze({
    label: "conv1",
    shipped: "fc5cc0fb415d66b0534396b92dafbb30b70cfedaed180bd5f20b919704fc2608",
    candidate: "8f4f5c3ca8a793fbb769c7b9af4f6e4ce148d94be4da21105b0d1c03758d8493",
  }),
  Object.freeze({
    label: "block-0-res-1-conv1",
    shipped: "d1ab226777b3813de4d432a9c95d4c25dab0cacb3e1e1b0c02c2c9ef51484780",
    candidate: "2038e1c7a740718ec45303422c1860f641a9a2e8ac362490a57d2b29a1779a26",
  }),
  Object.freeze({
    label: "block-0-res-2-conv1",
    shipped: "10ac855d884b23bcb0b340e740e72df4deaa57951aec91a6ac46ecf7df9a77e2",
    candidate: "e6ded7c797e0935e5dacfb86b84fc492957308513bc7821423c3ffc2f4dc86db",
  }),
  Object.freeze({
    label: "block-0-res-3-conv1",
    shipped: "12631c9722750a1f171fd51b3daab17eb5bfdbc57a4b6cf22fdd48df5b26fe9b",
    candidate: "b2f396034dd838fe6276f85d9965c5a4cfe2cd677e9685a2e2d98ae8ec8358e9",
  }),
  Object.freeze({
    label: "block-1-res-1-conv1",
    shipped: "53bdb272f31dbaf940680f87eeea993978a85f835119cbe1ea4a715b5eda1635",
    candidate: "3c9e65e4a253ed1db90a2d5e9102a3f293a297cf34e2813e7e86a99ed5d2ec89",
  }),
  Object.freeze({
    label: "block-1-res-2-conv1",
    shipped: "a576cef3c69a7557e6c7ef92c356e6ab92e63a46bf0aa28790476096b8b0b8ca",
    candidate: "6b0c5d93ca5aa1e80c07f21a484ae7ba1835811252e73168395ae43a0436ec34",
  }),
  Object.freeze({
    label: "block-1-res-3-conv1",
    shipped: "44eab03a664f6fcf4c3e51a1ec6e139c1ea3c7b0c754396e1fd9ea246accfb49",
    candidate: "5e63a4f42c0bb198443761780ac64710943b154e55f3105b29ffdf4fe68be091",
  }),
  Object.freeze({
    label: "block-2-res-1-conv1",
    shipped: "849fa502c6e1d06d0358f042ee7d51fd113bc5003eb867908aa266e5066e575b",
    candidate: "266365b9750c5c15a483da2c048b8f2b307e6165d9c565ac9f36c3cc03fe3cd7",
  }),
  Object.freeze({
    label: "block-2-res-2-conv1",
    shipped: "be8694bd30950d97083528081e63beac979ff97a3eae893c9f3aeb6143575a3a",
    candidate: "ceeb75b0591328c480e5611d2b0130630a2cd4d25fa8310ef5407d3bf2d2b8df",
  }),
  Object.freeze({
    label: "block-2-res-3-conv1",
    shipped: "f8cb5ed316a0d761e2f786dfc1a75da382c18318f5754292a96141fbf0dd3d81",
    candidate: "093d88c37ffbade8a8196aeaad2e9d987532842fa56595e766fdbee6db817f39",
  }),
  Object.freeze({
    label: "block-3-res-1-conv1",
    shipped: "8736c46ced66d150882713840d8e8d84e133f95eb87ad11c816d304aad122c80",
    candidate: "c65f9ee69450f4b0be31c69fe91314b467beea57b7cb8aac6c2a9660e9a1c37f",
  }),
  Object.freeze({
    label: "block-3-res-2-conv1",
    shipped: "47b07fb1a8e239aeac6f05df6258481b8848caee740809765dc7cfa016e46bba",
    candidate: "15c0f79c6738bf02b7c26f6e9fc76967182a292832dced0be6229b5bcde05009",
  }),
  Object.freeze({
    label: "block-3-res-3-conv1",
    shipped: "d02c2387bf09200dc8bf33e994e9de4e0fdbcf410c6de23b5c3a19e51311b6b3",
    candidate: "0027d7ee7f112e0bad835cc3e9ef813428adef8d105808fb0847f899b15d8b68",
  }),
  Object.freeze({
    label: "block-4-res-1-conv1",
    shipped: "6dd4ac75d87a6576802b1d46e04e4aede701b14ed2b9a37d6d89e7128f79eee6",
    candidate: "7c252c116990ccb9cd08681ecfcd8e5a98bc496eb846a6866d78b67578877de1",
  }),
  Object.freeze({
    label: "block-4-res-2-conv1",
    shipped: "606e5eb84110cdfbb1ff921e04783d495c87c516fd1596ac86a4d0a4608e95b5",
    candidate: "5205c7a4fad114af61c75b7e3ceff317fe390693d0348295fe0d453484504638",
  }),
  Object.freeze({
    label: "block-4-res-3-conv1",
    shipped: "400eaebc07fdbaa239525d4146801065d1964b24d6b2b0464b6c6f9c0b43fd19",
    candidate: "f9d91c63fb587583f16213b2b2f9d285acefc2b022a7efd42421feedcc0e3082",
  }),
] as const);

export function buildOpt0024Topology(
  inputFrames = C300_INPUT_FRAMES,
): readonly Opt0024Operation[] {
  if (!Number.isSafeInteger(inputFrames) || inputFrames < 1) {
    throw new RangeError("OPT-0024 input frames must be a positive integer");
  }
  const graph = planAceVaeDecoder(inputFrames);
  const quanta = planAceVaeDecoderQuanta(graph);
  const operations = graph.operations
    .map((operation, operationIndex) => ({ operation, operationIndex }))
    .filter((entry): entry is Readonly<{
      operation: AceVaeDecoderConvOperation;
      operationIndex: number;
    }> => entry.operation.kind === "conv1d" &&
      entry.operation.shape.kernelSize === 7 &&
      entry.operation.bias !== undefined)
    .map(({ operation, operationIndex }, ordinal) => {
      const plan = planAceFp16VaeConv1d(operation.shape, "float16");
      if (plan.inputChannels % 4 !== 0 ||
        plan.outputChannels % 128 !== 0) {
        throw new Error(`${operation.label} left OPT-0024 shape boundary`);
      }
      const ranges = quanta.quanta
        .filter((quantum) => quantum.operationIndex === operationIndex)
        .map((quantum, rangeIndex): ExactRange => {
          if (quantum.operationKind !== "conv1d" ||
            quantum.primitives.length !== 1) {
            throw new Error(`${operation.label} quantum topology changed`);
          }
          const primitive = quantum.primitives[0]!;
          if (primitive.firstOutputChannel !== 0 ||
            primitive.outputChannels !== plan.outputChannels ||
            primitive.outputBase !== quantum.logicalOutputBase ||
            primitive.outputCount !== quantum.logicalOutputCount ||
            primitive.outputBase % plan.outputChannels !== 0 ||
            primitive.outputCount % plan.outputChannels !== 0) {
            throw new Error(`${operation.label} primitive topology changed`);
          }
          return Object.freeze({
            rangeIndex,
            base: primitive.outputBase,
            count: primitive.outputCount,
            firstOutputRow: primitive.outputBase / plan.outputChannels,
            outputRowCount: primitive.outputCount / plan.outputChannels,
          });
        });
      const probes = buildProbes(operation.label, plan, ranges);
      return Object.freeze({
        ordinal,
        operationIndex,
        label: operation.label,
        shape: operation.shape,
        outputElements: plan.outputElements,
        ranges: Object.freeze(ranges),
        probes,
      });
    });
  if (operations.length !== C300_OPERATION_COUNT) {
    throw new Error("OPT-0024 biased K7 operation count changed");
  }
  if (inputFrames === C300_INPUT_FRAMES) {
    const ranges = operations.reduce((sum, operation) =>
      sum + operation.ranges.length, 0);
    const probes = operations.reduce((sum, operation) =>
      sum + operation.probes.length, 0);
    const unique = operations.flatMap((operation) => operation.probes)
      .reduce((sum, probe) => sum + probe.count, 0);
    if (ranges !== C300_EXACT_RANGE_COUNT || probes !== C300_PROBE_COUNT ||
      unique !== C300_UNIQUE_PROBE_U16_COUNT) {
      throw new Error("OPT-0024 exact C300 coverage changed");
    }
  }
  return Object.freeze(operations);
}

function buildProbes(
  label: string,
  plan: AceFp16VaeConv1dPlan,
  ranges: readonly ExactRange[],
): readonly Opt0024Probe[] {
  const first = ranges[0];
  const tail = ranges.at(-1);
  if (first === undefined || tail === undefined) {
    throw new Error(`${label} has no graph ranges`);
  }
  if (label === "conv1" && plan.inputFrames === C300_INPUT_FRAMES) {
    if (ranges.length !== 2 || first.base !== 0 || first.count !== 524_288 ||
      tail.base !== 524_288 || tail.count !== 90_112) {
      throw new Error("OPT-0024 conv1 graph probes changed");
    }
    const syntheticBase = 290_816;
    const syntheticCount = 32_768;
    return Object.freeze([
      probeFromRange("first", first),
      Object.freeze({
        id: "interior" as const,
        source: "synthetic-interior" as const,
        rangeIndex: null,
        base: syntheticBase,
        count: syntheticCount,
        firstOutputRow: syntheticBase / plan.outputChannels,
        outputRowCount: syntheticCount / plan.outputChannels,
      }),
      probeFromRange("tail", tail),
    ]);
  }
  const interior = ranges[Math.floor(ranges.length / 2)];
  if (interior === undefined) {
    throw new Error(`${label} interior graph probe is missing`);
  }
  return Object.freeze([
    probeFromRange("first", first),
    probeFromRange("interior", interior),
    probeFromRange("tail", tail),
  ]);
}

function probeFromRange(id: Opt0024ProbeId, range: ExactRange): Opt0024Probe {
  return Object.freeze({
    id,
    source: "graph" as const,
    rangeIndex: range.rangeIndex,
    base: range.base,
    count: range.count,
    firstOutputRow: range.firstOutputRow,
    outputRowCount: range.outputRowCount,
  });
}

export function buildOpt0024Representatives(
  topology: readonly Opt0024Operation[] = buildOpt0024Topology(),
): readonly Opt0024Representative[] {
  const result = REPRESENTATIVE_SPECS.map((spec) => {
    const operation = topology.find(({ label }) => label === spec.label);
    if (operation === undefined) {
      throw new Error(`OPT-0024 representative ${spec.label} missing`);
    }
    return Object.freeze({
      id: spec.id,
      operation,
      weight: spec.weight,
      probes: operation.probes,
    });
  });
  if (result.reduce((sum, item) => sum + item.weight, 0) !==
      TIMING_WEIGHT_TOTAL) {
    throw new Error("OPT-0024 C300 timing weights changed");
  }
  return Object.freeze(result);
}

export function buildOpt0024TimingOrders(): readonly Readonly<{
  tierIndex: number;
  sampleIndex: number;
  order: readonly Opt0024Arm[];
}>[] {
  return Object.freeze(REPRESENTATIVE_SPECS.flatMap((_, tierIndex) =>
    TIMING_PAIR_ORDERS.map((order, sampleIndex) => Object.freeze({
      tierIndex,
      sampleIndex,
      order,
    }))
  ));
}

export function summarizeOpt0024Timing(
  inputs: readonly Opt0024TimingInput[],
): Readonly<Record<string, unknown>> {
  if (inputs.length !== REPRESENTATIVE_SPECS.length) {
    throw new Error("OPT-0024 requires four timing tiers");
  }
  let shippedWeightedMilliseconds = 0;
  let candidateWeightedMilliseconds = 0;
  const tiers = inputs.map((input, index) => {
    const expected = REPRESENTATIVE_SPECS[index];
    if (expected === undefined || input.id !== expected.id ||
      input.operationLabel !== expected.label || input.weight !== expected.weight) {
      throw new Error("OPT-0024 timing tier identity changed");
    }
    const shippedMedianMilliseconds = median4(input.samples.shipped);
    const candidateMedianMilliseconds = median4(input.samples.candidate);
    shippedWeightedMilliseconds += shippedMedianMilliseconds * input.weight;
    candidateWeightedMilliseconds += candidateMedianMilliseconds * input.weight;
    return Object.freeze({
      id: input.id,
      operationLabel: input.operationLabel,
      weight: input.weight,
      samples: input.samples,
      shippedMedianMilliseconds,
      candidateMedianMilliseconds,
      candidateWon: candidateMedianMilliseconds < shippedMedianMilliseconds,
      speedup: shippedMedianMilliseconds / candidateMedianMilliseconds,
    });
  });
  const everyTierCandidateWon = tiers.every((tier) => tier.candidateWon);
  const c128 = tiers.find(({ id }) => id === "c128");
  if (c128 === undefined) throw new Error("OPT-0024 C128 tier missing");
  const weightedSpeedup = shippedWeightedMilliseconds /
    candidateWeightedMilliseconds;
  const c128ThresholdPassed = c128.speedup >= C128_SPEEDUP_THRESHOLD;
  const weightedThresholdPassed = weightedSpeedup >=
    WEIGHTED_SPEEDUP_THRESHOLD;
  const passed = everyTierCandidateWon && c128ThresholdPassed &&
    weightedThresholdPassed;
  return Object.freeze({
    exactC300WeightTotal: TIMING_WEIGHT_TOTAL,
    longConv1TimingWeight: 0,
    longScopeCountsUsedAsTimingWeights: false,
    tiers: Object.freeze(tiers),
    shippedWeightedMilliseconds,
    candidateWeightedMilliseconds,
    weightedSpeedup,
    everyTierCandidateWon,
    c128Speedup: c128.speedup,
    c128Threshold: C128_SPEEDUP_THRESHOLD,
    c128ThresholdPassed,
    weightedThreshold: WEIGHTED_SPEEDUP_THRESHOLD,
    weightedThresholdPassed,
    passed,
    decision: passed
      ? "positive-c512-escalation-authorized"
      : "negative-stop-primitive-gate",
  });
}

export function assertOpt0024NumericalThresholds(
  summary: Readonly<Record<string, unknown>>,
  label: string,
): void {
  const nrmse = summary["nrmse"] as number;
  const snrDb = summary["snrDb"] === "Infinity"
    ? Number.POSITIVE_INFINITY
    : summary["snrDb"] as number;
  const pearson = summary["pearson"] as number;
  const relativeMaximumAbsoluteError =
    summary["relativeMaximumAbsoluteError"] as number;
  const ranges = summary["numericOutputRanges"] as Readonly<{
    control: Readonly<{ minimum: number; maximum: number }>;
    candidate: Readonly<{ minimum: number; maximum: number }>;
  }> | undefined;
  if (!Number.isFinite(nrmse) || nrmse > NRMSE_LIMIT ||
    snrDb < SNR_MINIMUM_DB || !Number.isFinite(pearson) ||
    pearson < PEARSON_MINIMUM ||
    !Number.isFinite(relativeMaximumAbsoluteError) ||
    relativeMaximumAbsoluteError > RELATIVE_MAXIMUM_ABSOLUTE_ERROR_LIMIT ||
    ranges === undefined ||
    ![ranges.control.minimum, ranges.control.maximum,
      ranges.candidate.minimum, ranges.candidate.maximum]
      .every(Number.isFinite) ||
    ranges.control.minimum > ranges.control.maximum ||
    ranges.candidate.minimum > ranges.candidate.maximum) {
    throw new Error(`${label} rejected frozen OPT-0024 numerical thresholds`);
  }
}

interface StaticCounts {
  operationExecutions: number;
  graphRanges: number;
  physicalWorkgroups: number;
  logicalMacs: number;
  workgroupKCin4Instances: number;
  validInputVec4Loads: number;
  weightVec4Loads: number;
  subgroupVectorBroadcastCollectives: number;
  broadcastInvocationCalls: number;
  fp16Dot4Calls: number;
  fp32Vec4AccumulatorAdds: number;
}

function emptyStaticCounts(): StaticCounts {
  return {
    operationExecutions: 0,
    graphRanges: 0,
    physicalWorkgroups: 0,
    logicalMacs: 0,
    workgroupKCin4Instances: 0,
    validInputVec4Loads: 0,
    weightVec4Loads: 0,
    subgroupVectorBroadcastCollectives: 0,
    broadcastInvocationCalls: 0,
    fp16Dot4Calls: 0,
    fp32Vec4AccumulatorAdds: 0,
  };
}

function addStaticCounts(
  target: StaticCounts,
  source: Readonly<StaticCounts>,
  multiplier = 1,
): void {
  for (const key of Object.keys(target) as (keyof StaticCounts)[]) {
    target[key] += source[key] * multiplier;
  }
}

export function summarizeOpt0024StaticAccounting(
  inputFrames: number,
): Readonly<{
  inputFrames: number;
  total: Readonly<StaticCounts>;
  tiers: Readonly<Record<string, Readonly<StaticCounts>>>;
  operations: readonly Readonly<{
    label: string;
    tier: string;
    counts: Readonly<StaticCounts>;
  }>[];
}> {
  const operations = buildOpt0024Topology(inputFrames).map((operation) => {
    const counts = staticCountsForOperation(operation);
    return Object.freeze({
      label: operation.label,
      tier: tierForOperation(operation.label),
      counts: Object.freeze(counts),
    });
  });
  const total = emptyStaticCounts();
  const tiers: Record<string, StaticCounts> = {};
  for (const operation of operations) {
    addStaticCounts(total, operation.counts);
    const tier = tiers[operation.tier] ??= emptyStaticCounts();
    addStaticCounts(tier, operation.counts);
  }
  return Object.freeze({
    inputFrames,
    total: Object.freeze(total),
    tiers: Object.freeze(Object.fromEntries(Object.entries(tiers).map(
      ([tier, counts]) => [tier, Object.freeze(counts)],
    ))),
    operations: Object.freeze(operations),
  });
}

function staticCountsForOperation(operation: Opt0024Operation): StaticCounts {
  const counts = emptyStaticCounts();
  counts.operationExecutions = 1;
  counts.graphRanges = operation.ranges.length;
  const cin4 = operation.shape.inputChannels / 4;
  for (const range of operation.ranges) {
    const planned = planAceOpt0024VaeConv1dDirectDot4SubgroupRange(
      planAceFp16VaeConv1d(operation.shape, "float16"),
      range,
    );
    const workgroups = planned.workgroupsX * planned.workgroupsY;
    const instances = workgroups * 7 * cin4;
    const validRowsAcrossKernel = validRowsAcrossKernelForRange(
      operation.shape,
      range,
    );
    const inputLoads = validRowsAcrossKernel * planned.workgroupsY * cin4;
    counts.physicalWorkgroups += workgroups;
    counts.logicalMacs += range.count * operation.shape.inputChannels * 7;
    counts.workgroupKCin4Instances += instances;
    counts.validInputVec4Loads += inputLoads;
    counts.weightVec4Loads += instances * 4 * 32 * 4;
    counts.subgroupVectorBroadcastCollectives += instances * 4 * 8;
    counts.broadcastInvocationCalls += instances * 4 * 8 * 32;
    counts.fp16Dot4Calls += inputLoads * 32 * 4;
    counts.fp32Vec4AccumulatorAdds += inputLoads * 32;
  }
  return counts;
}

function validRowsAcrossKernelForRange(
  shape: AceVaeConv1dShape,
  range: ExactRange,
): number {
  const firstTime = range.firstOutputRow % shape.inputFrames;
  const lastTime = firstTime + range.outputRowCount - 1;
  let total = 0;
  for (let kernel = 0; kernel < 7; kernel += 1) {
    const validFirst = Math.max(
      firstTime,
      shape.padding - kernel * shape.dilation,
      0,
    );
    const validLast = Math.min(
      lastTime,
      shape.inputFrames - 1 + shape.padding - kernel * shape.dilation,
      shape.inputFrames - 1,
    );
    total += Math.max(0, validLast - validFirst + 1);
  }
  return total;
}

function tierForOperation(label: string): string {
  if (label === "conv1") return "conv1";
  if (label.startsWith("block-0-")) return "c1024";
  if (label.startsWith("block-1-")) return "c512";
  if (label.startsWith("block-2-")) return "c256";
  if (label.startsWith("block-3-") || label.startsWith("block-4-")) {
    return "c128";
  }
  throw new Error(`OPT-0024 unclassified operation ${label}`);
}

export function summarizeOpt0024LongAccounting(): Readonly<{
  windows: readonly Readonly<{ inputFrames: number; multiplicity: number }>[];
  total: Readonly<StaticCounts>;
  tiers: Readonly<Record<string, Readonly<StaticCounts>>>;
  logicalInputBytes: number;
  logicalWeightBytes: number;
  logicalOperandBytes: number;
  logicalOperandBytesPerMac: number;
  workgroupStorageBytes: 0;
  workgroupBarrierCount: 0;
  repackBytes: 0;
  longConv1GraphRanges: number;
  longConv1TimingWeight: 0;
}> {
  const windows = Object.freeze([
    Object.freeze({ inputFrames: 448, multiplicity: 1 }),
    Object.freeze({ inputFrames: 512, multiplicity: 10 }),
    Object.freeze({ inputFrames: 340, multiplicity: 1 }),
  ]);
  const total = emptyStaticCounts();
  const tiers: Record<string, StaticCounts> = {};
  for (const window of windows) {
    const accounting = summarizeOpt0024StaticAccounting(window.inputFrames);
    addStaticCounts(total, accounting.total, window.multiplicity);
    for (const [tierName, counts] of Object.entries(accounting.tiers)) {
      const tier = tiers[tierName] ??= emptyStaticCounts();
      addStaticCounts(tier, counts, window.multiplicity);
    }
  }
  const logicalInputBytes = total.validInputVec4Loads * 8;
  const logicalWeightBytes = total.weightVec4Loads * 8;
  const logicalOperandBytes = logicalInputBytes + logicalWeightBytes;
  return Object.freeze({
    windows,
    total: Object.freeze(total),
    tiers: Object.freeze(Object.fromEntries(Object.entries(tiers).map(
      ([tier, counts]) => [tier, Object.freeze(counts)],
    ))),
    logicalInputBytes,
    logicalWeightBytes,
    logicalOperandBytes,
    logicalOperandBytesPerMac: logicalOperandBytes / total.logicalMacs,
    workgroupStorageBytes: 0,
    workgroupBarrierCount: 0,
    repackBytes: 0,
    longConv1GraphRanges: tiers["conv1"]?.graphRanges ?? 0,
    longConv1TimingWeight: 0,
  });
}

export function parseOpt0024ThermalGate(
  parameters: URLSearchParams,
  preparedCompletedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0024ThermalGate {
  const source = requiredParameter(parameters, "thermalSource");
  const command = requiredParameter(parameters, "thermalCommand");
  const traceStartedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalTraceStartedAtEpochMilliseconds",
  );
  const traceStartObservationIndex = requiredFiniteParameter(
    parameters,
    "thermalTraceStartObservationIndex",
  );
  const gateStartedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalGateStartedAtEpochMilliseconds",
  );
  const gateStartObservationIndex = requiredFiniteParameter(
    parameters,
    "thermalGateStartObservationIndex",
  );
  const gateCompletedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalGateCompletedAtEpochMilliseconds",
  );
  const gateCompletedObservationIndex = requiredFiniteParameter(
    parameters,
    "thermalGateCompletedObservationIndex",
  );
  const observationCount = requiredFiniteParameter(
    parameters,
    "thermalGateObservations",
  );
  const pollMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalPollMilliseconds",
  );
  const maximumPollGapMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalGateMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = requiredFiniteParameter(
    parameters,
    "thermalGateNonNominalObservations",
  );
  const missingObservationCount = requiredFiniteParameter(
    parameters,
    "thermalGateMissingObservations",
  );
  const durationMilliseconds = gateCompletedAtEpochMilliseconds -
    gateStartedAtEpochMilliseconds;
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    gateCompletedAtEpochMilliseconds;
  const integerFields = [
    traceStartObservationIndex,
    gateStartObservationIndex,
    gateCompletedObservationIndex,
    observationCount,
    nonNominalObservationCount,
    missingObservationCount,
  ];
  if (source !== THERMAL_SOURCE || command !== THERMAL_COMMAND ||
    integerFields.some((value) => !Number.isSafeInteger(value) || value < 0) ||
    traceStartObservationIndex > gateStartObservationIndex ||
    gateCompletedObservationIndex - gateStartObservationIndex + 1 !==
      observationCount ||
    observationCount < Math.floor(durationMilliseconds / 1_000) + 1 ||
    pollMilliseconds !== THERMAL_POLL_MILLISECONDS ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS ||
    nonNominalObservationCount !== 0 || missingObservationCount !== 0 ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    traceStartedAtEpochMilliseconds > gateStartedAtEpochMilliseconds ||
    gateStartedAtEpochMilliseconds < preparedCompletedAtEpochMilliseconds ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS) {
    throw new Error(
      "OPT-0024 thermal gate is incomplete, stale, missing, or non-nominal",
    );
  }
  return Object.freeze({
    source,
    command,
    traceStartedAtEpochMilliseconds,
    traceStartObservationIndex,
    gateStartedAtEpochMilliseconds,
    gateStartObservationIndex,
    gateCompletedAtEpochMilliseconds,
    gateCompletedObservationIndex,
    observationCount,
    pollMilliseconds: THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount: 0,
    missingObservationCount: 0,
    launchDelayMilliseconds,
  });
}

export function parseOpt0024ThermalCompletion(
  parameters: URLSearchParams,
  gate: Opt0024ThermalGate,
  requiredThroughEpochMilliseconds: number,
): Readonly<Record<string, unknown>> {
  const schema = requiredParameter(parameters, "thermalTraceSchema");
  const rawTraceSha256 = requiredParameter(parameters, "thermalTraceSha256");
  const rawTraceByteLength = requiredFiniteParameter(
    parameters,
    "thermalTraceByteLength",
  );
  const completedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalTraceCompletedAtEpochMilliseconds",
  );
  const completedObservationIndex = requiredFiniteParameter(
    parameters,
    "thermalTraceCompletedObservationIndex",
  );
  const observationCount = requiredFiniteParameter(
    parameters,
    "thermalTraceObservations",
  );
  const maximumPollGapMilliseconds = requiredFiniteParameter(
    parameters,
    "thermalTraceMaximumPollGapMilliseconds",
  );
  const nonNominalObservationCount = requiredFiniteParameter(
    parameters,
    "thermalTraceNonNominalObservations",
  );
  const missingObservationCount = requiredFiniteParameter(
    parameters,
    "thermalTraceMissingObservations",
  );
  const initialLevel = requiredFiniteParameter(
    parameters,
    "thermalTraceInitialLevel",
  );
  const finalLevel = requiredFiniteParameter(
    parameters,
    "thermalTraceFinalLevel",
  );
  const transitions = parseThermalTransitions(requiredParameter(
    parameters,
    "thermalTraceTransitionsJson",
  ));
  const durationMilliseconds = completedAtEpochMilliseconds -
    gate.traceStartedAtEpochMilliseconds;
  const integers = [rawTraceByteLength, completedObservationIndex,
    observationCount, nonNominalObservationCount, missingObservationCount,
    initialLevel, finalLevel];
  if (schema !== THERMAL_TRACE_SCHEMA ||
    !/^[0-9a-f]{64}$/.test(rawTraceSha256) ||
    integers.some((value) => !Number.isSafeInteger(value) || value < 0) ||
    rawTraceByteLength < 1 ||
    completedAtEpochMilliseconds < requiredThroughEpochMilliseconds ||
    completedAtEpochMilliseconds < gate.gateCompletedAtEpochMilliseconds ||
    completedObservationIndex < gate.gateCompletedObservationIndex ||
    completedObservationIndex - gate.traceStartObservationIndex + 1 !==
      observationCount ||
    observationCount < gate.observationCount ||
    observationCount < Math.floor(durationMilliseconds / 1_000) + 1 ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS ||
    missingObservationCount !== 0 || initialLevel !== 0) {
    throw new Error("OPT-0024 complete thermal artifact is invalid");
  }
  let derivedFinalLevel = initialLevel;
  let previousTransitionEpoch = Number.NEGATIVE_INFINITY;
  let sawNonNominalLevel = initialLevel !== 0;
  for (const transition of transitions) {
    if (transition.atEpochMilliseconds < gate.traceStartedAtEpochMilliseconds ||
      transition.atEpochMilliseconds > completedAtEpochMilliseconds) {
      throw new Error("OPT-0024 thermal transition is outside the trace");
    }
    if (transition.atEpochMilliseconds <= previousTransitionEpoch ||
      transition.level > 3 || transition.level === derivedFinalLevel) {
      throw new Error("OPT-0024 thermal transitions are malformed");
    }
    previousTransitionEpoch = transition.atEpochMilliseconds;
    derivedFinalLevel = transition.level;
    if (transition.level !== 0) sawNonNominalLevel = true;
  }
  if ((nonNominalObservationCount === 0) !== !sawNonNominalLevel ||
    derivedFinalLevel !== finalLevel) {
    throw new Error("OPT-0024 thermal transition summary is inconsistent");
  }
  return Object.freeze({
    schema,
    source: gate.source,
    command: gate.command,
    rawTraceSha256,
    rawTraceByteLength,
    traceStartedAtEpochMilliseconds: gate.traceStartedAtEpochMilliseconds,
    traceStartObservationIndex: gate.traceStartObservationIndex,
    completedAtEpochMilliseconds,
    completedObservationIndex,
    observationCount,
    durationMilliseconds,
    pollMilliseconds: THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds,
    nonNominalObservationCount,
    missingObservationCount,
    initialLevel,
    finalLevel,
    transitions,
    coveredValidationAndCleanup: true,
    classification: nonNominalObservationCount === 0
      ? "complete-trace-nominal"
      : "complete-trace-thermally-evolved-balanced-order-retained",
  });
}

function parseThermalTransitions(value: string): readonly Readonly<{
  atEpochMilliseconds: number;
  level: number;
}>[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error("OPT-0024 thermal transitions are not valid JSON");
  }
  if (!Array.isArray(parsed)) {
    throw new Error("OPT-0024 thermal transitions must be an array");
  }
  return Object.freeze(parsed.map((entry) => {
    if (typeof entry !== "object" || entry === null ||
      !("atEpochMilliseconds" in entry) || !("level" in entry)) {
      throw new Error("OPT-0024 thermal transition is invalid");
    }
    const atEpochMilliseconds = Number(entry.atEpochMilliseconds);
    const level = Number(entry.level);
    if (!Number.isSafeInteger(atEpochMilliseconds) ||
      !Number.isSafeInteger(level) || level < 0 || level > 3) {
      throw new Error("OPT-0024 thermal transition is invalid");
    }
    return Object.freeze({ atEpochMilliseconds, level });
  }));
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
      throw new Error("OPT-0024 rejected an unbalanced unmap");
    }
    buffer.unmap();
    this.unmapCount += 1;
    this.activeMapCount -= 1;
    if (this.activeMapCount < 0) {
      throw new Error("OPT-0024 map accounting underflow");
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
  const gateFieldset = requireElement<HTMLFieldSetElement>("#thermal-gate");
  const completionFieldset = requireElement<HTMLFieldSetElement>(
    "#thermal-completion",
  );
  const run = requireElement<HTMLButtonElement>("#run");
  const finalize = requireElement<HTMLButtonElement>("#finalize");
  let prepared: PreparedGate | undefined;
  let pending: Readonly<{
    receipt: Readonly<Record<string, unknown>>;
    thermalGate: Opt0024ThermalGate;
    cleanupCompletedAtEpochMilliseconds: number;
  }> | undefined;
  void prepareGate((message) => {
    progress.textContent = message;
  }).then(
    (value) => {
      prepared = value;
      document.body.dataset.status = "ready";
      progress.textContent = "ready: enter one fresh 30-second nominal slice";
      gateFieldset.disabled = false;
      run.disabled = false;
    },
    (error: unknown) => finishPage("failed", failureReceipt(error)),
  );
  run.addEventListener("click", () => {
    if (prepared === undefined) return;
    run.disabled = true;
    gateFieldset.disabled = true;
    document.body.dataset.status = "running";
    progress.textContent = "running frozen 32-sample composite timing screen";
    const owned = prepared;
    prepared = undefined;
    void runTimedGate(owned).then(
      (value) => {
        pending = value;
        document.body.dataset.status = "awaiting-thermal-artifact";
        progress.textContent =
          "timing and cleanup complete: stop logger and enter full trace";
        completionFieldset.disabled = false;
        finalize.disabled = false;
      },
      (error: unknown) => finishPage("failed", failureReceipt(error)),
    );
  }, { once: true });
  finalize.addEventListener("click", () => {
    if (pending === undefined) return;
    finalize.disabled = true;
    completionFieldset.disabled = true;
    try {
      const thermal = parseOpt0024ThermalCompletion(
        fieldParameters("#thermal-completion"),
        pending.thermalGate,
        pending.cleanupCompletedAtEpochMilliseconds,
      );
      finishPage("passed", Object.freeze({
        ...pending.receipt,
        thermal,
      }));
    } catch (error) {
      finishPage("failed", failureReceipt(error));
    } finally {
      pending = undefined;
    }
  }, { once: true });
}

async function prepareGate(
  updateProgress: (message: string) => void,
): Promise<PreparedGate> {
  const preparationStartedAtEpochMilliseconds = Date.now();
  const preparationStarted = performance.now();
  const topology = buildOpt0024Topology();
  const representatives = buildOpt0024Representatives(topology);
  updateProgress("authenticating exhaustive current/candidate shader sources");
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
    label: "ace-opt-0024-direct-dot4-c300-gate-device",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits,
  });
  const uncapturedErrors: string[] = [];
  const deviceLosses: string[] = [];
  const onUncapturedError = (event: GPUUncapturedErrorEvent): void => {
    uncapturedErrors.push(`${event.error.constructor.name}: ${event.error.message}`);
  };
  device.addEventListener("uncapturederror", onUncapturedError);
  void device.lost.then((info) => {
    if (info.reason !== "destroyed") {
      deviceLosses.push(`${info.reason}: ${info.message}`);
    }
  });
  const tracker = new BufferTracker();
  const shippedKernel = AceFp16VaeConv1dSubgroupKernel.create(
    device,
    Object.freeze({ subgroupMinSize: 32, subgroupMaxSize: 32 }),
  );
  const candidateKernel =
    AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      device,
      Object.freeze({ subgroupMinSize: 32, subgroupMaxSize: 32 }),
    );
  let queueDrained = false;
  let cleanupReceipt: Readonly<Record<string, unknown>> | undefined;
  let shared: SharedResources | undefined;
  const operations: PreparedOperation[] = [];
  const cleanup = async (): Promise<Readonly<Record<string, unknown>>> => {
    if (cleanupReceipt !== undefined) {
      return Object.freeze({ ...cleanupReceipt, repeatedCall: true });
    }
    await device.queue.onSubmittedWorkDone().catch(() => undefined);
    queueDrained = true;
    await settlePostDrainEvents();
    shippedKernel.destroy();
    candidateKernel.destroy();
    shippedKernel.destroy();
    candidateKernel.destroy();
    const postDestroy = shared === undefined || operations.length === 0
      ? Object.freeze({ tested: false })
      : await verifyPostDestroyRejection(
        device,
        shippedKernel,
        candidateKernel,
        shared,
        operations[0]!,
      );
    tracker.destroyAll();
    tracker.destroyAll();
    device.removeEventListener("uncapturederror", onUncapturedError);
    device.destroy();
    cleanupReceipt = Object.freeze({
      ...tracker.receipt(),
      queueDrainedBeforeRelease: queueDrained,
      kernelDestroyIdempotent: true,
      postDestroy,
      zeroLiveResources: tracker.liveBytes === 0,
      deviceDestroyed: true,
      repeatedCall: false,
    });
    return cleanupReceipt;
  };
  try {
    updateProgress("allocating one bounded shared C300 resource set");
    shared = createSharedResources(device, tracker, topology);
    for (const [index, operation] of topology.entries()) {
      updateProgress(`compiling operation ${index + 1}/16: ${operation.label}`);
      operations.push(await prepareOperation(
        device,
        tracker,
        shippedKernel,
        candidateKernel,
        shared,
        operation,
      ));
      await yieldToBrowser();
    }
    updateProgress("running all 48 probes twice against same-run controls");
    const correctness = await runCorrectness(
      device,
      tracker,
      shared,
      operations,
      updateProgress,
    );
    await device.queue.onSubmittedWorkDone();
    queueDrained = true;
    await settlePostDrainEvents();
    requireNoGpuFailures(uncapturedErrors, deviceLosses, "correctness");
    const preparedRepresentatives = representatives.map((representative) => {
      const prepared = operations.find(({ topology: operation }) =>
        operation.label === representative.operation.label
      );
      if (prepared === undefined) {
        throw new Error(`${representative.operation.label} prep missing`);
      }
      return Object.freeze({ representative, prepared });
    });
    updateProgress("symmetrically warming exact three-dispatch composites");
    await warmRepresentativeComposites(device, preparedRepresentatives);
    await settlePostDrainEvents();
    requireNoGpuFailures(uncapturedErrors, deviceLosses, "warmup");
    const preparedCompletedAtEpochMilliseconds = Date.now();
    const preparedGate: PreparedGate = Object.freeze({
      adapter,
      device,
      tracker,
      shippedKernel,
      candidateKernel,
      topology,
      operations: Object.freeze(operations),
      representatives: Object.freeze(preparedRepresentatives),
      shared,
      correctness,
      sourceAuthority,
      preparation: Object.freeze({
        startedAtEpochMilliseconds: preparationStartedAtEpochMilliseconds,
        completedAtEpochMilliseconds: preparedCompletedAtEpochMilliseconds,
        totalMilliseconds: performance.now() - preparationStarted,
        all32GeneratedPipelinesCompiledByCorrectness: true,
        exactCompositeWarmupsPerArmPerTier: 2,
        warmupSubmitDrainEventCount: 16,
        warmupDispatchCount: 48,
        correctnessCompletedBeforeWarmup: true,
        memoryHighWaterBytes: tracker.maximumLiveBytes,
      }),
      uncapturedErrors,
      deviceLosses,
      preparedCompletedAtEpochMilliseconds,
      updateProgress,
      cleanup,
    });
    return preparedGate;
  } catch (error) {
    await cleanup();
    throw error;
  }
}

function createSharedResources(
  device: GPUDevice,
  tracker: BufferTracker,
  topology: readonly Opt0024Operation[],
): SharedResources {
  const plans = topology.map((operation) =>
    planAceFp16VaeConv1d(operation.shape, "float16")
  );
  const maximumInputBindingBytes = Math.max(...plans.map((plan) =>
    plan.inputBindingBytes
  ));
  const maximumBiasBindingBytes = Math.max(...plans.map((plan) =>
    plan.biasBindingBytes
  ));
  const maximumOutputBindingBytes = Math.max(...plans.map((plan) =>
    plan.outputBindingBytes
  ));
  const maximumProbeBytes = Math.max(...topology.flatMap((operation) =>
    operation.probes.map((probe) => probe.count * 2)
  ));
  const input = createPeriodicU16Buffer(
    device,
    tracker,
    "opt-0024-shared-input",
    maximumInputBindingBytes,
    INPUT_PATTERN,
    GPUBufferUsage.STORAGE,
  );
  const bias = createPeriodicU16Buffer(
    device,
    tracker,
    "opt-0024-shared-bias",
    maximumBiasBindingBytes,
    BIAS_PATTERN,
    GPUBufferUsage.STORAGE,
  );
  const outputs = Object.freeze(Object.fromEntries(ARMS.map((arm) => [
    arm,
    tracker.create(device, {
      label: `opt-0024-${arm}-guarded-output`,
      size: STORAGE_GUARD_BYTES + maximumOutputBindingBytes +
        STORAGE_GUARD_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    }),
  ]))) as Readonly<Record<Opt0024Arm, GPUBuffer>>;
  const prefill = tracker.create(device, {
    label: "opt-0024-output-qnan-prefill",
    size: maximumProbeBytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint16Array(prefill.getMappedRange()).fill(OUTPUT_PREFILL_QNAN_F16);
  tracker.unmap(prefill);
  const { buffer: controls, offsets: controlOffsets } = createControls(
    device,
    tracker,
    topology,
  );
  return Object.freeze({
    input,
    bias,
    outputs,
    prefill,
    controls,
    controlOffsets,
    maximumOutputBindingBytes,
  });
}

async function prepareOperation(
  device: GPUDevice,
  tracker: BufferTracker,
  shippedKernel: AceFp16VaeConv1dSubgroupKernel,
  candidateKernel: AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
  shared: SharedResources,
  topology: Opt0024Operation,
): Promise<PreparedOperation> {
  const plan = planAceFp16VaeConv1d(topology.shape, "float16");
  const weight = tracker.create(device, {
    label: `${topology.label}-native-o-k-i-weight`,
    size: plan.weightBindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const weightWords = new Uint16Array(weight.getMappedRange());
  for (let index = 0; index < plan.weightElements; index += 1) {
    weightWords[index] = deterministicWeightBits(topology.operationIndex, index);
  }
  tracker.unmap(weight);
  const dispatchEntries: [Opt0024ProbeId,
    Readonly<Record<Opt0024Arm, EncodableDispatch>>][] = [];
  for (const probe of topology.probes) {
    const controlOffset = shared.controlOffsets.get(probeKey(
      topology.label,
      probe.id,
    ));
    if (controlOffset === undefined) {
      throw new Error(`${topology.label} ${probe.id} control missing`);
    }
    const range = outputRangeBinding(
      shared.controls,
      controlOffset,
      probe.base,
      probe.count,
    );
    const common = Object.freeze({
      input: binding(shared.input, plan.inputBindingBytes),
      weight: binding(weight, plan.weightBindingBytes),
      bias: binding(shared.bias, plan.biasBindingBytes),
    });
    const shipped = await shippedKernel.createDispatch(
      `${topology.label}-${probe.id}-shipped-fixed32`,
      topology.shape,
      Object.freeze({
        ...common,
        output: guardedOutputBinding(shared.outputs.shipped, plan),
      }),
      "float16",
      range,
    );
    const candidate = await candidateKernel.createDispatch(
      `${topology.label}-${probe.id}-candidate-direct-dot4`,
      topology.shape,
      Object.freeze({
        ...common,
        output: guardedOutputBinding(shared.outputs.candidate, plan),
      }),
      "float16",
      range,
    );
    assertDispatch(topology, probe, plan, shipped, candidate);
    dispatchEntries.push([probe.id, Object.freeze({ shipped, candidate })]);
  }
  return Object.freeze({
    topology,
    plan,
    weight,
    dispatches: Object.freeze(Object.fromEntries(dispatchEntries)) as
      Readonly<Record<Opt0024ProbeId,
        Readonly<Record<Opt0024Arm, EncodableDispatch>>>>,
  });
}

function assertDispatch(
  operation: Opt0024Operation,
  probe: Opt0024Probe,
  plan: AceFp16VaeConv1dPlan,
  shipped: EncodableDispatch,
  candidate: EncodableDispatch,
): void {
  const range = Object.freeze({ base: probe.base, count: probe.count });
  const expectedShipped = planAceFp16VaeConv1dSubgroupRange(plan, range);
  const expectedCandidate =
    planAceOpt0024VaeConv1dDirectDot4SubgroupRange(plan, range);
  if (shipped.kernelId !== ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID ||
    candidate.kernelId !==
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID ||
    !sameRange(shipped.outputRange, expectedShipped) ||
    !sameRange(candidate.outputRange, expectedCandidate) ||
    !sameRange(shipped.outputRange, candidate.outputRange)) {
    throw new Error(`${operation.label} ${probe.id} dispatch changed`);
  }
}

async function runCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  shared: SharedResources,
  operations: readonly PreparedOperation[],
  updateProgress: (message: string) => void,
): Promise<Readonly<Record<string, unknown>>> {
  const aggregate = createNumericalAccumulator();
  const cases: Readonly<Record<string, unknown>>[] = [];
  let uniqueU16Count = 0;
  let comparedU16Count = 0;
  let executionCount = 0;
  for (const [operationIndex, operation] of operations.entries()) {
    const probes: Readonly<Record<string, unknown>>[] = [];
    for (const [probeIndex, probe] of operation.topology.probes.entries()) {
      updateProgress(
        `correctness ${operationIndex + 1}/16, probe ${probeIndex + 1}/3`,
      );
      uniqueU16Count += probe.count;
      const probeAggregate = createNumericalAccumulator();
      const executions: Readonly<Record<string, unknown>>[] = [];
      for (const [executionIndex, order] of CORRECTNESS_ORDERS.entries()) {
        const execution = await executeCorrectnessPair(
          device,
          tracker,
          shared,
          operation,
          probe,
          order,
          executionIndex,
        );
        executions.push(execution.receipt);
        mergeNumericalAccumulators(probeAggregate, execution.numerical);
        mergeNumericalAccumulators(aggregate, execution.numerical);
        comparedU16Count += probe.count;
        executionCount += 2;
      }
      const firstHashes = executions[0]!["armSha256"] as
        Readonly<Record<Opt0024Arm, string>>;
      const secondHashes = executions[1]!["armSha256"] as
        Readonly<Record<Opt0024Arm, string>>;
      if (firstHashes.candidate !== secondHashes.candidate) {
        throw new Error(
          `${operation.topology.label} ${probe.id} candidate rerun changed`,
        );
      }
      const numerical = summarizeNumericalAccumulator(probeAggregate);
      assertOpt0024NumericalThresholds(
        numerical,
        `${operation.topology.label} ${probe.id}`,
      );
      probes.push(Object.freeze({
        id: probe.id,
        source: probe.source,
        rangeIndex: probe.rangeIndex,
        base: probe.base,
        count: probe.count,
        firstOutputRow: probe.firstOutputRow,
        outputRowCount: probe.outputRowCount,
        candidateExecutionSha256: Object.freeze([
          firstHashes.candidate,
          secondHashes.candidate,
        ]),
        deterministicCandidateRerun: true,
        sameRunCandidateControlComparisons: probe.count * 2,
        executions: Object.freeze(executions),
        numerical,
      }));
    }
    cases.push(Object.freeze({
      ordinal: operation.topology.ordinal,
      operationIndex: operation.topology.operationIndex,
      label: operation.topology.label,
      shape: operation.topology.shape,
      probes: Object.freeze(probes),
    }));
    await yieldToBrowser();
  }
  if (uniqueU16Count !== C300_UNIQUE_PROBE_U16_COUNT ||
    comparedU16Count !== C300_CANDIDATE_CONTROL_COMPARISON_COUNT ||
    aggregate.count !== C300_CANDIDATE_CONTROL_COMPARISON_COUNT ||
    executionCount !== C300_PROBE_COUNT * 2 * 2) {
    throw new Error("OPT-0024 aggregate correctness accounting changed");
  }
  const aggregateNumerical = summarizeNumericalAccumulator(aggregate);
  assertOpt0024NumericalThresholds(aggregateNumerical, "OPT-0024 aggregate");
  return Object.freeze({
    operationCount: operations.length,
    probeCount: C300_PROBE_COUNT,
    uniqueRawU16ValueCount: uniqueU16Count,
    candidateExecutionsPerProbe: 2,
    shippedControlExecutionsPerProbe: 2,
    commandBufferAndMapCount: C300_PROBE_COUNT * 2,
    totalGpuDispatchCount: executionCount,
    sameRunCandidateControlComparisonCount: comparedU16Count,
    qNaNPrefillCompleteWrites: true,
    allOutputsFinite: true,
    guardsAndAdjacentCanariesUntouched: true,
    deterministicCandidateRerunHashes: true,
    fullFp16UlpDistributionReported: true,
    signedZeroDifferencesReported: true,
    thresholds: Object.freeze({
      nrmseMaximum: NRMSE_LIMIT,
      snrMinimumDb: SNR_MINIMUM_DB,
      pearsonMinimum: PEARSON_MINIMUM,
      relativeMaximumAbsoluteErrorMaximum:
        RELATIVE_MAXIMUM_ABSOLUTE_ERROR_LIMIT,
    }),
    aggregate: aggregateNumerical,
    cases: Object.freeze(cases),
    passed: true,
  });
}

async function executeCorrectnessPair(
  device: GPUDevice,
  tracker: BufferTracker,
  shared: SharedResources,
  prepared: PreparedOperation,
  probe: Opt0024Probe,
  order: readonly Opt0024Arm[],
  executionIndex: number,
): Promise<Readonly<{
  receipt: Readonly<Record<string, unknown>>;
  numerical: NumericalAccumulator;
}>> {
  const selectedStart = probe.base * 2;
  const selectedBytes = probe.count * 2;
  const beforeBytes = Math.min(STORAGE_GUARD_BYTES, selectedStart);
  const afterBytes = Math.min(
    STORAGE_GUARD_BYTES,
    prepared.plan.outputBindingBytes - selectedStart - selectedBytes,
  );
  const armBytes = STORAGE_GUARD_BYTES * 2 + selectedBytes + beforeBytes +
    afterBytes;
  const readback = tracker.create(device, {
    label: `${prepared.topology.label}-${probe.id}-${executionIndex}-readback`,
    size: armBytes * ARMS.length,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    for (const arm of ARMS) {
      primeOutput(
        device,
        shared.outputs[arm],
        prepared.plan,
        selectedStart,
        selectedBytes,
        beforeBytes,
        afterBytes,
      );
    }
    const encoder = device.createCommandEncoder({
      label: `${prepared.topology.label}-${probe.id}-${executionIndex}`,
    });
    for (const arm of ARMS) {
      encoder.copyBufferToBuffer(
        shared.prefill,
        0,
        shared.outputs[arm],
        STORAGE_GUARD_BYTES + selectedStart,
        selectedBytes,
      );
    }
    const pass = encoder.beginComputePass();
    for (const arm of order) prepared.dispatches[probe.id][arm].encode(pass);
    pass.end();
    for (const arm of ARMS) {
      encodeOutputReadback(
        encoder,
        shared.outputs[arm],
        readback,
        ARMS.indexOf(arm) * armBytes,
        prepared.plan,
        selectedStart,
        selectedBytes,
        beforeBytes,
        afterBytes,
      );
    }
    device.queue.submit([encoder.finish()]);
    await tracker.mapRead(readback);
    mapped = true;
    const mappedRange = readback.getMappedRange();
    const arms = {} as Record<Opt0024Arm, ReadbackArm>;
    for (const arm of ARMS) {
      arms[arm] = await scanOutputReadback(
        mappedRange,
        ARMS.indexOf(arm) * armBytes,
        probe.count,
        selectedBytes,
        beforeBytes,
        afterBytes,
      );
    }
    const numerical = compareFp16Outputs(
      arms.shipped.words,
      arms.candidate.words,
      prepared.topology,
      probe,
      executionIndex,
    );
    return Object.freeze({
      receipt: Object.freeze({
        executionIndex,
        order: order.join("/"),
        commandBufferCount: 1,
        computePassCount: 1,
        dispatchCount: 2,
        matchingQueueCompletionByMap: true,
        armSha256: Object.freeze({
          shipped: arms.shipped.sha256,
          candidate: arms.candidate.sha256,
        }),
        completeFiniteWrites: true,
        guardsAndAdjacentCanariesUntouched: true,
      }),
      numerical,
    });
  } finally {
    if (mapped) tracker.unmap(readback);
    tracker.destroy(readback);
  }
}

function createNumericalAccumulator(): NumericalAccumulator {
  return {
    count: 0,
    rawU16MismatchCount: 0,
    sumError: 0,
    sumAbsoluteError: 0,
    sumSquaredError: 0,
    sumControlSquared: 0,
    sumControl: 0,
    sumCandidate: 0,
    sumCandidateSquared: 0,
    sumProduct: 0,
    maximumAbsoluteError: 0,
    controlPeak: 0,
    controlMinimum: Number.POSITIVE_INFINITY,
    controlMaximum: Number.NEGATIVE_INFINITY,
    candidateMinimum: Number.POSITIVE_INFINITY,
    candidateMaximum: Number.NEGATIVE_INFINITY,
    signedZeroDifferenceCount: 0,
    ulpDistribution: new Map<number, number>(),
    firstDifference: null,
    worst: null,
  };
}

function compareFp16Outputs(
  controlWords: Uint16Array,
  candidateWords: Uint16Array,
  operation: Opt0024Operation,
  probe: Opt0024Probe,
  executionIndex: number,
): NumericalAccumulator {
  if (controlWords.length !== candidateWords.length ||
    controlWords.length !== probe.count) {
    throw new Error("OPT-0024 numerical comparison length changed");
  }
  const result = createNumericalAccumulator();
  for (let index = 0; index < controlWords.length; index += 1) {
    const controlBits = controlWords[index]!;
    const candidateBits = candidateWords[index]!;
    const control = fp16ToNumber(controlBits);
    const candidate = fp16ToNumber(candidateBits);
    if (!Number.isFinite(control) || !Number.isFinite(candidate)) {
      throw new Error("OPT-0024 numerical comparison found non-finite output");
    }
    const error = candidate - control;
    const absoluteError = Math.abs(error);
    const outputIndex = probe.base + index;
    const location = Object.freeze({
      executionIndex,
      localIndex: index,
      outputIndex,
      outputRow: Math.floor(outputIndex / operation.shape.outputChannels),
      outputChannel: outputIndex % operation.shape.outputChannels,
      controlBits,
      candidateBits,
      control,
      candidate,
      absoluteError,
    });
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
    if (result.worst === null ||
      absoluteError > result.maximumAbsoluteError) {
      result.maximumAbsoluteError = absoluteError;
      result.worst = location;
    }
    if (controlBits !== candidateBits) {
      result.rawU16MismatchCount += 1;
      result.firstDifference ??= location;
    }
    if ((controlBits === 0x0000 && candidateBits === 0x8000) ||
      (controlBits === 0x8000 && candidateBits === 0x0000)) {
      result.signedZeroDifferenceCount += 1;
    }
    const ulp = Math.abs(orderedFp16(controlBits) - orderedFp16(candidateBits));
    result.ulpDistribution.set(ulp,
      (result.ulpDistribution.get(ulp) ?? 0) + 1);
  }
  return result;
}

function mergeNumericalAccumulators(
  target: NumericalAccumulator,
  source: NumericalAccumulator,
): void {
  target.count += source.count;
  target.rawU16MismatchCount += source.rawU16MismatchCount;
  target.sumError += source.sumError;
  target.sumAbsoluteError += source.sumAbsoluteError;
  target.sumSquaredError += source.sumSquaredError;
  target.sumControlSquared += source.sumControlSquared;
  target.sumControl += source.sumControl;
  target.sumCandidate += source.sumCandidate;
  target.sumCandidateSquared += source.sumCandidateSquared;
  target.sumProduct += source.sumProduct;
  target.controlPeak = Math.max(target.controlPeak, source.controlPeak);
  target.controlMinimum = Math.min(target.controlMinimum,
    source.controlMinimum);
  target.controlMaximum = Math.max(target.controlMaximum,
    source.controlMaximum);
  target.candidateMinimum = Math.min(target.candidateMinimum,
    source.candidateMinimum);
  target.candidateMaximum = Math.max(target.candidateMaximum,
    source.candidateMaximum);
  target.signedZeroDifferenceCount += source.signedZeroDifferenceCount;
  target.firstDifference ??= source.firstDifference;
  if (target.worst === null ||
    source.maximumAbsoluteError > target.maximumAbsoluteError) {
    target.maximumAbsoluteError = source.maximumAbsoluteError;
    target.worst = source.worst;
  }
  for (const [ulp, count] of source.ulpDistribution) {
    target.ulpDistribution.set(ulp,
      (target.ulpDistribution.get(ulp) ?? 0) + count);
  }
}

function summarizeNumericalAccumulator(
  value: NumericalAccumulator,
): Readonly<Record<string, unknown>> {
  if (value.count <= 0) throw new Error("OPT-0024 empty numerical result");
  if (![value.controlMinimum, value.controlMaximum, value.candidateMinimum,
    value.candidateMaximum].every(Number.isFinite) ||
    value.controlMinimum > value.controlMaximum ||
    value.candidateMinimum > value.candidateMaximum) {
    throw new Error("OPT-0024 numerical output ranges are invalid");
  }
  const rmsError = Math.sqrt(value.sumSquaredError / value.count);
  const nrmse = Math.sqrt(
    value.sumSquaredError / Math.max(value.sumControlSquared, 1e-24),
  );
  const snrDb = value.sumSquaredError === 0
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
  const pearsonUnclamped = denominator === 0
    ? (value.sumSquaredError === 0 ? 1 : 0)
    : covariance / denominator;
  const pearson = Math.max(-1, Math.min(1, pearsonUnclamped));
  const fp16UlpDistribution = Object.freeze(Object.fromEntries(
    [...value.ulpDistribution.entries()]
      .sort(([left], [right]) => left - right)
      .map(([ulp, count]) => [String(ulp), count]),
  ));
  return Object.freeze({
    comparedValueCount: value.count,
    rawU16MismatchCount: value.rawU16MismatchCount,
    maximumAbsoluteError: value.maximumAbsoluteError,
    meanAbsoluteError: value.sumAbsoluteError / value.count,
    meanError: value.sumError / value.count,
    rmsError,
    nrmse,
    snrDb: Number.isFinite(snrDb) ? snrDb : "Infinity",
    pearson,
    controlPeak: value.controlPeak,
    relativeMaximumAbsoluteError: value.maximumAbsoluteError /
      Math.max(value.controlPeak, 1e-6),
    numericOutputRanges: Object.freeze({
      control: Object.freeze({
        minimum: value.controlMinimum,
        maximum: value.controlMaximum,
      }),
      candidate: Object.freeze({
        minimum: value.candidateMinimum,
        maximum: value.candidateMaximum,
      }),
    }),
    signedZeroDifferenceCount: value.signedZeroDifferenceCount,
    fp16UlpDistribution,
    fp16UlpDistributionCount: Object.values(fp16UlpDistribution)
      .reduce((sum, count) => sum + count, 0),
    firstDifference: value.firstDifference,
    worstLocation: value.worst,
  });
}

function orderedFp16(bits: number): number {
  return (bits & 0x8000) !== 0
    ? 0x8000 - (bits & 0x7fff)
    : 0x8000 + bits;
}

function fp16ToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1_024);
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1_024);
}

function primeOutput(
  device: GPUDevice,
  output: GPUBuffer,
  plan: AceFp16VaeConv1dPlan,
  selectedStart: number,
  selectedBytes: number,
  beforeBytes: number,
  afterBytes: number,
): void {
  writeU32Pattern(device, output, 0, STORAGE_GUARD_BYTES, STORAGE_GUARD_U32);
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

function encodeOutputReadback(
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
  encoder.copyBufferToBuffer(output, 0, readback, target,
    STORAGE_GUARD_BYTES);
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
  wordCount: number,
  selectedBytes: number,
  beforeBytes: number,
  afterBytes: number,
): Promise<ReadbackArm> {
  const prefix = new Uint32Array(mappedRange, base, STORAGE_GUARD_BYTES / 4);
  const suffix = new Uint32Array(
    mappedRange,
    base + STORAGE_GUARD_BYTES,
    STORAGE_GUARD_BYTES / 4,
  );
  const selectedOffset = base + STORAGE_GUARD_BYTES * 2;
  const words = new Uint16Array(
    mappedRange,
    selectedOffset,
    wordCount,
  ).slice();
  let adjacentOffset = selectedOffset + selectedBytes;
  const before = new Uint32Array(mappedRange, adjacentOffset, beforeBytes / 4);
  adjacentOffset += beforeBytes;
  const after = new Uint32Array(mappedRange, adjacentOffset, afterBytes / 4);
  let nonFiniteCount = 0;
  let prefillQNaNCount = 0;
  for (const word of words) {
    if ((word & 0x7c00) === 0x7c00) nonFiniteCount += 1;
    if (word === OUTPUT_PREFILL_QNAN_F16) prefillQNaNCount += 1;
  }
  const guardsUntouched = everyU32(prefix, STORAGE_GUARD_U32) &&
    everyU32(suffix, STORAGE_GUARD_U32) &&
    everyU32(before, ADJACENT_CANARY_U32) &&
    everyU32(after, ADJACENT_CANARY_U32);
  if (nonFiniteCount !== 0 || prefillQNaNCount !== 0 || !guardsUntouched) {
    throw new Error(
      `OPT-0024 output scan failed: nonfinite=${nonFiniteCount}, ` +
        `qnan=${prefillQNaNCount}, guards=${guardsUntouched}`,
    );
  }
  return Object.freeze({
    words,
    sha256: await sha256Bytes(new Uint8Array(words.buffer)),
    nonFiniteCount,
    prefillQNaNCount,
    guardsUntouched,
  });
}

async function warmRepresentativeComposites(
  device: GPUDevice,
  representatives: PreparedGate["representatives"],
): Promise<void> {
  for (const { representative, prepared } of representatives) {
    for (const order of CORRECTNESS_ORDERS) {
      for (const arm of order) {
        await executeComposite(
          device,
          representative.probes.map((probe) => prepared.dispatches[probe.id][arm]),
        );
      }
    }
  }
}

async function runTimedGate(
  prepared: PreparedGate,
): Promise<Readonly<{
  receipt: Readonly<Record<string, unknown>>;
  thermalGate: Opt0024ThermalGate;
  cleanupCompletedAtEpochMilliseconds: number;
}>> {
  const attributedLaunchAtEpochMilliseconds = Date.now();
  let submitDrainEventCount = 0;
  let dispatchCount = 0;
  let firstTimedSubmitAtEpochMilliseconds: number | null = null;
  let thermalGate: Opt0024ThermalGate | undefined;
  try {
    thermalGate = parseOpt0024ThermalGate(
      fieldParameters("#thermal-gate"),
      prepared.preparedCompletedAtEpochMilliseconds,
      attributedLaunchAtEpochMilliseconds,
    );
    const timingInputs: Opt0024TimingInput[] = prepared.representatives.map(
      ({ representative }) => ({
        id: representative.id,
        operationLabel: representative.operation.label,
        weight: representative.weight,
        samples: { shipped: [], candidate: [] },
      }),
    );
    for (const entry of buildOpt0024TimingOrders()) {
      const representative = prepared.representatives[entry.tierIndex];
      const timingInput = timingInputs[entry.tierIndex];
      if (representative === undefined || timingInput === undefined) {
        throw new Error("OPT-0024 timing order references a missing tier");
      }
      prepared.updateProgress(
        `timing tier ${entry.tierIndex + 1}/4, pair ${entry.sampleIndex + 1}/4`,
      );
      for (const arm of entry.order) {
        const dispatches = representative.representative.probes.map((probe) =>
          representative.prepared.dispatches[probe.id][arm]
        );
        const sample = await executeComposite(
          prepared.device,
          dispatches,
        );
        firstTimedSubmitAtEpochMilliseconds ??=
          sample.submittedAtEpochMilliseconds;
        (timingInput.samples[arm] as number[]).push(sample.wallMilliseconds);
        submitDrainEventCount += 1;
        dispatchCount += dispatches.length;
      }
      await yieldToBrowser();
    }
    const timingCompletedAtEpochMilliseconds = Date.now();
    if (firstTimedSubmitAtEpochMilliseconds === null ||
      firstTimedSubmitAtEpochMilliseconds -
          thermalGate.gateCompletedAtEpochMilliseconds >
        MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS ||
      firstTimedSubmitAtEpochMilliseconds <
        attributedLaunchAtEpochMilliseconds) {
      throw new Error("OPT-0024 attributed launch did not bind first submit");
    }
    await prepared.device.queue.onSubmittedWorkDone();
    await settlePostDrainEvents();
    requireNoGpuFailures(
      prepared.uncapturedErrors,
      prepared.deviceLosses,
      "timing",
    );
    if (submitDrainEventCount !== TIMED_SUBMIT_DRAIN_EVENTS ||
      dispatchCount !== TIMED_DISPATCH_COUNT) {
      throw new Error("OPT-0024 timed submission accounting changed");
    }
    const immutableInputs = timingInputs.map((input) => Object.freeze({
      ...input,
      samples: Object.freeze({
        shipped: Object.freeze([...input.samples.shipped]),
        candidate: Object.freeze([...input.samples.candidate]),
      }),
    }));
    const timing = summarizeOpt0024Timing(immutableInputs);
    const environment = environmentReceipt(prepared.adapter, prepared.device);
    const memoryBeforeCleanup = prepared.tracker.receipt();
    const cleanupStartedAtEpochMilliseconds = Date.now();
    const firstCleanup = await prepared.cleanup();
    const secondCleanup = await prepared.cleanup();
    const cleanupCompletedAtEpochMilliseconds = Date.now();
    const cleanup = Object.freeze({
      startedAtEpochMilliseconds: cleanupStartedAtEpochMilliseconds,
      completedAtEpochMilliseconds: cleanupCompletedAtEpochMilliseconds,
      firstCall: firstCleanup,
      secondCall: secondCleanup,
      drainBeforeRelease: firstCleanup["queueDrainedBeforeRelease"] === true,
      mapsBalanced: firstCleanup["mapsBalanced"] === true &&
        secondCleanup["mapsBalanced"] === true,
      zeroLiveResources: firstCleanup["zeroLiveResources"] === true &&
        secondCleanup["zeroLiveResources"] === true,
      idempotent: secondCleanup["repeatedCall"] === true,
      postDestroyRejection: firstCleanup["postDestroy"],
      deviceDestroyed: firstCleanup["deviceDestroyed"] === true,
    });
    if (!cleanup.drainBeforeRelease || !cleanup.mapsBalanced ||
      !cleanup.zeroLiveResources || !cleanup.idempotent ||
      !cleanup.deviceDestroyed) {
      throw new Error("OPT-0024 cleanup gate failed");
    }
    const receipt = Object.freeze({
      schema: "ace-opt-0024-vae-k7-direct-subgroup-fp16-dot4-c300-v1",
      status: "passed",
      experimentId: EXPERIMENT_ID,
      classification:
        "bounded-approximate-primitive-decision-gate-not-integrated",
      recordedAt: new Date().toISOString(),
      identity: prepared.sourceAuthority,
      environment,
      preparation: prepared.preparation,
      protocol: Object.freeze({
        gate: thermalGate,
        completeExternalThermalArtifact: "joined-at-finalization",
        correctnessCompletedBeforeTiming: true,
        symmetricCompositeWarmupCompletedBeforeFreshThermalGate: true,
        correctnessOrders: Object.freeze(["shipped/candidate", "candidate/shipped"]),
        timingPairOrderPerTier: Object.freeze([
          "shipped/candidate",
          "candidate/shipped",
          "candidate/shipped",
          "shipped/candidate",
        ]),
        timingCompositeDispatchOrder: Object.freeze([
          "first",
          "interior",
          "tail",
        ]),
        timingSampleTopology:
          "one command buffer, one compute pass, three dispatches, one submit, one matching queue drain",
        timingSamplesPerArmPerTier: TIMING_SAMPLES_PER_ARM_TIER,
        timedSubmitDrainEventCount: submitDrainEventCount,
        timedDispatchCount: dispatchCount,
        attributedLaunchAtEpochMilliseconds,
        firstTimedSubmitAtEpochMilliseconds,
        timingCompletedAtEpochMilliseconds,
        continuousExternalThermalTraceRequiredThroughEpochMilliseconds:
          cleanupCompletedAtEpochMilliseconds,
        unchangedThermalRetryPerformed: false,
        candidateTimedRetryPerformed: false,
      }),
      topology: Object.freeze({
        inputFrames: C300_INPUT_FRAMES,
        operationCount: prepared.topology.length,
        exactGraphRangeCount: prepared.topology.reduce((sum, operation) =>
          sum + operation.ranges.length, 0),
        probeCount: prepared.topology.reduce((sum, operation) =>
          sum + operation.probes.length, 0),
        operations: Object.freeze(prepared.topology.map(compactOperation)),
      }),
      staticLongScope: summarizeOpt0024LongAccounting(),
      correctness: prepared.correctness,
      timing: Object.freeze({
        ...timing,
        timingStartedAtEpochMilliseconds:
          firstTimedSubmitAtEpochMilliseconds,
        timingCompletedAtEpochMilliseconds,
        submitDrainEventCount,
        dispatchCount,
        compositeDispatchesPerSample: TIMED_COMPOSITE_DISPATCHES,
        caveat:
          "C300 separately drained three-range composites only; no package, decoder, waveform, listening, C4500, or product claim.",
      }),
      decision: Object.freeze({
        disposition: timing["decision"],
        c512SubsystemGateAuthorized: timing["passed"] === true,
        explicitBenchmarkProfileForC512Authorized: timing["passed"] === true,
        productionPackageProfileAuthorized: false,
        productionIntegrationAuthorized: false,
        waveformOrListeningAuthorized: false,
        c4500RunAuthorized: false,
        fullProductRunAuthorized: false,
      }),
      memoryBeforeCleanup,
      cleanup,
    });
    return Object.freeze({
      receipt,
      thermalGate,
      cleanupCompletedAtEpochMilliseconds,
    });
  } catch (error) {
    const cleanup = await prepared.cleanup();
    throw new Opt0024GateFailure(
      error instanceof Error ? error.message : String(error),
      Object.freeze({
        phase: submitDrainEventCount === 0
          ? "setup-only-preflight-rejection"
          : "timed-gate-failure",
        attributedLaunchAtEpochMilliseconds,
        firstTimedSubmitAtEpochMilliseconds,
        timedSubmitDrainEventCount: submitDrainEventCount,
        timedDispatchCount: dispatchCount,
        noTimedDispatches: submitDrainEventCount === 0,
        unchangedRetryPerformed: false,
        cleanup,
      }),
      { cause: error },
    );
  }
}

async function executeComposite(
  device: GPUDevice,
  dispatches: readonly EncodableDispatch[],
): Promise<Readonly<{
  submittedAtEpochMilliseconds: number;
  wallMilliseconds: number;
}>> {
  if (dispatches.length !== TIMED_COMPOSITE_DISPATCHES) {
    throw new Error("OPT-0024 composite must contain first/interior/tail");
  }
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (const dispatch of dispatches) dispatch.encode(pass);
  pass.end();
  const command = encoder.finish();
  const started = performance.now();
  const submittedAtEpochMilliseconds = Date.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  return Object.freeze({
    submittedAtEpochMilliseconds,
    wallMilliseconds: performance.now() - started,
  });
}

async function verifyPostDestroyRejection(
  device: GPUDevice,
  shippedKernel: AceFp16VaeConv1dSubgroupKernel,
  candidateKernel: AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
  shared: SharedResources,
  prepared: PreparedOperation,
): Promise<Readonly<Record<string, unknown>>> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  let shippedEncodeRejected = false;
  let candidateEncodeRejected = false;
  try {
    prepared.dispatches.first.shipped.encode(pass);
  } catch {
    shippedEncodeRejected = true;
  }
  try {
    prepared.dispatches.first.candidate.encode(pass);
  } catch {
    candidateEncodeRejected = true;
  }
  pass.end();
  const probe = prepared.topology.probes[0]!;
  const offset = shared.controlOffsets.get(probeKey(
    prepared.topology.label,
    probe.id,
  ));
  if (offset === undefined) throw new Error("OPT-0024 cleanup control missing");
  const range = outputRangeBinding(shared.controls, offset, probe.base, probe.count);
  const bindings = Object.freeze({
    input: binding(shared.input, prepared.plan.inputBindingBytes),
    weight: binding(prepared.weight, prepared.plan.weightBindingBytes),
    bias: binding(shared.bias, prepared.plan.biasBindingBytes),
    output: guardedOutputBinding(shared.outputs.shipped, prepared.plan),
  });
  const shippedCreateRejected = await shippedKernel.createDispatch(
    "post-destroy-shipped",
    prepared.topology.shape,
    bindings,
    "float16",
    range,
  ).then(() => false, () => true);
  const candidateCreateRejected = await candidateKernel.createDispatch(
    "post-destroy-candidate",
    prepared.topology.shape,
    bindings,
    "float16",
    range,
  ).then(() => false, () => true);
  if (!shippedEncodeRejected || !candidateEncodeRejected ||
    !shippedCreateRejected || !candidateCreateRejected) {
    throw new Error("OPT-0024 post-destroy rejection failed");
  }
  return Object.freeze({
    tested: true,
    shippedEncodeRejected,
    candidateEncodeRejected,
    shippedCreateRejected,
    candidateCreateRejected,
  });
}

async function buildSourceAuthority(
  topology: readonly Opt0024Operation[],
): Promise<Readonly<Record<string, unknown>>> {
  const sources = Object.freeze({
    fixed32CoreSourceSha256: await sha256Text(fixed32CoreSource),
    candidateCoreSourceSha256: await sha256Text(candidateCoreSource),
    conv1dCoreSourceSha256: await sha256Text(conv1dCoreSource),
    decoderCoreSourceSha256: await sha256Text(decoderCoreSource),
  });
  if (sources.fixed32CoreSourceSha256 !== FIXED32_CORE_SOURCE_SHA256 ||
    sources.candidateCoreSourceSha256 !== CANDIDATE_CORE_SOURCE_SHA256 ||
    sources.conv1dCoreSourceSha256 !== CONV1D_CORE_SOURCE_SHA256 ||
    sources.decoderCoreSourceSha256 !== DECODER_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0024 rejected unauthenticated source bytes");
  }
  const generated = await Promise.all(topology.map(async (operation) => {
    const shipped = aceFp16VaeConv1dSubgroupWgsl(
      operation.shape,
      true,
      "float16",
    );
    const candidate = aceOpt0024VaeConv1dDirectDot4SubgroupWgsl(
      operation.shape,
      true,
      "float16",
    );
    return Object.freeze({
      label: operation.label,
      shipped: await sha256Text(shipped),
      candidate: await sha256Text(candidate),
    });
  }));
  if (OPT_0024_EXPECTED_GENERATED_SHADER_SHA256.length !== topology.length ||
    generated.some((entry, index) => {
      const expected = OPT_0024_EXPECTED_GENERATED_SHADER_SHA256[index];
      return expected === undefined || entry.label !== expected.label ||
        entry.shipped !== expected.shipped ||
        entry.candidate !== expected.candidate;
    })) {
    throw new Error("OPT-0024 generated shader identity changed");
  }
  const longCoverage = [448, 512, 340].map((inputFrames) => {
    const operations = buildOpt0024Topology(inputFrames);
    return Object.freeze({
      inputFrames,
      generatedCandidateShaderCount: operations.length,
      conv1GraphRangeCount: operations.find(({ label }) => label === "conv1")
        ?.ranges.length ?? 0,
      candidateAggregateSha256: "runtime-static-contract-pinned",
    });
  });
  return Object.freeze({
    registrationCommit: REGISTRATION_COMMIT,
    registrationRecordSha256: REGISTRATION_RECORD_SHA256,
    ...sources,
    shippedKernelId: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
    candidateKernelId:
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
    generatedShaderCountPerArm: generated.length,
    generatedShaders: Object.freeze(generated),
    longWindowCompileAndCorrectnessCoverage: Object.freeze(longCoverage),
    longConv1RangeCount: 24,
    longConv1TimingWeight: 0,
  });
}

function requiredDeviceLimits(
  adapter: GPUAdapter,
  topology: readonly Opt0024Operation[],
): Record<string, number> {
  let maximumBuffer = 4;
  let maximumStorageBinding = 4;
  let maximumDispatch = 1;
  for (const operation of topology) {
    const plan = planAceFp16VaeConv1d(operation.shape, "float16");
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
      plan.biasBindingBytes,
      STORAGE_GUARD_BYTES + plan.outputBindingBytes + STORAGE_GUARD_BYTES,
    );
    for (const range of operation.ranges) {
      const planned = planAceOpt0024VaeConv1dDirectDot4SubgroupRange(
        plan,
        range,
      );
      maximumDispatch = Math.max(
        maximumDispatch,
        planned.workgroupsX,
        planned.workgroupsY,
      );
    }
  }
  const requested = {
    maxBufferSize: maximumBuffer,
    maxStorageBufferBindingSize: maximumStorageBinding,
    maxUniformBufferBindingSize: RANGE_CONTROL_BYTES,
    maxComputeInvocationsPerWorkgroup: 128,
    maxComputeWorkgroupSizeX: 128,
    maxComputeWorkgroupStorageSize: 0,
    maxComputeWorkgroupsPerDimension: maximumDispatch,
  };
  for (const [name, minimum] of Object.entries(requested)) {
    const actual = Number(adapter.limits[name as keyof GPUSupportedLimits]);
    if (!Number.isFinite(actual) || actual < minimum) {
      throw new RangeError(
        `OPT-0024 adapter ${name}=${actual} is below ${minimum}`,
      );
    }
  }
  return requested;
}

function requireAdapter(
  adapter: GPUAdapter,
  topology: readonly Opt0024Operation[],
): void {
  if (!adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups")) {
    throw new Error("OPT-0024 requires adapter shader-f16 and subgroups");
  }
  if (adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32) {
    throw new Error("OPT-0024 requires fixed 32-lane subgroups");
  }
  if (adapter.limits.minStorageBufferOffsetAlignment > STORAGE_GUARD_BYTES) {
    throw new Error("OPT-0024 storage guard is below adapter alignment");
  }
  requiredDeviceLimits(adapter, topology);
}

function environmentReceipt(adapter: GPUAdapter, device: GPUDevice) {
  return Object.freeze({
    userAgent: navigator.userAgent,
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

function createControls(
  device: GPUDevice,
  tracker: BufferTracker,
  topology: readonly Opt0024Operation[],
): Readonly<{ buffer: GPUBuffer; offsets: ReadonlyMap<string, number> }> {
  const alignment = Number(device.limits.minUniformBufferOffsetAlignment);
  if (!Number.isSafeInteger(alignment) || alignment < RANGE_CONTROL_BYTES ||
    !Number.isInteger(Math.log2(alignment))) {
    throw new RangeError("OPT-0024 uniform alignment is invalid");
  }
  const entries = topology.flatMap((operation) =>
    operation.probes.map((probe) => Object.freeze({
      key: probeKey(operation.label, probe.id),
      base: probe.base,
      count: probe.count,
    }))
  );
  if (entries.length !== C300_PROBE_COUNT) {
    throw new Error("OPT-0024 range control count changed");
  }
  const buffer = tracker.create(device, {
    label: "opt-0024-range-controls",
    size: entries.length * alignment,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  const words = new Uint32Array(buffer.getMappedRange());
  const offsets = new Map<string, number>();
  for (const [index, entry] of entries.entries()) {
    const byteOffset = index * alignment;
    const wordOffset = byteOffset / 4;
    words[wordOffset] = entry.base;
    words[wordOffset + 1] = entry.count;
    if (offsets.has(entry.key)) {
      throw new Error(`OPT-0024 duplicate control ${entry.key}`);
    }
    offsets.set(entry.key, byteOffset);
  }
  tracker.unmap(buffer);
  return Object.freeze({ buffer, offsets });
}

function createPeriodicU16Buffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  bytes: number,
  pattern: Uint16Array,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = tracker.create(device, {
    label,
    size: bytes,
    usage,
    mappedAtCreation: true,
  });
  try {
    fillPeriodic(new Uint16Array(buffer.getMappedRange()), pattern);
    tracker.unmap(buffer);
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") tracker.unmap(buffer);
    tracker.destroy(buffer);
    throw error;
  }
}

function fillPeriodic(destination: Uint16Array, pattern: Uint16Array): void {
  if (destination.length < 1 || pattern.length < 1) {
    throw new RangeError("OPT-0024 periodic upload geometry changed");
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
    throw new RangeError("OPT-0024 U32 pattern write is unaligned");
  }
  const words = new Uint32Array(bytes / 4);
  words.fill(value);
  device.queue.writeBuffer(buffer, offset, words);
}

function everyU32(words: Uint32Array, expected: number): boolean {
  for (const word of words) if (word !== expected) return false;
  return true;
}

function outputRangeBinding(
  controls: GPUBuffer,
  offset: number,
  base: number,
  count: number,
): AceVaeOutputRangeBinding {
  return Object.freeze({
    base,
    count,
    control: Object.freeze({
      buffer: controls,
      offset,
      size: RANGE_CONTROL_BYTES,
    }),
  });
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function guardedOutputBinding(
  buffer: GPUBuffer,
  plan: AceFp16VaeConv1dPlan,
): GPUBufferBinding {
  return Object.freeze({
    buffer,
    offset: STORAGE_GUARD_BYTES,
    size: plan.outputBindingBytes,
  });
}

function probeKey(label: string, id: Opt0024ProbeId): string {
  return `${label}:${id}`;
}

function sameRange(
  left: EncodableDispatch["outputRange"],
  right: EncodableDispatch["outputRange"],
): boolean {
  return left.base === right.base && left.count === right.count &&
    left.workgroupsX === right.workgroupsX &&
    left.workgroupsY === right.workgroupsY;
}

function compactOperation(operation: Opt0024Operation) {
  return Object.freeze({
    ordinal: operation.ordinal,
    operationIndex: operation.operationIndex,
    label: operation.label,
    shape: operation.shape,
    outputElements: operation.outputElements,
    exactGraphRangeCount: operation.ranges.length,
    probes: operation.probes,
  });
}

function median4(samples: readonly number[]): number {
  if (samples.length !== TIMING_SAMPLES_PER_ARM_TIER ||
    samples.some((sample) => !Number.isFinite(sample) || sample <= 0)) {
    throw new RangeError(
      "OPT-0024 requires four finite positive samples per arm/tier",
    );
  }
  const sorted = [...samples].sort((left, right) => left - right);
  return (sorted[1]! + sorted[2]!) / 2;
}

function requireNoGpuFailures(
  uncapturedErrors: readonly string[],
  deviceLosses: readonly string[],
  phase: string,
): void {
  if (uncapturedErrors.length !== 0 || deviceLosses.length !== 0) {
    throw new Error(
      `OPT-0024 ${phase} GPU failure: ` +
        JSON.stringify({ uncapturedErrors, deviceLosses }),
    );
  }
}

function fieldParameters(selector: string): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    `${selector} input[name]`,
  )) parameters.set(input.name, input.value);
  return parameters;
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.length === 0) {
    throw new Error(`OPT-0024 field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0024 field ${name} is not finite`);
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

function requireElement<ElementType extends Element>(
  selector: string,
): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing OPT-0024 element ${selector}`);
  return element;
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

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0024-vae-k7-direct-subgroup-fp16-dot4-c300-v1",
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

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, 0));
}

async function settlePostDrainEvents(): Promise<void> {
  await yieldToBrowser();
  await yieldToBrowser();
}
