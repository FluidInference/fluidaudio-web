/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import fixed32CoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.ts?raw";
import packed16x64CoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-subgroup.ts?raw";
import cooperativeCoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-cooperative-dot4.ts?raw";
import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
  AceFp16VaeConv1dSubgroupKernel,
  aceFp16VaeConv1dSubgroupWgsl,
  planAceFp16VaeConv1dSubgroupRange,
} from "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import {
  ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_REPACK_KERNEL_ID,
  AceOpt0014VaeConv1dPackedKioSubgroupKernel,
  aceOpt0014VaeConv1dPackedKioRepackWgsl,
  planAceOpt0014VaeConv1dPackedKioWeight,
  type AceOpt0014VaeConv1dRepackDispatch,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-subgroup.js";
import {
  ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID,
  AceOpt0017VaeConv1dCooperativeDot4Kernel,
  aceOpt0017VaeConv1dCooperativeDot4Wgsl,
  planAceOpt0017VaeConv1dCooperativeDot4Range,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-cooperative-dot4.js";
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

export type Opt0017RepresentativeArm = "fixed32" | "cooperativeDot4";
export type Opt0017CandidateArm = "cooperativeDot4";
export type Opt0017ProbeId = "first" | "interior" | "tail";
type FullSequenceArm = "fixed32" | "winner";

interface ExactRange {
  readonly rangeIndex: number;
  readonly base: number;
  readonly count: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
}

export interface Opt0017C300Operation {
  readonly operationIndex: number;
  readonly label: string;
  readonly shape: AceVaeConv1dShape;
  readonly outputStorage: AceFp16VaeConv1dOutputStorage;
  readonly outputElements: number;
  readonly ranges: readonly ExactRange[];
}

export interface Opt0017RepresentativeProbe extends ExactRange {
  readonly id: Opt0017ProbeId;
}

export interface Opt0017Representative {
  readonly id: string;
  readonly tier: "c1024" | "c512" | "c256" | "c128";
  readonly operation: Opt0017C300Operation;
  readonly weight: number;
  readonly probes: readonly Opt0017RepresentativeProbe[];
  readonly timingProbe: Opt0017RepresentativeProbe;
}

export interface Opt0017ThermalGate {
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

export interface Opt0017RepresentativeTimingInput {
  readonly representativeId: string;
  readonly tier: Opt0017Representative["tier"];
  readonly weight: number;
  readonly samples: Readonly<Record<Opt0017RepresentativeArm, readonly number[]>>;
}

export interface Opt0017FullSequenceRoundInput {
  readonly order: readonly FullSequenceArm[];
  readonly fixed32Milliseconds: number;
  readonly winnerConvolutionMilliseconds: number;
  readonly repackMilliseconds: number;
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
  readonly verificationExecutions: number;
  readonly verification: Readonly<Record<string, unknown>>;
}

interface PreparedRepresentative {
  readonly representative: Opt0017Representative;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly weight: PreparedWeight;
  readonly correctnessDispatches: Readonly<
    Record<Opt0017ProbeId, Readonly<Record<Opt0017RepresentativeArm, EncodableDispatch>>>
  >;
  readonly timingDispatches: Readonly<
    Record<Opt0017RepresentativeArm, EncodableDispatch>
  >;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface SharedResources {
  readonly input: GPUBuffer;
  readonly bias: GPUBuffer;
  readonly outputs: Readonly<Record<Opt0017RepresentativeArm, GPUBuffer>>;
  readonly stageF16Prefill: GPUBuffer;
  readonly repackPrefill: GPUBuffer;
  readonly controls: GPUBuffer;
  readonly controlOffsets: ReadonlyMap<string, number>;
  readonly maximumOutputBindingBytes: number;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly topology: readonly Opt0017C300Operation[];
  readonly representatives: readonly PreparedRepresentative[];
  readonly weights: Map<string, PreparedWeight>;
  readonly shared: SharedResources;
  readonly fixed32Kernel: AceFp16VaeConv1dSubgroupKernel;
  readonly repackKernel: AceOpt0014VaeConv1dPackedKioSubgroupKernel;
  readonly cooperativeKernel: AceOpt0017VaeConv1dCooperativeDot4Kernel;
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly repackCorrectness: Readonly<Record<string, unknown>>;
  readonly sourceAuthority: Readonly<Record<string, unknown>>;
  readonly preparedCompletedAtEpochMilliseconds: number;
  readonly updateProgress: (message: string) => void;
  destroy(): Readonly<Record<string, unknown>>;
}

interface FullPreparedOperation {
  readonly topology: Opt0017C300Operation;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly weight: PreparedWeight;
  readonly fixed32: readonly EncodableDispatch[];
  readonly winner: readonly EncodableDispatch[];
}

interface FullPreparedGate {
  readonly winner: Opt0017CandidateArm;
  readonly operations: readonly FullPreparedOperation[];
  readonly controls: GPUBuffer;
  readonly fixed32Output: GPUBuffer;
  readonly winnerOutput: GPUBuffer;
  readonly repackCorrectness: Readonly<Record<string, unknown>>;
  readonly commandBufferCountPerArm: number;
  readonly dispatchCountPerArm: number;
}

interface ReadbackArm {
  readonly words: Uint16Array;
  readonly nonFiniteCount: number;
  readonly prefillQNaNCount: number;
  readonly guardsUntouched: boolean;
  readonly sha256: string;
}

const EXPERIMENT_ID = "OPT-0017" as const;
const CORE_COMMIT =
  "b83f4fe94d56787ddb980629ea6f41804543ca69" as const;
const COOPERATIVE_CORE_SOURCE_SHA256 =
  "83987aa9b16e05a5b6f45c25ebfe33ed08bfb82ae94bd9dfb4fb624c625407b8" as const;
const PACKED_16X64_CORE_SOURCE_SHA256 =
  "802cb0ad1d2c57c0cc51cbd4a7c88632e00d543b526f2ed0b94e9fc393a3d8d8" as const;
const FIXED32_CORE_SOURCE_SHA256 =
  "7d218516d6b2c8d6e3332a53101be5fdeae1142096c442433915bfa58941ce32" as const;
const C300_INPUT_FRAMES = 300;
const C300_OPERATION_COUNT = 16;
const C300_EXACT_RANGE_COUNT = 2_399;
const C300_OUTPUT_WORD_COUNT = 424_550_400;
const REPRESENTATIVE_TIER_WEIGHT = 2_397;
const OMITTED_CONV1_RANGE_WEIGHT = 2;
const OMITTED_FINAL_CONV2_RANGE_WEIGHT = 5;
const REPRESENTATIVE_CORRECTNESS_PROBE_COUNT = 12;
const REPRESENTATIVE_CORRECTNESS_EXECUTIONS = 2;
const REPRESENTATIVE_COMPARED_U16_WORDS = 2_392_064;
const SELECTED_FOUR_UNIQUE_REPACK_U16_WORDS = 9_748_480;
const SELECTED_FOUR_REPACK_WORKGROUPS = 19_040;
const ALL_16_UNIQUE_REPACK_U16_WORDS = 30_507_008;
const FULL_REPACK_COMPARED_U16_WORDS = 40_255_488;
const ALL_16_PACKED_WEIGHT_BYTES = 61_014_016;
const ALL_16_REPACK_WORKGROUPS = 59_584;
const REPRESENTATIVE_QUALIFYING_SPEEDUP = 1.75;
const FULL_SEQUENCE_QUALIFYING_SPEEDUP = 2.0;
const REPRESENTATIVE_SAMPLES_PER_ARM = 4;
const MAX_QUANTA_PER_COMMAND_BUFFER = 8;
const FULL_SEQUENCE_COMMAND_BUFFERS_PER_ARM = 307;
const NRMSE_LIMIT = 0.001;
const SNR_MINIMUM_DB = 60;
const PEARSON_MINIMUM = 0.99999;
const RELATIVE_MAXIMUM_ABSOLUTE_ERROR_LIMIT = 0.01;
const RANGE_CONTROL_BYTES = 16;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const ADJACENT_CANARY_U32 = 0x5aa5_3cc3;
const OUTPUT_PREFILL_QNAN_F16 = 0x7e55;
const REPACK_PREFILL_QNAN_F16 = 0x7e6d;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const THERMAL_POLL_MILLISECONDS = 1_000;
const MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS = 1_250;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 5_000;

const REPRESENTATIVE_ARMS = Object.freeze([
  "fixed32",
  "cooperativeDot4",
] as const);
const CANDIDATE_ARMS = Object.freeze(["cooperativeDot4"] as const);
const SELECTED_FOUR_REPACK_SHA256 = Object.freeze({
  "block-0-res-1-conv1":
    "a5b263af7413fdcddb14d0cf4468360074e614f8cab1be22b4ce39e12633059b",
  "block-1-res-2-conv1":
    "ae8c34cb62c1b3c6a512328640377b7980971d5b8a508bc7bbb7403c6ed11564",
  "block-2-res-1-conv1":
    "baa781e62b263e89486a05b566635652db893da0064e6d9e02f9d01b97705c19",
  "block-4-res-3-conv1":
    "fd9eb42c7e1a36829ea952af49de1c20e2755354e9263bab53a04233aa2c8f3f",
} as const);
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

const EXPECTED_C300_TOPOLOGY = Object.freeze([
  expectedOperation("conv1", 300, 64, 2_048, 1, 3, 2),
  expectedOperation("block-0-res-1-conv1", 3_000, 1_024, 1_024, 1, 3, 94),
  expectedOperation("block-0-res-2-conv1", 3_000, 1_024, 1_024, 3, 9, 94),
  expectedOperation("block-0-res-3-conv1", 3_000, 1_024, 1_024, 9, 27, 94),
  expectedOperation("block-1-res-1-conv1", 18_000, 512, 512, 1, 3, 141),
  expectedOperation("block-1-res-2-conv1", 18_000, 512, 512, 3, 9, 141),
  expectedOperation("block-1-res-3-conv1", 18_000, 512, 512, 9, 27, 141),
  expectedOperation("block-2-res-1-conv1", 72_000, 256, 256, 1, 3, 141),
  expectedOperation("block-2-res-2-conv1", 72_000, 256, 256, 3, 9, 141),
  expectedOperation("block-2-res-3-conv1", 72_000, 256, 256, 9, 27, 141),
  expectedOperation("block-3-res-1-conv1", 288_000, 128, 128, 1, 3, 141),
  expectedOperation("block-3-res-2-conv1", 288_000, 128, 128, 3, 9, 141),
  expectedOperation("block-3-res-3-conv1", 288_000, 128, 128, 9, 27, 141),
  expectedOperation("block-4-res-1-conv1", 576_000, 128, 128, 1, 3, 282),
  expectedOperation("block-4-res-2-conv1", 576_000, 128, 128, 3, 9, 282),
  expectedOperation("block-4-res-3-conv1", 576_000, 128, 128, 9, 27, 282),
]);

const REPRESENTATIVE_SPECS = Object.freeze([
  representativeSpec("c1024-d1", "c1024", "block-0-res-1-conv1", 47, 282,
    [[0, 32_768], [1_540_096, 32_768], [3_047_424, 24_576]]),
  representativeSpec("c512-d3", "c512", "block-1-res-2-conv1", 70, 423,
    [[0, 65_536], [4_587_520, 65_536], [9_175_040, 40_960]]),
  representativeSpec("c256-d1", "c256", "block-2-res-1-conv1", 70, 423,
    [[0, 131_072], [9_175_040, 131_072], [18_350_080, 81_920]]),
  representativeSpec("c128-d9", "c128", "block-4-res-3-conv1", 141, 1_269,
    [[0, 262_144], [36_962_304, 262_144], [73_662_464, 65_536]]),
]);

export function buildOpt0017C300Topology(): readonly Opt0017C300Operation[] {
  const graph = planAceVaeDecoder(C300_INPUT_FRAMES);
  const cooperative = planAceVaeDecoderQuanta(graph);
  const operations = graph.operations
    .map((operation, operationIndex) => ({ operation, operationIndex }))
    .filter((entry): entry is Readonly<{
      operation: AceVaeDecoderConvOperation;
      operationIndex: number;
    }> => entry.operation.kind === "conv1d" &&
      entry.operation.shape.kernelSize === 7 &&
      entry.operation.bias !== undefined)
    .map(({ operation, operationIndex }, ordinal) => {
      const outputStorage = "float16" as const;
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
          return Object.freeze({
            rangeIndex,
            base: primitive.outputBase,
            count: primitive.outputCount,
            firstOutputRow: primitive.outputBase / plan.outputChannels,
            outputRowCount: primitive.outputCount / plan.outputChannels,
          });
        });
      const result = Object.freeze({
        operationIndex,
        label: operation.label,
        shape: operation.shape,
        outputStorage,
        outputElements: plan.outputElements,
        ranges: Object.freeze(ranges),
      });
      assertExpectedOperation(result, EXPECTED_C300_TOPOLOGY[ordinal]);
      return result;
    });
  const rangeCount = operations.reduce(
    (sum, operation) => sum + operation.ranges.length,
    0,
  );
  const outputWords = operations.reduce(
    (sum, operation) => sum + operation.outputElements,
    0,
  );
  if (operations.length !== C300_OPERATION_COUNT ||
    rangeCount !== C300_EXACT_RANGE_COUNT ||
    outputWords !== C300_OUTPUT_WORD_COUNT) {
    throw new Error("OPT-0017 exact C300 topology totals changed");
  }
  return Object.freeze(operations);
}

export function buildOpt0017Representatives(
  topology: readonly Opt0017C300Operation[] = buildOpt0017C300Topology(),
): readonly Opt0017Representative[] {
  const representatives = REPRESENTATIVE_SPECS.map((spec) => {
    const operation = topology.find(({ label }) => label === spec.label);
    if (operation === undefined) {
      throw new Error(`OPT-0017 representative ${spec.label} is missing`);
    }
    const [firstExpected, interiorExpected, tailExpected] = spec.probes;
    const first = operation.ranges[0];
    const interior = operation.ranges[spec.timingRangeIndex];
    const tail = operation.ranges.at(-1);
    if (first === undefined || interior === undefined || tail === undefined ||
      first.base !== firstExpected![0] || first.count !== firstExpected![1] ||
      interior.base !== interiorExpected![0] ||
      interior.count !== interiorExpected![1] ||
      tail.base !== tailExpected![0] || tail.count !== tailExpected![1]) {
      throw new Error(`OPT-0017 representative ${spec.label} range changed`);
    }
    const probes = Object.freeze([
      Object.freeze({ ...first, id: "first" as const }),
      Object.freeze({ ...interior, id: "interior" as const }),
      Object.freeze({ ...tail, id: "tail" as const }),
    ]);
    return Object.freeze({
      id: spec.id,
      tier: spec.tier,
      operation,
      weight: spec.weight,
      probes,
      timingProbe: probes[1]!,
    });
  });
  const totalWeight = representatives.reduce((sum, item) => sum + item.weight, 0);
  const probeCount = representatives.reduce((sum, item) =>
    sum + item.probes.length, 0);
  if (totalWeight !== REPRESENTATIVE_TIER_WEIGHT ||
    probeCount !== REPRESENTATIVE_CORRECTNESS_PROBE_COUNT) {
    throw new Error("OPT-0017 representative coverage changed");
  }
  return Object.freeze(representatives);
}

export function buildOpt0017RepresentativeOrders(): readonly Readonly<{
  tierIndex: number;
  sampleIndex: number;
  rotation: number;
  order: readonly Opt0017RepresentativeArm[];
}>[] {
  return Object.freeze(Array.from({ length: REPRESENTATIVE_SPECS.length },
    (_, tierIndex) => Array.from(
      { length: REPRESENTATIVE_SAMPLES_PER_ARM },
      (_, sampleIndex) => {
        const rotation = (tierIndex + sampleIndex) % REPRESENTATIVE_ARMS.length;
        return Object.freeze({
          tierIndex,
          sampleIndex,
          rotation,
          order: Object.freeze([
            ...REPRESENTATIVE_ARMS.slice(rotation),
            ...REPRESENTATIVE_ARMS.slice(0, rotation),
          ]),
        });
      },
    )).flat());
}

export function summarizeOpt0017RepresentativeTiming(
  inputs: readonly Opt0017RepresentativeTimingInput[],
): Readonly<Record<string, unknown>> {
  if (inputs.length !== REPRESENTATIVE_SPECS.length) {
    throw new Error("OPT-0017 requires exactly four representative timings");
  }
  const weightedMilliseconds = Object.fromEntries(
    REPRESENTATIVE_ARMS.map((arm) => [arm, 0]),
  ) as Record<Opt0017RepresentativeArm, number>;
  const strata = inputs.map((input, index) => {
    const expectedValue = REPRESENTATIVE_SPECS[index];
    if (expectedValue === undefined ||
      input.representativeId !== expectedValue.id ||
      input.tier !== expectedValue.tier || input.weight !== expectedValue.weight) {
      throw new Error("OPT-0017 representative timing identity changed");
    }
    const medians = {} as Record<Opt0017RepresentativeArm, number>;
    for (const arm of REPRESENTATIVE_ARMS) {
      const value = median4(input.samples[arm]);
      medians[arm] = value;
      weightedMilliseconds[arm] += value * input.weight;
    }
    return Object.freeze({
      representativeId: input.representativeId,
      tier: input.tier,
      weight: input.weight,
      samples: input.samples,
      medians: Object.freeze(medians),
    });
  });
  const speedups = Object.fromEntries(CANDIDATE_ARMS.map((arm) => [
    arm,
    weightedMilliseconds.fixed32 / weightedMilliseconds[arm],
  ])) as Record<Opt0017CandidateArm, number>;
  return Object.freeze({
    representedRangeWeight: REPRESENTATIVE_TIER_WEIGHT,
    omittedConv1RangeWeight: OMITTED_CONV1_RANGE_WEIGHT,
    omittedFinalConv2RangeWeight: OMITTED_FINAL_CONV2_RANGE_WEIGHT,
    fixed32WeightedMilliseconds: weightedMilliseconds.fixed32,
    candidateWeightedMilliseconds: Object.freeze(Object.fromEntries(
      CANDIDATE_ARMS.map((arm) => [arm, weightedMilliseconds[arm]]),
    )),
    speedups: Object.freeze(speedups),
    strata: Object.freeze(strata),
  });
}

export function selectOpt0017RepresentativeWinner(
  summary: Readonly<Record<string, unknown>>,
): Readonly<Record<string, unknown>> {
  const speedups = summary["speedups"] as Readonly<Record<
    Opt0017CandidateArm,
    number
  >>;
  const qualifiers = CANDIDATE_ARMS.filter((arm) =>
    Number.isFinite(speedups[arm]) &&
    speedups[arm] >= REPRESENTATIVE_QUALIFYING_SPEEDUP
  );
  if (qualifiers.length === 0) {
    return Object.freeze({
      status: "negative-stop-no-representative-qualifier",
      selectedWinner: null,
      qualifyingArms: Object.freeze([]),
      threshold: REPRESENTATIVE_QUALIFYING_SPEEDUP,
    });
  }
  const selectedWinner = qualifiers[0]!;
  return Object.freeze({
    status: "representative-qualifier-selected",
    selectedWinner,
    qualifyingArms: Object.freeze(qualifiers),
    selectedSpeedup: speedups[selectedWinner],
    threshold: REPRESENTATIVE_QUALIFYING_SPEEDUP,
  });
}

export function summarizeOpt0017FullSequence(
  winner: Opt0017CandidateArm,
  rounds: readonly Opt0017FullSequenceRoundInput[],
): Readonly<Record<string, unknown>> {
  if (rounds.length !== 2 ||
    rounds[0]!.order.join("-") !== "fixed32-winner" ||
    rounds[1]!.order.join("-") !== "winner-fixed32") {
    throw new Error("OPT-0017 full sequence requires balanced AB then BA");
  }
  const normalized = rounds.map((round) => {
    for (const value of [
      round.fixed32Milliseconds,
      round.winnerConvolutionMilliseconds,
      round.repackMilliseconds,
    ]) requireFinitePositive(value, "OPT-0017 full-sequence timing");
    const winnerTotalMilliseconds = round.winnerConvolutionMilliseconds +
      round.repackMilliseconds;
    return Object.freeze({
      order: round.order,
      fixed32Milliseconds: round.fixed32Milliseconds,
      winnerConvolutionMilliseconds: round.winnerConvolutionMilliseconds,
      repackMilliseconds: round.repackMilliseconds,
      winnerTotalMilliseconds,
      winnerPairWon: winnerTotalMilliseconds < round.fixed32Milliseconds,
    });
  });
  const fixed32TotalMilliseconds = normalized.reduce((sum, round) =>
    sum + round.fixed32Milliseconds, 0);
  const winnerTotalMilliseconds = normalized.reduce((sum, round) =>
    sum + round.winnerTotalMilliseconds, 0);
  const aggregateSpeedup = fixed32TotalMilliseconds / winnerTotalMilliseconds;
  const bothOrderWins = normalized.every((round) => round.winnerPairWon);
  const passed = bothOrderWins &&
    aggregateSpeedup >= FULL_SEQUENCE_QUALIFYING_SPEEDUP;
  return Object.freeze({
    winner,
    rounds: Object.freeze(normalized),
    fixed32TotalMilliseconds,
    winnerTotalMilliseconds,
    aggregateSpeedup,
    bothOrderWins,
    threshold: FULL_SEQUENCE_QUALIFYING_SPEEDUP,
    passed,
    decision: passed
      ? "positive-full-sequence-qualifier"
      : "negative-stop-full-sequence",
  });
}

export function parseOpt0017ThermalGate(
  parameters: URLSearchParams,
  preparedCompletedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0017ThermalGate {
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
    !Number.isSafeInteger(observationCount) || observationCount < 31 ||
    pollMilliseconds !== THERMAL_POLL_MILLISECONDS ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS ||
    nonNominalObservationCount !== 0 ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    startedAtEpochMilliseconds < preparedCompletedAtEpochMilliseconds ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS) {
    throw new Error("OPT-0017 thermal gate is incomplete, stale, or non-nominal");
  }
  return Object.freeze({
    source,
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
    progress.textContent = "running balanced representative sweep";
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

async function prepareGate(
  updateProgress: (message: string) => void,
): Promise<PreparedGate> {
  const topology = buildOpt0017C300Topology();
  const representatives = buildOpt0017Representatives(topology);
  const sourceAuthority = await buildSourceAuthority(topology);
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter, topology);
  const device = await adapter.requestDevice({
    label: "ace-opt-0017-vae-k7-cooperative-dot4-ab-device",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: requiredDeviceLimits(adapter, topology),
  });
  const capability = Object.freeze({
    subgroupMinSize: 32 as const,
    subgroupMaxSize: 32 as const,
  });
  const tracker = new BufferTracker();
  const fixed32Kernel = AceFp16VaeConv1dSubgroupKernel.create(
    device,
    capability,
  );
  const repackKernel =
    AceOpt0014VaeConv1dPackedKioSubgroupKernel.create(device, capability);
  const cooperativeKernel =
    AceOpt0017VaeConv1dCooperativeDot4Kernel.create(device);
  let destroyed = false;
  const destroy = (): Readonly<Record<string, unknown>> => {
    if (destroyed) {
      return Object.freeze({ ...tracker.receipt(), idempotent: true });
    }
    destroyed = true;
    fixed32Kernel.destroy();
    repackKernel.destroy();
    cooperativeKernel.destroy();
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
    updateProgress("allocating bounded representative resources");
    const shared = createSharedResources(device, tracker, topology,
      representatives);
    const weights = new Map<string, PreparedWeight>();
    const preparedRepresentatives: PreparedRepresentative[] = [];
    const repackCases: unknown[] = [];
    let uniqueRepackedU16WordCount = 0;
    let comparedRepackedU16WordCount = 0;
    let repackWorkgroupCount = 0;
    let selectedUniqueU16WordCount = 0;
    let selectedRepackWorkgroupCount = 0;
    const selectedLabels = new Set(representatives.map(
      (representative) => representative.operation.label,
    ));
    for (const [index, operation] of topology.entries()) {
      updateProgress(
        `repack verification ${index + 1}/${topology.length}: ` +
          operation.label,
      );
      const selected = selectedLabels.has(operation.label);
      const weight = await prepareWeight(
        device,
        tracker,
        repackKernel,
        shared.repackPrefill,
        operation,
        selected ? 2 : 1,
      );
      if (selected) {
        const acceptedRepackSha256 = SELECTED_FOUR_REPACK_SHA256[
          operation.label as keyof typeof SELECTED_FOUR_REPACK_SHA256
        ];
        if (acceptedRepackSha256 === undefined ||
          weight.verification["sha256"] !== acceptedRepackSha256) {
          throw new Error(`${operation.label} repack identity changed`);
        }
        selectedUniqueU16WordCount += weight.uniqueU16Words;
        selectedRepackWorkgroupCount += weight.repack.plan.repackWorkgroups;
      }
      weights.set(operation.label, weight);
      repackCases.push(weight.verification);
      uniqueRepackedU16WordCount += weight.uniqueU16Words;
      comparedRepackedU16WordCount += weight.uniqueU16Words *
        weight.verificationExecutions;
      repackWorkgroupCount += weight.repack.plan.repackWorkgroups;
      await yieldToBrowser();
    }
    for (const representative of representatives) {
      const weight = weights.get(representative.operation.label);
      if (weight === undefined) {
        throw new Error(`${representative.operation.label} weight missing`);
      }
      const prepared = await prepareRepresentative(
        fixed32Kernel,
        cooperativeKernel,
        shared,
        representative,
        weight,
      );
      preparedRepresentatives.push(prepared);
      await yieldToBrowser();
    }
    if (uniqueRepackedU16WordCount !== ALL_16_UNIQUE_REPACK_U16_WORDS ||
      comparedRepackedU16WordCount !== FULL_REPACK_COMPARED_U16_WORDS ||
      repackWorkgroupCount !== ALL_16_REPACK_WORKGROUPS ||
      selectedUniqueU16WordCount !==
        SELECTED_FOUR_UNIQUE_REPACK_U16_WORDS ||
      selectedRepackWorkgroupCount !== SELECTED_FOUR_REPACK_WORKGROUPS) {
      throw new Error("OPT-0017 all-16 repack totals changed");
    }
    updateProgress("running two representative numerical executions");
    const correctness = await runRepresentativeCorrectness(
      device,
      tracker,
      shared,
      preparedRepresentatives,
    );
    const repackCorrectness = Object.freeze({
      operationCount: topology.length,
      selectedOperationCount: representatives.length,
      selectedUniqueU16WordCount,
      selectedRepackWorkgroupCount,
      uniqueU16WordCount: uniqueRepackedU16WordCount,
      comparedU16WordCount: comparedRepackedU16WordCount,
      repackWorkgroupCount,
      mismatchCount: 0,
      qNaNPrefillCompleteWrites: true,
      redzonesUntouched: true,
      deterministicRerunHashes: true,
      cases: Object.freeze(repackCases),
    });
    await device.queue.onSubmittedWorkDone();
    return Object.freeze({
      adapter,
      device,
      tracker,
      topology,
      representatives: Object.freeze(preparedRepresentatives),
      weights,
      shared,
      fixed32Kernel,
      repackKernel,
      cooperativeKernel,
      correctness,
      repackCorrectness,
      sourceAuthority,
      preparedCompletedAtEpochMilliseconds: Date.now(),
      updateProgress,
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
  topology: readonly Opt0017C300Operation[],
  representatives: readonly Opt0017Representative[],
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
  const maximumF16ProbeBytes = Math.max(...representatives.flatMap((item) =>
    item.probes.map((probe) => probe.count * 2)
  ));
  const maximumSelectedWeightBytes = Math.max(...representatives.map((item) =>
    planAceOpt0014VaeConv1dPackedKioWeight(item.operation.shape)
      .packedBindingBytes
  ));
  const input = createPeriodicU16Buffer(
    device,
    tracker,
    "opt-0017-shared-input",
    maximumInputBindingBytes,
    INPUT_PATTERN,
  );
  const bias = createPeriodicU16Buffer(
    device,
    tracker,
    "opt-0017-shared-bias",
    maximumBiasBindingBytes,
    BIAS_PATTERN,
  );
  const outputs = Object.freeze(Object.fromEntries(REPRESENTATIVE_ARMS.map(
    (arm) => [arm, createUninitializedBuffer(
      device,
      tracker,
      `opt-0017-${arm}-output`,
      STORAGE_GUARD_BYTES + maximumOutputBindingBytes + STORAGE_GUARD_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    )],
  ))) as Readonly<Record<Opt0017RepresentativeArm, GPUBuffer>>;
  const stageF16Prefill = createRepeatedU32Buffer(
    device,
    tracker,
    "opt-0017-stage-f16-qnan-prefill",
    maximumF16ProbeBytes,
    OUTPUT_PREFILL_QNAN_F16 | (OUTPUT_PREFILL_QNAN_F16 << 16),
  );
  const repackPrefill = createRepeatedU32Buffer(
    device,
    tracker,
    "opt-0017-all-16-repack-qnan-prefill",
    maximumSelectedWeightBytes,
    REPACK_PREFILL_QNAN_F16 | (REPACK_PREFILL_QNAN_F16 << 16),
  );
  const controls = createRepresentativeControls(device, tracker,
    representatives);
  return Object.freeze({
    input,
    bias,
    outputs,
    stageF16Prefill,
    repackPrefill,
    controls: controls.buffer,
    controlOffsets: controls.offsets,
    maximumOutputBindingBytes,
  });
}

async function prepareRepresentative(
  fixed32Kernel: AceFp16VaeConv1dSubgroupKernel,
  cooperativeKernel: AceOpt0017VaeConv1dCooperativeDot4Kernel,
  shared: SharedResources,
  representative: Opt0017Representative,
  weight: PreparedWeight,
): Promise<PreparedRepresentative> {
  const operation = representative.operation;
  const plan = planAceFp16VaeConv1d(operation.shape, operation.outputStorage);
  const correctnessDispatches = {} as Record<
    Opt0017ProbeId,
    Record<Opt0017RepresentativeArm, EncodableDispatch>
  >;
  for (const probe of representative.probes) {
    const controlOffset = shared.controlOffsets.get(
      representativeProbeKey(operation.label, probe.id),
    );
    if (controlOffset === undefined) {
      throw new Error(`${operation.label} ${probe.id} control missing`);
    }
    const range = outputRangeBinding(
      shared.controls,
      controlOffset,
      probe.base,
      probe.count,
    );
    correctnessDispatches[probe.id] = await createRepresentativeDispatches(
      fixed32Kernel,
      cooperativeKernel,
      shared,
      operation,
      plan,
      weight,
      range,
      `${operation.label}-${probe.id}`,
    );
  }
  const frozenCorrectness = Object.freeze(Object.fromEntries(
    representative.probes.map((probe) => [
      probe.id,
      Object.freeze(correctnessDispatches[probe.id]),
    ]),
  )) as Readonly<
    Record<Opt0017ProbeId, Readonly<Record<Opt0017RepresentativeArm,
      EncodableDispatch>>>
  >;
  return Object.freeze({
    representative,
    plan,
    weight,
    correctnessDispatches: frozenCorrectness,
    timingDispatches: frozenCorrectness.interior,
    correctness: Object.freeze({ pending: true }),
  });
}

async function createRepresentativeDispatches(
  fixed32Kernel: AceFp16VaeConv1dSubgroupKernel,
  cooperativeKernel: AceOpt0017VaeConv1dCooperativeDot4Kernel,
  shared: SharedResources,
  operation: Opt0017C300Operation,
  plan: AceFp16VaeConv1dPlan,
  weight: PreparedWeight,
  range: AceVaeOutputRangeBinding,
  label: string,
): Promise<Record<Opt0017RepresentativeArm, EncodableDispatch>> {
  const common = Object.freeze({
    input: binding(shared.input, plan.inputBindingBytes),
    bias: binding(shared.bias, plan.biasBindingBytes),
  });
  const result = {} as Record<Opt0017RepresentativeArm, EncodableDispatch>;
  result.fixed32 = await fixed32Kernel.createDispatch(
    `${label}-fixed32`,
    operation.shape,
    Object.freeze({
      ...common,
      weight: binding(weight.native, plan.weightBindingBytes),
      output: outputBinding(shared.outputs.fixed32, plan),
    }),
    operation.outputStorage,
    range,
  );
  result.cooperativeDot4 = await cooperativeKernel.createDispatch(
    `${label}-cooperative-dot4`,
    operation.shape,
    Object.freeze({
      ...common,
      packedWeight: weight.packed.binding,
      output: outputBinding(shared.outputs.cooperativeDot4, plan),
    }),
    operation.outputStorage,
    range,
  );
  assertRepresentativeDispatches(operation, plan, range, result);
  return result;
}

async function prepareWeight(
  device: GPUDevice,
  tracker: BufferTracker,
  repackKernel: AceOpt0014VaeConv1dPackedKioSubgroupKernel,
  repackPrefill: GPUBuffer,
  topology: Opt0017C300Operation,
  verificationExecutions: 1 | 2,
): Promise<PreparedWeight> {
  const plan = planAceOpt0014VaeConv1dPackedKioWeight(topology.shape);
  const native = tracker.create(device, {
    label: `${topology.label}-native-weight`,
    size: plan.nativeBindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const nativeWords = new Uint16Array(native.getMappedRange());
  for (let index = 0;
    index < plan.inputChannels * 7 * plan.outputChannels;
    index += 1) {
    nativeWords[index] = deterministicWeightBits(topology.operationIndex,
      index);
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
  const repack = await repackKernel.createRepackDispatch(
    `${topology.label}-repack`,
    topology.shape,
    Object.freeze({
      nativeWeight: binding(native, plan.nativeBindingBytes),
      packedWeight: packed.binding,
    }),
  );
  const executions: Readonly<Record<string, unknown>>[] = [];
  for (let execution = 0; execution < verificationExecutions; execution += 1) {
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
  if (verificationExecutions === 2 &&
    executions[0]!["sha256"] !== executions[1]!["sha256"]) {
    throw new Error(`${topology.label} repack rerun changed`);
  }
  return Object.freeze({
    native,
    packed,
    repack,
    uniqueU16Words: plan.packedWordCount * 2,
    verificationExecutions,
    verification: Object.freeze({
      label: topology.label,
      uniqueU16WordCount: plan.packedWordCount * 2,
      comparedU16WordCount:
        plan.packedWordCount * 2 * verificationExecutions,
      packedBytes: plan.packedBindingBytes,
      repackWorkgroups: plan.repackWorkgroups,
      sha256: executions[0]!["sha256"],
      mismatchCount: 0,
      qNaNPrefillCount: 0,
      redzonesUntouched: true,
      deterministicRerun: verificationExecutions === 2,
    }),
  });
}

async function executeAndVerifyRepack(
  device: GPUDevice,
  tracker: BufferTracker,
  repackPrefill: GPUBuffer,
  topology: Opt0017C300Operation,
  packed: GuardedBinding,
  repack: AceOpt0014VaeConv1dRepackDispatch,
  execution: number,
): Promise<Readonly<Record<string, unknown>>> {
  const plan = planAceOpt0014VaeConv1dPackedKioWeight(topology.shape);
  const guardedBytes = STORAGE_GUARD_BYTES + plan.packedBindingBytes +
    STORAGE_GUARD_BYTES;
  writeU32Pattern(device, packed.buffer, 0, STORAGE_GUARD_BYTES,
    STORAGE_GUARD_U32);
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
    const encoder = device.createCommandEncoder();
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
    const mappedRange = readback.getMappedRange();
    const prefix = new Uint32Array(mappedRange, 0, STORAGE_GUARD_BYTES / 4);
    const payload = new Uint16Array(
      mappedRange,
      STORAGE_GUARD_BYTES,
      plan.packedBindingBytes / 2,
    );
    const suffix = new Uint32Array(
      mappedRange,
      STORAGE_GUARD_BYTES + plan.packedBindingBytes,
      STORAGE_GUARD_BYTES / 4,
    );
    if (!everyU32(prefix, STORAGE_GUARD_U32) ||
      !everyU32(suffix, STORAGE_GUARD_U32)) {
      throw new Error(`${topology.label} repack redzone changed`);
    }
    let prefillQNaNCount = 0;
    let mismatchCount = 0;
    let firstMismatchIndex: number | null = null;
    for (let packedIndex = 0; packedIndex < payload.length;
      packedIndex += 1) {
      const actual = payload[packedIndex]!;
      if (actual === REPACK_PREFILL_QNAN_F16) prefillQNaNCount += 1;
      const outputChannel = packedIndex % plan.outputChannels;
      const kernelInput = Math.floor(packedIndex / plan.outputChannels);
      const inputChannel = kernelInput % plan.inputChannels;
      const kernel = Math.floor(kernelInput / plan.inputChannels);
      const nativeIndex =
        (outputChannel * 7 + kernel) * plan.inputChannels + inputChannel;
      if (actual === deterministicWeightBits(topology.operationIndex,
        nativeIndex)) continue;
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
      prefillQNaNCount,
      redzonesUntouched: true,
    });
  } finally {
    if (mapped) readback.unmap();
    tracker.destroy(readback);
  }
}

async function runRepresentativeCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  shared: SharedResources,
  preparedRepresentatives: readonly PreparedRepresentative[],
): Promise<Readonly<Record<string, unknown>>> {
  const cases: unknown[] = [];
  let comparedU16WordCount = 0;
  const aggregate = createNumericalAccumulator();
  const orders = Object.freeze([
    REPRESENTATIVE_ARMS,
    Object.freeze([...REPRESENTATIVE_ARMS].reverse()),
  ]);
  for (const prepared of preparedRepresentatives) {
    const probeReceipts: unknown[] = [];
    for (const probe of prepared.representative.probes) {
      const executions: Readonly<Record<string, unknown>>[] = [];
      const probeAggregate = createNumericalAccumulator();
      for (const [executionIndex, order] of orders.entries()) {
        executions.push(await executeRepresentativeProbe(
          device,
          tracker,
          shared,
          prepared,
          probe,
          order,
          executionIndex,
        ));
        const numerical = executions.at(-1)!["numerical"] as
          NumericalAccumulator;
        mergeNumericalAccumulators(probeAggregate, numerical);
        mergeNumericalAccumulators(aggregate, numerical);
        comparedU16WordCount += probe.count;
      }
      const firstHashes = executions[0]!["hashes"] as Readonly<Record<
        Opt0017RepresentativeArm,
        string
      >>;
      const secondHashes = executions[1]!["hashes"] as Readonly<Record<
        Opt0017RepresentativeArm,
        string
      >>;
      if (firstHashes.cooperativeDot4 !== secondHashes.cooperativeDot4) {
        throw new Error(
          `${prepared.representative.operation.label} ${probe.id} ` +
            "candidate rerun changed",
        );
      }
      const numerical = summarizeNumericalAccumulator(probeAggregate);
      assertOpt0017NumericalThresholds(
        numerical,
        `${prepared.representative.operation.label} ${probe.id}`,
      );
      probeReceipts.push(Object.freeze({
        id: probe.id,
        rangeIndex: probe.rangeIndex,
        base: probe.base,
        count: probe.count,
        outputWordType: "u16",
        armSha256: firstHashes,
        candidateExecutionSha256: Object.freeze([
          firstHashes.cooperativeDot4,
          secondHashes.cooperativeDot4,
        ]),
        comparedCandidateControlValues: probeAggregate.count,
        deterministicCandidateRerun: true,
        numerical,
      }));
    }
    cases.push(Object.freeze({
      representativeId: prepared.representative.id,
      operationLabel: prepared.representative.operation.label,
      shape: prepared.representative.operation.shape,
      outputStorage: prepared.representative.operation.outputStorage,
      probes: Object.freeze(probeReceipts),
    }));
    await yieldToBrowser();
  }
  if (comparedU16WordCount !== REPRESENTATIVE_COMPARED_U16_WORDS ||
    aggregate.count !== REPRESENTATIVE_COMPARED_U16_WORDS) {
    throw new Error("OPT-0017 representative correctness totals changed");
  }
  const aggregateNumerical = summarizeNumericalAccumulator(aggregate);
  assertOpt0017NumericalThresholds(aggregateNumerical, "aggregate");
  return Object.freeze({
    representativeCount: preparedRepresentatives.length,
    probeCount: REPRESENTATIVE_CORRECTNESS_PROBE_COUNT,
    candidateCount: CANDIDATE_ARMS.length,
    executionsPerProbe: REPRESENTATIVE_CORRECTNESS_EXECUTIONS,
    comparedU16WordCount,
    candidateVersusShippedFixed32: true,
    qNaNPrefillCompleteWrites: true,
    guardsAndAdjacentCanariesUntouched: true,
    deterministicCandidateRerunHashes: true,
    thresholds: Object.freeze({
      nrmseMaximum: NRMSE_LIMIT,
      snrMinimumDb: SNR_MINIMUM_DB,
      pearsonMinimum: PEARSON_MINIMUM,
      relativeMaximumAbsoluteErrorMaximum:
        RELATIVE_MAXIMUM_ABSOLUTE_ERROR_LIMIT,
    }),
    aggregate: aggregateNumerical,
    cases: Object.freeze(cases),
  });
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
  sumError: number;
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

function createNumericalAccumulator(): NumericalAccumulator {
  return {
    count: 0,
    sumError: 0,
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

async function executeRepresentativeProbe(
  device: GPUDevice,
  tracker: BufferTracker,
  shared: SharedResources,
  prepared: PreparedRepresentative,
  probe: Opt0017RepresentativeProbe,
  order: readonly Opt0017RepresentativeArm[],
  executionIndex: number,
): Promise<Readonly<Record<string, unknown>>> {
  const operation = prepared.representative.operation;
  const plan = prepared.plan;
  const elementBytes = 2;
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
    label: `${operation.label}-${probe.id}-two-arm-readback`,
    size: armBytes * REPRESENTATIVE_ARMS.length,
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
      label: `${operation.label}-${probe.id}-correctness`,
    });
    for (const arm of order) {
      encoder.copyBufferToBuffer(
        shared.stageF16Prefill,
        0,
        shared.outputs[arm],
        STORAGE_GUARD_BYTES + selectedStart,
        selectedBytes,
      );
      const pass = encoder.beginComputePass();
      prepared.correctnessDispatches[probe.id][arm].encode(pass);
      pass.end();
      encodeProbeReadback(
        encoder,
        shared.outputs[arm],
        readback,
        REPRESENTATIVE_ARMS.indexOf(arm) * armBytes,
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
    const arms = {} as Record<Opt0017RepresentativeArm, ReadbackArm>;
    for (const arm of REPRESENTATIVE_ARMS) {
      arms[arm] = await scanOutputReadback(
        mappedRange,
        REPRESENTATIVE_ARMS.indexOf(arm) * armBytes,
        operation.outputStorage,
        probe.count,
        selectedBytes,
        beforeBytes,
        afterBytes,
      );
    }
    const numerical = compareFp16Outputs(
      arms.fixed32.words,
      arms.cooperativeDot4.words,
      operation,
      probe,
      executionIndex,
    );
    return Object.freeze({
      hashes: Object.freeze(Object.fromEntries(REPRESENTATIVE_ARMS.map(
        (arm) => [arm, arms[arm].sha256],
      ))),
      numerical,
      completeWrites: true,
      guardsAndAdjacentCanariesUntouched: true,
    });
  } finally {
    if (mapped) readback.unmap();
    tracker.destroy(readback);
  }
}

function compareFp16Outputs(
  controlWords: Uint16Array,
  candidateWords: Uint16Array,
  operation: Opt0017C300Operation,
  probe: Opt0017RepresentativeProbe,
  executionIndex: number,
): NumericalAccumulator {
  if (controlWords.length !== candidateWords.length ||
    controlWords.length !== probe.count) {
    throw new Error("OPT-0017 numerical comparison length changed");
  }
  const result = createNumericalAccumulator();
  for (let index = 0; index < controlWords.length; index += 1) {
    const controlBits = controlWords[index]!;
    const candidateBits = candidateWords[index]!;
    const control = fp16ToNumber(controlBits);
    const candidate = fp16ToNumber(candidateBits);
    if (!Number.isFinite(control) || !Number.isFinite(candidate)) {
      throw new Error("OPT-0017 numerical comparison found non-finite output");
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
    if (controlBits !== candidateBits) result.firstDifference ??= location;
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
  target.sumError += source.sumError;
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
  if (value.count <= 0) throw new Error("OPT-0017 empty numerical result");
  if (![value.controlMinimum, value.controlMaximum, value.candidateMinimum,
    value.candidateMaximum].every(Number.isFinite) ||
    value.controlMinimum > value.controlMaximum ||
    value.candidateMinimum > value.candidateMaximum) {
    throw new Error("OPT-0017 numerical output ranges are invalid");
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
  const ulpDistribution = Object.freeze(Object.fromEntries(
    [...value.ulpDistribution.entries()]
      .sort(([left], [right]) => left - right)
      .map(([ulp, count]) => [String(ulp), count]),
  ));
  return Object.freeze({
    comparedValueCount: value.count,
    maximumAbsoluteError: value.maximumAbsoluteError,
    meanError: value.sumError / value.count,
    rmsError,
    nrmse,
    snrDb: Number.isFinite(snrDb) ? snrDb : "Infinity",
    pearson,
    controlPeak: value.controlPeak,
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
    relativeMaximumAbsoluteError: value.maximumAbsoluteError /
      Math.max(value.controlPeak, 1e-6),
    signedZeroDifferenceCount: value.signedZeroDifferenceCount,
    fp16UlpDistribution: ulpDistribution,
    firstDifference: value.firstDifference,
    worstLocation: value.worst,
  });
}

export function assertOpt0017NumericalThresholds(
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
    throw new Error(`${label} rejected frozen numerical thresholds`);
  }
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
  if (exponent === 0) {
    return sign * 2 ** -14 * (fraction / 1_024);
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1_024);
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
  outputStorage: AceFp16VaeConv1dOutputStorage,
  wordCount: number,
  selectedBytes: number,
  beforeBytes: number,
  afterBytes: number,
): Promise<ReadbackArm> {
  if (outputStorage !== "float16") {
    throw new Error("OPT-0017 representative output must be FP16");
  }
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
      `OPT-0017 output scan failed: nonfinite=${nonFiniteCount}, ` +
        `qnan=${prefillQNaNCount}, guards=${guardsUntouched}`,
    );
  }
  return Object.freeze({
    words,
    nonFiniteCount,
    prefillQNaNCount,
    guardsUntouched,
    sha256: await sha256Hex(new Uint8Array(
      words.buffer,
      words.byteOffset,
      words.byteLength,
    )),
  });
}

async function runTimedGate(
  prepared: PreparedGate,
): Promise<Readonly<Record<string, unknown>>> {
  const launchedAtEpochMilliseconds = Date.now();
  const thermal = parseOpt0017ThermalGate(
    collectThermalParameters(),
    prepared.preparedCompletedAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const timingInputs = prepared.representatives.map((item) => ({
    representativeId: item.representative.id,
    tier: item.representative.tier,
    weight: item.representative.weight,
    samples: {
      fixed32: [],
      cooperativeDot4: [],
    } as Record<Opt0017RepresentativeArm, number[]>,
  }));
  const representativeTimingStartedAtEpochMilliseconds = Date.now();
  for (const entry of buildOpt0017RepresentativeOrders()) {
    const preparedRepresentative = prepared.representatives[entry.tierIndex];
    const input = timingInputs[entry.tierIndex];
    if (preparedRepresentative === undefined || input === undefined) {
      throw new Error("OPT-0017 representative timing topology changed");
    }
    prepared.updateProgress(
      `representative timing tier ${entry.tierIndex + 1}/4, ` +
        `sample ${entry.sampleIndex + 1}/4`,
    );
    for (const arm of entry.order) {
      input.samples[arm].push(await executeTimedDispatch(
        prepared.device,
        preparedRepresentative.timingDispatches[arm],
      ));
    }
    await yieldToBrowser();
  }
  const representativeTimingCompletedAtEpochMilliseconds = Date.now();
  const representativeTiming = summarizeOpt0017RepresentativeTiming(
    timingInputs.map((input) => Object.freeze({
      ...input,
      samples: Object.freeze(Object.fromEntries(REPRESENTATIVE_ARMS.map(
        (arm) => [arm, Object.freeze([...input.samples[arm]])],
      ))) as Readonly<Record<Opt0017RepresentativeArm, readonly number[]>>,
    })),
  );
  const selection = selectOpt0017RepresentativeWinner(representativeTiming);
  const selectedWinner = selection["selectedWinner"] as
    Opt0017CandidateArm | null;
  let fullSequence: Readonly<Record<string, unknown>> | null = null;
  let fullSequenceAllocationPerformed = false;
  if (selectedWinner !== null) {
    fullSequenceAllocationPerformed = true;
    prepared.updateProgress(
      `preparing conditional complete C300 gate for ${selectedWinner}`,
    );
    const full = await prepareFullGate(prepared, selectedWinner);
    fullSequence = await runFullTimedSequence(prepared, full);
  }
  const capability = capabilityReceipt(prepared.adapter, prepared.device);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanup = prepared.destroy();
  const finalDecision = selectedWinner === null
    ? "negative-stop-no-representative-qualifier"
    : String(fullSequence!["decision"]);
  return Object.freeze({
    schema: "ace-opt-0017-vae-k7-cooperative-dot4-ab-v1",
    status: "passed",
    experimentId: EXPERIMENT_ID,
    classification: "bounded-primitive-decision-gate-not-integrated",
    recordedAt: new Date().toISOString(),
    identity: Object.freeze({ sourceAuthority: prepared.sourceAuthority }),
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      page: window.location.href,
      ...capability,
    }),
    protocol: Object.freeze({
      thermal,
      numericalAndAll16RepackCompletedBeforeThermalGate: true,
      representativePipelinesCompiledAndWarmedBeforeTiming: true,
      representativeArmOrder: "alternating-AB/BA-by-tier-and-sample",
      representativePositionCounts: representativePositionCounts(),
      representativeSamplesPerArmPerTier:
        REPRESENTATIVE_SAMPLES_PER_ARM,
      representativeQualifyingSpeedup: REPRESENTATIVE_QUALIFYING_SPEEDUP,
      fullSequenceQualifyingSpeedup: FULL_SEQUENCE_QUALIFYING_SPEEDUP,
      authoritativeTiming: "performance.now-submit-through-queue-drain",
      oneThirtySecondNominalGate: true,
      unchangedThermalRetryPerformed: false,
      productionSelectorClaim: null,
      integratedDecoderWallClaim: null,
      qualityClaim: null,
      listeningClaim: null,
    }),
    topology: Object.freeze({
      decoderInputFrames: C300_INPUT_FRAMES,
      operationCount: C300_OPERATION_COUNT,
      exactRangeCount: C300_EXACT_RANGE_COUNT,
      outputWordCount: C300_OUTPUT_WORD_COUNT,
      representativeRangeWeight: REPRESENTATIVE_TIER_WEIGHT,
      omittedConv1RangeWeight: OMITTED_CONV1_RANGE_WEIGHT,
      omittedShippedFinalConv2RangeWeight: OMITTED_FINAL_CONV2_RANGE_WEIGHT,
      representativeProbes: Object.freeze(prepared.representatives.map(
        ({ representative }) => Object.freeze({
          id: representative.id,
          operationLabel: representative.operation.label,
          shape: representative.operation.shape,
          outputStorage: representative.operation.outputStorage,
          weight: representative.weight,
          probes: representative.probes,
        }),
      )),
    }),
    correctness: prepared.correctness,
    repack: Object.freeze({
      all16Correctness: prepared.repackCorrectness,
      packedPayloadBytes: ALL_16_PACKED_WEIGHT_BYTES,
      incrementalMemoryCostIfIntegratedBytes: ALL_16_PACKED_WEIGHT_BYTES,
    }),
    representativeTiming: Object.freeze({
      ...representativeTiming,
      timingStartedAtEpochMilliseconds:
        representativeTimingStartedAtEpochMilliseconds,
      timingCompletedAtEpochMilliseconds:
        representativeTimingCompletedAtEpochMilliseconds,
    }),
    selection,
    fullSequenceAllocationPerformed,
    fullSequence,
    decision: Object.freeze({
      disposition: finalDecision,
      productionIntegrationAuthorized: false,
    }),
    memory: memoryBeforeCleanup,
    cleanup,
  });
}

async function prepareFullGate(
  prepared: PreparedGate,
  winner: Opt0017CandidateArm,
): Promise<FullPreparedGate> {
  const fixed32Output = prepared.shared.outputs.fixed32;
  const winnerOutput = prepared.shared.outputs[winner];
  const controlsResult = createFullRangeControls(
    prepared.device,
    prepared.tracker,
    prepared.topology,
  );
  const operations: FullPreparedOperation[] = [];
  const repackCases: unknown[] = [];
  let uniqueRepackU16Words = 0;
  let comparedRepackU16Words = 0;
  let repackWorkgroups = 0;
  let dispatchCount = 0;
  let commandBufferCount = 0;
  let fixed32Workgroups = 0;
  let winnerWorkgroups = 0;
  for (const [operationOrdinal, topology] of prepared.topology.entries()) {
    prepared.updateProgress(
      `full C300 dispatch prep ${operationOrdinal + 1}/16: ${topology.label}`,
    );
    const weight = prepared.weights.get(topology.label);
    if (weight === undefined) throw new Error(`${topology.label} weight missing`);
    repackCases.push(weight.verification);
    uniqueRepackU16Words += weight.uniqueU16Words;
    comparedRepackU16Words += weight.uniqueU16Words *
      weight.verificationExecutions;
    repackWorkgroups += weight.repack.plan.repackWorkgroups;
    const plan = planAceFp16VaeConv1d(topology.shape, topology.outputStorage);
    const fixed32: EncodableDispatch[] = [];
    const selected: EncodableDispatch[] = [];
    for (const rangeValue of topology.ranges) {
      const controlOffset = controlsResult.offsets.get(
        fullRangeKey(topology.label, rangeValue.rangeIndex),
      );
      if (controlOffset === undefined) {
        throw new Error(`${topology.label} full range control missing`);
      }
      const range = outputRangeBinding(
        controlsResult.buffer,
        controlOffset,
        rangeValue.base,
        rangeValue.count,
      );
      const common = Object.freeze({
        input: binding(prepared.shared.input, plan.inputBindingBytes),
        bias: binding(prepared.shared.bias, plan.biasBindingBytes),
      });
      const fixedDispatch = await prepared.fixed32Kernel.createDispatch(
        `${topology.label}-range-${rangeValue.rangeIndex}-fixed32`,
        topology.shape,
        Object.freeze({
          ...common,
          weight: binding(weight.native, plan.weightBindingBytes),
          output: outputBinding(fixed32Output, plan),
        }),
        topology.outputStorage,
        range,
      );
      const winnerDispatch = await prepared.cooperativeKernel.createDispatch(
        `${topology.label}-range-${rangeValue.rangeIndex}-${winner}`,
        topology.shape,
        Object.freeze({
          ...common,
          packedWeight: weight.packed.binding,
          output: outputBinding(winnerOutput, plan),
        }),
        topology.outputStorage,
        range,
      );
      assertFullDispatches(topology, plan, rangeValue, winner, fixedDispatch,
        winnerDispatch);
      fixed32.push(fixedDispatch);
      selected.push(winnerDispatch);
      fixed32Workgroups += dispatchWorkgroupCount(fixedDispatch);
      winnerWorkgroups += dispatchWorkgroupCount(winnerDispatch);
    }
    dispatchCount += topology.ranges.length;
    commandBufferCount += Math.ceil(
      topology.ranges.length / MAX_QUANTA_PER_COMMAND_BUFFER,
    );
    operations.push(Object.freeze({
      topology,
      plan,
      weight,
      fixed32: Object.freeze(fixed32),
      winner: Object.freeze(selected),
    }));
    await yieldToBrowser();
  }
  if (uniqueRepackU16Words !== ALL_16_UNIQUE_REPACK_U16_WORDS ||
    comparedRepackU16Words !== FULL_REPACK_COMPARED_U16_WORDS ||
    repackWorkgroups !== ALL_16_REPACK_WORKGROUPS ||
    dispatchCount !== C300_EXACT_RANGE_COUNT ||
    commandBufferCount !== FULL_SEQUENCE_COMMAND_BUFFERS_PER_ARM ||
    fixed32Workgroups !== 103_672 ||
    winnerWorkgroups !== 103_684) {
    throw new Error("OPT-0017 full C300 dispatch accounting changed");
  }
  return Object.freeze({
    winner,
    operations: Object.freeze(operations),
    controls: controlsResult.buffer,
    fixed32Output,
    winnerOutput,
    repackCorrectness: Object.freeze({
      operationCount: C300_OPERATION_COUNT,
      uniqueU16WordCount: uniqueRepackU16Words,
      comparedU16WordCount: comparedRepackU16Words,
      packedPayloadBytes: ALL_16_PACKED_WEIGHT_BYTES,
      repackWorkgroupCount: repackWorkgroups,
      selectedFourVerifiedTwice: true,
      remainingTwelveVerifiedOnce: true,
      mismatchCount: 0,
      qNaNPrefillCompleteWrites: true,
      redzonesUntouched: true,
      cases: Object.freeze(repackCases),
    }),
    commandBufferCountPerArm: commandBufferCount,
    dispatchCountPerArm: dispatchCount,
  });
}

async function runFullTimedSequence(
  prepared: PreparedGate,
  full: FullPreparedGate,
): Promise<Readonly<Record<string, unknown>>> {
  prepared.updateProgress("warming complete C300 arms before timing");
  await executeTimedCommandSequence(prepared.device,
    encodeFullSequence(prepared.device, full.operations, "fixed32", -1));
  await executeTimedCommandSequence(prepared.device,
    encodeFullSequence(prepared.device, full.operations, "winner", -1));
  prepared.updateProgress("pre-encoding two balanced full-C300 sequences");
  const fixedRoundCommands = [
    encodeFullSequence(prepared.device, full.operations, "fixed32", 0),
    encodeFullSequence(prepared.device, full.operations, "fixed32", 1),
  ] as const;
  const winnerRoundCommands = [
    encodeFullSequence(prepared.device, full.operations, "winner", 0),
    encodeFullSequence(prepared.device, full.operations, "winner", 1),
  ] as const;
  for (const commands of [...fixedRoundCommands, ...winnerRoundCommands]) {
    if (commands.length !== FULL_SEQUENCE_COMMAND_BUFFERS_PER_ARM) {
      throw new Error("OPT-0017 preencoded full command count changed");
    }
  }
  const rounds: Opt0017FullSequenceRoundInput[] = [];
  const timingStartedAtEpochMilliseconds = Date.now();
  prepared.updateProgress("full C300 AB: fixed32 then repack+winner");
  const fixedAb = await executeTimedCommandSequence(
    prepared.device,
    fixedRoundCommands[0],
  );
  const repackAb = await executeTimedRepackBatch(
    prepared.device,
    full.operations.map((operation) => operation.weight.repack),
  );
  const winnerAb = await executeTimedCommandSequence(
    prepared.device,
    winnerRoundCommands[0],
  );
  rounds.push(Object.freeze({
    order: Object.freeze(["fixed32", "winner"] as const),
    fixed32Milliseconds: fixedAb,
    winnerConvolutionMilliseconds: winnerAb,
    repackMilliseconds: repackAb,
  }));
  await yieldToBrowser();
  prepared.updateProgress("full C300 BA: repack+winner then fixed32");
  const repackBa = await executeTimedRepackBatch(
    prepared.device,
    full.operations.map((operation) => operation.weight.repack),
  );
  const winnerBa = await executeTimedCommandSequence(
    prepared.device,
    winnerRoundCommands[1],
  );
  const fixedBa = await executeTimedCommandSequence(
    prepared.device,
    fixedRoundCommands[1],
  );
  rounds.push(Object.freeze({
    order: Object.freeze(["winner", "fixed32"] as const),
    fixed32Milliseconds: fixedBa,
    winnerConvolutionMilliseconds: winnerBa,
    repackMilliseconds: repackBa,
  }));
  const timingCompletedAtEpochMilliseconds = Date.now();
  const summary = summarizeOpt0017FullSequence(full.winner, rounds);
  return Object.freeze({
    ...summary,
    topology: Object.freeze({
      operationCount: C300_OPERATION_COUNT,
      exactRangeCount: C300_EXACT_RANGE_COUNT,
      dispatchCountPerArm: full.dispatchCountPerArm,
      commandBufferCountPerArmPerRound: full.commandBufferCountPerArm,
      maximumRangesPerOnePassCommandBuffer: MAX_QUANTA_PER_COMMAND_BUFFER,
      separatelyDrainedCommandBuffersPerArmPerRound: true,
    }),
    repack: Object.freeze({
      correctness: full.repackCorrectness,
      measuredAll16BeforeEachCandidateRun: true,
      dispatchCountPerMeasurement: C300_OPERATION_COUNT,
      workgroupCountPerMeasurement: ALL_16_REPACK_WORKGROUPS,
      commandBufferCountPerMeasurement: 1,
      submitAndDrainCountPerMeasurement: 1,
      packedPayloadBytes: ALL_16_PACKED_WEIGHT_BYTES,
    }),
    timingStartedAtEpochMilliseconds,
    timingCompletedAtEpochMilliseconds,
    compileAllocationUploadCorrectnessReadbackAndCleanupExcluded: true,
  });
}

function encodeFullSequence(
  device: GPUDevice,
  operations: readonly FullPreparedOperation[],
  arm: FullSequenceArm,
  round: number,
): readonly GPUCommandBuffer[] {
  return Object.freeze(operations.flatMap((operation) =>
    encodeBatchedDispatches(
      device,
      operation[arm],
      `${operation.topology.label}-${arm}-timing-round-${round}`,
    )
  ));
}

function encodeBatchedDispatches(
  device: GPUDevice,
  dispatches: readonly EncodableDispatch[],
  label: string,
): readonly GPUCommandBuffer[] {
  const commands: GPUCommandBuffer[] = [];
  for (let base = 0; base < dispatches.length;
    base += MAX_QUANTA_PER_COMMAND_BUFFER) {
    const encoder = device.createCommandEncoder({
      label: `${label}-batch-${commands.length}`,
    });
    const pass = encoder.beginComputePass();
    for (const dispatch of dispatches.slice(
      base,
      base + MAX_QUANTA_PER_COMMAND_BUFFER,
    )) dispatch.encode(pass);
    pass.end();
    commands.push(encoder.finish());
  }
  return Object.freeze(commands);
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

async function executeTimedCommandSequence(
  device: GPUDevice,
  commands: readonly GPUCommandBuffer[],
): Promise<number> {
  if (commands.length !== FULL_SEQUENCE_COMMAND_BUFFERS_PER_ARM) {
    throw new Error("OPT-0017 timed full sequence command count changed");
  }
  const started = performance.now();
  for (const command of commands) {
    device.queue.submit([command]);
    await device.queue.onSubmittedWorkDone();
  }
  return performance.now() - started;
}

async function executeTimedRepackBatch(
  device: GPUDevice,
  dispatches: readonly AceOpt0014VaeConv1dRepackDispatch[],
): Promise<number> {
  if (dispatches.length !== C300_OPERATION_COUNT ||
    dispatches.reduce((sum, dispatch) =>
      sum + dispatch.plan.repackWorkgroups, 0) !== ALL_16_REPACK_WORKGROUPS) {
    throw new Error("OPT-0017 all-16 repack timing topology changed");
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
  topology: readonly Opt0017C300Operation[],
): Promise<Readonly<Record<string, unknown>>> {
  const fixed32SourceSha256 = await sha256Text(fixed32CoreSource);
  const packed16x64SourceSha256 = await sha256Text(packed16x64CoreSource);
  const cooperativeSourceSha256 = await sha256Text(cooperativeCoreSource);
  if (fixed32SourceSha256 !== FIXED32_CORE_SOURCE_SHA256 ||
    packed16x64SourceSha256 !== PACKED_16X64_CORE_SOURCE_SHA256 ||
    cooperativeSourceSha256 !== COOPERATIVE_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0017 rejected unauthenticated kernel core source");
  }
  const fixed32Shaders: string[] = [];
  const repackShaders: string[] = [];
  const cooperativeShaders: string[] = [];
  for (const operation of topology) {
    const hasBias = operation.outputStorage === "float16";
    fixed32Shaders.push(
      `${operation.label}\n${aceFp16VaeConv1dSubgroupWgsl(
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
    cooperativeShaders.push(
      `${operation.label}\n${aceOpt0017VaeConv1dCooperativeDot4Wgsl(
        operation.shape,
        hasBias,
        operation.outputStorage,
      )}`,
    );
  }
  return Object.freeze({
    coreCommit: CORE_COMMIT,
    fixed32CoreSourceSha256: fixed32SourceSha256,
    packed16x64CoreSourceSha256: packed16x64SourceSha256,
    cooperativeCoreSourceSha256: cooperativeSourceSha256,
    fixed32KernelId: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
    repackKernelId: ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_REPACK_KERNEL_ID,
    cooperativeKernelId:
      ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID,
    generatedShaderCountPerArm: topology.length,
    fixed32GeneratedAggregateSha256: await aggregateShaderSha256(
      fixed32Shaders,
    ),
    repackGeneratedAggregateSha256: await aggregateShaderSha256(repackShaders),
    cooperativeGeneratedAggregateSha256: await aggregateShaderSha256(
      cooperativeShaders,
    ),
  });
}

function expectedOperation(
  label: string,
  inputFrames: number,
  inputChannels: number,
  outputChannels: number,
  dilation: number,
  padding: number,
  rangeCount: number,
  outputStorage: AceFp16VaeConv1dOutputStorage = "float16",
) {
  return Object.freeze({
    label,
    inputFrames,
    inputChannels,
    outputChannels,
    dilation,
    padding,
    rangeCount,
    outputStorage,
  });
}

function representativeSpec(
  id: string,
  tier: Opt0017Representative["tier"],
  label: string,
  timingRangeIndex: number,
  weight: number,
  probes: readonly (readonly [number, number])[],
) {
  return Object.freeze({
    id,
    tier,
    label,
    timingRangeIndex,
    weight,
    probes: Object.freeze(probes),
  });
}

function assertExpectedOperation(
  operation: Opt0017C300Operation,
  expectedValue: ReturnType<typeof expectedOperation> | undefined,
): void {
  if (expectedValue === undefined || operation.label !== expectedValue.label ||
    operation.shape.batch !== 1 || operation.shape.stride !== 1 ||
    operation.shape.kernelSize !== 7 ||
    operation.shape.inputFrames !== expectedValue.inputFrames ||
    operation.shape.inputChannels !== expectedValue.inputChannels ||
    operation.shape.outputChannels !== expectedValue.outputChannels ||
    operation.shape.dilation !== expectedValue.dilation ||
    operation.shape.padding !== expectedValue.padding ||
    operation.outputStorage !== expectedValue.outputStorage ||
    operation.ranges.length !== expectedValue.rangeCount) {
    throw new Error(`${operation.label} rejected unexpected C300 geometry`);
  }
}

function createRepresentativeControls(
  device: GPUDevice,
  tracker: BufferTracker,
  representatives: readonly Opt0017Representative[],
): Readonly<{ buffer: GPUBuffer; offsets: ReadonlyMap<string, number> }> {
  const entries = representatives.flatMap((representative) =>
    representative.probes.map((probe) => Object.freeze({
      key: representativeProbeKey(representative.operation.label, probe.id),
      base: probe.base,
      count: probe.count,
    }))
  );
  if (entries.length !== REPRESENTATIVE_CORRECTNESS_PROBE_COUNT) {
    throw new Error("OPT-0017 representative control count changed");
  }
  return createControls(device, tracker, "opt-0017-representative-controls",
    entries);
}

function createFullRangeControls(
  device: GPUDevice,
  tracker: BufferTracker,
  topology: readonly Opt0017C300Operation[],
): Readonly<{ buffer: GPUBuffer; offsets: ReadonlyMap<string, number> }> {
  const entries = topology.flatMap((operation) => operation.ranges.map(
    (range) => Object.freeze({
      key: fullRangeKey(operation.label, range.rangeIndex),
      base: range.base,
      count: range.count,
    }),
  ));
  if (entries.length !== C300_EXACT_RANGE_COUNT) {
    throw new Error("OPT-0017 full control count changed");
  }
  return createControls(device, tracker, "opt-0017-full-range-controls",
    entries);
}

function createControls(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  entries: readonly Readonly<{
    key: string;
    base: number;
    count: number;
  }>[]
): Readonly<{ buffer: GPUBuffer; offsets: ReadonlyMap<string, number> }> {
  const alignment = Number(device.limits.minUniformBufferOffsetAlignment);
  if (!Number.isSafeInteger(alignment) || alignment < RANGE_CONTROL_BYTES ||
    !Number.isInteger(Math.log2(alignment))) {
    throw new RangeError("OPT-0017 uniform alignment is invalid");
  }
  const buffer = tracker.create(device, {
    label,
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
      throw new Error(`OPT-0017 duplicate range control ${entry.key}`);
    }
    offsets.set(entry.key, byteOffset);
  }
  buffer.unmap();
  return Object.freeze({ buffer, offsets });
}

function outputRangeBinding(
  controls: GPUBuffer,
  controlOffset: number,
  base: number,
  count: number,
): AceVaeOutputRangeBinding {
  return Object.freeze({
    base,
    count,
    control: Object.freeze({
      buffer: controls,
      offset: controlOffset,
      size: RANGE_CONTROL_BYTES,
    }),
  });
}

function assertRepresentativeDispatches(
  operation: Opt0017C300Operation,
  plan: AceFp16VaeConv1dPlan,
  range: AceVaeOutputRangeBinding,
  dispatches: Readonly<Record<Opt0017RepresentativeArm,
    EncodableDispatch>>,
): void {
  const fixedExpected = planAceFp16VaeConv1dSubgroupRange(plan, range);
  const candidateExpected = planAceOpt0017VaeConv1dCooperativeDot4Range(
    plan,
    range,
  );
  if (dispatches.fixed32.kernelId !==
      ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID ||
    !sameDispatchRange(dispatches.fixed32.outputRange, fixedExpected) ||
    dispatches.cooperativeDot4.kernelId !==
      ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID ||
    !sameDispatchRange(
      dispatches.cooperativeDot4.outputRange,
      candidateExpected,
    )) {
    throw new Error(`${operation.label} representative dispatch changed`);
  }
}

function assertFullDispatches(
  operation: Opt0017C300Operation,
  plan: AceFp16VaeConv1dPlan,
  range: ExactRange,
  winner: Opt0017CandidateArm,
  fixed32: EncodableDispatch,
  selected: EncodableDispatch,
): void {
  const expectedFixed = planAceFp16VaeConv1dSubgroupRange(plan, range);
  const expectedWinner = planAceOpt0017VaeConv1dCooperativeDot4Range(
    plan,
    range,
  );
  if (winner !== "cooperativeDot4" ||
    fixed32.kernelId !== ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID ||
    !sameDispatchRange(fixed32.outputRange, expectedFixed) ||
    selected.kernelId !==
      ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID ||
    !sameDispatchRange(selected.outputRange, expectedWinner)) {
    throw new Error(
      `${operation.label} full range ${range.rangeIndex} dispatch changed`,
    );
  }
}

function sameDispatchRange(
  left: EncodableDispatch["outputRange"],
  right: EncodableDispatch["outputRange"],
): boolean {
  return left.base === right.base && left.count === right.count &&
    left.workgroupsX === right.workgroupsX &&
    left.workgroupsY === right.workgroupsY;
}

function dispatchWorkgroupCount(dispatch: EncodableDispatch): number {
  return dispatch.outputRange.workgroupsX * dispatch.outputRange.workgroupsY;
}

function requiredDeviceLimits(
  adapter: GPUAdapter,
  topology: readonly Opt0017C300Operation[],
): Record<string, number> {
  let maximumBuffer = 4;
  let maximumStorageBinding = 4;
  let maximumDispatch = 1;
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
    for (const range of operation.ranges) {
      const candidates = [
        planAceFp16VaeConv1dSubgroupRange(plan, range),
        planAceOpt0017VaeConv1dCooperativeDot4Range(plan, range),
      ];
      for (const candidate of candidates) {
        maximumDispatch = Math.max(
          maximumDispatch,
          candidate.workgroupsX,
          candidate.workgroupsY,
        );
      }
    }
  }
  const requested = {
    maxBufferSize: maximumBuffer,
    maxStorageBufferBindingSize: maximumStorageBinding,
    maxUniformBufferBindingSize: RANGE_CONTROL_BYTES,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupStorageSize: 10_496,
    maxComputeWorkgroupsPerDimension: maximumDispatch,
  };
  for (const [name, minimum] of Object.entries(requested)) {
    const actual = Number(adapter.limits[name as keyof GPUSupportedLimits]);
    if (!Number.isFinite(actual) || actual < minimum) {
      throw new RangeError(
        `OPT-0017 adapter ${name}=${actual} is below ${minimum}`,
      );
    }
  }
  return requested;
}

function requireAdapter(
  adapter: GPUAdapter,
  topology: readonly Opt0017C300Operation[],
): void {
  if (!adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups")) {
    throw new Error("OPT-0017 requires adapter shader-f16 and subgroups");
  }
  if (adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32) {
    throw new Error("OPT-0017 requires authenticated fixed 32-lane subgroups");
  }
  if (adapter.limits.minStorageBufferOffsetAlignment > STORAGE_GUARD_BYTES) {
    throw new Error("OPT-0017 storage redzone is below adapter alignment");
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
    fillPeriodic(new Uint16Array(buffer.getMappedRange()), pattern);
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

function fillPeriodic(destination: Uint16Array, pattern: Uint16Array): void {
  if (destination.length < 1 || pattern.length < 1) {
    throw new RangeError("OPT-0017 periodic upload geometry changed");
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
    throw new RangeError("OPT-0017 U32 pattern write is unaligned");
  }
  const words = new Uint32Array(bytes / 4);
  words.fill(value);
  device.queue.writeBuffer(buffer, offset, words);
}

function everyU32(words: Uint32Array, expectedValue: number): boolean {
  for (const word of words) if (word !== expectedValue) return false;
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

function representativeProbeKey(label: string, id: Opt0017ProbeId): string {
  return `${label}:${id}`;
}

function fullRangeKey(label: string, rangeIndex: number): string {
  return `${label}:range:${rangeIndex}`;
}

function representativePositionCounts(): Readonly<Record<
  Opt0017RepresentativeArm,
  readonly number[]
>> {
  const result = Object.fromEntries(REPRESENTATIVE_ARMS.map((arm) => [
    arm,
    Array.from({ length: REPRESENTATIVE_ARMS.length }, () => 0),
  ])) as Record<Opt0017RepresentativeArm, number[]>;
  for (const entry of buildOpt0017RepresentativeOrders()) {
    entry.order.forEach((arm, position) => {
      result[arm][position]! += 1;
    });
  }
  return Object.freeze(Object.fromEntries(REPRESENTATIVE_ARMS.map((arm) => [
    arm,
    Object.freeze([...result[arm]]),
  ]))) as Readonly<Record<Opt0017RepresentativeArm, readonly number[]>>;
}

function median4(samples: readonly number[]): number {
  if (samples.length !== REPRESENTATIVE_SAMPLES_PER_ARM ||
    samples.some((sample) => !Number.isFinite(sample) || sample <= 0)) {
    throw new RangeError(
      "OPT-0017 requires four finite positive timing samples per arm",
    );
  }
  const sorted = [...samples].sort((left, right) => left - right);
  return (sorted[1]! + sorted[2]!) / 2;
}

function requireFinitePositive(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${label} must be finite and positive`);
  }
}

function collectThermalParameters(): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    "#thermal-gate input[name]",
  )) parameters.set(input.name, input.value);
  return parameters;
}

async function aggregateShaderSha256(shaders: readonly string[]) {
  return sha256Text(shaders.join("\n\u0000\n"));
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
    throw new Error(`OPT-0017 thermal field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0017 thermal field ${name} is not finite`);
  }
  return value;
}

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, 0));
}

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0017-vae-k7-cooperative-dot4-ab-v1",
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
  if (element === null) throw new Error(`Missing OPT-0017 element ${selector}`);
  return element;
}
