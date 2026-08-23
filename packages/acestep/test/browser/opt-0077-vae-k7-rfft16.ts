/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import currentSelectorSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.ts?raw";
import currentRowReuseSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-16x64.ts?raw";
import scalarK7Source from
  "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.ts?raw";
import candidateSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-rfft16.ts?raw";
import candidateMathSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-rfft16-math.ts?raw";
import experimentSource from
  "../../optimization/experiments/OPT-0077-vae-k7-rfft16-transform-domain.md?raw";
import htmlSource from "./opt-0077-vae-k7-rfft16.html?raw";
import harnessSource from "./opt-0077-vae-k7-rfft16.ts?raw";

import {
  ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
  AceOpt0057VaeK7ShapeSelectorKernel,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import {
  aceOpt0051VaeConv1dK4RowReuse16x64Wgsl,
  packAceOpt0051VaeK7WeightU16,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-16x64.js";
import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
  AceFp16VaeConv1dSubgroupKernel,
  aceFp16VaeConv1dSubgroupWgsl,
  type AceFp16VaeConv1dFixedSubgroupCapability,
} from "../../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import {
  ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES,
  ACE_OPT_0077_VAE_K7_RFFT16_KERNEL_ID,
  AceOpt0077VaeK7Rfft16Kernel,
  aceOpt0077VaeK7Rfft16Wgsl,
  planAceOpt0077VaeK7Rfft16Range,
  transformAceOpt0077VaeK7WeightU16,
} from "../../src/webgpu/kernels/vae-conv1d-fp16-rfft16.js";
import {
  ACE_OPT_0077_RFFT16_COORDINATE_ORDER,
  ACE_OPT_0077_RFFT16_KERNEL_SIZE,
  ACE_OPT_0077_RFFT16_LENGTH,
  ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE,
  ACE_OPT_0077_RFFT16_OVERLAP,
  aceOpt0077DirectK7Correlation,
  aceOpt0077Rfft16CorrelateF32,
  aceOpt0077Rfft16DirectDftReference,
  aceOpt0077Rfft16ForwardF32,
  aceOpt0077TransformK7WeightF32,
} from "../../src/webgpu/kernels/vae-conv1d-fp16-rfft16-math.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dPlan,
} from "../../src/webgpu/kernels/vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../../src/webgpu/kernels/vae-primitives.js";

declare global {
  interface Window {
    __ACE_OPT0077_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

export type Opt0077Arm = "current" | "candidate";
export type Opt0077FixtureKind =
  | "production"
  | "signed-zero-subnormal"
  | "alternating-cancellation"
  | "finite-transform-amplification"
  | "partial-tile"
  | "dc"
  | "nyquist"
  | `cos${1 | 2 | 3 | 4 | 5 | 6 | 7}`
  | `sin${1 | 2 | 3 | 4 | 5 | 6 | 7}`;

type ExecutionArm = Opt0077Arm | "scalar-oracle";
type ProbeId = "first" | "interior" | "tail" | "full";

export interface Opt0077ProbeSpec {
  readonly id: ProbeId;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
  readonly base: number;
  readonly count: number;
}

export interface Opt0077CaseSpec {
  readonly id: string;
  readonly kind: Opt0077FixtureKind;
  readonly tier: "c1024" | "c512" | "c128" | "bounded-c128";
  readonly operationLabel: string;
  readonly representedOperationLabels: readonly string[];
  readonly dilation: 1 | 3 | 9;
  readonly timingMultiplicity: 0 | 2 | 3 | 9;
  readonly shape: AceVaeConv1dShape;
  readonly probes: readonly Opt0077ProbeSpec[];
}

export interface Opt0077TimestampSample {
  readonly gpuMilliseconds: number;
  readonly wallMilliseconds: number;
  readonly gpuElapsedNanoseconds: number;
  readonly dispatchCount: number;
  readonly commandBufferCount: 1;
  readonly queueDrainCount: 1;
  readonly submitAtEpochMilliseconds?: number;
  readonly fenceAtEpochMilliseconds?: number;
  readonly timestampBeginNanoseconds?: string;
  readonly timestampEndNanoseconds?: string;
}

export interface Opt0077StratumTimingInput {
  readonly id: string;
  readonly currentGpuMilliseconds: readonly number[];
  readonly candidateGpuMilliseconds: readonly number[];
}

export interface Opt0077TimingInput {
  readonly current: readonly Opt0077TimestampSample[];
  readonly candidate: readonly Opt0077TimestampSample[];
  readonly strata: readonly Opt0077StratumTimingInput[];
}

export interface Opt0077ThermalLaunchGate {
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

export interface Opt0077ThermalCompletion {
  readonly schema:
    "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1";
  readonly sha256: string;
  readonly byteLength: number;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly missingObservationCount: 0;
  readonly initialLevel: 0;
  readonly finalLevel: 0;
  readonly coversCleanup: true;
}

interface Encodable {
  readonly label: string;
  readonly kernelId: string;
  readonly stageDispatchCount: number;
  readonly outputRange: Readonly<{ readonly base: number; readonly count: number }>;
  readonly scratchBytes?: Readonly<{
    readonly inputSpectrum: number;
    readonly contractionSpectrum: number;
  }>;
  encode(pass: GPUComputePassEncoder): void;
}

interface GuardedBuffer {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly prefill: GPUBuffer;
  readonly logicalBytes: number;
  readonly totalBytes: number;
  readonly elementStorage: "f16" | "f32";
}

interface PreparedCase {
  readonly spec: Opt0077CaseSpec;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly output: GuardedBuffer;
  readonly inputSpectrum: GuardedBuffer;
  readonly contractionSpectrum: GuardedBuffer;
  readonly correctnessDispatches: Readonly<Record<ProbeId,
    Readonly<Record<ExecutionArm, Encodable>>>>;
  readonly timingDispatches?: Readonly<Record<Opt0077Arm, Encodable>>;
  readonly aggregateTimingMembers?: readonly PreparedTimingMember[];
  readonly ownedBuffers: readonly GPUBuffer[];
  readonly weightProof: Readonly<Record<string, unknown>>;
}

interface PreparedTimingMember {
  readonly operationLabel: string;
  readonly weightKey: string;
  readonly multiplicity: number;
  readonly dispatches: Readonly<Record<Opt0077Arm, Encodable>>;
  readonly weights: PreparedWeights;
}

interface PreparedHarness {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly currentKernel: AceOpt0057VaeK7ShapeSelectorKernel;
  readonly candidateKernel: AceOpt0077VaeK7Rfft16Kernel;
  readonly scalarOracleKernel: AceFp16VaeConv1dSubgroupKernel;
  readonly querySet: GPUQuerySet;
  readonly queryResolve: GPUBuffer;
  readonly queryReadback: GPUBuffer;
  readonly productionCases: readonly PreparedCase[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly identity: Readonly<Record<string, unknown>>;
  readonly resourceContract: Readonly<Record<string, unknown>>;
  readonly warmup: Readonly<Record<string, unknown>>;
  readonly readyAtEpochMilliseconds: number;
  readonly uncapturedErrors: readonly string[];
  readonly deviceLosses: readonly string[];
  cleanup(): Promise<Readonly<Record<string, unknown>>>;
}

interface PreparedWeights {
  readonly native: GPUBuffer;
  readonly currentPacked: GPUBuffer;
  readonly candidateTransformed: GPUBuffer;
  readonly nativeBytes: number;
  readonly currentPackedBytes: number;
  readonly candidateTransformedBytes: number;
  readonly proof: Readonly<Record<string, unknown>>;
}

export interface Opt0077ProductionWeightBindingSpec {
  readonly stratumId: string;
  readonly tier: "c1024" | "c512" | "c128";
  readonly dilation: 1 | 3 | 9;
  readonly operationLabel: string;
  readonly multiplicity: 2 | 3 | 6;
}

interface OutputSnapshot {
  readonly words: Uint16Array<ArrayBuffer>;
  readonly sha256: string;
  readonly selectedSha256: string;
  readonly nonFiniteCount: number;
  readonly qNaNPrefillCount: number;
  readonly prefixGuardIntact: boolean;
  readonly suffixGuardIntact: boolean;
  readonly adjacentBeforeIntact: boolean;
  readonly adjacentAfterIntact: boolean;
  readonly outOfRangeWriteCount: number;
  readonly selectedTailWritten: boolean;
  readonly scratch: Readonly<Record<string, unknown>> | null;
}

interface NumericalAccumulator {
  count: number;
  rawU16MismatchCount: number;
  signedZeroDifferenceCount: number;
  finiteClassChangeCount: number;
  sumError: number;
  sumAbsoluteError: number;
  sumSquaredError: number;
  sumOracleSquared: number;
  sumOracle: number;
  sumCandidate: number;
  sumCandidateSquared: number;
  sumProduct: number;
  maximumAbsoluteError: number;
  oraclePeak: number;
  oracleMinimum: number;
  oracleMaximum: number;
  candidateMinimum: number;
  candidateMaximum: number;
  ulpDistribution: Map<number, number>;
  finiteClassTransitions: Map<string, number>;
  firstDifference: Readonly<Record<string, unknown>> | null;
  worstDifference: Readonly<Record<string, unknown>> | null;
}

const EXPERIMENT_ID = "OPT-0077" as const;
const RECEIPT_SCHEMA = "ace-opt-0077-vae-k7-rfft16-primitive-v1";
const REGISTRATION_COMMIT = "3539a87";
const TIMING_FRAMES = 512;
const ADVERSARIAL_FRAMES = 37;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_F16 = 0x7e55;
const SCRATCH_PREFILL_QNAN_F32 = 0x7fc0_7755;
const TIMESTAMP_QUERY_BYTES = 16;
const TIMING_ROUNDS = Object.freeze([
  Object.freeze(["current", "candidate"] as const),
  Object.freeze(["candidate", "current"] as const),
  Object.freeze(["candidate", "current"] as const),
  Object.freeze(["current", "candidate"] as const),
  Object.freeze(["current", "candidate"] as const),
  Object.freeze(["candidate", "current"] as const),
]);
const CURRENT_AGGREGATE_DISPATCHES = 42;
const CANDIDATE_AGGREGATE_DISPATCHES = 126;
const REQUIRED_PAIR_WINS = 5;
const REQUIRED_SPEEDUP = 1.35;
const SCOPED_PLANNING_MILLISECONDS = 16_125.2;
const REQUIRED_PROJECTED_SAVING_MILLISECONDS = 4_000;
const CURRENT_PERSISTENT_WEIGHT_BYTES = 56_426_496;
const CANDIDATE_PERSISTENT_WEIGHT_BYTES = 128_974_848;
const PERSISTENT_WEIGHT_BYTE_INCREASE = 72_548_352;
const NRMSE_MAXIMUM = 0.001;
const SNR_MINIMUM_DB = 60;
const PEARSON_MINIMUM = 0.99999;
const RELATIVE_MAXIMUM_ABSOLUTE_ERROR_MAXIMUM = 0.01;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
const THERMAL_TRACE_SCHEMA =
  "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1" as const;
const THERMAL_POLL_MILLISECONDS = 1_000;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const MAXIMUM_THERMAL_GAP_MILLISECONDS = 1_250;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 5_000;

const PRODUCTION_PROBE_ROWS = Object.freeze([
  Object.freeze({ id: "first" as const, first: 0, count: 23 }),
  Object.freeze({ id: "interior" as const, first: 137, count: 113 }),
  Object.freeze({ id: "tail" as const, first: 489, count: 23 }),
]);

export function buildOpt0077ProductionStrata(): readonly Opt0077CaseSpec[] {
  const tiers = Object.freeze([
    Object.freeze({ tier: "c1024" as const, channels: 1_024,
      block: 0, multiplicity: 2 as const }),
    Object.freeze({ tier: "c512" as const, channels: 512,
      block: 1, multiplicity: 3 as const }),
    Object.freeze({ tier: "c128" as const, channels: 128,
      block: 4, multiplicity: 9 as const }),
  ]);
  const cases = tiers.flatMap((tier) => ([1, 3, 9] as const).map((dilation) => {
    const residual = dilation === 1 ? 1 : dilation === 3 ? 2 : 3;
    const operationLabel = `block-${tier.block}-res-${residual}-conv1`;
    const representedOperationLabels = tier.tier === "c128"
      ? Object.freeze([
          `block-3-res-${residual}-conv1`,
          `block-4-res-${residual}-conv1`,
        ])
      : Object.freeze([operationLabel]);
    const shape = k7Shape(tier.channels, TIMING_FRAMES, dilation);
    return Object.freeze({
      id: `${tier.tier}-d${dilation}`,
      kind: "production" as const,
      tier: tier.tier,
      operationLabel,
      representedOperationLabels,
      dilation,
      timingMultiplicity: tier.multiplicity,
      shape,
      probes: Object.freeze(PRODUCTION_PROBE_ROWS.map((entry) =>
        probe(entry.id, entry.first, entry.count, tier.channels)
      )),
    });
  }));
  if (cases.length !== 9 ||
    cases.reduce((sum, spec) => sum + spec.timingMultiplicity, 0) !== 42 ||
    cases.some((spec) => spec.probes.length !== 3)) {
    throw new Error("OPT-0077 production stratum topology changed");
  }
  return Object.freeze(cases);
}

export function buildOpt0077ProductionWeightBindingPlan():
readonly Opt0077ProductionWeightBindingSpec[] {
  const bindings = buildOpt0077ProductionStrata().flatMap((spec) =>
    productionWeightBindingsForStratum(spec)
  );
  if (bindings.length !== 12 ||
    new Set(bindings.map(({ operationLabel }) => operationLabel)).size !== 12 ||
    bindings.reduce((sum, { multiplicity }) => sum + multiplicity, 0) !== 42) {
    throw new Error("OPT-0077 production weight-binding topology changed");
  }
  return Object.freeze(bindings);
}

function productionWeightBindingsForStratum(
  spec: Opt0077CaseSpec,
): readonly Opt0077ProductionWeightBindingSpec[] {
  if (spec.kind !== "production" ||
    (spec.tier !== "c1024" && spec.tier !== "c512" && spec.tier !== "c128")) {
    throw new Error("OPT-0077 weight binding requires a production stratum");
  }
  if (spec.tier === "c128") {
    const [block3, block4] = spec.representedOperationLabels;
    if (block3 === undefined || block4 === undefined ||
      spec.representedOperationLabels.length !== 2 || spec.timingMultiplicity !== 9) {
      throw new Error("OPT-0077 C128 production split changed");
    }
    return Object.freeze([
      Object.freeze({ stratumId: spec.id, tier: spec.tier,
        dilation: spec.dilation, operationLabel: block3, multiplicity: 3 as const }),
      Object.freeze({ stratumId: spec.id, tier: spec.tier,
        dilation: spec.dilation, operationLabel: block4, multiplicity: 6 as const }),
    ]);
  }
  const multiplicity = spec.tier === "c1024" ? 2 as const : 3 as const;
  if (spec.representedOperationLabels.length !== 1 ||
    spec.representedOperationLabels[0] !== spec.operationLabel ||
    spec.timingMultiplicity !== multiplicity) {
    throw new Error("OPT-0077 high-tier production split changed");
  }
  return Object.freeze([
    Object.freeze({ stratumId: spec.id, tier: spec.tier,
      dilation: spec.dilation, operationLabel: spec.operationLabel, multiplicity }),
  ]);
}

export function buildOpt0077AdversarialCases(): readonly Opt0077CaseSpec[] {
  const kinds: readonly Opt0077FixtureKind[] = Object.freeze([
    "signed-zero-subnormal",
    "alternating-cancellation",
    "finite-transform-amplification",
    "partial-tile",
    "dc",
    "nyquist",
    ...([1, 2, 3, 4, 5, 6, 7] as const).flatMap((bin) =>
      [`cos${bin}` as const, `sin${bin}` as const]
    ),
  ]);
  const cases = kinds.map((kind): Opt0077CaseSpec => {
    const dilation: 1 | 3 | 9 = kind === "partial-tile" ? 9 : 1;
    const residual = dilation === 9 ? 3 : 1;
    const shape = k7Shape(128, ADVERSARIAL_FRAMES, dilation);
    return Object.freeze({
      id: `bounded-${kind}`,
      kind,
      tier: "bounded-c128",
      operationLabel: `block-4-res-${residual}-conv1`,
      representedOperationLabels: Object.freeze([
        `block-4-res-${residual}-conv1`,
      ]),
      dilation,
      timingMultiplicity: 0,
      shape,
      probes: Object.freeze([
        probe("full", 0, ADVERSARIAL_FRAMES, 128),
      ]),
    });
  });
  if (cases.length !== 20 ||
    cases.filter((spec) => spec.kind === "dc" || spec.kind === "nyquist")
        .length !== 2 ||
    cases.filter((spec) => /^cos[1-7]$|^sin[1-7]$/u.test(spec.kind)).length !==
      14) {
    throw new Error("OPT-0077 spectral/adversarial topology changed");
  }
  return Object.freeze(cases);
}

export function buildOpt0077CorrectnessCases(): readonly Opt0077CaseSpec[] {
  return Object.freeze([
    ...buildOpt0077ProductionStrata(),
    ...buildOpt0077AdversarialCases(),
  ]);
}

function k7Shape(
  channels: number,
  inputFrames: number,
  dilation: 1 | 3 | 9,
): AceVaeConv1dShape {
  return Object.freeze({ batch: 1, inputFrames, inputChannels: channels,
    outputChannels: channels, kernelSize: 7, stride: 1, dilation,
    padding: 3 * dilation });
}

function probe(
  id: ProbeId,
  firstOutputRow: number,
  outputRowCount: number,
  channels: number,
): Opt0077ProbeSpec {
  return Object.freeze({ id, firstOutputRow, outputRowCount,
    base: firstOutputRow * channels, count: outputRowCount * channels });
}

export function planOpt0077TileCount(
  frames: number,
  dilation: 1 | 3 | 9,
): number {
  if (!Number.isSafeInteger(frames) || frames <= 0) {
    throw new RangeError("OPT-0077 frames must be a positive integer");
  }
  let tiles = 0;
  for (let residue = 0; residue < dilation && residue < frames; residue += 1) {
    const streamLength = Math.floor((frames - 1 - residue) / dilation) + 1;
    tiles += Math.ceil(streamLength / ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE);
  }
  return tiles;
}

export function planOpt0077ScratchBytes(
  shape: AceVaeConv1dShape,
): Readonly<{ inputSpectrum: number; contractionSpectrum: number; total: number }> {
  const plan = planAceFp16VaeConv1d(shape, "float16");
  return planAceOpt0077VaeK7Rfft16Range(plan, {
    base: 0,
    count: plan.outputElements,
  }).scratchBytes;
}

export function buildOpt0077PersistentPayloadAccounting():
Readonly<Record<string, unknown>> {
  const routes = buildOpt0077ProductionStrata().flatMap((spec) =>
    spec.representedOperationLabels.map((operationLabel) => {
      const channelPairs = spec.shape.inputChannels * spec.shape.outputChannels;
      const currentBytes = channelPairs * ACE_OPT_0077_RFFT16_KERNEL_SIZE * 2;
      const candidateBytes = channelPairs *
        ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES * 2;
      return Object.freeze({ operationLabel, tier: spec.tier,
        inputChannels: spec.shape.inputChannels,
        outputChannels: spec.shape.outputChannels,
        currentCoordinates: ACE_OPT_0077_RFFT16_KERNEL_SIZE,
        candidateCoordinates: ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES,
        currentBytes, candidateBytes,
        increaseBytes: candidateBytes - currentBytes });
    })
  );
  const currentBytes = routes.reduce((sum, route) =>
    sum + route.currentBytes, 0);
  const candidateBytes = routes.reduce((sum, route) =>
    sum + route.candidateBytes, 0);
  const increaseBytes = candidateBytes - currentBytes;
  if (routes.length !== 12 ||
    new Set(routes.map(({ operationLabel }) => operationLabel)).size !== 12 ||
    currentBytes !== CURRENT_PERSISTENT_WEIGHT_BYTES ||
    candidateBytes !== CANDIDATE_PERSISTENT_WEIGHT_BYTES ||
    increaseBytes !== PERSISTENT_WEIGHT_BYTE_INCREASE) {
    throw new Error("OPT-0077 persistent package payload changed");
  }
  return Object.freeze({ routeCount: routes.length,
    routes: Object.freeze(routes), currentBytes, candidateBytes, increaseBytes,
    candidateToCurrentRatio: candidateBytes / currentBytes,
    productionPackagePolicy: "replace-native-do-not-duplicate",
    nativeAndTransformedDuplicateSurvives: false });
}

export function buildOpt0077ScratchAccounting(limits?: Readonly<{
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
}>): Readonly<Record<string, unknown>> {
  const strata = buildOpt0077ProductionStrata().map((spec) => {
    const scratch = planOpt0077ScratchBytes(spec.shape);
    const logicalOutputCount = spec.shape.inputFrames *
      spec.shape.outputChannels;
    return Object.freeze({ id: spec.id, dilation: spec.dilation,
      logicalOutputCount,
      inputSpectrumBytes: scratch.inputSpectrum,
      contractionSpectrumBytes: scratch.contractionSpectrum,
      totalScratchBytes: scratch.total,
      inputSpectrumBytesPerLogicalOutput:
        scratch.inputSpectrum / logicalOutputCount,
      contractionSpectrumBytesPerLogicalOutput:
        scratch.contractionSpectrum / logicalOutputCount,
      totalBytesPerLogicalOutput: scratch.total / logicalOutputCount,
      inputSpectrumBindingCount: 1,
      contractionSpectrumBindingCount: 1,
      totalScratchBindingCount: 2 });
  });
  const maximumInputSpectrumBytes = Math.max(...strata.map((entry) =>
    entry.inputSpectrumBytes));
  const maximumContractionSpectrumBytes = Math.max(...strata.map((entry) =>
    entry.contractionSpectrumBytes));
  const maximumTotalScratchBytes = Math.max(...strata.map((entry) =>
    entry.totalScratchBytes));
  const maximumScratchBindingBytes = Math.max(maximumInputSpectrumBytes,
    maximumContractionSpectrumBytes);
  const limitReceipt = limits === undefined ? null : Object.freeze({
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    maxBufferSize: limits.maxBufferSize,
    maximumScratchBindingBytes,
    storageBindingHeadroomBytes:
      limits.maxStorageBufferBindingSize - maximumScratchBindingBytes,
    bufferHeadroomBytes: limits.maxBufferSize -
      (maximumScratchBindingBytes + 2 * STORAGE_GUARD_BYTES),
    everyScratchBindingWithinAuthenticatedLimits:
      maximumScratchBindingBytes <= limits.maxStorageBufferBindingSize &&
      maximumScratchBindingBytes + 2 * STORAGE_GUARD_BYTES <=
        limits.maxBufferSize,
  });
  return Object.freeze({ strata: Object.freeze(strata),
    maximumInputSpectrumBytes, maximumContractionSpectrumBytes,
    maximumTotalScratchBytes, maximumScratchBindingBytes,
    maximumScratchBindingCount: 2,
    nominalInputSpectrumBytesPerLogicalOutputBeforeTail: 3.2,
    nominalContractionSpectrumBytesPerLogicalOutputBeforeTail: 6.4,
    authenticatedLimits: limitReceipt,
    passed: limitReceipt === null ||
      limitReceipt.everyScratchBindingWithinAuthenticatedLimits });
}

export function buildOpt0077MathProof(): Readonly<Record<string, unknown>> {
  if (
    ACE_OPT_0077_RFFT16_LENGTH !== 16 ||
    ACE_OPT_0077_RFFT16_KERNEL_SIZE !== 7 ||
    ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE !== 10 ||
    ACE_OPT_0077_RFFT16_OVERLAP !== 6 ||
    ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES !== 16 ||
    ACE_OPT_0077_RFFT16_COORDINATE_ORDER.length !== 16
  ) throw new Error("OPT-0077 RFFT16 mathematical constants changed");

  let maximumForwardBasisError = 0;
  let maximumWeightBasisError = 0;
  let maximumCorrelationBasisError = 0;
  let comparedForwardCoordinates = 0;
  let comparedWeightCoordinates = 0;
  let comparedCorrelationOutputs = 0;
  for (let inputIndex = 0; inputIndex < 16; inputIndex += 1) {
    const input = new Float32Array(16);
    input[inputIndex] = 1;
    const radix2 = aceOpt0077Rfft16ForwardF32(input);
    const direct = aceOpt0077Rfft16DirectDftReference(input);
    requireLength(radix2, 16, "radix-2 input basis");
    requireLength(direct, 16, "direct-DFT input basis");
    for (let coordinate = 0; coordinate < 16; coordinate += 1) {
      maximumForwardBasisError = Math.max(maximumForwardBasisError,
        Math.abs(radix2[coordinate]! - direct[coordinate]!));
      comparedForwardCoordinates += 1;
    }
  }
  for (let tap = 0; tap < 7; tap += 1) {
    const filter = new Float32Array(7);
    filter[tap] = 1;
    const transformed = aceOpt0077TransformK7WeightF32(filter);
    const padded = new Float32Array(16);
    padded.set(filter);
    const direct = aceOpt0077Rfft16DirectDftReference(padded);
    requireLength(transformed, 16, "radix-2 filter basis");
    for (let coordinate = 0; coordinate < 16; coordinate += 1) {
      maximumWeightBasisError = Math.max(maximumWeightBasisError,
        Math.abs(transformed[coordinate]! - direct[coordinate]!));
      comparedWeightCoordinates += 1;
    }
  }
  for (let inputIndex = 0; inputIndex < 16; inputIndex += 1) {
    const input = new Float32Array(16);
    input[inputIndex] = 1;
    const inputSpectrum = aceOpt0077Rfft16ForwardF32(input);
    for (let tap = 0; tap < 7; tap += 1) {
      const filter = new Float32Array(7);
      filter[tap] = 1;
      const weightSpectrum = aceOpt0077TransformK7WeightF32(filter);
      const transformed = aceOpt0077Rfft16CorrelateF32(
        inputSpectrum,
        weightSpectrum,
      );
      const direct = aceOpt0077DirectK7Correlation(input, filter);
      requireLength(transformed, 10, "RFFT correlation basis");
      requireLength(direct, 10, "direct K7 correlation basis");
      for (let output = 0; output < 10; output += 1) {
        maximumCorrelationBasisError = Math.max(
          maximumCorrelationBasisError,
          Math.abs(transformed[output]! - direct[output]!),
        );
        comparedCorrelationOutputs += 1;
      }
    }
  }
  const residueProof = proveResidueAndRangeMapping();
  const passed = maximumForwardBasisError <= 2e-6 &&
    maximumWeightBasisError <= 2e-6 &&
    maximumCorrelationBasisError <= 2e-5 &&
    residueProof["passed"] === true;
  if (!passed) throw new Error("OPT-0077 independent transform proof failed");
  return Object.freeze({
    transform: "orthonormal-radix2-rfft16-real-basis",
    coordinateOrder: ACE_OPT_0077_RFFT16_COORDINATE_ORDER,
    comparedForwardCoordinates,
    comparedWeightCoordinates,
    comparedCorrelationOutputs,
    maximumForwardBasisError,
    maximumWeightBasisError,
    maximumCorrelationBasisError,
    noExtraOverallScaleFactor: maximumCorrelationBasisError <= 2e-5,
    correlationSigns: "R=Xc*Gc+Xs*Gs; I=Xc*Gs-Xs*Gc",
    residueAndRangeMapping: residueProof,
    passed,
  });
}

function requireLength(
  values: ArrayLike<number>,
  expected: number,
  label: string,
): void {
  if (values.length !== expected) {
    throw new Error(`OPT-0077 ${label} length changed`);
  }
}

function proveResidueAndRangeMapping(): Readonly<Record<string, unknown>> {
  let plannerDomains = 0;
  let ranges = 0;
  let selectedRows = 0;
  let sourceCoordinateChecks = 0;
  for (const dilation of [1, 3, 9] as const) {
    for (let frames = 1; frames <= 64; frames += 1) {
      plannerDomains += 1;
      const tiles = enumerateLogicalTiles(frames, dilation);
      for (let base = 0; base < frames; base += 1) {
        for (let count = 1; count <= frames - base; count += 1) {
          ranges += 1;
          const covered = new Map<number, number>();
          for (const tile of tiles) {
            const intersects = tile.outputs.some((row) =>
              row >= base && row < base + count
            );
            if (!intersects) continue;
            for (const row of tile.outputs) {
              if (row >= base && row < base + count) {
                covered.set(row, (covered.get(row) ?? 0) + 1);
              }
            }
          }
          for (let row = base; row < base + count; row += 1) {
            if (covered.get(row) !== 1) {
              throw new Error("OPT-0077 tile/range intersection was not exact");
            }
            selectedRows += 1;
          }
          if ([...covered.keys()].some((row) => row < base || row >= base + count)) {
            throw new Error("OPT-0077 tile/range proof selected an extra row");
          }
        }
      }
      for (const tile of tiles) {
        for (let output = 0; output < tile.outputs.length; output += 1) {
          const outputRow = tile.outputs[output]!;
          for (let tap = 0; tap < 7; tap += 1) {
            const sourceViaTile = tile.residue + dilation *
              (tile.streamStart - 3 + output + tap);
            const sourceDirect = outputRow - 3 * dilation + tap * dilation;
            if (sourceViaTile !== sourceDirect) {
              throw new Error("OPT-0077 dilation residue source mapping changed");
            }
            sourceCoordinateChecks += 1;
          }
        }
      }
    }
  }
  return Object.freeze({ plannerDomains, ranges, selectedRows,
    sourceCoordinateChecks, dilations: Object.freeze([1, 3, 9]),
    maximumFrames: 64, exactOnceCoverage: true,
    partialFinalTilesCovered: true, passed: true });
}

function enumerateLogicalTiles(
  frames: number,
  dilation: 1 | 3 | 9,
): readonly Readonly<{
  residue: number;
  streamStart: number;
  outputs: readonly number[];
}>[] {
  const tiles: Readonly<{
    residue: number;
    streamStart: number;
    outputs: readonly number[];
  }>[] = [];
  for (let residue = 0; residue < dilation && residue < frames; residue += 1) {
    const streamLength = Math.floor((frames - 1 - residue) / dilation) + 1;
    for (let streamStart = 0; streamStart < streamLength;
      streamStart += ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE) {
      const outputs: number[] = [];
      for (let index = 0;
        index < ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE &&
        streamStart + index < streamLength; index += 1) {
        outputs.push(residue + dilation * (streamStart + index));
      }
      tiles.push(Object.freeze({ residue, streamStart,
        outputs: Object.freeze(outputs) }));
    }
  }
  return Object.freeze(tiles);
}

export function summarizeOpt0077Timing(
  input: Opt0077TimingInput,
): Readonly<Record<string, unknown>> {
  requireTimestampSamples(input.current, "current", CURRENT_AGGREGATE_DISPATCHES);
  requireTimestampSamples(
    input.candidate,
    "candidate",
    CANDIDATE_AGGREGATE_DISPATCHES,
  );
  const expectedIds = buildOpt0077ProductionStrata().map(({ id }) => id);
  if (input.strata.length !== 9 ||
    input.strata.map(({ id }) => id).join("\0") !== expectedIds.join("\0")) {
    throw new Error("OPT-0077 timing strata changed");
  }
  const strata = input.strata.map((stratum) => {
    requirePositiveSamples(stratum.currentGpuMilliseconds,
      `${stratum.id} current GPU`);
    requirePositiveSamples(stratum.candidateGpuMilliseconds,
      `${stratum.id} candidate GPU`);
    const currentMedian = median(stratum.currentGpuMilliseconds);
    const candidateMedian = median(stratum.candidateGpuMilliseconds);
    return Object.freeze({ id: stratum.id,
      currentGpuMilliseconds: stratum.currentGpuMilliseconds,
      candidateGpuMilliseconds: stratum.candidateGpuMilliseconds,
      currentMedianGpuMilliseconds: currentMedian,
      candidateMedianGpuMilliseconds: candidateMedian,
      gpuSpeedup: currentMedian / candidateMedian,
      nonSlower: candidateMedian <= currentMedian });
  });
  const summarizeArm = (samples: readonly Opt0077TimestampSample[]) => {
    const gpu = samples.map(({ gpuMilliseconds }) => gpuMilliseconds);
    const wall = samples.map(({ wallMilliseconds }) => wallMilliseconds);
    return Object.freeze({ samples, meanGpuMilliseconds: mean(gpu),
      medianGpuMilliseconds: median(gpu), meanWallMilliseconds: mean(wall),
      medianWallMilliseconds: median(wall), minimumGpuMilliseconds: Math.min(...gpu),
      maximumGpuMilliseconds: Math.max(...gpu), minimumWallMilliseconds: Math.min(...wall),
      maximumWallMilliseconds: Math.max(...wall) });
  };
  const current = summarizeArm(input.current);
  const candidate = summarizeArm(input.candidate);
  const pairedRounds = input.current.map((control, roundIndex) => {
    const contender = input.candidate[roundIndex]!;
    return Object.freeze({ roundIndex,
      currentGpuMilliseconds: control.gpuMilliseconds,
      candidateGpuMilliseconds: contender.gpuMilliseconds,
      gpuSpeedup: control.gpuMilliseconds / contender.gpuMilliseconds,
      gpuWin: contender.gpuMilliseconds < control.gpuMilliseconds,
      currentWallMilliseconds: control.wallMilliseconds,
      candidateWallMilliseconds: contender.wallMilliseconds,
      wallSpeedup: control.wallMilliseconds / contender.wallMilliseconds,
      wallWin: contender.wallMilliseconds < control.wallMilliseconds });
  });
  const meanGpuSpeedup = current.meanGpuMilliseconds /
    candidate.meanGpuMilliseconds;
  const medianGpuSpeedup = current.medianGpuMilliseconds /
    candidate.medianGpuMilliseconds;
  const meanWallSpeedup = current.meanWallMilliseconds /
    candidate.meanWallMilliseconds;
  const medianWallSpeedup = current.medianWallMilliseconds /
    candidate.medianWallMilliseconds;
  const lowerWallSpeedup = Math.min(meanWallSpeedup, medianWallSpeedup);
  const projectedSavingMilliseconds = SCOPED_PLANNING_MILLISECONDS *
    (1 - 1 / lowerWallSpeedup);
  const gpuPairWins = pairedRounds.filter(({ gpuWin }) => gpuWin).length;
  const wallPairWins = pairedRounds.filter(({ wallWin }) => wallWin).length;
  const gates = Object.freeze({
    everyStratumMedianGpuNonSlower: strata.every(({ nonSlower }) => nonSlower),
    requiredPairWins: REQUIRED_PAIR_WINS,
    gpuPairWins,
    wallPairWins,
    gpuPairWinsPassed: gpuPairWins >= REQUIRED_PAIR_WINS,
    wallPairWinsPassed: wallPairWins >= REQUIRED_PAIR_WINS,
    requiredMeanAndMedianSpeedup: REQUIRED_SPEEDUP,
    meanGpuSpeedupPassed: meanGpuSpeedup >= REQUIRED_SPEEDUP,
    medianGpuSpeedupPassed: medianGpuSpeedup >= REQUIRED_SPEEDUP,
    meanWallSpeedupPassed: meanWallSpeedup >= REQUIRED_SPEEDUP,
    medianWallSpeedupPassed: medianWallSpeedup >= REQUIRED_SPEEDUP,
    scopedPlanningMilliseconds: SCOPED_PLANNING_MILLISECONDS,
    lowerWallSpeedup,
    projectedSavingMilliseconds,
    requiredProjectedSavingMilliseconds: REQUIRED_PROJECTED_SAVING_MILLISECONDS,
    projectedSavingPassed:
      projectedSavingMilliseconds >= REQUIRED_PROJECTED_SAVING_MILLISECONDS,
  });
  const passed = gates.everyStratumMedianGpuNonSlower &&
    gates.gpuPairWinsPassed && gates.wallPairWinsPassed &&
    gates.meanGpuSpeedupPassed && gates.medianGpuSpeedupPassed &&
    gates.meanWallSpeedupPassed && gates.medianWallSpeedupPassed &&
    gates.projectedSavingPassed;
  return Object.freeze({ sampleCountPerArm: TIMING_ROUNDS.length,
    currentAggregateDispatches: CURRENT_AGGREGATE_DISPATCHES,
    candidateAggregateDispatches: CANDIDATE_AGGREGATE_DISPATCHES,
    strata: Object.freeze(strata), arms: Object.freeze({ current, candidate }),
    pairedRounds: Object.freeze(pairedRounds),
    speedup: Object.freeze({ meanGpu: meanGpuSpeedup,
      medianGpu: medianGpuSpeedup, meanWall: meanWallSpeedup,
      medianWall: medianWallSpeedup }), gates, passed });
}

function requireTimestampSamples(
  samples: readonly Opt0077TimestampSample[],
  label: string,
  dispatchCount: number,
): void {
  if (samples.length !== TIMING_ROUNDS.length || samples.some((sample) =>
    !Number.isFinite(sample.gpuMilliseconds) || sample.gpuMilliseconds <= 0 ||
    !Number.isFinite(sample.wallMilliseconds) || sample.wallMilliseconds <= 0 ||
    !Number.isSafeInteger(sample.gpuElapsedNanoseconds) ||
    sample.gpuElapsedNanoseconds <= 0 || sample.dispatchCount !== dispatchCount ||
    sample.commandBufferCount !== 1 || sample.queueDrainCount !== 1
  )) throw new Error(`OPT-0077 ${label} requires six valid timestamp samples`);
}

function requirePositiveSamples(values: readonly number[], label: string): void {
  if (values.length !== TIMING_ROUNDS.length ||
    values.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error(`OPT-0077 ${label} requires six positive samples`);
  }
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return (sorted[2]! + sorted[3]!) / 2;
}

export function parseOpt0077ThermalLaunchGate(
  parameters: URLSearchParams,
  readyAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0077ThermalLaunchGate {
  const source = requiredParameter(parameters, "thermalSource");
  const command = requiredParameter(parameters, "thermalCommand");
  const traceStartedAtEpochMilliseconds = requiredFiniteParameter(
    parameters, "thermalTraceStartedAtEpochMilliseconds");
  const gateStartedAtEpochMilliseconds = requiredFiniteParameter(
    parameters, "thermalGateStartedAtEpochMilliseconds");
  const gateCompletedAtEpochMilliseconds = requiredFiniteParameter(
    parameters, "thermalGateCompletedAtEpochMilliseconds");
  const observationCount = requiredIntegerParameter(
    parameters, "thermalGateObservations");
  const pollMilliseconds = requiredIntegerParameter(
    parameters, "thermalPollMilliseconds");
  const maximumPollGapMilliseconds = requiredFiniteParameter(
    parameters, "thermalGateMaximumPollGapMilliseconds");
  const nonNominalObservationCount = requiredIntegerParameter(
    parameters, "thermalGateNonNominalObservations");
  const missingObservationCount = requiredIntegerParameter(
    parameters, "thermalGateMissingObservations");
  const duration = gateCompletedAtEpochMilliseconds -
    gateStartedAtEpochMilliseconds;
  const minimumObservations = Math.floor(duration / THERMAL_POLL_MILLISECONDS) + 1;
  const readyToGateDelayMilliseconds = gateStartedAtEpochMilliseconds -
    readyAtEpochMilliseconds;
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    gateCompletedAtEpochMilliseconds;
  if (source !== THERMAL_SOURCE || command !== THERMAL_COMMAND ||
    pollMilliseconds !== THERMAL_POLL_MILLISECONDS ||
    traceStartedAtEpochMilliseconds > readyAtEpochMilliseconds ||
    readyToGateDelayMilliseconds < 0 || duration < MINIMUM_NOMINAL_MILLISECONDS ||
    observationCount < minimumObservations || maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_GAP_MILLISECONDS ||
    nonNominalObservationCount !== 0 || missingObservationCount !== 0 ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS) {
    throw new Error("OPT-0077 fresh continuous nominal thermal launch gate failed");
  }
  return Object.freeze({ source: THERMAL_SOURCE, command: THERMAL_COMMAND,
    traceStartedAtEpochMilliseconds, gateStartedAtEpochMilliseconds,
    gateCompletedAtEpochMilliseconds, observationCount,
    pollMilliseconds: THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds, nonNominalObservationCount: 0,
    missingObservationCount: 0, readyToGateDelayMilliseconds,
    launchDelayMilliseconds });
}

export function parseOpt0077ThermalCompletion(
  parameters: URLSearchParams,
  launch: Opt0077ThermalLaunchGate,
  cleanupCompletedAtEpochMilliseconds: number,
): Opt0077ThermalCompletion {
  const schema = requiredParameter(parameters, "thermalTraceSchema");
  const sha256 = requiredParameter(parameters, "thermalTraceSha256");
  const byteLength = requiredIntegerParameter(parameters, "thermalTraceByteLength");
  const completedAtEpochMilliseconds = requiredFiniteParameter(
    parameters, "thermalTraceCompletedAtEpochMilliseconds");
  const observationCount = requiredIntegerParameter(
    parameters, "thermalTraceObservations");
  const maximumPollGapMilliseconds = requiredFiniteParameter(
    parameters, "thermalTraceMaximumPollGapMilliseconds");
  const nonNominalObservationCount = requiredIntegerParameter(
    parameters, "thermalTraceNonNominalObservations");
  const missingObservationCount = requiredIntegerParameter(
    parameters, "thermalTraceMissingObservations");
  const initialLevel = requiredIntegerParameter(parameters,
    "thermalTraceInitialLevel");
  const finalLevel = requiredIntegerParameter(parameters, "thermalTraceFinalLevel");
  const duration = completedAtEpochMilliseconds -
    launch.traceStartedAtEpochMilliseconds;
  const minimumObservations = Math.floor(duration / THERMAL_POLL_MILLISECONDS) + 1;
  if (schema !== THERMAL_TRACE_SCHEMA || !/^[0-9a-f]{64}$/u.test(sha256) ||
    byteLength <= 0 || completedAtEpochMilliseconds <
      cleanupCompletedAtEpochMilliseconds || observationCount < minimumObservations ||
    maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_GAP_MILLISECONDS ||
    nonNominalObservationCount !== 0 || missingObservationCount !== 0 ||
    initialLevel !== 0 || finalLevel !== 0) {
    throw new Error("OPT-0077 complete through-cleanup thermal trace failed");
  }
  return Object.freeze({ schema: THERMAL_TRACE_SCHEMA, sha256, byteLength,
    completedAtEpochMilliseconds, observationCount, maximumPollGapMilliseconds,
    nonNominalObservationCount: 0, missingObservationCount: 0,
    initialLevel: 0, finalLevel: 0, coversCleanup: true });
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.trim() === "") {
    throw new Error(`OPT-0077 field ${name} is missing`);
  }
  return value.trim();
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) throw new Error(`OPT-0077 field ${name} invalid`);
  return value;
}

function requiredIntegerParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = requiredFiniteParameter(parameters, name);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`OPT-0077 field ${name} must be an integer`);
  }
  return value;
}

if (typeof document !== "undefined") install();

function install(): void {
  const progress = requireElement<HTMLElement>("#progress");
  const runButton = requireElement<HTMLButtonElement>("#run");
  const finalizeButton = requireElement<HTMLButtonElement>("#finalize");
  const launchFields = requireElement<HTMLFieldSetElement>("#thermal-gate");
  const completionFields = requireElement<HTMLFieldSetElement>(
    "#thermal-completion",
  );
  let active: PreparedHarness | undefined;
  let running: PreparedHarness | undefined;
  let pending: Readonly<Record<string, unknown>> | undefined;
  let started = false;

  void prepareHarness((message) => progress.textContent = message).then(
    (prepared) => {
      active = prepared;
      document.body.dataset.status = "ready";
      progress.textContent =
        "READY — transform proof, 29-case correctness, and symmetric warmup passed; timing has not run";
      launchFields.disabled = false;
      runButton.disabled = false;
    },
    (error: unknown) => publishFailure(error, "preparation"),
  );

  runButton.addEventListener("click", () => {
    if (started || active === undefined) return;
    started = true;
    runButton.disabled = true;
    launchFields.disabled = true;
    document.body.dataset.status = "running";
    const prepared = active;
    active = undefined;
    running = prepared;
    const launchedAtEpochMilliseconds = Date.now();
    let thermalLaunch: Opt0077ThermalLaunchGate;
    try {
      thermalLaunch = parseOpt0077ThermalLaunchGate(
        fieldParameters("#thermal-gate"),
        prepared.readyAtEpochMilliseconds,
        launchedAtEpochMilliseconds,
      );
    } catch (error) {
      void prepared.cleanup().then((cleanup) => {
        running = undefined;
        publishFailure(error, "thermal-launch", { cleanup });
      });
      return;
    }
    void runTiming(prepared, thermalLaunch,
      (message) => progress.textContent = message).then((receipt) => {
        running = undefined;
        pending = receipt;
        completionFields.disabled = false;
        finalizeButton.disabled = false;
        document.body.dataset.status = "awaiting-thermal-completion";
        progress.textContent =
          "measurement and cleanup complete — keep polling, enter the through-cleanup trace, then finalize";
      }, (error: unknown) => {
        void prepared.cleanup().then((cleanup) => {
          running = undefined;
          publishFailure(error, "timing", { thermalLaunch, cleanup });
        });
      });
  }, { once: true });

  finalizeButton.addEventListener("click", () => {
    if (pending === undefined) return;
    const cleanupCompletedAtEpochMilliseconds = Number(
      pending["cleanupCompletedAtEpochMilliseconds"],
    );
    const thermalLaunch = pending["thermalLaunch"] as Opt0077ThermalLaunchGate;
    try {
      const thermalCompletion = parseOpt0077ThermalCompletion(
        fieldParameters("#thermal-completion"),
        thermalLaunch,
        cleanupCompletedAtEpochMilliseconds,
      );
      const passed = pending["inPagePassed"] === true;
      const receipt = Object.freeze({ ...pending,
        schema: RECEIPT_SCHEMA, experiment: EXPERIMENT_ID,
        status: "completed", passed,
        thermal: Object.freeze({ launch: thermalLaunch,
          completion: thermalCompletion, passed: true }),
        decision: Object.freeze({
          disposition: passed
            ? "positive-primitive-package-c512-experiment-authorized"
            : "negative-stop-exact-rfft16-mechanism",
          packageC512FollowupAuthorized: passed,
          productionIntegrationAuthorized: false,
          productSpeedClaim: false,
          qualityOrListeningClaim: false,
        }),
      });
      publish(receipt, passed ? "passed" : "failed");
      progress.textContent = passed
        ? "completed — primitive gate passed; production remains unchanged"
        : "completed — exact RFFT16 primitive stopped at its frozen gate";
      pending = undefined;
      finalizeButton.disabled = true;
      completionFields.disabled = true;
    } catch (error) {
      publishFailure(error, "thermal-completion", { pending });
    }
  });

  window.addEventListener("beforeunload", () => {
    void active?.cleanup();
    void running?.cleanup();
    active = undefined;
    running = undefined;
  });
}

async function prepareHarness(
  update: (message: string) => void,
): Promise<PreparedHarness> {
  requireLittleEndianHost();
  update("proving fixed radix-2 RFFT16 against independent direct DFT");
  const mathProof = buildOpt0077MathProof();
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const cases = buildOpt0077CorrectnessCases();
  const resourceContract = Object.freeze({
    persistentPayload: buildOpt0077PersistentPayloadAccounting(),
    scratch: buildOpt0077ScratchAccounting(Object.freeze({
      maxStorageBufferBindingSize:
        adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    })),
  });
  if ((resourceContract.scratch as Readonly<Record<string, unknown>>)[
    "passed"] !== true) {
    throw new Error("OPT-0077 scratch exceeded authenticated adapter limits");
  }
  const required = maximumRequiredBindingBytes(cases);
  const device = await adapter.requestDevice({
    label: "ace-opt-0077-rfft16-primitive-device",
    requiredFeatures: ["shader-f16", "subgroups", "timestamp-query"],
    requiredLimits: {
      maxBufferSize: required.maxBufferSize,
      maxStorageBufferBindingSize: required.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 128,
    },
  });
  const tracker = new BufferTracker();
  const uncapturedErrors: string[] = [];
  const deviceLosses: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncapturedErrors.push(event.error.message);
  });
  void device.lost.then((loss) => {
    if (loss.reason !== "destroyed") {
      deviceLosses.push(`${loss.reason}: ${loss.message}`);
    }
  });
  const capability: AceFp16VaeConv1dFixedSubgroupCapability = Object.freeze({
    subgroupMinSize: 32,
    subgroupMaxSize: 32,
  });
  let currentKernel: AceOpt0057VaeK7ShapeSelectorKernel | undefined;
  let candidateKernel: AceOpt0077VaeK7Rfft16Kernel | undefined;
  let scalarOracleKernel: AceFp16VaeConv1dSubgroupKernel | undefined;
  let querySet: GPUQuerySet | undefined;
  let queryResolve: GPUBuffer | undefined;
  let queryReadback: GPUBuffer | undefined;
  let cleanupPromise: Promise<Readonly<Record<string, unknown>>> | undefined;
  let postDestroyProbe: (() => Promise<boolean>) | undefined;

  const cleanup = (): Promise<Readonly<Record<string, unknown>>> => {
    if (cleanupPromise !== undefined) return cleanupPromise;
    cleanupPromise = (async () => {
      const cleanupStartedAtEpochMilliseconds = Date.now();
      await device.queue.onSubmittedWorkDone().catch(() => undefined);
      currentKernel?.destroy();
      candidateKernel?.destroy();
      scalarOracleKernel?.destroy();
      querySet?.destroy();
      const postDestroyRejected = postDestroyProbe === undefined
        ? false
        : await postDestroyProbe();
      candidateKernel?.destroy();
      tracker.destroyAll();
      const memory = tracker.receipt();
      device.destroy();
      return Object.freeze({ ...memory, cleanupStartedAtEpochMilliseconds,
        cleanupCompletedAtEpochMilliseconds: Date.now(), queueDrained: true,
        ownersDestroyedIdempotently: true, postDestroyRejected,
        zeroLiveBuffers: memory.liveBufferCount === 0,
        zeroLiveBytes: memory.liveBytes === 0,
        createdEqualsDestroyed:
          memory.createdBufferCount === memory.destroyedBufferCount,
        mapsBalanced: memory.mapCount === memory.unmapCount &&
          memory.activeMapCount === 0 });
    })();
    return cleanupPromise;
  };

  try {
    update("authenticating source and generated-WGSL identities");
    const identity = await buildIdentity(adapter, cases, mathProof);
    currentKernel = AceOpt0057VaeK7ShapeSelectorKernel.create(
      device,
      capability,
    );
    candidateKernel = AceOpt0077VaeK7Rfft16Kernel.create(device, capability);
    scalarOracleKernel = AceFp16VaeConv1dSubgroupKernel.create(
      device,
      capability,
    );
    querySet = device.createQuerySet({
      label: "opt0077-rfft16-timestamps",
      type: "timestamp",
      count: 2,
    });
    queryResolve = tracker.create(device, {
      label: "opt0077-timestamp-resolve",
      size: TIMESTAMP_QUERY_BYTES,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    queryReadback = tracker.create(device, {
      label: "opt0077-timestamp-readback",
      size: TIMESTAMP_QUERY_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const aggregate = createNumericalAccumulator();
    const productionAggregate = createNumericalAccumulator();
    const adversarialAggregate = createNumericalAccumulator();
    const candidateCurrentAggregate = createNumericalAccumulator();
    const productionCandidateCurrentAggregate = createNumericalAccumulator();
    const adversarialCandidateCurrentAggregate = createNumericalAccumulator();
    const preparedProduction: PreparedCase[] = [];
    const weightCache = new Map<string, PreparedWeights>();
    const caseReceipts: Readonly<Record<string, unknown>>[] = [];
    let candidateScalarComparedU16Count = 0;
    let candidateRepeatComparedU16Count = 0;
    for (const [index, spec] of cases.entries()) {
      update(`correctness ${index + 1}/${cases.length}: ${spec.id}`);
      const prepared = await prepareCase(device, tracker, currentKernel,
        candidateKernel, scalarOracleKernel, spec, weightCache);
      const receipt = await verifyCase(device, tracker, prepared, aggregate,
        spec.kind === "production" ? productionAggregate : adversarialAggregate,
        candidateCurrentAggregate,
        spec.kind === "production" ? productionCandidateCurrentAggregate :
          adversarialCandidateCurrentAggregate);
      caseReceipts.push(receipt);
      candidateScalarComparedU16Count += Number(
        receipt["candidateScalarComparedU16Count"],
      );
      candidateRepeatComparedU16Count += Number(
        receipt["candidateRepeatComparedU16Count"],
      );
      if (spec.kind === "production") preparedProduction.push(prepared);
      else destroyPreparedCase(tracker, prepared);
      await yieldToBrowser();
    }
    await device.queue.onSubmittedWorkDone();
    await settlePostDrainEvents();
    const productionWorkingSet = buildPreparedProductionWorkingSetReceipt(
      preparedProduction,
    );
    const aggregateMetrics = summarizeNumerical(aggregate);
    const productionMetrics = summarizeNumerical(productionAggregate);
    const adversarialMetrics = summarizeNumerical(adversarialAggregate);
    const candidateCurrentMetrics = summarizeNumerical(
      candidateCurrentAggregate,
    );
    const productionCandidateCurrentMetrics = summarizeNumerical(
      productionCandidateCurrentAggregate,
    );
    const adversarialCandidateCurrentMetrics = summarizeNumerical(
      adversarialCandidateCurrentAggregate,
    );
    const resultHashes = await buildOpt0077ResultHashes(caseReceipts);
    const numericalEnvelopePassed = numericalPassed(aggregateMetrics);
    const correctness = Object.freeze({ caseCount: caseReceipts.length,
      productionStratumCount: preparedProduction.length,
      adversarialCaseCount: cases.length - preparedProduction.length,
      productionProbeCount: preparedProduction.reduce(
        (sum, item) => sum + item.spec.probes.length, 0),
      spectralCaseCount: cases.filter(({ kind }) =>
        kind === "dc" || kind === "nyquist" ||
        /^cos[1-7]$|^sin[1-7]$/u.test(kind)).length,
      candidateScalarComparedU16Count, candidateRepeatComparedU16Count,
      mathProof, resultHashes, numericalEnvelope: Object.freeze({
        thresholds: Object.freeze({ nrmseMaximum: NRMSE_MAXIMUM,
          snrMinimumDb: SNR_MINIMUM_DB, pearsonMinimum: PEARSON_MINIMUM,
          relativeMaximumAbsoluteErrorMaximum:
            RELATIVE_MAXIMUM_ABSOLUTE_ERROR_MAXIMUM }),
        aggregate: aggregateMetrics, production: productionMetrics,
        adversarialAndSpectral: adversarialMetrics,
        candidateVersusCurrent: Object.freeze({
          aggregate: candidateCurrentMetrics,
          production: productionCandidateCurrentMetrics,
          adversarialAndSpectral: adversarialCandidateCurrentMetrics,
        }),
        passed: numericalEnvelopePassed }),
      cases: Object.freeze(caseReceipts),
      allCasesPassed: caseReceipts.every((receipt) => receipt["passed"] === true),
      uncapturedGpuErrorCount: uncapturedErrors.length,
      deviceLossCount: deviceLosses.length,
      completedBeforeReady: true,
      passed: mathProof["passed"] === true && numericalEnvelopePassed &&
        caseReceipts.every((receipt) => receipt["passed"] === true) &&
        uncapturedErrors.length === 0 && deviceLosses.length === 0 });
    if (!correctness.passed) {
      throw new Error("OPT-0077 correctness gate did not pass");
    }
    update("symmetrically warming weighted 42/126-dispatch composites");
    const warmupStartedAtEpochMilliseconds = Date.now();
    await executeAggregate(device, preparedProduction, "current");
    await executeAggregate(device, preparedProduction, "candidate");
    const warmupCompletedAtEpochMilliseconds = Date.now();
    await settlePostDrainEvents();
    if (uncapturedErrors.length !== 0 || deviceLosses.length !== 0) {
      throw new Error("OPT-0077 GPU failure during warmup");
    }
    const first = preparedProduction[0]!;
    postDestroyProbe = async () => {
      try {
        const dispatch = first.timingDispatches!.candidate;
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        dispatch.encode(pass);
        pass.end();
        return false;
      } catch {
        return true;
      }
    };
    const readyAtEpochMilliseconds = Date.now();
    const authenticatedResourceContract = Object.freeze({
      ...resourceContract,
      productionWorkingSet,
    });
    return Object.freeze({ adapter, device, tracker, currentKernel,
      candidateKernel, scalarOracleKernel, querySet, queryResolve,
      queryReadback, productionCases: Object.freeze(preparedProduction),
      correctness, identity, resourceContract: authenticatedResourceContract,
      warmup: Object.freeze({ warmupStartedAtEpochMilliseconds,
        warmupCompletedAtEpochMilliseconds,
        currentAggregateDispatches: CURRENT_AGGREGATE_DISPATCHES,
        candidateAggregateDispatches: CANDIDATE_AGGREGATE_DISPATCHES,
        productionWeightBindingCount: 12,
        distinctProductionWeightKeys: 12,
        c128MultiplicitySplit: Object.freeze({ block3: 3, block4: 6 }),
        symmetric: true, completedBeforeReady: true }),
      readyAtEpochMilliseconds, uncapturedErrors, deviceLosses, cleanup });
  } catch (error) {
    await cleanup();
    throw error;
  }
}

function maximumRequiredBindingBytes(
  cases: readonly Opt0077CaseSpec[],
): Readonly<{ maxStorageBufferBindingSize: number; maxBufferSize: number }> {
  let maximum = 0;
  for (const spec of cases) {
    const plan = planAceFp16VaeConv1d(spec.shape, "float16");
    const fullScratch = planOpt0077ScratchBytes(spec.shape);
    const transformedWeightBytes = spec.shape.inputChannels *
      spec.shape.outputChannels * ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES * 2;
    maximum = Math.max(maximum, plan.inputBindingBytes,
      plan.weightBindingBytes, plan.biasBindingBytes, plan.outputBindingBytes,
      transformedWeightBytes, fullScratch.inputSpectrum,
      fullScratch.contractionSpectrum);
  }
  return Object.freeze({ maxStorageBufferBindingSize: maximum,
    maxBufferSize: maximum + 2 * STORAGE_GUARD_BYTES });
}

async function prepareCase(
  device: GPUDevice,
  tracker: BufferTracker,
  currentKernel: AceOpt0057VaeK7ShapeSelectorKernel,
  candidateKernel: AceOpt0077VaeK7Rfft16Kernel,
  scalarOracleKernel: AceFp16VaeConv1dSubgroupKernel,
  spec: Opt0077CaseSpec,
  weightCache: Map<string, PreparedWeights>,
): Promise<PreparedCase> {
  const plan = planAceFp16VaeConv1d(spec.shape, "float16");
  const inputWords = buildInputWords(spec, plan);
  const biasWords = buildBiasWords(spec, plan);
  const input = u16Buffer(device, tracker, `${spec.id}-input`,
    plan.inputBindingBytes, inputWords, GPUBufferUsage.STORAGE);
  const bias = u16Buffer(device, tracker, `${spec.id}-bias`,
    plan.biasBindingBytes, biasWords, GPUBufferUsage.STORAGE);
  const productionWeightBindings = spec.kind === "production"
    ? productionWeightBindingsForStratum(spec)
    : Object.freeze([]);
  const operationWeights = new Map<string, PreparedWeights>();
  const requiredOperationLabels = spec.kind === "production"
    ? productionWeightBindings.map(({ operationLabel }) => operationLabel)
    : Object.freeze([spec.operationLabel]);
  for (const operationLabel of requiredOperationLabels) {
    const cacheKey = weightCacheKey(spec, operationLabel);
    let operationWeight = weightCache.get(cacheKey);
    if (operationWeight === undefined) {
      operationWeight = await prepareWeights(
        device,
        tracker,
        spec,
        plan,
        operationLabel,
        cacheKey,
      );
      weightCache.set(cacheKey, operationWeight);
    }
    operationWeights.set(operationLabel, operationWeight);
  }
  const weights = operationWeights.get(spec.operationLabel);
  if (weights === undefined) {
    throw new Error(`${spec.id} representative weight binding was not prepared`);
  }
  const output = guardedBuffer(device, tracker, `${spec.id}-output`,
    plan.outputBindingBytes, "f16");
  const fullScratch = planOpt0077ScratchBytes(spec.shape);
  const inputSpectrum = guardedBuffer(device, tracker,
    `${spec.id}-input-spectrum`, fullScratch.inputSpectrum, "f16");
  const contractionSpectrum = guardedBuffer(device, tracker,
    `${spec.id}-contraction-spectrum`, fullScratch.contractionSpectrum, "f32");
  const controls = new Map<string, GPUBuffer>();
  const allProbes = spec.kind === "production"
    ? Object.freeze([...spec.probes,
        probe("full", 0, TIMING_FRAMES, spec.shape.outputChannels)])
    : spec.probes;
  for (const probeSpec of allProbes) {
    controls.set(probeSpec.id, rangeControlBuffer(device, tracker,
      `${spec.id}-${probeSpec.id}-range`, probeSpec));
  }
  const shared = Object.freeze({ input: binding(input, plan.inputBindingBytes),
    bias: binding(bias, plan.biasBindingBytes), output: output.binding });
  const dispatchEntries = await Promise.all(allProbes.map(async (probeSpec) => {
    const control = controls.get(probeSpec.id)!;
    const range = rangeBinding(control, probeSpec);
    const currentDispatch = await currentKernel.createDispatch(
      `${spec.id}-${probeSpec.id}-current`, spec.operationLabel, spec.shape,
      Object.freeze({ ...shared,
        weight: binding(weights.currentPacked, weights.currentPackedBytes) }),
      "float16", range,
    );
    const candidateDispatch = await candidateKernel.createDispatch(
      `${spec.id}-${probeSpec.id}-candidate`, spec.shape,
      Object.freeze({ input: shared.input,
        transformedWeight: binding(weights.candidateTransformed,
          weights.candidateTransformedBytes),
        bias: shared.bias, output: shared.output }),
      "float16", range,
      Object.freeze({ inputSpectrum: inputSpectrum.binding,
        contractionSpectrum: contractionSpectrum.binding }),
    );
    const scalarDispatch = await scalarOracleKernel.createDispatch(
      `${spec.id}-${probeSpec.id}-scalar-oracle`, spec.shape,
      Object.freeze({ ...shared,
        weight: binding(weights.native, weights.nativeBytes) }),
      "float16", range,
    );
    if (currentDispatch.selectorKernelId !==
        ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID ||
      currentDispatch.owner !== "row-reuse-k4" ||
      candidateDispatch.kernelId !== ACE_OPT_0077_VAE_K7_RFFT16_KERNEL_ID ||
      candidateDispatch.stageDispatchCount !== 3 ||
      scalarDispatch.kernelId !== ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID ||
      candidateDispatch.outputRange.base !== probeSpec.base ||
      candidateDispatch.outputRange.count !== probeSpec.count ||
      scalarDispatch.outputRange.base !== probeSpec.base ||
      scalarDispatch.outputRange.count !== probeSpec.count ||
      candidateDispatch.scratchBytes.inputSpectrum >
        inputSpectrum.logicalBytes ||
      candidateDispatch.scratchBytes.contractionSpectrum >
        contractionSpectrum.logicalBytes) {
      throw new Error(`${spec.id} ${probeSpec.id} dispatch identity changed`);
    }
    const entries: Readonly<Record<ExecutionArm, Encodable>> = Object.freeze({
      current: wrapDispatch(currentDispatch, probeSpec, 1),
      candidate: wrapDispatch(candidateDispatch, probeSpec, 3,
        candidateDispatch.scratchBytes),
      "scalar-oracle": wrapDispatch(scalarDispatch, probeSpec, 1),
    });
    return [probeSpec.id, entries] as const;
  }));
  const dispatches = Object.freeze(Object.fromEntries(dispatchEntries)) as
    Readonly<Record<ProbeId, Readonly<Record<ExecutionArm, Encodable>>>>;
  const timingDispatches = spec.kind === "production"
    ? Object.freeze({ current: dispatches.full.current,
        candidate: dispatches.full.candidate })
    : undefined;
  const aggregateTimingMembers = spec.kind === "production" &&
      timingDispatches !== undefined
    ? Object.freeze(await Promise.all(productionWeightBindings.map(
        async (member): Promise<PreparedTimingMember> => {
          const memberWeights = operationWeights.get(member.operationLabel);
          if (memberWeights === undefined) {
            throw new Error(`${member.operationLabel} weight binding was not prepared`);
          }
          const weightKey = weightCacheKey(spec, member.operationLabel);
          if (member.operationLabel === spec.operationLabel) {
            return Object.freeze({ operationLabel: member.operationLabel,
              weightKey, multiplicity: member.multiplicity,
              dispatches: timingDispatches, weights: memberWeights });
          }
          const fullProbe = allProbes.find(({ id }) => id === "full");
          const fullControl = controls.get("full");
          if (fullProbe === undefined || fullControl === undefined) {
            throw new Error(`${spec.id} full timing range was not prepared`);
          }
          const range = rangeBinding(fullControl, fullProbe);
          const currentDispatch = await currentKernel.createDispatch(
            `${spec.id}-${member.operationLabel}-full-current`,
            member.operationLabel,
            spec.shape,
            Object.freeze({ ...shared,
              weight: binding(memberWeights.currentPacked,
                memberWeights.currentPackedBytes) }),
            "float16",
            range,
          );
          const candidateDispatch = await candidateKernel.createDispatch(
            `${spec.id}-${member.operationLabel}-full-candidate`,
            spec.shape,
            Object.freeze({ input: shared.input,
              transformedWeight: binding(memberWeights.candidateTransformed,
                memberWeights.candidateTransformedBytes),
              bias: shared.bias, output: shared.output }),
            "float16",
            range,
            Object.freeze({ inputSpectrum: inputSpectrum.binding,
              contractionSpectrum: contractionSpectrum.binding }),
          );
          if (currentDispatch.selectorKernelId !==
              ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID ||
            currentDispatch.owner !== "row-reuse-k4" ||
            candidateDispatch.kernelId !== ACE_OPT_0077_VAE_K7_RFFT16_KERNEL_ID ||
            candidateDispatch.stageDispatchCount !== 3 ||
            candidateDispatch.outputRange.base !== fullProbe.base ||
            candidateDispatch.outputRange.count !== fullProbe.count) {
            throw new Error(`${member.operationLabel} timing dispatch changed`);
          }
          return Object.freeze({ operationLabel: member.operationLabel,
            weightKey, multiplicity: member.multiplicity,
            dispatches: Object.freeze({
              current: wrapDispatch(currentDispatch, fullProbe, 1),
              candidate: wrapDispatch(candidateDispatch, fullProbe, 3,
                candidateDispatch.scratchBytes),
            }), weights: memberWeights });
        },
      )))
    : undefined;
  return Object.freeze({ spec, plan, output, inputSpectrum,
    contractionSpectrum, correctnessDispatches: dispatches,
    ...(timingDispatches === undefined ? {} : { timingDispatches }),
    ...(aggregateTimingMembers === undefined ? {} : { aggregateTimingMembers }),
    ownedBuffers: Object.freeze([input, bias, output.buffer, output.prefill,
      inputSpectrum.buffer, inputSpectrum.prefill,
      contractionSpectrum.buffer, contractionSpectrum.prefill,
      ...controls.values()]), weightProof: weights.proof });
}

function buildPreparedProductionWorkingSetReceipt(
  cases: readonly PreparedCase[],
): Readonly<Record<string, unknown>> {
  const members = cases.flatMap((prepared) => {
    if (prepared.spec.kind !== "production" ||
      prepared.aggregateTimingMembers === undefined) {
      throw new Error("OPT-0077 production timing member was not prepared");
    }
    return prepared.aggregateTimingMembers.map((member) => Object.freeze({
      stratumId: prepared.spec.id,
      tier: prepared.spec.tier,
      dilation: prepared.spec.dilation,
      operationLabel: member.operationLabel,
      weightKey: member.weightKey,
      multiplicity: member.multiplicity,
      currentStageDispatches: member.dispatches.current.stageDispatchCount,
      candidateStageDispatches: member.dispatches.candidate.stageDispatchCount,
      nativeBytes: member.weights.nativeBytes,
      currentPackedBytes: member.weights.currentPackedBytes,
      candidateTransformedBytes: member.weights.candidateTransformedBytes,
      nativeSha256: member.weights.proof["nativeSha256"],
      currentPackedSha256: member.weights.proof["currentPackedSha256"],
      candidateTransformedSha256:
        member.weights.proof["candidateTransformedSha256"],
      proofCacheKey: member.weights.proof["cacheKey"],
      proofOperationLabel: member.weights.proof["operationLabel"],
      weights: member.weights,
    }));
  });
  const publicMapping = members.map(({ weights: _weights, ...member }) =>
    Object.freeze(member)
  );
  const planned = buildOpt0077ProductionWeightBindingPlan();
  const actualPlan = publicMapping.map(({ stratumId, tier, dilation,
    operationLabel, multiplicity }) => Object.freeze({ stratumId, tier,
    dilation, operationLabel, multiplicity }));
  const distinctWeightKeyCount = new Set(
    members.map(({ weightKey }) => weightKey),
  ).size;
  const distinctNativeBufferCount = new Set(
    members.map(({ weights }) => weights.native),
  ).size;
  const distinctCurrentPackedBufferCount = new Set(
    members.map(({ weights }) => weights.currentPacked),
  ).size;
  const distinctCandidateTransformedBufferCount = new Set(
    members.map(({ weights }) => weights.candidateTransformed),
  ).size;
  const distinctNativeSha256Count = new Set(
    members.map(({ nativeSha256 }) => String(nativeSha256)),
  ).size;
  const distinctCurrentPackedSha256Count = new Set(
    members.map(({ currentPackedSha256 }) => String(currentPackedSha256)),
  ).size;
  const distinctCandidateTransformedSha256Count = new Set(
    members.map(({ candidateTransformedSha256 }) =>
      String(candidateTransformedSha256)),
  ).size;
  const currentAggregateDispatches = members.reduce((sum, member) =>
    sum + member.multiplicity * member.currentStageDispatches, 0);
  const candidateAggregateDispatches = members.reduce((sum, member) =>
    sum + member.multiplicity * member.candidateStageDispatches, 0);
  const nativeBytes = members.reduce((sum, member) =>
    sum + member.nativeBytes, 0);
  const currentPackedBytes = members.reduce((sum, member) =>
    sum + member.currentPackedBytes, 0);
  const candidateTransformedBytes = members.reduce((sum, member) =>
    sum + member.candidateTransformedBytes, 0);
  if (members.length !== 12 || distinctWeightKeyCount !== 12 ||
    distinctNativeBufferCount !== 12 || distinctCurrentPackedBufferCount !== 12 ||
    distinctCandidateTransformedBufferCount !== 12 ||
    distinctNativeSha256Count !== 12 || distinctCurrentPackedSha256Count !== 12 ||
    distinctCandidateTransformedSha256Count !== 12 ||
    members.some((member) => member.proofCacheKey !== member.weightKey ||
      member.proofOperationLabel !== member.operationLabel) ||
    JSON.stringify(actualPlan) !== JSON.stringify(planned) ||
    currentAggregateDispatches !== CURRENT_AGGREGATE_DISPATCHES ||
    candidateAggregateDispatches !== CANDIDATE_AGGREGATE_DISPATCHES ||
    nativeBytes !== CURRENT_PERSISTENT_WEIGHT_BYTES ||
    currentPackedBytes !== CURRENT_PERSISTENT_WEIGHT_BYTES ||
    candidateTransformedBytes !== CANDIDATE_PERSISTENT_WEIGHT_BYTES) {
    throw new Error("OPT-0077 production working-set binding proof failed");
  }
  return Object.freeze({
    schema: "twelve-operation-distinct-native-current-transformed-v1",
    bindingCount: members.length,
    distinctWeightKeyCount,
    distinctNativeBufferCount,
    distinctCurrentPackedBufferCount,
    distinctCandidateTransformedBufferCount,
    distinctNativeSha256Count,
    distinctCurrentPackedSha256Count,
    distinctCandidateTransformedSha256Count,
    c128PerDilationMultiplicity: Object.freeze({ block3: 3, block4: 6,
      total: 9 }),
    currentAggregateDispatches,
    candidateAggregateDispatches,
    nativeBytes,
    currentPackedBytes,
    candidateTransformedBytes,
    mapping: Object.freeze(publicMapping),
    passed: true,
  });
}

function wrapDispatch(
  dispatch: Readonly<{ readonly label: string; readonly kernelId: string;
    encode(pass: GPUComputePassEncoder): void }>,
  probeSpec: Opt0077ProbeSpec,
  stageDispatchCount: number,
  scratchBytes?: Readonly<{
    readonly inputSpectrum: number;
    readonly contractionSpectrum: number;
  }>,
): Encodable {
  return Object.freeze({ label: dispatch.label, kernelId: dispatch.kernelId,
    stageDispatchCount,
    outputRange: Object.freeze({ base: probeSpec.base, count: probeSpec.count }),
    ...(scratchBytes === undefined ? {} : { scratchBytes }),
    encode(pass: GPUComputePassEncoder): void { dispatch.encode(pass); } });
}

function weightCacheKey(spec: Opt0077CaseSpec, operationLabel: string): string {
  if (spec.kind === "production") return `production-${operationLabel}`;
  if (spec.kind === "dc" || spec.kind === "nyquist" ||
    /^cos[1-7]$|^sin[1-7]$/u.test(spec.kind)) return "spectral-diagonal-c128";
  return `${spec.kind}-c128`;
}

async function prepareWeights(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0077CaseSpec,
  plan: AceFp16VaeConv1dPlan,
  operationLabel: string,
  cacheKey: string,
): Promise<PreparedWeights> {
  const nativeWords = buildNativeWeightWords(spec, plan, operationLabel);
  const currentWords = packAceOpt0051VaeK7WeightU16(nativeWords,
    plan.inputChannels, plan.outputChannels);
  const transformedWords = transformAceOpt0077VaeK7WeightU16(nativeWords,
    plan.inputChannels, plan.outputChannels);
  const expectedTransformedWords = plan.inputChannels * plan.outputChannels *
    ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES;
  if (currentWords.length !== nativeWords.length ||
    transformedWords.length !== expectedTransformedWords) {
    throw new Error(`${spec.id} transformed-weight length changed`);
  }
  let transformedNonFinite = 0;
  let transformedZeroCount = 0;
  let transformedSubnormalCount = 0;
  let transformedNormalCount = 0;
  let transformedMaximumAbsoluteValue = 0;
  let transformedMinimumNonzeroAbsoluteValue = Number.POSITIVE_INFINITY;
  for (const bits of transformedWords) {
    const finiteClass = fp16FiniteClass(bits);
    if (finiteClass === "non-finite") transformedNonFinite += 1;
    else if (finiteClass === "zero") transformedZeroCount += 1;
    else if (finiteClass === "subnormal") transformedSubnormalCount += 1;
    else transformedNormalCount += 1;
    const absoluteValue = Math.abs(float16BitsToNumber(bits));
    if (absoluteValue !== 0 && Number.isFinite(absoluteValue)) {
      transformedMaximumAbsoluteValue = Math.max(
        transformedMaximumAbsoluteValue, absoluteValue);
      transformedMinimumNonzeroAbsoluteValue = Math.min(
        transformedMinimumNonzeroAbsoluteValue, absoluteValue);
    }
  }
  if (transformedNonFinite !== 0) {
    throw new Error(`${spec.id} transformed weight contained non-finite FP16`);
  }
  if (spec.kind === "finite-transform-amplification" &&
    transformedMaximumAbsoluteValue < 1_024) {
    throw new Error(`${spec.id} did not exercise amplified finite spectra`);
  }
  const native = u16Buffer(device, tracker, `${operationLabel}-native-k7-weight`,
    nativeWords.byteLength, nativeWords, GPUBufferUsage.STORAGE);
  const currentPacked = u16Buffer(device, tracker,
    `${operationLabel}-current-row-reuse-weight`, currentWords.byteLength,
    currentWords, GPUBufferUsage.STORAGE);
  const candidateTransformed = u16Buffer(device, tracker,
    `${operationLabel}-rfft16-weight`, transformedWords.byteLength,
    transformedWords, GPUBufferUsage.STORAGE);
  const proof = Object.freeze({ cacheKey, operationLabel,
    nativeU16Count: nativeWords.length, currentPackedU16Count: currentWords.length,
    candidateTransformedU16Count: transformedWords.length,
    coordinateCount: ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES,
    payloadRatio: transformedWords.byteLength / nativeWords.byteLength,
    transformedFiniteClassCounts: Object.freeze({
      zero: transformedZeroCount, subnormal: transformedSubnormalCount,
      normal: transformedNormalCount, nonFinite: transformedNonFinite,
    }),
    transformedNonFiniteCount: transformedNonFinite,
    transformedMinimumNonzeroAbsoluteValue:
      Number.isFinite(transformedMinimumNonzeroAbsoluteValue)
        ? transformedMinimumNonzeroAbsoluteValue : 0,
    transformedMaximumAbsoluteValue,
    finiteTransformAmplificationStressPassed:
      spec.kind !== "finite-transform-amplification" ||
      transformedMaximumAbsoluteValue >= 1_024,
    nativeSha256: await sha256U16(nativeWords),
    currentPackedSha256: await sha256U16(currentWords),
    candidateTransformedSha256: await sha256U16(transformedWords),
    transformedBeforeReady: true, transformedInsideTiming: false });
  return Object.freeze({ native, currentPacked, candidateTransformed,
    nativeBytes: nativeWords.byteLength,
    currentPackedBytes: currentWords.byteLength,
    candidateTransformedBytes: transformedWords.byteLength, proof });
}

function buildNativeWeightWords(
  spec: Opt0077CaseSpec,
  plan: AceFp16VaeConv1dPlan,
  operationLabel: string,
): Uint16Array {
  const words = new Uint16Array(plan.weightElements);
  const spectral = spec.kind === "dc" || spec.kind === "nyquist" ||
    /^cos[1-7]$|^sin[1-7]$/u.test(spec.kind);
  if (spec.kind === "finite-transform-amplification") {
    const large = numberToFloat16Bits(4_096);
    for (let outputChannel = 0; outputChannel < plan.outputChannels;
      outputChannel += 1) {
      for (let tap = 0; tap < 7; tap += 1) {
        const index = (outputChannel * 7 + tap) * plan.inputChannels +
          outputChannel;
        words[index] = (tap & 1) === 0 ? large : large | 0x8000;
      }
    }
    return words;
  }
  if (spectral) {
    const half = numberToFloat16Bits(0.5);
    for (let channel = 0; channel < plan.outputChannels; channel += 1) {
      words[(channel * 7 + 3) * plan.inputChannels + channel] = half;
    }
    return words;
  }
  let state = 0x0077_5eed ^ plan.inputChannels ^ stableStringHash32(operationLabel);
  for (let index = 0; index < words.length; index += 1) {
    state = xorshift32(state + index);
    const sign = (state >>> 16) & 0x8000;
    words[index] = sign | 0x1000 | (state & 0x01ff);
  }
  return words;
}

function buildInputWords(
  spec: Opt0077CaseSpec,
  plan: AceFp16VaeConv1dPlan,
): Uint16Array {
  const words = new Uint16Array(plan.inputElements);
  const spectral = spectralFixture(spec.kind);
  if (spectral !== null) {
    for (let row = 0; row < plan.inputFrames; row += 1) {
      const phase = 2 * Math.PI * spectral.bin * row / 16;
      const value = spectral.component === "dc" ? 0.03125
        : spectral.component === "nyquist" ? ((row & 1) === 0 ? 0.03125 : -0.03125)
        : spectral.component === "cos" ? 0.03125 * Math.cos(phase)
        : 0.03125 * Math.sin(phase);
      const bits = numberToFloat16Bits(value);
      words.fill(bits, row * plan.inputChannels, (row + 1) * plan.inputChannels);
    }
    return words;
  }
  const signedZeroSubnormal = new Uint16Array([
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400,
  ]);
  const cancellation = new Uint16Array([
    0x3000, 0xb000, 0x2c00, 0xac00, 0x2800, 0xa800, 0x2400, 0xa400,
  ]);
  const finite = new Uint16Array([
    0x3800, 0xb800, 0x3600, 0xb600, 0x3400, 0xb400, 0x3000, 0xb000,
  ]);
  const pattern = spec.kind === "signed-zero-subnormal" ? signedZeroSubnormal
    : spec.kind === "alternating-cancellation" ? cancellation
    : spec.kind === "finite-transform-amplification" ? finite
    : new Uint16Array([
        0x2400, 0xa400, 0x2800, 0xa800, 0x2c00, 0xac00, 0x3000, 0xb000,
      ]);
  for (let index = 0; index < words.length; index += 1) {
    words[index] = pattern[(index * 13 + Math.floor(index / plan.inputChannels)) %
      pattern.length]!;
  }
  return words;
}

function spectralFixture(kind: Opt0077FixtureKind): Readonly<{
  component: "dc" | "nyquist" | "cos" | "sin";
  bin: number;
}> | null {
  if (kind === "dc") return Object.freeze({ component: "dc", bin: 0 });
  if (kind === "nyquist") {
    return Object.freeze({ component: "nyquist", bin: 8 });
  }
  const match = /^(cos|sin)([1-7])$/u.exec(kind);
  if (match === null) return null;
  return Object.freeze({ component: match[1] as "cos" | "sin",
    bin: Number(match[2]) });
}

function buildBiasWords(
  spec: Opt0077CaseSpec,
  plan: AceFp16VaeConv1dPlan,
): Uint16Array {
  const words = new Uint16Array(plan.outputChannels);
  const spectral = spectralFixture(spec.kind) !== null;
  if (spectral) return words;
  const pattern = new Uint16Array([
    0x0000, 0x8000, 0x1800, 0x9800, 0x1c00, 0x9c00, 0x2000, 0xa000,
  ]);
  for (let index = 0; index < words.length; index += 1) {
    words[index] = pattern[index % pattern.length]!;
  }
  return words;
}

async function verifyCase(
  device: GPUDevice,
  tracker: BufferTracker,
  prepared: PreparedCase,
  aggregate: NumericalAccumulator,
  categoryAggregate: NumericalAccumulator,
  candidateCurrentAggregate: NumericalAccumulator,
  categoryCandidateCurrentAggregate: NumericalAccumulator,
): Promise<Readonly<Record<string, unknown>>> {
  const probes: Readonly<Record<string, unknown>>[] = [];
  const hashLines: Record<"current" | "candidate" | "scalarOracle" |
    "candidateRepeat", string[]> = {
      current: [], candidate: [], scalarOracle: [], candidateRepeat: [],
    };
  let candidateScalarComparedU16Count = 0;
  let candidateRepeatComparedU16Count = 0;
  for (const probeSpec of prepared.spec.probes) {
    const dispatches = prepared.correctnessDispatches[probeSpec.id];
    const current = await executeCorrectness(device, tracker, prepared,
      dispatches.current, probeSpec, "current");
    const candidate = await executeCorrectness(device, tracker, prepared,
      dispatches.candidate, probeSpec, "candidate");
    const oracle = await executeCorrectness(device, tracker, prepared,
      dispatches["scalar-oracle"], probeSpec, "scalar-oracle");
    const candidateRepeat = await executeCorrectness(device, tracker, prepared,
      dispatches.candidate, probeSpec, "candidate");
    const snapshots = [current, candidate, oracle, candidateRepeat];
    const candidateRepeatComparison = compareRawSelected(
      candidate.words,
      candidateRepeat.words,
      probeSpec,
    );
    const candidateScalar = compareNumericalSelected(
      oracle.words,
      candidate.words,
      prepared.spec,
      probeSpec,
    );
    const currentScalar = compareNumericalSelected(
      oracle.words,
      current.words,
      prepared.spec,
      probeSpec,
    );
    const candidateCurrent = compareNumericalSelected(
      current.words,
      candidate.words,
      prepared.spec,
      probeSpec,
    );
    mergeNumerical(aggregate, candidateScalar);
    mergeNumerical(categoryAggregate, candidateScalar);
    mergeNumerical(candidateCurrentAggregate, candidateCurrent);
    mergeNumerical(categoryCandidateCurrentAggregate, candidateCurrent);
    const candidateScalarMetrics = summarizeNumerical(candidateScalar);
    const currentScalarMetrics = summarizeNumerical(currentScalar);
    const candidateCurrentMetrics = summarizeNumerical(candidateCurrent);
    const numericalEnvelopePassed = numericalPassed(candidateScalarMetrics);
    const complete = snapshots.every(snapshotPassed);
    const scratchPassed = candidate.scratch?.["passed"] === true &&
      candidateRepeat.scratch?.["passed"] === true;
    const passed = complete && scratchPassed &&
      candidateRepeatComparison.differingU16Count === 0 &&
      numericalEnvelopePassed;
    candidateScalarComparedU16Count += probeSpec.count;
    candidateRepeatComparedU16Count += probeSpec.count;
    hashLines.current.push(`${probeSpec.id}:${current.selectedSha256}`);
    hashLines.candidate.push(`${probeSpec.id}:${candidate.selectedSha256}`);
    hashLines.scalarOracle.push(`${probeSpec.id}:${oracle.selectedSha256}`);
    hashLines.candidateRepeat.push(
      `${probeSpec.id}:${candidateRepeat.selectedSha256}`,
    );
    probes.push(Object.freeze({ id: probeSpec.id, base: probeSpec.base,
      count: probeSpec.count, firstOutputRow: probeSpec.firstOutputRow,
      outputRowCount: probeSpec.outputRowCount,
      executionOrder: Object.freeze([
        "current", "candidate", "scalar-oracle", "candidate",
      ]),
      dispatchIdentity: Object.freeze({
        currentSelector: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
        candidateKernel: ACE_OPT_0077_VAE_K7_RFFT16_KERNEL_ID,
        scalarOracleKernel: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
        currentStageDispatches: dispatches.current.stageDispatchCount,
        candidateStageDispatches: dispatches.candidate.stageDispatchCount,
        scalarOracleStageDispatches:
          dispatches["scalar-oracle"].stageDispatchCount,
      }),
      hashes: Object.freeze({ current: current.selectedSha256,
        candidate: candidate.selectedSha256,
        scalarOracle: oracle.selectedSha256,
        candidateRepeat: candidateRepeat.selectedSha256 }),
      fullBindingHashes: Object.freeze({ current: current.sha256,
        candidate: candidate.sha256, scalarOracle: oracle.sha256,
        candidateRepeat: candidateRepeat.sha256 }),
      candidateRepeat: candidateRepeatComparison,
      candidateVersusScalarOracle: candidateScalarMetrics,
      currentVersusScalarOracle: currentScalarMetrics,
      candidateVersusCurrent: candidateCurrentMetrics,
      numericalEnvelopePassed,
      snapshots: Object.freeze(snapshots.map(snapshotReceipt)),
      scratchFiniteCompleteAndGuarded: scratchPassed,
      passed }));
  }
  const passed = probes.every((entry) => entry["passed"] === true);
  const resultHashes = Object.freeze({
    current: await sha256Text(hashLines.current.join("\n")),
    candidate: await sha256Text(hashLines.candidate.join("\n")),
    scalarOracle: await sha256Text(hashLines.scalarOracle.join("\n")),
    candidateRepeat: await sha256Text(hashLines.candidateRepeat.join("\n")),
  });
  return Object.freeze({ id: prepared.spec.id, kind: prepared.spec.kind,
    tier: prepared.spec.tier, dilation: prepared.spec.dilation,
    operationLabel: prepared.spec.operationLabel,
    representedOperationLabels: prepared.spec.representedOperationLabels,
    shape: prepared.spec.shape, probeCount: probes.length,
    candidateScalarComparedU16Count, candidateRepeatComparedU16Count,
    weightProof: prepared.weightProof, resultHashes,
    probes: Object.freeze(probes), passed });
}

async function executeCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  prepared: PreparedCase,
  dispatch: Encodable,
  probeSpec: Opt0077ProbeSpec,
  arm: ExecutionArm,
): Promise<OutputSnapshot> {
  const outputReadback = tracker.create(device, {
    label: `${prepared.spec.id}-${probeSpec.id}-${arm}-output-readback`,
    size: prepared.output.totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const scratchReadbacks = arm === "candidate" ? Object.freeze({
    inputSpectrum: tracker.create(device, {
      label: `${prepared.spec.id}-${probeSpec.id}-input-spectrum-readback`,
      size: prepared.inputSpectrum.totalBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    contractionSpectrum: tracker.create(device, {
      label: `${prepared.spec.id}-${probeSpec.id}-contraction-spectrum-readback`,
      size: prepared.contractionSpectrum.totalBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
  }) : undefined;
  const encoder = device.createCommandEncoder({
    label: `${prepared.spec.id}-${probeSpec.id}-${arm}-correctness`,
  });
  encoder.copyBufferToBuffer(prepared.output.prefill, 0,
    prepared.output.buffer, 0, prepared.output.totalBytes);
  if (arm === "candidate") {
    encoder.copyBufferToBuffer(prepared.inputSpectrum.prefill, 0,
      prepared.inputSpectrum.buffer, 0, prepared.inputSpectrum.totalBytes);
    encoder.copyBufferToBuffer(prepared.contractionSpectrum.prefill, 0,
      prepared.contractionSpectrum.buffer, 0,
      prepared.contractionSpectrum.totalBytes);
  }
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  encoder.copyBufferToBuffer(prepared.output.buffer, 0, outputReadback, 0,
    prepared.output.totalBytes);
  if (scratchReadbacks !== undefined) {
    encoder.copyBufferToBuffer(prepared.inputSpectrum.buffer, 0,
      scratchReadbacks.inputSpectrum, 0, prepared.inputSpectrum.totalBytes);
    encoder.copyBufferToBuffer(prepared.contractionSpectrum.buffer, 0,
      scratchReadbacks.contractionSpectrum, 0,
      prepared.contractionSpectrum.totalBytes);
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const outputBytes = await mapReadback(tracker, outputReadback);
  let scratch: Readonly<Record<string, unknown>> | null = null;
  if (scratchReadbacks !== undefined) {
    const inputBytes = await mapReadback(tracker,
      scratchReadbacks.inputSpectrum);
    const contractionBytes = await mapReadback(tracker,
      scratchReadbacks.contractionSpectrum);
    const inputCheck = inspectScratch(inputBytes, prepared.inputSpectrum,
      dispatch.scratchBytes!.inputSpectrum);
    const contractionCheck = inspectScratch(contractionBytes,
      prepared.contractionSpectrum,
      dispatch.scratchBytes!.contractionSpectrum);
    scratch = Object.freeze({ inputSpectrum: inputCheck,
      contractionSpectrum: contractionCheck,
      passed: inputCheck.passed && contractionCheck.passed });
  }
  const inspected = inspectOutput(outputBytes, prepared.output, prepared.plan,
    probeSpec, scratch);
  return Object.freeze({ ...inspected,
    sha256: await sha256U16(inspected.words),
    selectedSha256: await sha256U16(inspected.words.slice(
      probeSpec.base,
      probeSpec.base + probeSpec.count,
    )) });
}

async function mapReadback(
  tracker: BufferTracker,
  buffer: GPUBuffer,
): Promise<Uint8Array<ArrayBuffer>> {
  await buffer.mapAsync(GPUMapMode.READ);
  tracker.mapped();
  const bytes = new Uint8Array(buffer.getMappedRange()).slice();
  tracker.unmap(buffer);
  tracker.destroy(buffer);
  return bytes;
}

function inspectOutput(
  bytes: Uint8Array<ArrayBuffer>,
  guarded: GuardedBuffer,
  plan: AceFp16VaeConv1dPlan,
  probeSpec: Opt0077ProbeSpec,
  scratch: Readonly<Record<string, unknown>> | null,
): OutputSnapshot {
  const guardWords = STORAGE_GUARD_BYTES / 4;
  const allU32 = new Uint32Array(bytes.buffer);
  const words = new Uint16Array(bytes.buffer.slice(
    STORAGE_GUARD_BYTES,
    STORAGE_GUARD_BYTES + guarded.logicalBytes,
  ));
  let nonFiniteCount = 0;
  let qNaNPrefillCount = 0;
  let outOfRangeWriteCount = 0;
  const begin = probeSpec.base;
  const end = begin + probeSpec.count;
  for (let index = 0; index < plan.outputElements; index += 1) {
    const word = words[index]!;
    if (index >= begin && index < end) {
      if ((word & 0x7c00) === 0x7c00) nonFiniteCount += 1;
      if (word === OUTPUT_PREFILL_QNAN_F16) qNaNPrefillCount += 1;
    } else if (word !== OUTPUT_PREFILL_QNAN_F16) {
      outOfRangeWriteCount += 1;
    }
  }
  return Object.freeze({ words,
    sha256: "pending", selectedSha256: "pending",
    nonFiniteCount, qNaNPrefillCount,
    prefixGuardIntact: everyU32(allU32.subarray(0, guardWords),
      STORAGE_GUARD_U32),
    suffixGuardIntact: everyU32(
      allU32.subarray(allU32.length - guardWords), STORAGE_GUARD_U32),
    adjacentBeforeIntact: begin === 0 ||
      words[begin - 1] === OUTPUT_PREFILL_QNAN_F16,
    adjacentAfterIntact: end === plan.outputElements ||
      words[end] === OUTPUT_PREFILL_QNAN_F16,
    outOfRangeWriteCount,
    selectedTailWritten: words[end - 1] !== OUTPUT_PREFILL_QNAN_F16,
    scratch });
}

function inspectScratch(
  bytes: Uint8Array<ArrayBuffer>,
  guarded: GuardedBuffer,
  usedBytes: number,
): Readonly<{ passed: boolean; nonFiniteCount: number;
  prefillCount: number; prefixGuardIntact: boolean;
  suffixGuardIntact: boolean; unusedTailIntact: boolean }> {
  const guardWords = STORAGE_GUARD_BYTES / 4;
  const allU32 = new Uint32Array(bytes.buffer);
  let nonFiniteCount = 0;
  let prefillCount = 0;
  let unusedTailIntact = true;
  if (guarded.elementStorage === "f16") {
    const words = new Uint16Array(bytes.buffer, STORAGE_GUARD_BYTES,
      guarded.logicalBytes / 2);
    for (let index = 0; index < words.length; index += 1) {
      if (index * 2 < usedBytes) {
        if ((words[index]! & 0x7c00) === 0x7c00) nonFiniteCount += 1;
        if (words[index] === OUTPUT_PREFILL_QNAN_F16) prefillCount += 1;
      } else if (words[index] !== OUTPUT_PREFILL_QNAN_F16) {
        unusedTailIntact = false;
      }
    }
  } else {
    const words = new Uint32Array(bytes.buffer, STORAGE_GUARD_BYTES,
      guarded.logicalBytes / 4);
    for (let index = 0; index < words.length; index += 1) {
      if (index * 4 < usedBytes) {
        if ((words[index]! & 0x7f80_0000) === 0x7f80_0000) nonFiniteCount += 1;
        if (words[index] === SCRATCH_PREFILL_QNAN_F32) prefillCount += 1;
      } else if (words[index] !== SCRATCH_PREFILL_QNAN_F32) {
        unusedTailIntact = false;
      }
    }
  }
  const prefixGuardIntact = everyU32(allU32.subarray(0, guardWords),
    STORAGE_GUARD_U32);
  const suffixGuardIntact = everyU32(
    allU32.subarray(allU32.length - guardWords), STORAGE_GUARD_U32);
  return Object.freeze({ passed: nonFiniteCount === 0 && prefillCount === 0 &&
    prefixGuardIntact && suffixGuardIntact && unusedTailIntact,
    nonFiniteCount, prefillCount, prefixGuardIntact, suffixGuardIntact,
    unusedTailIntact });
}

function snapshotPassed(snapshot: OutputSnapshot): boolean {
  return snapshot.nonFiniteCount === 0 && snapshot.qNaNPrefillCount === 0 &&
    snapshot.prefixGuardIntact && snapshot.suffixGuardIntact &&
    snapshot.adjacentBeforeIntact && snapshot.adjacentAfterIntact &&
    snapshot.outOfRangeWriteCount === 0 && snapshot.selectedTailWritten;
}

function snapshotReceipt(snapshot: OutputSnapshot): Readonly<Record<string, unknown>> {
  return Object.freeze({ sha256: snapshot.sha256,
    selectedSha256: snapshot.selectedSha256,
    nonFiniteCount: snapshot.nonFiniteCount,
    qNaNPrefillCount: snapshot.qNaNPrefillCount,
    prefixGuardIntact: snapshot.prefixGuardIntact,
    suffixGuardIntact: snapshot.suffixGuardIntact,
    adjacentBeforeIntact: snapshot.adjacentBeforeIntact,
    adjacentAfterIntact: snapshot.adjacentAfterIntact,
    outOfRangeWriteCount: snapshot.outOfRangeWriteCount,
    selectedTailWritten: snapshot.selectedTailWritten,
    scratch: snapshot.scratch, passed: snapshotPassed(snapshot) });
}

function compareRawSelected(
  left: Uint16Array,
  right: Uint16Array,
  probeSpec: Opt0077ProbeSpec,
): Readonly<{ comparedU16Count: number; differingU16Count: number }> {
  let differingU16Count = 0;
  const end = probeSpec.base + probeSpec.count;
  for (let index = probeSpec.base; index < end; index += 1) {
    if (left[index] !== right[index]) differingU16Count += 1;
  }
  return Object.freeze({ comparedU16Count: probeSpec.count,
    differingU16Count });
}

function createNumericalAccumulator(): NumericalAccumulator {
  return { count: 0, rawU16MismatchCount: 0, signedZeroDifferenceCount: 0,
    finiteClassChangeCount: 0,
    sumError: 0, sumAbsoluteError: 0, sumSquaredError: 0,
    sumOracleSquared: 0, sumOracle: 0, sumCandidate: 0,
    sumCandidateSquared: 0, sumProduct: 0, maximumAbsoluteError: 0,
    oraclePeak: 0, oracleMinimum: Number.POSITIVE_INFINITY,
    oracleMaximum: Number.NEGATIVE_INFINITY,
    candidateMinimum: Number.POSITIVE_INFINITY,
    candidateMaximum: Number.NEGATIVE_INFINITY, ulpDistribution: new Map(),
    finiteClassTransitions: new Map(),
    firstDifference: null, worstDifference: null };
}

function compareNumericalSelected(
  oracleWords: Uint16Array,
  candidateWords: Uint16Array,
  spec: Opt0077CaseSpec,
  probeSpec: Opt0077ProbeSpec,
): NumericalAccumulator {
  const result = createNumericalAccumulator();
  const end = probeSpec.base + probeSpec.count;
  for (let index = probeSpec.base; index < end; index += 1) {
    const oracleBits = oracleWords[index]!;
    const candidateBits = candidateWords[index]!;
    const oracle = float16BitsToNumber(oracleBits);
    const candidate = float16BitsToNumber(candidateBits);
    const error = candidate - oracle;
    const absoluteError = Math.abs(error);
    result.count += 1;
    if (oracleBits !== candidateBits) {
      result.rawU16MismatchCount += 1;
      if (result.firstDifference === null) {
        result.firstDifference = Object.freeze({ caseId: spec.id,
          probeId: probeSpec.id, outputIndex: index,
          outputRow: Math.floor(index / spec.shape.outputChannels),
          outputChannel: index % spec.shape.outputChannels,
          oracleBits, candidateBits, oracle, candidate, absoluteError });
      }
    }
    if ((oracleBits & 0x7fff) === 0 && (candidateBits & 0x7fff) === 0 &&
      oracleBits !== candidateBits) result.signedZeroDifferenceCount += 1;
    const oracleClass = fp16FiniteClass(oracleBits);
    const candidateClass = fp16FiniteClass(candidateBits);
    if (oracleClass !== candidateClass) {
      const transition = `${oracleClass}->${candidateClass}`;
      result.finiteClassTransitions.set(transition,
        (result.finiteClassTransitions.get(transition) ?? 0) + 1);
      if (oracleClass !== "non-finite" && candidateClass !== "non-finite") {
        result.finiteClassChangeCount += 1;
      }
    }
    result.sumError += error;
    result.sumAbsoluteError += absoluteError;
    result.sumSquaredError += error * error;
    result.sumOracleSquared += oracle * oracle;
    result.sumOracle += oracle;
    result.sumCandidate += candidate;
    result.sumCandidateSquared += candidate * candidate;
    result.sumProduct += oracle * candidate;
    result.oraclePeak = Math.max(result.oraclePeak, Math.abs(oracle));
    result.oracleMinimum = Math.min(result.oracleMinimum, oracle);
    result.oracleMaximum = Math.max(result.oracleMaximum, oracle);
    result.candidateMinimum = Math.min(result.candidateMinimum, candidate);
    result.candidateMaximum = Math.max(result.candidateMaximum, candidate);
    const ulp = Math.abs(orderedF16(oracleBits) - orderedF16(candidateBits));
    result.ulpDistribution.set(ulp, (result.ulpDistribution.get(ulp) ?? 0) + 1);
    if (absoluteError > result.maximumAbsoluteError) {
      result.maximumAbsoluteError = absoluteError;
      result.worstDifference = Object.freeze({ caseId: spec.id,
        probeId: probeSpec.id, outputIndex: index,
        outputRow: Math.floor(index / spec.shape.outputChannels),
        outputChannel: index % spec.shape.outputChannels,
        oracleBits, candidateBits, oracle, candidate, absoluteError });
    }
  }
  return result;
}

function mergeNumerical(
  target: NumericalAccumulator,
  source: NumericalAccumulator,
): void {
  target.count += source.count;
  target.rawU16MismatchCount += source.rawU16MismatchCount;
  target.signedZeroDifferenceCount += source.signedZeroDifferenceCount;
  target.finiteClassChangeCount += source.finiteClassChangeCount;
  target.sumError += source.sumError;
  target.sumAbsoluteError += source.sumAbsoluteError;
  target.sumSquaredError += source.sumSquaredError;
  target.sumOracleSquared += source.sumOracleSquared;
  target.sumOracle += source.sumOracle;
  target.sumCandidate += source.sumCandidate;
  target.sumCandidateSquared += source.sumCandidateSquared;
  target.sumProduct += source.sumProduct;
  target.maximumAbsoluteError = Math.max(target.maximumAbsoluteError,
    source.maximumAbsoluteError);
  target.oraclePeak = Math.max(target.oraclePeak, source.oraclePeak);
  target.oracleMinimum = Math.min(target.oracleMinimum, source.oracleMinimum);
  target.oracleMaximum = Math.max(target.oracleMaximum, source.oracleMaximum);
  target.candidateMinimum = Math.min(target.candidateMinimum,
    source.candidateMinimum);
  target.candidateMaximum = Math.max(target.candidateMaximum,
    source.candidateMaximum);
  for (const [ulp, count] of source.ulpDistribution) {
    target.ulpDistribution.set(ulp,
      (target.ulpDistribution.get(ulp) ?? 0) + count);
  }
  for (const [transition, count] of source.finiteClassTransitions) {
    target.finiteClassTransitions.set(transition,
      (target.finiteClassTransitions.get(transition) ?? 0) + count);
  }
  if (target.firstDifference === null) target.firstDifference = source.firstDifference;
  if (source.maximumAbsoluteError >= target.maximumAbsoluteError &&
    source.worstDifference !== null) target.worstDifference = source.worstDifference;
}

function summarizeNumerical(
  accumulator: NumericalAccumulator,
): Readonly<Record<string, unknown>> {
  if (accumulator.count <= 0) throw new Error("OPT-0077 empty numerical result");
  const rmsError = Math.sqrt(accumulator.sumSquaredError / accumulator.count);
  const oracleRms = Math.sqrt(accumulator.sumOracleSquared / accumulator.count);
  const nrmse = rmsError / Math.max(oracleRms, 1e-12);
  const snr = accumulator.sumSquaredError === 0
    ? Number.POSITIVE_INFINITY
    : 10 * Math.log10(accumulator.sumOracleSquared /
      accumulator.sumSquaredError);
  const covariance = accumulator.sumProduct -
    accumulator.sumOracle * accumulator.sumCandidate / accumulator.count;
  const oracleVariance = accumulator.sumOracleSquared -
    accumulator.sumOracle ** 2 / accumulator.count;
  const candidateVariance = accumulator.sumCandidateSquared -
    accumulator.sumCandidate ** 2 / accumulator.count;
  const pearson = oracleVariance <= 0 || candidateVariance <= 0
    ? (accumulator.sumSquaredError === 0 ? 1 : 0)
    : covariance / Math.sqrt(oracleVariance * candidateVariance);
  const distribution = Object.freeze(Object.fromEntries(
    [...accumulator.ulpDistribution.entries()]
      .sort(([left], [right]) => left - right)
      .map(([ulp, count]) => [String(ulp), count]),
  ));
  const finiteClassTransitions = Object.freeze(Object.fromEntries(
    [...accumulator.finiteClassTransitions.entries()].sort(([left], [right]) =>
      left.localeCompare(right)),
  ));
  return Object.freeze({ comparedValueCount: accumulator.count,
    rawU16MismatchCount: accumulator.rawU16MismatchCount,
    signedZeroDifferenceCount: accumulator.signedZeroDifferenceCount,
    finiteClassChangeCount: accumulator.finiteClassChangeCount,
    finiteClassTransitionDistribution: finiteClassTransitions,
    finiteClassTransitionDistributionCount:
      [...accumulator.finiteClassTransitions.values()].reduce(
        (sum, count) => sum + count, 0),
    meanError: accumulator.sumError / accumulator.count,
    meanAbsoluteError: accumulator.sumAbsoluteError / accumulator.count,
    rmsError, oracleRms, nrmse,
    snrDb: Number.isFinite(snr) ? snr : "Infinity", pearson,
    maximumAbsoluteError: accumulator.maximumAbsoluteError,
    relativeMaximumAbsoluteError: accumulator.maximumAbsoluteError /
      Math.max(accumulator.oraclePeak, 1e-6),
    oraclePeak: accumulator.oraclePeak,
    oracleRange: Object.freeze({ minimum: accumulator.oracleMinimum,
      maximum: accumulator.oracleMaximum }),
    candidateRange: Object.freeze({ minimum: accumulator.candidateMinimum,
      maximum: accumulator.candidateMaximum }),
    fp16UlpDistribution: distribution,
    fp16UlpDistributionCount: [...accumulator.ulpDistribution.values()]
      .reduce((sum, count) => sum + count, 0),
    firstDifference: accumulator.firstDifference,
    worstDifference: accumulator.worstDifference });
}

function fp16FiniteClass(bits: number):
"zero" | "subnormal" | "normal" | "non-finite" {
  const magnitude = bits & 0x7fff;
  if (magnitude === 0) return "zero";
  const exponent = magnitude & 0x7c00;
  if (exponent === 0) return "subnormal";
  if (exponent === 0x7c00) return "non-finite";
  return "normal";
}

function numericalPassed(metrics: Readonly<Record<string, unknown>>): boolean {
  const snr = metrics["snrDb"] === "Infinity"
    ? Number.POSITIVE_INFINITY : Number(metrics["snrDb"]);
  return Number(metrics["nrmse"]) <= NRMSE_MAXIMUM &&
    snr >= SNR_MINIMUM_DB &&
    Number(metrics["pearson"]) >= PEARSON_MINIMUM &&
    Number(metrics["relativeMaximumAbsoluteError"]) <=
      RELATIVE_MAXIMUM_ABSOLUTE_ERROR_MAXIMUM;
}

function orderedF16(bits: number): number {
  return (bits & 0x8000) !== 0 ? 0x8000 - (bits & 0x7fff) : 0x8000 + bits;
}

async function runTiming(
  prepared: PreparedHarness,
  thermalLaunch: Opt0077ThermalLaunchGate,
  update: (message: string) => void,
): Promise<Readonly<Record<string, unknown>>> {
  const aggregate: Record<Opt0077Arm, Opt0077TimestampSample[]> = {
    current: [], candidate: [],
  };
  const stratumSamples = new Map<string, Record<Opt0077Arm, number[]>>();
  for (const item of prepared.productionCases) {
    stratumSamples.set(item.spec.id, { current: [], candidate: [] });
  }
  const rawSamples: Readonly<Record<string, unknown>>[] = [];
  const measurementStartedAtEpochMilliseconds = Date.now();
  for (const [roundIndex, order] of TIMING_ROUNDS.entries()) {
    for (const [armPosition, arm] of order.entries()) {
      for (const item of prepared.productionCases) {
        update(`round ${roundIndex + 1}/6 ${arm}: ${item.spec.id}`);
        const sample = await timeEncodings(prepared,
          `${item.spec.id}-${arm}`, 1,
          item.timingDispatches![arm]);
        stratumSamples.get(item.spec.id)![arm].push(sample.gpuMilliseconds);
        rawSamples.push(Object.freeze({ kind: "stratum", roundIndex,
          armPosition, arm, stratum: item.spec.id, ...sample }));
      }
      update(`round ${roundIndex + 1}/6 ${arm}: weighted aggregate`);
      const sample = await timeAggregate(prepared, arm);
      aggregate[arm].push(sample);
      rawSamples.push(Object.freeze({ kind: "weighted-aggregate", roundIndex,
        armPosition, arm, ...sample }));
    }
    await yieldToBrowser();
  }
  await prepared.device.queue.onSubmittedWorkDone();
  await settlePostDrainEvents();
  const measurementCompletedAtEpochMilliseconds = Date.now();
  if (prepared.uncapturedErrors.length !== 0 ||
    prepared.deviceLosses.length !== 0) {
    throw new Error("OPT-0077 observed a timing GPU error or device loss");
  }
  const timing = summarizeOpt0077Timing(Object.freeze({
    current: Object.freeze(aggregate.current),
    candidate: Object.freeze(aggregate.candidate),
    strata: Object.freeze(prepared.productionCases.map((item) => {
      const samples = stratumSamples.get(item.spec.id)!;
      return Object.freeze({ id: item.spec.id,
        currentGpuMilliseconds: Object.freeze(samples.current),
        candidateGpuMilliseconds: Object.freeze(samples.candidate) });
    })),
  }));
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanup = await prepared.cleanup();
  const cleanupCompletedAtEpochMilliseconds = Number(
    cleanup["cleanupCompletedAtEpochMilliseconds"],
  );
  const cleanupPassed = cleanup["zeroLiveBuffers"] === true &&
    cleanup["zeroLiveBytes"] === true &&
    cleanup["createdEqualsDestroyed"] === true &&
    cleanup["mapsBalanced"] === true &&
    cleanup["postDestroyRejected"] === true;
  const inPagePassed = timing["passed"] === true &&
    prepared.correctness["passed"] === true && cleanupPassed;
  return Object.freeze({ schema: RECEIPT_SCHEMA, experiment: EXPERIMENT_ID,
    status: "awaiting-external-thermal-completion", passed: false,
    inPagePassed, thermalLaunch, cleanupCompletedAtEpochMilliseconds,
    identity: prepared.identity, correctness: prepared.correctness,
    resources: prepared.resourceContract,
    warmup: prepared.warmup,
    protocol: Object.freeze({ correctnessAndWarmupBeforeReady: true,
      rounds: TIMING_ROUNDS.length, balancedOrders: TIMING_ROUNDS,
      completeOutputFramesPerStratum: TIMING_FRAMES,
      perStratumInstancesPerSample: 1,
      perStratumCurrentDispatches: 1,
      perStratumCandidateDispatches: 3,
      multiplicities: Object.freeze({ c1024: 2, c512: 3, c128: 9 }),
      c128OperationMultiplicitySplit: Object.freeze({ block3: 3, block4: 6 }),
      productionOperationWeightBindingCount: 12,
      distinctNativeCurrentAndTransformedBuffersPerOperation: true,
      productionWorkingSetProofIncluded: true,
      currentAggregateDispatches: CURRENT_AGGREGATE_DISPATCHES,
      candidateAggregateDispatches: CANDIDATE_AGGREGATE_DISPATCHES,
      onePassOneCommandBufferOneSubmitOneDrainPerSample: true,
      outputReadbackInsideTiming: false, transformedWeightInsideTiming: false,
      unchangedTimingRetryPerformed: false }),
    timing: Object.freeze({ ...timing, measurementStartedAtEpochMilliseconds,
      measurementCompletedAtEpochMilliseconds,
      rawSamples: Object.freeze(rawSamples) }),
    uncapturedGpuErrors: prepared.uncapturedErrors,
    deviceLosses: prepared.deviceLosses, memoryBeforeCleanup, cleanup });
}

async function executeAggregate(
  device: GPUDevice,
  cases: readonly PreparedCase[],
  arm: Opt0077Arm,
): Promise<void> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  encodeAggregate(pass, cases, arm);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

function encodeAggregate(
  pass: GPUComputePassEncoder,
  cases: readonly PreparedCase[],
  arm: Opt0077Arm,
): void {
  for (const item of cases) {
    if (item.aggregateTimingMembers === undefined) {
      throw new Error(`${item.spec.id} aggregate timing members were not prepared`);
    }
    for (const member of item.aggregateTimingMembers) {
      const dispatch = member.dispatches[arm];
      for (let repeat = 0; repeat < member.multiplicity; repeat += 1) {
        dispatch.encode(pass);
      }
    }
  }
}

async function timeAggregate(
  prepared: PreparedHarness,
  arm: Opt0077Arm,
): Promise<Opt0077TimestampSample> {
  return timePass(prepared, `aggregate-${arm}`,
    arm === "current" ? CURRENT_AGGREGATE_DISPATCHES :
      CANDIDATE_AGGREGATE_DISPATCHES,
    (pass) => encodeAggregate(pass, prepared.productionCases, arm));
}

async function timeEncodings(
  prepared: PreparedHarness,
  label: string,
  multiplicity: number,
  dispatch: Encodable,
): Promise<Opt0077TimestampSample> {
  return timePass(prepared, label, multiplicity * dispatch.stageDispatchCount,
    (pass) => {
      for (let repeat = 0; repeat < multiplicity; repeat += 1) {
        dispatch.encode(pass);
      }
    });
}

async function timePass(
  prepared: PreparedHarness,
  label: string,
  dispatchCount: number,
  encode: (pass: GPUComputePassEncoder) => void,
): Promise<Opt0077TimestampSample> {
  if (prepared.queryReadback.mapState !== "unmapped") {
    throw new Error("OPT-0077 timestamp readback remained mapped");
  }
  const encoder = prepared.device.createCommandEncoder({ label });
  const pass = encoder.beginComputePass({
    timestampWrites: { querySet: prepared.querySet,
      beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
  });
  encode(pass);
  pass.end();
  encoder.resolveQuerySet(prepared.querySet, 0, 2, prepared.queryResolve, 0);
  encoder.copyBufferToBuffer(prepared.queryResolve, 0,
    prepared.queryReadback, 0, TIMESTAMP_QUERY_BYTES);
  const command = encoder.finish();
  const submitAtPerformanceMilliseconds = performance.now();
  prepared.device.queue.submit([command]);
  await prepared.device.queue.onSubmittedWorkDone();
  const fenceAtPerformanceMilliseconds = performance.now();
  await prepared.queryReadback.mapAsync(GPUMapMode.READ);
  prepared.tracker.mapped();
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
  if (end <= begin) throw new Error("OPT-0077 timestamp interval was empty");
  const gpuElapsedNanoseconds = Number(end - begin);
  const gpuMilliseconds = gpuElapsedNanoseconds / 1_000_000;
  const wallMilliseconds = fenceAtPerformanceMilliseconds -
    submitAtPerformanceMilliseconds;
  return Object.freeze({ gpuMilliseconds, wallMilliseconds,
    gpuElapsedNanoseconds, dispatchCount, commandBufferCount: 1,
    queueDrainCount: 1,
    submitAtEpochMilliseconds: performance.timeOrigin +
      submitAtPerformanceMilliseconds,
    fenceAtEpochMilliseconds: performance.timeOrigin +
      fenceAtPerformanceMilliseconds,
    timestampBeginNanoseconds: begin.toString(),
    timestampEndNanoseconds: end.toString() });
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  private readonly sizes = new Map<GPUBuffer, number>();
  private created = 0;
  private destroyed = 0;
  private liveBytes = 0;
  private maximumLiveBytes = 0;
  private maps = 0;
  private unmaps = 0;
  private activeMaps = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    const bytes = Number(descriptor.size);
    this.live.add(buffer);
    this.sizes.set(buffer, bytes);
    this.created += 1;
    this.liveBytes += bytes;
    this.maximumLiveBytes = Math.max(this.maximumLiveBytes, this.liveBytes);
    if (descriptor.mappedAtCreation === true) this.mapped();
    return buffer;
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    if (buffer.mapState === "mapped") {
      this.unmap(buffer);
    }
    buffer.destroy();
    this.destroyed += 1;
    this.liveBytes -= this.sizes.get(buffer) ?? 0;
    this.sizes.delete(buffer);
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  mapped(): void { this.maps += 1; this.activeMaps += 1; }
  unmapped(): void { this.unmaps += 1; this.activeMaps -= 1; }
  unmap(buffer: GPUBuffer): void {
    if (buffer.mapState !== "mapped") {
      throw new Error("OPT-0077 attempted to unmap an unmapped buffer");
    }
    buffer.unmap();
    this.unmapped();
  }

  receipt(): Readonly<Record<string, number>> {
    return Object.freeze({ createdBufferCount: this.created,
      destroyedBufferCount: this.destroyed, liveBufferCount: this.live.size,
      liveBytes: this.liveBytes, maximumLiveBytes: this.maximumLiveBytes,
      mapCount: this.maps, unmapCount: this.unmaps,
      activeMapCount: this.activeMaps });
  }
}

function u16Buffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  bytes: number,
  payload: Uint16Array,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  if (bytes % 4 !== 0 || payload.byteLength > bytes) {
    throw new RangeError(`${label} payload does not fit its aligned buffer`);
  }
  const buffer = tracker.create(device, {
    label, size: bytes, usage, mappedAtCreation: true,
  });
  new Uint16Array(buffer.getMappedRange()).set(payload);
  tracker.unmap(buffer);
  return buffer;
}

function guardedBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  logicalBytes: number,
  elementStorage: "f16" | "f32",
): GuardedBuffer {
  const totalBytes = logicalBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = tracker.create(device, {
    label: `${label}-prefill`, size: totalBytes,
    usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true,
  });
  const all = new Uint32Array(prefill.getMappedRange());
  all.fill(STORAGE_GUARD_U32);
  if (elementStorage === "f16") {
    new Uint16Array(all.buffer, STORAGE_GUARD_BYTES, logicalBytes / 2)
      .fill(OUTPUT_PREFILL_QNAN_F16);
  } else {
    new Uint32Array(all.buffer, STORAGE_GUARD_BYTES, logicalBytes / 4)
      .fill(SCRATCH_PREFILL_QNAN_F32);
  }
  tracker.unmap(prefill);
  const buffer = tracker.create(device, {
    label, size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  return Object.freeze({ buffer,
    binding: Object.freeze({ buffer, offset: STORAGE_GUARD_BYTES,
      size: logicalBytes }), prefill, logicalBytes, totalBytes, elementStorage });
}

function rangeControlBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  probeSpec: Opt0077ProbeSpec,
): GPUBuffer {
  const buffer = tracker.create(device, {
    label, size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set([
    probeSpec.base, probeSpec.count, 0, 0,
  ]);
  tracker.unmap(buffer);
  return buffer;
}

function rangeBinding(
  control: GPUBuffer,
  probeSpec: Opt0077ProbeSpec,
): AceVaeOutputRangeBinding {
  return Object.freeze({ base: probeSpec.base, count: probeSpec.count,
    control: Object.freeze({ buffer: control, offset: 0, size: 16 }) });
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function destroyPreparedCase(
  tracker: BufferTracker,
  prepared: PreparedCase,
): void {
  for (const buffer of prepared.ownedBuffers) tracker.destroy(buffer);
}

function everyU32(values: Uint32Array, expected: number): boolean {
  for (const value of values) if (value !== expected) return false;
  return true;
}

function xorshift32(value: number): number {
  let state = value >>> 0;
  state ^= state << 13;
  state ^= state >>> 17;
  state ^= state << 5;
  return state >>> 0;
}

function stableStringHash32(value: string): number {
  let hash = 0x811c_9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x0100_0193);
  }
  return hash >>> 0;
}

function numberToFloat16Bits(value: number): number {
  const f32 = new Float32Array([value]);
  const bits = new Uint32Array(f32.buffer)[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7f_ffff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const significant = mantissa | 0x80_0000;
    const shift = 14 - halfExponent;
    const truncated = significant >>> shift;
    const remainder = significant & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    return sign | (truncated + (remainder > halfway ||
      (remainder === halfway && (truncated & 1) !== 0) ? 1 : 0));
  }
  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 ||
    (remainder === 0x1000 && (halfMantissa & 1) !== 0)) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      const nextExponent = halfExponent + 1;
      return sign | (nextExponent >= 0x1f ? 0x7c00 : nextExponent << 10);
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

function float16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) {
    return fraction === 0 ? sign * 0 : sign * 2 ** -14 * fraction / 1_024;
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1_024);
}

async function buildOpt0077ResultHashes(
  caseReceipts: readonly Readonly<Record<string, unknown>>[],
): Promise<Readonly<Record<string, unknown>>> {
  const arms = Object.freeze([
    "current", "candidate", "scalarOracle", "candidateRepeat",
  ] as const);
  const perCase = Object.freeze(Object.fromEntries(caseReceipts.map((receipt) =>
    [String(receipt["id"]), receipt["resultHashes"]]
  )));
  const production = caseReceipts.filter((receipt) =>
    receipt["kind"] === "production");
  const aggregate = async (
    receipts: readonly Readonly<Record<string, unknown>>[],
    arm: typeof arms[number],
  ): Promise<string> => sha256Text(receipts.map((receipt) => {
    const hashes = receipt["resultHashes"] as Readonly<Record<string, unknown>>;
    return `${String(receipt["id"])}:${String(hashes[arm])}`;
  }).join("\n"));
  const allCasesEntries = await Promise.all(arms.map(async (arm) =>
    [arm, await aggregate(caseReceipts, arm)] as const
  ));
  const productionEntries = await Promise.all(arms.map(async (arm) =>
    [arm, await aggregate(production, arm)] as const
  ));
  return Object.freeze({
    hashDefinition:
      "sha256-selected-u16-per-probe-then-id-ordered-sha256-manifest",
    perStratum: Object.freeze(Object.fromEntries(production.map((receipt) =>
      [String(receipt["id"]), receipt["resultHashes"]]
    ))),
    perCase,
    productionAggregate: Object.freeze(Object.fromEntries(productionEntries)),
    allCaseAggregate: Object.freeze(Object.fromEntries(allCasesEntries)),
  });
}

async function sha256U16(words: Uint16Array): Promise<string> {
  const copy = words.slice();
  const digest = await crypto.subtle.digest("SHA-256", copy.buffer);
  return hexadecimalDigest(digest);
}

async function sha256Text(value: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return hexadecimalDigest(digest);
}

function hexadecimalDigest(digest: ArrayBuffer): string {
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
}

function requireAdapter(adapter: GPUAdapter): void {
  if (!adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    !adapter.features.has("timestamp-query") ||
    Number(adapter.info.subgroupMinSize) !== 32 ||
    Number(adapter.info.subgroupMaxSize) !== 32 ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 128 ||
    adapter.limits.maxComputeWorkgroupSizeX < 128) {
    throw new Error(
      "OPT-0077 requires timestamp-query, shader-f16, fixed32 subgroups, and WG128",
    );
  }
}

function requireLittleEndianHost(): void {
  const words = new Uint16Array([0x0102]);
  if (new Uint8Array(words.buffer)[0] !== 0x02) {
    throw new Error("OPT-0077 fixtures require a little-endian host");
  }
}

function fieldParameters(selector: string): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    `${selector} input[name]`,
  )) parameters.set(input.name, input.value);
  return parameters;
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing ${selector}`);
  return element;
}

function publish(
  receipt: Readonly<Record<string, unknown>>,
  status: "passed" | "failed",
): void {
  window.__ACE_OPT0077_RESULT__ = receipt;
  document.body.dataset.status = status;
  requireElement<HTMLElement>("#result").textContent =
    JSON.stringify(receipt, null, 2);
}

function publishFailure(
  error: unknown,
  stage: string,
  evidence: Readonly<Record<string, unknown>> = Object.freeze({}),
): void {
  const failure = error instanceof Error
    ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
    : Object.freeze({ message: String(error) });
  publish(Object.freeze({ schema: RECEIPT_SCHEMA, experiment: EXPERIMENT_ID,
    status: "failed", passed: false, stage, failure, ...evidence,
    productionIntegrationAuthorized: false }), "failed");
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

async function settlePostDrainEvents(): Promise<void> {
  await Promise.resolve();
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

async function buildIdentity(
  adapter: GPUAdapter,
  cases: readonly Opt0077CaseSpec[],
  mathProof: Readonly<Record<string, unknown>>,
): Promise<Readonly<Record<string, unknown>>> {
  const generatedWgsl = await Promise.all(
    buildOpt0077ProductionStrata().map(async (spec) => Object.freeze({
      id: spec.id,
      currentSha256: await sha256Text(
        aceOpt0051VaeConv1dK4RowReuse16x64Wgsl(
          spec.shape,
          true,
          "float16",
        ),
      ),
      candidateSha256: await sha256Text(
        aceOpt0077VaeK7Rfft16Wgsl(spec.shape),
      ),
      scalarOracleSha256: await sha256Text(
        aceFp16VaeConv1dSubgroupWgsl(spec.shape, true, "float16"),
      ),
    })),
  );
  const sourceSha256 = Object.freeze({
    currentSelector: await sha256Text(currentSelectorSource),
    currentRowReuse: await sha256Text(currentRowReuseSource),
    scalarK7: await sha256Text(scalarK7Source),
    candidate: await sha256Text(candidateSource),
    candidateMath: await sha256Text(candidateMathSource),
    experimentRecord: await sha256Text(experimentSource),
    harness: await sha256Text(harnessSource),
    html: await sha256Text(htmlSource),
  });
  return Object.freeze({ registrationCommit: REGISTRATION_COMMIT,
    allocationBaselineCommit:
      "47c4dfed1a2b4826cafda14123282ce72d852cb2",
    sourceSha256, generatedWgsl: Object.freeze(generatedWgsl),
    generatedWgslAggregateSha256: await sha256Text(
      generatedWgsl.map((entry) => JSON.stringify(entry)).join("\n"),
    ),
    caseTopologySha256: await sha256Text(JSON.stringify(cases)),
    mathProofSha256: await sha256Text(JSON.stringify(mathProof)),
    adapterInfo: adapter.info,
    browser: navigator.userAgent,
    stockWebGpuOnly: true,
    experimentalBrowserFlags: false,
  });
}
