/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import currentCoreSource from
  "../../src/webgpu/kernels/dit-dense-fp16.ts?raw";
import exactCoreSource from
  "../../src/webgpu/kernels/dit-dense-fp16-cooperative-panels.ts?raw";
import dot4CoreSource from
  "../../src/webgpu/kernels/dit-dense-fp16-cooperative-dot4.ts?raw";
import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../../src/model/manifest.js";
import {
  AceOpt0009DenseGemmKernel,
  aceOpt0009DenseGemmWgsl,
  planAceOpt0009DenseGemm,
} from "../../src/webgpu/kernels/dit-dense-fp16.js";
import {
  ACE_OPT_0019_DENSE_COOPERATIVE_KERNEL_SET_ID,
  AceOpt0019DenseCooperativePanelsKernel,
  aceOpt0019DenseCooperativePanelsWgsl,
  planAceOpt0019DenseCooperativePanels,
} from "../../src/webgpu/kernels/dit-dense-fp16-cooperative-panels.js";
import {
  ACE_OPT_0020_DENSE_COOPERATIVE_DOT4_KERNEL_SET_ID,
  ACE_OPT_0020_DENSE_WORKGROUP_STORAGE_BYTES,
  AceOpt0020DenseCooperativeDot4Kernel,
  aceOpt0020DenseCooperativeDot4Wgsl,
  planAceOpt0020DenseCooperativeDot4,
} from "../../src/webgpu/kernels/dit-dense-fp16-cooperative-dot4.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";

export type Opt0020Arm = "A" | "B" | "C";

export interface Opt0020ShapeSpec {
  readonly id: "h-h" | "h-1024" | "h-6144" | "6144-h";
  readonly shape: AceGemmShape;
  readonly productionMultiplicity: 4 | 2 | 1;
  readonly feedForwardMultiplicity: 0 | 2 | 1;
  readonly fixture: Readonly<{
    ordinal: 0 | 1 | 2 | 3;
    activationSeed: number;
    weightSeed: number;
    activationSha256: string;
    packedWeightSha256: string;
  }>;
}

export interface Opt0020TimingInput {
  readonly id: Opt0020ShapeSpec["id"];
  readonly samples: Readonly<Record<Opt0020Arm, readonly number[]>>;
}

export interface Opt0020ThermalGate {
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

interface GuardedOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly prefill: GPUBuffer;
  readonly readback: GPUBuffer;
  readonly outputElements: number;
  readonly outputBytes: number;
  readonly totalBytes: number;
}

interface ReadbackSnapshot {
  readonly words: Uint32Array;
  readonly nonFiniteCount: number;
  readonly nonzeroCount: number;
  readonly qNaNPrefillCount: number;
  readonly prefixCanaryIntact: boolean;
  readonly suffixCanaryIntact: boolean;
  readonly tailRowWritten: boolean;
}

interface PreparedShape {
  readonly spec: Opt0020ShapeSpec;
  readonly activation: GPUBuffer;
  readonly weight: GPUBuffer;
  readonly output: GPUBuffer;
  readonly dispatches: Readonly<Record<Opt0020Arm, AceGemmDispatch>>;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly currentKernel: AceOpt0009DenseGemmKernel;
  readonly exactKernel: AceOpt0019DenseCooperativePanelsKernel;
  readonly dot4Kernel: AceOpt0020DenseCooperativeDot4Kernel;
  readonly shapes: readonly PreparedShape[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly sourceAuthority: Readonly<Record<string, unknown>>;
  readonly uncapturedErrors: readonly string[];
  readonly preparation: Readonly<Record<string, unknown>>;
  readonly preparedCompletedAtEpochMilliseconds: number;
  readonly updateProgress: (message: string) => void;
  destroy(): Readonly<Record<string, unknown>>;
}

interface NumericalAccumulator {
  count: number;
  controlMean: number;
  candidateMean: number;
  controlM2: number;
  candidateM2: number;
  covariance: number;
  controlSquareSum: number;
  errorSum: number;
  absoluteErrorSum: number;
  errorSquareSum: number;
  maximumAbsoluteControl: number;
  maximumAbsoluteError: number;
  controlMinimum: number;
  controlMaximum: number;
  candidateMinimum: number;
  candidateMaximum: number;
  errorMinimum: number;
  errorMaximum: number;
  differingCount: number;
  signedZeroDifferenceCount: number;
  firstDifference: Readonly<Record<string, unknown>> | null;
  worstDifference: Readonly<Record<string, unknown>> | null;
  ulpCounts: number[];
}

const EXPERIMENT_ID = "OPT-0020" as const;
const REGISTRATION_COMMIT =
  "fce77739841572942eca4e96cc6a9f48eb02a971" as const;
const CURRENT_CORE_SOURCE_SHA256 =
  "a238f67da07c6ba1097da9d9e9e97960ae97d2e1d5c129fcbabf69e962cbb6b3" as const;
const EXACT_CORE_SOURCE_SHA256 =
  "b5dad12724882d3fc942c7df7b10c7b7b89a4bed595125ff11a5905c03152a37" as const;
const DOT4_CORE_SOURCE_SHA256 =
  "466cf7b4c8f860ff55a89b03a5bb2ead99c0d849420918e11d6888e11482d28e" as const;
const ROWS = 2_250;
const HIDDEN = 2_048;
const EXPANDED = 6_144;
const CURRENT_TILE_ROWS = 32;
const CURRENT_TILE_COLUMNS = 256;
const CURRENT_TILE_INNER = 32;
const COOPERATIVE_TILE_ROWS = 64;
const COOPERATIVE_TILE_COLUMNS = 128;
const COOPERATIVE_TILE_INNER = 16;
const TIMING_ROUNDS = 6;
const B_TO_C_COMPLETE_SPEEDUP_THRESHOLD = 1.25;
const A_TO_C_COMPLETE_SPEEDUP_THRESHOLD = 1.55;
const B_TO_C_FEED_FORWARD_SPEEDUP_THRESHOLD = 1.25;
const NRMSE_THRESHOLD = 1e-5;
const SNR_DECIBELS_THRESHOLD = 100;
const PEARSON_THRESHOLD = 0.999999;
const RELATIVE_MAXIMUM_ERROR_THRESHOLD = 1e-3;
const MAXIMUM_ABSOLUTE_ERROR_THRESHOLD = 1e-4;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_2055;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const THERMAL_POLL_MILLISECONDS = 1_000;
const MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS = 1_250;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 5_000;
const FULL_CANDIDATE_CONTROL_COUNT = 25_344_000;
const ARMS = Object.freeze(["A", "B", "C"] as const);
const ARM_PERMUTATIONS = Object.freeze([
  Object.freeze(["A", "B", "C"] as const),
  Object.freeze(["A", "C", "B"] as const),
  Object.freeze(["B", "A", "C"] as const),
  Object.freeze(["B", "C", "A"] as const),
  Object.freeze(["C", "A", "B"] as const),
  Object.freeze(["C", "B", "A"] as const),
]);

const FIXTURE_GENERATOR_VERSION =
  "opt-0020-finite-fp16-cancellation-v1" as const;
const ACTIVATION_SEEDS = Object.freeze([
  0x3141_5926, 0x2718_2818, 0x6a09_e667, 0xbb67_ae85,
] as const);
const WEIGHT_SEEDS = Object.freeze([
  0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c,
] as const);
const FINITE_NON_POWER_FP16_MAGNITUDES = Object.freeze([
  0x2411, 0x28b5, 0x2d53, 0x31e7, 0x356b, 0x39ad,
] as const);
const FIXTURE_SUBJECT_MIX = 0x9e37_79b1;
const FIXTURE_GROUP_MIX = 0x85eb_ca6b;
const FIXTURE_OFFSET_MIX = 0xc2b2_ae35;
const ULP_BINS = Object.freeze([
  Object.freeze({ label: "0", minimum: 0, maximum: 0 }),
  Object.freeze({ label: "1", minimum: 1, maximum: 1 }),
  Object.freeze({ label: "2", minimum: 2, maximum: 2 }),
  Object.freeze({ label: "3", minimum: 3, maximum: 3 }),
  Object.freeze({ label: "4-7", minimum: 4, maximum: 7 }),
  Object.freeze({ label: "8-15", minimum: 8, maximum: 15 }),
  Object.freeze({ label: "16-31", minimum: 16, maximum: 31 }),
  Object.freeze({ label: "32-63", minimum: 32, maximum: 63 }),
  Object.freeze({ label: "64-127", minimum: 64, maximum: 127 }),
  Object.freeze({ label: "128-255", minimum: 128, maximum: 255 }),
  Object.freeze({ label: "256-511", minimum: 256, maximum: 511 }),
  Object.freeze({ label: "512-1023", minimum: 512, maximum: 1_023 }),
  Object.freeze({ label: "1024-4095", minimum: 1_024, maximum: 4_095 }),
  Object.freeze({ label: "4096-65535", minimum: 4_096, maximum: 65_535 }),
  Object.freeze({
    label: "65536-4294967295",
    minimum: 65_536,
    maximum: 0xffff_ffff,
  }),
]);

const SHAPE_SPECS = Object.freeze([
  shapeSpec("h-h", HIDDEN, HIDDEN, 4, 0, 0,
    "e66bc914da370971b0a717a3db8e9fa5b26820fe2d0bd5fa156b650f68fc5b99",
    "6e985fca3119b135d740c2f8814fe3ddbf538bdaabde96cfca1dd363b62d45eb"),
  shapeSpec("h-1024", HIDDEN, 1_024, 2, 0, 1,
    "47b9e4ecef742a678bb263443e783b5ea753dc513af2b986c15d97bbafb315cf",
    "3e4db4f1da0770dc2b2ffd6f2b8643e1778b2993495d4a2085bcc0bea7b383f3"),
  shapeSpec("h-6144", HIDDEN, EXPANDED, 2, 2, 2,
    "1469272232904084304f9834d5b9f1cf152ba940c7e25e82e0baac0708838c62",
    "898266fc61785f391230299844c4e50e6394daa631783e3a40f95ba436cb088c"),
  shapeSpec("6144-h", EXPANDED, HIDDEN, 1, 1, 3,
    "39604dec14faf7d5f8a2c2400f48ce162cbce268de9b176ac7acf2e84612b1aa",
    "e20d4d09edb58957967021c4f4957653be93ab3bcca5828d7a6c32403f2c24da"),
]);

export function buildOpt0020ShapeSpecs(): readonly Opt0020ShapeSpec[] {
  return SHAPE_SPECS;
}

export function buildOpt0020FixtureDeclaration(): Readonly<Record<string, unknown>> {
  return Object.freeze({
    version: FIXTURE_GENERATOR_VERSION,
    activationSeeds: ACTIVATION_SEEDS,
    weightSeeds: WEIGHT_SEEDS,
    finiteNonPowerFp16MagnitudeBits: FINITE_NON_POWER_FP16_MAGNITUDES,
    constants: Object.freeze({
      subjectMix: FIXTURE_SUBJECT_MIX,
      groupMix: FIXTURE_GROUP_MIX,
      offsetMix: FIXTURE_OFFSET_MIX,
      pairedCancellationGroups: "g and g^1 except every eighth group's breaker",
      withinK4ProductSignPattern: "+--+",
      groupSignFlip: "activation sign xor (floor(k/4) & 1)",
    }),
    shapes: Object.freeze(SHAPE_SPECS.map((spec) => Object.freeze({
      id: spec.id,
      shape: spec.shape,
      fixture: spec.fixture,
    }))),
  });
}

export function opt0020ActivationBitsAt(
  shapeIndex: number,
  row: number,
  innerIndex: number,
): number {
  const spec = requireShapeIndex(shapeIndex);
  requireFixtureCoordinate(row, spec.shape.rows, "activation row");
  requireFixtureCoordinate(innerIndex, spec.shape.inner, "activation K");
  const group = Math.floor(innerIndex / 4);
  const offset = innerIndex & 3;
  const rowSign = mix32(
    spec.fixture.activationSeed ^
      Math.imul(row + 1, FIXTURE_SUBJECT_MIX),
  ) >>> 31;
  const sign = rowSign ^ (group & 1);
  return fixtureMagnitudeBits(
    spec.fixture.activationSeed,
    row,
    group,
    offset,
  ) | (sign << 15);
}

export function opt0020PackedWeightBitsAt(
  shapeIndex: number,
  innerIndex: number,
  column: number,
): number {
  const spec = requireShapeIndex(shapeIndex);
  requireFixtureCoordinate(innerIndex, spec.shape.inner, "weight K");
  requireFixtureCoordinate(column, spec.shape.columns, "weight N");
  const group = Math.floor(innerIndex / 4);
  const offset = innerIndex & 3;
  const columnSign = mix32(
    spec.fixture.weightSeed ^
      Math.imul(column + 1, FIXTURE_SUBJECT_MIX),
  ) >>> 31;
  const withinK4Sign = offset === 1 || offset === 2 ? 1 : 0;
  const sign = columnSign ^ withinK4Sign;
  return fixtureMagnitudeBits(
    spec.fixture.weightSeed,
    column,
    group,
    offset,
  ) | (sign << 15);
}

export function fillOpt0020ActivationFixture(
  values: Float32Array,
  shapeIndex: number,
): void {
  const spec = requireShapeIndex(shapeIndex);
  const expected = spec.shape.rows * spec.shape.inner;
  if (values.length !== expected) {
    throw new RangeError(`OPT-0020 activation fixture requires ${expected} F32 values`);
  }
  let index = 0;
  for (let row = 0; row < spec.shape.rows; row += 1) {
    for (let inner = 0; inner < spec.shape.inner; inner += 1) {
      values[index] = finiteNormalHalfToNumber(
        opt0020ActivationBitsAt(shapeIndex, row, inner),
      );
      index += 1;
    }
  }
}

export function fillOpt0020PackedWeightFixture(
  values: Uint16Array,
  shapeIndex: number,
): void {
  const spec = requireShapeIndex(shapeIndex);
  const expected = spec.shape.inner * spec.shape.columns;
  if (values.length !== expected) {
    throw new RangeError(`OPT-0020 weight fixture requires ${expected} U16 values`);
  }
  const innerTiles = spec.shape.inner / 32;
  let physical = 0;
  for (let columnTile = 0; columnTile < spec.shape.columns / 256;
    columnTile += 1) {
    for (let innerTile = 0; innerTile < innerTiles; innerTile += 1) {
      for (let innerInTile = 0; innerInTile < 32; innerInTile += 1) {
        const inner = innerTile * 32 + innerInTile;
        for (let columnInTile = 0; columnInTile < 256; columnInTile += 1) {
          const column = columnTile * 256 + columnInTile;
          values[physical] = opt0020PackedWeightBitsAt(
            shapeIndex,
            inner,
            column,
          );
          physical += 1;
        }
      }
    }
  }
}

export function buildOpt0020TimingOrders(): readonly Readonly<{
  roundIndex: number;
  shapeIndex: number;
  shapeOrder: readonly number[];
  order: readonly Opt0020Arm[];
}>[] {
  const baseShapeOrder = SHAPE_SPECS.map((_, index) => index);
  return Object.freeze(ARM_PERMUTATIONS.flatMap((order, roundIndex) => {
    const rotation = roundIndex % SHAPE_SPECS.length;
    const shapeOrder = Object.freeze([
      ...baseShapeOrder.slice(rotation),
      ...baseShapeOrder.slice(0, rotation),
    ]);
    return shapeOrder.map((shapeIndex) => Object.freeze({
      roundIndex,
      shapeIndex,
      shapeOrder,
      order,
    }));
  }));
}

export function summarizeOpt0020Timing(
  inputs: readonly Opt0020TimingInput[],
): Readonly<Record<string, unknown>> {
  if (inputs.length !== SHAPE_SPECS.length) {
    throw new Error("OPT-0020 requires all four exact-shape timings");
  }
  const complete = { A: 0, B: 0, C: 0 };
  const feedForward = { A: 0, B: 0, C: 0 };
  const strata = inputs.map((input, index) => {
    const spec = SHAPE_SPECS[index];
    if (spec === undefined || input.id !== spec.id) {
      throw new Error("OPT-0020 timing shape order changed");
    }
    const medians = Object.freeze({
      A: median6(input.samples.A),
      B: median6(input.samples.B),
      C: median6(input.samples.C),
    });
    for (const arm of ARMS) {
      complete[arm] += medians[arm] * spec.productionMultiplicity;
      feedForward[arm] += medians[arm] * spec.feedForwardMultiplicity;
    }
    return Object.freeze({
      id: spec.id,
      shape: spec.shape,
      productionMultiplicity: spec.productionMultiplicity,
      feedForwardMultiplicity: spec.feedForwardMultiplicity,
      samples: input.samples,
      medians,
      cFasterThanB: medians.C < medians.B,
      bToCSpeedup: medians.B / medians.C,
      aToCSpeedup: medians.A / medians.C,
    });
  });
  const everyShapeCFasterThanB = strata.every((item) => item.cFasterThanB);
  const bToCCompleteSpeedup = complete.B / complete.C;
  const aToCCompleteSpeedup = complete.A / complete.C;
  const bToCFeedForwardSpeedup = feedForward.B / feedForward.C;
  const passed = everyShapeCFasterThanB &&
    bToCCompleteSpeedup >= B_TO_C_COMPLETE_SPEEDUP_THRESHOLD &&
    aToCCompleteSpeedup >= A_TO_C_COMPLETE_SPEEDUP_THRESHOLD &&
    bToCFeedForwardSpeedup >= B_TO_C_FEED_FORWARD_SPEEDUP_THRESHOLD;
  return Object.freeze({
    samplesPerArmPerShape: TIMING_ROUNDS,
    completeDense: Object.freeze({
      multiplicities: "4/2/2/1",
      milliseconds: Object.freeze(complete),
      bToCSpeedup: bToCCompleteSpeedup,
      bToCThreshold: B_TO_C_COMPLETE_SPEEDUP_THRESHOLD,
      aToCSpeedup: aToCCompleteSpeedup,
      aToCThreshold: A_TO_C_COMPLETE_SPEEDUP_THRESHOLD,
    }),
    feedForward: Object.freeze({
      multiplicities: "0/0/2/1",
      milliseconds: Object.freeze(feedForward),
      bToCSpeedup: bToCFeedForwardSpeedup,
      bToCThreshold: B_TO_C_FEED_FORWARD_SPEEDUP_THRESHOLD,
    }),
    everyShapeCFasterThanB,
    strata: Object.freeze(strata),
    passed,
    decision: passed
      ? "positive-package-layer-gate-authorized"
      : "negative-stop-primitive-gate",
  });
}

export function parseOpt0020ThermalGate(
  parameters: URLSearchParams,
  preparedCompletedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0020ThermalGate {
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
    maximumPollGapMilliseconds <= 0 ||
    nonNominalObservationCount !== 0 ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    startedAtEpochMilliseconds < preparedCompletedAtEpochMilliseconds ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS) {
    throw new Error("OPT-0020 thermal gate is incomplete, stale, or non-nominal");
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

export function compareOpt0020Numerics(
  controlWords: Uint32Array,
  candidateWords: Uint32Array,
  columns = controlWords.length,
): Readonly<Record<string, unknown>> {
  if (!Number.isSafeInteger(columns) || columns <= 0 ||
    controlWords.length !== candidateWords.length ||
    controlWords.length % columns !== 0) {
    throw new RangeError("OPT-0020 numerical comparison shape changed");
  }
  const accumulator = createNumericalAccumulator();
  accumulateNumerics(
    accumulator,
    controlWords,
    candidateWords,
    Object.freeze({ id: "test", shape: { columns } }),
  );
  return finalizeNumerics(accumulator);
}

function fixtureMagnitudeBits(
  seed: number,
  subject: number,
  group: number,
  offset: number,
): number {
  const cancellationGroup = group % 8 === 7 ? group : group & ~1;
  const symmetricOffset = group % 16 === 0
    ? 0
    : offset === 0 || offset === 3 ? 0 : 1;
  const mixed = mix32(
    seed ^
      Math.imul(subject + 1, FIXTURE_SUBJECT_MIX) ^
      Math.imul(cancellationGroup + 1, FIXTURE_GROUP_MIX) ^
      Math.imul(symmetricOffset + 1, FIXTURE_OFFSET_MIX),
  );
  return FINITE_NON_POWER_FP16_MAGNITUDES[
    mixed % FINITE_NON_POWER_FP16_MAGNITUDES.length
  ]!;
}

function finiteNormalHalfToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0 || exponent === 0x1f) {
    throw new RangeError("OPT-0020 fixture requires finite normal FP16 bits");
  }
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1_024);
}

function mix32(value: number): number {
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb_352d);
  value ^= value >>> 15;
  value = Math.imul(value, 0x846c_a68b);
  value ^= value >>> 16;
  return value >>> 0;
}

function requireShapeIndex(index: number): Opt0020ShapeSpec {
  if (!Number.isSafeInteger(index)) {
    throw new RangeError("OPT-0020 fixture shape index must be a safe integer");
  }
  const spec = SHAPE_SPECS[index];
  if (spec === undefined) {
    throw new RangeError(`OPT-0020 fixture rejects shape index ${index}`);
  }
  return spec;
}

function requireFixtureCoordinate(
  value: number,
  extent: number,
  label: string,
): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= extent) {
    throw new RangeError(`OPT-0020 ${label} ${value} is out of bounds`);
  }
}

function createNumericalAccumulator(): NumericalAccumulator {
  return {
    count: 0,
    controlMean: 0,
    candidateMean: 0,
    controlM2: 0,
    candidateM2: 0,
    covariance: 0,
    controlSquareSum: 0,
    errorSum: 0,
    absoluteErrorSum: 0,
    errorSquareSum: 0,
    maximumAbsoluteControl: 0,
    maximumAbsoluteError: 0,
    controlMinimum: Number.POSITIVE_INFINITY,
    controlMaximum: Number.NEGATIVE_INFINITY,
    candidateMinimum: Number.POSITIVE_INFINITY,
    candidateMaximum: Number.NEGATIVE_INFINITY,
    errorMinimum: Number.POSITIVE_INFINITY,
    errorMaximum: Number.NEGATIVE_INFINITY,
    differingCount: 0,
    signedZeroDifferenceCount: 0,
    firstDifference: null,
    worstDifference: null,
    ulpCounts: ULP_BINS.map(() => 0),
  };
}

function accumulateNumerics(
  accumulator: NumericalAccumulator,
  controlWords: Uint32Array,
  candidateWords: Uint32Array,
  spec: Readonly<{
    id: string;
    shape: Readonly<{ columns: number }>;
  }>,
): void {
  if (controlWords.length !== candidateWords.length) {
    throw new Error("OPT-0020 candidate/control output length changed");
  }
  const control = new Float32Array(
    controlWords.buffer,
    controlWords.byteOffset,
    controlWords.length,
  );
  const candidate = new Float32Array(
    candidateWords.buffer,
    candidateWords.byteOffset,
    candidateWords.length,
  );
  for (let index = 0; index < control.length; index += 1) {
    const a = control[index]!;
    const c = candidate[index]!;
    const error = c - a;
    const absoluteError = Math.abs(error);
    const absoluteControl = Math.abs(a);
    const controlWord = controlWords[index]!;
    const candidateWord = candidateWords[index]!;
    const ulpDistance = signAwareF32UlpDistance(controlWord, candidateWord);
    accumulator.ulpCounts[ulpBinIndex(ulpDistance)]! += 1;
    accumulator.count += 1;
    const count = accumulator.count;
    const controlDelta = a - accumulator.controlMean;
    accumulator.controlMean += controlDelta / count;
    const candidateDelta = c - accumulator.candidateMean;
    accumulator.candidateMean += candidateDelta / count;
    accumulator.controlM2 += controlDelta * (a - accumulator.controlMean);
    accumulator.candidateM2 += candidateDelta *
      (c - accumulator.candidateMean);
    accumulator.covariance += controlDelta *
      (c - accumulator.candidateMean);
    accumulator.controlSquareSum += a * a;
    accumulator.errorSum += error;
    accumulator.absoluteErrorSum += absoluteError;
    accumulator.errorSquareSum += error * error;
    accumulator.maximumAbsoluteControl = Math.max(
      accumulator.maximumAbsoluteControl,
      absoluteControl,
    );
    accumulator.controlMinimum = Math.min(accumulator.controlMinimum, a);
    accumulator.controlMaximum = Math.max(accumulator.controlMaximum, a);
    accumulator.candidateMinimum = Math.min(accumulator.candidateMinimum, c);
    accumulator.candidateMaximum = Math.max(accumulator.candidateMaximum, c);
    accumulator.errorMinimum = Math.min(accumulator.errorMinimum, error);
    accumulator.errorMaximum = Math.max(accumulator.errorMaximum, error);
    if (controlWord !== candidateWord) {
      accumulator.differingCount += 1;
      const location = Object.freeze({
        shapeId: spec.id,
        linearIndex: index,
        row: Math.floor(index / spec.shape.columns),
        column: index % spec.shape.columns,
        control: a,
        candidate: c,
        controlU32: controlWord,
        candidateU32: candidateWord,
        error,
        absoluteError,
        signAwareUlpDistance: ulpDistance,
      });
      accumulator.firstDifference ??= location;
      if (absoluteError > accumulator.maximumAbsoluteError ||
        (absoluteError === accumulator.maximumAbsoluteError &&
          ulpDistance > Number(
            accumulator.worstDifference?.["signAwareUlpDistance"] ?? -1,
          ))) {
        accumulator.maximumAbsoluteError = absoluteError;
        accumulator.worstDifference = location;
      }
    }
    if ((controlWord & 0x7fff_ffff) === 0 &&
      (candidateWord & 0x7fff_ffff) === 0 &&
      controlWord !== candidateWord) {
      accumulator.signedZeroDifferenceCount += 1;
    }
  }
}

function finalizeNumerics(
  accumulator: NumericalAccumulator,
): Readonly<Record<string, unknown>> {
  if (accumulator.count === 0) {
    throw new Error("OPT-0020 cannot finalize an empty numerical comparison");
  }
  const controlRms = Math.sqrt(
    accumulator.controlSquareSum / accumulator.count,
  );
  const errorRms = Math.sqrt(accumulator.errorSquareSum / accumulator.count);
  const nrmse = errorRms / Math.max(controlRms, 1e-12);
  const snrNumeric = errorRms === 0
    ? Number.POSITIVE_INFINITY
    : 20 * Math.log10(controlRms / errorRms);
  const pearsonDenominator = Math.sqrt(
    accumulator.controlM2 * accumulator.candidateM2,
  );
  const pearson = pearsonDenominator === 0
    ? accumulator.differingCount === 0 ? 1 : 0
    : accumulator.covariance / pearsonDenominator;
  const relativeMaximumError = accumulator.maximumAbsoluteError /
    Math.max(accumulator.maximumAbsoluteControl, 1e-6);
  const passed = nrmse <= NRMSE_THRESHOLD &&
    snrNumeric >= SNR_DECIBELS_THRESHOLD &&
    pearson >= PEARSON_THRESHOLD &&
    relativeMaximumError <= RELATIVE_MAXIMUM_ERROR_THRESHOLD &&
    accumulator.maximumAbsoluteError <= MAXIMUM_ABSOLUTE_ERROR_THRESHOLD;
  return Object.freeze({
    count: accumulator.count,
    differingCount: accumulator.differingCount,
    ranges: Object.freeze({
      control: Object.freeze([
        accumulator.controlMinimum,
        accumulator.controlMaximum,
      ]),
      candidate: Object.freeze([
        accumulator.candidateMinimum,
        accumulator.candidateMaximum,
      ]),
      error: Object.freeze([
        accumulator.errorMinimum,
        accumulator.errorMaximum,
      ]),
    }),
    signedMeanError: accumulator.errorSum / accumulator.count,
    meanAbsoluteError: accumulator.absoluteErrorSum / accumulator.count,
    rmsError: errorRms,
    controlRms,
    nrmse,
    snrDecibels: Number.isFinite(snrNumeric)
      ? snrNumeric
      : snrNumeric === Number.POSITIVE_INFINITY
      ? "positive-infinity"
      : snrNumeric === Number.NEGATIVE_INFINITY
      ? "negative-infinity"
      : "not-a-number",
    pearsonCorrelation: pearson,
    maximumAbsoluteControl: accumulator.maximumAbsoluteControl,
    maximumAbsoluteError: accumulator.maximumAbsoluteError,
    relativeMaximumError,
    firstDifference: accumulator.firstDifference,
    worstDifference: accumulator.worstDifference,
    signedZeroDifferenceCount: accumulator.signedZeroDifferenceCount,
    signAwareF32UlpDistribution: Object.freeze(ULP_BINS.map((bin, index) =>
      Object.freeze({ ...bin, count: accumulator.ulpCounts[index] }))
    ),
    thresholds: Object.freeze({
      nrmseMaximum: NRMSE_THRESHOLD,
      snrDecibelsMinimum: SNR_DECIBELS_THRESHOLD,
      pearsonCorrelationMinimum: PEARSON_THRESHOLD,
      relativeMaximumErrorMaximum: RELATIVE_MAXIMUM_ERROR_THRESHOLD,
      maximumAbsoluteErrorMaximum: MAXIMUM_ABSOLUTE_ERROR_THRESHOLD,
    }),
    passed,
  });
}

function mergeNumericalAccumulators(
  target: NumericalAccumulator,
  source: NumericalAccumulator,
): void {
  if (source.count === 0) return;
  if (target.count === 0) {
    Object.assign(target, {
      ...source,
      ulpCounts: [...source.ulpCounts],
    });
    return;
  }
  const targetCount = target.count;
  const sourceCount = source.count;
  const combinedCount = targetCount + sourceCount;
  const controlDelta = source.controlMean - target.controlMean;
  const candidateDelta = source.candidateMean - target.candidateMean;
  const crossWeight = targetCount * sourceCount / combinedCount;
  target.controlM2 += source.controlM2 +
    controlDelta * controlDelta * crossWeight;
  target.candidateM2 += source.candidateM2 +
    candidateDelta * candidateDelta * crossWeight;
  target.covariance += source.covariance +
    controlDelta * candidateDelta * crossWeight;
  target.controlMean += controlDelta * sourceCount / combinedCount;
  target.candidateMean += candidateDelta * sourceCount / combinedCount;
  target.count = combinedCount;
  target.controlSquareSum += source.controlSquareSum;
  target.errorSum += source.errorSum;
  target.absoluteErrorSum += source.absoluteErrorSum;
  target.errorSquareSum += source.errorSquareSum;
  target.maximumAbsoluteControl = Math.max(
    target.maximumAbsoluteControl,
    source.maximumAbsoluteControl,
  );
  target.controlMinimum = Math.min(target.controlMinimum, source.controlMinimum);
  target.controlMaximum = Math.max(target.controlMaximum, source.controlMaximum);
  target.candidateMinimum = Math.min(
    target.candidateMinimum,
    source.candidateMinimum,
  );
  target.candidateMaximum = Math.max(
    target.candidateMaximum,
    source.candidateMaximum,
  );
  target.errorMinimum = Math.min(target.errorMinimum, source.errorMinimum);
  target.errorMaximum = Math.max(target.errorMaximum, source.errorMaximum);
  target.differingCount += source.differingCount;
  target.signedZeroDifferenceCount += source.signedZeroDifferenceCount;
  target.firstDifference ??= source.firstDifference;
  const sourceWorstUlp = Number(
    source.worstDifference?.["signAwareUlpDistance"] ?? -1,
  );
  const targetWorstUlp = Number(
    target.worstDifference?.["signAwareUlpDistance"] ?? -1,
  );
  if (source.maximumAbsoluteError > target.maximumAbsoluteError ||
    (source.maximumAbsoluteError === target.maximumAbsoluteError &&
      sourceWorstUlp > targetWorstUlp)) {
    target.maximumAbsoluteError = source.maximumAbsoluteError;
    target.worstDifference = source.worstDifference;
  }
  for (let index = 0; index < target.ulpCounts.length; index += 1) {
    target.ulpCounts[index]! += source.ulpCounts[index]!;
  }
}

function signAwareF32UlpDistance(left: number, right: number): number {
  const leftOrdered = (left & 0x8000_0000) === 0
    ? (left ^ 0x8000_0000) >>> 0
    : (~left) >>> 0;
  const rightOrdered = (right & 0x8000_0000) === 0
    ? (right ^ 0x8000_0000) >>> 0
    : (~right) >>> 0;
  return Math.abs(leftOrdered - rightOrdered);
}

function ulpBinIndex(distance: number): number {
  for (let index = 0; index < ULP_BINS.length; index += 1) {
    if (distance <= ULP_BINS[index]!.maximum) return index;
  }
  throw new RangeError(`OPT-0020 invalid ULP distance ${distance}`);
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
      if (value.correctness["passed"] !== true) {
        finishPage("passed", buildCorrectnessStopReceipt(value));
        return;
      }
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
    progress.textContent = "running six rotated A/B/C timing rounds";
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

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  private readonly sizes = new Map<GPUBuffer, number>();
  created = 0;
  destroyed = 0;
  liveBytes = 0;
  maximumLiveBytes = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    const size = Number(descriptor.size);
    this.live.add(buffer);
    this.sizes.set(buffer, size);
    this.created += 1;
    this.liveBytes += size;
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
  requireLittleEndianHost();
  const preparationStartedAtEpochMilliseconds = Date.now();
  const preparationStarted = performance.now();
  updateProgress("authenticating all three frozen kernel owners");
  const sourceStarted = performance.now();
  const sourceAuthority = await buildSourceAuthority();
  const sourceAuthorityMilliseconds = performance.now() - sourceStarted;
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapterStarted = performance.now();
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const largestBindingBytes = maximumBindingBytes();
  const device = await adapter.requestDevice({
    label: "ace-opt-0020-dense-cooperative-dot4-abc-device",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: {
      maxBufferSize: largestBindingBytes + 2 * STORAGE_GUARD_BYTES,
      maxStorageBufferBindingSize: largestBindingBytes,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 128,
      maxComputeWorkgroupSizeY: 16,
      maxComputeWorkgroupStorageSize:
        ACE_OPT_0020_DENSE_WORKGROUP_STORAGE_BYTES,
    },
  });
  const adapterAndDeviceMilliseconds = performance.now() - adapterStarted;
  const tracker = new BufferTracker();
  const uncapturedErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncapturedErrors.push(event.error.message);
  });
  const capability = Object.freeze({
    subgroupMinSize: 32 as const,
    subgroupMaxSize: 32 as const,
  });
  const currentKernel = AceOpt0009DenseGemmKernel.create(device, capability);
  const exactKernel = AceOpt0019DenseCooperativePanelsKernel.create(device);
  const dot4Kernel = AceOpt0020DenseCooperativeDot4Kernel.create(device);
  const shapes: PreparedShape[] = [];
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
    currentKernel.destroy();
    exactKernel.destroy();
    dot4Kernel.destroy();
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
    const aggregateAccumulator = createNumericalAccumulator();
    for (const [index, spec] of SHAPE_SPECS.entries()) {
      updateProgress(`full-output A/B/C/C correctness ${index + 1}/4: ${spec.id}`);
      shapes.push(await prepareShape(
        device,
        tracker,
        currentKernel,
        exactKernel,
        dot4Kernel,
        spec,
        index,
        aggregateAccumulator,
      ));
      await yieldToBrowser();
    }
    const correctnessMilliseconds = performance.now() - correctnessStarted;
    const correctnessCases = shapes.map((shape) => shape.correctness);
    const aggregateNumerics = finalizeNumerics(aggregateAccumulator);
    const aggregateOutputHashManifestSha256 =
      await buildAggregateOutputHashManifest(correctnessCases);
    if (aggregateNumerics["count"] !== FULL_CANDIDATE_CONTROL_COUNT) {
      throw new Error("OPT-0020 aggregate candidate/control count changed");
    }
    const correctnessPassed = aggregateNumerics["passed"] === true &&
      correctnessCases.every((item) => item["passed"] === true);
    let warmupMilliseconds = 0;
    if (correctnessPassed) {
      updateProgress("symmetric warmup of A, B, and C on all four shapes");
      const warmupStarted = performance.now();
      for (const shape of shapes) {
        for (const arm of ARMS) {
          await executeTimedDispatch(device, shape.dispatches[arm]);
        }
        await yieldToBrowser();
      }
      warmupMilliseconds = performance.now() - warmupStarted;
    }
    await device.queue.onSubmittedWorkDone();
    queueDrained = true;
    await settlePostDrainEvents();
    if (uncapturedErrors.length !== 0) {
      throw new Error(`OPT-0020 observed ${uncapturedErrors.length} GPU errors`);
    }
    const preparedCompletedAtEpochMilliseconds = Date.now();
    const preparation = Object.freeze({
      startedAtEpochMilliseconds: preparationStartedAtEpochMilliseconds,
      completedAtEpochMilliseconds: preparedCompletedAtEpochMilliseconds,
      totalMilliseconds: performance.now() - preparationStarted,
      sourceAuthorityMilliseconds,
      adapterAndDeviceMilliseconds,
      allocationCompilationUploadAndCorrectnessMilliseconds:
        correctnessMilliseconds,
      symmetricWarmupMilliseconds: warmupMilliseconds,
      memoryHighWaterBytes: tracker.maximumLiveBytes,
    });
    return Object.freeze({
      adapter,
      device,
      tracker,
      currentKernel,
      exactKernel,
      dot4Kernel,
      shapes: Object.freeze(shapes),
      correctness: Object.freeze({
        shapeCount: shapes.length,
        executionCount: shapes.length * 4,
        exactComparisonCount: shapes.length * 2,
        aVersusBComparedU32Count: FULL_CANDIDATE_CONTROL_COUNT,
        cFirstVersusRerunComparedU32Count: FULL_CANDIDATE_CONTROL_COUNT,
        candidateVersusControlStatCount: FULL_CANDIDATE_CONTROL_COUNT,
        qNaNPrefillCompleteWrites: true,
        canariesUntouched: true,
        finiteOutputs: true,
        cDeterministicReruns: true,
        aVersusBRawU32Exact: true,
        uncapturedGpuErrorCount: 0,
        aggregateOutputHashManifestSha256,
        aggregateNumerics,
        cases: Object.freeze(correctnessCases),
        passed: correctnessPassed,
      }),
      sourceAuthority,
      uncapturedErrors,
      preparation,
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

async function buildAggregateOutputHashManifest(
  cases: readonly Readonly<Record<string, unknown>>[],
): Promise<Readonly<Record<string, unknown>>> {
  const keys = Object.freeze(["A", "B", "CFirst", "CRerun"] as const);
  const hashes = {} as Record<(typeof keys)[number], string>;
  for (const key of keys) {
    const manifest = cases.map((item) => {
      const shapeHashes = item["hashes"] as Readonly<Record<string, unknown>>;
      return `${String(item["id"])}:${String(shapeHashes[key])}`;
    }).join("\n");
    hashes[key] = await sha256Text(manifest);
  }
  return Object.freeze({
    algorithm:
      "sha256-utf8-of-ordered-shape-id-colon-raw-output-sha256-lines-v1",
    shapeOrder: Object.freeze(cases.map((item) => item["id"])),
    hashes: Object.freeze(hashes),
  });
}

async function prepareShape(
  device: GPUDevice,
  tracker: BufferTracker,
  currentKernel: AceOpt0009DenseGemmKernel,
  exactKernel: AceOpt0019DenseCooperativePanelsKernel,
  dot4Kernel: AceOpt0020DenseCooperativeDot4Kernel,
  spec: Opt0020ShapeSpec,
  shapeIndex: number,
  aggregateAccumulator: NumericalAccumulator,
): Promise<PreparedShape> {
  const currentPlan = planAceOpt0009DenseGemm(spec.shape);
  const exactPlan = planAceOpt0019DenseCooperativePanels(spec.shape);
  const dot4Plan = planAceOpt0020DenseCooperativeDot4(spec.shape);
  if (currentPlan.tileRows !== CURRENT_TILE_ROWS ||
    currentPlan.tileColumns !== CURRENT_TILE_COLUMNS ||
    currentPlan.tileInner !== CURRENT_TILE_INNER ||
    exactPlan.tileRows !== COOPERATIVE_TILE_ROWS ||
    exactPlan.tileColumns !== COOPERATIVE_TILE_COLUMNS ||
    exactPlan.tileInner !== COOPERATIVE_TILE_INNER ||
    dot4Plan.tileRows !== COOPERATIVE_TILE_ROWS ||
    dot4Plan.tileColumns !== COOPERATIVE_TILE_COLUMNS ||
    dot4Plan.tileInner !== COOPERATIVE_TILE_INNER ||
    exactPlan.workgroupCount !== dot4Plan.workgroupCount ||
    exactPlan.scheduledMultiplyAdds !== dot4Plan.scheduledMultiplyAdds ||
    exactPlan.estimatedGlobalOperandBytes !==
      dot4Plan.estimatedGlobalOperandBytes ||
    exactPlan.barrierEvents !== dot4Plan.barrierEvents) {
    throw new Error(`OPT-0020 ${spec.id} frozen three-arm topology changed`);
  }
  const activationFixture = await createActivationBuffer(
    device,
    tracker,
    spec,
    shapeIndex,
  );
  const weightFixture = await createWeightBuffer(
    device,
    tracker,
    spec,
    shapeIndex,
  );
  const guarded = createGuardedOutput(device, tracker, spec);
  const bindings: AceGemmBufferBindings = Object.freeze({
    activation: binding(
      activationFixture.buffer,
      currentPlan.activationElements * 4,
    ),
    weight: binding(weightFixture.buffer, currentPlan.weightElements * 2),
    output: guarded.binding,
  });
  const A = await currentKernel.createDispatch(
    `opt-0020-${spec.id}-A-current-opt-0009`,
    spec.shape,
    bindings,
  );
  const B = await exactKernel.createDispatch(
    `opt-0020-${spec.id}-B-exact-opt-0019`,
    spec.shape,
    bindings,
  );
  const C = await dot4Kernel.createDispatch(
    `opt-0020-${spec.id}-C-dot4`,
    spec.shape,
    bindings,
  );
  if (A.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT ||
    B.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT ||
    C.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT) {
    throw new Error(`OPT-0020 ${spec.id} package weight-layout identity changed`);
  }
  const correctness = await verifyCompleteShape(
    device,
    spec,
    guarded,
    Object.freeze({ A, B, C }),
    aggregateAccumulator,
  );
  tracker.destroy(guarded.prefill);
  tracker.destroy(guarded.readback);
  return Object.freeze({
    spec,
    activation: activationFixture.buffer,
    weight: weightFixture.buffer,
    output: guarded.buffer,
    dispatches: Object.freeze({ A, B, C }),
    correctness: Object.freeze({
      ...correctness,
      fixture: Object.freeze({
        generatorVersion: FIXTURE_GENERATOR_VERSION,
        activationSeed: spec.fixture.activationSeed,
        weightSeed: spec.fixture.weightSeed,
        activationSha256: activationFixture.sha256,
        packedWeightSha256: weightFixture.sha256,
        activationBytes: currentPlan.activationElements * 4,
        packedWeightBytes: currentPlan.weightElements * 2,
      }),
      plans: Object.freeze({
        A: compactPlan(currentPlan),
        B: compactPlan(exactPlan),
        C: compactPlan(dot4Plan),
      }),
    }),
  });
}

async function verifyCompleteShape(
  device: GPUDevice,
  spec: Opt0020ShapeSpec,
  output: GuardedOutput,
  dispatches: Readonly<Record<Opt0020Arm, AceGemmDispatch>>,
  aggregateAccumulator: NumericalAccumulator,
): Promise<Readonly<Record<string, unknown>>> {
  let control: ReadbackSnapshot | undefined = await executeCorrectnessDispatch(
    device,
    output,
    dispatches.A,
  );
  requireCompleteSnapshot(control, `${spec.id} A`);
  const controlSha256 = await sha256U32(control.words);

  let exact: ReadbackSnapshot | undefined = await executeCorrectnessDispatch(
    device,
    output,
    dispatches.B,
  );
  requireCompleteSnapshot(exact, `${spec.id} B`);
  requireExactComparison(control.words, exact.words, `${spec.id} A/B`);
  const exactSha256 = await sha256U32(exact.words);
  exact = undefined;

  let candidateFirst: ReadbackSnapshot | undefined =
    await executeCorrectnessDispatch(device, output, dispatches.C);
  requireCompleteSnapshot(candidateFirst, `${spec.id} C first`);
  const candidateFirstSha256 = await sha256U32(candidateFirst.words);
  const localAccumulator = createNumericalAccumulator();
  accumulateNumerics(
    localAccumulator,
    control.words,
    candidateFirst.words,
    spec,
  );
  const numerics = finalizeNumerics(localAccumulator);
  mergeNumericalAccumulators(aggregateAccumulator, localAccumulator);
  control = undefined;

  let candidateRerun: ReadbackSnapshot | undefined =
    await executeCorrectnessDispatch(device, output, dispatches.C);
  requireCompleteSnapshot(candidateRerun, `${spec.id} C rerun`);
  requireExactComparison(
    candidateFirst.words,
    candidateRerun.words,
    `${spec.id} C self-rerun`,
  );
  const candidateRerunSha256 = await sha256U32(candidateRerun.words);
  candidateFirst = undefined;
  candidateRerun = undefined;

  return Object.freeze({
    id: spec.id,
    shape: spec.shape,
    outputU32Count: output.outputElements,
    executionOrder: Object.freeze(["A", "B", "C", "C"] as const),
    executionCount: 4,
    exactComparisonCount: 2,
    aVersusBComparedU32Count: output.outputElements,
    cFirstVersusRerunComparedU32Count: output.outputElements,
    candidateVersusControlStatCount: output.outputElements,
    hashes: Object.freeze({
      A: controlSha256,
      B: exactSha256,
      CFirst: candidateFirstSha256,
      CRerun: candidateRerunSha256,
    }),
    aVersusBRawU32Exact: true,
    cFirstVersusRerunRawU32Exact: true,
    nonFiniteCount: 0,
    qNaNPrefillCount: 0,
    prefixCanaryIntact: true,
    suffixCanaryIntact: true,
    tailRowWritten: true,
    numerics,
    passed: numerics["passed"] === true,
  });
}

async function executeCorrectnessDispatch(
  device: GPUDevice,
  output: GuardedOutput,
  dispatch: AceGemmDispatch,
): Promise<ReadbackSnapshot> {
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(
    output.prefill,
    0,
    output.buffer,
    0,
    output.totalBytes,
  );
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  encoder.copyBufferToBuffer(
    output.buffer,
    0,
    output.readback,
    0,
    output.totalBytes,
  );
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await output.readback.mapAsync(GPUMapMode.READ);
  try {
    const all = new Uint32Array(output.readback.getMappedRange());
    const guardWords = STORAGE_GUARD_BYTES / 4;
    const firstOutput = guardWords;
    const lastOutput = firstOutput + output.outputElements;
    let prefixCanaryIntact = true;
    let suffixCanaryIntact = true;
    for (let index = 0; index < guardWords; index += 1) {
      prefixCanaryIntact &&= all[index] === STORAGE_GUARD_U32;
      suffixCanaryIntact &&=
        all[lastOutput + index] === STORAGE_GUARD_U32;
    }
    const words = all.slice(firstOutput, lastOutput);
    let nonFiniteCount = 0;
    let nonzeroCount = 0;
    let qNaNPrefillCount = 0;
    for (const word of words) {
      if ((word & 0x7f80_0000) === 0x7f80_0000) nonFiniteCount += 1;
      if ((word & 0x7fff_ffff) !== 0) nonzeroCount += 1;
      if (word === OUTPUT_PREFILL_QNAN_U32) qNaNPrefillCount += 1;
    }
    const tailStart = words.length - output.outputElements / ROWS;
    let tailRowWritten = true;
    for (let index = tailStart; index < words.length; index += 1) {
      if (words[index] === OUTPUT_PREFILL_QNAN_U32) {
        tailRowWritten = false;
        break;
      }
    }
    return Object.freeze({
      words,
      nonFiniteCount,
      nonzeroCount,
      qNaNPrefillCount,
      prefixCanaryIntact,
      suffixCanaryIntact,
      tailRowWritten,
    });
  } finally {
    output.readback.unmap();
  }
}

async function runTimedGate(
  prepared: PreparedGate,
): Promise<Readonly<Record<string, unknown>>> {
  const launchedAtEpochMilliseconds = Date.now();
  const thermal = parseOpt0020ThermalGate(
    collectThermalParameters(),
    prepared.preparedCompletedAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const timingInputs = prepared.shapes.map(({ spec }) => ({
    id: spec.id,
    samples: { A: [], B: [], C: [] } as Record<Opt0020Arm, number[]>,
  }));
  const dispatchSamples: Readonly<Record<string, unknown>>[] = [];
  const timingStartedAtEpochMilliseconds = Date.now();
  for (const entry of buildOpt0020TimingOrders()) {
    const shape = prepared.shapes[entry.shapeIndex];
    const input = timingInputs[entry.shapeIndex];
    if (shape === undefined || input === undefined) {
      throw new Error("OPT-0020 timing topology changed");
    }
    prepared.updateProgress(
      `timing round ${entry.roundIndex + 1}/6, shape ${entry.shapeIndex + 1}/4`,
    );
    for (const [armPosition, arm] of entry.order.entries()) {
      const sample = await executeTimedDispatch(
        prepared.device,
        shape.dispatches[arm],
      );
      input.samples[arm].push(sample.wallMilliseconds);
      dispatchSamples.push(Object.freeze({
        roundIndex: entry.roundIndex,
        rotatedShapeOrder: entry.shapeOrder,
        shapeIndex: entry.shapeIndex,
        shapeId: shape.spec.id,
        armOrder: entry.order,
        armPosition,
        arm,
        ...sample,
      }));
    }
    await yieldToBrowser();
  }
  const timingCompletedAtEpochMilliseconds = Date.now();
  await prepared.device.queue.onSubmittedWorkDone();
  await settlePostDrainEvents();
  if (prepared.uncapturedErrors.length !== 0) {
    throw new Error("OPT-0020 observed an uncaptured GPU error");
  }
  const timing = summarizeOpt0020Timing(timingInputs.map((input) =>
    Object.freeze({
      id: input.id,
      samples: Object.freeze({
        A: Object.freeze([...input.samples.A]),
        B: Object.freeze([...input.samples.B]),
        C: Object.freeze([...input.samples.C]),
      }),
    })
  ));
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
    zeroLiveResources: firstCleanup["liveBufferCount"] === 0 &&
      firstCleanup["liveBytes"] === 0 &&
      secondCleanup["liveBufferCount"] === 0 &&
      secondCleanup["liveBytes"] === 0,
  });
  if (!cleanup.zeroLiveResources) {
    throw new Error("OPT-0020 cleanup retained live GPU resources");
  }
  return Object.freeze({
    schema: "ace-opt-0020-dit-dense-cooperative-dot4-abc-v1",
    status: "passed",
    experimentId: EXPERIMENT_ID,
    classification: "bounded-reordered-rounding-primitive-decision-gate-not-integrated",
    recordedAt: new Date().toISOString(),
    identity: prepared.sourceAuthority,
    fixture: buildOpt0020FixtureDeclaration(),
    environment,
    preparation: prepared.preparation,
    protocol: Object.freeze({
      thermal,
      fullOutputCorrectnessCompletedBeforeThermalGate: true,
      allThreeKernelsCompiledAndWarmedBeforeThermalGate: true,
      timingRounds: TIMING_ROUNDS,
      timingOrder:
        "six exact A/B/C permutations; shape order rotated left by round mod 4",
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
    correctness: prepared.correctness,
    timing: Object.freeze({
      ...timing,
      timingStartedAtEpochMilliseconds,
      timingCompletedAtEpochMilliseconds,
      dispatchSampleCount: dispatchSamples.length,
      dispatchSamples: Object.freeze(dispatchSamples),
      caveat:
        "Primitive dense projections only; no package, C98, waveform, listening, or product claim.",
    }),
    decision: Object.freeze({
      disposition: timing["decision"],
      packageNativeLayerGateAuthorized: timing["passed"] === true,
      productionIntegrationAuthorized: false,
      c98TrajectoryRunAuthorized: false,
      m2250ProductRunAuthorized: false,
    }),
    memory: memoryBeforeCleanup,
    cleanup,
  });
}

async function executeTimedDispatch(
  device: GPUDevice,
  dispatch: AceGemmDispatch,
): Promise<Readonly<{
  submitAtPerformanceMilliseconds: number;
  fenceAtPerformanceMilliseconds: number;
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
  const submitAtPerformanceMilliseconds = performance.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  const fenceAtPerformanceMilliseconds = performance.now();
  return Object.freeze({
    submitAtPerformanceMilliseconds,
    fenceAtPerformanceMilliseconds,
    wallMilliseconds:
      fenceAtPerformanceMilliseconds - submitAtPerformanceMilliseconds,
    commandBufferCount: 1,
    dispatchCount: 1,
    matchingQueueCompletionFenceCount: 1,
  });
}

async function createActivationBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0020ShapeSpec,
  shapeIndex: number,
): Promise<Readonly<{ buffer: GPUBuffer; sha256: string }>> {
  const elements = spec.shape.rows * spec.shape.inner;
  const buffer = tracker.create(device, {
    label: `opt-0020-${spec.id}-activation`,
    size: elements * 4,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  let sha256: string;
  try {
    const range = buffer.getMappedRange();
    const values = new Float32Array(range);
    fillOpt0020ActivationFixture(values, shapeIndex);
    sha256 = await sha256Bytes(new Uint8Array(range));
  } finally {
    buffer.unmap();
  }
  if (sha256 !== spec.fixture.activationSha256) {
    throw new Error(`OPT-0020 ${spec.id} activation fixture hash changed`);
  }
  return Object.freeze({ buffer, sha256 });
}

async function createWeightBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0020ShapeSpec,
  shapeIndex: number,
): Promise<Readonly<{ buffer: GPUBuffer; sha256: string }>> {
  const elements = spec.shape.inner * spec.shape.columns;
  const buffer = tracker.create(device, {
    label: `opt-0020-${spec.id}-packed-weight`,
    size: elements * 2,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  let sha256: string;
  try {
    const range = buffer.getMappedRange();
    const values = new Uint16Array(range);
    fillOpt0020PackedWeightFixture(values, shapeIndex);
    sha256 = await sha256Bytes(new Uint8Array(range));
  } finally {
    buffer.unmap();
  }
  if (sha256 !== spec.fixture.packedWeightSha256) {
    throw new Error(`OPT-0020 ${spec.id} packed-weight fixture hash changed`);
  }
  return Object.freeze({ buffer, sha256 });
}

function createGuardedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0020ShapeSpec,
): GuardedOutput {
  const outputElements = spec.shape.rows * spec.shape.columns;
  const outputBytes = outputElements * 4;
  const totalBytes = outputBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = tracker.create(device, {
    label: `opt-0020-${spec.id}-output-prefill`,
    size: totalBytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  const words = new Uint32Array(prefill.getMappedRange());
  words.fill(STORAGE_GUARD_U32);
  words.fill(
    OUTPUT_PREFILL_QNAN_U32,
    STORAGE_GUARD_BYTES / 4,
    STORAGE_GUARD_BYTES / 4 + outputElements,
  );
  prefill.unmap();
  const buffer = tracker.create(device, {
    label: `opt-0020-${spec.id}-guarded-output`,
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  const readback = tracker.create(device, {
    label: `opt-0020-${spec.id}-readback`,
    size: totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  return Object.freeze({
    buffer,
    binding: Object.freeze({
      buffer,
      offset: STORAGE_GUARD_BYTES,
      size: outputBytes,
    }),
    prefill,
    readback,
    outputElements,
    outputBytes,
    totalBytes,
  });
}

async function buildSourceAuthority(): Promise<Readonly<Record<string, unknown>>> {
  const currentSourceSha256 = await sha256Text(currentCoreSource);
  const exactSourceSha256 = await sha256Text(exactCoreSource);
  const dot4SourceSha256 = await sha256Text(dot4CoreSource);
  if (currentSourceSha256 !== CURRENT_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0020 rejected unauthenticated current dense core");
  }
  if (exactSourceSha256 !== EXACT_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0020 rejected unauthenticated exact cooperative core");
  }
  if (dot4SourceSha256 !== DOT4_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0020 rejected unauthenticated dot4 cooperative core");
  }
  const generatedShaders = [];
  for (const spec of SHAPE_SPECS) {
    generatedShaders.push(Object.freeze({
      id: spec.id,
      A: await sha256Text(aceOpt0009DenseGemmWgsl(spec.shape)),
      B: await sha256Text(aceOpt0019DenseCooperativePanelsWgsl(spec.shape)),
      C: await sha256Text(aceOpt0020DenseCooperativeDot4Wgsl(spec.shape)),
    }));
  }
  return Object.freeze({
    registrationCommit: REGISTRATION_COMMIT,
    currentCoreSourceSha256: currentSourceSha256,
    exactCoreSourceSha256: exactSourceSha256,
    dot4CoreSourceSha256: dot4SourceSha256,
    exactKernelSetId: ACE_OPT_0019_DENSE_COOPERATIVE_KERNEL_SET_ID,
    dot4KernelSetId: ACE_OPT_0020_DENSE_COOPERATIVE_DOT4_KERNEL_SET_ID,
    payloadAuthority: Object.freeze({
      modelPackageLoaded: false,
      syntheticFixture: true,
      persistentWeightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
      converterOrManifestChange: false,
    }),
    generatedShaders: Object.freeze(generatedShaders),
  });
}

function buildCorrectnessStopReceipt(
  prepared: PreparedGate,
): Readonly<Record<string, unknown>> {
  const environment = environmentReceipt(prepared.adapter, prepared.device);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const firstCleanup = prepared.destroy();
  const secondCleanup = prepared.destroy();
  return Object.freeze({
    schema: "ace-opt-0020-dit-dense-cooperative-dot4-abc-v1",
    status: "passed",
    experimentId: EXPERIMENT_ID,
    classification: "bounded-reordered-rounding-primitive-correctness-stop",
    recordedAt: new Date().toISOString(),
    identity: prepared.sourceAuthority,
    fixture: buildOpt0020FixtureDeclaration(),
    environment,
    preparation: prepared.preparation,
    protocol: Object.freeze({
      fullOutputCorrectnessCompleted: true,
      timingSkippedBecauseNumericalGateFailed: true,
      unchangedThermalRetryPerformed: false,
    }),
    correctness: prepared.correctness,
    decision: Object.freeze({
      disposition: "negative-stop-numerical-gate",
      packageNativeLayerGateAuthorized: false,
      productionIntegrationAuthorized: false,
      c98TrajectoryRunAuthorized: false,
      m2250ProductRunAuthorized: false,
    }),
    memory: memoryBeforeCleanup,
    cleanup: Object.freeze({
      firstCall: firstCleanup,
      secondCall: secondCleanup,
      idempotent: true,
      zeroLiveResources: firstCleanup["liveBufferCount"] === 0 &&
        secondCleanup["liveBufferCount"] === 0,
    }),
  });
}

function requireAdapter(adapter: GPUAdapter): void {
  const info = adapter.info;
  const largestBindingBytes = maximumBindingBytes();
  if (!adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    info.subgroupMinSize !== 32 || info.subgroupMaxSize !== 32 ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 256 ||
    adapter.limits.maxComputeWorkgroupSizeX < 128 ||
    adapter.limits.maxComputeWorkgroupSizeY < 16 ||
    adapter.limits.maxComputeWorkgroupStorageSize <
      ACE_OPT_0020_DENSE_WORKGROUP_STORAGE_BYTES ||
    adapter.limits.maxStorageBufferBindingSize < largestBindingBytes ||
    adapter.limits.maxBufferSize < largestBindingBytes + 2 * STORAGE_GUARD_BYTES) {
    throw new Error(
      "OPT-0020 requires the authenticated fixed32 subgroup/WG256/6528-byte contract",
    );
  }
}

function environmentReceipt(
  adapter: GPUAdapter,
  device: GPUDevice,
): Readonly<Record<string, unknown>> {
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
    features: Object.freeze([...device.features].sort()),
    limits: Object.freeze({
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: device.limits.maxComputeWorkgroupSizeY,
      maxComputeWorkgroupStorageSize:
        device.limits.maxComputeWorkgroupStorageSize,
    }),
  });
}

function compactPlan(plan: Readonly<{
  rows: number;
  inner: number;
  columns: number;
  tileRows: number;
  tileColumns: number;
  tileInner: number;
  workgroupSize: number;
  rowTiles: number;
  columnTiles: number;
  innerTiles: number;
  workgroupCount: number;
  outputRanges: readonly Readonly<{ multiplyAdds: number }>[];
}>): Readonly<Record<string, unknown>> {
  return Object.freeze({
    rows: plan.rows,
    inner: plan.inner,
    columns: plan.columns,
    tileRows: plan.tileRows,
    tileColumns: plan.tileColumns,
    tileInner: plan.tileInner,
    workgroupSize: plan.workgroupSize,
    rowTiles: plan.rowTiles,
    columnTiles: plan.columnTiles,
    innerTiles: plan.innerTiles,
    workgroupCount: plan.workgroupCount,
    scheduledMultiplyAdds: plan.outputRanges.reduce(
      (sum, range) => sum + range.multiplyAdds,
      0,
    ),
  });
}

function requireCompleteSnapshot(snapshot: ReadbackSnapshot, label: string): void {
  if (snapshot.nonFiniteCount !== 0 || snapshot.nonzeroCount === 0 ||
    snapshot.qNaNPrefillCount !== 0 || !snapshot.prefixCanaryIntact ||
    !snapshot.suffixCanaryIntact || !snapshot.tailRowWritten) {
    throw new Error(`${label} failed complete-write, finite, tail, or canary gate`);
  }
}

function requireExactComparison(
  expected: Uint32Array,
  actual: Uint32Array,
  label: string,
): void {
  if (expected.length !== actual.length) {
    throw new Error(`${label} output length changed`);
  }
  for (let index = 0; index < expected.length; index += 1) {
    if (expected[index] !== actual[index]) {
      throw new Error(
        `${label} raw-U32 mismatch at ${index}: ` +
          `${expected[index]?.toString(16)} != ${actual[index]?.toString(16)}`,
      );
    }
  }
}

function shapeSpec(
  id: Opt0020ShapeSpec["id"],
  inner: number,
  columns: number,
  productionMultiplicity: Opt0020ShapeSpec["productionMultiplicity"],
  feedForwardMultiplicity: Opt0020ShapeSpec["feedForwardMultiplicity"],
  ordinal: 0 | 1 | 2 | 3,
  activationSha256: string,
  packedWeightSha256: string,
): Opt0020ShapeSpec {
  return Object.freeze({
    id,
    shape: Object.freeze({ rows: ROWS, inner, columns }),
    productionMultiplicity,
    feedForwardMultiplicity,
    fixture: Object.freeze({
      ordinal,
      activationSeed: ACTIVATION_SEEDS[ordinal],
      weightSeed: WEIGHT_SEEDS[ordinal],
      activationSha256,
      packedWeightSha256,
    }),
  });
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function maximumBindingBytes(): number {
  return Math.max(...SHAPE_SPECS.flatMap(({ shape }) => [
    shape.rows * shape.inner * 4,
    shape.inner * shape.columns * 2,
    shape.rows * shape.columns * 4,
  ]));
}

function median6(samples: readonly number[]): number {
  if (samples.length !== TIMING_ROUNDS ||
    samples.some((sample) => !Number.isFinite(sample) || sample <= 0)) {
    throw new Error("OPT-0020 requires six finite positive samples per arm");
  }
  const sorted = [...samples].sort((left, right) => left - right);
  return (sorted[2]! + sorted[3]!) / 2;
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
    throw new Error(`OPT-0020 thermal field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0020 thermal field ${name} is not finite`);
  }
  return value;
}

function requireLittleEndianHost(): void {
  const words = new Uint16Array([0x0102]);
  if (new Uint8Array(words.buffer)[0] !== 0x02) {
    throw new Error("OPT-0020 fixture hashes require a little-endian host");
  }
}

async function sha256Text(value: string): Promise<string> {
  return sha256Bytes(new TextEncoder().encode(value));
}

async function sha256U32(values: Uint32Array): Promise<string> {
  return sha256Bytes(new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  ));
}

async function sha256Bytes(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", Uint8Array.from(bytes));
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function requireElement<ElementType extends Element>(selector: string): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing browser element ${selector}`);
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
    __ACE_OPT0020_RESULT__?: Readonly<Record<string, unknown>>;
  }).__ACE_OPT0020_RESULT__ = receipt;
}

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0020-dit-dense-cooperative-dot4-abc-v1",
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
