/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import currentCoreSource from
  "../../src/webgpu/kernels/dit-dense-fp16.ts?raw";
import exactCoreSource from
  "../../src/webgpu/kernels/dit-dense-fp16-cooperative-panels.ts?raw";
import vec4CoreSource from
  "../../src/webgpu/kernels/dit-dense-fp16-cooperative-vec4-panels.ts?raw";
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
  ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID,
  ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES,
  AceOpt0021DenseCooperativeVec4PanelsKernel,
  aceOpt0021DenseCooperativeVec4PanelsWgsl,
  planAceOpt0021DenseCooperativeVec4Panels,
} from "../../src/webgpu/kernels/dit-dense-fp16-cooperative-vec4-panels.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";

export type Opt0021Arm = "A" | "B" | "C";

export interface Opt0021ShapeSpec {
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

export interface Opt0021TimingInput {
  readonly id: Opt0021ShapeSpec["id"];
  readonly samples: Readonly<Record<Opt0021Arm, readonly number[]>>;
}

export interface Opt0021ThermalGate {
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
  readonly spec: Opt0021ShapeSpec;
  readonly activation: GPUBuffer;
  readonly weight: GPUBuffer;
  readonly output: GPUBuffer;
  readonly dispatches: Readonly<Record<Opt0021Arm, AceGemmDispatch>>;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly currentKernel: AceOpt0009DenseGemmKernel;
  readonly exactKernel: AceOpt0019DenseCooperativePanelsKernel;
  readonly vec4Kernel: AceOpt0021DenseCooperativeVec4PanelsKernel;
  readonly shapes: readonly PreparedShape[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly sourceAuthority: Readonly<Record<string, unknown>>;
  readonly uncapturedErrors: readonly string[];
  readonly preparation: Readonly<Record<string, unknown>>;
  readonly preparedCompletedAtEpochMilliseconds: number;
  readonly updateProgress: (message: string) => void;
  destroy(): Readonly<Record<string, unknown>>;
}

const EXPERIMENT_ID = "OPT-0021" as const;
const REGISTRATION_COMMIT =
  "7dcbe50c7f04d1e07f6b30657da96372ca8574d1" as const;
const CURRENT_CORE_SOURCE_SHA256 =
  "a238f67da07c6ba1097da9d9e9e97960ae97d2e1d5c129fcbabf69e962cbb6b3" as const;
const EXACT_CORE_SOURCE_SHA256 =
  "b5dad12724882d3fc942c7df7b10c7b7b89a4bed595125ff11a5905c03152a37" as const;
const VEC4_CORE_SOURCE_SHA256 =
  "2229c55f8b7fe66d3770ef7683de68322632d17749fe2d3085d5d46dcdc22df1" as const;
const FIXTURE_MANIFEST_PROVENANCE_SHA256 =
  "cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb" as const;
const SYNTHETIC_FIXTURE_DECLARATION_SHA256 =
  "954aff0a07dcc2946ac8191c054e2ecf63473d05ed9ebf81dc7db6d535f80f0c" as const;
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
const B_TO_C_COMPLETE_SPEEDUP_THRESHOLD = 1.075;
const A_TO_C_COMPLETE_SAVING_THRESHOLD_MILLISECONDS = 52.0834;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_2155;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const THERMAL_POLL_MILLISECONDS = 1_000;
const MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS = 1_250;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 5_000;
const FULL_OUTPUT_U32_COUNT = 25_344_000;
const FULL_EXACT_COMPARED_U32_COUNT = 3 * FULL_OUTPUT_U32_COUNT;
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

export function buildOpt0021ShapeSpecs(): readonly Opt0021ShapeSpec[] {
  return SHAPE_SPECS;
}

export function buildOpt0021FixtureDeclaration(): Readonly<Record<string, unknown>> {
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

export function opt0021ActivationBitsAt(
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

export function opt0021PackedWeightBitsAt(
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

export function fillOpt0021ActivationFixture(
  values: Float32Array,
  shapeIndex: number,
): void {
  const spec = requireShapeIndex(shapeIndex);
  const expected = spec.shape.rows * spec.shape.inner;
  if (values.length !== expected) {
    throw new RangeError(`OPT-0021 activation fixture requires ${expected} F32 values`);
  }
  let index = 0;
  for (let row = 0; row < spec.shape.rows; row += 1) {
    for (let inner = 0; inner < spec.shape.inner; inner += 1) {
      values[index] = finiteNormalHalfToNumber(
        opt0021ActivationBitsAt(shapeIndex, row, inner),
      );
      index += 1;
    }
  }
}

export function fillOpt0021PackedWeightFixture(
  values: Uint16Array,
  shapeIndex: number,
): void {
  const spec = requireShapeIndex(shapeIndex);
  const expected = spec.shape.inner * spec.shape.columns;
  if (values.length !== expected) {
    throw new RangeError(`OPT-0021 weight fixture requires ${expected} U16 values`);
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
          values[physical] = opt0021PackedWeightBitsAt(
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

export function buildOpt0021TimingOrders(): readonly Readonly<{
  roundIndex: number;
  shapeIndex: number;
  shapeOrder: readonly number[];
  order: readonly Opt0021Arm[];
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

export function summarizeOpt0021Timing(
  inputs: readonly Opt0021TimingInput[],
): Readonly<Record<string, unknown>> {
  if (inputs.length !== SHAPE_SPECS.length) {
    throw new Error("OPT-0021 requires all four exact-shape timings");
  }
  const complete = { A: 0, B: 0, C: 0 };
  const feedForward = { A: 0, B: 0, C: 0 };
  const strata = inputs.map((input, index) => {
    const spec = SHAPE_SPECS[index];
    if (spec === undefined || input.id !== spec.id) {
      throw new Error("OPT-0021 timing shape order changed");
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
      aToCSavingMilliseconds: medians.A - medians.C,
    });
  });
  const everyShapeCFasterThanB = strata.every((item) => item.cFasterThanB);
  const bToCCompleteSpeedup = complete.B / complete.C;
  const aToCCompleteSavingMilliseconds = complete.A - complete.C;
  const passed = everyShapeCFasterThanB &&
    bToCCompleteSpeedup >= B_TO_C_COMPLETE_SPEEDUP_THRESHOLD &&
    aToCCompleteSavingMilliseconds >=
      A_TO_C_COMPLETE_SAVING_THRESHOLD_MILLISECONDS;
  return Object.freeze({
    samplesPerArmPerShape: TIMING_ROUNDS,
    completeDense: Object.freeze({
      multiplicities: "4/2/2/1",
      milliseconds: Object.freeze(complete),
      bToCSpeedup: bToCCompleteSpeedup,
      bToCThreshold: B_TO_C_COMPLETE_SPEEDUP_THRESHOLD,
      aToCSavingMilliseconds: aToCCompleteSavingMilliseconds,
      aToCSavingThresholdMilliseconds:
        A_TO_C_COMPLETE_SAVING_THRESHOLD_MILLISECONDS,
    }),
    feedForward: Object.freeze({
      multiplicities: "0/0/2/1",
      milliseconds: Object.freeze(feedForward),
      bToCSpeedup: feedForward.B / feedForward.C,
      aToCSavingMilliseconds: feedForward.A - feedForward.C,
    }),
    everyShapeCFasterThanB,
    strata: Object.freeze(strata),
    passed,
    decision: passed
      ? "positive-package-layer-gate-authorized"
      : "negative-stop-primitive-gate",
  });
}

export function parseOpt0021ThermalGate(
  parameters: URLSearchParams,
  preparedCompletedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0021ThermalGate {
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
    !Number.isSafeInteger(observationCount) || observationCount < 31 ||
    pollMilliseconds !== THERMAL_POLL_MILLISECONDS ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS ||
    maximumPollGapMilliseconds <= 0 ||
    nonNominalObservationCount !== 0 ||
    durationMilliseconds < MINIMUM_NOMINAL_MILLISECONDS ||
    startedAtEpochMilliseconds < preparedCompletedAtEpochMilliseconds ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS) {
    throw new Error("OPT-0021 thermal gate is incomplete, stale, or non-nominal");
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

export function compareOpt0021ExactWords(
  expected: Uint32Array,
  actual: Uint32Array,
): Readonly<Record<string, unknown>> {
  if (expected.length !== actual.length) {
    throw new RangeError("OPT-0021 exact comparison length changed");
  }
  let mismatchCount = 0;
  let firstMismatch: Readonly<Record<string, number>> | null = null;
  for (let index = 0; index < expected.length; index += 1) {
    const expectedWord = expected[index]!;
    const actualWord = actual[index]!;
    if (expectedWord === actualWord) continue;
    mismatchCount += 1;
    firstMismatch ??= Object.freeze({ index, expectedWord, actualWord });
  }
  return Object.freeze({
    comparedU32Count: expected.length,
    mismatchCount,
    firstMismatch,
    rawU32Exact: mismatchCount === 0,
  });
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
    throw new RangeError("OPT-0021 fixture requires finite normal FP16 bits");
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

function requireShapeIndex(index: number): Opt0021ShapeSpec {
  if (!Number.isSafeInteger(index)) {
    throw new RangeError("OPT-0021 fixture shape index must be a safe integer");
  }
  const spec = SHAPE_SPECS[index];
  if (spec === undefined) {
    throw new RangeError(`OPT-0021 fixture rejects shape index ${index}`);
  }
  return spec;
}

function requireFixtureCoordinate(
  value: number,
  extent: number,
  label: string,
): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= extent) {
    throw new RangeError(`OPT-0021 ${label} ${value} is out of bounds`);
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
    label: "ace-opt-0021-dense-vec4-panels-abc-device",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: {
      maxBufferSize: largestBindingBytes + 2 * STORAGE_GUARD_BYTES,
      maxStorageBufferBindingSize: largestBindingBytes,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 128,
      maxComputeWorkgroupSizeY: 16,
      maxComputeWorkgroupStorageSize:
        ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES,
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
  const vec4Kernel = AceOpt0021DenseCooperativeVec4PanelsKernel.create(device);
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
    vec4Kernel.destroy();
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
    for (const [index, spec] of SHAPE_SPECS.entries()) {
      updateProgress(`full-output A/B/C/C exactness ${index + 1}/4: ${spec.id}`);
      shapes.push(await prepareShape(
        device,
        tracker,
        currentKernel,
        exactKernel,
        vec4Kernel,
        spec,
        index,
      ));
      await yieldToBrowser();
    }
    const correctnessMilliseconds = performance.now() - correctnessStarted;
    const correctnessCases = shapes.map((shape) => shape.correctness);
    const comparedU32Count = correctnessCases.reduce(
      (sum, item) => sum + Number(item["comparedU32Count"]),
      0,
    );
    const mismatchCount = correctnessCases.reduce(
      (sum, item) => sum + Number(item["mismatchCount"]),
      0,
    );
    const aggregateOutputHashManifestSha256 =
      await buildAggregateOutputHashManifest(correctnessCases);
    if (comparedU32Count !== FULL_EXACT_COMPARED_U32_COUNT) {
      throw new Error("OPT-0021 aggregate exact comparison count changed");
    }
    const primitiveCorrectnessPassed = mismatchCount === 0 &&
      correctnessCases.every((item) => item["passed"] === true);
    let warmupMilliseconds = 0;
    if (primitiveCorrectnessPassed) {
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
    const correctnessPassed = primitiveCorrectnessPassed &&
      uncapturedErrors.length === 0;
    const aVersusBRawU32Exact = correctnessCases.every(
      (item) => item["aVersusBRawU32Exact"] === true,
    );
    const aVersusCFirstRawU32Exact = correctnessCases.every(
      (item) => item["aVersusCFirstRawU32Exact"] === true,
    );
    const cFirstVersusRerunRawU32Exact = correctnessCases.every(
      (item) => item["cFirstVersusRerunRawU32Exact"] === true,
    );
    const allFourExecutionsShareShapeHash = correctnessCases.every(
      (item) => item["allFourExecutionsShareShapeHash"] === true,
    );
    const qNaNPrefillCompleteWrites = correctnessCases.every(
      (item) => Number(item["qNaNPrefillCount"]) === 0,
    );
    const canariesUntouched = correctnessCases.every(
      (item) => item["prefixCanaryIntact"] === true &&
        item["suffixCanaryIntact"] === true,
    );
    const finiteOutputs = correctnessCases.every(
      (item) => Number(item["nonFiniteCount"]) === 0,
    );
    const tailRowsWritten = correctnessCases.every(
      (item) => item["tailRowWritten"] === true,
    );
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
      vec4Kernel,
      shapes: Object.freeze(shapes),
      correctness: Object.freeze({
        shapeCount: shapes.length,
        executionCount: shapes.length * 4,
        exactComparisonCount: shapes.length * 3,
        comparisonsPerOutputWord: 3,
        outputU32Count: FULL_OUTPUT_U32_COUNT,
        comparedU32Count,
        mismatchCount,
        aVersusBRawU32Exact,
        aVersusCFirstRawU32Exact,
        cFirstVersusRerunRawU32Exact,
        allFourExecutionsShareShapeHash,
        qNaNPrefillCompleteWrites,
        canariesUntouched,
        finiteOutputs,
        tailRowsWritten,
        uncapturedGpuErrorCount: uncapturedErrors.length,
        aggregateOutputHashManifestSha256,
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
    allFourExecutionsShareAggregateHash:
      hashes.A === hashes.B && hashes.A === hashes.CFirst &&
      hashes.A === hashes.CRerun,
  });
}

async function prepareShape(
  device: GPUDevice,
  tracker: BufferTracker,
  currentKernel: AceOpt0009DenseGemmKernel,
  exactKernel: AceOpt0019DenseCooperativePanelsKernel,
  vec4Kernel: AceOpt0021DenseCooperativeVec4PanelsKernel,
  spec: Opt0021ShapeSpec,
  shapeIndex: number,
): Promise<PreparedShape> {
  const currentPlan = planAceOpt0009DenseGemm(spec.shape);
  const exactPlan = planAceOpt0019DenseCooperativePanels(spec.shape);
  const vec4Plan = planAceOpt0021DenseCooperativeVec4Panels(spec.shape);
  if (currentPlan.tileRows !== CURRENT_TILE_ROWS ||
    currentPlan.tileColumns !== CURRENT_TILE_COLUMNS ||
    currentPlan.tileInner !== CURRENT_TILE_INNER ||
    exactPlan.tileRows !== COOPERATIVE_TILE_ROWS ||
    exactPlan.tileColumns !== COOPERATIVE_TILE_COLUMNS ||
    exactPlan.tileInner !== COOPERATIVE_TILE_INNER ||
    vec4Plan.tileRows !== COOPERATIVE_TILE_ROWS ||
    vec4Plan.tileColumns !== COOPERATIVE_TILE_COLUMNS ||
    vec4Plan.tileInner !== COOPERATIVE_TILE_INNER ||
    exactPlan.workgroupCount !== vec4Plan.workgroupCount ||
    exactPlan.scheduledMultiplyAdds !== vec4Plan.scheduledMultiplyAdds ||
    exactPlan.validMultiplyAdds !== vec4Plan.validMultiplyAdds ||
    exactPlan.estimatedGlobalActivationBytes !==
      vec4Plan.estimatedGlobalActivationBytes ||
    exactPlan.estimatedGlobalWeightBytes !==
      vec4Plan.estimatedGlobalWeightBytes ||
    exactPlan.estimatedGlobalOperandBytes !==
      vec4Plan.estimatedGlobalOperandBytes ||
    exactPlan.estimatedGlobalOutputBytes !==
      vec4Plan.estimatedGlobalOutputBytes ||
    exactPlan.barrierEvents !== vec4Plan.barrierEvents ||
    exactPlan.workgroupStorageBytes !==
      ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES ||
    vec4Plan.workgroupStorageBytes !==
      ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES ||
    JSON.stringify(exactPlan.packedWeightStorageShape) !==
      JSON.stringify(vec4Plan.packedWeightStorageShape)) {
    throw new Error(`OPT-0021 ${spec.id} frozen three-arm topology changed`);
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
    `opt-0021-${spec.id}-A-current-opt-0009`,
    spec.shape,
    bindings,
  );
  const B = await exactKernel.createDispatch(
    `opt-0021-${spec.id}-B-exact-opt-0019`,
    spec.shape,
    bindings,
  );
  const C = await vec4Kernel.createDispatch(
    `opt-0021-${spec.id}-C-vec4-panels`,
    spec.shape,
    bindings,
  );
  if (A.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT ||
    B.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT ||
    C.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT) {
    throw new Error(`OPT-0021 ${spec.id} package weight-layout identity changed`);
  }
  const correctness = await verifyCompleteShape(
    device,
    spec,
    guarded,
    Object.freeze({ A, B, C }),
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
        C: compactPlan(vec4Plan),
      }),
    }),
  });
}

async function verifyCompleteShape(
  device: GPUDevice,
  spec: Opt0021ShapeSpec,
  output: GuardedOutput,
  dispatches: Readonly<Record<Opt0021Arm, AceGemmDispatch>>,
): Promise<Readonly<Record<string, unknown>>> {
  let control: ReadbackSnapshot | undefined = await executeCorrectnessDispatch(
    device,
    output,
    dispatches.A,
  );
  const controlComplete = completeSnapshot(control);
  const controlSha256 = await sha256U32(control.words);

  let exact: ReadbackSnapshot | undefined = await executeCorrectnessDispatch(
    device,
    output,
    dispatches.B,
  );
  const exactComplete = completeSnapshot(exact);
  const aVersusB = compareExactWords(
    control.words,
    exact.words,
    `${spec.id} A/B`,
  );
  const exactSha256 = await sha256U32(exact.words);
  exact = undefined;

  let candidateFirst: ReadbackSnapshot | undefined =
    await executeCorrectnessDispatch(device, output, dispatches.C);
  const candidateFirstComplete = completeSnapshot(candidateFirst);
  const aVersusCFirst = compareExactWords(
    control.words,
    candidateFirst.words,
    `${spec.id} A/C-first`,
  );
  const candidateFirstSha256 = await sha256U32(candidateFirst.words);
  control = undefined;

  let candidateRerun: ReadbackSnapshot | undefined =
    await executeCorrectnessDispatch(device, output, dispatches.C);
  const candidateRerunComplete = completeSnapshot(candidateRerun);
  const cFirstVersusRerun = compareExactWords(
    candidateFirst.words,
    candidateRerun.words,
    `${spec.id} C-first/C-rerun`,
  );
  const candidateRerunSha256 = await sha256U32(candidateRerun.words);
  candidateFirst = undefined;
  candidateRerun = undefined;

  const allFourExecutionsShareShapeHash = controlSha256 === exactSha256 &&
    controlSha256 === candidateFirstSha256 &&
    controlSha256 === candidateRerunSha256;
  const comparedU32Count = output.outputElements * 3;
  const mismatchCount = Number(aVersusB["mismatchCount"]) +
    Number(aVersusCFirst["mismatchCount"]) +
    Number(cFirstVersusRerun["mismatchCount"]);
  const nonFiniteCount = controlComplete.nonFiniteCount +
    exactComplete.nonFiniteCount + candidateFirstComplete.nonFiniteCount +
    candidateRerunComplete.nonFiniteCount;
  const qNaNPrefillCount = controlComplete.qNaNPrefillCount +
    exactComplete.qNaNPrefillCount + candidateFirstComplete.qNaNPrefillCount +
    candidateRerunComplete.qNaNPrefillCount;
  const prefixCanaryIntact = controlComplete.prefixCanaryIntact &&
    exactComplete.prefixCanaryIntact && candidateFirstComplete.prefixCanaryIntact &&
    candidateRerunComplete.prefixCanaryIntact;
  const suffixCanaryIntact = controlComplete.suffixCanaryIntact &&
    exactComplete.suffixCanaryIntact && candidateFirstComplete.suffixCanaryIntact &&
    candidateRerunComplete.suffixCanaryIntact;
  const tailRowWritten = controlComplete.tailRowWritten &&
    exactComplete.tailRowWritten && candidateFirstComplete.tailRowWritten &&
    candidateRerunComplete.tailRowWritten;
  const finiteCompleteWrites = controlComplete.passed && exactComplete.passed &&
    candidateFirstComplete.passed && candidateRerunComplete.passed;
  return Object.freeze({
    id: spec.id,
    shape: spec.shape,
    outputU32Count: output.outputElements,
    executionOrder: Object.freeze(["A", "B", "C", "C"] as const),
    executionCount: 4,
    exactComparisonCount: 3,
    comparisonsPerOutputWord: 3,
    comparedU32Count,
    mismatchCount,
    hashes: Object.freeze({
      A: controlSha256,
      B: exactSha256,
      CFirst: candidateFirstSha256,
      CRerun: candidateRerunSha256,
    }),
    comparisons: Object.freeze({
      aVersusB,
      aVersusCFirst,
      cFirstVersusRerun,
    }),
    aVersusBRawU32Exact: aVersusB["rawU32Exact"],
    aVersusCFirstRawU32Exact: aVersusCFirst["rawU32Exact"],
    cFirstVersusRerunRawU32Exact: cFirstVersusRerun["rawU32Exact"],
    allFourExecutionsShareShapeHash,
    nonFiniteCount,
    qNaNPrefillCount,
    prefixCanaryIntact,
    suffixCanaryIntact,
    tailRowWritten,
    finiteCompleteWrites,
    passed: mismatchCount === 0 && allFourExecutionsShareShapeHash &&
      finiteCompleteWrites,
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
  const thermal = parseOpt0021ThermalGate(
    collectThermalParameters(),
    prepared.preparedCompletedAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const timingInputs = prepared.shapes.map(({ spec }) => ({
    id: spec.id,
    samples: { A: [], B: [], C: [] } as Record<Opt0021Arm, number[]>,
  }));
  const dispatchSamples: Readonly<Record<string, unknown>>[] = [];
  const timingStartedAtEpochMilliseconds = Date.now();
  for (const entry of buildOpt0021TimingOrders()) {
    const shape = prepared.shapes[entry.shapeIndex];
    const input = timingInputs[entry.shapeIndex];
    if (shape === undefined || input === undefined) {
      throw new Error("OPT-0021 timing topology changed");
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
    throw new Error("OPT-0021 observed an uncaptured GPU error");
  }
  const timing = summarizeOpt0021Timing(timingInputs.map((input) =>
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
    throw new Error("OPT-0021 cleanup retained live GPU resources");
  }
  return Object.freeze({
    schema: "ace-opt-0021-dit-dense-vec4-panels-abc-v1",
    status: "passed",
    experimentId: EXPERIMENT_ID,
    classification: "bounded-exact-primitive-decision-gate-not-integrated",
    recordedAt: new Date().toISOString(),
    identity: prepared.sourceAuthority,
    fixture: buildOpt0021FixtureDeclaration(),
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
        "Primitive dense projections only; no package, C98, VAE, waveform, listening, or product claim.",
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
  spec: Opt0021ShapeSpec,
  shapeIndex: number,
): Promise<Readonly<{ buffer: GPUBuffer; sha256: string }>> {
  const elements = spec.shape.rows * spec.shape.inner;
  const buffer = tracker.create(device, {
    label: `opt-0021-${spec.id}-activation`,
    size: elements * 4,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  let sha256: string;
  try {
    const range = buffer.getMappedRange();
    const values = new Float32Array(range);
    fillOpt0021ActivationFixture(values, shapeIndex);
    sha256 = await sha256Bytes(new Uint8Array(range));
  } finally {
    buffer.unmap();
  }
  if (sha256 !== spec.fixture.activationSha256) {
    throw new Error(`OPT-0021 ${spec.id} activation fixture hash changed`);
  }
  return Object.freeze({ buffer, sha256 });
}

async function createWeightBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0021ShapeSpec,
  shapeIndex: number,
): Promise<Readonly<{ buffer: GPUBuffer; sha256: string }>> {
  const elements = spec.shape.inner * spec.shape.columns;
  const buffer = tracker.create(device, {
    label: `opt-0021-${spec.id}-packed-weight`,
    size: elements * 2,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  let sha256: string;
  try {
    const range = buffer.getMappedRange();
    const values = new Uint16Array(range);
    fillOpt0021PackedWeightFixture(values, shapeIndex);
    sha256 = await sha256Bytes(new Uint8Array(range));
  } finally {
    buffer.unmap();
  }
  if (sha256 !== spec.fixture.packedWeightSha256) {
    throw new Error(`OPT-0021 ${spec.id} packed-weight fixture hash changed`);
  }
  return Object.freeze({ buffer, sha256 });
}

function createGuardedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0021ShapeSpec,
): GuardedOutput {
  const outputElements = spec.shape.rows * spec.shape.columns;
  const outputBytes = outputElements * 4;
  const totalBytes = outputBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = tracker.create(device, {
    label: `opt-0021-${spec.id}-output-prefill`,
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
    label: `opt-0021-${spec.id}-guarded-output`,
    size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  const readback = tracker.create(device, {
    label: `opt-0021-${spec.id}-readback`,
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
  const vec4SourceSha256 = await sha256Text(vec4CoreSource);
  const syntheticFixtureDeclarationSha256 = await sha256Text(
    canonicalJson(buildOpt0021FixtureDeclaration()),
  );
  if (currentSourceSha256 !== CURRENT_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0021 rejected unauthenticated current dense core");
  }
  if (exactSourceSha256 !== EXACT_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0021 rejected unauthenticated exact cooperative core");
  }
  if (vec4SourceSha256 !== VEC4_CORE_SOURCE_SHA256) {
    throw new Error("OPT-0021 rejected unauthenticated vec4 cooperative core");
  }
  if (syntheticFixtureDeclarationSha256 !==
    SYNTHETIC_FIXTURE_DECLARATION_SHA256) {
    throw new Error("OPT-0021 rejected changed synthetic fixture declaration");
  }
  const generatedShaders = [];
  for (const spec of SHAPE_SPECS) {
    generatedShaders.push(Object.freeze({
      id: spec.id,
      A: await sha256Text(aceOpt0009DenseGemmWgsl(spec.shape)),
      B: await sha256Text(aceOpt0019DenseCooperativePanelsWgsl(spec.shape)),
      C: await sha256Text(
        aceOpt0021DenseCooperativeVec4PanelsWgsl(spec.shape),
      ),
    }));
  }
  return Object.freeze({
    registrationCommit: REGISTRATION_COMMIT,
    currentCoreSourceSha256: currentSourceSha256,
    exactCoreSourceSha256: exactSourceSha256,
    vec4CoreSourceSha256: vec4SourceSha256,
    exactKernelSetId: ACE_OPT_0019_DENSE_COOPERATIVE_KERNEL_SET_ID,
    vec4KernelSetId:
      ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID,
    fixtureManifestProvenanceSha256: FIXTURE_MANIFEST_PROVENANCE_SHA256,
    syntheticFixtureDeclarationSha256,
    syntheticFixtureDeclarationHashAlgorithm:
      "sha256-canonical-json-recursive-sorted-object-keys-v1",
    payloadAuthority: Object.freeze({
      modelPackageLoaded: false,
      syntheticFixture: true,
      persistentWeightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
      converterOrManifestChange: false,
      productionRuntimeSelected: false,
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
    schema: "ace-opt-0021-dit-dense-vec4-panels-abc-v1",
    status: "passed",
    experimentId: EXPERIMENT_ID,
    classification: "bounded-exact-primitive-correctness-stop",
    recordedAt: new Date().toISOString(),
    identity: prepared.sourceAuthority,
    fixture: buildOpt0021FixtureDeclaration(),
    environment,
    preparation: prepared.preparation,
    protocol: Object.freeze({
      fullOutputCorrectnessCompleted: true,
      timingSkippedBecauseCorrectnessOrLifecycleGateFailed: true,
      unchangedThermalRetryPerformed: false,
    }),
    correctness: prepared.correctness,
    decision: Object.freeze({
      disposition: "negative-stop-exactness-gate",
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
        firstCleanup["liveBytes"] === 0 &&
        secondCleanup["liveBufferCount"] === 0 &&
        secondCleanup["liveBytes"] === 0,
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
      ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES ||
    adapter.limits.maxStorageBufferBindingSize < largestBindingBytes ||
    adapter.limits.maxBufferSize < largestBindingBytes + 2 * STORAGE_GUARD_BYTES) {
    throw new Error(
      "OPT-0021 requires the authenticated fixed32 subgroup/WG256/6400-byte contract",
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

function completeSnapshot(snapshot: ReadbackSnapshot): Readonly<{
  nonFiniteCount: number;
  qNaNPrefillCount: number;
  prefixCanaryIntact: boolean;
  suffixCanaryIntact: boolean;
  tailRowWritten: boolean;
  passed: boolean;
}> {
  return Object.freeze({
    nonFiniteCount: snapshot.nonFiniteCount,
    qNaNPrefillCount: snapshot.qNaNPrefillCount,
    prefixCanaryIntact: snapshot.prefixCanaryIntact,
    suffixCanaryIntact: snapshot.suffixCanaryIntact,
    tailRowWritten: snapshot.tailRowWritten,
    passed: snapshot.nonFiniteCount === 0 && snapshot.nonzeroCount !== 0 &&
      snapshot.qNaNPrefillCount === 0 && snapshot.prefixCanaryIntact &&
      snapshot.suffixCanaryIntact && snapshot.tailRowWritten,
  });
}

function compareExactWords(
  expected: Uint32Array,
  actual: Uint32Array,
  label: string,
): Readonly<Record<string, unknown>> {
  const comparison = compareOpt0021ExactWords(expected, actual);
  return Object.freeze({ label, ...comparison });
}

function shapeSpec(
  id: Opt0021ShapeSpec["id"],
  inner: number,
  columns: number,
  productionMultiplicity: Opt0021ShapeSpec["productionMultiplicity"],
  feedForwardMultiplicity: Opt0021ShapeSpec["feedForwardMultiplicity"],
  ordinal: 0 | 1 | 2 | 3,
  activationSha256: string,
  packedWeightSha256: string,
): Opt0021ShapeSpec {
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
    throw new Error("OPT-0021 requires six finite positive samples per arm");
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
    throw new Error(`OPT-0021 thermal field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0021 thermal field ${name} is not finite`);
  }
  return value;
}

function requireLittleEndianHost(): void {
  const words = new Uint16Array([0x0102]);
  if (new Uint8Array(words.buffer)[0] !== 0x02) {
    throw new Error("OPT-0021 fixture hashes require a little-endian host");
  }
}

async function sha256Text(value: string): Promise<string> {
  return sha256Bytes(new TextEncoder().encode(value));
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(sortJsonValue(value));
}

function sortJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortJsonValue);
  if (value !== null && typeof value === "object") {
    const record = value as Readonly<Record<string, unknown>>;
    return Object.fromEntries(Object.keys(record).sort().map((key) =>
      [key, sortJsonValue(record[key])]
    ));
  }
  return value;
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
    __ACE_OPT0021_RESULT__?: Readonly<Record<string, unknown>>;
  }).__ACE_OPT0021_RESULT__ = receipt;
}

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: "ace-opt-0021-dit-dense-vec4-panels-abc-v1",
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
