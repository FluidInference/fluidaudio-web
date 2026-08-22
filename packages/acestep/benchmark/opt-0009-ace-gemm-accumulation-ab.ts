/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import aceSubgroupGemmSource from
  "../src/webgpu/kernels/subgroup-gemm.ts?raw";

import {
  aceSubgroupGemmWgsl,
  planAceSubgroupGemm,
} from "../src/webgpu/kernels/subgroup-gemm.js";
import type { AceGemmShape } from "../src/webgpu/kernels/gemm.js";
import {
  OPT_0009_ALLOCATION_COMMIT,
  OPT_0009_PARAKEET_COMMIT,
  OPT_0009_PARAKEET_GEMM_SOURCE_SHA256,
  authenticateAceOpt0009ParakeetSources,
  parseAceOpt0009ThermalMetadata,
  requireAceOpt0009Fixed32Device,
} from "./opt-0009-parakeet-gemm-calibration.js";

export const OPT_0009_ACE_SUBGROUP_GEMM_SOURCE_SHA256 =
  "9ba0c589f975f19b7f7990aae5199581a83f87500a3b97ff15eb6ef5a43311ea" as const;

export const ACE_OPT_0009_ACCUMULATION_SCOPE = Object.freeze({
  comparison:
    "packed-bf16-fp32-vs-fp16-fp32-vs-native-fp16-accumulation",
  comparisonImplemented: true,
  requiresBrowserExecution: true,
  requiresNumericalReview: true,
  acceptanceThresholdApplied: false,
  closesExperiment: false,
} as const);

export type AceOpt0009AccumulationArm =
  | "packed-bf16-fp32-oracle"
  | "fp16-fp32-accum"
  | "fp16-native-accum";

export type AceOpt0009FixtureKind =
  | "signed-zero"
  | "cancellation"
  | "fp16-range-overflow"
  | "long-k-cancellation"
  | "benign-production";

export interface AceOpt0009AccumulationShape extends AceGemmShape {
  readonly id: string;
}

export interface AceOpt0009CorrectnessFixture
  extends AceOpt0009AccumulationShape {
  readonly fixtureKind: AceOpt0009FixtureKind;
  readonly coverage: readonly string[];
}

export interface AceOpt0009AdaptedPlan extends AceGemmShape {
  readonly tileRows: 32;
  readonly tileColumns: 256;
  readonly tileInner: 32;
  readonly workgroupSize: 128;
  readonly subgroupSize: 32;
  readonly rowTiles: number;
  readonly columnTiles: number;
  readonly innerTiles: number;
  readonly workgroups: readonly [number, number, 1];
  readonly scheduledRows: number;
  readonly validMacs: number;
  readonly scheduledMacs: number;
  readonly validFlops: number;
  readonly scheduledFlops: number;
  readonly activationBytes: number;
  readonly weightBytes: number;
  readonly outputBytes: number;
  readonly sourceClassification:
    "explicitly-adapted-parakeet-fixed32-direct-n256";
}

const ROWS = 2_250;
const HIDDEN = 2_048;
const TILE_ROWS = 32;
const TILE_COLUMNS = 256;
const TILE_INNER = 32;
const WORKGROUP_SIZE = 128;
const SUBGROUP_SIZE = 32;
const SUBGROUPS_PER_WORKGROUP = WORKGROUP_SIZE / SUBGROUP_SIZE;
const ROWS_PER_SUBGROUP = TILE_ROWS / SUBGROUPS_PER_WORKGROUP;
const SCALARS_PER_LANE = TILE_COLUMNS / SUBGROUP_SIZE;
const OUTPUT_SENTINEL_BITS = 0x7fa1_2345;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;

export const ACE_OPT_0009_ACCUMULATION_SHAPES = Object.freeze([
  exactShape("ace-m2250-k2048-n2048", ROWS, HIDDEN, HIDDEN),
  exactShape("ace-m2250-k2048-n1024", ROWS, HIDDEN, 1_024),
  exactShape("ace-m2250-k2048-n6144", ROWS, HIDDEN, 6_144),
  exactShape("ace-m2250-k6144-n2048", ROWS, 6_144, HIDDEN),
] satisfies readonly AceOpt0009AccumulationShape[]);

/** Six rounds: every arm occupies every order position exactly twice. */
export const ACE_OPT_0009_ACCUMULATION_ORDERS = Object.freeze([
  Object.freeze([
    "packed-bf16-fp32-oracle",
    "fp16-fp32-accum",
    "fp16-native-accum",
  ]),
  Object.freeze([
    "fp16-fp32-accum",
    "fp16-native-accum",
    "packed-bf16-fp32-oracle",
  ]),
  Object.freeze([
    "fp16-native-accum",
    "packed-bf16-fp32-oracle",
    "fp16-fp32-accum",
  ]),
  Object.freeze([
    "fp16-native-accum",
    "fp16-fp32-accum",
    "packed-bf16-fp32-oracle",
  ]),
  Object.freeze([
    "fp16-fp32-accum",
    "packed-bf16-fp32-oracle",
    "fp16-native-accum",
  ]),
  Object.freeze([
    "packed-bf16-fp32-oracle",
    "fp16-native-accum",
    "fp16-fp32-accum",
  ]),
] satisfies readonly (readonly AceOpt0009AccumulationArm[])[]);

export const ACE_OPT_0009_CORRECTNESS_FIXTURES = Object.freeze([
  correctnessFixture(
    "signed-zero-m33-k32-n256",
    33,
    32,
    "signed-zero",
    ["positive-zero", "negative-zero", "complete-output", "m-tail"],
  ),
  correctnessFixture(
    "cancellation-m33-k32-n256",
    33,
    32,
    "cancellation",
    ["cancellation-sensitive-magnitudes", "source-k-order", "m-tail"],
  ),
  correctnessFixture(
    "fp16-range-m33-k32-n256",
    33,
    32,
    "fp16-range-overflow",
    [
      "fp16-max-finite",
      "fp16-min-normal",
      "fp16-min-subnormal",
      "positive-overflow",
      "negative-overflow",
      "finite-nonfinite-classification",
      "m-tail",
    ],
  ),
  correctnessFixture(
    "long-k-m3-k6144-n256",
    3,
    6_144,
    "long-k-cancellation",
    ["long-k", "k6144", "cancellation-sensitive-magnitudes"],
  ),
  correctnessFixture(
    "benign-m33-k2048-n256",
    33,
    2_048,
    "benign-production",
    ["benign-production-shaped-probe", "k2048", "m-tail"],
  ),
] satisfies readonly AceOpt0009CorrectnessFixture[]);

export const ACE_OPT_0009_FAIRNESS_DISCLOSURE = Object.freeze({
  diagnosticPurpose:
    "separate operand-storage and accumulation effects without claiming topology identity",
  adaptedPair: Object.freeze({
    activation: "packed-fp16-row-major",
    weight: "packed-fp16-direct-tile-major-n256-k32",
    output: "fp32-row-major",
    topology: "m32-n256-k32-wg128-one-direct-dispatch",
    sharedInputBuffers: true,
  }),
  productionOracle: Object.freeze({
    activation: "fp32-row-major-production-contract",
    weight: "packed-bf16-direct-tile-major-n128-k32",
    output: "fp32-row-major",
    topology: "m32-n128-k32-wg128-production-bounded-ranges",
    source: "src/webgpu/kernels/subgroup-gemm.ts:aceSubgroupGemmWgsl",
  }),
  unavoidableDistinctions: Object.freeze([
    "production oracle activation storage is fp32 while adapted arms use fp16",
    "production oracle uses N128 and bounded ranged dispatches while adapted arms use N256",
    "only the two adapted arms share identical inputs, output storage, and topology",
  ]),
  timingInterpretation:
    "three-arm exact-shape timings are diagnostic; the adapted-pair delta isolates accumulation mode most directly",
} as const);

/** Filled after generator review; the browser additionally records every source hash. */
export const ACE_OPT_0009_EXACT_SHADER_SHA256 = Object.freeze({
  "ace-m2250-k2048-n2048:packed-bf16-fp32-oracle":
    "4008ffe0aaf0aa20dffd2a59eb0c06e79f13faf6e71c37a0e3f1cd26b4b19d21",
  "ace-m2250-k2048-n2048:fp16-fp32-accum":
    "c09031890ecc692f5ee8abfe19928414ee34bc2c7d1a02dda24982a44ba2c732",
  "ace-m2250-k2048-n2048:fp16-native-accum":
    "86f62d0484c5281cf567534d2805cfd8b8b013ea6cd3861d84704f9ec6f4cdbd",
  "ace-m2250-k2048-n1024:packed-bf16-fp32-oracle":
    "bbfe1fa1739208fbfb4b120e1e01bfa93b6d4f479857aa79ced6fcfcc8e96265",
  "ace-m2250-k2048-n1024:fp16-fp32-accum":
    "e1536de245742a16be1bb4b1648470a018dab346e3d302c0a6492b96bd3d9c31",
  "ace-m2250-k2048-n1024:fp16-native-accum":
    "6147c1101866e41cd720f4fac4c4c3315a53c319d2cc2ba9fcf612495e7792ed",
  "ace-m2250-k2048-n6144:packed-bf16-fp32-oracle":
    "713a785f1bae0638aa5f9fce37e21e3cbf359e09258b6c06f94324a163a29a40",
  "ace-m2250-k2048-n6144:fp16-fp32-accum":
    "8dad17bb82643d423383b60d49371975e90f52062ee4389046397c3d32111a43",
  "ace-m2250-k2048-n6144:fp16-native-accum":
    "63f1116d47efdb4b8123e9dc3c4486639b436846e2b199b602dc61fa59d991c8",
  "ace-m2250-k6144-n2048:packed-bf16-fp32-oracle":
    "9a8e12960f281a1462cc079aed8105ea9d6527735b659a84684f3596c6b38ea6",
  "ace-m2250-k6144-n2048:fp16-fp32-accum":
    "a3fedb521ed29c6ecb10b2ccd69154ace05c63944ba23fe6a9e5d56d4df18504",
  "ace-m2250-k6144-n2048:fp16-native-accum":
    "8326cd80e1ce25308e1c0abe6abf6cf5380dfea4fa14410ea8282480adcc6d11",
} as const satisfies Readonly<Record<string, string>>);

export function planAceOpt0009AdaptedGemm(
  shape: AceGemmShape,
): AceOpt0009AdaptedPlan {
  requirePositiveInteger(shape.rows, "rows");
  requirePositiveInteger(shape.inner, "inner");
  requirePositiveInteger(shape.columns, "columns");
  if (shape.inner % TILE_INNER !== 0) {
    throw new RangeError("OPT-0009 adapted direct GEMM requires K divisible by 32");
  }
  if (shape.columns % TILE_COLUMNS !== 0) {
    throw new RangeError("OPT-0009 adapted direct GEMM requires N divisible by 256");
  }
  const rowTiles = Math.ceil(shape.rows / TILE_ROWS);
  const columnTiles = shape.columns / TILE_COLUMNS;
  const innerTiles = shape.inner / TILE_INNER;
  const scheduledRows = rowTiles * TILE_ROWS;
  const validMacs = checkedProduct(
    checkedProduct(shape.rows, shape.inner, "valid M*K"),
    shape.columns,
    "valid M*K*N",
  );
  const scheduledMacs = checkedProduct(
    checkedProduct(scheduledRows, shape.inner, "scheduled M*K"),
    shape.columns,
    "scheduled M*K*N",
  );
  const activationElements = checkedProduct(shape.rows, shape.inner, "A elements");
  const weightElements = checkedProduct(shape.columns, shape.inner, "B elements");
  const outputElements = checkedProduct(shape.rows, shape.columns, "C elements");
  return Object.freeze({
    ...shape,
    tileRows: TILE_ROWS,
    tileColumns: TILE_COLUMNS,
    tileInner: TILE_INNER,
    workgroupSize: WORKGROUP_SIZE,
    subgroupSize: SUBGROUP_SIZE,
    rowTiles,
    columnTiles,
    innerTiles,
    workgroups: Object.freeze([columnTiles, rowTiles, 1]) as readonly [
      number,
      number,
      1,
    ],
    scheduledRows,
    validMacs,
    scheduledMacs,
    validFlops: validMacs * 2,
    scheduledFlops: scheduledMacs * 2,
    activationBytes: checkedProduct(activationElements, 2, "A bytes"),
    weightBytes: checkedProduct(weightElements, 2, "B bytes"),
    outputBytes: checkedProduct(outputElements, 4, "C bytes"),
    sourceClassification:
      "explicitly-adapted-parakeet-fixed32-direct-n256",
  });
}

/** Logical `[N,K]` scalar to adapted Parakeet-derived direct N256 physical scalar. */
export function aceOpt0009AdaptedWeightScalarIndex(
  shape: Pick<AceGemmShape, "inner" | "columns">,
  logicalScalarIndex: number,
): number {
  const plan = planAceOpt0009AdaptedGemm({ rows: 1, ...shape });
  const scalarCount = plan.inner * plan.columns;
  requireIndex(logicalScalarIndex, scalarCount, "logical FP16 weight");
  const column = Math.floor(logicalScalarIndex / plan.inner);
  const inner = logicalScalarIndex % plan.inner;
  const columnTile = Math.floor(column / TILE_COLUMNS);
  const columnInTile = column % TILE_COLUMNS;
  const innerTile = Math.floor(inner / TILE_INNER);
  const innerInTile = inner % TILE_INNER;
  return (
    ((columnTile * plan.innerTiles + innerTile) * TILE_INNER + innerInTile) *
      TILE_COLUMNS +
    columnInTile
  );
}

export function aceOpt0009AccumulationWgsl(
  shape: AceGemmShape,
  arm: AceOpt0009AccumulationArm,
): string {
  if (arm === "packed-bf16-fp32-oracle") {
    return aceSubgroupGemmWgsl(shape, false);
  }
  const plan = planAceOpt0009AdaptedGemm(shape);
  const fp32Accumulation = arm === "fp16-fp32-accum";
  const accumulatorType = fp32Accumulation ? "f32" : "f16";
  const zero = fp32Accumulation ? "0.0" : "0.0h";
  const declarations = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) =>
      `  var acc${row}_0 = vec4<${accumulatorType}>(${zero});\n` +
      `  var acc${row}_1 = vec4<${accumulatorType}>(${zero});`,
  ).join("\n");
  const contractions = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => {
      const a = `a${row}`;
      return fp32Accumulation
        ? `      acc${row}_0 = acc${row}_0 + vec4<f32>(f32(${a})) * vec4<f32>(b0);\n` +
            `      acc${row}_1 = acc${row}_1 + vec4<f32>(f32(${a})) * vec4<f32>(b1);`
        : `      acc${row}_0 = fma(vec4<f16>(${a}), b0, acc${row}_0);\n` +
            `      acc${row}_1 = fma(vec4<f16>(${a}), b1, acc${row}_1);`;
    },
  ).join("\n");
  const broadcasts = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `      let a${row} = subgroupBroadcast(lane_a, ${row}u);`,
  ).join("\n");
  const stores = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + ${row}u;
    if (row < ROWS) {
      let output_vector_base = row * (COLUMNS / 4u) + column_base / 4u;
      output[output_vector_base] = vec4<f32>(acc${row}_0);
      output[output_vector_base + 1u] = vec4<f32>(acc${row}_1);
    }
  }`,
  ).join("\n");

  return /* wgsl */ `
// OPT-0009 explicit adaptation of pinned Parakeet direct N256/K32 geometry.
// Adaptations: arbitrary K32, guarded M tail, common FP32 output, and selected
// ${fp32Accumulation ? "FP32" : "native FP16"} accumulation. This is not unchanged Parakeet source.
enable f16;
enable subgroups;

const ROWS = ${plan.rows}u;
const INNER = ${plan.inner}u;
const COLUMNS = ${plan.columns}u;
const INNER_TILES = ${plan.innerTiles}u;

@group(0) @binding(0) var<storage, read> activation: array<u32>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

fn load_activation(row: u32, inner: u32) -> f16 {
  let scalar = row * INNER + inner;
  let pair = unpack2x16float(activation[scalar >> 1u]);
  return f16(select(pair.x, pair.y, (scalar & 1u) != 0u));
}

fn unpack_f16x4(low: u32, high: u32) -> vec4<f16> {
  let low_pair = unpack2x16float(low);
  let high_pair = unpack2x16float(high);
  return vec4<f16>(
    f16(low_pair.x),
    f16(low_pair.y),
    f16(high_pair.x),
    f16(high_pair.y),
  );
}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (
    subgroup_size != ${SUBGROUP_SIZE}u ||
    group.x >= ${plan.columnTiles}u ||
    group.y >= ${plan.rowTiles}u ||
    group.z != 0u ||
    subgroup >= ${SUBGROUPS_PER_WORKGROUP}u
  ) {
    return;
  }
  let row_base =
    group.y * ${TILE_ROWS}u + subgroup * ${ROWS_PER_SUBGROUP}u;
  let column_base =
    group.x * ${TILE_COLUMNS}u + subgroup_lane * ${SCALARS_PER_LANE}u;
${declarations}

  for (var inner_tile = 0u; inner_tile < INNER_TILES; inner_tile += 1u) {
    let weight_tile_base =
      (group.x * INNER_TILES + inner_tile) * ${TILE_INNER * SUBGROUP_SIZE}u;
    for (var inner_in_tile = 0u; inner_in_tile < ${TILE_INNER}u; inner_in_tile += 1u) {
      let inner = inner_tile * ${TILE_INNER}u + inner_in_tile;
      var lane_a = 0.0h;
      let lane_row = row_base + subgroup_lane;
      if (subgroup_lane < ${ROWS_PER_SUBGROUP}u && lane_row < ROWS) {
        lane_a = load_activation(lane_row, inner);
      }
      let packed_b = weight[
        weight_tile_base + inner_in_tile * ${SUBGROUP_SIZE}u + subgroup_lane
      ];
      let b0 = unpack_f16x4(packed_b.x, packed_b.y);
      let b1 = unpack_f16x4(packed_b.z, packed_b.w);
${broadcasts}
${contractions}
    }
  }
${stores}
}
`;
}

export function aceOpt0009FixtureActivationValue(
  fixtureKind: AceOpt0009FixtureKind,
  row: number,
  inner: number,
): number {
  switch (fixtureKind) {
    case "signed-zero": {
      const values = [-0, +0, -1, 1, -(2 ** -24), 2 ** -24] as const;
      return values[(inner + row) % values.length]!;
    }
    case "cancellation": {
      const values = [2_048, 1, -2_048, 0.5, 1_024, -0.5, -1_024, 2 ** -10] as const;
      return values[(inner + row * 3) % values.length]!;
    }
    case "fp16-range-overflow": {
      if (inner === 0) return row % 2 === 0 ? 65_504 : -65_504;
      if (inner === 1) return 2 ** -14;
      if (inner === 2) return 2 ** -24;
      return inner % 2 === 0 ? +0 : -0;
    }
    case "long-k-cancellation": {
      const values = [
        1,
        2 ** -10,
        -1,
        2 ** -11,
        0.5,
        -(2 ** -12),
        -0.5,
        2 ** -13,
      ] as const;
      return values[(inner + row) % values.length]!;
    }
    case "benign-production":
      return Math.fround((((row * 17 + inner * 13 + 3) % 31) - 15) / 64);
  }
}

export function aceOpt0009FixtureWeightValue(
  fixtureKind: AceOpt0009FixtureKind,
  column: number,
  inner: number,
): number {
  switch (fixtureKind) {
    case "signed-zero": {
      const values = [1, -1, -0, +0, 0.5, -0.5] as const;
      return values[(inner + column) % values.length]!;
    }
    case "cancellation":
      return column % 4 < 2 ? 1 : -1;
    case "fp16-range-overflow": {
      if (inner === 0) return [2, -2, 1, 0.5][column % 4]!;
      if (inner === 1 || inner === 2) return column % 2 === 0 ? 1 : -1;
      return 0;
    }
    case "long-k-cancellation":
      return column % 4 === 0 ? 1 : column % 4 === 1 ? -1 : 0.5;
    case "benign-production":
      return Math.fround((((column * 13 + inner * 7 + 7) % 29) - 14) / 64);
  }
}

export function aceOpt0009CpuOutputValue(
  fixture: Pick<AceOpt0009CorrectnessFixture, "fixtureKind" | "inner" | "columns">,
  arm: AceOpt0009AccumulationArm,
  outputIndex: number,
  fp32Mode: "separate-product-add" | "contracted-expression" =
    "separate-product-add",
): number {
  const outputElements = checkedProduct(1, fixture.columns, "fixture columns");
  if (!Number.isSafeInteger(outputIndex) || outputIndex < 0) {
    throw new RangeError("OPT-0009 CPU output index must be non-negative");
  }
  const row = Math.floor(outputIndex / outputElements);
  const column = outputIndex % fixture.columns;
  let accumulator = arm === "fp16-native-accum" ? roundToFloat16(0) : 0;
  for (let inner = 0; inner < fixture.inner; inner += 1) {
    const sourceA = aceOpt0009FixtureActivationValue(
      fixture.fixtureKind,
      row,
      inner,
    );
    const sourceB = aceOpt0009FixtureWeightValue(
      fixture.fixtureKind,
      column,
      inner,
    );
    const a = arm === "packed-bf16-fp32-oracle"
      ? Math.fround(sourceA)
      : roundToFloat16(sourceA);
    const b = arm === "packed-bf16-fp32-oracle"
      ? bf16BitsToNumber(numberToBf16Bits(sourceB))
      : roundToFloat16(sourceB);
    if (arm === "fp16-native-accum") {
      accumulator = roundToFloat16(accumulator + a * b);
    } else if (fp32Mode === "separate-product-add") {
      accumulator = Math.fround(accumulator + Math.fround(a * b));
    } else {
      accumulator = Math.fround(accumulator + a * b);
    }
  }
  return accumulator;
}

export function summarizeAceOpt0009Samples(
  samples: readonly number[],
): Readonly<Record<string, unknown>> {
  if (
    samples.length === 0 ||
    samples.some((sample) => !Number.isFinite(sample) || sample < 0)
  ) {
    throw new RangeError("OPT-0009 samples must be finite non-negative values");
  }
  const retained = [...samples];
  const sorted = [...retained].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
  return Object.freeze({
    count: retained.length,
    samples: Object.freeze(retained),
    minimum: sorted[0]!,
    median,
    maximum: sorted.at(-1)!,
    range: sorted.at(-1)! - sorted[0]!,
  });
}

type Opt0009Mode = "correctness" | "timing";

interface PreparedArm {
  readonly id: AceOpt0009AccumulationArm;
  readonly source: string;
  readonly shaderSha256: string;
  readonly compilationMessages: readonly Readonly<Record<string, unknown>>[];
  readonly pipeline: GPUComputePipeline;
  readonly bindGroups: readonly GPUBindGroup[];
  readonly output: GPUBuffer;
  readonly shape: AceGemmShape;
  readonly dispatchesPerExecution: number;
  readonly counts: MutableExecutionCounts;
  encode(pass: GPUComputePassEncoder): void;
}

interface PreparedShape {
  readonly fixture: AceOpt0009CorrectnessFixture;
  readonly adaptedPlan: AceOpt0009AdaptedPlan;
  readonly oraclePlan: ReturnType<typeof planAceSubgroupGemm>;
  readonly arms: Readonly<Record<AceOpt0009AccumulationArm, PreparedArm>>;
  readonly preparation: Readonly<Record<string, unknown>>;
  destroy(): void;
}

interface MutableExecutionCounts {
  executions: number;
  encodedCommandBuffers: number;
  submissions: number;
  drains: number;
  dispatches: number;
}

interface OutputReadback {
  readonly scan: Readonly<Record<string, unknown>>;
  readonly rawFnv1a32: string;
  readonly values?: Float32Array;
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  created = 0;
  destroyed = 0;
  maximumLive = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    this.live.add(buffer);
    this.created += 1;
    this.maximumLive = Math.max(this.maximumLive, this.live.size);
    return buffer;
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    buffer.destroy();
    this.destroyed += 1;
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number>> {
    return Object.freeze({
      created: this.created,
      destroyed: this.destroyed,
      live: this.live.size,
      maximumLive: this.maximumLive,
    });
  }
}

if (typeof document !== "undefined") installBrowserUi();

function installBrowserUi(): void {
  const correctness = requireButton("#run-correctness");
  const timing = requireButton("#run-timing");
  const start = (mode: Opt0009Mode): void => {
    correctness.disabled = true;
    timing.disabled = true;
    document.body.dataset.status = "running";
    updateProgress(`starting ${mode}`);
    void runBrowser(mode).then(
      (result) => finish("passed", result),
      (error: unknown) => finish("failed", {
        schema: "ace-opt-0009-three-arm-accumulation-ab-v1",
        status: "failed",
        experimentId: "OPT-0009",
        mode,
        error: errorReceipt(error),
      }),
    );
  };
  correctness.addEventListener("click", () => start("correctness"), { once: true });
  timing.addEventListener("click", () => start("timing"), { once: true });
}

async function runBrowser(mode: Opt0009Mode): Promise<unknown> {
  const runStartedAtEpochMilliseconds = Date.now();
  const parameters = new URL(window.location.href).searchParams;
  const parakeetSourceIdentity = await authenticateAceOpt0009ParakeetSources(
    parameters,
  );
  const actualAceSourceHash = await sha256Hex(aceSubgroupGemmSource);
  if (actualAceSourceHash !== OPT_0009_ACE_SUBGROUP_GEMM_SOURCE_SHA256) {
    throw new Error("OPT-0009 rejected unauthenticated ACE subgroup GEMM source");
  }
  const thermal = mode === "timing"
    ? parseAceOpt0009ThermalMetadata(parameters)
    : null;
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAceOpt0009Fixed32Device(
    new Set(adapter.features),
    adapter.info.subgroupMinSize,
    adapter.info.subgroupMaxSize,
  );
  const maximumBindingBytes = maximumRequiredBindingBytes();
  requireAdapterLimits(adapter, maximumBindingBytes);
  const device = await adapter.requestDevice({
    label: "ace-opt-0009-three-arm-device",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: {
      maxBufferSize: maximumBindingBytes,
      maxStorageBufferBindingSize: maximumBindingBytes,
      maxComputeInvocationsPerWorkgroup: WORKGROUP_SIZE,
      maxComputeWorkgroupSizeX: WORKGROUP_SIZE,
    },
  });
  const tracker = new BufferTracker();
  const uncapturedErrors: Readonly<Record<string, unknown>>[] = [];
  const onUncapturedError = (event: GPUUncapturedErrorEvent): void => {
    uncapturedErrors.push(Object.freeze({
      name: event.error.constructor.name,
      message: event.error.message,
    }));
  };
  device.addEventListener("uncapturederror", onUncapturedError);
  let unexpectedDeviceLoss: Readonly<Record<string, unknown>> | null = null;
  let destroyingDevice = false;
  void device.lost.then((info) => {
    if (!destroyingDevice) {
      unexpectedDeviceLoss = Object.freeze({
        reason: info.reason,
        message: info.message,
      });
    }
  });
  const correctnessResults: unknown[] = [];
  const exactShapeResults: unknown[] = [];
  let cancellation: unknown = null;
  let firstTimedAtEpochMilliseconds: number | null = null;
  let lastTimedAtEpochMilliseconds: number | null = null;
  let cleanup: unknown;
  try {
    for (const [index, fixture] of ACE_OPT_0009_CORRECTNESS_FIXTURES.entries()) {
      updateProgress(
        `adversarial ${index + 1}/${ACE_OPT_0009_CORRECTNESS_FIXTURES.length}: ${fixture.id}`,
      );
      correctnessResults.push(await runCorrectnessFixture(device, tracker, fixture));
      await yieldToBrowser();
    }
    for (const [index, shape] of ACE_OPT_0009_ACCUMULATION_SHAPES.entries()) {
      updateProgress(
        `exact shape ${index + 1}/${ACE_OPT_0009_ACCUMULATION_SHAPES.length}: ${shape.id}`,
      );
      const fixture = correctnessFixture(
        shape.id,
        shape.rows,
        shape.inner,
        "benign-production",
        ["exact-ace-shape", "complete-write", "m2250-tail"],
        shape.columns,
      );
      const result = await runExactShape(
        device,
        tracker,
        fixture,
        mode,
        mode === "timing" &&
          index === ACE_OPT_0009_ACCUMULATION_SHAPES.length - 1,
        (started, completed) => {
          firstTimedAtEpochMilliseconds = firstTimedAtEpochMilliseconds === null
            ? started
            : Math.min(firstTimedAtEpochMilliseconds, started);
          lastTimedAtEpochMilliseconds = lastTimedAtEpochMilliseconds === null
            ? completed
            : Math.max(lastTimedAtEpochMilliseconds, completed);
        },
      );
      exactShapeResults.push(result.result);
      cancellation = result.cancellation ?? cancellation;
      await yieldToBrowser();
    }
    if (uncapturedErrors.length !== 0 || unexpectedDeviceLoss !== null) {
      throw new Error("OPT-0009 observed a WebGPU device event");
    }
  } finally {
    tracker.destroyAll();
    device.removeEventListener("uncapturederror", onUncapturedError);
    destroyingDevice = true;
    device.destroy();
    cleanup = Object.freeze({
      ...tracker.receipt(),
      deviceDestroyed: true,
      uncapturedErrors: Object.freeze([...uncapturedErrors]),
      unexpectedDeviceLoss,
      cleanupCompletedAtEpochMilliseconds: Date.now(),
    });
  }
  const completedAtEpochMilliseconds = Date.now();
  return Object.freeze({
    schema: "ace-opt-0009-three-arm-accumulation-ab-v1",
    status: "passed",
    experimentId: "OPT-0009",
    classification: "benchmark-only-three-arm-decision-evidence",
    mode,
    recordedAt: new Date(completedAtEpochMilliseconds).toISOString(),
    runStartedAtEpochMilliseconds,
    firstTimedAtEpochMilliseconds,
    lastTimedAtEpochMilliseconds,
    completedAtEpochMilliseconds,
    accumulationScope: ACE_OPT_0009_ACCUMULATION_SCOPE,
    fairnessDisclosure: ACE_OPT_0009_FAIRNESS_DISCLOSURE,
    sourceIdentity: Object.freeze({
      allocationCommit: OPT_0009_ALLOCATION_COMMIT,
      parakeetCommit: OPT_0009_PARAKEET_COMMIT,
      parakeetGemmSha256: OPT_0009_PARAKEET_GEMM_SOURCE_SHA256,
      authenticatedParakeetSources: parakeetSourceIdentity,
      aceSubgroupGemmSha256: actualAceSourceHash,
    }),
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      page: window.location.href,
      adapter: adapterReceipt(adapter),
    }),
    protocol: Object.freeze({
      thermal,
      embeddedThermalMetadataRole: "pre-gate-only",
      thermalTimingAuthority:
        "external-continuous-artifact-spanning-first-timing-fence-through-cleanup",
      thermalJoinEpochFields: Object.freeze([
        "runStartedAtEpochMilliseconds",
        "firstTimedAtEpochMilliseconds",
        "lastTimedAtEpochMilliseconds",
        "cleanup.cleanupCompletedAtEpochMilliseconds",
      ]),
      thermalSource: THERMAL_SOURCE,
      oneOutstandingCommandBuffer: true,
      queueDrainAfterEveryCommandBuffer: true,
      sourcePackingUploadCompilationExcludedFromTiming: true,
      timingOrders: ACE_OPT_0009_ACCUMULATION_ORDERS,
      fmaFlopConvention: 2,
      numericalAcceptanceThresholdApplied: false,
      nonFinitePolicy: "classify-and-report-without-discarding-samples",
    }),
    adversarialCorrectness: Object.freeze(correctnessResults),
    exactShapes: Object.freeze(exactShapeResults),
    cancellation,
    cleanup,
  });
}

async function runCorrectnessFixture(
  device: GPUDevice,
  tracker: BufferTracker,
  fixture: AceOpt0009CorrectnessFixture,
): Promise<unknown> {
  const prepared = await prepareShape(device, tracker, fixture);
  try {
    const arms: Partial<Record<AceOpt0009AccumulationArm, unknown>> = {};
    const firstOutputs = new Map<AceOpt0009AccumulationArm, Float32Array>();
    for (const armId of allArms()) {
      const arm = prepared.arms[armId];
      const firstExecution = await executeArm(device, arm, -2, "correctness", 0);
      const first = await readOutput(device, tracker, prepared, arm, true);
      const rerunExecution = await executeArm(
        device,
        arm,
        -1,
        "deterministic-rerun",
        0,
      );
      const rerun = await readOutput(device, tracker, prepared, arm, true);
      if (first.rawFnv1a32 !== rerun.rawFnv1a32) {
        throw new Error(`${fixture.id} ${armId} full output was nondeterministic`);
      }
      const firstValues = first.values!;
      firstOutputs.set(armId, firstValues);
      validateExecutionCounts(arm, 2);
      arms[armId] = Object.freeze({
        shaderSha256: arm.shaderSha256,
        shaderCompilationMessages: arm.compilationMessages,
        firstExecution,
        rerunExecution,
        firstScan: first.scan,
        rerunScan: rerun.scan,
        deterministicRawFingerprint: true,
        cpuFp32Diagnostics: cpuDiagnostics(fixture, armId, firstValues),
        executionCounts: Object.freeze({ ...arm.counts }),
      });
    }
    return Object.freeze({
      id: fixture.id,
      fixture,
      fullOutputReadback: true,
      everyOutputComparedToCpu: true,
      acceptanceThresholdApplied: false,
      plan: Object.freeze({
        adapted: prepared.adaptedPlan,
        oracle: prepared.oraclePlan,
      }),
      preparation: prepared.preparation,
      arms: Object.freeze(arms),
      pairwiseGpuDiagnostics: Object.freeze({
        oracleVsFp16Fp32: compareAceOpt0009GpuOutputs(
          firstOutputs.get("packed-bf16-fp32-oracle")!,
          firstOutputs.get("fp16-fp32-accum")!,
        ),
        fp16Fp32VsNativeFp16: compareAceOpt0009GpuOutputs(
          firstOutputs.get("fp16-fp32-accum")!,
          firstOutputs.get("fp16-native-accum")!,
        ),
        oracleVsNativeFp16: compareAceOpt0009GpuOutputs(
          firstOutputs.get("packed-bf16-fp32-oracle")!,
          firstOutputs.get("fp16-native-accum")!,
        ),
        acceptanceThresholdApplied: false,
      }),
    });
  } finally {
    prepared.destroy();
  }
}

async function runExactShape(
  device: GPUDevice,
  tracker: BufferTracker,
  fixture: AceOpt0009CorrectnessFixture,
  mode: Opt0009Mode,
  testPostTimingCancellation: boolean,
  observeTiming: (started: number, completed: number) => void,
): Promise<{ readonly result: unknown; readonly cancellation: unknown }> {
  const prepared = await prepareShape(device, tracker, fixture);
  try {
    const initialScans = new Map<AceOpt0009AccumulationArm, OutputReadback>();
    const rerunScans = new Map<AceOpt0009AccumulationArm, OutputReadback>();
    const initialExecutions = new Map<AceOpt0009AccumulationArm, unknown>();
    for (const armId of allArms()) {
      const arm = prepared.arms[armId];
      initialExecutions.set(
        armId,
        await executeArm(device, arm, -2, "correctness", 0),
      );
      initialScans.set(
        armId,
        await readOutput(device, tracker, prepared, arm, false),
      );
    }
    for (const armId of allArms()) {
      const arm = prepared.arms[armId];
      await executeArm(
        device,
        arm,
        -1,
        mode === "timing" ? "symmetric-warmup" : "deterministic-rerun",
        0,
      );
      const rerun = await readOutput(device, tracker, prepared, arm, false);
      rerunScans.set(armId, rerun);
      if (rerun.rawFnv1a32 !== initialScans.get(armId)!.rawFnv1a32) {
        throw new Error(`${fixture.id} ${armId} exact output was nondeterministic`);
      }
    }
    const samples = new Map<AceOpt0009AccumulationArm, unknown[]>();
    for (const armId of allArms()) samples.set(armId, []);
    if (mode === "timing") {
      for (const [roundIndex, order] of ACE_OPT_0009_ACCUMULATION_ORDERS.entries()) {
        for (const [orderPosition, armId] of order.entries()) {
          const sample = await executeArm(
            device,
            prepared.arms[armId],
            roundIndex,
            order.join("-"),
            orderPosition,
          );
          samples.get(armId)!.push(sample);
          observeTiming(
            sample.startedAtEpochMilliseconds,
            sample.completedAtEpochMilliseconds,
          );
          await yieldToBrowser();
        }
      }
    }
    const armResults: Partial<Record<AceOpt0009AccumulationArm, unknown>> = {};
    for (const armId of allArms()) {
      const arm = prepared.arms[armId];
      const post = mode === "timing"
        ? await readOutput(device, tracker, prepared, arm, false)
        : null;
      if (
        post !== null &&
        post.rawFnv1a32 !== initialScans.get(armId)!.rawFnv1a32
      ) {
        throw new Error(`${fixture.id} ${armId} post-timing output changed`);
      }
      const armSamples = samples.get(armId)! as readonly ExecutionTiming[];
      validateExecutionCounts(
        arm,
        2 + (mode === "timing" ? ACE_OPT_0009_ACCUMULATION_ORDERS.length : 0),
      );
      armResults[armId] = Object.freeze({
        shaderSha256: arm.shaderSha256,
        shaderCompilationMessages: arm.compilationMessages,
        initialExecution: initialExecutions.get(armId),
        initialScan: initialScans.get(armId)!.scan,
        rerunScan: rerunScans.get(armId)!.scan,
        postTimingScan: post?.scan ?? null,
        postTimingRawFingerprintValidated: post === null ? null : true,
        samples: Object.freeze([...armSamples]),
        summary: armSamples.length === 0
          ? null
          : Object.freeze({
              fencedWallMilliseconds: summarizeAceOpt0009Samples(
                armSamples.map((sample) => sample.wallMilliseconds),
              ),
              validLogicalTflops: summarizeAceOpt0009Samples(
                armSamples.map((sample) => sample.validLogicalTflops),
              ),
            }),
        executionCountsBeforeCancellation: Object.freeze({ ...arm.counts }),
      });
    }
    const fp32Samples = samples.get("fp16-fp32-accum")! as readonly ExecutionTiming[];
    const nativeSamples = samples.get("fp16-native-accum")! as readonly ExecutionTiming[];
    const postTimingCancellation = testPostTimingCancellation
      ? await runPostTimingCancellation(
          device,
          prepared.arms["fp16-native-accum"],
        )
      : null;
    return Object.freeze({
      result: Object.freeze({
        id: fixture.id,
        shape: Object.freeze({
          rows: fixture.rows,
          inner: fixture.inner,
          columns: fixture.columns,
        }),
        plan: Object.freeze({
          adapted: prepared.adaptedPlan,
          oracle: prepared.oraclePlan,
        }),
        preparation: prepared.preparation,
        arms: Object.freeze(armResults),
        adaptedPairTiming: mode === "timing"
          ? Object.freeze({
              fp16NativeMedianSpeedupOverFp16Fp32:
                medianOf(fp32Samples.map((sample) => sample.wallMilliseconds)) /
                medianOf(nativeSamples.map((sample) => sample.wallMilliseconds)),
              pairedRoundNativeWins: nativeSamples.filter((sample, index) =>
                sample.wallMilliseconds < fp32Samples[index]!.wallMilliseconds
              ).length,
              topologyAndStorageIdentical: true,
              acceptanceThresholdApplied: false,
            })
          : null,
      }),
      cancellation: postTimingCancellation,
    });
  } finally {
    prepared.destroy();
  }
}

interface ExecutionTiming {
  readonly roundIndex: number;
  readonly order: string;
  readonly orderPosition: number;
  readonly wallMilliseconds: number;
  readonly encodeMilliseconds: number;
  readonly submitMilliseconds: number;
  readonly drainMilliseconds: number;
  readonly validLogicalTflops: number;
  readonly scheduledLogicalTflops: number;
  readonly commandBufferCount: 1;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
}

async function executeArm(
  device: GPUDevice,
  arm: PreparedArm,
  roundIndex: number,
  order: string,
  orderPosition: number,
): Promise<ExecutionTiming> {
  const startedAtEpochMilliseconds = Date.now();
  const wallStarted = performance.now();
  const encodeStarted = performance.now();
  const encoder = device.createCommandEncoder({
    label: `${arm.id}-${order}-encoder`,
  });
  const pass = encoder.beginComputePass({ label: `${arm.id}-${order}-pass` });
  arm.encode(pass);
  pass.end();
  const command = encoder.finish();
  arm.counts.executions += 1;
  arm.counts.encodedCommandBuffers += 1;
  arm.counts.dispatches += arm.dispatchesPerExecution;
  const encodeMilliseconds = performance.now() - encodeStarted;
  const submitStarted = performance.now();
  device.queue.submit([command]);
  arm.counts.submissions += 1;
  const submitMilliseconds = performance.now() - submitStarted;
  const drainStarted = performance.now();
  await device.queue.onSubmittedWorkDone();
  arm.counts.drains += 1;
  const drainMilliseconds = performance.now() - drainStarted;
  const wallMilliseconds = performance.now() - wallStarted;
  const plan = planAceOpt0009AdaptedGemm(currentShapeFromArm(arm));
  return Object.freeze({
    roundIndex,
    order,
    orderPosition,
    wallMilliseconds,
    encodeMilliseconds,
    submitMilliseconds,
    drainMilliseconds,
    validLogicalTflops: plan.validFlops / wallMilliseconds / 1e9,
    scheduledLogicalTflops: plan.scheduledFlops / wallMilliseconds / 1e9,
    commandBufferCount: 1,
    startedAtEpochMilliseconds,
    completedAtEpochMilliseconds: Date.now(),
  });
}

async function prepareShape(
  device: GPUDevice,
  tracker: BufferTracker,
  fixture: AceOpt0009CorrectnessFixture,
): Promise<PreparedShape> {
  const started = performance.now();
  const shape = {
    rows: fixture.rows,
    inner: fixture.inner,
    columns: fixture.columns,
  };
  const adaptedPlan = planAceOpt0009AdaptedGemm(shape);
  const oraclePlan = planAceSubgroupGemm(shape);
  const owned: GPUBuffer[] = [];
  let destroyed = false;
  const create = (descriptor: GPUBufferDescriptor): GPUBuffer => {
    const buffer = tracker.create(device, descriptor);
    owned.push(buffer);
    return buffer;
  };
  try {
    const oracleActivation = createMappedStorage(
      device,
      tracker,
      owned,
      `${fixture.id}-oracle-activation`,
      oraclePlan.activationElements * 4,
    );
    const fp16Activation = createMappedStorage(
      device,
      tracker,
      owned,
      `${fixture.id}-fp16-activation`,
      adaptedPlan.activationBytes,
    );
    const oracleWeight = createMappedStorage(
      device,
      tracker,
      owned,
      `${fixture.id}-oracle-weight`,
      oraclePlan.packedWeightWords * 4,
    );
    const fp16Weight = createMappedStorage(
      device,
      tracker,
      owned,
      `${fixture.id}-fp16-weight`,
      adaptedPlan.weightBytes,
    );
    const packingStarted = performance.now();
    fillActivations(oracleActivation, fp16Activation, fixture);
    fillWeights(oracleWeight, fp16Weight, fixture);
    const packingMilliseconds = performance.now() - packingStarted;
    for (const buffer of [
      oracleActivation,
      fp16Activation,
      oracleWeight,
      fp16Weight,
    ]) buffer.unmap();
    const outputs = {
      "packed-bf16-fp32-oracle": createOutput(
        device,
        tracker,
        owned,
        `${fixture.id}-oracle-output`,
        adaptedPlan.outputBytes,
      ),
      "fp16-fp32-accum": createOutput(
        device,
        tracker,
        owned,
        `${fixture.id}-fp16-fp32-output`,
        adaptedPlan.outputBytes,
      ),
      "fp16-native-accum": createOutput(
        device,
        tracker,
        owned,
        `${fixture.id}-fp16-native-output`,
        adaptedPlan.outputBytes,
      ),
    } as const;
    const rangeStride = Math.max(
      256,
      device.limits.minUniformBufferOffsetAlignment,
    );
    const rangeParameters = create({
      label: `${fixture.id}-oracle-range-parameters`,
      size: Math.max(1, oraclePlan.outputRangeCount) * rangeStride,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    });
    const rangeMapped = rangeParameters.getMappedRange();
    for (let index = 0; index < oraclePlan.outputRanges.length; index += 1) {
      new Uint32Array(rangeMapped, index * rangeStride, 4)[0] =
        oraclePlan.outputRanges[index]!.firstWorkgroup;
    }
    rangeParameters.unmap();

    const compileStarted = performance.now();
    const oracleSource = aceOpt0009AccumulationWgsl(
      shape,
      "packed-bf16-fp32-oracle",
    );
    const fp32Source = aceOpt0009AccumulationWgsl(shape, "fp16-fp32-accum");
    const nativeSource = aceOpt0009AccumulationWgsl(shape, "fp16-native-accum");
    const oracleCompiled = await compilePipeline(device, fixture.id, "oracle", oracleSource);
    const fp32Compiled = await compilePipeline(device, fixture.id, "fp16-fp32", fp32Source);
    const nativeCompiled = await compilePipeline(device, fixture.id, "fp16-native", nativeSource);
    const compileMilliseconds = performance.now() - compileStarted;
    const oracleBindGroups = oraclePlan.outputRanges.map((_range, index) =>
      device.createBindGroup({
        label: `${fixture.id}-oracle-range-${index}-bindings`,
        layout: oracleCompiled.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: oracleActivation } },
          { binding: 1, resource: { buffer: oracleWeight } },
          { binding: 2, resource: { buffer: outputs["packed-bf16-fp32-oracle"] } },
          {
            binding: 3,
            resource: {
              buffer: rangeParameters,
              offset: index * rangeStride,
              size: 16,
            },
          },
        ],
      })
    );
    const adaptedBindGroup = (
      pipeline: GPUComputePipeline,
      output: GPUBuffer,
      label: string,
    ): GPUBindGroup => device.createBindGroup({
      label,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: fp16Activation } },
        { binding: 1, resource: { buffer: fp16Weight } },
        { binding: 2, resource: { buffer: output } },
      ],
    });
    const fp32BindGroup = adaptedBindGroup(
      fp32Compiled.pipeline,
      outputs["fp16-fp32-accum"],
      `${fixture.id}-fp16-fp32-bindings`,
    );
    const nativeBindGroup = adaptedBindGroup(
      nativeCompiled.pipeline,
      outputs["fp16-native-accum"],
      `${fixture.id}-fp16-native-bindings`,
    );
    const arms = Object.freeze({
      "packed-bf16-fp32-oracle": preparedArm(
        "packed-bf16-fp32-oracle",
        oracleSource,
        oracleCompiled,
        oracleBindGroups,
        outputs["packed-bf16-fp32-oracle"],
        oraclePlan.outputRangeCount,
        (pass) => {
          pass.setPipeline(oracleCompiled.pipeline);
          for (let index = 0; index < oraclePlan.outputRanges.length; index += 1) {
            pass.setBindGroup(0, oracleBindGroups[index]!);
            pass.dispatchWorkgroups(
              oraclePlan.outputRanges[index]!.workgroupCount,
              1,
              1,
            );
          }
        },
        shape,
      ),
      "fp16-fp32-accum": preparedArm(
        "fp16-fp32-accum",
        fp32Source,
        fp32Compiled,
        [fp32BindGroup],
        outputs["fp16-fp32-accum"],
        1,
        (pass) => {
          pass.setPipeline(fp32Compiled.pipeline);
          pass.setBindGroup(0, fp32BindGroup);
          pass.dispatchWorkgroups(...adaptedPlan.workgroups);
        },
        shape,
      ),
      "fp16-native-accum": preparedArm(
        "fp16-native-accum",
        nativeSource,
        nativeCompiled,
        [nativeBindGroup],
        outputs["fp16-native-accum"],
        1,
        (pass) => {
          pass.setPipeline(nativeCompiled.pipeline);
          pass.setBindGroup(0, nativeBindGroup);
          pass.dispatchWorkgroups(...adaptedPlan.workgroups);
        },
        shape,
      ),
    });
    verifyExactShaderHashes(fixture.id, arms);
    return Object.freeze({
      fixture,
      adaptedPlan,
      oraclePlan,
      arms,
      preparation: Object.freeze({
        totalMilliseconds: performance.now() - started,
        packingMilliseconds,
        compileMilliseconds,
        excludedFromTiming: true,
        sharedAdaptedActivationAndWeightBuffers: true,
        oracleRangeCount: oraclePlan.outputRangeCount,
        adaptedDispatchesPerExecution: 1,
        sourceBytes: Object.freeze(Object.fromEntries(
          allArms().map((armId) => [
            armId,
            new TextEncoder().encode(arms[armId].source).byteLength,
          ]),
        )),
      }),
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
        for (const buffer of owned) tracker.destroy(buffer);
      },
    });
  } catch (error) {
    for (const buffer of owned) {
      if (buffer.mapState === "mapped") buffer.unmap();
      tracker.destroy(buffer);
    }
    throw error;
  }
}

interface CompiledPipeline {
  readonly pipeline: GPUComputePipeline;
  readonly shaderSha256: string;
  readonly messages: readonly Readonly<Record<string, unknown>>[];
}

async function compilePipeline(
  device: GPUDevice,
  fixtureId: string,
  armLabel: string,
  source: string,
): Promise<CompiledPipeline> {
  device.pushErrorScope("validation");
  let scopeOpen = true;
  try {
    const module = device.createShaderModule({
      label: `${fixtureId}-${armLabel}-module`,
      code: source,
    });
    const info = await module.getCompilationInfo();
    const messages = Object.freeze(info.messages.map((message) => Object.freeze({
      type: message.type,
      message: message.message,
      lineNum: message.lineNum,
      linePos: message.linePos,
    })));
    if (info.messages.some((message) => message.type === "error")) {
      const details = messages
        .filter((message) => message.type === "error")
        .map((message) =>
          `${String(message.lineNum)}:${String(message.linePos)} ${String(message.message)}`
        )
        .join(" | ");
      throw new Error(
        `${fixtureId} ${armLabel} shader compilation failed: ${details}`,
      );
    }
    const pipeline = await device.createComputePipelineAsync({
      label: `${fixtureId}-${armLabel}-pipeline`,
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const validation = await device.popErrorScope();
    scopeOpen = false;
    if (validation !== null) throw validation;
    return Object.freeze({
      pipeline,
      shaderSha256: await sha256Hex(source),
      messages,
    });
  } catch (error) {
    if (scopeOpen) void device.popErrorScope().catch(() => undefined);
    throw error;
  }
}

function preparedArm(
  id: AceOpt0009AccumulationArm,
  source: string,
  compiled: CompiledPipeline,
  bindGroups: readonly GPUBindGroup[],
  output: GPUBuffer,
  dispatchesPerExecution: number,
  encode: (pass: GPUComputePassEncoder) => void,
  shape: AceGemmShape,
): PreparedArm {
  const counts: MutableExecutionCounts = {
    executions: 0,
    encodedCommandBuffers: 0,
    submissions: 0,
    drains: 0,
    dispatches: 0,
  };
  return Object.freeze({
    id,
    source,
    shaderSha256: compiled.shaderSha256,
    compilationMessages: compiled.messages,
    pipeline: compiled.pipeline,
    bindGroups: Object.freeze([...bindGroups]),
    output,
    shape: Object.freeze({ ...shape }),
    dispatchesPerExecution,
    counts,
    encode,
  });
}

function currentShapeFromArm(arm: PreparedArm): AceGemmShape {
  return arm.shape;
}

function createMappedStorage(
  device: GPUDevice,
  tracker: BufferTracker,
  owned: GPUBuffer[],
  label: string,
  size: number,
): GPUBuffer {
  const buffer = tracker.create(device, {
    label,
    size,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  owned.push(buffer);
  return buffer;
}

function createOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  owned: GPUBuffer[],
  label: string,
  size: number,
): GPUBuffer {
  const output = tracker.create(device, {
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  owned.push(output);
  new Uint32Array(output.getMappedRange()).fill(OUTPUT_SENTINEL_BITS);
  output.unmap();
  return output;
}

function fillActivations(
  oracle: GPUBuffer,
  fp16: GPUBuffer,
  fixture: AceOpt0009CorrectnessFixture,
): void {
  const oracleValues = new Float32Array(oracle.getMappedRange());
  const fp16Words = new Uint32Array(fp16.getMappedRange());
  for (let row = 0; row < fixture.rows; row += 1) {
    for (let inner = 0; inner < fixture.inner; inner += 1) {
      const scalar = row * fixture.inner + inner;
      const value = aceOpt0009FixtureActivationValue(
        fixture.fixtureKind,
        row,
        inner,
      );
      oracleValues[scalar] = Math.fround(value);
      setPacked16(fp16Words, scalar, numberToFloat16Bits(value));
    }
  }
}

function fillWeights(
  oracle: GPUBuffer,
  fp16: GPUBuffer,
  fixture: AceOpt0009CorrectnessFixture,
): void {
  const oracleWords = new Uint32Array(oracle.getMappedRange());
  const fp16Words = new Uint32Array(fp16.getMappedRange());
  const innerTiles = fixture.inner / TILE_INNER;
  for (let column = 0; column < fixture.columns; column += 1) {
    for (let inner = 0; inner < fixture.inner; inner += 1) {
      const value = aceOpt0009FixtureWeightValue(
        fixture.fixtureKind,
        column,
        inner,
      );
      const oraclePhysical = oracleWeightScalarIndex(
        fixture.columns,
        innerTiles,
        column,
        inner,
      );
      setPacked16(oracleWords, oraclePhysical, numberToBf16Bits(value));
      const logical = column * fixture.inner + inner;
      const fp16Physical = aceOpt0009AdaptedWeightScalarIndex(fixture, logical);
      setPacked16(fp16Words, fp16Physical, numberToFloat16Bits(value));
    }
  }
}

function oracleWeightScalarIndex(
  columns: number,
  innerTiles: number,
  column: number,
  inner: number,
): number {
  if (columns % 128 !== 0) throw new RangeError("oracle N must divide 128");
  const columnTile = Math.floor(column / 128);
  const columnInTile = column % 128;
  const innerTile = Math.floor(inner / TILE_INNER);
  const innerInTile = inner % TILE_INNER;
  return (
    ((columnTile * innerTiles + innerTile) * TILE_INNER + innerInTile) * 128 +
    columnInTile
  );
}

async function readOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  prepared: PreparedShape,
  arm: PreparedArm,
  retainValues: boolean,
): Promise<OutputReadback> {
  const bytes = prepared.adaptedPlan.outputBytes;
  const readback = tracker.create(device, {
    label: `${prepared.fixture.id}-${arm.id}-readback`,
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder({
      label: `${prepared.fixture.id}-${arm.id}-readback-encoder`,
    });
    encoder.copyBufferToBuffer(arm.output, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    const range = readback.getMappedRange();
    const bits = new Uint32Array(range);
    const values = new Float32Array(range);
    let finiteCount = 0;
    let positiveInfinityCount = 0;
    let negativeInfinityCount = 0;
    let nanCount = 0;
    let positiveZeroCount = 0;
    let negativeZeroCount = 0;
    let nonzeroFiniteCount = 0;
    let sentinelCount = 0;
    let minimumFinite = Infinity;
    let maximumFinite = -Infinity;
    let hash = 0x811c_9dc5;
    for (let index = 0; index < bits.length; index += 1) {
      const raw = bits[index]!;
      const value = values[index]!;
      hash = Math.imul(hash ^ raw, 0x0100_0193) >>> 0;
      if (raw === OUTPUT_SENTINEL_BITS) sentinelCount += 1;
      if (Number.isFinite(value)) {
        finiteCount += 1;
        minimumFinite = Math.min(minimumFinite, value);
        maximumFinite = Math.max(maximumFinite, value);
        if (value === 0) {
          if ((raw >>> 31) === 0) positiveZeroCount += 1;
          else negativeZeroCount += 1;
        } else {
          nonzeroFiniteCount += 1;
        }
      } else if (Number.isNaN(value)) {
        nanCount += 1;
      } else if (value > 0) {
        positiveInfinityCount += 1;
      } else {
        negativeInfinityCount += 1;
      }
    }
    if (sentinelCount !== 0 || bits.length !== prepared.adaptedPlan.rows * prepared.adaptedPlan.columns) {
      throw new Error(`${prepared.fixture.id} ${arm.id} failed complete-write accounting`);
    }
    const sentinels = outputSentinelIndices(prepared.fixture).map((index) =>
      Object.freeze({
        index,
        gpu: values[index]!,
        cpuSeparateProductAdd: aceOpt0009CpuOutputValue(
          prepared.fixture,
          arm.id,
          index,
          "separate-product-add",
        ),
        cpuContractedExpression: aceOpt0009CpuOutputValue(
          prepared.fixture,
          arm.id,
          index,
          "contracted-expression",
        ),
      })
    );
    const retained = retainValues ? new Float32Array(values) : undefined;
    return Object.freeze({
      scan: Object.freeze({
        elementCount: bits.length,
        finiteCount,
        nonFiniteCount:
          positiveInfinityCount + negativeInfinityCount + nanCount,
        positiveInfinityCount,
        negativeInfinityCount,
        nanCount,
        positiveZeroCount,
        negativeZeroCount,
        nonzeroFiniteCount,
        sentinelCount,
        minimumFinite: finiteCount === 0 ? null : minimumFinite,
        maximumFinite: finiteCount === 0 ? null : maximumFinite,
        rawFnv1a32: hash.toString(16).padStart(8, "0"),
        overflowClassification:
          positiveInfinityCount + negativeInfinityCount === 0
            ? "none"
            : "observed",
        cpuSentinels: Object.freeze(sentinels),
      }),
      rawFnv1a32: hash.toString(16).padStart(8, "0"),
      ...(retained === undefined ? {} : { values: retained }),
    });
  } finally {
    if (mapped) readback.unmap();
    tracker.destroy(readback);
  }
}

function cpuDiagnostics(
  fixture: AceOpt0009CorrectnessFixture,
  arm: AceOpt0009AccumulationArm,
  gpu: Float32Array,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    separateProductAdd: compareCpuReference(
      fixture,
      arm,
      gpu,
      "separate-product-add",
    ),
    contractedExpression: compareCpuReference(
      fixture,
      arm,
      gpu,
      "contracted-expression",
    ),
    fp16OperandsFp32AccumulationReference: Object.freeze({
      separateProductAdd: compareCpuReference(
        fixture,
        "fp16-fp32-accum",
        gpu,
        "separate-product-add",
      ),
      contractedExpression: compareCpuReference(
        fixture,
        "fp16-fp32-accum",
        gpu,
        "contracted-expression",
      ),
    }),
    interpretation:
      "diagnostic only: WebGPU may contract multiply-add; no numerical threshold is applied",
  });
}

export function compareAceOpt0009GpuOutputs(
  left: Float32Array,
  right: Float32Array,
): Readonly<Record<string, unknown>> {
  if (left.length !== right.length) {
    throw new Error("OPT-0009 pairwise GPU outputs have different lengths");
  }
  let finitePairCount = 0;
  let classificationMismatchCount = 0;
  let signedZeroMismatchCount = 0;
  let bitExactCount = 0;
  let maximumAbsoluteDifference = 0;
  let maximumRelativeDifference = 0;
  let squaredDifference = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index]!;
    const rightValue = right[index]!;
    if (float32Bits(leftValue) === float32Bits(rightValue)) bitExactCount += 1;
    if (numberClass(leftValue) !== numberClass(rightValue)) {
      classificationMismatchCount += 1;
      continue;
    }
    if (
      leftValue === 0 &&
      rightValue === 0 &&
      Object.is(leftValue, -0) !== Object.is(rightValue, -0)
    ) {
      signedZeroMismatchCount += 1;
    }
    if (!Number.isFinite(leftValue) || !Number.isFinite(rightValue)) continue;
    const absolute = Math.abs(leftValue - rightValue);
    const relative = absolute / Math.max(Math.abs(leftValue), Math.abs(rightValue), 1e-30);
    finitePairCount += 1;
    maximumAbsoluteDifference = Math.max(maximumAbsoluteDifference, absolute);
    maximumRelativeDifference = Math.max(maximumRelativeDifference, relative);
    squaredDifference += absolute * absolute;
  }
  return Object.freeze({
    elementCount: left.length,
    bitExactCount,
    finitePairCount,
    classificationMismatchCount,
    signedZeroMismatchCount,
    maximumAbsoluteDifference,
    maximumRelativeDifference,
    rootMeanSquareDifference: finitePairCount === 0
      ? null
      : Math.sqrt(squaredDifference / finitePairCount),
    acceptanceThresholdApplied: false,
  });
}

function validateExecutionCounts(
  arm: PreparedArm,
  expectedExecutions: number,
): void {
  const counts = arm.counts;
  if (
    counts.executions !== expectedExecutions ||
    counts.encodedCommandBuffers !== expectedExecutions ||
    counts.submissions !== expectedExecutions ||
    counts.drains !== expectedExecutions ||
    counts.dispatches !== expectedExecutions * arm.dispatchesPerExecution
  ) {
    throw new Error(`OPT-0009 ${arm.id} execution counts do not reconcile`);
  }
}

function compareCpuReference(
  fixture: AceOpt0009CorrectnessFixture,
  arm: AceOpt0009AccumulationArm,
  gpu: Float32Array,
  mode: "separate-product-add" | "contracted-expression",
): Readonly<Record<string, unknown>> {
  let finitePairCount = 0;
  let classificationMismatchCount = 0;
  let signedZeroMismatchCount = 0;
  let maximumAbsoluteError = 0;
  let maximumRelativeError = 0;
  let squaredError = 0;
  for (let index = 0; index < gpu.length; index += 1) {
    const actual = gpu[index]!;
    const expected = aceOpt0009CpuOutputValue(fixture, arm, index, mode);
    const actualClass = numberClass(actual);
    const expectedClass = numberClass(expected);
    if (actualClass !== expectedClass) {
      classificationMismatchCount += 1;
      continue;
    }
    if (actual === 0 && expected === 0 && Object.is(actual, -0) !== Object.is(expected, -0)) {
      signedZeroMismatchCount += 1;
    }
    if (!Number.isFinite(actual) || !Number.isFinite(expected)) continue;
    const absolute = Math.abs(actual - expected);
    const relative = absolute / Math.max(Math.abs(expected), 1e-30);
    finitePairCount += 1;
    maximumAbsoluteError = Math.max(maximumAbsoluteError, absolute);
    maximumRelativeError = Math.max(maximumRelativeError, relative);
    squaredError += absolute * absolute;
  }
  return Object.freeze({
    elementCount: gpu.length,
    finitePairCount,
    classificationMismatchCount,
    signedZeroMismatchCount,
    maximumAbsoluteError,
    maximumRelativeError,
    rootMeanSquareError: finitePairCount === 0
      ? null
      : Math.sqrt(squaredError / finitePairCount),
    acceptanceThresholdApplied: false,
  });
}

async function runPostTimingCancellation(
  device: GPUDevice,
  arm: PreparedArm,
): Promise<unknown> {
  await device.queue.onSubmittedWorkDone();
  const before = Object.freeze({ ...arm.counts });
  const controller = new AbortController();
  const plannedExecutions = 3;
  let encodedExecutions = 0;
  let skippedAfterAbort = 0;
  for (let index = 0; index < plannedExecutions; index += 1) {
    if (controller.signal.aborted) {
      skippedAfterAbort += 1;
      continue;
    }
    await executeArm(
      device,
      arm,
      -3,
      "post-timing-cancellation",
      index,
    );
    encodedExecutions += 1;
    if (index === 0) controller.abort("cancel-after-first-completion-fence");
  }
  const after = Object.freeze({ ...arm.counts });
  if (
    !controller.signal.aborted ||
    encodedExecutions !== 1 ||
    skippedAfterAbort !== plannedExecutions - 1 ||
    after.executions - before.executions !== 1 ||
    after.encodedCommandBuffers - before.encodedCommandBuffers !== 1 ||
    after.submissions - before.submissions !== 1 ||
    after.drains - before.drains !== 1 ||
    after.dispatches - before.dispatches !== arm.dispatchesPerExecution
  ) {
    throw new Error("OPT-0009 post-timing cancellation did not stop encoding");
  }
  return Object.freeze({
    testedAtEpochMilliseconds: Date.now(),
    arm: arm.id,
    cancellationPoint: "after-first-completion-fence",
    plannedExecutions,
    encodedExecutions,
    skippedAfterAbort,
    signalAborted: controller.signal.aborted,
    countsBefore: before,
    countsAfter: after,
    noPostAbortEncoding: true,
  });
}

function verifyExactShaderHashes(
  fixtureId: string,
  arms: Readonly<Record<AceOpt0009AccumulationArm, PreparedArm>>,
): void {
  if (!ACE_OPT_0009_ACCUMULATION_SHAPES.some((shape) => shape.id === fixtureId)) {
    return;
  }
  for (const armId of allArms()) {
    const key = `${fixtureId}:${armId}`;
    const expected = (
      ACE_OPT_0009_EXACT_SHADER_SHA256 as Readonly<Record<string, string>>
    )[key];
    if (expected === undefined || expected === "PENDING") {
      throw new Error(`OPT-0009 exact shader hash ${key} is not frozen`);
    }
    if (arms[armId].shaderSha256 !== expected) {
      throw new Error(`OPT-0009 exact shader hash ${key} changed`);
    }
  }
}

function outputSentinelIndices(shape: AceGemmShape): readonly number[] {
  const count = shape.rows * shape.columns;
  return Object.freeze([...new Set([
    0,
    shape.columns - 1,
    shape.columns,
    Math.floor(count / 2),
    (shape.rows - 1) * shape.columns,
    count - 1,
  ].filter((index) => index >= 0 && index < count))].sort((a, b) => a - b));
}

function maximumRequiredBindingBytes(): number {
  const shapes = [
    ...ACE_OPT_0009_ACCUMULATION_SHAPES,
    ...ACE_OPT_0009_CORRECTNESS_FIXTURES,
  ];
  return Math.max(...shapes.flatMap((shape) => {
    const adapted = planAceOpt0009AdaptedGemm(shape);
    const oracle = planAceSubgroupGemm(shape);
    return [
      adapted.activationBytes,
      adapted.weightBytes,
      adapted.outputBytes,
      oracle.activationElements * 4,
      oracle.packedWeightWords * 4,
    ];
  }));
}

function requireAdapterLimits(adapter: GPUAdapter, bytes: number): void {
  if (
    adapter.limits.maxBufferSize < bytes ||
    adapter.limits.maxStorageBufferBindingSize < bytes ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE ||
    adapter.limits.maxComputeWorkgroupSizeX < WORKGROUP_SIZE
  ) {
    throw new Error("OPT-0009 adapter cannot satisfy exact three-arm resources");
  }
}

function adapterReceipt(adapter: GPUAdapter): Readonly<Record<string, unknown>> {
  return Object.freeze({
    vendor: adapter.info.vendor,
    architecture: adapter.info.architecture,
    device: adapter.info.device,
    description: adapter.info.description,
    subgroupMinSize: adapter.info.subgroupMinSize,
    subgroupMaxSize: adapter.info.subgroupMaxSize,
    features: Object.freeze([...adapter.features].sort()),
  });
}

function allArms(): readonly AceOpt0009AccumulationArm[] {
  return [
    "packed-bf16-fp32-oracle",
    "fp16-fp32-accum",
    "fp16-native-accum",
  ] as const;
}

function exactShape(
  id: string,
  rows: number,
  inner: number,
  columns: number,
): AceOpt0009AccumulationShape {
  return Object.freeze({ id, rows, inner, columns });
}

function correctnessFixture(
  id: string,
  rows: number,
  inner: number,
  fixtureKind: AceOpt0009FixtureKind,
  coverage: readonly string[],
  columns = TILE_COLUMNS,
): AceOpt0009CorrectnessFixture {
  return Object.freeze({
    id,
    rows,
    inner,
    columns,
    fixtureKind,
    coverage: Object.freeze([...coverage]),
  });
}

function setPacked16(words: Uint32Array, scalar: number, bits: number): void {
  const word = scalar >>> 1;
  if ((scalar & 1) === 0) {
    words[word] = ((words[word] ?? 0) & 0xffff_0000) | (bits & 0xffff);
  } else {
    words[word] = ((words[word] ?? 0) & 0x0000_ffff) | ((bits & 0xffff) << 16);
  }
}

function numberToBf16Bits(value: number): number {
  FLOAT32_SCRATCH[0] = value;
  const source = UINT32_SCRATCH[0]!;
  const exponent = source & 0x7f80_0000;
  const mantissa = source & 0x007f_ffff;
  if (exponent === 0x7f80_0000) {
    return mantissa === 0 ? source >>> 16 : ((source >>> 16) | 0x0040) & 0xffff;
  }
  return ((source + 0x7fff + ((source >>> 16) & 1)) >>> 16) & 0xffff;
}

function bf16BitsToNumber(bits: number): number {
  UINT32_SCRATCH[0] = (bits & 0xffff) << 16;
  return FLOAT32_SCRATCH[0]!;
}

function numberToFloat16Bits(value: number): number {
  FLOAT32_SCRATCH[0] = value;
  const bits = UINT32_SCRATCH[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7f_ffff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const normalized = mantissa | 0x80_0000;
    const shift = 14 - halfExponent;
    const truncated = normalized >>> shift;
    const remainder = normalized & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    return sign | (truncated + (
      remainder > halfway || (remainder === halfway && (truncated & 1) !== 0)
        ? 1
        : 0
    ));
  }
  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1) !== 0)) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      const nextExponent = halfExponent + 1;
      return nextExponent >= 0x1f ? sign | 0x7c00 : sign | (nextExponent << 10);
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

function float16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) {
    return mantissa === 0 ? (sign < 0 ? -0 : 0) : sign * mantissa * 2 ** -24;
  }
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

function roundToFloat16(value: number): number {
  return float16BitsToNumber(numberToFloat16Bits(value));
}

function numberClass(value: number): string {
  if (Number.isNaN(value)) return "nan";
  if (value === Infinity) return "positive-infinity";
  if (value === -Infinity) return "negative-infinity";
  if (value === 0) return "zero";
  return "finite-nonzero";
}

function float32Bits(value: number): number {
  FLOAT32_SCRATCH[0] = value;
  return UINT32_SCRATCH[0]!;
}

function medianOf(samples: readonly number[]): number {
  const sorted = [...samples].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (
    !Number.isSafeInteger(left) ||
    left <= 0 ||
    !Number.isSafeInteger(right) ||
    right <= 0 ||
    !Number.isSafeInteger(product)
  ) {
    throw new RangeError(`OPT-0009 ${label} exceeds positive safe arithmetic`);
  }
  return product;
}

function requirePositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`OPT-0009 ${label} must be a positive safe integer`);
  }
}

function requireIndex(index: number, count: number, label: string): void {
  if (!Number.isSafeInteger(index) || index < 0 || index >= count) {
    throw new RangeError(`OPT-0009 ${label} index ${index} exceeds ${count}`);
  }
}

async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function requireButton(selector: string): HTMLButtonElement {
  const button = document.querySelector<HTMLButtonElement>(selector);
  if (button === null) throw new Error(`Missing OPT-0009 button ${selector}`);
  return button;
}

function updateProgress(message: string): void {
  const progress = document.querySelector<HTMLElement>("#progress");
  if (progress !== null) progress.textContent = message;
}

function finish(status: "passed" | "failed", result: unknown): void {
  document.body.dataset.status = status;
  updateProgress(status);
  const output = document.querySelector<HTMLElement>("#result");
  if (output !== null) output.textContent = JSON.stringify(result, null, 2);
}

function errorReceipt(error: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({
    name: error instanceof Error ? error.name : "Error",
    message: error instanceof Error ? error.message : String(error),
    ...(error instanceof Error && error.stack !== undefined
      ? { stack: error.stack }
      : {}),
  });
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
}

const FLOAT32_SCRATCH = new Float32Array(1);
const UINT32_SCRATCH = new Uint32Array(FLOAT32_SCRATCH.buffer);
