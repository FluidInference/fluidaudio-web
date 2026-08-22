import {
  requireAceBindingBytes,
  requireAceDisjointOutput,
} from "./correctness-utils.js";
import type {
  AceCooperativeGemmPlan,
  AceGemmOutputRange,
  AceGemmShape,
} from "./gemm.js";

/** Benchmark-only OPT-0058 identity. This is not a package layout. */
export const ACE_OPT_0058_DENSE_INT8_DP4A_KERNEL_ID =
  "ace-opt-0058-dense-int8-dp4a-fixed32-wg128-m32-n128-v1";
export const ACE_OPT_0058_DENSE_INT8_DP4A_WEIGHT_LAYOUT =
  "ace-opt-0058-b-n128-group-packed-k-output4-lane32-i8-v1";
export const ACE_OPT_0058_DENSE_INT8_DP4A_SCALE_LAYOUT =
  "ace-opt-0058-b-scale-n128-group-lane32-output4-f32-v1";
export const ACE_OPT_0058_PACKED_DOT_LANGUAGE_FEATURE =
  "packed_4x8_integer_dot_product";
export const ACE_OPT_0058_TILE_ROWS = 32;
export const ACE_OPT_0058_TILE_COLUMNS = 128;
export const ACE_OPT_0058_WORKGROUP_SIZE = 128;
export const ACE_OPT_0058_SUBGROUP_SIZE = 32;
export const ACE_OPT_0058_QUANTIZER_MAX_WORKGROUP_SIZE = 128;
export const ACE_OPT_0058_GROUP_SIZES = Object.freeze([32, 64, 128] as const);

export type AceOpt0058GroupSize = 32 | 64 | 128;

const PACKED_VALUES = 4;
const OUTPUTS_PER_LANE = 4;
const SUBGROUPS_PER_WORKGROUP =
  ACE_OPT_0058_WORKGROUP_SIZE / ACE_OPT_0058_SUBGROUP_SIZE;
const ROWS_PER_SUBGROUP = ACE_OPT_0058_TILE_ROWS / SUBGROUPS_PER_WORKGROUP;
const MAX_DISPATCH_DIMENSION = 65_535;
const MAX_WGSL_U32 = 0xffff_ffff;
const PRODUCTION_DENSE_SHAPES = new Set([
  "2048x2048",
  "2048x1024",
  "2048x6144",
  "6144x2048",
]);

export interface AceOpt0058DenseInt8Plan extends AceCooperativeGemmPlan {
  readonly kernelId: typeof ACE_OPT_0058_DENSE_INT8_DP4A_KERNEL_ID;
  readonly weightLayout: typeof ACE_OPT_0058_DENSE_INT8_DP4A_WEIGHT_LAYOUT;
  readonly scaleLayout: typeof ACE_OPT_0058_DENSE_INT8_DP4A_SCALE_LAYOUT;
  readonly groupSize: AceOpt0058GroupSize;
  readonly tileRows: typeof ACE_OPT_0058_TILE_ROWS;
  readonly tileColumns: typeof ACE_OPT_0058_TILE_COLUMNS;
  readonly tileInner: AceOpt0058GroupSize;
  readonly workgroupSize: typeof ACE_OPT_0058_WORKGROUP_SIZE;
  readonly subgroupSize: typeof ACE_OPT_0058_SUBGROUP_SIZE;
  readonly rowTiles: number;
  readonly columnTiles: number;
  readonly innerGroups: number;
  readonly packedKPerGroup: number;
  /** Worst symmetric-int8 magnitude before the per-group i32 state resets. */
  readonly maximumIntegerPartialMagnitude: number;
  readonly activationElements: number;
  readonly packedActivationWords: number;
  readonly activationScaleElements: number;
  readonly weightElements: number;
  readonly packedWeightWords: number;
  readonly weightScaleElements: number;
  readonly outputElements: number;
  /** [N/128,K/group,group/4,output4,lane32]. */
  readonly packedWeightStorageShape: readonly [number, number, number, 4, 32];
  /** [N/128,K/group,lane32,output4]. */
  readonly weightScaleStorageShape: readonly [number, number, 32, 4];
  /** [M,K/group,group/4]. */
  readonly packedActivationStorageShape: readonly [number, number, number];
  /** [M,K/group]. */
  readonly activationScaleStorageShape: readonly [number, number];
}

export interface AceOpt0058DenseInt8Bindings {
  readonly activation: GPUBufferBinding;
  readonly packedActivation: GPUBufferBinding;
  readonly activationScale: GPUBufferBinding;
  readonly weight: GPUBufferBinding;
  readonly weightScale: GPUBufferBinding;
  /** One u32 atomic. Nonzero means a non-finite activation was rejected. */
  readonly quantizationStatus: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceOpt0058DenseInt8Dispatch {
  readonly label: string;
  readonly kernelId: typeof ACE_OPT_0058_DENSE_INT8_DP4A_KERNEL_ID;
  readonly weightLayout: typeof ACE_OPT_0058_DENSE_INT8_DP4A_WEIGHT_LAYOUT;
  readonly scaleLayout: typeof ACE_OPT_0058_DENSE_INT8_DP4A_SCALE_LAYOUT;
  readonly groupSize: AceOpt0058GroupSize;
  readonly plan: AceOpt0058DenseInt8Plan;
  /** Quantization must be encoded immediately before contraction. */
  encodeQuantize(pass: GPUComputePassEncoder): void;
  encodeContraction(pass: GPUComputePassEncoder): void;
  /** Conservative product boundary: exactly one quantization per GEMM. */
  encodeComplete(pass: GPUComputePassEncoder): void;
  /** Ceiling only. The caller must have populated the quantized activation. */
  encodePrequantized(pass: GPUComputePassEncoder): void;
}

export interface AceOpt0058QuantizedWeight {
  readonly groupSize: AceOpt0058GroupSize;
  readonly inner: number;
  readonly columns: number;
  readonly packed: Uint32Array<ArrayBuffer>;
  readonly scales: Float32Array<ArrayBuffer>;
  readonly saturationCount: number;
  readonly zeroGroupCount: number;
}

export class AceOpt0058DenseInt8Dp4aKernel {
  private readonly compiled = new Map<
    string,
    Promise<Readonly<{ quantize: GPUComputePipeline; contraction: GPUComputePipeline }>>
  >();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
      packed4x8IntegerDotProduct?: boolean;
    }>,
  ): AceOpt0058DenseInt8Dp4aKernel {
    if (
      !device.features.has("subgroups") ||
      capability.subgroupMinSize !== ACE_OPT_0058_SUBGROUP_SIZE ||
      capability.subgroupMaxSize !== ACE_OPT_0058_SUBGROUP_SIZE ||
      capability.packed4x8IntegerDotProduct !== true
    ) {
      throw new Error(
        "OPT-0058 DP4a requires fixed 32-lane subgroups and the packed_4x8_integer_dot_product WGSL language feature",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0058_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX < ACE_OPT_0058_WORKGROUP_SIZE
    ) {
      throw new Error(`OPT-0058 DP4a requires WG${ACE_OPT_0058_WORKGROUP_SIZE}`);
    }
    return new AceOpt0058DenseInt8Dp4aKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    groupSize: AceOpt0058GroupSize,
    bindings: AceOpt0058DenseInt8Bindings,
  ): Promise<AceOpt0058DenseInt8Dispatch> {
    if (this.destroyed) throw new Error("OPT-0058 DP4a kernel was destroyed");
    const plan = planAceOpt0058DenseInt8(shape, groupSize);
    const activationBytes = checkedBytes(plan.activationElements, 4, "activation");
    const packedActivationBytes = checkedBytes(
      plan.packedActivationWords,
      4,
      "packed activation",
    );
    const activationScaleBytes = checkedBytes(
      plan.activationScaleElements,
      4,
      "activation scale",
    );
    const weightBytes = checkedBytes(plan.packedWeightWords, 4, "weight");
    const weightScaleBytes = checkedBytes(
      plan.weightScaleElements,
      4,
      "weight scale",
    );
    const outputBytes = checkedBytes(plan.outputElements, 4, "output");
    requireAceBindingBytes(bindings.activation, activationBytes, `${label} activation`);
    requireAceBindingBytes(
      bindings.packedActivation,
      packedActivationBytes,
      `${label} packed activation`,
    );
    requireAceBindingBytes(
      bindings.activationScale,
      activationScaleBytes,
      `${label} activation scale`,
    );
    requireAceBindingBytes(bindings.weight, weightBytes, `${label} weight`);
    requireAceBindingBytes(
      bindings.weightScale,
      weightScaleBytes,
      `${label} weight scale`,
    );
    requireAceBindingBytes(
      bindings.quantizationStatus,
      4,
      `${label} quantization status`,
    );
    requireAceBindingBytes(bindings.output, outputBytes, `${label} output`);
    requireAceDisjointOutput(
      exactBinding(bindings.output, outputBytes),
      [
        exactBinding(bindings.activation, activationBytes),
        exactBinding(bindings.packedActivation, packedActivationBytes),
        exactBinding(bindings.activationScale, activationScaleBytes),
        exactBinding(bindings.weight, weightBytes),
        exactBinding(bindings.weightScale, weightScaleBytes),
        exactBinding(bindings.quantizationStatus, 4),
      ],
      label,
    );
    const pipelines = await this.pipelinesFor(shape, groupSize);
    if (this.destroyed) {
      throw new Error("OPT-0058 DP4a kernel was destroyed while compiling");
    }
    const quantizeBindings = this.device.createBindGroup({
      label: `${label}-opt0058-quantize-bindings`,
      layout: pipelines.quantize.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: exactBinding(bindings.activation, activationBytes) },
        {
          binding: 1,
          resource: exactBinding(bindings.packedActivation, packedActivationBytes),
        },
        {
          binding: 2,
          resource: exactBinding(bindings.activationScale, activationScaleBytes),
        },
        { binding: 3, resource: exactBinding(bindings.quantizationStatus, 4) },
      ],
    });
    const contractionBindings = this.device.createBindGroup({
      label: `${label}-opt0058-contraction-bindings`,
      layout: pipelines.contraction.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: exactBinding(bindings.packedActivation, packedActivationBytes),
        },
        {
          binding: 1,
          resource: exactBinding(bindings.activationScale, activationScaleBytes),
        },
        { binding: 2, resource: exactBinding(bindings.weight, weightBytes) },
        {
          binding: 3,
          resource: exactBinding(bindings.weightScale, weightScaleBytes),
        },
        { binding: 4, resource: exactBinding(bindings.quantizationStatus, 4) },
        { binding: 5, resource: exactBinding(bindings.output, outputBytes) },
      ],
    });
    const requireLive = (): void => {
      if (this.destroyed) {
        throw new Error(`OPT-0058 ${label} dispatch was destroyed`);
      }
    };
    const encodeQuantize = (pass: GPUComputePassEncoder): void => {
      requireLive();
      pass.setPipeline(pipelines.quantize);
      pass.setBindGroup(0, quantizeBindings);
      pass.dispatchWorkgroups(plan.innerGroups, plan.rows, 1);
    };
    const encodeContraction = (pass: GPUComputePassEncoder): void => {
      requireLive();
      pass.setPipeline(pipelines.contraction);
      pass.setBindGroup(0, contractionBindings);
      pass.dispatchWorkgroups(plan.columnTiles, plan.rowTiles, 1);
    };
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0058_DENSE_INT8_DP4A_KERNEL_ID,
      weightLayout: ACE_OPT_0058_DENSE_INT8_DP4A_WEIGHT_LAYOUT,
      scaleLayout: ACE_OPT_0058_DENSE_INT8_DP4A_SCALE_LAYOUT,
      groupSize,
      plan,
      encodeQuantize,
      encodeContraction,
      encodeComplete(pass: GPUComputePassEncoder): void {
        encodeQuantize(pass);
        encodeContraction(pass);
      },
      encodePrequantized(pass: GPUComputePassEncoder): void {
        encodeContraction(pass);
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.compiled.clear();
  }

  private pipelinesFor(
    shape: AceGemmShape,
    groupSize: AceOpt0058GroupSize,
  ): Promise<Readonly<{ quantize: GPUComputePipeline; contraction: GPUComputePipeline }>> {
    const key = `${shape.rows}x${shape.inner}x${shape.columns}-g${groupSize}`;
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceOpt0058DenseInt8(this.device, shape, groupSize);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

export function planAceOpt0058DenseInt8(
  shape: AceGemmShape,
  groupSize: AceOpt0058GroupSize,
): AceOpt0058DenseInt8Plan {
  const { rows, inner, columns } = shape;
  requirePositiveSafeInteger(rows, "rows");
  requirePositiveSafeInteger(inner, "inner");
  requirePositiveSafeInteger(columns, "columns");
  requireGroupSize(groupSize);
  if (!PRODUCTION_DENSE_SHAPES.has(`${inner}x${columns}`)) {
    throw new RangeError(
      `OPT-0058 DP4a rejects non-production K${inner}/N${columns}`,
    );
  }
  if (inner % groupSize !== 0) {
    throw new RangeError(`OPT-0058 DP4a requires K divisible by group ${groupSize}`);
  }
  if (columns % ACE_OPT_0058_TILE_COLUMNS !== 0) {
    throw new RangeError("OPT-0058 DP4a requires N divisible by 128");
  }
  const rowTiles = Math.ceil(rows / ACE_OPT_0058_TILE_ROWS);
  const columnTiles = columns / ACE_OPT_0058_TILE_COLUMNS;
  const innerGroups = inner / groupSize;
  const packedKPerGroup = groupSize / PACKED_VALUES;
  if (
    rowTiles > MAX_DISPATCH_DIMENSION ||
    columnTiles > MAX_DISPATCH_DIMENSION ||
    innerGroups > MAX_DISPATCH_DIMENSION ||
    rows > MAX_DISPATCH_DIMENSION
  ) {
    throw new RangeError("OPT-0058 DP4a exceeds WebGPU dispatch dimensions");
  }
  const activationElements = checkedProduct(rows, inner, "activation");
  const packedActivationWords = activationElements / PACKED_VALUES;
  const activationScaleElements = checkedProduct(rows, innerGroups, "activation scales");
  const weightElements = checkedProduct(inner, columns, "weight");
  const packedWeightWords = weightElements / PACKED_VALUES;
  const weightScaleElements = checkedProduct(columns, innerGroups, "weight scales");
  const outputElements = checkedProduct(rows, columns, "output");
  for (const [elements, label] of [
    [activationElements, "activation"],
    [packedActivationWords, "packed activation"],
    [activationScaleElements, "activation scales"],
    [weightElements, "weight"],
    [packedWeightWords, "packed weight"],
    [weightScaleElements, "weight scales"],
    [outputElements, "output"],
  ] as const) requireWgslIndexable(elements, label);
  const workgroupCount = checkedProduct(rowTiles, columnTiles, "workgroups");
  const scheduledRows = checkedProduct(rowTiles, ACE_OPT_0058_TILE_ROWS, "scheduled rows");
  const scheduledMultiplyAdds = checkedProduct(
    checkedProduct(scheduledRows, inner, "scheduled row-inner"),
    columns,
    "scheduled multiply-adds",
  );
  const outputRanges: readonly AceGemmOutputRange[] = Object.freeze([
    Object.freeze({
      firstOutput: 0,
      outputCount: outputElements,
      firstWorkgroup: 0,
      workgroupCount,
      multiplyAdds: scheduledMultiplyAdds,
    }),
  ]);
  return Object.freeze({
    kernelId: ACE_OPT_0058_DENSE_INT8_DP4A_KERNEL_ID,
    weightLayout: ACE_OPT_0058_DENSE_INT8_DP4A_WEIGHT_LAYOUT,
    scaleLayout: ACE_OPT_0058_DENSE_INT8_DP4A_SCALE_LAYOUT,
    rows,
    inner,
    columns,
    workgroupsX: columnTiles,
    workgroupsY: rowTiles,
    tileRows: ACE_OPT_0058_TILE_ROWS,
    tileColumns: ACE_OPT_0058_TILE_COLUMNS,
    tileInner: groupSize,
    workgroupSize: ACE_OPT_0058_WORKGROUP_SIZE,
    subgroupSize: ACE_OPT_0058_SUBGROUP_SIZE,
    groupSize,
    rowTiles,
    columnTiles,
    innerGroups,
    packedKPerGroup,
    maximumIntegerPartialMagnitude: 127 * 127 * groupSize,
    activationElements,
    packedActivationWords,
    activationScaleElements,
    weightElements,
    packedWeightWords,
    weightScaleElements,
    outputElements,
    outputRangeCount: 1,
    outputRanges,
    packedWeightStorageShape: Object.freeze([
      columnTiles,
      innerGroups,
      packedKPerGroup,
      OUTPUTS_PER_LANE,
      ACE_OPT_0058_SUBGROUP_SIZE,
    ]) as readonly [number, number, number, 4, 32],
    weightScaleStorageShape: Object.freeze([
      columnTiles,
      innerGroups,
      ACE_OPT_0058_SUBGROUP_SIZE,
      OUTPUTS_PER_LANE,
    ]) as readonly [number, number, 32, 4],
    packedActivationStorageShape: Object.freeze([
      rows,
      innerGroups,
      packedKPerGroup,
    ]) as readonly [number, number, number],
    activationScaleStorageShape: Object.freeze([
      rows,
      innerGroups,
    ]) as readonly [number, number],
  });
}

export function quantizeAndPackAceOpt0058DenseWeight(
  logical: Float32Array,
  inner: number,
  columns: number,
  groupSize: AceOpt0058GroupSize,
): AceOpt0058QuantizedWeight {
  const plan = planPack(inner, columns, groupSize);
  if (logical.length !== plan.weightElements) {
    throw new RangeError(
      `OPT-0058 weight pack expected ${plan.weightElements} values, got ${logical.length}`,
    );
  }
  const packed = new Uint32Array(plan.packedWeightWords);
  const scales = new Float32Array(plan.weightScaleElements);
  let saturationCount = 0;
  let zeroGroupCount = 0;
  for (let column = 0; column < columns; column += 1) {
    for (let group = 0; group < plan.innerGroups; group += 1) {
      let maximum = 0;
      const innerBase = group * groupSize;
      for (let offset = 0; offset < groupSize; offset += 1) {
        const value = logical[(innerBase + offset) * columns + column]!;
        if (!Number.isFinite(value)) {
          throw new RangeError(
            `OPT-0058 weight pack rejects non-finite value at K${innerBase + offset}/N${column}`,
          );
        }
        maximum = Math.max(maximum, Math.abs(value));
      }
      const scale = maximum === 0 ? 0 : Math.fround(maximum / 127);
      scales[aceOpt0058WeightScaleIndex(column, group, inner, columns, groupSize)] =
        scale;
      if (scale === 0) zeroGroupCount += 1;
      for (let packedK = 0; packedK < plan.packedKPerGroup; packedK += 1) {
        const bytes = [0, 1, 2, 3].map((byte) => {
          const innerIndex = innerBase + packedK * PACKED_VALUES + byte;
          const value = logical[innerIndex * columns + column]!;
          const quantized = scale === 0 ? 0 : symmetricInt8(value, scale);
          if (Math.abs(quantized) === 127) saturationCount += 1;
          return quantized;
        });
        packed[aceOpt0058PackedWeightWordIndex(
          group,
          packedK,
          column,
          inner,
          columns,
          groupSize,
        )] = packAceOpt0058SignedI8x4(bytes[0]!, bytes[1]!, bytes[2]!, bytes[3]!);
      }
    }
  }
  return Object.freeze({
    groupSize,
    inner,
    columns,
    packed,
    scales,
    saturationCount,
    zeroGroupCount,
  });
}

/** Returns logical row-major B[K,N] quantized signed bytes. */
export function unpackAceOpt0058DenseWeightI8(
  packed: Uint32Array,
  inner: number,
  columns: number,
  groupSize: AceOpt0058GroupSize,
): Int8Array<ArrayBuffer> {
  const plan = planPack(inner, columns, groupSize);
  if (packed.length !== plan.packedWeightWords) {
    throw new RangeError(
      `OPT-0058 weight unpack expected ${plan.packedWeightWords} words, got ${packed.length}`,
    );
  }
  const logical = new Int8Array(plan.weightElements);
  for (let k = 0; k < inner; k += 1) {
    for (let column = 0; column < columns; column += 1) {
      const group = Math.floor(k / groupSize);
      const withinGroup = k % groupSize;
      const packedK = Math.floor(withinGroup / PACKED_VALUES);
      const byte = withinGroup % PACKED_VALUES;
      const word = packed[aceOpt0058PackedWeightWordIndex(
        group,
        packedK,
        column,
        inner,
        columns,
        groupSize,
      )]!;
      logical[k * columns + column] = unpackAceOpt0058SignedI8(word, byte);
    }
  }
  return logical;
}

export function aceOpt0058PackedWeightWordIndex(
  group: number,
  packedK: number,
  column: number,
  inner: number,
  columns: number,
  groupSize: AceOpt0058GroupSize,
): number {
  const plan = planPack(inner, columns, groupSize);
  requireCoordinate(group, plan.innerGroups, "weight group");
  requireCoordinate(packedK, plan.packedKPerGroup, "packed K");
  requireCoordinate(column, columns, "weight column");
  const columnTile = Math.floor(column / ACE_OPT_0058_TILE_COLUMNS);
  const columnInTile = column % ACE_OPT_0058_TILE_COLUMNS;
  const outputInLane = columnInTile % OUTPUTS_PER_LANE;
  const lane = Math.floor(columnInTile / OUTPUTS_PER_LANE);
  return ((((
    columnTile * plan.innerGroups + group
  ) * plan.packedKPerGroup + packedK) * OUTPUTS_PER_LANE + outputInLane) *
    ACE_OPT_0058_SUBGROUP_SIZE) + lane;
}

export function aceOpt0058WeightScaleIndex(
  column: number,
  group: number,
  inner: number,
  columns: number,
  groupSize: AceOpt0058GroupSize,
): number {
  const plan = planPack(inner, columns, groupSize);
  requireCoordinate(column, columns, "scale column");
  requireCoordinate(group, plan.innerGroups, "scale group");
  const columnTile = Math.floor(column / ACE_OPT_0058_TILE_COLUMNS);
  const columnInTile = column % ACE_OPT_0058_TILE_COLUMNS;
  const outputInLane = columnInTile % OUTPUTS_PER_LANE;
  const lane = Math.floor(columnInTile / OUTPUTS_PER_LANE);
  return (((columnTile * plan.innerGroups + group) *
    ACE_OPT_0058_SUBGROUP_SIZE + lane) * OUTPUTS_PER_LANE) + outputInLane;
}

export function packAceOpt0058SignedI8x4(
  x: number,
  y: number,
  z: number,
  w: number,
): number {
  const values = [x, y, z, w];
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < -127 || value > 127) {
      throw new RangeError("OPT-0058 symmetric signed byte must be in [-127,127]");
    }
  }
  return (
    (x & 0xff) |
    ((y & 0xff) << 8) |
    ((z & 0xff) << 16) |
    ((w & 0xff) << 24)
  ) >>> 0;
}

export function unpackAceOpt0058SignedI8(word: number, byte: number): number {
  if (!Number.isSafeInteger(word) || word < 0 || word > 0xffff_ffff) {
    throw new RangeError("OPT-0058 packed word must be u32");
  }
  if (!Number.isSafeInteger(byte) || byte < 0 || byte >= PACKED_VALUES) {
    throw new RangeError("OPT-0058 packed byte index is out of bounds");
  }
  return (word << (24 - byte * 8)) >> 24;
}

export function aceOpt0058DynamicQuantizerWgsl(
  shape: AceGemmShape,
  groupSize: AceOpt0058GroupSize,
): string {
  const plan = planAceOpt0058DenseInt8(shape, groupSize);
  return /* wgsl */ `
enable subgroups;

const ROWS = ${plan.rows}u;
const INNER = ${plan.inner}u;
const INNER_GROUPS = ${plan.innerGroups}u;
const GROUP_SIZE = ${groupSize}u;
const PACKED_K_PER_GROUP = ${plan.packedKPerGroup}u;
const SUBGROUP_COUNT = ${groupSize / ACE_OPT_0058_SUBGROUP_SIZE}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed_activation: array<u32>;
@group(0) @binding(2) var<storage, read_write> activation_scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>, 1>;

var<workgroup> subgroup_maximum: array<f32, 4>;
var<workgroup> group_scale: f32;

fn finite_or_zero(value: f32) -> f32 {
  if (value == value && abs(value) <= 3.402823466e+38) {
    return value;
  }
  return 0.0;
}

fn quantize(value: f32, scale: f32) -> i32 {
  if (scale == 0.0) {
    return 0;
  }
  return clamp(i32(round(value / scale)), -127, 127);
}

fn packed_byte(value: i32, shift: u32) -> u32 {
  return (u32(value) & 0xffu) << shift;
}

@compute @workgroup_size(${groupSize}, 1, 1)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != ${ACE_OPT_0058_SUBGROUP_SIZE}u ||
    group.x >= INNER_GROUPS || group.y >= ROWS || group.z != 0u) {
    return;
  }
  let input_base = group.y * INNER + group.x * GROUP_SIZE;
  let raw = activation[input_base + lane];
  if (!(raw == raw && abs(raw) <= 3.402823466e+38)) {
    atomicAdd(&status[0], 1u);
  }
  let value = finite_or_zero(raw);
  let local_maximum = subgroupMax(abs(value));
  if (subgroup_lane == 0u) {
    subgroup_maximum[subgroup] = local_maximum;
  }
  workgroupBarrier();
  if (lane == 0u) {
    var maximum = 0.0;
    for (var index = 0u; index < SUBGROUP_COUNT; index += 1u) {
      maximum = max(maximum, subgroup_maximum[index]);
    }
    group_scale = select(0.0, maximum / 127.0, maximum > 0.0);
    activation_scale[group.y * INNER_GROUPS + group.x] = group_scale;
  }
  workgroupBarrier();
  if (lane < PACKED_K_PER_GROUP) {
    let k = input_base + lane * 4u;
    let q0 = quantize(finite_or_zero(activation[k]), group_scale);
    let q1 = quantize(finite_or_zero(activation[k + 1u]), group_scale);
    let q2 = quantize(finite_or_zero(activation[k + 2u]), group_scale);
    let q3 = quantize(finite_or_zero(activation[k + 3u]), group_scale);
    packed_activation[
      (group.y * INNER_GROUPS + group.x) * PACKED_K_PER_GROUP + lane
    ] = packed_byte(q0, 0u) | packed_byte(q1, 8u) |
      packed_byte(q2, 16u) | packed_byte(q3, 24u);
  }
}
`;
}

export function aceOpt0058DenseInt8Dp4aWgsl(
  shape: AceGemmShape,
  groupSize: AceOpt0058GroupSize,
): string {
  const plan = planAceOpt0058DenseInt8(shape, groupSize);
  const declarations = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `  var acc${row} = vec4<f32>(0.0);`,
  ).join("\n");
  const partialDeclarations = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `    var partial${row} = vec4<i32>(0);`,
  ).join("\n");
  const activationBroadcasts = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `      let a${row} = subgroupBroadcast(lane_a, ${row}u);`,
  ).join("\n");
  const contractions = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `      partial${row} += vec4<i32>(
        dot4I8Packed(a${row}, b0),
        dot4I8Packed(a${row}, b1),
        dot4I8Packed(a${row}, b2),
        dot4I8Packed(a${row}, b3)
      );`,
  ).join("\n");
  const scaledUpdates = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `    let a_scale${row} = subgroupBroadcast(lane_scale, ${row}u);
    acc${row} += vec4<f32>(partial${row}) *
      (vec4<f32>(a_scale${row}) * weight_group_scale);`,
  ).join("\n");
  const stores = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `
  {
    let row = row_base + ${row}u;
    if (row < ROWS) {
      output[row * (COLUMNS / 4u) + column_vector] = select(
        acc${row},
        bitcast<vec4<f32>>(vec4<u32>(failure_bits)),
        failed,
      );
    }
  }`,
  ).join("\n");
  return /* wgsl */ `
enable subgroups;
requires ${ACE_OPT_0058_PACKED_DOT_LANGUAGE_FEATURE};

const ROWS = ${plan.rows}u;
const COLUMNS = ${plan.columns}u;
const INNER_GROUPS = ${plan.innerGroups}u;
const PACKED_K_PER_GROUP = ${plan.packedKPerGroup}u;

@group(0) @binding(0) var<storage, read> packed_activation: array<u32>;
@group(0) @binding(1) var<storage, read> activation_scale: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;
@group(0) @binding(3) var<storage, read> weight_scale: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> status: array<atomic<u32>, 1>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(${ACE_OPT_0058_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (
    subgroup_size != ${ACE_OPT_0058_SUBGROUP_SIZE}u ||
    group.x >= ${plan.columnTiles}u ||
    group.y >= ${plan.rowTiles}u ||
    group.z != 0u ||
    subgroup >= ${SUBGROUPS_PER_WORKGROUP}u
  ) {
    return;
  }
  let row_base = group.y * ${ACE_OPT_0058_TILE_ROWS}u +
    subgroup * ${ROWS_PER_SUBGROUP}u;
  let column_vector = group.x * ${ACE_OPT_0058_SUBGROUP_SIZE}u + subgroup_lane;
  let lane_row = row_base + subgroup_lane;
  let failed = atomicLoad(&status[0]) != 0u;
  // Keep the required qNaN payload runtime-dependent. Tint correctly rejects a
  // constant-expression NaN as an unrepresentable f32 shader-creation value.
  let failure_bits = 0x7fc05800u | (atomicLoad(&status[0]) & 0xffu);
${declarations}

  for (var inner_group = 0u; inner_group < INNER_GROUPS; inner_group += 1u) {
${partialDeclarations}
    var lane_scale = 0.0;
    if (subgroup_lane < ${ROWS_PER_SUBGROUP}u && lane_row < ROWS) {
      lane_scale = activation_scale[lane_row * INNER_GROUPS + inner_group];
    }
    let weight_group_scale = weight_scale[
      (group.x * INNER_GROUPS + inner_group) *
      ${ACE_OPT_0058_SUBGROUP_SIZE}u + subgroup_lane
    ];
    for (var packed_k = 0u; packed_k < PACKED_K_PER_GROUP; packed_k += 1u) {
      var lane_a = 0u;
      if (subgroup_lane < ${ROWS_PER_SUBGROUP}u && lane_row < ROWS) {
        lane_a = packed_activation[
          (lane_row * INNER_GROUPS + inner_group) *
          PACKED_K_PER_GROUP + packed_k
        ];
      }
      let weight_base = ((((group.x * INNER_GROUPS + inner_group) *
        PACKED_K_PER_GROUP + packed_k) * ${OUTPUTS_PER_LANE}u) *
        ${ACE_OPT_0058_SUBGROUP_SIZE}u) + subgroup_lane;
      let b0 = weight[weight_base];
      let b1 = weight[weight_base + ${ACE_OPT_0058_SUBGROUP_SIZE}u];
      let b2 = weight[weight_base + ${2 * ACE_OPT_0058_SUBGROUP_SIZE}u];
      let b3 = weight[weight_base + ${3 * ACE_OPT_0058_SUBGROUP_SIZE}u];
${activationBroadcasts}
${contractions}
    }
${scaledUpdates}
  }
${stores}
}
`;
}

async function compileAceOpt0058DenseInt8(
  device: GPUDevice,
  shape: AceGemmShape,
  groupSize: AceOpt0058GroupSize,
): Promise<Readonly<{ quantize: GPUComputePipeline; contraction: GPUComputePipeline }>> {
  const label = `ace-opt-0058-dp4a-${shape.rows}x${shape.inner}x${shape.columns}-g${groupSize}`;
  const quantizeModule = device.createShaderModule({
    label: `${label}-dynamic-quantizer`,
    code: aceOpt0058DynamicQuantizerWgsl(shape, groupSize),
  });
  const contractionModule = device.createShaderModule({
    label: `${label}-contraction`,
    code: aceOpt0058DenseInt8Dp4aWgsl(shape, groupSize),
  });
  const [quantizeInfo, contractionInfo] = await Promise.all([
    quantizeModule.getCompilationInfo(),
    contractionModule.getCompilationInfo(),
  ]);
  requireCleanCompilation(`${label} dynamic quantizer`, quantizeInfo);
  requireCleanCompilation(`${label} contraction`, contractionInfo);
  const [quantize, contraction] = await Promise.all([
    device.createComputePipelineAsync({
      label: `${label}-dynamic-quantizer`,
      layout: "auto",
      compute: { module: quantizeModule, entryPoint: "main" },
    }),
    device.createComputePipelineAsync({
      label: `${label}-contraction`,
      layout: "auto",
      compute: { module: contractionModule, entryPoint: "main" },
    }),
  ]);
  return Object.freeze({ quantize, contraction });
}

function requireCleanCompilation(label: string, info: GPUCompilationInfo): void {
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length === 0) return;
  throw new Error(
    `${label} WGSL compilation failed:\n` + errors.map((message) =>
      `${message.lineNum}:${message.linePos} ${message.message}`
    ).join("\n"),
  );
}

function planPack(
  inner: number,
  columns: number,
  groupSize: AceOpt0058GroupSize,
): Readonly<{
  innerGroups: number;
  packedKPerGroup: number;
  weightElements: number;
  packedWeightWords: number;
  weightScaleElements: number;
}> {
  requirePositiveSafeInteger(inner, "pack inner");
  requirePositiveSafeInteger(columns, "pack columns");
  requireGroupSize(groupSize);
  if (inner % groupSize !== 0 || columns % ACE_OPT_0058_TILE_COLUMNS !== 0) {
    throw new RangeError("OPT-0058 weight pack requires K%group=0 and N%128=0");
  }
  const weightElements = checkedProduct(inner, columns, "pack elements");
  return Object.freeze({
    innerGroups: inner / groupSize,
    packedKPerGroup: groupSize / PACKED_VALUES,
    weightElements,
    packedWeightWords: weightElements / PACKED_VALUES,
    weightScaleElements: checkedProduct(columns, inner / groupSize, "pack scales"),
  });
}

function symmetricInt8(value: number, scale: number): number {
  const normalized = Math.fround(value / scale);
  const rounded = Math.sign(normalized) * Math.floor(Math.abs(normalized) + 0.5);
  return Math.max(-127, Math.min(127, rounded));
}

function exactBinding(
  binding: GPUBufferBinding,
  requiredBytes: number,
): GPUBufferBinding {
  return Object.freeze({
    buffer: binding.buffer,
    offset: binding.offset ?? 0,
    size: requiredBytes,
  });
}

function requireGroupSize(value: number): asserts value is AceOpt0058GroupSize {
  if (!(ACE_OPT_0058_GROUP_SIZES as readonly number[]).includes(value)) {
    throw new RangeError("OPT-0058 group size must be 32, 64, or 128");
  }
}

function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`OPT-0058 ${label} must be a positive safe integer`);
  }
}

function requireCoordinate(value: number, extent: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= extent) {
    throw new RangeError(`OPT-0058 ${label} ${value} is out of bounds`);
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(`OPT-0058 ${label} is not a safe integer`);
  }
  return product;
}

function checkedBytes(elements: number, itemBytes: number, label: string): number {
  return checkedProduct(elements, itemBytes, `${label} bytes`);
}

function requireWgslIndexable(elements: number, label: string): void {
  if (elements > MAX_WGSL_U32) {
    throw new RangeError(`OPT-0058 ${label} exceeds WGSL's u32 indexing domain`);
  }
}
