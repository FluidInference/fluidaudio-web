import {
  requireAceBindingBytes,
  requireAceDisjointOutput,
} from "./correctness-utils.js";
import type {
  AceCooperativeGemmPlan,
  AceGemmBufferBindings,
  AceGemmOutputRange,
  AceGemmShape,
} from "./gemm.js";

/** Benchmark-only OPT-0032 identity; this is not a model-package layout. */
export const ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID =
  "ace-opt-0032-dense-fp16-k4-partials-fixed32-wg128-m32-n128-v1";
export const ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT =
  "ace-opt-0032-b-n128-k4-output4-lane32-k4-v1";
export const ACE_OPT_0074_DENSE_K2_PARTIALS_KERNEL_ID =
  "ace-opt-0074-dense-fp16-k2-partials-fixed32-wg128-m32-n128-v1";
export const ACE_OPT_0074_DENSE_K2_PARTIALS_WEIGHT_LAYOUT =
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT;
export const ACE_OPT_0032_DENSE_TILE_ROWS = 32;
export const ACE_OPT_0032_DENSE_TILE_COLUMNS = 128;
export const ACE_OPT_0032_DENSE_TILE_INNER = 4;
export const ACE_OPT_0032_DENSE_WORKGROUP_SIZE = 128;
export const ACE_OPT_0032_DENSE_SUBGROUP_SIZE = 32;

const SUBGROUPS_PER_WORKGROUP =
  ACE_OPT_0032_DENSE_WORKGROUP_SIZE / ACE_OPT_0032_DENSE_SUBGROUP_SIZE;
const ROWS_PER_SUBGROUP =
  ACE_OPT_0032_DENSE_TILE_ROWS / SUBGROUPS_PER_WORKGROUP;
const COLUMNS_PER_LANE =
  ACE_OPT_0032_DENSE_TILE_COLUMNS / ACE_OPT_0032_DENSE_SUBGROUP_SIZE;
const MAX_DISPATCH_DIMENSION = 65_535;
const MAX_WGSL_U32 = 0xffff_ffff;

const PRODUCTION_DENSE_SHAPES = new Set([
  "2048x2048",
  "2048x1024",
  "2048x6144",
  "6144x2048",
]);

export interface AceOpt0032DenseK4PartialsPlan
  extends AceCooperativeGemmPlan {
  readonly tileRows: typeof ACE_OPT_0032_DENSE_TILE_ROWS;
  readonly tileColumns: typeof ACE_OPT_0032_DENSE_TILE_COLUMNS;
  readonly tileInner: typeof ACE_OPT_0032_DENSE_TILE_INNER;
  readonly workgroupSize: typeof ACE_OPT_0032_DENSE_WORKGROUP_SIZE;
  readonly subgroupSize: typeof ACE_OPT_0032_DENSE_SUBGROUP_SIZE;
  readonly rowTiles: number;
  readonly columnTiles: number;
  readonly innerK4Groups: number;
  readonly workgroupCount: number;
  readonly activationElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  /** [N/128, K/4, four outputs per lane, 32 lanes, four K values]. */
  readonly packedWeightStorageShape: readonly [
    number,
    number,
    4,
    32,
    4,
  ];
}

export interface AceOpt0032DenseK4PartialsDispatch {
  readonly label: string;
  readonly kernelId: typeof ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID;
  readonly weightLayout: typeof ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT;
  readonly plan: AceOpt0032DenseK4PartialsPlan;
  readonly rangeCount: 1;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Measured barrier-free K4-partial owner. OPT-0037 binds this immutable
 * arithmetic through a production package-layout adapter. Each native FP16
 * dot is widened exactly once into a vec4<f32>; running state stays FP32.
 */
export class AceOpt0032DenseK4PartialsKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0032DenseK4PartialsKernel {
    if (
      !device.features.has("shader-f16") ||
      !device.features.has("subgroups") ||
      capability.subgroupMinSize !== ACE_OPT_0032_DENSE_SUBGROUP_SIZE ||
      capability.subgroupMaxSize !== ACE_OPT_0032_DENSE_SUBGROUP_SIZE
    ) {
      throw new Error(
        "OPT-0032 dense K4 partials require shader-f16 and fixed 32-lane subgroups",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0032_DENSE_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX <
        ACE_OPT_0032_DENSE_WORKGROUP_SIZE
    ) {
      throw new Error(
        `OPT-0032 dense K4 partials require WG${ACE_OPT_0032_DENSE_WORKGROUP_SIZE}`,
      );
    }
    return new AceOpt0032DenseK4PartialsKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0032DenseK4PartialsDispatch> {
    if (this.destroyed) {
      throw new Error("OPT-0032 dense K4 partials kernel was destroyed");
    }
    if (bindings.bias !== undefined) {
      throw new Error("OPT-0032 repeated-layer dense GEMMs do not accept bias");
    }
    const plan = planAceOpt0032DenseK4Partials(shape);
    const activationBytes = checkedBytes(plan.activationElements, 4, "activation");
    const weightBytes = checkedBytes(plan.weightElements, 2, "weight");
    const outputBytes = checkedBytes(plan.outputElements, 4, "output");
    requireAceBindingBytes(bindings.activation, activationBytes, `${label} activation`);
    requireAceBindingBytes(bindings.weight, weightBytes, `${label} weight`);
    requireAceBindingBytes(bindings.output, outputBytes, `${label} output`);
    requireAceDisjointOutput(
      exactBinding(bindings.output, outputBytes),
      [
        exactBinding(bindings.activation, activationBytes),
        exactBinding(bindings.weight, weightBytes),
      ],
      label,
    );
    const pipeline = await this.pipelineFor(shape);
    if (this.destroyed) {
      throw new Error(
        "OPT-0032 dense K4 partials kernel was destroyed while compiling",
      );
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0032-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: exactBinding(bindings.activation, activationBytes) },
        { binding: 1, resource: exactBinding(bindings.weight, weightBytes) },
        { binding: 2, resource: exactBinding(bindings.output, outputBytes) },
      ],
    });
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0032_DENSE_K4_PARTIALS_KERNEL_ID,
      weightLayout: ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
      plan,
      rangeCount: 1 as const,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(`${label} OPT-0032 dense range must be zero`);
        }
        encodeDispatch(pass, pipeline, bindGroup, plan);
      },
      encode(pass: GPUComputePassEncoder): void {
        encodeDispatch(pass, pipeline, bindGroup, plan);
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.compiled.clear();
  }

  private pipelineFor(shape: AceGemmShape): Promise<GPUComputePipeline> {
    const key = `${shape.rows}x${shape.inner}x${shape.columns}`;
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceOpt0032DenseK4Partials(this.device, shape);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

/**
 * Benchmark-only K2 arithmetic owner. It consumes the immutable OPT-0032 K4
 * layout, but widens each consecutive two-term FP16 dot before advancing to
 * the second pair. FP32 running state therefore observes K0/K1 before K2/K3.
 */
export class AceOpt0074DenseK2PartialsKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0074DenseK2PartialsKernel {
    if (
      !device.features.has("shader-f16") ||
      !device.features.has("subgroups") ||
      capability.subgroupMinSize !== ACE_OPT_0032_DENSE_SUBGROUP_SIZE ||
      capability.subgroupMaxSize !== ACE_OPT_0032_DENSE_SUBGROUP_SIZE
    ) {
      throw new Error(
        "OPT-0074 dense K2 partials require shader-f16 and fixed 32-lane subgroups",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0032_DENSE_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX <
        ACE_OPT_0032_DENSE_WORKGROUP_SIZE
    ) {
      throw new Error(
        `OPT-0074 dense K2 partials require WG${ACE_OPT_0032_DENSE_WORKGROUP_SIZE}`,
      );
    }
    return new AceOpt0074DenseK2PartialsKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0074DenseK2PartialsDispatch> {
    if (this.destroyed) {
      throw new Error("OPT-0074 dense K2 partials kernel was destroyed");
    }
    if (bindings.bias !== undefined) {
      throw new Error("OPT-0074 repeated-layer dense GEMMs do not accept bias");
    }
    const plan = planAceOpt0032DenseK4Partials(shape);
    const activationBytes = checkedBytes(plan.activationElements, 4, "activation");
    const weightBytes = checkedBytes(plan.weightElements, 2, "weight");
    const outputBytes = checkedBytes(plan.outputElements, 4, "output");
    requireAceBindingBytes(bindings.activation, activationBytes, `${label} activation`);
    requireAceBindingBytes(bindings.weight, weightBytes, `${label} weight`);
    requireAceBindingBytes(bindings.output, outputBytes, `${label} output`);
    requireAceDisjointOutput(
      exactBinding(bindings.output, outputBytes),
      [
        exactBinding(bindings.activation, activationBytes),
        exactBinding(bindings.weight, weightBytes),
      ],
      label,
    );
    const pipeline = await this.pipelineFor(shape);
    if (this.destroyed) {
      throw new Error(
        "OPT-0074 dense K2 partials kernel was destroyed while compiling",
      );
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0074-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: exactBinding(bindings.activation, activationBytes) },
        { binding: 1, resource: exactBinding(bindings.weight, weightBytes) },
        { binding: 2, resource: exactBinding(bindings.output, outputBytes) },
      ],
    });
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0074_DENSE_K2_PARTIALS_KERNEL_ID,
      weightLayout: ACE_OPT_0074_DENSE_K2_PARTIALS_WEIGHT_LAYOUT,
      plan,
      rangeCount: 1 as const,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(`${label} OPT-0074 dense range must be zero`);
        }
        encodeDispatch(pass, pipeline, bindGroup, plan);
      },
      encode(pass: GPUComputePassEncoder): void {
        encodeDispatch(pass, pipeline, bindGroup, plan);
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.compiled.clear();
  }

  private pipelineFor(shape: AceGemmShape): Promise<GPUComputePipeline> {
    const key = `${shape.rows}x${shape.inner}x${shape.columns}`;
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceOpt0074DenseK2Partials(this.device, shape);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

export interface AceOpt0074DenseK2PartialsDispatch {
  readonly label: string;
  readonly kernelId: typeof ACE_OPT_0074_DENSE_K2_PARTIALS_KERNEL_ID;
  readonly weightLayout: typeof ACE_OPT_0074_DENSE_K2_PARTIALS_WEIGHT_LAYOUT;
  readonly plan: AceOpt0032DenseK4PartialsPlan;
  readonly rangeCount: 1;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
  encode(pass: GPUComputePassEncoder): void;
}

export function planAceOpt0032DenseK4Partials(
  shape: AceGemmShape,
): AceOpt0032DenseK4PartialsPlan {
  const { rows, inner, columns } = shape;
  requirePositiveSafeInteger(rows, "rows");
  requirePositiveSafeInteger(inner, "inner");
  requirePositiveSafeInteger(columns, "columns");
  if (!PRODUCTION_DENSE_SHAPES.has(`${inner}x${columns}`)) {
    throw new RangeError(
      `OPT-0032 dense K4 partials reject non-production K${inner}/N${columns}`,
    );
  }
  if (inner % ACE_OPT_0032_DENSE_TILE_INNER !== 0) {
    throw new RangeError("OPT-0032 dense K4 partials require K divisible by 4");
  }
  if (columns % ACE_OPT_0032_DENSE_TILE_COLUMNS !== 0) {
    throw new RangeError("OPT-0032 dense K4 partials require N divisible by 128");
  }
  const activationElements = checkedProduct(rows, inner, "activation");
  const weightElements = checkedProduct(columns, inner, "weight");
  const outputElements = checkedProduct(rows, columns, "output");
  requireWgslIndexable(activationElements, "activation");
  requireWgslIndexable(weightElements, "weight");
  requireWgslIndexable(outputElements, "output");
  const rowTiles = Math.ceil(rows / ACE_OPT_0032_DENSE_TILE_ROWS);
  const columnTiles = columns / ACE_OPT_0032_DENSE_TILE_COLUMNS;
  const innerK4Groups = inner / ACE_OPT_0032_DENSE_TILE_INNER;
  if (rowTiles > MAX_DISPATCH_DIMENSION || columnTiles > MAX_DISPATCH_DIMENSION) {
    throw new RangeError(
      "OPT-0032 dense K4 partials exceed WebGPU dispatch dimensions",
    );
  }
  const workgroupCount = checkedProduct(rowTiles, columnTiles, "workgroups");
  const scheduledRows = checkedProduct(
    rowTiles,
    ACE_OPT_0032_DENSE_TILE_ROWS,
    "scheduled rows",
  );
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
    rows,
    inner,
    columns,
    workgroupsX: columnTiles,
    workgroupsY: rowTiles,
    tileRows: ACE_OPT_0032_DENSE_TILE_ROWS,
    tileColumns: ACE_OPT_0032_DENSE_TILE_COLUMNS,
    tileInner: ACE_OPT_0032_DENSE_TILE_INNER,
    workgroupSize: ACE_OPT_0032_DENSE_WORKGROUP_SIZE,
    subgroupSize: ACE_OPT_0032_DENSE_SUBGROUP_SIZE,
    rowTiles,
    columnTiles,
    innerK4Groups,
    workgroupCount,
    activationElements,
    weightElements,
    outputElements,
    outputRangeCount: 1,
    outputRanges,
    packedWeightStorageShape: Object.freeze([
      columnTiles,
      innerK4Groups,
      COLUMNS_PER_LANE,
      ACE_OPT_0032_DENSE_SUBGROUP_SIZE,
      ACE_OPT_0032_DENSE_TILE_INNER,
    ]) as readonly [number, number, 4, 32, 4],
  });
}

/**
 * Convert logical row-major B[K,N] FP16 bits to
 * [N/128,K/4,output4,lane32,K4]. This helper intentionally accepts any
 * K%4/N%128 shape so focused layout tests need not allocate production B.
 */
export function packAceOpt0032DenseWeightU16(
  logical: Uint16Array,
  inner: number,
  columns: number,
): Uint16Array {
  requirePositiveSafeInteger(inner, "pack inner");
  requirePositiveSafeInteger(columns, "pack columns");
  if (inner % ACE_OPT_0032_DENSE_TILE_INNER !== 0) {
    throw new RangeError("OPT-0032 pack requires K divisible by 4");
  }
  if (columns % ACE_OPT_0032_DENSE_TILE_COLUMNS !== 0) {
    throw new RangeError("OPT-0032 pack requires N divisible by 128");
  }
  const elements = checkedProduct(inner, columns, "pack elements");
  if (logical.length !== elements) {
    throw new RangeError(
      `OPT-0032 pack expected ${elements} FP16 elements, got ${logical.length}`,
    );
  }
  const packed = new Uint16Array(elements);
  for (let k = 0; k < inner; k += 1) {
    for (let column = 0; column < columns; column += 1) {
      packed[aceOpt0032PackedWeightIndex(k, column, inner, columns)] =
        logical[k * columns + column]!;
    }
  }
  return packed;
}

export function aceOpt0032PackedWeightIndex(
  innerIndex: number,
  column: number,
  inner: number,
  columns: number,
): number {
  requirePositiveSafeInteger(inner, "index inner");
  requirePositiveSafeInteger(columns, "index columns");
  if (
    inner % ACE_OPT_0032_DENSE_TILE_INNER !== 0 ||
    columns % ACE_OPT_0032_DENSE_TILE_COLUMNS !== 0
  ) {
    throw new RangeError("OPT-0032 packed index requires K%4=0 and N%128=0");
  }
  requireCoordinate(innerIndex, inner, "inner index");
  requireCoordinate(column, columns, "column");
  const columnTile = Math.floor(column / ACE_OPT_0032_DENSE_TILE_COLUMNS);
  const columnInTile = column % ACE_OPT_0032_DENSE_TILE_COLUMNS;
  const outputInLane = columnInTile % COLUMNS_PER_LANE;
  const lane = Math.floor(columnInTile / COLUMNS_PER_LANE);
  const innerK4 = Math.floor(innerIndex / ACE_OPT_0032_DENSE_TILE_INNER);
  const innerInK4 = innerIndex % ACE_OPT_0032_DENSE_TILE_INNER;
  const innerK4Groups = inner / ACE_OPT_0032_DENSE_TILE_INNER;
  return (((
    (columnTile * innerK4Groups + innerK4) * COLUMNS_PER_LANE +
    outputInLane
  ) * ACE_OPT_0032_DENSE_SUBGROUP_SIZE + lane) *
    ACE_OPT_0032_DENSE_TILE_INNER) + innerInK4;
}

export function aceOpt0032DenseK4PartialsWgsl(shape: AceGemmShape): string {
  const plan = planAceOpt0032DenseK4Partials(shape);
  const declarations = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `  var acc${row} = vec4<f32>(0.0);`,
  ).join("\n");
  const broadcasts = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `    let a${row} = subgroupBroadcast(lane_a, ${row}u);`,
  ).join("\n");
  const contractions = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => /* wgsl */ `
    let partial${row} = vec4<f16>(
      dot(a${row}, b0),
      dot(a${row}, b1),
      dot(a${row}, b2),
      dot(a${row}, b3)
    );
    acc${row} = acc${row} + vec4<f32>(partial${row});`,
  ).join("\n");
  const stores = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + ${row}u;
    if (row < ROWS) {
      output[row * (COLUMNS / 4u) + column_vector] = acc${row};
    }
  }`,
  ).join("\n");
  return /* wgsl */ `
enable f16;
enable subgroups;

const ROWS = ${plan.rows}u;
const INNER = ${plan.inner}u;
const COLUMNS = ${plan.columns}u;
const INNER_K4_GROUPS = ${plan.innerK4Groups}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(${ACE_OPT_0032_DENSE_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (
    subgroup_size != ${ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u ||
    group.x >= ${plan.columnTiles}u ||
    group.y >= ${plan.rowTiles}u ||
    group.z != 0u ||
    subgroup >= ${SUBGROUPS_PER_WORKGROUP}u
  ) {
    return;
  }
  let row_base =
    group.y * ${ACE_OPT_0032_DENSE_TILE_ROWS}u +
    subgroup * ${ROWS_PER_SUBGROUP}u;
  let column_vector =
    group.x * ${ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u + subgroup_lane;
${declarations}

  for (var inner_k4 = 0u; inner_k4 < INNER_K4_GROUPS; inner_k4 += 1u) {
    let inner_base = inner_k4 * ${ACE_OPT_0032_DENSE_TILE_INNER}u;
    var lane_a = vec4<f16>(0.0h);
    let lane_row = row_base + subgroup_lane;
    if (subgroup_lane < ${ROWS_PER_SUBGROUP}u && lane_row < ROWS) {
      let activation_base = lane_row * INNER + inner_base;
      lane_a = vec4<f16>(
        f16(activation[activation_base]),
        f16(activation[activation_base + 1u]),
        f16(activation[activation_base + 2u]),
        f16(activation[activation_base + 3u])
      );
    }
    let weight_base =
      ((group.x * INNER_K4_GROUPS + inner_k4) *
      ${COLUMNS_PER_LANE}u) * ${ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u +
      subgroup_lane;
    let b0 = weight[weight_base];
    let b1 = weight[weight_base + ${ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u];
    let b2 = weight[weight_base + ${2 * ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u];
    let b3 = weight[weight_base + ${3 * ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u];
${broadcasts}
${contractions}
  }
${stores}
}
`;
}

export function aceOpt0074DenseK2PartialsWgsl(shape: AceGemmShape): string {
  const plan = planAceOpt0032DenseK4Partials(shape);
  const declarations = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `  var acc${row} = vec4<f32>(0.0);`,
  ).join("\n");
  const broadcasts = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `    let a${row} = subgroupBroadcast(lane_a, ${row}u);`,
  ).join("\n");
  const contractions = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => /* wgsl */ `
    let partial${row}_01 = vec4<f16>(
      dot(a${row}.xy, b0.xy),
      dot(a${row}.xy, b1.xy),
      dot(a${row}.xy, b2.xy),
      dot(a${row}.xy, b3.xy)
    );
    acc${row} = acc${row} + vec4<f32>(partial${row}_01);
    let partial${row}_23 = vec4<f16>(
      dot(a${row}.zw, b0.zw),
      dot(a${row}.zw, b1.zw),
      dot(a${row}.zw, b2.zw),
      dot(a${row}.zw, b3.zw)
    );
    acc${row} = acc${row} + vec4<f32>(partial${row}_23);`,
  ).join("\n");
  const stores = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + ${row}u;
    if (row < ROWS) {
      output[row * (COLUMNS / 4u) + column_vector] = acc${row};
    }
  }`,
  ).join("\n");
  return /* wgsl */ `
enable f16;
enable subgroups;

const ROWS = ${plan.rows}u;
const INNER = ${plan.inner}u;
const COLUMNS = ${plan.columns}u;
const INNER_K4_GROUPS = ${plan.innerK4Groups}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(${ACE_OPT_0032_DENSE_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (
    subgroup_size != ${ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u ||
    group.x >= ${plan.columnTiles}u ||
    group.y >= ${plan.rowTiles}u ||
    group.z != 0u ||
    subgroup >= ${SUBGROUPS_PER_WORKGROUP}u
  ) {
    return;
  }
  let row_base =
    group.y * ${ACE_OPT_0032_DENSE_TILE_ROWS}u +
    subgroup * ${ROWS_PER_SUBGROUP}u;
  let column_vector =
    group.x * ${ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u + subgroup_lane;
${declarations}

  for (var inner_k4 = 0u; inner_k4 < INNER_K4_GROUPS; inner_k4 += 1u) {
    let inner_base = inner_k4 * ${ACE_OPT_0032_DENSE_TILE_INNER}u;
    var lane_a = vec4<f16>(0.0h);
    let lane_row = row_base + subgroup_lane;
    if (subgroup_lane < ${ROWS_PER_SUBGROUP}u && lane_row < ROWS) {
      let activation_base = lane_row * INNER + inner_base;
      lane_a = vec4<f16>(
        f16(activation[activation_base]),
        f16(activation[activation_base + 1u]),
        f16(activation[activation_base + 2u]),
        f16(activation[activation_base + 3u])
      );
    }
    let weight_base =
      ((group.x * INNER_K4_GROUPS + inner_k4) *
      ${COLUMNS_PER_LANE}u) * ${ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u +
      subgroup_lane;
    let b0 = weight[weight_base];
    let b1 = weight[weight_base + ${ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u];
    let b2 = weight[weight_base + ${2 * ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u];
    let b3 = weight[weight_base + ${3 * ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u];
${broadcasts}
${contractions}
  }
${stores}
}
`;
}

async function compileAceOpt0032DenseK4Partials(
  device: GPUDevice,
  shape: AceGemmShape,
): Promise<GPUComputePipeline> {
  const label =
    `ace-opt-0032-dense-k4-${shape.rows}x${shape.inner}x${shape.columns}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0032DenseK4PartialsWgsl(shape),
  });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length !== 0) {
    throw new Error(
      `${label} WGSL compilation failed:\n` + errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("\n"),
    );
  }
  return await device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

async function compileAceOpt0074DenseK2Partials(
  device: GPUDevice,
  shape: AceGemmShape,
): Promise<GPUComputePipeline> {
  const label =
    `ace-opt-0074-dense-k2-${shape.rows}x${shape.inner}x${shape.columns}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0074DenseK2PartialsWgsl(shape),
  });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length !== 0) {
    throw new Error(
      `${label} WGSL compilation failed:\n` + errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("\n"),
    );
  }
  return await device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

function encodeDispatch(
  pass: GPUComputePassEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  plan: AceOpt0032DenseK4PartialsPlan,
): void {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(plan.columnTiles, plan.rowTiles, 1);
}

function exactBinding(
  binding: GPUBufferBinding,
  requiredBytes: number,
): GPUBufferBinding {
  return {
    buffer: binding.buffer,
    offset: binding.offset ?? 0,
    size: requiredBytes,
  };
}

function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(
      `OPT-0032 dense K4 partials ${label} must be a positive safe integer`,
    );
  }
}

function requireCoordinate(value: number, extent: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= extent) {
    throw new RangeError(`OPT-0032 ${label} ${value} is out of bounds`);
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(
      `OPT-0032 dense K4 partials ${label} is not a safe integer`,
    );
  }
  return product;
}

function checkedBytes(elements: number, itemBytes: number, label: string): number {
  return checkedProduct(elements, itemBytes, `${label} bytes`);
}

function requireWgslIndexable(elements: number, label: string): void {
  if (elements > MAX_WGSL_U32) {
    throw new RangeError(
      `OPT-0032 dense K4 partials ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}
