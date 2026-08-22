import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../../model/manifest.js";
import { requireAceDisjointOutput } from "./correctness-utils.js";
import type {
  AceCooperativeGemmPlan,
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmKernel,
  AceGemmOutputRange,
  AceGemmShape,
} from "./gemm.js";

export const ACE_OPT_0079_DENSE_DECODED_HALF_TILE_KERNEL_SET_ID =
  "opt-0079-m64-n128-k32-wg256-decoded-half-tile-fp16-fp32-v1" as const;
export const ACE_OPT_0079_DENSE_TILE_ROWS = 64;
export const ACE_OPT_0079_DENSE_TILE_COLUMNS = 128;
export const ACE_OPT_0079_DENSE_TILE_INNER = 32;
export const ACE_OPT_0079_DENSE_WORKGROUP_SIZE = 256;
export const ACE_OPT_0079_DENSE_SUBGROUP_SIZE = 32;
export const ACE_OPT_0079_DENSE_SUBGROUPS_PER_WORKGROUP = 8;
export const ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP = 8;
export const ACE_OPT_0079_DENSE_COLUMNS_PER_LANE = 4;
export const ACE_OPT_0079_DENSE_ACCUMULATORS_PER_LANE = 32;
export const ACE_OPT_0079_DENSE_PACKED_RECORDS_PER_INNER_TILE = 512;
export const ACE_OPT_0079_DENSE_PACKED_RECORDS_PER_LANE = 2;
export const ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_PACKED_RECORD = 2;
export const ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_INNER_TILE = 1_024;
export const ACE_OPT_0079_DENSE_BARRIERS_PER_INNER_TILE = 2;
export const ACE_OPT_0079_DENSE_WORKGROUP_STORAGE_BYTES =
  ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_INNER_TILE * 8;

const M2250 = 2_250;
const FLOAT16_BYTES = 2;
const FLOAT32_BYTES = 4;
const GPU_BUFFER_ALIGNMENT = 4;
const PACKED_COLUMNS_PER_RECORD = 8;
const PHYSICAL_TILE_COLUMNS = 256;
const PHYSICAL_PACKED_RECORDS_PER_K_ROW =
  PHYSICAL_TILE_COLUMNS / PACKED_COLUMNS_PER_RECORD;
const LOGICAL_PACKED_RECORDS_PER_K_ROW =
  ACE_OPT_0079_DENSE_TILE_COLUMNS / PACKED_COLUMNS_PER_RECORD;
const MAX_WGSL_U32 = 0xffff_ffff;

const PRODUCTION_DENSE_SHAPES = new Set([
  "2048x2048",
  "2048x1024",
  "2048x6144",
  "6144x2048",
]);

export interface AceOpt0079DenseDecodedHalfTilePlan
  extends AceCooperativeGemmPlan {
  readonly kernelSetId:
    typeof ACE_OPT_0079_DENSE_DECODED_HALF_TILE_KERNEL_SET_ID;
  readonly tileRows: typeof ACE_OPT_0079_DENSE_TILE_ROWS;
  readonly tileColumns: typeof ACE_OPT_0079_DENSE_TILE_COLUMNS;
  readonly tileInner: typeof ACE_OPT_0079_DENSE_TILE_INNER;
  readonly workgroupSize: typeof ACE_OPT_0079_DENSE_WORKGROUP_SIZE;
  readonly subgroupSize: typeof ACE_OPT_0079_DENSE_SUBGROUP_SIZE;
  readonly subgroupsPerWorkgroup:
    typeof ACE_OPT_0079_DENSE_SUBGROUPS_PER_WORKGROUP;
  readonly rowsPerSubgroup: typeof ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP;
  readonly columnsPerLane: typeof ACE_OPT_0079_DENSE_COLUMNS_PER_LANE;
  readonly accumulatorsPerLane:
    typeof ACE_OPT_0079_DENSE_ACCUMULATORS_PER_LANE;
  readonly packedRecordsPerInnerTile:
    typeof ACE_OPT_0079_DENSE_PACKED_RECORDS_PER_INNER_TILE;
  readonly packedRecordsPerLane:
    typeof ACE_OPT_0079_DENSE_PACKED_RECORDS_PER_LANE;
  readonly decodedVectorsPerPackedRecord:
    typeof ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_PACKED_RECORD;
  readonly decodedVectorsPerInnerTile:
    typeof ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_INNER_TILE;
  readonly workgroupStorageBytes:
    typeof ACE_OPT_0079_DENSE_WORKGROUP_STORAGE_BYTES;
  readonly rowTiles: number;
  readonly columnTiles: number;
  readonly physicalColumnTiles: number;
  readonly innerTiles: number;
  readonly workgroupCount: number;
  readonly scheduledRows: number;
  readonly scheduledMultiplyAdds: number;
  readonly validMultiplyAdds: number;
  readonly activationElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly barriersPerWorkgroup: number;
  readonly barrierEvents: number;
  readonly packedRecordLoadsPerWorkgroup: number;
  readonly decodedVectorWritesPerWorkgroup: number;
  readonly estimatedGlobalActivationBytes: number;
  readonly estimatedGlobalWeightBytes: number;
  readonly estimatedGlobalOperandBytes: number;
  readonly estimatedGlobalOutputBytes: number;
  readonly packedWeightStorageShape: readonly [number, number, 32, 256];
}

export interface AceOpt0079DenseDecodedHalfTileDispatch
  extends AceGemmDispatch {
  readonly kernelSetId:
    typeof ACE_OPT_0079_DENSE_DECODED_HALF_TILE_KERNEL_SET_ID;
  readonly weightLayout: typeof ACE_DIT_DENSE_FP16_TILE_LAYOUT;
  readonly plan: AceOpt0079DenseDecodedHalfTilePlan;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Isolated OPT-0079 benchmark owner. It stages one converter-native N128/K32
 * half-tile as typed FP16 vectors and is not selectable by the production graph.
 */
export class AceOpt0079DenseDecodedHalfTileKernel implements AceGemmKernel {
  private readonly pipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0079DenseDecodedHalfTileKernel {
    requireKernelDevice(device, capability);
    return new AceOpt0079DenseDecodedHalfTileKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0079DenseDecodedHalfTileDispatch> {
    this.requireLive();
    if (bindings.bias !== undefined) {
      throw new RangeError("OPT-0079 decoded half-tile dense rejects bias");
    }
    const plan = planAceOpt0079DenseDecodedHalfTile(shape);
    requireDispatchDimensions(this.device, plan.workgroupsX, plan.workgroupsY);
    const activationBytes = checkedProduct(
      plan.activationElements,
      FLOAT32_BYTES,
      "activation bytes",
    );
    const weightBytes = checkedProduct(
      plan.weightElements,
      FLOAT16_BYTES,
      "weight bytes",
    );
    const outputBytes = checkedProduct(
      plan.outputElements,
      FLOAT32_BYTES,
      "output bytes",
    );
    requireBufferLimits(this.device, [
      ["activation", activationBytes],
      ["weight", weightBytes],
      ["output", outputBytes],
    ]);
    const activation = requireStorageBinding(
      this.device,
      bindings.activation,
      activationBytes,
      `${label} activation`,
    );
    const weight = requireStorageBinding(
      this.device,
      bindings.weight,
      weightBytes,
      `${label} weight`,
    );
    const output = requireStorageBinding(
      this.device,
      bindings.output,
      outputBytes,
      `${label} output`,
    );
    requireAceDisjointOutput(output, [activation, weight], label);

    const compiled = await this.pipelineFor(plan);
    this.requireLive();
    const resources = Object.freeze([activation, weight, output]);
    const bindGroup = this.bindGroupFor(
      shapeKey(plan),
      `${label}-opt-0079-decoded-half-tile-bindings`,
      compiled.bindGroupLayout,
      resources,
    );
    const owner = this;
    const encode = (pass: GPUComputePassEncoder): void => {
      owner.requireLive();
      pass.setPipeline(compiled.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(plan.columnTiles, plan.rowTiles, 1);
    };
    return Object.freeze({
      label,
      kernelSetId: ACE_OPT_0079_DENSE_DECODED_HALF_TILE_KERNEL_SET_ID,
      weightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
      plan,
      rangeCount: 1,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(`${label} OPT-0079 dense range must be zero`);
        }
        encode(pass);
      },
      encode,
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.pipelines.clear();
    this.bindGroups.clear();
  }

  private pipelineFor(
    plan: AceOpt0079DenseDecodedHalfTilePlan,
  ): Promise<CompiledKernel> {
    const key = shapeKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileDecodedHalfTile(this.device, plan);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
  }

  private bindGroupFor(
    shape: string,
    label: string,
    layout: GPUBindGroupLayout,
    resources: readonly GPUBufferBinding[],
  ): GPUBindGroup {
    const key = `${shape}:${resources.map((binding) =>
      this.bindingKey(binding)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(key);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label,
        layout,
        entries: resources.map((resource, binding) => ({ binding, resource })),
      });
      this.bindGroups.set(key, bindGroup);
    }
    return bindGroup;
  }

  private bindingKey(binding: GPUBufferBinding): string {
    let id = this.bufferIds.get(binding.buffer);
    if (id === undefined) {
      id = this.nextBufferId;
      this.nextBufferId += 1;
      this.bufferIds.set(binding.buffer, id);
    }
    return `${id}:${binding.offset ?? 0}:${binding.size ?? -1}`;
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0079 decoded half-tile dense kernel was destroyed");
    }
  }
}

export function planAceOpt0079DenseDecodedHalfTile(
  shape: AceGemmShape,
): AceOpt0079DenseDecodedHalfTilePlan {
  const { rows, inner, columns } = shape;
  requirePositiveSafeInteger(rows, "rows");
  requirePositiveSafeInteger(inner, "inner");
  requirePositiveSafeInteger(columns, "columns");
  if (rows !== M2250) {
    throw new RangeError("OPT-0079 decoded half-tile dense requires exact M2250");
  }
  if (!PRODUCTION_DENSE_SHAPES.has(`${inner}x${columns}`)) {
    throw new RangeError(
      `OPT-0079 decoded half-tile dense rejects non-production K${inner}/N${columns}`,
    );
  }
  if (
    inner % ACE_OPT_0079_DENSE_TILE_INNER !== 0 ||
    columns % PHYSICAL_TILE_COLUMNS !== 0
  ) {
    throw new RangeError(
      "OPT-0079 decoded half-tile dense requires exact K32/N256 payload multiples",
    );
  }

  const activationElements = checkedProduct(rows, inner, "activation elements");
  const weightElements = checkedProduct(inner, columns, "weight elements");
  const outputElements = checkedProduct(rows, columns, "output elements");
  requireWgslIndexable(activationElements, "activation");
  requireWgslIndexable(weightElements, "weight");
  requireWgslIndexable(outputElements, "output");
  const rowTiles = Math.ceil(rows / ACE_OPT_0079_DENSE_TILE_ROWS);
  const columnTiles = columns / ACE_OPT_0079_DENSE_TILE_COLUMNS;
  const physicalColumnTiles = columns / PHYSICAL_TILE_COLUMNS;
  const innerTiles = inner / ACE_OPT_0079_DENSE_TILE_INNER;
  const workgroupCount = checkedProduct(rowTiles, columnTiles, "workgroups");
  const scheduledRows = checkedProduct(
    rowTiles,
    ACE_OPT_0079_DENSE_TILE_ROWS,
    "scheduled rows",
  );
  const scheduledMultiplyAdds = checkedProduct(
    checkedProduct(scheduledRows, inner, "scheduled row-inner"),
    columns,
    "scheduled multiply-adds",
  );
  const validMultiplyAdds = checkedProduct(
    activationElements,
    columns,
    "valid multiply-adds",
  );
  const barriersPerWorkgroup = checkedProduct(
    innerTiles,
    ACE_OPT_0079_DENSE_BARRIERS_PER_INNER_TILE,
    "barriers per workgroup",
  );
  const barrierEvents = checkedProduct(
    workgroupCount,
    barriersPerWorkgroup,
    "barrier events",
  );
  const packedRecordLoadsPerWorkgroup = checkedProduct(
    innerTiles,
    ACE_OPT_0079_DENSE_PACKED_RECORDS_PER_INNER_TILE,
    "packed-record loads per workgroup",
  );
  const decodedVectorWritesPerWorkgroup = checkedProduct(
    innerTiles,
    ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_INNER_TILE,
    "decoded-vector writes per workgroup",
  );
  const estimatedGlobalActivationBytes = checkedProduct(
    checkedProduct(scheduledRows, inner, "scheduled activation elements"),
    checkedProduct(columnTiles, FLOAT32_BYTES, "activation column-tile bytes"),
    "global activation request bytes",
  );
  const estimatedGlobalWeightBytes = checkedProduct(
    checkedProduct(weightElements, FLOAT16_BYTES, "weight payload bytes"),
    rowTiles,
    "global weight request bytes",
  );
  const estimatedGlobalOperandBytes = checkedSum(
    estimatedGlobalActivationBytes,
    estimatedGlobalWeightBytes,
    "global operand request bytes",
  );
  const estimatedGlobalOutputBytes = checkedProduct(
    outputElements,
    FLOAT32_BYTES,
    "global output bytes",
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
    kernelSetId: ACE_OPT_0079_DENSE_DECODED_HALF_TILE_KERNEL_SET_ID,
    rows,
    inner,
    columns,
    workgroupsX: columnTiles,
    workgroupsY: rowTiles,
    tileRows: ACE_OPT_0079_DENSE_TILE_ROWS,
    tileColumns: ACE_OPT_0079_DENSE_TILE_COLUMNS,
    tileInner: ACE_OPT_0079_DENSE_TILE_INNER,
    workgroupSize: ACE_OPT_0079_DENSE_WORKGROUP_SIZE,
    subgroupSize: ACE_OPT_0079_DENSE_SUBGROUP_SIZE,
    subgroupsPerWorkgroup: ACE_OPT_0079_DENSE_SUBGROUPS_PER_WORKGROUP,
    rowsPerSubgroup: ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP,
    columnsPerLane: ACE_OPT_0079_DENSE_COLUMNS_PER_LANE,
    accumulatorsPerLane: ACE_OPT_0079_DENSE_ACCUMULATORS_PER_LANE,
    packedRecordsPerInnerTile:
      ACE_OPT_0079_DENSE_PACKED_RECORDS_PER_INNER_TILE,
    packedRecordsPerLane: ACE_OPT_0079_DENSE_PACKED_RECORDS_PER_LANE,
    decodedVectorsPerPackedRecord:
      ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_PACKED_RECORD,
    decodedVectorsPerInnerTile:
      ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_INNER_TILE,
    workgroupStorageBytes: ACE_OPT_0079_DENSE_WORKGROUP_STORAGE_BYTES,
    rowTiles,
    columnTiles,
    physicalColumnTiles,
    innerTiles,
    workgroupCount,
    scheduledRows,
    scheduledMultiplyAdds,
    validMultiplyAdds,
    activationElements,
    weightElements,
    outputElements,
    outputRangeCount: 1,
    outputRanges,
    barriersPerWorkgroup,
    barrierEvents,
    packedRecordLoadsPerWorkgroup,
    decodedVectorWritesPerWorkgroup,
    estimatedGlobalActivationBytes,
    estimatedGlobalWeightBytes,
    estimatedGlobalOperandBytes,
    estimatedGlobalOutputBytes,
    packedWeightStorageShape: Object.freeze([
      physicalColumnTiles,
      innerTiles,
      32,
      256,
    ]) as readonly [number, number, 32, 256],
  });
}

export function aceOpt0079DenseDecodedHalfTileWgsl(
  shape: AceGemmShape,
): string {
  const plan = planAceOpt0079DenseDecodedHalfTile(shape);
  const accumulators = Array.from(
    { length: ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP },
    (_, row) => `  var acc${row} = vec4<f32>(0.0);`,
  ).join("\n");
  const broadcasts = Array.from(
    { length: ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP },
    (_, row) => `      let a${row} = subgroupBroadcast(lane_a, ${row}u);`,
  ).join("\n");
  const contractions = Array.from(
    { length: ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP },
    (_, row) =>
      `      acc${row} = acc${row} + vec4<f32>(f32(a${row})) * vec4<f32>(b);`,
  ).join("\n");
  const stores = Array.from(
    { length: ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + ${row}u;
    if (row < ROWS) {
      let vector_base = row * (COLUMNS / 4u) + column_base / 4u;
      output[vector_base] = acc${row};
    }
  }`,
  ).join("");
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0079_DENSE_DECODED_HALF_TILE_KERNEL_SET_ID}
// reduction-semantics: strict-increasing-k-fp32-sum-plus-product
// payload: N256/K32; staged half-tile: N128/K32, 512 records -> 1024 vec4<f16>
enable f16;
enable subgroups;

const ROWS: u32 = ${plan.rows}u;
const INNER: u32 = ${plan.inner}u;
const COLUMNS: u32 = ${plan.columns}u;
const INNER_TILES: u32 = ${plan.innerTiles}u;
const TILE_ROWS: u32 = ${ACE_OPT_0079_DENSE_TILE_ROWS}u;
const TILE_COLUMNS: u32 = ${ACE_OPT_0079_DENSE_TILE_COLUMNS}u;
const TILE_INNER: u32 = ${ACE_OPT_0079_DENSE_TILE_INNER}u;
const PHYSICAL_RECORDS_PER_TILE: u32 = 1024u;
const PHYSICAL_RECORDS_PER_K_ROW: u32 = ${PHYSICAL_PACKED_RECORDS_PER_K_ROW}u;
const HALF_RECORDS_PER_K_ROW: u32 = ${LOGICAL_PACKED_RECORDS_PER_K_ROW}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

var<workgroup> weight_panel: array<vec4<f16>, ${ACE_OPT_0079_DENSE_DECODED_VECTORS_PER_INNER_TILE}>;

fn decode_f16x4(low: u32, high: u32) -> vec4<f16> {
  let low_pair = unpack2x16float(low);
  let high_pair = unpack2x16float(high);
  return vec4<f16>(
    f16(low_pair.x), f16(low_pair.y), f16(high_pair.x), f16(high_pair.y)
  );
}

@compute @workgroup_size(${ACE_OPT_0079_DENSE_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
) {
  let row_base = group.y * TILE_ROWS + subgroup * ${ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP}u;
  let column_base =
    group.x * TILE_COLUMNS + subgroup_lane * ${ACE_OPT_0079_DENSE_COLUMNS_PER_LANE}u;
  let physical_column_tile = group.x / 2u;
  let half_record_offset = (group.x % 2u) * HALF_RECORDS_PER_K_ROW;
${accumulators}

  // All 256 invocations execute both barriers. Each invocation loads exactly
  // two native packed records and decodes each once into two typed-FP16 vectors.
  for (var inner_tile = 0u; inner_tile < INNER_TILES; inner_tile += 1u) {
    let physical_tile_base =
      (physical_column_tile * INNER_TILES + inner_tile) * PHYSICAL_RECORDS_PER_TILE;

    let logical_record0 = local_index;
    let physical_record0 = physical_tile_base +
      (logical_record0 / HALF_RECORDS_PER_K_ROW) * PHYSICAL_RECORDS_PER_K_ROW +
      half_record_offset + logical_record0 % HALF_RECORDS_PER_K_ROW;
    let packed0 = weight[physical_record0];
    weight_panel[logical_record0 * 2u] = decode_f16x4(packed0.x, packed0.y);
    weight_panel[logical_record0 * 2u + 1u] = decode_f16x4(packed0.z, packed0.w);

    let logical_record1 = local_index + 256u;
    let physical_record1 = physical_tile_base +
      (logical_record1 / HALF_RECORDS_PER_K_ROW) * PHYSICAL_RECORDS_PER_K_ROW +
      half_record_offset + logical_record1 % HALF_RECORDS_PER_K_ROW;
    let packed1 = weight[physical_record1];
    weight_panel[logical_record1 * 2u] = decode_f16x4(packed1.x, packed1.y);
    weight_panel[logical_record1 * 2u + 1u] = decode_f16x4(packed1.z, packed1.w);

    workgroupBarrier();
    for (var inner_in_tile = 0u; inner_in_tile < TILE_INNER; inner_in_tile += 1u) {
      let inner = inner_tile * TILE_INNER + inner_in_tile;
      var lane_a = 0.0h;
      let lane_row = row_base + subgroup_lane;
      if (
        subgroup_lane < ${ACE_OPT_0079_DENSE_ROWS_PER_SUBGROUP}u &&
        lane_row < ROWS
      ) {
        lane_a = f16(activation[lane_row * INNER + inner]);
      }
      let b = weight_panel[inner_in_tile * 32u + subgroup_lane];
${broadcasts}
${contractions}
    }
    workgroupBarrier();
  }
${stores}
}
`;
}

async function compileDecodedHalfTile(
  device: GPUDevice,
  plan: AceOpt0079DenseDecodedHalfTilePlan,
): Promise<CompiledKernel> {
  const label =
    `ace-opt-0079-dense-${plan.rows}x${plan.inner}x${plan.columns}`;
  const module = await checkedShaderModule(
    device,
    label,
    aceOpt0079DenseDecodedHalfTileWgsl(plan),
  );
  const bindingBytes = [
    checkedProduct(plan.activationElements, FLOAT32_BYTES, "activation bytes"),
    checkedProduct(plan.weightElements, FLOAT16_BYTES, "weight bytes"),
    checkedProduct(plan.outputElements, FLOAT32_BYTES, "output bytes"),
  ];
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: bindingBytes.map((minBindingSize, binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: binding === 2 ? "storage" as const : "read-only-storage" as const,
        minBindingSize,
      },
    })),
  });
  const pipelineLayout = device.createPipelineLayout({
    label: `${label}-layout`,
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: pipelineLayout,
    compute: { module, entryPoint: "main" },
  });
  return Object.freeze({ pipeline, bindGroupLayout });
}

async function checkedShaderModule(
  device: GPUDevice,
  label: string,
  code: string,
): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ label, code });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0079 decoded half-tile dense WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  return module;
}

function requireKernelDevice(
  device: GPUDevice,
  capability: Readonly<{
    subgroupMinSize?: number;
    subgroupMaxSize?: number;
  }>,
): void {
  if (
    !device.features.has("shader-f16") ||
    !device.features.has("subgroups") ||
    capability.subgroupMinSize !== ACE_OPT_0079_DENSE_SUBGROUP_SIZE ||
    capability.subgroupMaxSize !== ACE_OPT_0079_DENSE_SUBGROUP_SIZE
  ) {
    throw new Error(
      "OPT-0079 decoded half-tile dense requires shader-f16 and fixed 32-lane subgroups",
    );
  }
  const invocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const sizeX = device.limits.maxComputeWorkgroupSizeX;
  const storage = device.limits.maxComputeWorkgroupStorageSize;
  if (
    !Number.isSafeInteger(invocations) ||
    !Number.isSafeInteger(sizeX) ||
    invocations < ACE_OPT_0079_DENSE_WORKGROUP_SIZE ||
    sizeX < ACE_OPT_0079_DENSE_WORKGROUP_SIZE
  ) {
    throw new Error("OPT-0079 decoded half-tile dense requires a 256x1 workgroup");
  }
  if (
    !Number.isSafeInteger(storage) ||
    storage < ACE_OPT_0079_DENSE_WORKGROUP_STORAGE_BYTES
  ) {
    throw new Error(
      `OPT-0079 decoded half-tile dense requires ${ACE_OPT_0079_DENSE_WORKGROUP_STORAGE_BYTES} workgroup-storage bytes`,
    );
  }
}

function requireDispatchDimensions(
  device: GPUDevice,
  workgroupsX: number,
  workgroupsY: number,
): void {
  const maximum = device.limits.maxComputeWorkgroupsPerDimension;
  if (
    !Number.isSafeInteger(maximum) ||
    maximum < 1 ||
    workgroupsX > maximum ||
    workgroupsY > maximum
  ) {
    throw new RangeError(
      "OPT-0079 decoded half-tile dense exceeds the dispatch dimension",
    );
  }
}

function requireBufferLimits(
  device: GPUDevice,
  resources: readonly (readonly [string, number])[],
): void {
  const maximumBinding = Number(device.limits.maxStorageBufferBindingSize);
  const maximumBuffer = Number(device.limits.maxBufferSize);
  if (
    !Number.isSafeInteger(maximumBinding) ||
    maximumBinding < 1 ||
    !Number.isSafeInteger(maximumBuffer) ||
    maximumBuffer < 1
  ) {
    throw new RangeError(
      "OPT-0079 decoded half-tile dense device reported invalid buffer limits",
    );
  }
  for (const [name, bytes] of resources) {
    if (bytes > maximumBinding || bytes > maximumBuffer) {
      throw new RangeError(
        `OPT-0079 decoded half-tile dense ${name} exceeds the device buffer limits`,
      );
    }
  }
}

function requireStorageBinding(
  device: GPUDevice,
  binding: GPUBufferBinding,
  requiredBytes: number,
  label: string,
): GPUBufferBinding {
  const alignment = device.limits.minStorageBufferOffsetAlignment;
  if (!isValidGpuAlignment(alignment)) {
    throw new Error(
      "OPT-0079 decoded half-tile dense device reported invalid alignment",
    );
  }
  const bufferBytes = Number(binding.buffer.size);
  const offset = binding.offset ?? 0;
  const available = binding.size ?? bufferBytes - offset;
  if (
    !Number.isSafeInteger(bufferBytes) ||
    bufferBytes < requiredBytes ||
    bufferBytes % GPU_BUFFER_ALIGNMENT !== 0 ||
    !Number.isSafeInteger(offset) ||
    offset < 0 ||
    offset % alignment !== 0 ||
    !Number.isSafeInteger(available) ||
    available < requiredBytes ||
    available % GPU_BUFFER_ALIGNMENT !== 0 ||
    !Number.isSafeInteger(offset + available) ||
    offset + available > bufferBytes
  ) {
    throw new RangeError(
      `${label} does not expose an aligned ${requiredBytes}-byte binding`,
    );
  }
  return Object.freeze({ buffer: binding.buffer, offset, size: requiredBytes });
}

function shapeKey(shape: AceGemmShape): string {
  return `${shape.rows}x${shape.inner}x${shape.columns}`;
}

function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(
      `OPT-0079 decoded half-tile dense ${label} must be a positive safe integer`,
    );
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(
      `OPT-0079 decoded half-tile dense ${label} is not a safe integer`,
    );
  }
  return product;
}

function checkedSum(left: number, right: number, label: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw new RangeError(
      `OPT-0079 decoded half-tile dense ${label} is not a safe integer`,
    );
  }
  return sum;
}

function requireWgslIndexable(elements: number, label: string): void {
  if (elements > MAX_WGSL_U32) {
    throw new RangeError(
      `OPT-0079 decoded half-tile dense ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
