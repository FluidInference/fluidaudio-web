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

export const ACE_OPT_0020_DENSE_COOPERATIVE_DOT4_KERNEL_SET_ID =
  "opt-0020-m64-n128-k16-cooperative-dot4-fp16-fp32-v1" as const;
export const ACE_OPT_0020_DENSE_TILE_ROWS = 64;
export const ACE_OPT_0020_DENSE_TILE_COLUMNS = 128;
export const ACE_OPT_0020_DENSE_TILE_INNER = 16;
export const ACE_OPT_0020_DENSE_WORKGROUP_SIZE_X = 16;
export const ACE_OPT_0020_DENSE_WORKGROUP_SIZE_Y = 16;
export const ACE_OPT_0020_DENSE_WORKGROUP_SIZE = 256;
export const ACE_OPT_0020_DENSE_ROWS_PER_THREAD = 4;
export const ACE_OPT_0020_DENSE_COLUMNS_PER_THREAD = 8;
export const ACE_OPT_0020_DENSE_ACCUMULATORS_PER_THREAD = 32;
export const ACE_OPT_0020_DENSE_INPUT_PANEL_STRIDE = 17;
export const ACE_OPT_0020_DENSE_WEIGHT_PANEL_STRIDE = 17;
export const ACE_OPT_0020_DENSE_DOT_WIDTH = 4;
export const ACE_OPT_0020_DENSE_BARRIERS_PER_INNER_TILE = 2;

const FLOAT16_BYTES = 2;
const FLOAT32_BYTES = 4;
const GPU_BUFFER_ALIGNMENT = 4;
const PACKED_TILE_COLUMNS = 256;
const PACKED_TILE_INNER = 32;
const PACKED_COLUMNS_PER_RECORD = 8;
const PACKED_RECORDS_PER_ROW =
  PACKED_TILE_COLUMNS / PACKED_COLUMNS_PER_RECORD;
const INPUT_PANEL_ELEMENTS =
  ACE_OPT_0020_DENSE_TILE_ROWS * ACE_OPT_0020_DENSE_INPUT_PANEL_STRIDE;
const WEIGHT_PANEL_ELEMENTS =
  ACE_OPT_0020_DENSE_TILE_COLUMNS * ACE_OPT_0020_DENSE_WEIGHT_PANEL_STRIDE;
export const ACE_OPT_0020_DENSE_WORKGROUP_STORAGE_BYTES =
  (INPUT_PANEL_ELEMENTS + WEIGHT_PANEL_ELEMENTS) * FLOAT16_BYTES;
const MAX_WGSL_U32 = 0xffff_ffff;

const PRODUCTION_DENSE_SHAPES = new Set([
  "2048x2048",
  "2048x1024",
  "2048x6144",
  "6144x2048",
]);

export interface AceOpt0020DenseCooperativeDot4Plan
  extends AceCooperativeGemmPlan {
  readonly kernelSetId:
    typeof ACE_OPT_0020_DENSE_COOPERATIVE_DOT4_KERNEL_SET_ID;
  readonly tileRows: typeof ACE_OPT_0020_DENSE_TILE_ROWS;
  readonly tileColumns: typeof ACE_OPT_0020_DENSE_TILE_COLUMNS;
  readonly tileInner: typeof ACE_OPT_0020_DENSE_TILE_INNER;
  readonly workgroupSize: typeof ACE_OPT_0020_DENSE_WORKGROUP_SIZE;
  readonly workgroupSizeX: typeof ACE_OPT_0020_DENSE_WORKGROUP_SIZE_X;
  readonly workgroupSizeY: typeof ACE_OPT_0020_DENSE_WORKGROUP_SIZE_Y;
  readonly rowsPerThread: typeof ACE_OPT_0020_DENSE_ROWS_PER_THREAD;
  readonly columnsPerThread: typeof ACE_OPT_0020_DENSE_COLUMNS_PER_THREAD;
  readonly accumulatorsPerThread:
    typeof ACE_OPT_0020_DENSE_ACCUMULATORS_PER_THREAD;
  readonly dotWidth: typeof ACE_OPT_0020_DENSE_DOT_WIDTH;
  readonly rowTiles: number;
  readonly columnTiles: number;
  readonly innerTiles: number;
  readonly workgroupCount: number;
  readonly scheduledRows: number;
  readonly scheduledMultiplyAdds: number;
  readonly validMultiplyAdds: number;
  readonly activationElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly inputPanelStride:
    typeof ACE_OPT_0020_DENSE_INPUT_PANEL_STRIDE;
  readonly inputPanelElements: typeof INPUT_PANEL_ELEMENTS;
  readonly weightPanelStride:
    typeof ACE_OPT_0020_DENSE_WEIGHT_PANEL_STRIDE;
  readonly weightPanelElements: typeof WEIGHT_PANEL_ELEMENTS;
  readonly workgroupStorageBytes:
    typeof ACE_OPT_0020_DENSE_WORKGROUP_STORAGE_BYTES;
  readonly barriersPerWorkgroup: number;
  readonly barrierEvents: number;
  readonly estimatedGlobalActivationBytes: number;
  readonly estimatedGlobalWeightBytes: number;
  readonly estimatedGlobalOperandBytes: number;
  readonly estimatedGlobalOutputBytes: number;
  readonly packedWeightStorageShape: readonly [number, number, 32, 256];
}

export interface AceOpt0020DenseCooperativeDot4Dispatch
  extends AceGemmDispatch {
  readonly kernelSetId:
    typeof ACE_OPT_0020_DENSE_COOPERATIVE_DOT4_KERNEL_SET_ID;
  readonly weightLayout: typeof ACE_DIT_DENSE_FP16_TILE_LAYOUT;
  readonly plan: AceOpt0020DenseCooperativeDot4Plan;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Isolated OPT-0020 benchmark owner. It consumes the existing N256/K32 FP16
 * payload, transposes exact N128/K16 quadrants into cooperative storage, and
 * performs ordered K4 FP32 dot updates. It is never a production fallback.
 */
export class AceOpt0020DenseCooperativeDot4Kernel implements AceGemmKernel {
  private readonly pipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(device: GPUDevice): AceOpt0020DenseCooperativeDot4Kernel {
    requireKernelDevice(device);
    return new AceOpt0020DenseCooperativeDot4Kernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0020DenseCooperativeDot4Dispatch> {
    this.requireLive();
    if (bindings.bias !== undefined) {
      throw new RangeError(
        "OPT-0020 repeated-layer cooperative dot4 dense GEMMs reject bias",
      );
    }
    const plan = planAceOpt0020DenseCooperativeDot4(shape);
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
      `${label}-opt-0020-cooperative-dot4-bindings`,
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
      kernelSetId: ACE_OPT_0020_DENSE_COOPERATIVE_DOT4_KERNEL_SET_ID,
      weightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
      plan,
      rangeCount: 1,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(`${label} OPT-0020 dense range must be zero`);
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
    plan: AceOpt0020DenseCooperativeDot4Plan,
  ): Promise<CompiledKernel> {
    const key = shapeKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileCooperativeDot4(this.device, plan);
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
        entries: resources.map((resource, binding) => ({
          binding,
          resource,
        })),
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
      throw new Error("OPT-0020 cooperative dot4 dense kernel was destroyed");
    }
  }
}

export function planAceOpt0020DenseCooperativeDot4(
  shape: AceGemmShape,
): AceOpt0020DenseCooperativeDot4Plan {
  const { rows, inner, columns } = shape;
  requirePositiveSafeInteger(rows, "rows");
  requirePositiveSafeInteger(inner, "inner");
  requirePositiveSafeInteger(columns, "columns");
  if (rows !== 2_250) {
    throw new RangeError("OPT-0020 cooperative dot4 dense requires exact M2250");
  }
  if (!PRODUCTION_DENSE_SHAPES.has(`${inner}x${columns}`)) {
    throw new RangeError(
      `OPT-0020 cooperative dot4 dense rejects non-production K${inner}/N${columns}`,
    );
  }
  if (
    inner % ACE_OPT_0020_DENSE_TILE_INNER !== 0 ||
    inner % PACKED_TILE_INNER !== 0 ||
    columns % ACE_OPT_0020_DENSE_TILE_COLUMNS !== 0 ||
    columns % PACKED_TILE_COLUMNS !== 0
  ) {
    throw new RangeError(
      "OPT-0020 cooperative dot4 dense requires exact K16/N128 panel multiples and N256/K32 packed tiles",
    );
  }

  const activationElements = checkedProduct(rows, inner, "activation elements");
  const weightElements = checkedProduct(inner, columns, "weight elements");
  const outputElements = checkedProduct(rows, columns, "output elements");
  requireWgslIndexable(activationElements, "activation");
  requireWgslIndexable(weightElements, "weight");
  requireWgslIndexable(outputElements, "output");
  const rowTiles = Math.ceil(rows / ACE_OPT_0020_DENSE_TILE_ROWS);
  const columnTiles = columns / ACE_OPT_0020_DENSE_TILE_COLUMNS;
  const innerTiles = inner / ACE_OPT_0020_DENSE_TILE_INNER;
  const workgroupCount = checkedProduct(rowTiles, columnTiles, "workgroups");
  const scheduledRows = checkedProduct(
    rowTiles,
    ACE_OPT_0020_DENSE_TILE_ROWS,
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
    ACE_OPT_0020_DENSE_BARRIERS_PER_INNER_TILE,
    "barriers per workgroup",
  );
  const barrierEvents = checkedProduct(
    workgroupCount,
    barriersPerWorkgroup,
    "barrier events",
  );
  const estimatedGlobalActivationBytes = checkedProduct(
    checkedProduct(activationElements, FLOAT32_BYTES, "activation request bytes"),
    columnTiles,
    "global activation request bytes",
  );
  const estimatedGlobalWeightBytes = checkedProduct(
    checkedProduct(weightElements, FLOAT16_BYTES, "weight request bytes"),
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
    kernelSetId: ACE_OPT_0020_DENSE_COOPERATIVE_DOT4_KERNEL_SET_ID,
    rows,
    inner,
    columns,
    workgroupsX: columnTiles,
    workgroupsY: rowTiles,
    tileRows: ACE_OPT_0020_DENSE_TILE_ROWS,
    tileColumns: ACE_OPT_0020_DENSE_TILE_COLUMNS,
    tileInner: ACE_OPT_0020_DENSE_TILE_INNER,
    workgroupSize: ACE_OPT_0020_DENSE_WORKGROUP_SIZE,
    workgroupSizeX: ACE_OPT_0020_DENSE_WORKGROUP_SIZE_X,
    workgroupSizeY: ACE_OPT_0020_DENSE_WORKGROUP_SIZE_Y,
    rowsPerThread: ACE_OPT_0020_DENSE_ROWS_PER_THREAD,
    columnsPerThread: ACE_OPT_0020_DENSE_COLUMNS_PER_THREAD,
    accumulatorsPerThread: ACE_OPT_0020_DENSE_ACCUMULATORS_PER_THREAD,
    dotWidth: ACE_OPT_0020_DENSE_DOT_WIDTH,
    rowTiles,
    columnTiles,
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
    inputPanelStride: ACE_OPT_0020_DENSE_INPUT_PANEL_STRIDE,
    inputPanelElements: INPUT_PANEL_ELEMENTS,
    weightPanelStride: ACE_OPT_0020_DENSE_WEIGHT_PANEL_STRIDE,
    weightPanelElements: WEIGHT_PANEL_ELEMENTS,
    workgroupStorageBytes: ACE_OPT_0020_DENSE_WORKGROUP_STORAGE_BYTES,
    barriersPerWorkgroup,
    barrierEvents,
    estimatedGlobalActivationBytes,
    estimatedGlobalWeightBytes,
    estimatedGlobalOperandBytes,
    estimatedGlobalOutputBytes,
    packedWeightStorageShape: Object.freeze([
      columns / PACKED_TILE_COLUMNS,
      inner / PACKED_TILE_INNER,
      32,
      256,
    ]) as readonly [number, number, 32, 256],
  });
}

export function aceOpt0020DenseCooperativeDot4Wgsl(
  shape: AceGemmShape,
): string {
  const plan = planAceOpt0020DenseCooperativeDot4(shape);
  const accumulators = Array.from(
    { length: ACE_OPT_0020_DENSE_ROWS_PER_THREAD },
    (_, row) => Array.from(
      { length: ACE_OPT_0020_DENSE_COLUMNS_PER_THREAD },
      (_, column) => `  var acc${row}_${column}: f32 = 0.0;`,
    ).join("\n"),
  ).join("\n");
  const activationVectors = Array.from(
    { length: ACE_OPT_0020_DENSE_ROWS_PER_THREAD },
    (_, row) => /* wgsl */ `
      let a${row} = vec4<f32>(
        f32(input_panel[
          (owned_panel_row + ${row}u) * INPUT_PANEL_STRIDE + inner_k4
        ]),
        f32(input_panel[
          (owned_panel_row + ${row}u) * INPUT_PANEL_STRIDE + inner_k4 + 1u
        ]),
        f32(input_panel[
          (owned_panel_row + ${row}u) * INPUT_PANEL_STRIDE + inner_k4 + 2u
        ]),
        f32(input_panel[
          (owned_panel_row + ${row}u) * INPUT_PANEL_STRIDE + inner_k4 + 3u
        ])
      );`,
  ).join("");
  const contractions = Array.from(
    { length: ACE_OPT_0020_DENSE_COLUMNS_PER_THREAD },
    (_, column) => {
      const updates = Array.from(
        { length: ACE_OPT_0020_DENSE_ROWS_PER_THREAD },
        (_, row) =>
          `      acc${row}_${column} = acc${row}_${column} + dot(a${row}, b${column});`,
      ).join("\n");
      return /* wgsl */ `
      let weight${column}_base =
        (owned_panel_column + ${column}u) * WEIGHT_PANEL_STRIDE + inner_k4;
      let b${column} = vec4<f32>(
        f32(weight_panel[weight${column}_base]),
        f32(weight_panel[weight${column}_base + 1u]),
        f32(weight_panel[weight${column}_base + 2u]),
        f32(weight_panel[weight${column}_base + 3u])
      );
${updates}`;
    },
  ).join("");
  const stores = Array.from(
    { length: ACE_OPT_0020_DENSE_ROWS_PER_THREAD },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + ${row}u;
    if (row < ROWS && column_base + 7u < COLUMNS) {
      let vector_base = row * (COLUMNS / 4u) + column_base / 4u;
      output[vector_base] = vec4<f32>(
        acc${row}_0, acc${row}_1, acc${row}_2, acc${row}_3
      );
      output[vector_base + 1u] = vec4<f32>(
        acc${row}_4, acc${row}_5, acc${row}_6, acc${row}_7
      );
    }
  }`,
  ).join("");
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0020_DENSE_COOPERATIVE_DOT4_KERNEL_SET_ID}
// reduction-semantics: increasing-k4-fp32-widened-dot-plus-scalar-accumulator
// existing payload: N256/K32; cooperative panels: A[64][17], transposed B[128][17]
enable f16;

const ROWS: u32 = ${plan.rows}u;
const INNER: u32 = ${plan.inner}u;
const COLUMNS: u32 = ${plan.columns}u;
const PACKED_INNER_TILES: u32 = ${plan.inner / PACKED_TILE_INNER}u;
const TILE_ROWS: u32 = ${ACE_OPT_0020_DENSE_TILE_ROWS}u;
const TILE_COLUMNS: u32 = ${ACE_OPT_0020_DENSE_TILE_COLUMNS}u;
const TILE_INNER: u32 = ${ACE_OPT_0020_DENSE_TILE_INNER}u;
const INPUT_PANEL_STRIDE: u32 = ${ACE_OPT_0020_DENSE_INPUT_PANEL_STRIDE}u;
const WEIGHT_PANEL_STRIDE: u32 = ${ACE_OPT_0020_DENSE_WEIGHT_PANEL_STRIDE}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

var<workgroup> input_panel: array<f16, ${INPUT_PANEL_ELEMENTS}>;
var<workgroup> weight_panel: array<f16, ${WEIGHT_PANEL_ELEMENTS}>;

fn unpack_f16x4(low: u32, high: u32) -> vec4<f16> {
  let low_pair = unpack2x16float(low);
  let high_pair = unpack2x16float(high);
  return vec4<f16>(
    f16(low_pair.x), f16(low_pair.y), f16(high_pair.x), f16(high_pair.y)
  );
}

@compute @workgroup_size(
  ${ACE_OPT_0020_DENSE_WORKGROUP_SIZE_X},
  ${ACE_OPT_0020_DENSE_WORKGROUP_SIZE_Y},
  1
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
) {
  let row_block = local_id.y;
  let column_block = local_id.x;
  let row_base = group.y * TILE_ROWS + row_block * 4u;
  let column_base = group.x * TILE_COLUMNS + column_block * 8u;
  let owned_panel_row = row_block * 4u;
  let owned_panel_column = column_block * 8u;
${accumulators}

  // Each iteration consumes exactly K=[tile*16, tile*16+15]. Both barriers
  // are unconditional, and each accumulator visits K4 bases 0,4,8,12.
  for (var inner_tile = 0u; inner_tile < ${plan.innerTiles}u; inner_tile += 1u) {
    // A is row-major FP32. Each invocation rounds one contiguous vec4 to FP16.
    let panel_a_row = local_index / 4u;
    let panel_a_k4 = (local_index % 4u) * 4u;
    let global_a_row = group.y * TILE_ROWS + panel_a_row;
    let global_a_inner = inner_tile * TILE_INNER + panel_a_k4;
    var a_value = vec4<f16>(0.0h);
    if (global_a_row < ROWS) {
      let activation_base = global_a_row * INNER + global_a_inner;
      a_value = vec4<f16>(
        f16(activation[activation_base]),
        f16(activation[activation_base + 1u]),
        f16(activation[activation_base + 2u]),
        f16(activation[activation_base + 3u])
      );
    }
    let input_panel_base =
      panel_a_row * INPUT_PANEL_STRIDE + panel_a_k4;
    input_panel[input_panel_base] = a_value.x;
    input_panel[input_panel_base + 1u] = a_value.y;
    input_panel[input_panel_base + 2u] = a_value.z;
    input_panel[input_panel_base + 3u] = a_value.w;

    // The 16x16 grid owns the 16 K rows x 16 packed eight-column records
    // in one N128/K16 quadrant, then transposes them into B[128][17].
    let packed_inner = inner_tile * TILE_INNER + local_id.y;
    let packed_n256_tile = group.x / 2u;
    let packed_n128_half = group.x % 2u;
    let packed_k32_tile = packed_inner / 32u;
    let packed_k_in_tile = packed_inner % 32u;
    let packed_n8_record = packed_n128_half * 16u + local_id.x;
    let packed_record_index =
      ((packed_n256_tile * PACKED_INNER_TILES + packed_k32_tile) * 32u +
        packed_k_in_tile) * ${PACKED_RECORDS_PER_ROW}u + packed_n8_record;
    let packed = weight[packed_record_index];
    let packed_low = unpack_f16x4(packed.x, packed.y);
    let packed_high = unpack_f16x4(packed.z, packed.w);
    let weight_panel_column = local_id.x * 8u;
    let weight_panel_inner = local_id.y;
    weight_panel[
      weight_panel_column * WEIGHT_PANEL_STRIDE + weight_panel_inner
    ] = packed_low.x;
    weight_panel[
      (weight_panel_column + 1u) * WEIGHT_PANEL_STRIDE + weight_panel_inner
    ] = packed_low.y;
    weight_panel[
      (weight_panel_column + 2u) * WEIGHT_PANEL_STRIDE + weight_panel_inner
    ] = packed_low.z;
    weight_panel[
      (weight_panel_column + 3u) * WEIGHT_PANEL_STRIDE + weight_panel_inner
    ] = packed_low.w;
    weight_panel[
      (weight_panel_column + 4u) * WEIGHT_PANEL_STRIDE + weight_panel_inner
    ] = packed_high.x;
    weight_panel[
      (weight_panel_column + 5u) * WEIGHT_PANEL_STRIDE + weight_panel_inner
    ] = packed_high.y;
    weight_panel[
      (weight_panel_column + 6u) * WEIGHT_PANEL_STRIDE + weight_panel_inner
    ] = packed_high.z;
    weight_panel[
      (weight_panel_column + 7u) * WEIGHT_PANEL_STRIDE + weight_panel_inner
    ] = packed_high.w;

    workgroupBarrier();
    for (var inner_k4 = 0u; inner_k4 < TILE_INNER; inner_k4 += 4u) {${activationVectors}
${contractions}
    }
    workgroupBarrier();
  }
${stores}
}
`;
}

async function compileCooperativeDot4(
  device: GPUDevice,
  plan: AceOpt0020DenseCooperativeDot4Plan,
): Promise<CompiledKernel> {
  const label = `ace-opt-0020-dense-${plan.rows}x${plan.inner}x${plan.columns}`;
  const module = await checkedShaderModule(
    device,
    label,
    aceOpt0020DenseCooperativeDot4Wgsl(plan),
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
        type: binding === 2
          ? "storage" as const
          : "read-only-storage" as const,
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
      `OPT-0020 cooperative dot4 dense WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  return module;
}

function requireKernelDevice(device: GPUDevice): void {
  if (!device.features.has("shader-f16")) {
    throw new Error(
      "OPT-0020 cooperative dot4 dense requires WebGPU shader-f16",
    );
  }
  const invocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const sizeX = device.limits.maxComputeWorkgroupSizeX;
  const sizeY = device.limits.maxComputeWorkgroupSizeY;
  const storage = device.limits.maxComputeWorkgroupStorageSize;
  if (
    !Number.isSafeInteger(invocations) ||
    !Number.isSafeInteger(sizeX) ||
    !Number.isSafeInteger(sizeY) ||
    invocations < ACE_OPT_0020_DENSE_WORKGROUP_SIZE ||
    sizeX < ACE_OPT_0020_DENSE_WORKGROUP_SIZE_X ||
    sizeY < ACE_OPT_0020_DENSE_WORKGROUP_SIZE_Y
  ) {
    throw new Error(
      "OPT-0020 cooperative dot4 dense requires WG256 with a 16x16 workgroup",
    );
  }
  if (
    !Number.isSafeInteger(storage) ||
    storage < ACE_OPT_0020_DENSE_WORKGROUP_STORAGE_BYTES
  ) {
    throw new Error(
      `OPT-0020 cooperative dot4 dense requires ${ACE_OPT_0020_DENSE_WORKGROUP_STORAGE_BYTES} workgroup-storage bytes`,
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
      "OPT-0020 cooperative dot4 dense exceeds the device dispatch dimension",
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
      "OPT-0020 cooperative dot4 dense device reported invalid buffer limits",
    );
  }
  for (const [name, bytes] of resources) {
    if (bytes > maximumBinding || bytes > maximumBuffer) {
      throw new RangeError(
        `OPT-0020 cooperative dot4 dense ${name} exceeds the device buffer limits`,
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
      "OPT-0020 cooperative dot4 dense device reported an invalid storage alignment",
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
      `OPT-0020 cooperative dot4 dense ${label} must be a positive safe integer`,
    );
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(
      `OPT-0020 cooperative dot4 dense ${label} is not a safe integer`,
    );
  }
  return product;
}

function checkedSum(left: number, right: number, label: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw new RangeError(
      `OPT-0020 cooperative dot4 dense ${label} is not a safe integer`,
    );
  }
  return sum;
}

function requireWgslIndexable(elements: number, label: string): void {
  if (elements - 1 > MAX_WGSL_U32) {
    throw new RangeError(
      `OPT-0020 cooperative dot4 dense ${label} exceeds WGSL u32 indexing`,
    );
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    value % GPU_BUFFER_ALIGNMENT === 0 &&
    (value & (value - 1)) === 0;
}
