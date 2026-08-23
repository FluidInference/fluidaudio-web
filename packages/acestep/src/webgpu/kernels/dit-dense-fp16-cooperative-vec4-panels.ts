import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../../model/manifest.js";
import {
  requireAceDisjointOutput,
} from "./correctness-utils.js";
import type {
  AceCooperativeGemmPlan,
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmKernel,
  AceGemmOutputRange,
  AceGemmShape,
} from "./gemm.js";

export const ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID =
  "opt-0021-m64-n128-k16-cooperative-vec4-panels-fp16-fp32-v1" as const;
export const ACE_OPT_0021_DENSE_TILE_ROWS = 64;
export const ACE_OPT_0021_DENSE_TILE_COLUMNS = 128;
export const ACE_OPT_0021_DENSE_TILE_INNER = 16;
export const ACE_OPT_0021_DENSE_WORKGROUP_SIZE_X = 16;
export const ACE_OPT_0021_DENSE_WORKGROUP_SIZE_Y = 16;
export const ACE_OPT_0021_DENSE_WORKGROUP_SIZE = 256;
export const ACE_OPT_0021_DENSE_ROWS_PER_THREAD = 4;
export const ACE_OPT_0021_DENSE_COLUMNS_PER_THREAD = 8;
export const ACE_OPT_0021_DENSE_ACCUMULATORS_PER_THREAD = 32;
export const ACE_OPT_0021_DENSE_INPUT_PANEL_STRIDE = 17;
export const ACE_OPT_0021_DENSE_WEIGHT_PANEL_STRIDE = 33;
export const ACE_OPT_0021_DENSE_BARRIERS_PER_INNER_TILE = 2;

const FLOAT16_BYTES = 2;
const FLOAT16_VECTOR_WIDTH = 4;
const FLOAT16_VECTOR_BYTES = FLOAT16_BYTES * FLOAT16_VECTOR_WIDTH;
const FLOAT32_BYTES = 4;
const GPU_BUFFER_ALIGNMENT = 4;
const PACKED_TILE_COLUMNS = 256;
const PACKED_TILE_INNER = 32;
const PACKED_COLUMNS_PER_RECORD = 8;
const PACKED_RECORDS_PER_ROW =
  PACKED_TILE_COLUMNS / PACKED_COLUMNS_PER_RECORD;
const INPUT_PANEL_ELEMENTS =
  ACE_OPT_0021_DENSE_TILE_INNER * ACE_OPT_0021_DENSE_INPUT_PANEL_STRIDE;
const WEIGHT_PANEL_ELEMENTS =
  ACE_OPT_0021_DENSE_TILE_INNER * ACE_OPT_0021_DENSE_WEIGHT_PANEL_STRIDE;
export const ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES =
  (INPUT_PANEL_ELEMENTS + WEIGHT_PANEL_ELEMENTS) * FLOAT16_VECTOR_BYTES;
const MAX_WGSL_U32 = 0xffff_ffff;

const PRODUCTION_DENSE_SHAPES = new Set([
  "2048x2048",
  "2048x1024",
  "2048x6144",
  "6144x2048",
]);

export interface AceOpt0021DenseCooperativeVec4PanelsPlan
  extends AceCooperativeGemmPlan {
  readonly kernelSetId:
    typeof ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID;
  readonly tileRows: typeof ACE_OPT_0021_DENSE_TILE_ROWS;
  readonly tileColumns: typeof ACE_OPT_0021_DENSE_TILE_COLUMNS;
  readonly tileInner: typeof ACE_OPT_0021_DENSE_TILE_INNER;
  readonly workgroupSize: typeof ACE_OPT_0021_DENSE_WORKGROUP_SIZE;
  readonly workgroupSizeX: typeof ACE_OPT_0021_DENSE_WORKGROUP_SIZE_X;
  readonly workgroupSizeY: typeof ACE_OPT_0021_DENSE_WORKGROUP_SIZE_Y;
  readonly rowsPerThread: typeof ACE_OPT_0021_DENSE_ROWS_PER_THREAD;
  readonly columnsPerThread: typeof ACE_OPT_0021_DENSE_COLUMNS_PER_THREAD;
  readonly accumulatorsPerThread:
    typeof ACE_OPT_0021_DENSE_ACCUMULATORS_PER_THREAD;
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
    typeof ACE_OPT_0021_DENSE_INPUT_PANEL_STRIDE;
  readonly inputPanelElements: typeof INPUT_PANEL_ELEMENTS;
  readonly weightPanelStride:
    typeof ACE_OPT_0021_DENSE_WEIGHT_PANEL_STRIDE;
  readonly weightPanelElements: typeof WEIGHT_PANEL_ELEMENTS;
  readonly workgroupStorageBytes:
    typeof ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES;
  readonly barriersPerWorkgroup: number;
  readonly barrierEvents: number;
  readonly estimatedGlobalActivationBytes: number;
  readonly estimatedGlobalWeightBytes: number;
  readonly estimatedGlobalOperandBytes: number;
  readonly estimatedGlobalOutputBytes: number;
  readonly packedWeightStorageShape: readonly [number, number, 32, 256];
}

export interface AceOpt0021DenseCooperativeVec4PanelsDispatch
  extends AceGemmDispatch {
  readonly kernelSetId:
    typeof ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID;
  readonly weightLayout: typeof ACE_DIT_DENSE_FP16_TILE_LAYOUT;
  readonly plan: AceOpt0021DenseCooperativeVec4PanelsPlan;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Isolated OPT-0021 benchmark owner. It consumes the existing N256/K32 FP16
 * payload, stages exact N128/K16 quadrants, and never participates in a
 * production selector or fallback.
 */
export class AceOpt0021DenseCooperativeVec4PanelsKernel
  implements AceGemmKernel {
  private readonly pipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
  ): AceOpt0021DenseCooperativeVec4PanelsKernel {
    requireKernelDevice(device);
    return new AceOpt0021DenseCooperativeVec4PanelsKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0021DenseCooperativeVec4PanelsDispatch> {
    this.requireLive();
    if (bindings.bias !== undefined) {
      throw new RangeError(
        "OPT-0021 repeated-layer cooperative dense GEMMs reject bias",
      );
    }
    const plan = planAceOpt0021DenseCooperativeVec4Panels(shape);
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
      `${label}-opt-0021-cooperative-bindings`,
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
      kernelSetId: ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID,
      weightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
      plan,
      rangeCount: 1,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(`${label} OPT-0021 dense range must be zero`);
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
    plan: AceOpt0021DenseCooperativeVec4PanelsPlan,
  ): Promise<CompiledKernel> {
    const key = shapeKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileCooperativeVec4Panels(this.device, plan);
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
      throw new Error("OPT-0021 cooperative dense kernel was destroyed");
    }
  }
}

export function planAceOpt0021DenseCooperativeVec4Panels(
  shape: AceGemmShape,
): AceOpt0021DenseCooperativeVec4PanelsPlan {
  const { rows, inner, columns } = shape;
  requirePositiveSafeInteger(rows, "rows");
  requirePositiveSafeInteger(inner, "inner");
  requirePositiveSafeInteger(columns, "columns");
  if (rows !== 2_250) {
    throw new RangeError("OPT-0021 cooperative dense requires exact M2250");
  }
  if (!PRODUCTION_DENSE_SHAPES.has(`${inner}x${columns}`)) {
    throw new RangeError(
      `OPT-0021 cooperative dense rejects non-production K${inner}/N${columns}`,
    );
  }
  if (
    inner % ACE_OPT_0021_DENSE_TILE_INNER !== 0 ||
    inner % PACKED_TILE_INNER !== 0 ||
    columns % ACE_OPT_0021_DENSE_TILE_COLUMNS !== 0 ||
    columns % PACKED_TILE_COLUMNS !== 0
  ) {
    throw new RangeError(
      "OPT-0021 cooperative dense requires exact K16/N128 panel multiples and N256/K32 packed tiles",
    );
  }

  const activationElements = checkedProduct(rows, inner, "activation elements");
  const weightElements = checkedProduct(inner, columns, "weight elements");
  const outputElements = checkedProduct(rows, columns, "output elements");
  requireWgslIndexable(activationElements, "activation");
  requireWgslIndexable(weightElements, "weight");
  requireWgslIndexable(outputElements, "output");
  const rowTiles = Math.ceil(rows / ACE_OPT_0021_DENSE_TILE_ROWS);
  const columnTiles = columns / ACE_OPT_0021_DENSE_TILE_COLUMNS;
  const innerTiles = inner / ACE_OPT_0021_DENSE_TILE_INNER;
  const workgroupCount = checkedProduct(rowTiles, columnTiles, "workgroups");
  const scheduledRows = checkedProduct(
    rowTiles,
    ACE_OPT_0021_DENSE_TILE_ROWS,
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
    ACE_OPT_0021_DENSE_BARRIERS_PER_INNER_TILE,
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
    kernelSetId: ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID,
    rows,
    inner,
    columns,
    workgroupsX: columnTiles,
    workgroupsY: rowTiles,
    tileRows: ACE_OPT_0021_DENSE_TILE_ROWS,
    tileColumns: ACE_OPT_0021_DENSE_TILE_COLUMNS,
    tileInner: ACE_OPT_0021_DENSE_TILE_INNER,
    workgroupSize: ACE_OPT_0021_DENSE_WORKGROUP_SIZE,
    workgroupSizeX: ACE_OPT_0021_DENSE_WORKGROUP_SIZE_X,
    workgroupSizeY: ACE_OPT_0021_DENSE_WORKGROUP_SIZE_Y,
    rowsPerThread: ACE_OPT_0021_DENSE_ROWS_PER_THREAD,
    columnsPerThread: ACE_OPT_0021_DENSE_COLUMNS_PER_THREAD,
    accumulatorsPerThread: ACE_OPT_0021_DENSE_ACCUMULATORS_PER_THREAD,
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
    inputPanelStride: ACE_OPT_0021_DENSE_INPUT_PANEL_STRIDE,
    inputPanelElements: INPUT_PANEL_ELEMENTS,
    weightPanelStride: ACE_OPT_0021_DENSE_WEIGHT_PANEL_STRIDE,
    weightPanelElements: WEIGHT_PANEL_ELEMENTS,
    workgroupStorageBytes: ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES,
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

export function aceOpt0021DenseCooperativeVec4PanelsWgsl(
  shape: AceGemmShape,
): string {
  const plan = planAceOpt0021DenseCooperativeVec4Panels(shape);
  const rowComponents = ["x", "y", "z", "w"] as const;
  const accumulators = Array.from(
    { length: ACE_OPT_0021_DENSE_ROWS_PER_THREAD },
    (_, row) =>
      `  var acc${row}_0 = vec4<f32>(0.0);\n` +
      `  var acc${row}_1 = vec4<f32>(0.0);`,
  ).join("\n");
  const k4Groups = [0, 4, 8, 12].map((base) => {
    const activationLoads = Array.from({ length: 4 }, (_, offset) => {
      const inner = base + offset;
      return /* wgsl */ `      let a${inner} = vec4<f32>(
        input_panel[${inner}u * INPUT_PANEL_STRIDE + row_block]
      );`;
    }).join("\n");
    const weightLoads = Array.from({ length: 4 }, (_, offset) => {
      const inner = base + offset;
      return /* wgsl */ `      let b${inner}_0 = vec4<f32>(
        weight_panel[
          ${inner}u * WEIGHT_PANEL_STRIDE + column_block * 2u
        ]
      );
      let b${inner}_1 = vec4<f32>(
        weight_panel[
          ${inner}u * WEIGHT_PANEL_STRIDE + column_block * 2u + 1u
        ]
      );`;
    }).join("\n");
    const orderedUpdates = Array.from({ length: 4 }, (_, offset) => {
      const inner = base + offset;
      return Array.from(
        { length: ACE_OPT_0021_DENSE_ROWS_PER_THREAD },
        (_, row) => /* wgsl */ `      acc${row}_0 = acc${row}_0 +
        vec4<f32>(a${inner}.${rowComponents[row]!}) * b${inner}_0;
      acc${row}_1 = acc${row}_1 +
        vec4<f32>(a${inner}.${rowComponents[row]!}) * b${inner}_1;`,
      ).join("\n");
    }).join("\n");
    return /* wgsl */ `
    {
      // lexical-k4-base: ${base}
${activationLoads}
${weightLoads}
${orderedUpdates}
    }`;
  }).join("");
  const stores = Array.from(
    { length: ACE_OPT_0021_DENSE_ROWS_PER_THREAD },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + ${row}u;
    if (row < ROWS && column_base + 7u < COLUMNS) {
      let vector_base = row * (COLUMNS / 4u) + column_base / 4u;
      output[vector_base] = acc${row}_0;
      output[vector_base + 1u] = acc${row}_1;
    }
  }`,
  ).join("");
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0021_DENSE_COOPERATIVE_VEC4_PANELS_KERNEL_SET_ID}
// reduction-semantics: strict-increasing-k-fp32-sum-plus-product
// existing payload: N256/K32; cooperative panel: M64/N128/K16
enable f16;

const ROWS: u32 = ${plan.rows}u;
const INNER: u32 = ${plan.inner}u;
const COLUMNS: u32 = ${plan.columns}u;
const PACKED_INNER_TILES: u32 = ${plan.inner / PACKED_TILE_INNER}u;
const TILE_ROWS: u32 = ${ACE_OPT_0021_DENSE_TILE_ROWS}u;
const TILE_COLUMNS: u32 = ${ACE_OPT_0021_DENSE_TILE_COLUMNS}u;
const TILE_INNER: u32 = ${ACE_OPT_0021_DENSE_TILE_INNER}u;
const INPUT_PANEL_STRIDE: u32 = ${ACE_OPT_0021_DENSE_INPUT_PANEL_STRIDE}u;
const WEIGHT_PANEL_STRIDE: u32 = ${ACE_OPT_0021_DENSE_WEIGHT_PANEL_STRIDE}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

var<workgroup> input_panel: array<vec4<f16>, ${INPUT_PANEL_ELEMENTS}>;
var<workgroup> weight_panel: array<vec4<f16>, ${WEIGHT_PANEL_ELEMENTS}>;

fn unpack_f16x4(low: u32, high: u32) -> vec4<f16> {
  let low_pair = unpack2x16float(low);
  let high_pair = unpack2x16float(high);
  return vec4<f16>(
    f16(low_pair.x), f16(low_pair.y), f16(high_pair.x), f16(high_pair.y)
  );
}

@compute @workgroup_size(
  ${ACE_OPT_0021_DENSE_WORKGROUP_SIZE_X},
  ${ACE_OPT_0021_DENSE_WORKGROUP_SIZE_Y},
  1
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row_block = local_id.y;
  let column_block = local_id.x;
  let row_base = group.y * TILE_ROWS + row_block * 4u;
  let column_base = group.x * TILE_COLUMNS + column_block * 8u;
${accumulators}

  // Each iteration consumes exactly K=[tile*16, tile*16+15]. Both barriers
  // are unconditional, and every final accumulator sees increasing K.
  for (var inner_tile = 0u; inner_tile < ${plan.innerTiles}u; inner_tile += 1u) {
    // Invocation (x,y) owns A[K=x][row-quad=y]. Tail rows are independently
    // guarded so no robust out-of-bounds behavior belongs to the mechanism.
    let global_a_inner = inner_tile * TILE_INNER + local_id.x;
    let global_a_row = group.y * TILE_ROWS + local_id.y * 4u;
    var a_value = vec4<f16>(0.0h);
    if (global_a_row < ROWS) {
      a_value.x = f16(activation[global_a_row * INNER + global_a_inner]);
    }
    if (global_a_row + 1u < ROWS) {
      a_value.y = f16(activation[
        (global_a_row + 1u) * INNER + global_a_inner
      ]);
    }
    if (global_a_row + 2u < ROWS) {
      a_value.z = f16(activation[
        (global_a_row + 2u) * INNER + global_a_inner
      ]);
    }
    if (global_a_row + 3u < ROWS) {
      a_value.w = f16(activation[
        (global_a_row + 3u) * INNER + global_a_inner
      ]);
    }
    input_panel[
      local_id.x * INPUT_PANEL_STRIDE + local_id.y
    ] = a_value;

    // The 16x16 invocation grid exactly owns the 16 K rows x 16 packed
    // eight-column records in one N128/K16 quadrant of the N256/K32 payload.
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
    let weight_panel_base =
      local_id.y * WEIGHT_PANEL_STRIDE + local_id.x * 2u;
    weight_panel[weight_panel_base] = packed_low;
    weight_panel[weight_panel_base + 1u] = packed_high;

    workgroupBarrier();
${k4Groups}
    workgroupBarrier();
  }
${stores}
}
`;
}

async function compileCooperativeVec4Panels(
  device: GPUDevice,
  plan: AceOpt0021DenseCooperativeVec4PanelsPlan,
): Promise<CompiledKernel> {
  const label = `ace-opt-0021-dense-${plan.rows}x${plan.inner}x${plan.columns}`;
  const module = await checkedShaderModule(
    device,
    label,
    aceOpt0021DenseCooperativeVec4PanelsWgsl(plan),
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
      `OPT-0021 cooperative dense WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  return module;
}

function requireKernelDevice(device: GPUDevice): void {
  if (!device.features.has("shader-f16")) {
    throw new Error("OPT-0021 cooperative dense requires WebGPU shader-f16");
  }
  const invocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const sizeX = device.limits.maxComputeWorkgroupSizeX;
  const sizeY = device.limits.maxComputeWorkgroupSizeY;
  const storage = device.limits.maxComputeWorkgroupStorageSize;
  if (
    !Number.isSafeInteger(invocations) ||
    !Number.isSafeInteger(sizeX) ||
    !Number.isSafeInteger(sizeY) ||
    invocations < ACE_OPT_0021_DENSE_WORKGROUP_SIZE ||
    sizeX < ACE_OPT_0021_DENSE_WORKGROUP_SIZE_X ||
    sizeY < ACE_OPT_0021_DENSE_WORKGROUP_SIZE_Y
  ) {
    throw new Error(
      "OPT-0021 cooperative dense requires WG256 with a 16x16 workgroup",
    );
  }
  if (
    !Number.isSafeInteger(storage) ||
    storage < ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES
  ) {
    throw new Error(
      `OPT-0021 cooperative dense requires ${ACE_OPT_0021_DENSE_WORKGROUP_STORAGE_BYTES} workgroup-storage bytes`,
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
      "OPT-0021 cooperative dense exceeds the device dispatch dimension",
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
      "OPT-0021 cooperative dense device reported invalid buffer limits",
    );
  }
  for (const [name, bytes] of resources) {
    if (bytes > maximumBinding || bytes > maximumBuffer) {
      throw new RangeError(
        `OPT-0021 cooperative dense ${name} exceeds the device buffer limits`,
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
      "OPT-0021 cooperative dense device reported an invalid storage alignment",
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
      `OPT-0021 cooperative dense ${label} must be a positive safe integer`,
    );
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(
      `OPT-0021 cooperative dense ${label} is not a safe integer`,
    );
  }
  return product;
}

function checkedSum(left: number, right: number, label: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw new RangeError(
      `OPT-0021 cooperative dense ${label} is not a safe integer`,
    );
  }
  return sum;
}

function requireWgslIndexable(elements: number, label: string): void {
  if (elements > MAX_WGSL_U32) {
    throw new RangeError(
      `OPT-0021 cooperative dense ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
