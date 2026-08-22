import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../../model/manifest.js";
import {
  requireAceBindingBytes,
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

export const ACE_OPT_0009_DENSE_TILE_ROWS = 32;
export const ACE_OPT_0009_DENSE_TILE_COLUMNS = 256;
export const ACE_OPT_0009_DENSE_TILE_INNER = 32;
export const ACE_OPT_0009_DENSE_WORKGROUP_SIZE = 128;
export const ACE_OPT_0009_DENSE_SUBGROUP_SIZE = 32;

const SUBGROUPS_PER_WORKGROUP =
  ACE_OPT_0009_DENSE_WORKGROUP_SIZE / ACE_OPT_0009_DENSE_SUBGROUP_SIZE;
const ROWS_PER_SUBGROUP =
  ACE_OPT_0009_DENSE_TILE_ROWS / SUBGROUPS_PER_WORKGROUP;
const SCALARS_PER_LANE =
  ACE_OPT_0009_DENSE_TILE_COLUMNS / ACE_OPT_0009_DENSE_SUBGROUP_SIZE;
const MAX_DISPATCH_DIMENSION = 65_535;
const MAX_WGSL_U32 = 0xffff_ffff;

const PRODUCTION_DENSE_SHAPES = new Set([
  "2048x2048",
  "2048x1024",
  "2048x6144",
  "6144x2048",
]);

export interface AceOpt0009DenseGemmPlan extends AceCooperativeGemmPlan {
  readonly tileRows: typeof ACE_OPT_0009_DENSE_TILE_ROWS;
  readonly tileColumns: typeof ACE_OPT_0009_DENSE_TILE_COLUMNS;
  readonly tileInner: typeof ACE_OPT_0009_DENSE_TILE_INNER;
  readonly workgroupSize: typeof ACE_OPT_0009_DENSE_WORKGROUP_SIZE;
  readonly subgroupSize: typeof ACE_OPT_0009_DENSE_SUBGROUP_SIZE;
  readonly rowTiles: number;
  readonly columnTiles: number;
  readonly innerTiles: number;
  readonly workgroupCount: number;
  readonly activationElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly packedWeightStorageShape: readonly [number, number, 32, 256];
}

export interface AceOpt0009DenseGemmDispatch extends AceGemmDispatch {
  readonly weightLayout: typeof ACE_DIT_DENSE_FP16_TILE_LAYOUT;
  readonly plan: AceOpt0009DenseGemmPlan;
}

/**
 * OPT-0009's selected N256/K32 subgroup geometry: IEEE-FP16 operands,
 * source-order FP32 accumulation, and FP32 output. Production activations stay
 * in the reference graph's FP32 arena and are rounded to FP16 at the load.
 */
export class AceOpt0009DenseGemmKernel implements AceGemmKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0009DenseGemmKernel {
    if (
      !device.features.has("shader-f16") ||
      !device.features.has("subgroups") ||
      capability.subgroupMinSize !== ACE_OPT_0009_DENSE_SUBGROUP_SIZE ||
      capability.subgroupMaxSize !== ACE_OPT_0009_DENSE_SUBGROUP_SIZE
    ) {
      throw new Error(
        "OPT-0009 dense GEMM requires shader-f16 and fixed 32-lane subgroups",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0009_DENSE_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX <
        ACE_OPT_0009_DENSE_WORKGROUP_SIZE
    ) {
      throw new Error(
        `OPT-0009 dense GEMM requires WG${ACE_OPT_0009_DENSE_WORKGROUP_SIZE}`,
      );
    }
    return new AceOpt0009DenseGemmKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0009DenseGemmDispatch> {
    if (this.destroyed) {
      throw new Error("OPT-0009 dense GEMM kernel was destroyed");
    }
    if (bindings.bias !== undefined) {
      throw new Error("OPT-0009 repeated-layer dense GEMMs do not accept bias");
    }
    const plan = planAceOpt0009DenseGemm(shape);
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
    const pipeline = await this.pipelineFor(plan);
    if (this.destroyed) {
      throw new Error("OPT-0009 dense GEMM kernel was destroyed while compiling");
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0009-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: bindings.activation },
        { binding: 1, resource: bindings.weight },
        { binding: 2, resource: bindings.output },
      ],
    });
    return Object.freeze({
      label,
      weightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
      plan,
      rangeCount: 1,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(`${label} OPT-0009 dense range must be zero`);
        }
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(plan.columnTiles, plan.rowTiles, 1);
      },
      encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(plan.columnTiles, plan.rowTiles, 1);
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
    const created = compileAceOpt0009DenseGemm(this.device, shape);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

export function planAceOpt0009DenseGemm(
  shape: AceGemmShape,
): AceOpt0009DenseGemmPlan {
  const { rows, inner, columns } = shape;
  requirePositiveSafeInteger(rows, "rows");
  requirePositiveSafeInteger(inner, "inner");
  requirePositiveSafeInteger(columns, "columns");
  if (!PRODUCTION_DENSE_SHAPES.has(`${inner}x${columns}`)) {
    throw new RangeError(
      `OPT-0009 dense GEMM rejects non-production K${inner}/N${columns}`,
    );
  }
  const activationElements = checkedProduct(rows, inner, "activation");
  const weightElements = checkedProduct(columns, inner, "weight");
  const outputElements = checkedProduct(rows, columns, "output");
  requireWgslIndexable(activationElements, "activation");
  requireWgslIndexable(weightElements, "weight");
  requireWgslIndexable(outputElements, "output");
  const rowTiles = Math.ceil(rows / ACE_OPT_0009_DENSE_TILE_ROWS);
  const columnTiles = columns / ACE_OPT_0009_DENSE_TILE_COLUMNS;
  const innerTiles = inner / ACE_OPT_0009_DENSE_TILE_INNER;
  if (rowTiles > MAX_DISPATCH_DIMENSION || columnTiles > MAX_DISPATCH_DIMENSION) {
    throw new RangeError("OPT-0009 dense GEMM exceeds WebGPU dispatch dimensions");
  }
  const workgroupCount = checkedProduct(rowTiles, columnTiles, "workgroups");
  const scheduledRows = checkedProduct(
    rowTiles,
    ACE_OPT_0009_DENSE_TILE_ROWS,
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
    tileRows: ACE_OPT_0009_DENSE_TILE_ROWS,
    tileColumns: ACE_OPT_0009_DENSE_TILE_COLUMNS,
    tileInner: ACE_OPT_0009_DENSE_TILE_INNER,
    workgroupSize: ACE_OPT_0009_DENSE_WORKGROUP_SIZE,
    subgroupSize: ACE_OPT_0009_DENSE_SUBGROUP_SIZE,
    rowTiles,
    columnTiles,
    innerTiles,
    workgroupCount,
    activationElements,
    weightElements,
    outputElements,
    outputRangeCount: 1,
    outputRanges,
    packedWeightStorageShape: Object.freeze([
      columnTiles,
      innerTiles,
      32,
      256,
    ]) as readonly [number, number, 32, 256],
  });
}

export function aceOpt0009DenseGemmWgsl(shape: AceGemmShape): string {
  const plan = planAceOpt0009DenseGemm(shape);
  const declarations = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) =>
      `  var acc${row}_0 = vec4<f32>(0.0);\n` +
      `  var acc${row}_1 = vec4<f32>(0.0);`,
  ).join("\n");
  const broadcasts = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `      let a${row} = subgroupBroadcast(lane_a, ${row}u);`,
  ).join("\n");
  const contractions = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) =>
      `      acc${row}_0 = acc${row}_0 + vec4<f32>(f32(a${row})) * vec4<f32>(b0);\n` +
      `      acc${row}_1 = acc${row}_1 + vec4<f32>(f32(a${row})) * vec4<f32>(b1);`,
  ).join("\n");
  const stores = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + ${row}u;
    if (row < ROWS) {
      let vector_base = row * (COLUMNS / 4u) + column_base / 4u;
      output[vector_base] = acc${row}_0;
      output[vector_base + 1u] = acc${row}_1;
    }
  }`,
  ).join("\n");
  return /* wgsl */ `
enable f16;
enable subgroups;

const ROWS = ${plan.rows}u;
const INNER = ${plan.inner}u;
const COLUMNS = ${plan.columns}u;
const INNER_TILES = ${plan.innerTiles}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

fn unpack_f16x4(low: u32, high: u32) -> vec4<f16> {
  let low_pair = unpack2x16float(low);
  let high_pair = unpack2x16float(high);
  return vec4<f16>(
    f16(low_pair.x), f16(low_pair.y), f16(high_pair.x), f16(high_pair.y)
  );
}

@compute @workgroup_size(${ACE_OPT_0009_DENSE_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (
    subgroup_size != ${ACE_OPT_0009_DENSE_SUBGROUP_SIZE}u ||
    group.x >= ${plan.columnTiles}u ||
    group.y >= ${plan.rowTiles}u ||
    group.z != 0u ||
    subgroup >= ${SUBGROUPS_PER_WORKGROUP}u
  ) {
    return;
  }
  let row_base =
    group.y * ${ACE_OPT_0009_DENSE_TILE_ROWS}u +
    subgroup * ${ROWS_PER_SUBGROUP}u;
  let column_base =
    group.x * ${ACE_OPT_0009_DENSE_TILE_COLUMNS}u +
    subgroup_lane * ${SCALARS_PER_LANE}u;
${declarations}

  for (var inner_tile = 0u; inner_tile < INNER_TILES; inner_tile += 1u) {
    let weight_tile_base =
      (group.x * INNER_TILES + inner_tile) *
      ${ACE_OPT_0009_DENSE_TILE_INNER * ACE_OPT_0009_DENSE_SUBGROUP_SIZE}u;
    for (
      var inner_in_tile = 0u;
      inner_in_tile < ${ACE_OPT_0009_DENSE_TILE_INNER}u;
      inner_in_tile += 1u
    ) {
      let inner = inner_tile * ${ACE_OPT_0009_DENSE_TILE_INNER}u + inner_in_tile;
      var lane_a = 0.0h;
      let lane_row = row_base + subgroup_lane;
      if (subgroup_lane < ${ROWS_PER_SUBGROUP}u && lane_row < ROWS) {
        lane_a = f16(activation[lane_row * INNER + inner]);
      }
      let packed_b = weight[
        weight_tile_base +
        inner_in_tile * ${ACE_OPT_0009_DENSE_SUBGROUP_SIZE}u +
        subgroup_lane
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

async function compileAceOpt0009DenseGemm(
  device: GPUDevice,
  shape: AceGemmShape,
): Promise<GPUComputePipeline> {
  const label = `ace-opt-0009-dense-${shape.rows}x${shape.inner}x${shape.columns}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0009DenseGemmWgsl(shape),
  });
  return await device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
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
      `OPT-0009 dense GEMM ${label} must be a positive safe integer`,
    );
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(`OPT-0009 dense GEMM ${label} is not a safe integer`);
  }
  return product;
}

function checkedBytes(elements: number, itemBytes: number, label: string): number {
  return checkedProduct(elements, itemBytes, `${label} bytes`);
}

function requireWgslIndexable(elements: number, label: string): void {
  if (elements > MAX_WGSL_U32) {
    throw new RangeError(
      `OPT-0009 dense GEMM ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}
