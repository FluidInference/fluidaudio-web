import { ACE_DIT_DENSE_K4_FP16_LAYOUT } from "../../model/manifest.js";
import {
  requireAceBindingBytes,
  requireAceDisjointOutput,
} from "./correctness-utils.js";
import {
  ACE_OPT_0032_DENSE_SUBGROUP_SIZE,
  ACE_OPT_0032_DENSE_TILE_COLUMNS,
  ACE_OPT_0032_DENSE_TILE_INNER,
  ACE_OPT_0032_DENSE_TILE_ROWS,
  ACE_OPT_0032_DENSE_WORKGROUP_SIZE,
  planAceOpt0032DenseK4Partials,
  type AceOpt0032DenseK4PartialsPlan,
} from "./dit-dense-fp16-k4-partials.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmKernel,
  AceGemmShape,
} from "./gemm.js";

export const ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID =
  "ace-opt-0056-dense-k4-exact-increasing-k-fp32-fixed32-wg128-m32-n128-v1";
export const ACE_OPT_0056_DENSE_K4_EXACT_WEIGHT_LAYOUT =
  ACE_DIT_DENSE_K4_FP16_LAYOUT;

const SUBGROUPS_PER_WORKGROUP =
  ACE_OPT_0032_DENSE_WORKGROUP_SIZE / ACE_OPT_0032_DENSE_SUBGROUP_SIZE;
const ROWS_PER_SUBGROUP =
  ACE_OPT_0032_DENSE_TILE_ROWS / SUBGROUPS_PER_WORKGROUP;
const OUTPUTS_PER_LANE =
  ACE_OPT_0032_DENSE_TILE_COLUMNS / ACE_OPT_0032_DENSE_SUBGROUP_SIZE;

export interface AceOpt0056DenseK4ExactDispatch extends AceGemmDispatch {
  readonly kernelId: typeof ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID;
  readonly weightLayout: typeof ACE_OPT_0056_DENSE_K4_EXACT_WEIGHT_LAYOUT;
  readonly plan: AceOpt0032DenseK4PartialsPlan;
}

/**
 * Diagnostic exact-arithmetic owner for the authenticated revision-8 K4
 * package. It changes only physical weight addressing: every FP16 activation
 * and weight product is widened before multiplication and accumulated in
 * strictly increasing logical K order into FP32 running state.
 */
export class AceOpt0056DenseK4ExactKernel implements AceGemmKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0056DenseK4ExactKernel {
    if (
      !device.features.has("shader-f16") ||
      !device.features.has("subgroups") ||
      capability.subgroupMinSize !== ACE_OPT_0032_DENSE_SUBGROUP_SIZE ||
      capability.subgroupMaxSize !== ACE_OPT_0032_DENSE_SUBGROUP_SIZE
    ) {
      throw new Error(
        "OPT-0056 exact K4 dense GEMM requires shader-f16 and fixed 32-lane subgroups",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0032_DENSE_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX <
        ACE_OPT_0032_DENSE_WORKGROUP_SIZE
    ) {
      throw new Error(
        `OPT-0056 exact K4 dense GEMM requires WG${ACE_OPT_0032_DENSE_WORKGROUP_SIZE}`,
      );
    }
    return new AceOpt0056DenseK4ExactKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0056DenseK4ExactDispatch> {
    if (this.destroyed) {
      throw new Error("OPT-0056 exact K4 dense GEMM kernel was destroyed");
    }
    if (bindings.bias !== undefined) {
      throw new Error("OPT-0056 repeated-layer dense GEMMs do not accept bias");
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
        "OPT-0056 exact K4 dense GEMM kernel was destroyed while compiling",
      );
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0056-exact-k4-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: exactBinding(bindings.activation, activationBytes) },
        { binding: 1, resource: exactBinding(bindings.weight, weightBytes) },
        { binding: 2, resource: exactBinding(bindings.output, outputBytes) },
      ],
    });
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID,
      weightLayout: ACE_OPT_0056_DENSE_K4_EXACT_WEIGHT_LAYOUT,
      plan,
      rangeCount: 1 as const,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(`${label} OPT-0056 exact K4 range must be zero`);
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
    const created = compileAceOpt0056DenseK4Exact(this.device, shape);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

export function planAceOpt0056DenseK4Exact(
  shape: AceGemmShape,
): AceOpt0032DenseK4PartialsPlan {
  return planAceOpt0032DenseK4Partials(shape);
}

/** Logical B[K,N] FP16 bits to revision-8 `[N/128,K/4,output4,lane32,K4]`. */
export function packAceOpt0056DenseK4ExactWeightU16(
  logical: Uint16Array,
  inner: number,
  columns: number,
): Uint16Array {
  requireLayoutShape(inner, columns);
  const elements = checkedProduct(inner, columns, "pack elements");
  if (logical.length !== elements) {
    throw new RangeError(
      `OPT-0056 pack expected ${elements} FP16 elements, got ${logical.length}`,
    );
  }
  const packed = new Uint16Array(elements);
  for (let k = 0; k < inner; k += 1) {
    for (let column = 0; column < columns; column += 1) {
      packed[aceOpt0056DenseK4PackedWeightIndex(k, column, inner, columns)] =
        logical[k * columns + column]!;
    }
  }
  return packed;
}

export function unpackAceOpt0056DenseK4ExactWeightU16(
  packed: Uint16Array,
  inner: number,
  columns: number,
): Uint16Array {
  requireLayoutShape(inner, columns);
  const elements = checkedProduct(inner, columns, "unpack elements");
  if (packed.length !== elements) {
    throw new RangeError(
      `OPT-0056 unpack expected ${elements} FP16 elements, got ${packed.length}`,
    );
  }
  const logical = new Uint16Array(elements);
  for (let physical = 0; physical < elements; physical += 1) {
    const { innerIndex, column } =
      aceOpt0056DenseK4LogicalCoordinates(physical, inner, columns);
    logical[innerIndex * columns + column] = packed[physical]!;
  }
  return logical;
}

export function aceOpt0056DenseK4PackedWeightIndex(
  innerIndex: number,
  column: number,
  inner: number,
  columns: number,
): number {
  requireLayoutShape(inner, columns);
  requireCoordinate(innerIndex, inner, "inner index");
  requireCoordinate(column, columns, "column");
  const columnTile = Math.floor(column / ACE_OPT_0032_DENSE_TILE_COLUMNS);
  const columnInTile = column % ACE_OPT_0032_DENSE_TILE_COLUMNS;
  const outputInLane = columnInTile % OUTPUTS_PER_LANE;
  const lane = Math.floor(columnInTile / OUTPUTS_PER_LANE);
  const innerK4 = Math.floor(innerIndex / ACE_OPT_0032_DENSE_TILE_INNER);
  const innerInK4 = innerIndex % ACE_OPT_0032_DENSE_TILE_INNER;
  const innerK4Groups = inner / ACE_OPT_0032_DENSE_TILE_INNER;
  return (((
    (columnTile * innerK4Groups + innerK4) * OUTPUTS_PER_LANE +
    outputInLane
  ) * ACE_OPT_0032_DENSE_SUBGROUP_SIZE + lane) *
    ACE_OPT_0032_DENSE_TILE_INNER) + innerInK4;
}

/** Exact inverse of the revision-8 physical scalar index. */
export function aceOpt0056DenseK4LogicalCoordinates(
  physicalIndex: number,
  inner: number,
  columns: number,
): Readonly<{ innerIndex: number; column: number }> {
  requireLayoutShape(inner, columns);
  const elements = checkedProduct(inner, columns, "inverse elements");
  requireCoordinate(physicalIndex, elements, "physical index");
  let remaining = physicalIndex;
  const innerInK4 = remaining % ACE_OPT_0032_DENSE_TILE_INNER;
  remaining = Math.floor(remaining / ACE_OPT_0032_DENSE_TILE_INNER);
  const lane = remaining % ACE_OPT_0032_DENSE_SUBGROUP_SIZE;
  remaining = Math.floor(remaining / ACE_OPT_0032_DENSE_SUBGROUP_SIZE);
  const outputInLane = remaining % OUTPUTS_PER_LANE;
  remaining = Math.floor(remaining / OUTPUTS_PER_LANE);
  const innerK4Groups = inner / ACE_OPT_0032_DENSE_TILE_INNER;
  const innerK4 = remaining % innerK4Groups;
  const columnTile = Math.floor(remaining / innerK4Groups);
  return Object.freeze({
    innerIndex: innerK4 * ACE_OPT_0032_DENSE_TILE_INNER + innerInK4,
    column:
      columnTile * ACE_OPT_0032_DENSE_TILE_COLUMNS +
      lane * OUTPUTS_PER_LANE + outputInLane,
  });
}

export function aceOpt0056DenseK4ExactWgsl(shape: AceGemmShape): string {
  const plan = planAceOpt0056DenseK4Exact(shape);
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
    acc${row} = acc${row} + vec4<f32>(f32(a${row}.x)) * vec4<f32>(
      f32(b0.x), f32(b1.x), f32(b2.x), f32(b3.x)
    );
    acc${row} = acc${row} + vec4<f32>(f32(a${row}.y)) * vec4<f32>(
      f32(b0.y), f32(b1.y), f32(b2.y), f32(b3.y)
    );
    acc${row} = acc${row} + vec4<f32>(f32(a${row}.z)) * vec4<f32>(
      f32(b0.z), f32(b1.z), f32(b2.z), f32(b3.z)
    );
    acc${row} = acc${row} + vec4<f32>(f32(a${row}.w)) * vec4<f32>(
      f32(b0.w), f32(b1.w), f32(b2.w), f32(b3.w)
    );`,
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
      ${OUTPUTS_PER_LANE}u) * ${ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u +
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

async function compileAceOpt0056DenseK4Exact(
  device: GPUDevice,
  shape: AceGemmShape,
): Promise<GPUComputePipeline> {
  const label =
    `ace-opt-0056-dense-k4-exact-${shape.rows}x${shape.inner}x${shape.columns}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0056DenseK4ExactWgsl(shape),
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

function requireLayoutShape(inner: number, columns: number): void {
  requirePositiveSafeInteger(inner, "layout inner");
  requirePositiveSafeInteger(columns, "layout columns");
  if (
    inner % ACE_OPT_0032_DENSE_TILE_INNER !== 0 ||
    columns % ACE_OPT_0032_DENSE_TILE_COLUMNS !== 0
  ) {
    throw new RangeError("OPT-0056 K4 layout requires K%4=0 and N%128=0");
  }
}

function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`OPT-0056 ${label} must be a positive safe integer`);
  }
}

function requireCoordinate(value: number, extent: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= extent) {
    throw new RangeError(`OPT-0056 ${label} ${value} is out of bounds`);
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(`OPT-0056 ${label} is not a safe integer`);
  }
  return product;
}

function checkedBytes(elements: number, itemBytes: number, label: string): number {
  return checkedProduct(elements, itemBytes, `${label} bytes`);
}
