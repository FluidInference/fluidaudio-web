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
  AceGemmShape,
} from "./gemm.js";

export const ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_KERNEL_ID =
  "ace-opt-0050-dense-fp16-k4-output-vector-fma-fixed32-wg128-m32-n128-v1" as const;
export const ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_WEIGHT_LAYOUT =
  "ace-opt-0050-b-n128-k4-k4-lane32-output4-v1" as const;
export const ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_REDUCTION =
  "increasing-k4-four-output-vector-f16-fma-then-fp32-running-state" as const;

const SUBGROUPS_PER_WORKGROUP = ACE_OPT_0032_DENSE_WORKGROUP_SIZE /
  ACE_OPT_0032_DENSE_SUBGROUP_SIZE;
const ROWS_PER_SUBGROUP = ACE_OPT_0032_DENSE_TILE_ROWS /
  SUBGROUPS_PER_WORKGROUP;
const OUTPUTS_PER_LANE = ACE_OPT_0032_DENSE_TILE_COLUMNS /
  ACE_OPT_0032_DENSE_SUBGROUP_SIZE;

export interface AceOpt0050DenseK4OutputVectorFmaPlan
  extends Omit<AceOpt0032DenseK4PartialsPlan, "packedWeightStorageShape"> {
  readonly kernelId:
    typeof ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_KERNEL_ID;
  readonly weightLayout:
    typeof ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_WEIGHT_LAYOUT;
  readonly reductionSemantics:
    typeof ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_REDUCTION;
  /** [N/128, K/4, K4, lane32, four contiguous outputs]. */
  readonly packedWeightStorageShape: readonly [
    number,
    number,
    4,
    32,
    4,
  ];
}

export interface AceOpt0050DenseK4OutputVectorFmaDispatch {
  readonly label: string;
  readonly kernelId:
    typeof ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_KERNEL_ID;
  readonly weightLayout:
    typeof ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_WEIGHT_LAYOUT;
  readonly plan: AceOpt0050DenseK4OutputVectorFmaPlan;
  readonly rangeCount: 1;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Benchmark-only OPT-0050 owner. It changes OPT-0032 only along the local K4
 * vector direction: four FP16 output-vector FMAs form one bounded partial,
 * which is widened once into the unchanged FP32 running accumulator.
 */
export class AceOpt0050DenseK4OutputVectorFmaKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0050DenseK4OutputVectorFmaKernel {
    if (
      !device.features.has("shader-f16") ||
      !device.features.has("subgroups") ||
      capability.subgroupMinSize !== ACE_OPT_0032_DENSE_SUBGROUP_SIZE ||
      capability.subgroupMaxSize !== ACE_OPT_0032_DENSE_SUBGROUP_SIZE
    ) {
      throw new Error(
        "OPT-0050 dense K4 output-vector FMA requires shader-f16 and fixed 32-lane subgroups",
      );
    }
    if (
      Number(device.limits.maxComputeInvocationsPerWorkgroup) <
        ACE_OPT_0032_DENSE_WORKGROUP_SIZE ||
      Number(device.limits.maxComputeWorkgroupSizeX) <
        ACE_OPT_0032_DENSE_WORKGROUP_SIZE
    ) {
      throw new Error(
        `OPT-0050 dense K4 output-vector FMA requires WG${ACE_OPT_0032_DENSE_WORKGROUP_SIZE}`,
      );
    }
    return new AceOpt0050DenseK4OutputVectorFmaKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0050DenseK4OutputVectorFmaDispatch> {
    this.requireLive();
    if (bindings.bias !== undefined) {
      throw new Error("OPT-0050 repeated-layer dense GEMMs do not accept bias");
    }
    const plan = planAceOpt0050DenseK4OutputVectorFma(shape);
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
    this.requireLive();
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0050-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: exactBinding(bindings.activation, activationBytes) },
        { binding: 1, resource: exactBinding(bindings.weight, weightBytes) },
        { binding: 2, resource: exactBinding(bindings.output, outputBytes) },
      ],
    });
    const owner = this;
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_KERNEL_ID,
      weightLayout: ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_WEIGHT_LAYOUT,
      plan,
      rangeCount: 1 as const,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        owner.requireLive();
        if (rangeIndex !== 0) {
          throw new RangeError(`${label} OPT-0050 dense range must be zero`);
        }
        encodeDispatch(pass, pipeline, bindGroup, plan);
      },
      encode(pass: GPUComputePassEncoder): void {
        owner.requireLive();
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
    const created = compileKernel(this.device, shape);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0050 dense K4 output-vector FMA was destroyed");
    }
  }
}

export function planAceOpt0050DenseK4OutputVectorFma(
  shape: AceGemmShape,
): AceOpt0050DenseK4OutputVectorFmaPlan {
  const base = planAceOpt0032DenseK4Partials(shape);
  return Object.freeze({
    ...base,
    kernelId: ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_KERNEL_ID,
    weightLayout: ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_WEIGHT_LAYOUT,
    reductionSemantics: ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_REDUCTION,
    packedWeightStorageShape: Object.freeze([
      base.columnTiles,
      base.innerK4Groups,
      ACE_OPT_0032_DENSE_TILE_INNER,
      ACE_OPT_0032_DENSE_SUBGROUP_SIZE,
      OUTPUTS_PER_LANE,
    ]) as readonly [number, number, 4, 32, 4],
  });
}

/** Pack logical row-major B[K,N] into `[N/128,K/4,K4,lane32,output4]`. */
export function packAceOpt0050DenseWeightU16(
  logical: Uint16Array,
  inner: number,
  columns: number,
): Uint16Array {
  const dimensions = requirePackDimensions(logical.length, inner, columns);
  const packed = new Uint16Array(logical.length);
  let physical = 0;
  for (let columnTile = 0; columnTile < dimensions.columnTiles; columnTile += 1) {
    for (let innerK4 = 0; innerK4 < dimensions.innerK4Groups; innerK4 += 1) {
      for (let innerInK4 = 0; innerInK4 < 4; innerInK4 += 1) {
        const innerIndex = innerK4 * 4 + innerInK4;
        for (let lane = 0; lane < 32; lane += 1) {
          const columnBase = columnTile * 128 + lane * 4;
          for (let output = 0; output < 4; output += 1) {
            packed[physical] = logical[
              innerIndex * columns + columnBase + output
            ]!;
            physical += 1;
          }
        }
      }
    }
  }
  requireCompleteTraversal(physical, logical.length);
  return packed;
}

export function unpackAceOpt0050DenseWeightU16(
  packed: Uint16Array,
  inner: number,
  columns: number,
): Uint16Array {
  const dimensions = requirePackDimensions(packed.length, inner, columns);
  const logical = new Uint16Array(packed.length);
  let physical = 0;
  for (let columnTile = 0; columnTile < dimensions.columnTiles; columnTile += 1) {
    for (let innerK4 = 0; innerK4 < dimensions.innerK4Groups; innerK4 += 1) {
      for (let innerInK4 = 0; innerInK4 < 4; innerInK4 += 1) {
        const innerIndex = innerK4 * 4 + innerInK4;
        for (let lane = 0; lane < 32; lane += 1) {
          const columnBase = columnTile * 128 + lane * 4;
          for (let output = 0; output < 4; output += 1) {
            logical[innerIndex * columns + columnBase + output] =
              packed[physical]!;
            physical += 1;
          }
        }
      }
    }
  }
  requireCompleteTraversal(physical, packed.length);
  return logical;
}

export function aceOpt0050PackedWeightIndex(
  innerIndex: number,
  column: number,
  inner: number,
  columns: number,
): number {
  const dimensions = requirePackDimensions(inner * columns, inner, columns);
  requireCoordinate(innerIndex, inner, "inner index");
  requireCoordinate(column, columns, "column");
  const columnTile = Math.floor(column / 128);
  const columnInTile = column % 128;
  const lane = Math.floor(columnInTile / 4);
  const output = columnInTile % 4;
  const innerK4 = Math.floor(innerIndex / 4);
  const innerInK4 = innerIndex % 4;
  return (((columnTile * dimensions.innerK4Groups + innerK4) * 4 +
    innerInK4) * 32 + lane) * 4 + output;
}

export function aceOpt0050DenseK4OutputVectorFmaWgsl(
  shape: AceGemmShape,
): string {
  const plan = planAceOpt0050DenseK4OutputVectorFma(shape);
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
    var partial${row} = vec4<f16>(0.0h);
    partial${row} = fma(vec4<f16>(a${row}.x), b0, partial${row});
    partial${row} = fma(vec4<f16>(a${row}.y), b1, partial${row});
    partial${row} = fma(vec4<f16>(a${row}.z), b2, partial${row});
    partial${row} = fma(vec4<f16>(a${row}.w), b3, partial${row});
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
// kernel-id: ${ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_KERNEL_ID}
// weight-layout: ${ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_WEIGHT_LAYOUT}
// reduction-semantics: ${ACE_OPT_0050_DENSE_K4_OUTPUT_VECTOR_FMA_REDUCTION}
enable f16;
enable subgroups;

const ROWS: u32 = ${plan.rows}u;
const INNER: u32 = ${plan.inner}u;
const COLUMNS: u32 = ${plan.columns}u;
const INNER_K4_GROUPS: u32 = ${plan.innerK4Groups}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(${ACE_OPT_0032_DENSE_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(subgroup_invocation_id) lane: u32,
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
  ) { return; }
  let row_base = group.y * ${ACE_OPT_0032_DENSE_TILE_ROWS}u +
    subgroup * ${ROWS_PER_SUBGROUP}u;
  let column_vector = group.x * 32u + lane;
${declarations}

  for (var inner_k4 = 0u; inner_k4 < INNER_K4_GROUPS; inner_k4 += 1u) {
    let inner_base = inner_k4 * 4u;
    var lane_a = vec4<f16>(0.0h);
    let lane_row = row_base + lane;
    if (lane < ${ROWS_PER_SUBGROUP}u && lane_row < ROWS) {
      let activation_base = lane_row * INNER + inner_base;
      lane_a = vec4<f16>(
        f16(activation[activation_base]),
        f16(activation[activation_base + 1u]),
        f16(activation[activation_base + 2u]),
        f16(activation[activation_base + 3u])
      );
    }
    let weight_base =
      ((group.x * INNER_K4_GROUPS + inner_k4) * 4u) * 32u + lane;
    let b0 = weight[weight_base];
    let b1 = weight[weight_base + 32u];
    let b2 = weight[weight_base + 64u];
    let b3 = weight[weight_base + 96u];
${broadcasts}
${contractions}
  }
${stores}
}
`;
}

async function compileKernel(
  device: GPUDevice,
  shape: AceGemmShape,
): Promise<GPUComputePipeline> {
  const label = `ace-opt-0050-dense-k4-output-fma-${shape.rows}x${shape.inner}x${shape.columns}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0050DenseK4OutputVectorFmaWgsl(shape),
  });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(({ type }) => type === "error");
  if (errors.length !== 0) {
    throw new Error(`${label} WGSL compilation failed:\n` + errors.map(
      ({ lineNum, linePos, message }) => `${lineNum}:${linePos} ${message}`,
    ).join("\n"));
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
  plan: AceOpt0050DenseK4OutputVectorFmaPlan,
): void {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(plan.columnTiles, plan.rowTiles, 1);
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

function requirePackDimensions(
  length: number,
  inner: number,
  columns: number,
): Readonly<{ innerK4Groups: number; columnTiles: number }> {
  requirePositiveSafeInteger(inner, "inner");
  requirePositiveSafeInteger(columns, "columns");
  if (inner % 4 !== 0 || columns % 128 !== 0) {
    throw new RangeError("OPT-0050 pack requires K%4=0 and N%128=0");
  }
  const elements = checkedProduct(inner, columns, "pack elements");
  if (length !== elements) {
    throw new RangeError(
      `OPT-0050 pack expected ${elements} FP16 words, got ${length}`,
    );
  }
  return Object.freeze({
    innerK4Groups: inner / 4,
    columnTiles: columns / 128,
  });
}

function requireCompleteTraversal(actual: number, expected: number): void {
  if (actual !== expected) {
    throw new Error(`OPT-0050 packed traversal ${actual} != ${expected}`);
  }
}

function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`OPT-0050 ${label} must be a positive safe integer`);
  }
}

function requireCoordinate(value: number, extent: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= extent) {
    throw new RangeError(`OPT-0050 ${label} ${value} is out of bounds`);
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(`OPT-0050 ${label} is not a safe integer`);
  }
  return product;
}

function checkedBytes(elements: number, itemBytes: number, label: string): number {
  return checkedProduct(elements, itemBytes, `${label} bytes`);
}
