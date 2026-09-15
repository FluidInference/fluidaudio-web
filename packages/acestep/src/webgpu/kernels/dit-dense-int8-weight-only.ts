import { ACE_DIT_DENSE_INT8_TILE_LAYOUT } from "../../model/manifest.js";
import {
  checkedAceProduct,
  requireAceBindingBytes,
  requireAceDisjointOutput,
} from "./correctness-utils.js";
import {
  ACE_OPT_0009_DENSE_SUBGROUP_SIZE,
  ACE_OPT_0009_DENSE_TILE_COLUMNS,
  ACE_OPT_0009_DENSE_TILE_INNER,
  ACE_OPT_0009_DENSE_TILE_ROWS,
  ACE_OPT_0009_DENSE_WORKGROUP_SIZE,
  planAceOpt0009DenseGemm,
  type AceOpt0009DenseGemmPlan,
} from "./dit-dense-fp16.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmKernel,
  AceGemmShape,
} from "./gemm.js";

export const ACE_OPT_0091_DENSE_INT8_KERNEL_ID =
  "opt-0091-dense-int8-fp16-scale-fp32-fixed32-v1";
export const ACE_OPT_0091_DENSE_INT8_PORTABLE_KERNEL_ID =
  "opt-0091-dense-int8-fp16-scale-fp32-portable-v1";
export const ACE_OPT_0091_DENSE_INT8_WEIGHT_LAYOUT =
  ACE_DIT_DENSE_INT8_TILE_LAYOUT;
export const ACE_OPT_0091_DENSE_INT8_TILE_QUANTIZED_BYTES =
  ACE_OPT_0009_DENSE_TILE_INNER * ACE_OPT_0009_DENSE_TILE_COLUMNS;
export const ACE_OPT_0091_DENSE_INT8_TILE_SCALE_BYTES =
  ACE_OPT_0009_DENSE_TILE_COLUMNS * 2;
export const ACE_OPT_0091_DENSE_INT8_TILE_BYTES =
  ACE_OPT_0091_DENSE_INT8_TILE_QUANTIZED_BYTES +
  ACE_OPT_0091_DENSE_INT8_TILE_SCALE_BYTES;
export const ACE_OPT_0091_DENSE_INT8_TILE_WORDS =
  ACE_OPT_0091_DENSE_INT8_TILE_BYTES / 4;

const LANES = ACE_OPT_0009_DENSE_SUBGROUP_SIZE;
const SLICES = ACE_OPT_0009_DENSE_WORKGROUP_SIZE / LANES;
const ROWS_PER_SLICE = ACE_OPT_0009_DENSE_TILE_ROWS / SLICES;
const OUTPUTS_PER_LANE = ACE_OPT_0009_DENSE_TILE_COLUMNS / LANES;
const QUANTIZED_WORDS_PER_INNER = ACE_OPT_0009_DENSE_TILE_COLUMNS / 4;
const QUANTIZED_WORDS_PER_TILE =
  ACE_OPT_0091_DENSE_INT8_TILE_QUANTIZED_BYTES / 4;

export interface AceOpt0091DenseInt8Plan extends Omit<
  AceOpt0009DenseGemmPlan,
  "packedWeightStorageShape"
> {
  readonly packedWeightBytes: number;
  readonly packedWeightStorageShape: readonly [number, number, number];
}

export interface AceOpt0091DenseInt8Dispatch extends AceGemmDispatch {
  readonly kernelId:
    | typeof ACE_OPT_0091_DENSE_INT8_KERNEL_ID
    | typeof ACE_OPT_0091_DENSE_INT8_PORTABLE_KERNEL_ID;
  readonly weightLayout: typeof ACE_OPT_0091_DENSE_INT8_WEIGHT_LAYOUT;
  readonly plan: AceOpt0091DenseInt8Plan;
}

export class AceOpt0091DenseInt8Kernel implements AceGemmKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly portable: boolean,
  ) {}

  static create(
    device: GPUDevice,
    configuration: Readonly<{
      portable: boolean;
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0091DenseInt8Kernel {
    if (!device.features.has("shader-f16")) {
      throw new Error("OPT-0091 dense INT8 requires WebGPU shader-f16");
    }
    if (
      !configuration.portable &&
      (!device.features.has("subgroups") ||
        configuration.subgroupMinSize !== LANES ||
        configuration.subgroupMaxSize !== LANES)
    ) {
      throw new Error(
        "OPT-0091 subgroup dense INT8 requires fixed 32-lane subgroups",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0009_DENSE_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX < ACE_OPT_0009_DENSE_WORKGROUP_SIZE
    ) {
      throw new Error(
        `OPT-0091 dense INT8 requires WG${ACE_OPT_0009_DENSE_WORKGROUP_SIZE}`,
      );
    }
    return new AceOpt0091DenseInt8Kernel(device, configuration.portable);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0091DenseInt8Dispatch> {
    if (this.destroyed)
      throw new Error("OPT-0091 dense INT8 kernel was destroyed");
    if (bindings.bias !== undefined) {
      throw new Error("OPT-0091 repeated-layer dense GEMMs do not accept bias");
    }
    const base = planAceOpt0009DenseGemm(shape);
    const packedWeightBytes = checkedAceProduct(
      [base.columnTiles, base.innerTiles, ACE_OPT_0091_DENSE_INT8_TILE_BYTES],
      `${label} packed weight bytes`,
    );
    const plan: AceOpt0091DenseInt8Plan = Object.freeze({
      ...base,
      packedWeightBytes,
      packedWeightStorageShape: Object.freeze([
        base.columnTiles,
        base.innerTiles,
        ACE_OPT_0091_DENSE_INT8_TILE_WORDS,
      ]) as readonly [number, number, number],
    });
    const activationBytes = checkedAceProduct(
      [plan.activationElements, 4],
      `${label} activation bytes`,
    );
    const outputBytes = checkedAceProduct(
      [plan.outputElements, 4],
      `${label} output bytes`,
    );
    requireAceBindingBytes(
      bindings.activation,
      activationBytes,
      `${label} activation`,
    );
    requireAceBindingBytes(
      bindings.weight,
      packedWeightBytes,
      `${label} weight`,
    );
    requireAceBindingBytes(bindings.output, outputBytes, `${label} output`);
    requireAceDisjointOutput(
      exactBinding(bindings.output, outputBytes),
      [
        exactBinding(bindings.activation, activationBytes),
        exactBinding(bindings.weight, packedWeightBytes),
      ],
      label,
    );
    const pipeline = await this.pipelineFor(shape);
    if (this.destroyed) {
      throw new Error(
        "OPT-0091 dense INT8 kernel was destroyed while compiling",
      );
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0091-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: exactBinding(bindings.activation, activationBytes),
        },
        {
          binding: 1,
          resource: exactBinding(bindings.weight, packedWeightBytes),
        },
        { binding: 2, resource: exactBinding(bindings.output, outputBytes) },
      ],
    });
    const kernelId = this.portable
      ? ACE_OPT_0091_DENSE_INT8_PORTABLE_KERNEL_ID
      : ACE_OPT_0091_DENSE_INT8_KERNEL_ID;
    return Object.freeze({
      label,
      kernelId,
      weightLayout: ACE_OPT_0091_DENSE_INT8_WEIGHT_LAYOUT,
      plan,
      rangeCount: 1,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0)
          throw new RangeError(`${label} OPT-0091 range must be zero`);
        encode(pass, pipeline, bindGroup, plan);
      },
      encode(pass: GPUComputePassEncoder): void {
        encode(pass, pipeline, bindGroup, plan);
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
    const created = compile(this.device, shape, this.portable);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

export function aceOpt0091DenseInt8Wgsl(
  shape: AceGemmShape,
  portable: boolean,
): string {
  const plan = planAceOpt0009DenseGemm(shape);
  const declarations = Array.from(
    { length: ROWS_PER_SLICE },
    (_, row) =>
      `  var acc${row}_0 = vec4<f32>(0.0);\n  var acc${row}_1 = vec4<f32>(0.0);`,
  ).join("\n");
  const reads = Array.from({ length: ROWS_PER_SLICE }, (_, row) =>
    portable
      ? `      let a${row} = staged_a[staged_base + ${row}u];`
      : `      let a${row} = subgroupBroadcast(lane_a, ${row}u);`,
  ).join("\n");
  const contractions = Array.from(
    { length: ROWS_PER_SLICE },
    (_, row) =>
      `      acc${row}_0 = acc${row}_0 + vec4<f32>(f32(a${row})) * vec4<f32>(b0);\n` +
      `      acc${row}_1 = acc${row}_1 + vec4<f32>(f32(a${row})) * vec4<f32>(b1);`,
  ).join("\n");
  const stores = Array.from(
    { length: ROWS_PER_SLICE },
    (_, row) => `
  {
    let row = row_base + ${row}u;
    if (row < ROWS) {
      let vector_base = row * (COLUMNS / 4u) + column_base / 4u;
      output[vector_base] = acc${row}_0;
      output[vector_base + 1u] = acc${row}_1;
    }
  }`,
  ).join("\n");
  const feature = portable ? "" : "enable subgroups;\n";
  const builtins = portable
    ? "@builtin(local_invocation_index) local_index: u32,"
    : "@builtin(subgroup_invocation_id) lane: u32,\n  @builtin(subgroup_id) slice: u32,\n  @builtin(subgroup_size) subgroup_size: u32,";
  const laneSetup = portable
    ? `  let slice = local_index / ${LANES}u;\n  let lane = local_index % ${LANES}u;`
    : `  if (subgroup_size != ${LANES}u || slice >= ${SLICES}u) { return; }`;
  const stagingDeclaration = portable
    ? `var<workgroup> staged_a: array<f16, ${SLICES * ROWS_PER_SLICE}>;`
    : "";
  const stageWrite = portable
    ? `      if (lane < ${ROWS_PER_SLICE}u) { staged_a[staged_base + lane] = lane_a; }\n      workgroupBarrier();`
    : "";
  const stageEnd = portable ? "      workgroupBarrier();" : "";
  return `
enable f16;
${feature}
const ROWS = ${plan.rows}u;
const INNER = ${plan.inner}u;
const COLUMNS = ${plan.columns}u;
const INNER_TILES = ${plan.innerTiles}u;
@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
${stagingDeclaration}
fn unpack_i8x4(word: u32) -> vec4<f16> {
  let raw = vec4<u32>(word & 255u, (word >> 8u) & 255u, (word >> 16u) & 255u, word >> 24u);
  let positive = vec4<i32>(raw);
  let signed = select(positive, positive - vec4<i32>(256), raw >= vec4<u32>(128));
  return vec4<f16>(signed);
}
@compute @workgroup_size(${ACE_OPT_0009_DENSE_WORKGROUP_SIZE}, 1, 1)
fn main(
  ${builtins}
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (group.x >= ${plan.columnTiles}u || group.y >= ${plan.rowTiles}u || group.z != 0u) { return; }
${laneSetup}
  let staged_base = slice * ${ROWS_PER_SLICE}u;
  let row_base = group.y * ${ACE_OPT_0009_DENSE_TILE_ROWS}u + slice * ${ROWS_PER_SLICE}u;
  let column_base = group.x * ${ACE_OPT_0009_DENSE_TILE_COLUMNS}u + lane * ${OUTPUTS_PER_LANE}u;
${declarations}
  for (var inner_tile = 0u; inner_tile < INNER_TILES; inner_tile += 1u) {
    let tile_base = (group.x * INNER_TILES + inner_tile) * ${ACE_OPT_0091_DENSE_INT8_TILE_WORDS}u;
    let scale_base = tile_base + ${QUANTIZED_WORDS_PER_TILE}u + lane * 4u;
    let s01 = unpack2x16float(weight[scale_base]);
    let s23 = unpack2x16float(weight[scale_base + 1u]);
    let s45 = unpack2x16float(weight[scale_base + 2u]);
    let s67 = unpack2x16float(weight[scale_base + 3u]);
    let scale0 = vec4<f16>(f16(s01.x), f16(s01.y), f16(s23.x), f16(s23.y));
    let scale1 = vec4<f16>(f16(s45.x), f16(s45.y), f16(s67.x), f16(s67.y));
    for (var inner_in_tile = 0u; inner_in_tile < ${ACE_OPT_0009_DENSE_TILE_INNER}u; inner_in_tile += 1u) {
      let inner = inner_tile * ${ACE_OPT_0009_DENSE_TILE_INNER}u + inner_in_tile;
      var lane_a = 0.0h;
      let lane_row = row_base + lane;
      if (lane < ${ROWS_PER_SLICE}u && lane_row < ROWS) { lane_a = f16(activation[lane_row * INNER + inner]); }
${stageWrite}
      let q_base = tile_base + inner_in_tile * ${QUANTIZED_WORDS_PER_INNER}u + lane * 2u;
      let b0 = unpack_i8x4(weight[q_base]) * scale0;
      let b1 = unpack_i8x4(weight[q_base + 1u]) * scale1;
${reads}
${contractions}
${stageEnd}
    }
  }
${stores}
}
`;
}

async function compile(
  device: GPUDevice,
  shape: AceGemmShape,
  portable: boolean,
): Promise<GPUComputePipeline> {
  const label = `ace-opt-0091-${portable ? "portable" : "subgroup"}-${shape.rows}x${shape.inner}x${shape.columns}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0091DenseInt8Wgsl(shape, portable),
  });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length !== 0) {
    throw new Error(
      `${label} WGSL compilation failed:\n${errors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`).join("\n")}`,
    );
  }
  return await device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

function encode(
  pass: GPUComputePassEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  plan: AceOpt0091DenseInt8Plan,
): void {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(plan.columnTiles, plan.rowTiles, 1);
}

function exactBinding(
  binding: GPUBufferBinding,
  size: number,
): GPUBufferBinding {
  return { buffer: binding.buffer, offset: binding.offset ?? 0, size };
}
