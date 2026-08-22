import {
  requireAceBindingBytes,
  requireAceDisjointOutput,
} from "./correctness-utils.js";
import {
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
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

export type AceOpt0038DensePartialVariant = "k8" | "k16";

export const ACE_OPT_0038_DENSE_K8_PARTIALS_KERNEL_ID =
  "ace-opt-0038-dense-fp16-k8-partials-fixed32-wg128-m32-n128-v1";
export const ACE_OPT_0038_DENSE_K16_PARTIALS_KERNEL_ID =
  "ace-opt-0038-dense-fp16-k16-partials-fixed32-wg128-m32-n128-v1";
/** OPT-0038 deliberately consumes the exact OPT-0032 K4-native B layout. */
export const ACE_OPT_0038_DENSE_WEIGHT_LAYOUT =
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT;

const SUBGROUPS_PER_WORKGROUP =
  ACE_OPT_0032_DENSE_WORKGROUP_SIZE / ACE_OPT_0032_DENSE_SUBGROUP_SIZE;
const ROWS_PER_SUBGROUP =
  ACE_OPT_0032_DENSE_TILE_ROWS / SUBGROUPS_PER_WORKGROUP;
const COLUMNS_PER_LANE =
  ACE_OPT_0032_DENSE_TILE_COLUMNS / ACE_OPT_0032_DENSE_SUBGROUP_SIZE;

type AceOpt0038DenseKernelId =
  typeof ACE_OPT_0038_DENSE_K8_PARTIALS_KERNEL_ID |
  typeof ACE_OPT_0038_DENSE_K16_PARTIALS_KERNEL_ID;

export interface AceOpt0038DenseBoundedPartialsPlan
  extends AceOpt0032DenseK4PartialsPlan {
  readonly variant: AceOpt0038DensePartialVariant;
  readonly partialInner: 8 | 16;
  readonly k4GroupsPerPartial: 2 | 4;
  readonly innerPartialBlocks: number;
}

export interface AceOpt0038DenseBoundedPartialsDispatch {
  readonly label: string;
  readonly kernelId: AceOpt0038DenseKernelId;
  readonly weightLayout: typeof ACE_OPT_0038_DENSE_WEIGHT_LAYOUT;
  readonly plan: AceOpt0038DenseBoundedPartialsPlan;
  readonly rangeCount: 1;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Benchmark-only OPT-0038 K8/K16 variants. FP16 state exists only inside one
 * bounded partial; every inter-partial running accumulator and output is FP32.
 */
export class AceOpt0038DenseBoundedPartialsKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly variant: AceOpt0038DensePartialVariant,
  ) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
    variant: AceOpt0038DensePartialVariant,
  ): AceOpt0038DenseBoundedPartialsKernel {
    requireVariant(variant);
    if (
      !device.features.has("shader-f16") ||
      !device.features.has("subgroups") ||
      capability.subgroupMinSize !== ACE_OPT_0032_DENSE_SUBGROUP_SIZE ||
      capability.subgroupMaxSize !== ACE_OPT_0032_DENSE_SUBGROUP_SIZE
    ) {
      throw new Error(
        "OPT-0038 dense bounded partials require shader-f16 and fixed 32-lane subgroups",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0032_DENSE_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX <
        ACE_OPT_0032_DENSE_WORKGROUP_SIZE
    ) {
      throw new Error(
        `OPT-0038 dense bounded partials require WG${ACE_OPT_0032_DENSE_WORKGROUP_SIZE}`,
      );
    }
    return new AceOpt0038DenseBoundedPartialsKernel(device, variant);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0038DenseBoundedPartialsDispatch> {
    if (this.destroyed) {
      throw new Error("OPT-0038 dense bounded partials kernel was destroyed");
    }
    if (bindings.bias !== undefined) {
      throw new Error("OPT-0038 repeated-layer dense GEMMs do not accept bias");
    }
    const plan = planAceOpt0038DenseBoundedPartials(shape, this.variant);
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
        "OPT-0038 dense bounded partials kernel was destroyed while compiling",
      );
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0038-${this.variant}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: exactBinding(bindings.activation, activationBytes) },
        { binding: 1, resource: exactBinding(bindings.weight, weightBytes) },
        { binding: 2, resource: exactBinding(bindings.output, outputBytes) },
      ],
    });
    const kernelId = kernelIdFor(this.variant);
    return Object.freeze({
      label,
      kernelId,
      weightLayout: ACE_OPT_0038_DENSE_WEIGHT_LAYOUT,
      plan,
      rangeCount: 1 as const,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(`${label} OPT-0038 dense range must be zero`);
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
    const created = compileAceOpt0038DenseBoundedPartials(
      this.device,
      shape,
      this.variant,
    );
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

export function planAceOpt0038DenseBoundedPartials(
  shape: AceGemmShape,
  variant: AceOpt0038DensePartialVariant,
): AceOpt0038DenseBoundedPartialsPlan {
  requireVariant(variant);
  const base = planAceOpt0032DenseK4Partials(shape);
  const partialInner = variant === "k8" ? 8 : 16;
  const k4GroupsPerPartial = variant === "k8" ? 2 : 4;
  if (shape.inner % partialInner !== 0) {
    throw new RangeError(
      `OPT-0038 ${variant} requires K divisible by ${partialInner}`,
    );
  }
  return Object.freeze({
    ...base,
    variant,
    partialInner,
    k4GroupsPerPartial,
    innerPartialBlocks: shape.inner / partialInner,
  });
}

export function aceOpt0038DenseBoundedPartialsWgsl(
  shape: AceGemmShape,
  variant: AceOpt0038DensePartialVariant,
): string {
  const plan = planAceOpt0038DenseBoundedPartials(shape, variant);
  const accumulators = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `  var acc${row} = vec4<f32>(0.0);`,
  ).join("\n");
  const partials = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `    var partial${row}: vec4<f16>;`,
  ).join("\n");
  const steps = Array.from(
    { length: plan.k4GroupsPerPartial },
    (_, step) => {
      const broadcasts = Array.from(
        { length: ROWS_PER_SUBGROUP },
        (_, row) =>
          `    let a${row}_${step} = subgroupBroadcast(lane_a_${step}, ${row}u);`,
      ).join("\n");
      const contractions = Array.from(
        { length: ROWS_PER_SUBGROUP },
        (_, row) => {
          const update = step === 0
            ? `partial${row} =`
            : `partial${row} = partial${row} +`;
          return /* wgsl */ `
    ${update} vec4<f16>(
      dot(a${row}_${step}, b0_${step}),
      dot(a${row}_${step}, b1_${step}),
      dot(a${row}_${step}, b2_${step}),
      dot(a${row}_${step}, b3_${step})
    );`;
        },
      ).join("\n");
      return /* wgsl */ `
    {
    let inner_k4_${step} =
      inner_partial * ${plan.k4GroupsPerPartial}u + ${step}u;
    let inner_base_${step} =
      inner_k4_${step} * ${ACE_OPT_0032_DENSE_TILE_INNER}u;
    var lane_a_${step} = vec4<f16>(0.0h);
    if (subgroup_lane < ${ROWS_PER_SUBGROUP}u && lane_row < ROWS) {
      let activation_base_${step} = lane_row * INNER + inner_base_${step};
      lane_a_${step} = vec4<f16>(
        f16(activation[activation_base_${step}]),
        f16(activation[activation_base_${step} + 1u]),
        f16(activation[activation_base_${step} + 2u]),
        f16(activation[activation_base_${step} + 3u])
      );
    }
    let weight_base_${step} =
      ((group.x * INNER_K4_GROUPS + inner_k4_${step}) *
      ${COLUMNS_PER_LANE}u) * ${ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u +
      subgroup_lane;
    let b0_${step} = weight[weight_base_${step}];
    let b1_${step} =
      weight[weight_base_${step} + ${ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u];
    let b2_${step} =
      weight[weight_base_${step} + ${2 * ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u];
    let b3_${step} =
      weight[weight_base_${step} + ${3 * ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u];
${broadcasts}
${contractions}
    }`;
    },
  ).join("\n");
  const widen = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `    acc${row} = acc${row} + vec4<f32>(partial${row});`,
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
const INNER_PARTIAL_BLOCKS = ${plan.innerPartialBlocks}u;

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
  let lane_row = row_base + subgroup_lane;
  let column_vector =
    group.x * ${ACE_OPT_0032_DENSE_SUBGROUP_SIZE}u + subgroup_lane;
${accumulators}

  for (
    var inner_partial = 0u;
    inner_partial < INNER_PARTIAL_BLOCKS;
    inner_partial += 1u
  ) {
${partials}
${steps}
${widen}
  }
${stores}
}
`;
}

async function compileAceOpt0038DenseBoundedPartials(
  device: GPUDevice,
  shape: AceGemmShape,
  variant: AceOpt0038DensePartialVariant,
): Promise<GPUComputePipeline> {
  const label =
    `ace-opt-0038-dense-${variant}-${shape.rows}x${shape.inner}x${shape.columns}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0038DenseBoundedPartialsWgsl(shape, variant),
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
  plan: AceOpt0038DenseBoundedPartialsPlan,
): void {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(plan.columnTiles, plan.rowTiles, 1);
}

function kernelIdFor(
  variant: AceOpt0038DensePartialVariant,
): AceOpt0038DenseKernelId {
  return variant === "k8"
    ? ACE_OPT_0038_DENSE_K8_PARTIALS_KERNEL_ID
    : ACE_OPT_0038_DENSE_K16_PARTIALS_KERNEL_ID;
}

function requireVariant(
  variant: AceOpt0038DensePartialVariant,
): void {
  if (variant !== "k8" && variant !== "k16") {
    throw new RangeError(`OPT-0038 unknown dense partial variant ${variant}`);
  }
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

function checkedBytes(elements: number, itemBytes: number, label: string): number {
  const bytes = elements * itemBytes;
  if (!Number.isSafeInteger(bytes)) {
    throw new RangeError(
      `OPT-0038 dense bounded partials ${label} bytes is not a safe integer`,
    );
  }
  return bytes;
}
