import type { AceModelProfileId } from "../capabilities.js";
import {
  requireAceDisjointOutput,
} from "./correctness-utils.js";
import {
  ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE,
  ACE_GEMM_MAX_OUTPUTS_PER_RANGE,
  planAceGemm,
  type AceCooperativeGemmPlan,
  type AceGemmBufferBindings,
  type AceGemmDispatch,
  type AceGemmKernel,
  type AceGemmShape,
  type AceGemmWeightLayout,
} from "./gemm.js";

export const ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID =
  "opt-0083-planner-m1-m2-n128-k64-wg128-source-row-major-bf16-fp32-v1" as const;
export const ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID =
  "opt-0083-planner-direct-m1-m2-n128-wg128-source-row-major-bf16-fp32-v1" as const;
export const ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_MAX_ROWS = 2;
export const ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS = 128;
export const ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER = 64;
export const ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE = 128;
export const ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_STRIDE = 129;
export const ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_WORDS =
  32 * ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_STRIDE;
export const ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_BYTES =
  ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_WORDS *
  Uint32Array.BYTES_PER_ELEMENT;
export const ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_MAX_WORKGROUP_BYTES =
  ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_BYTES +
  ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_MAX_ROWS *
    ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER *
    Float32Array.BYTES_PER_ELEMENT;

const MAX_DISPATCH_DIMENSION = 65_535;
const GPU_BUFFER_ALIGNMENT = 4;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const PACKED_BF16_PAIR_BYTES = Uint32Array.BYTES_PER_ELEMENT;

export interface AceOpt0083PlannerDirectLowRowBf16GemvPlan
  extends AceCooperativeGemmPlan {
  readonly kernelId:
    typeof ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID;
  readonly tileRows:
    typeof ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_MAX_ROWS;
  readonly tileColumns:
    typeof ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS;
  /** One packed pair, consumed low half before high half. */
  readonly tileInner: 2;
  readonly workgroupSize:
    typeof ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE;
  readonly columnTiles: number;
  readonly innerPairs: number;
  readonly workgroupCount: number;
  readonly scheduledColumns: number;
  readonly scheduledMultiplyAdds: number;
  readonly validMultiplyAdds: number;
  readonly packedWeightStorageShape: readonly [number, number];
  readonly activationBytes: number;
  readonly weightBytes: number;
  readonly outputBytes: number;
  readonly workgroupStorageBytes: 0;
  readonly barriersPerWorkgroup: 0;
  readonly estimatedGlobalActivationBytes: number;
  readonly estimatedGlobalWeightBytes: number;
  readonly estimatedGlobalOperandBytes: number;
}

export interface AceOpt0083PlannerDirectLowRowBf16GemvDispatch
  extends AceGemmDispatch {
  readonly kernelId:
    typeof ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID;
  readonly weightLayout: "source-row-major";
  readonly plan: AceOpt0083PlannerDirectLowRowBf16GemvPlan;
}

export interface AceOpt0083PlannerLowRowBf16GemvPlan
  extends AceCooperativeGemmPlan {
  readonly kernelId:
    typeof ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID;
  readonly tileRows:
    typeof ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_MAX_ROWS;
  readonly tileColumns:
    typeof ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS;
  readonly tileInner:
    typeof ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER;
  readonly workgroupSize:
    typeof ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE;
  readonly columnTiles: number;
  readonly innerTiles: number;
  readonly workgroupCount: number;
  readonly scheduledColumns: number;
  readonly scheduledInner: number;
  readonly scheduledMultiplyAdds: number;
  readonly validMultiplyAdds: number;
  readonly packedWeightStorageShape: readonly [number, number];
  readonly activationBytes: number;
  readonly weightBytes: number;
  readonly outputBytes: number;
  readonly workgroupStorageBytes: number;
  readonly barriersPerWorkgroup: number;
  readonly estimatedGlobalActivationBytes: number;
  readonly estimatedGlobalWeightBytes: number;
  readonly estimatedGlobalOperandBytes: number;
}

export interface AceOpt0083PlannerLowRowBf16GemvDispatch
  extends AceGemmDispatch {
  readonly kernelId:
    typeof ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID;
  readonly weightLayout: "source-row-major";
  readonly plan: AceOpt0083PlannerLowRowBf16GemvPlan;
}

interface CompiledPlannerLowRowBf16Gemv {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/** OPT-0083 arm B: direct, sequential packed-BF16 M1/M2 GEMV. */
export class AceOpt0083PlannerDirectLowRowBf16GemvKernel
  implements AceGemmKernel {
  readonly modelProfile = "reference-bf16" as const;
  readonly weightLayout = "source-row-major" as const;

  private readonly pipelines = new Map<
    string,
    Promise<CompiledPlannerLowRowBf16Gemv>
  >();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
    weightLayout: AceGemmWeightLayout = "source-row-major",
  ): AceOpt0083PlannerDirectLowRowBf16GemvKernel {
    requirePlannerLowRowProfileAndLayout(modelProfile, weightLayout, "direct");
    requireDirectKernelDevice(device);
    return new AceOpt0083PlannerDirectLowRowBf16GemvKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0083PlannerDirectLowRowBf16GemvDispatch> {
    this.requireLive();
    if (bindings.bias !== undefined) {
      throw new RangeError(
        "OPT-0083 planner direct low-row GEMV rejects bias",
      );
    }
    const plan = planAceOpt0083PlannerDirectLowRowBf16Gemv(shape);
    requireDispatchDimensions(this.device, plan.workgroupsX);
    requireBufferLimits(this.device, [
      ["activation", plan.activationBytes],
      ["weight", plan.weightBytes],
      ["output", plan.outputBytes],
    ]);
    const activation = requireStorageBinding(
      this.device,
      bindings.activation,
      plan.activationBytes,
      `${label} activation`,
    );
    const weight = requireStorageBinding(
      this.device,
      bindings.weight,
      plan.weightBytes,
      `${label} weight`,
    );
    const output = requireStorageBinding(
      this.device,
      bindings.output,
      plan.outputBytes,
      `${label} output`,
    );
    requireAceDisjointOutput(output, [activation, weight], label);

    const compiled = await this.pipelineFor(plan);
    this.requireLive();
    const bindGroup = createPlannerLowRowBindGroup(
      this.device,
      `${label}-opt-0083-planner-direct-low-row-bf16-gemv-bindings`,
      compiled.bindGroupLayout,
      activation,
      weight,
      output,
    );
    const owner = this;
    const encode = (pass: GPUComputePassEncoder): void => {
      owner.requireLive();
      pass.setPipeline(compiled.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(plan.workgroupsX, 1, 1);
    };
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID,
      weightLayout: "source-row-major" as const,
      plan,
      rangeCount: 1,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(
            `${label} OPT-0083 planner direct low-row GEMV range must be zero`,
          );
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
  }

  private pipelineFor(
    plan: AceOpt0083PlannerDirectLowRowBf16GemvPlan,
  ): Promise<CompiledPlannerLowRowBf16Gemv> {
    const key = shapeKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compilePlannerDirectLowRowBf16Gemv(this.device, plan);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error(
        "OPT-0083 planner direct low-row GEMV kernel was destroyed",
      );
    }
  }
}

/**
 * OPT-0083 arm C: a benchmark-only M1/M2 packed-BF16 planner GEMV.
 *
 * A lane owns one N column for all active rows. The workgroup transposes one
 * source-row-major N128/K64 weight panel into bank-padded shared storage, then
 * every owner contracts K strictly low-half then high-half in source order.
 * The class is intentionally not selected by any production graph yet.
 */
export class AceOpt0083PlannerLowRowBf16GemvKernel
  implements AceGemmKernel {
  readonly modelProfile = "reference-bf16" as const;
  readonly weightLayout = "source-row-major" as const;

  private readonly pipelines = new Map<
    string,
    Promise<CompiledPlannerLowRowBf16Gemv>
  >();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
    weightLayout: AceGemmWeightLayout = "source-row-major",
  ): AceOpt0083PlannerLowRowBf16GemvKernel {
    if (modelProfile !== "reference-bf16") {
      throw new Error(
        "OPT-0083 planner low-row GEMV requires reference-bf16",
      );
    }
    if (weightLayout !== "source-row-major") {
      throw new Error(
        "OPT-0083 planner low-row GEMV requires source-row-major weights",
      );
    }
    requireKernelDevice(device);
    return new AceOpt0083PlannerLowRowBf16GemvKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0083PlannerLowRowBf16GemvDispatch> {
    this.requireLive();
    if (bindings.bias !== undefined) {
      throw new RangeError("OPT-0083 planner low-row GEMV rejects bias");
    }
    const plan = planAceOpt0083PlannerLowRowBf16Gemv(shape);
    requireDispatchDimensions(this.device, plan.workgroupsX);
    requireBufferLimits(this.device, [
      ["activation", plan.activationBytes],
      ["weight", plan.weightBytes],
      ["output", plan.outputBytes],
    ]);
    const activation = requireStorageBinding(
      this.device,
      bindings.activation,
      plan.activationBytes,
      `${label} activation`,
    );
    const weight = requireStorageBinding(
      this.device,
      bindings.weight,
      plan.weightBytes,
      `${label} weight`,
    );
    const output = requireStorageBinding(
      this.device,
      bindings.output,
      plan.outputBytes,
      `${label} output`,
    );
    requireAceDisjointOutput(output, [activation, weight], label);

    const compiled = await this.pipelineFor(plan);
    this.requireLive();
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0083-planner-low-row-bf16-gemv-bindings`,
      layout: compiled.bindGroupLayout,
      entries: [activation, weight, output].map((resource, binding) => ({
        binding,
        resource,
      })),
    });
    const owner = this;
    const encode = (pass: GPUComputePassEncoder): void => {
      owner.requireLive();
      pass.setPipeline(compiled.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(plan.workgroupsX, 1, 1);
    };
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID,
      weightLayout: "source-row-major" as const,
      plan,
      rangeCount: 1,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(
            `${label} OPT-0083 planner low-row GEMV range must be zero`,
          );
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
  }

  private pipelineFor(
    plan: AceOpt0083PlannerLowRowBf16GemvPlan,
  ): Promise<CompiledPlannerLowRowBf16Gemv> {
    const key = shapeKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compilePlannerLowRowBf16Gemv(this.device, plan);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0083 planner low-row GEMV kernel was destroyed");
    }
  }
}

export function planAceOpt0083PlannerDirectLowRowBf16Gemv(
  shape: AceGemmShape,
): AceOpt0083PlannerDirectLowRowBf16GemvPlan {
  const scalar = planAceGemm(shape);
  requirePlannerLowRowShape(scalar);
  const columnTiles = Math.ceil(
    scalar.columns / ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS,
  );
  if (columnTiles > MAX_DISPATCH_DIMENSION) {
    throw new RangeError(
      "OPT-0083 planner direct low-row GEMV exceeds the portable dispatch domain",
    );
  }
  const scheduledColumns = checkedProduct(
    columnTiles,
    ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS,
    "direct scheduled columns",
  );
  const scheduledMultiplyAdds = checkedProduct(
    checkedProduct(
      scalar.rows,
      scheduledColumns,
      "direct scheduled outputs",
    ),
    scalar.inner,
    "direct scheduled multiply-adds",
  );
  const validMultiplyAdds = checkedProduct(
    scalar.outputElements,
    scalar.inner,
    "direct valid multiply-adds",
  );
  if (
    scalar.outputElements > ACE_GEMM_MAX_OUTPUTS_PER_RANGE ||
    scheduledMultiplyAdds > ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE
  ) {
    throw new RangeError(
      "OPT-0083 planner direct low-row GEMV exceeds one bounded output range",
    );
  }
  const activationBytes = checkedProduct(
    scalar.activationElements,
    FLOAT32_BYTES,
    "direct activation bytes",
  );
  const weightBytes = checkedProduct(
    scalar.weightElements / 2,
    PACKED_BF16_PAIR_BYTES,
    "direct packed BF16 weight bytes",
  );
  const outputBytes = checkedProduct(
    scalar.outputElements,
    FLOAT32_BYTES,
    "direct output bytes",
  );
  const estimatedGlobalActivationBytes = checkedProduct(
    activationBytes,
    scalar.columns,
    "direct estimated global activation bytes",
  );
  const estimatedGlobalWeightBytes = weightBytes;
  const estimatedGlobalOperandBytes = checkedSum(
    estimatedGlobalActivationBytes,
    estimatedGlobalWeightBytes,
    "direct estimated global operand bytes",
  );
  const outputRange = Object.freeze({
    firstOutput: 0,
    outputCount: scalar.outputElements,
    firstWorkgroup: 0,
    workgroupCount: columnTiles,
    multiplyAdds: scheduledMultiplyAdds,
  });
  return Object.freeze({
    ...scalar,
    kernelId: ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID,
    workgroupsX: columnTiles,
    workgroupsY: 1,
    tileRows: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_MAX_ROWS,
    tileColumns: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS,
    tileInner: 2 as const,
    workgroupSize: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE,
    columnTiles,
    innerPairs: scalar.inner / 2,
    workgroupCount: columnTiles,
    scheduledColumns,
    scheduledMultiplyAdds,
    validMultiplyAdds,
    packedWeightStorageShape: Object.freeze([
      scalar.columns,
      scalar.inner / 2,
    ] as const),
    activationBytes,
    weightBytes,
    outputBytes,
    workgroupStorageBytes: 0 as const,
    barriersPerWorkgroup: 0 as const,
    estimatedGlobalActivationBytes,
    estimatedGlobalWeightBytes,
    estimatedGlobalOperandBytes,
    outputRangeCount: 1,
    outputRanges: Object.freeze([outputRange]),
  });
}

export function planAceOpt0083PlannerLowRowBf16Gemv(
  shape: AceGemmShape,
): AceOpt0083PlannerLowRowBf16GemvPlan {
  const scalar = planAceGemm(shape);
  requirePlannerLowRowShape(scalar);

  const columnTiles = Math.ceil(
    scalar.columns / ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS,
  );
  const innerTiles = Math.ceil(
    scalar.inner / ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER,
  );
  if (columnTiles > MAX_DISPATCH_DIMENSION) {
    throw new RangeError(
      "OPT-0083 planner low-row GEMV exceeds the portable dispatch domain",
    );
  }
  const scheduledColumns = checkedProduct(
    columnTiles,
    ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS,
    "scheduled columns",
  );
  const scheduledInner = checkedProduct(
    innerTiles,
    ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER,
    "scheduled inner",
  );
  const scheduledMultiplyAdds = checkedProduct(
    checkedProduct(scalar.rows, scheduledColumns, "scheduled outputs"),
    scheduledInner,
    "scheduled multiply-adds",
  );
  const validMultiplyAdds = checkedProduct(
    scalar.outputElements,
    scalar.inner,
    "valid multiply-adds",
  );
  if (
    scalar.outputElements > ACE_GEMM_MAX_OUTPUTS_PER_RANGE ||
    scheduledMultiplyAdds > ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE
  ) {
    throw new RangeError(
      "OPT-0083 planner low-row GEMV exceeds one bounded output range",
    );
  }

  const activationBytes = checkedProduct(
    scalar.activationElements,
    FLOAT32_BYTES,
    "activation bytes",
  );
  const weightBytes = checkedProduct(
    scalar.weightElements / 2,
    PACKED_BF16_PAIR_BYTES,
    "packed BF16 weight bytes",
  );
  const outputBytes = checkedProduct(
    scalar.outputElements,
    FLOAT32_BYTES,
    "output bytes",
  );
  const activationPanelBytes = checkedProduct(
    checkedProduct(
      scalar.rows,
      ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER,
      "activation panel elements",
    ),
    FLOAT32_BYTES,
    "activation panel bytes",
  );
  const workgroupStorageBytes = checkedSum(
    ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_BYTES,
    activationPanelBytes,
    "workgroup storage bytes",
  );
  const estimatedGlobalActivationBytes = checkedProduct(
    activationBytes,
    columnTiles,
    "estimated global activation bytes",
  );
  const estimatedGlobalWeightBytes = weightBytes;
  const estimatedGlobalOperandBytes = checkedSum(
    estimatedGlobalActivationBytes,
    estimatedGlobalWeightBytes,
    "estimated global operand bytes",
  );
  const outputRange = Object.freeze({
    firstOutput: 0,
    outputCount: scalar.outputElements,
    firstWorkgroup: 0,
    workgroupCount: columnTiles,
    multiplyAdds: scheduledMultiplyAdds,
  });
  return Object.freeze({
    ...scalar,
    kernelId: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID,
    workgroupsX: columnTiles,
    workgroupsY: 1,
    tileRows: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_MAX_ROWS,
    tileColumns: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS,
    tileInner: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER,
    workgroupSize: ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE,
    columnTiles,
    innerTiles,
    workgroupCount: columnTiles,
    scheduledColumns,
    scheduledInner,
    scheduledMultiplyAdds,
    validMultiplyAdds,
    packedWeightStorageShape: Object.freeze([
      scalar.columns,
      scalar.inner / 2,
    ] as const),
    activationBytes,
    weightBytes,
    outputBytes,
    workgroupStorageBytes,
    barriersPerWorkgroup: innerTiles * 2,
    estimatedGlobalActivationBytes,
    estimatedGlobalWeightBytes,
    estimatedGlobalOperandBytes,
    outputRangeCount: 1,
    outputRanges: Object.freeze([outputRange]),
  });
}

export function aceOpt0083PlannerDirectLowRowBf16GemvWgsl(
  shape: AceGemmShape,
): string {
  const plan = planAceOpt0083PlannerDirectLowRowBf16Gemv(shape);
  const secondAccumulator = plan.rows === 2 ? "var sum1 = 0.0;" : "";
  const secondLow = plan.rows === 2
    ? "sum1 = sum1 + activation[INNER + inner] * weight_low;"
    : "";
  const secondHigh = plan.rows === 2
    ? "sum1 = sum1 + activation[INNER + inner + 1u] * weight_high;"
    : "";
  const secondStore = plan.rows === 2
    ? "output[COLUMNS + column] = sum1;"
    : "";
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0083_PLANNER_DIRECT_LOW_ROW_BF16_GEMV_KERNEL_ID}
// weight-layout: source-row-major-packed-bf16-[N,K/2]
// reduction-semantics: strict-increasing-k-fp32-sum-plus-product
const INNER: u32 = ${plan.inner}u;
const INNER_PAIRS: u32 = ${plan.innerPairs}u;
const COLUMNS: u32 = ${plan.columns}u;
const TILE_COLUMNS: u32 = ${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn decode_bf16_low(pair: u32) -> f32 {
  return bitcast<f32>((pair & 0xffffu) << 16u);
}

fn decode_bf16_high(pair: u32) -> f32 {
  return bitcast<f32>(pair & 0xffff0000u);
}

@compute @workgroup_size(${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let column = group.x * TILE_COLUMNS + lane;
  var sum0 = 0.0;
  ${secondAccumulator}
  if (column < COLUMNS) {
    let weight_base = column * INNER_PAIRS;
    for (var pair_index = 0u; pair_index < INNER_PAIRS; pair_index += 1u) {
      let packed = weight[weight_base + pair_index];
      let weight_low = decode_bf16_low(packed);
      let weight_high = decode_bf16_high(packed);
      let inner = pair_index * 2u;
      sum0 = sum0 + activation[inner] * weight_low;
      ${secondLow}
      sum0 = sum0 + activation[inner + 1u] * weight_high;
      ${secondHigh}
    }
    output[column] = sum0;
    ${secondStore}
  }
}
`;
}

export function aceOpt0083PlannerLowRowBf16GemvWgsl(
  shape: AceGemmShape,
): string {
  const plan = planAceOpt0083PlannerLowRowBf16Gemv(shape);
  const secondAccumulator = plan.rows === 2 ? "var sum1 = 0.0;" : "";
  const secondLow = plan.rows === 2
    ? `sum1 = sum1 + activation_panel[${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER}u + local_inner] * weight_low;`
    : "";
  const secondHigh = plan.rows === 2
    ? `sum1 = sum1 + activation_panel[${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER}u + local_inner + 1u] * weight_high;`
    : "";
  const secondStore = plan.rows === 2
    ? "output[COLUMNS + column] = sum1;"
    : "";
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_KERNEL_ID}
// weight-layout: source-row-major-packed-bf16-[N,K/2]
// reduction-semantics: strict-increasing-k-fp32-sum-plus-product
const ROWS: u32 = ${plan.rows}u;
const INNER: u32 = ${plan.inner}u;
const INNER_PAIRS: u32 = ${plan.inner / 2}u;
const COLUMNS: u32 = ${plan.columns}u;
const TILE_COLUMNS: u32 = ${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_COLUMNS}u;
const TILE_INNER: u32 = ${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER}u;
const TILE_INNER_PAIRS: u32 = ${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER / 2}u;
const WEIGHT_PANEL_STRIDE: u32 = ${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WEIGHT_PANEL_STRIDE}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// K64 is 32 packed pairs; the +1 column pad makes both cooperative stores
// and per-column reads conflict-free for a 32-bank workgroup memory.
var<workgroup> weight_panel: array<u32, 32 * 129>;
var<workgroup> activation_panel: array<f32, ${plan.rows * ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_TILE_INNER}>;

fn decode_bf16_low(pair: u32) -> f32 {
  return bitcast<f32>((pair & 0xffffu) << 16u);
}

fn decode_bf16_high(pair: u32) -> f32 {
  return bitcast<f32>(pair & 0xffff0000u);
}

@compute @workgroup_size(${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let column_base = group.x * TILE_COLUMNS;
  let column = column_base + lane;
  var sum0 = 0.0;
  ${secondAccumulator}

  for (var k_base = 0u; k_base < INNER; k_base += TILE_INNER) {
    for (var item = lane; item < ROWS * TILE_INNER; item += ${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE}u) {
      let local_row = item / TILE_INNER;
      let local_inner = item % TILE_INNER;
      let source_inner = k_base + local_inner;
      if (source_inner < INNER) {
        activation_panel[item] = activation[local_row * INNER + source_inner];
      } else {
        activation_panel[item] = 0.0;
      }
    }

    for (var item = lane; item < TILE_COLUMNS * TILE_INNER_PAIRS; item += ${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE}u) {
      let local_column = item / TILE_INNER_PAIRS;
      let local_pair = item % TILE_INNER_PAIRS;
      let source_column = column_base + local_column;
      let source_pair = (k_base >> 1u) + local_pair;
      var packed = 0u;
      if (source_column < COLUMNS && source_pair < INNER_PAIRS) {
        packed = weight[source_column * INNER_PAIRS + source_pair];
      }
      weight_panel[local_pair * WEIGHT_PANEL_STRIDE + local_column] = packed;
    }
    workgroupBarrier();

    if (column < COLUMNS) {
      for (var local_pair = 0u; local_pair < TILE_INNER_PAIRS; local_pair += 1u) {
        let packed = weight_panel[local_pair * WEIGHT_PANEL_STRIDE + lane];
        let weight_low = decode_bf16_low(packed);
        let weight_high = decode_bf16_high(packed);
        let local_inner = local_pair * 2u;
        sum0 = sum0 + activation_panel[local_inner] * weight_low;
        ${secondLow}
        sum0 = sum0 + activation_panel[local_inner + 1u] * weight_high;
        ${secondHigh}
      }
    }
    workgroupBarrier();
  }

  if (column < COLUMNS) {
    output[column] = sum0;
    ${secondStore}
  }
}
`;
}

async function compilePlannerDirectLowRowBf16Gemv(
  device: GPUDevice,
  plan: AceOpt0083PlannerDirectLowRowBf16GemvPlan,
): Promise<CompiledPlannerLowRowBf16Gemv> {
  const label =
    `ace-opt-0083-planner-direct-low-row-bf16-gemv-` + shapeKey(plan);
  const module = device.createShaderModule({
    label,
    code: aceOpt0083PlannerDirectLowRowBf16GemvWgsl(plan),
  });
  await requireShaderCompilation(
    module,
    "OPT-0083 planner direct low-row GEMV",
  );
  return compilePlannerLowRowPipeline(
    device,
    label,
    module,
    plan.activationBytes,
    plan.weightBytes,
    plan.outputBytes,
  );
}

function createPlannerLowRowBindGroup(
  device: GPUDevice,
  label: string,
  layout: GPUBindGroupLayout,
  activation: GPUBufferBinding,
  weight: GPUBufferBinding,
  output: GPUBufferBinding,
): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout,
    entries: [activation, weight, output].map((resource, binding) => ({
      binding,
      resource,
    })),
  });
}

async function compilePlannerLowRowBf16Gemv(
  device: GPUDevice,
  plan: AceOpt0083PlannerLowRowBf16GemvPlan,
): Promise<CompiledPlannerLowRowBf16Gemv> {
  const label =
    `ace-opt-0083-planner-low-row-bf16-gemv-` + shapeKey(plan);
  const module = device.createShaderModule({
    label,
    code: aceOpt0083PlannerLowRowBf16GemvWgsl(plan),
  });
  await requireShaderCompilation(module, "OPT-0083 planner low-row GEMV");
  return compilePlannerLowRowPipeline(
    device,
    label,
    module,
    plan.activationBytes,
    plan.weightBytes,
    plan.outputBytes,
  );
}

async function compilePlannerLowRowPipeline(
  device: GPUDevice,
  label: string,
  module: GPUShaderModule,
  activationBytes: number,
  weightBytes: number,
  outputBytes: number,
): Promise<CompiledPlannerLowRowBf16Gemv> {
  const bindingBytes = [activationBytes, weightBytes, outputBytes];
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

async function requireShaderCompilation(
  module: GPUShaderModule,
  label: string,
): Promise<void> {
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `${label} WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
}

function requireKernelDevice(device: GPUDevice): void {
  const invocations = Number(device.limits.maxComputeInvocationsPerWorkgroup);
  const sizeX = Number(device.limits.maxComputeWorkgroupSizeX);
  const storage = Number(device.limits.maxComputeWorkgroupStorageSize);
  if (
    !Number.isSafeInteger(invocations) ||
    !Number.isSafeInteger(sizeX) ||
    invocations < ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE ||
    sizeX < ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE
  ) {
    throw new Error("OPT-0083 planner low-row GEMV requires a 128x1 workgroup");
  }
  if (
    !Number.isSafeInteger(storage) ||
    storage < ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_MAX_WORKGROUP_BYTES
  ) {
    throw new Error(
      `OPT-0083 planner low-row GEMV requires ${ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_MAX_WORKGROUP_BYTES} workgroup-storage bytes`,
    );
  }
}

function requireDirectKernelDevice(device: GPUDevice): void {
  const invocations = Number(device.limits.maxComputeInvocationsPerWorkgroup);
  const sizeX = Number(device.limits.maxComputeWorkgroupSizeX);
  if (
    !Number.isSafeInteger(invocations) ||
    !Number.isSafeInteger(sizeX) ||
    invocations < ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE ||
    sizeX < ACE_OPT_0083_PLANNER_LOW_ROW_BF16_GEMV_WORKGROUP_SIZE
  ) {
    throw new Error(
      "OPT-0083 planner direct low-row GEMV requires a 128x1 workgroup",
    );
  }
}

function requireDispatchDimensions(device: GPUDevice, workgroupsX: number): void {
  const maximum = Number(device.limits.maxComputeWorkgroupsPerDimension);
  if (
    !Number.isSafeInteger(maximum) ||
    maximum < 1 ||
    workgroupsX > maximum
  ) {
    throw new RangeError(
      "OPT-0083 planner low-row GEMV exceeds the dispatch dimension",
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
      "OPT-0083 planner low-row GEMV device reported invalid buffer limits",
    );
  }
  for (const [name, bytes] of resources) {
    if (bytes > maximumBinding || bytes > maximumBuffer) {
      throw new RangeError(
        `OPT-0083 planner low-row GEMV ${name} exceeds the device buffer limits`,
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
  const alignment = Number(device.limits.minStorageBufferOffsetAlignment);
  if (!isValidGpuAlignment(alignment)) {
    throw new Error(
      "OPT-0083 planner low-row GEMV device reported invalid alignment",
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

function requirePlannerLowRowProfileAndLayout(
  modelProfile: AceModelProfileId,
  weightLayout: AceGemmWeightLayout,
  arm: string,
): void {
  if (modelProfile !== "reference-bf16") {
    throw new Error(
      `OPT-0083 planner ${arm} low-row GEMV requires reference-bf16`,
    );
  }
  if (weightLayout !== "source-row-major") {
    throw new Error(
      `OPT-0083 planner ${arm} low-row GEMV requires source-row-major weights`,
    );
  }
}

function requirePlannerLowRowShape(
  shape: Readonly<{
    rows: number;
    inner: number;
    activationElements: number;
    weightElements: number;
    outputElements: number;
  }>,
): void {
  if (shape.rows !== 1 && shape.rows !== 2) {
    throw new RangeError("OPT-0083 planner low-row GEMV requires M1 or M2");
  }
  if (shape.inner % 2 !== 0) {
    throw new RangeError("OPT-0083 planner low-row GEMV requires even K");
  }
  requireU32Elements(shape.activationElements, "activation");
  requireU32Elements(shape.weightElements, "weight");
  requireU32Elements(shape.outputElements, "output");
}

function requireU32Elements(elements: number, label: string): void {
  if (elements > 0xffff_ffff) {
    throw new RangeError(
      `OPT-0083 planner low-row GEMV ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(
      `OPT-0083 planner low-row GEMV ${label} is not a safe integer`,
    );
  }
  return product;
}

function checkedSum(left: number, right: number, label: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw new RangeError(
      `OPT-0083 planner low-row GEMV ${label} is not a safe integer`,
    );
  }
  return sum;
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}

function shapeKey(shape: AceGemmShape): string {
  return `${shape.rows}x${shape.inner}x${shape.columns}`;
}
