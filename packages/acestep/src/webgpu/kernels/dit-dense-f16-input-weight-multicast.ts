import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../../model/manifest.js";
import { requireAceDisjointOutput } from "./correctness-utils.js";
import {
  ACE_OPT_0078_DENSE_ACCUMULATORS_PER_LANE,
  ACE_OPT_0078_DENSE_BARRIERS_PER_INNER_TILE,
  ACE_OPT_0078_DENSE_COLUMNS_PER_LANE,
  ACE_OPT_0078_DENSE_PACKED_RECORDS_PER_INNER_TILE,
  ACE_OPT_0078_DENSE_PACKED_RECORDS_PER_LANE,
  ACE_OPT_0078_DENSE_ROWS_PER_SUBGROUP,
  ACE_OPT_0078_DENSE_SUBGROUP_SIZE,
  ACE_OPT_0078_DENSE_SUBGROUPS_PER_WORKGROUP,
  ACE_OPT_0078_DENSE_TILE_COLUMNS,
  ACE_OPT_0078_DENSE_TILE_INNER,
  ACE_OPT_0078_DENSE_TILE_ROWS,
  ACE_OPT_0078_DENSE_WEIGHT_MULTICAST_KERNEL_SET_ID,
  ACE_OPT_0078_DENSE_WORKGROUP_SIZE,
  ACE_OPT_0078_DENSE_WORKGROUP_STORAGE_BYTES,
  aceOpt0078DenseWeightMulticastWgsl,
  planAceOpt0078DenseWeightMulticast,
  type AceOpt0078DenseWeightMulticastPlan,
} from "./dit-dense-fp16-weight-multicast.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmKernel,
  AceGemmShape,
} from "./gemm.js";

export const ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_KERNEL_SET_ID =
  "opt-0081-m32-n256-k32-wg256-weight-multicast-typed-f16-input-fp32-output-v1" as const;
export const ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_TILE_ROWS =
  ACE_OPT_0078_DENSE_TILE_ROWS;
export const ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_TILE_COLUMNS =
  ACE_OPT_0078_DENSE_TILE_COLUMNS;
export const ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_TILE_INNER =
  ACE_OPT_0078_DENSE_TILE_INNER;
export const ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_WORKGROUP_SIZE =
  ACE_OPT_0078_DENSE_WORKGROUP_SIZE;
export const ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_SUBGROUP_SIZE =
  ACE_OPT_0078_DENSE_SUBGROUP_SIZE;
export const
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_SUBGROUPS_PER_WORKGROUP =
    ACE_OPT_0078_DENSE_SUBGROUPS_PER_WORKGROUP;
export const ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_ROWS_PER_SUBGROUP =
  ACE_OPT_0078_DENSE_ROWS_PER_SUBGROUP;
export const ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_COLUMNS_PER_LANE =
  ACE_OPT_0078_DENSE_COLUMNS_PER_LANE;
export const
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_ACCUMULATORS_PER_LANE =
    ACE_OPT_0078_DENSE_ACCUMULATORS_PER_LANE;
export const
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_PACKED_RECORDS_PER_INNER_TILE =
    ACE_OPT_0078_DENSE_PACKED_RECORDS_PER_INNER_TILE;
export const
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_PACKED_RECORDS_PER_LANE =
    ACE_OPT_0078_DENSE_PACKED_RECORDS_PER_LANE;
export const
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_BARRIERS_PER_INNER_TILE =
    ACE_OPT_0078_DENSE_BARRIERS_PER_INNER_TILE;
export const
  ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_WORKGROUP_STORAGE_BYTES =
    ACE_OPT_0078_DENSE_WORKGROUP_STORAGE_BYTES;

const FLOAT16_BYTES = 2;
const FLOAT32_BYTES = 4;
const GPU_BUFFER_ALIGNMENT = 4;

export type AceOpt0081DenseF16InputWeightMulticastPlan = Readonly<
  Omit<
    AceOpt0078DenseWeightMulticastPlan,
    | "kernelSetId"
    | "estimatedGlobalActivationBytes"
    | "estimatedGlobalOperandBytes"
  > & {
    readonly kernelSetId:
      typeof ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_KERNEL_SET_ID;
    readonly referenceKernelSetId:
      typeof ACE_OPT_0078_DENSE_WEIGHT_MULTICAST_KERNEL_SET_ID;
    readonly activationStorage: "scalar-f16";
    readonly activationElementBytes: typeof FLOAT16_BYTES;
    readonly weightElementBytes: typeof FLOAT16_BYTES;
    readonly outputElementBytes: typeof FLOAT32_BYTES;
    readonly activationBytes: number;
    readonly weightBytes: number;
    readonly outputBytes: number;
    readonly estimatedGlobalActivationBytes: number;
    readonly estimatedGlobalOperandBytes: number;
  }
>;

export interface AceOpt0081DenseF16InputWeightMulticastDispatch
  extends AceGemmDispatch {
  readonly kernelSetId:
    typeof ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_KERNEL_SET_ID;
  readonly weightLayout: typeof ACE_DIT_DENSE_FP16_TILE_LAYOUT;
  readonly plan: AceOpt0081DenseF16InputWeightMulticastPlan;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/**
 * OPT-0081 arm C. The owner preserves OPT-0078's exact output and packed-weight
 * ownership, but consumes the scalar typed-F16 value emitted by the selected
 * producer boundary instead of loading an FP32 activation and casting it.
 * It is deliberately not selectable by the production graph.
 */
export class AceOpt0081DenseF16InputWeightMulticastKernel
  implements AceGemmKernel {
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
  ): AceOpt0081DenseF16InputWeightMulticastKernel {
    requireKernelDevice(device, capability);
    return new AceOpt0081DenseF16InputWeightMulticastKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0081DenseF16InputWeightMulticastDispatch> {
    this.requireLive();
    if (bindings.bias !== undefined) {
      throw new RangeError(
        "OPT-0081 typed-F16 weight-multicast dense rejects bias",
      );
    }
    const plan = planAceOpt0081DenseF16InputWeightMulticast(shape);
    requireDispatchDimensions(this.device, plan.workgroupsX, plan.workgroupsY);
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
    const resources = Object.freeze([activation, weight, output]);
    const bindGroup = this.bindGroupFor(
      shapeKey(plan),
      `${label}-opt-0081-typed-f16-weight-multicast-bindings`,
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
      kernelSetId:
        ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_KERNEL_SET_ID,
      weightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
      plan,
      rangeCount: 1,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(
            `${label} OPT-0081 typed-F16 weight-multicast dense range must be zero`,
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
    this.bindGroups.clear();
  }

  private pipelineFor(
    plan: AceOpt0081DenseF16InputWeightMulticastPlan,
  ): Promise<CompiledKernel> {
    const key = shapeKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileTypedF16WeightMulticast(this.device, plan);
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
      throw new Error(
        "OPT-0081 typed-F16 weight-multicast dense kernel was destroyed",
      );
    }
  }
}

export function planAceOpt0081DenseF16InputWeightMulticast(
  shape: AceGemmShape,
): AceOpt0081DenseF16InputWeightMulticastPlan {
  const reference = planAceOpt0078DenseWeightMulticast(shape);
  const activationBytes = checkedProduct(
    reference.activationElements,
    FLOAT16_BYTES,
    "activation bytes",
  );
  const weightBytes = checkedProduct(
    reference.weightElements,
    FLOAT16_BYTES,
    "weight bytes",
  );
  const outputBytes = checkedProduct(
    reference.outputElements,
    FLOAT32_BYTES,
    "output bytes",
  );
  const estimatedGlobalActivationBytes = checkedProduct(
    checkedProduct(
      reference.scheduledRows,
      reference.inner,
      "scheduled activation elements",
    ),
    checkedProduct(
      reference.columnTiles,
      FLOAT16_BYTES,
      "activation column-tile bytes",
    ),
    "global activation request bytes",
  );
  const estimatedGlobalOperandBytes = checkedSum(
    estimatedGlobalActivationBytes,
    reference.estimatedGlobalWeightBytes,
    "global operand request bytes",
  );
  return Object.freeze({
    ...reference,
    kernelSetId:
      ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_KERNEL_SET_ID,
    referenceKernelSetId:
      ACE_OPT_0078_DENSE_WEIGHT_MULTICAST_KERNEL_SET_ID,
    activationStorage: "scalar-f16",
    activationElementBytes: FLOAT16_BYTES,
    weightElementBytes: FLOAT16_BYTES,
    outputElementBytes: FLOAT32_BYTES,
    activationBytes,
    weightBytes,
    outputBytes,
    estimatedGlobalActivationBytes,
    estimatedGlobalOperandBytes,
  });
}

export function aceOpt0081DenseF16InputWeightMulticastWgsl(
  shape: AceGemmShape,
): string {
  let source = aceOpt0078DenseWeightMulticastWgsl(shape);
  source = replaceExactlyOnce(
    source,
    `// kernel-id: ${ACE_OPT_0078_DENSE_WEIGHT_MULTICAST_KERNEL_SET_ID}`,
    `// kernel-id: ${ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_KERNEL_SET_ID}`,
    "kernel identity",
  );
  source = replaceExactlyOnce(
    source,
    "@group(0) @binding(0) var<storage, read> activation: array<f32>;",
    "@group(0) @binding(0) var<storage, read> activation: array<f16>;",
    "typed activation declaration",
  );
  source = replaceExactlyOnce(
    source,
    "lane_a = f16(activation[lane_row * INNER + inner]);",
    "lane_a = activation[lane_row * INNER + inner];",
    "native typed activation load",
  );
  return source;
}

async function compileTypedF16WeightMulticast(
  device: GPUDevice,
  plan: AceOpt0081DenseF16InputWeightMulticastPlan,
): Promise<CompiledKernel> {
  const label =
    `ace-opt-0081-typed-f16-weight-multicast-${plan.rows}x${plan.inner}x${plan.columns}`;
  const module = await checkedShaderModule(
    device,
    label,
    aceOpt0081DenseF16InputWeightMulticastWgsl(plan),
  );
  const bindingBytes = [
    plan.activationBytes,
    plan.weightBytes,
    plan.outputBytes,
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
      `OPT-0081 typed-F16 weight-multicast dense WGSL failed: ${errors.map(
        (message) =>
          `${message.lineNum}:${message.linePos} ${message.message}`,
      ).join("; ")}`,
    );
  }
  return module;
}

function replaceExactlyOnce(
  source: string,
  expected: string,
  replacement: string,
  label: string,
): string {
  const first = source.indexOf(expected);
  if (
    first < 0 ||
    source.indexOf(expected, first + expected.length) >= 0
  ) {
    throw new Error(
      `OPT-0081 typed-F16 weight-multicast ${label} source changed`,
    );
  }
  return source.slice(0, first) + replacement +
    source.slice(first + expected.length);
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
    capability.subgroupMinSize !==
      ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_SUBGROUP_SIZE ||
    capability.subgroupMaxSize !==
      ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_SUBGROUP_SIZE
  ) {
    throw new Error(
      "OPT-0081 typed-F16 weight-multicast dense requires shader-f16 and fixed 32-lane subgroups",
    );
  }
  const invocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const sizeX = device.limits.maxComputeWorkgroupSizeX;
  const storage = device.limits.maxComputeWorkgroupStorageSize;
  if (
    !Number.isSafeInteger(invocations) ||
    !Number.isSafeInteger(sizeX) ||
    invocations <
      ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_WORKGROUP_SIZE ||
    sizeX < ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_WORKGROUP_SIZE
  ) {
    throw new Error(
      "OPT-0081 typed-F16 weight-multicast dense requires a 256x1 workgroup",
    );
  }
  if (
    !Number.isSafeInteger(storage) ||
    storage <
      ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_WORKGROUP_STORAGE_BYTES
  ) {
    throw new Error(
      `OPT-0081 typed-F16 weight-multicast dense requires ${ACE_OPT_0081_DENSE_F16_INPUT_WEIGHT_MULTICAST_WORKGROUP_STORAGE_BYTES} workgroup-storage bytes`,
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
      "OPT-0081 typed-F16 weight-multicast dense exceeds the dispatch dimension",
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
      "OPT-0081 typed-F16 weight-multicast dense device reported invalid buffer limits",
    );
  }
  for (const [name, bytes] of resources) {
    if (bytes > maximumBinding || bytes > maximumBuffer) {
      throw new RangeError(
        `OPT-0081 typed-F16 weight-multicast dense ${name} exceeds the device buffer limits`,
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
      "OPT-0081 typed-F16 weight-multicast dense device reported invalid alignment",
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

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(
      `OPT-0081 typed-F16 weight-multicast dense ${label} is not a safe integer`,
    );
  }
  return product;
}

function checkedSum(left: number, right: number, label: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw new RangeError(
      `OPT-0081 typed-F16 weight-multicast dense ${label} is not a safe integer`,
    );
  }
  return sum;
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
