import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../../model/manifest.js";
import { requireAceDisjointOutput } from "./correctness-utils.js";
import {
  ACE_OPT_0009_DENSE_SUBGROUP_SIZE,
  ACE_OPT_0009_DENSE_TILE_COLUMNS,
  ACE_OPT_0009_DENSE_TILE_INNER,
  ACE_OPT_0009_DENSE_TILE_ROWS,
  ACE_OPT_0009_DENSE_WORKGROUP_SIZE,
  aceOpt0009DenseGemmWgsl,
  planAceOpt0009DenseGemm,
  type AceOpt0009DenseGemmPlan,
} from "./dit-dense-fp16.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmKernel,
  AceGemmShape,
} from "./gemm.js";

export const ACE_OPT_0081_DENSE_F16_INPUT_KERNEL_SET_ID =
  "opt-0081-m32-n256-k32-wg128-scalar-f16-input-fp32-output-v1" as const;
export const ACE_OPT_0081_DENSE_F16_INPUT_TILE_ROWS =
  ACE_OPT_0009_DENSE_TILE_ROWS;
export const ACE_OPT_0081_DENSE_F16_INPUT_TILE_COLUMNS =
  ACE_OPT_0009_DENSE_TILE_COLUMNS;
export const ACE_OPT_0081_DENSE_F16_INPUT_TILE_INNER =
  ACE_OPT_0009_DENSE_TILE_INNER;
export const ACE_OPT_0081_DENSE_F16_INPUT_WORKGROUP_SIZE =
  ACE_OPT_0009_DENSE_WORKGROUP_SIZE;
export const ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUP_SIZE =
  ACE_OPT_0009_DENSE_SUBGROUP_SIZE;
export const ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUPS_PER_WORKGROUP =
  ACE_OPT_0081_DENSE_F16_INPUT_WORKGROUP_SIZE /
  ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUP_SIZE;
export const ACE_OPT_0081_DENSE_F16_INPUT_ROWS_PER_SUBGROUP =
  ACE_OPT_0081_DENSE_F16_INPUT_TILE_ROWS /
  ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUPS_PER_WORKGROUP;
export const ACE_OPT_0081_DENSE_F16_INPUT_COLUMNS_PER_LANE =
  ACE_OPT_0081_DENSE_F16_INPUT_TILE_COLUMNS /
  ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUP_SIZE;
export const ACE_OPT_0081_DENSE_F16_INPUT_ACCUMULATORS_PER_LANE =
  ACE_OPT_0081_DENSE_F16_INPUT_ROWS_PER_SUBGROUP *
  ACE_OPT_0081_DENSE_F16_INPUT_COLUMNS_PER_LANE;
export const ACE_OPT_0081_DENSE_F16_INPUT_ACTIVATION_ELEMENT_BYTES = 2;

const M2250 = 2_250;
const FLOAT16_BYTES = 2;
const FLOAT32_BYTES = 4;
const GPU_BUFFER_ALIGNMENT = 4;
const PACKED_RECORDS_PER_INNER_TILE_PER_SUBGROUP =
  ACE_OPT_0081_DENSE_F16_INPUT_TILE_INNER *
  ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUP_SIZE;

export interface AceOpt0081DenseF16InputPlan
  extends AceOpt0009DenseGemmPlan {
  readonly kernelSetId: typeof ACE_OPT_0081_DENSE_F16_INPUT_KERNEL_SET_ID;
  readonly subgroupsPerWorkgroup:
    typeof ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUPS_PER_WORKGROUP;
  readonly rowsPerSubgroup:
    typeof ACE_OPT_0081_DENSE_F16_INPUT_ROWS_PER_SUBGROUP;
  readonly columnsPerLane:
    typeof ACE_OPT_0081_DENSE_F16_INPUT_COLUMNS_PER_LANE;
  readonly accumulatorsPerLane:
    typeof ACE_OPT_0081_DENSE_F16_INPUT_ACCUMULATORS_PER_LANE;
  readonly activationElementBytes:
    typeof ACE_OPT_0081_DENSE_F16_INPUT_ACTIVATION_ELEMENT_BYTES;
  readonly scheduledRows: number;
  readonly scheduledMultiplyAdds: number;
  readonly validMultiplyAdds: number;
  readonly packedRecordLoadsPerWorkgroup: number;
  readonly estimatedGlobalActivationBytes: number;
  readonly estimatedGlobalWeightBytes: number;
  readonly estimatedGlobalOperandBytes: number;
  readonly estimatedGlobalOutputBytes: number;
}

export interface AceOpt0081DenseF16InputDispatch extends AceGemmDispatch {
  readonly kernelSetId: typeof ACE_OPT_0081_DENSE_F16_INPUT_KERNEL_SET_ID;
  readonly weightLayout: typeof ACE_DIT_DENSE_FP16_TILE_LAYOUT;
  readonly plan: AceOpt0081DenseF16InputPlan;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/**
 * OPT-0081 arm B. It preserves OPT-0009's exact M32/N256/K32/WG128
 * ownership and increasing-K FP32 accumulation while consuming a scalar
 * typed-F16 activation binding produced at the preceding operation's store.
 * This isolated owner is not selectable by the production graph.
 */
export class AceOpt0081DenseF16InputKernel implements AceGemmKernel {
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
  ): AceOpt0081DenseF16InputKernel {
    requireKernelDevice(device, capability);
    return new AceOpt0081DenseF16InputKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0081DenseF16InputDispatch> {
    this.requireLive();
    if (bindings.bias !== undefined) {
      throw new RangeError("OPT-0081 typed-F16 input dense rejects bias");
    }
    const plan = planAceOpt0081DenseF16Input(shape);
    requireDispatchDimensions(this.device, plan.workgroupsX, plan.workgroupsY);
    const activationBytes = checkedProduct(
      plan.activationElements,
      FLOAT16_BYTES,
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
      `${label}-opt-0081-typed-f16-input-bindings`,
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
      kernelSetId: ACE_OPT_0081_DENSE_F16_INPUT_KERNEL_SET_ID,
      weightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
      plan,
      rangeCount: 1,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(
            `${label} OPT-0081 typed-F16 input dense range must be zero`,
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
    plan: AceOpt0081DenseF16InputPlan,
  ): Promise<CompiledKernel> {
    const key = shapeKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileDenseF16Input(this.device, plan);
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
      throw new Error("OPT-0081 typed-F16 input dense kernel was destroyed");
    }
  }
}

export function planAceOpt0081DenseF16Input(
  shape: AceGemmShape,
): AceOpt0081DenseF16InputPlan {
  if (shape.rows !== M2250) {
    throw new RangeError("OPT-0081 typed-F16 input dense requires exact M2250");
  }
  const reference = planAceOpt0009DenseGemm(shape);
  if (
    reference.tileRows !== ACE_OPT_0081_DENSE_F16_INPUT_TILE_ROWS ||
    reference.tileColumns !== ACE_OPT_0081_DENSE_F16_INPUT_TILE_COLUMNS ||
    reference.tileInner !== ACE_OPT_0081_DENSE_F16_INPUT_TILE_INNER ||
    reference.workgroupSize !==
      ACE_OPT_0081_DENSE_F16_INPUT_WORKGROUP_SIZE ||
    reference.subgroupSize !== ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUP_SIZE ||
    reference.workgroupsX !== reference.columnTiles ||
    reference.workgroupsY !== reference.rowTiles ||
    reference.outputRangeCount !== 1 ||
    reference.outputRanges.length !== 1 ||
    reference.packedWeightStorageShape[2] !== 32 ||
    reference.packedWeightStorageShape[3] !== 256
  ) {
    throw new Error(
      "OPT-0081 typed-F16 input dense OPT-0009 reference geometry changed",
    );
  }
  const scheduledRows = checkedProduct(
    reference.rowTiles,
    ACE_OPT_0081_DENSE_F16_INPUT_TILE_ROWS,
    "scheduled rows",
  );
  const scheduledMultiplyAdds = checkedProduct(
    checkedProduct(scheduledRows, reference.inner, "scheduled row-inner"),
    reference.columns,
    "scheduled multiply-adds",
  );
  const validMultiplyAdds = checkedProduct(
    reference.activationElements,
    reference.columns,
    "valid multiply-adds",
  );
  const packedRecordLoadsPerWorkgroup = checkedProduct(
    reference.innerTiles,
    checkedProduct(
      PACKED_RECORDS_PER_INNER_TILE_PER_SUBGROUP,
      ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUPS_PER_WORKGROUP,
      "packed-record loads per inner tile",
    ),
    "packed-record loads per workgroup",
  );
  const estimatedGlobalActivationBytes = checkedProduct(
    checkedProduct(
      scheduledRows,
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
  const estimatedGlobalWeightBytes = checkedProduct(
    checkedProduct(
      reference.weightElements,
      FLOAT16_BYTES,
      "weight payload bytes",
    ),
    checkedProduct(
      reference.rowTiles,
      ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUPS_PER_WORKGROUP,
      "weight row-subgroup repeats",
    ),
    "global weight request bytes",
  );
  const estimatedGlobalOperandBytes = checkedSum(
    estimatedGlobalActivationBytes,
    estimatedGlobalWeightBytes,
    "global operand request bytes",
  );
  const estimatedGlobalOutputBytes = checkedProduct(
    reference.outputElements,
    FLOAT32_BYTES,
    "global output bytes",
  );
  const outputRange = reference.outputRanges[0]!;
  if (
    outputRange.firstOutput !== 0 ||
    outputRange.outputCount !== reference.outputElements ||
    outputRange.firstWorkgroup !== 0 ||
    outputRange.workgroupCount !== reference.workgroupCount ||
    outputRange.multiplyAdds !== scheduledMultiplyAdds
  ) {
    throw new Error(
      "OPT-0081 typed-F16 input dense OPT-0009 output range changed",
    );
  }
  return Object.freeze({
    ...reference,
    kernelSetId: ACE_OPT_0081_DENSE_F16_INPUT_KERNEL_SET_ID,
    subgroupsPerWorkgroup:
      ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUPS_PER_WORKGROUP,
    rowsPerSubgroup: ACE_OPT_0081_DENSE_F16_INPUT_ROWS_PER_SUBGROUP,
    columnsPerLane: ACE_OPT_0081_DENSE_F16_INPUT_COLUMNS_PER_LANE,
    accumulatorsPerLane: ACE_OPT_0081_DENSE_F16_INPUT_ACCUMULATORS_PER_LANE,
    activationElementBytes:
      ACE_OPT_0081_DENSE_F16_INPUT_ACTIVATION_ELEMENT_BYTES,
    scheduledRows,
    scheduledMultiplyAdds,
    validMultiplyAdds,
    packedRecordLoadsPerWorkgroup,
    estimatedGlobalActivationBytes,
    estimatedGlobalWeightBytes,
    estimatedGlobalOperandBytes,
    estimatedGlobalOutputBytes,
  });
}

export function aceOpt0081DenseF16InputWgsl(shape: AceGemmShape): string {
  planAceOpt0081DenseF16Input(shape);
  let source =
    `// kernel-id: ${ACE_OPT_0081_DENSE_F16_INPUT_KERNEL_SET_ID}\n` +
    "// reference-owner: OPT-0009-exact-body\n" +
    "// activation-storage: scalar-array-f16-producer-boundary\n" +
    "// reduction-semantics: strict-increasing-k-fp32-sum-plus-product\n" +
    aceOpt0009DenseGemmWgsl(shape);
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

async function compileDenseF16Input(
  device: GPUDevice,
  plan: AceOpt0081DenseF16InputPlan,
): Promise<CompiledKernel> {
  const label =
    `ace-opt-0081-dense-f16-input-${plan.rows}x${plan.inner}x${plan.columns}`;
  const module = await checkedShaderModule(
    device,
    label,
    aceOpt0081DenseF16InputWgsl(plan),
  );
  const bindingBytes = [
    checkedProduct(plan.activationElements, FLOAT16_BYTES, "activation bytes"),
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
      `OPT-0081 typed-F16 input dense WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
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
    throw new Error(`OPT-0081 typed-F16 input ${label} source changed`);
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
    capability.subgroupMinSize !== ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUP_SIZE ||
    capability.subgroupMaxSize !== ACE_OPT_0081_DENSE_F16_INPUT_SUBGROUP_SIZE
  ) {
    throw new Error(
      "OPT-0081 typed-F16 input dense requires shader-f16 and fixed 32-lane subgroups",
    );
  }
  const invocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const sizeX = device.limits.maxComputeWorkgroupSizeX;
  if (
    !Number.isSafeInteger(invocations) ||
    !Number.isSafeInteger(sizeX) ||
    invocations < ACE_OPT_0081_DENSE_F16_INPUT_WORKGROUP_SIZE ||
    sizeX < ACE_OPT_0081_DENSE_F16_INPUT_WORKGROUP_SIZE
  ) {
    throw new Error("OPT-0081 typed-F16 input dense requires a 128x1 workgroup");
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
      "OPT-0081 typed-F16 input dense exceeds the dispatch dimension",
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
      "OPT-0081 typed-F16 input dense device reported invalid buffer limits",
    );
  }
  for (const [name, bytes] of resources) {
    if (bytes > maximumBinding || bytes > maximumBuffer) {
      throw new RangeError(
        `OPT-0081 typed-F16 input dense ${name} exceeds the device buffer limits`,
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
      "OPT-0081 typed-F16 input dense device reported invalid alignment",
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
      `OPT-0081 typed-F16 input dense ${label} is not a safe integer`,
    );
  }
  return product;
}

function checkedSum(left: number, right: number, label: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw new RangeError(
      `OPT-0081 typed-F16 input dense ${label} is not a safe integer`,
    );
  }
  return sum;
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
