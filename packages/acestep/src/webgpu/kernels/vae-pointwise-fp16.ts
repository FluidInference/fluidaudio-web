import {
  ACE_CORRECTNESS_WORKGROUP_SIZE,
  checkedAceProduct,
  planAceLinearDispatch,
} from "./correctness-utils.js";
import type {
  AceVaeOutputRangeBinding,
  AceVaePointwiseShape,
} from "./vae-primitives.js";

export const ACE_FP16_VAE_POINTWISE_WORKGROUP_SIZE =
  ACE_CORRECTNESS_WORKGROUP_SIZE;
export const ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID =
  "ace-vae-fp16-portable-ingress-v1" as const;
export const ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID =
  "ace-vae-fp16-portable-add-v1" as const;

const FLOAT16_BYTES = 2;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const GPU_BUFFER_ALIGNMENT = 4;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_CONTROL_BYTES = 16;

export interface AceFp16VaePointwisePlan extends AceVaePointwiseShape {
  readonly operation: "ingress" | "add";
  readonly elements: number;
  readonly sourceStorageBytes: number;
  readonly sourceBindingBytes: number;
  readonly outputStorageBytes: number;
  readonly outputBindingBytes: number;
  readonly workgroupSize: typeof ACE_FP16_VAE_POINTWISE_WORKGROUP_SIZE;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceFp16VaeIngressPlan extends AceFp16VaePointwisePlan {
  readonly operation: "ingress";
}

export interface AceFp16VaeAddPlan extends AceFp16VaePointwisePlan {
  readonly operation: "add";
}

export interface AceFp16VaePointwiseRangePlan {
  readonly base: number;
  readonly count: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceFp16VaeIngressBindings {
  /** FP32 decoder input in frame-major NLC order. */
  readonly input: GPUBufferBinding;
  /** FP16 decoder ingress activation in the same NLC order. */
  readonly output: GPUBufferBinding;
}

export interface AceFp16VaeAddBindings {
  readonly left: GPUBufferBinding;
  readonly right: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export type AceFp16VaePointwiseKernelId =
  | typeof ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID
  | typeof ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID;

export interface AceFp16VaePointwiseDispatch<
  Plan extends AceFp16VaePointwisePlan,
  KernelId extends AceFp16VaePointwiseKernelId,
> {
  readonly label: string;
  readonly kernelId: KernelId;
  readonly plan: Plan;
  readonly outputRange: AceFp16VaePointwiseRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

export type AceFp16VaeIngressDispatch = AceFp16VaePointwiseDispatch<
  AceFp16VaeIngressPlan,
  typeof ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID
>;

export type AceFp16VaeAddDispatch = AceFp16VaePointwiseDispatch<
  AceFp16VaeAddPlan,
  typeof ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID
>;

interface CompiledAceFp16VaePointwise {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

interface NamedBinding {
  readonly name: "input" | "left" | "right" | "output" | "range control";
  readonly binding: GPUBufferBinding;
}

/**
 * Portable production `shader-f16` ingress and residual Add for OPT-0011.
 *
 * Both kernels keep the Stage-1 decoder's immutable ranged-dispatch control
 * records. Add expands both FP16 operands into FP32 registers, performs one
 * FP32 addition, then rounds exactly once at the FP16 store boundary.
 */
export class AceFp16VaePointwiseKernel {
  private readonly pipelines = new Map<
    string,
    Promise<CompiledAceFp16VaePointwise>
  >();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {
    requireKernelDevice(device);
  }

  static create(device: GPUDevice): AceFp16VaePointwiseKernel {
    return new AceFp16VaePointwiseKernel(device);
  }

  async createIngressDispatch(
    label: string,
    shape: AceVaePointwiseShape,
    bindings: AceFp16VaeIngressBindings,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceFp16VaeIngressDispatch> {
    this.requireLive();
    const plan = planAceFp16VaeIngress(shape);
    const outputRange = planAceFp16VaePointwiseRange(plan, range);
    this.requireDeviceLimits(plan, outputRange);
    const normalized = Object.freeze([
      {
        name: "input" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.input,
          plan.sourceStorageBytes,
          plan.sourceBindingBytes,
          `${label} input`,
        ),
      },
      {
        name: "output" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.output,
          plan.outputStorageBytes,
          plan.outputBindingBytes,
          `${label} output`,
        ),
      },
      {
        name: "range control" as const,
        binding: requireRangeControlBinding(
          this.device,
          range.control,
          `${label} range control`,
        ),
      },
    ] satisfies readonly NamedBinding[]);
    requireDisjointBindings(normalized, label);
    return this.createDispatch(
      label,
      ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
      plan,
      outputRange,
      normalized,
      aceFp16VaeIngressWgsl(),
    );
  }

  async createAddDispatch(
    label: string,
    shape: AceVaePointwiseShape,
    bindings: AceFp16VaeAddBindings,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceFp16VaeAddDispatch> {
    this.requireLive();
    const plan = planAceFp16VaeAdd(shape);
    const outputRange = planAceFp16VaePointwiseRange(plan, range);
    this.requireDeviceLimits(plan, outputRange);
    const normalized = Object.freeze([
      {
        name: "left" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.left,
          plan.sourceStorageBytes,
          plan.sourceBindingBytes,
          `${label} left`,
        ),
      },
      {
        name: "right" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.right,
          plan.sourceStorageBytes,
          plan.sourceBindingBytes,
          `${label} right`,
        ),
      },
      {
        name: "output" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.output,
          plan.outputStorageBytes,
          plan.outputBindingBytes,
          `${label} output`,
        ),
      },
      {
        name: "range control" as const,
        binding: requireRangeControlBinding(
          this.device,
          range.control,
          `${label} range control`,
        ),
      },
    ] satisfies readonly NamedBinding[]);
    requireDisjointBindings(normalized, label);
    return this.createDispatch(
      label,
      ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
      plan,
      outputRange,
      normalized,
      aceFp16VaeAddWgsl(),
    );
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.bindGroups.clear();
    this.pipelines.clear();
  }

  private async createDispatch<
    Plan extends AceFp16VaePointwisePlan,
    KernelId extends AceFp16VaePointwiseKernelId,
  >(
    label: string,
    kernelId: KernelId,
    plan: Plan,
    outputRange: AceFp16VaePointwiseRangePlan,
    normalized: readonly NamedBinding[],
    code: string,
  ): Promise<AceFp16VaePointwiseDispatch<Plan, KernelId>> {
    const key = pointwiseKey(plan);
    const compiled = await this.pipelineFor(key, plan, code);
    this.requireLive();

    const rangeBindingIndex = normalized.length - 1;
    const controlOffset = normalized[rangeBindingIndex]!.binding.offset ?? 0;
    const bindGroupResources = normalized.map(({ binding }, index) =>
      index === rangeBindingIndex
        ? Object.freeze({
            buffer: binding.buffer,
            offset: 0,
            size: OUTPUT_RANGE_CONTROL_BYTES,
          })
        : binding
    );
    const bindGroupKey = `${key}:${bindGroupResources.map((binding) =>
      this.bindingKey(binding)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-fp16-${plan.operation}-bindings`,
        layout: compiled.bindGroupLayout,
        entries: bindGroupResources.map((resource, binding) => ({
          binding,
          resource,
        })),
      });
      this.bindGroups.set(bindGroupKey, bindGroup);
    }

    const owner = this;
    return Object.freeze({
      label,
      kernelId,
      plan,
      outputRange,
      encode(pass: GPUComputePassEncoder): void {
        owner.requireLive();
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroup, [controlOffset]);
        pass.dispatchWorkgroups(
          outputRange.workgroupsX,
          outputRange.workgroupsY,
          1,
        );
      },
    });
  }

  private pipelineFor(
    key: string,
    plan: AceFp16VaePointwisePlan,
    code: string,
  ): Promise<CompiledAceFp16VaePointwise> {
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceFp16VaePointwise(this.device, plan, code);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
  }

  private requireDeviceLimits(
    plan: AceFp16VaePointwisePlan,
    range: AceFp16VaePointwiseRangePlan,
  ): void {
    const maximumDispatch = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (
      !Number.isSafeInteger(maximumDispatch) || maximumDispatch < 1 ||
      range.workgroupsX > maximumDispatch ||
      range.workgroupsY > maximumDispatch
    ) {
      throw new RangeError(
        `ACE FP16 VAE ${plan.operation} range exceeds the device dispatch dimension`,
      );
    }
    const maximumStorage = Number(
      this.device.limits.maxStorageBufferBindingSize,
    );
    const maximumUniform = Number(
      this.device.limits.maxUniformBufferBindingSize,
    );
    const maximumBuffer = Number(this.device.limits.maxBufferSize);
    if (
      !Number.isSafeInteger(maximumStorage) || maximumStorage < 1 ||
      !Number.isSafeInteger(maximumUniform) ||
      maximumUniform < OUTPUT_RANGE_CONTROL_BYTES ||
      !Number.isSafeInteger(maximumBuffer) || maximumBuffer < 1
    ) {
      throw new RangeError(
        `ACE FP16 VAE ${plan.operation} device reported invalid buffer limits`,
      );
    }
    for (const [name, bytes] of [
      ["source", plan.sourceBindingBytes],
      ["output", plan.outputBindingBytes],
    ] as const) {
      if (bytes > maximumStorage) {
        throw new RangeError(
          `ACE FP16 VAE ${plan.operation} ${name} exceeds the device storage binding limit`,
        );
      }
      if (bytes > maximumBuffer) {
        throw new RangeError(
          `ACE FP16 VAE ${plan.operation} ${name} exceeds the device buffer limit`,
        );
      }
    }
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
      throw new Error("ACE FP16 VAE pointwise kernel was destroyed");
    }
  }
}

export function planAceFp16VaeIngress(
  shape: AceVaePointwiseShape,
): AceFp16VaeIngressPlan {
  return planPointwise(shape, "ingress") as AceFp16VaeIngressPlan;
}

export function planAceFp16VaeAdd(
  shape: AceVaePointwiseShape,
): AceFp16VaeAddPlan {
  return planPointwise(shape, "add") as AceFp16VaeAddPlan;
}

export function planAceFp16VaePointwiseRange(
  plan: AceFp16VaePointwisePlan,
  range: Readonly<{ base: number; count: number }>,
): AceFp16VaePointwiseRangePlan {
  if (
    !Number.isSafeInteger(range.base) || range.base < 0 ||
    !Number.isSafeInteger(range.count) || range.count <= 0 ||
    range.base > MAX_WGSL_U32 || range.count > MAX_WGSL_U32 ||
    !Number.isSafeInteger(range.base + range.count) ||
    range.base + range.count > plan.elements
  ) {
    throw new RangeError(
      `ACE FP16 VAE ${plan.operation} range must be non-empty and inside its complete output domain`,
    );
  }
  const dispatch = planAceLinearDispatch(
    range.count,
    `ACE FP16 VAE ${plan.operation} range`,
  );
  return Object.freeze({
    base: range.base,
    count: range.count,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

export function aceFp16VaeIngressWgsl(): string {
  return pointwiseWgsl(
    ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
    "@group(0) @binding(0) var<storage, read> input: array<f32>;\n" +
      "@group(0) @binding(1) var<storage, read_write> output: array<f16>;",
    2,
    "output[index] = f16(input[index]);",
  );
}

export function aceFp16VaeAddWgsl(): string {
  return pointwiseWgsl(
    ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
    "@group(0) @binding(0) var<storage, read> left: array<f16>;\n" +
      "@group(0) @binding(1) var<storage, read> right: array<f16>;\n" +
      "@group(0) @binding(2) var<storage, read_write> output: array<f16>;",
    3,
    `let left_operand: f32 = f32(left[index]);
    let right_operand: f32 = f32(right[index]);
    let sum: f32 = left_operand + right_operand;
    output[index] = f16(sum);`,
  );
}

function pointwiseWgsl(
  kernelId: AceFp16VaePointwiseKernelId,
  bindings: string,
  rangeBinding: number,
  body: string,
): string {
  return /* wgsl */ `
// kernel-id: ${kernelId}
enable f16;

${bindings}

struct OutputRangeParameters {
  first_output: u32,
  output_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(${rangeBinding}) var<uniform>
  output_range: OutputRangeParameters;

fn quantum_workgroup_index(
  workgroup_id: vec3<u32>,
  num_workgroups: vec3<u32>,
) -> u32 {
  return workgroup_id.y * num_workgroups.x + workgroup_id.x;
}

@compute @workgroup_size(
  ${ACE_FP16_VAE_POINTWISE_WORKGROUP_SIZE},
  1,
  1,
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let workgroup = quantum_workgroup_index(group, num_workgroups);
  let full_workgroups = output_range.output_count /
    ${ACE_FP16_VAE_POINTWISE_WORKGROUP_SIZE}u;
  let tail_workgroup = select(
    0u,
    1u,
    (output_range.output_count % ${ACE_FP16_VAE_POINTWISE_WORKGROUP_SIZE}u) != 0u,
  );
  let active_workgroups = full_workgroups + tail_workgroup;
  if (workgroup < active_workgroups) {
    let quantum_index =
      workgroup * ${ACE_FP16_VAE_POINTWISE_WORKGROUP_SIZE}u + lane;
    if (quantum_index < output_range.output_count) {
      let index = output_range.first_output + quantum_index;
      ${body}
    }
  }
}
`;
}

function planPointwise(
  shape: AceVaePointwiseShape,
  operation: "ingress" | "add",
): AceFp16VaePointwisePlan {
  const snapshot = {
    batch: shape.batch,
    frames: shape.frames,
    channels: shape.channels,
  };
  const elements = checkedAceProduct(
    [snapshot.batch, snapshot.frames, snapshot.channels],
    `ACE FP16 VAE ${operation}`,
  );
  requireWgslIndexable(elements, `${operation} elements`);
  const dispatch = planAceLinearDispatch(
    elements,
    `ACE FP16 VAE ${operation}`,
  );
  const sourceStorageBytes = checkedStorageBytes(
    elements,
    operation === "ingress" ? FLOAT32_BYTES : FLOAT16_BYTES,
    `${operation} source`,
  );
  const outputStorageBytes = checkedStorageBytes(
    elements,
    FLOAT16_BYTES,
    `${operation} output`,
  );
  return Object.freeze({
    ...snapshot,
    operation,
    elements,
    sourceStorageBytes,
    sourceBindingBytes: alignGpuBindingBytes(
      sourceStorageBytes,
      `${operation} source`,
    ),
    outputStorageBytes,
    outputBindingBytes: alignGpuBindingBytes(
      outputStorageBytes,
      `${operation} output`,
    ),
    workgroupSize: ACE_FP16_VAE_POINTWISE_WORKGROUP_SIZE,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

async function compileAceFp16VaePointwise(
  device: GPUDevice,
  plan: AceFp16VaePointwisePlan,
  code: string,
): Promise<CompiledAceFp16VaePointwise> {
  const key = pointwiseKey(plan);
  const label = `ace-fp16-vae-${key}`;
  const module = device.createShaderModule({ label, code });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `ACE FP16 VAE ${plan.operation} WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  const dataBindingSizes = plan.operation === "ingress"
    ? [plan.sourceBindingBytes, plan.outputBindingBytes]
    : [
        plan.sourceBindingBytes,
        plan.sourceBindingBytes,
        plan.outputBindingBytes,
      ];
  const outputBinding = dataBindingSizes.length - 1;
  const rangeBinding = dataBindingSizes.length;
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: [
      ...dataBindingSizes.map((minBindingSize, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: binding === outputBinding
            ? "storage" as const
            : "read-only-storage" as const,
          minBindingSize,
        },
      })),
      {
        binding: rangeBinding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform" as const,
          hasDynamicOffset: true,
          minBindingSize: OUTPUT_RANGE_CONTROL_BYTES,
        },
      },
    ],
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

function pointwiseKey(plan: AceFp16VaePointwisePlan): string {
  return [
    plan.operation,
    plan.elements,
    plan.sourceBindingBytes,
    plan.outputBindingBytes,
  ].join("x");
}

function requireKernelDevice(device: GPUDevice): void {
  if (!device.features.has("shader-f16")) {
    throw new Error("ACE FP16 VAE pointwise kernel requires WebGPU shader-f16");
  }
  const maximumInvocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const maximumSizeX = device.limits.maxComputeWorkgroupSizeX;
  if (
    !Number.isSafeInteger(maximumInvocations) ||
    !Number.isSafeInteger(maximumSizeX) ||
    maximumInvocations < ACE_FP16_VAE_POINTWISE_WORKGROUP_SIZE ||
    maximumSizeX < ACE_FP16_VAE_POINTWISE_WORKGROUP_SIZE
  ) {
    throw new Error(
      `ACE FP16 VAE pointwise kernels require a ${ACE_FP16_VAE_POINTWISE_WORKGROUP_SIZE}-lane compute workgroup`,
    );
  }
}

function requireStorageBinding(
  device: GPUDevice,
  binding: GPUBufferBinding,
  requiredStorageBytes: number,
  requiredBindingBytes: number,
  label: string,
): GPUBufferBinding {
  const alignment = device.limits.minStorageBufferOffsetAlignment;
  if (!isValidGpuAlignment(alignment)) {
    throw new Error(
      "ACE FP16 VAE pointwise device reported an invalid storage alignment",
    );
  }
  const maximumBuffer = Number(device.limits.maxBufferSize);
  const bufferBytes = Number(binding.buffer.size);
  const offset = binding.offset ?? 0;
  const available = binding.size ?? bufferBytes - offset;
  if (
    !Number.isSafeInteger(maximumBuffer) || maximumBuffer < 1 ||
    !Number.isSafeInteger(bufferBytes) || bufferBytes < 1 ||
    bufferBytes > maximumBuffer ||
    !Number.isSafeInteger(offset) || offset < 0 ||
    !Number.isSafeInteger(available) || available < requiredBindingBytes ||
    !Number.isSafeInteger(offset + available) ||
    offset + available > bufferBytes ||
    offset % alignment !== 0 ||
    available % GPU_BUFFER_ALIGNMENT !== 0 ||
    bufferBytes % GPU_BUFFER_ALIGNMENT !== 0
  ) {
    throw new RangeError(
      `${label} binding does not expose an aligned ${requiredStorageBytes}-byte storage payload in ${requiredBindingBytes} binding bytes`,
    );
  }
  return Object.freeze({
    buffer: binding.buffer,
    offset,
    size: requiredBindingBytes,
  });
}

function requireRangeControlBinding(
  device: GPUDevice,
  binding: GPUBufferBinding,
  label: string,
): GPUBufferBinding {
  const alignment = device.limits.minUniformBufferOffsetAlignment;
  if (!isValidGpuAlignment(alignment)) {
    throw new Error(
      "ACE FP16 VAE pointwise device reported an invalid uniform alignment",
    );
  }
  const maximumBuffer = Number(device.limits.maxBufferSize);
  const bufferBytes = Number(binding.buffer.size);
  const offset = binding.offset ?? 0;
  const available = binding.size ?? bufferBytes - offset;
  if (
    !Number.isSafeInteger(maximumBuffer) || maximumBuffer < 1 ||
    !Number.isSafeInteger(bufferBytes) || bufferBytes < 1 ||
    bufferBytes > maximumBuffer ||
    !Number.isSafeInteger(offset) || offset < 0 || offset > MAX_WGSL_U32 ||
    !Number.isSafeInteger(available) ||
    available < OUTPUT_RANGE_CONTROL_BYTES ||
    offset + OUTPUT_RANGE_CONTROL_BYTES > bufferBytes ||
    offset % alignment !== 0
  ) {
    throw new RangeError(
      `${label} binding must expose an aligned ${OUTPUT_RANGE_CONTROL_BYTES}-byte immutable record`,
    );
  }
  return Object.freeze({
    buffer: binding.buffer,
    offset,
    size: OUTPUT_RANGE_CONTROL_BYTES,
  });
}

function requireDisjointBindings(
  bindings: readonly NamedBinding[],
  label: string,
): void {
  for (let leftIndex = 0; leftIndex < bindings.length; leftIndex += 1) {
    const left = bindings[leftIndex]!;
    const leftStart = left.binding.offset ?? 0;
    const leftEnd = leftStart + (left.binding.size ?? 0);
    for (
      let rightIndex = leftIndex + 1;
      rightIndex < bindings.length;
      rightIndex += 1
    ) {
      const right = bindings[rightIndex]!;
      if (left.binding.buffer !== right.binding.buffer) continue;
      const rightStart = right.binding.offset ?? 0;
      const rightEnd = rightStart + (right.binding.size ?? 0);
      if (leftStart < rightEnd && rightStart < leftEnd) {
        throw new RangeError(
          `${label} ${left.name} and ${right.name} bindings must not overlap`,
        );
      }
    }
  }
}

function checkedStorageBytes(
  elements: number,
  bytesPerElement: number,
  label: string,
): number {
  const bytes = elements * bytesPerElement;
  if (!Number.isSafeInteger(bytes) || bytes < 1) {
    throw new RangeError(
      `ACE FP16 VAE pointwise ${label} bytes are not a positive safe integer`,
    );
  }
  return bytes;
}

function alignGpuBindingBytes(bytes: number, label: string): number {
  const rounded = Math.ceil(bytes / GPU_BUFFER_ALIGNMENT) *
    GPU_BUFFER_ALIGNMENT;
  if (!Number.isSafeInteger(rounded) || rounded < bytes) {
    throw new RangeError(
      `ACE FP16 VAE pointwise ${label} binding bytes are not a safe integer`,
    );
  }
  return rounded;
}

function requireWgslIndexable(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > MAX_WGSL_U32) {
    throw new RangeError(
      `ACE FP16 VAE pointwise ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
