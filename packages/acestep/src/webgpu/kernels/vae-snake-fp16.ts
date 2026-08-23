import {
  ACE_CORRECTNESS_WORKGROUP_SIZE,
  checkedAceProduct,
  planAceLinearDispatch,
} from "./correctness-utils.js";
import type {
  AceVaeOutputRangeBinding,
  AceVaePointwiseShape,
} from "./vae-primitives.js";

export const ACE_FP16_VAE_SNAKE_WORKGROUP_SIZE =
  ACE_CORRECTNESS_WORKGROUP_SIZE;
export const ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID =
  "ace-vae-fp16-portable-snake-v1" as const;

const FLOAT16_BYTES = 2;
const GPU_BUFFER_ALIGNMENT = 4;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_CONTROL_BYTES = 16;
const REQUIRED_STORAGE_BUFFERS = 4;

export interface AceFp16VaeSnakeBindings {
  /** FP16 activation in frame-major NLC order. */
  readonly input: GPUBufferBinding;
  /** FP16 flattened log-scale parameter `[channels]`. */
  readonly alpha: GPUBufferBinding;
  /** FP16 flattened log-scale parameter `[channels]`. */
  readonly beta: GPUBufferBinding;
  /** FP16 activation in frame-major NLC order. */
  readonly output: GPUBufferBinding;
}

export interface AceFp16VaeSnakePlan extends AceVaePointwiseShape {
  readonly elements: number;
  readonly inputStorageBytes: number;
  readonly inputBindingBytes: number;
  readonly alphaStorageBytes: number;
  readonly alphaBindingBytes: number;
  readonly betaStorageBytes: number;
  readonly betaBindingBytes: number;
  readonly outputStorageBytes: number;
  readonly outputBindingBytes: number;
  readonly workgroupSize: typeof ACE_FP16_VAE_SNAKE_WORKGROUP_SIZE;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceFp16VaeSnakeRangePlan {
  readonly base: number;
  readonly count: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceFp16VaeSnakeDispatch {
  readonly label: string;
  readonly kernelId: typeof ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID;
  readonly plan: AceFp16VaeSnakePlan;
  readonly outputRange: AceFp16VaeSnakeRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledAceFp16VaeSnake {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

interface NamedBinding {
  readonly name: "input" | "alpha" | "beta" | "output" | "range control";
  readonly binding: GPUBufferBinding;
}

/**
 * Portable production `shader-f16` Snake for the OPT-0011 VAE profile.
 *
 * Input and learned log scales are expanded from FP16 storage into explicit
 * FP32 registers. The two exponentials, sine, epsilon denominator, reciprocal,
 * and residual expression retain the registered source order, followed by one
 * round-to-nearest-even FP16 store at the internal activation boundary.
 */
export class AceFp16VaeSnakeKernel {
  private readonly pipelines = new Map<
    string,
    Promise<CompiledAceFp16VaeSnake>
  >();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {
    requireKernelDevice(device);
  }

  static create(device: GPUDevice): AceFp16VaeSnakeKernel {
    return new AceFp16VaeSnakeKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaePointwiseShape,
    bindings: AceFp16VaeSnakeBindings,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceFp16VaeSnakeDispatch> {
    this.requireLive();
    const plan = planAceFp16VaeSnake(shape);
    const outputRange = planAceFp16VaeSnakeRange(plan, range);
    this.requireDeviceLimits(plan, outputRange);
    const normalized = this.requireBindings(label, plan, bindings, range);
    const compiled = await this.pipelineFor(plan);
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
    const key = snakeKey(plan);
    const bindGroupKey = `${key}:${bindGroupResources.map((binding) =>
      this.bindingKey(binding)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-fp16-snake-bindings`,
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
      kernelId: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
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

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.bindGroups.clear();
    this.pipelines.clear();
  }

  private pipelineFor(
    plan: AceFp16VaeSnakePlan,
  ): Promise<CompiledAceFp16VaeSnake> {
    const key = snakeKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceFp16VaeSnake(this.device, plan);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
  }

  private requireBindings(
    label: string,
    plan: AceFp16VaeSnakePlan,
    bindings: AceFp16VaeSnakeBindings,
    range: AceVaeOutputRangeBinding,
  ): readonly NamedBinding[] {
    const normalized = Object.freeze([
      {
        name: "input" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.input,
          plan.inputStorageBytes,
          plan.inputBindingBytes,
          `${label} input`,
        ),
      },
      {
        name: "alpha" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.alpha,
          plan.alphaStorageBytes,
          plan.alphaBindingBytes,
          `${label} alpha`,
        ),
      },
      {
        name: "beta" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.beta,
          plan.betaStorageBytes,
          plan.betaBindingBytes,
          `${label} beta`,
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
    return normalized;
  }

  private requireDeviceLimits(
    plan: AceFp16VaeSnakePlan,
    range: AceFp16VaeSnakeRangePlan,
  ): void {
    const maximumDispatch = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (
      !Number.isSafeInteger(maximumDispatch) || maximumDispatch < 1 ||
      range.workgroupsX > maximumDispatch ||
      range.workgroupsY > maximumDispatch
    ) {
      throw new RangeError(
        "ACE FP16 VAE Snake range exceeds the device dispatch dimension",
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
        "ACE FP16 VAE Snake device reported invalid buffer limits",
      );
    }
    for (const [name, bytes] of [
      ["input", plan.inputBindingBytes],
      ["alpha", plan.alphaBindingBytes],
      ["beta", plan.betaBindingBytes],
      ["output", plan.outputBindingBytes],
    ] as const) {
      if (bytes > maximumStorage) {
        throw new RangeError(
          `ACE FP16 VAE Snake ${name} exceeds the device storage binding limit`,
        );
      }
      if (bytes > maximumBuffer) {
        throw new RangeError(
          `ACE FP16 VAE Snake ${name} exceeds the device buffer limit`,
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
      throw new Error("ACE FP16 VAE Snake kernel was destroyed");
    }
  }
}

export function planAceFp16VaeSnake(
  shape: AceVaePointwiseShape,
): AceFp16VaeSnakePlan {
  const snapshot = {
    batch: shape.batch,
    frames: shape.frames,
    channels: shape.channels,
  };
  const elements = checkedAceProduct(
    [snapshot.batch, snapshot.frames, snapshot.channels],
    "ACE FP16 VAE Snake",
  );
  requireWgslIndexable(elements, "elements");
  const dispatch = planAceLinearDispatch(elements, "ACE FP16 VAE Snake");
  const inputStorageBytes = checkedStorageBytes(
    elements,
    FLOAT16_BYTES,
    "input",
  );
  const parameterStorageBytes = checkedStorageBytes(
    snapshot.channels,
    FLOAT16_BYTES,
    "channel parameter",
  );
  const outputStorageBytes = checkedStorageBytes(
    elements,
    FLOAT16_BYTES,
    "output",
  );
  const parameterBindingBytes = alignGpuBindingBytes(
    parameterStorageBytes,
    "channel parameter",
  );
  return Object.freeze({
    ...snapshot,
    elements,
    inputStorageBytes,
    inputBindingBytes: alignGpuBindingBytes(inputStorageBytes, "input"),
    alphaStorageBytes: parameterStorageBytes,
    alphaBindingBytes: parameterBindingBytes,
    betaStorageBytes: parameterStorageBytes,
    betaBindingBytes: parameterBindingBytes,
    outputStorageBytes,
    outputBindingBytes: alignGpuBindingBytes(outputStorageBytes, "output"),
    workgroupSize: ACE_FP16_VAE_SNAKE_WORKGROUP_SIZE,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

export function planAceFp16VaeSnakeRange(
  plan: AceFp16VaeSnakePlan,
  range: Readonly<{ base: number; count: number }>,
): AceFp16VaeSnakeRangePlan {
  if (
    !Number.isSafeInteger(range.base) || range.base < 0 ||
    !Number.isSafeInteger(range.count) || range.count <= 0 ||
    range.base > MAX_WGSL_U32 || range.count > MAX_WGSL_U32 ||
    !Number.isSafeInteger(range.base + range.count) ||
    range.base + range.count > plan.elements
  ) {
    throw new RangeError(
      "ACE FP16 VAE Snake range must be non-empty and inside its complete output domain",
    );
  }
  const dispatch = planAceLinearDispatch(
    range.count,
    "ACE FP16 VAE Snake range",
  );
  return Object.freeze({
    base: range.base,
    count: range.count,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

export function aceFp16VaeSnakeWgsl(
  shape: AceVaePointwiseShape,
): string {
  const plan = planAceFp16VaeSnake(shape);
  return /* wgsl */ `
// kernel-id: ${ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID}
enable f16;

const CHANNELS: u32 = ${plan.channels}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> alpha: array<f16>;
@group(0) @binding(2) var<storage, read> beta: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

struct OutputRangeParameters {
  first_output: u32,
  output_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(4) var<uniform>
  output_range: OutputRangeParameters;

fn quantum_workgroup_index(
  workgroup_id: vec3<u32>,
  num_workgroups: vec3<u32>,
) -> u32 {
  return workgroup_id.y * num_workgroups.x + workgroup_id.x;
}

@compute @workgroup_size(${ACE_FP16_VAE_SNAKE_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let workgroup = quantum_workgroup_index(group, num_workgroups);
  let full_workgroups = output_range.output_count /
    ${ACE_FP16_VAE_SNAKE_WORKGROUP_SIZE}u;
  let tail_workgroup = select(
    0u,
    1u,
    (output_range.output_count % ${ACE_FP16_VAE_SNAKE_WORKGROUP_SIZE}u) != 0u,
  );
  let active_workgroups = full_workgroups + tail_workgroup;
  if (workgroup < active_workgroups) {
    let quantum_index =
      workgroup * ${ACE_FP16_VAE_SNAKE_WORKGROUP_SIZE}u + lane;
    if (quantum_index < output_range.output_count) {
      let index = output_range.first_output + quantum_index;
      let channel = index % CHANNELS;
      let value: f32 = f32(input[index]);
      let alpha_log_scale: f32 = f32(alpha[channel]);
      let beta_log_scale: f32 = f32(beta[channel]);
      let alpha_value: f32 = exp(alpha_log_scale);
      let beta_value: f32 = exp(beta_log_scale);
      let periodic: f32 = sin(alpha_value * value);
      let reciprocal_beta: f32 = 1.0 / (beta_value + 1e-9);
      let result: f32 =
        value + reciprocal_beta * periodic * periodic;
      output[index] = f16(result);
    }
  }
}
`;
}

async function compileAceFp16VaeSnake(
  device: GPUDevice,
  plan: AceFp16VaeSnakePlan,
): Promise<CompiledAceFp16VaeSnake> {
  const key = snakeKey(plan);
  const label = `ace-fp16-vae-${key}`;
  const module = device.createShaderModule({
    label,
    code: aceFp16VaeSnakeWgsl(plan),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `ACE FP16 VAE Snake WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  const bindingSizes = [
    plan.inputBindingBytes,
    plan.alphaBindingBytes,
    plan.betaBindingBytes,
    plan.outputBindingBytes,
  ];
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: [
      ...bindingSizes.map((minBindingSize, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: binding === 3
            ? "storage" as const
            : "read-only-storage" as const,
          minBindingSize,
        },
      })),
      {
        binding: 4,
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

function snakeKey(plan: AceFp16VaeSnakePlan): string {
  return [
    "snake",
    plan.channels,
    plan.elements,
    plan.inputBindingBytes,
    plan.alphaBindingBytes,
    plan.outputBindingBytes,
  ].join("x");
}

function requireKernelDevice(device: GPUDevice): void {
  if (!device.features.has("shader-f16")) {
    throw new Error("ACE FP16 VAE Snake kernel requires WebGPU shader-f16");
  }
  const maximumInvocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const maximumSizeX = device.limits.maxComputeWorkgroupSizeX;
  if (
    !Number.isSafeInteger(maximumInvocations) ||
    !Number.isSafeInteger(maximumSizeX) ||
    maximumInvocations < ACE_FP16_VAE_SNAKE_WORKGROUP_SIZE ||
    maximumSizeX < ACE_FP16_VAE_SNAKE_WORKGROUP_SIZE
  ) {
    throw new Error(
      `ACE FP16 VAE Snake requires a ${ACE_FP16_VAE_SNAKE_WORKGROUP_SIZE}-lane compute workgroup`,
    );
  }
  for (const [name, value, minimum] of [
    ["maxBindGroups", device.limits.maxBindGroups, 1],
    ["maxBindingsPerBindGroup", device.limits.maxBindingsPerBindGroup, 5],
    [
      "maxStorageBuffersPerShaderStage",
      device.limits.maxStorageBuffersPerShaderStage,
      REQUIRED_STORAGE_BUFFERS,
    ],
    ["maxUniformBuffersPerShaderStage", device.limits.maxUniformBuffersPerShaderStage, 1],
    [
      "maxDynamicUniformBuffersPerPipelineLayout",
      device.limits.maxDynamicUniformBuffersPerPipelineLayout,
      1,
    ],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < minimum) {
      throw new Error(
        `ACE FP16 VAE Snake requires ${name} >= ${minimum}`,
      );
    }
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
      "ACE FP16 VAE Snake device reported an invalid storage alignment",
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
      "ACE FP16 VAE Snake device reported an invalid uniform alignment",
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
      `ACE FP16 VAE Snake ${label} bytes are not a positive safe integer`,
    );
  }
  return bytes;
}

function alignGpuBindingBytes(bytes: number, label: string): number {
  const rounded = Math.ceil(bytes / GPU_BUFFER_ALIGNMENT) *
    GPU_BUFFER_ALIGNMENT;
  if (!Number.isSafeInteger(rounded) || rounded < bytes) {
    throw new RangeError(
      `ACE FP16 VAE Snake ${label} binding bytes are not a safe integer`,
    );
  }
  return rounded;
}

function requireWgslIndexable(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > MAX_WGSL_U32) {
    throw new RangeError(
      `ACE FP16 VAE Snake ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
