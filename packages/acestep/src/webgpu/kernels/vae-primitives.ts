import {
  ACE_CORRECTNESS_WORKGROUP_SIZE,
  aceLinearInvocationWgsl,
  checkedAceProduct,
  planAceLinearDispatch,
  requireAceBindingBytes,
  requireAceDisjointOutput,
  requirePositiveSafeInteger,
} from "./correctness-utils.js";

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;

export interface AceVaeConv1dShape {
  readonly batch: number;
  readonly inputFrames: number;
  readonly inputChannels: number;
  readonly outputChannels: number;
  readonly kernelSize: number;
  readonly stride: number;
  readonly dilation: number;
  readonly padding: number;
}

export interface AceVaeConvTranspose1dShape extends AceVaeConv1dShape {
  readonly outputPadding: number;
}

/** One output-axis package part of a logical ConvTranspose1d weight. */
export interface AceVaeConvTranspose1dPartShape
  extends Omit<AceVaeConvTranspose1dShape, "outputChannels"> {
  /** Complete logical output width, before output-axis row sharding. */
  readonly outputChannels: number;
  readonly firstOutputChannel: number;
  readonly partOutputChannels: number;
}

export interface AceVaePointwiseShape {
  readonly batch: number;
  readonly frames: number;
  readonly channels: number;
}

export interface AceVaeConvPlan {
  readonly batch: number;
  readonly inputFrames: number;
  readonly outputFrames: number;
  readonly inputChannels: number;
  readonly outputChannels: number;
  readonly kernelSize: number;
  readonly stride: number;
  readonly dilation: number;
  readonly padding: number;
  readonly outputPadding: number;
  readonly inputElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceVaePointwisePlan extends AceVaePointwiseShape {
  readonly elements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

/** A non-empty, half-open subrange of one primitive's output domain. */
export interface AceVaeOutputRange {
  readonly base: number;
  readonly count: number;
}

/**
 * Immutable GPU control for a ranged dispatch. The first two U32 words of the
 * binding must equal `base,count`; the remaining two words are padding.
 */
export interface AceVaeOutputRangeBinding extends AceVaeOutputRange {
  readonly control: GPUBufferBinding;
}

export interface AceVaeConvBindings {
  /** FP32 activation in `[batch,frames,inputChannels]` (NLC) order. */
  readonly input: GPUBufferBinding;
  /** FP32 package weight in converter-native `[out,kernel,in]` order. */
  readonly weight: GPUBufferBinding;
  /** FP32 `[out]`. Omitted only for `decoder.conv2`. */
  readonly bias?: GPUBufferBinding;
  /** FP32 activation in `[batch,frames,outputChannels]` (NLC) order. */
  readonly output: GPUBufferBinding;
}

export interface AceVaeSnakeBindings {
  readonly input: GPUBufferBinding;
  /** FP32 flattened source parameter `[channels]`. */
  readonly alpha: GPUBufferBinding;
  /** FP32 flattened source parameter `[channels]`. */
  readonly beta: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceVaeAddBindings {
  readonly left: GPUBufferBinding;
  readonly right: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceVaeDispatch<Plan> {
  readonly label: string;
  readonly plan: Plan;
  /** Complete-domain range, or part-local range for a transpose weight part. */
  readonly outputRange: AceVaeOutputRange;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledVaePrimitive {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Scalar, unfused FP32 Oobleck primitives for Stage 1 correctness.
 *
 * The VAE package is deliberately FP32 in both model profiles. One invocation
 * owns one output scalar and performs its reduction in source order. These are
 * conventional oracle kernels, not the eventual performance implementation.
 */
export class AceCorrectnessVaePrimitiveKernel {
  private readonly pipelines = new Map<string, Promise<CompiledVaePrimitive>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(device: GPUDevice): AceCorrectnessVaePrimitiveKernel {
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_CORRECTNESS_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX < ACE_CORRECTNESS_WORKGROUP_SIZE
    ) {
      throw new Error(
        `ACE VAE correctness primitives require ${ACE_CORRECTNESS_WORKGROUP_SIZE} compute lanes`,
      );
    }
    return new AceCorrectnessVaePrimitiveKernel(device);
  }

  async createConv1dDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceVaeConvBindings,
    range?: AceVaeOutputRangeBinding,
  ): Promise<AceVaeDispatch<AceVaeConvPlan>> {
    this.requireLive();
    const plan = planAceVaeConv1d(shape);
    requireConvBindings(label, plan, bindings);
    return await this.createDispatch(
      label,
      `conv:${convKey(plan)}:${bindings.bias === undefined ? 0 : 1}:${range === undefined ? "full" : "ranged"}`,
      plan,
      plan.outputElements,
      aceCorrectnessVaeConv1dWgsl(
        shape,
        bindings.bias !== undefined,
        range !== undefined,
      ),
      appendRangeBinding(bindings.bias === undefined
        ? [bindings.input, bindings.weight, bindings.output]
        : [bindings.input, bindings.weight, bindings.bias, bindings.output], range),
      range,
    );
  }

  async createConvTranspose1dDispatch(
    label: string,
    shape: AceVaeConvTranspose1dShape,
    bindings: AceVaeConvBindings,
    range?: AceVaeOutputRangeBinding,
  ): Promise<AceVaeDispatch<AceVaeConvPlan>> {
    this.requireLive();
    const plan = planAceVaeConvTranspose1d(shape);
    requireConvBindings(label, plan, bindings);
    return await this.createDispatch(
      label,
      `transpose:${convKey(plan)}:${bindings.bias === undefined ? 0 : 1}:${range === undefined ? "full" : "ranged"}`,
      plan,
      plan.outputElements,
      aceCorrectnessVaeConvTranspose1dWgsl(
        shape,
        bindings.bias !== undefined,
        range !== undefined,
      ),
      appendRangeBinding(bindings.bias === undefined
        ? [bindings.input, bindings.weight, bindings.output]
        : [bindings.input, bindings.weight, bindings.bias, bindings.output], range),
      range,
    );
  }

  async createConvTranspose1dPartDispatch(
    label: string,
    shape: AceVaeConvTranspose1dPartShape,
    bindings: AceVaeConvBindings,
    range?: AceVaeOutputRangeBinding,
  ): Promise<AceVaeDispatch<AceVaeConvPlan>> {
    this.requireLive();
    validateTransposePart(shape);
    const fullPlan = planAceVaeConvTranspose1d(shape);
    const partOutputElements = checkedAceProduct(
      [shape.batch, fullPlan.outputFrames, shape.partOutputChannels],
      `${label} output part`,
    );
    const partDispatch = planAceLinearDispatch(
      partOutputElements,
      `${label} output part`,
    );
    const partPlan: AceVaeConvPlan = Object.freeze({
      ...fullPlan,
      weightElements: checkedAceProduct(
        [shape.partOutputChannels, shape.kernelSize, shape.inputChannels],
        `${label} weight part`,
      ),
      // This is the invocation domain; bindings still expose full output.
      outputElements: partOutputElements,
      workgroupsX: partDispatch.workgroupsX,
      workgroupsY: partDispatch.workgroupsY,
    });
    requireAceBindingBytes(
      bindings.input,
      fullPlan.inputElements * FLOAT32_BYTES,
      `${label} input`,
    );
    requireAceBindingBytes(
      bindings.weight,
      partPlan.weightElements * FLOAT32_BYTES,
      `${label} weight`,
    );
    if (bindings.bias !== undefined) {
      requireAceBindingBytes(
        bindings.bias,
        shape.outputChannels * FLOAT32_BYTES,
        `${label} complete bias`,
      );
    }
    requireAceBindingBytes(
      bindings.output,
      fullPlan.outputElements * FLOAT32_BYTES,
      `${label} complete output`,
    );
    requireAceDisjointOutput(
      bindings.output,
      [
        bindings.input,
        bindings.weight,
        ...(bindings.bias === undefined ? [] : [bindings.bias]),
      ],
      label,
    );
    const dispatch = await this.createDispatch(
      label,
      `transpose-part:${convKey(fullPlan)}:${shape.firstOutputChannel}x${shape.partOutputChannels}:${bindings.bias === undefined ? 0 : 1}:${range === undefined ? "full" : "ranged"}`,
      partPlan,
      partOutputElements,
      aceCorrectnessVaeConvTranspose1dPartWgsl(
        shape,
        bindings.bias !== undefined,
        range !== undefined,
      ),
      appendRangeBinding(bindings.bias === undefined
        ? [bindings.input, bindings.weight, bindings.output]
        : [bindings.input, bindings.weight, bindings.bias, bindings.output], range),
      range,
    );
    // Expose the complete logical output geometry while preserving the part's
    // physical invocation domain inside the encoded closure.
    return Object.freeze({
      label: dispatch.label,
      plan: fullPlan,
      outputRange: dispatch.outputRange,
      encode(pass: GPUComputePassEncoder): void {
        dispatch.encode(pass);
      },
    });
  }

  async createSnakeDispatch(
    label: string,
    shape: AceVaePointwiseShape,
    bindings: AceVaeSnakeBindings,
    range?: AceVaeOutputRangeBinding,
  ): Promise<AceVaeDispatch<AceVaePointwisePlan>> {
    this.requireLive();
    const plan = planAceVaePointwise(shape, "Snake");
    requireAceBindingBytes(
      bindings.input,
      plan.elements * FLOAT32_BYTES,
      `${label} input`,
    );
    requireAceBindingBytes(
      bindings.alpha,
      plan.channels * FLOAT32_BYTES,
      `${label} alpha`,
    );
    requireAceBindingBytes(
      bindings.beta,
      plan.channels * FLOAT32_BYTES,
      `${label} beta`,
    );
    requireAceBindingBytes(
      bindings.output,
      plan.elements * FLOAT32_BYTES,
      `${label} output`,
    );
    requireAceDisjointOutput(
      bindings.output,
      [bindings.input, bindings.alpha, bindings.beta],
      label,
    );
    return await this.createDispatch(
      label,
      `snake:${shape.batch}x${shape.frames}x${shape.channels}:${range === undefined ? "full" : "ranged"}`,
      plan,
      plan.elements,
      aceCorrectnessVaeSnakeWgsl(shape, range !== undefined),
      appendRangeBinding(
        [bindings.input, bindings.alpha, bindings.beta, bindings.output],
        range,
      ),
      range,
    );
  }

  async createAddDispatch(
    label: string,
    shape: AceVaePointwiseShape,
    bindings: AceVaeAddBindings,
    range?: AceVaeOutputRangeBinding,
  ): Promise<AceVaeDispatch<AceVaePointwisePlan>> {
    this.requireLive();
    const plan = planAceVaePointwise(shape, "residual add");
    const bytes = plan.elements * FLOAT32_BYTES;
    requireAceBindingBytes(bindings.left, bytes, `${label} left`);
    requireAceBindingBytes(bindings.right, bytes, `${label} right`);
    requireAceBindingBytes(bindings.output, bytes, `${label} output`);
    requireAceDisjointOutput(
      bindings.output,
      [bindings.left, bindings.right],
      label,
    );
    return await this.createDispatch(
      label,
      `add:${shape.batch}x${shape.frames}x${shape.channels}:${range === undefined ? "full" : "ranged"}`,
      plan,
      plan.elements,
      aceCorrectnessVaeAddWgsl(shape, range !== undefined),
      appendRangeBinding([bindings.left, bindings.right, bindings.output], range),
      range,
    );
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.bindGroups.clear();
    this.pipelines.clear();
  }

  private async createDispatch<Plan extends {
    readonly workgroupsX: number;
    readonly workgroupsY: number;
  }>(
    label: string,
    key: string,
    plan: Plan,
    outputElements: number,
    code: string,
    bindings: readonly GPUBufferBinding[],
    requestedRange?: AceVaeOutputRangeBinding,
  ): Promise<AceVaeDispatch<Plan>> {
    const outputRange = resolveOutputRange(
      requestedRange,
      outputElements,
      `${label} output`,
      this.device.limits.minUniformBufferOffsetAlignment,
    );
    const dispatchPlan = requestedRange === undefined
      ? plan
      : planAceLinearDispatch(outputRange.count, `${label} quantum`);
    if (requestedRange !== undefined) {
      requireAceDisjointOutput(
        bindings[bindings.length - 2]!,
        [requestedRange.control],
        `${label} range control`,
      );
    }
    let compiledPromise = this.pipelines.get(key);
    if (compiledPromise === undefined) {
      compiledPromise = compilePrimitive(
        this.device,
        `ace-vae-${key}`,
        code,
        requestedRange === undefined ? undefined : bindings.length,
      );
      this.pipelines.set(key, compiledPromise);
      void compiledPromise.catch(() => {
        if (this.pipelines.get(key) === compiledPromise) {
          this.pipelines.delete(key);
        }
      });
    }
    const compiled = await compiledPromise;
    this.requireLive();
    const bindGroupBindings = requestedRange === undefined
      ? bindings
      : bindings.map((binding, index) =>
        index === bindings.length - 1
          ? Object.freeze({ buffer: binding.buffer, offset: 0, size: 16 })
          : binding
      );
    const bindGroupKey = `${key}:${bindGroupBindings.map((binding) =>
      this.bindingKey(binding)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-bindings`,
        layout: compiled.bindGroupLayout,
        entries: bindGroupBindings.map((resource, index) => ({
          binding: index,
          resource,
        })),
      });
      this.bindGroups.set(bindGroupKey, bindGroup);
    }
    const dynamicOffsets = requestedRange === undefined
      ? undefined
      : [requestedRange.control.offset ?? 0];
    return Object.freeze({
      label,
      plan,
      outputRange,
      encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(compiled.pipeline);
        if (dynamicOffsets === undefined) {
          pass.setBindGroup(0, bindGroup);
        } else {
          pass.setBindGroup(0, bindGroup, dynamicOffsets);
        }
        pass.dispatchWorkgroups(
          dispatchPlan.workgroupsX,
          dispatchPlan.workgroupsY,
          1,
        );
      },
    });
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("ACE VAE correctness primitive kernel was destroyed");
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
}

export function planAceVaeConv1d(shape: AceVaeConv1dShape): AceVaeConvPlan {
  validateConvShape(shape, "Conv1d");
  const effectiveKernel = shape.dilation * (shape.kernelSize - 1) + 1;
  const numerator = shape.inputFrames + 2 * shape.padding - effectiveKernel;
  if (numerator < 0) {
    throw new RangeError("ACE VAE Conv1d kernel exceeds its padded input");
  }
  const outputFrames = Math.floor(numerator / shape.stride) + 1;
  return makeConvPlan(shape, outputFrames, 0);
}

export function planAceVaeConvTranspose1d(
  shape: AceVaeConvTranspose1dShape,
): AceVaeConvPlan {
  validateConvShape(shape, "ConvTranspose1d");
  if (
    !Number.isSafeInteger(shape.outputPadding) ||
    shape.outputPadding < 0 ||
    shape.outputPadding >= shape.stride
  ) {
    throw new RangeError(
      "ACE VAE ConvTranspose1d output padding must be an integer below stride",
    );
  }
  const outputFrames =
    (shape.inputFrames - 1) * shape.stride -
    2 * shape.padding +
    shape.dilation * (shape.kernelSize - 1) +
    shape.outputPadding +
    1;
  if (!Number.isSafeInteger(outputFrames) || outputFrames <= 0) {
    throw new RangeError("ACE VAE ConvTranspose1d has no output frames");
  }
  return makeConvPlan(shape, outputFrames, shape.outputPadding);
}

export function planAceVaePointwise(
  shape: AceVaePointwiseShape,
  operation = "pointwise",
): AceVaePointwisePlan {
  const elements = checkedAceProduct(
    [shape.batch, shape.frames, shape.channels],
    `ACE VAE ${operation}`,
  );
  const dispatch = planAceLinearDispatch(elements, `ACE VAE ${operation}`);
  return Object.freeze({ ...shape, ...dispatch });
}

export function aceCorrectnessVaeConv1dWgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  ranged = false,
): string {
  const plan = planAceVaeConv1d(shape);
  return convPrelude(plan, hasBias, ranged) + /* wgsl */ `
@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
  ${vaeNumWorkgroupsParameterWgsl(ranged)}
) {
  ${vaeOutputIndexWgsl("output_index", ranged)}
  let output_channel = output_index % OUTPUT_CHANNELS;
  let row = output_index / OUTPUT_CHANNELS;
  let output_time = row % OUTPUT_FRAMES;
  let batch = row / OUTPUT_FRAMES;
  var sum = ${hasBias ? "bias[output_channel]" : "0.0"};
  for (var kernel = 0u; kernel < KERNEL_SIZE; kernel += 1u) {
    let padded_time = output_time * STRIDE + kernel * DILATION;
    if (padded_time < PADDING) { continue; }
    let input_time = padded_time - PADDING;
    if (input_time >= INPUT_FRAMES) { continue; }
    let input_base = (batch * INPUT_FRAMES + input_time) * INPUT_CHANNELS;
    let weight_base = (output_channel * KERNEL_SIZE + kernel) * INPUT_CHANNELS;
    for (var input_channel = 0u; input_channel < INPUT_CHANNELS; input_channel += 1u) {
      sum = sum + input[input_base + input_channel] *
        weight[weight_base + input_channel];
    }
  }
  output[output_index] = sum;
}
`;
}

export function aceCorrectnessVaeConvTranspose1dWgsl(
  shape: AceVaeConvTranspose1dShape,
  hasBias: boolean,
  ranged = false,
): string {
  const plan = planAceVaeConvTranspose1d(shape);
  return convPrelude(plan, hasBias, ranged) + /* wgsl */ `
@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
  ${vaeNumWorkgroupsParameterWgsl(ranged)}
) {
  ${vaeOutputIndexWgsl("output_index", ranged)}
  let output_channel = output_index % OUTPUT_CHANNELS;
  let row = output_index / OUTPUT_CHANNELS;
  let output_time = row % OUTPUT_FRAMES;
  let batch = row / OUTPUT_FRAMES;
  var sum = ${hasBias ? "bias[output_channel]" : "0.0"};
  for (var kernel = 0u; kernel < KERNEL_SIZE; kernel += 1u) {
    let kernel_time = kernel * DILATION;
    let padded_output_time = output_time + PADDING;
    if (padded_output_time < kernel_time) { continue; }
    let input_numerator = padded_output_time - kernel_time;
    if ((input_numerator % STRIDE) != 0u) { continue; }
    let input_time = input_numerator / STRIDE;
    if (input_time >= INPUT_FRAMES) { continue; }
    let input_base = (batch * INPUT_FRAMES + input_time) * INPUT_CHANNELS;
    let weight_base = (output_channel * KERNEL_SIZE + kernel) * INPUT_CHANNELS;
    for (var input_channel = 0u; input_channel < INPUT_CHANNELS; input_channel += 1u) {
      sum = sum + input[input_base + input_channel] *
        weight[weight_base + input_channel];
    }
  }
  output[output_index] = sum;
}
`;
}

export function aceCorrectnessVaeConvTranspose1dPartWgsl(
  shape: AceVaeConvTranspose1dPartShape,
  hasBias: boolean,
  ranged = false,
): string {
  validateTransposePart(shape);
  const fullPlan = planAceVaeConvTranspose1d(shape);
  const partElements = checkedAceProduct(
    [shape.batch, fullPlan.outputFrames, shape.partOutputChannels],
    "ACE VAE ConvTranspose1d part output",
  );
  const dispatch = planAceLinearDispatch(
    partElements,
    "ACE VAE ConvTranspose1d part",
  );
  return /* wgsl */ `
const INPUT_FRAMES: u32 = ${fullPlan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${fullPlan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${fullPlan.inputChannels}u;
const TOTAL_OUTPUT_CHANNELS: u32 = ${fullPlan.outputChannels}u;
const FIRST_OUTPUT_CHANNEL: u32 = ${shape.firstOutputChannel}u;
const PART_OUTPUT_CHANNELS: u32 = ${shape.partOutputChannels}u;
const OUTPUT_ELEMENTS: u32 = ${partElements}u;
const KERNEL_SIZE: u32 = ${fullPlan.kernelSize}u;
const STRIDE: u32 = ${fullPlan.stride}u;
const DILATION: u32 = ${fullPlan.dilation}u;
const PADDING: u32 = ${fullPlan.padding}u;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
${vaeConvOutputBindingsWgsl(hasBias)}

${vaeInvocationWgsl(
  partElements,
  dispatch.workgroupsX,
  ranged,
  hasBias ? 4 : 3,
)}

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
  ${vaeNumWorkgroupsParameterWgsl(ranged)}
) {
  ${vaeOutputIndexWgsl("part_index", ranged)}
  let local_output_channel = part_index % PART_OUTPUT_CHANNELS;
  let row = part_index / PART_OUTPUT_CHANNELS;
  let output_time = row % OUTPUT_FRAMES;
  let batch = row / OUTPUT_FRAMES;
  let output_channel = FIRST_OUTPUT_CHANNEL + local_output_channel;
  var sum = ${hasBias ? "bias[output_channel]" : "0.0"};
  for (var kernel = 0u; kernel < KERNEL_SIZE; kernel += 1u) {
    let kernel_time = kernel * DILATION;
    let padded_output_time = output_time + PADDING;
    if (padded_output_time < kernel_time) { continue; }
    let input_numerator = padded_output_time - kernel_time;
    if ((input_numerator % STRIDE) != 0u) { continue; }
    let input_time = input_numerator / STRIDE;
    if (input_time >= INPUT_FRAMES) { continue; }
    let input_base = (batch * INPUT_FRAMES + input_time) * INPUT_CHANNELS;
    let weight_base =
      (local_output_channel * KERNEL_SIZE + kernel) * INPUT_CHANNELS;
    for (var input_channel = 0u; input_channel < INPUT_CHANNELS; input_channel += 1u) {
      sum = sum + input[input_base + input_channel] *
        weight[weight_base + input_channel];
    }
  }
  let output_index =
    (batch * OUTPUT_FRAMES + output_time) * TOTAL_OUTPUT_CHANNELS + output_channel;
  output[output_index] = sum;
}
`;
}

export function aceCorrectnessVaeSnakeWgsl(
  shape: AceVaePointwiseShape,
  ranged = false,
): string {
  const plan = planAceVaePointwise(shape, "Snake");
  return /* wgsl */ `
const CHANNELS: u32 = ${plan.channels}u;
const OUTPUT_ELEMENTS: u32 = ${plan.elements}u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

${vaeInvocationWgsl(plan.elements, plan.workgroupsX, ranged, 4)}

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
  ${vaeNumWorkgroupsParameterWgsl(ranged)}
) {
  ${vaeOutputIndexWgsl("index", ranged)}
  let channel = index % CHANNELS;
  let value = input[index];
  let alpha_value = exp(alpha[channel]);
  let beta_value = exp(beta[channel]);
  let periodic = sin(alpha_value * value);
  let reciprocal_beta = 1.0 / (beta_value + 1e-9);
  output[index] = value + reciprocal_beta * periodic * periodic;
}
`;
}

export function aceCorrectnessVaeAddWgsl(
  shape: AceVaePointwiseShape,
  ranged = false,
): string {
  const plan = planAceVaePointwise(shape, "residual add");
  return /* wgsl */ `
const OUTPUT_ELEMENTS: u32 = ${plan.elements}u;

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

${vaeInvocationWgsl(plan.elements, plan.workgroupsX, ranged, 3)}

@compute @workgroup_size(${ACE_CORRECTNESS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
  ${vaeNumWorkgroupsParameterWgsl(ranged)}
) {
  ${vaeOutputIndexWgsl("index", ranged)}
  output[index] = left[index] + right[index];
}
`;
}

function makeConvPlan(
  shape: AceVaeConv1dShape,
  outputFrames: number,
  outputPadding: number,
): AceVaeConvPlan {
  const inputElements = checkedAceProduct(
    [shape.batch, shape.inputFrames, shape.inputChannels],
    "ACE VAE convolution input",
  );
  const weightElements = checkedAceProduct(
    [shape.outputChannels, shape.kernelSize, shape.inputChannels],
    "ACE VAE convolution weight",
  );
  const outputElements = checkedAceProduct(
    [shape.batch, outputFrames, shape.outputChannels],
    "ACE VAE convolution output",
  );
  const dispatch = planAceLinearDispatch(
    outputElements,
    "ACE VAE convolution",
  );
  return Object.freeze({
    ...shape,
    outputFrames,
    outputPadding,
    inputElements,
    weightElements,
    outputElements,
    workgroupsX: dispatch.workgroupsX,
    workgroupsY: dispatch.workgroupsY,
  });
}

function validateConvShape(
  shape: AceVaeConv1dShape,
  operation: string,
): void {
  for (const [name, value] of Object.entries({
    batch: shape.batch,
    inputFrames: shape.inputFrames,
    inputChannels: shape.inputChannels,
    outputChannels: shape.outputChannels,
    kernelSize: shape.kernelSize,
    stride: shape.stride,
    dilation: shape.dilation,
  })) {
    requirePositiveSafeInteger(value, `ACE VAE ${operation} ${name}`);
  }
  if (!Number.isSafeInteger(shape.padding) || shape.padding < 0) {
    throw new RangeError(
      `ACE VAE ${operation} padding must be a non-negative safe integer`,
    );
  }
}

function validateTransposePart(shape: AceVaeConvTranspose1dPartShape): void {
  planAceVaeConvTranspose1d(shape);
  if (
    !Number.isSafeInteger(shape.firstOutputChannel) ||
    shape.firstOutputChannel < 0
  ) {
    throw new RangeError(
      "ACE VAE ConvTranspose1d first output channel must be non-negative",
    );
  }
  requirePositiveSafeInteger(
    shape.partOutputChannels,
    "ACE VAE ConvTranspose1d part output channels",
  );
  if (
    shape.firstOutputChannel + shape.partOutputChannels >
    shape.outputChannels
  ) {
    throw new RangeError(
      "ACE VAE ConvTranspose1d part exceeds total output channels",
    );
  }
}

function requireConvBindings(
  label: string,
  plan: AceVaeConvPlan,
  bindings: AceVaeConvBindings,
): void {
  requireAceBindingBytes(
    bindings.input,
    plan.inputElements * FLOAT32_BYTES,
    `${label} input`,
  );
  requireAceBindingBytes(
    bindings.weight,
    plan.weightElements * FLOAT32_BYTES,
    `${label} weight`,
  );
  if (bindings.bias !== undefined) {
    requireAceBindingBytes(
      bindings.bias,
      plan.outputChannels * FLOAT32_BYTES,
      `${label} bias`,
    );
  }
  requireAceBindingBytes(
    bindings.output,
    plan.outputElements * FLOAT32_BYTES,
    `${label} output`,
  );
  requireAceDisjointOutput(
    bindings.output,
    [
      bindings.input,
      bindings.weight,
      ...(bindings.bias === undefined ? [] : [bindings.bias]),
    ],
    label,
  );
}

function convPrelude(
  plan: AceVaeConvPlan,
  hasBias: boolean,
  ranged: boolean,
): string {
  return /* wgsl */ `
const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const KERNEL_SIZE: u32 = ${plan.kernelSize}u;
const STRIDE: u32 = ${plan.stride}u;
const DILATION: u32 = ${plan.dilation}u;
const PADDING: u32 = ${plan.padding}u;
const OUTPUT_ELEMENTS: u32 = ${plan.outputElements}u;
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
${vaeConvOutputBindingsWgsl(hasBias)}

${vaeInvocationWgsl(
  plan.outputElements,
  plan.workgroupsX,
  ranged,
  hasBias ? 4 : 3,
)}
`;
}

function convKey(plan: AceVaeConvPlan): string {
  return [
    plan.batch,
    plan.inputFrames,
    plan.inputChannels,
    plan.outputChannels,
    plan.kernelSize,
    plan.stride,
    plan.dilation,
    plan.padding,
    plan.outputPadding,
  ].join("x");
}

function vaeConvOutputBindingsWgsl(hasBias: boolean): string {
  return hasBias
    ? `@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;`
    : "@group(0) @binding(2) var<storage, read_write> output: array<f32>;";
}

function appendRangeBinding(
  bindings: readonly GPUBufferBinding[],
  range: AceVaeOutputRangeBinding | undefined,
): readonly GPUBufferBinding[] {
  return range === undefined ? bindings : [...bindings, range.control];
}

function resolveOutputRange(
  range: AceVaeOutputRangeBinding | undefined,
  outputElements: number,
  label: string,
  uniformOffsetAlignment: number,
): AceVaeOutputRange {
  requirePositiveSafeInteger(outputElements, `${label} domain`);
  if (outputElements > 0xffff_ffff) {
    throw new RangeError(`${label} domain exceeds WGSL's u32 indexing domain`);
  }
  if (range === undefined) {
    return Object.freeze({ base: 0, count: outputElements });
  }
  if (!Number.isSafeInteger(range.base) || range.base < 0) {
    throw new RangeError(`${label} range base must be a non-negative safe integer`);
  }
  requirePositiveSafeInteger(range.count, `${label} range count`);
  if (
    range.base > 0xffff_ffff ||
    range.count > 0xffff_ffff ||
    range.base + range.count > outputElements
  ) {
    throw new RangeError(`${label} range exceeds its complete output domain`);
  }
  requireAceBindingBytes(range.control, 16, `${label} range control`);
  const controlOffset = range.control.offset ?? 0;
  if (controlOffset > 0xffff_ffff) {
    throw new RangeError(
      `${label} range control offset exceeds WebGPU's dynamic-offset domain`,
    );
  }
  if (
    !Number.isSafeInteger(uniformOffsetAlignment) ||
    uniformOffsetAlignment <= 0 ||
    controlOffset % uniformOffsetAlignment !== 0
  ) {
    throw new RangeError(
      `${label} range control offset must satisfy uniform-buffer alignment`,
    );
  }
  return Object.freeze({ base: range.base, count: range.count });
}

function vaeInvocationWgsl(
  elements: number,
  workgroupsX: number,
  ranged: boolean,
  rangeBinding: number,
): string {
  if (!ranged) return aceLinearInvocationWgsl(elements, workgroupsX);
  return /* wgsl */ `
struct AceVaeOutputRangeControl {
  base: u32,
  count: u32,
  padding0: u32,
  padding1: u32,
};

@group(0) @binding(${rangeBinding})
var<uniform> output_range: AceVaeOutputRangeControl;

fn quantum_linear_index(
  workgroup_id: vec3<u32>,
  num_workgroups: vec3<u32>,
  lane: u32,
) -> u32 {
  let workgroup = workgroup_id.y * num_workgroups.x + workgroup_id.x;
  return workgroup * ${ACE_CORRECTNESS_WORKGROUP_SIZE}u + lane;
}
`;
}

function vaeNumWorkgroupsParameterWgsl(ranged: boolean): string {
  return ranged
    ? "@builtin(num_workgroups) num_workgroups: vec3<u32>,"
    : "";
}

function vaeOutputIndexWgsl(name: string, ranged: boolean): string {
  return ranged
    ? `let local_output_index = quantum_linear_index(
    workgroup_id,
    num_workgroups,
    lane,
  );
  if (local_output_index >= output_range.count) { return; }
  let ${name} = output_range.base + local_output_index;
  if (${name} >= OUTPUT_ELEMENTS) { return; }`
    : `let ${name} = linear_index(workgroup_id, lane);
  if (${name} >= OUTPUT_ELEMENTS) { return; }`;
}

async function compilePrimitive(
  device: GPUDevice,
  label: string,
  code: string,
  rangedBindingCount?: number,
): Promise<CompiledVaePrimitive> {
  const module = device.createShaderModule({ label, code });
  const bindGroupLayout = rangedBindingCount === undefined
    ? undefined
    : device.createBindGroupLayout({
        label: `${label}-bindings`,
        entries: Array.from({ length: rangedBindingCount }, (_, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: binding === rangedBindingCount - 1
            ? { type: "uniform" as const, hasDynamicOffset: true, minBindingSize: 16 }
            : binding === rangedBindingCount - 2
            ? { type: "storage" as const }
            : { type: "read-only-storage" as const },
        })),
      });
  const pipelineLayout = bindGroupLayout === undefined
    ? "auto" as const
    : device.createPipelineLayout({
        label: `${label}-layout`,
        bindGroupLayouts: [bindGroupLayout],
      });
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: pipelineLayout,
    compute: { module, entryPoint: "main" },
  });
  return Object.freeze({
    pipeline,
    bindGroupLayout: bindGroupLayout ?? pipeline.getBindGroupLayout(0),
  });
}
