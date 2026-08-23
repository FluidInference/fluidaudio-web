import {
  requireAceBindingBytes,
  requireAceDisjointOutput,
} from "./correctness-utils.js";
import {
  planAceVaeConv1d,
  type AceVaeConv1dShape,
  type AceVaeConvBindings,
  type AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_TILED_VAE_CONV1D_KERNEL_SIZE = 7;
export const ACE_TILED_VAE_CONV1D_TILE_FRAMES = 16;
export const ACE_TILED_VAE_CONV1D_TILE_CHANNELS = 8;
export const ACE_TILED_VAE_CONV1D_INPUT_TILE_FRAMES = 22;
/** One padding slot prevents a bank-aligned time-row stride. */
export const ACE_TILED_VAE_CONV1D_INPUT_TILE_STRIDE = 23;
export const ACE_TILED_VAE_CONV1D_WORKGROUP_SIZE_X = 16;
export const ACE_TILED_VAE_CONV1D_WORKGROUP_SIZE_Y = 8;
export const ACE_TILED_VAE_CONV1D_WORKGROUP_SIZE = 128;

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const MAX_WGSL_U32 = 0xffff_ffff;
const RANGE_CONTROL_BYTES = 16;

export interface AceTiledVaeConv1dPlan extends AceVaeConv1dShape {
  readonly outputFrames: number;
  readonly inputElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly inputTileElements: number;
  readonly weightTileElements: number;
  readonly inputTileBytes: number;
  readonly weightTileBytes: number;
  readonly workgroupStorageBytes: number;
  readonly weightTileStride: number;
}

export interface AceTiledVaeConv1dSelection {
  readonly eligible: boolean;
  readonly reason:
    | "eligible"
    | "unsupported-math"
    | "unsupported-workgroup"
    | "workgroup-storage"
    | "unaligned-range"
    | "batch-crossing-range"
    | "dispatch-limit"
    | "storage-binding-limit"
    | "wgsl-index-limit";
  readonly plan?: AceTiledVaeConv1dPlan;
  readonly firstOutputRow?: number;
  readonly outputRowCount?: number;
  readonly workgroupsX?: number;
  readonly workgroupsY?: number;
}

export interface AceTiledVaeConv1dDispatch {
  readonly label: string;
  readonly plan: AceTiledVaeConv1dPlan;
  readonly outputRange: Readonly<{ base: number; count: number }>;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledTiledVaeConv1d {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Production OPT-0004 K7/stride-one/dilation-one Conv1D specialization.
 *
 * The scalar primitive remains the universal implementation. Callers select
 * this kernel only through `selectAceTiledVaeConv1d`, which fails closed for
 * unsupported math, device limits, or ranges that are not complete NLC rows.
 */
export class AceTiledVaeConv1dKernel {
  private readonly pipelines = new Map<
    string,
    Promise<CompiledTiledVaeConv1d>
  >();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(device: GPUDevice): AceTiledVaeConv1dKernel {
    return new AceTiledVaeConv1dKernel(device);
  }

  select(
    shape: AceVaeConv1dShape,
    range: Pick<AceVaeOutputRangeBinding, "base" | "count">,
  ): AceTiledVaeConv1dSelection {
    return selectAceTiledVaeConv1d(this.device.limits, shape, range);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceVaeConvBindings,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceTiledVaeConv1dDispatch> {
    this.requireLive();
    const selection = this.select(shape, range);
    if (!selection.eligible || selection.plan === undefined) {
      throw new RangeError(
        `${label} is not eligible for tiled VAE Conv1D: ${selection.reason}`,
      );
    }
    const plan = selection.plan;
    const firstOutputRow = selection.firstOutputRow!;
    const outputRowCount = selection.outputRowCount!;
    const workgroupsX = selection.workgroupsX!;
    const workgroupsY = selection.workgroupsY!;
    const inputBytes = checkedBytes(plan.inputElements, "input");
    const weightBytes = checkedBytes(plan.weightElements, "weight");
    const outputBytes = checkedBytes(plan.outputElements, "output");
    const biasBytes = checkedBytes(plan.outputChannels, "bias");
    requireAceBindingBytes(bindings.input, inputBytes, `${label} input`);
    requireAceBindingBytes(bindings.weight, weightBytes, `${label} weight`);
    requireAceBindingBytes(bindings.output, outputBytes, `${label} output`);
    requireAceBindingBytes(range.control, RANGE_CONTROL_BYTES, `${label} range control`);
    if (bindings.bias !== undefined) {
      requireAceBindingBytes(bindings.bias, biasBytes, `${label} bias`);
    }
    requireAceDisjointOutput(
      exactBinding(bindings.output, outputBytes),
      [
        exactBinding(bindings.input, inputBytes),
        exactBinding(bindings.weight, weightBytes),
        ...(bindings.bias === undefined
          ? []
          : [exactBinding(bindings.bias, biasBytes)]),
        range.control,
      ],
      label,
    );
    const alignment = this.device.limits.minUniformBufferOffsetAlignment;
    const controlOffset = range.control.offset ?? 0;
    if (
      !Number.isSafeInteger(alignment) || alignment <= 0 ||
      !Number.isSafeInteger(controlOffset) || controlOffset < 0 ||
      controlOffset > MAX_WGSL_U32 || controlOffset % alignment !== 0
    ) {
      throw new RangeError(
        `${label} range control offset must satisfy uniform-buffer alignment`,
      );
    }

    const hasBias = bindings.bias !== undefined;
    const compiled = await this.pipelineFor(plan, hasBias);
    this.requireLive();
    const rangeBinding = hasBias ? 4 : 3;
    const resources = hasBias
      ? [
          bindings.input,
          bindings.weight,
          bindings.bias!,
          bindings.output,
          range.control,
        ]
      : [bindings.input, bindings.weight, bindings.output, range.control];
    const bindGroupResources = resources.map((binding, index) =>
      index === rangeBinding
        ? Object.freeze({ buffer: binding.buffer, offset: 0, size: RANGE_CONTROL_BYTES })
        : binding
    );
    const key = `${convKey(plan, hasBias)}:${bindGroupResources.map((binding) =>
      this.bindingKey(binding)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(key);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-bindings`,
        layout: compiled.bindGroupLayout,
        entries: bindGroupResources.map((resource, binding) => ({
          binding,
          resource,
        })),
      });
      this.bindGroups.set(key, bindGroup);
    }
    const dynamicOffsets = [controlOffset];
    return Object.freeze({
      label,
      plan,
      outputRange: Object.freeze({ base: range.base, count: range.count }),
      firstOutputRow,
      outputRowCount,
      encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroup, dynamicOffsets);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
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
    plan: AceTiledVaeConv1dPlan,
    hasBias: boolean,
  ): Promise<CompiledTiledVaeConv1d> {
    const key = convKey(plan, hasBias);
    let compiled = this.pipelines.get(key);
    if (compiled !== undefined) return compiled;
    compiled = compileTiledVaeConv1d(this.device, plan, hasBias);
    this.pipelines.set(key, compiled);
    void compiled.catch(() => {
      if (this.pipelines.get(key) === compiled) this.pipelines.delete(key);
    });
    return compiled;
  }

  private bindingKey(binding: GPUBufferBinding): string {
    let id = this.bufferIds.get(binding.buffer);
    if (id === undefined) {
      id = this.nextBufferId++;
      this.bufferIds.set(binding.buffer, id);
    }
    return `${id}:${binding.offset ?? 0}:${binding.size ?? -1}`;
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("ACE tiled VAE Conv1D kernel was destroyed");
    }
  }
}

export function planAceTiledVaeConv1d(
  shape: AceVaeConv1dShape,
): AceTiledVaeConv1dPlan {
  const portable = planAceVaeConv1d(shape);
  if (
    shape.kernelSize !== ACE_TILED_VAE_CONV1D_KERNEL_SIZE ||
    shape.stride !== 1 || shape.dilation !== 1
  ) {
    throw new RangeError(
      "ACE tiled VAE Conv1D requires K7, stride one, and dilation one",
    );
  }
  const inputTileElements = checkedProduct(
    [shape.inputChannels, ACE_TILED_VAE_CONV1D_INPUT_TILE_STRIDE],
    "input tile",
  );
  const weightTileStride = shape.inputChannels + 1;
  if (!Number.isSafeInteger(weightTileStride)) {
    throw new RangeError("ACE tiled VAE Conv1D weight stride is unsafe");
  }
  const weightTileElements = checkedProduct(
    [ACE_TILED_VAE_CONV1D_TILE_CHANNELS, weightTileStride],
    "weight tile",
  );
  const inputTileBytes = checkedBytes(inputTileElements, "input tile");
  const weightTileBytes = checkedBytes(weightTileElements, "weight tile");
  return Object.freeze({
    ...shape,
    outputFrames: portable.outputFrames,
    inputElements: portable.inputElements,
    weightElements: portable.weightElements,
    outputElements: portable.outputElements,
    inputTileElements,
    weightTileElements,
    inputTileBytes,
    weightTileBytes,
    workgroupStorageBytes: checkedSum(
      inputTileBytes,
      weightTileBytes,
      "workgroup storage",
    ),
    weightTileStride,
  });
}

export function selectAceTiledVaeConv1d(
  limits: Pick<
    GPUSupportedLimits,
    | "maxComputeInvocationsPerWorkgroup"
    | "maxComputeWorkgroupSizeX"
    | "maxComputeWorkgroupSizeY"
    | "maxComputeWorkgroupStorageSize"
    | "maxComputeWorkgroupsPerDimension"
    | "maxStorageBufferBindingSize"
  >,
  shape: AceVaeConv1dShape,
  range: Readonly<{ base: number; count: number }>,
): AceTiledVaeConv1dSelection {
  if (
    shape.kernelSize !== ACE_TILED_VAE_CONV1D_KERNEL_SIZE ||
    shape.stride !== 1 || shape.dilation !== 1
  ) {
    return Object.freeze({ eligible: false, reason: "unsupported-math" });
  }
  let plan: AceTiledVaeConv1dPlan;
  try {
    plan = planAceTiledVaeConv1d(shape);
  } catch {
    return Object.freeze({ eligible: false, reason: "wgsl-index-limit" });
  }
  if (
    limits.maxComputeInvocationsPerWorkgroup <
      ACE_TILED_VAE_CONV1D_WORKGROUP_SIZE ||
    limits.maxComputeWorkgroupSizeX < ACE_TILED_VAE_CONV1D_WORKGROUP_SIZE_X ||
    limits.maxComputeWorkgroupSizeY < ACE_TILED_VAE_CONV1D_WORKGROUP_SIZE_Y
  ) {
    return Object.freeze({ eligible: false, reason: "unsupported-workgroup", plan });
  }
  if (plan.workgroupStorageBytes > limits.maxComputeWorkgroupStorageSize) {
    return Object.freeze({ eligible: false, reason: "workgroup-storage", plan });
  }
  if (
    !Number.isSafeInteger(range.base) || range.base < 0 ||
    !Number.isSafeInteger(range.count) || range.count <= 0 ||
    range.base + range.count > plan.outputElements ||
    range.base % plan.outputChannels !== 0 ||
    range.count % plan.outputChannels !== 0
  ) {
    return Object.freeze({ eligible: false, reason: "unaligned-range", plan });
  }
  const firstOutputRow = range.base / plan.outputChannels;
  const outputRowCount = range.count / plan.outputChannels;
  const firstOutputTime = firstOutputRow % plan.outputFrames;
  if (firstOutputTime + outputRowCount > plan.outputFrames) {
    return Object.freeze({
      eligible: false,
      reason: "batch-crossing-range",
      plan,
      firstOutputRow,
      outputRowCount,
    });
  }
  const workgroupsX = Math.ceil(
    outputRowCount / ACE_TILED_VAE_CONV1D_TILE_FRAMES,
  );
  const workgroupsY = Math.ceil(
    plan.outputChannels / ACE_TILED_VAE_CONV1D_TILE_CHANNELS,
  );
  if (
    workgroupsX > limits.maxComputeWorkgroupsPerDimension ||
    workgroupsY > limits.maxComputeWorkgroupsPerDimension
  ) {
    return Object.freeze({ eligible: false, reason: "dispatch-limit", plan });
  }
  if ([
    plan.inputElements,
    plan.weightElements,
    plan.outputElements,
    plan.outputChannels,
  ].some((elements) =>
    elements * FLOAT32_BYTES > limits.maxStorageBufferBindingSize
  )) {
    return Object.freeze({
      eligible: false,
      reason: "storage-binding-limit",
      plan,
    });
  }
  const rangeFirstTime = firstOutputTime;
  const lastStagedTime = rangeFirstTime +
    (workgroupsX - 1) * ACE_TILED_VAE_CONV1D_TILE_FRAMES +
    ACE_TILED_VAE_CONV1D_INPUT_TILE_FRAMES - 1;
  if (
    plan.inputElements > MAX_WGSL_U32 ||
    plan.weightElements > MAX_WGSL_U32 ||
    plan.outputElements > MAX_WGSL_U32 ||
    firstOutputRow > MAX_WGSL_U32 ||
    firstOutputTime + outputRowCount > MAX_WGSL_U32 ||
    lastStagedTime > MAX_WGSL_U32 ||
    shape.padding > MAX_WGSL_U32
  ) {
    return Object.freeze({ eligible: false, reason: "wgsl-index-limit", plan });
  }
  return Object.freeze({
    eligible: true,
    reason: "eligible",
    plan,
    firstOutputRow,
    outputRowCount,
    workgroupsX,
    workgroupsY,
  });
}

export function aceTiledVaeConv1dWgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
): string {
  const plan = planAceTiledVaeConv1d(shape);
  const outputBinding = hasBias ? 3 : 2;
  const rangeBinding = hasBias ? 4 : 3;
  const biasDeclaration = hasBias
    ? "@group(0) @binding(2) var<storage, read> bias: array<f32>;"
    : "";
  const initialSum = hasBias ? "bias[output_channel]" : "0.0";
  return /* wgsl */ `
const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const PADDING: u32 = ${plan.padding}u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
${biasDeclaration}
@group(0) @binding(${outputBinding}) var<storage, read_write>
  output: array<f32>;

struct AceTiledVaeConv1dRange {
  first_output: u32,
  output_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(${rangeBinding}) var<uniform>
  output_range: AceTiledVaeConv1dRange;

var<workgroup> input_tile: array<f32, ${plan.inputTileElements}>;
var<workgroup> weight_tile: array<f32, ${plan.weightTileElements}>;

@compute @workgroup_size(
  ${ACE_TILED_VAE_CONV1D_WORKGROUP_SIZE_X},
  ${ACE_TILED_VAE_CONV1D_WORKGROUP_SIZE_Y},
  1,
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
  let range_first_time = first_output_row % OUTPUT_FRAMES;
  let batch = first_output_row / OUTPUT_FRAMES;
  let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
  let tile_first_time =
    range_first_time + group.x * ${ACE_TILED_VAE_CONV1D_TILE_FRAMES}u;

  for (
    var tile_index = lane;
    tile_index < ${ACE_TILED_VAE_CONV1D_INPUT_TILE_FRAMES * plan.inputChannels}u;
    tile_index += ${ACE_TILED_VAE_CONV1D_WORKGROUP_SIZE}u
  ) {
    let tile_time = tile_index / INPUT_CHANNELS;
    let input_channel = tile_index % INPUT_CHANNELS;
    let padded_time = tile_first_time + tile_time;
    var value = 0.0;
    if (padded_time >= PADDING) {
      let input_time = padded_time - PADDING;
      if (input_time < INPUT_FRAMES) {
        value = input[
          (batch * INPUT_FRAMES + input_time) * INPUT_CHANNELS + input_channel
        ];
      }
    }
    input_tile[
      input_channel * ${ACE_TILED_VAE_CONV1D_INPUT_TILE_STRIDE}u + tile_time
    ] = value;
  }
  workgroupBarrier();

  let output_time = tile_first_time + local.x;
  let output_channel =
    group.y * ${ACE_TILED_VAE_CONV1D_TILE_CHANNELS}u + local.y;
  let range_end_time = range_first_time + output_row_count;
  let output_active =
    output_time < range_end_time && output_channel < OUTPUT_CHANNELS;
  var sum = 0.0;
  if (output_active) { sum = ${initialSum}; }

  for (
    var kernel = 0u;
    kernel < ${ACE_TILED_VAE_CONV1D_KERNEL_SIZE}u;
    kernel += 1u
  ) {
    for (
      var tile_index = lane;
      tile_index < ${ACE_TILED_VAE_CONV1D_TILE_CHANNELS * plan.inputChannels}u;
      tile_index += ${ACE_TILED_VAE_CONV1D_WORKGROUP_SIZE}u
    ) {
      let tile_output_channel = tile_index / INPUT_CHANNELS;
      let input_channel = tile_index % INPUT_CHANNELS;
      let weight_output_channel =
        group.y * ${ACE_TILED_VAE_CONV1D_TILE_CHANNELS}u + tile_output_channel;
      var value = 0.0;
      if (weight_output_channel < OUTPUT_CHANNELS) {
        value = weight[
          (weight_output_channel * ${ACE_TILED_VAE_CONV1D_KERNEL_SIZE}u +
            kernel) * INPUT_CHANNELS + input_channel
        ];
      }
      weight_tile[
        tile_output_channel * ${plan.weightTileStride}u + input_channel
      ] = value;
    }
    workgroupBarrier();

    if (output_active) {
      let padded_time = output_time + kernel;
      if (padded_time >= PADDING) {
        let input_time = padded_time - PADDING;
        if (input_time < INPUT_FRAMES) {
          let input_time_in_tile = local.x + kernel;
          let weight_base = local.y * ${plan.weightTileStride}u;
          for (
            var input_channel = 0u;
            input_channel < INPUT_CHANNELS;
            input_channel += 1u
          ) {
            sum = sum + input_tile[
              input_channel * ${ACE_TILED_VAE_CONV1D_INPUT_TILE_STRIDE}u +
              input_time_in_tile
            ] * weight_tile[weight_base + input_channel];
          }
        }
      }
    }
    workgroupBarrier();
  }

  if (output_active) {
    let output_row = batch * OUTPUT_FRAMES + output_time;
    output[output_row * OUTPUT_CHANNELS + output_channel] = sum;
  }
}
`;
}

async function compileTiledVaeConv1d(
  device: GPUDevice,
  plan: AceTiledVaeConv1dPlan,
  hasBias: boolean,
): Promise<CompiledTiledVaeConv1d> {
  const label = `ace-tiled-vae-conv1d-${convKey(plan, hasBias)}`;
  const module = device.createShaderModule({
    label,
    code: aceTiledVaeConv1dWgsl(plan, hasBias),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `ACE tiled VAE Conv1D WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  const bindingCount = hasBias ? 5 : 4;
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: Array.from({ length: bindingCount }, (_, binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: binding === bindingCount - 1
        ? { type: "uniform" as const, hasDynamicOffset: true, minBindingSize: 16 }
        : binding === bindingCount - 2
        ? { type: "storage" as const }
        : { type: "read-only-storage" as const },
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

function convKey(shape: AceVaeConv1dShape, hasBias: boolean): string {
  return [
    shape.batch,
    shape.inputFrames,
    shape.inputChannels,
    shape.outputChannels,
    shape.kernelSize,
    shape.stride,
    shape.dilation,
    shape.padding,
    hasBias ? "bias" : "no-bias",
  ].join("x");
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

function checkedProduct(values: readonly number[], label: string): number {
  let product = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`${label} must contain positive safe integers`);
    }
    product *= value;
    if (!Number.isSafeInteger(product)) {
      throw new RangeError(`${label} is not a safe integer`);
    }
  }
  return product;
}

function checkedSum(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    throw new RangeError(`${label} is not a safe integer`);
  }
  return result;
}

function checkedBytes(elements: number, label: string): number {
  return checkedProduct([elements, FLOAT32_BYTES], `${label} bytes`);
}
