import {
  planAceVaeConv1d,
  type AceVaeConv1dShape,
  type AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_FP16_VAE_CONV1D_K1_KERNEL_SIZE = 1;
export const ACE_FP16_VAE_CONV1D_K7_KERNEL_SIZE = 7;
export const ACE_FP16_VAE_CONV1D_K7_SUPPORTED_DILATIONS =
  Object.freeze([1, 3, 9] as const);
export const ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK = 64;
export const ACE_FP16_VAE_CONV1D_TILE_FRAMES = 16;
export const ACE_FP16_VAE_CONV1D_TILE_CHANNELS = 8;
/** One unused time slot avoids a bank-aligned shared-input row stride. */
export const ACE_FP16_VAE_CONV1D_INPUT_TILE_STRIDE = 17;
/** One unused channel slot avoids a bank-aligned shared-weight row stride. */
export const ACE_FP16_VAE_CONV1D_WEIGHT_TILE_STRIDE = 65;
export const ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_X = 16;
export const ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_Y = 8;
export const ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE = 128;
export const ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID =
  "ace-vae-fp16-portable-conv1d-v1" as const;

const FLOAT16_BYTES = 2;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const GPU_BUFFER_ALIGNMENT = 4;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_CONTROL_BYTES = 16;

export type AceFp16VaeConv1dFamily = "k1" | "k7";
export type AceFp16VaeConv1dOutputStorage = "float16" | "float32";

export interface AceFp16VaeConv1dBindings {
  /** FP16 activation in frame-major NLC order. */
  readonly input: GPUBufferBinding;
  /** FP16 package weight in converter-native `[out,kernel,in]` order. */
  readonly weight: GPUBufferBinding;
  /** FP16 `[out]`; required for K1 and omitted only by final K7 Conv1D. */
  readonly bias?: GPUBufferBinding;
  /** FP16 internal activation or FP32 final raw-waveform boundary. */
  readonly output: GPUBufferBinding;
}

export interface AceFp16VaeConv1dPlan extends AceVaeConv1dShape {
  readonly family: AceFp16VaeConv1dFamily;
  readonly outputStorage: AceFp16VaeConv1dOutputStorage;
  readonly outputFrames: number;
  readonly inputElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly inputStorageBytes: number;
  readonly inputBindingBytes: number;
  readonly weightStorageBytes: number;
  readonly weightBindingBytes: number;
  readonly biasStorageBytes: number;
  readonly biasBindingBytes: number;
  readonly outputStorageBytes: number;
  readonly outputBindingBytes: number;
  readonly inputChannelChunk:
    typeof ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK;
  readonly inputChannelChunkCount: number;
  readonly tileFrames: typeof ACE_FP16_VAE_CONV1D_TILE_FRAMES;
  readonly tileChannels: typeof ACE_FP16_VAE_CONV1D_TILE_CHANNELS;
  readonly inputTileStride:
    typeof ACE_FP16_VAE_CONV1D_INPUT_TILE_STRIDE;
  readonly weightTileStride:
    typeof ACE_FP16_VAE_CONV1D_WEIGHT_TILE_STRIDE;
  readonly workgroupSizeX: typeof ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_X;
  readonly workgroupSizeY: typeof ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_Y;
  readonly workgroupSize: typeof ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE;
  readonly inputTileElements: number;
  readonly weightTileElements: number;
  readonly inputTileBytes: number;
  readonly weightTileBytes: number;
  readonly workgroupStorageBytes: number;
}

export interface AceFp16VaeConv1dRangePlan {
  readonly base: number;
  readonly count: number;
  readonly batch: number;
  readonly firstOutputTime: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceFp16VaeConv1dDispatch {
  readonly label: string;
  readonly kernelId: typeof ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly outputRange: AceFp16VaeConv1dRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledAceFp16VaeConv1d {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

interface NamedBinding {
  readonly name: "input" | "weight" | "bias" | "output" | "range control";
  readonly binding: GPUBufferBinding;
}

/**
 * Portable production `shader-f16` Conv1D set for the OPT-0011 VAE profile.
 *
 * Each invocation owns one output. K7 retains the audited K-outer, increasing
 * chunk, increasing-Cin FP32 accumulation order. K1 uses the same portable
 * workgroup-memory tiling for the decoder's biased residual projections.
 */
export class AceFp16VaeConv1dKernel {
  private readonly pipelines = new Map<
    string,
    Promise<CompiledAceFp16VaeConv1d>
  >();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {
    requireKernelDevice(device);
  }

  static create(device: GPUDevice): AceFp16VaeConv1dKernel {
    return new AceFp16VaeConv1dKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceFp16VaeConv1dBindings,
    outputStorage: AceFp16VaeConv1dOutputStorage,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceFp16VaeConv1dDispatch> {
    this.requireLive();
    const plan = planAceFp16VaeConv1d(shape, outputStorage);
    requireBiasBoundary(plan, bindings.bias !== undefined);
    const outputRange = planAceFp16VaeConv1dRange(plan, range);
    this.requireDeviceLimits(plan, outputRange);
    const normalized = this.requireBindings(label, plan, bindings, range);
    const hasBias = bindings.bias !== undefined;
    const compiled = await this.pipelineFor(plan, hasBias);
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
    const bindGroupKey = `${convKey(plan, hasBias)}:${bindGroupResources.map(
      (binding) => this.bindingKey(binding),
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-fp16-conv1d-bindings`,
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
      kernelId: ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID,
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
    plan: AceFp16VaeConv1dPlan,
    hasBias: boolean,
  ): Promise<CompiledAceFp16VaeConv1d> {
    const key = convKey(plan, hasBias);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceFp16VaeConv1d(this.device, plan, hasBias);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
  }

  private requireBindings(
    label: string,
    plan: AceFp16VaeConv1dPlan,
    bindings: AceFp16VaeConv1dBindings,
    range: AceVaeOutputRangeBinding,
  ): readonly NamedBinding[] {
    const normalized: NamedBinding[] = [
      {
        name: "input",
        binding: requireStorageBinding(
          this.device,
          bindings.input,
          plan.inputStorageBytes,
          plan.inputBindingBytes,
          `${label} input`,
        ),
      },
      {
        name: "weight",
        binding: requireStorageBinding(
          this.device,
          bindings.weight,
          plan.weightStorageBytes,
          plan.weightBindingBytes,
          `${label} weight`,
        ),
      },
    ];
    if (bindings.bias !== undefined) {
      normalized.push({
        name: "bias",
        binding: requireStorageBinding(
          this.device,
          bindings.bias,
          plan.biasStorageBytes,
          plan.biasBindingBytes,
          `${label} bias`,
        ),
      });
    }
    normalized.push({
      name: "output",
      binding: requireStorageBinding(
        this.device,
        bindings.output,
        plan.outputStorageBytes,
        plan.outputBindingBytes,
        `${label} output`,
      ),
    });
    normalized.push({
      name: "range control",
      binding: requireRangeControlBinding(
        this.device,
        range.control,
        `${label} range control`,
      ),
    });
    requireDisjointBindings(normalized, label);
    return Object.freeze(normalized);
  }

  private requireDeviceLimits(
    plan: AceFp16VaeConv1dPlan,
    range: AceFp16VaeConv1dRangePlan,
  ): void {
    const maximumStorage = this.device.limits.maxComputeWorkgroupStorageSize;
    if (
      !Number.isSafeInteger(maximumStorage) ||
      maximumStorage < plan.workgroupStorageBytes
    ) {
      throw new RangeError(
        `ACE FP16 VAE Conv1D requires ${plan.workgroupStorageBytes} workgroup-storage bytes`,
      );
    }
    const maximumDispatch = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (
      !Number.isSafeInteger(maximumDispatch) || maximumDispatch < 1 ||
      range.workgroupsX > maximumDispatch ||
      range.workgroupsY > maximumDispatch
    ) {
      throw new RangeError(
        "ACE FP16 VAE Conv1D range exceeds the device dispatch dimension",
      );
    }
    const maximumBinding = Number(
      this.device.limits.maxStorageBufferBindingSize,
    );
    const maximumBuffer = Number(this.device.limits.maxBufferSize);
    if (
      !Number.isSafeInteger(maximumBinding) || maximumBinding < 1 ||
      !Number.isSafeInteger(maximumBuffer) || maximumBuffer < 1
    ) {
      throw new RangeError(
        "ACE FP16 VAE Conv1D device reported invalid buffer limits",
      );
    }
    for (const [name, bytes] of [
      ["input", plan.inputBindingBytes],
      ["weight", plan.weightBindingBytes],
      ["bias", plan.biasBindingBytes],
      ["output", plan.outputBindingBytes],
    ] as const) {
      if (bytes > maximumBinding) {
        throw new RangeError(
          `ACE FP16 VAE Conv1D ${name} exceeds the device storage binding limit`,
        );
      }
      if (bytes > maximumBuffer) {
        throw new RangeError(
          `ACE FP16 VAE Conv1D ${name} exceeds the device buffer limit`,
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
      throw new Error("ACE FP16 VAE Conv1D kernel was destroyed");
    }
  }
}

export function planAceFp16VaeConv1d(
  shape: AceVaeConv1dShape,
  outputStorage: AceFp16VaeConv1dOutputStorage,
): AceFp16VaeConv1dPlan {
  requireOutputStorage(outputStorage);
  const portable = planAceVaeConv1d(shape);
  const family = requireSupportedShape(shape);
  if (family === "k1" && outputStorage !== "float16") {
    throw new RangeError(
      "ACE FP16 VAE K1 Conv1D requires an FP16 internal output",
    );
  }
  if (portable.outputFrames !== shape.inputFrames) {
    throw new Error("ACE FP16 VAE Conv1D lost same-length frame identity");
  }
  for (const [name, value] of [
    ["input", portable.inputElements],
    ["weight", portable.weightElements],
    ["output", portable.outputElements],
    ["padding", shape.padding],
  ] as const) requireWgslIndexable(value, name);

  const inputChannelChunkCount = Math.ceil(
    shape.inputChannels / ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK,
  );
  requireWgslIndexable(
    inputChannelChunkCount,
    "input channel chunk count",
  );
  const inputTileElements = checkedProduct([
    ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK,
    ACE_FP16_VAE_CONV1D_INPUT_TILE_STRIDE,
  ], "input tile");
  const weightTileElements = checkedProduct([
    ACE_FP16_VAE_CONV1D_TILE_CHANNELS,
    ACE_FP16_VAE_CONV1D_WEIGHT_TILE_STRIDE,
  ], "weight tile");
  const inputTileBytes = checkedStorageBytes(
    inputTileElements,
    FLOAT16_BYTES,
    "input tile",
  );
  const weightTileBytes = checkedStorageBytes(
    weightTileElements,
    FLOAT16_BYTES,
    "weight tile",
  );
  const workgroupStorageBytes = checkedSum(
    inputTileBytes,
    weightTileBytes,
    "workgroup storage",
  );
  const inputStorageBytes = checkedStorageBytes(
    portable.inputElements,
    FLOAT16_BYTES,
    "input",
  );
  const weightStorageBytes = checkedStorageBytes(
    portable.weightElements,
    FLOAT16_BYTES,
    "weight",
  );
  const biasStorageBytes = checkedStorageBytes(
    shape.outputChannels,
    FLOAT16_BYTES,
    "bias",
  );
  const outputStorageBytes = checkedStorageBytes(
    portable.outputElements,
    outputStorage === "float16" ? FLOAT16_BYTES : FLOAT32_BYTES,
    "output",
  );
  return Object.freeze({
    ...shape,
    family,
    outputStorage,
    outputFrames: portable.outputFrames,
    inputElements: portable.inputElements,
    weightElements: portable.weightElements,
    outputElements: portable.outputElements,
    inputStorageBytes,
    inputBindingBytes: alignGpuBindingBytes(inputStorageBytes, "input"),
    weightStorageBytes,
    weightBindingBytes: alignGpuBindingBytes(weightStorageBytes, "weight"),
    biasStorageBytes,
    biasBindingBytes: alignGpuBindingBytes(biasStorageBytes, "bias"),
    outputStorageBytes,
    outputBindingBytes: alignGpuBindingBytes(outputStorageBytes, "output"),
    inputChannelChunk: ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK,
    inputChannelChunkCount,
    tileFrames: ACE_FP16_VAE_CONV1D_TILE_FRAMES,
    tileChannels: ACE_FP16_VAE_CONV1D_TILE_CHANNELS,
    inputTileStride: ACE_FP16_VAE_CONV1D_INPUT_TILE_STRIDE,
    weightTileStride: ACE_FP16_VAE_CONV1D_WEIGHT_TILE_STRIDE,
    workgroupSizeX: ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_X,
    workgroupSizeY: ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_Y,
    workgroupSize: ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE,
    inputTileElements,
    weightTileElements,
    inputTileBytes,
    weightTileBytes,
    workgroupStorageBytes,
  });
}

export function planAceFp16VaeConv1dRange(
  plan: AceFp16VaeConv1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceFp16VaeConv1dRangePlan {
  if (
    !Number.isSafeInteger(range.base) || range.base < 0 ||
    !Number.isSafeInteger(range.count) || range.count <= 0 ||
    range.base + range.count > plan.outputElements ||
    range.base % plan.outputChannels !== 0 ||
    range.count % plan.outputChannels !== 0
  ) {
    throw new RangeError(
      "ACE FP16 VAE Conv1D range must contain complete in-bounds NLC rows",
    );
  }
  const firstOutputRow = range.base / plan.outputChannels;
  const outputRowCount = range.count / plan.outputChannels;
  const firstOutputTime = firstOutputRow % plan.outputFrames;
  if (firstOutputTime + outputRowCount > plan.outputFrames) {
    throw new RangeError(
      "ACE FP16 VAE Conv1D range must not cross a batch boundary",
    );
  }
  const batch = Math.floor(firstOutputRow / plan.outputFrames);
  const workgroupsX = Math.ceil(
    outputRowCount / ACE_FP16_VAE_CONV1D_TILE_FRAMES,
  );
  const workgroupsY = Math.ceil(
    plan.outputChannels / ACE_FP16_VAE_CONV1D_TILE_CHANNELS,
  );
  for (const [name, value] of [
    ["range first output", range.base],
    ["range output count", range.count],
    ["range first output row", firstOutputRow],
    ["range end time", firstOutputTime + outputRowCount],
    [
      "last staged input time",
      firstOutputTime +
        (workgroupsX - 1) * ACE_FP16_VAE_CONV1D_TILE_FRAMES +
        (ACE_FP16_VAE_CONV1D_TILE_FRAMES - 1) +
        (plan.kernelSize - 1) * plan.dilation,
    ],
  ] as const) requireWgslIndexable(value, name);
  return Object.freeze({
    base: range.base,
    count: range.count,
    batch,
    firstOutputTime,
    firstOutputRow,
    outputRowCount,
    workgroupsX,
    workgroupsY,
  });
}

export function aceFp16VaeConv1dWgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceFp16VaeConv1dOutputStorage,
): string {
  const plan = planAceFp16VaeConv1d(shape, outputStorage);
  requireBiasBoundary(plan, hasBias);
  const outputBinding = hasBias ? 3 : 2;
  const rangeBinding = hasBias ? 4 : 3;
  const biasDeclaration = hasBias
    ? "@group(0) @binding(2) var<storage, read> bias: array<f16>;"
    : "";
  const outputElementType = outputStorage === "float16" ? "f16" : "f32";
  return /* wgsl */ `
// kernel-id: ${ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID}
enable f16;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const PADDING: u32 = ${plan.padding}u;
const DILATION: u32 = ${plan.dilation}u;
const INPUT_CHANNEL_CHUNKS: u32 = ${plan.inputChannelChunkCount}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
${biasDeclaration}
@group(0) @binding(${outputBinding}) var<storage, read_write>
  output: array<${outputElementType}>;

struct OutputRangeParameters {
  first_output: u32,
  output_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(${rangeBinding}) var<uniform>
  output_range: OutputRangeParameters;

var<workgroup> input_tile: array<f16, ${plan.inputTileElements}>;
var<workgroup> weight_tile: array<f16, ${plan.weightTileElements}>;

@compute @workgroup_size(
  ${ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_X},
  ${ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_Y},
  1,
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
  let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
  let range_first_time = first_output_row % OUTPUT_FRAMES;
  let batch = first_output_row / OUTPUT_FRAMES;
  let tile_first_time =
    range_first_time + group.x * ${ACE_FP16_VAE_CONV1D_TILE_FRAMES}u;
  let output_time = tile_first_time + local.x;
  let output_channel =
    group.y * ${ACE_FP16_VAE_CONV1D_TILE_CHANNELS}u + local.y;
  let range_end_time = range_first_time + output_row_count;
  let output_active =
    output_time < range_end_time && output_channel < OUTPUT_CHANNELS;
  var sum: f32 = 0.0;
  if (output_active) { sum = ${initialSumWgsl(hasBias)}; }

  // Increasing chunks concatenate into the exact K-outer, increasing-Cin
  // scalar order while f16 workgroup storage reuses operands.
  for (var kernel = 0u; kernel < ${plan.kernelSize}u; kernel += 1u) {
    for (
      var input_channel_chunk = 0u;
      input_channel_chunk < INPUT_CHANNEL_CHUNKS;
      input_channel_chunk += 1u
    ) {
      let chunk_first_channel = input_channel_chunk *
        ${ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK}u;
      for (
        var tile_index = lane;
        tile_index < ${ACE_FP16_VAE_CONV1D_TILE_FRAMES * ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK}u;
        tile_index += ${ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE}u
      ) {
        let tile_time = tile_index /
          ${ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK}u;
        let chunk_channel = tile_index %
          ${ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK}u;
        let input_channel = chunk_first_channel + chunk_channel;
        let padded_time = tile_first_time + tile_time + kernel * DILATION;
        var value: f16 = f16(0.0);
        if (input_channel < INPUT_CHANNELS && padded_time >= PADDING) {
          let input_time = padded_time - PADDING;
          if (input_time < INPUT_FRAMES) {
            value = input[
              (batch * INPUT_FRAMES + input_time) * INPUT_CHANNELS +
              input_channel
            ];
          }
        }
        input_tile[
          chunk_channel * ${ACE_FP16_VAE_CONV1D_INPUT_TILE_STRIDE}u +
          tile_time
        ] = value;
      }
      for (
        var tile_index = lane;
        tile_index < ${ACE_FP16_VAE_CONV1D_TILE_CHANNELS * ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK}u;
        tile_index += ${ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE}u
      ) {
        let tile_output_channel = tile_index /
          ${ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK}u;
        let chunk_channel = tile_index %
          ${ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK}u;
        let input_channel = chunk_first_channel + chunk_channel;
        let weight_output_channel =
          group.y * ${ACE_FP16_VAE_CONV1D_TILE_CHANNELS}u +
          tile_output_channel;
        var value: f16 = f16(0.0);
        if (
          weight_output_channel < OUTPUT_CHANNELS &&
          input_channel < INPUT_CHANNELS
        ) {
          // Native [output_channel, kernel, input_channel] weight order.
          value = weight[
            (weight_output_channel * ${plan.kernelSize}u + kernel) *
              INPUT_CHANNELS + input_channel
          ];
        }
        weight_tile[
          tile_output_channel *
            ${ACE_FP16_VAE_CONV1D_WEIGHT_TILE_STRIDE}u +
          chunk_channel
        ] = value;
      }
      workgroupBarrier();

      if (output_active) {
        let padded_time = output_time + kernel * DILATION;
        // Invalid padding skips every channel operation for this tap.
        if (padded_time >= PADDING) {
          let input_time = padded_time - PADDING;
          if (input_time < INPUT_FRAMES) {
            let chunk_channel_count = min(
              ${ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK}u,
              INPUT_CHANNELS - chunk_first_channel
            );
            let weight_base = local.y *
              ${ACE_FP16_VAE_CONV1D_WEIGHT_TILE_STRIDE}u;
            for (
              var chunk_channel = 0u;
              chunk_channel < chunk_channel_count;
              chunk_channel += 1u
            ) {
              let input_operand = f32(input_tile[
                chunk_channel *
                  ${ACE_FP16_VAE_CONV1D_INPUT_TILE_STRIDE}u + local.x
              ]);
              let weight_operand = f32(
                weight_tile[weight_base + chunk_channel]
              );
              sum = sum + input_operand * weight_operand;
            }
          }
        }
      }
      workgroupBarrier();
    }
  }

  if (output_active) {
    let output_row = batch * OUTPUT_FRAMES + output_time;
    output[output_row * OUTPUT_CHANNELS + output_channel] =
      ${outputValueWgsl(hasBias)};
  }
}
`;
}

async function compileAceFp16VaeConv1d(
  device: GPUDevice,
  plan: AceFp16VaeConv1dPlan,
  hasBias: boolean,
): Promise<CompiledAceFp16VaeConv1d> {
  const label = `ace-fp16-vae-conv1d-${convKey(plan, hasBias)}`;
  const module = device.createShaderModule({
    label,
    code: aceFp16VaeConv1dWgsl(plan, hasBias, plan.outputStorage),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `ACE FP16 VAE Conv1D WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  const dataBindingSizes = hasBias
    ? [
        plan.inputBindingBytes,
        plan.weightBindingBytes,
        plan.biasBindingBytes,
        plan.outputBindingBytes,
      ]
    : [
        plan.inputBindingBytes,
        plan.weightBindingBytes,
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

function requireSupportedShape(
  shape: AceVaeConv1dShape,
): AceFp16VaeConv1dFamily {
  if (
    shape.kernelSize === ACE_FP16_VAE_CONV1D_K1_KERNEL_SIZE &&
    shape.stride === 1 && shape.dilation === 1 && shape.padding === 0
  ) return "k1";
  if (
    shape.kernelSize === ACE_FP16_VAE_CONV1D_K7_KERNEL_SIZE &&
    shape.stride === 1 &&
    ACE_FP16_VAE_CONV1D_K7_SUPPORTED_DILATIONS.includes(
      shape.dilation as 1 | 3 | 9,
    ) &&
    shape.padding === shape.dilation * 3
  ) return "k7";
  throw new RangeError(
    "ACE FP16 VAE Conv1D requires biased K1/stride1/dilation1/padding0 or K7/stride1/dilation1,3,9/padding=dilation*3",
  );
}

function requireBiasBoundary(
  plan: AceFp16VaeConv1dPlan,
  hasBias: boolean,
): void {
  if (plan.family === "k1" && !hasBias) {
    throw new RangeError(
      "ACE FP16 VAE K1 Conv1D requires the residual-projection bias",
    );
  }
  if (hasBias && plan.outputStorage === "float32") {
    throw new RangeError(
      "ACE FP16 VAE Conv1D FP32 output is reserved for the final no-bias raw-waveform boundary",
    );
  }
  if (!hasBias && plan.outputStorage !== "float32") {
    throw new RangeError(
      "ACE FP16 VAE Conv1D bias may be omitted only at the final FP32 raw-waveform boundary",
    );
  }
}

function requireOutputStorage(
  outputStorage: AceFp16VaeConv1dOutputStorage,
): void {
  if (outputStorage !== "float16" && outputStorage !== "float32") {
    throw new TypeError(
      `ACE FP16 VAE Conv1D has unknown output storage ${String(outputStorage)}`,
    );
  }
}

function initialSumWgsl(hasBias: boolean): string {
  return hasBias ? "f32(bias[output_channel])" : "0.0";
}

function outputValueWgsl(hasBias: boolean): string {
  return hasBias
    ? "f16(sum)"
    : "select(sum, bitcast<f32>(0u), (bitcast<u32>(sum) & 0x7fffffffu) == 0u)";
}

function requireKernelDevice(device: GPUDevice): void {
  if (!device.features.has("shader-f16")) {
    throw new Error("ACE FP16 VAE Conv1D requires WebGPU shader-f16");
  }
  const maximumInvocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const maximumSizeX = device.limits.maxComputeWorkgroupSizeX;
  const maximumSizeY = device.limits.maxComputeWorkgroupSizeY;
  if (
    !Number.isSafeInteger(maximumInvocations) ||
    !Number.isSafeInteger(maximumSizeX) ||
    !Number.isSafeInteger(maximumSizeY) ||
    maximumInvocations < ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE ||
    maximumSizeX < ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_X ||
    maximumSizeY < ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_Y
  ) {
    throw new Error(
      "ACE FP16 VAE Conv1D requires a 16x8 (128-lane) compute workgroup",
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
      "ACE FP16 VAE Conv1D device reported an invalid storage alignment",
    );
  }
  const bufferBytes = Number(binding.buffer.size);
  const offset = binding.offset ?? 0;
  const available = binding.size ?? bufferBytes - offset;
  if (
    !Number.isSafeInteger(bufferBytes) || bufferBytes < 1 ||
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
      "ACE FP16 VAE Conv1D device reported an invalid uniform alignment",
    );
  }
  const bufferBytes = Number(binding.buffer.size);
  const offset = binding.offset ?? 0;
  const available = binding.size ?? bufferBytes - offset;
  if (
    !Number.isSafeInteger(bufferBytes) || bufferBytes < 1 ||
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

function convKey(plan: AceFp16VaeConv1dPlan, hasBias: boolean): string {
  return [
    plan.family,
    plan.batch,
    plan.inputFrames,
    plan.inputChannels,
    plan.outputChannels,
    plan.kernelSize,
    plan.stride,
    plan.dilation,
    plan.padding,
    plan.outputStorage,
    hasBias ? "bias" : "no-bias",
  ].join("x");
}

function checkedProduct(values: readonly number[], label: string): number {
  let product = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(
        `ACE FP16 VAE Conv1D ${label} operand is not a non-negative safe integer`,
      );
    }
    product *= value;
    if (!Number.isSafeInteger(product)) {
      throw new RangeError(
        `ACE FP16 VAE Conv1D ${label} is not a safe integer`,
      );
    }
  }
  return product;
}

function checkedSum(left: number, right: number, label: string): number {
  if (
    !Number.isSafeInteger(left) || left < 0 ||
    !Number.isSafeInteger(right) || right < 0
  ) {
    throw new RangeError(
      `ACE FP16 VAE Conv1D ${label} operand is not a non-negative safe integer`,
    );
  }
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw new RangeError(
      `ACE FP16 VAE Conv1D ${label} is not a safe integer`,
    );
  }
  return sum;
}

function checkedStorageBytes(
  elements: number,
  bytesPerElement: number,
  label: string,
): number {
  return checkedProduct([elements, bytesPerElement], `${label} storage bytes`);
}

function alignGpuBindingBytes(bytes: number, label: string): number {
  const rounded = Math.ceil(bytes / GPU_BUFFER_ALIGNMENT) *
    GPU_BUFFER_ALIGNMENT;
  if (!Number.isSafeInteger(rounded) || rounded < bytes) {
    throw new RangeError(
      `ACE FP16 VAE Conv1D ${label} binding bytes are not a safe integer`,
    );
  }
  return rounded;
}

function requireWgslIndexable(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > MAX_WGSL_U32) {
    throw new RangeError(
      `ACE FP16 VAE Conv1D ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
