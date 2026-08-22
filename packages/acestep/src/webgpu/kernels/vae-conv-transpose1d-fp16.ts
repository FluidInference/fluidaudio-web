import { checkedAceProduct } from "./correctness-utils.js";
import {
  planAceVaeConvTranspose1d,
  type AceVaeConvTranspose1dShape,
  type AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_FP16_VAE_CONV_TRANSPOSE1D_SUPPORTED_STRIDES =
  Object.freeze([2, 4, 6, 10] as const);
export const ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK = 64;
export const ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES = 16;
export const ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS = 8;
/** One unused time slot avoids a bank-aligned shared-input row stride. */
export const ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE = 17;
/** One unused channel slot avoids a bank-aligned shared-weight row stride. */
export const ACE_FP16_VAE_CONV_TRANSPOSE1D_WEIGHT_TILE_STRIDE = 65;
export const ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_X = 16;
export const ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_Y = 8;
export const ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE = 128;
export const ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID =
  "ace-vae-fp16-portable-conv-transpose1d-v1" as const;
export const ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID =
  "ace-vae-fp16-congruent-two-tap-conv-transpose1d-v1" as const;

export type AceFp16VaeConvTranspose1dKernelId =
  | typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID
  | typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID;

const FLOAT16_BYTES = 2;
const GPU_BUFFER_ALIGNMENT = 4;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_CONTROL_BYTES = 16;

export interface AceFp16VaeConvTranspose1dBindings {
  /** FP16 activation in frame-major NLC order. */
  readonly input: GPUBufferBinding;
  /** FP16 package weight in converter-native `[out,kernel,in]` order. */
  readonly weight: GPUBufferBinding;
  /** FP16 `[out]`; every frozen decoder transpose is biased. */
  readonly bias: GPUBufferBinding;
  /** FP16 activation in frame-major NLC order. */
  readonly output: GPUBufferBinding;
}

export interface AceFp16VaeConvTranspose1dPlan
  extends AceVaeConvTranspose1dShape {
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
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK;
  readonly inputChannelChunkCount: number;
  readonly tileFrames: typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES;
  readonly tileChannels: typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS;
  readonly inputTileStride:
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE;
  readonly weightTileStride:
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_WEIGHT_TILE_STRIDE;
  readonly workgroupSizeX:
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_X;
  readonly workgroupSizeY:
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_Y;
  readonly workgroupSize:
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE;
  readonly inputTileElements: number;
  readonly weightTileElements: number;
  readonly inputTileBytes: number;
  readonly weightTileBytes: number;
  readonly workgroupStorageBytes: number;
}

export interface AceFp16VaeConvTranspose1dRangePlan {
  readonly base: number;
  readonly count: number;
  readonly batch: number;
  readonly firstOutputTime: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceFp16VaeConvTranspose1dCongruentRangePlan
  extends AceFp16VaeConvTranspose1dRangePlan {
  /** One dispatch layer per stride phase. */
  readonly workgroupsZ: number;
}

type AceFp16VaeConvTranspose1dRangePlanFor<
  KernelId extends AceFp16VaeConvTranspose1dKernelId,
> = KernelId extends
  typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
  ? AceFp16VaeConvTranspose1dCongruentRangePlan
  : AceFp16VaeConvTranspose1dRangePlan;

export interface AceFp16VaeConvTranspose1dDispatch<
  KernelId extends AceFp16VaeConvTranspose1dKernelId =
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
> {
  readonly label: string;
  readonly kernelId: KernelId;
  readonly plan: AceFp16VaeConvTranspose1dPlan;
  readonly outputRange: AceFp16VaeConvTranspose1dRangePlanFor<KernelId>;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledAceFp16VaeConvTranspose1d {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

interface NamedBinding {
  readonly name: "input" | "weight" | "bias" | "output" | "range control";
  readonly binding: GPUBufferBinding;
}

/**
 * Portable production `shader-f16` ConvTranspose1D for the OPT-0011 VAE.
 *
 * Each invocation owns one output. The kernel skips invalid inverse-stride
 * taps before its increasing-Cin loop, then stores one RNE FP16 result after
 * the exact kernel-ascending, input-channel-ascending FP32 reduction.
 */
export class AceFp16VaeConvTranspose1dKernel<
  KernelId extends AceFp16VaeConvTranspose1dKernelId =
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
> {
  private readonly pipelines = new Map<
    string,
    Promise<CompiledAceFp16VaeConvTranspose1d>
  >();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly kernelId: KernelId,
  ) {
    requireKernelDevice(device);
  }

  static create(
    device: GPUDevice,
  ): AceFp16VaeConvTranspose1dKernel<
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID
  > {
    return new AceFp16VaeConvTranspose1dKernel(
      device,
      ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
    );
  }

  static createCongruent(
    device: GPUDevice,
  ): AceFp16VaeConvTranspose1dKernel<
    typeof ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
  > {
    return new AceFp16VaeConvTranspose1dKernel(
      device,
      ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID,
    );
  }

  async createDispatch(
    label: string,
    shape: AceVaeConvTranspose1dShape,
    bindings: AceFp16VaeConvTranspose1dBindings,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceFp16VaeConvTranspose1dDispatch<KernelId>> {
    this.requireLive();
    const plan = planAceFp16VaeConvTranspose1d(shape);
    const outputRange = this.kernelId ===
        ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
      ? planAceFp16VaeConvTranspose1dCongruentRange(plan, range)
      : planAceFp16VaeConvTranspose1dRange(plan, range);
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
    const key = `${this.kernelId}:${transposeKey(plan)}`;
    const bindGroupKey = `${key}:${bindGroupResources.map((binding) =>
      this.bindingKey(binding)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-fp16-conv-transpose1d-bindings`,
        layout: compiled.bindGroupLayout,
        entries: bindGroupResources.map((resource, binding) => ({
          binding,
          resource,
        })),
      });
      this.bindGroups.set(bindGroupKey, bindGroup);
    }

    const workgroupsZ = this.kernelId ===
        ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
      ? (outputRange as AceFp16VaeConvTranspose1dCongruentRangePlan)
        .workgroupsZ
      : 1;
    const owner = this;
    return Object.freeze({
      label,
      kernelId: this.kernelId,
      plan,
      outputRange: outputRange as
        AceFp16VaeConvTranspose1dRangePlanFor<KernelId>,
      encode(pass: GPUComputePassEncoder): void {
        owner.requireLive();
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroup, [controlOffset]);
        pass.dispatchWorkgroups(
          outputRange.workgroupsX,
          outputRange.workgroupsY,
          workgroupsZ,
        );
      },
    }) as AceFp16VaeConvTranspose1dDispatch<KernelId>;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.bindGroups.clear();
    this.pipelines.clear();
  }

  private pipelineFor(
    plan: AceFp16VaeConvTranspose1dPlan,
  ): Promise<CompiledAceFp16VaeConvTranspose1d> {
    const key = `${this.kernelId}:${transposeKey(plan)}`;
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = this.kernelId ===
        ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID
      ? compileAceFp16VaeCongruentConvTranspose1d(this.device, plan)
      : compileAceFp16VaeConvTranspose1d(this.device, plan);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
  }

  private requireBindings(
    label: string,
    plan: AceFp16VaeConvTranspose1dPlan,
    bindings: AceFp16VaeConvTranspose1dBindings,
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
        name: "weight" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.weight,
          plan.weightStorageBytes,
          plan.weightBindingBytes,
          `${label} weight`,
        ),
      },
      {
        name: "bias" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.bias,
          plan.biasStorageBytes,
          plan.biasBindingBytes,
          `${label} bias`,
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
    plan: AceFp16VaeConvTranspose1dPlan,
    range:
      | AceFp16VaeConvTranspose1dRangePlan
      | AceFp16VaeConvTranspose1dCongruentRangePlan,
  ): void {
    const maximumWorkgroupStorage = Number(
      this.device.limits.maxComputeWorkgroupStorageSize,
    );
    if (
      !Number.isSafeInteger(maximumWorkgroupStorage) ||
      maximumWorkgroupStorage < plan.workgroupStorageBytes
    ) {
      throw new RangeError(
        `ACE FP16 VAE ConvTranspose1D requires ${plan.workgroupStorageBytes} workgroup-storage bytes`,
      );
    }
    const maximumDispatch = Number(
      this.device.limits.maxComputeWorkgroupsPerDimension,
    );
    if (
      !Number.isSafeInteger(maximumDispatch) || maximumDispatch < 1 ||
      range.workgroupsX > maximumDispatch ||
      range.workgroupsY > maximumDispatch ||
      ("workgroupsZ" in range && range.workgroupsZ > maximumDispatch)
    ) {
      throw new RangeError(
        "ACE FP16 VAE ConvTranspose1D range exceeds the device dispatch dimension",
      );
    }
    const maximumStorageBinding = Number(
      this.device.limits.maxStorageBufferBindingSize,
    );
    const maximumUniformBinding = Number(
      this.device.limits.maxUniformBufferBindingSize,
    );
    const maximumBuffer = Number(this.device.limits.maxBufferSize);
    if (
      !Number.isSafeInteger(maximumStorageBinding) ||
      maximumStorageBinding < 1 ||
      !Number.isSafeInteger(maximumUniformBinding) ||
      maximumUniformBinding < OUTPUT_RANGE_CONTROL_BYTES ||
      !Number.isSafeInteger(maximumBuffer) || maximumBuffer < 1
    ) {
      throw new RangeError(
        "ACE FP16 VAE ConvTranspose1D device reported invalid buffer limits",
      );
    }
    for (const [name, bytes] of [
      ["input", plan.inputBindingBytes],
      ["weight", plan.weightBindingBytes],
      ["bias", plan.biasBindingBytes],
      ["output", plan.outputBindingBytes],
    ] as const) {
      if (bytes > maximumStorageBinding) {
        throw new RangeError(
          `ACE FP16 VAE ConvTranspose1D ${name} exceeds the device storage binding limit`,
        );
      }
      if (bytes > maximumBuffer) {
        throw new RangeError(
          `ACE FP16 VAE ConvTranspose1D ${name} exceeds the device buffer limit`,
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
      throw new Error("ACE FP16 VAE ConvTranspose1D kernel was destroyed");
    }
  }
}

export function planAceFp16VaeConvTranspose1d(
  shape: AceVaeConvTranspose1dShape,
): AceFp16VaeConvTranspose1dPlan {
  const snapshot: AceVaeConvTranspose1dShape = {
    batch: shape.batch,
    inputFrames: shape.inputFrames,
    inputChannels: shape.inputChannels,
    outputChannels: shape.outputChannels,
    kernelSize: shape.kernelSize,
    stride: shape.stride,
    dilation: shape.dilation,
    padding: shape.padding,
    outputPadding: shape.outputPadding,
  };
  requireSupportedShape(snapshot);
  const portable = planAceVaeConvTranspose1d(snapshot);
  for (const [name, value] of [
    ["input", portable.inputElements],
    ["weight", portable.weightElements],
    ["output", portable.outputElements],
    ["input frames", portable.inputFrames],
    ["output frames", portable.outputFrames],
    ["input channels", portable.inputChannels],
    ["output channels", portable.outputChannels],
    ["kernel size", portable.kernelSize],
    ["stride", portable.stride],
    ["dilation", portable.dilation],
    ["padding", portable.padding],
  ] as const) requireWgslIndexable(value, name);
  requireWgslIndexable(
    (portable.kernelSize - 1) * portable.dilation,
    "maximum kernel time",
  );
  requireWgslIndexable(
    (portable.outputFrames - 1) + portable.padding,
    "maximum padded output time",
  );

  const inputChannelChunkCount = Math.ceil(
    portable.inputChannels /
      ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK,
  );
  requireWgslIndexable(
    inputChannelChunkCount,
    "input channel chunk count",
  );
  const inputTileElements = checkedAceProduct([
    ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK,
    ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE,
  ], "ACE FP16 VAE ConvTranspose1D input tile");
  const weightTileElements = checkedAceProduct([
    ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS,
    ACE_FP16_VAE_CONV_TRANSPOSE1D_WEIGHT_TILE_STRIDE,
  ], "ACE FP16 VAE ConvTranspose1D weight tile");
  const inputTileBytes = checkedStorageBytes(
    inputTileElements,
    "input tile",
  );
  const weightTileBytes = checkedStorageBytes(
    weightTileElements,
    "weight tile",
  );
  const workgroupStorageBytes = checkedSum(
    inputTileBytes,
    weightTileBytes,
    "workgroup storage",
  );
  const inputStorageBytes = checkedStorageBytes(
    portable.inputElements,
    "input",
  );
  const weightStorageBytes = checkedStorageBytes(
    portable.weightElements,
    "weight",
  );
  const biasStorageBytes = checkedStorageBytes(
    portable.outputChannels,
    "bias",
  );
  const outputStorageBytes = checkedStorageBytes(
    portable.outputElements,
    "output",
  );
  return Object.freeze({
    ...snapshot,
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
    inputChannelChunk:
      ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK,
    inputChannelChunkCount,
    tileFrames: ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES,
    tileChannels: ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS,
    inputTileStride: ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE,
    weightTileStride: ACE_FP16_VAE_CONV_TRANSPOSE1D_WEIGHT_TILE_STRIDE,
    workgroupSizeX: ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_X,
    workgroupSizeY: ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_Y,
    workgroupSize: ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE,
    inputTileElements,
    weightTileElements,
    inputTileBytes,
    weightTileBytes,
    workgroupStorageBytes,
  });
}

export function planAceFp16VaeConvTranspose1dRange(
  plan: AceFp16VaeConvTranspose1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceFp16VaeConvTranspose1dRangePlan {
  if (
    !Number.isSafeInteger(range.base) || range.base < 0 ||
    !Number.isSafeInteger(range.count) || range.count <= 0 ||
    range.base > MAX_WGSL_U32 || range.count > MAX_WGSL_U32 ||
    !Number.isSafeInteger(range.base + range.count) ||
    range.base + range.count > plan.outputElements ||
    range.base % plan.outputChannels !== 0 ||
    range.count % plan.outputChannels !== 0
  ) {
    throw new RangeError(
      "ACE FP16 VAE ConvTranspose1D range must contain complete in-bounds NLC rows",
    );
  }
  const firstOutputRow = range.base / plan.outputChannels;
  const outputRowCount = range.count / plan.outputChannels;
  const firstOutputTime = firstOutputRow % plan.outputFrames;
  if (firstOutputTime + outputRowCount > plan.outputFrames) {
    throw new RangeError(
      "ACE FP16 VAE ConvTranspose1D range must not cross a batch boundary",
    );
  }
  const batch = Math.floor(firstOutputRow / plan.outputFrames);
  const workgroupsX = Math.ceil(
    outputRowCount / ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES,
  );
  const workgroupsY = Math.ceil(
    plan.outputChannels / ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS,
  );
  const lastStagedOutputTime = firstOutputTime +
    (workgroupsX - 1) * ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES +
    (ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES - 1);
  for (const [name, value] of [
    ["range first output", range.base],
    ["range output count", range.count],
    ["range first output row", firstOutputRow],
    ["range end time", firstOutputTime + outputRowCount],
    ["last staged output time", lastStagedOutputTime],
    [
      "last staged padded output time",
      lastStagedOutputTime + plan.padding,
    ],
    ["range workgroups X", workgroupsX],
    ["range workgroups Y", workgroupsY],
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

export function planAceFp16VaeConvTranspose1dCongruentRange(
  plan: AceFp16VaeConvTranspose1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceFp16VaeConvTranspose1dCongruentRangePlan {
  const portable = planAceFp16VaeConvTranspose1dRange(plan, range);
  const workgroupsX = Math.ceil(
    portable.outputRowCount /
      (ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES * plan.stride),
  );
  const workgroupsZ = plan.stride;
  const lastStagedOutputTime = portable.firstOutputTime +
    workgroupsX * ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES * plan.stride - 1;
  for (const [name, value] of [
    ["congruent range workgroups X", workgroupsX],
    ["congruent range workgroups Z", workgroupsZ],
    ["last congruent staged output time", lastStagedOutputTime],
    [
      "last congruent staged padded output time",
      lastStagedOutputTime + plan.padding,
    ],
  ] as const) requireWgslIndexable(value, name);
  return Object.freeze({
    ...portable,
    workgroupsX,
    workgroupsZ,
  });
}

export function aceFp16VaeConvTranspose1dWgsl(
  shape: AceVaeConvTranspose1dShape,
): string {
  const plan = planAceFp16VaeConvTranspose1d(shape);
  return /* wgsl */ `
// kernel-id: ${ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID}
enable f16;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const KERNEL_SIZE: u32 = ${plan.kernelSize}u;
const STRIDE: u32 = ${plan.stride}u;
const DILATION: u32 = ${plan.dilation}u;
const PADDING: u32 = ${plan.padding}u;
const INPUT_CHANNEL_CHUNKS: u32 = ${plan.inputChannelChunkCount}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

struct OutputRangeParameters {
  first_output: u32,
  output_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(4) var<uniform>
  output_range: OutputRangeParameters;

var<workgroup> input_tile: array<f16, ${plan.inputTileElements}>;
var<workgroup> weight_tile: array<f16, ${plan.weightTileElements}>;

@compute @workgroup_size(
  ${ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_X},
  ${ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_Y},
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
    range_first_time + group.x * ${ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES}u;
  let output_time = tile_first_time + local.x;
  let output_channel =
    group.y * ${ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS}u + local.y;
  let range_end_time = range_first_time + output_row_count;
  let output_active =
    output_time < range_end_time && output_channel < OUTPUT_CHANNELS;
  var sum: f32 = 0.0;
  if (output_active) {
    let bias_operand: f32 = f32(bias[output_channel]);
    sum = bias_operand;
  }

  // Increasing chunks concatenate into the exact kernel-outer,
  // increasing-Cin scalar order while f16 workgroup storage reuses operands.
  for (var kernel = 0u; kernel < KERNEL_SIZE; kernel += 1u) {
    let kernel_time = kernel * DILATION;
    for (
      var input_channel_chunk = 0u;
      input_channel_chunk < INPUT_CHANNEL_CHUNKS;
      input_channel_chunk += 1u
    ) {
      let chunk_first_channel = input_channel_chunk *
        ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
      for (
        var tile_index = lane;
        tile_index < ${ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES * ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
        tile_index += ${ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE}u
      ) {
        let tile_time = tile_index /
          ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
        let chunk_channel = tile_index %
          ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
        let input_channel = chunk_first_channel + chunk_channel;
        let staged_output_time = tile_first_time + tile_time;
        let padded_output_time = staged_output_time + PADDING;
        var value: f16 = f16(0.0);
        if (
          input_channel < INPUT_CHANNELS &&
          padded_output_time >= kernel_time
        ) {
          let input_numerator = padded_output_time - kernel_time;
          if ((input_numerator % STRIDE) == 0u) {
            let input_time = input_numerator / STRIDE;
            if (input_time < INPUT_FRAMES) {
              value = input[
                (batch * INPUT_FRAMES + input_time) * INPUT_CHANNELS +
                input_channel
              ];
            }
          }
        }
        input_tile[
          chunk_channel *
            ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE}u +
          tile_time
        ] = value;
      }
      for (
        var tile_index = lane;
        tile_index < ${ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS * ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
        tile_index += ${ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE}u
      ) {
        let tile_output_channel = tile_index /
          ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
        let chunk_channel = tile_index %
          ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
        let input_channel = chunk_first_channel + chunk_channel;
        let weight_output_channel =
          group.y * ${ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS}u +
          tile_output_channel;
        var value: f16 = f16(0.0);
        if (
          weight_output_channel < OUTPUT_CHANNELS &&
          input_channel < INPUT_CHANNELS
        ) {
          // Converter-native [output_channel, kernel, input_channel] order.
          value = weight[
            (weight_output_channel * KERNEL_SIZE + kernel) * INPUT_CHANNELS +
            input_channel
          ];
        }
        weight_tile[
          tile_output_channel *
            ${ACE_FP16_VAE_CONV_TRANSPOSE1D_WEIGHT_TILE_STRIDE}u +
          chunk_channel
        ] = value;
      }
      workgroupBarrier();

      if (output_active) {
        let padded_output_time = output_time + PADDING;
        // Invalid inverse-stride and padded taps skip the Cin loop entirely;
        // they are never represented as multiply-by-zero operations.
        if (padded_output_time >= kernel_time) {
          let input_numerator = padded_output_time - kernel_time;
          if ((input_numerator % STRIDE) == 0u) {
            let input_time = input_numerator / STRIDE;
            if (input_time < INPUT_FRAMES) {
              let chunk_channel_count = min(
                ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u,
                INPUT_CHANNELS - chunk_first_channel
              );
              let weight_base = local.y *
                ${ACE_FP16_VAE_CONV_TRANSPOSE1D_WEIGHT_TILE_STRIDE}u;
              for (
                var chunk_channel = 0u;
                chunk_channel < chunk_channel_count;
                chunk_channel += 1u
              ) {
                let input_operand: f32 = f32(input_tile[
                  chunk_channel *
                    ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE}u +
                  local.x
                ]);
                let weight_operand: f32 = f32(
                  weight_tile[weight_base + chunk_channel]
                );
                sum = sum + input_operand * weight_operand;
              }
            }
          }
        }
      }
      workgroupBarrier();
    }
  }

  if (output_active) {
    let output_row = batch * OUTPUT_FRAMES + output_time;
    output[output_row * OUTPUT_CHANNELS + output_channel] = f16(sum);
  }
}
`;
}

/**
 * Exact polyphase specialization for the frozen K=2*stride geometry.
 *
 * Every output has at most two congruent taps: `r`, then `r + stride`.
 * Workgroup Z selects `r` indirectly through an output-time phase, so the 16
 * rows in a tile share both weights while retaining the portable kernel's
 * increasing valid-kernel then increasing-Cin FP32 accumulation order.
 */
export function aceFp16VaeCongruentConvTranspose1dWgsl(
  shape: AceVaeConvTranspose1dShape,
): string {
  const plan = planAceFp16VaeConvTranspose1d(shape);
  return /* wgsl */ `
// kernel-id: ${ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID}
enable f16;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const KERNEL_SIZE: u32 = ${plan.kernelSize}u;
const STRIDE: u32 = ${plan.stride}u;
const PADDING: u32 = ${plan.padding}u;
const INPUT_CHANNEL_CHUNKS: u32 = ${plan.inputChannelChunkCount}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

struct OutputRangeParameters {
  first_output: u32,
  output_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(4) var<uniform>
  output_range: OutputRangeParameters;

var<workgroup> input_tile: array<f16, ${plan.inputTileElements}>;
var<workgroup> weight_tile: array<f16, ${plan.weightTileElements}>;

@compute @workgroup_size(
  ${ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_X},
  ${ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_Y},
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

  // Z selects an output-time phase relative to this immutable range. Rows in
  // one workgroup advance by STRIDE, hence share the same congruent kernels.
  let phase = group.z;
  let phase_first_padded_time = range_first_time + phase + PADDING;
  let congruent_kernel = phase_first_padded_time % STRIDE;
  let phase_first_input_time = phase_first_padded_time / STRIDE;
  let tile_first_phase_row =
    group.x * ${ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES}u;
  let output_phase_row = tile_first_phase_row + local.x;
  let output_range_offset = phase + output_phase_row * STRIDE;
  let output_time = range_first_time + output_range_offset;
  let output_channel =
    group.y * ${ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS}u + local.y;
  let output_active =
    output_range_offset < output_row_count &&
    output_channel < OUTPUT_CHANNELS;
  var sum: f32 = 0.0;
  if (output_active) {
    sum = f32(bias[output_channel]);
  }

  // For KERNEL_SIZE == 2*STRIDE these are exactly the only possible valid
  // taps, in the same increasing-kernel order as the portable oracle.
  for (var tap = 0u; tap < 2u; tap += 1u) {
    let kernel = congruent_kernel + tap * STRIDE;
    for (
      var input_channel_chunk = 0u;
      input_channel_chunk < INPUT_CHANNEL_CHUNKS;
      input_channel_chunk += 1u
    ) {
      let chunk_first_channel = input_channel_chunk *
        ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
      for (
        var tile_index = lane;
        tile_index < ${ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES * ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
        tile_index += ${ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE}u
      ) {
        let tile_time = tile_index /
          ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
        let chunk_channel = tile_index %
          ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
        let input_channel = chunk_first_channel + chunk_channel;
        let tile_phase_row = tile_first_phase_row + tile_time;
        let tile_range_offset = phase + tile_phase_row * STRIDE;
        var value: f16 = f16(0.0);
        if (
          tile_range_offset < output_row_count &&
          input_channel < INPUT_CHANNELS
        ) {
          let first_input_time = phase_first_input_time + tile_phase_row;
          if (tap == 0u) {
            if (first_input_time < INPUT_FRAMES) {
              value = input[
                (batch * INPUT_FRAMES + first_input_time) * INPUT_CHANNELS +
                input_channel
              ];
            }
          } else if (first_input_time > 0u) {
            let input_time = first_input_time - 1u;
            if (input_time < INPUT_FRAMES) {
              value = input[
                (batch * INPUT_FRAMES + input_time) * INPUT_CHANNELS +
                input_channel
              ];
            }
          }
        }
        input_tile[
          chunk_channel *
            ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE}u +
          tile_time
        ] = value;
      }
      for (
        var tile_index = lane;
        tile_index < ${ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS * ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
        tile_index += ${ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE}u
      ) {
        let tile_output_channel = tile_index /
          ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
        let chunk_channel = tile_index %
          ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
        let input_channel = chunk_first_channel + chunk_channel;
        let weight_output_channel =
          group.y * ${ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS}u +
          tile_output_channel;
        var value: f16 = f16(0.0);
        if (
          weight_output_channel < OUTPUT_CHANNELS &&
          input_channel < INPUT_CHANNELS
        ) {
          value = weight[
            (weight_output_channel * KERNEL_SIZE + kernel) * INPUT_CHANNELS +
            input_channel
          ];
        }
        weight_tile[
          tile_output_channel *
            ${ACE_FP16_VAE_CONV_TRANSPOSE1D_WEIGHT_TILE_STRIDE}u +
          chunk_channel
        ] = value;
      }
      workgroupBarrier();

      var input_valid = false;
      let first_input_time = phase_first_input_time + output_phase_row;
      if (tap == 0u) {
        input_valid = first_input_time < INPUT_FRAMES;
      } else if (first_input_time > 0u) {
        input_valid = first_input_time - 1u < INPUT_FRAMES;
      }
      if (output_active && input_valid) {
        let chunk_channel_count = min(
          ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u,
          INPUT_CHANNELS - chunk_first_channel
        );
        let weight_base = local.y *
          ${ACE_FP16_VAE_CONV_TRANSPOSE1D_WEIGHT_TILE_STRIDE}u;
        for (
          var chunk_channel = 0u;
          chunk_channel < chunk_channel_count;
          chunk_channel += 1u
        ) {
          let input_operand = f32(input_tile[
            chunk_channel *
              ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE}u +
            local.x
          ]);
          let weight_operand = f32(
            weight_tile[weight_base + chunk_channel]
          );
          sum = sum + input_operand * weight_operand;
        }
      }
      workgroupBarrier();
    }
  }

  if (output_active) {
    let output_row = batch * OUTPUT_FRAMES + output_time;
    output[output_row * OUTPUT_CHANNELS + output_channel] = f16(sum);
  }
}
`;
}

async function compileAceFp16VaeConvTranspose1d(
  device: GPUDevice,
  plan: AceFp16VaeConvTranspose1dPlan,
): Promise<CompiledAceFp16VaeConvTranspose1d> {
  return compileAceFp16VaeConvTranspose1dVariant(
    device,
    plan,
    ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
    aceFp16VaeConvTranspose1dWgsl(plan),
  );
}

async function compileAceFp16VaeCongruentConvTranspose1d(
  device: GPUDevice,
  plan: AceFp16VaeConvTranspose1dPlan,
): Promise<CompiledAceFp16VaeConvTranspose1d> {
  return compileAceFp16VaeConvTranspose1dVariant(
    device,
    plan,
    ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID,
    aceFp16VaeCongruentConvTranspose1dWgsl(plan),
  );
}

async function compileAceFp16VaeConvTranspose1dVariant(
  device: GPUDevice,
  plan: AceFp16VaeConvTranspose1dPlan,
  kernelId: AceFp16VaeConvTranspose1dKernelId,
  code: string,
): Promise<CompiledAceFp16VaeConvTranspose1d> {
  const label = `${kernelId}-${transposeKey(plan)}`;
  const module = device.createShaderModule({
    label,
    code,
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `ACE FP16 VAE ConvTranspose1D WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  const dataBindingSizes = [
    plan.inputBindingBytes,
    plan.weightBindingBytes,
    plan.biasBindingBytes,
    plan.outputBindingBytes,
  ];
  const outputBinding = 3;
  const rangeBinding = 4;
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

function requireSupportedShape(shape: AceVaeConvTranspose1dShape): void {
  if (
    !ACE_FP16_VAE_CONV_TRANSPOSE1D_SUPPORTED_STRIDES.includes(
      shape.stride as 2 | 4 | 6 | 10,
    ) ||
    shape.kernelSize !== 2 * shape.stride ||
    shape.dilation !== 1 ||
    shape.padding !== Math.ceil(shape.stride / 2) ||
    shape.outputPadding !== 0
  ) {
    throw new RangeError(
      "ACE FP16 VAE ConvTranspose1D requires stride 2,4,6,10 with kernel=2*stride, dilation1, padding=ceil(stride/2), and outputPadding0",
    );
  }
}

function requireKernelDevice(device: GPUDevice): void {
  if (!device.features.has("shader-f16")) {
    throw new Error(
      "ACE FP16 VAE ConvTranspose1D requires WebGPU shader-f16",
    );
  }
  const maximumInvocations = Number(
    device.limits.maxComputeInvocationsPerWorkgroup,
  );
  const maximumSizeX = Number(device.limits.maxComputeWorkgroupSizeX);
  const maximumSizeY = Number(device.limits.maxComputeWorkgroupSizeY);
  if (
    !Number.isSafeInteger(maximumInvocations) ||
    !Number.isSafeInteger(maximumSizeX) ||
    !Number.isSafeInteger(maximumSizeY) ||
    maximumInvocations < ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE ||
    maximumSizeX < ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_X ||
    maximumSizeY < ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_Y
  ) {
    throw new Error(
      "ACE FP16 VAE ConvTranspose1D requires a 16x8 (128-lane) compute workgroup",
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
  const alignment = Number(device.limits.minStorageBufferOffsetAlignment);
  if (!isValidGpuAlignment(alignment)) {
    throw new Error(
      "ACE FP16 VAE ConvTranspose1D device reported an invalid storage alignment",
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
  const alignment = Number(device.limits.minUniformBufferOffsetAlignment);
  if (!isValidGpuAlignment(alignment)) {
    throw new Error(
      "ACE FP16 VAE ConvTranspose1D device reported an invalid uniform alignment",
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

function transposeKey(plan: AceFp16VaeConvTranspose1dPlan): string {
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

function checkedStorageBytes(elements: number, label: string): number {
  const bytes = elements * FLOAT16_BYTES;
  if (!Number.isSafeInteger(bytes) || bytes < 1) {
    throw new RangeError(
      `ACE FP16 VAE ConvTranspose1D ${label} storage bytes are not a positive safe integer`,
    );
  }
  return bytes;
}

function checkedSum(left: number, right: number, label: string): number {
  if (
    !Number.isSafeInteger(left) || left < 0 ||
    !Number.isSafeInteger(right) || right < 0 ||
    !Number.isSafeInteger(left + right)
  ) {
    throw new RangeError(
      `ACE FP16 VAE ConvTranspose1D ${label} is not a safe integer`,
    );
  }
  return left + right;
}

function alignGpuBindingBytes(bytes: number, label: string): number {
  const rounded = Math.ceil(bytes / GPU_BUFFER_ALIGNMENT) *
    GPU_BUFFER_ALIGNMENT;
  if (!Number.isSafeInteger(rounded) || rounded < bytes) {
    throw new RangeError(
      `ACE FP16 VAE ConvTranspose1D ${label} binding bytes are not a safe integer`,
    );
  }
  return rounded;
}

function requireWgslIndexable(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > MAX_WGSL_U32) {
    throw new RangeError(
      `ACE FP16 VAE ConvTranspose1D ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
