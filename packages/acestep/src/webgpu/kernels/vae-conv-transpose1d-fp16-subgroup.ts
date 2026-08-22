import {
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dCongruentRange,
  type AceFp16VaeConvTranspose1dPlan,
} from "./vae-conv-transpose1d-fp16.js";
import type {
  AceVaeConvTranspose1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID =
  "ace-vae-fp16-fixed32-subgroup-polyphase-conv-transpose1d-v1" as const;
export const ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID =
  "conv-transpose1d-phase-tap-input-output-f16-v1" as const;
export const ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE = 32;
export const ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE = 128;
export const ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP = 4;
export const ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP = 16;
export const ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP = 32;
export const ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP = 128;
export const ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_TAPS = 2;
export const ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK = 8;

const GPU_BUFFER_ALIGNMENT = 4;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_CONTROL_BYTES = 16;

export interface AceOpt0022VaeConvTranspose1dFixedSubgroupCapability {
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
}

export interface AceOpt0022VaeConvTranspose1dBindings {
  /** FP16 activation in frame-major NLC order. */
  readonly input: GPUBufferBinding;
  /** FP16 package weight in `[phase,2,input,output]` physical order. */
  readonly polyphaseWeight: GPUBufferBinding;
  /** FP16 `[output]`; every frozen decoder transpose is biased. */
  readonly bias: GPUBufferBinding;
  /** FP16 activation in frame-major NLC order. */
  readonly output: GPUBufferBinding;
}

export interface AceOpt0022VaeConvTranspose1dPlan
  extends AceVaeConvTranspose1dShape {
  readonly kernelId:
    typeof ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID;
  readonly weightLayout:
    typeof ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID;
  readonly outputFrames: number;
  readonly inputElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly inputStorageBytes: number;
  readonly inputBindingBytes: number;
  readonly polyphaseWeightStorageBytes: number;
  readonly polyphaseWeightBindingBytes: number;
  readonly biasStorageBytes: number;
  readonly biasBindingBytes: number;
  readonly outputStorageBytes: number;
  readonly outputBindingBytes: number;
  readonly subgroupSize:
    typeof ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE;
  readonly workgroupSize:
    typeof ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE;
  readonly subgroupsPerWorkgroup:
    typeof ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP;
  readonly rowsPerSubgroup:
    typeof ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP;
  readonly channelsPerSubgroup:
    typeof ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP;
  readonly channelsPerWorkgroup:
    typeof ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP;
  readonly taps: typeof ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_TAPS;
  readonly inputChannelChunk:
    typeof ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK;
  readonly inputChannelChunkCount: number;
  readonly workgroupStorageBytes: 0;
  readonly workgroupBarrierCount: 0;
}

export interface AceOpt0022VaeConvTranspose1dRangePlan {
  readonly base: number;
  readonly count: number;
  readonly batch: number;
  readonly firstOutputTime: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
  readonly workgroupsZ: number;
}

export interface AceOpt0022VaeConvTranspose1dDispatch {
  readonly label: string;
  readonly kernelId:
    typeof ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID;
  readonly plan: AceOpt0022VaeConvTranspose1dPlan;
  readonly outputRange: AceOpt0022VaeConvTranspose1dRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

interface NamedBinding {
  readonly name:
    | "input"
    | "polyphase weight"
    | "bias"
    | "output"
    | "range control";
  readonly binding: GPUBufferBinding;
}

/**
 * Isolated exact OPT-0022 benchmark owner.
 *
 * It accepts only converter-native polyphase weights and an explicitly
 * authenticated fixed-32 subgroup device. There is deliberately no native
 * O-K-I repack, portable fallback, decoder selection, or production profile.
 */
export class AceOpt0022VaeConvTranspose1dSubgroupKernel {
  private readonly pipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: AceOpt0022VaeConvTranspose1dFixedSubgroupCapability,
  ): AceOpt0022VaeConvTranspose1dSubgroupKernel {
    requireFixed32SubgroupDevice(device, capability);
    return new AceOpt0022VaeConvTranspose1dSubgroupKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConvTranspose1dShape,
    bindings: AceOpt0022VaeConvTranspose1dBindings,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0022VaeConvTranspose1dDispatch> {
    this.requireLive();
    const plan = planAceOpt0022VaeConvTranspose1d(shape);
    const outputRange = planAceOpt0022VaeConvTranspose1dRange(plan, range);
    requireDeviceLimits(this.device, plan, outputRange);
    const normalized = requireBindings(
      this.device,
      label,
      plan,
      bindings,
      range,
    );
    const compiled = await this.pipelineFor(plan);
    this.requireLive();

    const rangeBindingIndex = normalized.length - 1;
    const controlOffset = normalized[rangeBindingIndex]!.binding.offset ?? 0;
    const resources = normalized.map(({ binding }, index) =>
      index === rangeBindingIndex
        ? Object.freeze({
            buffer: binding.buffer,
            offset: 0,
            size: OUTPUT_RANGE_CONTROL_BYTES,
          })
        : binding
    );
    const pipelineKey = planKey(plan);
    const bindGroupKey = `${pipelineKey}:${resources.map((binding) =>
      this.bindingKey(binding)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-opt-0022-subgroup-polyphase-bindings`,
        layout: compiled.bindGroupLayout,
        entries: resources.map((resource, binding) => ({
          binding,
          resource,
        })),
      });
      this.bindGroups.set(bindGroupKey, bindGroup);
    }

    const owner = this;
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID,
      plan,
      outputRange,
      encode(pass: GPUComputePassEncoder): void {
        owner.requireLive();
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroup, [controlOffset]);
        pass.dispatchWorkgroups(
          outputRange.workgroupsX,
          outputRange.workgroupsY,
          outputRange.workgroupsZ,
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
    plan: AceOpt0022VaeConvTranspose1dPlan,
  ): Promise<CompiledKernel> {
    const key = planKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileKernel(this.device, plan);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
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
      throw new Error("OPT-0022 subgroup ConvTranspose1D kernel was destroyed");
    }
  }
}

export function planAceOpt0022VaeConvTranspose1d(
  shape: AceVaeConvTranspose1dShape,
): AceOpt0022VaeConvTranspose1dPlan {
  const base = planAceFp16VaeConvTranspose1d(shape);
  const inputChannelChunkCount = Math.ceil(
    base.inputChannels /
      ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK,
  );
  requireWgslIndexable(
    inputChannelChunkCount,
    "input channel chunk count",
  );
  return Object.freeze({
    batch: base.batch,
    inputFrames: base.inputFrames,
    inputChannels: base.inputChannels,
    outputChannels: base.outputChannels,
    kernelSize: base.kernelSize,
    stride: base.stride,
    dilation: base.dilation,
    padding: base.padding,
    outputPadding: base.outputPadding,
    kernelId: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID,
    weightLayout: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
    outputFrames: base.outputFrames,
    inputElements: base.inputElements,
    weightElements: base.weightElements,
    outputElements: base.outputElements,
    inputStorageBytes: base.inputStorageBytes,
    inputBindingBytes: base.inputBindingBytes,
    polyphaseWeightStorageBytes: base.weightStorageBytes,
    polyphaseWeightBindingBytes: base.weightBindingBytes,
    biasStorageBytes: base.biasStorageBytes,
    biasBindingBytes: base.biasBindingBytes,
    outputStorageBytes: base.outputStorageBytes,
    outputBindingBytes: base.outputBindingBytes,
    subgroupSize: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE,
    workgroupSize: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE,
    subgroupsPerWorkgroup:
      ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP,
    rowsPerSubgroup: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP,
    channelsPerSubgroup:
      ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP,
    channelsPerWorkgroup:
      ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP,
    taps: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_TAPS,
    inputChannelChunk:
      ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK,
    inputChannelChunkCount,
    workgroupStorageBytes: 0,
    workgroupBarrierCount: 0,
  });
}

export function planAceOpt0022VaeConvTranspose1dRange(
  plan: AceOpt0022VaeConvTranspose1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceOpt0022VaeConvTranspose1dRangePlan {
  const base = planAceFp16VaeConvTranspose1dCongruentRange(
    asBasePlan(plan),
    range,
  );
  const workgroupsY = Math.ceil(
    plan.outputChannels /
      ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP,
  );
  requireWgslIndexable(workgroupsY, "workgroups Y");
  return Object.freeze({
    base: base.base,
    count: base.count,
    batch: base.batch,
    firstOutputTime: base.firstOutputTime,
    firstOutputRow: base.firstOutputRow,
    outputRowCount: base.outputRowCount,
    workgroupsX: base.workgroupsX,
    workgroupsY,
    workgroupsZ: base.workgroupsZ,
  });
}

/** Emit the fixed WG128/four-subgroup exact candidate for one frozen shape. */
export function aceOpt0022VaeConvTranspose1dSubgroupWgsl(
  shape: AceVaeConvTranspose1dShape,
): string {
  const plan = planAceOpt0022VaeConvTranspose1d(shape);
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID}
// weight-layout: ${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID}
enable f16;
enable subgroups;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const STRIDE: u32 = ${plan.stride}u;
const PADDING: u32 = ${plan.padding}u;
const INPUT_CHANNEL_CHUNKS: u32 = ${plan.inputChannelChunkCount}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> polyphase_weight: array<f16>;
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

@compute @workgroup_size(${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (
    subgroup_size != ${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE}u ||
    subgroup >= ${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP}u
  ) {
    return;
  }

  let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
  let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
  let range_first_time = first_output_row % OUTPUT_FRAMES;
  let batch = first_output_row / OUTPUT_FRAMES;
  let phase = group.z;
  let phase_first_padded_time = range_first_time + phase + PADDING;
  let congruent_kernel = phase_first_padded_time % STRIDE;
  let phase_first_input_time = phase_first_padded_time / STRIDE;
  let tile_first_phase_row =
    group.x * ${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP}u;
  let output_channel =
    group.y * ${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP}u +
    subgroup * ${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP}u +
    subgroup_lane;
  let output_channel_active = output_channel < OUTPUT_CHANNELS;
  var initial_sum: f32 = 0.0;
  if (output_channel_active) {
    initial_sum = f32(bias[output_channel]);
  }
${initialRowsWgsl()}

  // Tap then increasing-Cin is the exact OPT-0015 scalar reduction order.
  for (var tap = 0u; tap < 2u; tap += 1u) {
${tapValidityWgsl()}
    for (
      var input_channel_chunk = 0u;
      input_channel_chunk < INPUT_CHANNEL_CHUNKS;
      input_channel_chunk += 1u
    ) {
      let chunk_first_channel = input_channel_chunk *
        ${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
${weightLoadsWgsl()}
${rowBroadcastsAndAddsWgsl()}
    }
  }

${storesWgsl()}
}
`;
}

function initialRowsWgsl(): string {
  return Array.from(
    { length: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => `
  let output_phase_row${row} = tile_first_phase_row + ${row}u;
  let output_range_offset${row} =
    phase + output_phase_row${row} * STRIDE;
  let output_time${row} = range_first_time + output_range_offset${row};
  let output_row_active${row} =
    output_range_offset${row} < output_row_count;
  let output_active${row} =
    output_channel_active && output_row_active${row};
  var sum${row}: f32 = initial_sum;`,
  ).join("");
}

function tapValidityWgsl(): string {
  return Array.from(
    { length: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => `
    let first_input_time${row} =
      phase_first_input_time + output_phase_row${row};
    var input_time${row}: u32 = 0u;
    var input_valid${row} = false;
    if (tap == 0u) {
      if (first_input_time${row} < INPUT_FRAMES) {
        input_time${row} = first_input_time${row};
        input_valid${row} = true;
      }
    } else if (first_input_time${row} > 0u) {
      input_time${row} = first_input_time${row} - 1u;
      input_valid${row} = input_time${row} < INPUT_FRAMES;
    }`,
  ).join("");
}

function weightLoadsWgsl(): string {
  return Array.from(
    { length: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK },
    (_, member) => `
      let input_channel${member} = chunk_first_channel + ${member}u;
      var weight_operand${member}: f16 = f16(0.0);
      if (
        output_channel_active && input_channel${member} < INPUT_CHANNELS
      ) {
        // Physical [phase, tap, input, output], contiguous in output.
        weight_operand${member} = polyphase_weight[
          ((congruent_kernel * 2u + tap) * INPUT_CHANNELS +
            input_channel${member}) * OUTPUT_CHANNELS + output_channel
        ];
      }`,
  ).join("");
}

function rowBroadcastsAndAddsWgsl(): string {
  return Array.from(
    { length: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => {
      const broadcasts = Array.from(
        { length: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK },
        (_, member) => `
      let input_operand${row}_${member} =
        subgroupBroadcast(lane_input${row}, ${member}u);
      if (
        output_active${row} && input_valid${row} &&
        input_channel${member} < INPUT_CHANNELS
      ) {
        sum${row} = sum${row} +
          input_operand${row}_${member} * f32(weight_operand${member});
      }`,
      ).join("");
      return `
      var lane_input${row}: f32 = 0.0;
      if (
        output_row_active${row} && input_valid${row} &&
        subgroup_lane <
          ${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u &&
        chunk_first_channel + subgroup_lane < INPUT_CHANNELS
      ) {
        lane_input${row} = f32(input[
          (batch * INPUT_FRAMES + input_time${row}) * INPUT_CHANNELS +
          chunk_first_channel + subgroup_lane
        ]);
      }
${broadcasts}`;
    },
  ).join("");
}

function storesWgsl(): string {
  return Array.from(
    { length: ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => `
  if (output_active${row}) {
    output[
      (batch * OUTPUT_FRAMES + output_time${row}) * OUTPUT_CHANNELS +
      output_channel
    ] = f16(sum${row});
  }`,
  ).join("");
}

async function compileKernel(
  device: GPUDevice,
  plan: AceOpt0022VaeConvTranspose1dPlan,
): Promise<CompiledKernel> {
  const label = `${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_KERNEL_ID}-${planKey(plan)}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0022VaeConvTranspose1dSubgroupWgsl(plan),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0022 subgroup ConvTranspose1D WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: [
      ...[
        plan.inputBindingBytes,
        plan.polyphaseWeightBindingBytes,
        plan.biasBindingBytes,
        plan.outputBindingBytes,
      ].map((minBindingSize, binding) => ({
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

function requireFixed32SubgroupDevice(
  device: GPUDevice,
  capability: AceOpt0022VaeConvTranspose1dFixedSubgroupCapability,
): void {
  if (!device.features.has("shader-f16")) {
    throw new Error(
      "OPT-0022 subgroup ConvTranspose1D requires WebGPU shader-f16",
    );
  }
  if (
    !device.features.has("subgroups") ||
    capability.subgroupMinSize !==
      ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE ||
    capability.subgroupMaxSize !==
      ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE
  ) {
    throw new Error(
      "OPT-0022 subgroup ConvTranspose1D requires reported fixed 32-lane subgroups",
    );
  }
  const maximumInvocations = Number(
    device.limits.maxComputeInvocationsPerWorkgroup,
  );
  const maximumSizeX = Number(device.limits.maxComputeWorkgroupSizeX);
  if (
    !Number.isSafeInteger(maximumInvocations) ||
    !Number.isSafeInteger(maximumSizeX) ||
    maximumInvocations < ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE ||
    maximumSizeX < ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE
  ) {
    throw new Error(
      `OPT-0022 subgroup ConvTranspose1D requires WG${ACE_OPT_0022_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE} in X`,
    );
  }
}

function requireDeviceLimits(
  device: GPUDevice,
  plan: AceOpt0022VaeConvTranspose1dPlan,
  range: AceOpt0022VaeConvTranspose1dRangePlan,
): void {
  const maximumDispatch = Number(
    device.limits.maxComputeWorkgroupsPerDimension,
  );
  if (
    !Number.isSafeInteger(maximumDispatch) || maximumDispatch < 1 ||
    range.workgroupsX > maximumDispatch ||
    range.workgroupsY > maximumDispatch ||
    range.workgroupsZ > maximumDispatch
  ) {
    throw new RangeError(
      "OPT-0022 subgroup ConvTranspose1D range exceeds the device dispatch dimension",
    );
  }
  const maximumStorageBinding = Number(
    device.limits.maxStorageBufferBindingSize,
  );
  const maximumUniformBinding = Number(
    device.limits.maxUniformBufferBindingSize,
  );
  const maximumBuffer = Number(device.limits.maxBufferSize);
  if (
    !Number.isSafeInteger(maximumStorageBinding) ||
    maximumStorageBinding < 1 ||
    !Number.isSafeInteger(maximumUniformBinding) ||
    maximumUniformBinding < OUTPUT_RANGE_CONTROL_BYTES ||
    !Number.isSafeInteger(maximumBuffer) || maximumBuffer < 1
  ) {
    throw new RangeError(
      "OPT-0022 subgroup ConvTranspose1D device reported invalid buffer limits",
    );
  }
  for (const [name, bytes] of [
    ["input", plan.inputBindingBytes],
    ["polyphase weight", plan.polyphaseWeightBindingBytes],
    ["bias", plan.biasBindingBytes],
    ["output", plan.outputBindingBytes],
  ] as const) {
    if (bytes > maximumStorageBinding || bytes > maximumBuffer) {
      throw new RangeError(
        `OPT-0022 subgroup ConvTranspose1D ${name} exceeds the device buffer limits`,
      );
    }
  }
}

function requireBindings(
  device: GPUDevice,
  label: string,
  plan: AceOpt0022VaeConvTranspose1dPlan,
  bindings: AceOpt0022VaeConvTranspose1dBindings,
  range: AceVaeOutputRangeBinding,
): readonly NamedBinding[] {
  const normalized = Object.freeze([
    Object.freeze({
      name: "input" as const,
      binding: requireStorageBinding(
        device,
        bindings.input,
        plan.inputStorageBytes,
        plan.inputBindingBytes,
        `${label} input`,
      ),
    }),
    Object.freeze({
      name: "polyphase weight" as const,
      binding: requireStorageBinding(
        device,
        bindings.polyphaseWeight,
        plan.polyphaseWeightStorageBytes,
        plan.polyphaseWeightBindingBytes,
        `${label} polyphase weight`,
      ),
    }),
    Object.freeze({
      name: "bias" as const,
      binding: requireStorageBinding(
        device,
        bindings.bias,
        plan.biasStorageBytes,
        plan.biasBindingBytes,
        `${label} bias`,
      ),
    }),
    Object.freeze({
      name: "output" as const,
      binding: requireStorageBinding(
        device,
        bindings.output,
        plan.outputStorageBytes,
        plan.outputBindingBytes,
        `${label} output`,
      ),
    }),
    Object.freeze({
      name: "range control" as const,
      binding: requireRangeControlBinding(
        device,
        range.control,
        `${label} range control`,
      ),
    }),
  ] satisfies readonly NamedBinding[]);
  requireDisjointBindings(normalized, label);
  return normalized;
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
      "OPT-0022 subgroup ConvTranspose1D device reported an invalid storage alignment",
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
      `${label} does not expose an aligned ${requiredStorageBytes}-byte payload in ${requiredBindingBytes} binding bytes`,
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
      "OPT-0022 subgroup ConvTranspose1D device reported an invalid uniform alignment",
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
      `${label} must expose an aligned ${OUTPUT_RANGE_CONTROL_BYTES}-byte immutable record`,
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

function asBasePlan(
  plan: AceOpt0022VaeConvTranspose1dPlan,
): AceFp16VaeConvTranspose1dPlan {
  return planAceFp16VaeConvTranspose1d(plan);
}

function planKey(plan: AceOpt0022VaeConvTranspose1dPlan): string {
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

function requireWgslIndexable(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > MAX_WGSL_U32) {
    throw new RangeError(
      `OPT-0022 subgroup ConvTranspose1D ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
