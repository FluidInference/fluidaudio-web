import {
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dCongruentRange,
} from "./vae-conv-transpose1d-fp16.js";
import type {
  AceVaeConvTranspose1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID =
  "ace-vae-fp16-fixed32-subgroup-4row-128channel-conv-transpose1d-v1" as const;
export const ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID =
  "conv-transpose1d-phase-tap-input-output-f16-v1" as const;
export const ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE = 32;
export const ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP = 4;
export const ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE = 128;
export const ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP = 4;
export const ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP = 16;
export const ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE = 4;
export const ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP = 128;
export const ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK = 8;

const OUTPUT_RANGE_CONTROL_BYTES = 16;

export interface AceOpt0026VaeConvTranspose1dPlan
  extends AceVaeConvTranspose1dShape {
  readonly kernelId: typeof ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID;
  readonly weightLayout:
    typeof ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID;
  readonly outputFrames: number;
  readonly inputElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly inputBindingBytes: number;
  readonly weightBindingBytes: number;
  readonly biasBindingBytes: number;
  readonly outputBindingBytes: number;
  readonly inputStorageBytes: number;
  readonly weightStorageBytes: number;
  readonly biasStorageBytes: number;
  readonly outputStorageBytes: number;
  readonly inputChannelChunkCount: number;
  readonly rowsPerSubgroup:
    typeof ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP;
  readonly rowsPerWorkgroup:
    typeof ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP;
  readonly channelsPerLane:
    typeof ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE;
  readonly channelsPerSubgroup:
    typeof ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP;
  readonly workgroupSize:
    typeof ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE;
  readonly workgroupStorageBytes: 0;
  readonly workgroupBarrierCount: 0;
}

export interface AceOpt0026VaeConvTranspose1dRangePlan {
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

export interface AceOpt0026VaeConvTranspose1dBindings {
  readonly input: GPUBufferBinding;
  /** Converter-native FP16 `[phase,2,input,output]`. */
  readonly polyphaseWeight: GPUBufferBinding;
  readonly bias: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceOpt0026VaeConvTranspose1dDispatch {
  readonly label: string;
  readonly kernelId: typeof ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID;
  readonly plan: AceOpt0026VaeConvTranspose1dPlan;
  readonly outputRange: AceOpt0026VaeConvTranspose1dRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/** Exact measured OPT-0026 kernel owner for production cooperative ranges. */
export class AceOpt0026VaeConvTranspose1dKernel {
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
  ): AceOpt0026VaeConvTranspose1dKernel {
    if (
      !device.features.has("shader-f16") ||
      !device.features.has("subgroups") ||
      capability.subgroupMinSize !==
        ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE ||
      capability.subgroupMaxSize !==
        ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE ||
      Number(device.limits.maxComputeInvocationsPerWorkgroup) <
        ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE ||
      Number(device.limits.maxComputeWorkgroupSizeX) <
        ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE
    ) {
      throw new Error(
        "OPT-0026 ConvTranspose1D requires shader-f16, fixed32 subgroups, and WG128",
      );
    }
    return new AceOpt0026VaeConvTranspose1dKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConvTranspose1dShape,
    bindings: AceOpt0026VaeConvTranspose1dBindings,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0026VaeConvTranspose1dDispatch> {
    this.requireLive();
    const plan = planAceOpt0026VaeConvTranspose1d(shape);
    const outputRange = planAceOpt0026VaeConvTranspose1dRange(plan, range);
    const maximumDispatch = Number(
      this.device.limits.maxComputeWorkgroupsPerDimension,
    );
    if (
      !Number.isSafeInteger(maximumDispatch) || maximumDispatch < 1 ||
      outputRange.workgroupsX > maximumDispatch ||
      outputRange.workgroupsY > maximumDispatch ||
      outputRange.workgroupsZ > maximumDispatch
    ) {
      throw new RangeError("OPT-0026 dispatch exceeds the device dimension");
    }
    const resources = [
      normalizeStorageBinding(this.device, bindings.input, plan.inputBindingBytes, `${label} input`),
      normalizeStorageBinding(this.device, bindings.polyphaseWeight, plan.weightBindingBytes, `${label} polyphase weight`),
      normalizeStorageBinding(this.device, bindings.bias, plan.biasBindingBytes, `${label} bias`),
      normalizeStorageBinding(this.device, bindings.output, plan.outputBindingBytes, `${label} output`),
    ] as const;
    const controlOffset = normalizeRangeOffset(this.device, range.control, label);
    const compiled = await this.pipelineFor(plan);
    this.requireLive();
    const controlResource = Object.freeze({
      buffer: range.control.buffer,
      offset: 0,
      size: OUTPUT_RANGE_CONTROL_BYTES,
    });
    const key = `${planKey(plan)}:${[...resources, controlResource].map((binding) =>
      this.bindingKey(binding)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(key);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-opt-0026-bindings`,
        layout: compiled.bindGroupLayout,
        entries: [...resources, controlResource].map((resource, binding) => ({
          binding,
          resource,
        })),
      });
      this.bindGroups.set(key, bindGroup);
    }
    const owner = this;
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
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
    plan: AceOpt0026VaeConvTranspose1dPlan,
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
      id = this.nextBufferId++;
      this.bufferIds.set(binding.buffer, id);
    }
    return `${id}:${binding.offset ?? 0}:${binding.size ?? -1}`;
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0026 ConvTranspose1D kernel was destroyed");
    }
  }
}

/**
 * Benchmark-only OPT-0026 ownership plan.
 *
 * Unlike OPT-0022, every subgroup spans the complete 128-channel output tile.
 * Its 32 lanes own four adjacent channels each and four phase-congruent rows;
 * the four subgroups own disjoint rows. One broadcast input operand therefore
 * feeds 128 output channels instead of 32.
 */
export function planAceOpt0026VaeConvTranspose1d(
  shape: AceVaeConvTranspose1dShape,
): AceOpt0026VaeConvTranspose1dPlan {
  const base = planAceFp16VaeConvTranspose1d(shape);
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
    kernelId: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
    weightLayout: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID,
    outputFrames: base.outputFrames,
    inputElements: base.inputElements,
    weightElements: base.weightElements,
    outputElements: base.outputElements,
    inputBindingBytes: base.inputBindingBytes,
    weightBindingBytes: base.weightBindingBytes,
    biasBindingBytes: base.biasBindingBytes,
    outputBindingBytes: base.outputBindingBytes,
    inputStorageBytes: base.inputStorageBytes,
    weightStorageBytes: base.weightStorageBytes,
    biasStorageBytes: base.biasStorageBytes,
    outputStorageBytes: base.outputStorageBytes,
    inputChannelChunkCount: Math.ceil(
      base.inputChannels /
        ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK,
    ),
    rowsPerSubgroup: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP,
    rowsPerWorkgroup: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP,
    channelsPerLane: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE,
    channelsPerSubgroup:
      ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP,
    workgroupSize: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE,
    workgroupStorageBytes: 0,
    workgroupBarrierCount: 0,
  });
}

export function planAceOpt0026VaeConvTranspose1dRange(
  plan: AceOpt0026VaeConvTranspose1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceOpt0026VaeConvTranspose1dRangePlan {
  const base = planAceFp16VaeConvTranspose1dCongruentRange(
    planAceFp16VaeConvTranspose1d(plan),
    range,
  );
  return Object.freeze({
    base: base.base,
    count: base.count,
    batch: base.batch,
    firstOutputTime: base.firstOutputTime,
    firstOutputRow: base.firstOutputRow,
    outputRowCount: base.outputRowCount,
    workgroupsX: Math.ceil(
      base.outputRowCount /
        (ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP * plan.stride),
    ),
    workgroupsY: Math.ceil(
      plan.outputChannels /
        ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP,
    ),
    workgroupsZ: plan.stride,
  });
}

export function aceOpt0026NativeWeightIndex(
  shape: AceVaeConvTranspose1dShape,
  outputChannel: number,
  kernel: number,
  inputChannel: number,
): number {
  requireCoordinate(outputChannel, shape.outputChannels, "output channel");
  requireCoordinate(kernel, shape.kernelSize, "kernel");
  requireCoordinate(inputChannel, shape.inputChannels, "input channel");
  return (outputChannel * shape.kernelSize + kernel) * shape.inputChannels +
    inputChannel;
}

export function aceOpt0026PolyphaseWeightIndex(
  shape: AceVaeConvTranspose1dShape,
  phase: number,
  tap: number,
  inputChannel: number,
  outputChannel: number,
): number {
  requireCoordinate(phase, shape.stride, "phase");
  requireCoordinate(tap, 2, "tap");
  requireCoordinate(inputChannel, shape.inputChannels, "input channel");
  requireCoordinate(outputChannel, shape.outputChannels, "output channel");
  return ((phase * 2 + tap) * shape.inputChannels + inputChannel) *
    shape.outputChannels + outputChannel;
}

/** Exact raw-U16 O-K-I -> phase/tap/input/output permutation. */
export function packAceOpt0026VaeConvTranspose1dWeights(
  native: Uint16Array,
  shape: AceVaeConvTranspose1dShape,
): Uint16Array {
  requireWeightWords(native, shape, "native");
  const packed = new Uint16Array(native.length);
  for (let phase = 0; phase < shape.stride; phase += 1) {
    for (let tap = 0; tap < 2; tap += 1) {
      const kernel = phase + tap * shape.stride;
      for (let inputChannel = 0; inputChannel < shape.inputChannels;
        inputChannel += 1) {
        for (let outputChannel = 0; outputChannel < shape.outputChannels;
          outputChannel += 1) {
          packed[aceOpt0026PolyphaseWeightIndex(
            shape,
            phase,
            tap,
            inputChannel,
            outputChannel,
          )] = native[aceOpt0026NativeWeightIndex(
            shape,
            outputChannel,
            kernel,
            inputChannel,
          )]!;
        }
      }
    }
  }
  return packed;
}

/** Exact inverse permutation, used by the primitive gate. */
export function unpackAceOpt0026VaeConvTranspose1dWeights(
  packed: Uint16Array,
  shape: AceVaeConvTranspose1dShape,
): Uint16Array {
  requireWeightWords(packed, shape, "polyphase");
  const native = new Uint16Array(packed.length);
  for (let phase = 0; phase < shape.stride; phase += 1) {
    for (let tap = 0; tap < 2; tap += 1) {
      const kernel = phase + tap * shape.stride;
      for (let inputChannel = 0; inputChannel < shape.inputChannels;
        inputChannel += 1) {
        for (let outputChannel = 0; outputChannel < shape.outputChannels;
          outputChannel += 1) {
          native[aceOpt0026NativeWeightIndex(
            shape,
            outputChannel,
            kernel,
            inputChannel,
          )] = packed[aceOpt0026PolyphaseWeightIndex(
            shape,
            phase,
            tap,
            inputChannel,
            outputChannel,
          )]!;
        }
      }
    }
  }
  return native;
}

export function aceOpt0026VaeConvTranspose1dWgsl(
  shape: AceVaeConvTranspose1dShape,
): string {
  const plan = planAceOpt0026VaeConvTranspose1d(shape);
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID}
// weight-layout: ${ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_POLYPHASE_LAYOUT_ID}
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
@group(0) @binding(4) var<uniform> output_range: OutputRangeParameters;

fn load_bias4(channel: u32) -> vec4<f32> {
  var result = vec4<f32>(0.0);
${loadVectorComponentsWgsl("result", "bias", "channel", false)}
  return result;
}

fn load_weight4(base: u32, channel: u32) -> vec4<f32> {
  var result = vec4<f32>(0.0);
${loadVectorComponentsWgsl("result", "polyphase_weight", "channel", true)}
  return result;
}

@compute @workgroup_size(${ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (
    subgroup_size != ${ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE}u ||
    subgroup >= ${ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP}u
  ) { return; }

  let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
  let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
  let range_first_time = first_output_row % OUTPUT_FRAMES;
  let batch = first_output_row / OUTPUT_FRAMES;
  let phase = group.z;
  let phase_first_padded_time = range_first_time + phase + PADDING;
  let congruent_kernel = phase_first_padded_time % STRIDE;
  let phase_first_input_time = phase_first_padded_time / STRIDE;
  let subgroup_first_phase_row =
    group.x * ${ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP}u +
    subgroup * ${ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP}u;
  let output_channel =
    group.y * ${ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_SUBGROUP}u +
    lane * ${ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE}u;
  let initial_sum = load_bias4(output_channel);
${initialRowsWgsl()}

  // Each component keeps the exact tap-then-increasing-Cin FP32 term order.
  for (var tap = 0u; tap < 2u; tap += 1u) {
${tapValidityWgsl()}
    for (var chunk = 0u; chunk < INPUT_CHANNEL_CHUNKS; chunk += 1u) {
      let chunk_first_channel = chunk *
        ${ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK}u;
${inputChannelMembersWgsl()}
    }
  }
${storesWgsl()}
}
`;
}

function loadVectorComponentsWgsl(
  destination: string,
  source: string,
  channel: string,
  useBase: boolean,
): string {
  return ["x", "y", "z", "w"].map((component, offset) => `
  if (${channel} + ${offset}u < OUTPUT_CHANNELS) {
    ${destination}.${component} = f32(${source}[
      ${useBase ? `base + ${channel} + ${offset}u` : `${channel} + ${offset}u`}
    ]);
  }`).join("");
}

function initialRowsWgsl(): string {
  return Array.from(
    { length: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => `
  let phase_row${row} = subgroup_first_phase_row + ${row}u;
  let range_offset${row} = phase + phase_row${row} * STRIDE;
  let output_time${row} = range_first_time + range_offset${row};
  let row_active${row} = range_offset${row} < output_row_count;
  var sum${row} = initial_sum;`,
  ).join("");
}

function tapValidityWgsl(): string {
  return Array.from(
    { length: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => `
    let first_input_time${row} = phase_first_input_time + phase_row${row};
    var input_time${row} = 0u;
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

function inputChannelMembersWgsl(): string {
  return Array.from(
    { length: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK },
    (_, member) => `
      let input_channel${member} = chunk_first_channel + ${member}u;
      let weight_base${member} =
        ((congruent_kernel * 2u + tap) * INPUT_CHANNELS +
          input_channel${member}) * OUTPUT_CHANNELS;
      var weight${member} = vec4<f32>(0.0);
      if (input_channel${member} < INPUT_CHANNELS) {
        weight${member} = load_weight4(weight_base${member}, output_channel);
      }
${rowBroadcastAndAddWgsl(member)}`,
  ).join("");
}

function rowBroadcastAndAddWgsl(member: number): string {
  return Array.from(
    { length: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => `
      var lane_input${row}_${member} = 0.0;
      if (
        row_active${row} && input_valid${row} &&
        lane == ${member}u && input_channel${member} < INPUT_CHANNELS
      ) {
        lane_input${row}_${member} = f32(input[
          (batch * INPUT_FRAMES + input_time${row}) * INPUT_CHANNELS +
          input_channel${member}
        ]);
      }
      let input_operand${row}_${member} =
        subgroupBroadcast(lane_input${row}_${member}, ${member}u);
      if (
        row_active${row} && input_valid${row} &&
        input_channel${member} < INPUT_CHANNELS
      ) {
        sum${row} = sum${row} +
          vec4<f32>(input_operand${row}_${member}) * weight${member};
      }`,
  ).join("");
}

function storesWgsl(): string {
  return Array.from(
    { length: ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => ["x", "y", "z", "w"].map((component, offset) => `
  if (row_active${row} && output_channel + ${offset}u < OUTPUT_CHANNELS) {
    output[
      (batch * OUTPUT_FRAMES + output_time${row}) * OUTPUT_CHANNELS +
      output_channel + ${offset}u
    ] = f16(sum${row}.${component});
  }`).join(""),
  ).join("");
}

async function compileKernel(
  device: GPUDevice,
  plan: AceOpt0026VaeConvTranspose1dPlan,
): Promise<CompiledKernel> {
  const label = `${ACE_OPT_0026_VAE_CONV_TRANSPOSE1D_KERNEL_ID}-${planKey(plan)}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0026VaeConvTranspose1dWgsl(plan),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0026 ConvTranspose1D WGSL failed: ${errors.map(({ lineNum, linePos, message }) =>
        `${lineNum}:${linePos} ${message}`
      ).join("; ")}`,
    );
  }
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: [
      ...[
        plan.inputBindingBytes,
        plan.weightBindingBytes,
        plan.biasBindingBytes,
        plan.outputBindingBytes,
      ].map((minBindingSize, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: binding === 3 ? "storage" as const : "read-only-storage" as const,
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
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: device.createPipelineLayout({
      label: `${label}-layout`,
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: { module, entryPoint: "main" },
  });
  return Object.freeze({ pipeline, bindGroupLayout });
}

function normalizeStorageBinding(
  device: GPUDevice,
  binding: GPUBufferBinding,
  requiredBytes: number,
  label: string,
): GPUBufferBinding {
  const offset = Number(binding.offset ?? 0);
  const available = Number(binding.size ?? binding.buffer.size - offset);
  const alignment = Number(device.limits.minStorageBufferOffsetAlignment);
  if (
    !Number.isSafeInteger(offset) || offset < 0 ||
    !Number.isSafeInteger(available) || available < requiredBytes ||
    offset + requiredBytes > binding.buffer.size ||
    !Number.isSafeInteger(alignment) || alignment < 4 ||
    offset % alignment !== 0
  ) {
    throw new RangeError(`${label} does not expose ${requiredBytes} aligned bytes`);
  }
  return Object.freeze({ buffer: binding.buffer, offset, size: requiredBytes });
}

function normalizeRangeOffset(
  device: GPUDevice,
  binding: GPUBufferBinding,
  label: string,
): number {
  const offset = Number(binding.offset ?? 0);
  const available = Number(binding.size ?? binding.buffer.size - offset);
  const alignment = Number(device.limits.minUniformBufferOffsetAlignment);
  if (
    !Number.isSafeInteger(offset) || offset < 0 || offset > 0xffff_ffff ||
    !Number.isSafeInteger(available) || available < OUTPUT_RANGE_CONTROL_BYTES ||
    offset + OUTPUT_RANGE_CONTROL_BYTES > binding.buffer.size ||
    !Number.isSafeInteger(alignment) || alignment < 4 || offset % alignment !== 0
  ) {
    throw new RangeError(`${label} range control is not dynamically aligned`);
  }
  return offset;
}

function planKey(plan: AceOpt0026VaeConvTranspose1dPlan): string {
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

function requireWeightWords(
  words: Uint16Array,
  shape: AceVaeConvTranspose1dShape,
  label: string,
): void {
  planAceFp16VaeConvTranspose1d(shape);
  const expected = shape.outputChannels * shape.kernelSize * shape.inputChannels;
  if (words.length !== expected) {
    throw new RangeError(`OPT-0026 ${label} weights require ${expected} U16 words`);
  }
}

function requireCoordinate(value: number, bound: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= bound) {
    throw new RangeError(`OPT-0026 ${label} is out of range`);
  }
}
