import {
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
} from "../../model/manifest.js";
import {
  ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_WEIGHT_TILE_STRIDE,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_X,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_Y,
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dCongruentRange,
  type AceFp16VaeConvTranspose1dCongruentRangePlan,
  type AceFp16VaeConvTranspose1dPlan,
} from "./vae-conv-transpose1d-fp16.js";
import type {
  AceVaeConvTranspose1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID =
  "ace-vae-fp16-portable-packed-polyphase-conv-transpose1d-v1" as const;
export const ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_LAYOUT_ID =
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT;

const OUTPUT_RANGE_CONTROL_BYTES = 16;

export interface AceOpt0028VaeConvTranspose1dPortablePackedPlan
  extends AceFp16VaeConvTranspose1dPlan {
  readonly kernelId:
    typeof ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID;
  readonly weightLayout:
    typeof ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_LAYOUT_ID;
  readonly packedWeightStorageShape: readonly [number, 2, number, number];
}

export interface AceOpt0028VaeConvTranspose1dPortablePackedBindings {
  readonly input: GPUBufferBinding;
  /** Revision-6 FP16 `[phase,tap,Cin,Cout]`. */
  readonly polyphaseWeight: GPUBufferBinding;
  readonly bias: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceOpt0028VaeConvTranspose1dPortablePackedDispatch {
  readonly label: string;
  readonly kernelId:
    typeof ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID;
  readonly plan: AceOpt0028VaeConvTranspose1dPortablePackedPlan;
  readonly outputRange: AceFp16VaeConvTranspose1dCongruentRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/** Portable workgroup-memory owner for revision-6 polyphase weights. */
export class AceOpt0028VaeConvTranspose1dPortablePackedKernel {
  private readonly pipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {
    requireKernelDevice(device);
  }

  static create(
    device: GPUDevice,
  ): AceOpt0028VaeConvTranspose1dPortablePackedKernel {
    return new AceOpt0028VaeConvTranspose1dPortablePackedKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConvTranspose1dShape,
    bindings: AceOpt0028VaeConvTranspose1dPortablePackedBindings,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0028VaeConvTranspose1dPortablePackedDispatch> {
    this.requireLive();
    const plan = planAceOpt0028VaeConvTranspose1dPortablePacked(shape);
    const outputRange = planAceFp16VaeConvTranspose1dCongruentRange(
      plan,
      range,
    );
    requireDevicePlan(this.device, plan, outputRange);
    const resources = Object.freeze([
      normalizeStorageBinding(
        this.device,
        bindings.input,
        plan.inputBindingBytes,
        `${label} input`,
      ),
      normalizeStorageBinding(
        this.device,
        bindings.polyphaseWeight,
        plan.weightBindingBytes,
        `${label} polyphase weight`,
      ),
      normalizeStorageBinding(
        this.device,
        bindings.bias,
        plan.biasBindingBytes,
        `${label} bias`,
      ),
      normalizeStorageBinding(
        this.device,
        bindings.output,
        plan.outputBindingBytes,
        `${label} output`,
      ),
    ] as const);
    requireDisjointOutput(resources[3], resources.slice(0, 3), label);
    const controlOffset = normalizeRangeOffset(this.device, range.control, label);
    const controlResource = Object.freeze({
      buffer: range.control.buffer,
      offset: 0,
      size: OUTPUT_RANGE_CONTROL_BYTES,
    });
    const compiled = await this.pipelineFor(plan);
    this.requireLive();
    const bindGroupResources = Object.freeze([
      ...resources,
      controlResource,
    ] as const);
    const key = `${planKey(plan)}:${bindGroupResources.map((binding) =>
      this.bindingKey(binding)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(key);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-portable-packed-transpose-bindings`,
        layout: compiled.bindGroupLayout,
        entries: bindGroupResources.map((resource, binding) => ({
          binding,
          resource,
        })),
      });
      this.bindGroups.set(key, bindGroup);
    }
    const owner = this;
    return Object.freeze({
      label,
      kernelId:
        ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID,
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
    plan: AceOpt0028VaeConvTranspose1dPortablePackedPlan,
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
      throw new Error(
        "OPT-0028 portable packed ConvTranspose1D kernel was destroyed",
      );
    }
  }
}

export function planAceOpt0028VaeConvTranspose1dPortablePacked(
  shape: AceVaeConvTranspose1dShape,
): AceOpt0028VaeConvTranspose1dPortablePackedPlan {
  const portable = planAceFp16VaeConvTranspose1d(shape);
  return Object.freeze({
    ...portable,
    kernelId:
      ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID,
    weightLayout:
      ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_LAYOUT_ID,
    packedWeightStorageShape: Object.freeze([
      shape.stride,
      2,
      shape.inputChannels,
      shape.outputChannels,
    ]) as readonly [number, 2, number, number],
  });
}

export function aceOpt0028VaeConvTranspose1dPortablePackedWeightIndex(
  shape: AceVaeConvTranspose1dShape,
  phase: number,
  tap: number,
  inputChannel: number,
  outputChannel: number,
): number {
  const plan = planAceOpt0028VaeConvTranspose1dPortablePacked(shape);
  requireCoordinate(phase, plan.stride, "phase");
  requireCoordinate(tap, 2, "tap");
  requireCoordinate(inputChannel, plan.inputChannels, "input channel");
  requireCoordinate(outputChannel, plan.outputChannels, "output channel");
  return ((phase * 2 + tap) * plan.inputChannels + inputChannel) *
    plan.outputChannels + outputChannel;
}

export function aceOpt0028VaeConvTranspose1dPortablePackedWgsl(
  shape: AceVaeConvTranspose1dShape,
): string {
  const plan = planAceOpt0028VaeConvTranspose1dPortablePacked(shape);
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID}
// weight-layout: ${ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_LAYOUT_ID}
enable f16;

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
  let phase = group.z;
  let phase_first_padded_time = range_first_time + phase + PADDING;
  let congruent_phase = phase_first_padded_time % STRIDE;
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
  if (output_active) { sum = f32(bias[output_channel]); }

  // Tap 0 then tap 1 are the only valid kernels, in increasing-kernel order.
  // Increasing chunks and members retain increasing-Cin FP32 accumulation.
  for (var tap = 0u; tap < 2u; tap += 1u) {
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
            ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE}u + tile_time
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
          let packed_index =
            ((congruent_phase * 2u + tap) * INPUT_CHANNELS + input_channel) *
              OUTPUT_CHANNELS + weight_output_channel;
          value = polyphase_weight[packed_index];
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
              ${ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE}u + local.x
          ]);
          let weight_operand = f32(weight_tile[weight_base + chunk_channel]);
          sum = sum + input_operand * weight_operand;
        }
      }
      workgroupBarrier();
    }
  }

  if (output_active) {
    output[(batch * OUTPUT_FRAMES + output_time) * OUTPUT_CHANNELS +
      output_channel] = f16(sum);
  }
}
`;
}

async function compileKernel(
  device: GPUDevice,
  plan: AceOpt0028VaeConvTranspose1dPortablePackedPlan,
): Promise<CompiledKernel> {
  const label =
    `${ACE_OPT_0028_VAE_CONV_TRANSPOSE1D_PORTABLE_PACKED_KERNEL_ID}-${planKey(plan)}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0028VaeConvTranspose1dPortablePackedWgsl(plan),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0028 portable packed ConvTranspose1D WGSL failed: ${errors.map(
        (message) => `${message.lineNum}:${message.linePos} ${message.message}`,
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

function requireKernelDevice(device: GPUDevice): void {
  if (!device.features.has("shader-f16")) {
    throw new Error(
      "OPT-0028 portable packed ConvTranspose1D requires shader-f16",
    );
  }
  if (
    Number(device.limits.maxComputeInvocationsPerWorkgroup) <
      ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE ||
    Number(device.limits.maxComputeWorkgroupSizeX) <
      ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_X ||
    Number(device.limits.maxComputeWorkgroupSizeY) <
      ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_Y
  ) {
    throw new Error(
      "OPT-0028 portable packed ConvTranspose1D requires a 16x8 workgroup",
    );
  }
}

function requireDevicePlan(
  device: GPUDevice,
  plan: AceOpt0028VaeConvTranspose1dPortablePackedPlan,
  range: AceFp16VaeConvTranspose1dCongruentRangePlan,
): void {
  if (
    Number(device.limits.maxComputeWorkgroupStorageSize) <
      plan.workgroupStorageBytes
  ) {
    throw new RangeError(
      `OPT-0028 portable packed ConvTranspose1D requires ${plan.workgroupStorageBytes} workgroup bytes`,
    );
  }
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
      "OPT-0028 portable packed ConvTranspose1D dispatch exceeds limits",
    );
  }
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
    !Number.isSafeInteger(available) ||
    available < OUTPUT_RANGE_CONTROL_BYTES ||
    offset + OUTPUT_RANGE_CONTROL_BYTES > binding.buffer.size ||
    !Number.isSafeInteger(alignment) || alignment < 4 ||
    offset % alignment !== 0
  ) {
    throw new RangeError(`${label} range control is not dynamically aligned`);
  }
  return offset;
}

function requireDisjointOutput(
  output: GPUBufferBinding,
  inputs: readonly GPUBufferBinding[],
  label: string,
): void {
  const outputStart = Number(output.offset ?? 0);
  const outputEnd = outputStart + Number(output.size);
  for (const input of inputs) {
    if (input.buffer !== output.buffer) continue;
    const inputStart = Number(input.offset ?? 0);
    const inputEnd = inputStart + Number(input.size);
    if (inputStart < outputEnd && outputStart < inputEnd) {
      throw new RangeError(`${label} output overlaps an input`);
    }
  }
}

function planKey(
  plan: AceOpt0028VaeConvTranspose1dPortablePackedPlan,
): string {
  return [
    plan.batch,
    plan.inputFrames,
    plan.inputChannels,
    plan.outputChannels,
    plan.kernelSize,
    plan.stride,
    plan.padding,
  ].join("x");
}

function requireCoordinate(value: number, bound: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= bound) {
    throw new RangeError(
      `OPT-0028 portable packed ConvTranspose1D ${label} is out of range`,
    );
  }
}
