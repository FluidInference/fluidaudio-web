import {
  planAceFp16VaeConv1d,
  planAceFp16VaeConv1dRange,
  type AceFp16VaeConv1dOutputStorage,
  type AceFp16VaeConv1dPlan,
} from "./vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_REPACK_KERNEL_ID =
  "ace-vae-fp16-opt-0014-o-k-i-to-k-i-o-u16-repack-v1" as const;
export const ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_KERNEL_ID =
  "ace-vae-fp16-opt-0014-fixed32-packed-kio-k7-conv1d-v1" as const;
export const ACE_OPT_0014_VAE_CONV1D_SUBGROUP_SIZE = 32;
export const ACE_OPT_0014_VAE_CONV1D_WORKGROUP_SIZE = 128;
export const ACE_OPT_0014_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP = 4;
export const ACE_OPT_0014_VAE_CONV1D_ROWS_PER_SUBGROUP = 16;
export const ACE_OPT_0014_VAE_CONV1D_OUTPUTS_PER_LANE = 2;
export const ACE_OPT_0014_VAE_CONV1D_CHANNELS_PER_SUBGROUP = 64;
export const ACE_OPT_0014_VAE_CONV1D_MAX_CHANNEL_BANDS = 4;
export const ACE_OPT_0014_VAE_CONV1D_REPACK_WORKGROUP_SIZE = 256;

const GPU_BUFFER_ALIGNMENT = 4;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_CONTROL_BYTES = 16;

export interface AceOpt0014VaeConv1dFixedSubgroupCapability {
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
}

export interface AceOpt0014VaeConv1dPackedWeightPlan {
  readonly inputChannels: number;
  readonly outputChannels: number;
  readonly kernelSize: 7;
  readonly outputChannelPairs: number;
  readonly channelBands: 1 | 2 | 4;
  readonly rowBands: 4 | 2 | 1;
  readonly tileRows: 64 | 32 | 16;
  readonly tileChannels: 64 | 128 | 256;
  readonly packedWordCount: number;
  readonly nativeStorageBytes: number;
  readonly nativeBindingBytes: number;
  readonly packedStorageBytes: number;
  readonly packedBindingBytes: number;
  readonly repackWorkgroupSize:
    typeof ACE_OPT_0014_VAE_CONV1D_REPACK_WORKGROUP_SIZE;
  readonly repackWorkgroups: number;
}

export interface AceOpt0014VaeConv1dPackedKioRangePlan {
  readonly base: number;
  readonly count: number;
  readonly batch: number;
  readonly firstOutputTime: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
  readonly channelBands: 1 | 2 | 4;
  readonly rowBands: 4 | 2 | 1;
  readonly tileRows: 64 | 32 | 16;
  readonly tileChannels: 64 | 128 | 256;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceOpt0014VaeConv1dRepackBindings {
  /** Converter-native raw FP16 bits in `[out,kernel,in]` order. */
  readonly nativeWeight: GPUBufferBinding;
  /** Destination raw FP16 bits in packed `[kernel,in,out]` pair order. */
  readonly packedWeight: GPUBufferBinding;
}

export interface AceOpt0014VaeConv1dPackedKioBindings {
  /** FP16 activation in frame-major NLC order. */
  readonly input: GPUBufferBinding;
  /** GPU-repacked FP16 pairs in `[kernel,in,out]` order. */
  readonly packedWeight: GPUBufferBinding;
  /** FP16 `[out]`; omitted only by the final FP32 Conv1D. */
  readonly bias?: GPUBufferBinding;
  /** FP16 internal activation or FP32 final raw-waveform boundary. */
  readonly output: GPUBufferBinding;
}

export interface AceOpt0014VaeConv1dRepackDispatch {
  readonly label: string;
  readonly kernelId:
    typeof ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_REPACK_KERNEL_ID;
  readonly plan: AceOpt0014VaeConv1dPackedWeightPlan;
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceOpt0014VaeConv1dPackedKioDispatch {
  readonly label: string;
  readonly kernelId: typeof ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_KERNEL_ID;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly packedWeightPlan: AceOpt0014VaeConv1dPackedWeightPlan;
  readonly outputRange: AceOpt0014VaeConv1dPackedKioRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

interface NamedBinding {
  readonly name:
    | "native weight"
    | "packed weight"
    | "input"
    | "bias"
    | "output"
    | "range control";
  readonly binding: GPUBufferBinding;
}

/**
 * Benchmark-only OPT-0014 owner.
 *
 * The preparation pass copies raw U16 payloads from converter-native O-K-I
 * order into K-I-O pairs without converting through `f16`. The fixed-32
 * Conv1D pass then maps each subgroup to a 16-row by 64-channel tile. Compile-
 * time channel/row bands keep all four subgroups useful at the exact 2, 128,
 * and 256+-Cout production shapes. Each lane owns two adjacent output
 * channels and walks K then Cin in the production arithmetic order. No
 * fallback or production selection is attached to this candidate.
 */
export class AceOpt0014VaeConv1dPackedKioSubgroupKernel {
  private readonly repackPipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly convPipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: AceOpt0014VaeConv1dFixedSubgroupCapability,
  ): AceOpt0014VaeConv1dPackedKioSubgroupKernel {
    requireFixedSubgroupDevice(device, capability);
    return new AceOpt0014VaeConv1dPackedKioSubgroupKernel(device);
  }

  async createRepackDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceOpt0014VaeConv1dRepackBindings,
  ): Promise<AceOpt0014VaeConv1dRepackDispatch> {
    this.requireLive();
    const plan = planAceOpt0014VaeConv1dPackedKioWeight(shape);
    requireDispatchDimension(
      this.device,
      plan.repackWorkgroups,
      1,
      "repack",
    );
    requireBufferLimits(this.device, [
      ["native weight", plan.nativeBindingBytes],
      ["packed weight", plan.packedBindingBytes],
    ]);
    const normalized = Object.freeze([
      Object.freeze({
        name: "native weight" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.nativeWeight,
          plan.nativeStorageBytes,
          plan.nativeBindingBytes,
          `${label} native weight`,
        ),
      }),
      Object.freeze({
        name: "packed weight" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.packedWeight,
          plan.packedStorageBytes,
          plan.packedBindingBytes,
          `${label} packed weight`,
        ),
      }),
    ] satisfies readonly NamedBinding[]);
    requireDisjointBindings(normalized, label);
    const key = packedWeightKey(plan);
    const compiled = await this.repackPipelineFor(plan);
    this.requireLive();
    const bindGroup = this.bindGroupFor(
      `${ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_REPACK_KERNEL_ID}:${key}`,
      `${label}-opt-0014-kio-repack-bindings`,
      compiled.bindGroupLayout,
      normalized.map(({ binding }) => binding),
    );
    const owner = this;
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_REPACK_KERNEL_ID,
      plan,
      encode(pass: GPUComputePassEncoder): void {
        owner.requireLive();
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(plan.repackWorkgroups, 1, 1);
      },
    });
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceOpt0014VaeConv1dPackedKioBindings,
    outputStorage: AceFp16VaeConv1dOutputStorage,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0014VaeConv1dPackedKioDispatch> {
    this.requireLive();
    const plan = planAceFp16VaeConv1d(shape, outputStorage);
    const packedWeightPlan = requirePackedK7(plan);
    requireBiasBoundary(plan, bindings.bias !== undefined);
    const outputRange = planAceOpt0014VaeConv1dPackedKioRange(plan, range);
    requireDispatchDimension(
      this.device,
      outputRange.workgroupsX,
      outputRange.workgroupsY,
      "Conv1D",
    );
    requireBufferLimits(this.device, [
      ["input", plan.inputBindingBytes],
      ["packed weight", packedWeightPlan.packedBindingBytes],
      ["bias", plan.biasBindingBytes],
      ["output", plan.outputBindingBytes],
    ]);
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
        name: "packed weight",
        binding: requireStorageBinding(
          this.device,
          bindings.packedWeight,
          packedWeightPlan.packedStorageBytes,
          packedWeightPlan.packedBindingBytes,
          `${label} packed weight`,
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
    const hasBias = bindings.bias !== undefined;
    const compiled = await this.convPipelineFor(plan, hasBias);
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
    const bindGroup = this.bindGroupFor(
      `${ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_KERNEL_ID}:${convKey(plan, hasBias)}`,
      `${label}-opt-0014-packed-kio-conv1d-bindings`,
      compiled.bindGroupLayout,
      resources,
    );
    const owner = this;
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_KERNEL_ID,
      plan,
      packedWeightPlan,
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
    this.repackPipelines.clear();
    this.convPipelines.clear();
    this.bindGroups.clear();
  }

  private repackPipelineFor(
    plan: AceOpt0014VaeConv1dPackedWeightPlan,
  ): Promise<CompiledKernel> {
    const key = packedWeightKey(plan);
    const existing = this.repackPipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileRepack(this.device, plan);
    this.repackPipelines.set(key, created);
    void created.catch(() => {
      if (this.repackPipelines.get(key) === created) {
        this.repackPipelines.delete(key);
      }
    });
    return created;
  }

  private convPipelineFor(
    plan: AceFp16VaeConv1dPlan,
    hasBias: boolean,
  ): Promise<CompiledKernel> {
    const key = convKey(plan, hasBias);
    const existing = this.convPipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compilePackedConv(this.device, plan, hasBias);
    this.convPipelines.set(key, created);
    void created.catch(() => {
      if (this.convPipelines.get(key) === created) {
        this.convPipelines.delete(key);
      }
    });
    return created;
  }

  private bindGroupFor(
    prefix: string,
    label: string,
    layout: GPUBindGroupLayout,
    resources: readonly GPUBufferBinding[],
  ): GPUBindGroup {
    const key = `${prefix}:${resources.map((binding) =>
      this.bindingKey(binding)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(key);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label,
        layout,
        entries: resources.map((resource, binding) => ({
          binding,
          resource,
        })),
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
      throw new Error("OPT-0014 packed-KIO Conv1D kernel was destroyed");
    }
  }
}

export function planAceOpt0014VaeConv1dPackedKioWeight(
  shape: AceVaeConv1dShape,
): AceOpt0014VaeConv1dPackedWeightPlan {
  const plan = planAceFp16VaeConv1d(shape, "float16");
  if (plan.family !== "k7") {
    throw new RangeError("OPT-0014 packed-KIO Conv1D supports only K7");
  }
  if (plan.outputChannels % 2 !== 0) {
    throw new RangeError(
      "OPT-0014 packed-KIO Conv1D requires output-channel pairs",
    );
  }
  const outputChannelPairs = plan.outputChannels / 2;
  const geometry = packedGeometry(plan.outputChannels);
  const packedWordCount = plan.weightElements / 2;
  const repackWorkgroups = Math.ceil(
    packedWordCount / ACE_OPT_0014_VAE_CONV1D_REPACK_WORKGROUP_SIZE,
  );
  for (const [name, value] of [
    ["output channel pairs", outputChannelPairs],
    ["packed word count", packedWordCount],
    ["repack workgroups", repackWorkgroups],
  ] as const) requireWgslIndexable(value, name);
  return Object.freeze({
    inputChannels: plan.inputChannels,
    outputChannels: plan.outputChannels,
    kernelSize: 7,
    outputChannelPairs,
    ...geometry,
    packedWordCount,
    nativeStorageBytes: plan.weightStorageBytes,
    nativeBindingBytes: plan.weightBindingBytes,
    packedStorageBytes: plan.weightStorageBytes,
    packedBindingBytes: plan.weightBindingBytes,
    repackWorkgroupSize: ACE_OPT_0014_VAE_CONV1D_REPACK_WORKGROUP_SIZE,
    repackWorkgroups,
  });
}

export function planAceOpt0014VaeConv1dPackedKioRange(
  plan: AceFp16VaeConv1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceOpt0014VaeConv1dPackedKioRangePlan {
  const packed = requirePackedK7(plan);
  const portable = planAceFp16VaeConv1dRange(plan, range);
  const workgroupsX = Math.ceil(
    portable.outputRowCount / packed.tileRows,
  );
  const workgroupsY = Math.ceil(
    plan.outputChannels / packed.tileChannels,
  );
  requireWgslIndexable(workgroupsX, "Conv1D workgroups X");
  requireWgslIndexable(workgroupsY, "Conv1D workgroups Y");
  return Object.freeze({
    base: portable.base,
    count: portable.count,
    batch: portable.batch,
    firstOutputTime: portable.firstOutputTime,
    firstOutputRow: portable.firstOutputRow,
    outputRowCount: portable.outputRowCount,
    channelBands: packed.channelBands,
    rowBands: packed.rowBands,
    tileRows: packed.tileRows,
    tileChannels: packed.tileChannels,
    workgroupsX,
    workgroupsY,
  });
}

export function aceOpt0014VaeConv1dPackedKioRepackWgsl(
  shape: AceVaeConv1dShape,
): string {
  const plan = planAceOpt0014VaeConv1dPackedKioWeight(shape);
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_REPACK_KERNEL_ID}
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNEL_PAIRS: u32 = ${plan.outputChannelPairs}u;
const PACKED_WORD_COUNT: u32 = ${plan.packedWordCount}u;

@group(0) @binding(0) var<storage, read> native_weight: array<u32>;
@group(0) @binding(1) var<storage, read_write> packed_weight: array<u32>;

fn load_native_u16(scalar_index: u32) -> u32 {
  let word = native_weight[scalar_index >> 1u];
  let shift = (scalar_index & 1u) * 16u;
  return (word >> shift) & 0xffffu;
}

@compute @workgroup_size(${ACE_OPT_0014_VAE_CONV1D_REPACK_WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let packed_word = global.x;
  if (packed_word < PACKED_WORD_COUNT) {
    let output_pair = packed_word % OUTPUT_CHANNEL_PAIRS;
    let kernel_input = packed_word / OUTPUT_CHANNEL_PAIRS;
    let input_channel = kernel_input % INPUT_CHANNELS;
    let kernel = kernel_input / INPUT_CHANNELS;
    let output_channel = output_pair * 2u;
    let native_index0 =
      (output_channel * 7u + kernel) * INPUT_CHANNELS + input_channel;
    let native_index1 =
      ((output_channel + 1u) * 7u + kernel) * INPUT_CHANNELS + input_channel;
    let low = load_native_u16(native_index0);
    let high = load_native_u16(native_index1);
    packed_weight[packed_word] = low | (high << 16u);
  }
}
`;
}

export function aceOpt0014VaeConv1dPackedKioWgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceFp16VaeConv1dOutputStorage,
): string {
  const plan = planAceFp16VaeConv1d(shape, outputStorage);
  const packed = requirePackedK7(plan);
  requireBiasBoundary(plan, hasBias);
  const outputBinding = hasBias ? 3 : 2;
  const rangeBinding = hasBias ? 4 : 3;
  const biasDeclaration = hasBias
    ? "@group(0) @binding(2) var<storage, read> bias: array<f16>;"
    : "";
  const outputElementType = outputStorage === "float16" ? "f16" : "f32";
  const initialSums = Array.from(
    { length: ACE_OPT_0014_VAE_CONV1D_ROWS_PER_SUBGROUP },
    (_, row) => `  var sum${row}: vec2<f32> = initial_sum;`,
  ).join("\n");
  const inputValidity = Array.from(
    { length: ACE_OPT_0014_VAE_CONV1D_ROWS_PER_SUBGROUP },
    (_, row) => `
      let output_time${row} = tile_first_time + ${row}u;
      let padded_time${row} = output_time${row} + kernel * DILATION;
      var input_valid${row} = false;
      if (output_time${row} < range_end_time && padded_time${row} >= PADDING) {
        input_valid${row} = padded_time${row} - PADDING < INPUT_FRAMES;
      }`,
  ).join("");
  const broadcastsAndAdds = Array.from(
    { length: ACE_OPT_0014_VAE_CONV1D_ROWS_PER_SUBGROUP },
    (_, row) => `
      let input_operand${row} = subgroupBroadcast(lane_input, ${row}u);
      if (input_valid${row}) {
        sum${row} = sum${row} +
          vec2<f32>(input_operand${row}) * weight_operands;
      }`,
  ).join("");
  const stores = Array.from(
    { length: ACE_OPT_0014_VAE_CONV1D_ROWS_PER_SUBGROUP },
    (_, row) => `
    let output_time${row} = tile_first_time + ${row}u;
    if (output_time${row} < range_end_time) {
${outputStoresForRow(row, hasBias)}
    }`,
  ).join("");
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_KERNEL_ID}
enable f16;
enable subgroups;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const OUTPUT_CHANNEL_PAIRS: u32 = ${packed.outputChannelPairs}u;
const CHANNEL_BANDS: u32 = ${packed.channelBands}u;
const PADDING: u32 = ${plan.padding}u;
const DILATION: u32 = ${plan.dilation}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> packed_weight: array<u32>;
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

@compute @workgroup_size(${ACE_OPT_0014_VAE_CONV1D_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (subgroup_size == ${ACE_OPT_0014_VAE_CONV1D_SUBGROUP_SIZE}u) {
    let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
    let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
    let range_first_time = first_output_row % OUTPUT_FRAMES;
    let batch = first_output_row / OUTPUT_FRAMES;
    let range_end_time = range_first_time + output_row_count;
    let channel_band = subgroup % CHANNEL_BANDS;
    let row_band = subgroup / CHANNEL_BANDS;
    let tile_first_time = range_first_time +
      group.x * ${packed.tileRows}u +
      row_band * ${ACE_OPT_0014_VAE_CONV1D_ROWS_PER_SUBGROUP}u;
    let output_channel_base =
      group.y * ${packed.tileChannels}u +
      channel_band * ${ACE_OPT_0014_VAE_CONV1D_CHANNELS_PER_SUBGROUP}u +
      subgroup_lane * ${ACE_OPT_0014_VAE_CONV1D_OUTPUTS_PER_LANE}u;

${initialSumVectorWgsl(hasBias)}
${initialSums}

    // Exact source K-outer, increasing-Cin FP32 accumulation order. Each
    // subgroup owns one 16-row x 64-Cout tile and loads a contiguous FP16 pair.
    for (var kernel = 0u; kernel < 7u; kernel += 1u) {
      for (
        var input_channel = 0u;
        input_channel < INPUT_CHANNELS;
        input_channel += 1u
      ) {
${inputValidity}
        var lane_input: f32 = 0.0;
        if (subgroup_lane < ${ACE_OPT_0014_VAE_CONV1D_ROWS_PER_SUBGROUP}u) {
          let lane_output_time = tile_first_time + subgroup_lane;
          let lane_padded_time = lane_output_time + kernel * DILATION;
          if (
            lane_output_time < range_end_time &&
            lane_padded_time >= PADDING
          ) {
            let lane_input_time = lane_padded_time - PADDING;
            if (lane_input_time < INPUT_FRAMES) {
              lane_input = f32(input[
                (batch * INPUT_FRAMES + lane_input_time) * INPUT_CHANNELS +
                input_channel
              ]);
            }
          }
        }
        var weight_operands = vec2<f32>(0.0);
        if (output_channel_base < OUTPUT_CHANNELS) {
          let packed_index =
            (kernel * INPUT_CHANNELS + input_channel) *
              OUTPUT_CHANNEL_PAIRS +
            output_channel_base / 2u;
          weight_operands = unpack2x16float(packed_weight[packed_index]);
        }
${broadcastsAndAdds}
      }
    }

${stores}
  }
}
`;
}

async function compileRepack(
  device: GPUDevice,
  plan: AceOpt0014VaeConv1dPackedWeightPlan,
): Promise<CompiledKernel> {
  const label = `ace-opt-0014-vae-k7-repack-${packedWeightKey(plan)}`;
  const module = await checkedShaderModule(
    device,
    label,
    aceOpt0014VaeConv1dPackedKioRepackWgsl({
      batch: 1,
      inputFrames: 1,
      inputChannels: plan.inputChannels,
      outputChannels: plan.outputChannels,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 3,
    }),
  );
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
          minBindingSize: plan.nativeBindingBytes,
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
          minBindingSize: plan.packedBindingBytes,
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

async function compilePackedConv(
  device: GPUDevice,
  plan: AceFp16VaeConv1dPlan,
  hasBias: boolean,
): Promise<CompiledKernel> {
  const label = `ace-opt-0014-vae-k7-conv-${convKey(plan, hasBias)}`;
  const module = await checkedShaderModule(
    device,
    label,
    aceOpt0014VaeConv1dPackedKioWgsl(plan, hasBias, plan.outputStorage),
  );
  const packed = requirePackedK7(plan);
  const dataBindingSizes = hasBias
    ? [
        plan.inputBindingBytes,
        packed.packedBindingBytes,
        plan.biasBindingBytes,
        plan.outputBindingBytes,
      ]
    : [
        plan.inputBindingBytes,
        packed.packedBindingBytes,
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
      `OPT-0014 packed-KIO WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  return module;
}

function packedGeometry(outputChannels: number): Readonly<{
  channelBands: 1 | 2 | 4;
  rowBands: 4 | 2 | 1;
  tileRows: 64 | 32 | 16;
  tileChannels: 64 | 128 | 256;
}> {
  const channelBands = Math.min(
    ACE_OPT_0014_VAE_CONV1D_MAX_CHANNEL_BANDS,
    Math.ceil(outputChannels / ACE_OPT_0014_VAE_CONV1D_CHANNELS_PER_SUBGROUP),
  );
  switch (channelBands) {
    case 1:
      return Object.freeze({
        channelBands: 1,
        rowBands: 4,
        tileRows: 64,
        tileChannels: 64,
      });
    case 2:
      return Object.freeze({
        channelBands: 2,
        rowBands: 2,
        tileRows: 32,
        tileChannels: 128,
      });
    case 4:
      return Object.freeze({
        channelBands: 4,
        rowBands: 1,
        tileRows: 16,
        tileChannels: 256,
      });
    default:
      throw new RangeError(
        "OPT-0014 packed-KIO Conv1D requires 1, 2, or 4 channel bands",
      );
  }
}

function requirePackedK7(
  plan: AceFp16VaeConv1dPlan,
): AceOpt0014VaeConv1dPackedWeightPlan {
  return planAceOpt0014VaeConv1dPackedKioWeight(plan);
}

function requireBiasBoundary(
  plan: AceFp16VaeConv1dPlan,
  hasBias: boolean,
): void {
  if (hasBias && plan.outputStorage === "float32") {
    throw new RangeError(
      "OPT-0014 packed-KIO FP32 output is reserved for the final no-bias boundary",
    );
  }
  if (!hasBias && plan.outputStorage !== "float32") {
    throw new RangeError(
      "OPT-0014 packed-KIO bias may be omitted only at the final FP32 boundary",
    );
  }
}

function initialSumVectorWgsl(hasBias: boolean): string {
  if (!hasBias) return "  let initial_sum = vec2<f32>(0.0);";
  return `  var bias0: f32 = 0.0;
  var bias1: f32 = 0.0;
  if (output_channel_base < OUTPUT_CHANNELS) {
    bias0 = f32(bias[output_channel_base]);
  }
  if (output_channel_base + 1u < OUTPUT_CHANNELS) {
    bias1 = f32(bias[output_channel_base + 1u]);
  }
  let initial_sum = vec2<f32>(bias0, bias1);`;
}

function outputStoresForRow(row: number, hasBias: boolean): string {
  return ["x", "y"].map((swizzle, component) => {
    const suffix = component === 0 ? "" : " + 1u";
    const value = hasBias
      ? `f16(sum${row}.${swizzle})`
      : `select(
          sum${row}.${swizzle},
          bitcast<f32>(0u),
          (bitcast<u32>(sum${row}.${swizzle}) & 0x7fffffffu) == 0u
        )`;
    return `      if (output_channel_base${suffix} < OUTPUT_CHANNELS) {
        output[
          (batch * OUTPUT_FRAMES + output_time${row}) * OUTPUT_CHANNELS +
          output_channel_base${suffix}
        ] = ${value};
      }`;
  }).join("\n");
}

function requireFixedSubgroupDevice(
  device: GPUDevice,
  capability: AceOpt0014VaeConv1dFixedSubgroupCapability,
): void {
  if (!device.features.has("shader-f16")) {
    throw new Error("OPT-0014 packed-KIO Conv1D requires WebGPU shader-f16");
  }
  if (
    !device.features.has("subgroups") ||
    capability.subgroupMinSize !== ACE_OPT_0014_VAE_CONV1D_SUBGROUP_SIZE ||
    capability.subgroupMaxSize !== ACE_OPT_0014_VAE_CONV1D_SUBGROUP_SIZE
  ) {
    throw new Error(
      "OPT-0014 packed-KIO Conv1D requires reported fixed 32-lane subgroups",
    );
  }
  const maximumInvocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const maximumSizeX = device.limits.maxComputeWorkgroupSizeX;
  if (
    !Number.isSafeInteger(maximumInvocations) ||
    !Number.isSafeInteger(maximumSizeX) ||
    maximumInvocations < ACE_OPT_0014_VAE_CONV1D_REPACK_WORKGROUP_SIZE ||
    maximumSizeX < ACE_OPT_0014_VAE_CONV1D_REPACK_WORKGROUP_SIZE
  ) {
    throw new Error(
      `OPT-0014 packed-KIO Conv1D requires WG${ACE_OPT_0014_VAE_CONV1D_REPACK_WORKGROUP_SIZE}`,
    );
  }
}

function requireDispatchDimension(
  device: GPUDevice,
  workgroupsX: number,
  workgroupsY: number,
  family: string,
): void {
  const maximum = device.limits.maxComputeWorkgroupsPerDimension;
  if (
    !Number.isSafeInteger(maximum) || maximum < 1 ||
    workgroupsX > maximum || workgroupsY > maximum
  ) {
    throw new RangeError(
      `OPT-0014 packed-KIO ${family} exceeds the device dispatch dimension`,
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
    !Number.isSafeInteger(maximumBinding) || maximumBinding < 1 ||
    !Number.isSafeInteger(maximumBuffer) || maximumBuffer < 1
  ) {
    throw new RangeError(
      "OPT-0014 packed-KIO device reported invalid buffer limits",
    );
  }
  for (const [name, bytes] of resources) {
    if (bytes > maximumBinding || bytes > maximumBuffer) {
      throw new RangeError(
        `OPT-0014 packed-KIO ${name} exceeds the device buffer limits`,
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
      "OPT-0014 packed-KIO device reported an invalid storage alignment",
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
  const alignment = device.limits.minUniformBufferOffsetAlignment;
  if (!isValidGpuAlignment(alignment)) {
    throw new Error(
      "OPT-0014 packed-KIO device reported an invalid uniform alignment",
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

function packedWeightKey(
  plan: AceOpt0014VaeConv1dPackedWeightPlan,
): string {
  return [
    plan.inputChannels,
    plan.outputChannels,
    plan.kernelSize,
  ].join("x");
}

function convKey(plan: AceFp16VaeConv1dPlan, hasBias: boolean): string {
  return [
    "fixed32-packed-kio",
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

function requireWgslIndexable(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > MAX_WGSL_U32) {
    throw new RangeError(
      `OPT-0014 packed-KIO ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
