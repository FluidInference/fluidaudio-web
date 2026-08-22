import {
  planAceFp16VaeConv1d,
  planAceFp16VaeConv1dRange,
  type AceFp16VaeConv1dOutputStorage,
  type AceFp16VaeConv1dPlan,
} from "./vae-conv1d-fp16.js";
import {
  planAceOpt0014VaeConv1dPackedKioWeight,
  type AceOpt0014VaeConv1dPackedWeightPlan,
} from "./vae-conv1d-fp16-packed-kio-subgroup.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_8X64_KERNEL_ID =
  "ace-vae-fp16-opt-0016-fixed32-packed-kio-k7-conv1d-8x64-v1" as const;
export const ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_16X32_KERNEL_ID =
  "ace-vae-fp16-opt-0016-fixed32-packed-kio-k7-conv1d-16x32-v1" as const;
export const ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_8X32_KERNEL_ID =
  "ace-vae-fp16-opt-0016-fixed32-packed-kio-k7-conv1d-8x32-v1" as const;
export const ACE_OPT_0016_VAE_CONV1D_SUBGROUP_SIZE = 32;
export const ACE_OPT_0016_VAE_CONV1D_WORKGROUP_SIZE = 128;
export const ACE_OPT_0016_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP = 4;
export const ACE_OPT_0016_VAE_CONV1D_MAX_CHANNEL_BANDS = 4;
export const ACE_OPT_0016_VAE_CONV1D_PRIMARY_VARIANT = "16x32" as const;
export const ACE_OPT_0016_VAE_CONV1D_VARIANTS = Object.freeze([
  "8x64",
  "16x32",
  "8x32",
] as const);

const GPU_BUFFER_ALIGNMENT = 4;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_CONTROL_BYTES = 16;

export type AceOpt0016VaeConv1dMicrotileVariant =
  typeof ACE_OPT_0016_VAE_CONV1D_VARIANTS[number];

export type AceOpt0016VaeConv1dMicrotileKernelId =
  | typeof ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_8X64_KERNEL_ID
  | typeof ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_16X32_KERNEL_ID
  | typeof ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_8X32_KERNEL_ID;

export interface AceOpt0016VaeConv1dFixedSubgroupCapability {
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
}

export interface AceOpt0016VaeConv1dPackedKioBindings {
  /** FP16 activation in frame-major NLC order. */
  readonly input: GPUBufferBinding;
  /** OPT-0014's bit-preserving FP16 `[kernel,input,output]` payload. */
  readonly packedWeight: GPUBufferBinding;
  /** FP16 `[output]`; omitted only by the final FP32 Conv1D. */
  readonly bias?: GPUBufferBinding;
  /** FP16 internal activation or FP32 final raw-waveform boundary. */
  readonly output: GPUBufferBinding;
}

export interface AceOpt0016VaeConv1dMicrotilePlan {
  readonly variant: AceOpt0016VaeConv1dMicrotileVariant;
  readonly kernelId: AceOpt0016VaeConv1dMicrotileKernelId;
  readonly rowsPerSubgroup: 8 | 16;
  readonly outputsPerLane: 1 | 2;
  readonly channelsPerSubgroup: 32 | 64;
  readonly accumulatorsPerLane: 8 | 16;
  readonly weightView: "scalar-f16" | "vec2-f16";
  readonly channelBands: 1 | 2 | 4;
  readonly rowBands: 4 | 2 | 1;
  readonly tileRows: number;
  readonly tileChannels: number;
  readonly packedWeightPlan: AceOpt0014VaeConv1dPackedWeightPlan;
}

export interface AceOpt0016VaeConv1dMicrotileRangePlan {
  readonly variant: AceOpt0016VaeConv1dMicrotileVariant;
  readonly kernelId: AceOpt0016VaeConv1dMicrotileKernelId;
  readonly base: number;
  readonly count: number;
  readonly batch: number;
  readonly firstOutputTime: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
  readonly channelBands: 1 | 2 | 4;
  readonly rowBands: 4 | 2 | 1;
  readonly tileRows: number;
  readonly tileChannels: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceOpt0016VaeConv1dMicrotileDispatch {
  readonly label: string;
  readonly variant: AceOpt0016VaeConv1dMicrotileVariant;
  readonly kernelId: AceOpt0016VaeConv1dMicrotileKernelId;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly microtilePlan: AceOpt0016VaeConv1dMicrotilePlan;
  readonly outputRange: AceOpt0016VaeConv1dMicrotileRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface VariantSpec {
  readonly variant: AceOpt0016VaeConv1dMicrotileVariant;
  readonly kernelId: AceOpt0016VaeConv1dMicrotileKernelId;
  readonly rowsPerSubgroup: 8 | 16;
  readonly outputsPerLane: 1 | 2;
  readonly channelsPerSubgroup: 32 | 64;
  readonly accumulatorsPerLane: 8 | 16;
  readonly weightView: "scalar-f16" | "vec2-f16";
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

interface NamedBinding {
  readonly name: "input" | "packed weight" | "bias" | "output" |
    "range control";
  readonly binding: GPUBufferBinding;
}

/**
 * Isolated benchmark-only owner for OPT-0016's final exact-order K7 sweep.
 *
 * All three variants consume the exact OPT-0014 KIO bytes. They differ only
 * in per-subgroup row/channel ownership and the typed FP16 view used to load
 * those bytes. There is no repack, allocation, fallback, or production route.
 */
export class AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel {
  private readonly pipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: AceOpt0016VaeConv1dFixedSubgroupCapability,
  ): AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel {
    requireFixedSubgroupDevice(device, capability);
    return new AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel(device);
  }

  async createDispatch(
    label: string,
    variant: AceOpt0016VaeConv1dMicrotileVariant,
    shape: AceVaeConv1dShape,
    bindings: AceOpt0016VaeConv1dPackedKioBindings,
    outputStorage: AceFp16VaeConv1dOutputStorage,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0016VaeConv1dMicrotileDispatch> {
    this.requireLive();
    requireVariantSpec(variant);
    const plan = planAceFp16VaeConv1d(shape, outputStorage);
    const microtilePlan = planAceOpt0016VaeConv1dPackedKioMicrotile(
      plan,
      variant,
    );
    requireBiasBoundary(plan, bindings.bias !== undefined);
    const outputRange = planAceOpt0016VaeConv1dPackedKioMicrotileRange(
      plan,
      variant,
      range,
    );
    requireDispatchDimension(
      this.device,
      outputRange.workgroupsX,
      outputRange.workgroupsY,
    );
    requireBufferLimits(this.device, [
      ["input", plan.inputBindingBytes],
      ["packed weight", microtilePlan.packedWeightPlan.packedBindingBytes],
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
          microtilePlan.packedWeightPlan.packedStorageBytes,
          microtilePlan.packedWeightPlan.packedBindingBytes,
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
    const compiled = await this.pipelineFor(plan, variant, hasBias);
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
      `${microtilePlan.kernelId}:${convKey(plan, variant, hasBias)}`,
      `${label}-opt-0016-${variant}-packed-kio-bindings`,
      compiled.bindGroupLayout,
      resources,
    );
    const owner = this;
    return Object.freeze({
      label,
      variant,
      kernelId: microtilePlan.kernelId,
      plan,
      microtilePlan,
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
    this.pipelines.clear();
    this.bindGroups.clear();
  }

  private pipelineFor(
    plan: AceFp16VaeConv1dPlan,
    variant: AceOpt0016VaeConv1dMicrotileVariant,
    hasBias: boolean,
  ): Promise<CompiledKernel> {
    const key = convKey(plan, variant, hasBias);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileMicrotile(this.device, plan, variant, hasBias);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) {
        this.pipelines.delete(key);
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
      throw new Error("OPT-0016 packed-KIO microtile Conv1D kernel was destroyed");
    }
  }
}

export function planAceOpt0016VaeConv1dPackedKioMicrotile(
  shape: AceVaeConv1dShape,
  variant: AceOpt0016VaeConv1dMicrotileVariant,
): AceOpt0016VaeConv1dMicrotilePlan {
  const spec = requireVariantSpec(variant);
  const plan = planAceFp16VaeConv1d(shape, "float16");
  if (plan.family !== "k7") {
    throw new RangeError("OPT-0016 packed-KIO microtiles support only K7");
  }
  const packedWeightPlan = planAceOpt0014VaeConv1dPackedKioWeight(plan);
  const channelBandCount = Math.min(
    ACE_OPT_0016_VAE_CONV1D_MAX_CHANNEL_BANDS,
    Math.ceil(plan.outputChannels / spec.channelsPerSubgroup),
  );
  const channelBands = requireChannelBands(channelBandCount, variant);
  const rowBands = requireRowBands(channelBands);
  const tileRows = spec.rowsPerSubgroup * rowBands;
  const tileChannels = spec.channelsPerSubgroup * channelBands;
  for (const [name, value] of [
    ["tile rows", tileRows],
    ["tile channels", tileChannels],
  ] as const) requireWgslIndexable(value, name);
  return Object.freeze({
    ...spec,
    channelBands,
    rowBands,
    tileRows,
    tileChannels,
    packedWeightPlan,
  });
}

export function planAceOpt0016VaeConv1dPackedKioMicrotileRange(
  plan: AceFp16VaeConv1dPlan,
  variant: AceOpt0016VaeConv1dMicrotileVariant,
  range: Readonly<{ base: number; count: number }>,
): AceOpt0016VaeConv1dMicrotileRangePlan {
  const microtile = planAceOpt0016VaeConv1dPackedKioMicrotile(plan, variant);
  const portable = planAceFp16VaeConv1dRange(plan, range);
  const workgroupsX = Math.ceil(portable.outputRowCount / microtile.tileRows);
  const workgroupsY = Math.ceil(plan.outputChannels / microtile.tileChannels);
  requireWgslIndexable(workgroupsX, "Conv1D workgroups X");
  requireWgslIndexable(workgroupsY, "Conv1D workgroups Y");
  return Object.freeze({
    variant,
    kernelId: microtile.kernelId,
    base: portable.base,
    count: portable.count,
    batch: portable.batch,
    firstOutputTime: portable.firstOutputTime,
    firstOutputRow: portable.firstOutputRow,
    outputRowCount: portable.outputRowCount,
    channelBands: microtile.channelBands,
    rowBands: microtile.rowBands,
    tileRows: microtile.tileRows,
    tileChannels: microtile.tileChannels,
    workgroupsX,
    workgroupsY,
  });
}

export function aceOpt0016VaeConv1dPackedKioMicrotileWgsl(
  shape: AceVaeConv1dShape,
  variant: AceOpt0016VaeConv1dMicrotileVariant,
  hasBias: boolean,
  outputStorage: AceFp16VaeConv1dOutputStorage,
): string {
  const spec = requireVariantSpec(variant);
  const plan = planAceFp16VaeConv1d(shape, outputStorage);
  const microtile = planAceOpt0016VaeConv1dPackedKioMicrotile(plan, variant);
  requireBiasBoundary(plan, hasBias);
  const outputBinding = hasBias ? 3 : 2;
  const rangeBinding = hasBias ? 4 : 3;
  const biasDeclaration = hasBias
    ? "@group(0) @binding(2) var<storage, read> bias: array<f16>;"
    : "";
  const outputElementType = outputStorage === "float16" ? "f16" : "f32";
  const weightElementType = spec.weightView === "vec2-f16"
    ? "vec2<f16>"
    : "f16";
  const accumulatorType = spec.outputsPerLane === 2 ? "vec2<f32>" : "f32";
  const initialSums = Array.from(
    { length: spec.rowsPerSubgroup },
    (_, row) => `  var sum${row}: ${accumulatorType} = initial_sum;`,
  ).join("\n");
  const inputValidity = Array.from(
    { length: spec.rowsPerSubgroup },
    (_, row) => `
      let output_time${row} = tile_first_time + ${row}u;
      let padded_time${row} = output_time${row} + kernel * DILATION;
      var input_valid${row} = false;
      if (output_time${row} < range_end_time && padded_time${row} >= PADDING) {
        input_valid${row} = padded_time${row} - PADDING < INPUT_FRAMES;
      }`,
  ).join("");
  const operandType = spec.outputsPerLane === 2 ? "vec2<f32>" : "f32";
  const broadcastsAndAdds = Array.from(
    { length: spec.rowsPerSubgroup },
    (_, row) => `
      let input_operand${row} = subgroupBroadcast(lane_input, ${row}u);
      if (input_valid${row}) {
        sum${row} = sum${row} +
          ${operandType}(input_operand${row}) * weight_operands;
      }`,
  ).join("");
  const stores = Array.from(
    { length: spec.rowsPerSubgroup },
    (_, row) => `
    let output_time${row} = tile_first_time + ${row}u;
    if (output_time${row} < range_end_time) {
${outputStoresForRow(row, spec.outputsPerLane, hasBias)}
    }`,
  ).join("");
  return /* wgsl */ `
// kernel-id: ${spec.kernelId}
// variant: ${variant}; packed-weight-view: ${spec.weightView}
enable f16;
enable subgroups;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const OUTPUT_CHANNEL_PAIRS: u32 =
  ${microtile.packedWeightPlan.outputChannelPairs}u;
const CHANNEL_BANDS: u32 = ${microtile.channelBands}u;
const PADDING: u32 = ${plan.padding}u;
const DILATION: u32 = ${plan.dilation}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read>
  packed_weight: array<${weightElementType}>;
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

@compute @workgroup_size(${ACE_OPT_0016_VAE_CONV1D_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (subgroup_size == ${ACE_OPT_0016_VAE_CONV1D_SUBGROUP_SIZE}u) {
    let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
    let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
    let range_first_time = first_output_row % OUTPUT_FRAMES;
    let batch = first_output_row / OUTPUT_FRAMES;
    let range_end_time = range_first_time + output_row_count;
    let channel_band = subgroup % CHANNEL_BANDS;
    let row_band = subgroup / CHANNEL_BANDS;
    let tile_first_time = range_first_time +
      group.x * ${microtile.tileRows}u +
      row_band * ${spec.rowsPerSubgroup}u;
    let output_channel_base =
      group.y * ${microtile.tileChannels}u +
      channel_band * ${spec.channelsPerSubgroup}u +
      subgroup_lane * ${spec.outputsPerLane}u;

${initialSumWgsl(spec, hasBias)}
${initialSums}

    // Exact source K-outer, increasing-Cin FP32 accumulation order. The typed
    // FP16 view consumes OPT-0014's unchanged bit-preserving KIO bytes.
    for (var kernel = 0u; kernel < 7u; kernel += 1u) {
      for (
        var input_channel = 0u;
        input_channel < INPUT_CHANNELS;
        input_channel += 1u
      ) {
${inputValidity}
        var lane_input: f32 = 0.0;
        if (subgroup_lane < ${spec.rowsPerSubgroup}u) {
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
${weightLoadWgsl(spec)}
${broadcastsAndAdds}
      }
    }

${stores}
  }
}
`;
}

async function compileMicrotile(
  device: GPUDevice,
  plan: AceFp16VaeConv1dPlan,
  variant: AceOpt0016VaeConv1dMicrotileVariant,
  hasBias: boolean,
): Promise<CompiledKernel> {
  const microtile = planAceOpt0016VaeConv1dPackedKioMicrotile(plan, variant);
  const label = `ace-opt-0016-vae-k7-${variant}-${convKey(
    plan,
    variant,
    hasBias,
  )}`;
  const module = await checkedShaderModule(
    device,
    label,
    aceOpt0016VaeConv1dPackedKioMicrotileWgsl(
      plan,
      variant,
      hasBias,
      plan.outputStorage,
    ),
  );
  const dataBindingSizes = hasBias
    ? [
        plan.inputBindingBytes,
        microtile.packedWeightPlan.packedBindingBytes,
        plan.biasBindingBytes,
        plan.outputBindingBytes,
      ]
    : [
        plan.inputBindingBytes,
        microtile.packedWeightPlan.packedBindingBytes,
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
      `OPT-0016 packed-KIO microtile WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  return module;
}

function requireVariantSpec(
  variant: AceOpt0016VaeConv1dMicrotileVariant,
): VariantSpec {
  switch (variant) {
    case "8x64":
      return Object.freeze({
        variant,
        kernelId: ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_8X64_KERNEL_ID,
        rowsPerSubgroup: 8,
        outputsPerLane: 2,
        channelsPerSubgroup: 64,
        accumulatorsPerLane: 16,
        weightView: "vec2-f16",
      });
    case "16x32":
      return Object.freeze({
        variant,
        kernelId: ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_16X32_KERNEL_ID,
        rowsPerSubgroup: 16,
        outputsPerLane: 1,
        channelsPerSubgroup: 32,
        accumulatorsPerLane: 16,
        weightView: "scalar-f16",
      });
    case "8x32":
      return Object.freeze({
        variant,
        kernelId: ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_8X32_KERNEL_ID,
        rowsPerSubgroup: 8,
        outputsPerLane: 1,
        channelsPerSubgroup: 32,
        accumulatorsPerLane: 8,
        weightView: "scalar-f16",
      });
    default:
      throw new TypeError(
        `OPT-0016 packed-KIO microtile has unknown variant ${String(variant)}`,
      );
  }
}

function requireChannelBands(
  value: number,
  variant: AceOpt0016VaeConv1dMicrotileVariant,
): 1 | 2 | 4 {
  if (value === 1 || value === 2 || value === 4) return value;
  throw new RangeError(
    `OPT-0016 ${variant} requires 1, 2, or 4 compile-time channel bands`,
  );
}

function requireRowBands(channelBands: 1 | 2 | 4): 4 | 2 | 1 {
  switch (channelBands) {
    case 1:
      return 4;
    case 2:
      return 2;
    case 4:
      return 1;
  }
}

function requireBiasBoundary(
  plan: AceFp16VaeConv1dPlan,
  hasBias: boolean,
): void {
  if (hasBias && plan.outputStorage === "float32") {
    throw new RangeError(
      "OPT-0016 packed-KIO microtile FP32 output is reserved for the final no-bias boundary",
    );
  }
  if (!hasBias && plan.outputStorage !== "float32") {
    throw new RangeError(
      "OPT-0016 packed-KIO microtile bias may be omitted only at the final FP32 boundary",
    );
  }
}

function initialSumWgsl(spec: VariantSpec, hasBias: boolean): string {
  if (spec.outputsPerLane === 1) {
    if (!hasBias) return "  let initial_sum: f32 = 0.0;";
    return `  var initial_sum: f32 = 0.0;
  if (output_channel_base < OUTPUT_CHANNELS) {
    initial_sum = f32(bias[output_channel_base]);
  }`;
  }
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

function weightLoadWgsl(spec: VariantSpec): string {
  if (spec.outputsPerLane === 1) {
    return `        var weight_operands: f32 = 0.0;
        if (output_channel_base < OUTPUT_CHANNELS) {
          let packed_index =
            (kernel * INPUT_CHANNELS + input_channel) * OUTPUT_CHANNELS +
            output_channel_base;
          weight_operands = f32(packed_weight[packed_index]);
        }`;
  }
  return `        var weight_operands = vec2<f32>(0.0);
        if (output_channel_base < OUTPUT_CHANNELS) {
          let packed_index =
            (kernel * INPUT_CHANNELS + input_channel) *
              OUTPUT_CHANNEL_PAIRS +
            output_channel_base / 2u;
          weight_operands = vec2<f32>(packed_weight[packed_index]);
        }`;
}

function outputStoresForRow(
  row: number,
  outputsPerLane: 1 | 2,
  hasBias: boolean,
): string {
  if (outputsPerLane === 1) {
    const value = hasBias
      ? `f16(sum${row})`
      : `select(
          sum${row},
          bitcast<f32>(0u),
          (bitcast<u32>(sum${row}) & 0x7fffffffu) == 0u
        )`;
    return `      if (output_channel_base < OUTPUT_CHANNELS) {
        output[
          (batch * OUTPUT_FRAMES + output_time${row}) * OUTPUT_CHANNELS +
          output_channel_base
        ] = ${value};
      }`;
  }
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
  capability: AceOpt0016VaeConv1dFixedSubgroupCapability,
): void {
  if (!device.features.has("shader-f16")) {
    throw new Error(
      "OPT-0016 packed-KIO microtile Conv1D requires WebGPU shader-f16",
    );
  }
  if (
    !device.features.has("subgroups") ||
    capability.subgroupMinSize !== ACE_OPT_0016_VAE_CONV1D_SUBGROUP_SIZE ||
    capability.subgroupMaxSize !== ACE_OPT_0016_VAE_CONV1D_SUBGROUP_SIZE
  ) {
    throw new Error(
      "OPT-0016 packed-KIO microtile Conv1D requires reported fixed 32-lane subgroups",
    );
  }
  const maximumInvocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const maximumSizeX = device.limits.maxComputeWorkgroupSizeX;
  if (
    !Number.isSafeInteger(maximumInvocations) ||
    !Number.isSafeInteger(maximumSizeX) ||
    maximumInvocations < ACE_OPT_0016_VAE_CONV1D_WORKGROUP_SIZE ||
    maximumSizeX < ACE_OPT_0016_VAE_CONV1D_WORKGROUP_SIZE
  ) {
    throw new Error(
      `OPT-0016 packed-KIO microtile Conv1D requires WG${ACE_OPT_0016_VAE_CONV1D_WORKGROUP_SIZE}`,
    );
  }
}

function requireDispatchDimension(
  device: GPUDevice,
  workgroupsX: number,
  workgroupsY: number,
): void {
  const maximum = device.limits.maxComputeWorkgroupsPerDimension;
  if (
    !Number.isSafeInteger(maximum) || maximum < 1 ||
    workgroupsX > maximum || workgroupsY > maximum
  ) {
    throw new RangeError(
      "OPT-0016 packed-KIO microtile Conv1D exceeds the device dispatch dimension",
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
      "OPT-0016 packed-KIO microtile device reported invalid buffer limits",
    );
  }
  for (const [name, bytes] of resources) {
    if (bytes > maximumBinding || bytes > maximumBuffer) {
      throw new RangeError(
        `OPT-0016 packed-KIO microtile ${name} exceeds the device buffer limits`,
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
      "OPT-0016 packed-KIO microtile device reported an invalid storage alignment",
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
      "OPT-0016 packed-KIO microtile device reported an invalid uniform alignment",
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

function convKey(
  plan: AceFp16VaeConv1dPlan,
  variant: AceOpt0016VaeConv1dMicrotileVariant,
  hasBias: boolean,
): string {
  return [
    "fixed32-packed-kio-microtile",
    variant,
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
      `OPT-0016 packed-KIO microtile ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
