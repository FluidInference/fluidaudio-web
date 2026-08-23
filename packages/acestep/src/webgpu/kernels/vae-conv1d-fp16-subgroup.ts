import {
  planAceFp16VaeConv1d,
  planAceFp16VaeConv1dRange,
  type AceFp16VaeConv1dBindings,
  type AceFp16VaeConv1dOutputStorage,
  type AceFp16VaeConv1dPlan,
} from "./vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_FP16_VAE_CONV1D_SUBGROUP_SIZE = 32;
export const ACE_FP16_VAE_CONV1D_SUBGROUP_WORKGROUP_SIZE = 128;
export const ACE_FP16_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP = 4;
export const ACE_FP16_VAE_CONV1D_SUBGROUP_ROWS = 8;
export const ACE_FP16_VAE_CONV1D_SUBGROUP_OUTPUTS_PER_LANE = 4;
export const ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_ROWS = 32;
export const ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_CHANNELS = 128;
export const ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID =
  "ace-vae-fp16-fixed32-subgroup-k7-conv1d-v1" as const;

const GPU_BUFFER_ALIGNMENT = 4;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_CONTROL_BYTES = 16;

export interface AceFp16VaeConv1dFixedSubgroupCapability {
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
}

export interface AceFp16VaeConv1dSubgroupRangePlan {
  readonly base: number;
  readonly count: number;
  readonly batch: number;
  readonly firstOutputTime: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceFp16VaeConv1dSubgroupDispatch {
  readonly label: string;
  readonly kernelId: typeof ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly outputRange: AceFp16VaeConv1dSubgroupRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledAceFp16VaeConv1dSubgroup {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

interface NamedBinding {
  readonly name: "input" | "weight" | "bias" | "output" | "range control";
  readonly binding: GPUBufferBinding;
}

/**
 * Fixed-32 subgroup implicit-im2col K7 Conv1D for the FP16 VAE.
 *
 * One subgroup owns eight time rows and 128 output channels. Each lane keeps
 * four output channels in FP32 registers, while lanes 0..7 load one input row
 * and broadcast it across the subgroup. The kernel walks K then Cin in the
 * same increasing order as the portable correctness implementation, but does
 * not stage either operand through workgroup memory or execute an inner-loop
 * workgroup barrier.
 *
 * There is deliberately no fallback. Callers must pass the adapter's exact
 * subgroup bounds, and this owner rejects anything except a fixed 32-lane
 * device before shader compilation.
 */
export class AceFp16VaeConv1dSubgroupKernel {
  private readonly pipelines = new Map<
    string,
    Promise<CompiledAceFp16VaeConv1dSubgroup>
  >();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: AceFp16VaeConv1dFixedSubgroupCapability,
  ): AceFp16VaeConv1dSubgroupKernel {
    requireFixedSubgroupDevice(device, capability);
    return new AceFp16VaeConv1dSubgroupKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceFp16VaeConv1dBindings,
    outputStorage: AceFp16VaeConv1dOutputStorage,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceFp16VaeConv1dSubgroupDispatch> {
    this.requireLive();
    const plan = planAceFp16VaeConv1d(shape, outputStorage);
    requireSubgroupK7(plan);
    requireBiasBoundary(plan, bindings.bias !== undefined);
    const outputRange = planAceFp16VaeConv1dSubgroupRange(plan, range);
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
        label: `${label}-fp16-subgroup-conv1d-bindings`,
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
      kernelId: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
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
  ): Promise<CompiledAceFp16VaeConv1dSubgroup> {
    const key = convKey(plan, hasBias);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceFp16VaeConv1dSubgroup(
      this.device,
      plan,
      hasBias,
    );
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
    range: AceFp16VaeConv1dSubgroupRangePlan,
  ): void {
    const maximumDispatch = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (
      !Number.isSafeInteger(maximumDispatch) || maximumDispatch < 1 ||
      range.workgroupsX > maximumDispatch ||
      range.workgroupsY > maximumDispatch
    ) {
      throw new RangeError(
        "ACE FP16 VAE subgroup Conv1D range exceeds the device dispatch dimension",
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
        "ACE FP16 VAE subgroup Conv1D device reported invalid buffer limits",
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
          `ACE FP16 VAE subgroup Conv1D ${name} exceeds the device storage binding limit`,
        );
      }
      if (bytes > maximumBuffer) {
        throw new RangeError(
          `ACE FP16 VAE subgroup Conv1D ${name} exceeds the device buffer limit`,
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
      throw new Error("ACE FP16 VAE subgroup Conv1D kernel was destroyed");
    }
  }
}

export function planAceFp16VaeConv1dSubgroupRange(
  plan: AceFp16VaeConv1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceFp16VaeConv1dSubgroupRangePlan {
  requireSubgroupK7(plan);
  const portableRange = planAceFp16VaeConv1dRange(plan, range);
  const workgroupsX = Math.ceil(
    portableRange.outputRowCount / ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_ROWS,
  );
  const workgroupsY = Math.ceil(
    plan.outputChannels / ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_CHANNELS,
  );
  requireWgslIndexable(workgroupsX, "subgroup workgroups X");
  requireWgslIndexable(workgroupsY, "subgroup workgroups Y");
  return Object.freeze({
    base: portableRange.base,
    count: portableRange.count,
    batch: portableRange.batch,
    firstOutputTime: portableRange.firstOutputTime,
    firstOutputRow: portableRange.firstOutputRow,
    outputRowCount: portableRange.outputRowCount,
    workgroupsX,
    workgroupsY,
  });
}

export function aceFp16VaeConv1dSubgroupWgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceFp16VaeConv1dOutputStorage,
): string {
  const plan = planAceFp16VaeConv1d(shape, outputStorage);
  requireSubgroupK7(plan);
  requireBiasBoundary(plan, hasBias);
  const outputBinding = hasBias ? 3 : 2;
  const rangeBinding = hasBias ? 4 : 3;
  const biasDeclaration = hasBias
    ? "@group(0) @binding(2) var<storage, read> bias: array<f16>;"
    : "";
  const outputElementType = outputStorage === "float16" ? "f16" : "f32";
  const initialSums = Array.from(
    { length: ACE_FP16_VAE_CONV1D_SUBGROUP_ROWS },
    (_, row) => `  var sum${row}: vec4<f32> = initial_sum;`,
  ).join("\n");
  const inputValidity = Array.from(
    { length: ACE_FP16_VAE_CONV1D_SUBGROUP_ROWS },
    (_, row) => `
      let output_time${row} = tile_first_time + ${row}u;
      let padded_time${row} = output_time${row} + kernel * DILATION;
      var input_valid${row} = false;
      if (output_time${row} < range_end_time && padded_time${row} >= PADDING) {
        input_valid${row} = padded_time${row} - PADDING < INPUT_FRAMES;
      }`,
  ).join("");
  const broadcastsAndAdds = Array.from(
    { length: ACE_FP16_VAE_CONV1D_SUBGROUP_ROWS },
    (_, row) => `
      let input_operand${row} = subgroupBroadcast(lane_input, ${row}u);
      if (input_valid${row}) {
        sum${row} = sum${row} +
          vec4<f32>(input_operand${row}) * weight_operands;
      }`,
  ).join("");
  const stores = Array.from(
    { length: ACE_FP16_VAE_CONV1D_SUBGROUP_ROWS },
    (_, row) => `
    let output_time${row} = tile_first_time + ${row}u;
    if (output_time${row} < range_end_time) {
${outputStoresForRow(row, hasBias)}
    }`,
  ).join("");
  return /* wgsl */ `
// kernel-id: ${ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID}
enable f16;
enable subgroups;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const PADDING: u32 = ${plan.padding}u;
const DILATION: u32 = ${plan.dilation}u;

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

@compute @workgroup_size(
  ${ACE_FP16_VAE_CONV1D_SUBGROUP_WORKGROUP_SIZE},
  1,
  1,
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (subgroup_size == ${ACE_FP16_VAE_CONV1D_SUBGROUP_SIZE}u) {
    let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
    let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
    let range_first_time = first_output_row % OUTPUT_FRAMES;
    let batch = first_output_row / OUTPUT_FRAMES;
    let range_end_time = range_first_time + output_row_count;
    let tile_first_time = range_first_time +
      group.x * ${ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_ROWS}u +
      subgroup * ${ACE_FP16_VAE_CONV1D_SUBGROUP_ROWS}u;
    let output_channel_base =
      group.y * ${ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_CHANNELS}u +
      subgroup_lane * ${ACE_FP16_VAE_CONV1D_SUBGROUP_OUTPUTS_PER_LANE}u;

${initialSumVectorWgsl(hasBias)}
${initialSums}

    // Exact K-outer, increasing-Cin FP32 accumulation order. Inputs are
    // broadcast across 128 output channels; native O-K-I weights stay direct.
    for (var kernel = 0u; kernel < 7u; kernel += 1u) {
      for (
        var input_channel = 0u;
        input_channel < INPUT_CHANNELS;
        input_channel += 1u
      ) {
${inputValidity}
        var lane_input: f32 = 0.0;
        if (subgroup_lane < ${ACE_FP16_VAE_CONV1D_SUBGROUP_ROWS}u) {
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
${weightVectorWgsl()}
${broadcastsAndAdds}
      }
    }

${stores}
  }
}
`;
}

async function compileAceFp16VaeConv1dSubgroup(
  device: GPUDevice,
  plan: AceFp16VaeConv1dPlan,
  hasBias: boolean,
): Promise<CompiledAceFp16VaeConv1dSubgroup> {
  const label = `ace-fp16-vae-subgroup-conv1d-${convKey(plan, hasBias)}`;
  const module = device.createShaderModule({
    label,
    code: aceFp16VaeConv1dSubgroupWgsl(
      plan,
      hasBias,
      plan.outputStorage,
    ),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `ACE FP16 VAE subgroup Conv1D WGSL failed: ${errors.map((message) =>
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

function initialSumVectorWgsl(hasBias: boolean): string {
  if (!hasBias) return "  let initial_sum = vec4<f32>(0.0);";
  return `  var bias0: f32 = 0.0;
  var bias1: f32 = 0.0;
  var bias2: f32 = 0.0;
  var bias3: f32 = 0.0;
  if (output_channel_base < OUTPUT_CHANNELS) {
    bias0 = f32(bias[output_channel_base]);
  }
  if (output_channel_base + 1u < OUTPUT_CHANNELS) {
    bias1 = f32(bias[output_channel_base + 1u]);
  }
  if (output_channel_base + 2u < OUTPUT_CHANNELS) {
    bias2 = f32(bias[output_channel_base + 2u]);
  }
  if (output_channel_base + 3u < OUTPUT_CHANNELS) {
    bias3 = f32(bias[output_channel_base + 3u]);
  }
  let initial_sum = vec4<f32>(bias0, bias1, bias2, bias3);`;
}

function weightVectorWgsl(): string {
  return `        var weight0: f32 = 0.0;
        var weight1: f32 = 0.0;
        var weight2: f32 = 0.0;
        var weight3: f32 = 0.0;
        if (output_channel_base < OUTPUT_CHANNELS) {
          weight0 = f32(weight[
            (output_channel_base * 7u + kernel) * INPUT_CHANNELS +
            input_channel
          ]);
        }
        if (output_channel_base + 1u < OUTPUT_CHANNELS) {
          weight1 = f32(weight[
            ((output_channel_base + 1u) * 7u + kernel) * INPUT_CHANNELS +
            input_channel
          ]);
        }
        if (output_channel_base + 2u < OUTPUT_CHANNELS) {
          weight2 = f32(weight[
            ((output_channel_base + 2u) * 7u + kernel) * INPUT_CHANNELS +
            input_channel
          ]);
        }
        if (output_channel_base + 3u < OUTPUT_CHANNELS) {
          weight3 = f32(weight[
            ((output_channel_base + 3u) * 7u + kernel) * INPUT_CHANNELS +
            input_channel
          ]);
        }
        let weight_operands = vec4<f32>(
          weight0,
          weight1,
          weight2,
          weight3,
        );`;
}

function outputStoresForRow(row: number, hasBias: boolean): string {
  return Array.from(
    { length: ACE_FP16_VAE_CONV1D_SUBGROUP_OUTPUTS_PER_LANE },
    (_, component) => {
      const suffix = component === 0 ? "" : ` + ${component}u`;
      const swizzle = ["x", "y", "z", "w"][component]!;
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
    },
  ).join("\n");
}

function requireFixedSubgroupDevice(
  device: GPUDevice,
  capability: AceFp16VaeConv1dFixedSubgroupCapability,
): void {
  if (!device.features.has("shader-f16")) {
    throw new Error(
      "ACE FP16 VAE subgroup Conv1D requires WebGPU shader-f16",
    );
  }
  if (
    !device.features.has("subgroups") ||
    capability.subgroupMinSize !== ACE_FP16_VAE_CONV1D_SUBGROUP_SIZE ||
    capability.subgroupMaxSize !== ACE_FP16_VAE_CONV1D_SUBGROUP_SIZE
  ) {
    throw new Error(
      "ACE FP16 VAE subgroup Conv1D requires reported fixed 32-lane subgroups",
    );
  }
  const maximumInvocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const maximumSizeX = device.limits.maxComputeWorkgroupSizeX;
  if (
    !Number.isSafeInteger(maximumInvocations) ||
    !Number.isSafeInteger(maximumSizeX) ||
    maximumInvocations < ACE_FP16_VAE_CONV1D_SUBGROUP_WORKGROUP_SIZE ||
    maximumSizeX < ACE_FP16_VAE_CONV1D_SUBGROUP_WORKGROUP_SIZE
  ) {
    throw new Error(
      `ACE FP16 VAE subgroup Conv1D requires WG${ACE_FP16_VAE_CONV1D_SUBGROUP_WORKGROUP_SIZE}`,
    );
  }
}

function requireSubgroupK7(plan: AceFp16VaeConv1dPlan): void {
  if (plan.family !== "k7") {
    throw new RangeError(
      "ACE FP16 VAE fixed-32 subgroup Conv1D supports only K7",
    );
  }
}

function requireBiasBoundary(
  plan: AceFp16VaeConv1dPlan,
  hasBias: boolean,
): void {
  if (hasBias && plan.outputStorage === "float32") {
    throw new RangeError(
      "ACE FP16 VAE subgroup Conv1D FP32 output is reserved for the final no-bias raw-waveform boundary",
    );
  }
  if (!hasBias && plan.outputStorage !== "float32") {
    throw new RangeError(
      "ACE FP16 VAE subgroup Conv1D bias may be omitted only at the final FP32 raw-waveform boundary",
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
      "ACE FP16 VAE subgroup Conv1D device reported an invalid storage alignment",
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
      "ACE FP16 VAE subgroup Conv1D device reported an invalid uniform alignment",
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
    "fixed32-subgroup",
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
      `ACE FP16 VAE subgroup Conv1D ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
