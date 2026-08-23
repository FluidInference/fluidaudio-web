import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dBindings,
  type AceFp16VaeConv1dOutputStorage,
  type AceFp16VaeConv1dPlan,
} from "./vae-conv1d-fp16.js";
import {
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_CIN_VECTOR_WIDTH,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_OUTPUTS_PER_LANE,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_ROWS,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_SIZE,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_CHANNELS,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_ROWS,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_WORKGROUP_SIZE,
  planAceOpt0024VaeConv1dDirectDot4SubgroupRange,
} from "./vae-conv1d-fp16-direct-dot4-subgroup.js";
import type {
  AceFp16VaeConv1dFixedSubgroupCapability,
  AceFp16VaeConv1dSubgroupRangePlan,
} from "./vae-conv1d-fp16-subgroup.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export type AceOpt0041VaeK7PartialVariant = "k8" | "k16";

export const ACE_OPT_0041_VAE_K7_K8_PARTIALS_KERNEL_ID =
  "ace-vae-fp16-opt-0041-direct-native-oki-fp16-k8-partials-k7-conv1d-v1" as const;
export const ACE_OPT_0041_VAE_K7_K16_PARTIALS_KERNEL_ID =
  "ace-vae-fp16-opt-0041-direct-native-oki-fp16-k16-partials-k7-conv1d-v1" as const;
export const ACE_OPT_0041_VAE_K7_NATIVE_WEIGHT_LAYOUT =
  "native-o-k-i-fp16" as const;

type AceOpt0041VaeK7KernelId =
  typeof ACE_OPT_0041_VAE_K7_K8_PARTIALS_KERNEL_ID |
  typeof ACE_OPT_0041_VAE_K7_K16_PARTIALS_KERNEL_ID;

const GPU_BUFFER_ALIGNMENT = 4;
const F16_VEC4_ALIGNMENT = 8;
const OUTPUT_RANGE_CONTROL_BYTES = 16;
const MAX_WGSL_U32 = 0xffff_ffff;

export interface AceOpt0041VaeK7BoundedPartialsPlan
  extends AceFp16VaeConv1dPlan {
  readonly variant: AceOpt0041VaeK7PartialVariant;
  readonly partialInner: 8 | 16;
  readonly cin4GroupsPerPartial: 2 | 4;
  readonly inputChannelPartialBlocks: number;
  readonly finalPartialCin4Groups: 1 | 2 | 3 | 4;
  readonly weightLayout: typeof ACE_OPT_0041_VAE_K7_NATIVE_WEIGHT_LAYOUT;
}

export interface AceOpt0041VaeK7BoundedPartialsDispatch {
  readonly label: string;
  readonly kernelId: AceOpt0041VaeK7KernelId;
  readonly plan: AceOpt0041VaeK7BoundedPartialsPlan;
  readonly outputRange: AceFp16VaeConv1dSubgroupRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

interface NamedBinding {
  readonly name: "input" | "weight" | "bias" | "output" | "range control";
  readonly binding: GPUBufferBinding;
}

/**
 * Benchmark-only OPT-0041 K8/K16 variants of OPT-0024. The input and native
 * O-K-I weight payloads, WG128/fixed32 ownership, kernel order, and FP32
 * running state are unchanged. Only two or four adjacent FP16 dot4 results
 * live in one bounded local FP16 partial before it is widened once.
 */
export class AceOpt0041VaeK7BoundedPartialsKernel {
  private readonly pipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly variant: AceOpt0041VaeK7PartialVariant,
  ) {}

  static create(
    device: GPUDevice,
    capability: AceFp16VaeConv1dFixedSubgroupCapability,
    variant: AceOpt0041VaeK7PartialVariant,
  ): AceOpt0041VaeK7BoundedPartialsKernel {
    requireVariant(variant);
    requireFixedSubgroupDevice(device);
    requireFixedSubgroupCapability(capability);
    return new AceOpt0041VaeK7BoundedPartialsKernel(device, variant);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceFp16VaeConv1dBindings,
    outputStorage: AceFp16VaeConv1dOutputStorage,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0041VaeK7BoundedPartialsDispatch> {
    this.requireLive();
    const plan = planAceOpt0041VaeK7BoundedPartials(
      shape,
      outputStorage,
      this.variant,
    );
    const outputRange = planAceOpt0041VaeK7BoundedPartialsRange(plan, range);
    this.requireDeviceLimits(plan, outputRange);
    const normalized = this.requireBindings(label, plan, bindings, range);
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
    const bindGroupKey = `${planKey(plan)}:${resources.map((binding) =>
      this.bindingKey(binding)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-opt-0041-${this.variant}-bindings`,
        layout: compiled.bindGroupLayout,
        entries: resources.map((resource, binding) => ({ binding, resource })),
      });
      this.bindGroups.set(bindGroupKey, bindGroup);
    }

    const owner = this;
    return Object.freeze({
      label,
      kernelId: kernelIdFor(this.variant),
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
    plan: AceOpt0041VaeK7BoundedPartialsPlan,
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

  private requireBindings(
    label: string,
    plan: AceOpt0041VaeK7BoundedPartialsPlan,
    bindings: AceFp16VaeConv1dBindings,
    range: AceVaeOutputRangeBinding,
  ): readonly NamedBinding[] {
    if (bindings.bias === undefined) {
      throw new RangeError("OPT-0041 K7 bounded partials require bias");
    }
    const normalized = Object.freeze([
      namedBinding("input", requireStorageBinding(
        this.device,
        bindings.input,
        plan.inputStorageBytes,
        plan.inputBindingBytes,
        F16_VEC4_ALIGNMENT,
        `${label} input`,
      )),
      namedBinding("weight", requireStorageBinding(
        this.device,
        bindings.weight,
        plan.weightStorageBytes,
        plan.weightBindingBytes,
        F16_VEC4_ALIGNMENT,
        `${label} weight`,
      )),
      namedBinding("bias", requireStorageBinding(
        this.device,
        bindings.bias,
        plan.biasStorageBytes,
        plan.biasBindingBytes,
        GPU_BUFFER_ALIGNMENT,
        `${label} bias`,
      )),
      namedBinding("output", requireStorageBinding(
        this.device,
        bindings.output,
        plan.outputStorageBytes,
        plan.outputBindingBytes,
        GPU_BUFFER_ALIGNMENT,
        `${label} output`,
      )),
      namedBinding("range control", requireRangeControlBinding(
        this.device,
        range.control,
        `${label} range control`,
      )),
    ] satisfies readonly NamedBinding[]);
    requireDisjointBindings(normalized, label);
    return normalized;
  }

  private requireDeviceLimits(
    plan: AceOpt0041VaeK7BoundedPartialsPlan,
    range: AceFp16VaeConv1dSubgroupRangePlan,
  ): void {
    const maximumDispatch = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (
      !Number.isSafeInteger(maximumDispatch) || maximumDispatch < 1 ||
      range.workgroupsX > maximumDispatch ||
      range.workgroupsY > maximumDispatch
    ) {
      throw new RangeError(
        "OPT-0041 K7 bounded partials range exceeds the device dispatch dimension",
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
        "OPT-0041 K7 bounded partials device reported invalid buffer limits",
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
          `OPT-0041 K7 bounded partials ${name} exceeds the device storage binding limit`,
        );
      }
      if (bytes > maximumBuffer) {
        throw new RangeError(
          `OPT-0041 K7 bounded partials ${name} exceeds the device buffer limit`,
        );
      }
    }
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
      throw new Error("OPT-0041 K7 bounded partials kernel was destroyed");
    }
  }
}

export function planAceOpt0041VaeK7BoundedPartials(
  shape: AceVaeConv1dShape,
  outputStorage: AceFp16VaeConv1dOutputStorage,
  variant: AceOpt0041VaeK7PartialVariant,
): AceOpt0041VaeK7BoundedPartialsPlan {
  requireVariant(variant);
  const base = planAceFp16VaeConv1d(shape, outputStorage);
  requireCandidateBoundary(base, true);
  const cin4GroupsPerPartial = variant === "k8" ? 2 : 4;
  const inputChannelVec4s =
    base.inputChannels /
    ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_CIN_VECTOR_WIDTH;
  const inputChannelPartialBlocks = Math.ceil(
    inputChannelVec4s / cin4GroupsPerPartial,
  );
  const finalPartialCin4Groups =
    inputChannelVec4s -
    (inputChannelPartialBlocks - 1) * cin4GroupsPerPartial;
  return Object.freeze({
    ...base,
    variant,
    partialInner: variant === "k8" ? 8 : 16,
    cin4GroupsPerPartial,
    inputChannelPartialBlocks,
    finalPartialCin4Groups: finalPartialCin4Groups as 1 | 2 | 3 | 4,
    weightLayout: ACE_OPT_0041_VAE_K7_NATIVE_WEIGHT_LAYOUT,
  });
}

export function planAceOpt0041VaeK7BoundedPartialsRange(
  plan: AceFp16VaeConv1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceFp16VaeConv1dSubgroupRangePlan {
  requireCandidateBoundary(plan, true);
  return planAceOpt0024VaeConv1dDirectDot4SubgroupRange(plan, range);
}

export function aceOpt0041VaeK7BoundedPartialsWgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceFp16VaeConv1dOutputStorage,
  variant: AceOpt0041VaeK7PartialVariant,
): string {
  const plan = planAceOpt0041VaeK7BoundedPartials(
    shape,
    outputStorage,
    variant,
  );
  requireCandidateBoundary(plan, hasBias);
  const rows = ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_ROWS;
  const initialSums = Array.from(
    { length: rows },
    (_, row) => `  var sum${row}: vec4<f32> = initial_sum;`,
  ).join("\n");
  const rowValidity = Array.from(
    { length: rows },
    (_, row) => /* wgsl */ `
      let output_time${row} = tile_first_time + ${row}u;
      let padded_time${row} = output_time${row} + kernel * DILATION;
      var input_valid${row} = false;
      if (output_time${row} < range_end_time && padded_time${row} >= PADDING) {
        input_valid${row} = padded_time${row} - PADDING < INPUT_FRAMES;
      }`,
  ).join("");
  const partialDeclarations = Array.from(
    { length: rows },
    (_, row) => `        var partial${row}: vec4<f16>;`,
  ).join("\n");
  const steps = Array.from(
    { length: plan.cin4GroupsPerPartial },
    (_, step) => boundedStepWgsl(step, rows),
  ).join("\n");
  const widens = Array.from(
    { length: rows },
    (_, row) => /* wgsl */ `
        if (input_valid${row}) {
          sum${row} = sum${row} + vec4<f32>(partial${row});
        }`,
  ).join("");
  const stores = Array.from(
    { length: rows },
    (_, row) => /* wgsl */ `
    let output_time${row} = tile_first_time + ${row}u;
    if (output_time${row} < range_end_time) {
${outputStoresForRow(row)}
    }`,
  ).join("");

  return /* wgsl */ `
// kernel-id: ${kernelIdFor(variant)}
// reduction-semantics: increasing-k-bounded-${variant}-fp16-local-partials-fp32-running-state
// native-layout: input=NLC-vec4-f16; weight=OKI-vec4-f16
enable f16;
enable subgroups;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNEL_VEC4S: u32 = ${plan.inputChannels / 4}u;
const INPUT_CHANNEL_PARTIAL_BLOCKS: u32 = ${plan.inputChannelPartialBlocks}u;
const CIN4_GROUPS_PER_PARTIAL: u32 = ${plan.cin4GroupsPerPartial}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const PADDING: u32 = ${plan.padding}u;
const DILATION: u32 = ${plan.dilation}u;

@group(0) @binding(0) var<storage, read> input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<f16>>;
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

@compute @workgroup_size(
  ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_WORKGROUP_SIZE},
  1,
  1,
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (subgroup_size == ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_SIZE}u) {
    let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
    let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
    let range_first_time = first_output_row % OUTPUT_FRAMES;
    let batch = first_output_row / OUTPUT_FRAMES;
    let range_end_time = range_first_time + output_row_count;
    let tile_first_time = range_first_time +
      group.x * ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_ROWS}u +
      subgroup * ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_ROWS}u;
    let output_channel_base =
      group.y * ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_CHANNELS}u +
      subgroup_lane *
        ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_OUTPUTS_PER_LANE}u;

    let initial_sum = vec4<f32>(
      f32(bias[output_channel_base]),
      f32(bias[output_channel_base + 1u]),
      f32(bias[output_channel_base + 2u]),
      f32(bias[output_channel_base + 3u])
    );
${initialSums}

    // K and Cin-block order remain increasing. At most ${plan.partialInner}
    // products live in FP16; every inter-block running accumulator is FP32.
    for (var kernel = 0u; kernel < 7u; kernel += 1u) {${rowValidity}
      for (
        var input_channel_partial = 0u;
        input_channel_partial < INPUT_CHANNEL_PARTIAL_BLOCKS;
        input_channel_partial += 1u
      ) {
${partialDeclarations}
${steps}
${widens}
      }
    }
${stores}
  }
}
`;
}

function boundedStepWgsl(step: number, rows: number): string {
  const broadcasts = Array.from(
    { length: rows },
    (_, row) =>
      `          let input_operand${row}_${step} = subgroupBroadcast(lane_input_${step}, ${row}u);`,
  ).join("\n");
  const contractions = Array.from(
    { length: rows },
    (_, row) => {
      const operator = step === 0 ? "=" : `= partial${row} +`;
      return /* wgsl */ `
          if (input_valid${row}) {
            partial${row} ${operator} vec4<f16>(
              dot(input_operand${row}_${step}, weight0_${step}),
              dot(input_operand${row}_${step}, weight1_${step}),
              dot(input_operand${row}_${step}, weight2_${step}),
              dot(input_operand${row}_${step}, weight3_${step})
            );
          }`;
    },
  ).join("");
  return /* wgsl */ `
        {
          let input_channel4_${step} =
            input_channel_partial * CIN4_GROUPS_PER_PARTIAL + ${step}u;
          if (input_channel4_${step} < INPUT_CHANNEL_VEC4S) {
            var lane_input_${step} = vec4<f16>(0.0h);
            if (subgroup_lane <
              ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_ROWS}u)
            {
              let lane_output_time_${step} = tile_first_time + subgroup_lane;
              let lane_padded_time_${step} =
                lane_output_time_${step} + kernel * DILATION;
              if (
                lane_output_time_${step} < range_end_time &&
                lane_padded_time_${step} >= PADDING
              ) {
                let lane_input_time_${step} = lane_padded_time_${step} - PADDING;
                if (lane_input_time_${step} < INPUT_FRAMES) {
                  lane_input_${step} = input[
                    (batch * INPUT_FRAMES + lane_input_time_${step}) *
                      INPUT_CHANNEL_VEC4S + input_channel4_${step}
                  ];
                }
              }
            }
${broadcasts}

            let weight0_${step} = weight[
              (output_channel_base * 7u + kernel) * INPUT_CHANNEL_VEC4S +
                input_channel4_${step}
            ];
            let weight1_${step} = weight[
              ((output_channel_base + 1u) * 7u + kernel) *
                INPUT_CHANNEL_VEC4S + input_channel4_${step}
            ];
            let weight2_${step} = weight[
              ((output_channel_base + 2u) * 7u + kernel) *
                INPUT_CHANNEL_VEC4S + input_channel4_${step}
            ];
            let weight3_${step} = weight[
              ((output_channel_base + 3u) * 7u + kernel) *
                INPUT_CHANNEL_VEC4S + input_channel4_${step}
            ];${contractions}
          }
        }`;
}

async function compileKernel(
  device: GPUDevice,
  plan: AceOpt0041VaeK7BoundedPartialsPlan,
): Promise<CompiledKernel> {
  const label = `ace-opt-0041-vae-${plan.variant}-${planKey(plan)}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0041VaeK7BoundedPartialsWgsl(
      plan,
      true,
      "float16",
      plan.variant,
    ),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length !== 0) {
    throw new Error(
      `OPT-0041 ${plan.variant} K7 WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  const bindingSizes = [
    plan.inputBindingBytes,
    plan.weightBindingBytes,
    plan.biasBindingBytes,
    plan.outputBindingBytes,
  ];
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: [
      ...bindingSizes.map((minBindingSize, binding) => ({
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

function outputStoresForRow(row: number): string {
  return Array.from(
    {
      length:
        ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_OUTPUTS_PER_LANE,
    },
    (_, component) => {
      const suffix = component === 0 ? "" : ` + ${component}u`;
      const swizzle = ["x", "y", "z", "w"][component]!;
      return `      output[
        (batch * OUTPUT_FRAMES + output_time${row}) * OUTPUT_CHANNELS +
          output_channel_base${suffix}
      ] = f16(sum${row}.${swizzle});`;
    },
  ).join("\n");
}

function requireFixedSubgroupDevice(device: GPUDevice): void {
  if (!device.features.has("shader-f16") || !device.features.has("subgroups")) {
    throw new Error(
      "OPT-0041 K7 bounded partials require shader-f16 and fixed 32-lane subgroups",
    );
  }
  if (
    device.limits.maxComputeInvocationsPerWorkgroup <
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_WORKGROUP_SIZE ||
    device.limits.maxComputeWorkgroupSizeX <
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_WORKGROUP_SIZE
  ) {
    throw new Error(
      `OPT-0041 K7 bounded partials require WG${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_WORKGROUP_SIZE}`,
    );
  }
}

function requireFixedSubgroupCapability(
  capability: AceFp16VaeConv1dFixedSubgroupCapability,
): void {
  if (
    capability.subgroupMinSize !==
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_SIZE ||
    capability.subgroupMaxSize !==
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_SIZE
  ) {
    throw new Error(
      "OPT-0041 K7 bounded partials require shader-f16 and fixed 32-lane subgroups",
    );
  }
}

function requireCandidateBoundary(
  plan: AceFp16VaeConv1dPlan,
  hasBias: boolean,
): void {
  if (plan.family !== "k7") {
    throw new RangeError("OPT-0041 K7 bounded partials support only K7");
  }
  if (!hasBias) {
    throw new RangeError("OPT-0041 K7 bounded partials require bias");
  }
  if (plan.outputStorage !== "float16") {
    throw new RangeError(
      "OPT-0041 K7 bounded partials require FP16 internal output",
    );
  }
  if (
    plan.inputChannels %
        ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_CIN_VECTOR_WIDTH !==
      0
  ) {
    throw new RangeError(
      "OPT-0041 K7 bounded partials require Cin divisible by 4",
    );
  }
  if (
    plan.outputChannels %
        ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_CHANNELS !==
      0
  ) {
    throw new RangeError(
      "OPT-0041 K7 bounded partials require Cout divisible by 128",
    );
  }
}

function requireVariant(variant: AceOpt0041VaeK7PartialVariant): void {
  if (variant !== "k8" && variant !== "k16") {
    throw new RangeError(`OPT-0041 unknown K7 partial variant ${variant}`);
  }
}

function kernelIdFor(
  variant: AceOpt0041VaeK7PartialVariant,
): AceOpt0041VaeK7KernelId {
  return variant === "k8"
    ? ACE_OPT_0041_VAE_K7_K8_PARTIALS_KERNEL_ID
    : ACE_OPT_0041_VAE_K7_K16_PARTIALS_KERNEL_ID;
}

function planKey(plan: AceOpt0041VaeK7BoundedPartialsPlan): string {
  return [
    "opt-0041",
    plan.variant,
    plan.batch,
    plan.inputFrames,
    plan.inputChannels,
    plan.outputChannels,
    plan.kernelSize,
    plan.stride,
    plan.dilation,
    plan.padding,
    plan.outputStorage,
    "bias",
  ].join("x");
}

function namedBinding(
  name: NamedBinding["name"],
  binding: GPUBufferBinding,
): NamedBinding {
  return Object.freeze({ name, binding });
}

function requireStorageBinding(
  device: GPUDevice,
  binding: GPUBufferBinding,
  requiredStorageBytes: number,
  requiredBindingBytes: number,
  requiredViewAlignment: number,
  label: string,
): GPUBufferBinding {
  const deviceAlignment = device.limits.minStorageBufferOffsetAlignment;
  if (!isValidGpuAlignment(deviceAlignment)) {
    throw new Error(
      "OPT-0041 K7 bounded partials device reported an invalid storage alignment",
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
    offset % deviceAlignment !== 0 ||
    offset % requiredViewAlignment !== 0 ||
    available % GPU_BUFFER_ALIGNMENT !== 0 ||
    bufferBytes % GPU_BUFFER_ALIGNMENT !== 0 ||
    requiredStorageBytes % requiredViewAlignment !== 0 ||
    requiredBindingBytes % requiredViewAlignment !== 0
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
      "OPT-0041 K7 bounded partials device reported an invalid uniform alignment",
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

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
