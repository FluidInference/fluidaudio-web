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

export const ACE_OPT_0046_VAE_CONV1D_BRANCH_FREE_INTERIOR_KERNEL_ID =
  "ace-vae-fp16-opt-0046-branch-free-interior-native-oki-fp16-dot4-k7-conv1d-v1" as const;
export const ACE_OPT_0046_VAE_CONV1D_BRANCH_FREE_INTERIOR_SEMANTICS =
  "opt-0024-k-cin4-order-with-all-seven-taps-proven-valid" as const;

const GPU_BUFFER_ALIGNMENT = 4;
const F16_VEC4_ALIGNMENT = 8;
const OUTPUT_RANGE_CONTROL_BYTES = 16;
const MAX_WGSL_U32 = 0xffff_ffff;

export type AceOpt0046VaeConv1dPartitionKind =
  | "prefix-boundary"
  | "interior"
  | "suffix-boundary";

export interface AceOpt0046VaeConv1dPartitionSegment {
  readonly kind: AceOpt0046VaeConv1dPartitionKind;
  readonly base: number;
  readonly count: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
}

export interface AceOpt0046VaeConv1dPartition {
  readonly base: number;
  readonly count: number;
  readonly batch: number;
  readonly firstOutputTime: number;
  readonly outputRowCount: number;
  readonly safeInteriorFirstTime: number;
  readonly safeInteriorEndTime: number;
  readonly segments: readonly AceOpt0046VaeConv1dPartitionSegment[];
  readonly prefix?: AceOpt0046VaeConv1dPartitionSegment;
  readonly interior?: AceOpt0046VaeConv1dPartitionSegment;
  readonly suffix?: AceOpt0046VaeConv1dPartitionSegment;
}

export interface AceOpt0046VaeConv1dBranchFreeInteriorDispatch {
  readonly label: string;
  readonly kernelId:
    typeof ACE_OPT_0046_VAE_CONV1D_BRANCH_FREE_INTERIOR_KERNEL_ID;
  readonly plan: AceFp16VaeConv1dPlan;
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
 * Split one OPT-0024 range into unchanged boundary work and branch-free,
 * complete 32-row interior tiles. Empty pieces are omitted. The interior is
 * the largest prefix of the all-taps-valid intersection that the frozen
 * 32-row ownership can execute without padded lanes.
 */
export function planAceOpt0046VaeConv1dBranchFreePartition(
  plan: AceFp16VaeConv1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceOpt0046VaeConv1dPartition {
  const whole = planAceOpt0024VaeConv1dDirectDot4SubgroupRange(plan, range);
  const safeInteriorFirstTime = plan.padding;
  const safeInteriorEndTime = plan.inputFrames - plan.padding;
  const requestedFirstTime = whole.firstOutputTime;
  const requestedEndTime = requestedFirstTime + whole.outputRowCount;
  const eligibleFirstTime = Math.min(
    requestedEndTime,
    Math.max(requestedFirstTime, safeInteriorFirstTime),
  );
  const eligibleEndTime = Math.max(
    eligibleFirstTime,
    Math.min(requestedEndTime, safeInteriorEndTime),
  );
  const eligibleRows = Math.max(0, eligibleEndTime - eligibleFirstTime);
  const interiorRows = Math.floor(
    eligibleRows /
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_ROWS,
  ) * ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_ROWS;
  const interiorEndTime = eligibleFirstTime + interiorRows;
  const batchRowBase = whole.batch * plan.outputFrames;
  const makeSegment = (
    kind: AceOpt0046VaeConv1dPartitionKind,
    firstTime: number,
    endTime: number,
  ): AceOpt0046VaeConv1dPartitionSegment | undefined => {
    const rows = endTime - firstTime;
    if (rows === 0) return undefined;
    if (rows < 0) throw new Error("OPT-0046 partition produced negative rows");
    const firstOutputRow = batchRowBase + firstTime;
    return Object.freeze({
      kind,
      base: firstOutputRow * plan.outputChannels,
      count: rows * plan.outputChannels,
      firstOutputRow,
      outputRowCount: rows,
    });
  };
  const noInteriorBoundary = interiorRows === 0
    ? makeSegment(
        requestedFirstTime < safeInteriorFirstTime
          ? "prefix-boundary"
          : "suffix-boundary",
        requestedFirstTime,
        requestedEndTime,
      )
    : undefined;
  const prefix = interiorRows === 0
    ? noInteriorBoundary?.kind === "prefix-boundary"
      ? noInteriorBoundary
      : undefined
    : makeSegment(
        "prefix-boundary",
        requestedFirstTime,
        eligibleFirstTime,
      );
  const interior = interiorRows === 0 ? undefined : makeSegment(
    "interior",
    eligibleFirstTime,
    interiorEndTime,
  );
  const suffix = interiorRows === 0
    ? noInteriorBoundary?.kind === "suffix-boundary"
      ? noInteriorBoundary
      : undefined
    : makeSegment(
        "suffix-boundary",
        interiorEndTime,
        requestedEndTime,
      );
  const segments = Object.freeze(
    [prefix, interior, suffix].filter(
      (segment): segment is AceOpt0046VaeConv1dPartitionSegment =>
        segment !== undefined,
    ),
  );
  if (
    segments.length === 0 ||
    segments[0]!.base !== whole.base ||
    segments.reduce((sum, segment) => sum + segment.count, 0) !== whole.count ||
    segments.some((segment, index) =>
      index > 0 &&
      segments[index - 1]!.base + segments[index - 1]!.count !== segment.base
    ) ||
    (interior !== undefined &&
      (interior.outputRowCount %
          ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_ROWS !== 0 ||
        interior.firstOutputRow % plan.outputFrames < safeInteriorFirstTime ||
        interior.firstOutputRow % plan.outputFrames +
            interior.outputRowCount > safeInteriorEndTime))
  ) {
    throw new Error("OPT-0046 partition lost disjoint complete coverage");
  }
  return Object.freeze({
    base: whole.base,
    count: whole.count,
    batch: whole.batch,
    firstOutputTime: whole.firstOutputTime,
    outputRowCount: whole.outputRowCount,
    safeInteriorFirstTime,
    safeInteriorEndTime,
    segments,
    ...(prefix === undefined ? {} : { prefix }),
    ...(interior === undefined ? {} : { interior }),
    ...(suffix === undefined ? {} : { suffix }),
  });
}

export function planAceOpt0046VaeConv1dBranchFreeInteriorRange(
  plan: AceFp16VaeConv1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceFp16VaeConv1dSubgroupRangePlan {
  const planned = planAceOpt0024VaeConv1dDirectDot4SubgroupRange(plan, range);
  if (
    planned.outputRowCount %
        ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_ROWS !== 0 ||
    planned.firstOutputTime < plan.padding ||
    planned.firstOutputTime + planned.outputRowCount >
      plan.inputFrames - plan.padding
  ) {
    throw new RangeError(
      "OPT-0046 interior range must contain complete 32-row all-taps-valid tiles",
    );
  }
  return planned;
}

/** Isolated benchmark-only owner for the interior part of OPT-0046. */
export class AceOpt0046VaeConv1dBranchFreeInteriorKernel {
  private readonly pipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: AceFp16VaeConv1dFixedSubgroupCapability,
  ): AceOpt0046VaeConv1dBranchFreeInteriorKernel {
    requireFixed32Device(device, capability);
    return new AceOpt0046VaeConv1dBranchFreeInteriorKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceFp16VaeConv1dBindings,
    outputStorage: AceFp16VaeConv1dOutputStorage,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0046VaeConv1dBranchFreeInteriorDispatch> {
    this.requireLive();
    const plan = planAceFp16VaeConv1d(shape, outputStorage);
    requireCandidate(plan, bindings.bias !== undefined);
    const outputRange = planAceOpt0046VaeConv1dBranchFreeInteriorRange(
      plan,
      range,
    );
    this.requireDeviceLimits(plan, outputRange);
    const normalized = this.requireBindings(label, plan, bindings, range);
    const compiled = await this.pipelineFor(plan);
    this.requireLive();

    const rangeIndex = normalized.length - 1;
    const controlOffset = normalized[rangeIndex]!.binding.offset ?? 0;
    const resources = normalized.map(({ binding }, index) =>
      index === rangeIndex
        ? Object.freeze({
            buffer: binding.buffer,
            offset: 0,
            size: OUTPUT_RANGE_CONTROL_BYTES,
          })
        : binding
    );
    const bindGroupKey = `${convKey(plan)}:${resources.map((resource) =>
      this.bindingKey(resource)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-opt-0046-interior-bindings`,
        layout: compiled.bindGroupLayout,
        entries: resources.map((resource, binding) => ({ binding, resource })),
      });
      this.bindGroups.set(bindGroupKey, bindGroup);
    }
    const owner = this;
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0046_VAE_CONV1D_BRANCH_FREE_INTERIOR_KERNEL_ID,
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
    this.pipelines.clear();
    this.bindGroups.clear();
  }

  private pipelineFor(plan: AceFp16VaeConv1dPlan): Promise<CompiledKernel> {
    const key = convKey(plan);
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
    plan: AceFp16VaeConv1dPlan,
    bindings: AceFp16VaeConv1dBindings,
    range: AceVaeOutputRangeBinding,
  ): readonly NamedBinding[] {
    if (bindings.bias === undefined) {
      throw new RangeError("OPT-0046 interior Conv1D requires bias");
    }
    const normalized = Object.freeze([
      named("input", requireStorageBinding(
        this.device,
        bindings.input,
        plan.inputStorageBytes,
        plan.inputBindingBytes,
        F16_VEC4_ALIGNMENT,
        `${label} input`,
      )),
      named("weight", requireStorageBinding(
        this.device,
        bindings.weight,
        plan.weightStorageBytes,
        plan.weightBindingBytes,
        F16_VEC4_ALIGNMENT,
        `${label} weight`,
      )),
      named("bias", requireStorageBinding(
        this.device,
        bindings.bias,
        plan.biasStorageBytes,
        plan.biasBindingBytes,
        GPU_BUFFER_ALIGNMENT,
        `${label} bias`,
      )),
      named("output", requireStorageBinding(
        this.device,
        bindings.output,
        plan.outputStorageBytes,
        plan.outputBindingBytes,
        GPU_BUFFER_ALIGNMENT,
        `${label} output`,
      )),
      named("range control", requireControlBinding(
        this.device,
        range.control,
        `${label} range control`,
      )),
    ] satisfies readonly NamedBinding[]);
    requireDisjoint(normalized, label);
    return normalized;
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
    ) throw new RangeError("OPT-0046 interior dispatch exceeds device limits");
    const maximumBinding = Number(this.device.limits.maxStorageBufferBindingSize);
    const maximumBuffer = Number(this.device.limits.maxBufferSize);
    for (const [name, bytes] of [
      ["input", plan.inputBindingBytes],
      ["weight", plan.weightBindingBytes],
      ["bias", plan.biasBindingBytes],
      ["output", plan.outputBindingBytes],
    ] as const) {
      if (!Number.isSafeInteger(maximumBinding) || bytes > maximumBinding) {
        throw new RangeError(`OPT-0046 interior ${name} exceeds binding limit`);
      }
      if (!Number.isSafeInteger(maximumBuffer) || bytes > maximumBuffer) {
        throw new RangeError(`OPT-0046 interior ${name} exceeds buffer limit`);
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
    if (this.destroyed) throw new Error("OPT-0046 interior kernel was destroyed");
  }
}

export function aceOpt0046VaeConv1dBranchFreeInteriorWgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceFp16VaeConv1dOutputStorage,
): string {
  const plan = planAceFp16VaeConv1d(shape, outputStorage);
  requireCandidate(plan, hasBias);
  const initialSums = Array.from(
    { length: ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_ROWS },
    (_, row) => `  var sum${row}: vec4<f32> = initial_sum;`,
  ).join("\n");
  const broadcasts = Array.from(
    { length: ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_ROWS },
    (_, row) =>
      `      let input_operand${row} = subgroupBroadcast(lane_input, ${row}u);`,
  ).join("\n");
  const partialAdds = Array.from(
    { length: ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_ROWS },
    (_, row) => /* wgsl */ `
      let partial${row} = vec4<f16>(
        dot(input_operand${row}, weight0),
        dot(input_operand${row}, weight1),
        dot(input_operand${row}, weight2),
        dot(input_operand${row}, weight3)
      );
      sum${row} = sum${row} + vec4<f32>(partial${row});`,
  ).join("");
  const stores = Array.from(
    { length: ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_ROWS },
    (_, row) => /* wgsl */ `
    let output_time${row} = tile_first_time + ${row}u;
${outputStores(row)}`,
  ).join("");
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0046_VAE_CONV1D_BRANCH_FREE_INTERIOR_KERNEL_ID}
// reduction-semantics: ${ACE_OPT_0046_VAE_CONV1D_BRANCH_FREE_INTERIOR_SEMANTICS}
// boundary-owner: ace-vae-fp16-opt-0024-direct-native-oki-fp16-dot4-k7-conv1d-v1
enable f16;
enable subgroups;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNEL_VEC4S: u32 = ${
    plan.inputChannels /
    ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_CIN_VECTOR_WIDTH
  }u;
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
@group(0) @binding(4) var<uniform> output_range: OutputRangeParameters;

@compute @workgroup_size(
  ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_WORKGROUP_SIZE}, 1, 1,
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (subgroup_size == ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_SIZE}u) {
    let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
    let range_first_time = first_output_row % OUTPUT_FRAMES;
    let batch = first_output_row / OUTPUT_FRAMES;
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

    // The range planner proves every owned row has all seven taps. K then
    // Cin4 and every FP16 dot / FP32 accumulator update match OPT-0024.
    for (var kernel = 0u; kernel < 7u; kernel += 1u) {
      for (
        var input_channel4 = 0u;
        input_channel4 < INPUT_CHANNEL_VEC4S;
        input_channel4 += 1u
      ) {
        var lane_input: vec4<f16>;
        if (subgroup_lane <
          ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_ROWS}u)
        {
          let lane_output_time = tile_first_time + subgroup_lane;
          let lane_input_time =
            lane_output_time + kernel * DILATION - PADDING;
          lane_input = input[
            (batch * INPUT_FRAMES + lane_input_time) *
              INPUT_CHANNEL_VEC4S + input_channel4
          ];
        }
${broadcasts}
        let weight0 = weight[
          (output_channel_base * 7u + kernel) * INPUT_CHANNEL_VEC4S +
            input_channel4
        ];
        let weight1 = weight[
          ((output_channel_base + 1u) * 7u + kernel) *
            INPUT_CHANNEL_VEC4S + input_channel4
        ];
        let weight2 = weight[
          ((output_channel_base + 2u) * 7u + kernel) *
            INPUT_CHANNEL_VEC4S + input_channel4
        ];
        let weight3 = weight[
          ((output_channel_base + 3u) * 7u + kernel) *
            INPUT_CHANNEL_VEC4S + input_channel4
        ];${partialAdds}
      }
    }
${stores}
  }
}
`;
}

async function compileKernel(
  device: GPUDevice,
  plan: AceFp16VaeConv1dPlan,
): Promise<CompiledKernel> {
  const label = `ace-opt-0046-branch-free-interior-${convKey(plan)}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0046VaeConv1dBranchFreeInteriorWgsl(plan, true, "float16"),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length !== 0) {
    throw new Error(`OPT-0046 interior WGSL failed: ${errors.map((message) =>
      `${message.lineNum}:${message.linePos} ${message.message}`
    ).join("; ")}`);
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

function outputStores(row: number): string {
  return Array.from(
    {
      length:
        ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_OUTPUTS_PER_LANE,
    },
    (_, component) => {
      const offset = component === 0 ? "" : ` + ${component}u`;
      const swizzle = ["x", "y", "z", "w"][component]!;
      return `    output[
      (batch * OUTPUT_FRAMES + output_time${row}) * OUTPUT_CHANNELS +
        output_channel_base${offset}
    ] = f16(sum${row}.${swizzle});`;
    },
  ).join("\n");
}

function requireCandidate(plan: AceFp16VaeConv1dPlan, hasBias: boolean): void {
  if (plan.family !== "k7") {
    throw new RangeError("OPT-0046 interior Conv1D supports only K7");
  }
  if (!hasBias) throw new RangeError("OPT-0046 interior Conv1D requires bias");
  if (plan.outputStorage !== "float16") {
    throw new RangeError("OPT-0046 interior Conv1D requires FP16 output");
  }
  if (
    plan.inputChannels %
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_CIN_VECTOR_WIDTH !== 0
  ) throw new RangeError("OPT-0046 interior Conv1D requires Cin divisible by 4");
  if (
    plan.outputChannels %
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_CHANNELS !== 0
  ) throw new RangeError("OPT-0046 interior Conv1D requires Cout divisible by 128");
}

function requireFixed32Device(
  device: GPUDevice,
  capability: AceFp16VaeConv1dFixedSubgroupCapability,
): void {
  if (!device.features.has("shader-f16") || !device.features.has("subgroups")) {
    throw new Error("OPT-0046 requires shader-f16 and subgroups");
  }
  if (
    capability.subgroupMinSize !==
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_SIZE ||
    capability.subgroupMaxSize !==
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_SIZE
  ) throw new Error("OPT-0046 requires reported fixed 32-lane subgroups");
  if (
    device.limits.maxComputeInvocationsPerWorkgroup <
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_WORKGROUP_SIZE ||
    device.limits.maxComputeWorkgroupSizeX <
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_WORKGROUP_SIZE
  ) throw new Error("OPT-0046 requires WG128");
}

function named(
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
  viewAlignment: number,
  label: string,
): GPUBufferBinding {
  const deviceAlignment = device.limits.minStorageBufferOffsetAlignment;
  const bufferBytes = Number(binding.buffer.size);
  const offset = binding.offset ?? 0;
  const available = binding.size ?? bufferBytes - offset;
  if (
    !validAlignment(deviceAlignment) ||
    !Number.isSafeInteger(bufferBytes) || bufferBytes < 1 ||
    !Number.isSafeInteger(offset) || offset < 0 ||
    !Number.isSafeInteger(available) || available < requiredBindingBytes ||
    offset + available > bufferBytes ||
    offset % deviceAlignment !== 0 || offset % viewAlignment !== 0 ||
    available % GPU_BUFFER_ALIGNMENT !== 0 ||
    requiredStorageBytes % viewAlignment !== 0 ||
    requiredBindingBytes % viewAlignment !== 0
  ) {
    throw new RangeError(`${label} binding does not expose aligned storage`);
  }
  return Object.freeze({ buffer: binding.buffer, offset, size: requiredBindingBytes });
}

function requireControlBinding(
  device: GPUDevice,
  binding: GPUBufferBinding,
  label: string,
): GPUBufferBinding {
  const alignment = device.limits.minUniformBufferOffsetAlignment;
  const bufferBytes = Number(binding.buffer.size);
  const offset = binding.offset ?? 0;
  const available = binding.size ?? bufferBytes - offset;
  if (
    !validAlignment(alignment) ||
    !Number.isSafeInteger(bufferBytes) || bufferBytes < 1 ||
    !Number.isSafeInteger(offset) || offset < 0 || offset > MAX_WGSL_U32 ||
    !Number.isSafeInteger(available) || available < OUTPUT_RANGE_CONTROL_BYTES ||
    offset + OUTPUT_RANGE_CONTROL_BYTES > bufferBytes || offset % alignment !== 0
  ) throw new RangeError(`${label} must expose one aligned 16-byte record`);
  return Object.freeze({ buffer: binding.buffer, offset, size: OUTPUT_RANGE_CONTROL_BYTES });
}

function requireDisjoint(bindings: readonly NamedBinding[], label: string): void {
  for (let leftIndex = 0; leftIndex < bindings.length; leftIndex += 1) {
    const left = bindings[leftIndex]!;
    const leftStart = left.binding.offset ?? 0;
    const leftEnd = leftStart + (left.binding.size ?? 0);
    for (let rightIndex = leftIndex + 1; rightIndex < bindings.length; rightIndex += 1) {
      const right = bindings[rightIndex]!;
      if (left.binding.buffer !== right.binding.buffer) continue;
      const rightStart = right.binding.offset ?? 0;
      const rightEnd = rightStart + (right.binding.size ?? 0);
      if (leftStart < rightEnd && rightStart < leftEnd) {
        throw new RangeError(`${label} ${left.name} and ${right.name} must not overlap`);
      }
    }
  }
}

function convKey(plan: AceFp16VaeConv1dPlan): string {
  return [
    "opt-0046-branch-free-interior",
    plan.batch,
    plan.inputFrames,
    plan.inputChannels,
    plan.outputChannels,
    plan.dilation,
    plan.padding,
    plan.outputStorage,
  ].join("x");
}

function validAlignment(value: number): boolean {
  return Number.isSafeInteger(value) && value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
