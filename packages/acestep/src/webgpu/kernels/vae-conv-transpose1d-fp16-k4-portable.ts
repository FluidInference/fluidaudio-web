import { checkedAceProduct, requireAceDisjointOutput } from "./correctness-utils.js";
import {
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  type AceOpt0036VaeConvTranspose1dRangePlan,
} from "./vae-conv-transpose1d-fp16-reuse-axis-subgroup.js";
import {
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT,
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID,
  planAceOpt0048VaeConvTranspose1dK4,
  planAceOpt0048VaeConvTranspose1dK4Range,
  type AceOpt0048VaeConvTranspose1dK4Plan,
} from "./vae-conv-transpose1d-fp16-k4-partials.js";
import type {
  AceVaeConvTranspose1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R4C8_K4_PORTABLE_KERNEL_ID =
  "opt-0088-vae-conv-transpose1d-r4c8-k4-portable-v1" as const;
export const ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R8C4_K4_PORTABLE_KERNEL_ID =
  "opt-0088-vae-conv-transpose1d-r8c4-k4-portable-v1" as const;
/** Former 32-lane hardware groups become fixed workgroup-memory slices. */
export const ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SLICE_SIZE =
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE;
export const ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SLICES =
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP;
export const ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_WORKGROUP_SIZE =
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE;
const MAX_WORKGROUP_STORAGE_BYTES = 16_384;
const STAGED_INPUT_BYTES_PER_SLOT = 8;
const OUTPUT_RANGE_CONTROL_BYTES = 16;

export type AceOpt0088VaeConvTranspose1dK4PortableKernelId =
  | typeof ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R4C8_K4_PORTABLE_KERNEL_ID
  | typeof ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R8C4_K4_PORTABLE_KERNEL_ID;

/**
 * Portable twin of the OPT-0048 plan. Every layout, ownership, and reduction
 * field is inherited unchanged; only the kernel identity and the workgroup
 * staging accounting differ from the subgroup owner.
 */
export interface AceOpt0088VaeConvTranspose1dK4PortablePlan
  extends Omit<
    AceOpt0048VaeConvTranspose1dK4Plan,
    "kernelId" | "workgroupStorageBytes" | "workgroupBarrierCount"
  > {
  readonly kernelId: AceOpt0088VaeConvTranspose1dK4PortableKernelId;
  readonly emulatedSliceCount: 4;
  readonly emulatedSliceSize: 32;
  readonly workgroupStorageBytes: number;
  readonly workgroupBarrierCount: number;
}

export interface AceOpt0088VaeConvTranspose1dK4PortableBindings {
  readonly input: GPUBufferBinding;
  /** OPT-0048 packed `[phase,tap,Cin4,CoutTile,lane32,output/lane,K4]`. */
  readonly weight: GPUBufferBinding;
  readonly bias: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceOpt0088VaeConvTranspose1dK4PortableDispatch {
  readonly label: string;
  readonly kernelId: AceOpt0088VaeConvTranspose1dK4PortableKernelId;
  readonly operationLabel:
    AceOpt0088VaeConvTranspose1dK4PortablePlan["operationLabel"];
  readonly reuseAxis: "channel" | "row";
  readonly plan: AceOpt0088VaeConvTranspose1dK4PortablePlan;
  readonly outputRange: AceOpt0036VaeConvTranspose1dRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Portable no-subgroups owner for the revision-7 OPT-0048 K4 ConvTranspose1D
 * layout (Safari/Firefox class devices exposing `shader-f16` only).
 *
 * Numerical contract: bit-identical to the OPT-0048 subgroup kernel on the
 * same package bytes. The per-output arithmetic is unchanged — native FP16
 * dot4 partials widened exactly once into FP32 running state, in
 * `tap-ascending-then-input-channel-groups-of-four-ascending` order, with the
 * identical single `f16(sum)` store rounding. The hardware 32-lane broadcast
 * of each row's input vector is replaced by value-identical workgroup-memory
 * staging (one `vec4<f16>` slot per row per 32-lane slice) fenced by
 * `workgroupBarrier()`; no other statement changes.
 */
export class AceOpt0088VaeConvTranspose1dK4PortableKernel {
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
  ): AceOpt0088VaeConvTranspose1dK4PortableKernel {
    return new AceOpt0088VaeConvTranspose1dK4PortableKernel(device);
  }

  async createDispatch(
    label: string,
    operationLabel: string,
    shape: AceVaeConvTranspose1dShape,
    bindings: AceOpt0088VaeConvTranspose1dK4PortableBindings,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0088VaeConvTranspose1dK4PortableDispatch> {
    this.requireLive();
    const plan = planAceOpt0088VaeConvTranspose1dK4Portable(
      operationLabel,
      shape,
    );
    const outputRange = planAceOpt0088VaeConvTranspose1dK4PortableRange(
      plan,
      range,
    );
    requireDevicePlan(this.device, plan, outputRange);
    const resources = [
      normalizeStorageBinding(
        this.device,
        bindings.input,
        plan.inputBindingBytes,
        `${label} input`,
      ),
      normalizeStorageBinding(
        this.device,
        bindings.weight,
        plan.weightBindingBytes,
        `${label} K4 weight`,
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
    ] as const;
    requireAceDisjointOutput(resources[3], [
      resources[0],
      resources[1],
      resources[2],
      range.control,
    ], label);
    const controlOffset = normalizeRangeOffset(this.device, range.control, label);
    const compiled = await this.pipelineFor(plan);
    this.requireLive();
    const controlResource = Object.freeze({
      buffer: range.control.buffer,
      offset: 0,
      size: OUTPUT_RANGE_CONTROL_BYTES,
    });
    const bindGroupKey = `${planKey(plan)}:${[
      ...resources,
      controlResource,
    ].map((binding) => this.bindingKey(binding)).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-opt-0088-k4-portable-bindings`,
        layout: compiled.bindGroupLayout,
        entries: [...resources, controlResource].map((resource, binding) => ({
          binding,
          resource,
        })),
      });
      this.bindGroups.set(bindGroupKey, bindGroup);
    }
    const owner = this;
    return Object.freeze({
      label,
      kernelId: plan.kernelId,
      operationLabel: plan.operationLabel,
      reuseAxis: plan.reuseAxis,
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
    plan: AceOpt0088VaeConvTranspose1dK4PortablePlan,
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
      throw new Error("OPT-0088 portable K4 ConvTranspose1D kernel was destroyed");
    }
  }
}

/**
 * Plan the portable owner over the identical OPT-0048 selection. Every field
 * except the kernel identity and the staging accounting is the frozen OPT-0048
 * plan for the same authenticated operation, so packed weights, ownership, and
 * dispatch geometry cannot diverge from the subgroup owner.
 */
export function planAceOpt0088VaeConvTranspose1dK4Portable(
  operationLabel: string,
  shape: AceVaeConvTranspose1dShape,
): AceOpt0088VaeConvTranspose1dK4PortablePlan {
  const base = planAceOpt0048VaeConvTranspose1dK4(operationLabel, shape);
  const workgroupStorageBytes = checkedAceProduct([
    ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SLICES,
    base.rowsPerSubgroup,
    STAGED_INPUT_BYTES_PER_SLOT,
  ], "OPT-0088 staged input bytes");
  if (workgroupStorageBytes > MAX_WORKGROUP_STORAGE_BYTES) {
    throw new RangeError(
      `OPT-0088 staged input requires ${workgroupStorageBytes} workgroup bytes, over ${MAX_WORKGROUP_STORAGE_BYTES}`,
    );
  }
  const workgroupBarrierCount = checkedAceProduct(
    [2, 2, base.inputChannelK4Groups],
    "OPT-0088 barrier count",
  );
  return Object.freeze({
    ...base,
    kernelId:
      base.kernelId === ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID
        ? ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R4C8_K4_PORTABLE_KERNEL_ID
        : ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R8C4_K4_PORTABLE_KERNEL_ID,
    emulatedSliceCount: ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SLICES,
    emulatedSliceSize: ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SLICE_SIZE,
    workgroupStorageBytes,
    workgroupBarrierCount,
  });
}

/** Delegate to the OPT-0048 range planner so ownership stays byte-identical. */
export function planAceOpt0088VaeConvTranspose1dK4PortableRange(
  plan: AceOpt0088VaeConvTranspose1dK4PortablePlan,
  range: Readonly<{ base: number; count: number }>,
): AceOpt0036VaeConvTranspose1dRangePlan {
  return planAceOpt0048VaeConvTranspose1dK4Range(
    planAceOpt0048VaeConvTranspose1dK4(plan.operationLabel, plan),
    range,
  );
}

/**
 * Generate the portable WGSL. The emitted arithmetic statements — tap
 * validity, guarded input loads, `weight_base`, weight loads, dot4 partial
 * contractions, and stores — are byte-identical to the OPT-0048 subgroup
 * kernel; the only mechanism change is workgroup-memory staging in place of
 * the 32-lane hardware broadcast. The module requires `enable f16;` only.
 */
export function aceOpt0088VaeConvTranspose1dK4PortableWgsl(
  operationLabel: string,
  shape: AceVaeConvTranspose1dShape,
): string {
  const plan = planAceOpt0088VaeConvTranspose1dK4Portable(operationLabel, shape);
  const rows = plan.rowsPerSubgroup;
  const outputs = plan.channelsPerLane;
  const declarations = Array.from({ length: rows }, (_, row) =>
    outputs === 4
      ? `  var sum${row} = bias_value0;`
      : `  var sum${row}_0 = bias_value0;\n  var sum${row}_1 = bias_value1;`
  ).join("\n");
  const rowState = Array.from({ length: rows }, (_, row) => `
  let phase_row${row} = slice_first_phase_row + ${row}u;
  let range_offset${row} = phase + phase_row${row} * STRIDE;
  let output_time${row} = range_first_time + range_offset${row};
  let row_active${row} = range_offset${row} < output_row_count;
  let first_input_time${row} = phase_first_input_time + phase_row${row};
  var input_time${row} = 0u;
  var input_valid${row} = false;`).join("");
  const tapValidity = Array.from({ length: rows }, (_, row) => `
    if (tap == 0u) {
      input_time${row} = first_input_time${row};
      input_valid${row} = input_time${row} < INPUT_FRAMES;
    } else if (first_input_time${row} > 0u) {
      input_time${row} = first_input_time${row} - 1u;
      input_valid${row} = input_time${row} < INPUT_FRAMES;
    } else {
      input_valid${row} = false;
    }`).join("");
  const inputLoads = Array.from({ length: rows }, (_, row) => `
      if (lane == ${row}u && row_active${row} && input_valid${row}) {
        lane_input = input[
          (batch * INPUT_FRAMES + input_time${row}) * INPUT_CHANNEL_K4_GROUPS +
            input_channel4
        ];
      }`).join("");
  const contractions = Array.from({ length: rows }, (_, row) =>
    contractionForRow(row, outputs)
  ).join("");
  const stores = Array.from({ length: rows }, (_, row) =>
    storesForRow(row, outputs)
  ).join("");
  const weightLoads = Array.from({ length: outputs }, (_, output) =>
    `      let weight${output} = packed_weight[weight_base + ${output}u];`
  ).join("\n");
  return /* wgsl */ `
// kernel-id: ${plan.kernelId}
// weight-layout: ${ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT}
// reduction-semantics: ${plan.reductionSemantics}
enable f16;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNEL_K4_GROUPS: u32 = ${plan.inputChannelK4Groups}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const OUTPUT_CHANNEL_TILES: u32 = ${plan.outputChannelTiles}u;
const OUTPUTS_PER_LANE: u32 = ${outputs}u;
const STRIDE: u32 = ${plan.stride}u;
const PADDING: u32 = ${plan.padding}u;

@group(0) @binding(0) var<storage, read> input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> packed_weight: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

struct OutputRangeParameters {
  first_output: u32,
  output_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(4) var<uniform> output_range: OutputRangeParameters;

// One vec4<f16> row-input slot per row per 32-lane slice of the workgroup.
var<workgroup> staged_input:
  array<vec4<f16>, ${ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SLICES * rows}>;

@compute @workgroup_size(
  ${ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_WORKGROUP_SIZE}, 1, 1,
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
) {
  let lane = local_index % ${ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SLICE_SIZE}u;
  let slice = local_index / ${ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SLICE_SIZE}u;
  let slice_base = slice * ${rows}u;
  let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
  let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
  let range_first_time = first_output_row % OUTPUT_FRAMES;
  let batch = first_output_row / OUTPUT_FRAMES;
  let phase = group.z;
  let phase_first_padded_time = range_first_time + phase + PADDING;
  let congruent_kernel = phase_first_padded_time % STRIDE;
  let phase_first_input_time = phase_first_padded_time / STRIDE;
  let slice_first_phase_row =
    group.x * ${plan.rowsPerWorkgroup}u + slice * ${rows}u;
  let output_channel =
    group.y * ${plan.channelsPerWorkgroup}u + lane * ${outputs}u;
  let bias_value0 = vec4<f32>(
    f32(bias[output_channel]),
    f32(bias[output_channel + 1u]),
    f32(bias[output_channel + 2u]),
    f32(bias[output_channel + 3u])
  );
${outputs === 8 ? `  let bias_value1 = vec4<f32>(
    f32(bias[output_channel + 4u]),
    f32(bias[output_channel + 5u]),
    f32(bias[output_channel + 6u]),
    f32(bias[output_channel + 7u])
  );` : ""}
${declarations}
${rowState}

  // Tap, then Cin4, remain increasing. Each native FP16 dot4 is widened
  // exactly once; the running state across Cin4 groups remains FP32.
  for (var tap = 0u; tap < 2u; tap += 1u) {
${tapValidity}
    for (var input_channel4 = 0u;
      input_channel4 < INPUT_CHANNEL_K4_GROUPS;
      input_channel4 += 1u) {
      var lane_input = vec4<f16>(0.0h);
${inputLoads}
      if (lane < ${rows}u) {
        staged_input[slice_base + lane] = lane_input;
      }
      workgroupBarrier();
      let weight_base = (((((congruent_kernel * 2u + tap) *
        INPUT_CHANNEL_K4_GROUPS + input_channel4) * OUTPUT_CHANNEL_TILES +
        group.y) * 32u + lane) * OUTPUTS_PER_LANE);
${weightLoads}
${contractions}
      workgroupBarrier();
    }
  }
${stores}
}
`;
}

function contractionForRow(row: number, outputs: 4 | 8): string {
  const dots = (start: number) => Array.from({ length: 4 }, (_, index) =>
    `dot(input_operand${row}, weight${start + index})`
  ).join(",\n          ");
  return `
      let input_operand${row} = staged_input[slice_base + ${row}u];
      if (row_active${row} && input_valid${row}) {
        let partial${row}_0 = vec4<f16>(
          ${dots(0)}
        );
        sum${row}${outputs === 8 ? "_0" : ""} =
          sum${row}${outputs === 8 ? "_0" : ""} + vec4<f32>(partial${row}_0);
${outputs === 8 ? `        let partial${row}_1 = vec4<f16>(
          ${dots(4)}
        );
        sum${row}_1 = sum${row}_1 + vec4<f32>(partial${row}_1);` : ""}
      }`;
}

function storesForRow(row: number, outputs: 4 | 8): string {
  const stores = Array.from({ length: outputs }, (_, output) => {
    const vector = outputs === 8 ? Math.floor(output / 4) : 0;
    const component = ["x", "y", "z", "w"][output % 4]!;
    const sum = outputs === 8 ? `sum${row}_${vector}` : `sum${row}`;
    return `    output[output_base + ${output}u] = f16(${sum}.${component});`;
  }).join("\n");
  return `
  if (row_active${row}) {
    let output_base =
      (batch * OUTPUT_FRAMES + output_time${row}) * OUTPUT_CHANNELS +
      output_channel;
${stores}
  }`;
}

async function compileKernel(
  device: GPUDevice,
  plan: AceOpt0088VaeConvTranspose1dK4PortablePlan,
): Promise<CompiledKernel> {
  const label = `${plan.kernelId}-${planKey(plan)}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0088VaeConvTranspose1dK4PortableWgsl(
      plan.operationLabel,
      plan,
    ),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0088 portable K4 ConvTranspose1D WGSL failed: ${errors.map(
        ({ lineNum, linePos, message }) => `${lineNum}:${linePos} ${message}`,
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
    throw new Error("OPT-0088 portable K4 ConvTranspose1D requires shader-f16");
  }
  if (
    Number(device.limits.maxComputeInvocationsPerWorkgroup) <
      ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_WORKGROUP_SIZE ||
    Number(device.limits.maxComputeWorkgroupSizeX) <
      ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_WORKGROUP_SIZE
  ) {
    throw new Error(
      "OPT-0088 portable K4 ConvTranspose1D requires 128 compute lanes",
    );
  }
}

function requireDevicePlan(
  device: GPUDevice,
  plan: AceOpt0088VaeConvTranspose1dK4PortablePlan,
  range: AceOpt0036VaeConvTranspose1dRangePlan,
): void {
  if (
    Number(device.limits.maxComputeWorkgroupStorageSize) <
      plan.workgroupStorageBytes
  ) {
    throw new RangeError(
      `OPT-0088 portable K4 ConvTranspose1D requires ${plan.workgroupStorageBytes} workgroup bytes`,
    );
  }
  const maximum = Number(device.limits.maxComputeWorkgroupsPerDimension);
  if (
    !Number.isSafeInteger(maximum) || maximum < 1 ||
    range.workgroupsX > maximum ||
    range.workgroupsY > maximum ||
    range.workgroupsZ > maximum
  ) {
    throw new RangeError("OPT-0088 dispatch exceeds the device dimension");
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
    offset % alignment !== 0 ||
    requiredBytes > Number(device.limits.maxStorageBufferBindingSize)
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
    !Number.isSafeInteger(alignment) || alignment < 4 ||
    offset % alignment !== 0
  ) {
    throw new RangeError(`${label} range control is not dynamically aligned`);
  }
  return offset;
}

function planKey(plan: AceOpt0088VaeConvTranspose1dK4PortablePlan): string {
  return [
    plan.kernelId,
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
