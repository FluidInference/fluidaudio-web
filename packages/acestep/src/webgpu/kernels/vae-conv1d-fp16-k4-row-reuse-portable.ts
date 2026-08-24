import {
  requireAceU32,
  requirePositiveSafeInteger,
} from "./correctness-utils.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dBindings,
  type AceFp16VaeConv1dOutputStorage,
  type AceFp16VaeConv1dPlan,
} from "./vae-conv1d-fp16.js";
import {
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_CHANNELS_PER_SUBGROUP,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_CIN_VECTOR_WIDTH,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_LAYOUT,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_OUTPUTS_PER_LANE,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_REDUCTION_SEMANTICS,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_ROWS,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_SIZE,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_WORKGROUP_SIZE,
  aceOpt0051VaeK7NativeWeightIndex,
  aceOpt0051VaeK7PackedWeightCoordinate,
  aceOpt0051VaeK7PackedWeightIndex,
  packAceOpt0051VaeK7WeightU16,
  planAceOpt0051VaeConv1dK4RowReuse16x64Range,
  unpackAceOpt0051VaeK7WeightU16,
  type AceOpt0051VaeConv1dK4RowReuse16x64RangePlan,
  type AceOpt0051VaeK7WeightCoordinate,
} from "./vae-conv1d-fp16-k4-row-reuse-16x64.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_KERNEL_ID =
  "opt-0088-vae-conv1d-k4-row-reuse-portable-v1" as const;
/** Identical revision-7 packed layout consumed by the OPT-0051 owner. */
export const ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_LAYOUT =
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_LAYOUT;
export const ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_REDUCTION_SEMANTICS =
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_REDUCTION_SEMANTICS;
export const ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_SIZE =
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_WORKGROUP_SIZE;
/** Each former 32-lane hardware group becomes one 32-lane workgroup slice. */
export const ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_SLICE_LANES =
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_SIZE;
export const ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_SLICES =
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_WORKGROUP_SIZE /
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_SIZE;
export const ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_ROWS =
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_ROWS;
export const ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_OUTPUTS_PER_LANE =
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_OUTPUTS_PER_LANE;
export const ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_CHANNELS_PER_SLICE =
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_CHANNELS_PER_SUBGROUP;
export const ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_CIN_VECTOR_WIDTH =
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_CIN_VECTOR_WIDTH;
/** Staged shared-input rows: 4 slices x 16 rows of `vec4<f16>`. */
export const ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_STAGED_VEC4S =
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_SLICES *
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_ROWS;
export const
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_STORAGE_BYTES =
    ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_STAGED_VEC4S * 8;

/**
 * The portable owner consumes the byte-identical hosted revision-7 K7
 * row-reuse weight package, so the index bijection and U16 pack helpers are
 * the OPT-0051 exports re-exported without change.
 */
export const aceOpt0088VaeK7NativeWeightIndex = aceOpt0051VaeK7NativeWeightIndex;
export const aceOpt0088VaeK7PackedWeightIndex = aceOpt0051VaeK7PackedWeightIndex;
export const aceOpt0088VaeK7PackedWeightCoordinate =
  aceOpt0051VaeK7PackedWeightCoordinate;
export const packAceOpt0088VaeK7WeightU16 = packAceOpt0051VaeK7WeightU16;
export const unpackAceOpt0088VaeK7WeightU16 = unpackAceOpt0051VaeK7WeightU16;
export type AceOpt0088VaeK7WeightCoordinate = AceOpt0051VaeK7WeightCoordinate;

/** Identical tile geometry to the OPT-0051 range plan, shared type. */
export type AceOpt0088VaeConv1dK4RowReusePortableRangePlan =
  AceOpt0051VaeConv1dK4RowReuse16x64RangePlan;

const GPU_BUFFER_ALIGNMENT = 4;
const F16_VEC4_ALIGNMENT = 8;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_CONTROL_BYTES = 16;

export interface AceOpt0088VaeConv1dK4RowReusePortableDispatch {
  readonly label: string;
  readonly kernelId:
    typeof ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_KERNEL_ID;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly outputRange: AceOpt0088VaeConv1dK4RowReusePortableRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledAceOpt0088VaeConv1dK4RowReusePortable {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

interface NamedBinding {
  readonly name: "input" | "weight" | "bias" | "output" | "range control";
  readonly binding: GPUBufferBinding;
}

/**
 * Portable OPT-0088 owner for biased FP16-output K7 Conv1D on devices without
 * the WebGPU `subgroups` feature (Safari/Firefox), requiring `shader-f16`
 * only.
 *
 * Numerical contract: bit-identical output to the OPT-0051 subgroup owner on
 * the same device. The row-sharing transport changes from a hardware lane
 * broadcast to workgroup-memory staging fenced by `workgroupBarrier()`; both
 * transports deliver the exact same staged `vec4<f16>` values. Every
 * per-output arithmetic step is unchanged: the same
 * `kernel-ascending-then-input-channel-groups-of-four-ascending` reduction
 * order, the same FP16 dot4 partials widened once into vec2 FP32 running
 * accumulators, the same bias initialisation, and the same single `f16(...)`
 * rounding on store. The consumed weight bytes are the identical hosted
 * revision-7 `[K7, Cin/4, CoutBand64, lane32, output2, CinElement4]` package.
 */
export class AceOpt0088VaeConv1dK4RowReusePortableKernel {
  private readonly pipelines = new Map<
    string,
    Promise<CompiledAceOpt0088VaeConv1dK4RowReusePortable>
  >();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
  ): AceOpt0088VaeConv1dK4RowReusePortableKernel {
    requirePortableDevice(device);
    return new AceOpt0088VaeConv1dK4RowReusePortableKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceFp16VaeConv1dBindings,
    outputStorage: AceFp16VaeConv1dOutputStorage,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0088VaeConv1dK4RowReusePortableDispatch> {
    this.requireLive();
    const plan = planAceFp16VaeConv1d(shape, outputStorage);
    requireCandidateBoundary(plan, bindings.bias !== undefined);
    const outputRange = planAceOpt0088VaeConv1dK4RowReusePortableRange(
      plan,
      range,
    );
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
    const bindGroupKey = `${convKey(plan)}:${bindGroupResources.map(
      (binding) => this.bindingKey(binding),
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-opt-0088-k4-row-reuse-portable-bindings`,
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
      kernelId: ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_KERNEL_ID,
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
  ): Promise<CompiledAceOpt0088VaeConv1dK4RowReusePortable> {
    const key = convKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceOpt0088VaeConv1dK4RowReusePortable(
      this.device,
      plan,
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
    const bias = bindings.bias;
    if (bias === undefined) {
      throw new RangeError(
        "OPT-0088 portable k4-row-reuse Conv1D requires bias",
      );
    }
    const normalized = Object.freeze([
      Object.freeze({
        name: "input" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.input,
          plan.inputStorageBytes,
          plan.inputBindingBytes,
          F16_VEC4_ALIGNMENT,
          `${label} input`,
        ),
      }),
      Object.freeze({
        name: "weight" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.weight,
          plan.weightStorageBytes,
          plan.weightBindingBytes,
          F16_VEC4_ALIGNMENT,
          `${label} weight`,
        ),
      }),
      Object.freeze({
        name: "bias" as const,
        binding: requireStorageBinding(
          this.device,
          bias,
          plan.biasStorageBytes,
          plan.biasBindingBytes,
          GPU_BUFFER_ALIGNMENT,
          `${label} bias`,
        ),
      }),
      Object.freeze({
        name: "output" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.output,
          plan.outputStorageBytes,
          plan.outputBindingBytes,
          GPU_BUFFER_ALIGNMENT,
          `${label} output`,
        ),
      }),
      Object.freeze({
        name: "range control" as const,
        binding: requireRangeControlBinding(
          this.device,
          range.control,
          `${label} range control`,
        ),
      }),
    ] satisfies readonly NamedBinding[]);
    requireDisjointBindings(normalized, label);
    return normalized;
  }

  private requireDeviceLimits(
    plan: AceFp16VaeConv1dPlan,
    range: AceOpt0088VaeConv1dK4RowReusePortableRangePlan,
  ): void {
    const maximumDispatch = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (
      !Number.isSafeInteger(maximumDispatch) || maximumDispatch < 1 ||
      range.workgroupsX > maximumDispatch ||
      range.workgroupsY > maximumDispatch
    ) {
      throw new RangeError(
        "OPT-0088 portable k4-row-reuse Conv1D range exceeds the device dispatch dimension",
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
        "OPT-0088 portable k4-row-reuse Conv1D device reported invalid buffer limits",
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
          `OPT-0088 portable k4-row-reuse Conv1D ${name} exceeds the device storage binding limit`,
        );
      }
      if (bytes > maximumBuffer) {
        throw new RangeError(
          `OPT-0088 portable k4-row-reuse Conv1D ${name} exceeds the device buffer limit`,
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
      throw new Error(
        "OPT-0088 portable k4-row-reuse Conv1D kernel was destroyed",
      );
    }
  }
}

/**
 * Range plan delegated verbatim to the OPT-0051 planner so the tile geometry
 * (16 rows x 64 channels per slice, WG128, C128 dual row band) can never
 * drift between the two transports.
 */
export function planAceOpt0088VaeConv1dK4RowReusePortableRange(
  plan: AceFp16VaeConv1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceOpt0088VaeConv1dK4RowReusePortableRangePlan {
  requireCandidateBoundary(plan, true);
  return planAceOpt0051VaeConv1dK4RowReuse16x64Range(plan, range);
}

/**
 * WGSL with the exact OPT-0051 per-output arithmetic. The only difference is
 * the shared-input transport: staging lanes publish their row vectors into
 * `staged_input` and every lane of the same 32-lane slice consumes them after
 * a barrier, instead of a hardware lane broadcast. Contains no `enable`
 * directive beyond f16 and no hardware-lane builtins.
 */
export function aceOpt0088VaeConv1dK4RowReusePortableWgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceFp16VaeConv1dOutputStorage,
): string {
  const plan = planAceFp16VaeConv1d(shape, outputStorage);
  requireCandidateBoundary(plan, hasBias);
  const geometry = planAceOpt0088VaeConv1dK4RowReusePortableRange(
    plan,
    { base: 0, count: plan.outputElements },
  );
  requireAceU32(plan.inputChannels, "OPT-0088 input channels");
  requireAceU32(plan.outputChannels, "OPT-0088 output channels");
  const initialSums = Array.from(
    { length: ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_ROWS },
    (_, row) => `  var sum${row}: vec2<f32> = initial_sum;`,
  ).join("\n");
  const rowValidity = Array.from(
    { length: ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_ROWS },
    (_, row) => /* wgsl */ `
      let output_time${row} = tile_first_time + ${row}u;
      let padded_time${row} = output_time${row} + kernel * DILATION;
      var input_valid${row} = false;
      if (output_time${row} < range_end_time && padded_time${row} >= PADDING) {
        input_valid${row} = padded_time${row} - PADDING < INPUT_FRAMES;
      }`,
  ).join("");
  const rowDots = Array.from(
    { length: ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_ROWS },
    (_, row) => /* wgsl */ `
      let input_operand${row} = staged_input[stage_base + ${row}u];
      if (input_valid${row}) {
        let partial${row} = vec2<f16>(
          dot(input_operand${row}, weight0),
          dot(input_operand${row}, weight1)
        );
        sum${row} = sum${row} + vec2<f32>(partial${row});
      }`,
  ).join("");
  const stores = Array.from(
    { length: ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_ROWS },
    (_, row) => /* wgsl */ `
    let output_time${row} = tile_first_time + ${row}u;
    if (output_time${row} < range_end_time) {
${outputStoresForRow(row)}
    }`,
  ).join("");
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_KERNEL_ID}
// reduction-semantics: ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_REDUCTION_SEMANTICS}
// input-layout: NLC-vec4-f16
// weight-layout: ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_LAYOUT}
enable f16;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const INPUT_CHANNEL_VEC4S: u32 = ${
    plan.inputChannels /
    ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_CIN_VECTOR_WIDTH
  }u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const OUTPUT_CHANNEL_BANDS: u32 = ${plan.outputChannels / 64}u;
const CHANNEL_BANDS_PER_WORKGROUP: u32 = ${geometry.channelBands}u;
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

// One 16-row staging region per 32-lane slice; 512 bytes total.
var<workgroup> staged_input: array<
  vec4<f16>,
  ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_STAGED_VEC4S}
>;

@compute @workgroup_size(
  ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_SIZE},
  1,
  1,
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
) {
  let lane = local_index %
    ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_SLICE_LANES}u;
  let slice = local_index /
    ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_SLICE_LANES}u;
  let stage_base = slice *
    ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_ROWS}u;
  let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
  let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
  let range_first_time = first_output_row % OUTPUT_FRAMES;
  let batch = first_output_row / OUTPUT_FRAMES;
  let range_end_time = range_first_time + output_row_count;
  let channel_band = slice % CHANNEL_BANDS_PER_WORKGROUP;
  let row_band = slice / CHANNEL_BANDS_PER_WORKGROUP;
  let tile_first_time = range_first_time +
    group.x * ${geometry.tileRows}u +
    row_band * ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_ROWS}u;
  let output_channel_band =
    group.y * CHANNEL_BANDS_PER_WORKGROUP + channel_band;
  let output_channel_base =
    output_channel_band *
      ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_CHANNELS_PER_SLICE}u +
    lane *
      ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_OUTPUTS_PER_LANE}u;

  let initial_sum = vec2<f32>(
    f32(bias[output_channel_base]),
    f32(bias[output_channel_base + 1u])
  );
${initialSums}

  // K then Cin4 stay increasing. Each FP16 dot4 partial is widened once;
  // only the accumulation across Cin4 groups remains FP32. Both loop bounds
  // are shader constants, so every barrier sits in uniform control flow.
  for (var kernel = 0u; kernel < 7u; kernel += 1u) {${rowValidity}
    for (
      var input_channel4 = 0u;
      input_channel4 < INPUT_CHANNEL_VEC4S;
      input_channel4 += 1u
    ) {
      var lane_input = vec4<f16>(0.0h);
      if (lane <
        ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_ROWS}u)
      {
        let lane_output_time = tile_first_time + lane;
        let lane_padded_time = lane_output_time + kernel * DILATION;
        if (
          lane_output_time < range_end_time &&
          lane_padded_time >= PADDING
        ) {
          let lane_input_time = lane_padded_time - PADDING;
          if (lane_input_time < INPUT_FRAMES) {
            lane_input = input[
              (batch * INPUT_FRAMES + lane_input_time) *
                INPUT_CHANNEL_VEC4S + input_channel4
            ];
          }
        }
      }

      // First barrier retires the previous iteration's staged reads before
      // this iteration's staging lanes republish their rows.
      workgroupBarrier();
      if (lane <
        ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_ROWS}u)
      {
        staged_input[stage_base + lane] = lane_input;
      }
      workgroupBarrier();

      // At fixed K/Cin4/output-band, all 64 slice vectors are one
      // contiguous region; each lane's two owned outputs are adjacent.
      let packed_weight_base = ((((kernel * INPUT_CHANNEL_VEC4S +
        input_channel4) * OUTPUT_CHANNEL_BANDS + output_channel_band) *
        32u + lane) * 2u);
      let weight0 = weight[packed_weight_base];
      let weight1 = weight[packed_weight_base + 1u];
      // Consume one staged row at a time so the 32 FP32 running
      // accumulators do not overlap 16 simultaneously-live input vectors.
${rowDots}
    }
  }
${stores}
}
`;
}

async function compileAceOpt0088VaeConv1dK4RowReusePortable(
  device: GPUDevice,
  plan: AceFp16VaeConv1dPlan,
): Promise<CompiledAceOpt0088VaeConv1dK4RowReusePortable> {
  requireCandidateBoundary(plan, true);
  const label = `ace-opt-0088-vae-k4-row-reuse-portable-${convKey(plan)}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0088VaeConv1dK4RowReusePortableWgsl(
      plan,
      true,
      "float16",
    ),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0088 portable k4-row-reuse Conv1D WGSL failed: ${errors.map(
        (message) =>
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
        ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_OUTPUTS_PER_LANE,
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

function requirePortableDevice(device: GPUDevice): void {
  if (!device.features.has("shader-f16")) {
    throw new Error(
      "OPT-0088 portable k4-row-reuse Conv1D requires WebGPU shader-f16",
    );
  }
  const maximumInvocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const maximumSizeX = device.limits.maxComputeWorkgroupSizeX;
  const maximumWorkgroupStorage = device.limits.maxComputeWorkgroupStorageSize;
  if (
    !Number.isSafeInteger(maximumInvocations) ||
    !Number.isSafeInteger(maximumSizeX) ||
    maximumInvocations <
      ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_SIZE ||
    maximumSizeX <
      ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_SIZE
  ) {
    throw new Error(
      `OPT-0088 portable k4-row-reuse Conv1D requires WG${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_SIZE}`,
    );
  }
  if (
    !Number.isSafeInteger(maximumWorkgroupStorage) ||
    maximumWorkgroupStorage <
      ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_STORAGE_BYTES
  ) {
    throw new Error(
      `OPT-0088 portable k4-row-reuse Conv1D requires ${ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_STORAGE_BYTES} workgroup bytes`,
    );
  }
}

function requireCandidateBoundary(
  plan: AceFp16VaeConv1dPlan,
  hasBias: boolean,
): void {
  if (plan.family !== "k7") {
    throw new RangeError(
      "OPT-0088 portable k4-row-reuse Conv1D supports only K7",
    );
  }
  if (!hasBias) {
    throw new RangeError(
      "OPT-0088 portable k4-row-reuse Conv1D requires bias",
    );
  }
  if (plan.outputStorage !== "float16") {
    throw new RangeError(
      "OPT-0088 portable k4-row-reuse Conv1D requires FP16 internal output",
    );
  }
  if (
    plan.inputChannels %
        ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_CIN_VECTOR_WIDTH !==
      0
  ) {
    throw new RangeError(
      "OPT-0088 portable k4-row-reuse Conv1D requires Cin divisible by 4",
    );
  }
  if (
    plan.outputChannels !== 128 && plan.outputChannels % 256 !== 0
  ) {
    throw new RangeError(
      "OPT-0088 portable k4-row-reuse Conv1D requires Cout 128 or divisible by 256",
    );
  }
  requirePositiveSafeInteger(
    plan.inputChannels,
    "OPT-0088 portable k4-row-reuse Conv1D input channels",
  );
  requirePositiveSafeInteger(
    plan.outputChannels,
    "OPT-0088 portable k4-row-reuse Conv1D output channels",
  );
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
      "OPT-0088 portable k4-row-reuse Conv1D device reported an invalid storage alignment",
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
      "OPT-0088 portable k4-row-reuse Conv1D device reported an invalid uniform alignment",
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

function convKey(plan: AceFp16VaeConv1dPlan): string {
  return [
    "opt-0088-portable-k4-row-reuse",
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

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
