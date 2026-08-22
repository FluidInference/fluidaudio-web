import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dBindings,
  type AceFp16VaeConv1dOutputStorage,
  type AceFp16VaeConv1dPlan,
} from "./vae-conv1d-fp16.js";
import {
  planAceFp16VaeConv1dSubgroupRange,
  type AceFp16VaeConv1dFixedSubgroupCapability,
  type AceFp16VaeConv1dSubgroupRangePlan,
} from "./vae-conv1d-fp16-subgroup.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_KERNEL_ID =
  "ace-vae-fp16-opt-0047-output-major-k4-fp16-dot4-k7-conv1d-v1" as const;
export const
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_REDUCTION_SEMANTICS =
    "increasing-k-cin4-fp16-dot4-partials-fp32-accumulator" as const;
export const ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_LAYOUT =
  "k7-cin4-cout-tile128-lane32-output4-cin-element4" as const;
export const ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_SIZE = 32;
export const ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_WORKGROUP_SIZE = 128;
export const
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_SUBGROUPS_PER_WORKGROUP = 4;
export const ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_ROWS = 8;
export const
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_OUTPUTS_PER_LANE = 4;
export const ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_TILE_ROWS = 32;
export const ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_TILE_CHANNELS = 128;
export const ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_CIN_VECTOR_WIDTH = 4;

const GPU_BUFFER_ALIGNMENT = 4;
const F16_VEC4_ALIGNMENT = 8;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_CONTROL_BYTES = 16;

export interface AceOpt0047VaeK7WeightCoordinate {
  readonly outputChannel: number;
  readonly kernel: number;
  readonly inputChannel: number;
}

/** Scalar-U16 index in the package's native `[Cout, K7, Cin]` layout. */
export function aceOpt0047VaeK7NativeWeightIndex(
  inputChannels: number,
  outputChannels: number,
  coordinate: AceOpt0047VaeK7WeightCoordinate,
): number {
  requireWeightLayoutDimensions(inputChannels, outputChannels);
  requireWeightCoordinate(inputChannels, outputChannels, coordinate);
  return nativeWeightIndexUnchecked(inputChannels, coordinate);
}

/**
 * Scalar-U16 index in
 * `[K7, Cin4, CoutTile128, lane32, output4, CinElement4]`.
 */
export function aceOpt0047VaeK7PackedWeightIndex(
  inputChannels: number,
  outputChannels: number,
  coordinate: AceOpt0047VaeK7WeightCoordinate,
): number {
  requireWeightLayoutDimensions(inputChannels, outputChannels);
  requireWeightCoordinate(inputChannels, outputChannels, coordinate);
  return packedWeightIndexUnchecked(
    inputChannels,
    outputChannels,
    coordinate,
  );
}

/** Exact inverse of `aceOpt0047VaeK7PackedWeightIndex`. */
export function aceOpt0047VaeK7PackedWeightCoordinate(
  inputChannels: number,
  outputChannels: number,
  packedScalarIndex: number,
): AceOpt0047VaeK7WeightCoordinate {
  const elementCount = requireWeightLayoutDimensions(
    inputChannels,
    outputChannels,
  );
  if (!Number.isSafeInteger(packedScalarIndex) || packedScalarIndex < 0 ||
    packedScalarIndex >= elementCount) {
    throw new RangeError("OPT-0047 packed weight index is out of bounds");
  }
  const coordinate = packedWeightCoordinateUnchecked(
    inputChannels,
    outputChannels,
    packedScalarIndex,
  );
  requireWeightCoordinate(inputChannels, outputChannels, coordinate);
  return coordinate;
}

/** Bit-preserving native O-K-I U16 to OPT-0047 output-major packing. */
export function packAceOpt0047VaeK7WeightU16(
  nativeWeight: Uint16Array,
  inputChannels: number,
  outputChannels: number,
): Uint16Array {
  const elementCount = requireWeightLayoutDimensions(
    inputChannels,
    outputChannels,
  );
  if (nativeWeight.length !== elementCount) {
    throw new RangeError("OPT-0047 native weight U16 length changed");
  }
  const packed = new Uint16Array(elementCount);
  const cin4s = inputChannels / 4;
  const outputTiles = outputChannels / 128;
  for (let kernel = 0; kernel < 7; kernel += 1) {
    for (let cin4 = 0; cin4 < cin4s; cin4 += 1) {
      for (let tile = 0; tile < outputTiles; tile += 1) {
        for (let lane = 0; lane < 32; lane += 1) {
          for (let output4 = 0; output4 < 4; output4 += 1) {
            const outputChannel = tile * 128 + lane * 4 + output4;
            const nativeBase = (outputChannel * 7 + kernel) * inputChannels +
              cin4 * 4;
            const packedBase = (((((kernel * cin4s + cin4) * outputTiles +
              tile) * 32 + lane) * 4 + output4) * 4);
            packed[packedBase] = nativeWeight[nativeBase]!;
            packed[packedBase + 1] = nativeWeight[nativeBase + 1]!;
            packed[packedBase + 2] = nativeWeight[nativeBase + 2]!;
            packed[packedBase + 3] = nativeWeight[nativeBase + 3]!;
          }
        }
      }
    }
  }
  return packed;
}

/** Bit-preserving exact inverse of `packAceOpt0047VaeK7WeightU16`. */
export function unpackAceOpt0047VaeK7WeightU16(
  packedWeight: Uint16Array,
  inputChannels: number,
  outputChannels: number,
): Uint16Array {
  const elementCount = requireWeightLayoutDimensions(
    inputChannels,
    outputChannels,
  );
  if (packedWeight.length !== elementCount) {
    throw new RangeError("OPT-0047 packed weight U16 length changed");
  }
  const nativeWeight = new Uint16Array(elementCount);
  const cin4s = inputChannels / 4;
  const outputTiles = outputChannels / 128;
  for (let kernel = 0; kernel < 7; kernel += 1) {
    for (let cin4 = 0; cin4 < cin4s; cin4 += 1) {
      for (let tile = 0; tile < outputTiles; tile += 1) {
        for (let lane = 0; lane < 32; lane += 1) {
          for (let output4 = 0; output4 < 4; output4 += 1) {
            const outputChannel = tile * 128 + lane * 4 + output4;
            const nativeBase = (outputChannel * 7 + kernel) * inputChannels +
              cin4 * 4;
            const packedBase = (((((kernel * cin4s + cin4) * outputTiles +
              tile) * 32 + lane) * 4 + output4) * 4);
            nativeWeight[nativeBase] = packedWeight[packedBase]!;
            nativeWeight[nativeBase + 1] = packedWeight[packedBase + 1]!;
            nativeWeight[nativeBase + 2] = packedWeight[packedBase + 2]!;
            nativeWeight[nativeBase + 3] = packedWeight[packedBase + 3]!;
          }
        }
      }
    }
  }
  return nativeWeight;
}

export interface AceOpt0047VaeConv1dOutputMajorK4Dispatch {
  readonly label: string;
  readonly kernelId:
    typeof ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_KERNEL_ID;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly outputRange: AceFp16VaeConv1dSubgroupRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledAceOpt0047VaeConv1dOutputMajorK4 {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

interface NamedBinding {
  readonly name: "input" | "weight" | "bias" | "output" | "range control";
  readonly binding: GPUBufferBinding;
}

/**
 * Isolated OPT-0047 benchmark owner for biased FP16-output K7 Conv1D.
 *
 * This keeps the OPT-0024 fixed32 8-row by 128-channel subgroup ownership,
 * native NLC input, boundary predicates, and reduction order. Only the weight
 * address changes to the bit-preserving output-major layout. There is no
 * fallback or production route in this owner.
 */
export class AceOpt0047VaeConv1dOutputMajorK4Kernel {
  private readonly pipelines = new Map<
    string,
    Promise<CompiledAceOpt0047VaeConv1dOutputMajorK4>
  >();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: AceFp16VaeConv1dFixedSubgroupCapability,
  ): AceOpt0047VaeConv1dOutputMajorK4Kernel {
    requireFixedSubgroupDevice(device, capability);
    return new AceOpt0047VaeConv1dOutputMajorK4Kernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceFp16VaeConv1dBindings,
    outputStorage: AceFp16VaeConv1dOutputStorage,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0047VaeConv1dOutputMajorK4Dispatch> {
    this.requireLive();
    const plan = planAceFp16VaeConv1d(shape, outputStorage);
    requireCandidateBoundary(plan, bindings.bias !== undefined);
    const outputRange = planAceOpt0047VaeConv1dOutputMajorK4Range(
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
        label: `${label}-opt-0047-output-major-k4-bindings`,
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
      kernelId: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_KERNEL_ID,
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
  ): Promise<CompiledAceOpt0047VaeConv1dOutputMajorK4> {
    const key = convKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceOpt0047VaeConv1dOutputMajorK4(
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
      throw new RangeError("OPT-0047 direct-output-major-k4 Conv1D requires bias");
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
    range: AceFp16VaeConv1dSubgroupRangePlan,
  ): void {
    const maximumDispatch = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (
      !Number.isSafeInteger(maximumDispatch) || maximumDispatch < 1 ||
      range.workgroupsX > maximumDispatch ||
      range.workgroupsY > maximumDispatch
    ) {
      throw new RangeError(
        "OPT-0047 direct-output-major-k4 Conv1D range exceeds the device dispatch dimension",
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
        "OPT-0047 direct-output-major-k4 Conv1D device reported invalid buffer limits",
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
          `OPT-0047 direct-output-major-k4 Conv1D ${name} exceeds the device storage binding limit`,
        );
      }
      if (bytes > maximumBuffer) {
        throw new RangeError(
          `OPT-0047 direct-output-major-k4 Conv1D ${name} exceeds the device buffer limit`,
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
      throw new Error("OPT-0047 direct-output-major-k4 Conv1D kernel was destroyed");
    }
  }
}

export function planAceOpt0047VaeConv1dOutputMajorK4Range(
  plan: AceFp16VaeConv1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceFp16VaeConv1dSubgroupRangePlan {
  requireCandidateBoundary(plan, true);
  return planAceFp16VaeConv1dSubgroupRange(plan, range);
}

export function aceOpt0047VaeConv1dOutputMajorK4Wgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceFp16VaeConv1dOutputStorage,
): string {
  const plan = planAceFp16VaeConv1d(shape, outputStorage);
  requireCandidateBoundary(plan, hasBias);
  const initialSums = Array.from(
    { length: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_ROWS },
    (_, row) => `  var sum${row}: vec4<f32> = initial_sum;`,
  ).join("\n");
  const rowValidity = Array.from(
    { length: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_ROWS },
    (_, row) => /* wgsl */ `
      let output_time${row} = tile_first_time + ${row}u;
      let padded_time${row} = output_time${row} + kernel * DILATION;
      var input_valid${row} = false;
      if (output_time${row} < range_end_time && padded_time${row} >= PADDING) {
        input_valid${row} = padded_time${row} - PADDING < INPUT_FRAMES;
      }`,
  ).join("");
  const broadcasts = Array.from(
    { length: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_ROWS },
    (_, row) =>
      `      let input_operand${row} = subgroupBroadcast(lane_input, ${row}u);`,
  ).join("\n");
  const partialAdds = Array.from(
    { length: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_ROWS },
    (_, row) => /* wgsl */ `
      if (input_valid${row}) {
        let partial${row} = vec4<f16>(
          dot(input_operand${row}, weight0),
          dot(input_operand${row}, weight1),
          dot(input_operand${row}, weight2),
          dot(input_operand${row}, weight3)
        );
        sum${row} = sum${row} + vec4<f32>(partial${row});
      }`,
  ).join("");
  const stores = Array.from(
    { length: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_ROWS },
    (_, row) => /* wgsl */ `
    let output_time${row} = tile_first_time + ${row}u;
    if (output_time${row} < range_end_time) {
${outputStoresForRow(row)}
    }`,
  ).join("");
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_KERNEL_ID}
// reduction-semantics: ${ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_REDUCTION_SEMANTICS}
// input-layout: NLC-vec4-f16
// weight-layout: ${ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_LAYOUT}
enable f16;
enable subgroups;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const INPUT_CHANNEL_VEC4S: u32 = ${
    plan.inputChannels /
    ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_CIN_VECTOR_WIDTH
  }u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const OUTPUT_CHANNEL_TILES: u32 = ${
    plan.outputChannels /
    ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_TILE_CHANNELS
  }u;
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
  ${ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_WORKGROUP_SIZE},
  1,
  1,
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (subgroup_size == ${ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_SIZE}u) {
    let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
    let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
    let range_first_time = first_output_row % OUTPUT_FRAMES;
    let batch = first_output_row / OUTPUT_FRAMES;
    let range_end_time = range_first_time + output_row_count;
    let tile_first_time = range_first_time +
      group.x * ${ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_TILE_ROWS}u +
      subgroup * ${ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_ROWS}u;
    let output_channel_base =
      group.y * ${ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_TILE_CHANNELS}u +
      subgroup_lane *
        ${ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_OUTPUTS_PER_LANE}u;

    let initial_sum = vec4<f32>(
      f32(bias[output_channel_base]),
      f32(bias[output_channel_base + 1u]),
      f32(bias[output_channel_base + 2u]),
      f32(bias[output_channel_base + 3u])
    );
${initialSums}

    // K then Cin4 stay increasing. Each FP16 dot4 partial is widened once;
    // only the accumulation across Cin4 groups remains FP32.
    for (var kernel = 0u; kernel < 7u; kernel += 1u) {${rowValidity}
      for (
        var input_channel4 = 0u;
        input_channel4 < INPUT_CHANNEL_VEC4S;
        input_channel4 += 1u
      ) {
        var lane_input = vec4<f16>(0.0h);
        if (subgroup_lane <
          ${ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_ROWS}u)
        {
          let lane_output_time = tile_first_time + subgroup_lane;
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
${broadcasts}

        // At fixed K/Cin4/output-tile, all 128 subgroup vectors are one
        // contiguous region; each lane's four owned outputs are adjacent.
        let packed_weight_base = ((((kernel * INPUT_CHANNEL_VEC4S +
          input_channel4) * OUTPUT_CHANNEL_TILES + group.y) * 32u +
          subgroup_lane) * 4u);
        let weight0 = weight[packed_weight_base];
        let weight1 = weight[packed_weight_base + 1u];
        let weight2 = weight[packed_weight_base + 2u];
        let weight3 = weight[packed_weight_base + 3u];${partialAdds}
      }
    }
${stores}
  }
}
`;
}

async function compileAceOpt0047VaeConv1dOutputMajorK4(
  device: GPUDevice,
  plan: AceFp16VaeConv1dPlan,
): Promise<CompiledAceOpt0047VaeConv1dOutputMajorK4> {
  requireCandidateBoundary(plan, true);
  const label = `ace-opt-0047-vae-output-major-k4-${convKey(plan)}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0047VaeConv1dOutputMajorK4Wgsl(
      plan,
      true,
      "float16",
    ),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0047 direct-output-major-k4 Conv1D WGSL failed: ${errors.map((message) =>
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
        ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_OUTPUTS_PER_LANE,
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

function requireFixedSubgroupDevice(
  device: GPUDevice,
  capability: AceFp16VaeConv1dFixedSubgroupCapability,
): void {
  if (!device.features.has("shader-f16")) {
    throw new Error("OPT-0047 direct-output-major-k4 Conv1D requires WebGPU shader-f16");
  }
  if (
    !device.features.has("subgroups") ||
    capability.subgroupMinSize !==
      ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_SIZE ||
    capability.subgroupMaxSize !==
      ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_SIZE
  ) {
    throw new Error(
      "OPT-0047 direct-output-major-k4 Conv1D requires reported fixed 32-lane subgroups",
    );
  }
  const maximumInvocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const maximumSizeX = device.limits.maxComputeWorkgroupSizeX;
  if (
    !Number.isSafeInteger(maximumInvocations) ||
    !Number.isSafeInteger(maximumSizeX) ||
    maximumInvocations <
      ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_WORKGROUP_SIZE ||
    maximumSizeX <
      ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_WORKGROUP_SIZE
  ) {
    throw new Error(
      `OPT-0047 direct-output-major-k4 Conv1D requires WG${ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_WORKGROUP_SIZE}`,
    );
  }
}

function requireCandidateBoundary(
  plan: AceFp16VaeConv1dPlan,
  hasBias: boolean,
): void {
  if (plan.family !== "k7") {
    throw new RangeError("OPT-0047 direct-output-major-k4 Conv1D supports only K7");
  }
  if (!hasBias) {
    throw new RangeError("OPT-0047 direct-output-major-k4 Conv1D requires bias");
  }
  if (plan.outputStorage !== "float16") {
    throw new RangeError(
      "OPT-0047 direct-output-major-k4 Conv1D requires FP16 internal output",
    );
  }
  if (
    plan.inputChannels %
        ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_CIN_VECTOR_WIDTH !==
      0
  ) {
    throw new RangeError(
      "OPT-0047 direct-output-major-k4 Conv1D requires Cin divisible by 4",
    );
  }
  if (
    plan.outputChannels %
        ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_TILE_CHANNELS !==
      0
  ) {
    throw new RangeError(
      "OPT-0047 direct-output-major-k4 Conv1D requires Cout divisible by 128",
    );
  }
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
      "OPT-0047 direct-output-major-k4 Conv1D device reported an invalid storage alignment",
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
      "OPT-0047 direct-output-major-k4 Conv1D device reported an invalid uniform alignment",
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
    "opt-0047-direct-output-major-k4-fixed32-subgroup-output-major-k4",
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

function nativeWeightIndexUnchecked(
  inputChannels: number,
  coordinate: AceOpt0047VaeK7WeightCoordinate,
): number {
  return (coordinate.outputChannel * 7 + coordinate.kernel) * inputChannels +
    coordinate.inputChannel;
}

function packedWeightIndexUnchecked(
  inputChannels: number,
  outputChannels: number,
  coordinate: AceOpt0047VaeK7WeightCoordinate,
): number {
  const cin4 = Math.floor(coordinate.inputChannel / 4);
  const cinElement4 = coordinate.inputChannel % 4;
  const outputTile = Math.floor(coordinate.outputChannel / 128);
  const outputWithinTile = coordinate.outputChannel % 128;
  const lane = Math.floor(outputWithinTile / 4);
  const output4 = outputWithinTile % 4;
  return (((((coordinate.kernel * (inputChannels / 4) + cin4) *
    (outputChannels / 128) + outputTile) * 32 + lane) * 4 + output4) * 4 +
    cinElement4);
}

function packedWeightCoordinateUnchecked(
  inputChannels: number,
  outputChannels: number,
  packedScalarIndex: number,
): AceOpt0047VaeK7WeightCoordinate {
  let remaining = packedScalarIndex;
  const cinElement4 = remaining % 4;
  remaining = Math.floor(remaining / 4);
  const output4 = remaining % 4;
  remaining = Math.floor(remaining / 4);
  const lane = remaining % 32;
  remaining = Math.floor(remaining / 32);
  const outputTile = remaining % (outputChannels / 128);
  remaining = Math.floor(remaining / (outputChannels / 128));
  const cin4 = remaining % (inputChannels / 4);
  remaining = Math.floor(remaining / (inputChannels / 4));
  return Object.freeze({
    outputChannel: outputTile * 128 + lane * 4 + output4,
    kernel: remaining,
    inputChannel: cin4 * 4 + cinElement4,
  });
}

function requireWeightLayoutDimensions(
  inputChannels: number,
  outputChannels: number,
): number {
  if (!Number.isSafeInteger(inputChannels) || inputChannels < 1 ||
    inputChannels % 4 !== 0) {
    throw new RangeError("OPT-0047 weight layout requires Cin divisible by 4");
  }
  if (!Number.isSafeInteger(outputChannels) || outputChannels < 1 ||
    outputChannels % 128 !== 0) {
    throw new RangeError(
      "OPT-0047 weight layout requires Cout divisible by 128",
    );
  }
  const elementCount = inputChannels * outputChannels * 7;
  if (!Number.isSafeInteger(elementCount)) {
    throw new RangeError("OPT-0047 weight layout exceeds safe indexing");
  }
  return elementCount;
}

function requireWeightCoordinate(
  inputChannels: number,
  outputChannels: number,
  coordinate: AceOpt0047VaeK7WeightCoordinate,
): void {
  for (const [name, value, limit] of [
    ["output channel", coordinate.outputChannel, outputChannels],
    ["kernel", coordinate.kernel, 7],
    ["input channel", coordinate.inputChannel, inputChannels],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 0 || value >= limit) {
      throw new RangeError(`OPT-0047 weight ${name} is out of bounds`);
    }
  }
}
