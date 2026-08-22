import {
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dCongruentRange,
} from "./vae-conv-transpose1d-fp16.js";
import type { AceVaeConvTranspose1dShape } from "./vae-primitives.js";

export const ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_KERNEL_ID =
  "ace-vae-fp16-fixed32-subgroup-32row-256channel-conv-transpose1d-v1" as const;
export const ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE = 32;
export const ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE = 128;
export const ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP = 8;
export const ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP = 32;
export const ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE = 8;
export const ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP = 256;

export interface AceOpt0029VaeConvTranspose1dPlan
  extends AceVaeConvTranspose1dShape {
  readonly outputFrames: number;
  readonly inputElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly inputBindingBytes: number;
  readonly weightBindingBytes: number;
  readonly biasBindingBytes: number;
  readonly outputBindingBytes: number;
  readonly kernelId: typeof ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_KERNEL_ID;
  readonly rowsPerSubgroup: 8;
  readonly rowsPerWorkgroup: 32;
  readonly channelsPerLane: 8;
  readonly channelsPerWorkgroup: 256;
  readonly workgroupSize: 128;
}

export interface AceOpt0029VaeConvTranspose1dRangePlan {
  readonly base: number;
  readonly count: number;
  readonly firstOutputTime: number;
  readonly outputRowCount: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
  readonly workgroupsZ: number;
}

export function planAceOpt0029VaeConvTranspose1d(
  shape: AceVaeConvTranspose1dShape,
): AceOpt0029VaeConvTranspose1dPlan {
  const base = planAceFp16VaeConvTranspose1d(shape);
  if (
    base.outputChannels % 128 !== 0 ||
    base.kernelSize !== base.stride * 2 ||
    base.dilation !== 1 ||
    base.outputPadding !== 0
  ) {
    throw new RangeError("OPT-0029 requires the production polyphase geometry");
  }
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
    outputFrames: base.outputFrames,
    inputElements: base.inputElements,
    weightElements: base.weightElements,
    outputElements: base.outputElements,
    inputBindingBytes: base.inputBindingBytes,
    weightBindingBytes: base.weightBindingBytes,
    biasBindingBytes: base.biasBindingBytes,
    outputBindingBytes: base.outputBindingBytes,
    kernelId: ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_KERNEL_ID,
    rowsPerSubgroup: ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP,
    rowsPerWorkgroup: ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP,
    channelsPerLane: ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE,
    channelsPerWorkgroup:
      ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP,
    workgroupSize: ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  });
}

export function planAceOpt0029VaeConvTranspose1dRange(
  plan: AceOpt0029VaeConvTranspose1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceOpt0029VaeConvTranspose1dRangePlan {
  const base = planAceFp16VaeConvTranspose1dCongruentRange(
    planAceFp16VaeConvTranspose1d(plan),
    range,
  );
  return Object.freeze({
    base: base.base,
    count: base.count,
    firstOutputTime: base.firstOutputTime,
    outputRowCount: base.outputRowCount,
    workgroupsX: Math.ceil(
      base.outputRowCount /
        (ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP * plan.stride),
    ),
    workgroupsY: Math.ceil(
      plan.outputChannels /
        ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP,
    ),
    workgroupsZ: plan.stride,
  });
}

export function aceOpt0029VaeConvTranspose1dWgsl(
  shape: AceVaeConvTranspose1dShape,
): string {
  const plan = planAceOpt0029VaeConvTranspose1d(shape);
  const declarations = Array.from(
    { length: ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => `
  let phase_row${row} = subgroup_first_phase_row + ${row}u;
  let range_offset${row} = phase + phase_row${row} * STRIDE;
  let output_time${row} = range_first_time + range_offset${row};
  let row_active${row} = range_offset${row} < output_row_count;
  let first_input_time${row} = phase_first_input_time + phase_row${row};
  var input_time${row} = 0u;
  var input_valid${row} = false;
  var sum${row}_0 = bias_value.low;
  var sum${row}_1 = bias_value.high;`,
  ).join("");
  const tapInputs = Array.from(
    { length: ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => `
    if (tap == 0u) {
      input_time${row} = first_input_time${row};
      input_valid${row} = input_time${row} < INPUT_FRAMES;
    } else if (first_input_time${row} > 0u) {
      input_time${row} = first_input_time${row} - 1u;
      input_valid${row} = input_time${row} < INPUT_FRAMES;
    } else {
      input_valid${row} = false;
    }`,
  ).join("");
  const laneLoads = Array.from(
    { length: ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => `
      if (lane == ${row}u && row_active${row} && input_valid${row}) {
        lane_input = input[
          (batch * INPUT_FRAMES + input_time${row}) * INPUT_CHANNELS + inner
        ];
      }`,
  ).join("");
  const contractions = Array.from(
    { length: ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => `
      if (row_active${row} && input_valid${row}) {
        let a${row} = f32(subgroupBroadcast(lane_input, ${row}u));
        sum${row}_0 = sum${row}_0 + vec4<f32>(a${row}) * weight_value.low;
        sum${row}_1 = sum${row}_1 + vec4<f32>(a${row}) * weight_value.high;
      }`,
  ).join("");
  const stores = Array.from(
    { length: ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP },
    (_, row) => `
  if (row_active${row}) {
    let output_base =
      (batch * OUTPUT_FRAMES + output_time${row}) * OUTPUT_CHANNELS +
      output_channel;
${storeVector("sum" + row + "_0", "output_base", 0)}
${storeVector("sum" + row + "_1", "output_base", 4)}
  }`,
  ).join("");

  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_KERNEL_ID}
// weight-layout: conv-transpose1d-phase-tap-input-output-f16-v1
enable f16;
enable subgroups;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const STRIDE: u32 = ${plan.stride}u;
const PADDING: u32 = ${plan.padding}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> polyphase_weight: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

struct OutputRangeParameters {
  first_output: u32,
  output_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(4) var<uniform> output_range: OutputRangeParameters;

fn unpack_f16x4(low: u32, high: u32) -> vec4<f32> {
  let low_pair = unpack2x16float(low);
  let high_pair = unpack2x16float(high);
  return vec4<f32>(low_pair.x, low_pair.y, high_pair.x, high_pair.y);
}

struct F32x8 {
  low: vec4<f32>,
  high: vec4<f32>,
}

fn load_f16x8(packed: vec4<u32>) -> F32x8 {
  return F32x8(
    unpack_f16x4(packed.x, packed.y),
    unpack_f16x4(packed.z, packed.w),
  );
}

@compute @workgroup_size(${ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (subgroup_size != 32u || subgroup >= 4u) { return; }
  let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
  let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
  let range_first_time = first_output_row % OUTPUT_FRAMES;
  let batch = first_output_row / OUTPUT_FRAMES;
  let phase = group.z;
  let phase_first_padded_time = range_first_time + phase + PADDING;
  let congruent_kernel = phase_first_padded_time % STRIDE;
  let phase_first_input_time = phase_first_padded_time / STRIDE;
  let subgroup_first_phase_row = group.x * 32u + subgroup * 8u;
  let output_channel = group.y * 256u + lane * 8u;
  var bias_value = F32x8(vec4<f32>(0.0), vec4<f32>(0.0));
  if (output_channel < OUTPUT_CHANNELS) {
    bias_value = load_f16x8(bias[output_channel / 8u]);
  }
${declarations}

  // Exact source order: tap 0 Cin 0..K-1, then tap 1 Cin 0..K-1.
  for (var tap = 0u; tap < 2u; tap += 1u) {
${tapInputs}
    for (var inner = 0u; inner < INPUT_CHANNELS; inner += 1u) {
      var lane_input = 0.0h;
${laneLoads}
      var weight_value = F32x8(vec4<f32>(0.0), vec4<f32>(0.0));
      if (output_channel < OUTPUT_CHANNELS) {
        let weight_word =
          (((congruent_kernel * 2u + tap) * INPUT_CHANNELS + inner) *
            OUTPUT_CHANNELS + output_channel) / 8u;
        weight_value = load_f16x8(polyphase_weight[weight_word]);
      }
${contractions}
    }
  }
${stores}
}
`;
}

function storeVector(value: string, base: string, offset: number): string {
  return ["x", "y", "z", "w"].map((component, index) => {
    const channelOffset = offset + index;
    return `    if (output_channel + ${channelOffset}u < OUTPUT_CHANNELS) {
      output[${base} + ${channelOffset}u] = f16(${value}.${component});
    }`;
  }).join("\n");
}
