import {
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
} from "../../model/manifest.js";
import {
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dCongruentRange,
} from "./vae-conv-transpose1d-fp16.js";
import type { AceVaeConvTranspose1dShape } from "./vae-primitives.js";

export const ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID =
  "ace-vae-fp16-fixed32-subgroup-32row-128channel-conv-transpose1d-v1" as const;
export const ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID =
  "ace-vae-fp16-fixed32-subgroup-16row-256channel-conv-transpose1d-v1" as const;
export const ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_LAYOUT_ID =
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT;
export const ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE = 32;
export const ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP = 4;
export const ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE = 128;

export type AceOpt0036VaeConvTranspose1dKernelId =
  | typeof ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID
  | typeof ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID;

export interface AceOpt0036VaeConvTranspose1dPlan
  extends AceVaeConvTranspose1dShape {
  readonly kernelId: AceOpt0036VaeConvTranspose1dKernelId;
  readonly weightLayout: typeof ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_LAYOUT_ID;
  readonly outputFrames: number;
  readonly inputElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly inputBindingBytes: number;
  readonly weightBindingBytes: number;
  readonly biasBindingBytes: number;
  readonly outputBindingBytes: number;
  readonly rowsPerSubgroup: 4 | 8;
  readonly rowsPerWorkgroup: 16 | 32;
  readonly channelsPerLane: 4 | 8;
  readonly channelsPerWorkgroup: 128 | 256;
  readonly workgroupSize: 128;
  readonly accumulatorCountPerLane: 32;
  readonly workgroupStorageBytes: 0;
  readonly workgroupBarrierCount: 0;
}

export interface AceOpt0036VaeConvTranspose1dRangePlan {
  readonly base: number;
  readonly count: number;
  readonly batch: number;
  readonly firstOutputTime: number;
  readonly outputRowCount: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
  readonly workgroupsZ: number;
}

interface Variant {
  readonly kernelId: AceOpt0036VaeConvTranspose1dKernelId;
  readonly rowsPerSubgroup: 4 | 8;
  readonly rowsPerWorkgroup: 16 | 32;
  readonly channelsPerLane: 4 | 8;
  readonly channelsPerWorkgroup: 128 | 256;
}

const R8C4 = Object.freeze({
  kernelId: ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
  rowsPerSubgroup: 8,
  rowsPerWorkgroup: 32,
  channelsPerLane: 4,
  channelsPerWorkgroup: 128,
} as const satisfies Variant);

const R4C8 = Object.freeze({
  kernelId: ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
  rowsPerSubgroup: 4,
  rowsPerWorkgroup: 16,
  channelsPerLane: 8,
  channelsPerWorkgroup: 256,
} as const satisfies Variant);

export function planAceOpt0036VaeConvTranspose1dR8C4(
  shape: AceVaeConvTranspose1dShape,
): AceOpt0036VaeConvTranspose1dPlan {
  return planVariant(shape, R8C4);
}

export function planAceOpt0036VaeConvTranspose1dR4C8(
  shape: AceVaeConvTranspose1dShape,
): AceOpt0036VaeConvTranspose1dPlan {
  return planVariant(shape, R4C8);
}

export function planAceOpt0036VaeConvTranspose1dRange(
  plan: AceOpt0036VaeConvTranspose1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceOpt0036VaeConvTranspose1dRangePlan {
  const base = planAceFp16VaeConvTranspose1dCongruentRange(
    planAceFp16VaeConvTranspose1d(plan),
    range,
  );
  return Object.freeze({
    base: base.base,
    count: base.count,
    batch: base.batch,
    firstOutputTime: base.firstOutputTime,
    outputRowCount: base.outputRowCount,
    workgroupsX: Math.ceil(
      base.outputRowCount / (plan.rowsPerWorkgroup * plan.stride),
    ),
    workgroupsY: Math.ceil(
      plan.outputChannels / plan.channelsPerWorkgroup,
    ),
    workgroupsZ: plan.stride,
  });
}

export function aceOpt0036VaeConvTranspose1dR8C4Wgsl(
  shape: AceVaeConvTranspose1dShape,
): string {
  return variantWgsl(shape, R8C4);
}

export function aceOpt0036VaeConvTranspose1dR4C8Wgsl(
  shape: AceVaeConvTranspose1dShape,
): string {
  return variantWgsl(shape, R4C8);
}

function planVariant(
  shape: AceVaeConvTranspose1dShape,
  variant: Variant,
): AceOpt0036VaeConvTranspose1dPlan {
  const base = planAceFp16VaeConvTranspose1d(shape);
  if (
    base.outputChannels % 128 !== 0 ||
    base.kernelSize !== base.stride * 2 ||
    base.dilation !== 1 ||
    base.outputPadding !== 0
  ) {
    throw new RangeError(
      "OPT-0036 requires the exact production polyphase geometry",
    );
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
    kernelId: variant.kernelId,
    weightLayout: ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_LAYOUT_ID,
    outputFrames: base.outputFrames,
    inputElements: base.inputElements,
    weightElements: base.weightElements,
    outputElements: base.outputElements,
    inputBindingBytes: base.inputBindingBytes,
    weightBindingBytes: base.weightBindingBytes,
    biasBindingBytes: base.biasBindingBytes,
    outputBindingBytes: base.outputBindingBytes,
    rowsPerSubgroup: variant.rowsPerSubgroup,
    rowsPerWorkgroup: variant.rowsPerWorkgroup,
    channelsPerLane: variant.channelsPerLane,
    channelsPerWorkgroup: variant.channelsPerWorkgroup,
    workgroupSize: ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE,
    accumulatorCountPerLane: 32,
    workgroupStorageBytes: 0,
    workgroupBarrierCount: 0,
  });
}

function variantWgsl(
  shape: AceVaeConvTranspose1dShape,
  variant: Variant,
): string {
  const plan = planVariant(shape, variant);
  const rows = variant.rowsPerSubgroup;
  const declarations = Array.from({ length: rows }, (_, row) => `
  let phase_row${row} = subgroup_first_phase_row + ${row}u;
  let range_offset${row} = phase + phase_row${row} * STRIDE;
  let output_time${row} = range_first_time + range_offset${row};
  let row_active${row} = range_offset${row} < output_row_count;
  let first_input_time${row} = phase_first_input_time + phase_row${row};
  var input_time${row} = 0u;
  var input_valid${row} = false;
${sumDeclarations(row, variant.channelsPerLane)}`)
    .join("");
  const tapInputs = Array.from({ length: rows }, (_, row) => `
    if (tap == 0u) {
      input_time${row} = first_input_time${row};
      input_valid${row} = input_time${row} < INPUT_FRAMES;
    } else if (first_input_time${row} > 0u) {
      input_time${row} = first_input_time${row} - 1u;
      input_valid${row} = input_time${row} < INPUT_FRAMES;
    } else {
      input_valid${row} = false;
    }`).join("");
  const laneLoads = Array.from({ length: rows }, (_, row) => `
      if (lane == ${row}u && row_active${row} && input_valid${row}) {
        lane_input = input[
          (batch * INPUT_FRAMES + input_time${row}) * INPUT_CHANNELS + inner
        ];
      }`).join("");
  const contractions = Array.from({ length: rows }, (_, row) =>
    contractionsWgsl(row, variant.channelsPerLane)
  ).join("");
  const stores = Array.from({ length: rows }, (_, row) =>
    storesWgsl(row, variant.channelsPerLane)
  ).join("");

  return /* wgsl */ `
// kernel-id: ${variant.kernelId}
// weight-layout: ${ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_LAYOUT_ID}
enable f16;
enable subgroups;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const STRIDE: u32 = ${plan.stride}u;
const PADDING: u32 = ${plan.padding}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> polyphase_weight:
  ${packedArrayType(variant.channelsPerLane)};
@group(0) @binding(2) var<storage, read> bias:
  ${packedArrayType(variant.channelsPerLane)};
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

struct OutputRangeParameters {
  first_output: u32,
  output_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(4) var<uniform> output_range: OutputRangeParameters;

fn unpack_f16x4(packed: vec2<u32>) -> vec4<f32> {
  let low = unpack2x16float(packed.x);
  let high = unpack2x16float(packed.y);
  return vec4<f32>(low.x, low.y, high.x, high.y);
}
${variant.channelsPerLane === 8 ? `
struct F32x8 {
  low: vec4<f32>,
  high: vec4<f32>,
}

fn load_f16x8(packed: vec4<u32>) -> F32x8 {
  return F32x8(
    unpack_f16x4(packed.xy),
    unpack_f16x4(packed.zw),
  );
}
` : ""}
@compute @workgroup_size(
  ${ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE}, 1, 1,
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
) {
  if (
    subgroup_size != ${ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE}u ||
    subgroup >= ${ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP}u
  ) { return; }
  let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
  let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
  let range_first_time = first_output_row % OUTPUT_FRAMES;
  let batch = first_output_row / OUTPUT_FRAMES;
  let phase = group.z;
  let phase_first_padded_time = range_first_time + phase + PADDING;
  let congruent_kernel = phase_first_padded_time % STRIDE;
  let phase_first_input_time = phase_first_padded_time / STRIDE;
  let subgroup_first_phase_row =
    group.x * ${variant.rowsPerWorkgroup}u +
    subgroup * ${variant.rowsPerSubgroup}u;
  let output_channel =
    group.y * ${variant.channelsPerWorkgroup}u +
    lane * ${variant.channelsPerLane}u;
${biasLoadWgsl(variant.channelsPerLane)}
${declarations}

  // Exact source order: tap 0 Cin 0..K-1, then tap 1 Cin 0..K-1.
  for (var tap = 0u; tap < 2u; tap += 1u) {
${tapInputs}
    for (var inner = 0u; inner < INPUT_CHANNELS; inner += 1u) {
      var lane_input: f16 = 0.0h;
${laneLoads}
${weightLoadWgsl(variant.channelsPerLane)}
${contractions}
    }
  }
${stores}
}
`;
}

function packedArrayType(channelsPerLane: 4 | 8): string {
  return channelsPerLane === 4 ? "array<vec2<u32>>" : "array<vec4<u32>>";
}

function sumDeclarations(row: number, channelsPerLane: 4 | 8): string {
  return channelsPerLane === 4
    ? `  var sum${row} = bias_value;`
    : `  var sum${row}_0 = bias_value.low;
  var sum${row}_1 = bias_value.high;`;
}

function biasLoadWgsl(channelsPerLane: 4 | 8): string {
  if (channelsPerLane === 4) {
    return `  var bias_value = vec4<f32>(0.0);
  if (output_channel < OUTPUT_CHANNELS) {
    bias_value = unpack_f16x4(bias[output_channel / 4u]);
  }`;
  }
  return `  var bias_value = F32x8(vec4<f32>(0.0), vec4<f32>(0.0));
  if (output_channel < OUTPUT_CHANNELS) {
    bias_value = load_f16x8(bias[output_channel / 8u]);
  }`;
}

function weightLoadWgsl(channelsPerLane: 4 | 8): string {
  const zero = channelsPerLane === 4
    ? "vec4<f32>(0.0)"
    : "F32x8(vec4<f32>(0.0), vec4<f32>(0.0))";
  const load = channelsPerLane === 4
    ? "unpack_f16x4(polyphase_weight[weight_word])"
    : "load_f16x8(polyphase_weight[weight_word])";
  return `      var weight_value = ${zero};
      if (output_channel < OUTPUT_CHANNELS) {
        let weight_word =
          (((congruent_kernel * 2u + tap) * INPUT_CHANNELS + inner) *
            OUTPUT_CHANNELS + output_channel) / ${channelsPerLane}u;
        weight_value = ${load};
      }`;
}

function contractionsWgsl(row: number, channelsPerLane: 4 | 8): string {
  if (channelsPerLane === 4) {
    return `
      if (row_active${row} && input_valid${row}) {
        let a${row} = f32(subgroupBroadcast(lane_input, ${row}u));
        sum${row} = sum${row} + vec4<f32>(a${row}) * weight_value;
      }`;
  }
  return `
      if (row_active${row} && input_valid${row}) {
        let a${row} = f32(subgroupBroadcast(lane_input, ${row}u));
        sum${row}_0 = sum${row}_0 + vec4<f32>(a${row}) * weight_value.low;
        sum${row}_1 = sum${row}_1 + vec4<f32>(a${row}) * weight_value.high;
      }`;
}

function storesWgsl(row: number, channelsPerLane: 4 | 8): string {
  const vectors = channelsPerLane === 4
    ? storeVector(`sum${row}`, "output_base", 0)
    : [
        storeVector(`sum${row}_0`, "output_base", 0),
        storeVector(`sum${row}_1`, "output_base", 4),
      ].join("\n");
  return `
  if (row_active${row}) {
    let output_base =
      (batch * OUTPUT_FRAMES + output_time${row}) * OUTPUT_CHANNELS +
      output_channel;
${vectors}
  }`;
}

function storeVector(value: string, base: string, offset: number): string {
  return ["x", "y", "z", "w"].map((component, index) => {
    const channelOffset = offset + index;
    return `    if (output_channel + ${channelOffset}u < OUTPUT_CHANNELS) {
      output[${base} + ${channelOffset}u] = f16(${value}.${component});
    }`;
  }).join("\n");
}
