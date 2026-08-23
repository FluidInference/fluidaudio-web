import {
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  type AceOpt0036VaeConvTranspose1dPlan,
  type AceOpt0036VaeConvTranspose1dRangePlan,
} from "./vae-conv-transpose1d-fp16-reuse-axis-subgroup.js";
import {
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dCongruentRange,
} from "./vae-conv-transpose1d-fp16.js";
import {
  selectAceOpt0040VaeConvTranspose1d,
  type AceOpt0040VaeConvTranspose1dOperationLabel,
} from "./vae-conv-transpose1d-fp16-shape-selector.js";
import type { AceVaeConvTranspose1dShape } from "./vae-primitives.js";

/** Benchmark-only payload; it is deliberately not a revision-6 package layout. */
export const ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT =
  "ace-opt-0048-phase-tap-cin4-cout-tile-lane-output-k4-f16-v1" as const;
export const ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID =
  "ace-opt-0048-vae-convtranspose-r4c8-fp16-k4-partials-v1" as const;
export const ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID =
  "ace-opt-0048-vae-convtranspose-r8c4-fp16-k4-partials-v1" as const;
export const ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WIDTH = 4;

export type AceOpt0048VaeConvTranspose1dK4KernelId =
  | typeof ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID
  | typeof ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID;

export interface AceOpt0048VaeConvTranspose1dK4WeightPlan {
  readonly weightLayout:
    typeof ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT;
  readonly stride: number;
  readonly inputChannels: number;
  readonly outputChannels: number;
  readonly reuseAxis: "channel" | "row";
  readonly tapCountPerPhase: 2;
  readonly inputChannelK4Groups: number;
  readonly outputChannelsPerLane: 4 | 8;
  readonly outputChannelsPerTile: 128 | 256;
  readonly outputChannelTiles: number;
  readonly logicalWeightElements: number;
  /** [phase, tap, Cin4, Cout tile, lane32, output/lane, K4]. */
  readonly packedWeightStorageShape: readonly [
    number,
    2,
    number,
    number,
    32,
    4 | 8,
    4,
  ];
}

export interface AceOpt0048VaeConvTranspose1dK4Plan
  extends Omit<AceOpt0036VaeConvTranspose1dPlan, "kernelId" | "weightLayout"> {
  readonly operationLabel: AceOpt0040VaeConvTranspose1dOperationLabel;
  readonly reuseAxis: "channel" | "row";
  readonly kernelId: AceOpt0048VaeConvTranspose1dK4KernelId;
  readonly weightLayout:
    typeof ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT;
  readonly reductionSemantics:
    "increasing-tap-cin4-fp16-dot4-partials-fp32-running-state";
  readonly inputChannelK4Groups: number;
  readonly outputChannelTiles: number;
  readonly packedWeightStorageShape:
    AceOpt0048VaeConvTranspose1dK4WeightPlan["packedWeightStorageShape"];
}

/**
 * Close the benchmark over OPT-0040's measured per-operation ownership. The
 * only changed mechanism is the four-Cin arithmetic and benchmark weight view.
 */
export function planAceOpt0048VaeConvTranspose1dK4(
  operationLabel: string,
  shape: AceVaeConvTranspose1dShape,
): AceOpt0048VaeConvTranspose1dK4Plan {
  const selected = selectAceOpt0040VaeConvTranspose1d(operationLabel, shape);
  const weight = planAceOpt0048VaeConvTranspose1dK4Weight(
    shape,
    selected.reuseAxis,
  );
  return Object.freeze({
    ...selected.plan,
    operationLabel: selected.operationLabel,
    reuseAxis: selected.reuseAxis,
    kernelId: selected.reuseAxis === "channel"
      ? ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID
      : ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID,
    weightLayout: ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT,
    reductionSemantics:
      "increasing-tap-cin4-fp16-dot4-partials-fp32-running-state" as const,
    inputChannelK4Groups: weight.inputChannelK4Groups,
    outputChannelTiles: weight.outputChannelTiles,
    packedWeightStorageShape: weight.packedWeightStorageShape,
  });
}

export function planAceOpt0048VaeConvTranspose1dK4Range(
  plan: AceOpt0048VaeConvTranspose1dK4Plan,
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

/**
 * Layout-only planner kept independent of the production route so exhaustive
 * pack tests can use bounded tensors without allocating a production weight.
 */
export function planAceOpt0048VaeConvTranspose1dK4Weight(
  shape: Pick<
    AceVaeConvTranspose1dShape,
    "kernelSize" | "stride" | "dilation" | "outputPadding" |
      "inputChannels" | "outputChannels"
  >,
  reuseAxis: "channel" | "row",
): AceOpt0048VaeConvTranspose1dK4WeightPlan {
  for (const [name, value] of [
    ["stride", shape.stride],
    ["kernel size", shape.kernelSize],
    ["input channels", shape.inputChannels],
    ["output channels", shape.outputChannels],
  ] as const) requirePositiveSafeInteger(value, name);
  if (
    shape.kernelSize !== shape.stride * 2 || shape.dilation !== 1 ||
    shape.outputPadding !== 0
  ) {
    throw new RangeError("OPT-0048 pack requires revision-6 polyphase geometry");
  }
  if (shape.inputChannels % ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WIDTH !== 0) {
    throw new RangeError("OPT-0048 pack requires Cin divisible by 4");
  }
  const outputChannelsPerLane = reuseAxis === "channel" ? 8 : 4;
  const outputChannelsPerTile = (outputChannelsPerLane *
    ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE) as 128 | 256;
  if (shape.outputChannels % outputChannelsPerTile !== 0) {
    throw new RangeError(
      `OPT-0048 ${reuseAxis} pack requires Cout divisible by ${outputChannelsPerTile}`,
    );
  }
  const inputChannelK4Groups = shape.inputChannels /
    ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WIDTH;
  const outputChannelTiles = shape.outputChannels / outputChannelsPerTile;
  const logicalWeightElements = checkedProduct([
    shape.stride,
    2,
    shape.inputChannels,
    shape.outputChannels,
  ], "OPT-0048 logical weight elements");
  return Object.freeze({
    weightLayout: ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT,
    stride: shape.stride,
    inputChannels: shape.inputChannels,
    outputChannels: shape.outputChannels,
    reuseAxis,
    tapCountPerPhase: 2,
    inputChannelK4Groups,
    outputChannelsPerLane,
    outputChannelsPerTile,
    outputChannelTiles,
    logicalWeightElements,
    packedWeightStorageShape: Object.freeze([
      shape.stride,
      2,
      inputChannelK4Groups,
      outputChannelTiles,
      ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE,
      outputChannelsPerLane,
      ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WIDTH,
    ]) as AceOpt0048VaeConvTranspose1dK4WeightPlan["packedWeightStorageShape"],
  });
}

/** Pack revision-6 `[phase,tap,Cin,Cout]` FP16 words without changing bits. */
export function packAceOpt0048VaeConvTranspose1dK4WeightU16(
  logical: Uint16Array,
  plan: AceOpt0048VaeConvTranspose1dK4WeightPlan,
): Uint16Array {
  if (logical.length !== plan.logicalWeightElements) {
    throw new RangeError(
      `OPT-0048 pack expected ${plan.logicalWeightElements} FP16 words, got ${logical.length}`,
    );
  }
  const packed = new Uint16Array(logical.length);
  visitPackedCoordinates(plan, (
    physical,
    phase,
    tap,
    inputChannel,
    outputChannel,
  ) => {
    const logicalIndex = (((phase * 2 + tap) * plan.inputChannels +
      inputChannel) * plan.outputChannels + outputChannel);
    packed[physical] = logical[logicalIndex]!;
  });
  return packed;
}

/** Inverse used only by the benchmark's authenticated layout proof. */
export function unpackAceOpt0048VaeConvTranspose1dK4WeightU16(
  packed: Uint16Array,
  plan: AceOpt0048VaeConvTranspose1dK4WeightPlan,
): Uint16Array {
  if (packed.length !== plan.logicalWeightElements) {
    throw new RangeError(
      `OPT-0048 inverse expected ${plan.logicalWeightElements} FP16 words, got ${packed.length}`,
    );
  }
  const logical = new Uint16Array(packed.length);
  visitPackedCoordinates(plan, (
    physical,
    phase,
    tap,
    inputChannel,
    outputChannel,
  ) => {
    const logicalIndex = (((phase * 2 + tap) * plan.inputChannels +
      inputChannel) * plan.outputChannels + outputChannel);
    logical[logicalIndex] = packed[physical]!;
  });
  return logical;
}

export function aceOpt0048VaeConvTranspose1dK4PackedWeightIndex(
  phase: number,
  tap: number,
  inputChannel: number,
  outputChannel: number,
  plan: AceOpt0048VaeConvTranspose1dK4WeightPlan,
): number {
  requireCoordinate(phase, plan.stride, "phase");
  requireCoordinate(tap, 2, "tap");
  requireCoordinate(inputChannel, plan.inputChannels, "input channel");
  requireCoordinate(outputChannel, plan.outputChannels, "output channel");
  const cin4 = Math.floor(inputChannel / 4);
  const cinElement4 = inputChannel % 4;
  const coutTile = Math.floor(outputChannel / plan.outputChannelsPerTile);
  const outputInTile = outputChannel % plan.outputChannelsPerTile;
  const lane = Math.floor(outputInTile / plan.outputChannelsPerLane);
  const outputWithinLane = outputInTile % plan.outputChannelsPerLane;
  return (((((((phase * 2 + tap) * plan.inputChannelK4Groups + cin4) *
    plan.outputChannelTiles + coutTile) *
    ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE + lane) *
    plan.outputChannelsPerLane + outputWithinLane) * 4) + cinElement4);
}

export function aceOpt0048VaeConvTranspose1dK4Wgsl(
  operationLabel: string,
  shape: AceVaeConvTranspose1dShape,
): string {
  const plan = planAceOpt0048VaeConvTranspose1dK4(operationLabel, shape);
  const rows = plan.rowsPerSubgroup;
  const outputs = plan.channelsPerLane;
  const declarations = Array.from({ length: rows }, (_, row) =>
    outputs === 4
      ? `  var sum${row} = bias_value0;`
      : `  var sum${row}_0 = bias_value0;\n  var sum${row}_1 = bias_value1;`
  ).join("\n");
  const rowState = Array.from({ length: rows }, (_, row) => `
  let phase_row${row} = subgroup_first_phase_row + ${row}u;
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
enable subgroups;

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

@compute @workgroup_size(
  ${ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE}, 1, 1,
)
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
  let subgroup_first_phase_row =
    group.x * ${plan.rowsPerWorkgroup}u + subgroup * ${rows}u;
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
      let weight_base = (((((congruent_kernel * 2u + tap) *
        INPUT_CHANNEL_K4_GROUPS + input_channel4) * OUTPUT_CHANNEL_TILES +
        group.y) * 32u + lane) * OUTPUTS_PER_LANE);
${weightLoads}
${contractions}
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
      let input_operand${row} = subgroupBroadcast(lane_input, ${row}u);
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

function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`OPT-0048 ${label} must be a positive safe integer`);
  }
}

function requireCoordinate(value: number, extent: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= extent) {
    throw new RangeError(`OPT-0048 ${label} ${value} is out of bounds [0, ${extent})`);
  }
}

function checkedProduct(values: readonly number[], label: string): number {
  let product = 1;
  for (const value of values) {
    requirePositiveSafeInteger(value, label);
    product *= value;
    if (!Number.isSafeInteger(product)) {
      throw new RangeError(`${label} exceeds JavaScript's exact integer domain`);
    }
  }
  return product;
}

function visitPackedCoordinates(
  plan: AceOpt0048VaeConvTranspose1dK4WeightPlan,
  visit: (
    physical: number,
    phase: number,
    tap: number,
    inputChannel: number,
    outputChannel: number,
  ) => void,
): void {
  let physical = 0;
  for (let phase = 0; phase < plan.stride; phase += 1) {
    for (let tap = 0; tap < 2; tap += 1) {
      for (let cin4 = 0;
        cin4 < plan.inputChannelK4Groups;
        cin4 += 1) {
        for (let coutTile = 0;
          coutTile < plan.outputChannelTiles;
          coutTile += 1) {
          for (let lane = 0; lane < 32; lane += 1) {
            for (let outputWithinLane = 0;
              outputWithinLane < plan.outputChannelsPerLane;
              outputWithinLane += 1) {
              const outputChannel = coutTile * plan.outputChannelsPerTile +
                lane * plan.outputChannelsPerLane + outputWithinLane;
              for (let cinElement4 = 0; cinElement4 < 4; cinElement4 += 1) {
                visit(
                  physical,
                  phase,
                  tap,
                  cin4 * 4 + cinElement4,
                  outputChannel,
                );
                physical += 1;
              }
            }
          }
        }
      }
    }
  }
  if (physical !== plan.logicalWeightElements) {
    throw new Error("OPT-0048 packed traversal did not cover the payload");
  }
}
