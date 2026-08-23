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

export const ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID =
  "ace-vae-fp16-opt-0017-reordered-dot4-packed-kio-k7-conv1d-v1" as const;
export const ACE_OPT_0017_VAE_CONV1D_REDUCTION_SEMANTICS =
  "reordered-rounding-dot4" as const;
export const ACE_OPT_0017_VAE_CONV1D_WORKGROUP_SIZE = 256;
export const ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE = 32;
export const ACE_OPT_0017_VAE_CONV1D_WEIGHT_PANEL_STRIDE = 33;
export const ACE_OPT_0017_VAE_CONV1D_ROWS_PER_THREAD = 4;
export const ACE_OPT_0017_VAE_CONV1D_CHANNELS_PER_THREAD = 4;
export const ACE_OPT_0017_VAE_CONV1D_ACCUMULATORS_PER_THREAD = 16;
export const ACE_OPT_0017_VAE_CONV1D_BARRIERS_PER_REDUCTION_TILE = 2;

const FLOAT16_BYTES = 2;
const GPU_BUFFER_ALIGNMENT = 4;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_CONTROL_BYTES = 16;

export type AceOpt0017VaeConv1dTile = "m32n128" | "m64n64";

export interface AceOpt0017VaeConv1dPackedKioBindings {
  /** FP16 activation in frame-major NLC order. */
  readonly input: GPUBufferBinding;
  /** OPT-0014's bit-preserving FP16 `[kernel,input,output]` payload. */
  readonly packedWeight: GPUBufferBinding;
  /** Required FP16 `[output]` bias for every OPT-0017 candidate operation. */
  readonly bias: GPUBufferBinding;
  /** FP16 internal activation. The final FP32 Conv1D is not a candidate. */
  readonly output: GPUBufferBinding;
}

export interface AceOpt0017VaeConv1dCooperativePlan {
  readonly kernelId:
    typeof ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID;
  readonly reductionSemantics:
    typeof ACE_OPT_0017_VAE_CONV1D_REDUCTION_SEMANTICS;
  readonly tile: AceOpt0017VaeConv1dTile;
  readonly tileRows: 32 | 64;
  readonly tileChannels: 128 | 64;
  readonly rowBlocks: 8 | 16;
  readonly channelBlocks: 32 | 16;
  readonly rowsPerThread:
    typeof ACE_OPT_0017_VAE_CONV1D_ROWS_PER_THREAD;
  readonly channelsPerThread:
    typeof ACE_OPT_0017_VAE_CONV1D_CHANNELS_PER_THREAD;
  readonly accumulatorsPerThread:
    typeof ACE_OPT_0017_VAE_CONV1D_ACCUMULATORS_PER_THREAD;
  readonly reductionTile:
    typeof ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE;
  readonly reductionTileCount: number;
  readonly barriersPerWorkgroup: number;
  readonly inputPanelElements: number;
  readonly weightPanelStride:
    typeof ACE_OPT_0017_VAE_CONV1D_WEIGHT_PANEL_STRIDE;
  /** R32 x N values loaded contiguously from KIO before transposition. */
  readonly weightTileElements: number;
  /** N x stride-33 workgroup storage, including one unused value per output. */
  readonly weightPanelElements: number;
  readonly workgroupStorageBytes: 10_496 | 8_320;
  readonly estimatedGlobalOperandBytesPerWorkgroup: number;
  readonly packedWeightPlan: AceOpt0014VaeConv1dPackedWeightPlan;
}

export interface AceOpt0017VaeConv1dCooperativeRangePlan {
  readonly kernelId:
    typeof ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID;
  readonly base: number;
  readonly count: number;
  readonly batch: number;
  readonly firstOutputTime: number;
  readonly firstOutputRow: number;
  readonly outputRowCount: number;
  readonly tile: AceOpt0017VaeConv1dTile;
  readonly tileRows: 32 | 64;
  readonly tileChannels: 128 | 64;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
  readonly physicalWorkgroups: number;
  readonly barrierEvents: number;
  readonly estimatedGlobalOperandBytes: number;
}

export interface AceOpt0017VaeConv1dCooperativeDispatch {
  readonly label: string;
  readonly kernelId:
    typeof ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly cooperativePlan: AceOpt0017VaeConv1dCooperativePlan;
  readonly outputRange: AceOpt0017VaeConv1dCooperativeRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface TileSpec {
  readonly tile: AceOpt0017VaeConv1dTile;
  readonly tileRows: 32 | 64;
  readonly tileChannels: 128 | 64;
  readonly rowBlocks: 8 | 16;
  readonly channelBlocks: 32 | 16;
  readonly workgroupStorageBytes: 10_496 | 8_320;
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
 * Isolated benchmark-only owner for OPT-0017's reordered-rounding K7 tile.
 *
 * The owner consumes OPT-0014's unchanged KIO bytes and deliberately groups
 * each increasing-R run into FP32 dot4 additions. It has no repack,
 * allocation, fallback, production route, or final FP32 Conv1D path.
 */
export class AceOpt0017VaeConv1dCooperativeDot4Kernel {
  private readonly pipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(device: GPUDevice): AceOpt0017VaeConv1dCooperativeDot4Kernel {
    requireKernelDevice(device);
    return new AceOpt0017VaeConv1dCooperativeDot4Kernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceOpt0017VaeConv1dPackedKioBindings,
    outputStorage: AceFp16VaeConv1dOutputStorage,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0017VaeConv1dCooperativeDispatch> {
    this.requireLive();
    const plan = planAceFp16VaeConv1d(shape, outputStorage);
    requireCandidateBoundary(plan, bindings.bias !== undefined);
    const cooperativePlan = planAceOpt0017VaeConv1dCooperativeDot4(plan);
    const outputRange = planAceOpt0017VaeConv1dCooperativeDot4Range(
      plan,
      range,
    );
    requireDispatchDimension(
      this.device,
      outputRange.workgroupsX,
      outputRange.workgroupsY,
    );
    requireWorkgroupStorage(this.device, cooperativePlan.workgroupStorageBytes);
    requireBufferLimits(this.device, [
      ["input", plan.inputBindingBytes],
      ["packed weight", cooperativePlan.packedWeightPlan.packedBindingBytes],
      ["bias", plan.biasBindingBytes],
      ["output", plan.outputBindingBytes],
    ]);
    const normalized = Object.freeze([
      Object.freeze({
        name: "input" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.input,
          plan.inputStorageBytes,
          plan.inputBindingBytes,
          `${label} input`,
        ),
      }),
      Object.freeze({
        name: "packed weight" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.packedWeight,
          cooperativePlan.packedWeightPlan.packedStorageBytes,
          cooperativePlan.packedWeightPlan.packedBindingBytes,
          `${label} packed weight`,
        ),
      }),
      Object.freeze({
        name: "bias" as const,
        binding: requireStorageBinding(
          this.device,
          bindings.bias,
          plan.biasStorageBytes,
          plan.biasBindingBytes,
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

    const compiled = await this.pipelineFor(plan);
    this.requireLive();
    const controlOffset = normalized[4]!.binding.offset ?? 0;
    const resources = normalized.map(({ binding }, index) =>
      index === 4
        ? Object.freeze({
            buffer: binding.buffer,
            offset: 0,
            size: OUTPUT_RANGE_CONTROL_BYTES,
          })
        : binding
    );
    const bindGroup = this.bindGroupFor(
      `${ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID}:${convKey(plan)}`,
      `${label}-opt-0017-cooperative-dot4-bindings`,
      compiled.bindGroupLayout,
      resources,
    );
    const owner = this;
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID,
      plan,
      cooperativePlan,
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
    const created = compileCooperativeDot4(this.device, plan);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
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
      throw new Error("OPT-0017 cooperative-dot4 Conv1D kernel was destroyed");
    }
  }
}

export function planAceOpt0017VaeConv1dCooperativeDot4(
  shape: AceVaeConv1dShape,
): AceOpt0017VaeConv1dCooperativePlan {
  const plan = planAceFp16VaeConv1d(shape, "float16");
  requireCandidateShape(plan);
  const tile = selectTile(plan);
  const packedWeightPlan = planAceOpt0014VaeConv1dPackedKioWeight(plan);
  const reductionTileCount = checkedInteger(
    7 * (plan.inputChannels / ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE),
    "reduction-tile count",
  );
  const barriersPerWorkgroup = checkedInteger(
    reductionTileCount * ACE_OPT_0017_VAE_CONV1D_BARRIERS_PER_REDUCTION_TILE,
    "barriers per workgroup",
  );
  const inputPanelElements = checkedInteger(
    tile.tileRows * ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE,
    "input-panel elements",
  );
  const weightPanelElements = checkedInteger(
    tile.tileChannels * ACE_OPT_0017_VAE_CONV1D_WEIGHT_PANEL_STRIDE,
    "weight-panel elements",
  );
  const weightTileElements = checkedInteger(
    tile.tileChannels * ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE,
    "weight-tile elements",
  );
  const estimatedGlobalOperandBytesPerWorkgroup = checkedInteger(
    reductionTileCount * ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE *
      (tile.tileRows + tile.tileChannels) * FLOAT16_BYTES,
    "estimated global operand bytes per workgroup",
  );
  return Object.freeze({
    kernelId: ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID,
    reductionSemantics: ACE_OPT_0017_VAE_CONV1D_REDUCTION_SEMANTICS,
    ...tile,
    rowsPerThread: ACE_OPT_0017_VAE_CONV1D_ROWS_PER_THREAD,
    channelsPerThread: ACE_OPT_0017_VAE_CONV1D_CHANNELS_PER_THREAD,
    accumulatorsPerThread: ACE_OPT_0017_VAE_CONV1D_ACCUMULATORS_PER_THREAD,
    reductionTile: ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE,
    reductionTileCount,
    barriersPerWorkgroup,
    inputPanelElements,
    weightPanelStride: ACE_OPT_0017_VAE_CONV1D_WEIGHT_PANEL_STRIDE,
    weightTileElements,
    weightPanelElements,
    estimatedGlobalOperandBytesPerWorkgroup,
    packedWeightPlan,
  });
}

export function planAceOpt0017VaeConv1dCooperativeDot4Range(
  plan: AceFp16VaeConv1dPlan,
  range: Readonly<{ base: number; count: number }>,
): AceOpt0017VaeConv1dCooperativeRangePlan {
  requireCandidateBoundary(plan, true);
  const cooperative = planAceOpt0017VaeConv1dCooperativeDot4(plan);
  const portable = planAceFp16VaeConv1dRange(plan, range);
  const workgroupsX = checkedInteger(
    Math.ceil(portable.outputRowCount / cooperative.tileRows),
    "Conv1D workgroups X",
  );
  const workgroupsY = checkedInteger(
    Math.ceil(plan.outputChannels / cooperative.tileChannels),
    "Conv1D workgroups Y",
  );
  const physicalWorkgroups = checkedInteger(
    workgroupsX * workgroupsY,
    "physical workgroups",
  );
  const barrierEvents = checkedInteger(
    physicalWorkgroups * cooperative.barriersPerWorkgroup,
    "barrier events",
  );
  // Weight panels stage a complete N x R32 tile for every physical
  // workgroup. Dense A panels omit global loads for padded tail rows, so their
  // estimate uses the range's exact valid-row count instead of ceil(rows/M)*M.
  const estimatedWeightBytes = physicalWorkgroups *
    cooperative.reductionTileCount * cooperative.tileChannels *
    ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE * FLOAT16_BYTES;
  const estimatedInputBytes = workgroupsY * portable.outputRowCount *
    cooperative.reductionTileCount *
    ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE * FLOAT16_BYTES;
  const estimatedGlobalOperandBytes = checkedInteger(
    estimatedWeightBytes + estimatedInputBytes,
    "estimated global operand bytes",
  );
  return Object.freeze({
    kernelId: ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID,
    base: portable.base,
    count: portable.count,
    batch: portable.batch,
    firstOutputTime: portable.firstOutputTime,
    firstOutputRow: portable.firstOutputRow,
    outputRowCount: portable.outputRowCount,
    tile: cooperative.tile,
    tileRows: cooperative.tileRows,
    tileChannels: cooperative.tileChannels,
    workgroupsX,
    workgroupsY,
    physicalWorkgroups,
    barrierEvents,
    estimatedGlobalOperandBytes,
  });
}

export function aceOpt0017VaeConv1dCooperativeDot4Wgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceFp16VaeConv1dOutputStorage,
): string {
  const plan = planAceFp16VaeConv1d(shape, outputStorage);
  requireCandidateBoundary(plan, hasBias);
  const cooperative = planAceOpt0017VaeConv1dCooperativeDot4(plan);
  const rowValidity = Array.from(
    { length: ACE_OPT_0017_VAE_CONV1D_ROWS_PER_THREAD },
    (_, row) => `
    let output_time${row} = owned_first_time + ${row}u;
    let padded_time${row} = output_time${row} + kernel * DILATION;
    var row_valid${row} = false;
    if (output_time${row} < range_end_time && padded_time${row} >= PADDING) {
      row_valid${row} = padded_time${row} - PADDING < INPUT_FRAMES;
    }`,
  ).join("");
  const sums = Array.from(
    { length: ACE_OPT_0017_VAE_CONV1D_ROWS_PER_THREAD },
    (_, row) => `  var sum${row} = initial_sum;`,
  ).join("\n");
  const dotGroups = Array.from(
    { length: ACE_OPT_0017_VAE_CONV1D_ROWS_PER_THREAD },
    (_, row) => dotGroupsForRow(row),
  ).join("\n");
  const stores = Array.from(
    { length: ACE_OPT_0017_VAE_CONV1D_ROWS_PER_THREAD },
    (_, row) => storesForRow(row),
  ).join("\n");
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID}
// reduction-semantics: ${ACE_OPT_0017_VAE_CONV1D_REDUCTION_SEMANTICS}
// tile: ${cooperative.tile}; each thread owns 4 rows x 4 output channels
enable f16;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const TILE_ROWS: u32 = ${cooperative.tileRows}u;
const TILE_CHANNELS: u32 = ${cooperative.tileChannels}u;
const CHANNEL_BLOCKS: u32 = ${cooperative.channelBlocks}u;
const REDUCTION_TILE: u32 = ${ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE}u;
const WEIGHT_PANEL_STRIDE: u32 =
  ${ACE_OPT_0017_VAE_CONV1D_WEIGHT_PANEL_STRIDE}u;
const INPUT_PANEL_ELEMENTS: u32 = ${cooperative.inputPanelElements}u;
const WEIGHT_TILE_ELEMENTS: u32 = ${cooperative.weightTileElements}u;
const WEIGHT_PANEL_ELEMENTS: u32 = ${cooperative.weightPanelElements}u;
const PADDING: u32 = ${plan.padding}u;
const DILATION: u32 = ${plan.dilation}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> packed_weight: array<f16>;
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

var<workgroup> input_panel: array<f16, ${cooperative.inputPanelElements}>;
var<workgroup> weight_panel: array<f16, ${cooperative.weightPanelElements}>;

@compute @workgroup_size(${ACE_OPT_0017_VAE_CONV1D_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
) {
  let first_output_row = output_range.first_output / OUTPUT_CHANNELS;
  let output_row_count = output_range.output_count / OUTPUT_CHANNELS;
  let range_first_time = first_output_row % OUTPUT_FRAMES;
  let batch = first_output_row / OUTPUT_FRAMES;
  let range_end_time = range_first_time + output_row_count;
  let row_block = local_index / CHANNEL_BLOCKS;
  let channel_block = local_index % CHANNEL_BLOCKS;
  let tile_first_time = range_first_time + group.x * TILE_ROWS;
  let owned_first_time = tile_first_time + row_block * 4u;
  let output_channel_base =
    group.y * TILE_CHANNELS + channel_block * 4u;

  var initial_sum = vec4<f32>(0.0);
  if (output_channel_base + 3u < OUTPUT_CHANNELS) {
    initial_sum = vec4<f32>(
      f32(bias[output_channel_base]),
      f32(bias[output_channel_base + 1u]),
      f32(bias[output_channel_base + 2u]),
      f32(bias[output_channel_base + 3u])
    );
  }
${sums}

  // This is deliberately not the shipped scalar source-order reduction.
  // Increasing R is grouped into vec4<f32> dot products and each dot result
  // is added directly to its final FP32 accumulator.
  for (var kernel = 0u; kernel < 7u; kernel += 1u) {
${rowValidity}
    for (
      var input_channel_base = 0u;
      input_channel_base < INPUT_CHANNELS;
      input_channel_base += REDUCTION_TILE
    ) {
      for (
        var panel_index = local_index;
        panel_index < INPUT_PANEL_ELEMENTS;
        panel_index += ${ACE_OPT_0017_VAE_CONV1D_WORKGROUP_SIZE}u
      ) {
        let panel_row = panel_index / REDUCTION_TILE;
        let panel_r = panel_index % REDUCTION_TILE;
        let panel_output_time = tile_first_time + panel_row;
        let panel_padded_time = panel_output_time + kernel * DILATION;
        var value = f16(0.0);
        if (
          panel_output_time < range_end_time &&
          panel_padded_time >= PADDING
        ) {
          let panel_input_time = panel_padded_time - PADDING;
          if (panel_input_time < INPUT_FRAMES) {
            value = input[
              (batch * INPUT_FRAMES + panel_input_time) * INPUT_CHANNELS +
              input_channel_base + panel_r
            ];
          }
        }
        input_panel[panel_index] = value;
      }

      for (
        var load_index = local_index;
        load_index < WEIGHT_TILE_ELEMENTS;
        load_index += ${ACE_OPT_0017_VAE_CONV1D_WORKGROUP_SIZE}u
      ) {
        // KIO is R-major, so adjacent lanes load adjacent output channels.
        // The store transposes that coalesced load into the stride-33 panel.
        let panel_r = load_index / TILE_CHANNELS;
        let panel_channel = load_index % TILE_CHANNELS;
        let output_channel = group.y * TILE_CHANNELS + panel_channel;
        var value = f16(0.0);
        if (output_channel < OUTPUT_CHANNELS) {
          let packed_index =
            (kernel * INPUT_CHANNELS + input_channel_base + panel_r) *
              OUTPUT_CHANNELS +
            output_channel;
          value = packed_weight[packed_index];
        }
        weight_panel[
          panel_channel * WEIGHT_PANEL_STRIDE + panel_r
        ] = value;
      }

      // Both barriers are unconditional for every WG256 invocation.
      workgroupBarrier();
${dotGroups}
      workgroupBarrier();
    }
  }

${stores}
}
`;
}

async function compileCooperativeDot4(
  device: GPUDevice,
  plan: AceFp16VaeConv1dPlan,
): Promise<CompiledKernel> {
  const cooperative = planAceOpt0017VaeConv1dCooperativeDot4(plan);
  const label = `ace-opt-0017-vae-k7-${cooperative.tile}-${convKey(plan)}`;
  const module = await checkedShaderModule(
    device,
    label,
    aceOpt0017VaeConv1dCooperativeDot4Wgsl(plan, true, "float16"),
  );
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: [
      ...[
        plan.inputBindingBytes,
        cooperative.packedWeightPlan.packedBindingBytes,
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
      `OPT-0017 cooperative-dot4 WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  return module;
}

function dotGroupsForRow(row: number): string {
  const channelAdds = ["x", "y", "z", "w"].map(
    (component, channel) => `
          let weight_operand${row}${channel} = vec4<f32>(
            f32(weight_panel[(owned_channel_base + ${channel}u) *
              WEIGHT_PANEL_STRIDE + r4]),
            f32(weight_panel[(owned_channel_base + ${channel}u) *
              WEIGHT_PANEL_STRIDE + r4 + 1u]),
            f32(weight_panel[(owned_channel_base + ${channel}u) *
              WEIGHT_PANEL_STRIDE + r4 + 2u]),
            f32(weight_panel[(owned_channel_base + ${channel}u) *
              WEIGHT_PANEL_STRIDE + r4 + 3u])
          );
          sum${row}.${component} = sum${row}.${component} +
            dot(input_operand${row}, weight_operand${row}${channel});`,
  ).join("");
  return `
      if (row_valid${row} && output_channel_base + 3u < OUTPUT_CHANNELS) {
        let owned_panel_row = row_block * 4u + ${row}u;
        let owned_channel_base = channel_block * 4u;
        for (var r4 = 0u; r4 < REDUCTION_TILE; r4 += 4u) {
          let input_operand${row} = vec4<f32>(
            f32(input_panel[owned_panel_row * REDUCTION_TILE + r4]),
            f32(input_panel[owned_panel_row * REDUCTION_TILE + r4 + 1u]),
            f32(input_panel[owned_panel_row * REDUCTION_TILE + r4 + 2u]),
            f32(input_panel[owned_panel_row * REDUCTION_TILE + r4 + 3u])
          );${channelAdds}
        }
      }`;
}

function storesForRow(row: number): string {
  return `
  let store_time${row} = owned_first_time + ${row}u;
  if (
    store_time${row} < range_end_time &&
    output_channel_base + 3u < OUTPUT_CHANNELS
  ) {
    let store_base${row} =
      (batch * OUTPUT_FRAMES + store_time${row}) * OUTPUT_CHANNELS +
      output_channel_base;
    output[store_base${row}] = f16(sum${row}.x);
    output[store_base${row} + 1u] = f16(sum${row}.y);
    output[store_base${row} + 2u] = f16(sum${row}.z);
    output[store_base${row} + 3u] = f16(sum${row}.w);
  }`;
}

function selectTile(plan: AceFp16VaeConv1dPlan): TileSpec {
  if (plan.inputChannels === 1_024 && plan.outputChannels === 1_024) {
    return Object.freeze({
      tile: "m32n128",
      tileRows: 32,
      tileChannels: 128,
      rowBlocks: 8,
      channelBlocks: 32,
      workgroupStorageBytes: 10_496,
    });
  }
  return Object.freeze({
    tile: "m64n64",
    tileRows: 64,
    tileChannels: 64,
    rowBlocks: 16,
    channelBlocks: 16,
    workgroupStorageBytes: 8_320,
  });
}

function requireCandidateShape(plan: AceFp16VaeConv1dPlan): void {
  if (plan.family !== "k7") {
    throw new RangeError("OPT-0017 cooperative-dot4 supports only K7");
  }
  if (plan.inputChannels % ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE !== 0) {
    throw new RangeError(
      "OPT-0017 cooperative-dot4 requires input channels divisible by R32",
    );
  }
  const productionChannels =
    (plan.inputChannels === 64 && plan.outputChannels === 2_048) ||
    (plan.inputChannels === plan.outputChannels &&
      [128, 256, 512, 1_024].includes(plan.inputChannels));
  if (!productionChannels) {
    throw new RangeError(
      "OPT-0017 cooperative-dot4 supports only the 16 biased production K7 channel shapes",
    );
  }
}

function requireCandidateBoundary(
  plan: AceFp16VaeConv1dPlan,
  hasBias: boolean,
): void {
  if (plan.outputStorage !== "float16" || !hasBias) {
    throw new RangeError(
      "OPT-0017 cooperative-dot4 supports only biased FP16-output K7 operations; final FP32 conv2 remains shipped fixed32",
    );
  }
  requireCandidateShape(plan);
}

function requireKernelDevice(device: GPUDevice): void {
  if (!device.features.has("shader-f16")) {
    throw new Error(
      "OPT-0017 cooperative-dot4 Conv1D requires WebGPU shader-f16",
    );
  }
  const maximumInvocations = device.limits.maxComputeInvocationsPerWorkgroup;
  const maximumSizeX = device.limits.maxComputeWorkgroupSizeX;
  if (
    !Number.isSafeInteger(maximumInvocations) ||
    !Number.isSafeInteger(maximumSizeX) ||
    maximumInvocations < ACE_OPT_0017_VAE_CONV1D_WORKGROUP_SIZE ||
    maximumSizeX < ACE_OPT_0017_VAE_CONV1D_WORKGROUP_SIZE
  ) {
    throw new Error(
      `OPT-0017 cooperative-dot4 Conv1D requires WG${ACE_OPT_0017_VAE_CONV1D_WORKGROUP_SIZE}`,
    );
  }
}

function requireWorkgroupStorage(device: GPUDevice, requiredBytes: number): void {
  const maximum = device.limits.maxComputeWorkgroupStorageSize;
  if (!Number.isSafeInteger(maximum) || maximum < requiredBytes) {
    throw new RangeError(
      `OPT-0017 cooperative-dot4 Conv1D requires ${requiredBytes} workgroup-storage bytes`,
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
      "OPT-0017 cooperative-dot4 Conv1D exceeds the device dispatch dimension",
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
      "OPT-0017 cooperative-dot4 device reported invalid buffer limits",
    );
  }
  for (const [name, bytes] of resources) {
    if (bytes > maximumBinding || bytes > maximumBuffer) {
      throw new RangeError(
        `OPT-0017 cooperative-dot4 ${name} exceeds the device buffer limits`,
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
      "OPT-0017 cooperative-dot4 device reported an invalid storage alignment",
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
      "OPT-0017 cooperative-dot4 device reported an invalid uniform alignment",
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

function convKey(plan: AceFp16VaeConv1dPlan): string {
  return [
    "cooperative-dot4-packed-kio",
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

function checkedInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(
      `OPT-0017 cooperative-dot4 ${label} exceeds JavaScript's exact integer domain`,
    );
  }
  return value;
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
