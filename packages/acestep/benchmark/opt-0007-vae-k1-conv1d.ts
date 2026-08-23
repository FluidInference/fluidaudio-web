import {
  requireAceBindingBytes,
  requireAceDisjointOutput,
} from "../src/webgpu/kernels/correctness-utils.js";
import {
  planAceVaeConv1d,
  type AceVaeConv1dShape,
  type AceVaeConvBindings,
} from "../src/webgpu/kernels/vae-primitives.js";
import { createAceScopedBuffers } from
  "../src/webgpu/scoped-buffer-allocation.js";
import { ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY } from
  "../src/webgpu/vae-decoder.js";

export const ACE_OPT_0007_VAE_K1_CONV1D_KERNEL_SIZE = 1;
export const ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK = 64;
export const ACE_OPT_0007_VAE_K1_CONV1D_TILE_FRAMES = 16;
export const ACE_OPT_0007_VAE_K1_CONV1D_TILE_CHANNELS = 8;
/** One unused time slot avoids a bank-aligned shared-input row stride. */
export const ACE_OPT_0007_VAE_K1_CONV1D_INPUT_TILE_STRIDE = 17;
/** One unused channel slot avoids a bank-aligned shared-weight row stride. */
export const ACE_OPT_0007_VAE_K1_CONV1D_WEIGHT_TILE_STRIDE = 65;
export const ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_X = 16;
export const ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_Y = 8;
export const ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE = 128;

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const MAX_DISPATCH_DIMENSION = 65_535;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_PARAMETER_BYTES = 16;
const MINIMUM_UNIFORM_STRIDE = 256;

export interface AceOpt0007VaeK1Conv1dOutputRange {
  readonly batch: number;
  readonly firstOutputTime: number;
  /** Global NLC row, before multiplying by the complete channel width. */
  readonly firstOutputRow: number;
  /** Complete-channel rows; a range never crosses a batch boundary. */
  readonly outputRowCount: number;
  readonly firstOutput: number;
  readonly outputCount: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
  readonly workgroupCount: number;
  readonly multiplyAdds: number;
}

export interface AceOpt0007VaeK1Conv1dPlan extends AceVaeConv1dShape {
  readonly outputFrames: number;
  readonly inputElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly inputChannelChunk:
    typeof ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK;
  readonly inputChannelChunkCount: number;
  readonly tileFrames: typeof ACE_OPT_0007_VAE_K1_CONV1D_TILE_FRAMES;
  readonly tileChannels: typeof ACE_OPT_0007_VAE_K1_CONV1D_TILE_CHANNELS;
  readonly inputTileStride:
    typeof ACE_OPT_0007_VAE_K1_CONV1D_INPUT_TILE_STRIDE;
  readonly weightTileStride:
    typeof ACE_OPT_0007_VAE_K1_CONV1D_WEIGHT_TILE_STRIDE;
  readonly workgroupSizeX:
    typeof ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_X;
  readonly workgroupSizeY:
    typeof ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_Y;
  readonly workgroupSize: typeof ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE;
  readonly inputTileElements: number;
  readonly weightTileElements: number;
  readonly inputTileBytes: number;
  readonly weightTileBytes: number;
  readonly workgroupStorageBytes: number;
  readonly outputRangeCount: number;
  readonly outputRanges: readonly AceOpt0007VaeK1Conv1dOutputRange[];
}

export interface AceOpt0007VaeK1Conv1dDispatch {
  readonly label: string;
  readonly plan: AceOpt0007VaeK1Conv1dPlan;
  readonly rangeCount: number;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledAceOpt0007VaeK1Conv1d {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly outputRangeParameters: GPUBuffer;
  readonly outputRangeParameterStride: number;
  destroy(): void;
}

/**
 * Benchmark-only exact-loop pointwise FP32 Conv1D candidate for OPT-0007.
 *
 * One WG(16,8) owns 16 frames by 8 output channels. Increasing 64-channel
 * chunks stage one input and weight tile while every invocation remains the
 * sole owner of one output scalar and visits real input channels in the same
 * global order as the scalar authority.
 */
export class AceOpt0007VaeK1Conv1dKernel {
  private readonly compiled = new Map<
    string,
    Promise<CompiledAceOpt0007VaeK1Conv1d>
  >();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(device: GPUDevice): AceOpt0007VaeK1Conv1dKernel {
    const maximumInvocations =
      device.limits.maxComputeInvocationsPerWorkgroup;
    const maximumSizeX = device.limits.maxComputeWorkgroupSizeX;
    const maximumSizeY = device.limits.maxComputeWorkgroupSizeY;
    if (
      !Number.isSafeInteger(maximumInvocations) ||
      !Number.isSafeInteger(maximumSizeX) ||
      !Number.isSafeInteger(maximumSizeY) ||
      maximumInvocations <
        ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE ||
      maximumSizeX <
        ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_X ||
      maximumSizeY <
        ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_Y
    ) {
      throw new Error(
        "OPT-0007 VAE K1 Conv1D requires a 16x8 (128-lane) compute workgroup",
      );
    }
    return new AceOpt0007VaeK1Conv1dKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceVaeConvBindings,
  ): Promise<AceOpt0007VaeK1Conv1dDispatch> {
    this.requireLive();
    const plan = planAceOpt0007VaeK1Conv1d(shape);
    this.requireDeviceLimits(plan);

    const inputBytes = checkedBytes(plan.inputElements, "input");
    const weightBytes = checkedBytes(plan.weightElements, "weight");
    const outputBytes = checkedBytes(plan.outputElements, "output");
    const biasBytes = checkedBytes(plan.outputChannels, "bias");
    requireAceBindingBytes(bindings.input, inputBytes, `${label} input`);
    requireAceBindingBytes(bindings.weight, weightBytes, `${label} weight`);
    requireAceBindingBytes(bindings.output, outputBytes, `${label} output`);
    if (bindings.bias !== undefined) {
      requireAceBindingBytes(bindings.bias, biasBytes, `${label} bias`);
    }
    requireAceDisjointOutput(
      exactBinding(bindings.output, outputBytes),
      [
        exactBinding(bindings.input, inputBytes),
        exactBinding(bindings.weight, weightBytes),
        ...(bindings.bias === undefined
          ? []
          : [exactBinding(bindings.bias, biasBytes)]),
      ],
      label,
    );

    const hasBias = bindings.bias !== undefined;
    const compiled = await this.pipelineFor(plan, hasBias);
    this.requireLive();
    const rangeBinding = hasBias ? 4 : 3;
    const resources = hasBias
      ? [bindings.input, bindings.weight, bindings.bias!, bindings.output]
      : [bindings.input, bindings.weight, bindings.output];
    const bindGroupKey = `${convKey(plan, hasBias)}:${resources.map((binding) =>
      this.bindingKey(binding)
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-opt-0007-bindings`,
        layout: compiled.bindGroupLayout,
        entries: [
          ...resources.map((resource, binding) => ({ binding, resource })),
          {
            binding: rangeBinding,
            resource: {
              buffer: compiled.outputRangeParameters,
              offset: 0,
              size: OUTPUT_RANGE_PARAMETER_BYTES,
            },
          },
        ],
      });
      this.bindGroups.set(bindGroupKey, bindGroup);
    }

    return Object.freeze({
      label,
      plan,
      rangeCount: plan.outputRangeCount,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        const range = plan.outputRanges[rangeIndex];
        if (range === undefined) {
          throw new RangeError(
            `${label} OPT-0007 range ${rangeIndex} is outside [0, ${plan.outputRangeCount})`,
          );
        }
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroup, [
          rangeIndex * compiled.outputRangeParameterStride,
        ]);
        pass.dispatchWorkgroups(range.workgroupsX, range.workgroupsY, 1);
      },
      encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(compiled.pipeline);
        for (let index = 0; index < plan.outputRanges.length; index += 1) {
          const range = plan.outputRanges[index]!;
          pass.setBindGroup(0, bindGroup, [
            index * compiled.outputRangeParameterStride,
          ]);
          pass.dispatchWorkgroups(range.workgroupsX, range.workgroupsY, 1);
        }
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.bindGroups.clear();
    for (const compiled of this.compiled.values()) {
      void compiled.then((resources) => resources.destroy(), () => undefined);
    }
    this.compiled.clear();
  }

  private pipelineFor(
    plan: AceOpt0007VaeK1Conv1dPlan,
    hasBias: boolean,
  ): Promise<CompiledAceOpt0007VaeK1Conv1d> {
    const key = convKey(plan, hasBias);
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceOpt0007VaeK1Conv1d(this.device, plan, hasBias);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
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

  private requireDeviceLimits(plan: AceOpt0007VaeK1Conv1dPlan): void {
    const maximumStorage =
      this.device.limits.maxComputeWorkgroupStorageSize;
    if (
      !Number.isSafeInteger(maximumStorage) ||
      plan.workgroupStorageBytes > maximumStorage
    ) {
      throw new RangeError(
        `OPT-0007 VAE K1 Conv1D requires ${plan.workgroupStorageBytes} workgroup-storage bytes`,
      );
    }
    const maximumDispatch = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (
      !Number.isSafeInteger(maximumDispatch) ||
      maximumDispatch < 1 ||
      plan.outputRanges.some(({ workgroupsX, workgroupsY }) =>
      workgroupsX > maximumDispatch || workgroupsY > maximumDispatch
      )
    ) {
      throw new RangeError(
        "OPT-0007 VAE K1 Conv1D exceeds the device dispatch dimension",
      );
    }
    const maximumBinding = Number(
      this.device.limits.maxStorageBufferBindingSize,
    );
    for (const [name, elements] of [
      ["input", plan.inputElements],
      ["weight", plan.weightElements],
      ["output", plan.outputElements],
      ["bias", plan.outputChannels],
    ] as const) {
      if (
        !Number.isSafeInteger(maximumBinding) ||
        maximumBinding < 1 ||
        checkedBytes(elements, name) > maximumBinding
      ) {
        throw new RangeError(
          `OPT-0007 VAE K1 Conv1D ${name} exceeds the device storage binding limit`,
        );
      }
    }
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0007 VAE K1 Conv1D kernel was destroyed");
    }
  }
}

export function planAceOpt0007VaeK1Conv1d(
  shape: AceVaeConv1dShape,
): AceOpt0007VaeK1Conv1dPlan {
  const portable = planAceVaeConv1d(shape);
  if (
    shape.kernelSize !== ACE_OPT_0007_VAE_K1_CONV1D_KERNEL_SIZE ||
    shape.stride !== 1 ||
    shape.dilation !== 1 ||
    shape.padding !== 0
  ) {
    throw new RangeError(
      "OPT-0007 VAE K1 Conv1D requires K1, stride one, dilation one, and zero padding",
    );
  }
  if (portable.outputFrames !== shape.inputFrames) {
    throw new Error("OPT-0007 VAE K1 Conv1D lost pointwise frame identity");
  }
  for (const [name, elements] of [
    ["input", portable.inputElements],
    ["weight", portable.weightElements],
    ["output", portable.outputElements],
  ] as const) requireWgslIndexable(elements, name);

  const inputChannelChunkCount = Math.ceil(
    shape.inputChannels / ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK,
  );
  requireWgslIndexable(inputChannelChunkCount, "input channel chunk count");
  const inputTileElements = checkedProduct([
    ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK,
    ACE_OPT_0007_VAE_K1_CONV1D_INPUT_TILE_STRIDE,
  ], "input tile");
  const weightTileElements = checkedProduct([
    ACE_OPT_0007_VAE_K1_CONV1D_TILE_CHANNELS,
    ACE_OPT_0007_VAE_K1_CONV1D_WEIGHT_TILE_STRIDE,
  ], "weight tile");
  const inputTileBytes = checkedBytes(inputTileElements, "input tile");
  const weightTileBytes = checkedBytes(weightTileElements, "weight tile");
  const workgroupStorageBytes = checkedSum(
    inputTileBytes,
    weightTileBytes,
    "workgroup storage",
  );
  const workgroupsY = Math.ceil(
    shape.outputChannels / ACE_OPT_0007_VAE_K1_CONV1D_TILE_CHANNELS,
  );
  if (workgroupsY > MAX_DISPATCH_DIMENSION) {
    throw new RangeError(
      "OPT-0007 VAE K1 Conv1D output channels exceed the 2D dispatch domain",
    );
  }
  const multiplyAddsPerRow = checkedProduct([
    shape.outputChannels,
    shape.inputChannels,
  ], "multiply-adds per output row");
  const maximumRowsPerRange = Math.min(
    Math.floor(
      ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY.maximumOutputElements /
        shape.outputChannels,
    ),
    Math.floor(
      ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY
        .maximumConvolutionMultiplyAccumulates / multiplyAddsPerRow,
    ),
    MAX_DISPATCH_DIMENSION * ACE_OPT_0007_VAE_K1_CONV1D_TILE_FRAMES,
  );
  if (maximumRowsPerRange < 1) {
    throw new RangeError(
      "OPT-0007 VAE K1 Conv1D cannot fit one complete-channel output row in a bounded range",
    );
  }

  const outputRanges: AceOpt0007VaeK1Conv1dOutputRange[] = [];
  for (let batch = 0; batch < shape.batch; batch += 1) {
    for (
      let firstOutputTime = 0;
      firstOutputTime < portable.outputFrames;
      firstOutputTime += maximumRowsPerRange
    ) {
      const outputRowCount = Math.min(
        maximumRowsPerRange,
        portable.outputFrames - firstOutputTime,
      );
      const firstOutputRow = checkedSum(
        checkedProduct([batch, portable.outputFrames], "range batch row"),
        firstOutputTime,
        "range first row",
      );
      const outputCount = checkedProduct(
        [outputRowCount, shape.outputChannels],
        "range output count",
      );
      const workgroupsX = Math.ceil(
        outputRowCount / ACE_OPT_0007_VAE_K1_CONV1D_TILE_FRAMES,
      );
      const lastStagedTime = checkedSum(
        firstOutputTime,
        checkedSum(
          checkedProduct([
            workgroupsX - 1,
            ACE_OPT_0007_VAE_K1_CONV1D_TILE_FRAMES,
          ], "last staged tile offset"),
          ACE_OPT_0007_VAE_K1_CONV1D_TILE_FRAMES - 1,
          "last staged tile time",
        ),
        "last staged input time",
      );
      requireWgslIndexable(lastStagedTime, "last staged input time");
      requireWgslIndexable(
        firstOutputTime + outputRowCount,
        "range end time",
      );
      outputRanges.push(Object.freeze({
        batch,
        firstOutputTime,
        firstOutputRow,
        outputRowCount,
        firstOutput: checkedProduct(
          [firstOutputRow, shape.outputChannels],
          "range first output",
        ),
        outputCount,
        workgroupsX,
        workgroupsY,
        workgroupCount: checkedProduct(
          [workgroupsX, workgroupsY],
          "range workgroups",
        ),
        multiplyAdds: checkedProduct(
          [outputRowCount, multiplyAddsPerRow],
          "range multiply-adds",
        ),
      }));
    }
  }
  const emittedOutputs = outputRanges.reduce(
    (sum, range) => checkedSum(sum, range.outputCount, "emitted outputs"),
    0,
  );
  if (emittedOutputs !== portable.outputElements) {
    throw new Error("OPT-0007 VAE K1 Conv1D range planner lost outputs");
  }

  return Object.freeze({
    ...shape,
    outputFrames: portable.outputFrames,
    inputElements: portable.inputElements,
    weightElements: portable.weightElements,
    outputElements: portable.outputElements,
    inputChannelChunk: ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK,
    inputChannelChunkCount,
    tileFrames: ACE_OPT_0007_VAE_K1_CONV1D_TILE_FRAMES,
    tileChannels: ACE_OPT_0007_VAE_K1_CONV1D_TILE_CHANNELS,
    inputTileStride: ACE_OPT_0007_VAE_K1_CONV1D_INPUT_TILE_STRIDE,
    weightTileStride: ACE_OPT_0007_VAE_K1_CONV1D_WEIGHT_TILE_STRIDE,
    workgroupSizeX: ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_X,
    workgroupSizeY: ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_Y,
    workgroupSize: ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE,
    inputTileElements,
    weightTileElements,
    inputTileBytes,
    weightTileBytes,
    workgroupStorageBytes,
    outputRangeCount: outputRanges.length,
    outputRanges: Object.freeze(outputRanges),
  });
}

export function aceOpt0007VaeK1Conv1dWgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
): string {
  const plan = planAceOpt0007VaeK1Conv1d(shape);
  const outputBinding = hasBias ? 3 : 2;
  const rangeBinding = hasBias ? 4 : 3;
  const biasDeclaration = hasBias
    ? "@group(0) @binding(2) var<storage, read> bias: array<f32>;"
    : "";
  const initialSum = hasBias ? "bias[output_channel]" : "0.0";
  return /* wgsl */ `
const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const INPUT_CHANNEL_CHUNKS: u32 = ${plan.inputChannelChunkCount}u;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
${biasDeclaration}
@group(0) @binding(${outputBinding}) var<storage, read_write>
  output: array<f32>;

struct OutputRangeParameters {
  first_output_row: u32,
  output_row_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(${rangeBinding}) var<uniform>
  output_range: OutputRangeParameters;

var<workgroup> input_tile: array<f32, ${plan.inputTileElements}>;
var<workgroup> weight_tile: array<f32, ${plan.weightTileElements}>;

@compute @workgroup_size(
  ${ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_X},
  ${ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_Y},
  1,
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let range_first_time = output_range.first_output_row % OUTPUT_FRAMES;
  let batch = output_range.first_output_row / OUTPUT_FRAMES;
  let tile_first_time =
    range_first_time + group.x * ${ACE_OPT_0007_VAE_K1_CONV1D_TILE_FRAMES}u;
  let output_time = tile_first_time + local.x;
  let output_channel =
    group.y * ${ACE_OPT_0007_VAE_K1_CONV1D_TILE_CHANNELS}u + local.y;
  let range_end_time = range_first_time + output_range.output_row_count;
  let output_active =
    output_time < range_end_time && output_channel < OUTPUT_CHANNELS;
  var sum = 0.0;
  if (output_active) { sum = ${initialSum}; }

  // Concatenating increasing chunks preserves scalar Cin order exactly.
  for (
    var input_channel_chunk = 0u;
    input_channel_chunk < INPUT_CHANNEL_CHUNKS;
    input_channel_chunk += 1u
  ) {
    let chunk_first_channel =
      input_channel_chunk * ${ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK}u;
    for (
      var tile_index = lane;
      tile_index < ${ACE_OPT_0007_VAE_K1_CONV1D_TILE_FRAMES * ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK}u;
      tile_index += ${ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE}u
    ) {
      let tile_time =
        tile_index / ${ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK}u;
      let chunk_channel =
        tile_index % ${ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK}u;
      let input_channel = chunk_first_channel + chunk_channel;
      let input_time = tile_first_time + tile_time;
      var value = 0.0;
      if (input_channel < INPUT_CHANNELS && input_time < INPUT_FRAMES) {
        value = input[
          (batch * INPUT_FRAMES + input_time) * INPUT_CHANNELS +
          input_channel
        ];
      }
      input_tile[
        chunk_channel * ${ACE_OPT_0007_VAE_K1_CONV1D_INPUT_TILE_STRIDE}u +
        tile_time
      ] = value;
    }
    for (
      var tile_index = lane;
      tile_index < ${ACE_OPT_0007_VAE_K1_CONV1D_TILE_CHANNELS * ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK}u;
      tile_index += ${ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE}u
    ) {
      let tile_output_channel =
        tile_index / ${ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK}u;
      let chunk_channel =
        tile_index % ${ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK}u;
      let input_channel = chunk_first_channel + chunk_channel;
      let weight_output_channel =
        group.y * ${ACE_OPT_0007_VAE_K1_CONV1D_TILE_CHANNELS}u +
        tile_output_channel;
      var value = 0.0;
      if (
        weight_output_channel < OUTPUT_CHANNELS &&
        input_channel < INPUT_CHANNELS
      ) {
        // Native [output_channel, kernel=0, input_channel] layout.
        value = weight[
          weight_output_channel * INPUT_CHANNELS + input_channel
        ];
      }
      weight_tile[
        tile_output_channel * ${ACE_OPT_0007_VAE_K1_CONV1D_WEIGHT_TILE_STRIDE}u +
        chunk_channel
      ] = value;
    }
    workgroupBarrier();

    if (output_active) {
      let chunk_channel_count = min(
        ${ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK}u,
        INPUT_CHANNELS - chunk_first_channel
      );
      let weight_base =
        local.y * ${ACE_OPT_0007_VAE_K1_CONV1D_WEIGHT_TILE_STRIDE}u;
      for (
        var chunk_channel = 0u;
        chunk_channel < chunk_channel_count;
        chunk_channel += 1u
      ) {
        sum = sum + input_tile[
          chunk_channel * ${ACE_OPT_0007_VAE_K1_CONV1D_INPUT_TILE_STRIDE}u +
          local.x
        ] * weight_tile[weight_base + chunk_channel];
      }
    }
    workgroupBarrier();
  }

  if (output_active) {
    let output_row = batch * OUTPUT_FRAMES + output_time;
    output[output_row * OUTPUT_CHANNELS + output_channel] = sum;
  }
}
`;
}

async function compileAceOpt0007VaeK1Conv1d(
  device: GPUDevice,
  plan: AceOpt0007VaeK1Conv1dPlan,
  hasBias: boolean,
): Promise<CompiledAceOpt0007VaeK1Conv1d> {
  const label = `ace-opt-0007-vae-k1-conv1d-${convKey(plan, hasBias)}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0007VaeK1Conv1dWgsl(plan, hasBias),
  });
  const compilation = await module.getCompilationInfo();
  const shaderErrors = compilation.messages.filter(
    ({ type }) => type === "error",
  );
  if (shaderErrors.length > 0) {
    throw new Error(
      `OPT-0007 VAE K1 Conv1D WGSL failed: ${shaderErrors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  const bindingCount = hasBias ? 5 : 4;
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: Array.from({ length: bindingCount }, (_, binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: binding === bindingCount - 1
        ? {
            type: "uniform" as const,
            hasDynamicOffset: true,
            minBindingSize: OUTPUT_RANGE_PARAMETER_BYTES,
          }
        : binding === bindingCount - 2
        ? { type: "storage" as const }
        : { type: "read-only-storage" as const },
    })),
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
  const reportedAlignment = device.limits.minUniformBufferOffsetAlignment;
  const outputRangeParameterStride = Math.max(
    MINIMUM_UNIFORM_STRIDE,
    reportedAlignment,
  );
  if (
    !Number.isSafeInteger(reportedAlignment) ||
    reportedAlignment <= 0 ||
    !Number.isSafeInteger(outputRangeParameterStride) ||
    outputRangeParameterStride % reportedAlignment !== 0 ||
    outputRangeParameterStride % Uint32Array.BYTES_PER_ELEMENT !== 0
  ) {
    throw new Error("OPT-0007 device reported an invalid uniform alignment");
  }
  const parameterBytes = checkedProduct([
    Math.max(1, plan.outputRangeCount),
    outputRangeParameterStride,
  ], "range parameter bytes");
  const maximumDynamicOffset =
    (Math.max(1, plan.outputRangeCount) - 1) * outputRangeParameterStride;
  const maximumBufferSize = Number(device.limits.maxBufferSize);
  if (
    !Number.isSafeInteger(maximumBufferSize) ||
    maximumBufferSize < parameterBytes ||
    maximumDynamicOffset > MAX_WGSL_U32
  ) {
    throw new RangeError(
      "OPT-0007 VAE K1 Conv1D range controls exceed the device buffer limit",
    );
  }
  const allocated = await createAceScopedBuffers(
    device,
    [{
      label: `${label}-output-range-parameters`,
      size: parameterBytes,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    }],
    `${label} output range parameters`,
  );
  const outputRangeParameters = allocated[0];
  if (outputRangeParameters === undefined) {
    throw new Error(`${label} output range allocation returned no buffer`);
  }
  try {
    const mapped = outputRangeParameters.getMappedRange();
    for (let index = 0; index < plan.outputRanges.length; index += 1) {
      const words = new Uint32Array(
        mapped,
        index * outputRangeParameterStride,
        OUTPUT_RANGE_PARAMETER_BYTES / Uint32Array.BYTES_PER_ELEMENT,
      );
      const range = plan.outputRanges[index]!;
      words[0] = range.firstOutputRow;
      words[1] = range.outputRowCount;
    }
    outputRangeParameters.unmap();
    return Object.freeze({
      pipeline,
      bindGroupLayout,
      outputRangeParameters,
      outputRangeParameterStride,
      destroy(): void {
        outputRangeParameters.destroy();
      },
    });
  } catch (error) {
    outputRangeParameters.destroy();
    throw error;
  }
}

function convKey(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
): string {
  return [
    shape.batch,
    shape.inputFrames,
    shape.inputChannels,
    shape.outputChannels,
    shape.kernelSize,
    shape.stride,
    shape.dilation,
    shape.padding,
    hasBias ? "bias" : "no-bias",
  ].join("x");
}

function exactBinding(
  binding: GPUBufferBinding,
  requiredBytes: number,
): GPUBufferBinding {
  return {
    buffer: binding.buffer,
    offset: binding.offset ?? 0,
    size: requiredBytes,
  };
}

function checkedProduct(values: readonly number[], label: string): number {
  let product = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(
        `OPT-0007 VAE K1 Conv1D ${label} operand is not a non-negative safe integer`,
      );
    }
    product *= value;
    if (!Number.isSafeInteger(product)) {
      throw new RangeError(
        `OPT-0007 VAE K1 Conv1D ${label} is not a safe integer`,
      );
    }
  }
  return product;
}

function checkedSum(left: number, right: number, label: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw new RangeError(
      `OPT-0007 VAE K1 Conv1D ${label} is not a safe integer`,
    );
  }
  return sum;
}

function checkedBytes(elements: number, label: string): number {
  return checkedProduct([elements, FLOAT32_BYTES], `${label} bytes`);
}

function requireWgslIndexable(value: number, label: string): void {
  if (value > MAX_WGSL_U32) {
    throw new RangeError(
      `OPT-0007 VAE K1 Conv1D ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}
