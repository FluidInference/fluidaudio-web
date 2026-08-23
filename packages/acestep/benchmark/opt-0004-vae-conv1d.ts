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

export const ACE_OPT_0004_VAE_CONV1D_KERNEL_SIZE = 7;
export const ACE_OPT_0004_VAE_CONV1D_TILE_FRAMES = 16;
export const ACE_OPT_0004_VAE_CONV1D_TILE_CHANNELS = 8;
export const ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_FRAMES = 22;
/** One unused time slot avoids a bank-aligned shared-input row stride. */
export const ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_STRIDE = 23;
export const ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_X = 16;
export const ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_Y = 8;
export const ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE = 128;

const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const MAX_DISPATCH_DIMENSION = 65_535;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_PARAMETER_BYTES = 16;
const MINIMUM_UNIFORM_STRIDE = 256;

export interface AceOpt0004VaeConv1dOutputRange {
  readonly batch: number;
  readonly firstOutputTime: number;
  /** Global NLC row, before multiplying by the complete output-channel width. */
  readonly firstOutputRow: number;
  /** Complete-channel rows; ranges never split or cross a batch. */
  readonly outputRowCount: number;
  readonly firstOutput: number;
  readonly outputCount: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
  readonly workgroupCount: number;
  /** Conservative full K7 contraction work, including padded edge taps. */
  readonly multiplyAdds: number;
}

export interface AceOpt0004VaeConv1dPlan extends AceVaeConv1dShape {
  readonly outputFrames: number;
  readonly inputElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly tileFrames: typeof ACE_OPT_0004_VAE_CONV1D_TILE_FRAMES;
  readonly tileChannels: typeof ACE_OPT_0004_VAE_CONV1D_TILE_CHANNELS;
  readonly inputTileFrames: typeof ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_FRAMES;
  readonly inputTileStride: typeof ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_STRIDE;
  readonly weightTileStride: number;
  readonly workgroupSizeX: typeof ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_X;
  readonly workgroupSizeY: typeof ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_Y;
  readonly workgroupSize: typeof ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE;
  readonly inputTileElements: number;
  readonly weightTileElements: number;
  readonly inputTileBytes: number;
  readonly weightTileBytes: number;
  readonly workgroupStorageBytes: number;
  readonly outputRangeCount: number;
  readonly outputRanges: readonly AceOpt0004VaeConv1dOutputRange[];
}

export interface AceOpt0004VaeConv1dDispatch {
  readonly label: string;
  readonly plan: AceOpt0004VaeConv1dPlan;
  readonly rangeCount: number;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledAceOpt0004VaeConv1d {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly outputRangeParameters: GPUBuffer;
  readonly outputRangeParameterStride: number;
  destroy(): void;
}

/**
 * Benchmark-only exact-loop K7 Conv1D candidate for OPT-0004.
 *
 * One WG(16,8) owns 16 output frames by 8 output channels. It stages the
 * 22-frame input halo once and the current kernel tap's 8-channel weight tile
 * before visiting input channels in the converter/source order.
 */
export class AceOpt0004VaeConv1dKernel {
  private readonly compiled = new Map<
    string,
    Promise<CompiledAceOpt0004VaeConv1d>
  >();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(device: GPUDevice): AceOpt0004VaeConv1dKernel {
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX <
        ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_X ||
      device.limits.maxComputeWorkgroupSizeY <
        ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_Y
    ) {
      throw new Error(
        "OPT-0004 VAE Conv1D requires a 16x8 (128-lane) compute workgroup",
      );
    }
    return new AceOpt0004VaeConv1dKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceVaeConvBindings,
  ): Promise<AceOpt0004VaeConv1dDispatch> {
    this.requireLive();
    const plan = planAceOpt0004VaeConv1d(shape);
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

    const compiled = await this.pipelineFor(
      plan,
      bindings.bias !== undefined,
    );
    if (this.destroyed) {
      throw new Error(
        "OPT-0004 VAE Conv1D kernel was destroyed while compiling",
      );
    }
    const rangeBinding = bindings.bias === undefined ? 3 : 4;
    const bindGroups = plan.outputRanges.map((_, rangeIndex) =>
      this.device.createBindGroup({
        label: `${label}-opt-0004-range-${rangeIndex}-bindings`,
        layout: compiled.bindGroupLayout,
        entries: [
          { binding: 0, resource: bindings.input },
          { binding: 1, resource: bindings.weight },
          ...(bindings.bias === undefined
            ? [{ binding: 2, resource: bindings.output }]
            : [
                { binding: 2, resource: bindings.bias },
                { binding: 3, resource: bindings.output },
              ]),
          {
            binding: rangeBinding,
            resource: {
              buffer: compiled.outputRangeParameters,
              offset: rangeIndex * compiled.outputRangeParameterStride,
              size: OUTPUT_RANGE_PARAMETER_BYTES,
            },
          },
        ],
      }),
    );

    return Object.freeze({
      label,
      plan,
      rangeCount: plan.outputRangeCount,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        const range = plan.outputRanges[rangeIndex];
        if (range === undefined) {
          throw new RangeError(
            `${label} OPT-0004 range ${rangeIndex} is outside [0, ${plan.outputRangeCount})`,
          );
        }
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroups[rangeIndex]!);
        pass.dispatchWorkgroups(range.workgroupsX, range.workgroupsY, 1);
      },
      encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(compiled.pipeline);
        for (let index = 0; index < plan.outputRanges.length; index += 1) {
          const range = plan.outputRanges[index]!;
          pass.setBindGroup(0, bindGroups[index]!);
          pass.dispatchWorkgroups(range.workgroupsX, range.workgroupsY, 1);
        }
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const compiled of this.compiled.values()) {
      void compiled.then(
        (resources) => resources.destroy(),
        () => undefined,
      );
    }
    this.compiled.clear();
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0004 VAE Conv1D kernel was destroyed");
    }
  }

  private requireDeviceLimits(plan: AceOpt0004VaeConv1dPlan): void {
    if (
      plan.workgroupStorageBytes >
        this.device.limits.maxComputeWorkgroupStorageSize
    ) {
      throw new RangeError(
        `OPT-0004 VAE Conv1D requires ${plan.workgroupStorageBytes} ` +
          "workgroup-storage bytes",
      );
    }
    const maximumDispatch = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (plan.outputRanges.some(
      ({ workgroupsX, workgroupsY }) =>
        workgroupsX > maximumDispatch || workgroupsY > maximumDispatch,
    )) {
      throw new RangeError(
        "OPT-0004 VAE Conv1D exceeds the device dispatch dimension",
      );
    }
    const maximumBinding = Number(this.device.limits.maxStorageBufferBindingSize);
    for (const [name, elements] of [
      ["input", plan.inputElements],
      ["weight", plan.weightElements],
      ["output", plan.outputElements],
      ["bias", plan.outputChannels],
    ] as const) {
      if (checkedBytes(elements, name) > maximumBinding) {
        throw new RangeError(
          `OPT-0004 VAE Conv1D ${name} exceeds the device storage binding limit`,
        );
      }
    }
  }

  private pipelineFor(
    plan: AceOpt0004VaeConv1dPlan,
    hasBias: boolean,
  ): Promise<CompiledAceOpt0004VaeConv1d> {
    const key = convKey(plan, hasBias);
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceOpt0004VaeConv1d(this.device, plan, hasBias);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

export function planAceOpt0004VaeConv1d(
  shape: AceVaeConv1dShape,
): AceOpt0004VaeConv1dPlan {
  const portable = planAceVaeConv1d(shape);
  if (shape.kernelSize !== ACE_OPT_0004_VAE_CONV1D_KERNEL_SIZE) {
    throw new RangeError("OPT-0004 VAE Conv1D requires kernel size 7");
  }
  if (shape.stride !== 1 || shape.dilation !== 1) {
    throw new RangeError(
      "OPT-0004 VAE Conv1D requires stride 1 and dilation 1",
    );
  }
  for (const [name, elements] of [
    ["input", portable.inputElements],
    ["weight", portable.weightElements],
    ["output", portable.outputElements],
  ] as const) {
    requireWgslIndexable(elements, name);
  }
  const outputRows = checkedProduct(
    [shape.batch, portable.outputFrames],
    "output rows",
  );
  requireWgslIndexable(outputRows, "output rows");
  requireWgslIndexable(shape.padding, "padding");
  const maximumPaddedTime = portable.outputFrames - 1 + shape.kernelSize - 1;
  requireWgslIndexable(maximumPaddedTime, "padded time");

  const inputTileElements = checkedProduct(
    [shape.inputChannels, ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_STRIDE],
    "input tile",
  );
  const weightTileStride = shape.inputChannels + 1;
  if (!Number.isSafeInteger(weightTileStride)) {
    throw new RangeError(
      "OPT-0004 VAE Conv1D weight tile stride is not a safe integer",
    );
  }
  const weightTileElements = checkedProduct(
    [ACE_OPT_0004_VAE_CONV1D_TILE_CHANNELS, weightTileStride],
    "weight tile",
  );
  const inputTileBytes = checkedBytes(inputTileElements, "input tile");
  const weightTileBytes = checkedBytes(weightTileElements, "weight tile");
  const workgroupStorageBytes = checkedSum(
    inputTileBytes,
    weightTileBytes,
    "workgroup storage",
  );
  const workgroupsY = Math.ceil(
    shape.outputChannels / ACE_OPT_0004_VAE_CONV1D_TILE_CHANNELS,
  );
  if (workgroupsY > MAX_DISPATCH_DIMENSION) {
    throw new RangeError(
      "OPT-0004 VAE Conv1D output channels exceed the 2D dispatch domain",
    );
  }
  const multiplyAddsPerRow = checkedProduct(
    [
      shape.outputChannels,
      ACE_OPT_0004_VAE_CONV1D_KERNEL_SIZE,
      shape.inputChannels,
    ],
    "multiply-adds per output row",
  );
  const maximumRowsPerRange = Math.min(
    Math.floor(
      ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY.maximumOutputElements /
        shape.outputChannels,
    ),
    Math.floor(
      ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY
        .maximumConvolutionMultiplyAccumulates / multiplyAddsPerRow,
    ),
    MAX_DISPATCH_DIMENSION * ACE_OPT_0004_VAE_CONV1D_TILE_FRAMES,
  );
  if (maximumRowsPerRange < 1) {
    throw new RangeError(
      "OPT-0004 VAE Conv1D cannot fit one complete-channel output row in a bounded range",
    );
  }

  const outputRanges: AceOpt0004VaeConv1dOutputRange[] = [];
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
      const firstOutputRow =
        batch * portable.outputFrames + firstOutputTime;
      const outputCount = checkedProduct(
        [outputRowCount, shape.outputChannels],
        "range output count",
      );
      const workgroupsX = Math.ceil(
        outputRowCount / ACE_OPT_0004_VAE_CONV1D_TILE_FRAMES,
      );
      const lastStagedTime =
        firstOutputTime +
        (workgroupsX - 1) * ACE_OPT_0004_VAE_CONV1D_TILE_FRAMES +
        ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_FRAMES - 1;
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
    throw new Error("OPT-0004 VAE Conv1D range planner lost outputs");
  }

  return Object.freeze({
    batch: shape.batch,
    inputFrames: shape.inputFrames,
    inputChannels: shape.inputChannels,
    outputChannels: shape.outputChannels,
    kernelSize: shape.kernelSize,
    stride: shape.stride,
    dilation: shape.dilation,
    padding: shape.padding,
    outputFrames: portable.outputFrames,
    inputElements: portable.inputElements,
    weightElements: portable.weightElements,
    outputElements: portable.outputElements,
    tileFrames: ACE_OPT_0004_VAE_CONV1D_TILE_FRAMES,
    tileChannels: ACE_OPT_0004_VAE_CONV1D_TILE_CHANNELS,
    inputTileFrames: ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_FRAMES,
    inputTileStride: ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_STRIDE,
    weightTileStride,
    workgroupSizeX: ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_X,
    workgroupSizeY: ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_Y,
    workgroupSize: ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE,
    inputTileElements,
    weightTileElements,
    inputTileBytes,
    weightTileBytes,
    workgroupStorageBytes,
    outputRangeCount: outputRanges.length,
    outputRanges: Object.freeze(outputRanges),
  });
}

export function aceOpt0004VaeConv1dWgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
): string {
  const plan = planAceOpt0004VaeConv1d(shape);
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
const PADDING: u32 = ${plan.padding}u;

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
  ${ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_X},
  ${ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_Y},
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
    range_first_time + group.x * ${ACE_OPT_0004_VAE_CONV1D_TILE_FRAMES}u;

  // Stage the complete 16-output K7 halo. Padded slots are initialized but
  // their arithmetic remains skipped below, matching the source primitive.
  for (
    var tile_index = lane;
    tile_index < ${ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_FRAMES * plan.inputChannels}u;
    tile_index += ${ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE}u
  ) {
    let tile_time = tile_index / INPUT_CHANNELS;
    let input_channel = tile_index % INPUT_CHANNELS;
    let padded_time = tile_first_time + tile_time;
    var value = 0.0;
    if (padded_time >= PADDING) {
      let input_time = padded_time - PADDING;
      if (input_time < INPUT_FRAMES) {
        value = input[
          (batch * INPUT_FRAMES + input_time) * INPUT_CHANNELS +
          input_channel
        ];
      }
    }
    input_tile[
      input_channel * ${ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_STRIDE}u + tile_time
    ] = value;
  }
  workgroupBarrier();

  let output_time = tile_first_time + local.x;
  let output_channel =
    group.y * ${ACE_OPT_0004_VAE_CONV1D_TILE_CHANNELS}u + local.y;
  let range_end_time = range_first_time + output_range.output_row_count;
  let output_active =
    output_time < range_end_time && output_channel < OUTPUT_CHANNELS;
  var sum = 0.0;
  if (output_active) {
    sum = ${initialSum};
  }

  // Every lane reaches both barriers for every K. Active outputs retain the
  // source order: bias/zero, then kernel 0..6, then input channel increasing.
  for (
    var kernel = 0u;
    kernel < ${ACE_OPT_0004_VAE_CONV1D_KERNEL_SIZE}u;
    kernel += 1u
  ) {
    for (
      var tile_index = lane;
      tile_index < ${ACE_OPT_0004_VAE_CONV1D_TILE_CHANNELS * plan.inputChannels}u;
      tile_index += ${ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE}u
    ) {
      let tile_output_channel = tile_index / INPUT_CHANNELS;
      let input_channel = tile_index % INPUT_CHANNELS;
      let weight_output_channel =
        group.y * ${ACE_OPT_0004_VAE_CONV1D_TILE_CHANNELS}u +
        tile_output_channel;
      var value = 0.0;
      if (weight_output_channel < OUTPUT_CHANNELS) {
        value = weight[
          (weight_output_channel * ${ACE_OPT_0004_VAE_CONV1D_KERNEL_SIZE}u +
            kernel) * INPUT_CHANNELS + input_channel
        ];
      }
      weight_tile[
        tile_output_channel * ${plan.weightTileStride}u + input_channel
      ] = value;
    }
    workgroupBarrier();

    if (output_active) {
      let padded_time = output_time + kernel;
      if (padded_time >= PADDING) {
        let input_time = padded_time - PADDING;
        if (input_time < INPUT_FRAMES) {
          let input_time_in_tile = local.x + kernel;
          let weight_base = local.y * ${plan.weightTileStride}u;
          for (
            var input_channel = 0u;
            input_channel < INPUT_CHANNELS;
            input_channel += 1u
          ) {
            sum = sum + input_tile[
              input_channel * ${ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_STRIDE}u +
              input_time_in_tile
            ] *
              weight_tile[weight_base + input_channel];
          }
        }
      }
    }
    workgroupBarrier();
  }

  if (output_active) {
    let output_row =
      batch * OUTPUT_FRAMES + output_time;
    output[output_row * OUTPUT_CHANNELS + output_channel] = sum;
  }
}
`;
}

async function compileAceOpt0004VaeConv1d(
  device: GPUDevice,
  plan: AceOpt0004VaeConv1dPlan,
  hasBias: boolean,
): Promise<CompiledAceOpt0004VaeConv1d> {
  const label = `ace-opt-0004-vae-conv1d-${convKey(plan, hasBias)}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0004VaeConv1dWgsl(plan, hasBias),
  });
  const compilation = await module.getCompilationInfo();
  const shaderErrors = compilation.messages.filter(
    (message) => message.type === "error",
  );
  if (shaderErrors.length > 0) {
    throw new Error(
      `OPT-0004 VAE Conv1D WGSL failed: ${shaderErrors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const outputRangeParameterStride = Math.max(
    MINIMUM_UNIFORM_STRIDE,
    device.limits.minUniformBufferOffsetAlignment,
  );
  if (
    !Number.isSafeInteger(outputRangeParameterStride) ||
    outputRangeParameterStride <= 0
  ) {
    throw new Error("OPT-0004 device reported an invalid uniform alignment");
  }
  const parameterBytes = checkedProduct(
    [Math.max(1, plan.outputRangeCount), outputRangeParameterStride],
    "range parameter bytes",
  );
  if (parameterBytes > Number(device.limits.maxBufferSize)) {
    throw new RangeError(
      "OPT-0004 VAE Conv1D range controls exceed the device buffer limit",
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
      bindGroupLayout: pipeline.getBindGroupLayout(0),
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
        `OPT-0004 VAE Conv1D ${label} operand is not a non-negative safe integer`,
      );
    }
    product *= value;
    if (!Number.isSafeInteger(product)) {
      throw new RangeError(
        `OPT-0004 VAE Conv1D ${label} is not a safe integer`,
      );
    }
  }
  return product;
}

function checkedSum(left: number, right: number, label: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw new RangeError(
      `OPT-0004 VAE Conv1D ${label} is not a safe integer`,
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
      `OPT-0004 VAE Conv1D ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}
