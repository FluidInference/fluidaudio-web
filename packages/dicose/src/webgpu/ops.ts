import type { GpuWeightTensor } from "../model/package.js";
import { createF16Tensor, destroyTensors, type GpuTensor } from "./tensor.js";

const PARAMETER_BYTES = 256;
const PARAMETER_SLOTS = 32_768;

export type Activation = "none" | "gelu" | "tanh";
export type AttentionKernel = "query8" | "q64" | "flash";
export type PackedAccumulation = "exact" | "k2" | "k4";

export interface AttentionGeometry {
  readonly sequences: number;
  readonly tokens: number;
  readonly strided?: boolean;
}

export interface AttentionDescriptor extends AttentionGeometry {
  readonly kernel?: AttentionKernel;
  readonly gates?: GpuTensor;
  readonly rotatedKeys?: boolean;
}

export interface LinearDescriptor {
  readonly rows: number;
  readonly inner: number;
  readonly columns: number;
  readonly activation?: Activation;
  /** Output ownership width; may split converter-native N256 tiles in half. */
  readonly outputTileColumns?: 128 | 256;
  /** Load and source-unroll four adjacent K operands while retaining FP32 FMA order. */
  readonly vectorizeK?: boolean;
  /** Packed-owner reduction arithmetic; approximate modes retain FP32 running state. */
  readonly accumulation?: PackedAccumulation;
  /** Add after the projection's f16 rounding, preserving the former add pass. */
  readonly residual?: GpuTensor;
  /** Rotate the K slice of a packed 3×512 QKV projection after f16 rounding. */
  readonly rotaryKeys?: AttentionGeometry;
}

interface DynamicBindGroup {
  readonly bindGroup: GPUBindGroup;
  readonly parameterOffset: number;
}

/**
 * The small primitive set used by both BS-RoFormer graphs. All activation and
 * weight storage is f16; reductions and attention softmax use f32.
 */
export class GpuOps {
  private readonly parameters: GPUBuffer;
  private parameterCursor = 0;
  private readonly zero: GPUBuffer;
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private readonly layouts = new Map<string, GPUBindGroupLayout>();
  private readonly rotaryTables: GpuTensor[] = [];
  private rotaryTable: GpuTensor | undefined;
  private rotaryTableTokens = 0;
  private destroyed = false;

  constructor(
    readonly device: GPUDevice,
    private readonly defaultAttentionKernel: AttentionKernel = "flash",
    private readonly defaultPackedAccumulation: PackedAccumulation = "exact",
  ) {
    this.parameters = device.createBuffer({
      label: "dicose-dynamic-parameters",
      size: PARAMETER_BYTES * PARAMETER_SLOTS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.zero = device.createBuffer({
      label: "dicose-zero-bias",
      size: 16_384,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  beginGraph(): void {
    this.requireAlive();
    this.parameterCursor = 0;
  }

  linear(
    pass: GPUComputePassEncoder,
    input: GpuTensor,
    weight: GpuWeightTensor,
    bias: GpuWeightTensor | undefined,
    output: GpuTensor,
    descriptor: LinearDescriptor,
  ): void {
    const activation = descriptor.activation ?? "none";
    const packedTileColumns = packedLinearTileColumns(weight.layout);
    if (packedTileColumns !== undefined) {
      this.packedLinear(
        pass,
        input,
        weight,
        bias,
        output,
        descriptor,
        activation,
        packedTileColumns,
      );
      return;
    }
    if (descriptor.outputTileColumns !== undefined) {
      throw new Error(`Packed output ownership requires a packed weight: ${weight.name}`);
    }
    if (descriptor.accumulation !== undefined && descriptor.accumulation !== "exact") {
      throw new Error(`Approximate accumulation requires a packed weight: ${weight.name}`);
    }
    if (descriptor.residual !== undefined || descriptor.rotaryKeys !== undefined) {
      throw new Error(`Fused packed-linear post-op requires a packed weight: ${weight.name}`);
    }
    const pipeline = this.pipeline("linear", LINEAR_WGSL);
    const params = this.parametersFor([
      descriptor.rows,
      descriptor.inner,
      descriptor.columns,
      activationCode(activation),
    ]);
    const bindings = this.bind("linear", [
      storage(input.buffer, input.byteLength),
      storage(weight.buffer, weight.byteLength, weight.offset),
      storage(bias?.buffer ?? this.zero, bias?.byteLength ?? 16_384, bias?.offset ?? 0),
      storage(output.buffer, output.byteLength),
      dynamicUniform(this.parameters),
    ]);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings.bindGroup, [params]);
    pass.dispatchWorkgroups(
      Math.ceil(descriptor.columns / 16),
      Math.ceil(descriptor.rows / 16),
    );
  }

  private packedLinear(
    pass: GPUComputePassEncoder,
    input: GpuTensor,
    weight: GpuWeightTensor,
    bias: GpuWeightTensor | undefined,
    output: GpuTensor,
    descriptor: LinearDescriptor,
    activation: Activation,
    tileColumns: 128 | 256,
  ): void {
    if (
      descriptor.inner % 32 !== 0 || descriptor.columns % tileColumns !== 0 ||
      weight.shape.length !== 2 || weight.shape[0] !== descriptor.inner ||
      weight.shape[1] !== descriptor.columns
    ) {
      throw new RangeError(
        `Invalid packed linear ${descriptor.rows}x${descriptor.inner}x${descriptor.columns} for ${weight.name}`,
      );
    }
    const rotaryKeys = descriptor.rotaryKeys;
    const optimizeFullRowTiles = descriptor.rows >= 32;
    const outputTileColumns = descriptor.outputTileColumns ?? (
      optimizeFullRowTiles ? 128 : tileColumns
    );
    const vectorizeK = descriptor.vectorizeK ?? (
      optimizeFullRowTiles && outputTileColumns === 128
    );
    if (outputTileColumns > tileColumns || tileColumns % outputTileColumns !== 0) {
      throw new RangeError(`Invalid packed output ownership for ${weight.name}`);
    }
    const requestedAccumulation = descriptor.accumulation ?? this.defaultPackedAccumulation;
    const approximateEligible = outputTileColumns === 128 && vectorizeK;
    if (
      descriptor.accumulation !== undefined && requestedAccumulation !== "exact" &&
      !approximateEligible
    ) {
      throw new RangeError(
        `Packed ${requestedAccumulation} accumulation requires owner128/K4 loads for ${weight.name}`,
      );
    }
    // A graph-wide approximate default deliberately leaves incompatible small-row
    // and N256 owners exact. Explicit incompatible requests remain programmer errors.
    const accumulation = approximateEligible ? requestedAccumulation : "exact";
    if (rotaryKeys !== undefined) {
      if (
        descriptor.residual !== undefined || activation !== "none" ||
        descriptor.columns !== 1_536 || tileColumns !== 256 ||
        !Number.isSafeInteger(rotaryKeys.sequences) || rotaryKeys.sequences <= 0 ||
        !Number.isSafeInteger(rotaryKeys.tokens) || rotaryKeys.tokens <= 0 ||
        rotaryKeys.sequences * rotaryKeys.tokens !== descriptor.rows
      ) {
        throw new RangeError(`Invalid fused K rotation for ${weight.name}`);
      }
    }
    const pipelineKey = `linear-${tileColumns}x32-owner${outputTileColumns}-${vectorizeK ? "loadk4" : "loadk1"}-${accumulation}-${descriptor.inner}x${descriptor.columns}-${activation}`;
    const params = this.parametersFor([
      descriptor.rows,
      rotaryKeys?.sequences ?? 0,
      rotaryKeys?.tokens ?? 0,
      rotaryKeys?.strided === true ? 1 : 0,
    ]);
    const commonBindings = [
      storage(input.buffer, input.byteLength),
      storage(weight.buffer, weight.byteLength, weight.offset),
      storage(bias?.buffer ?? this.zero, bias?.byteLength ?? 16_384, bias?.offset ?? 0),
      storage(output.buffer, output.byteLength),
    ];
    if (rotaryKeys !== undefined) {
      const rotaryTable = this.prepareRotaryTable(pass, rotaryKeys.tokens);
      const normalBindings = this.bind("linear-packed", [
        ...commonBindings,
        dynamicUniform(this.parameters),
      ]);
      const rotaryBindings = this.bind("linear-packed-rotary-keys", [
        ...commonBindings,
        storage(rotaryTable.buffer, rotaryTable.byteLength),
        dynamicUniformAt(this.parameters, 5),
      ], 5);
      const rowWorkgroups = Math.ceil(descriptor.rows / 32);
      const sliceWorkgroups = 512 / outputTileColumns;
      const dispatch = (
        pipeline: GPUComputePipeline,
        bindings: DynamicBindGroup,
      ): void => {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindings.bindGroup, [params]);
        pass.dispatchWorkgroups(sliceWorkgroups, rowWorkgroups);
      };
      dispatch(this.pipeline(
        pipelineKey,
        packedLinearWgsl(descriptor.inner, descriptor.columns, tileColumns, outputTileColumns, vectorizeK, accumulation, activation, false, false),
        "linear-packed",
      ), normalBindings);
      dispatch(this.pipeline(
        `${pipelineKey}-rotary-keys-offset-2`,
        packedLinearWgsl(descriptor.inner, descriptor.columns, tileColumns, outputTileColumns, vectorizeK, accumulation, activation, false, true, sliceWorkgroups),
        "linear-packed-rotary-keys",
      ), rotaryBindings);
      dispatch(this.pipeline(
        `${pipelineKey}-offset-4`,
        packedLinearWgsl(descriptor.inner, descriptor.columns, tileColumns, outputTileColumns, vectorizeK, accumulation, activation, false, false, sliceWorkgroups * 2),
        "linear-packed",
      ), normalBindings);
      return;
    }
    const hasResidual = descriptor.residual !== undefined;
    const packedLayout = hasResidual ? "linear-packed-residual" : "linear-packed";
    const pipeline = this.pipeline(
      `${pipelineKey}${hasResidual ? "-residual" : ""}`,
      packedLinearWgsl(
        descriptor.inner,
        descriptor.columns,
        tileColumns,
        outputTileColumns,
        vectorizeK,
        accumulation,
        activation,
        hasResidual,
        false,
      ),
      packedLayout,
    );
    const bindings = this.bind(packedLayout, [
      ...commonBindings,
      ...(descriptor.residual === undefined ? [] : [
        storage(descriptor.residual.buffer, descriptor.residual.byteLength),
      ]),
      hasResidual ? dynamicUniformAt(this.parameters, 5) : dynamicUniform(this.parameters),
    ], hasResidual ? 5 : 4);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings.bindGroup, [params]);
    pass.dispatchWorkgroups(
      descriptor.columns / outputTileColumns,
      Math.ceil(descriptor.rows / 32),
    );
  }

  rmsNorm(
    pass: GPUComputePassEncoder,
    input: GpuTensor,
    gamma: GpuWeightTensor,
    output: GpuTensor,
    rows: number,
    columns: number,
    scaleShift?: GpuTensor,
    owner: "auto" | "row1" = "auto",
    workgroupWidthLimit = this.device.limits.maxComputeWorkgroupsPerDimension,
  ): void {
    if (
      !Number.isSafeInteger(workgroupWidthLimit) ||
      workgroupWidthLimit <= 0 ||
      workgroupWidthLimit > this.device.limits.maxComputeWorkgroupsPerDimension
    ) {
      throw new RangeError(`Invalid RMSNorm workgroup-width limit ${workgroupWidthLimit}`);
    }
    const rowsPerWorkgroup = owner === "auto" ? 8 : 1;
    const pipeline = this.pipeline(
      owner === "auto" ? "rmsnorm-rows8" : "rmsnorm-row1",
      owner === "auto" ? RMSNORM_ROWS8_WGSL : RMSNORM_WGSL,
      "rmsnorm",
    );
    const workgroups = Math.ceil(rows / rowsPerWorkgroup);
    const workgroupWidth = Math.min(workgroups, workgroupWidthLimit);
    const workgroupHeight = Math.ceil(workgroups / workgroupWidth);
    if (workgroupHeight > this.device.limits.maxComputeWorkgroupsPerDimension) {
      throw new RangeError(`DiCoSe RMSNorm dispatch exceeds the device workgroup grid for ${rows} rows`);
    }
    const params = this.parametersFor([rows, columns, scaleShift === undefined ? 0 : 1, workgroupWidth]);
    const bindings = this.bind("rmsnorm", [
      storage(input.buffer, input.byteLength),
      storage(gamma.buffer, gamma.byteLength, gamma.offset),
      storage(scaleShift?.buffer ?? this.zero, scaleShift?.byteLength ?? 16_384),
      storage(output.buffer, output.byteLength),
      dynamicUniform(this.parameters),
    ]);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings.bindGroup, [params]);
    pass.dispatchWorkgroups(workgroupWidth, workgroupHeight);
  }

  add(
    pass: GPUComputePassEncoder,
    source: GpuTensor,
    destination: GpuTensor,
    elements: number,
  ): void {
    this.elementwise(pass, "add", ADD_WGSL, [
      storage(source.buffer, source.byteLength),
      storage(destination.buffer, destination.byteLength),
      dynamicUniform(this.parameters),
    ], [elements, 0, 0, 0]);
  }

  copy(
    pass: GPUComputePassEncoder,
    source: GpuTensor,
    destination: GpuTensor,
    elements: number,
  ): void {
    this.elementwise(pass, "copy", COPY_WGSL, [
      storage(source.buffer, source.byteLength),
      storage(destination.buffer, destination.byteLength),
      dynamicUniform(this.parameters),
    ], [elements, 0, 0, 0]);
  }

  /** Capture evenly spaced f16 words without reading a production tensor back in full. */
  sampleEven(
    pass: GPUComputePassEncoder,
    input: GpuTensor,
    output: GpuTensor,
    elements: number,
    samples: number,
  ): void {
    if (
      !Number.isSafeInteger(elements) || elements <= 0 ||
      !Number.isSafeInteger(samples) || samples <= 0 || samples > elements ||
      output.byteLength !== samples * 2
    ) {
      throw new RangeError("Invalid evenly sampled tensor geometry");
    }
    this.elementwise(pass, "sample-even", SAMPLE_EVEN_WGSL, [
      storage(input.buffer, input.byteLength),
      storage(output.buffer, output.byteLength),
      dynamicUniform(this.parameters),
    ], [elements, samples, 0, 0], Math.ceil(samples / 256));
  }

  transposeTB(
    pass: GPUComputePassEncoder,
    input: GpuTensor,
    output: GpuTensor,
    time: number,
    bands: number,
    dim: number,
  ): void {
    this.elementwise(pass, "transpose-tb", TRANSPOSE_TB_WGSL, [
      storage(input.buffer, input.byteLength),
      storage(output.buffer, output.byteLength),
      dynamicUniform(this.parameters),
    ], [time, bands, dim, 0], Math.ceil(time * bands * dim / 256));
  }

  gatherSlice(
    pass: GPUComputePassEncoder,
    source: GpuTensor,
    output: GpuTensor,
    rows: number,
    sourceWidth: number,
    offset: number,
    width: number,
  ): void {
    this.elementwise(pass, "gather-slice", GATHER_SLICE_WGSL, [
      storage(source.buffer, source.byteLength),
      storage(output.buffer, output.byteLength),
      dynamicUniform(this.parameters),
    ], [rows, sourceWidth, offset, width], Math.ceil(rows * width / 256));
  }

  scatterSlice(
    pass: GPUComputePassEncoder,
    source: GpuTensor,
    output: GpuTensor,
    rows: number,
    destinationWidth: number,
    offset: number,
    width: number,
  ): void {
    this.elementwise(pass, "scatter-slice", SCATTER_SLICE_WGSL, [
      storage(source.buffer, source.byteLength),
      storage(output.buffer, output.byteLength),
      dynamicUniform(this.parameters),
    ], [rows, destinationWidth, offset, width], Math.ceil(rows * width / 256));
  }

  spectralToPixels(
    pass: GPUComputePassEncoder,
    source: GpuTensor,
    output: GpuTensor,
    time: number,
    frequencies: number,
    channels: number,
  ): void {
    this.elementwise(pass, "spectral-to-pixels", SPECTRAL_TO_PIXELS_WGSL, [
      storage(source.buffer, source.byteLength),
      storage(output.buffer, output.byteLength),
      dynamicUniform(this.parameters),
    ], [time, frequencies, channels, 0], Math.ceil(time * frequencies * channels / 256));
  }

  pixelsToSpectral(
    pass: GPUComputePassEncoder,
    source: GpuTensor,
    output: GpuTensor,
    time: number,
    frequencies: number,
    channels: number,
  ): void {
    this.elementwise(pass, "pixels-to-spectral", PIXELS_TO_SPECTRAL_WGSL, [
      storage(source.buffer, source.byteLength),
      storage(output.buffer, output.byteLength),
      dynamicUniform(this.parameters),
    ], [time, frequencies, channels, 0], Math.ceil(time * frequencies * channels / 256));
  }

  attention(
    pass: GPUComputePassEncoder,
    qkv: GpuTensor,
    output: GpuTensor,
    descriptor: AttentionDescriptor,
  ): void {
    const {
      sequences,
      tokens,
      kernel = this.defaultAttentionKernel,
      gates,
      strided = false,
      rotatedKeys = false,
    } = descriptor;
    if (
      !Number.isSafeInteger(sequences) || sequences <= 0 ||
      !Number.isSafeInteger(tokens) || tokens <= 0
    ) {
      throw new RangeError("Invalid attention geometry");
    }
    const grouped = kernel !== "query8";
    if (!grouped && (strided || rotatedKeys)) {
      throw new Error("Strided or pre-rotated-key attention requires a grouped kernel");
    }
    const pipelineName = grouped
      ? `${kernel}${rotatedKeys ? "-rotated-keys" : ""}`
      : "attention";
    const pipeline = this.pipeline(
      pipelineName,
      kernel === "flash"
        ? attentionFlashWgsl(rotatedKeys)
        : kernel === "q64" ? attentionQ64Wgsl(rotatedKeys) : ATTENTION_WGSL,
      grouped ? "attention-grouped" : "attention",
    );
    const parameterOffset = this.parametersFor([
      sequences,
      tokens,
      gates === undefined ? 0 : 1,
      strided ? 1 : 0,
    ]);
    const rotaryTable = grouped ? this.prepareRotaryTable(pass, tokens) : undefined;
    const bindings = this.bind(grouped ? "attention-grouped" : "attention", [
      storage(qkv.buffer, qkv.byteLength),
      storage(output.buffer, output.byteLength),
      ...(rotaryTable === undefined ? [] : [storage(rotaryTable.buffer, rotaryTable.byteLength)]),
      ...(grouped ? [storage(gates?.buffer ?? this.zero, gates?.byteLength ?? 16_384)] : []),
      dynamicUniform(this.parameters),
    ]);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings.bindGroup, [parameterOffset]);
    const queriesPerWorkgroup = grouped ? 64 : 8;
    pass.dispatchWorkgroups(Math.ceil(tokens / queriesPerWorkgroup), 8, sequences);
  }

  applyGates(
    pass: GPUComputePassEncoder,
    context: GpuTensor,
    gates: GpuTensor,
    rows: number,
  ): void {
    this.elementwise(pass, "gates", GATES_WGSL, [
      storage(context.buffer, context.byteLength),
      storage(gates.buffer, gates.byteLength),
      dynamicUniform(this.parameters),
    ], [rows, 0, 0, 0], Math.ceil(rows * 512 / 256));
  }

  complexMultiply(
    pass: GPUComputePassEncoder,
    input: GpuTensor,
    mask: GpuTensor,
    output: GpuTensor,
    complexes: number,
  ): void {
    this.elementwise(pass, "complex-multiply", COMPLEX_MULTIPLY_WGSL, [
      storage(input.buffer, input.byteLength),
      storage(mask.buffer, mask.byteLength),
      storage(output.buffer, output.byteLength),
      dynamicUniform(this.parameters),
    ], [complexes, 0, 0, 0]);
  }

  affine(
    pass: GPUComputePassEncoder,
    modelOutput: GpuTensor,
    noisyInput: GpuTensor,
    output: GpuTensor,
    elements: number,
    cOut: number,
    cSkip: number,
  ): void {
    const params = new ArrayBuffer(16);
    const u32 = new Uint32Array(params);
    const f32 = new Float32Array(params);
    u32[0] = elements;
    f32[1] = cOut;
    f32[2] = cSkip;
    this.elementwise(pass, "affine", AFFINE_WGSL, [
      storage(modelOutput.buffer, modelOutput.byteLength),
      storage(noisyInput.buffer, noisyInput.byteLength),
      storage(output.buffer, output.byteLength),
      dynamicUniform(this.parameters),
    ], new Uint32Array(params));
  }

  conv2d(
    pass: GPUComputePassEncoder,
    input: GpuTensor,
    weight: GpuWeightTensor,
    bias: GpuWeightTensor | undefined,
    output: GpuTensor,
    height: number,
    width: number,
    inChannels: number,
    outChannels: number,
    kernel: 1 | 3,
    owner: "auto" | "generic" = "auto",
  ): void {
    if (owner === "auto" && kernel === 3 && inChannels === 4 && outChannels === 128) {
      const pipeline = this.pipeline("conv3x3-4x128", CONV3X3_4X128_WGSL, "conv2d");
      const rows = height * width;
      const parameterOffset = this.parametersFor([rows, width, height, 0]);
      const bindings = this.bind("conv2d", [
        storage(input.buffer, input.byteLength),
        storage(weight.buffer, weight.byteLength, weight.offset),
        storage(bias?.buffer ?? this.zero, bias?.byteLength ?? 16_384, bias?.offset ?? 0),
        storage(output.buffer, output.byteLength),
        dynamicUniform(this.parameters),
      ]);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindings.bindGroup, [parameterOffset]);
      pass.dispatchWorkgroups(1, Math.ceil(rows / 32));
      return;
    }
    if (owner === "auto" && kernel === 1 && inChannels === 128 && outChannels === 128) {
      const pipeline = this.pipeline("conv1x1-128", CONV1X1_128_WGSL, "conv2d");
      const rows = height * width;
      const parameterOffset = this.parametersFor([rows, 0, 0, 0]);
      const bindings = this.bind("conv2d", [
        storage(input.buffer, input.byteLength),
        storage(weight.buffer, weight.byteLength, weight.offset),
        storage(bias?.buffer ?? this.zero, bias?.byteLength ?? 16_384, bias?.offset ?? 0),
        storage(output.buffer, output.byteLength),
        dynamicUniform(this.parameters),
      ]);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindings.bindGroup, [parameterOffset]);
      pass.dispatchWorkgroups(1, Math.ceil(rows / 32));
      return;
    }
    const pipeline = this.pipeline("conv2d", CONV2D_WGSL);
    const parameterOffset = this.parametersFor([
      height,
      width,
      inChannels,
      outChannels,
      kernel,
      0,
      0,
      0,
    ]);
    const bindings = this.bind("conv2d", [
      storage(input.buffer, input.byteLength),
      storage(weight.buffer, weight.byteLength, weight.offset),
      storage(bias?.buffer ?? this.zero, bias?.byteLength ?? 16_384, bias?.offset ?? 0),
      storage(output.buffer, output.byteLength),
      dynamicUniform(this.parameters),
    ]);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings.bindGroup, [parameterOffset]);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), outChannels);
  }

  geluInPlace(pass: GPUComputePassEncoder, tensor: GpuTensor, elements: number): void {
    this.elementwise(pass, "gelu", GELU_WGSL, [
      storage(tensor.buffer, tensor.byteLength),
      dynamicUniform(this.parameters),
    ], [elements, 0, 0, 0]);
  }

  tanhInPlace(pass: GPUComputePassEncoder, tensor: GpuTensor, elements: number): void {
    this.elementwise(pass, "tanh", TANH_WGSL, [
      storage(tensor.buffer, tensor.byteLength),
      dynamicUniform(this.parameters),
    ], [elements, 0, 0, 0]);
  }

  gluInPlace(
    pass: GPUComputePassEncoder,
    input: GpuTensor,
    output: GpuTensor,
    rows: number,
    columns: number,
  ): void {
    this.elementwise(pass, "glu", GLU_WGSL, [
      storage(input.buffer, input.byteLength),
      storage(output.buffer, output.byteLength),
      dynamicUniform(this.parameters),
    ], [rows, columns, 0, 0], Math.ceil(rows * columns / 256));
  }

  createF16(elements: number, label: string): GpuTensor {
    return createF16Tensor(this.device, elements, label);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.parameters.destroy();
    this.zero.destroy();
    destroyTensors(this.rotaryTables);
    this.pipelines.clear();
    this.layouts.clear();
  }

  /**
   * Materialize the fixed RoPE sin/cos table once on the GPU. The attention
   * graph previously re-evaluated pow/sin/cos for every key and every
   * 32-query block; the longest table for the supplied WAV is only ~304 KiB.
   */
  private prepareRotaryTable(pass: GPUComputePassEncoder, tokens: number): GpuTensor {
    if (this.rotaryTable !== undefined && this.rotaryTableTokens >= tokens) {
      return this.rotaryTable;
    }
    const table = createF16Tensor(this.device, tokens * 32 * 4, `dicose-rope-table-${tokens}`);
    // createF16Tensor sizes in two-byte elements; four such elements reserve
    // one vec2<f32> record per (position, rotary pair).
    this.rotaryTables.push(table);
    this.rotaryTable = table;
    this.rotaryTableTokens = tokens;
    const pipeline = this.pipeline("rotary-table", ROTARY_TABLE_WGSL);
    const parameterOffset = this.parametersFor([tokens, 0, 0, 0]);
    const bindings = this.bind("rotary-table", [
      storage(table.buffer, table.byteLength),
      dynamicUniform(this.parameters),
    ]);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings.bindGroup, [parameterOffset]);
    pass.dispatchWorkgroups(Math.ceil(tokens * 32 / 256));
    return table;
  }

  private elementwise(
    pass: GPUComputePassEncoder,
    name: string,
    code: string,
    entries: readonly GPUBindGroupEntry[],
    values: readonly number[] | Uint32Array,
    workgroups?: number,
  ): void {
    const pipeline = this.pipeline(name, code);
    const parameterOffset = this.parametersFor(values);
    const bindings = this.bind(name, entries);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings.bindGroup, [parameterOffset]);
    this.dispatchElementwise(pass, workgroups ?? Math.ceil(Number(values[0]) / 256));
  }

  /**
   * Chrome/Metal caps a single dispatch dimension at 65,535 workgroups.  The
   * frequency-axis feature tensor for the supplied WAV is already larger than
   * that when processed by a 256-lane elementwise kernel, so flatten over X/Y
   * rather than relying on an invalid one-dimensional dispatch.
   */
  private dispatchElementwise(pass: GPUComputePassEncoder, workgroups: number): void {
    const limit = this.device.limits.maxComputeWorkgroupsPerDimension;
    const width = Math.min(workgroups, limit);
    const height = Math.ceil(workgroups / width);
    if (height > limit) {
      throw new RangeError(`DiCoSe elementwise dispatch exceeds the device workgroup grid for ${workgroups} workgroups`);
    }
    pass.dispatchWorkgroups(width, height);
  }

  private pipeline(name: string, code: string, layoutName = name): GPUComputePipeline {
    const existing = this.pipelines.get(name);
    if (existing !== undefined) return existing;
    const layout = this.layout(layoutName, bindingsFor(layoutName));
    const pipeline = this.device.createComputePipeline({
      label: `dicose-${name}`,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: this.device.createShaderModule({ label: `dicose-${name}-wgsl`, code }), entryPoint: "main" },
    });
    this.pipelines.set(name, pipeline);
    return pipeline;
  }

  private layout(name: string, entries: readonly GPUBindGroupLayoutEntry[]): GPUBindGroupLayout {
    const existing = this.layouts.get(name);
    if (existing !== undefined) return existing;
    const layout = this.device.createBindGroupLayout({ label: `dicose-${name}-layout`, entries });
    this.layouts.set(name, layout);
    return layout;
  }

  private bind(
    name: string,
    entries: readonly GPUBindGroupEntry[],
    uniformBinding = 4,
  ): DynamicBindGroup {
    let nextStorageBinding = 0;
    const normalized = entries.map((entry) => {
      if (entry.binding === uniformBinding) return entry;
      const binding = nextStorageBinding;
      nextStorageBinding += 1;
      return { ...entry, binding };
    });
    return {
      bindGroup: this.device.createBindGroup({
        label: `dicose-${name}-bindings`,
        layout: this.layout(name, bindingsFor(name)),
        entries: normalized,
      }),
      parameterOffset: 0,
    };
  }

  private parametersFor(values: readonly number[] | Uint32Array): number {
    this.requireAlive();
    if (this.parameterCursor >= PARAMETER_SLOTS) throw new Error("DiCoSe uniform pool exhausted");
    const bytes = new ArrayBuffer(PARAMETER_BYTES);
    const u32 = new Uint32Array(bytes);
    for (let index = 0; index < values.length; index += 1) u32[index] = values[index] ?? 0;
    const offset = this.parameterCursor * PARAMETER_BYTES;
    this.device.queue.writeBuffer(this.parameters, offset, bytes);
    this.parameterCursor += 1;
    return offset;
  }

  private requireAlive(): void {
    if (this.destroyed) throw new Error("DiCoSe GPU ops were destroyed");
  }
}

function storage(buffer: GPUBuffer, size: number, offset = 0): GPUBindGroupEntry {
  return { binding: 0, resource: { buffer, offset, size: align4(size) } };
}

function dynamicUniform(buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding: 4, resource: { buffer, size: PARAMETER_BYTES } };
}

function dynamicUniformAt(buffer: GPUBuffer, binding: number): GPUBindGroupEntry {
  return { binding, resource: { buffer, size: PARAMETER_BYTES } };
}

function bindingsFor(name: string): readonly GPUBindGroupLayoutEntry[] {
  const storageRead = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } });
  const storageWrite = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
  const uniform = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAMETER_BYTES } });
  switch (name) {
    case "linear": return [storageRead(0), storageRead(1), storageRead(2), storageWrite(3), uniform(4)];
    case "linear-packed": return [storageRead(0), storageRead(1), storageRead(2), storageWrite(3), uniform(4)];
    case "linear-packed-residual": return [storageRead(0), storageRead(1), storageRead(2), storageWrite(3), storageRead(4), uniform(5)];
    case "linear-packed-rotary-keys": return [storageRead(0), storageRead(1), storageRead(2), storageWrite(3), storageRead(4), uniform(5)];
    case "rmsnorm": return [storageRead(0), storageRead(1), storageRead(2), storageWrite(3), uniform(4)];
    case "add": return [storageRead(0), storageWrite(1), uniform(4)];
    case "copy": return [storageRead(0), storageWrite(1), uniform(4)];
    case "sample-even": return [storageRead(0), storageWrite(1), uniform(4)];
    case "transpose-tb": return [storageRead(0), storageWrite(1), uniform(4)];
    case "gather-slice": return [storageRead(0), storageWrite(1), uniform(4)];
    case "scatter-slice": return [storageRead(0), storageWrite(1), uniform(4)];
    case "spectral-to-pixels": return [storageRead(0), storageWrite(1), uniform(4)];
    case "pixels-to-spectral": return [storageRead(0), storageWrite(1), uniform(4)];
    case "attention": return [storageRead(0), storageWrite(1), uniform(4)];
    case "attention-grouped": return [storageRead(0), storageWrite(1), storageRead(2), storageRead(3), uniform(4)];
    case "rotary-table": return [storageWrite(0), uniform(4)];
    case "gates": return [storageWrite(0), storageRead(1), uniform(4)];
    case "complex-multiply": return [storageRead(0), storageRead(1), storageWrite(2), uniform(4)];
    case "affine": return [storageRead(0), storageRead(1), storageWrite(2), uniform(4)];
    case "conv2d": return [storageRead(0), storageRead(1), storageRead(2), storageWrite(3), uniform(4)];
    case "gelu": return [storageWrite(0), uniform(4)];
    case "tanh": return [storageWrite(0), uniform(4)];
    case "glu": return [storageRead(0), storageWrite(1), uniform(4)];
    default: throw new Error(`Unknown DiCoSe pipeline ${name}`);
  }
}

function activationCode(value: Activation): number {
  if (value === "gelu") return 1;
  if (value === "tanh") return 2;
  return 0;
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}

function packedLinearTileColumns(layout: GpuWeightTensor["layout"]): 128 | 256 | undefined {
  if (layout === "linear-tile-n128-k32") return 128;
  if (layout === "linear-tile-n256-k32") return 256;
  return undefined;
}

/**
 * Emit the two production dense owners. Four fixed-32 subgroups each own
 * eight rows; every lane owns one N128 or two N256 vec4 columns. Converter-
 * native [N-tile, K-tile, K32, N] weights let each loaded vector serve eight
 * rows without workgroup staging or barriers. The exact arm visits K in source
 * order with FP32 FMA. Owner128/K4-load experiments may instead form bounded
 * native-f16 K2 or K4 dot partials and immediately widen into FP32 state.
 */
function packedLinearWgsl(
  inner: number,
  columns: number,
  storageTileColumns: 128 | 256,
  outputTileColumns: 128 | 256,
  vectorizeK: boolean,
  accumulation: PackedAccumulation,
  activation: Activation,
  hasResidual: boolean,
  hasRotaryKeys: boolean,
  tileOffset = 0,
): string {
  if (accumulation !== "exact" && (outputTileColumns !== 128 || !vectorizeK)) {
    throw new RangeError(`${accumulation} accumulation requires owner128/K4 loads`);
  }
  const rowsPerSubgroup = 8;
  const vectorsPerLane = outputTileColumns / 128;
  const weightVectorsPerInner = storageTileColumns / 4;
  const ownerTilesPerStorageTile = storageTileColumns / outputTileColumns;
  const declarations = Array.from({ length: rowsPerSubgroup }, (_, row) =>
    Array.from({ length: vectorsPerLane }, (_, vector) =>
      `  var acc${row}_${vector} = vec4<f32>(0.0);`,
    ).join("\n"),
  ).join("\n");
  const broadcasts = (suffix: string, laneValue: string): string =>
    Array.from({ length: rowsPerSubgroup }, (_, row) =>
      `      let a${row}${suffix} = subgroupBroadcast(${laneValue}, ${row}u);`,
    ).join("\n");
  const weightLoads = (suffix: string, weightBase: string): string =>
    Array.from({ length: vectorsPerLane }, (_, vector) =>
      `      let b${vector}${suffix} = vec4<f32>(weight[
        ${weightBase} + subgroup_lane * ${vectorsPerLane}u + ${vector}u
      ]);`,
    ).join("\n");
  const contractions = (suffix: string): string =>
    Array.from({ length: rowsPerSubgroup }, (_, row) =>
      Array.from({ length: vectorsPerLane }, (_, vector) =>
        `      acc${row}_${vector} = fma(vec4<f32>(f32(a${row}${suffix})), b${vector}${suffix}, acc${row}_${vector});`,
      ).join("\n"),
    ).join("\n");
  const approximateContractions = Array.from({ length: rowsPerSubgroup }, (_, row) => {
    const broadcast = `      let a${row} = subgroupBroadcast(lane_a, ${row}u);`;
    if (accumulation === "k2") {
      return `${broadcast}
      let partial${row}_01 = vec4<f16>(
        dot(a${row}.xy, vec2<f16>(b0.x, b1.x)),
        dot(a${row}.xy, vec2<f16>(b0.y, b1.y)),
        dot(a${row}.xy, vec2<f16>(b0.z, b1.z)),
        dot(a${row}.xy, vec2<f16>(b0.w, b1.w))
      );
      acc${row}_0 = acc${row}_0 + vec4<f32>(partial${row}_01);
      let partial${row}_23 = vec4<f16>(
        dot(a${row}.zw, vec2<f16>(b2.x, b3.x)),
        dot(a${row}.zw, vec2<f16>(b2.y, b3.y)),
        dot(a${row}.zw, vec2<f16>(b2.z, b3.z)),
        dot(a${row}.zw, vec2<f16>(b2.w, b3.w))
      );
      acc${row}_0 = acc${row}_0 + vec4<f32>(partial${row}_23);`;
    }
    return `${broadcast}
      let partial${row} = vec4<f16>(
        dot(a${row}, vec4<f16>(b0.x, b1.x, b2.x, b3.x)),
        dot(a${row}, vec4<f16>(b0.y, b1.y, b2.y, b3.y)),
        dot(a${row}, vec4<f16>(b0.z, b1.z, b2.z, b3.z)),
        dot(a${row}, vec4<f16>(b0.w, b1.w, b2.w, b3.w))
      );
      acc${row}_0 = acc${row}_0 + vec4<f32>(partial${row});`;
  }).join("\n");
  const approximateTraversal = `    for (var inner_in_tile = 0u; inner_in_tile < 32u; inner_in_tile += 4u) {
      let inner_index = inner_tile * 32u + inner_in_tile;
      var lane_a = vec4<f16>(0.0h);
      let lane_row = row_base + subgroup_lane;
      if (subgroup_lane < 8u && lane_row < params.rows) {
        lane_a = input[(lane_row * INNER + inner_index) / 4u];
      }
      let weight_base0 = tile_base + inner_in_tile * WEIGHT_VECTORS_PER_INNER + storage_column_offset;
      let weight_base1 = weight_base0 + WEIGHT_VECTORS_PER_INNER;
      let weight_base2 = weight_base1 + WEIGHT_VECTORS_PER_INNER;
      let weight_base3 = weight_base2 + WEIGHT_VECTORS_PER_INNER;
      let b0 = weight[weight_base0 + subgroup_lane];
      let b1 = weight[weight_base1 + subgroup_lane];
      let b2 = weight[weight_base2 + subgroup_lane];
      let b3 = weight[weight_base3 + subgroup_lane];
${approximateContractions}
    }`;
  const exactInnerTraversal = vectorizeK
    ? `    for (var inner_in_tile = 0u; inner_in_tile < 32u; inner_in_tile += 4u) {
      let inner_index = inner_tile * 32u + inner_in_tile;
      var lane_a = vec4<f16>(0.0h);
      let lane_row = row_base + subgroup_lane;
      if (subgroup_lane < 8u && lane_row < params.rows) {
        lane_a = input[(lane_row * INNER + inner_index) / 4u];
      }
${["x", "y", "z", "w"].map((component, index) => {
    const suffix = `_${index}`;
    const weightBase = `weight_base${suffix}`;
    return `      let ${weightBase} = tile_base + (inner_in_tile + ${index}u) * WEIGHT_VECTORS_PER_INNER + storage_column_offset;
${weightLoads(suffix, weightBase)}
${broadcasts(suffix, `lane_a.${component}`)}
${contractions(suffix)}`;
  }).join("\n")}
    }`
    : `    for (var inner_in_tile = 0u; inner_in_tile < 32u; inner_in_tile += 1u) {
      let inner_index = inner_tile * 32u + inner_in_tile;
      var lane_a = 0.0h;
      let lane_row = row_base + subgroup_lane;
      if (subgroup_lane < 8u && lane_row < params.rows) {
        lane_a = input[lane_row * INNER + inner_index];
      }
      let weight_base = tile_base + inner_in_tile * WEIGHT_VECTORS_PER_INNER + storage_column_offset;
${weightLoads("", "weight_base")}
${broadcasts("", "lane_a")}
${contractions("")}
    }`;
  const innerTraversal = accumulation === "exact"
    ? exactInnerTraversal
    : approximateTraversal;
  const applyActivation = (value: string): string => {
    if (activation === "gelu") {
      return `    ${value} = vec4<f32>(gelu(${value}.x), gelu(${value}.y), gelu(${value}.z), gelu(${value}.w));`;
    }
    if (activation === "tanh") {
      return `    ${value} = vec4<f32>(tanh(${value}.x), tanh(${value}.y), tanh(${value}.z), tanh(${value}.w));`;
    }
    return "";
  };
  const stores = Array.from({ length: rowsPerSubgroup }, (_, row) =>
    Array.from({ length: vectorsPerLane }, (_, vector) => {
      const value = `value${row}_${vector}`;
      const store = hasResidual
        ? `let rounded = vec4<f16>(${value});
      output[row * COLUMN_VECTORS + column_vector] = vec4<f16>(
        vec4<f32>(rounded) + vec4<f32>(residual[row * COLUMN_VECTORS + column_vector])
      );`
        : hasRotaryKeys
          ? `output[row * COLUMN_VECTORS + column_vector] = rotate_key_vector(${value}, row, column_vector);`
          : `output[row * COLUMN_VECTORS + column_vector] = vec4<f16>(${value});`;
      return `
  {
    let row = row_base + ${row}u;
    if (row < params.rows) {
      let column_vector = column_vector_base + ${vector}u;
      var ${value} = acc${row}_${vector} + vec4<f32>(bias[column_vector]);
${applyActivation(value)}
      ${store}
    }
  }`;
    }).join("\n"),
  ).join("\n");
  return `${COMMON_WGSL}
struct Params {
  rows: u32,
  sequences: u32,
  tokens: u32,
  strided: u32,
}
const INNER: u32 = ${inner}u;
const INNER_TILES: u32 = ${inner / 32}u;
const COLUMN_VECTORS: u32 = ${columns / 4}u;
const TILE_VECTORS: u32 = ${outputTileColumns / 4}u;
const WEIGHT_VECTORS_PER_INNER: u32 = ${weightVectorsPerInner}u;
@group(0) @binding(0) var<storage, read> input: array<${vectorizeK ? "vec4<f16>" : "f16"}>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f16>>;
${hasResidual ? "@group(0) @binding(4) var<storage, read> residual: array<vec4<f16>>;" : hasRotaryKeys ? "@group(0) @binding(4) var<storage, read> rotary_table: array<vec2<f32>>;" : ""}
@group(0) @binding(${hasResidual || hasRotaryKeys ? 5 : 4}) var<uniform> params: Params;

${hasRotaryKeys ? `fn rotate_key_vector(value: vec4<f32>, row: u32, column_vector: u32) -> vec4<f16> {
  let rounded = vec4<f16>(value);
  let dimension = ((column_vector - 128u) * 4u) % 64u;
  let position = select(row % params.tokens, row / params.sequences, params.strided != 0u);
  let rotation0 = rotary_table[position * 32u + dimension / 2u];
  let rotation1 = rotary_table[position * 32u + dimension / 2u + 1u];
  return vec4<f16>(
    f16(f32(rounded.x) * rotation0.x - f32(rounded.y) * rotation0.y),
    f16(f32(rounded.x) * rotation0.y + f32(rounded.y) * rotation0.x),
    f16(f32(rounded.z) * rotation1.x - f32(rounded.w) * rotation1.y),
    f16(f32(rounded.z) * rotation1.y + f32(rounded.w) * rotation1.x)
  );
}` : ""}

@compute @workgroup_size(128, 1, 1)
fn main(
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != 32u) { return; }
  let row_base = group.y * 32u + subgroup * 8u;
  let column_tile = group.x + ${tileOffset}u;
  let column_vector_base = column_tile * TILE_VECTORS + subgroup_lane * ${vectorsPerLane}u;
${declarations}
  for (var inner_tile = 0u; inner_tile < INNER_TILES; inner_tile += 1u) {
    let storage_column_tile = column_tile / ${ownerTilesPerStorageTile}u;
    let storage_column_offset = (column_tile % ${ownerTilesPerStorageTile}u) * TILE_VECTORS;
    let tile_base =
      (storage_column_tile * INNER_TILES + inner_tile) * 32u * WEIGHT_VECTORS_PER_INNER;
${innerTraversal}
  }
${stores}
}`;
}

const COMMON_WGSL = /* wgsl */ `
enable f16;
enable subgroups;

fn flat_element_index(id: vec3<u32>, workgroups: vec3<u32>) -> u32 {
  return id.x + id.y * workgroups.x * 256u + id.z * workgroups.x * workgroups.y * 256u;
}

fn gelu(x: f32) -> f32 {
  let sign = select(-1.0, 1.0, x >= 0.0);
  // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2))). The previous kernel fed
  // x directly to erf, making every deterministic and CD GELU too steep.
  let ax = abs(x) * 0.7071067811865476;
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let erf = sign * (1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * exp(-ax * ax));
  return 0.5 * x * (1.0 + erf);
}
`;

const LINEAR_WGSL = `${COMMON_WGSL}
struct Params { rows: u32, inner: u32, columns: u32, activation: u32, }
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> a_tile: array<f16, 256>;
var<workgroup> b_tile: array<f16, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let row = group.y * 16u + local.y;
  let column = group.x * 16u + local.x;
  var sum = 0.0;
  for (var base = 0u; base < params.inner; base += 16u) {
    let a_column = base + local.x;
    let b_row = base + local.y;
    let local_index = local.y * 16u + local.x;
    a_tile[local_index] = select(0.0h, input[row * params.inner + a_column], row < params.rows && a_column < params.inner);
    b_tile[local_index] = select(0.0h, weight[b_row * params.columns + column], b_row < params.inner && column < params.columns);
    workgroupBarrier();
    for (var k = 0u; k < 16u; k += 1u) {
      sum = fma(f32(a_tile[local.y * 16u + k]), f32(b_tile[k * 16u + local.x]), sum);
    }
    workgroupBarrier();
  }
  if (row < params.rows && column < params.columns) {
    var value = sum + f32(bias[column]);
    if (params.activation == 1u) { value = gelu(value); }
    if (params.activation == 2u) { value = tanh(value); }
    output[row * params.columns + column] = f16(value);
  }
}`;

const RMSNORM_WGSL = `${COMMON_WGSL}
struct Params { rows: u32, columns: u32, has_mapping: u32, workgroup_width: u32, }
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> gamma: array<f16>;
@group(0) @binding(2) var<storage, read> mapping: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> partials: array<f32, 8>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(subgroup_invocation_id) subgroup_lane: u32, @builtin(subgroup_id) subgroup: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let row = group.x + group.y * params.workgroup_width;
  if (row >= params.rows) { return; }
  var sum = 0.0;
  for (var c = lane; c < params.columns; c += 256u) { let x = f32(input[row * params.columns + c]); sum = fma(x, x, sum); }
  let sub = subgroupAdd(sum);
  if (subgroup_lane == 0u) { partials[subgroup] = sub; }
  workgroupBarrier();
  if (lane == 0u) { var total = 0.0; for (var i = 0u; i < 8u; i += 1u) { total += partials[i]; } partials[0] = inverseSqrt(max(total, 1e-24)) * sqrt(f32(params.columns)); }
  workgroupBarrier();
  let factor = partials[0];
  for (var c = lane; c < params.columns; c += 256u) {
    var value = f32(input[row * params.columns + c]) * factor * f32(gamma[c]);
    if (params.has_mapping != 0u) { value = value * (f32(mapping[c]) + 1.0) + f32(mapping[params.columns + c]); }
    output[row * params.columns + c] = f16(value);
  }
}`;

const RMSNORM_ROWS8_WGSL = `${COMMON_WGSL}
struct Params { rows: u32, columns: u32, has_mapping: u32, workgroup_width: u32, }
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> gamma: array<f16>;
@group(0) @binding(2) var<storage, read> mapping: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != 32u) { return; }
  let workgroup = group.x + group.y * params.workgroup_width;
  let row = workgroup * 8u + subgroup;
  if (row >= params.rows) { return; }
  var total = 0.0;
  for (var original_subgroup = 0u; original_subgroup < 8u; original_subgroup += 1u) {
    let original_lane = original_subgroup * 32u + subgroup_lane;
    var partial = 0.0;
    for (var column = original_lane; column < params.columns; column += 256u) {
      let value = f32(input[row * params.columns + column]);
      partial = fma(value, value, partial);
    }
    total += subgroupAdd(partial);
  }
  let factor = inverseSqrt(max(total, 1e-24)) * sqrt(f32(params.columns));
  for (var column = subgroup_lane; column < params.columns; column += 32u) {
    var value = f32(input[row * params.columns + column]) * factor * f32(gamma[column]);
    if (params.has_mapping != 0u) {
      value = value * (f32(mapping[column]) + 1.0) + f32(mapping[params.columns + column]);
    }
    output[row * params.columns + column] = f16(value);
  }
}`;

const ADD_WGSL = `${COMMON_WGSL}
struct Params { elements: u32, _a: u32, _b: u32, _c: u32, }
@group(0) @binding(0) var<storage, read> source: array<f16>;
@group(0) @binding(1) var<storage, read_write> destination: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) { let index = flat_element_index(id, workgroups); if (index < params.elements) { destination[index] = f16(f32(destination[index]) + f32(source[index])); } }`;

const COPY_WGSL = `${COMMON_WGSL}
struct Params { elements: u32, _a: u32, _b: u32, _c: u32, }
@group(0) @binding(0) var<storage, read> source: array<f16>;
@group(0) @binding(1) var<storage, read_write> destination: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) { let index = flat_element_index(id, workgroups); if (index < params.elements) { destination[index] = source[index]; } }`;

const SAMPLE_EVEN_WGSL = `${COMMON_WGSL}
struct Params { elements: u32, samples: u32, _a: u32, _b: u32, }
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) {
  let index = flat_element_index(id, workgroups);
  if (index >= params.samples) { return; }
  let quotient = params.elements / params.samples;
  let remainder = params.elements % params.samples;
  let source_index = index * quotient + (index * remainder) / params.samples;
  output[index] = input[source_index];
}`;

const TRANSPOSE_TB_WGSL = `${COMMON_WGSL}
struct Params { time: u32, bands: u32, dim: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) {
  let index = flat_element_index(id, workgroups);
  let total = params.time * params.bands * params.dim;
  if (index >= total) { return; }
  let d = index % params.dim;
  let temp = index / params.dim;
  let band = temp % params.bands;
  let time = temp / params.bands;
  output[(band * params.time + time) * params.dim + d] = input[index];
}`;

const GATHER_SLICE_WGSL = `${COMMON_WGSL}
struct Params { rows: u32, source_width: u32, offset: u32, width: u32, }
@group(0) @binding(0) var<storage, read> source: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) { let index = flat_element_index(id, workgroups); let total = params.rows * params.width; if (index < total) { let row = index / params.width; let column = index % params.width; output[index] = source[row * params.source_width + params.offset + column]; } }`;

const SCATTER_SLICE_WGSL = `${COMMON_WGSL}
struct Params { rows: u32, destination_width: u32, offset: u32, width: u32, }
@group(0) @binding(0) var<storage, read> source: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) { let index = flat_element_index(id, workgroups); let total = params.rows * params.width; if (index < total) { let row = index / params.width; let column = index % params.width; output[row * params.destination_width + params.offset + column] = source[index]; } }`;

const SPECTRAL_TO_PIXELS_WGSL = `${COMMON_WGSL}
struct Params { time: u32, frequencies: u32, channels: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> source: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) { let index = flat_element_index(id, workgroups); let total = params.time * params.frequencies * params.channels; if (index < total) { let channel = index % params.channels; let pixel = index / params.channels; let frequency = pixel / params.time; let time = pixel % params.time; output[index] = source[(time * params.frequencies + frequency) * params.channels + channel]; } }`;

const PIXELS_TO_SPECTRAL_WGSL = `${COMMON_WGSL}
struct Params { time: u32, frequencies: u32, channels: u32, _pad: u32, }
@group(0) @binding(0) var<storage, read> source: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) { let index = flat_element_index(id, workgroups); let total = params.time * params.frequencies * params.channels; if (index < total) { let channel = index % params.channels; let pixel = index / params.channels; let frequency = pixel / params.time; let time = pixel % params.time; output[(time * params.frequencies + frequency) * params.channels + channel] = source[index]; } }`;

const ROTARY_TABLE_WGSL = `${COMMON_WGSL}
struct Params { tokens: u32, _a: u32, _b: u32, _c: u32, }
@group(0) @binding(0) var<storage, read_write> table: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.tokens * 32u) { return; }
  let pair = index % 32u;
  let position = index / 32u;
  let freq = pow(0.0001, f32(pair) / 32.0);
  let theta = f32(position) * freq;
  table[index] = vec2<f32>(cos(theta), sin(theta));
}`;

const ATTENTION_WGSL = `${COMMON_WGSL}
struct Params { sequences: u32, tokens: u32, _a: u32, _b: u32, }
const MODEL_DIM: u32 = 384u;
const HEAD: u32 = 64u;
const INNER_DIM: u32 = 512u;
const QKV_DIM: u32 = INNER_DIM * 3u;
@group(0) @binding(0) var<storage, read> qkv: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> keys: array<f16, 64>;
var<workgroup> values: array<f16, 64>;
fn rotary(value_even: f32, value_odd: f32, dimension: u32, position: u32) -> f32 {
  let pair = dimension / 2u;
  let freq = pow(0.0001, f32(pair) / 32.0);
  let theta = f32(position) * freq;
  return select(value_even * cos(theta) - value_odd * sin(theta), value_even * sin(theta) + value_odd * cos(theta), (dimension & 1u) == 1u);
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(subgroup_invocation_id) sublane: u32, @builtin(subgroup_id) subgroup: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let query = group.x * 8u + subgroup;
  let head = group.y;
  let sequence = group.z;
  if (sequence >= params.sequences) { return; }
  let qrow = sequence * params.tokens + query;
  let d0 = sublane;
  let d1 = sublane + 32u;
  var q0 = 0.0; var q1 = 0.0;
  if (query < params.tokens) {
    let qbase = qrow * QKV_DIM + head * HEAD;
    q0 = rotary(f32(qkv[qbase + (d0 & 0xfffffffeu)]), f32(qkv[qbase + (d0 | 1u)]), d0, query);
    q1 = rotary(f32(qkv[qbase + (d1 & 0xfffffffeu)]), f32(qkv[qbase + (d1 | 1u)]), d1, query);
  }
  var max_score = -3.402823466e38;
  var denominator = 0.0;
  var value0 = 0.0; var value1 = 0.0;
  for (var key = 0u; key < params.tokens; key += 1u) {
    if (lane < HEAD) {
      let kbase = (sequence * params.tokens + key) * QKV_DIM + INNER_DIM + head * HEAD;
      let even = lane & 0xfffffffeu;
      keys[lane] = f16(rotary(f32(qkv[kbase + even]), f32(qkv[kbase + (even | 1u)]), lane, key));
      values[lane] = qkv[(sequence * params.tokens + key) * QKV_DIM + INNER_DIM * 2u + head * HEAD + lane];
    }
    workgroupBarrier();
    let score = subgroupAdd(q0 * f32(keys[d0]) + q1 * f32(keys[d1])) * 0.125;
    let next_max = max(max_score, score);
    let alpha = exp(max_score - next_max);
    let beta = exp(score - next_max);
    denominator = denominator * alpha + beta;
    value0 = value0 * alpha + beta * f32(values[d0]);
    value1 = value1 * alpha + beta * f32(values[d1]);
    max_score = next_max;
    workgroupBarrier();
  }
  if (query < params.tokens && denominator > 0.0) {
    let out = qrow * INNER_DIM + head * HEAD;
    output[out + d0] = f16(value0 / denominator);
    output[out + d1] = f16(value1 / denominator);
  }
}`;

/**
 * Flash-style Q64×K16 owner. A lane owns one query/key score and one vec4
 * output column for each of four query rows. This turns the QK contraction
 * into 64 source-ordered FP32 FMAs per lane instead of one subgroup reduction
 * after every two FMAs. Softmax state is merged once per K16 block and the
 * score tile is immediately consumed by the matching P×V contraction.
 */
function attentionFlashWgsl(rotatedKeys: boolean): string {
  const queryTile = 64;
  const streams = 4;
  const vectorType = "vec4";
  const components = ["x", "y", "z", "w"] as const;
  const queryLocals = Array.from({ length: streams }, (_, stream) =>
    stream === 0
      ? "  let query_local0 = lane / KEY_TILE;"
      : `  let query_local${stream} = query_local0 + ${stream * 16}u;`,
  ).join("\n");
  const outputDeclarations = Array.from({ length: streams }, (_, stream) =>
    `  var output${stream} = vec4<f32>(0.0);`,
  ).join("\n");
  const scoreQueryLoads = Array.from({ length: streams }, (_, stream) =>
    `      let query${stream} = query_tile[query_local${stream} * HEAD_VECTORS + score_dimension];`,
  ).join("\n");
  const scoreFmas = components.map((component) => `      score_values = fma(
        ${vectorType}<f32>(${Array.from({ length: streams }, (_, stream) => `query${stream}.${component}`).join(", ")}),
        ${vectorType}<f32>(f32(key_vector.${component})),
        score_values
      );`).join("\n");
  const scoreDeclarations = Array.from({ length: streams }, (_, stream) =>
    `    let score${stream} = select(
      NEG_MAX,
      score_values.${components[stream]} * 0.125,
      query_base + query_local${stream} < params.tokens && valid_key
    );`,
  ).join("\n");
  const scoreStores = Array.from({ length: streams }, (_, stream) =>
    `    scores[query_local${stream} * KEY_TILE + key_or_dimension] = score${stream};`,
  ).join("\n");
  const blockDeclarations = Array.from({ length: streams }, (_, stream) =>
    `    var block${stream} = vec4<f32>(0.0);`,
  ).join("\n");
  const blockUpdates = Array.from({ length: streams }, (_, stream) =>
    `      block${stream} = fma(
        vec4<f32>(scores[query_local${stream} * KEY_TILE + value_key]),
        value_vector,
        block${stream}
      );`,
  ).join("\n");
  const outputMerges = Array.from({ length: streams }, (_, stream) =>
    `    if (query_base + query_local${stream} < params.tokens) {
      output${stream} = output${stream} * merge_scale[query_local${stream}] + block${stream};
    }`,
  ).join("\n");
  const outputStores = Array.from({ length: streams }, (_, stream) =>
    `  if (query_base + query_local${stream} < params.tokens) {
    let row = sequence_base + (query_base + query_local${stream}) * row_stride;
    output[row * INNER_VECTORS + output_vector] = context_value(
      output${stream} / vec4<f32>(denominators[query_local${stream}]), row, head
    );
  }`,
  ).join("\n");
  return `${COMMON_WGSL}
struct Params {
  sequences: u32,
  tokens: u32,
  has_gates: u32,
  strided: u32,
}
const HEAD: u32 = 64u;
const HEAD_VECTORS: u32 = 16u;
const INNER_VECTORS: u32 = 128u;
const QKV_VECTORS: u32 = 384u;
const QUERY_TILE: u32 = ${queryTile}u;
const KEY_TILE: u32 = 16u;
const NEG_MAX: f32 = -3.402823466e38;
@group(0) @binding(0) var<storage, read> qkv: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> rotary_table: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> gates: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> query_tile: array<vec4<f32>, ${queryTile * 16}>;
var<workgroup> key_tile: array<vec4<f16>, 256>;
var<workgroup> value_tile: array<vec4<f16>, 256>;
var<workgroup> scores: array<f32, ${queryTile * 16}>;
var<workgroup> maxima: array<f32, ${queryTile}>;
var<workgroup> denominators: array<f32, ${queryTile}>;
var<workgroup> merge_scale: array<f32, ${queryTile}>;

fn rotate_f32(value: vec4<f32>, position: u32, dimension_vector: u32) -> vec4<f32> {
  let rotation0 = rotary_table[position * 32u + dimension_vector * 2u];
  let rotation1 = rotary_table[position * 32u + dimension_vector * 2u + 1u];
  return vec4<f32>(
    value.x * rotation0.x - value.y * rotation0.y,
    value.x * rotation0.y + value.y * rotation0.x,
    value.z * rotation1.x - value.w * rotation1.y,
    value.z * rotation1.y + value.w * rotation1.x
  );
}

fn rotate_f16(value: vec4<f16>, position: u32, dimension_vector: u32) -> vec4<f16> {
  return vec4<f16>(rotate_f32(vec4<f32>(value), position, dimension_vector));
}

fn context_value(value: vec4<f32>, row: u32, head: u32) -> vec4<f16> {
  let rounded = vec4<f16>(value);
  if (params.has_gates != 0u) {
    let denominator = 1.0 + exp(-f32(gates[row * 8u + head]));
    return vec4<f16>(vec4<f32>(rounded) / vec4<f32>(denominator));
  }
  return rounded;
}

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != 32u) { return; }
  let query_base = group.x * QUERY_TILE;
  let head = group.y;
  let sequence = group.z;
  if (sequence >= params.sequences) { return; }
  let row_stride = select(1u, params.sequences, params.strided != 0u);
  let sequence_base = select(sequence * params.tokens, sequence, params.strided != 0u);

  if (lane < QUERY_TILE) {
    maxima[lane] = NEG_MAX;
    denominators[lane] = 0.0;
    merge_scale[lane] = 0.0;
  }
  for (var load = lane; load < QUERY_TILE * HEAD_VECTORS; load += 256u) {
    let query_local = load / HEAD_VECTORS;
    let dimension_vector = load % HEAD_VECTORS;
    let query = query_base + query_local;
    var value = vec4<f32>(0.0);
    if (query < params.tokens) {
      let row = sequence_base + query * row_stride;
      value = rotate_f32(
        vec4<f32>(qkv[row * QKV_VECTORS + head * HEAD_VECTORS + dimension_vector]),
        query,
        dimension_vector
      );
    }
    query_tile[load] = value;
  }
  workgroupBarrier();

${queryLocals}
  let key_or_dimension = lane % KEY_TILE;
${outputDeclarations}

  for (var key_base = 0u; key_base < params.tokens; key_base += KEY_TILE) {
    let key_local = lane / HEAD_VECTORS;
    let dimension_vector = lane % HEAD_VECTORS;
    let key = key_base + key_local;
    var key_value = vec4<f16>(0.0h);
    var value_value = vec4<f16>(0.0h);
    if (key < params.tokens) {
      let row = sequence_base + key * row_stride;
      let loaded_key = qkv[
        row * QKV_VECTORS + INNER_VECTORS + head * HEAD_VECTORS + dimension_vector
      ];
      key_value = ${rotatedKeys ? "loaded_key" : "rotate_f16(loaded_key, key, dimension_vector)"};
      value_value = qkv[
        row * QKV_VECTORS + INNER_VECTORS * 2u + head * HEAD_VECTORS + dimension_vector
      ];
    }
    key_tile[lane] = key_value;
    value_tile[lane] = value_value;
    workgroupBarrier();

    let valid_key = key_base + key_or_dimension < params.tokens;
    var score_values = ${vectorType}<f32>(0.0);
    for (var score_dimension = 0u; score_dimension < HEAD_VECTORS; score_dimension += 1u) {
      let key_vector = key_tile[key_or_dimension * HEAD_VECTORS + score_dimension];
${scoreQueryLoads}
${scoreFmas}
    }
${scoreDeclarations}
${scoreStores}
    workgroupBarrier();

    if (lane < QUERY_TILE) {
      let query = query_base + lane;
      if (query < params.tokens) {
        var block_max = NEG_MAX;
        for (var key_in_block = 0u; key_in_block < KEY_TILE; key_in_block += 1u) {
          if (key_base + key_in_block < params.tokens) {
            block_max = max(block_max, scores[lane * KEY_TILE + key_in_block]);
          }
        }
        let next_max = max(maxima[lane], block_max);
        let alpha = exp(maxima[lane] - next_max);
        var block_sum = 0.0;
        for (var key_in_block = 0u; key_in_block < KEY_TILE; key_in_block += 1u) {
          var probability = 0.0;
          if (key_base + key_in_block < params.tokens) {
            probability = exp(scores[lane * KEY_TILE + key_in_block] - next_max);
            block_sum += probability;
          }
          scores[lane * KEY_TILE + key_in_block] = probability;
        }
        maxima[lane] = next_max;
        denominators[lane] = denominators[lane] * alpha + block_sum;
        merge_scale[lane] = alpha;
      } else {
        merge_scale[lane] = 1.0;
      }
    }
    workgroupBarrier();

${blockDeclarations}
    for (var value_key = 0u; value_key < KEY_TILE; value_key += 1u) {
      let value_vector = vec4<f32>(value_tile[value_key * HEAD_VECTORS + key_or_dimension]);
${blockUpdates}
    }
${outputMerges}
    workgroupBarrier();
  }

  let output_vector = head * HEAD_VECTORS + key_or_dimension;
${outputStores}
}`;
}

/**
 * Eight query streams per fixed-32 subgroup. Keeping the 256-thread workgroup
 * while widening ownership to Q64 halves workgroups. Eight-key shared tiles
 * then amortize synchronization while retaining ascending-key update order.
 * Each stream retains the query8 scalar update order; the only algebraic
 * simplification is replacing the online softmax's exp(0) with literal 1.
 */
function attentionQ64Wgsl(rotatedKeys: boolean): string {
  const streams = 8;
  const queryTile = 64;
  const keyTile = 8;
  const queryDeclarations = Array.from({ length: streams }, (_, stream) =>
    stream === 0
      ? `  let query0 = group.x * ${queryTile}u + subgroup;`
      : `  let query${stream} = query0 + ${stream * 8}u;`,
  ).join("\n");
  const rowDeclarations = Array.from({ length: streams }, (_, stream) =>
    `  let qrow${stream} = sequence_base + query${stream} * row_stride;`,
  ).join("\n");
  const queryState = Array.from({ length: streams }, (_, stream) =>
    `  var q${stream}0 = 0.0; var q${stream}1 = 0.0;`,
  ).join("\n");
  const queryLoads = Array.from({ length: streams }, (_, stream) => `  if (query${stream} < params.tokens) {
    let qbase = qrow${stream} * QKV_DIM + head * HEAD;
    q${stream}0 = rotary(f32(qkv[qbase + (d0 & 0xfffffffeu)]), f32(qkv[qbase + (d0 | 1u)]), d0, query${stream});
    q${stream}1 = rotary(f32(qkv[qbase + (d1 & 0xfffffffeu)]), f32(qkv[qbase + (d1 | 1u)]), d1, query${stream});
  }`).join("\n");
  const reductionState = Array.from({ length: streams }, (_, stream) =>
    `  var max_score${stream} = -3.402823466e38; var denominator${stream} = 0.0;
  var value${stream}0 = 0.0; var value${stream}1 = 0.0;`,
  ).join("\n");
  const keyIndex = "key_index + ";
  const updates = Array.from({ length: streams }, (_, stream) => `    let score${stream} = subgroupAdd(q${stream}0 * f32(keys[${keyIndex}d0]) + q${stream}1 * f32(keys[${keyIndex}d1])) * 0.125;
    let next_max${stream} = max(max_score${stream}, score${stream});
    let score_is_new_max${stream} = score${stream} > max_score${stream};
    let scale${stream} = exp(select(score${stream} - max_score${stream}, max_score${stream} - score${stream}, score_is_new_max${stream}));
    let alpha${stream} = select(1.0, scale${stream}, score_is_new_max${stream});
    let beta${stream} = select(scale${stream}, 1.0, score_is_new_max${stream});
    denominator${stream} = denominator${stream} * alpha${stream} + beta${stream};
    value${stream}0 = value${stream}0 * alpha${stream} + beta${stream} * f32(values[${keyIndex}d0]);
    value${stream}1 = value${stream}1 * alpha${stream} + beta${stream} * f32(values[${keyIndex}d1]);
    max_score${stream} = next_max${stream};`).join("\n");
  const stores = Array.from({ length: streams }, (_, stream) => `  if (query${stream} < params.tokens && denominator${stream} > 0.0) {
    let out = qrow${stream} * INNER_DIM + head * HEAD;
    output[out + d0] = context_value(value${stream}0 / denominator${stream}, qrow${stream}, head);
    output[out + d1] = context_value(value${stream}1 / denominator${stream}, qrow${stream}, head);
  }`).join("\n");
  const tiledKeyLoads = Array.from({ length: keyTile / 4 }, (_, load) => `    {
      let load_index = lane + ${load * 256}u;
      let load_key = key_base + load_index / HEAD;
      let dimension = load_index % HEAD;
      if (load_key < params.tokens) {
        let key_row = sequence_base + load_key * row_stride;
        let kbase = key_row * QKV_DIM + INNER_DIM + head * HEAD;
${rotatedKeys
  ? "        keys[load_index] = qkv[kbase + dimension];"
  : "        let even = dimension & 0xfffffffeu;\n        keys[load_index] = f16(rotary(f32(qkv[kbase + even]), f32(qkv[kbase + (even | 1u)]), dimension, load_key));"}
        values[load_index] = qkv[key_row * QKV_DIM + INNER_DIM * 2u + head * HEAD + dimension];
      }
    }`).join("\n");
  const keyLoop = `  for (var key_base = 0u; key_base < params.tokens; key_base += ${keyTile}u) {
${tiledKeyLoads}
    workgroupBarrier();
${Array.from({ length: keyTile }, (_, keyInTile) => `    if (key_base + ${keyInTile}u < params.tokens) {
      let key_index = ${keyInTile * 64}u;
${updates}
    }`).join("\n")}
    workgroupBarrier();
  }`;
  return `${COMMON_WGSL}
struct Params {
  sequences: u32,
  tokens: u32,
  has_gates: u32,
  strided: u32,
}
const HEAD: u32 = 64u;
const INNER_DIM: u32 = 512u;
const QKV_DIM: u32 = INNER_DIM * 3u;
@group(0) @binding(0) var<storage, read> qkv: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;
@group(0) @binding(2) var<storage, read> rotary_table: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> gates: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> keys: array<f16, ${64 * keyTile}>;
var<workgroup> values: array<f16, ${64 * keyTile}>;
fn rotary(value_even: f32, value_odd: f32, dimension: u32, position: u32) -> f32 {
  let pair = dimension / 2u;
  let rotation = rotary_table[position * 32u + pair];
  return select(value_even * rotation.x - value_odd * rotation.y, value_even * rotation.y + value_odd * rotation.x, (dimension & 1u) == 1u);
}
fn context_value(value: f32, row: u32, head: u32) -> f16 {
  let rounded = f16(value);
  if (params.has_gates != 0u) {
    return f16(f32(rounded) / (1.0 + exp(-f32(gates[row * 8u + head]))));
  }
  return rounded;
}
@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) sublane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != 32u) { return; }
${queryDeclarations}
  let head = group.y;
  let sequence = group.z;
  if (sequence >= params.sequences) { return; }
  let row_stride = select(1u, params.sequences, params.strided != 0u);
  let sequence_base = select(sequence * params.tokens, sequence, params.strided != 0u);
${rowDeclarations}
  let d0 = sublane;
  let d1 = sublane + 32u;
${queryState}
${queryLoads}
${reductionState}
${keyLoop}
${stores}
}`;
}

/**
 * Four query streams per fixed-32 subgroup. Each stream intentionally keeps
 * query8's two-element dot expression, ascending-key online-softmax update,
 * and f16 output rounding; only the ownership schedule changes. The eight
 * subgroups therefore reuse each loaded K/V row across 32 queries per
 * workgroup, rather than the original eight.
 */
const GATES_WGSL = `${COMMON_WGSL}
struct Params { rows: u32, _a: u32, _b: u32, _c: u32, }
@group(0) @binding(0) var<storage, read_write> context: array<f16>;
@group(0) @binding(1) var<storage, read> gates: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) { let index = flat_element_index(id, workgroups); let total = params.rows * 512u; if (index >= total) { return; } let d = index % 512u; let row = index / 512u; let gate = f32(gates[row * 8u + d / 64u]); context[index] = f16(f32(context[index]) / (1.0 + exp(-gate))); }`;

const COMPLEX_MULTIPLY_WGSL = `${COMMON_WGSL}
struct Params { complexes: u32, _a: u32, _b: u32, _c: u32, }
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> mask: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) { let index = flat_element_index(id, workgroups); if (index >= params.complexes) { return; } let base = index * 2u; let ar = f32(input[base]); let ai = f32(input[base + 1u]); let br = f32(mask[base]); let bi = f32(mask[base + 1u]); output[base] = f16(ar * br - ai * bi); output[base + 1u] = f16(ar * bi + ai * br); }`;

const AFFINE_WGSL = `${COMMON_WGSL}
struct Params { elements: u32, c_out: f32, c_skip: f32, _pad: u32, }
@group(0) @binding(0) var<storage, read> model_output: array<f16>;
@group(0) @binding(1) var<storage, read> noisy_input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) { let index = flat_element_index(id, workgroups); if (index < params.elements) { output[index] = f16(params.c_out * f32(model_output[index]) + params.c_skip * f32(noisy_input[index])); } }`;

const CONV3X3_4X128_WGSL = `${COMMON_WGSL}
struct Params { rows: u32, width: u32, height: u32, _a: u32, }
const INPUT_CHANNELS: u32 = 4u;
const OUTPUT_CHANNELS: u32 = 128u;
const COLUMN_VECTORS: u32 = 32u;
const TAPS: u32 = 36u;
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> weight_tile: array<vec4<f16>, 1152>;
@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_index) local_lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != 32u) { return; }
  for (var load = local_lane; load < TAPS * COLUMN_VECTORS; load += 128u) {
    let tap = load / COLUMN_VECTORS;
    let column = (load % COLUMN_VECTORS) * 4u;
    weight_tile[load] = vec4<f16>(
      weight[column * TAPS + tap],
      weight[(column + 1u) * TAPS + tap],
      weight[(column + 2u) * TAPS + tap],
      weight[(column + 3u) * TAPS + tap]
    );
  }
  workgroupBarrier();

  let row_base = group.y * 32u + subgroup * 8u;
  var acc0 = vec4<f32>(bias[subgroup_lane]);
  var acc1 = acc0;
  var acc2 = acc0;
  var acc3 = acc0;
  var acc4 = acc0;
  var acc5 = acc0;
  var acc6 = acc0;
  var acc7 = acc0;
  let lane_row = row_base + subgroup_lane;
  var lane_y = 0u;
  var lane_x = 0u;
  var lane_row_valid = 0u;
  if (subgroup_lane < 8u && lane_row < params.rows) {
    lane_y = lane_row / params.width;
    lane_x = lane_row % params.width;
    lane_row_valid = 1u;
  }
  for (var tap = 0u; tap < TAPS; tap += 1u) {
    let channel = tap / 9u;
    let spatial = tap % 9u;
    let dy = i32(spatial / 3u) - 1;
    let dx = i32(spatial % 3u) - 1;
    var lane_a = 0.0h;
    var lane_valid = 0u;
    if (lane_row_valid != 0u) {
      let yy = i32(lane_y) + dy;
      let xx = i32(lane_x) + dx;
      if (yy >= 0 && xx >= 0 && yy < i32(params.height) && xx < i32(params.width)) {
        lane_a = input[(u32(yy) * params.width + u32(xx)) * INPUT_CHANNELS + channel];
        lane_valid = 1u;
      }
    }
    let b = vec4<f32>(weight_tile[tap * COLUMN_VECTORS + subgroup_lane]);
    let a0 = subgroupBroadcast(lane_a, 0u);
    let a1 = subgroupBroadcast(lane_a, 1u);
    let a2 = subgroupBroadcast(lane_a, 2u);
    let a3 = subgroupBroadcast(lane_a, 3u);
    let a4 = subgroupBroadcast(lane_a, 4u);
    let a5 = subgroupBroadcast(lane_a, 5u);
    let a6 = subgroupBroadcast(lane_a, 6u);
    let a7 = subgroupBroadcast(lane_a, 7u);
    let valid0 = subgroupBroadcast(lane_valid, 0u);
    let valid1 = subgroupBroadcast(lane_valid, 1u);
    let valid2 = subgroupBroadcast(lane_valid, 2u);
    let valid3 = subgroupBroadcast(lane_valid, 3u);
    let valid4 = subgroupBroadcast(lane_valid, 4u);
    let valid5 = subgroupBroadcast(lane_valid, 5u);
    let valid6 = subgroupBroadcast(lane_valid, 6u);
    let valid7 = subgroupBroadcast(lane_valid, 7u);
    if (valid0 != 0u) { acc0 = fma(vec4<f32>(f32(a0)), b, acc0); }
    if (valid1 != 0u) { acc1 = fma(vec4<f32>(f32(a1)), b, acc1); }
    if (valid2 != 0u) { acc2 = fma(vec4<f32>(f32(a2)), b, acc2); }
    if (valid3 != 0u) { acc3 = fma(vec4<f32>(f32(a3)), b, acc3); }
    if (valid4 != 0u) { acc4 = fma(vec4<f32>(f32(a4)), b, acc4); }
    if (valid5 != 0u) { acc5 = fma(vec4<f32>(f32(a5)), b, acc5); }
    if (valid6 != 0u) { acc6 = fma(vec4<f32>(f32(a6)), b, acc6); }
    if (valid7 != 0u) { acc7 = fma(vec4<f32>(f32(a7)), b, acc7); }
  }
  let row0 = row_base;
  if (row0 < params.rows) { output[row0 * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc0); }
  if (row0 + 1u < params.rows) { output[(row0 + 1u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc1); }
  if (row0 + 2u < params.rows) { output[(row0 + 2u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc2); }
  if (row0 + 3u < params.rows) { output[(row0 + 3u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc3); }
  if (row0 + 4u < params.rows) { output[(row0 + 4u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc4); }
  if (row0 + 5u < params.rows) { output[(row0 + 5u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc5); }
  if (row0 + 6u < params.rows) { output[(row0 + 6u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc6); }
  if (row0 + 7u < params.rows) { output[(row0 + 7u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc7); }
}`;

const CONV1X1_128_WGSL = `${COMMON_WGSL}
struct Params { rows: u32, _a: u32, _b: u32, _c: u32, }
const CHANNELS: u32 = 128u;
const COLUMN_VECTORS: u32 = 32u;
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f16>>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> weight_tile: array<vec4<f16>, 1024>;
@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_index) local_lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != 32u) { return; }
  let row_base = group.y * 32u + subgroup * 8u;
  var acc0 = vec4<f32>(bias[subgroup_lane]);
  var acc1 = acc0;
  var acc2 = acc0;
  var acc3 = acc0;
  var acc4 = acc0;
  var acc5 = acc0;
  var acc6 = acc0;
  var acc7 = acc0;
  for (var inner_tile = 0u; inner_tile < 4u; inner_tile += 1u) {
    for (var load = local_lane; load < 1024u; load += 128u) {
      let inner_in_tile = load / COLUMN_VECTORS;
      let column_vector = load % COLUMN_VECTORS;
      let column = column_vector * 4u;
      let inner = inner_tile * 32u + inner_in_tile;
      weight_tile[load] = vec4<f16>(
        weight[column * CHANNELS + inner],
        weight[(column + 1u) * CHANNELS + inner],
        weight[(column + 2u) * CHANNELS + inner],
        weight[(column + 3u) * CHANNELS + inner]
      );
    }
    workgroupBarrier();
    for (var inner_in_tile = 0u; inner_in_tile < 32u; inner_in_tile += 1u) {
      let inner = inner_tile * 32u + inner_in_tile;
      var lane_a = 0.0h;
      let lane_row = row_base + subgroup_lane;
      if (subgroup_lane < 8u && lane_row < params.rows) {
        lane_a = input[lane_row * CHANNELS + inner];
      }
      let a0 = subgroupBroadcast(lane_a, 0u);
      let a1 = subgroupBroadcast(lane_a, 1u);
      let a2 = subgroupBroadcast(lane_a, 2u);
      let a3 = subgroupBroadcast(lane_a, 3u);
      let a4 = subgroupBroadcast(lane_a, 4u);
      let a5 = subgroupBroadcast(lane_a, 5u);
      let a6 = subgroupBroadcast(lane_a, 6u);
      let a7 = subgroupBroadcast(lane_a, 7u);
      let b = vec4<f32>(weight_tile[inner_in_tile * COLUMN_VECTORS + subgroup_lane]);
      acc0 = fma(vec4<f32>(f32(a0)), b, acc0);
      acc1 = fma(vec4<f32>(f32(a1)), b, acc1);
      acc2 = fma(vec4<f32>(f32(a2)), b, acc2);
      acc3 = fma(vec4<f32>(f32(a3)), b, acc3);
      acc4 = fma(vec4<f32>(f32(a4)), b, acc4);
      acc5 = fma(vec4<f32>(f32(a5)), b, acc5);
      acc6 = fma(vec4<f32>(f32(a6)), b, acc6);
      acc7 = fma(vec4<f32>(f32(a7)), b, acc7);
    }
    workgroupBarrier();
  }
  let row0 = row_base;
  if (row0 < params.rows) { output[row0 * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc0); }
  if (row0 + 1u < params.rows) { output[(row0 + 1u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc1); }
  if (row0 + 2u < params.rows) { output[(row0 + 2u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc2); }
  if (row0 + 3u < params.rows) { output[(row0 + 3u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc3); }
  if (row0 + 4u < params.rows) { output[(row0 + 4u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc4); }
  if (row0 + 5u < params.rows) { output[(row0 + 5u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc5); }
  if (row0 + 6u < params.rows) { output[(row0 + 6u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc6); }
  if (row0 + 7u < params.rows) { output[(row0 + 7u) * COLUMN_VECTORS + subgroup_lane] = vec4<f16>(acc7); }
}`;

const CONV2D_WGSL = `${COMMON_WGSL}
struct Params { height: u32, width: u32, in_channels: u32, out_channels: u32, kernel: u32, _a: u32, _b: u32, _c: u32, }
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(8, 8, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x; let y = id.y; let out_channel = id.z;
  if (x >= params.width || y >= params.height || out_channel >= params.out_channels) { return; }
  let radius = i32(params.kernel / 2u);
  var sum = f32(bias[out_channel]);
  for (var channel = 0u; channel < params.in_channels; channel += 1u) {
    for (var ky = -radius; ky <= radius; ky += 1) { for (var kx = -radius; kx <= radius; kx += 1) {
      let yy = i32(y) + ky; let xx = i32(x) + kx;
      if (yy >= 0 && xx >= 0 && yy < i32(params.height) && xx < i32(params.width)) {
        let input_index = (u32(yy) * params.width + u32(xx)) * params.in_channels + channel;
        let weight_index = (((out_channel * params.in_channels + channel) * params.kernel + u32(ky + radius)) * params.kernel + u32(kx + radius));
        sum = fma(f32(input[input_index]), f32(weight[weight_index]), sum);
      }
    } }
  }
  output[(y * params.width + x) * params.out_channels + out_channel] = f16(sum);
}`;

const GELU_WGSL = `${COMMON_WGSL}
struct Params { elements: u32, _a: u32, _b: u32, _c: u32, }
@group(0) @binding(0) var<storage, read_write> value: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) { let index = flat_element_index(id, workgroups); if (index < params.elements) { value[index] = f16(gelu(f32(value[index]))); } }`;

const TANH_WGSL = `${COMMON_WGSL}
struct Params { elements: u32, _a: u32, _b: u32, _c: u32, }
@group(0) @binding(0) var<storage, read_write> value: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) { let index = flat_element_index(id, workgroups); if (index < params.elements) { value[index] = f16(tanh(f32(value[index]))); } }`;

const GLU_WGSL = `${COMMON_WGSL}
struct Params { rows: u32, columns: u32, _a: u32, _b: u32, }
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) workgroups: vec3<u32>) { let index = flat_element_index(id, workgroups); let total = params.rows * params.columns; if (index < total) { let row = index / params.columns; let column = index % params.columns; let base = row * params.columns * 2u + column; output[index] = f16(f32(input[base]) / (1.0 + exp(-f32(input[base + params.columns])))); } }`;
