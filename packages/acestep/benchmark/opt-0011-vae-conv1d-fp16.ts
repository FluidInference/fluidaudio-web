import {
  planAceVaeConv1d,
  type AceVaeConv1dShape,
} from "../src/webgpu/kernels/vae-primitives.js";
import { createAceScopedBuffers } from
  "../src/webgpu/scoped-buffer-allocation.js";
import { ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY } from
  "../src/webgpu/vae-decoder.js";

export const ACE_OPT_0011_VAE_CONV1D_FP16_KERNEL_SIZE = 7;
export const ACE_OPT_0011_VAE_CONV1D_FP16_SUPPORTED_DILATIONS =
  Object.freeze([1, 3, 9] as const);
export const ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK = 64;
export const ACE_OPT_0011_VAE_CONV1D_FP16_TILE_FRAMES = 16;
export const ACE_OPT_0011_VAE_CONV1D_FP16_TILE_CHANNELS = 8;
/** One unused time slot avoids a bank-aligned shared-input row stride. */
export const ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_TILE_STRIDE = 17;
/** One unused channel slot avoids a bank-aligned shared-weight row stride. */
export const ACE_OPT_0011_VAE_CONV1D_FP16_WEIGHT_TILE_STRIDE = 65;
export const ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_X = 16;
export const ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_Y = 8;
export const ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE = 128;
export const ACE_OPT_0011_VAE_CONV1D_FP16_SCALAR_ORACLE_ID =
  "opt-0011-vae-conv1d-fp16-scalar-oracle-v1" as const;
export const ACE_OPT_0011_VAE_CONV1D_FP16_PORTABLE_WORKGROUP_ID =
  "opt-0011-vae-conv1d-fp16-portable-workgroup-v1" as const;

const FLOAT16_BYTES = 2;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const GPU_BUFFER_ALIGNMENT = 4;
const MAX_DISPATCH_DIMENSION = 65_535;
const MAX_OUTPUT_RANGE_COUNT = 65_535;
const MAX_WGSL_U32 = 0xffff_ffff;
const OUTPUT_RANGE_PARAMETER_BYTES = 16;
const MINIMUM_UNIFORM_STRIDE = 256;

export type AceOpt0011VaeConv1dFp16OutputStorage = "float16" | "float32";
export type AceOpt0011VaeConv1dFp16KernelId =
  | typeof ACE_OPT_0011_VAE_CONV1D_FP16_SCALAR_ORACLE_ID
  | typeof ACE_OPT_0011_VAE_CONV1D_FP16_PORTABLE_WORKGROUP_ID;

export interface AceOpt0011VaeConv1dFp16Bindings {
  /** FP16 activation in frame-major NLC order. */
  readonly input: GPUBufferBinding;
  /** FP16 package weight in converter-native `[out,kernel,in]` order. */
  readonly weight: GPUBufferBinding;
  /** FP16 `[out]`; omitted by the final raw-waveform convolution. */
  readonly bias?: GPUBufferBinding;
  /** FP16 internal activation or FP32 raw-waveform boundary. */
  readonly output: GPUBufferBinding;
}

export interface AceOpt0011VaeConv1dFp16OutputRange {
  readonly batch: number;
  readonly firstOutputTime: number;
  /** Global NLC row before multiplying by the complete channel width. */
  readonly firstOutputRow: number;
  /** Complete-channel rows; one range never crosses a batch boundary. */
  readonly outputRowCount: number;
  readonly firstOutput: number;
  readonly outputCount: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
  readonly workgroupCount: number;
  readonly multiplyAdds: number;
}

export interface AceOpt0011VaeConv1dFp16Plan extends AceVaeConv1dShape {
  readonly outputStorage: AceOpt0011VaeConv1dFp16OutputStorage;
  readonly outputFrames: number;
  readonly inputElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly inputStorageBytes: number;
  readonly inputBindingBytes: number;
  readonly weightStorageBytes: number;
  readonly weightBindingBytes: number;
  readonly biasStorageBytes: number;
  readonly biasBindingBytes: number;
  readonly outputStorageBytes: number;
  readonly outputBindingBytes: number;
  readonly inputChannelChunk:
    typeof ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK;
  readonly inputChannelChunkCount: number;
  readonly tileFrames: typeof ACE_OPT_0011_VAE_CONV1D_FP16_TILE_FRAMES;
  readonly tileChannels: typeof ACE_OPT_0011_VAE_CONV1D_FP16_TILE_CHANNELS;
  readonly inputTileStride:
    typeof ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_TILE_STRIDE;
  readonly weightTileStride:
    typeof ACE_OPT_0011_VAE_CONV1D_FP16_WEIGHT_TILE_STRIDE;
  readonly workgroupSizeX:
    typeof ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_X;
  readonly workgroupSizeY:
    typeof ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_Y;
  readonly workgroupSize: typeof ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE;
  readonly inputTileElements: number;
  readonly weightTileElements: number;
  readonly inputTileBytes: number;
  readonly weightTileBytes: number;
  readonly workgroupStorageBytes: number;
  readonly outputRangeCount: number;
  readonly outputRanges: readonly AceOpt0011VaeConv1dFp16OutputRange[];
}

export interface AceOpt0011VaeConv1dFp16Dispatch {
  readonly label: string;
  readonly kernelId: AceOpt0011VaeConv1dFp16KernelId;
  readonly outputStorage: AceOpt0011VaeConv1dFp16OutputStorage;
  readonly plan: AceOpt0011VaeConv1dFp16Plan;
  readonly rangeCount: number;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
  encode(pass: GPUComputePassEncoder): void;
}

interface AceOpt0011VaeConv1dFp16ControlPlan {
  readonly stride: number;
  readonly parameterBytes: number;
}

interface CompiledAceOpt0011VaeConv1dFp16 {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly outputRangeParameters: GPUBuffer;
  readonly outputRangeParameterStride: number;
  destroy(): void;
}

interface NamedBinding {
  readonly name: "input" | "weight" | "bias" | "output";
  readonly binding: GPUBufferBinding;
}

/**
 * Benchmark-only scalar authority for the registered OPT-0011 K7 contract.
 *
 * Every invocation owns one output and directly loads FP16 operands. It uses
 * the same K-outer, chunk, increasing-Cin FP32 accumulation order and the same
 * explicit output boundary as the portable workgroup candidate.
 */
export class AceOpt0011VaeConv1dFp16ScalarOracleKernel {
  private constructor(
    private readonly core: AceOpt0011VaeConv1dFp16KernelCore,
  ) {}

  static create(device: GPUDevice):
    AceOpt0011VaeConv1dFp16ScalarOracleKernel {
    return new AceOpt0011VaeConv1dFp16ScalarOracleKernel(
      new AceOpt0011VaeConv1dFp16KernelCore(
        device,
        ACE_OPT_0011_VAE_CONV1D_FP16_SCALAR_ORACLE_ID,
      ),
    );
  }

  createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceOpt0011VaeConv1dFp16Bindings,
    outputStorage: AceOpt0011VaeConv1dFp16OutputStorage,
  ): Promise<AceOpt0011VaeConv1dFp16Dispatch> {
    return this.core.createDispatch(label, shape, bindings, outputStorage);
  }

  destroy(): void {
    this.core.destroy();
  }
}

/**
 * Benchmark-only portable `shader-f16` workgroup-memory OPT-0011 candidate.
 */
export class AceOpt0011VaeConv1dFp16PortableWorkgroupKernel {
  private constructor(
    private readonly core: AceOpt0011VaeConv1dFp16KernelCore,
  ) {}

  static create(device: GPUDevice):
    AceOpt0011VaeConv1dFp16PortableWorkgroupKernel {
    return new AceOpt0011VaeConv1dFp16PortableWorkgroupKernel(
      new AceOpt0011VaeConv1dFp16KernelCore(
        device,
        ACE_OPT_0011_VAE_CONV1D_FP16_PORTABLE_WORKGROUP_ID,
      ),
    );
  }

  createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceOpt0011VaeConv1dFp16Bindings,
    outputStorage: AceOpt0011VaeConv1dFp16OutputStorage,
  ): Promise<AceOpt0011VaeConv1dFp16Dispatch> {
    return this.core.createDispatch(label, shape, bindings, outputStorage);
  }

  destroy(): void {
    this.core.destroy();
  }
}

class AceOpt0011VaeConv1dFp16KernelCore {
  private readonly compiled = new Map<
    string,
    Promise<CompiledAceOpt0011VaeConv1dFp16>
  >();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    readonly kernelId: AceOpt0011VaeConv1dFp16KernelId,
  ) {
    requireKernelDevice(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceOpt0011VaeConv1dFp16Bindings,
    outputStorage: AceOpt0011VaeConv1dFp16OutputStorage,
  ): Promise<AceOpt0011VaeConv1dFp16Dispatch> {
    this.requireLive();
    requireOutputStorage(outputStorage);
    requireOutputBoundary(bindings.bias !== undefined, outputStorage);
    const plan = planAceOpt0011VaeConv1dFp16(shape, outputStorage);
    const controlPlan = this.requireDeviceLimits(plan);
    const normalized = this.requireBindings(label, plan, bindings);

    const hasBias = bindings.bias !== undefined;
    const compiled = await this.pipelineFor(plan, hasBias, controlPlan);
    this.requireLive();
    const rangeBinding = normalized.length;
    const bindGroupKey = `${convKey(plan, hasBias)}:${normalized.map(
      ({ binding }) => this.bindingKey(binding),
    ).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-${this.kernelId}-bindings`,
        layout: compiled.bindGroupLayout,
        entries: [
          ...normalized.map(({ binding: resource }, binding) => ({
            binding,
            resource,
          })),
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

    const owner = this;
    return Object.freeze({
      label,
      kernelId: this.kernelId,
      outputStorage,
      plan,
      rangeCount: plan.outputRangeCount,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        owner.requireLive();
        if (!Number.isSafeInteger(rangeIndex) || rangeIndex < 0) {
          throw new RangeError(
            `${label} OPT-0011 range index must be a non-negative safe integer`,
          );
        }
        const range = plan.outputRanges[rangeIndex];
        if (range === undefined) {
          throw new RangeError(
            `${label} OPT-0011 range ${rangeIndex} is outside [0, ${plan.outputRangeCount})`,
          );
        }
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroup, [
          rangeIndex * compiled.outputRangeParameterStride,
        ]);
        pass.dispatchWorkgroups(range.workgroupsX, range.workgroupsY, 1);
      },
      encode(pass: GPUComputePassEncoder): void {
        owner.requireLive();
        pass.setPipeline(compiled.pipeline);
        for (let index = 0; index < plan.outputRanges.length; index += 1) {
          owner.requireLive();
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
    plan: AceOpt0011VaeConv1dFp16Plan,
    hasBias: boolean,
    controlPlan: AceOpt0011VaeConv1dFp16ControlPlan,
  ): Promise<CompiledAceOpt0011VaeConv1dFp16> {
    const key = convKey(plan, hasBias);
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const created = this.compile(plan, hasBias, controlPlan);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }

  private async compile(
    plan: AceOpt0011VaeConv1dFp16Plan,
    hasBias: boolean,
    controlPlan: AceOpt0011VaeConv1dFp16ControlPlan,
  ): Promise<CompiledAceOpt0011VaeConv1dFp16> {
    const label = `ace-${this.kernelId}-${convKey(plan, hasBias)}`;
    const code = this.kernelId ===
        ACE_OPT_0011_VAE_CONV1D_FP16_SCALAR_ORACLE_ID
      ? aceOpt0011VaeConv1dFp16ScalarOracleWgsl(
        plan,
        hasBias,
        plan.outputStorage,
      )
      : aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl(
        plan,
        hasBias,
        plan.outputStorage,
      );
    const module = this.device.createShaderModule({ label, code });
    const compilation = await module.getCompilationInfo();
    this.requireLive();
    const shaderErrors = compilation.messages.filter(
      ({ type }) => type === "error",
    );
    if (shaderErrors.length > 0) {
      throw new Error(
        `OPT-0011 FP16 VAE Conv1D WGSL failed: ${shaderErrors.map((message) =>
          `${message.lineNum}:${message.linePos} ${message.message}`
        ).join("; ")}`,
      );
    }

    const dataBindingSizes = hasBias
      ? [
        plan.inputBindingBytes,
        plan.weightBindingBytes,
        plan.biasBindingBytes,
        plan.outputBindingBytes,
      ]
      : [
        plan.inputBindingBytes,
        plan.weightBindingBytes,
        plan.outputBindingBytes,
      ];
    const outputBinding = dataBindingSizes.length - 1;
    const rangeBinding = dataBindingSizes.length;
    const bindGroupLayout = this.device.createBindGroupLayout({
      label: `${label}-bindings`,
      entries: [
        ...dataBindingSizes.map((minBindingSize, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: binding === outputBinding
              ? "storage" as const
              : "read-only-storage" as const,
            minBindingSize,
          },
        })),
        {
          binding: rangeBinding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform" as const,
            hasDynamicOffset: true,
            minBindingSize: OUTPUT_RANGE_PARAMETER_BYTES,
          },
        },
      ],
    });
    const pipelineLayout = this.device.createPipelineLayout({
      label: `${label}-layout`,
      bindGroupLayouts: [bindGroupLayout],
    });
    const pipeline = await this.device.createComputePipelineAsync({
      label,
      layout: pipelineLayout,
      compute: { module, entryPoint: "main" },
    });
    this.requireLive();

    const allocated = await createAceScopedBuffers(
      this.device,
      [{
        label: `${label}-output-range-parameters`,
        size: controlPlan.parameterBytes,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
      }],
      `${label} output range parameters`,
    );
    const outputRangeParameters = allocated[0];
    if (outputRangeParameters === undefined) {
      throw new Error(`${label} output range allocation returned no buffer`);
    }
    let mapped = false;
    try {
      this.requireLive();
      const bytes = outputRangeParameters.getMappedRange();
      mapped = true;
      for (let index = 0; index < plan.outputRanges.length; index += 1) {
        const words = new Uint32Array(
          bytes,
          index * controlPlan.stride,
          OUTPUT_RANGE_PARAMETER_BYTES / Uint32Array.BYTES_PER_ELEMENT,
        );
        const range = plan.outputRanges[index]!;
        words[0] = range.firstOutputRow;
        words[1] = range.outputRowCount;
      }
      outputRangeParameters.unmap();
      mapped = false;
      this.requireLive();
      let resourcesDestroyed = false;
      return Object.freeze({
        pipeline,
        bindGroupLayout,
        outputRangeParameters,
        outputRangeParameterStride: controlPlan.stride,
        destroy(): void {
          if (resourcesDestroyed) return;
          resourcesDestroyed = true;
          outputRangeParameters.destroy();
        },
      });
    } catch (error) {
      if (mapped) {
        try {
          outputRangeParameters.unmap();
        } catch {
          // Destruction below is the final bounded cleanup path.
        }
      }
      outputRangeParameters.destroy();
      throw error;
    }
  }

  private requireBindings(
    label: string,
    plan: AceOpt0011VaeConv1dFp16Plan,
    bindings: AceOpt0011VaeConv1dFp16Bindings,
  ): readonly NamedBinding[] {
    const normalized: NamedBinding[] = [
      {
        name: "input",
        binding: requireStorageBinding(
          this.device,
          bindings.input,
          plan.inputStorageBytes,
          plan.inputBindingBytes,
          `${label} input`,
        ),
      },
      {
        name: "weight",
        binding: requireStorageBinding(
          this.device,
          bindings.weight,
          plan.weightStorageBytes,
          plan.weightBindingBytes,
          `${label} weight`,
        ),
      },
    ];
    if (bindings.bias !== undefined) {
      normalized.push({
        name: "bias",
        binding: requireStorageBinding(
          this.device,
          bindings.bias,
          plan.biasStorageBytes,
          plan.biasBindingBytes,
          `${label} bias`,
        ),
      });
    }
    normalized.push({
      name: "output",
      binding: requireStorageBinding(
        this.device,
        bindings.output,
        plan.outputStorageBytes,
        plan.outputBindingBytes,
        `${label} output`,
      ),
    });
    requireDisjointBindings(normalized, label);
    return Object.freeze(normalized);
  }

  private requireDeviceLimits(
    plan: AceOpt0011VaeConv1dFp16Plan,
  ): AceOpt0011VaeConv1dFp16ControlPlan {
    const maximumStorage =
      this.device.limits.maxComputeWorkgroupStorageSize;
    if (!Number.isSafeInteger(maximumStorage) || maximumStorage < 0) {
      throw new RangeError(
        "OPT-0011 FP16 VAE Conv1D device reported invalid workgroup storage",
      );
    }
    if (
      this.kernelId === ACE_OPT_0011_VAE_CONV1D_FP16_PORTABLE_WORKGROUP_ID &&
      plan.workgroupStorageBytes > maximumStorage
    ) {
      throw new RangeError(
        `OPT-0011 FP16 VAE Conv1D requires ${plan.workgroupStorageBytes} workgroup-storage bytes`,
      );
    }

    const maximumDispatch =
      this.device.limits.maxComputeWorkgroupsPerDimension;
    if (
      !Number.isSafeInteger(maximumDispatch) ||
      maximumDispatch < 1 ||
      plan.outputRanges.some(({ workgroupsX, workgroupsY }) =>
        workgroupsX > maximumDispatch || workgroupsY > maximumDispatch
      )
    ) {
      throw new RangeError(
        "OPT-0011 FP16 VAE Conv1D exceeds the device dispatch dimension",
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
        "OPT-0011 FP16 VAE Conv1D device reported invalid buffer limits",
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
          `OPT-0011 FP16 VAE Conv1D ${name} exceeds the device storage binding limit`,
        );
      }
      if (bytes > maximumBuffer) {
        throw new RangeError(
          `OPT-0011 FP16 VAE Conv1D ${name} exceeds the device buffer limit`,
        );
      }
    }

    const reportedAlignment =
      this.device.limits.minUniformBufferOffsetAlignment;
    if (!isValidGpuAlignment(reportedAlignment)) {
      throw new Error(
        "OPT-0011 FP16 VAE Conv1D device reported an invalid uniform alignment",
      );
    }
    const stride = Math.max(MINIMUM_UNIFORM_STRIDE, reportedAlignment);
    const parameterBytes = checkedProduct([
      Math.max(1, plan.outputRangeCount),
      stride,
    ], "range parameter bytes");
    const maximumDynamicOffset = checkedProduct([
      Math.max(0, plan.outputRangeCount - 1),
      stride,
    ], "maximum dynamic offset");
    if (
      stride % reportedAlignment !== 0 ||
      stride % Uint32Array.BYTES_PER_ELEMENT !== 0
    ) {
      throw new Error(
        "OPT-0011 FP16 VAE Conv1D device reported an invalid uniform alignment",
      );
    }
    if (
      parameterBytes > maximumBuffer ||
      maximumDynamicOffset > MAX_WGSL_U32
    ) {
      throw new RangeError(
        "OPT-0011 FP16 VAE Conv1D range controls exceed the device buffer or dynamic-offset limit",
      );
    }
    return Object.freeze({ stride, parameterBytes });
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
      throw new Error(`OPT-0011 FP16 VAE Conv1D ${this.kernelId} was destroyed`);
    }
  }
}

export function planAceOpt0011VaeConv1dFp16(
  shape: AceVaeConv1dShape,
  outputStorage: AceOpt0011VaeConv1dFp16OutputStorage,
): AceOpt0011VaeConv1dFp16Plan {
  requireOutputStorage(outputStorage);
  const portable = planAceVaeConv1d(shape);
  if (
    shape.kernelSize !== ACE_OPT_0011_VAE_CONV1D_FP16_KERNEL_SIZE ||
    shape.stride !== 1 ||
    !ACE_OPT_0011_VAE_CONV1D_FP16_SUPPORTED_DILATIONS.includes(
      shape.dilation as 1 | 3 | 9,
    ) ||
    shape.padding !== shape.dilation * 3
  ) {
    throw new RangeError(
      "OPT-0011 FP16 VAE Conv1D requires K7, stride one, dilation 1/3/9, and padding dilation*3",
    );
  }
  if (portable.outputFrames !== shape.inputFrames) {
    throw new Error("OPT-0011 FP16 VAE Conv1D lost same-length frame identity");
  }
  for (const [name, value] of [
    ["input", portable.inputElements],
    ["weight", portable.weightElements],
    ["output", portable.outputElements],
    ["padding", shape.padding],
  ] as const) requireWgslIndexable(value, name);

  const inputChannelChunkCount = Math.ceil(
    shape.inputChannels /
      ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK,
  );
  requireWgslIndexable(
    inputChannelChunkCount,
    "input channel chunk count",
  );
  const inputTileElements = checkedProduct([
    ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK,
    ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_TILE_STRIDE,
  ], "input tile");
  const weightTileElements = checkedProduct([
    ACE_OPT_0011_VAE_CONV1D_FP16_TILE_CHANNELS,
    ACE_OPT_0011_VAE_CONV1D_FP16_WEIGHT_TILE_STRIDE,
  ], "weight tile");
  const inputTileBytes = checkedStorageBytes(
    inputTileElements,
    FLOAT16_BYTES,
    "input tile",
  );
  const weightTileBytes = checkedStorageBytes(
    weightTileElements,
    FLOAT16_BYTES,
    "weight tile",
  );
  const workgroupStorageBytes = checkedSum(
    inputTileBytes,
    weightTileBytes,
    "workgroup storage",
  );
  const workgroupsY = Math.ceil(
    shape.outputChannels / ACE_OPT_0011_VAE_CONV1D_FP16_TILE_CHANNELS,
  );
  if (workgroupsY > MAX_DISPATCH_DIMENSION) {
    throw new RangeError(
      "OPT-0011 FP16 VAE Conv1D output channels exceed the 2D dispatch domain",
    );
  }

  const multiplyAddsPerRow = checkedProduct([
    shape.outputChannels,
    shape.kernelSize,
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
    MAX_DISPATCH_DIMENSION * ACE_OPT_0011_VAE_CONV1D_FP16_TILE_FRAMES,
  );
  if (maximumRowsPerRange < 1) {
    throw new RangeError(
      "OPT-0011 FP16 VAE Conv1D cannot fit one complete-channel output row in a production range",
    );
  }
  const rangesPerBatch = Math.ceil(
    portable.outputFrames / maximumRowsPerRange,
  );
  const plannedRangeCount = checkedProduct(
    [shape.batch, rangesPerBatch],
    "output range count",
  );
  if (plannedRangeCount > MAX_OUTPUT_RANGE_COUNT) {
    throw new RangeError(
      `OPT-0011 FP16 VAE Conv1D output range count exceeds ${MAX_OUTPUT_RANGE_COUNT}`,
    );
  }

  const outputRanges: AceOpt0011VaeConv1dFp16OutputRange[] = [];
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
        checkedProduct(
          [batch, portable.outputFrames],
          "range batch row",
        ),
        firstOutputTime,
        "range first row",
      );
      const outputCount = checkedProduct(
        [outputRowCount, shape.outputChannels],
        "range output count",
      );
      const workgroupsX = Math.ceil(
        outputRowCount / ACE_OPT_0011_VAE_CONV1D_FP16_TILE_FRAMES,
      );
      const lastStagedTime = checkedSum(
        firstOutputTime,
        checkedSum(
          checkedProduct([
            workgroupsX - 1,
            ACE_OPT_0011_VAE_CONV1D_FP16_TILE_FRAMES,
          ], "last staged tile offset"),
          checkedSum(
            ACE_OPT_0011_VAE_CONV1D_FP16_TILE_FRAMES - 1,
            checkedProduct([
              shape.kernelSize - 1,
              shape.dilation,
            ], "last staged kernel offset"),
            "last staged tile and kernel offset",
          ),
          "last staged range offset",
        ),
        "last staged input time",
      );
      requireWgslIndexable(lastStagedTime, "last staged input time");
      requireWgslIndexable(
        checkedSum(
          firstOutputTime,
          outputRowCount,
          "range end time",
        ),
        "range end time",
      );
      requireWgslIndexable(firstOutputRow, "range first output row");
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
  if (outputRanges.length !== plannedRangeCount) {
    throw new Error("OPT-0011 FP16 VAE Conv1D range count changed while planning");
  }
  const emittedOutputs = outputRanges.reduce(
    (sum, range) => checkedSum(sum, range.outputCount, "emitted outputs"),
    0,
  );
  if (emittedOutputs !== portable.outputElements) {
    throw new Error("OPT-0011 FP16 VAE Conv1D range planner lost outputs");
  }

  const inputStorageBytes = checkedStorageBytes(
    portable.inputElements,
    FLOAT16_BYTES,
    "input",
  );
  const weightStorageBytes = checkedStorageBytes(
    portable.weightElements,
    FLOAT16_BYTES,
    "weight",
  );
  const biasStorageBytes = checkedStorageBytes(
    shape.outputChannels,
    FLOAT16_BYTES,
    "bias",
  );
  const outputStorageBytes = checkedStorageBytes(
    portable.outputElements,
    outputStorage === "float16" ? FLOAT16_BYTES : FLOAT32_BYTES,
    "output",
  );
  return Object.freeze({
    ...shape,
    outputStorage,
    outputFrames: portable.outputFrames,
    inputElements: portable.inputElements,
    weightElements: portable.weightElements,
    outputElements: portable.outputElements,
    inputStorageBytes,
    inputBindingBytes: alignGpuBindingBytes(inputStorageBytes, "input"),
    weightStorageBytes,
    weightBindingBytes: alignGpuBindingBytes(weightStorageBytes, "weight"),
    biasStorageBytes,
    biasBindingBytes: alignGpuBindingBytes(biasStorageBytes, "bias"),
    outputStorageBytes,
    outputBindingBytes: alignGpuBindingBytes(outputStorageBytes, "output"),
    inputChannelChunk:
      ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK,
    inputChannelChunkCount,
    tileFrames: ACE_OPT_0011_VAE_CONV1D_FP16_TILE_FRAMES,
    tileChannels: ACE_OPT_0011_VAE_CONV1D_FP16_TILE_CHANNELS,
    inputTileStride: ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_TILE_STRIDE,
    weightTileStride: ACE_OPT_0011_VAE_CONV1D_FP16_WEIGHT_TILE_STRIDE,
    workgroupSizeX: ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_X,
    workgroupSizeY: ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_Y,
    workgroupSize: ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE,
    inputTileElements,
    weightTileElements,
    inputTileBytes,
    weightTileBytes,
    workgroupStorageBytes,
    outputRangeCount: outputRanges.length,
    outputRanges: Object.freeze(outputRanges),
  });
}

export function aceOpt0011VaeConv1dFp16ScalarOracleWgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceOpt0011VaeConv1dFp16OutputStorage,
): string {
  requireOutputStorage(outputStorage);
  requireOutputBoundary(hasBias, outputStorage);
  const plan = planAceOpt0011VaeConv1dFp16(shape, outputStorage);
  return fp16ConvPrelude(
    plan,
    hasBias,
    ACE_OPT_0011_VAE_CONV1D_FP16_SCALAR_ORACLE_ID,
  ) + /* wgsl */ `
@compute @workgroup_size(
  ${ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_X},
  ${ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_Y},
  1,
)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>,
) {
  let range_first_time = output_range.first_output_row % OUTPUT_FRAMES;
  let batch = output_range.first_output_row / OUTPUT_FRAMES;
  let tile_first_time =
    range_first_time + group.x * ${ACE_OPT_0011_VAE_CONV1D_FP16_TILE_FRAMES}u;
  let output_time = tile_first_time + local.x;
  let output_channel =
    group.y * ${ACE_OPT_0011_VAE_CONV1D_FP16_TILE_CHANNELS}u + local.y;
  let range_end_time = range_first_time + output_range.output_row_count;
  let output_active =
    output_time < range_end_time && output_channel < OUTPUT_CHANNELS;
  var sum: f32 = 0.0;
  if (output_active) { sum = ${initialSumWgsl(hasBias)}; }

  if (output_active) {
    for (var kernel = 0u; kernel < 7u; kernel += 1u) {
      let padded_time = output_time + kernel * DILATION;
      // Invalid padding skips every channel operation for this tap.
      if (padded_time >= PADDING) {
        let input_time = padded_time - PADDING;
        if (input_time < INPUT_FRAMES) {
          for (
            var input_channel_chunk = 0u;
            input_channel_chunk < INPUT_CHANNEL_CHUNKS;
            input_channel_chunk += 1u
          ) {
            let chunk_first_channel = input_channel_chunk *
              ${ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK}u;
            let chunk_channel_count = min(
              ${ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK}u,
              INPUT_CHANNELS - chunk_first_channel
            );
            for (
              var chunk_channel = 0u;
              chunk_channel < chunk_channel_count;
              chunk_channel += 1u
            ) {
              let input_channel = chunk_first_channel + chunk_channel;
              let input_operand = f32(input[
                (batch * INPUT_FRAMES + input_time) * INPUT_CHANNELS +
                input_channel
              ]);
              // Native [output_channel, kernel, input_channel] weight order.
              let weight_operand = f32(weight[
                (output_channel * 7u + kernel) * INPUT_CHANNELS +
                input_channel
              ]);
              sum = sum + input_operand * weight_operand;
            }
          }
        }
      }
    }
  }

  if (output_active) {
    let output_row = batch * OUTPUT_FRAMES + output_time;
    output[output_row * OUTPUT_CHANNELS + output_channel] =
      ${outputValueWgsl(hasBias, outputStorage)};
  }
}
`;
}

export function aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl(
  shape: AceVaeConv1dShape,
  hasBias: boolean,
  outputStorage: AceOpt0011VaeConv1dFp16OutputStorage,
): string {
  requireOutputStorage(outputStorage);
  requireOutputBoundary(hasBias, outputStorage);
  const plan = planAceOpt0011VaeConv1dFp16(shape, outputStorage);
  return fp16ConvPrelude(
    plan,
    hasBias,
    ACE_OPT_0011_VAE_CONV1D_FP16_PORTABLE_WORKGROUP_ID,
  ) + /* wgsl */ `
var<workgroup> input_tile: array<f16, ${plan.inputTileElements}>;
var<workgroup> weight_tile: array<f16, ${plan.weightTileElements}>;

@compute @workgroup_size(
  ${ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_X},
  ${ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_Y},
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
    range_first_time + group.x * ${ACE_OPT_0011_VAE_CONV1D_FP16_TILE_FRAMES}u;
  let output_time = tile_first_time + local.x;
  let output_channel =
    group.y * ${ACE_OPT_0011_VAE_CONV1D_FP16_TILE_CHANNELS}u + local.y;
  let range_end_time = range_first_time + output_range.output_row_count;
  let output_active =
    output_time < range_end_time && output_channel < OUTPUT_CHANNELS;
  var sum: f32 = 0.0;
  if (output_active) { sum = ${initialSumWgsl(hasBias)}; }

  // Increasing chunks concatenate into the exact K-outer, increasing-Cin
  // scalar order while f16 workgroup storage reuses operands.
  for (var kernel = 0u; kernel < 7u; kernel += 1u) {
    for (
      var input_channel_chunk = 0u;
      input_channel_chunk < INPUT_CHANNEL_CHUNKS;
      input_channel_chunk += 1u
    ) {
      let chunk_first_channel = input_channel_chunk *
        ${ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK}u;
      for (
        var tile_index = lane;
        tile_index < ${ACE_OPT_0011_VAE_CONV1D_FP16_TILE_FRAMES * ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK}u;
        tile_index += ${ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE}u
      ) {
        let tile_time = tile_index /
          ${ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK}u;
        let chunk_channel = tile_index %
          ${ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK}u;
        let input_channel = chunk_first_channel + chunk_channel;
        let padded_time = tile_first_time + tile_time + kernel * DILATION;
        var value: f16 = f16(0.0);
        if (input_channel < INPUT_CHANNELS && padded_time >= PADDING) {
          let input_time = padded_time - PADDING;
          if (input_time < INPUT_FRAMES) {
            value = input[
              (batch * INPUT_FRAMES + input_time) * INPUT_CHANNELS +
              input_channel
            ];
          }
        }
        input_tile[
          chunk_channel * ${ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_TILE_STRIDE}u +
          tile_time
        ] = value;
      }
      for (
        var tile_index = lane;
        tile_index < ${ACE_OPT_0011_VAE_CONV1D_FP16_TILE_CHANNELS * ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK}u;
        tile_index += ${ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE}u
      ) {
        let tile_output_channel = tile_index /
          ${ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK}u;
        let chunk_channel = tile_index %
          ${ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK}u;
        let input_channel = chunk_first_channel + chunk_channel;
        let weight_output_channel =
          group.y * ${ACE_OPT_0011_VAE_CONV1D_FP16_TILE_CHANNELS}u +
          tile_output_channel;
        var value: f16 = f16(0.0);
        if (
          weight_output_channel < OUTPUT_CHANNELS &&
          input_channel < INPUT_CHANNELS
        ) {
          // Native [output_channel, kernel, input_channel] weight order.
          value = weight[
            (weight_output_channel * 7u + kernel) * INPUT_CHANNELS +
            input_channel
          ];
        }
        weight_tile[
          tile_output_channel *
            ${ACE_OPT_0011_VAE_CONV1D_FP16_WEIGHT_TILE_STRIDE}u +
          chunk_channel
        ] = value;
      }
      workgroupBarrier();

      if (output_active) {
        let padded_time = output_time + kernel * DILATION;
        // Invalid padding skips every channel operation for this tap.
        if (padded_time >= PADDING) {
          let input_time = padded_time - PADDING;
          if (input_time < INPUT_FRAMES) {
            let chunk_channel_count = min(
              ${ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK}u,
              INPUT_CHANNELS - chunk_first_channel
            );
            let weight_base = local.y *
              ${ACE_OPT_0011_VAE_CONV1D_FP16_WEIGHT_TILE_STRIDE}u;
            for (
              var chunk_channel = 0u;
              chunk_channel < chunk_channel_count;
              chunk_channel += 1u
            ) {
              let input_operand = f32(input_tile[
                chunk_channel *
                  ${ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_TILE_STRIDE}u +
                local.x
              ]);
              let weight_operand = f32(
                weight_tile[weight_base + chunk_channel]
              );
              sum = sum + input_operand * weight_operand;
            }
          }
        }
      }
      workgroupBarrier();
    }
  }

  if (output_active) {
    let output_row = batch * OUTPUT_FRAMES + output_time;
    output[output_row * OUTPUT_CHANNELS + output_channel] =
      ${outputValueWgsl(hasBias, outputStorage)};
  }
}
`;
}

function fp16ConvPrelude(
  plan: AceOpt0011VaeConv1dFp16Plan,
  hasBias: boolean,
  kernelId: AceOpt0011VaeConv1dFp16KernelId,
): string {
  const outputBinding = hasBias ? 3 : 2;
  const rangeBinding = hasBias ? 4 : 3;
  const biasDeclaration = hasBias
    ? "@group(0) @binding(2) var<storage, read> bias: array<f16>;"
    : "";
  const outputElementType = plan.outputStorage === "float16" ? "f16" : "f32";
  return /* wgsl */ `
// kernel-id: ${kernelId}
enable f16;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const OUTPUT_FRAMES: u32 = ${plan.outputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;
const PADDING: u32 = ${plan.padding}u;
const DILATION: u32 = ${plan.dilation}u;
const INPUT_CHANNEL_CHUNKS: u32 = ${plan.inputChannelChunkCount}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
${biasDeclaration}
@group(0) @binding(${outputBinding}) var<storage, read_write>
  output: array<${outputElementType}>;

struct OutputRangeParameters {
  first_output_row: u32,
  output_row_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(${rangeBinding}) var<uniform>
  output_range: OutputRangeParameters;
`;
}

function initialSumWgsl(hasBias: boolean): string {
  return hasBias ? "f32(bias[output_channel])" : "0.0";
}

function outputValueWgsl(
  hasBias: boolean,
  outputStorage: AceOpt0011VaeConv1dFp16OutputStorage,
): string {
  const sum = hasBias
    ? "sum"
    : "select(sum, bitcast<f32>(0u), (bitcast<u32>(sum) & 0x7fffffffu) == 0u)";
  return outputStorage === "float16" ? `f16(${sum})` : sum;
}

function requireKernelDevice(device: GPUDevice): void {
  if (!device.features.has("shader-f16")) {
    throw new Error(
      "OPT-0011 FP16 VAE Conv1D requires WebGPU shader-f16",
    );
  }
  const maximumInvocations =
    device.limits.maxComputeInvocationsPerWorkgroup;
  const maximumSizeX = device.limits.maxComputeWorkgroupSizeX;
  const maximumSizeY = device.limits.maxComputeWorkgroupSizeY;
  if (
    !Number.isSafeInteger(maximumInvocations) ||
    !Number.isSafeInteger(maximumSizeX) ||
    !Number.isSafeInteger(maximumSizeY) ||
    maximumInvocations < ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE ||
    maximumSizeX < ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_X ||
    maximumSizeY < ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_Y
  ) {
    throw new Error(
      "OPT-0011 FP16 VAE Conv1D requires a 16x8 (128-lane) compute workgroup",
    );
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
      "OPT-0011 FP16 VAE Conv1D device reported an invalid storage alignment",
    );
  }
  const maximumBinding = Number(device.limits.maxStorageBufferBindingSize);
  const maximumBuffer = Number(device.limits.maxBufferSize);
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
      `${label} binding does not expose an aligned ${requiredStorageBytes}-byte storage payload in ${requiredBindingBytes} binding bytes`,
    );
  }
  if (
    !Number.isSafeInteger(maximumBinding) || maximumBinding < 1 ||
    available > maximumBinding
  ) {
    throw new RangeError(`${label} binding exceeds the device storage binding limit`);
  }
  if (
    !Number.isSafeInteger(maximumBuffer) || maximumBuffer < 1 ||
    bufferBytes > maximumBuffer
  ) {
    throw new RangeError(`${label} buffer exceeds the device buffer limit`);
  }
  return Object.freeze({
    buffer: binding.buffer,
    offset,
    size: requiredBindingBytes,
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

function convKey(
  plan: AceOpt0011VaeConv1dFp16Plan,
  hasBias: boolean,
): string {
  return [
    plan.batch,
    plan.inputFrames,
    plan.inputChannels,
    plan.outputChannels,
    plan.kernelSize,
    plan.stride,
    plan.dilation,
    plan.padding,
    plan.outputStorage,
    hasBias ? "bias" : "no-bias",
  ].join("x");
}

function requireOutputStorage(
  outputStorage: AceOpt0011VaeConv1dFp16OutputStorage,
): void {
  if (outputStorage !== "float16" && outputStorage !== "float32") {
    throw new TypeError(
      `OPT-0011 FP16 VAE Conv1D has unknown output storage ${String(outputStorage)}`,
    );
  }
}

function requireOutputBoundary(
  hasBias: boolean,
  outputStorage: AceOpt0011VaeConv1dFp16OutputStorage,
): void {
  if (hasBias && outputStorage === "float32") {
    throw new RangeError(
      "OPT-0011 FP16 VAE Conv1D float32 output is reserved for the final no-bias raw-waveform boundary",
    );
  }
}

function checkedProduct(values: readonly number[], label: string): number {
  let product = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(
        `OPT-0011 FP16 VAE Conv1D ${label} operand is not a non-negative safe integer`,
      );
    }
    product *= value;
    if (!Number.isSafeInteger(product)) {
      throw new RangeError(
        `OPT-0011 FP16 VAE Conv1D ${label} is not a safe integer`,
      );
    }
  }
  return product;
}

function checkedSum(left: number, right: number, label: string): number {
  if (
    !Number.isSafeInteger(left) || left < 0 ||
    !Number.isSafeInteger(right) || right < 0
  ) {
    throw new RangeError(
      `OPT-0011 FP16 VAE Conv1D ${label} operand is not a non-negative safe integer`,
    );
  }
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw new RangeError(
      `OPT-0011 FP16 VAE Conv1D ${label} is not a safe integer`,
    );
  }
  return sum;
}

function checkedStorageBytes(
  elements: number,
  bytesPerElement: number,
  label: string,
): number {
  return checkedProduct([elements, bytesPerElement], `${label} storage bytes`);
}

function alignGpuBindingBytes(bytes: number, label: string): number {
  const rounded = Math.ceil(bytes / GPU_BUFFER_ALIGNMENT) *
    GPU_BUFFER_ALIGNMENT;
  if (!Number.isSafeInteger(rounded) || rounded < bytes) {
    throw new RangeError(
      `OPT-0011 FP16 VAE Conv1D ${label} binding bytes are not a safe integer`,
    );
  }
  return rounded;
}

function requireWgslIndexable(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > MAX_WGSL_U32) {
    throw new RangeError(
      `OPT-0011 FP16 VAE Conv1D ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}

function isValidGpuAlignment(value: number): boolean {
  return Number.isSafeInteger(value) &&
    value >= GPU_BUFFER_ALIGNMENT &&
    Number.isInteger(Math.log2(value));
}
