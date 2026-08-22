import {
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  aceOpt0036VaeConvTranspose1dR4C8Wgsl,
  aceOpt0036VaeConvTranspose1dR8C4Wgsl,
  planAceOpt0036VaeConvTranspose1dR4C8,
  planAceOpt0036VaeConvTranspose1dR8C4,
  planAceOpt0036VaeConvTranspose1dRange,
  type AceOpt0036VaeConvTranspose1dKernelId,
  type AceOpt0036VaeConvTranspose1dPlan,
  type AceOpt0036VaeConvTranspose1dRangePlan,
} from "./vae-conv-transpose1d-fp16-reuse-axis-subgroup.js";
import type {
  AceVaeConvTranspose1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID =
  "ace-vae-fp16-fixed32-subgroup-shape-selected-conv-transpose1d-v1" as const;

const OUTPUT_RANGE_CONTROL_BYTES = 16;

export type AceOpt0040VaeConvTranspose1dOperationLabel =
  | "block-0-conv-t1"
  | "block-1-conv-t1"
  | "block-2-conv-t1"
  | "block-3-conv-t1"
  | "block-4-conv-t1";

export interface AceOpt0040VaeConvTranspose1dRoute {
  readonly operationLabel: AceOpt0040VaeConvTranspose1dOperationLabel;
  readonly reuseAxis: "channel" | "row";
  readonly kernelId: AceOpt0036VaeConvTranspose1dKernelId;
  readonly inputChannels: number;
  readonly outputChannels: number;
  readonly stride: number;
}

export const ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_ROUTES = Object.freeze([
  route("block-0-conv-t1", "channel", 2_048, 1_024, 10),
  route("block-1-conv-t1", "channel", 1_024, 512, 6),
  route("block-2-conv-t1", "channel", 512, 256, 4),
  route("block-3-conv-t1", "row", 256, 128, 4),
  route("block-4-conv-t1", "row", 128, 128, 2),
] as const);

export interface AceOpt0040VaeConvTranspose1dSelection {
  readonly selectorKernelId:
    typeof ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID;
  readonly operationLabel: AceOpt0040VaeConvTranspose1dOperationLabel;
  readonly reuseAxis: "channel" | "row";
  readonly kernelId: AceOpt0036VaeConvTranspose1dKernelId;
  readonly plan: AceOpt0036VaeConvTranspose1dPlan;
}

export interface AceOpt0040VaeConvTranspose1dBindings {
  readonly input: GPUBufferBinding;
  /** Converter-native FP16 `[phase,2,input,output]`. */
  readonly polyphaseWeight: GPUBufferBinding;
  readonly bias: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceOpt0040VaeConvTranspose1dDispatch {
  readonly label: string;
  readonly selectorKernelId:
    typeof ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID;
  readonly operationLabel: AceOpt0040VaeConvTranspose1dOperationLabel;
  readonly reuseAxis: "channel" | "row";
  readonly kernelId: AceOpt0036VaeConvTranspose1dKernelId;
  readonly plan: AceOpt0036VaeConvTranspose1dPlan;
  readonly outputRange: AceOpt0036VaeConvTranspose1dRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Exact OPT-0040 production owner. The selector is closed over the five
 * authenticated decoder operations and exposes the real OPT-0036 kernel ID on
 * every dispatch; it does not provide a geometric or portable fallback.
 */
export class AceOpt0040VaeConvTranspose1dShapeSelectorKernel {
  private readonly pipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0040VaeConvTranspose1dShapeSelectorKernel {
    if (
      !device.features.has("shader-f16") ||
      !device.features.has("subgroups") ||
      capability.subgroupMinSize !==
        ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE ||
      capability.subgroupMaxSize !==
        ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE ||
      Number(device.limits.maxComputeInvocationsPerWorkgroup) <
        ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE ||
      Number(device.limits.maxComputeWorkgroupSizeX) <
        ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE
    ) {
      throw new Error(
        "OPT-0040 ConvTranspose1D requires shader-f16, fixed32 subgroups, and WG128",
      );
    }
    return new AceOpt0040VaeConvTranspose1dShapeSelectorKernel(device);
  }

  async createDispatch(
    label: string,
    operationLabel: string,
    shape: AceVaeConvTranspose1dShape,
    bindings: AceOpt0040VaeConvTranspose1dBindings,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0040VaeConvTranspose1dDispatch> {
    this.requireLive();
    const selection = selectAceOpt0040VaeConvTranspose1d(
      operationLabel,
      shape,
    );
    const outputRange = planAceOpt0036VaeConvTranspose1dRange(
      selection.plan,
      range,
    );
    const maximumDispatch = Number(
      this.device.limits.maxComputeWorkgroupsPerDimension,
    );
    if (
      !Number.isSafeInteger(maximumDispatch) || maximumDispatch < 1 ||
      outputRange.workgroupsX > maximumDispatch ||
      outputRange.workgroupsY > maximumDispatch ||
      outputRange.workgroupsZ > maximumDispatch
    ) {
      throw new RangeError("OPT-0040 dispatch exceeds the device dimension");
    }
    const resources = [
      normalizeStorageBinding(
        this.device,
        bindings.input,
        selection.plan.inputBindingBytes,
        `${label} input`,
      ),
      normalizeStorageBinding(
        this.device,
        bindings.polyphaseWeight,
        selection.plan.weightBindingBytes,
        `${label} polyphase weight`,
      ),
      normalizeStorageBinding(
        this.device,
        bindings.bias,
        selection.plan.biasBindingBytes,
        `${label} bias`,
      ),
      normalizeStorageBinding(
        this.device,
        bindings.output,
        selection.plan.outputBindingBytes,
        `${label} output`,
      ),
    ] as const;
    const controlOffset = normalizeRangeOffset(this.device, range.control, label);
    const compiled = await this.pipelineFor(selection.plan);
    this.requireLive();
    const controlResource = Object.freeze({
      buffer: range.control.buffer,
      offset: 0,
      size: OUTPUT_RANGE_CONTROL_BYTES,
    });
    const key = `${planKey(selection.plan)}:${[
      ...resources,
      controlResource,
    ].map((binding) => this.bindingKey(binding)).join("|")}`;
    let bindGroup = this.bindGroups.get(key);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-opt-0040-bindings`,
        layout: compiled.bindGroupLayout,
        entries: [...resources, controlResource].map((resource, binding) => ({
          binding,
          resource,
        })),
      });
      this.bindGroups.set(key, bindGroup);
    }
    const owner = this;
    return Object.freeze({
      label,
      selectorKernelId:
        ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID,
      operationLabel: selection.operationLabel,
      reuseAxis: selection.reuseAxis,
      kernelId: selection.kernelId,
      plan: selection.plan,
      outputRange,
      encode(pass: GPUComputePassEncoder): void {
        owner.requireLive();
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroup, [controlOffset]);
        pass.dispatchWorkgroups(
          outputRange.workgroupsX,
          outputRange.workgroupsY,
          outputRange.workgroupsZ,
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
    plan: AceOpt0036VaeConvTranspose1dPlan,
  ): Promise<CompiledKernel> {
    const key = planKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileKernel(this.device, plan);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
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

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0040 ConvTranspose1D selector was destroyed");
    }
  }
}

export function selectAceOpt0040VaeConvTranspose1d(
  operationLabel: string,
  shape: AceVaeConvTranspose1dShape,
): AceOpt0040VaeConvTranspose1dSelection {
  const selected = ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_ROUTES.find((candidate) =>
    candidate.operationLabel === operationLabel
  );
  if (selected === undefined) {
    throw new RangeError(
      `OPT-0040 has no ConvTranspose1D route for ${operationLabel}`,
    );
  }
  if (
    shape.batch !== 1 ||
    shape.inputChannels !== selected.inputChannels ||
    shape.outputChannels !== selected.outputChannels ||
    shape.kernelSize !== selected.stride * 2 ||
    shape.stride !== selected.stride ||
    shape.dilation !== 1 ||
    shape.padding !== Math.ceil(selected.stride / 2) ||
    shape.outputPadding !== 0
  ) {
    throw new RangeError(
      `OPT-0040 ${operationLabel} changed its authenticated shape`,
    );
  }
  const plan = selected.reuseAxis === "channel"
    ? planAceOpt0036VaeConvTranspose1dR4C8(shape)
    : planAceOpt0036VaeConvTranspose1dR8C4(shape);
  if (plan.kernelId !== selected.kernelId) {
    throw new Error(`OPT-0040 ${operationLabel} selected the wrong kernel`);
  }
  return Object.freeze({
    selectorKernelId:
      ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID,
    operationLabel: selected.operationLabel,
    reuseAxis: selected.reuseAxis,
    kernelId: selected.kernelId,
    plan,
  });
}

function route(
  operationLabel: AceOpt0040VaeConvTranspose1dOperationLabel,
  reuseAxis: "channel" | "row",
  inputChannels: number,
  outputChannels: number,
  stride: number,
): Readonly<AceOpt0040VaeConvTranspose1dRoute> {
  return Object.freeze({
    operationLabel,
    reuseAxis,
    kernelId: reuseAxis === "channel"
      ? ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID
      : ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
    inputChannels,
    outputChannels,
    stride,
  });
}

async function compileKernel(
  device: GPUDevice,
  plan: AceOpt0036VaeConvTranspose1dPlan,
): Promise<CompiledKernel> {
  const label =
    `${ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID}-${planKey(plan)}`;
  const module = device.createShaderModule({
    label,
    code: plan.kernelId === ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID
      ? aceOpt0036VaeConvTranspose1dR4C8Wgsl(plan)
      : aceOpt0036VaeConvTranspose1dR8C4Wgsl(plan),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0040 ConvTranspose1D WGSL failed: ${errors.map(
        ({ lineNum, linePos, message }) => `${lineNum}:${linePos} ${message}`,
      ).join("; ")}`,
    );
  }
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: [
      ...[
        plan.inputBindingBytes,
        plan.weightBindingBytes,
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
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: device.createPipelineLayout({
      label: `${label}-layout`,
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: { module, entryPoint: "main" },
  });
  return Object.freeze({ pipeline, bindGroupLayout });
}

function normalizeStorageBinding(
  device: GPUDevice,
  binding: GPUBufferBinding,
  requiredBytes: number,
  label: string,
): GPUBufferBinding {
  const offset = Number(binding.offset ?? 0);
  const available = Number(binding.size ?? binding.buffer.size - offset);
  const alignment = Number(device.limits.minStorageBufferOffsetAlignment);
  if (
    !Number.isSafeInteger(offset) || offset < 0 ||
    !Number.isSafeInteger(available) || available < requiredBytes ||
    offset + requiredBytes > binding.buffer.size ||
    !Number.isSafeInteger(alignment) || alignment < 4 ||
    offset % alignment !== 0
  ) {
    throw new RangeError(`${label} does not expose ${requiredBytes} aligned bytes`);
  }
  return Object.freeze({ buffer: binding.buffer, offset, size: requiredBytes });
}

function normalizeRangeOffset(
  device: GPUDevice,
  binding: GPUBufferBinding,
  label: string,
): number {
  const offset = Number(binding.offset ?? 0);
  const available = Number(binding.size ?? binding.buffer.size - offset);
  const alignment = Number(device.limits.minUniformBufferOffsetAlignment);
  if (
    !Number.isSafeInteger(offset) || offset < 0 || offset > 0xffff_ffff ||
    !Number.isSafeInteger(available) || available < OUTPUT_RANGE_CONTROL_BYTES ||
    offset + OUTPUT_RANGE_CONTROL_BYTES > binding.buffer.size ||
    !Number.isSafeInteger(alignment) || alignment < 4 ||
    offset % alignment !== 0
  ) {
    throw new RangeError(`${label} range control is not dynamically aligned`);
  }
  return offset;
}

function planKey(plan: AceOpt0036VaeConvTranspose1dPlan): string {
  return [
    plan.kernelId,
    plan.batch,
    plan.inputFrames,
    plan.inputChannels,
    plan.outputChannels,
    plan.kernelSize,
    plan.stride,
    plan.dilation,
    plan.padding,
    plan.outputPadding,
  ].join("x");
}
