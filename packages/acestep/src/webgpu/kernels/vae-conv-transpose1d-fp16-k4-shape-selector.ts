import {
  ACE_VAE_REVISION7_POLYPHASE_TRANSPOSE_TENSOR,
  ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS,
} from "../../model/manifest.js";
import { requireAceDisjointOutput } from "./correctness-utils.js";
import {
  type AceOpt0036VaeConvTranspose1dKernelId,
  type AceOpt0036VaeConvTranspose1dRangePlan,
} from "./vae-conv-transpose1d-fp16-reuse-axis-subgroup.js";
import {
  AceOpt0040VaeConvTranspose1dShapeSelectorKernel,
  selectAceOpt0040VaeConvTranspose1d,
  type AceOpt0040VaeConvTranspose1dOperationLabel,
} from "./vae-conv-transpose1d-fp16-shape-selector.js";
import {
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID,
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID,
  aceOpt0048VaeConvTranspose1dK4Wgsl,
  planAceOpt0048VaeConvTranspose1dK4,
  planAceOpt0048VaeConvTranspose1dK4Range,
  type AceOpt0048VaeConvTranspose1dK4KernelId,
  type AceOpt0048VaeConvTranspose1dK4Plan,
} from "./vae-conv-transpose1d-fp16-k4-partials.js";
import type {
  AceVaeConvTranspose1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID =
  "ace-opt-0052-vae-convtranspose-exact-k4-shape-selector-v1" as const;

const OUTPUT_RANGE_CONTROL_BYTES = 16;

export type AceOpt0052VaeConvTranspose1dOwner =
  | "revision6-polyphase"
  | "k4-channel-reuse"
  | "k4-row-reuse";

export interface AceOpt0052VaeConvTranspose1dBindings {
  readonly input: GPUBufferBinding;
  readonly weight: GPUBufferBinding;
  readonly bias: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceOpt0052VaeConvTranspose1dSelection {
  readonly selectorKernelId:
    typeof ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID;
  readonly operationLabel: AceOpt0040VaeConvTranspose1dOperationLabel;
  readonly owner: AceOpt0052VaeConvTranspose1dOwner;
  readonly kernelId:
    | AceOpt0036VaeConvTranspose1dKernelId
    | AceOpt0048VaeConvTranspose1dK4KernelId;
  readonly plan: AceOpt0048VaeConvTranspose1dK4Plan | null;
}

export interface AceOpt0052VaeConvTranspose1dDispatch {
  readonly label: string;
  readonly selectorKernelId:
    typeof ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID;
  readonly operationLabel: AceOpt0040VaeConvTranspose1dOperationLabel;
  readonly owner: AceOpt0052VaeConvTranspose1dOwner;
  readonly kernelId:
    | AceOpt0036VaeConvTranspose1dKernelId
    | AceOpt0048VaeConvTranspose1dK4KernelId;
  readonly outputRange: AceOpt0036VaeConvTranspose1dRangePlan;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledK4Kernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Revision-7 mixed-layout ConvTranspose owner. Block 0 delegates to the
 * unchanged OPT-0040 polyphase owner; blocks 1-4 consume the package-native
 * OPT-0048 K4 layout selected by the frozen OPT-0052 route table.
 */
export class AceOpt0052VaeConvTranspose1dK4ShapeSelectorKernel {
  private readonly pipelines = new Map<string, Promise<CompiledK4Kernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly block0: AceOpt0040VaeConvTranspose1dShapeSelectorKernel,
  ) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0052VaeConvTranspose1dK4ShapeSelectorKernel {
    const block0 = AceOpt0040VaeConvTranspose1dShapeSelectorKernel.create(
      device,
      capability,
    );
    return new AceOpt0052VaeConvTranspose1dK4ShapeSelectorKernel(
      device,
      block0,
    );
  }

  async createDispatch(
    label: string,
    operationLabel: string,
    shape: AceVaeConvTranspose1dShape,
    bindings: AceOpt0052VaeConvTranspose1dBindings,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0052VaeConvTranspose1dDispatch> {
    this.requireLive();
    const selection = selectAceOpt0052VaeConvTranspose1d(
      operationLabel,
      shape,
    );
    if (selection.owner === "revision6-polyphase") {
      const dispatch = await this.block0.createDispatch(
        label,
        operationLabel,
        shape,
        {
          input: bindings.input,
          polyphaseWeight: bindings.weight,
          bias: bindings.bias,
          output: bindings.output,
        },
        range,
      );
      this.requireLive();
      return Object.freeze({
        label,
        selectorKernelId:
          ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
        operationLabel: dispatch.operationLabel,
        owner: selection.owner,
        kernelId: dispatch.kernelId,
        outputRange: dispatch.outputRange,
        encode: (pass: GPUComputePassEncoder): void => dispatch.encode(pass),
      });
    }

    const plan = selection.plan!;
    const outputRange = planAceOpt0048VaeConvTranspose1dK4Range(plan, range);
    requireDispatchDimensions(this.device, outputRange);
    const resources = [
      normalizeStorageBinding(
        this.device,
        bindings.input,
        plan.inputBindingBytes,
        `${label} input`,
      ),
      normalizeStorageBinding(
        this.device,
        bindings.weight,
        plan.weightBindingBytes,
        `${label} K4 weight`,
      ),
      normalizeStorageBinding(
        this.device,
        bindings.bias,
        plan.biasBindingBytes,
        `${label} bias`,
      ),
      normalizeStorageBinding(
        this.device,
        bindings.output,
        plan.outputBindingBytes,
        `${label} output`,
      ),
    ] as const;
    requireAceDisjointOutput(resources[3], [
      resources[0],
      resources[1],
      resources[2],
      range.control,
    ], label);
    const controlOffset = normalizeRangeOffset(this.device, range.control, label);
    const compiled = await this.pipelineFor(plan);
    this.requireLive();
    const controlResource = Object.freeze({
      buffer: range.control.buffer,
      offset: 0,
      size: OUTPUT_RANGE_CONTROL_BYTES,
    });
    const bindGroupKey = `${planKey(plan)}:${[
      ...resources,
      controlResource,
    ].map((binding) => this.bindingKey(binding)).join("|")}`;
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-opt-0052-k4-bindings`,
        layout: compiled.bindGroupLayout,
        entries: [...resources, controlResource].map((resource, binding) => ({
          binding,
          resource,
        })),
      });
      this.bindGroups.set(bindGroupKey, bindGroup);
    }
    const owner = this;
    return Object.freeze({
      label,
      selectorKernelId:
        ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
      operationLabel: plan.operationLabel,
      owner: selection.owner,
      kernelId: plan.kernelId,
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
    this.block0.destroy();
  }

  private pipelineFor(
    plan: AceOpt0048VaeConvTranspose1dK4Plan,
  ): Promise<CompiledK4Kernel> {
    const key = planKey(plan);
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileK4Kernel(this.device, plan);
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
      throw new Error("OPT-0052 ConvTranspose1D selector was destroyed");
    }
  }
}

export function selectAceOpt0052VaeConvTranspose1d(
  operationLabel: string,
  shape: AceVaeConvTranspose1dShape,
): AceOpt0052VaeConvTranspose1dSelection {
  if (operationLabel === "block-0-conv-t1") {
    const selected = selectAceOpt0040VaeConvTranspose1d(operationLabel, shape);
    if (
      ACE_VAE_REVISION7_POLYPHASE_TRANSPOSE_TENSOR !==
        "vae.decoder.block.0.conv_t1.weight"
    ) {
      throw new Error("OPT-0052 block-0 package contract changed");
    }
    return Object.freeze({
      selectorKernelId:
        ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
      operationLabel: selected.operationLabel,
      owner: "revision6-polyphase",
      kernelId: selected.kernelId,
      plan: null,
    });
  }
  const contract = ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.find((candidate) =>
    candidate.operationLabel === operationLabel
  );
  if (contract === undefined) {
    throw new RangeError(
      `OPT-0052 has no ConvTranspose1D route for ${operationLabel}`,
    );
  }
  const plan = planAceOpt0048VaeConvTranspose1dK4(operationLabel, shape);
  if (
    plan.inputChannels !== contract.inputChannels ||
    plan.outputChannels !== contract.outputChannels ||
    plan.stride !== contract.stride ||
    plan.reuseAxis !== contract.reuseAxis ||
    plan.kernelId !== (contract.reuseAxis === "channel"
      ? ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R4C8_K4_KERNEL_ID
      : ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID)
  ) {
    throw new Error(`OPT-0052 ${operationLabel} selected the wrong K4 owner`);
  }
  return Object.freeze({
    selectorKernelId:
      ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
    operationLabel: plan.operationLabel,
    owner: contract.reuseAxis === "channel"
      ? "k4-channel-reuse"
      : "k4-row-reuse",
    kernelId: plan.kernelId,
    plan,
  });
}

async function compileK4Kernel(
  device: GPUDevice,
  plan: AceOpt0048VaeConvTranspose1dK4Plan,
): Promise<CompiledK4Kernel> {
  const label = `${ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID}-${planKey(plan)}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0048VaeConvTranspose1dK4Wgsl(plan.operationLabel, plan),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0052 ConvTranspose1D K4 WGSL failed: ${errors.map(
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

function requireDispatchDimensions(
  device: GPUDevice,
  range: AceOpt0036VaeConvTranspose1dRangePlan,
): void {
  const maximum = Number(device.limits.maxComputeWorkgroupsPerDimension);
  if (
    !Number.isSafeInteger(maximum) || maximum < 1 ||
    range.workgroupsX > maximum ||
    range.workgroupsY > maximum ||
    range.workgroupsZ > maximum
  ) {
    throw new RangeError("OPT-0052 dispatch exceeds the device dimension");
  }
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
    offset % alignment !== 0 ||
    requiredBytes > Number(device.limits.maxStorageBufferBindingSize)
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

function planKey(plan: AceOpt0048VaeConvTranspose1dK4Plan): string {
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
