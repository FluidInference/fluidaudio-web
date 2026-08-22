import { describe, expect, it, vi } from "vitest";
import {
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID,
} from "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-16x64.js";
import {
  AceOpt0057VaeK7ShapeSelectorKernel,
} from "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import {
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID,
  planAceOpt0048VaeConvTranspose1dK4,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-partials.js";
import {
  AceOpt0052VaeConvTranspose1dK4ShapeSelectorKernel,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-shape-selector.js";
import { planAceFp16VaeConv1d } from
  "../src/webgpu/kernels/vae-conv1d-fp16.js";
import { planAceVaeDecoder } from "../src/webgpu/vae-decoder.js";

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const FIXED32 = Object.freeze({ subgroupMinSize: 32, subgroupMaxSize: 32 });

describe("OPT-0054 revision-7 selector owners", () => {
  it("caches one selected K7 pipeline/bind group and destroys both owners", async () => {
    const device = fakeDevice();
    const owner = AceOpt0057VaeK7ShapeSelectorKernel.create(device, FIXED32);
    const shape = {
      batch: 1,
      inputFrames: 33,
      inputChannels: 128,
      outputChannels: 128,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 3,
    } as const;
    const plan = planAceFp16VaeConv1d(shape, "float16");
    const bindings = {
      input: fakeBinding(plan.inputBindingBytes),
      weight: fakeBinding(plan.weightBindingBytes),
      bias: fakeBinding(plan.biasBindingBytes),
      output: fakeBinding(plan.outputBindingBytes),
    };
    const control = fakeBuffer(512);
    const first = await owner.createDispatch(
      "first",
      "block-3-res-1-conv1",
      shape,
      bindings,
      "float16",
      { base: 0, count: plan.outputElements, control: range(control, 256) },
    );
    const second = await owner.createDispatch(
      "second",
      "block-3-res-1-conv1",
      shape,
      bindings,
      "float16",
      { base: 0, count: plan.outputElements, control: range(control, 0) },
    );
    expect(first.kernelId).toBe(
      ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID,
    );
    expect(second.kernelId).toBe(first.kernelId);
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();
    const pass = fakePass();
    first.encode(pass);
    second.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([[256], [0]]);

    owner.destroy();
    owner.destroy();
    expect(() => first.encode(pass)).toThrow(/destroyed/);
    await expect(owner.createDispatch(
      "after-destroy",
      "block-3-res-1-conv1",
      shape,
      bindings,
      "float16",
      { base: 0, count: plan.outputElements, control: range(control, 0) },
    )).rejects.toThrow(/destroyed/);
  });

  it("caches one K4 transpose pipeline/bind group and destroys both owners", async () => {
    const device = fakeDevice();
    const owner = AceOpt0052VaeConvTranspose1dK4ShapeSelectorKernel.create(
      device,
      FIXED32,
    );
    const operation = planAceVaeDecoder(256).operations.find(({ label }) =>
      label === "block-4-conv-t1"
    )!;
    if (operation.kind !== "conv-transpose1d") {
      throw new Error("block-4 transpose fixture changed");
    }
    const plan = planAceOpt0048VaeConvTranspose1dK4(
      operation.label,
      operation.shape,
    );
    const bindings = {
      input: fakeBinding(plan.inputBindingBytes),
      weight: fakeBinding(plan.weightBindingBytes),
      bias: fakeBinding(plan.biasBindingBytes),
      output: fakeBinding(plan.outputBindingBytes),
    };
    const control = fakeBuffer(512);
    const first = await owner.createDispatch(
      "first",
      operation.label,
      operation.shape,
      bindings,
      { base: 0, count: plan.outputElements, control: range(control, 256) },
    );
    const second = await owner.createDispatch(
      "second",
      operation.label,
      operation.shape,
      bindings,
      { base: 0, count: plan.outputElements, control: range(control, 0) },
    );
    expect(first.kernelId).toBe(
      ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_R8C4_K4_KERNEL_ID,
    );
    expect(second.kernelId).toBe(first.kernelId);
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();
    const pass = fakePass();
    first.encode(pass);
    second.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([[256], [0]]);

    owner.destroy();
    owner.destroy();
    expect(() => first.encode(pass)).toThrow(/destroyed/);
    await expect(owner.createDispatch(
      "after-destroy",
      operation.label,
      operation.shape,
      bindings,
      { base: 0, count: plan.outputElements, control: range(control, 0) },
    )).rejects.toThrow(/destroyed/);
  });
});

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: fakeBuffer(size), offset: 0, size };
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function range(buffer: GPUBuffer, offset: number): GPUBufferBinding {
  return { buffer, offset, size: 16 };
}

function fakeDevice(): GPUDevice & {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
} {
  return {
    features: new Set(["shader-f16", "subgroups"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxStorageBufferBindingSize: 1_073_741_824,
      maxBufferSize: 1_073_741_824,
      minUniformBufferOffsetAlignment: 256,
      minStorageBufferOffsetAlignment: 256,
    },
    createShaderModule: vi.fn(() => ({
      getCompilationInfo: vi.fn(async () => ({ messages: [] })),
    })),
    createBindGroupLayout: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    createComputePipelineAsync: vi.fn(async () => ({})),
    createBindGroup: vi.fn(() => ({})),
  } as unknown as GPUDevice & {
    readonly createShaderModule: ReturnType<typeof vi.fn>;
    readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
    readonly createBindGroup: ReturnType<typeof vi.fn>;
  };
}

function fakePass(): GPUComputePassEncoder & {
  readonly setBindGroup: ReturnType<typeof vi.fn>;
} {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
  } as unknown as GPUComputePassEncoder & {
    readonly setBindGroup: ReturnType<typeof vi.fn>;
  };
}
