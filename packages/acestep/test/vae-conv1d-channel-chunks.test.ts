import { describe, expect, it, vi } from "vitest";

import {
  ACE_CHANNEL_CHUNKED_VAE_CONV1D_INPUT_CHANNEL_CHUNK,
  ACE_CHANNEL_CHUNKED_VAE_CONV1D_WORKGROUP_SIZE,
  AceChannelChunkedVaeConv1dKernel,
  aceChannelChunkedVaeConv1dWgsl,
  planAceChannelChunkedVaeConv1d,
  selectAceChannelChunkedVaeConv1d,
} from "../src/webgpu/kernels/vae-conv1d-channel-chunks.js";
import type {
  AceVaeConv1dShape,
  AceVaeConvBindings,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";

describe("production channel-chunked FP32 VAE Conv1D", () => {
  it("plans the exact 64-channel source-order layout", () => {
    const plan = planAceChannelChunkedVaeConv1d(shape());
    expect(plan).toMatchObject({
      outputFrames: 4_097,
      inputElements: 4_195_328,
      weightElements: 7_340_032,
      outputElements: 4_195_328,
      inputChannelChunk: 64,
      inputChannelChunkCount: 16,
      inputTileElements: 1_088,
      weightTileElements: 520,
      inputTileBytes: 4_352,
      weightTileBytes: 2_080,
      workgroupStorageBytes: 6_432,
    });
    const source = aceChannelChunkedVaeConv1dWgsl(shape(), true);
    expect(source).toContain("@compute @workgroup_size(\n  16,\n  8,\n  1,\n)");
    expect(source).toContain("const INPUT_CHANNEL_CHUNKS: u32 = 16u;");
    expect(source).toContain("chunk_channel * 17u +");
    expect(source).toContain("local.y * 65u");
    expect(source).toContain("output_range.first_output / OUTPUT_CHANNELS");
    expect(source.indexOf("var kernel = 0u")).toBeLessThan(
      source.indexOf("var input_channel_chunk = 0u"),
    );
    expect(source.indexOf("var input_channel_chunk = 0u")).toBeLessThan(
      source.indexOf("var chunk_channel = 0u"),
    );
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toContain("return;");
    expect(Object.isFrozen(plan)).toBe(true);
  });

  it("selects K7 d1/d3/d9 complete rows under the production limits", () => {
    for (const dilation of [1, 3, 9] as const) {
      expect(selectAceChannelChunkedVaeConv1d(
        limits(),
        { ...shape(), dilation, padding: dilation * 3 },
        range(),
      )).toMatchObject({
        eligible: true,
        reason: "eligible",
        firstOutputRow: 0,
        outputRowCount: 2_048,
        workgroupsX: 128,
        workgroupsY: 128,
        plan: { workgroupStorageBytes: 6_432 },
      });
    }
    expect(planAceChannelChunkedVaeConv1d({
      ...shape(),
      inputChannels: 65,
    }).inputChannelChunkCount).toBe(2);
  });

  it("fails closed for unsupported math, malformed ranges, and limits", () => {
    expect(selectAceChannelChunkedVaeConv1d(limits(), {
      ...shape(),
      kernelSize: 1,
      padding: 0,
    }, range())).toMatchObject({
      eligible: false,
      reason: "unsupported-math",
    });
    expect(selectAceChannelChunkedVaeConv1d(limits(), {
      ...shape(),
      dilation: 2,
      padding: 6,
    }, range())).toMatchObject({
      eligible: false,
      reason: "unsupported-math",
    });
    expect(selectAceChannelChunkedVaeConv1d(limits(), shape(), {
      base: 1,
      count: 1_024,
    })).toMatchObject({ eligible: false, reason: "unaligned-range" });
    expect(selectAceChannelChunkedVaeConv1d(limits(), {
      ...shape(),
      batch: 2,
    }, {
      base: (4_097 - 1) * 1_024,
      count: 2 * 1_024,
    })).toMatchObject({
      eligible: false,
      reason: "batch-crossing-range",
    });
    expect(selectAceChannelChunkedVaeConv1d({
      ...limits(),
      maxComputeWorkgroupStorageSize: 6_400,
    }, shape(), range())).toMatchObject({
      eligible: false,
      reason: "workgroup-storage",
    });
    expect(selectAceChannelChunkedVaeConv1d({
      ...limits(),
      maxComputeInvocationsPerWorkgroup: 64,
    }, shape(), range())).toMatchObject({
      eligible: false,
      reason: "unsupported-workgroup",
    });
    expect(selectAceChannelChunkedVaeConv1d({
      ...limits(),
      maxStorageBufferBindingSize: 1_024,
    }, shape(), range())).toMatchObject({
      eligible: false,
      reason: "storage-binding-limit",
    });
  });

  it("encodes one existing cooperative range with its dynamic control offset", async () => {
    const device = fakeDevice();
    const kernel = AceChannelChunkedVaeConv1dKernel.create(device);
    const dispatch = await kernel.createDispatch(
      "production-k7-d3",
      shape(),
      bindings(),
      controlRange(),
    );
    expect(dispatch).toMatchObject({
      firstOutputRow: 0,
      outputRowCount: 2_048,
      outputRange: range(),
    });
    const layout = device.createBindGroupLayout.mock.calls[0]?.[0] as
      GPUBindGroupLayoutDescriptor;
    expect(Array.from(layout.entries).at(-1)?.buffer).toEqual({
      type: "uniform",
      hasDynamicOffset: true,
      minBindingSize: 16,
    });
    const group = device.createBindGroup.mock.calls[0]?.[0] as
      GPUBindGroupDescriptor;
    const rangeResource = Array.from(group.entries).at(-1)?.resource as
      GPUBufferBinding;
    expect(rangeResource).toMatchObject({ offset: 0, size: 16 });

    const pass = fakePass();
    dispatch.encode(pass);
    expect(pass.setBindGroup).toHaveBeenCalledWith(0, expect.anything(), [256]);
    expect(pass.dispatchWorkgroups).toHaveBeenCalledWith(128, 128, 1);

    await kernel.createDispatch(
      "reuse-next-range",
      shape(),
      bindingsForReuse,
      { ...controlRangeForReuse, control: { ...controlRangeForReuse.control, offset: 0 } },
    );
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    kernel.destroy();
    await expect(kernel.createDispatch(
      "destroyed",
      shape(),
      bindings(),
      controlRange(),
    )).rejects.toThrow(/destroyed/);
  });

  it("caches bind groups, evicts compile failures, and rejects aliases", async () => {
    const failure = new Error("synthetic compile failure");
    const device = fakeDevice({
      pipelineResults: [Promise.reject(failure), Promise.resolve(fakePipeline())],
    });
    const kernel = AceChannelChunkedVaeConv1dKernel.create(device);
    await expect(kernel.createDispatch(
      "fails",
      shape(),
      bindingsForReuse,
      controlRangeForReuse,
    )).rejects.toBe(failure);
    await kernel.createDispatch(
      "retry",
      shape(),
      bindingsForReuse,
      controlRangeForReuse,
    );
    await kernel.createDispatch(
      "reuse",
      shape(),
      bindingsForReuse,
      controlRangeForReuse,
    );
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createBindGroup).toHaveBeenCalledOnce();

    const planned = planAceChannelChunkedVaeConv1d(shape());
    const shared = fakeBuffer(planned.outputElements * 4 + 4_096);
    await expect(kernel.createDispatch("alias", shape(), {
      ...bindingsForReuse,
      input: { buffer: shared, offset: 0, size: planned.inputElements * 4 },
      output: { buffer: shared, offset: 1_024, size: planned.outputElements * 4 },
    }, controlRangeForReuse)).rejects.toThrow(/must not overlap/);
  });

  it("rejects malformed control bindings before compiling", async () => {
    const device = fakeDevice();
    const kernel = AceChannelChunkedVaeConv1dKernel.create(device);
    await expect(kernel.createDispatch(
      "unaligned-control",
      shape(),
      bindingsForReuse,
      {
        ...controlRangeForReuse,
        control: { ...controlRangeForReuse.control, offset: 4 },
      },
    )).rejects.toThrow(/uniform-buffer alignment/);
    expect(device.createShaderModule).not.toHaveBeenCalled();
  });
});

expect(ACE_CHANNEL_CHUNKED_VAE_CONV1D_INPUT_CHANNEL_CHUNK).toBe(64);
expect(ACE_CHANNEL_CHUNKED_VAE_CONV1D_WORKGROUP_SIZE).toBe(128);

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

function shape(): AceVaeConv1dShape {
  return {
    batch: 1,
    inputFrames: 4_097,
    inputChannels: 1_024,
    outputChannels: 1_024,
    kernelSize: 7,
    stride: 1,
    dilation: 3,
    padding: 9,
  };
}

function range(): Readonly<{ base: number; count: number }> {
  return { base: 0, count: 2_048 * 1_024 };
}

function limits(): Pick<
  GPUSupportedLimits,
  | "maxComputeInvocationsPerWorkgroup"
  | "maxComputeWorkgroupSizeX"
  | "maxComputeWorkgroupSizeY"
  | "maxComputeWorkgroupStorageSize"
  | "maxComputeWorkgroupsPerDimension"
  | "maxStorageBufferBindingSize"
> {
  return {
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupStorageSize: 16 * 1_024,
    maxComputeWorkgroupsPerDimension: 65_535,
    maxStorageBufferBindingSize: 256 * 1_024 * 1_024,
  };
}

function controlRange(): AceVaeOutputRangeBinding {
  return {
    ...range(),
    control: { buffer: fakeBuffer(512), offset: 256, size: 16 },
  };
}

const plannedForReuse = planAceChannelChunkedVaeConv1d(shape());
const bindingsForReuse = makeBindings(plannedForReuse);
const controlRangeForReuse: AceVaeOutputRangeBinding = {
  ...range(),
  control: { buffer: fakeBuffer(512), offset: 256, size: 16 },
};

function bindings(): AceVaeConvBindings {
  return makeBindings(planAceChannelChunkedVaeConv1d(shape()));
}

function makeBindings(
  plan: AceChannelChunkedVaeConv1dPlanForTest,
): AceVaeConvBindings {
  return {
    input: fakeBinding(plan.inputElements * 4),
    weight: fakeBinding(plan.weightElements * 4),
    bias: fakeBinding(plan.outputChannels * 4),
    output: fakeBinding(plan.outputElements * 4),
  };
}

type AceChannelChunkedVaeConv1dPlanForTest = ReturnType<
  typeof planAceChannelChunkedVaeConv1d
>;

interface FakeDeviceDiagnostics {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createBindGroupLayout: ReturnType<typeof vi.fn>;
  readonly createPipelineLayout: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
}

type FakeDevice = GPUDevice & FakeDeviceDiagnostics;

function fakeDevice(options: {
  readonly pipelineResults?: readonly Promise<GPUComputePipeline>[];
} = {}): FakeDevice {
  const pipelineResults = [...(options.pipelineResults ?? [
    Promise.resolve(fakePipeline()),
  ])];
  return {
    limits: { ...limits(), minUniformBufferOffsetAlignment: 256 },
    createShaderModule: vi.fn(() => ({
      label: "module",
      getCompilationInfo: vi.fn(async () => ({ messages: [] })),
    })),
    createBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    createPipelineLayout: vi.fn(() => ({ label: "pipeline-layout" })),
    createComputePipelineAsync: vi.fn(() => {
      const result = pipelineResults.shift();
      if (result === undefined) throw new Error("fake pipelines exhausted");
      return result;
    }),
    createBindGroup: vi.fn(() => ({ label: "bind-group" })),
  } as unknown as FakeDevice;
}

function fakePipeline(): GPUComputePipeline {
  return { label: "pipeline" } as GPUComputePipeline;
}

function fakePass(): GPUComputePassEncoder & {
  readonly setBindGroup: ReturnType<typeof vi.fn>;
  readonly dispatchWorkgroups: ReturnType<typeof vi.fn>;
} {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
  } as unknown as GPUComputePassEncoder & {
    readonly setBindGroup: ReturnType<typeof vi.fn>;
    readonly dispatchWorkgroups: ReturnType<typeof vi.fn>;
  };
}

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: fakeBuffer(size), offset: 0, size };
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}
