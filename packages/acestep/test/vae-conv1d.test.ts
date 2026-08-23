import { describe, expect, it, vi } from "vitest";

import {
  ACE_TILED_VAE_CONV1D_INPUT_TILE_STRIDE,
  ACE_TILED_VAE_CONV1D_WORKGROUP_SIZE,
  AceTiledVaeConv1dKernel,
  aceTiledVaeConv1dWgsl,
  planAceTiledVaeConv1d,
  selectAceTiledVaeConv1d,
} from "../src/webgpu/kernels/vae-conv1d.js";
import {
  AceChannelChunkedVaeConv1dKernel,
  planAceChannelChunkedVaeConv1d,
} from "../src/webgpu/kernels/vae-conv1d-channel-chunks.js";
import type {
  AceVaeConv1dShape,
  AceVaeConvBindings,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";
import { AceCorrectnessVaePrimitiveKernel } from
  "../src/webgpu/kernels/vae-primitives.js";
import {
  AceCorrectnessVaeDecoderRuntime,
  planAceVaeDecoder,
  type AceVaeDecoderConfig,
  type AceVaeDecoderBindings,
} from "../src/webgpu/vae-decoder.js";

describe("production tiled FP32 VAE Conv1D", () => {
  it("plans the proven M3 layout without changing package weight order", () => {
    const plan = planAceTiledVaeConv1d(shape());
    expect(plan).toMatchObject({
      outputFrames: 4_097,
      inputElements: 524_416,
      weightElements: 114_688,
      outputElements: 524_416,
      inputTileElements: 2_944,
      weightTileElements: 1_032,
      inputTileBytes: 11_776,
      weightTileBytes: 4_128,
      workgroupStorageBytes: 15_904,
      weightTileStride: 129,
    });
    const source = aceTiledVaeConv1dWgsl(shape(), true);
    expect(source).toContain("@compute @workgroup_size(\n  16,\n  8,\n  1,\n)");
    expect(source).toContain("input_channel * 23u + tile_time");
    expect(source).toContain("tile_output_channel * 129u + input_channel");
    expect(source).toContain(
      "(weight_output_channel * 7u +\n            kernel) * INPUT_CHANNELS",
    );
    expect(source).toContain("output_range.first_output / OUTPUT_CHANNELS");
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(3);
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toContain("return;");
    expect(Object.isFrozen(plan)).toBe(true);
  });

  it("selects only complete, batch-contained rows within device limits", () => {
    expect(selectAceTiledVaeConv1d(limits(), shape(), {
      base: 128,
      count: 2_048 * 128,
    })).toMatchObject({
      eligible: true,
      reason: "eligible",
      firstOutputRow: 1,
      outputRowCount: 2_048,
      workgroupsX: 128,
      workgroupsY: 16,
    });
    expect(selectAceTiledVaeConv1d(limits(), shape(), {
      base: 1,
      count: 128,
    })).toMatchObject({ eligible: false, reason: "unaligned-range" });
    expect(selectAceTiledVaeConv1d(limits(), {
      ...shape(),
      batch: 2,
    }, {
      base: (4_097 - 1) * 128,
      count: 2 * 128,
    })).toMatchObject({ eligible: false, reason: "batch-crossing-range" });
  });

  it("fails closed for unsupported math and high-channel storage", () => {
    expect(selectAceTiledVaeConv1d(limits(), {
      ...shape(),
      dilation: 3,
      padding: 9,
    }, range())).toMatchObject({
      eligible: false,
      reason: "unsupported-math",
    });
    expect(selectAceTiledVaeConv1d(limits(), {
      ...shape(),
      inputChannels: 1_024,
    }, range())).toMatchObject({
      eligible: false,
      reason: "workgroup-storage",
      plan: { workgroupStorageBytes: 127_008 },
    });
    expect(selectAceTiledVaeConv1d({
      ...limits(),
      maxComputeInvocationsPerWorkgroup: 64,
    }, shape(), range())).toMatchObject({
      eligible: false,
      reason: "unsupported-workgroup",
    });
  });

  it("pins the existing production 16 KiB device contract selection boundary", () => {
    const production = { ...limits(), maxComputeWorkgroupStorageSize: 16_384 };
    for (const [inputChannels, expected] of [
      [64, true],
      [128, true],
      [256, false],
      [512, false],
      [1_024, false],
    ] as const) {
      expect(selectAceTiledVaeConv1d(production, {
        ...shape(),
        inputChannels,
      }, range()).eligible).toBe(expected);
    }
  });

  it("encodes one existing OPT-0002 range with its dynamic control offset", async () => {
    const device = fakeDevice();
    const kernel = AceTiledVaeConv1dKernel.create(device);
    const dispatch = await kernel.createDispatch(
      "production-k7",
      shape(),
      bindings(),
      controlRange(),
    );
    expect(dispatch).toMatchObject({
      firstOutputRow: 0,
      outputRowCount: 2_048,
      outputRange: range(),
    });
    expect(device.createBindGroupLayout).toHaveBeenCalledOnce();
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
    expect(rangeResource.offset).toBe(0);
    expect(rangeResource.size).toBe(16);

    const pass = fakePass();
    dispatch.encode(pass);
    expect(pass.setBindGroup).toHaveBeenCalledWith(
      0,
      expect.anything(),
      [256],
    );
    expect(pass.dispatchWorkgroups).toHaveBeenCalledWith(128, 16, 1);
    kernel.destroy();
    await expect(kernel.createDispatch(
      "destroyed",
      shape(),
      bindings(),
      controlRange(),
    )).rejects.toThrow(/destroyed/);
  });

  it("caches pipelines/bind groups, evicts failures, and validates aliases", async () => {
    const failure = new Error("synthetic compile failure");
    const device = fakeDevice({
      pipelineResults: [Promise.reject(failure), Promise.resolve(fakePipeline())],
    });
    const kernel = AceTiledVaeConv1dKernel.create(device);
    const stableBindings = bindings();
    const stableRange = controlRange();
    await expect(kernel.createDispatch(
      "fails",
      shape(),
      stableBindings,
      stableRange,
    )).rejects.toBe(failure);
    await kernel.createDispatch("retry", shape(), stableBindings, stableRange);
    await kernel.createDispatch("reuse", shape(), stableBindings, stableRange);
    await kernel.createDispatch("next-offset", shape(), stableBindings, {
      ...stableRange,
      control: { ...stableRange.control, offset: 0 },
    });
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createBindGroup).toHaveBeenCalledOnce();

    const bytes = shape().inputFrames * shape().inputChannels * 4;
    const shared = fakeBuffer(bytes + 2_000_000);
    await expect(kernel.createDispatch("alias", shape(), {
      ...stableBindings,
      input: { buffer: shared, offset: 0, size: bytes },
      output: { buffer: shared, offset: 1_024, size: bytes },
    }, controlRange())).rejects.toThrow(/must not overlap/);
  });

  it("rejects malformed control bindings without compiling", async () => {
    const device = fakeDevice();
    const kernel = AceTiledVaeConv1dKernel.create(device);
    await expect(kernel.createDispatch(
      "unaligned-control",
      shape(),
      bindings(),
      {
        ...controlRange(),
        control: { ...controlRange().control, offset: 4 },
      },
    )).rejects.toThrow(/uniform-buffer alignment/);
    expect(device.createShaderModule).not.toHaveBeenCalled();
  });

  it("preserves FIFO quanta while selecting tiled math and forcing portable", async () => {
    const device = fakeRuntimeDevice();
    const config: AceVaeDecoderConfig = Object.freeze({
      id: "tiled-selection-toy",
      decoderInputChannels: 2,
      decoderChannels: 1,
      audioChannels: 2,
      channelMultiples: Object.freeze([1]),
      downsamplingRatios: Object.freeze([2]),
      sampleRateHz: 48_000,
    });
    const graph = planAceVaeDecoder(3, config);
    const bindings = decoderBindings(graph);
    const creationEvents: string[] = [];
    const tiledDispatch = vi.spyOn(
      AceTiledVaeConv1dKernel.prototype,
      "createDispatch",
    ).mockImplementation(async (label, plannedShape, _bindings, plannedRange) => {
      creationEvents.push(`tiled:${label}`);
      return {
        label,
        plan: planAceTiledVaeConv1d(plannedShape),
        outputRange: { base: plannedRange.base, count: plannedRange.count },
        firstOutputRow: plannedRange.base / plannedShape.outputChannels,
        outputRowCount: plannedRange.count / plannedShape.outputChannels,
        encode: vi.fn(),
      };
    });
    const portableConv = vi.spyOn(
      AceCorrectnessVaePrimitiveKernel.prototype,
      "createConv1dDispatch",
    ).mockImplementation(async (label) => {
      creationEvents.push(`portable:${label}`);
      return fakePrimitiveDispatch(label);
    });
    const channelChunkedDispatch = vi.spyOn(
      AceChannelChunkedVaeConv1dKernel.prototype,
      "createDispatch",
    ).mockImplementation(async (label, plannedShape, _bindings, plannedRange) => {
      creationEvents.push(`channel-chunked:${label}`);
      return {
        label,
        plan: planAceChannelChunkedVaeConv1d(plannedShape),
        outputRange: { base: plannedRange.base, count: plannedRange.count },
        firstOutputRow: plannedRange.base / plannedShape.outputChannels,
        outputRowCount: plannedRange.count / plannedShape.outputChannels,
        encode: vi.fn(),
      };
    });
    const transpose = vi.spyOn(
      AceCorrectnessVaePrimitiveKernel.prototype,
      "createConvTranspose1dPartDispatch",
    ).mockImplementation(async (label) => fakePrimitiveDispatch(label));
    const snake = vi.spyOn(
      AceCorrectnessVaePrimitiveKernel.prototype,
      "createSnakeDispatch",
    ).mockImplementation(async (label) => fakePrimitiveDispatch(label));
    const add = vi.spyOn(
      AceCorrectnessVaePrimitiveKernel.prototype,
      "createAddDispatch",
    ).mockImplementation(async (label) => fakePrimitiveDispatch(label));
    const tiledDestroy = vi.spyOn(
      AceTiledVaeConv1dKernel.prototype,
      "destroy",
    );
    const channelChunkedDestroy = vi.spyOn(
      AceChannelChunkedVaeConv1dKernel.prototype,
      "destroy",
    );
    const runtime = AceCorrectnessVaeDecoderRuntime.create(device);
    try {
      const optimized = await runtime.createDecoderDispatch(
        "optimized",
        3,
        bindings,
        config,
        1,
        {
          quantumWorkPolicy: {
            maximumConvolutionMultiplyAccumulates: 1_000_000,
            maximumOutputElements: 2,
          },
        },
      );
      const portable = await runtime.createDecoderDispatch(
        "portable",
        3,
        bindings,
        config,
        1,
        {
          quantumWorkPolicy: {
            maximumConvolutionMultiplyAccumulates: 1_000_000,
            maximumOutputElements: 2,
          },
          conv1dProfile: "portable",
        },
      );
      expect(optimized.quanta.map(({ operationIndex, logicalOutputBase }) =>
        [operationIndex, logicalOutputBase]
      )).toEqual(portable.quanta.map(({ operationIndex, logicalOutputBase }) =>
        [operationIndex, logicalOutputBase]
      ));
      expect(optimized.conv1dSelection).toMatchObject({
        profile: "optimized-when-eligible",
        tiledOperationLabels: [
          "conv1",
          "block-0-res-1-conv1",
          "conv2",
        ],
        channelChunkedOperationLabels: [
          "block-0-res-2-conv1",
          "block-0-res-3-conv1",
        ],
        portableOperationLabels: [
          "block-0-res-1-conv2",
          "block-0-res-2-conv2",
          "block-0-res-3-conv2",
        ],
      });
      expect(optimized.conv1dSelection?.fallbackReasons).toEqual({
        "tiled:unsupported-math;channel-chunked:unsupported-math": 9,
      });
      expect(portable.conv1dSelection).toMatchObject({
        profile: "portable",
        tiledQuantumCount: 0,
        channelChunkedQuantumCount: 0,
        fallbackReasons: { "profile-portable": 26 },
      });
      expect(creationEvents[0]).toContain("tiled:optimized-operation-0-conv1");
      expect(creationEvents.some((event) =>
        event.startsWith("channel-chunked:optimized-") &&
        event.includes("block-0-res-2-conv1")
      )).toBe(true);
    } finally {
      runtime.destroy();
      tiledDispatch.mockRestore();
      channelChunkedDispatch.mockRestore();
      portableConv.mockRestore();
      transpose.mockRestore();
      snake.mockRestore();
      add.mockRestore();
    }
    expect(tiledDestroy).toHaveBeenCalledOnce();
    tiledDestroy.mockRestore();
    expect(channelChunkedDestroy).toHaveBeenCalledOnce();
    channelChunkedDestroy.mockRestore();
  });
});

expect(ACE_TILED_VAE_CONV1D_INPUT_TILE_STRIDE).toBe(23);
expect(ACE_TILED_VAE_CONV1D_WORKGROUP_SIZE).toBe(128);

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });
vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1 << 6, COPY_DST: 1 << 3 });

function shape(): AceVaeConv1dShape {
  return {
    batch: 1,
    inputFrames: 4_097,
    inputChannels: 128,
    outputChannels: 128,
    kernelSize: 7,
    stride: 1,
    dilation: 1,
    padding: 3,
  };
}

function range(): Readonly<{ base: number; count: number }> {
  return { base: 0, count: 2_048 * 128 };
}

function controlRange(): AceVaeOutputRangeBinding {
  return {
    ...range(),
    control: {
      buffer: fakeBuffer(512),
      offset: 256,
      size: 16,
    },
  };
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
    maxComputeInvocationsPerWorkgroup: 1_024,
    maxComputeWorkgroupSizeX: 1_024,
    maxComputeWorkgroupSizeY: 1_024,
    maxComputeWorkgroupStorageSize: 32_768,
    maxComputeWorkgroupsPerDimension: 65_535,
    maxStorageBufferBindingSize: 0xffff_fffc,
  };
}

function bindings(): AceVaeConvBindings {
  const planned = planAceTiledVaeConv1d(shape());
  return {
    input: fakeBinding(planned.inputElements * 4),
    weight: fakeBinding(planned.weightElements * 4),
    bias: fakeBinding(planned.outputChannels * 4),
    output: fakeBinding(planned.outputElements * 4),
  };
}

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
    limits: {
      ...limits(),
      minUniformBufferOffsetAlignment: 256,
    },
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

function fakePrimitiveDispatch(label: string) {
  return {
    label,
    plan: { workgroupsX: 1, workgroupsY: 1 },
    outputRange: { base: 0, count: 1 },
    encode: vi.fn(),
  } as never;
}

function fakeRuntimeDevice(): GPUDevice {
  let nextBuffer = 0;
  return {
    limits: {
      ...limits(),
      maxComputeWorkgroupStorageSize: 16 * 1_024,
      maxBufferSize: 256 * 1_024 * 1_024,
      maxUniformBufferBindingSize: 64 * 1_024,
      minUniformBufferOffsetAlignment: 256,
    },
    queue: { writeBuffer: vi.fn() },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => null),
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => ({
      label: `range-${nextBuffer++}`,
      size: descriptor.size,
      destroy: vi.fn(),
    })),
  } as unknown as GPUDevice;
}

function decoderBindings(
  graph: ReturnType<typeof planAceVaeDecoder>,
): AceVaeDecoderBindings {
  const binding = (size: number): GPUBufferBinding => ({
    buffer: fakeBuffer(size),
    offset: 0,
    size,
  });
  const tensorBytes = Math.max(graph.parameterBytes, 4);
  return {
    input: binding(graph.inputElements * 4),
    output: binding(graph.outputElements * 4),
    workspaces: [
      binding(graph.workspaceBytes),
      binding(graph.workspaceBytes),
      binding(graph.workspaceBytes),
    ],
    tensors: Object.fromEntries(
      graph.requiredTensorNames.map((name) => [name, binding(tensorBytes)]),
    ),
  };
}
