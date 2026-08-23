import { describe, expect, it, vi } from "vitest";
import type {
  AceVaeConv1dShape,
  AceVaeConvBindings,
} from "../src/webgpu/kernels/vae-primitives.js";

import {
  ACE_OPT_0005_VAE_CONV1D_INPUT_CHANNEL_CHUNK,
  ACE_OPT_0005_VAE_CONV1D_INPUT_TILE_STRIDE,
  ACE_OPT_0005_VAE_CONV1D_KERNEL_SIZE,
  ACE_OPT_0005_VAE_CONV1D_SUPPORTED_DILATIONS,
  ACE_OPT_0005_VAE_CONV1D_TILE_CHANNELS,
  ACE_OPT_0005_VAE_CONV1D_TILE_FRAMES,
  ACE_OPT_0005_VAE_CONV1D_WEIGHT_TILE_STRIDE,
  ACE_OPT_0005_VAE_CONV1D_WORKGROUP_SIZE,
  ACE_OPT_0005_VAE_CONV1D_WORKGROUP_SIZE_X,
  ACE_OPT_0005_VAE_CONV1D_WORKGROUP_SIZE_Y,
  AceOpt0005VaeConv1dKernel,
  aceOpt0005VaeConv1dWgsl,
  planAceOpt0005VaeConv1d,
} from "../benchmark/opt-0005-vae-conv1d.js";

describe("OPT-0005 channel-chunked FP32 VAE Conv1D candidate", () => {
  it("plans the principal C1024 operation in exact 32-row ranges using 6,432 bytes", () => {
    const plan = planAceOpt0005VaeConv1d(principalShape());
    expect(plan).toMatchObject({
      batch: 1,
      inputFrames: 2_560,
      outputFrames: 2_560,
      inputChannels: 1_024,
      outputChannels: 1_024,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 3,
      inputElements: 2_621_440,
      weightElements: 7_340_032,
      outputElements: 2_621_440,
      inputChannelChunk: 64,
      inputChannelChunkCount: 16,
      tileFrames: 16,
      tileChannels: 8,
      inputTileStride: 17,
      weightTileStride: 65,
      workgroupSizeX: 16,
      workgroupSizeY: 8,
      workgroupSize: 128,
      inputTileElements: 1_088,
      weightTileElements: 520,
      inputTileBytes: 4_352,
      weightTileBytes: 2_080,
      workgroupStorageBytes: 6_432,
      outputRangeCount: 80,
    });
    expect(plan.outputRanges[0]).toEqual({
      batch: 0,
      firstOutputTime: 0,
      firstOutputRow: 0,
      outputRowCount: 32,
      firstOutput: 0,
      outputCount: 32_768,
      workgroupsX: 2,
      workgroupsY: 128,
      workgroupCount: 256,
      multiplyAdds: 234_881_024,
    });
    expect(plan.outputRanges.at(-1)).toEqual({
      batch: 0,
      firstOutputTime: 2_528,
      firstOutputRow: 2_528,
      outputRowCount: 32,
      firstOutput: 2_588_672,
      outputCount: 32_768,
      workgroupsX: 2,
      workgroupsY: 128,
      workgroupCount: 256,
      multiplyAdds: 234_881_024,
    });
    expect(plan.outputRanges.reduce(
      (sum, range) => sum + range.outputCount,
      0,
    )).toBe(plan.outputElements);
    expect(Object.isFrozen(plan)).toBe(true);
    expect(Object.isFrozen(plan.outputRanges)).toBe(true);
    expect(plan.outputRanges.every(Object.isFrozen)).toBe(true);
  });

  it.each([
    [1, 3],
    [3, 9],
    [9, 27],
  ] as const)("accepts K7 dilation %i with same-length padding %i", (
    dilation,
    padding,
  ) => {
    const shape = {
      ...tailShape(65),
      dilation,
      padding,
    };
    const plan = planAceOpt0005VaeConv1d(shape);
    expect(plan.outputFrames).toBe(shape.inputFrames);
    expect(plan.inputChannelChunkCount).toBe(2);
    expect(plan.workgroupStorageBytes).toBe(6_432);
    const source = aceOpt0005VaeConv1dWgsl(shape, true);
    expect(source).toContain(`const DILATION: u32 = ${dilation}u;`);
    expect(source).toContain("var kernel = 0u;");
    expect(source).toContain("var input_channel_chunk = 0u;");
  });

  it("covers C63, C64, and C65 without changing the fixed shared geometry", () => {
    for (const [inputChannels, chunks] of [
      [63, 1],
      [64, 1],
      [65, 2],
    ] as const) {
      const plan = planAceOpt0005VaeConv1d(tailShape(inputChannels));
      expect(plan.inputChannelChunkCount).toBe(chunks);
      expect(plan.inputTileElements).toBe(64 * 17);
      expect(plan.weightTileElements).toBe(8 * 65);
      expect(plan.workgroupStorageBytes).toBe(6_432);
    }
    const source = aceOpt0005VaeConv1dWgsl(tailShape(65), false);
    expect(source).toContain("const INPUT_CHANNEL_CHUNKS: u32 = 2u;");
    expect(source).toContain("input_channel < INPUT_CHANNELS");
    expect(source).toContain("INPUT_CHANNELS - chunk_first_channel");
  });

  it("keeps generic tails on complete-channel rows and inside one batch", () => {
    const plan = planAceOpt0005VaeConv1d({
      ...tailShape(65),
      batch: 2,
      inputFrames: 33,
      outputChannels: 10,
      dilation: 9,
      padding: 27,
    });
    expect(plan.outputRanges).toEqual([
      {
        batch: 0,
        firstOutputTime: 0,
        firstOutputRow: 0,
        outputRowCount: 33,
        firstOutput: 0,
        outputCount: 330,
        workgroupsX: 3,
        workgroupsY: 2,
        workgroupCount: 6,
        multiplyAdds: 150_150,
      },
      {
        batch: 1,
        firstOutputTime: 0,
        firstOutputRow: 33,
        outputRowCount: 33,
        firstOutput: 330,
        outputCount: 330,
        workgroupsX: 3,
        workgroupsY: 2,
        workgroupCount: 6,
        multiplyAdds: 150_150,
      },
    ]);
    for (const range of plan.outputRanges) {
      expect(range.firstOutput % plan.outputChannels).toBe(0);
      expect(range.outputCount % plan.outputChannels).toBe(0);
      expect(range.firstOutputTime + range.outputRowCount)
        .toBeLessThanOrEqual(plan.outputFrames);
    }
  });

  it("emits K-then-chunk-then-channel FP32 adds and skips whole padded chunks", () => {
    const source = aceOpt0005VaeConv1dWgsl(tailShape(65), true);
    expect(source).toContain("var<workgroup> input_tile: array<f32, 1088>;");
    expect(source).toContain("var<workgroup> weight_tile: array<f32, 520>;");
    expect(source).toContain("@compute @workgroup_size(\n  16,\n  8,\n  1,\n)");
    expect(source).toContain("chunk_channel * 17u +\n          tile_time");
    expect(source).toContain("tile_output_channel * 65u +\n          chunk_channel");
    expect(source).toContain("local.y * 65u");
    expect(source).toContain(
      "(weight_output_channel * 7u + kernel) * INPUT_CHANNELS",
    );
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    const kernel = source.indexOf("var kernel = 0u;");
    const chunk = source.indexOf("var input_channel_chunk = 0u;");
    const channel = source.indexOf("var chunk_channel = 0u;", chunk);
    const paddedGuard = source.indexOf(
      "if (padded_time >= PADDING)",
      chunk,
    );
    expect(kernel).toBeLessThan(chunk);
    expect(chunk).toBeLessThan(channel);
    expect(paddedGuard).toBeLessThan(channel);
    expect(source.indexOf("sum = bias[output_channel];")).toBeLessThan(kernel);
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toContain("return;");
    expect(source).not.toContain("enable f16");
  });

  it("uses the no-bias binding layout without changing contraction order", () => {
    const source = aceOpt0005VaeConv1dWgsl(tailShape(63), false);
    expect(source).toContain(
      "@group(0) @binding(2) var<storage, read_write>",
    );
    expect(source).toContain("@group(0) @binding(3) var<uniform>");
    expect(source).toContain("if (output_active) { sum = 0.0; }");
    expect(source).not.toContain("var<storage, read> bias");
  });

  it.each([
    [{ ...tailShape(65), kernelSize: 5 }, /kernel size 7/],
    [{ ...tailShape(65), stride: 2 }, /stride 1 and dilation 1, 3, or 9/],
    [{ ...tailShape(65), dilation: 2 }, /stride 1 and dilation 1, 3, or 9/],
    [{ ...tailShape(65), batch: 0 }, /positive safe integer/],
    [{
      ...tailShape(65),
      inputFrames: 7,
      outputChannels: 8 * 65_535 + 1,
    }, /dispatch domain/],
    [{
      ...tailShape(4_000_000),
      inputFrames: 7,
    }, /complete-channel output row/],
  ])("rejects unsupported or unbounded geometry", (shape, reason) => {
    expect(() => planAceOpt0005VaeConv1d(shape)).toThrow(reason);
  });

  it("fails closed when an inactive tail tile's staged tap would wrap u32", () => {
    expect(() => planAceOpt0005VaeConv1d({
      batch: 1,
      inputFrames: 1,
      inputChannels: 1,
      outputChannels: 1,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 2_147_483_643,
    })).toThrow(/last staged input time exceeds WGSL's u32 indexing domain/);
  });

  it("fails closed on insufficient workgroup, storage, and dispatch limits", async () => {
    expect(() => AceOpt0005VaeConv1dKernel.create(fakeDevice({
      maximumWorkgroupSizeX: 8,
    }))).toThrow(/16x8/);
    expect(() => AceOpt0005VaeConv1dKernel.create(fakeDevice({
      maximumWorkgroupSizeY: 4,
    }))).toThrow(/16x8/);
    expect(() => AceOpt0005VaeConv1dKernel.create(fakeDevice({
      maximumInvocations: 64,
    }))).toThrow(/128-lane/);

    const storageDevice = fakeDevice({ maximumWorkgroupStorage: 6_431 });
    const storageKernel = AceOpt0005VaeConv1dKernel.create(storageDevice);
    await expect(storageKernel.createDispatch(
      "storage",
      principalShape(),
      bindingsFor(principalShape()),
    )).rejects.toThrow(/6432 workgroup-storage bytes/);
    expect(storageDevice.createShaderModule).not.toHaveBeenCalled();

    const dispatchDevice = fakeDevice({ maximumDispatch: 1 });
    const dispatchKernel = AceOpt0005VaeConv1dKernel.create(dispatchDevice);
    await expect(dispatchKernel.createDispatch(
      "dispatch",
      principalShape(),
      bindingsFor(principalShape()),
    )).rejects.toThrow(/dispatch dimension/);
    expect(dispatchDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("compiles once, binds once, and encodes ranges with dynamic offsets", async () => {
    const device = fakeDevice();
    const kernel = AceOpt0005VaeConv1dKernel.create(device);
    const shape = { ...principalShape(), inputFrames: 65 };
    const stableBindings = bindingsFor(shape);
    const dispatch = await kernel.createDispatch("candidate", shape, stableBindings);
    const reused = await kernel.createDispatch("reuse", shape, stableBindings);
    expect(dispatch.rangeCount).toBe(3);
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();
    const controls = device.createdBuffers[0]!;
    expect(Array.from(new Uint32Array(controls.mapped, 0, 2))).toEqual([0, 32]);
    expect(Array.from(new Uint32Array(controls.mapped, 256, 2))).toEqual([32, 32]);
    expect(Array.from(new Uint32Array(controls.mapped, 512, 2))).toEqual([64, 1]);
    const layout = device.createBindGroupLayout.mock.calls[0]?.[0] as
      GPUBindGroupLayoutDescriptor;
    expect(Array.from(layout.entries).at(-1)?.buffer).toEqual({
      type: "uniform",
      hasDynamicOffset: true,
      minBindingSize: 16,
    });
    const group = device.createBindGroup.mock.calls[0]?.[0] as
      GPUBindGroupDescriptor;
    expect((Array.from(group.entries).at(-1)?.resource as GPUBufferBinding))
      .toMatchObject({ offset: 0, size: 16 });

    const pass = fakePass();
    dispatch.encodeRange(pass, 2);
    expect(pass.setBindGroup).toHaveBeenLastCalledWith(
      0,
      expect.anything(),
      [512],
    );
    expect(pass.dispatchWorkgroups).toHaveBeenLastCalledWith(1, 128, 1);
    expect(() => dispatch.encodeRange(pass, 3)).toThrow(/outside \[0, 3\)/);
    reused.encode(pass);
    expect(pass.dispatchWorkgroups).toHaveBeenCalledTimes(4);

    kernel.destroy();
    await Promise.resolve();
    expect(controls.destroy).toHaveBeenCalledOnce();
    await expect(kernel.createDispatch(
      "after-destroy",
      shape,
      stableBindings,
    )).rejects.toThrow(/was destroyed/);
  });

  it("uses distinct bias layouts and evicts a failed compilation", async () => {
    const failure = new Error("synthetic compile failure");
    const device = fakeDevice({
      pipelineResults: [
        Promise.reject(failure),
        Promise.resolve(fakePipeline()),
        Promise.resolve(fakePipeline()),
      ],
    });
    const kernel = AceOpt0005VaeConv1dKernel.create(device);
    const shape = tailShape(65);
    const noBias = bindingsFor(shape);
    await expect(kernel.createDispatch("failure", shape, noBias))
      .rejects.toBe(failure);
    await kernel.createDispatch("retry", shape, noBias);
    await kernel.createDispatch("bias", shape, {
      ...noBias,
      bias: fakeBinding(shape.outputChannels * 4),
    });
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(3);
    expect(device.createdBuffers).toHaveLength(2);
    const layouts = device.createBindGroupLayout.mock.calls.map(
      ([descriptor]) => Array.from(
        (descriptor as GPUBindGroupLayoutDescriptor).entries,
      ).length,
    );
    expect(layouts).toEqual([4, 4, 5]);
  });

  it("reports shader diagnostics and retries allocation failures cleanly", async () => {
    const diagnostic = fakeDevice({
      compilationMessages: [{
        message: "synthetic uniformity error",
        type: "error",
        lineNum: 42,
        linePos: 7,
      }],
    });
    const diagnosticKernel = AceOpt0005VaeConv1dKernel.create(diagnostic);
    await expect(diagnosticKernel.createDispatch(
      "shader-failure",
      tailShape(65),
      bindingsFor(tailShape(65)),
    )).rejects.toThrow(/42:7 synthetic uniformity error/);
    expect(diagnostic.createComputePipelineAsync).not.toHaveBeenCalled();
    expect(diagnostic.createdBuffers).toHaveLength(0);

    const allocationFailure = {
      message: "synthetic control allocation failure",
    } as GPUError;
    const allocation = fakeDevice({
      pipelineResults: [
        Promise.resolve(fakePipeline()),
        Promise.resolve(fakePipeline()),
      ],
      scopeResults: [allocationFailure, null, null, null, null, null],
    });
    const allocationKernel = AceOpt0005VaeConv1dKernel.create(allocation);
    const shape = tailShape(65);
    await expect(allocationKernel.createDispatch(
      "allocation-failure",
      shape,
      bindingsFor(shape),
    )).rejects.toThrow(/control allocation failure/);
    expect(allocation.createdBuffers[0]?.destroy).toHaveBeenCalledOnce();
    await expect(allocationKernel.createDispatch(
      "allocation-retry",
      shape,
      bindingsFor(shape),
    )).resolves.toMatchObject({ rangeCount: 1 });
    expect(allocation.pushErrorScope).toHaveBeenCalledTimes(6);
    expect(allocation.popErrorScope).toHaveBeenCalledTimes(6);
  });

  it("destroys controls that finish compiling after kernel destruction", async () => {
    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pending = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const device = fakeDevice({ pipelineResults: [pending] });
    const kernel = AceOpt0005VaeConv1dKernel.create(device);
    const shape = tailShape(65);
    const dispatch = kernel.createDispatch("pending", shape, bindingsFor(shape));
    kernel.destroy();
    resolvePipeline(fakePipeline());
    await expect(dispatch).rejects.toThrow(/destroyed/);
    await Promise.resolve();
    expect(device.createdBuffers[0]?.destroy).toHaveBeenCalledOnce();
  });

  it("rejects undersized, oversized-device, and overlapping bindings before compilation", async () => {
    const shape = tailShape(65);
    const device = fakeDevice();
    const kernel = AceOpt0005VaeConv1dKernel.create(device);
    const bindings = bindingsFor(shape);
    await expect(kernel.createDispatch("short", shape, {
      ...bindings,
      weight: fakeBinding(shape.outputChannels * 7 * shape.inputChannels * 4 - 4),
    })).rejects.toThrow(/weight binding/);

    const inputBytes = shape.batch * shape.inputFrames * shape.inputChannels * 4;
    const outputBytes = shape.batch * shape.inputFrames * shape.outputChannels * 4;
    const shared = fakeBuffer(inputBytes + outputBytes);
    await expect(kernel.createDispatch("alias", shape, {
      ...bindings,
      input: { buffer: shared, offset: 0, size: inputBytes },
      output: { buffer: shared, offset: 256, size: outputBytes },
    })).rejects.toThrow(/must not overlap/);
    expect(device.createShaderModule).not.toHaveBeenCalled();

    const bindingDevice = fakeDevice({ maximumStorageBinding: 1_024 });
    const bindingKernel = AceOpt0005VaeConv1dKernel.create(bindingDevice);
    await expect(bindingKernel.createDispatch(
      "binding-limit",
      shape,
      bindingsFor(shape),
    )).rejects.toThrow(/storage binding limit/);
    expect(bindingDevice.createShaderModule).not.toHaveBeenCalled();
  });
});

expect(ACE_OPT_0005_VAE_CONV1D_KERNEL_SIZE).toBe(7);
expect(ACE_OPT_0005_VAE_CONV1D_SUPPORTED_DILATIONS).toEqual([1, 3, 9]);
expect(ACE_OPT_0005_VAE_CONV1D_INPUT_CHANNEL_CHUNK).toBe(64);
expect(ACE_OPT_0005_VAE_CONV1D_TILE_FRAMES).toBe(16);
expect(ACE_OPT_0005_VAE_CONV1D_TILE_CHANNELS).toBe(8);
expect(ACE_OPT_0005_VAE_CONV1D_INPUT_TILE_STRIDE).toBe(17);
expect(ACE_OPT_0005_VAE_CONV1D_WEIGHT_TILE_STRIDE).toBe(65);
expect(ACE_OPT_0005_VAE_CONV1D_WORKGROUP_SIZE_X).toBe(16);
expect(ACE_OPT_0005_VAE_CONV1D_WORKGROUP_SIZE_Y).toBe(8);
expect(ACE_OPT_0005_VAE_CONV1D_WORKGROUP_SIZE).toBe(128);

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });
vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1 << 6 });

function principalShape(): AceVaeConv1dShape {
  return {
    batch: 1,
    inputFrames: 2_560,
    inputChannels: 1_024,
    outputChannels: 1_024,
    kernelSize: 7,
    stride: 1,
    dilation: 1,
    padding: 3,
  };
}

function tailShape(inputChannels: number): AceVaeConv1dShape {
  return {
    batch: 1,
    inputFrames: 17,
    inputChannels,
    outputChannels: 9,
    kernelSize: 7,
    stride: 1,
    dilation: 1,
    padding: 3,
  };
}

function bindingsFor(shape: AceVaeConv1dShape): AceVaeConvBindings {
  const effectiveKernel = shape.dilation * (shape.kernelSize - 1) + 1;
  const outputFrames = Math.floor(
    (shape.inputFrames + 2 * shape.padding - effectiveKernel) / shape.stride,
  ) + 1;
  return {
    input: fakeBinding(shape.batch * shape.inputFrames * shape.inputChannels * 4),
    weight: fakeBinding(shape.outputChannels * 7 * shape.inputChannels * 4),
    output: fakeBinding(shape.batch * outputFrames * shape.outputChannels * 4),
  };
}

interface FakeDeviceDiagnostics {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createBindGroupLayout: ReturnType<typeof vi.fn>;
  readonly createPipelineLayout: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
  readonly pushErrorScope: ReturnType<typeof vi.fn>;
  readonly popErrorScope: ReturnType<typeof vi.fn>;
  readonly createdBuffers: ReturnType<typeof fakeMappedBuffer>[];
}

type FakeDevice = GPUDevice & FakeDeviceDiagnostics;

function fakeDevice(options: {
  readonly maximumInvocations?: number;
  readonly maximumWorkgroupSizeX?: number;
  readonly maximumWorkgroupSizeY?: number;
  readonly maximumWorkgroupStorage?: number;
  readonly maximumDispatch?: number;
  readonly maximumStorageBinding?: number;
  readonly pipelineResults?: readonly Promise<GPUComputePipeline>[];
  readonly scopeResults?: readonly (GPUError | null)[];
  readonly compilationMessages?: readonly Partial<GPUCompilationMessage>[];
} = {}): FakeDevice {
  const createdBuffers: ReturnType<typeof fakeMappedBuffer>[] = [];
  const pipelineResults = [...(options.pipelineResults ?? [
    Promise.resolve(fakePipeline()),
  ])];
  const scopeResults = [...(options.scopeResults ?? [null, null, null])];
  return {
    limits: {
      maxComputeInvocationsPerWorkgroup: options.maximumInvocations ?? 256,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSizeX ?? 256,
      maxComputeWorkgroupSizeY: options.maximumWorkgroupSizeY ?? 256,
      maxComputeWorkgroupStorageSize: options.maximumWorkgroupStorage ?? 32_768,
      maxComputeWorkgroupsPerDimension: options.maximumDispatch ?? 65_535,
      maxStorageBufferBindingSize: options.maximumStorageBinding ?? 1_073_741_824,
      maxBufferSize: 1_073_741_824,
      minUniformBufferOffsetAlignment: 256,
    },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => scopeResults.shift() ?? null),
    createShaderModule: vi.fn(() => ({
      label: "module",
      getCompilationInfo: vi.fn(async () => ({
        messages: options.compilationMessages ?? [],
      })),
    })),
    createBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    createPipelineLayout: vi.fn(() => ({ label: "pipeline-layout" })),
    createComputePipelineAsync: vi.fn(() => {
      const result = pipelineResults.shift();
      if (result === undefined) throw new Error("fake pipelines exhausted");
      return result;
    }),
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      const buffer = fakeMappedBuffer(Number(descriptor.size));
      createdBuffers.push(buffer);
      return buffer;
    }),
    createBindGroup: vi.fn(() => ({ label: "bind-group" })),
    createdBuffers,
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

function fakeMappedBuffer(size: number) {
  const mapped = new ArrayBuffer(size);
  return {
    size,
    mapped,
    getMappedRange: vi.fn(() => mapped),
    unmap: vi.fn(),
    destroy: vi.fn(),
  } as unknown as GPUBuffer & {
    readonly mapped: ArrayBuffer;
    readonly destroy: ReturnType<typeof vi.fn>;
  };
}
