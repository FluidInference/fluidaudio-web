import { describe, expect, it, vi } from "vitest";
import type {
  AceVaeConv1dShape,
  AceVaeConvBindings,
} from "../src/webgpu/kernels/vae-primitives.js";

import {
  ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK,
  ACE_OPT_0007_VAE_K1_CONV1D_INPUT_TILE_STRIDE,
  ACE_OPT_0007_VAE_K1_CONV1D_KERNEL_SIZE,
  ACE_OPT_0007_VAE_K1_CONV1D_TILE_CHANNELS,
  ACE_OPT_0007_VAE_K1_CONV1D_TILE_FRAMES,
  ACE_OPT_0007_VAE_K1_CONV1D_WEIGHT_TILE_STRIDE,
  ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE,
  ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_X,
  ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_Y,
  AceOpt0007VaeK1Conv1dKernel,
  aceOpt0007VaeK1Conv1dWgsl,
  planAceOpt0007VaeK1Conv1d,
} from "../benchmark/opt-0007-vae-k1-conv1d.js";

describe("OPT-0007 pointwise FP32 VAE Conv1D candidate", () => {
  it("plans the authenticated C1024 endpoint as twelve bounded ranges", () => {
    const plan = planAceOpt0007VaeK1Conv1d(principalShape());
    expect(plan).toMatchObject({
      batch: 1,
      inputFrames: 2_560,
      outputFrames: 2_560,
      inputChannels: 1_024,
      outputChannels: 1_024,
      kernelSize: 1,
      stride: 1,
      dilation: 1,
      padding: 0,
      inputElements: 2_621_440,
      weightElements: 1_048_576,
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
      outputRangeCount: 12,
    });
    expect(plan.outputRanges[0]).toEqual({
      batch: 0,
      firstOutputTime: 0,
      firstOutputRow: 0,
      outputRowCount: 224,
      firstOutput: 0,
      outputCount: 229_376,
      workgroupsX: 14,
      workgroupsY: 128,
      workgroupCount: 1_792,
      multiplyAdds: 234_881_024,
    });
    expect(plan.outputRanges.at(-1)).toEqual({
      batch: 0,
      firstOutputTime: 2_464,
      firstOutputRow: 2_464,
      outputRowCount: 96,
      firstOutput: 2_523_136,
      outputCount: 98_304,
      workgroupsX: 6,
      workgroupsY: 128,
      workgroupCount: 768,
      multiplyAdds: 100_663_296,
    });
    expect(plan.outputRanges.reduce(
      (sum, range) => sum + range.outputCount,
      0,
    )).toBe(plan.outputElements);
    expect(Object.isFrozen(plan)).toBe(true);
    expect(Object.isFrozen(plan.outputRanges)).toBe(true);
    expect(plan.outputRanges.every(Object.isFrozen)).toBe(true);
  });

  it("plans the authenticated C128 endpoint as sixty complete ranges", () => {
    const plan = planAceOpt0007VaeK1Conv1d({
      batch: 1,
      inputFrames: 491_520,
      inputChannels: 128,
      outputChannels: 128,
      kernelSize: 1,
      stride: 1,
      dilation: 1,
      padding: 0,
    });
    expect(plan.outputRangeCount).toBe(60);
    expect(plan.outputRanges[0]).toMatchObject({
      outputRowCount: 8_192,
      outputCount: 1_048_576,
      workgroupsX: 512,
      workgroupsY: 16,
      workgroupCount: 8_192,
      multiplyAdds: 134_217_728,
    });
    expect(plan.outputRanges.at(-1)).toMatchObject({
      firstOutputTime: 483_328,
      outputRowCount: 8_192,
    });
  });

  it("keeps frame, channel, and batch tails on complete output rows", () => {
    const plan = planAceOpt0007VaeK1Conv1d({
      batch: 2,
      inputFrames: 33,
      inputChannels: 65,
      outputChannels: 10,
      kernelSize: 1,
      stride: 1,
      dilation: 1,
      padding: 0,
    });
    expect(plan.inputChannelChunkCount).toBe(2);
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
        multiplyAdds: 21_450,
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
        multiplyAdds: 21_450,
      },
    ]);
    for (const range of plan.outputRanges) {
      expect(range.firstOutput % plan.outputChannels).toBe(0);
      expect(range.outputCount % plan.outputChannels).toBe(0);
      expect(range.firstOutputTime + range.outputRowCount)
        .toBeLessThanOrEqual(plan.outputFrames);
    }
  });

  it("covers C63, C64, and C65 without changing shared geometry", () => {
    for (const [inputChannels, chunks] of [
      [63, 1],
      [64, 1],
      [65, 2],
    ] as const) {
      const plan = planAceOpt0007VaeK1Conv1d(tailShape(inputChannels));
      expect(plan.inputChannelChunkCount).toBe(chunks);
      expect(plan.inputTileElements).toBe(64 * 17);
      expect(plan.weightTileElements).toBe(8 * 65);
      expect(plan.workgroupStorageBytes).toBe(6_432);
    }
  });

  it("emits one-owner K1 native weights and globally increasing Cin adds", () => {
    const source = aceOpt0007VaeK1Conv1dWgsl(tailShape(65), true);
    expect(source).toContain("var<workgroup> input_tile: array<f32, 1088>;");
    expect(source).toContain("var<workgroup> weight_tile: array<f32, 520>;");
    expect(source).toContain("@compute @workgroup_size(\n  16,\n  8,\n  1,\n)");
    expect(source).toContain("var input_channel_chunk = 0u;");
    expect(source).toContain("var chunk_channel = 0u;");
    expect(source).toContain(
      "weight_output_channel * INPUT_CHANNELS + input_channel",
    );
    expect(source).toContain("chunk_channel * 17u +\n          local.x");
    expect(source).toContain("local.y * 65u");
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    const initialize = source.indexOf("sum = bias[output_channel];");
    const chunk = source.indexOf("var input_channel_chunk = 0u;");
    const channel = source.indexOf("var chunk_channel = 0u;", chunk);
    const write = source.lastIndexOf("output[output_row * OUTPUT_CHANNELS");
    expect(initialize).toBeLessThan(chunk);
    expect(chunk).toBeLessThan(channel);
    expect(channel).toBeLessThan(write);
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toContain("return;");
    expect(source).not.toContain("enable f16");
    expect(source.match(/output\[output_row \* OUTPUT_CHANNELS/g)).toHaveLength(1);
  });

  it("uses the no-bias layout and positive-zero initialization", () => {
    const source = aceOpt0007VaeK1Conv1dWgsl(tailShape(63), false);
    expect(source).toContain(
      "@group(0) @binding(2) var<storage, read_write>",
    );
    expect(source).toContain("@group(0) @binding(3) var<uniform>");
    expect(source).toContain("if (output_active) { sum = 0.0; }");
    expect(source).not.toContain("var<storage, read> bias");
  });

  it.each([
    [{ ...tailShape(65), kernelSize: 3 }, /requires K1/],
    [{ ...tailShape(65), stride: 2 }, /requires K1/],
    [{ ...tailShape(65), dilation: 2 }, /requires K1/],
    [{ ...tailShape(65), padding: 1 }, /zero padding/],
    [{ ...tailShape(65), batch: 0 }, /positive safe integer/],
    [{
      ...tailShape(65),
      inputFrames: 1,
      outputChannels: 8 * 65_535 + 1,
    }, /dispatch domain/],
    [{
      ...tailShape(1),
      inputFrames: 65_536,
      outputChannels: 65_536,
    }, /WGSL's u32 indexing domain/],
    [{
      ...tailShape(30_000_000),
      inputFrames: 1,
    }, /complete-channel output row/],
  ])("rejects unsupported or unbounded geometry", (shape, reason) => {
    expect(() => planAceOpt0007VaeK1Conv1d(shape)).toThrow(reason);
  });

  it("fails closed on insufficient workgroup, storage, and dispatch limits", async () => {
    expect(() => AceOpt0007VaeK1Conv1dKernel.create(fakeDevice({
      maximumWorkgroupSizeX: 8,
    }))).toThrow(/16x8/);
    expect(() => AceOpt0007VaeK1Conv1dKernel.create(fakeDevice({
      maximumWorkgroupSizeY: 4,
    }))).toThrow(/16x8/);
    expect(() => AceOpt0007VaeK1Conv1dKernel.create(fakeDevice({
      maximumInvocations: 64,
    }))).toThrow(/128-lane/);

    const storageDevice = fakeDevice({ maximumWorkgroupStorage: 6_431 });
    const storageKernel = AceOpt0007VaeK1Conv1dKernel.create(storageDevice);
    await expect(storageKernel.createDispatch(
      "storage",
      principalShape(),
      bindingsFor(principalShape()),
    )).rejects.toThrow(/6432 workgroup-storage bytes/);
    expect(storageDevice.createShaderModule).not.toHaveBeenCalled();

    const dispatchDevice = fakeDevice({ maximumDispatch: 1 });
    const dispatchKernel = AceOpt0007VaeK1Conv1dKernel.create(dispatchDevice);
    await expect(dispatchKernel.createDispatch(
      "dispatch",
      principalShape(),
      bindingsFor(principalShape()),
    )).rejects.toThrow(/dispatch dimension/);
    expect(dispatchDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("fails closed on invalid control alignment and buffer limits", async () => {
    const alignmentDevice = fakeDevice({ uniformAlignment: 12 });
    const alignmentKernel = AceOpt0007VaeK1Conv1dKernel.create(
      alignmentDevice,
    );
    await expect(alignmentKernel.createDispatch(
      "alignment",
      tailShape(65),
      bindingsFor(tailShape(65)),
    )).rejects.toThrow(/invalid uniform alignment/);
    expect(alignmentDevice.createdBuffers).toHaveLength(0);

    const bufferDevice = fakeDevice({ maximumBuffer: 256 });
    const bufferKernel = AceOpt0007VaeK1Conv1dKernel.create(bufferDevice);
    await expect(bufferKernel.createDispatch(
      "control-limit",
      principalShape(),
      bindingsFor(principalShape()),
    )).rejects.toThrow(/range controls exceed the device buffer limit/);
    expect(bufferDevice.createdBuffers).toHaveLength(0);
  });

  it("compiles once, binds once, and encodes dynamic complete-row ranges", async () => {
    const device = fakeDevice();
    const kernel = AceOpt0007VaeK1Conv1dKernel.create(device);
    const shape = { ...principalShape(), inputFrames: 449 };
    const stableBindings = bindingsFor(shape);
    const dispatch = await kernel.createDispatch("candidate", shape, stableBindings);
    const reused = await kernel.createDispatch("reuse", shape, stableBindings);
    expect(dispatch.rangeCount).toBe(3);
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();
    const controls = device.createdBuffers[0]!;
    expect(Array.from(new Uint32Array(controls.mapped, 0, 2))).toEqual([0, 224]);
    expect(Array.from(new Uint32Array(controls.mapped, 256, 2)))
      .toEqual([224, 224]);
    expect(Array.from(new Uint32Array(controls.mapped, 512, 2)))
      .toEqual([448, 1]);
    const layout = device.createBindGroupLayout.mock.calls[0]?.[0] as
      GPUBindGroupLayoutDescriptor;
    expect(Array.from(layout.entries).at(-1)?.buffer).toEqual({
      type: "uniform",
      hasDynamicOffset: true,
      minBindingSize: 16,
    });

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
    const failure = new Error("synthetic K1 compile failure");
    const device = fakeDevice({
      pipelineResults: [
        Promise.reject(failure),
        Promise.resolve(fakePipeline()),
        Promise.resolve(fakePipeline()),
      ],
    });
    const kernel = AceOpt0007VaeK1Conv1dKernel.create(device);
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

  it("reports diagnostics and retries scoped control allocation failures", async () => {
    const diagnostic = fakeDevice({
      compilationMessages: [{
        message: "synthetic uniformity error",
        type: "error",
        lineNum: 42,
        linePos: 7,
      }],
    });
    const diagnosticKernel = AceOpt0007VaeK1Conv1dKernel.create(diagnostic);
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
    const allocationKernel = AceOpt0007VaeK1Conv1dKernel.create(allocation);
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

  it("destroys controls whose compilation finishes after destruction", async () => {
    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pending = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const device = fakeDevice({ pipelineResults: [pending] });
    const kernel = AceOpt0007VaeK1Conv1dKernel.create(device);
    const shape = tailShape(65);
    const dispatch = kernel.createDispatch("pending", shape, bindingsFor(shape));
    kernel.destroy();
    resolvePipeline(fakePipeline());
    await expect(dispatch).rejects.toThrow(/destroyed/);
    await Promise.resolve();
    expect(device.createdBuffers[0]?.destroy).toHaveBeenCalledOnce();
  });

  it("rejects undersized, oversized-device, and overlapping bindings", async () => {
    const shape = tailShape(65);
    const device = fakeDevice();
    const kernel = AceOpt0007VaeK1Conv1dKernel.create(device);
    const bindings = bindingsFor(shape);
    await expect(kernel.createDispatch("short", shape, {
      ...bindings,
      weight: fakeBinding(shape.outputChannels * shape.inputChannels * 4 - 4),
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
    const bindingKernel = AceOpt0007VaeK1Conv1dKernel.create(bindingDevice);
    await expect(bindingKernel.createDispatch(
      "binding-limit",
      shape,
      bindingsFor(shape),
    )).rejects.toThrow(/storage binding limit/);
    expect(bindingDevice.createShaderModule).not.toHaveBeenCalled();
  });
});

expect(ACE_OPT_0007_VAE_K1_CONV1D_KERNEL_SIZE).toBe(1);
expect(ACE_OPT_0007_VAE_K1_CONV1D_INPUT_CHANNEL_CHUNK).toBe(64);
expect(ACE_OPT_0007_VAE_K1_CONV1D_TILE_FRAMES).toBe(16);
expect(ACE_OPT_0007_VAE_K1_CONV1D_TILE_CHANNELS).toBe(8);
expect(ACE_OPT_0007_VAE_K1_CONV1D_INPUT_TILE_STRIDE).toBe(17);
expect(ACE_OPT_0007_VAE_K1_CONV1D_WEIGHT_TILE_STRIDE).toBe(65);
expect(ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_X).toBe(16);
expect(ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE_Y).toBe(8);
expect(ACE_OPT_0007_VAE_K1_CONV1D_WORKGROUP_SIZE).toBe(128);

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });
vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1 << 6 });

function principalShape(): AceVaeConv1dShape {
  return {
    batch: 1,
    inputFrames: 2_560,
    inputChannels: 1_024,
    outputChannels: 1_024,
    kernelSize: 1,
    stride: 1,
    dilation: 1,
    padding: 0,
  };
}

function tailShape(inputChannels: number): AceVaeConv1dShape {
  return {
    batch: 1,
    inputFrames: 17,
    inputChannels,
    outputChannels: 9,
    kernelSize: 1,
    stride: 1,
    dilation: 1,
    padding: 0,
  };
}

function bindingsFor(shape: AceVaeConv1dShape): AceVaeConvBindings {
  const outputFrames = shape.inputFrames;
  return {
    input: fakeBinding(shape.batch * shape.inputFrames * shape.inputChannels * 4),
    weight: fakeBinding(shape.outputChannels * shape.inputChannels * 4),
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
  readonly maximumBuffer?: number;
  readonly uniformAlignment?: number;
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
      maxBufferSize: options.maximumBuffer ?? 1_073_741_824,
      minUniformBufferOffsetAlignment: options.uniformAlignment ?? 256,
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
