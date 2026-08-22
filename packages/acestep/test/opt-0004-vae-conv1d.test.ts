import { describe, expect, it, vi } from "vitest";
import type { AceVaeConv1dShape } from
  "../src/webgpu/kernels/vae-primitives.js";

import {
  ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_FRAMES,
  ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_STRIDE,
  ACE_OPT_0004_VAE_CONV1D_KERNEL_SIZE,
  ACE_OPT_0004_VAE_CONV1D_TILE_CHANNELS,
  ACE_OPT_0004_VAE_CONV1D_TILE_FRAMES,
  ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE,
  ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_X,
  ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_Y,
  AceOpt0004VaeConv1dKernel,
  aceOpt0004VaeConv1dWgsl,
  planAceOpt0004VaeConv1d,
} from "../benchmark/opt-0004-vae-conv1d.js";

describe("OPT-0004 tiled FP32 VAE Conv1D candidate", () => {
  it("plans exact 2,048-row bounded ranges for the principal K7 shape", () => {
    const plan = planAceOpt0004VaeConv1d(principalShape());
    expect(plan).toMatchObject({
      batch: 1,
      inputFrames: 491_520,
      outputFrames: 491_520,
      inputChannels: 128,
      outputChannels: 128,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 3,
      inputElements: 62_914_560,
      weightElements: 114_688,
      outputElements: 62_914_560,
      tileFrames: 16,
      tileChannels: 8,
      inputTileFrames: 22,
      inputTileStride: 23,
      weightTileStride: 129,
      workgroupSizeX: 16,
      workgroupSizeY: 8,
      workgroupSize: 128,
      inputTileElements: 2_944,
      weightTileElements: 1_032,
      inputTileBytes: 11_776,
      weightTileBytes: 4_128,
      workgroupStorageBytes: 15_904,
      outputRangeCount: 240,
    });
    expect(plan.outputRanges[0]).toEqual({
      batch: 0,
      firstOutputTime: 0,
      firstOutputRow: 0,
      outputRowCount: 2_048,
      firstOutput: 0,
      outputCount: 262_144,
      workgroupsX: 128,
      workgroupsY: 16,
      workgroupCount: 2_048,
      multiplyAdds: 234_881_024,
    });
    expect(plan.outputRanges.at(-1)).toEqual({
      batch: 0,
      firstOutputTime: 489_472,
      firstOutputRow: 489_472,
      outputRowCount: 2_048,
      firstOutput: 62_652_416,
      outputCount: 262_144,
      workgroupsX: 128,
      workgroupsY: 16,
      workgroupCount: 2_048,
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

  it("keeps generic tail ranges on complete-channel rows and inside one batch", () => {
    const plan = planAceOpt0004VaeConv1d({
      batch: 2,
      inputFrames: 33,
      inputChannels: 3,
      outputChannels: 10,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 3,
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
        multiplyAdds: 6_930,
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
        multiplyAdds: 6_930,
      },
    ]);
    for (const range of plan.outputRanges) {
      expect(range.firstOutput % plan.outputChannels).toBe(0);
      expect(range.outputCount % plan.outputChannels).toBe(0);
      expect(range.firstOutputTime + range.outputRowCount)
        .toBeLessThanOrEqual(plan.outputFrames);
    }
  });

  it("preserves a final single-row quantum instead of padding the range", () => {
    const plan = planAceOpt0004VaeConv1d({
      ...principalShape(),
      inputFrames: 4_097,
    });
    expect(plan.outputRanges.map(({ outputRowCount }) => outputRowCount)).toEqual([
      2_048, 2_048, 1,
    ]);
    expect(plan.outputRanges.at(-1)).toMatchObject({
      firstOutputTime: 4_096,
      outputCount: 128,
      workgroupsX: 1,
      workgroupsY: 16,
      multiplyAdds: 114_688,
    });
  });

  it.each([
    [{ ...principalShape(), kernelSize: 5 }, /kernel size 7/],
    [{ ...principalShape(), stride: 2 }, /stride 1 and dilation 1/],
    [{ ...principalShape(), dilation: 2 }, /stride 1 and dilation 1/],
    [{ ...principalShape(), batch: 0 }, /positive safe integer/],
    [{
      ...principalShape(),
      inputFrames: 7,
      outputChannels: 8 * 65_535 + 1,
    }, /dispatch domain/],
    [{
      ...principalShape(),
      inputFrames: 7,
      inputChannels: 300_000,
    }, /complete-channel output row/],
  ])("rejects unsupported or unbounded geometry", (shape, reason) => {
    expect(() => planAceOpt0004VaeConv1d(shape)).toThrow(reason);
  });

  it("fails closed when an inactive tail tile's staged halo would wrap u32", () => {
    expect(() => planAceOpt0004VaeConv1d({
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

  it("emits transposed/padded tiles, uniform barriers, and source-order FP32 adds", () => {
    const source = aceOpt0004VaeConv1dWgsl(principalShape(), true);
    expect(source).toContain("var<workgroup> input_tile: array<f32, 2944>;");
    expect(source).toContain("var<workgroup> weight_tile: array<f32, 1032>;");
    expect(source).toContain("@compute @workgroup_size(\n  16,\n  8,\n  1,\n)");
    expect(source).toContain("input_channel * 23u + tile_time");
    expect(source).toContain("tile_output_channel * 129u + input_channel");
    expect(source).toContain("let weight_base = local.y * 129u;");
    expect(source).toContain("var kernel = 0u;");
    expect(source).toContain("var input_channel = 0u;");
    expect(source).toContain(
      "sum = sum + input_tile[\n              input_channel * 23u +",
    );
    expect(source).toContain(
      "(weight_output_channel * 7u +\n            kernel) * INPUT_CHANNELS",
    );
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(3);
    expect(source.indexOf("sum = bias[output_channel];")).toBeLessThan(
      source.indexOf("var kernel = 0u;"),
    );
    expect(source.indexOf("if (padded_time >= PADDING)", source.indexOf("var kernel")))
      .toBeLessThan(source.indexOf("var input_channel = 0u;"));
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toContain("return;");
    expect(source).not.toMatch(/\blet active\b/);
  });

  it("uses the no-bias binding layout without changing the contraction", () => {
    const source = aceOpt0004VaeConv1dWgsl({
      ...principalShape(),
      inputFrames: 17,
      inputChannels: 7,
      outputChannels: 9,
    }, false);
    expect(source).toContain(
      "@group(0) @binding(2) var<storage, read_write>",
    );
    expect(source).toContain("@group(0) @binding(3) var<uniform>");
    expect(source).toContain("var sum = 0.0;");
    expect(source).toContain("sum = 0.0;");
    expect(source).not.toContain("var<storage, read> bias");
  });

  it("fails closed on insufficient workgroup geometry or storage", async () => {
    expect(() => AceOpt0004VaeConv1dKernel.create(fakeDevice({
      maximumWorkgroupSizeX: 8,
    }))).toThrow(/16x8/);
    expect(() => AceOpt0004VaeConv1dKernel.create(fakeDevice({
      maximumWorkgroupSizeY: 4,
    }))).toThrow(/16x8/);
    expect(() => AceOpt0004VaeConv1dKernel.create(fakeDevice({
      maximumInvocations: 64,
    }))).toThrow(/128-lane/);

    const device = fakeDevice({ maximumWorkgroupStorage: 15_903 });
    const kernel = AceOpt0004VaeConv1dKernel.create(device);
    await expect(kernel.createDispatch(
      "too-much-shared-memory",
      principalShape(),
      bindingsFor(principalShape()),
    )).rejects.toThrow(/15904 workgroup-storage bytes/);
    expect(device.createShaderModule).not.toHaveBeenCalled();
  });

  it("compiles, binds, and encodes every complete-row range", async () => {
    const device = fakeDevice();
    const kernel = AceOpt0004VaeConv1dKernel.create(device);
    const shape = { ...principalShape(), inputFrames: 4_097 };
    const dispatch = await kernel.createDispatch(
      "candidate",
      shape,
      bindingsFor(shape),
    );
    expect(dispatch.rangeCount).toBe(3);
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledTimes(3);
    const controls = device.createdBuffers[0]!;
    expect(Array.from(new Uint32Array(controls.mapped, 0, 2))).toEqual([0, 2_048]);
    expect(Array.from(new Uint32Array(controls.mapped, 256, 2)))
      .toEqual([2_048, 2_048]);
    expect(Array.from(new Uint32Array(controls.mapped, 512, 2)))
      .toEqual([4_096, 1]);

    const pass = fakePass();
    dispatch.encodeRange(pass, 2);
    expect(pass.dispatchWorkgroups).toHaveBeenLastCalledWith(1, 16, 1);
    expect(() => dispatch.encodeRange(pass, 3)).toThrow(/outside \[0, 3\)/);
    dispatch.encode(pass);
    expect(pass.dispatchWorkgroups).toHaveBeenCalledTimes(4);

    kernel.destroy();
    await Promise.resolve();
    expect(controls.destroy).toHaveBeenCalledOnce();
    await expect(kernel.createDispatch(
      "after-destroy",
      shape,
      bindingsFor(shape),
    )).rejects.toThrow(/was destroyed/);
  });

  it("uses distinct bias/no-bias layouts and caches each specialization", async () => {
    const device = fakeDevice({
      pipelineResults: [
        Promise.resolve(fakePipeline()),
        Promise.resolve(fakePipeline()),
      ],
      scopeResults: [null, null, null, null, null, null],
    });
    const kernel = AceOpt0004VaeConv1dKernel.create(device);
    const shape = { ...principalShape(), inputFrames: 17 };
    const noBiasBindings = bindingsFor(shape);
    await kernel.createDispatch("no-bias", shape, noBiasBindings);
    await kernel.createDispatch("no-bias-reuse", shape, noBiasBindings);
    await kernel.createDispatch("bias", shape, {
      ...noBiasBindings,
      bias: fakeBinding(shape.outputChannels * 4),
    });
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createdBuffers).toHaveLength(2);
    const calls = device.createBindGroup.mock.calls;
    const noBias = calls[0]?.[0] as GPUBindGroupDescriptor;
    const bias = calls[2]?.[0] as GPUBindGroupDescriptor;
    expect(Array.from(noBias.entries, ({ binding }) => binding)).toEqual([
      0, 1, 2, 3,
    ]);
    expect(Array.from(bias.entries, ({ binding }) => binding)).toEqual([
      0, 1, 2, 3, 4,
    ]);
    kernel.destroy();
    await Promise.resolve();
    expect(device.createdBuffers.every(
      (buffer) => vi.mocked(buffer.destroy).mock.calls.length === 1,
    )).toBe(true);
  });

  it("evicts a failed compilation and retries the same specialization", async () => {
    const failure = new Error("synthetic Conv1D compile failure");
    const device = fakeDevice({
      pipelineResults: [Promise.reject(failure), Promise.resolve(fakePipeline())],
    });
    const kernel = AceOpt0004VaeConv1dKernel.create(device);
    const shape = { ...principalShape(), inputFrames: 17 };
    const bindings = bindingsFor(shape);
    await expect(kernel.createDispatch("failure", shape, bindings))
      .rejects.toBe(failure);
    await expect(kernel.createDispatch("retry", shape, bindings))
      .resolves.toMatchObject({ label: "retry", rangeCount: 1 });
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createdBuffers).toHaveLength(1);
  });

  it("reports WGSL compilation diagnostics before pipeline creation", async () => {
    const device = fakeDevice({
      compilationMessages: [{
        message: "synthetic uniformity error",
        type: "error",
        lineNum: 42,
        linePos: 7,
      }],
    });
    const kernel = AceOpt0004VaeConv1dKernel.create(device);
    const shape = { ...principalShape(), inputFrames: 17 };
    await expect(kernel.createDispatch(
      "shader-failure",
      shape,
      bindingsFor(shape),
    )).rejects.toThrow(/42:7 synthetic uniformity error/);
    expect(device.createComputePipelineAsync).not.toHaveBeenCalled();
    expect(device.createdBuffers).toHaveLength(0);
  });

  it("destroys scoped controls after allocation failure and retries", async () => {
    const allocationFailure = {
      message: "synthetic range-control validation failure",
    } as GPUError;
    const device = fakeDevice({
      pipelineResults: [
        Promise.resolve(fakePipeline()),
        Promise.resolve(fakePipeline()),
      ],
      scopeResults: [
        allocationFailure, null, null,
        null, null, null,
      ],
    });
    const kernel = AceOpt0004VaeConv1dKernel.create(device);
    const shape = { ...principalShape(), inputFrames: 17 };
    const bindings = bindingsFor(shape);
    await expect(kernel.createDispatch("allocation-fails", shape, bindings))
      .rejects.toThrow(/validation failure/);
    expect(device.createdBuffers[0]?.destroy).toHaveBeenCalledOnce();
    await expect(kernel.createDispatch("allocation-retry", shape, bindings))
      .resolves.toMatchObject({ label: "allocation-retry", rangeCount: 1 });
    expect(device.pushErrorScope).toHaveBeenCalledTimes(6);
    expect(device.popErrorScope).toHaveBeenCalledTimes(6);
    expect(device.createdBuffers).toHaveLength(2);
  });

  it("destroys controls compiled after kernel destruction", async () => {
    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pending = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const device = fakeDevice({ pipelineResults: [pending] });
    const kernel = AceOpt0004VaeConv1dKernel.create(device);
    const shape = { ...principalShape(), inputFrames: 17 };
    const dispatch = kernel.createDispatch("pending", shape, bindingsFor(shape));
    kernel.destroy();
    resolvePipeline(fakePipeline());
    await expect(dispatch).rejects.toThrow(/destroyed while compiling/);
    await Promise.resolve();
    expect(device.createdBuffers[0]?.destroy).toHaveBeenCalledOnce();
  });

  it("rejects undersized and overlapping bindings before compilation", async () => {
    const device = fakeDevice();
    const kernel = AceOpt0004VaeConv1dKernel.create(device);
    const shape = { ...principalShape(), inputFrames: 17 };
    const bindings = bindingsFor(shape);
    await expect(kernel.createDispatch("short", shape, {
      ...bindings,
      weight: fakeBinding(shape.outputChannels * 7 * shape.inputChannels * 4 - 4),
    })).rejects.toThrow(/weight binding/);

    const inputBytes = shape.batch * shape.inputFrames * shape.inputChannels * 4;
    const outputBytes = shape.batch * shape.inputFrames * shape.outputChannels * 4;
    const shared = fakeBuffer(inputBytes + outputBytes);
    await expect(kernel.createDispatch("aliased", shape, {
      ...bindings,
      input: { buffer: shared, offset: 0, size: inputBytes },
      output: { buffer: shared, offset: 1_024, size: outputBytes },
    })).rejects.toThrow(/must not overlap/);
    expect(device.createShaderModule).not.toHaveBeenCalled();
  });
});

expect(ACE_OPT_0004_VAE_CONV1D_KERNEL_SIZE).toBe(7);
expect(ACE_OPT_0004_VAE_CONV1D_TILE_FRAMES).toBe(16);
expect(ACE_OPT_0004_VAE_CONV1D_TILE_CHANNELS).toBe(8);
expect(ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_FRAMES).toBe(22);
expect(ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_STRIDE).toBe(23);
expect(ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_X).toBe(16);
expect(ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE_Y).toBe(8);
expect(ACE_OPT_0004_VAE_CONV1D_WORKGROUP_SIZE).toBe(128);

vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1 << 6 });

function principalShape(): AceVaeConv1dShape {
  return {
    batch: 1,
    inputFrames: 491_520,
    inputChannels: 128,
    outputChannels: 128,
    kernelSize: 7,
    stride: 1,
    dilation: 1,
    padding: 3,
  };
}

function bindingsFor(shape: AceVaeConv1dShape) {
  const outputFrames = shape.inputFrames + 2 * shape.padding - 6;
  return {
    input: fakeBinding(shape.batch * shape.inputFrames * shape.inputChannels * 4),
    weight: fakeBinding(shape.outputChannels * 7 * shape.inputChannels * 4),
    output: fakeBinding(shape.batch * outputFrames * shape.outputChannels * 4),
  };
}

interface FakeDeviceDiagnostics {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
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
      maxComputeWorkgroupsPerDimension: 65_535,
      maxStorageBufferBindingSize: 1_073_741_824,
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
    createComputePipelineAsync: vi.fn(() => {
      const result = pipelineResults.shift();
      if (result === undefined) {
        throw new Error("fake device exhausted pipeline results");
      }
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
  return {
    getBindGroupLayout: vi.fn(() => ({ label: "layout" })),
  } as unknown as GPUComputePipeline;
}

function fakePass(): GPUComputePassEncoder & {
  readonly dispatchWorkgroups: ReturnType<typeof vi.fn>;
} {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
  } as unknown as GPUComputePassEncoder & {
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
