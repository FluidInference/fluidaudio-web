import { describe, expect, it, vi } from "vitest";

import {
  ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK,
  ACE_FP16_VAE_CONV1D_INPUT_TILE_STRIDE,
  ACE_FP16_VAE_CONV1D_K1_KERNEL_SIZE,
  ACE_FP16_VAE_CONV1D_K7_KERNEL_SIZE,
  ACE_FP16_VAE_CONV1D_K7_SUPPORTED_DILATIONS,
  ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID,
  ACE_FP16_VAE_CONV1D_TILE_CHANNELS,
  ACE_FP16_VAE_CONV1D_TILE_FRAMES,
  ACE_FP16_VAE_CONV1D_WEIGHT_TILE_STRIDE,
  ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE,
  ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_X,
  ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_Y,
  AceFp16VaeConv1dKernel,
  aceFp16VaeConv1dWgsl,
  planAceFp16VaeConv1d,
  planAceFp16VaeConv1dRange,
  type AceFp16VaeConv1dBindings,
  type AceFp16VaeConv1dOutputStorage,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderConvOperation,
} from "../src/webgpu/vae-decoder.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";

describe("production portable FP16 VAE Conv1D set", () => {
  it("covers all 15 biased K1 residual projections in the frozen B-256 graph", () => {
    const operations = planAceVaeDecoder(256).operations.filter(
      (operation): operation is AceVaeDecoderConvOperation =>
        operation.kind === "conv1d" && operation.shape.kernelSize === 1,
    );
    expect(operations).toHaveLength(15);
    for (const operation of operations) {
      expect(operation.bias).toBeTypeOf("string");
      expect(planAceFp16VaeConv1d(operation.shape, "float16"))
        .toMatchObject({
          family: "k1",
          outputStorage: "float16",
          outputFrames: operation.shape.inputFrames,
          workgroupStorageBytes: 3_216,
        });
      expect(() => aceFp16VaeConv1dWgsl(
        operation.shape,
        true,
        "float16",
      )).not.toThrow();
    }
  });

  it("covers all 17 K7 graph operations and preserves the final FP32 boundary", () => {
    const operations = planAceVaeDecoder(256).operations.filter(
      (operation): operation is AceVaeDecoderConvOperation =>
        operation.kind === "conv1d" && operation.shape.kernelSize === 7,
    );
    expect(operations).toHaveLength(17);
    expect(operations.at(-1)?.bias).toBeUndefined();
    for (const [index, operation] of operations.entries()) {
      const outputStorage = index === operations.length - 1
        ? "float32"
        : "float16";
      expect(planAceFp16VaeConv1d(operation.shape, outputStorage))
        .toMatchObject({
          family: "k7",
          outputStorage,
          outputFrames: operation.shape.inputFrames,
          workgroupStorageBytes: 3_216,
        });
      expect(() => aceFp16VaeConv1dWgsl(
        operation.shape,
        operation.bias !== undefined,
        outputStorage,
      )).not.toThrow();
    }
  });

  it("accepts every existing B-256 decoder Conv1D quantum without repartitioning", () => {
    const graph = planAceVaeDecoder(256);
    const cooperative = planAceVaeDecoderQuanta(graph);
    let k1Ranges = 0;
    let k7Ranges = 0;
    for (const quantum of cooperative.quanta) {
      const operation = graph.operations[quantum.operationIndex]!;
      if (operation.kind !== "conv1d") continue;
      expect(quantum.primitives).toHaveLength(1);
      const primitive = quantum.primitives[0]!;
      const outputStorage = operation.bias === undefined
        ? "float32"
        : "float16";
      const plan = planAceFp16VaeConv1d(operation.shape, outputStorage);
      const range = planAceFp16VaeConv1dRange(plan, {
        base: primitive.outputBase,
        count: primitive.outputCount,
      });
      expect(range.base).toBe(quantum.logicalOutputBase);
      expect(range.count).toBe(quantum.logicalOutputCount);
      expect(range.outputRowCount * operation.shape.outputChannels)
        .toBe(primitive.outputCount);
      if (plan.family === "k1") k1Ranges += 1;
      else k7Ranges += 1;
    }
    expect(k1Ranges).toBeGreaterThan(15);
    expect(k7Ranges).toBeGreaterThan(17);
  });

  it("pins FP16 payload sizes separately from four-byte binding padding", () => {
    const k7 = planAceFp16VaeConv1d(k7TailShape(63), "float16");
    expect(k7).toMatchObject({
      family: "k7",
      inputStorageBytes: 2_142,
      inputBindingBytes: 2_144,
      weightStorageBytes: 7_938,
      weightBindingBytes: 7_940,
      biasStorageBytes: 18,
      biasBindingBytes: 20,
      outputStorageBytes: 306,
      outputBindingBytes: 308,
      inputChannelChunkCount: 1,
      inputTileElements: 1_088,
      weightTileElements: 520,
      inputTileBytes: 2_176,
      weightTileBytes: 1_040,
      workgroupStorageBytes: 3_216,
    });
    const final = planAceFp16VaeConv1d(k7TailShape(63), "float32");
    expect(final.outputStorageBytes).toBe(612);
    expect(final.outputBindingBytes).toBe(612);

    const k1 = planAceFp16VaeConv1d(k1TailShape(65), "float16");
    expect(k1).toMatchObject({
      family: "k1",
      inputChannelChunkCount: 2,
      workgroupStorageBytes: 3_216,
    });
  });

  it("plans only exact complete-row ranges within one batch", () => {
    const plan = planAceFp16VaeConv1d({
      ...k7TailShape(65),
      batch: 2,
      inputFrames: 33,
      outputChannels: 10,
      dilation: 9,
      padding: 27,
    }, "float16");
    expect(planAceFp16VaeConv1dRange(plan, {
      base: 330,
      count: 170,
    })).toEqual({
      base: 330,
      count: 170,
      batch: 1,
      firstOutputTime: 0,
      firstOutputRow: 33,
      outputRowCount: 17,
      workgroupsX: 2,
      workgroupsY: 2,
    });
    expect(() => planAceFp16VaeConv1dRange(plan, {
      base: 1,
      count: 10,
    })).toThrow(/complete in-bounds NLC rows/);
    expect(() => planAceFp16VaeConv1dRange(plan, {
      base: 320,
      count: 20,
    })).toThrow(/batch boundary/);
    expect(() => planAceFp16VaeConv1dRange(plan, {
      base: 0,
      count: 0,
    })).toThrow(/complete in-bounds NLC rows/);
  });

  it("emits the audited K7 K-outer/chunk/Cin FP32 accumulation order", () => {
    const source = aceFp16VaeConv1dWgsl(
      k7TailShape(65),
      true,
      "float16",
    );
    expect(source).toContain(
      `// kernel-id: ${ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID}`,
    );
    expect(source).toContain("enable f16;");
    expect(source).toContain("input: array<f16>;");
    expect(source).toContain("weight: array<f16>;");
    expect(source).toContain("bias: array<f16>;");
    expect(source).toContain("output: array<f16>;");
    expect(source).toContain("var<workgroup> input_tile: array<f16, 1088>;");
    expect(source).toContain("var<workgroup> weight_tile: array<f16, 520>;");
    expect(source).toContain("var sum: f32 = 0.0;");
    expect(source).toContain("sum = f32(bias[output_channel]);");
    expect(source).toContain("for (var kernel = 0u; kernel < 7u;");
    expect(source).toContain(
      "(weight_output_channel * 7u + kernel) *\n" +
        "              INPUT_CHANNELS + input_channel",
    );
    expect(source).toContain("let input_operand = f32(input_tile[");
    expect(source).toContain("let weight_operand = f32(");
    expect(source).toContain("sum = sum + input_operand * weight_operand;");
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    expect(source.match(/=\s*f16\(sum\);/g)).toHaveLength(1);

    const kernel = source.indexOf("var kernel = 0u;");
    const chunk = source.indexOf("var input_channel_chunk = 0u;", kernel);
    const paddingGuard = source.indexOf(
      "if (padded_time >= PADDING)",
      source.indexOf("if (output_active)", chunk),
    );
    const channel = source.indexOf("var chunk_channel = 0u;", paddingGuard);
    const add = source.indexOf(
      "sum = sum + input_operand * weight_operand;",
      channel,
    );
    expect(kernel).toBeLessThan(chunk);
    expect(chunk).toBeLessThan(paddingGuard);
    expect(paddingGuard).toBeLessThan(channel);
    expect(channel).toBeLessThan(add);
    expectForbiddenMathAbsent(source);
  });

  it("emits biased K1 with native weights and the same increasing-Cin loop", () => {
    const source = aceFp16VaeConv1dWgsl(
      k1TailShape(65),
      true,
      "float16",
    );
    expect(source).toContain("for (var kernel = 0u; kernel < 1u;");
    expect(source).toContain(
      "(weight_output_channel * 1u + kernel) *\n" +
        "              INPUT_CHANNELS + input_channel",
    );
    expect(source).toContain("sum = f32(bias[output_channel]);");
    expect(source).toContain("output: array<f16>;");
    expect(source).toContain("sum = sum + input_operand * weight_operand;");
    expectForbiddenMathAbsent(source);
    expect(() => aceFp16VaeConv1dWgsl(
      k1TailShape(65),
      false,
      "float16",
    )).toThrow(/requires the residual-projection bias/);
    expect(() => planAceFp16VaeConv1d(
      k1TailShape(65),
      "float32",
    )).toThrow(/requires an FP16 internal output/);
  });

  it("canonicalizes either sign of exact zero at the no-bias FP32 boundary", () => {
    const source = aceFp16VaeConv1dWgsl(
      k7TailShape(63),
      false,
      "float32",
    );
    expect(source).not.toContain("bias: array<f16>");
    expect(source).toContain("if (output_active) { sum = 0.0; }");
    expect(source).toContain(
      "select(sum, bitcast<f32>(0u), " +
        "(bitcast<u32>(sum) & 0x7fffffffu) == 0u)",
    );
    expect(source).toContain("output: array<f32>;");
    expect(() => aceFp16VaeConv1dWgsl(
      k7TailShape(63),
      false,
      "float16",
    )).toThrow(/bias may be omitted only at the final FP32/);
    expect(() => aceFp16VaeConv1dWgsl(
      k7TailShape(63),
      true,
      "float32",
    )).toThrow(/final no-bias raw-waveform boundary/);
  });

  it.each([
    [{ ...k7TailShape(65), kernelSize: 5 }, /requires biased K1/],
    [{ ...k7TailShape(65), stride: 2 }, /stride1/],
    [{ ...k7TailShape(65), dilation: 2, padding: 6 }, /dilation1,3,9/],
    [{ ...k7TailShape(65), padding: 4 }, /padding=dilation\*3/],
    [{ ...k1TailShape(65), dilation: 2 }, /requires biased K1/],
    [{ ...k1TailShape(65), padding: 1 }, /padding0/],
  ])("rejects unsupported production geometry", (shape, reason) => {
    expect(() => planAceFp16VaeConv1d(shape, "float16")).toThrow(reason);
  });

  it("compiles once, shares one bind group, and encodes exact dynamic controls", async () => {
    const device = fakeDevice();
    const kernel = AceFp16VaeConv1dKernel.create(device);
    const shape = k7PrincipalShape();
    const plan = planAceFp16VaeConv1d(shape, "float16");
    const bindings = bindingsFor(shape, "float16", true);
    const control = fakeBuffer(1_024);
    const first = await kernel.createDispatch(
      "first",
      shape,
      bindings,
      "float16",
      rangeBinding(control, 256, 0, 32 * shape.outputChannels),
    );
    const second = await kernel.createDispatch(
      "second",
      shape,
      bindings,
      "float16",
      rangeBinding(control, 512, 32 * shape.outputChannels, 16 * shape.outputChannels),
    );
    expect(first.plan).toMatchObject({ family: "k7" });
    expect(first.outputRange).toMatchObject({
      firstOutputRow: 0,
      outputRowCount: 32,
      workgroupsX: 2,
      workgroupsY: 128,
    });
    expect(second.outputRange).toMatchObject({
      firstOutputRow: 32,
      outputRowCount: 16,
      workgroupsX: 1,
      workgroupsY: 128,
    });
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();

    const layout = device.createBindGroupLayout.mock.calls[0]?.[0] as
      GPUBindGroupLayoutDescriptor;
    expect(Array.from(layout.entries).map(({ buffer }) => buffer?.minBindingSize))
      .toEqual([
        plan.inputBindingBytes,
        plan.weightBindingBytes,
        plan.biasBindingBytes,
        plan.outputBindingBytes,
        16,
      ]);
    expect(Array.from(layout.entries).at(-1)?.buffer).toEqual({
      type: "uniform",
      hasDynamicOffset: true,
      minBindingSize: 16,
    });
    const group = device.createBindGroup.mock.calls[0]?.[0] as
      GPUBindGroupDescriptor;
    expect((Array.from(group.entries).at(-1)?.resource as GPUBufferBinding))
      .toEqual({ buffer: control, offset: 0, size: 16 });

    const pass = fakePass();
    first.encode(pass);
    second.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([[256], [512]]);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [2, 128, 1],
      [1, 128, 1],
    ]);

    kernel.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
    await expect(kernel.createDispatch(
      "after-destroy",
      shape,
      bindings,
      "float16",
      rangeBinding(control, 256, 0, 32 * shape.outputChannels),
    )).rejects.toThrow(/was destroyed/);
  });

  it("keys K1, K7, bias, and output storage as distinct pipelines", async () => {
    const device = fakeDevice();
    const kernel = AceFp16VaeConv1dKernel.create(device);
    const control = fakeBuffer(1_024);
    const k1 = k1TailShape(65);
    const k7 = k7TailShape(65);
    await kernel.createDispatch(
      "k1",
      k1,
      bindingsFor(k1, "float16", true),
      "float16",
      fullRange(control, 0, k1),
    );
    await kernel.createDispatch(
      "k7-biased",
      k7,
      bindingsFor(k7, "float16", true),
      "float16",
      fullRange(control, 256, k7),
    );
    await kernel.createDispatch(
      "k7-final",
      k7,
      bindingsFor(k7, "float32", false),
      "float32",
      fullRange(control, 512, k7),
    );
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(3);
    const sources = device.createShaderModule.mock.calls.map(([descriptor]) =>
      (descriptor as GPUShaderModuleDescriptor).code
    );
    expect(sources[0]).toContain("kernel < 1u");
    expect(sources[1]).toContain("kernel < 7u");
    expect(sources[2]).toContain("output: array<f32>;");
  });

  it("fails closed before compilation on features, limits, controls, and aliases", async () => {
    expect(() => AceFp16VaeConv1dKernel.create(fakeDevice({
      shaderF16: false,
    }))).toThrow(/requires WebGPU shader-f16/);
    expect(() => AceFp16VaeConv1dKernel.create(fakeDevice({
      maximumInvocations: 127,
    }))).toThrow(/128-lane/);
    expect(() => AceFp16VaeConv1dKernel.create(fakeDevice({
      maximumWorkgroupSizeY: 7,
    }))).toThrow(/16x8/);

    const shape = k7TailShape(65);
    const bindings = bindingsFor(shape, "float16", true);
    const storageDevice = fakeDevice({ maximumWorkgroupStorage: 3_215 });
    await expect(AceFp16VaeConv1dKernel.create(storageDevice).createDispatch(
      "storage",
      shape,
      bindings,
      "float16",
      fullRange(fakeBuffer(256), 0, shape),
    )).rejects.toThrow(/3216 workgroup-storage bytes/);
    expect(storageDevice.createShaderModule).not.toHaveBeenCalled();

    const alignmentDevice = fakeDevice({ uniformAlignment: 512 });
    await expect(AceFp16VaeConv1dKernel.create(alignmentDevice).createDispatch(
      "unaligned-control",
      shape,
      bindings,
      "float16",
      fullRange(fakeBuffer(1_024), 256, shape),
    )).rejects.toThrow(/aligned 16-byte immutable record/);
    expect(alignmentDevice.createShaderModule).not.toHaveBeenCalled();

    const aliasDevice = fakeDevice();
    const plan = planAceFp16VaeConv1d(shape, "float16");
    const shared = fakeBuffer(plan.inputBindingBytes + plan.outputBindingBytes);
    await expect(AceFp16VaeConv1dKernel.create(aliasDevice).createDispatch(
      "alias",
      shape,
      {
        ...bindings,
        input: { buffer: shared, offset: 0, size: plan.inputBindingBytes },
        output: { buffer: shared, offset: 256, size: plan.outputBindingBytes },
      },
      "float16",
      fullRange(fakeBuffer(256), 0, shape),
    )).rejects.toThrow(/input and output bindings must not overlap/);
    expect(aliasDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("reports shader diagnostics and evicts failed compilation for retry", async () => {
    const device = fakeDevice({
      compilationMessageBatches: [[{
        message: "synthetic FP16 production diagnostic",
        type: "error",
        lineNum: 41,
        linePos: 9,
      }], []],
    });
    const kernel = AceFp16VaeConv1dKernel.create(device);
    const shape = k1TailShape(65);
    const bindings = bindingsFor(shape, "float16", true);
    const range = fullRange(fakeBuffer(256), 0, shape);
    await expect(kernel.createDispatch(
      "diagnostic",
      shape,
      bindings,
      "float16",
      range,
    )).rejects.toThrow(/41:9 synthetic FP16 production diagnostic/);
    expect(device.createComputePipelineAsync).not.toHaveBeenCalled();
    await expect(kernel.createDispatch(
      "retry",
      shape,
      bindings,
      "float16",
      range,
    )).resolves.toMatchObject({
      kernelId: ACE_FP16_VAE_CONV1D_PORTABLE_KERNEL_ID,
    });
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
  });
});

expect(ACE_FP16_VAE_CONV1D_K1_KERNEL_SIZE).toBe(1);
expect(ACE_FP16_VAE_CONV1D_K7_KERNEL_SIZE).toBe(7);
expect(ACE_FP16_VAE_CONV1D_K7_SUPPORTED_DILATIONS).toEqual([1, 3, 9]);
expect(ACE_FP16_VAE_CONV1D_INPUT_CHANNEL_CHUNK).toBe(64);
expect(ACE_FP16_VAE_CONV1D_TILE_FRAMES).toBe(16);
expect(ACE_FP16_VAE_CONV1D_TILE_CHANNELS).toBe(8);
expect(ACE_FP16_VAE_CONV1D_INPUT_TILE_STRIDE).toBe(17);
expect(ACE_FP16_VAE_CONV1D_WEIGHT_TILE_STRIDE).toBe(65);
expect(ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_X).toBe(16);
expect(ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE_Y).toBe(8);
expect(ACE_FP16_VAE_CONV1D_WORKGROUP_SIZE).toBe(128);

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

function k7PrincipalShape(): AceVaeConv1dShape {
  return {
    batch: 1,
    inputFrames: 256,
    inputChannels: 1_024,
    outputChannels: 1_024,
    kernelSize: 7,
    stride: 1,
    dilation: 1,
    padding: 3,
  };
}

function k7TailShape(inputChannels: number): AceVaeConv1dShape {
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

function k1TailShape(inputChannels: number): AceVaeConv1dShape {
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

function bindingsFor(
  shape: AceVaeConv1dShape,
  outputStorage: AceFp16VaeConv1dOutputStorage,
  hasBias: boolean,
): AceFp16VaeConv1dBindings {
  const plan = planAceFp16VaeConv1d(shape, outputStorage);
  const bindings = {
    input: fakeBinding(plan.inputBindingBytes),
    weight: fakeBinding(plan.weightBindingBytes),
    output: fakeBinding(plan.outputBindingBytes),
  };
  return hasBias
    ? { ...bindings, bias: fakeBinding(plan.biasBindingBytes) }
    : bindings;
}

function fullRange(
  buffer: GPUBuffer,
  offset: number,
  shape: AceVaeConv1dShape,
): AceVaeOutputRangeBinding {
  return rangeBinding(
    buffer,
    offset,
    0,
    shape.batch * shape.inputFrames * shape.outputChannels,
  );
}

function rangeBinding(
  buffer: GPUBuffer,
  offset: number,
  base: number,
  count: number,
): AceVaeOutputRangeBinding {
  return {
    base,
    count,
    control: { buffer, offset, size: 16 },
  };
}

function expectForbiddenMathAbsent(source: string): void {
  expect(source).not.toMatch(/\bfma\s*\(/);
  expect(source).not.toMatch(/\bdot\s*\(/);
  expect(source).not.toMatch(/\batomic\w*\s*\(/);
  expect(source).not.toContain("subgroup");
  expect(source).not.toContain("return;");
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
  readonly shaderF16?: boolean;
  readonly maximumInvocations?: number;
  readonly maximumWorkgroupSizeX?: number;
  readonly maximumWorkgroupSizeY?: number;
  readonly maximumWorkgroupStorage?: number;
  readonly maximumDispatch?: number;
  readonly maximumStorageBinding?: number;
  readonly maximumBuffer?: number;
  readonly uniformAlignment?: number;
  readonly storageAlignment?: number;
  readonly compilationMessageBatches?: readonly (
    readonly Partial<GPUCompilationMessage>[]
  )[];
} = {}): FakeDevice {
  const compilationMessageBatches = [
    ...(options.compilationMessageBatches ?? [[]]),
  ];
  return {
    features: new Set(options.shaderF16 === false ? [] : ["shader-f16"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: options.maximumInvocations ?? 256,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSizeX ?? 256,
      maxComputeWorkgroupSizeY: options.maximumWorkgroupSizeY ?? 256,
      maxComputeWorkgroupStorageSize: options.maximumWorkgroupStorage ?? 32_768,
      maxComputeWorkgroupsPerDimension: options.maximumDispatch ?? 65_535,
      maxStorageBufferBindingSize:
        options.maximumStorageBinding ?? 1_073_741_824,
      maxBufferSize: options.maximumBuffer ?? 1_073_741_824,
      minUniformBufferOffsetAlignment: options.uniformAlignment ?? 256,
      minStorageBufferOffsetAlignment: options.storageAlignment ?? 256,
    },
    createShaderModule: vi.fn(() => {
      const messages = compilationMessageBatches.shift() ?? [];
      return {
        label: "module",
        getCompilationInfo: vi.fn(async () => ({ messages })),
      };
    }),
    createBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    createPipelineLayout: vi.fn(() => ({ label: "pipeline-layout" })),
    createComputePipelineAsync: vi.fn(async () => fakePipeline()),
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
