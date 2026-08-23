import { describe, expect, it, vi } from "vitest";

import {
  ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID,
  ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
  ACE_FP16_VAE_POINTWISE_WORKGROUP_SIZE,
  AceFp16VaePointwiseKernel,
  aceFp16VaeAddWgsl,
  aceFp16VaeIngressWgsl,
  planAceFp16VaeAdd,
  planAceFp16VaeIngress,
  planAceFp16VaePointwiseRange,
  type AceFp16VaeAddBindings,
  type AceFp16VaeIngressBindings,
} from "../src/webgpu/kernels/vae-pointwise-fp16.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderAddOperation,
} from "../src/webgpu/vae-decoder.js";
import type {
  AceVaeOutputRangeBinding,
  AceVaePointwiseShape,
} from "../src/webgpu/kernels/vae-primitives.js";

describe("production portable FP16 VAE ingress and Add", () => {
  it("covers all 15 Add operations with the exact frozen B-256 quanta", () => {
    const graph = planAceVaeDecoder(256);
    const cooperative = planAceVaeDecoderQuanta(graph);
    const operations = graph.operations
      .map((operation, index) => ({ operation, index }))
      .filter((entry): entry is {
        readonly operation: AceVaeDecoderAddOperation;
        readonly index: number;
      } => entry.operation.kind === "add");
    expect(operations).toHaveLength(15);

    let quantumCount = 0;
    for (const { operation, index } of operations) {
      const plan = planAceFp16VaeAdd(operation.shape);
      expect(plan).toMatchObject({
        operation: "add",
        batch: operation.shape.batch,
        frames: operation.shape.frames,
        channels: operation.shape.channels,
        elements:
          operation.shape.batch * operation.shape.frames *
          operation.shape.channels,
        sourceStorageBytes:
          operation.shape.batch * operation.shape.frames *
          operation.shape.channels * 2,
        outputStorageBytes:
          operation.shape.batch * operation.shape.frames *
          operation.shape.channels * 2,
        workgroupSize: 256,
      });
      expect(Object.isFrozen(plan)).toBe(true);

      const quanta = cooperative.quanta.filter((quantum) =>
        quantum.operationIndex === index
      );
      let cursor = 0;
      for (const quantum of quanta) {
        expect(quantum.operationKind).toBe("add");
        expect(quantum.operationLabel).toBe(operation.label);
        expect(quantum.estimatedMaximumMultiplyAccumulates).toBe(0);
        expect(quantum.primitives).toHaveLength(1);
        const primitive = quantum.primitives[0]!;
        expect(primitive.outputBase).toBe(cursor);
        expect(primitive.outputBase).toBe(quantum.logicalOutputBase);
        expect(primitive.outputCount).toBe(quantum.logicalOutputCount);
        const range = planAceFp16VaePointwiseRange(plan, {
          base: primitive.outputBase,
          count: primitive.outputCount,
        });
        expect(range).toEqual({
          base: primitive.outputBase,
          count: primitive.outputCount,
          workgroupsX: Math.min(
            Math.ceil(primitive.outputCount / 256),
            65_535,
          ),
          workgroupsY: Math.ceil(
            Math.ceil(primitive.outputCount / 256) / 65_535,
          ),
        });
        cursor += primitive.outputCount;
        quantumCount += 1;
      }
      expect(cursor).toBe(plan.elements);
      expect(() => aceFp16VaeAddWgsl()).not.toThrow();
    }
    expect(quantumCount).toBe(348);
  });

  it("plans the complete B-256 FP32-to-FP16 decoder ingress", () => {
    const graph = planAceVaeDecoder(256);
    const shape = {
      batch: graph.batch,
      frames: graph.inputFrames,
      channels: graph.config.decoderInputChannels,
    };
    const plan = planAceFp16VaeIngress(shape);
    expect(plan).toEqual({
      operation: "ingress",
      batch: 1,
      frames: 256,
      channels: 64,
      elements: 16_384,
      sourceStorageBytes: 65_536,
      sourceBindingBytes: 65_536,
      outputStorageBytes: 32_768,
      outputBindingBytes: 32_768,
      workgroupSize: 256,
      workgroupsX: 64,
      workgroupsY: 1,
    });
    expect(planAceFp16VaePointwiseRange(plan, {
      base: 0,
      count: graph.inputElements,
    })).toEqual({
      base: 0,
      count: 16_384,
      workgroupsX: 64,
      workgroupsY: 1,
    });
  });

  it("separates odd FP16 payload bytes from four-byte binding padding", () => {
    expect(planAceFp16VaeIngress({
      batch: 1,
      frames: 1,
      channels: 3,
    })).toMatchObject({
      sourceStorageBytes: 12,
      sourceBindingBytes: 12,
      outputStorageBytes: 6,
      outputBindingBytes: 8,
    });
    expect(planAceFp16VaeAdd({
      batch: 1,
      frames: 1,
      channels: 3,
    })).toMatchObject({
      sourceStorageBytes: 6,
      sourceBindingBytes: 8,
      outputStorageBytes: 6,
      outputBindingBytes: 8,
    });
  });

  it("pins FP16 ingress rounding, signed zero, and subnormal boundaries", () => {
    const source = aceFp16VaeIngressWgsl();
    expect(source).toContain(
      `// kernel-id: ${ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID}`,
    );
    expect(source).toContain("enable f16;");
    expect(source).toContain("input: array<f32>;");
    expect(source).toContain("output: array<f16>;");
    expect(source).toContain("output[index] = f16(input[index]);");
    expect(source).toContain(
      "if (quantum_index < output_range.output_count)",
    );
    expectForbiddenMathAbsent(source);

    expect(numberToFloat16Bits(-0)).toBe(0x8000);
    expect(Object.is(float16BitsToNumber(0x8000), -0)).toBe(true);
    expect(numberToFloat16Bits(2 ** -24)).toBe(0x0001);
    expect(numberToFloat16Bits(1 + 2 ** -11)).toBe(0x3c00);
    expect(numberToFloat16Bits(1 + 3 * 2 ** -11)).toBe(0x3c02);
  });

  it("pins FP16 Add to FP32 register arithmetic and one FP16 store", () => {
    const source = aceFp16VaeAddWgsl();
    expect(source).toContain(
      `// kernel-id: ${ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID}`,
    );
    expect(source).toContain("left: array<f16>;");
    expect(source).toContain("right: array<f16>;");
    expect(source).toContain("output: array<f16>;");
    expect(source).toContain("let left_operand: f32 = f32(left[index]);");
    expect(source).toContain("let right_operand: f32 = f32(right[index]);");
    expect(source).toContain("let sum: f32 = left_operand + right_operand;");
    expect(source).toContain("output[index] = f16(sum);");
    expect(source.match(/f16\(sum\)/g)).toHaveLength(1);
    expectForbiddenMathAbsent(source);

    expect(addFloat16Bits(0x8000, 0x8000)).toBe(0x8000);
    expect(addFloat16Bits(0x0001, 0x0001)).toBe(0x0002);
    expect(addFloat16Bits(0x3c00, 0x1000)).toBe(0x3c00);
    expect(addFloat16Bits(0x3c01, 0x1000)).toBe(0x3c02);
  });

  it("retains one guarded invocation for an odd tail range", () => {
    const plan = planAceFp16VaeAdd({ batch: 1, frames: 1, channels: 257 });
    expect(planAceFp16VaePointwiseRange(plan, {
      base: 256,
      count: 1,
    })).toEqual({ base: 256, count: 1, workgroupsX: 1, workgroupsY: 1 });
    expect(aceFp16VaeAddWgsl()).toContain(
      "let index = output_range.first_output + quantum_index;",
    );
  });

  it("guards rectangular dispatch padding before u32 index multiplication", () => {
    const maximumU32 = 0xffff_ffff;
    const plan = planAceFp16VaeAdd({
      batch: 1,
      frames: 1,
      channels: maximumU32,
    });
    const range = planAceFp16VaePointwiseRange(plan, {
      base: 0,
      count: maximumU32,
    });
    expect(range).toEqual({
      base: 0,
      count: maximumU32,
      workgroupsX: 65_535,
      workgroupsY: 257,
    });

    const source = aceFp16VaeAddWgsl();
    const activeGuard = source.indexOf("if (workgroup < active_workgroups)");
    const indexMultiply = source.indexOf("workgroup * 256u + lane");
    expect(activeGuard).toBeGreaterThan(0);
    expect(indexMultiply).toBeGreaterThan(activeGuard);
    expect(source).toContain(
      "(output_range.output_count % 256u) != 0u",
    );
  });

  it("caches exact layouts and bind groups while using dynamic controls", async () => {
    const device = fakeDevice();
    const kernel = AceFp16VaePointwiseKernel.create(device);
    const shape = { batch: 1, frames: 1, channels: 257 };
    const ingressPlan = planAceFp16VaeIngress(shape);
    const ingressBindings = ingressBindingsFor(shape);
    const control = fakeBuffer(1_024);
    const first = await kernel.createIngressDispatch(
      "ingress-first",
      shape,
      ingressBindings,
      rangeBinding(control, 256, 0, 256),
    );
    const second = await kernel.createIngressDispatch(
      "ingress-tail",
      shape,
      ingressBindings,
      rangeBinding(control, 512, 256, 1),
    );
    const add = await kernel.createAddDispatch(
      "add",
      shape,
      addBindingsFor(shape),
      rangeBinding(control, 768, 0, 257),
    );

    expect(first.kernelId).toBe(ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID);
    expect(add.kernelId).toBe(ACE_FP16_VAE_ADD_PORTABLE_KERNEL_ID);
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createBindGroup).toHaveBeenCalledTimes(2);
    const ingressLayout = device.createBindGroupLayout.mock.calls[0]?.[0] as
      GPUBindGroupLayoutDescriptor;
    expect(Array.from(ingressLayout.entries).map(({ buffer }) =>
      buffer?.minBindingSize
    )).toEqual([
      ingressPlan.sourceBindingBytes,
      ingressPlan.outputBindingBytes,
      16,
    ]);
    expect(Array.from(ingressLayout.entries).at(-1)?.buffer).toEqual({
      type: "uniform",
      hasDynamicOffset: true,
      minBindingSize: 16,
    });
    const ingressGroup = device.createBindGroup.mock.calls[0]?.[0] as
      GPUBindGroupDescriptor;
    expect((Array.from(ingressGroup.entries).at(-1)?.resource as
      GPUBufferBinding)).toEqual({ buffer: control, offset: 0, size: 16 });

    const pass = fakePass();
    first.encode(pass);
    second.encode(pass);
    add.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2])).toEqual([
      [256],
      [512],
      [768],
    ]);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [1, 1, 1],
      [1, 1, 1],
      [2, 1, 1],
    ]);
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(first.outputRange)).toBe(true);
  });

  it("rejects malformed shapes and ranges before shader compilation", async () => {
    expect(() => planAceFp16VaeAdd({
      batch: 0,
      frames: 1,
      channels: 1,
    })).toThrow(/positive safe integer/);
    expect(() => planAceFp16VaeIngress({
      batch: 0xffff_ffff,
      frames: 2,
      channels: 1,
    })).toThrow(/u32 indexing domain/);

    const plan = planAceFp16VaeAdd({ batch: 1, frames: 2, channels: 3 });
    for (const range of [
      { base: -1, count: 1 },
      { base: 0, count: 0 },
      { base: 6, count: 1 },
      { base: 5, count: 2 },
      { base: 0xffff_ffff, count: 1 },
    ]) {
      expect(() => planAceFp16VaePointwiseRange(plan, range)).toThrow(
        /non-empty and inside/,
      );
    }

    const device = fakeDevice();
    const kernel = AceFp16VaePointwiseKernel.create(device);
    await expect(kernel.createAddDispatch(
      "bad-range",
      plan,
      addBindingsFor(plan),
      rangeBinding(fakeBuffer(256), 0, 5, 2),
    )).rejects.toThrow(/non-empty and inside/);
    expect(device.createShaderModule).not.toHaveBeenCalled();
  });

  it("fails closed on features, limits, binding spans, controls, and aliases", async () => {
    expect(() => AceFp16VaePointwiseKernel.create(fakeDevice({
      shaderF16: false,
    }))).toThrow(/requires WebGPU shader-f16/);
    expect(() => AceFp16VaePointwiseKernel.create(fakeDevice({
      maximumInvocations: 255,
    }))).toThrow(/256-lane/);
    expect(() => AceFp16VaePointwiseKernel.create(fakeDevice({
      maximumWorkgroupSizeX: 255,
    }))).toThrow(/256-lane/);

    const shape = { batch: 1, frames: 1, channels: 257 };
    const plan = planAceFp16VaeAdd(shape);
    const bindings = addBindingsFor(shape);
    const control = fakeBuffer(1_024);
    const cases: readonly [FakeDevice, AceFp16VaeAddBindings,
      AceVaeOutputRangeBinding, RegExp][] = [
        [
          fakeDevice({ maximumDispatch: 1 }),
          bindings,
          rangeBinding(control, 0, 0, 257),
          /dispatch dimension/,
        ],
        [
          fakeDevice({ maximumStorageBinding: plan.sourceBindingBytes - 1 }),
          bindings,
          rangeBinding(control, 0, 0, 257),
          /storage binding limit/,
        ],
        [
          fakeDevice({ maximumUniformBinding: 15 }),
          bindings,
          rangeBinding(control, 0, 0, 257),
          /invalid buffer limits/,
        ],
        [
          fakeDevice({ maximumBuffer: plan.sourceBindingBytes - 1 }),
          bindings,
          rangeBinding(control, 0, 0, 257),
          /buffer limit/,
        ],
        [
          fakeDevice({ storageAlignment: 3 }),
          bindings,
          rangeBinding(control, 0, 0, 257),
          /invalid storage alignment/,
        ],
        [
          fakeDevice({ uniformAlignment: 6 }),
          bindings,
          rangeBinding(control, 0, 0, 257),
          /invalid uniform alignment/,
        ],
        [
          fakeDevice(),
          { ...bindings, left: fakeBinding(plan.sourceBindingBytes - 4) },
          rangeBinding(control, 0, 0, 257),
          /does not expose an aligned/,
        ],
        [
          fakeDevice(),
          {
            ...bindings,
            left: {
              buffer: fakeBuffer(plan.sourceBindingBytes + 256),
              offset: 4,
              size: plan.sourceBindingBytes,
            },
          },
          rangeBinding(control, 0, 0, 257),
          /does not expose an aligned/,
        ],
        [
          fakeDevice(),
          bindings,
          rangeBinding(control, 4, 0, 257),
          /aligned 16-byte immutable record/,
        ],
        [
          fakeDevice({ maximumBuffer: 0x1_0000_0200 }),
          bindings,
          rangeBinding(
            fakeBuffer(0x1_0000_0200),
            0x1_0000_0000,
            0,
            257,
          ),
          /aligned 16-byte immutable record/,
        ],
      ];
    for (const [device, candidateBindings, range, reason] of cases) {
      await expect(AceFp16VaePointwiseKernel.create(device).createAddDispatch(
        "invalid",
        shape,
        candidateBindings,
        range,
      )).rejects.toThrow(reason);
      expect(device.createShaderModule).not.toHaveBeenCalled();
    }

    const shared = fakeBuffer(plan.sourceBindingBytes * 3);
    const aliases: readonly AceFp16VaeAddBindings[] = [
      {
        left: { buffer: shared, offset: 0, size: plan.sourceBindingBytes },
        right: { buffer: shared, offset: 0, size: plan.sourceBindingBytes },
        output: fakeBinding(plan.outputBindingBytes),
      },
      {
        left: { buffer: shared, offset: 0, size: plan.sourceBindingBytes },
        right: fakeBinding(plan.sourceBindingBytes),
        output: { buffer: shared, offset: 0, size: plan.outputBindingBytes },
      },
    ];
    for (const alias of aliases) {
      const aliasDevice = fakeDevice();
      await expect(AceFp16VaePointwiseKernel.create(aliasDevice)
        .createAddDispatch(
          "alias",
          shape,
          alias,
          rangeBinding(control, 0, 0, 257),
        )).rejects.toThrow(/must not overlap/);
      expect(aliasDevice.createShaderModule).not.toHaveBeenCalled();
    }

    const controlAliasDevice = fakeDevice();
    await expect(AceFp16VaePointwiseKernel.create(controlAliasDevice)
      .createAddDispatch(
        "control-alias",
        shape,
        {
          left: { buffer: shared, offset: 0, size: plan.sourceBindingBytes },
          right: fakeBinding(plan.sourceBindingBytes),
          output: fakeBinding(plan.outputBindingBytes),
        },
        rangeBinding(shared, 0, 0, 257),
      )).rejects.toThrow(/left and range control bindings must not overlap/);
    expect(controlAliasDevice.createShaderModule).not.toHaveBeenCalled();

    const ingressPlan = planAceFp16VaeIngress(shape);
    const ingressShared = fakeBuffer(
      ingressPlan.sourceBindingBytes + ingressPlan.outputBindingBytes,
    );
    const ingressAliasDevice = fakeDevice();
    await expect(AceFp16VaePointwiseKernel.create(ingressAliasDevice)
      .createIngressDispatch(
        "ingress-alias",
        shape,
        {
          input: {
            buffer: ingressShared,
            offset: 0,
            size: ingressPlan.sourceBindingBytes,
          },
          output: {
            buffer: ingressShared,
            offset: 0,
            size: ingressPlan.outputBindingBytes,
          },
        },
        rangeBinding(control, 0, 0, 257),
      )).rejects.toThrow(/input and output bindings must not overlap/);
    expect(ingressAliasDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("reports diagnostics and evicts failed compilation for retry", async () => {
    const device = fakeDevice({
      compilationMessageBatches: [[{
        message: "synthetic FP16 pointwise diagnostic",
        type: "error",
        lineNum: 17,
        linePos: 5,
      }], []],
    });
    const kernel = AceFp16VaePointwiseKernel.create(device);
    const shape = { batch: 1, frames: 1, channels: 3 };
    const bindings = ingressBindingsFor(shape);
    const range = fullRange(fakeBuffer(256), 0, shape);
    await expect(kernel.createIngressDispatch(
      "diagnostic",
      shape,
      bindings,
      range,
    )).rejects.toThrow(/17:5 synthetic FP16 pointwise diagnostic/);
    expect(device.createComputePipelineAsync).not.toHaveBeenCalled();
    await expect(kernel.createIngressDispatch(
      "retry",
      shape,
      bindings,
      range,
    )).resolves.toMatchObject({
      kernelId: ACE_FP16_VAE_INGRESS_PORTABLE_KERNEL_ID,
    });
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
  });

  it("fails a destroy race before bind-group creation and invalidates dispatches", async () => {
    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pipelinePromise = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const device = fakeDevice({ pipelinePromise });
    const kernel = AceFp16VaePointwiseKernel.create(device);
    const shape = { batch: 1, frames: 1, channels: 3 };
    const pending = kernel.createAddDispatch(
      "destroy-race",
      shape,
      addBindingsFor(shape),
      fullRange(fakeBuffer(256), 0, shape),
    );
    await vi.waitFor(() => {
      expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    });
    kernel.destroy();
    kernel.destroy();
    resolvePipeline(fakePipeline());
    await expect(pending).rejects.toThrow(/was destroyed/);
    expect(device.createBindGroup).not.toHaveBeenCalled();

    const liveKernel = AceFp16VaePointwiseKernel.create(fakeDevice());
    const dispatch = await liveKernel.createIngressDispatch(
      "live",
      shape,
      ingressBindingsFor(shape),
      fullRange(fakeBuffer(256), 0, shape),
    );
    liveKernel.destroy();
    expect(() => dispatch.encode(fakePass())).toThrow(/was destroyed/);
    await expect(liveKernel.createIngressDispatch(
      "dead",
      shape,
      ingressBindingsFor(shape),
      fullRange(fakeBuffer(256), 0, shape),
    )).rejects.toThrow(/was destroyed/);
  });
});

expect(ACE_FP16_VAE_POINTWISE_WORKGROUP_SIZE).toBe(256);

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

function ingressBindingsFor(
  shape: AceVaePointwiseShape,
): AceFp16VaeIngressBindings {
  const plan = planAceFp16VaeIngress(shape);
  return {
    input: fakeBinding(plan.sourceBindingBytes),
    output: fakeBinding(plan.outputBindingBytes),
  };
}

function addBindingsFor(shape: AceVaePointwiseShape): AceFp16VaeAddBindings {
  const plan = planAceFp16VaeAdd(shape);
  return {
    left: fakeBinding(plan.sourceBindingBytes),
    right: fakeBinding(plan.sourceBindingBytes),
    output: fakeBinding(plan.outputBindingBytes),
  };
}

function fullRange(
  buffer: GPUBuffer,
  offset: number,
  shape: AceVaePointwiseShape,
): AceVaeOutputRangeBinding {
  return rangeBinding(
    buffer,
    offset,
    0,
    shape.batch * shape.frames * shape.channels,
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
  expect(source).not.toContain("workgroupBarrier");
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
  readonly maximumDispatch?: number;
  readonly maximumStorageBinding?: number;
  readonly maximumUniformBinding?: number;
  readonly maximumBuffer?: number;
  readonly uniformAlignment?: number;
  readonly storageAlignment?: number;
  readonly compilationMessageBatches?: readonly (
    readonly Partial<GPUCompilationMessage>[]
  )[];
  readonly pipelinePromise?: Promise<GPUComputePipeline>;
} = {}): FakeDevice {
  const compilationMessageBatches = [
    ...(options.compilationMessageBatches ?? [[]]),
  ];
  return {
    features: new Set(options.shaderF16 === false ? [] : ["shader-f16"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: options.maximumInvocations ?? 256,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSizeX ?? 256,
      maxComputeWorkgroupsPerDimension: options.maximumDispatch ?? 65_535,
      maxStorageBufferBindingSize:
        options.maximumStorageBinding ?? 1_073_741_824,
      maxUniformBufferBindingSize: options.maximumUniformBinding ?? 65_536,
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
    createComputePipelineAsync: vi.fn(
      () => options.pipelinePromise ?? Promise.resolve(fakePipeline()),
    ),
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

function addFloat16Bits(leftBits: number, rightBits: number): number {
  return numberToFloat16Bits(Math.fround(
    float16BitsToNumber(leftBits) + float16BitsToNumber(rightBits),
  ));
}

function numberToFloat16Bits(value: number): number {
  floatScratch.setFloat32(0, Math.fround(value), false);
  const bits = floatScratch.getUint32(0, false);
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7f_ffff;
  if (exponent === 0xff) {
    return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  }
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const significand = mantissa | 0x80_0000;
    const shift = 14 - halfExponent;
    const truncated = significand >>> shift;
    const remainder = significand & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    return sign | (truncated + (
      remainder > halfway ||
        (remainder === halfway && (truncated & 1) !== 0)
        ? 1
        : 0
    ));
  }
  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (
    remainder > 0x1000 ||
    (remainder === 0x1000 && (halfMantissa & 1) !== 0)
  ) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      const nextExponent = halfExponent + 1;
      return sign | (nextExponent >= 0x1f ? 0x7c00 : nextExponent << 10);
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

function float16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) {
    return mantissa === 0 ? sign * 0 : sign * mantissa * 2 ** -24;
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Infinity : Number.NaN;
  }
  return sign * (1 + mantissa / 1_024) * 2 ** (exponent - 15);
}

const floatScratch = new DataView(new ArrayBuffer(4));
