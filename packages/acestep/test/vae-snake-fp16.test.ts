import { createHash } from "node:crypto";

import { describe, expect, it, vi } from "vitest";

import {
  ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
  ACE_FP16_VAE_SNAKE_WORKGROUP_SIZE,
  AceFp16VaeSnakeKernel,
  aceFp16VaeSnakeWgsl,
  planAceFp16VaeSnake,
  planAceFp16VaeSnakeRange,
  type AceFp16VaeSnakeBindings,
} from "../src/webgpu/kernels/vae-snake-fp16.js";
import type {
  AceVaeOutputRangeBinding,
  AceVaePointwiseShape,
} from "../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderSnakeOperation,
} from "../src/webgpu/vae-decoder.js";

describe("production portable FP16 VAE Snake", () => {
  it("enumerates all 36 exact B-256 graph operations", () => {
    const graph = planAceVaeDecoder(256);
    const operations = graph.operations
      .map((operation, index) => ({ operation, index }))
      .filter((entry): entry is {
        readonly operation: AceVaeDecoderSnakeOperation;
        readonly index: number;
      } => entry.operation.kind === "snake");

    expect(operations.map(({ operation, index }) => [
      index,
      operation.label,
      operation.shape.frames,
      operation.shape.channels,
    ])).toEqual(EXPECTED_B256_SNAKE_OPERATIONS.map((entry) =>
      entry.slice(0, 4)
    ));
    expect(operations).toHaveLength(36);

    for (const { operation } of operations) {
      const plan = planAceFp16VaeSnake(operation.shape);
      expect(plan).toMatchObject({
        batch: 1,
        frames: operation.shape.frames,
        channels: operation.shape.channels,
        elements: operation.shape.frames * operation.shape.channels,
        inputStorageBytes:
          operation.shape.frames * operation.shape.channels * 2,
        alphaStorageBytes: operation.shape.channels * 2,
        betaStorageBytes: operation.shape.channels * 2,
        outputStorageBytes:
          operation.shape.frames * operation.shape.channels * 2,
        workgroupSize: 256,
      });
      expect(Object.isFrozen(plan)).toBe(true);
      expect(() => aceFp16VaeSnakeWgsl(operation.shape)).not.toThrow();
    }

    const largest = planAceFp16VaeSnake(
      operations.at(-1)!.operation.shape,
    );
    expect(largest).toMatchObject({
      elements: 62_914_560,
      inputBindingBytes: 125_829_120,
      alphaBindingBytes: 256,
      betaBindingBytes: 256,
      outputBindingBytes: 125_829_120,
      workgroupsX: 65_535,
      workgroupsY: 4,
    });
  });

  it("enumerates and authenticates all 813 exact B-256 Snake quanta", () => {
    const graph = planAceVaeDecoder(256);
    const cooperative = planAceVaeDecoderQuanta(graph);
    let quantumCount = 0;

    for (const expected of EXPECTED_B256_SNAKE_OPERATIONS) {
      const [operationIndex, label, frames, channels, expectedCount, tail] =
        expected;
      const operation = graph.operations[operationIndex]!;
      expect(operation).toMatchObject({
        kind: "snake",
        label,
        shape: { batch: 1, frames, channels },
      });
      if (operation.kind !== "snake") throw new Error("unreachable");
      const plan = planAceFp16VaeSnake(operation.shape);
      const quanta = cooperative.quanta.filter((quantum) =>
        quantum.operationIndex === operationIndex
      );
      expect(quanta).toHaveLength(expectedCount);

      let cursor = 0;
      for (let operationQuantumIndex = 0;
        operationQuantumIndex < quanta.length;
        operationQuantumIndex += 1) {
        const quantum = quanta[operationQuantumIndex]!;
        const expectedOutputCount = operationQuantumIndex === expectedCount - 1
          ? tail
          : 1_048_576;
        expect(quantum).toMatchObject({
          id:
            `operation-${operationIndex}-${label}-quantum-${operationQuantumIndex}`,
          operationIndex,
          operationLabel: label,
          operationKind: "snake",
          logicalOutputBase: cursor,
          logicalOutputCount: expectedOutputCount,
          estimatedMaximumMultiplyAccumulates: 0,
        });
        expect(quantum.primitives).toHaveLength(1);
        const primitive = quantum.primitives[0]!;
        expect(primitive).toMatchObject({
          firstOutputChannel: 0,
          outputChannels: channels,
          outputBase: cursor,
          outputCount: expectedOutputCount,
        });
        expect(primitive.outputBase).toBe(quantum.logicalOutputBase);
        expect(primitive.outputCount).toBe(quantum.logicalOutputCount);

        const range = planAceFp16VaeSnakeRange(plan, {
          base: primitive.outputBase,
          count: primitive.outputCount,
        });
        expect(range).toEqual({
          base: cursor,
          count: expectedOutputCount,
          workgroupsX: Math.min(
            Math.ceil(expectedOutputCount / 256),
            65_535,
          ),
          workgroupsY: Math.ceil(
            Math.ceil(expectedOutputCount / 256) / 65_535,
          ),
        });
        cursor += expectedOutputCount;
        quantumCount += 1;
      }
      expect(cursor).toBe(frames * channels);
    }
    expect(quantumCount).toBe(813);

    const canonical = cooperative.quanta
      .filter((quantum) => quantum.operationKind === "snake")
      .map((quantum) => ({
        index: quantum.index,
        id: quantum.id,
        operationIndex: quantum.operationIndex,
        operationLabel: quantum.operationLabel,
        operationKind: quantum.operationKind,
        logicalOutputBase: quantum.logicalOutputBase,
        logicalOutputCount: quantum.logicalOutputCount,
        estimatedMaximumMultiplyAccumulates:
          quantum.estimatedMaximumMultiplyAccumulates,
        primitives: quantum.primitives,
      }));
    expect(createHash("sha256").update(JSON.stringify(canonical)).digest("hex"))
      .toBe("ec79060be88fba5d0a2579826f1ca50730dfba16410da09ffc048963f2623bf3");
  });

  it("separates odd FP16 payload bytes from four-byte binding padding", () => {
    expect(planAceFp16VaeSnake({
      batch: 1,
      frames: 1,
      channels: 3,
    })).toMatchObject({
      elements: 3,
      inputStorageBytes: 6,
      inputBindingBytes: 8,
      alphaStorageBytes: 6,
      alphaBindingBytes: 8,
      betaStorageBytes: 6,
      betaBindingBytes: 8,
      outputStorageBytes: 6,
      outputBindingBytes: 8,
      workgroupsX: 1,
      workgroupsY: 1,
    });
  });

  it("pins the exact FP32 nonlinear island and single FP16 boundary store", () => {
    const source = aceFp16VaeSnakeWgsl({
      batch: 1,
      frames: 2,
      channels: 3,
    });
    expect(source).toContain(
      `// kernel-id: ${ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID}`,
    );
    expect(source).toContain("enable f16;");
    expect(source).toContain("const CHANNELS: u32 = 3u;");
    expect(source).toContain("input: array<f16>;");
    expect(source).toContain("alpha: array<f16>;");
    expect(source).toContain("beta: array<f16>;");
    expect(source).toContain("output: array<f16>;");
    expect(source).not.toContain("array<f32>");
    expect(source).toContain("let value: f32 = f32(input[index]);");
    expect(source).toContain(
      "let alpha_log_scale: f32 = f32(alpha[channel]);",
    );
    expect(source).toContain(
      "let beta_log_scale: f32 = f32(beta[channel]);",
    );
    expect(source).toContain(
      "let alpha_value: f32 = exp(alpha_log_scale);",
    );
    expect(source).toContain(
      "let beta_value: f32 = exp(beta_log_scale);",
    );
    expect(source).toContain(
      "let periodic: f32 = sin(alpha_value * value);",
    );
    expect(source).toContain(
      "let reciprocal_beta: f32 = 1.0 / (beta_value + 1e-9);",
    );
    expect(source).toContain(
      "value + reciprocal_beta * periodic * periodic;",
    );
    expect(source).toContain("output[index] = f16(result);");
    expect(source.match(/\bexp\s*\(/g)).toHaveLength(2);
    expect(source.match(/\bsin\s*\(/g)).toHaveLength(1);
    expect(source.match(/f16\(result\)/g)).toHaveLength(1);

    const alpha = source.indexOf("exp(alpha_log_scale)");
    const beta = source.indexOf("exp(beta_log_scale)");
    const periodic = source.indexOf("sin(alpha_value * value)");
    const reciprocal = source.indexOf("1.0 / (beta_value + 1e-9)");
    const residual = source.indexOf(
      "value + reciprocal_beta * periodic * periodic",
    );
    const store = source.indexOf("output[index] = f16(result)");
    expect(alpha).toBeLessThan(beta);
    expect(beta).toBeLessThan(periodic);
    expect(periodic).toBeLessThan(reciprocal);
    expect(reciprocal).toBeLessThan(residual);
    expect(residual).toBeLessThan(store);
    expectForbiddenMathAbsent(source);
  });

  it("pins signed zero, subnormal, RNE, and extreme finite semantics", () => {
    expect(numberToFloat16Bits(-0)).toBe(0x8000);
    expect(Object.is(float16BitsToNumber(0x8000), -0)).toBe(true);
    expect(mixedSnakeFloat16Bits(0x0000, 0x8000, 0x0000)).toBe(0x0000);
    expect(mixedSnakeFloat16Bits(0x8000, 0x0000, 0x0000)).toBe(0x0000);
    expect(mixedSnakeFloat16Bits(0x0001, 0xcc00, 0x0000)).toBe(0x0001);
    expect(numberToFloat16Bits(1 + 2 ** -11)).toBe(0x3c00);
    expect(numberToFloat16Bits(1 + 3 * 2 ** -11)).toBe(0x3c02);
    expect(mixedSnakeFloat16Bits(0x3800, 0x4500, 0xc500)).toBe(0x5805);
    expect(mixedSnakeFloat16Bits(0x7bff, 0xbc00, 0x4000)).toBe(0x7bff);
    expect(mixedSnakeFloat16Bits(0xfbff, 0xbc00, 0x4000)).toBe(0xfbff);

    // A legitimate high-gain result overflows the FP16 boundary. The kernel
    // must expose that result instead of hiding it behind a clamp.
    expect(mixedSnakeFloat16Bits(0x3800, 0x4500, 0xca00)).toBe(0x7c00);
    const source = aceFp16VaeSnakeWgsl({
      batch: 1,
      frames: 1,
      channels: 1,
    });
    expect(source).not.toMatch(/\b(?:clamp|min|max)\s*\(/);
    expect(source).not.toContain("isFinite");
    expect(source).not.toContain("isNan");
  });

  it("retains guarded invocations for odd range and channel tails", () => {
    const plan = planAceFp16VaeSnake({
      batch: 1,
      frames: 1,
      channels: 257,
    });
    expect(planAceFp16VaeSnakeRange(plan, {
      base: 256,
      count: 1,
    })).toEqual({ base: 256, count: 1, workgroupsX: 1, workgroupsY: 1 });
    const source = aceFp16VaeSnakeWgsl(plan);
    expect(source).toContain(
      "if (quantum_index < output_range.output_count)",
    );
    expect(source).toContain(
      "let index = output_range.first_output + quantum_index;",
    );
    expect(source).toContain("let channel = index % CHANNELS;");
  });

  it("guards rectangular dispatch padding before u32 index multiplication", () => {
    const maximumU32 = 0xffff_ffff;
    const plan = planAceFp16VaeSnake({
      batch: 1,
      frames: 1,
      channels: maximumU32,
    });
    const range = planAceFp16VaeSnakeRange(plan, {
      base: 0,
      count: maximumU32,
    });
    expect(range).toEqual({
      base: 0,
      count: maximumU32,
      workgroupsX: 65_535,
      workgroupsY: 257,
    });

    const source = aceFp16VaeSnakeWgsl(plan);
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
    const kernel = AceFp16VaeSnakeKernel.create(device);
    const shape = { batch: 1, frames: 1, channels: 257 };
    const plan = planAceFp16VaeSnake(shape);
    const bindings = bindingsFor(shape);
    const control = fakeBuffer(1_024);
    const first = await kernel.createDispatch(
      "snake-first",
      shape,
      bindings,
      rangeBinding(control, 256, 0, 256),
    );
    const tail = await kernel.createDispatch(
      "snake-tail",
      shape,
      bindings,
      rangeBinding(control, 512, 256, 1),
    );

    expect(first.kernelId).toBe(ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID);
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroupLayout).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();
    const layout = device.createBindGroupLayout.mock.calls[0]?.[0] as
      GPUBindGroupLayoutDescriptor;
    expect(Array.from(layout.entries).map(({ buffer }) =>
      buffer?.minBindingSize
    )).toEqual([
      plan.inputBindingBytes,
      plan.alphaBindingBytes,
      plan.betaBindingBytes,
      plan.outputBindingBytes,
      16,
    ]);
    expect(Array.from(layout.entries).map(({ buffer }) => buffer?.type))
      .toEqual([
        "read-only-storage",
        "read-only-storage",
        "read-only-storage",
        "storage",
        "uniform",
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
    tail.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([[256], [512]]);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [1, 1, 1],
      [1, 1, 1],
    ]);
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(first.outputRange)).toBe(true);
  });

  it("accepts disjoint aligned slices while rejecting every overlap", async () => {
    const shape = { batch: 1, frames: 1, channels: 257 };
    const plan = planAceFp16VaeSnake(shape);
    const shared = fakeBuffer(4_096);
    const slicedBindings: AceFp16VaeSnakeBindings = {
      input: {
        buffer: shared,
        offset: 0,
        size: plan.inputBindingBytes,
      },
      alpha: {
        buffer: shared,
        offset: 768,
        size: plan.alphaBindingBytes,
      },
      beta: {
        buffer: shared,
        offset: 1_536,
        size: plan.betaBindingBytes,
      },
      output: {
        buffer: shared,
        offset: 2_304,
        size: plan.outputBindingBytes,
      },
    };
    await expect(AceFp16VaeSnakeKernel.create(fakeDevice()).createDispatch(
      "disjoint-slices",
      shape,
      slicedBindings,
      rangeBinding(shared, 3_072, 0, 257),
    )).resolves.toMatchObject({
      kernelId: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
    });

    const independent = bindingsFor(shape);
    const aliases: readonly [
      AceFp16VaeSnakeBindings,
      AceVaeOutputRangeBinding,
      RegExp,
    ][] = [
      [
        { ...independent, output: independent.input },
        fullRange(fakeBuffer(256), 0, shape),
        /input and output bindings must not overlap/,
      ],
      [
        { ...independent, beta: independent.alpha },
        fullRange(fakeBuffer(256), 0, shape),
        /alpha and beta bindings must not overlap/,
      ],
      [
        { ...independent, alpha: independent.input },
        fullRange(fakeBuffer(256), 0, shape),
        /input and alpha bindings must not overlap/,
      ],
      [
        independent,
        fullRange(independent.output.buffer, 0, shape),
        /output and range control bindings must not overlap/,
      ],
    ];
    for (const [bindings, range, reason] of aliases) {
      const device = fakeDevice();
      await expect(AceFp16VaeSnakeKernel.create(device).createDispatch(
        "alias",
        shape,
        bindings,
        range,
      )).rejects.toThrow(reason);
      expect(device.createShaderModule).not.toHaveBeenCalled();
    }
  });

  it("rejects malformed shapes and ranges before shader compilation", async () => {
    expect(() => planAceFp16VaeSnake({
      batch: 0,
      frames: 1,
      channels: 1,
    })).toThrow(/positive safe integer/);
    expect(() => planAceFp16VaeSnake({
      batch: 0xffff_ffff,
      frames: 2,
      channels: 1,
    })).toThrow(/u32 indexing domain/);

    const plan = planAceFp16VaeSnake({ batch: 1, frames: 2, channels: 3 });
    for (const range of [
      { base: -1, count: 1 },
      { base: 0, count: 0 },
      { base: 6, count: 1 },
      { base: 5, count: 2 },
      { base: 0xffff_ffff, count: 1 },
    ]) {
      expect(() => planAceFp16VaeSnakeRange(plan, range)).toThrow(
        /non-empty and inside/,
      );
    }

    const device = fakeDevice();
    await expect(AceFp16VaeSnakeKernel.create(device).createDispatch(
      "bad-range",
      plan,
      bindingsFor(plan),
      rangeBinding(fakeBuffer(256), 0, 5, 2),
    )).rejects.toThrow(/non-empty and inside/);
    expect(device.createShaderModule).not.toHaveBeenCalled();
  });

  it("fails closed on features, topology, limits, bindings, and controls", async () => {
    expect(() => AceFp16VaeSnakeKernel.create(fakeDevice({
      shaderF16: false,
    }))).toThrow(/requires WebGPU shader-f16/);
    expect(() => AceFp16VaeSnakeKernel.create(fakeDevice({
      maximumInvocations: 255,
    }))).toThrow(/256-lane/);
    expect(() => AceFp16VaeSnakeKernel.create(fakeDevice({
      maximumWorkgroupSizeX: 255,
    }))).toThrow(/256-lane/);
    for (const options of [
      { maximumBindGroups: 0 },
      { maximumBindingsPerBindGroup: 4 },
      { maximumStorageBuffers: 3 },
      { maximumUniformBuffers: 0 },
      { maximumDynamicUniformBuffers: 0 },
    ]) {
      expect(() => AceFp16VaeSnakeKernel.create(fakeDevice(options)))
        .toThrow(/requires max/);
    }

    const shape = { batch: 1, frames: 1, channels: 257 };
    const plan = planAceFp16VaeSnake(shape);
    const bindings = bindingsFor(shape);
    const control = fakeBuffer(1_024);
    const cases: readonly [
      FakeDevice,
      AceFp16VaeSnakeBindings,
      AceVaeOutputRangeBinding,
      RegExp,
    ][] = [
      [
        fakeDevice({ maximumDispatch: 1 }),
        bindings,
        rangeBinding(control, 0, 0, 257),
        /dispatch dimension/,
      ],
      [
        fakeDevice({ maximumStorageBinding: plan.inputBindingBytes - 1 }),
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
        fakeDevice({ maximumBuffer: plan.inputBindingBytes - 1 }),
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
        { ...bindings, input: fakeBinding(plan.inputBindingBytes - 4) },
        rangeBinding(control, 0, 0, 257),
        /does not expose an aligned/,
      ],
      [
        fakeDevice(),
        {
          ...bindings,
          alpha: {
            buffer: fakeBuffer(plan.alphaBindingBytes + 256),
            offset: 4,
            size: plan.alphaBindingBytes,
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
      await expect(AceFp16VaeSnakeKernel.create(device).createDispatch(
        "invalid",
        shape,
        candidateBindings,
        range,
      )).rejects.toThrow(reason);
      expect(device.createShaderModule).not.toHaveBeenCalled();
    }
  });

  it("reports diagnostics and evicts failed compilation for retry", async () => {
    const device = fakeDevice({
      compilationMessageBatches: [[{
        message: "synthetic FP16 Snake diagnostic",
        type: "error",
        lineNum: 41,
        linePos: 7,
      }], []],
    });
    const kernel = AceFp16VaeSnakeKernel.create(device);
    const shape = { batch: 1, frames: 1, channels: 3 };
    const bindings = bindingsFor(shape);
    const range = fullRange(fakeBuffer(256), 0, shape);
    await expect(kernel.createDispatch(
      "diagnostic",
      shape,
      bindings,
      range,
    )).rejects.toThrow(/41:7 synthetic FP16 Snake diagnostic/);
    expect(device.createComputePipelineAsync).not.toHaveBeenCalled();
    await expect(kernel.createDispatch(
      "retry",
      shape,
      bindings,
      range,
    )).resolves.toMatchObject({
      kernelId: ACE_FP16_VAE_SNAKE_PORTABLE_KERNEL_ID,
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
    const kernel = AceFp16VaeSnakeKernel.create(device);
    const shape = { batch: 1, frames: 1, channels: 3 };
    const pending = kernel.createDispatch(
      "destroy-race",
      shape,
      bindingsFor(shape),
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

    const liveKernel = AceFp16VaeSnakeKernel.create(fakeDevice());
    const dispatch = await liveKernel.createDispatch(
      "live",
      shape,
      bindingsFor(shape),
      fullRange(fakeBuffer(256), 0, shape),
    );
    liveKernel.destroy();
    expect(() => dispatch.encode(fakePass())).toThrow(/was destroyed/);
    await expect(liveKernel.createDispatch(
      "dead",
      shape,
      bindingsFor(shape),
      fullRange(fakeBuffer(256), 0, shape),
    )).rejects.toThrow(/was destroyed/);
  });
});

expect(ACE_FP16_VAE_SNAKE_WORKGROUP_SIZE).toBe(256);

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const EXPECTED_B256_SNAKE_OPERATIONS = Object.freeze([
  [1, "block-0-snake1", 256, 2_048, 1, 524_288],
  [3, "block-0-res-1-snake1", 2_560, 1_024, 3, 524_288],
  [5, "block-0-res-1-snake2", 2_560, 1_024, 3, 524_288],
  [8, "block-0-res-2-snake1", 2_560, 1_024, 3, 524_288],
  [10, "block-0-res-2-snake2", 2_560, 1_024, 3, 524_288],
  [13, "block-0-res-3-snake1", 2_560, 1_024, 3, 524_288],
  [15, "block-0-res-3-snake2", 2_560, 1_024, 3, 524_288],
  [18, "block-1-snake1", 2_560, 1_024, 3, 524_288],
  [20, "block-1-res-1-snake1", 15_360, 512, 8, 524_288],
  [22, "block-1-res-1-snake2", 15_360, 512, 8, 524_288],
  [25, "block-1-res-2-snake1", 15_360, 512, 8, 524_288],
  [27, "block-1-res-2-snake2", 15_360, 512, 8, 524_288],
  [30, "block-1-res-3-snake1", 15_360, 512, 8, 524_288],
  [32, "block-1-res-3-snake2", 15_360, 512, 8, 524_288],
  [35, "block-2-snake1", 15_360, 512, 8, 524_288],
  [37, "block-2-res-1-snake1", 61_440, 256, 15, 1_048_576],
  [39, "block-2-res-1-snake2", 61_440, 256, 15, 1_048_576],
  [42, "block-2-res-2-snake1", 61_440, 256, 15, 1_048_576],
  [44, "block-2-res-2-snake2", 61_440, 256, 15, 1_048_576],
  [47, "block-2-res-3-snake1", 61_440, 256, 15, 1_048_576],
  [49, "block-2-res-3-snake2", 61_440, 256, 15, 1_048_576],
  [52, "block-3-snake1", 61_440, 256, 15, 1_048_576],
  [54, "block-3-res-1-snake1", 245_760, 128, 30, 1_048_576],
  [56, "block-3-res-1-snake2", 245_760, 128, 30, 1_048_576],
  [59, "block-3-res-2-snake1", 245_760, 128, 30, 1_048_576],
  [61, "block-3-res-2-snake2", 245_760, 128, 30, 1_048_576],
  [64, "block-3-res-3-snake1", 245_760, 128, 30, 1_048_576],
  [66, "block-3-res-3-snake2", 245_760, 128, 30, 1_048_576],
  [69, "block-4-snake1", 245_760, 128, 30, 1_048_576],
  [71, "block-4-res-1-snake1", 491_520, 128, 60, 1_048_576],
  [73, "block-4-res-1-snake2", 491_520, 128, 60, 1_048_576],
  [76, "block-4-res-2-snake1", 491_520, 128, 60, 1_048_576],
  [78, "block-4-res-2-snake2", 491_520, 128, 60, 1_048_576],
  [81, "block-4-res-3-snake1", 491_520, 128, 60, 1_048_576],
  [83, "block-4-res-3-snake2", 491_520, 128, 60, 1_048_576],
  [86, "snake1", 491_520, 128, 60, 1_048_576],
] as const);

function bindingsFor(shape: AceVaePointwiseShape): AceFp16VaeSnakeBindings {
  const plan = planAceFp16VaeSnake(shape);
  return {
    input: fakeBinding(plan.inputBindingBytes),
    alpha: fakeBinding(plan.alphaBindingBytes),
    beta: fakeBinding(plan.betaBindingBytes),
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
  readonly maximumBindGroups?: number;
  readonly maximumBindingsPerBindGroup?: number;
  readonly maximumStorageBuffers?: number;
  readonly maximumUniformBuffers?: number;
  readonly maximumDynamicUniformBuffers?: number;
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
      maxBindGroups: options.maximumBindGroups ?? 4,
      maxBindingsPerBindGroup: options.maximumBindingsPerBindGroup ?? 1_000,
      maxStorageBuffersPerShaderStage: options.maximumStorageBuffers ?? 10,
      maxUniformBuffersPerShaderStage: options.maximumUniformBuffers ?? 12,
      maxDynamicUniformBuffersPerPipelineLayout:
        options.maximumDynamicUniformBuffers ?? 10,
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

function mixedSnakeFloat16Bits(
  inputBits: number,
  alphaBits: number,
  betaBits: number,
): number {
  const value = Math.fround(float16BitsToNumber(inputBits));
  const alphaLogScale = Math.fround(float16BitsToNumber(alphaBits));
  const betaLogScale = Math.fround(float16BitsToNumber(betaBits));
  const alphaValue = Math.fround(Math.exp(alphaLogScale));
  const betaValue = Math.fround(Math.exp(betaLogScale));
  const periodicArgument = Math.fround(alphaValue * value);
  const periodic = Math.fround(Math.sin(periodicArgument));
  const denominator = Math.fround(betaValue + Math.fround(1e-9));
  const reciprocalBeta = Math.fround(1 / denominator);
  const firstProduct = Math.fround(reciprocalBeta * periodic);
  const periodicTerm = Math.fround(firstProduct * periodic);
  const result = Math.fround(value + periodicTerm);
  return numberToFloat16Bits(result);
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
