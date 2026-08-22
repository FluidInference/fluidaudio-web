import { describe, expect, it, vi } from "vitest";

import {
  ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_SUPPORTED_STRIDES,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_WEIGHT_TILE_STRIDE,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_X,
  ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_Y,
  AceFp16VaeConvTranspose1dKernel,
  aceFp16VaeCongruentConvTranspose1dWgsl,
  aceFp16VaeConvTranspose1dWgsl,
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dCongruentRange,
  planAceFp16VaeConvTranspose1dRange,
  type AceFp16VaeConvTranspose1dBindings,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import type {
  AceVaeConvTranspose1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderConvTransposeOperation,
} from "../src/webgpu/vae-decoder.js";

describe("production portable FP16 VAE ConvTranspose1D", () => {
  it("covers the five exact B-256 graph operations and package-sized boundaries", () => {
    const operations = transposeOperations();
    expect(operations).toHaveLength(5);
    expect(operations.map(({ shape }) => ({
      inputFrames: shape.inputFrames,
      inputChannels: shape.inputChannels,
      outputChannels: shape.outputChannels,
      kernelSize: shape.kernelSize,
      stride: shape.stride,
      padding: shape.padding,
    }))).toEqual([
      {
        inputFrames: 256,
        inputChannels: 2_048,
        outputChannels: 1_024,
        kernelSize: 20,
        stride: 10,
        padding: 5,
      },
      {
        inputFrames: 2_560,
        inputChannels: 1_024,
        outputChannels: 512,
        kernelSize: 12,
        stride: 6,
        padding: 3,
      },
      {
        inputFrames: 15_360,
        inputChannels: 512,
        outputChannels: 256,
        kernelSize: 8,
        stride: 4,
        padding: 2,
      },
      {
        inputFrames: 61_440,
        inputChannels: 256,
        outputChannels: 128,
        kernelSize: 8,
        stride: 4,
        padding: 2,
      },
      {
        inputFrames: 245_760,
        inputChannels: 128,
        outputChannels: 128,
        kernelSize: 4,
        stride: 2,
        padding: 1,
      },
    ]);

    for (const operation of operations) {
      const plan = planAceFp16VaeConvTranspose1d(operation.shape);
      expect(plan.outputFrames).toBe(
        operation.shape.inputFrames * operation.shape.stride,
      );
      expect(plan.workgroupStorageBytes).toBe(3_216);
      expect(plan.inputChannelChunkCount).toBe(
        operation.shape.inputChannels / 64,
      );
      expect(() => aceFp16VaeConvTranspose1dWgsl(operation.shape))
        .not.toThrow();
    }

    expect(planAceFp16VaeConvTranspose1d(operations[0]!.shape))
      .toMatchObject({
        inputStorageBytes: 1_048_576,
        weightStorageBytes: 83_886_080,
        biasStorageBytes: 2_048,
        outputStorageBytes: 5_242_880,
      });
    expect(planAceFp16VaeConvTranspose1d(operations[4]!.shape))
      .toMatchObject({
        outputFrames: 491_520,
        outputStorageBytes: 125_829_120,
        outputBindingBytes: 125_829_120,
      });
  });

  it("authenticates all 322 existing B-256 row quanta without repartition", () => {
    const graph = planAceVaeDecoder(256);
    const cooperative = planAceVaeDecoderQuanta(graph);
    const counts = new Map<string, number>();
    let transposeQuanta = 0;
    for (const quantum of cooperative.quanta) {
      const operation = graph.operations[quantum.operationIndex]!;
      if (operation.kind !== "conv-transpose1d") continue;
      transposeQuanta += 1;
      counts.set(operation.label, (counts.get(operation.label) ?? 0) + 1);
      expect(quantum.primitives).toHaveLength(1);
      const primitive = quantum.primitives[0]!;
      expect(primitive.physicalPartIndex).toBe(0);
      expect(primitive.firstOutputChannel).toBe(0);
      expect(primitive.outputChannels).toBe(operation.shape.outputChannels);
      expect(primitive.outputBase).toBe(quantum.logicalOutputBase);
      expect(primitive.outputCount).toBe(quantum.logicalOutputCount);

      const plan = planAceFp16VaeConvTranspose1d(operation.shape);
      const range = planAceFp16VaeConvTranspose1dRange(plan, {
        base: primitive.outputBase,
        count: primitive.outputCount,
      });
      expect(range.base).toBe(quantum.logicalOutputBase);
      expect(range.count).toBe(quantum.logicalOutputCount);
      expect(range.outputRowCount * operation.shape.outputChannels)
        .toBe(quantum.logicalOutputCount);
      expect(range.batch).toBe(0);
    }
    expect(transposeQuanta).toBe(322);
    expect([...counts.values()]).toEqual([46, 69, 69, 69, 69]);
  });

  it("pins odd FP16 payload padding and complete-row, single-batch ranges", () => {
    const plan = planAceFp16VaeConvTranspose1d(tailShape(65, 9, 6, 2));
    expect(plan).toMatchObject({
      batch: 2,
      outputFrames: 102,
      inputStorageBytes: 4_420,
      inputBindingBytes: 4_420,
      weightStorageBytes: 14_040,
      weightBindingBytes: 14_040,
      biasStorageBytes: 18,
      biasBindingBytes: 20,
      outputStorageBytes: 3_672,
      outputBindingBytes: 3_672,
      inputChannelChunkCount: 2,
      inputTileElements: 1_088,
      weightTileElements: 520,
      inputTileBytes: 2_176,
      weightTileBytes: 1_040,
      workgroupStorageBytes: 3_216,
    });
    expect(planAceFp16VaeConvTranspose1dRange(plan, {
      base: 102 * 9,
      count: 17 * 9,
    })).toEqual({
      base: 918,
      count: 153,
      batch: 1,
      firstOutputTime: 0,
      firstOutputRow: 102,
      outputRowCount: 17,
      workgroupsX: 2,
      workgroupsY: 2,
    });
    expect(() => planAceFp16VaeConvTranspose1dRange(plan, {
      base: 1,
      count: 9,
    })).toThrow(/complete in-bounds NLC rows/);
    expect(() => planAceFp16VaeConvTranspose1dRange(plan, {
      base: 101 * 9,
      count: 2 * 9,
    })).toThrow(/batch boundary/);
    expect(() => planAceFp16VaeConvTranspose1dRange(plan, {
      base: 0,
      count: 0,
    })).toThrow(/complete in-bounds NLC rows/);
  });

  it("emits inverse-stride guards before the exact K-then-Cin FP32 reduction", () => {
    const source = aceFp16VaeConvTranspose1dWgsl(
      tailShape(65, 9, 6),
    );
    expect(source).toContain(
      `// kernel-id: ${ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID}`,
    );
    expect(source).toContain("enable f16;");
    expect(source).toContain("const KERNEL_SIZE: u32 = 12u;");
    expect(source).toContain("const STRIDE: u32 = 6u;");
    expect(source).toContain("const PADDING: u32 = 3u;");
    expect(source).toContain("input: array<f16>;");
    expect(source).toContain("weight: array<f16>;");
    expect(source).toContain("bias: array<f16>;");
    expect(source).toContain("output: array<f16>;");
    expect(source).toContain("var<workgroup> input_tile: array<f16, 1088>;");
    expect(source).toContain("var<workgroup> weight_tile: array<f16, 520>;");
    expect(source).toContain("let bias_operand: f32 = f32(bias[output_channel]);");
    expect(source).toContain("let input_operand: f32 = f32(input_tile[");
    expect(source).toContain("let weight_operand: f32 = f32(");
    expect(source).toContain("sum = sum + input_operand * weight_operand;");
    expect(source).toContain(
      "(weight_output_channel * KERNEL_SIZE + kernel) * INPUT_CHANNELS +\n" +
        "            input_channel",
    );
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    expect(source.match(/output\[[^\]]+\][^=]*= f16\(sum\);/gs))
      .toHaveLength(1);

    const kernel = source.indexOf("var kernel = 0u;");
    const chunk = source.indexOf("var input_channel_chunk = 0u;", kernel);
    const skip = source.indexOf("Invalid inverse-stride", chunk);
    const paddingGuard = source.indexOf(
      "if (padded_output_time >= kernel_time)",
      skip,
    );
    const strideGuard = source.indexOf(
      "if ((input_numerator % STRIDE) == 0u)",
      paddingGuard,
    );
    const inputGuard = source.indexOf(
      "if (input_time < INPUT_FRAMES)",
      strideGuard,
    );
    const channel = source.indexOf("var chunk_channel = 0u;", inputGuard);
    const add = source.indexOf(
      "sum = sum + input_operand * weight_operand;",
      channel,
    );
    expect(kernel).toBeLessThan(chunk);
    expect(chunk).toBeLessThan(paddingGuard);
    expect(paddingGuard).toBeLessThan(strideGuard);
    expect(strideGuard).toBeLessThan(inputGuard);
    expect(inputGuard).toBeLessThan(channel);
    expect(channel).toBeLessThan(add);
    expectForbiddenMathAbsent(source);
  });

  it("guards output-time, output-channel, input-channel, and tile tails", () => {
    const source = aceFp16VaeConvTranspose1dWgsl(
      tailShape(65, 9, 2),
    );
    expect(source).toContain(
      "output_time < range_end_time && output_channel < OUTPUT_CHANNELS",
    );
    expect(source).toContain("input_channel < INPUT_CHANNELS");
    expect(source).toContain("weight_output_channel < OUTPUT_CHANNELS");
    expect(source).toContain(
      "INPUT_CHANNELS - chunk_first_channel",
    );
    expect(source).toContain(
      "chunk_channel *\n                    17u +\n                  local.x",
    );
    const plan = planAceFp16VaeConvTranspose1d(tailShape(65, 9, 2));
    expect(planAceFp16VaeConvTranspose1dRange(plan, {
      base: (plan.outputFrames - 7) * plan.outputChannels,
      count: 7 * plan.outputChannels,
    })).toMatchObject({
      firstOutputTime: plan.outputFrames - 7,
      outputRowCount: 7,
      workgroupsX: 1,
      workgroupsY: 2,
    });
  });

  it.each([
    [{ ...tailShape(64, 8, 4), stride: 3, kernelSize: 6, padding: 2 }, /stride 2,4,6,10/],
    [{ ...tailShape(64, 8, 4), kernelSize: 7 }, /kernel=2\*stride/],
    [{ ...tailShape(64, 8, 4), dilation: 2 }, /dilation1/],
    [{ ...tailShape(64, 8, 4), padding: 3 }, /padding=ceil/],
    [{ ...tailShape(64, 8, 4), outputPadding: 1 }, /outputPadding0/],
  ])("rejects unsupported production geometry", (shape, reason) => {
    expect(() => planAceFp16VaeConvTranspose1d(shape)).toThrow(reason);
  });

  it("fails closed when shape or range arithmetic leaves WGSL u32", () => {
    expect(() => planAceFp16VaeConvTranspose1d({
      ...tailShape(1, 1, 2),
      inputFrames: 0x8000_0000,
    })).toThrow(/u32 indexing domain/);
    const plan = planAceFp16VaeConvTranspose1d(tailShape(64, 8, 2));
    expect(() => planAceFp16VaeConvTranspose1dRange(plan, {
      base: Number.MAX_SAFE_INTEGER,
      count: 8,
    })).toThrow(/complete in-bounds NLC rows/);
  });

  it("compiles once, shares one bind group, and encodes exact immutable controls", async () => {
    const device = fakeDevice();
    const kernel = AceFp16VaeConvTranspose1dKernel.create(device);
    const shape = transposeOperations()[0]!.shape;
    const plan = planAceFp16VaeConvTranspose1d(shape);
    const bindings = bindingsFor(shape);
    const control = fakeBuffer(1_024);
    const first = await kernel.createDispatch(
      "first",
      shape,
      bindings,
      rangeBinding(control, 256, 0, 56 * shape.outputChannels),
    );
    const second = await kernel.createDispatch(
      "second",
      shape,
      bindings,
      rangeBinding(
        control,
        512,
        56 * shape.outputChannels,
        32 * shape.outputChannels,
      ),
    );
    expect(first.outputRange).toMatchObject({
      firstOutputTime: 0,
      outputRowCount: 56,
      workgroupsX: 4,
      workgroupsY: 128,
    });
    expect(second.outputRange).toMatchObject({
      firstOutputTime: 56,
      outputRowCount: 32,
      workgroupsX: 2,
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
      [4, 128, 1],
      [2, 128, 1],
    ]);

    kernel.destroy();
    kernel.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
    await expect(kernel.createDispatch(
      "after-destroy",
      shape,
      bindings,
      rangeBinding(control, 256, 0, 56 * shape.outputChannels),
    )).rejects.toThrow(/was destroyed/);
  });

  it("keys distinct transpose geometries as distinct pipelines", async () => {
    const device = fakeDevice();
    const kernel = AceFp16VaeConvTranspose1dKernel.create(device);
    const control = fakeBuffer(1_024);
    for (const [index, shape] of [
      tailShape(64, 8, 2),
      tailShape(64, 8, 4),
      tailShape(64, 8, 6),
      tailShape(64, 8, 10),
    ].entries()) {
      await kernel.createDispatch(
        `stride-${shape.stride}`,
        shape,
        bindingsFor(shape),
        fullRange(control, index * 256, shape),
      );
    }
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(4);
    const sources = device.createShaderModule.mock.calls.map(([descriptor]) =>
      (descriptor as GPUShaderModuleDescriptor).code
    );
    expect(sources.map((source) => source.match(/const STRIDE: u32 = (\d+)u/)?.[1]))
      .toEqual(["2", "4", "6", "10"]);
  });

  it("fails closed before compilation on features, limits, bindings, and aliases", async () => {
    expect(() => AceFp16VaeConvTranspose1dKernel.create(fakeDevice({
      shaderF16: false,
    }))).toThrow(/requires WebGPU shader-f16/);
    expect(() => AceFp16VaeConvTranspose1dKernel.create(fakeDevice({
      maximumInvocations: 127,
    }))).toThrow(/128-lane/);
    expect(() => AceFp16VaeConvTranspose1dKernel.create(fakeDevice({
      maximumWorkgroupSizeY: 7,
    }))).toThrow(/16x8/);

    const shape = tailShape(65, 9, 6);
    const plan = planAceFp16VaeConvTranspose1d(shape);
    const bindings = bindingsFor(shape);
    const storageDevice = fakeDevice({ maximumWorkgroupStorage: 3_215 });
    await expect(AceFp16VaeConvTranspose1dKernel.create(storageDevice)
      .createDispatch(
        "workgroup-storage",
        shape,
        bindings,
        fullRange(fakeBuffer(256), 0, shape),
      )).rejects.toThrow(/3216 workgroup-storage bytes/);
    expect(storageDevice.createShaderModule).not.toHaveBeenCalled();

    const dispatchDevice = fakeDevice({ maximumDispatch: 6 });
    await expect(AceFp16VaeConvTranspose1dKernel.create(dispatchDevice)
      .createDispatch(
        "dispatch",
        shape,
        bindings,
        fullRange(fakeBuffer(256), 0, shape),
      )).rejects.toThrow(/dispatch dimension/);
    expect(dispatchDevice.createShaderModule).not.toHaveBeenCalled();

    const uniformDevice = fakeDevice({ maximumUniformBinding: 15 });
    await expect(AceFp16VaeConvTranspose1dKernel.create(uniformDevice)
      .createDispatch(
        "uniform-limit",
        shape,
        bindings,
        fullRange(fakeBuffer(256), 0, shape),
      )).rejects.toThrow(/invalid buffer limits/);
    expect(uniformDevice.createShaderModule).not.toHaveBeenCalled();

    const storageBindingDevice = fakeDevice({
      maximumStorageBinding: plan.weightBindingBytes - 4,
    });
    await expect(AceFp16VaeConvTranspose1dKernel.create(storageBindingDevice)
      .createDispatch(
        "storage-binding-limit",
        shape,
        bindings,
        fullRange(fakeBuffer(256), 0, shape),
      )).rejects.toThrow(/weight exceeds the device storage binding limit/);
    expect(storageBindingDevice.createShaderModule).not.toHaveBeenCalled();

    const shortDevice = fakeDevice();
    await expect(AceFp16VaeConvTranspose1dKernel.create(shortDevice)
      .createDispatch(
        "short-input",
        shape,
        {
          ...bindings,
          input: fakeBinding(plan.inputBindingBytes - 4),
        },
        fullRange(fakeBuffer(256), 0, shape),
      )).rejects.toThrow(/input binding does not expose/);
    expect(shortDevice.createShaderModule).not.toHaveBeenCalled();

    const alignmentDevice = fakeDevice({ uniformAlignment: 512 });
    await expect(AceFp16VaeConvTranspose1dKernel.create(alignmentDevice)
      .createDispatch(
        "unaligned-control",
        shape,
        bindings,
        fullRange(fakeBuffer(1_024), 256, shape),
      )).rejects.toThrow(/aligned 16-byte immutable record/);
    expect(alignmentDevice.createShaderModule).not.toHaveBeenCalled();

    const aliasDevice = fakeDevice();
    const shared = fakeBuffer(
      plan.inputBindingBytes + plan.outputBindingBytes,
    );
    await expect(AceFp16VaeConvTranspose1dKernel.create(aliasDevice)
      .createDispatch(
        "alias",
        shape,
        {
          ...bindings,
          input: {
            buffer: shared,
            offset: 0,
            size: plan.inputBindingBytes,
          },
          output: {
            buffer: shared,
            offset: 256,
            size: plan.outputBindingBytes,
          },
        },
        fullRange(fakeBuffer(256), 0, shape),
      )).rejects.toThrow(/input and output bindings must not overlap/);
    expect(aliasDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("reports diagnostics, evicts failures, and rejects an in-flight destroy race", async () => {
    const diagnosticDevice = fakeDevice({
      compilationMessageBatches: [[{
        message: "synthetic transpose diagnostic",
        type: "error",
        lineNum: 57,
        linePos: 11,
      }], []],
    });
    const diagnosticKernel =
      AceFp16VaeConvTranspose1dKernel.create(diagnosticDevice);
    const shape = tailShape(65, 9, 6);
    const bindings = bindingsFor(shape);
    const range = fullRange(fakeBuffer(256), 0, shape);
    await expect(diagnosticKernel.createDispatch(
      "diagnostic",
      shape,
      bindings,
      range,
    )).rejects.toThrow(/57:11 synthetic transpose diagnostic/);
    expect(diagnosticDevice.createComputePipelineAsync).not.toHaveBeenCalled();
    await expect(diagnosticKernel.createDispatch(
      "retry",
      shape,
      bindings,
      range,
    )).resolves.toMatchObject({
      kernelId: ACE_FP16_VAE_CONV_TRANSPOSE1D_PORTABLE_KERNEL_ID,
    });
    expect(diagnosticDevice.createShaderModule).toHaveBeenCalledTimes(2);
    expect(diagnosticDevice.createComputePipelineAsync).toHaveBeenCalledOnce();

    const gate = deferred<void>();
    const raceDevice = fakeDevice({ pipelineGate: gate.promise });
    const raceKernel = AceFp16VaeConvTranspose1dKernel.create(raceDevice);
    const pending = raceKernel.createDispatch(
      "race",
      shape,
      bindingsFor(shape),
      fullRange(fakeBuffer(256), 0, shape),
    );
    for (let index = 0; index < 8; index += 1) await Promise.resolve();
    expect(raceDevice.createComputePipelineAsync).toHaveBeenCalledOnce();
    raceKernel.destroy();
    gate.resolve();
    await expect(pending).rejects.toThrow(/was destroyed/);
    expect(raceDevice.createBindGroup).not.toHaveBeenCalled();
  });
});

describe("exact congruent two-tap FP16 VAE ConvTranspose1D", () => {
  it("enumerates exactly the reference valid kernels for all five operations", () => {
    const expectedTapPairs = [
      5_110,
      30_714,
      122_876,
      491_516,
      983_038,
    ];
    for (const [operationIndex, operation] of transposeOperations().entries()) {
      const shape = operation.shape;
      let tapPairs = 0;
      for (
        let outputTime = 0;
        outputTime < shape.inputFrames * shape.stride;
        outputTime += 1
      ) {
        const reference: number[] = [];
        const paddedOutputTime = outputTime + shape.padding;
        for (let kernel = 0; kernel < shape.kernelSize; kernel += 1) {
          if (paddedOutputTime < kernel) continue;
          const numerator = paddedOutputTime - kernel;
          if (numerator % shape.stride !== 0) continue;
          if (numerator / shape.stride >= shape.inputFrames) continue;
          reference.push(kernel);
        }
        const congruentKernel = paddedOutputTime % shape.stride;
        const congruent = [
          congruentKernel,
          congruentKernel + shape.stride,
        ].filter((kernel) => {
          if (paddedOutputTime < kernel) return false;
          return (paddedOutputTime - kernel) / shape.stride <
            shape.inputFrames;
        });
        expect(congruent).toEqual(reference);
        tapPairs += congruent.length;
      }
      expect(tapPairs).toBe(expectedTapPairs[operationIndex]);
    }
  });

  it("emits phase-row tiling and only two source-ordered tap/chunk loops", () => {
    const source = aceFp16VaeCongruentConvTranspose1dWgsl(
      tailShape(65, 9, 6),
    );
    expect(source).toContain(
      `// kernel-id: ${ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID}`,
    );
    expect(source).toContain("let phase = group.z;");
    expect(source).toContain(
      "let congruent_kernel = phase_first_padded_time % STRIDE;",
    );
    expect(source).toContain("for (var tap = 0u; tap < 2u; tap += 1u)");
    expect(source).toContain(
      "let kernel = congruent_kernel + tap * STRIDE;",
    );
    expect(source).not.toContain("var kernel = 0u;");
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    expect(source).toContain(
      "tile_range_offset < output_row_count &&\n" +
        "          input_channel < INPUT_CHANNELS",
    );
    expect(source).toContain(
      "(weight_output_channel * KERNEL_SIZE + kernel) * INPUT_CHANNELS +\n" +
        "            input_channel",
    );
    const tap = source.indexOf("var tap = 0u;");
    const chunk = source.indexOf("var input_channel_chunk = 0u;", tap);
    const channel = source.indexOf("var chunk_channel = 0u;", chunk);
    const add = source.indexOf(
      "sum = sum + input_operand * weight_operand;",
      channel,
    );
    expect(tap).toBeLessThan(chunk);
    expect(chunk).toBeLessThan(channel);
    expect(channel).toBeLessThan(add);
    expectForbiddenMathAbsent(source);
  });

  it("preserves every B-256 quantum while removing most staged tap loops", () => {
    const graph = planAceVaeDecoder(256);
    const cooperative = planAceVaeDecoderQuanta(graph);
    let portableWorkgroups = 0;
    let congruentWorkgroups = 0;
    let portableBarrierInstances = 0;
    let congruentBarrierInstances = 0;
    let portableStagedFp16Writes = 0;
    let congruentStagedFp16Writes = 0;
    let transposeQuanta = 0;
    for (const quantum of cooperative.quanta) {
      const operation = graph.operations[quantum.operationIndex]!;
      if (operation.kind !== "conv-transpose1d") continue;
      transposeQuanta += 1;
      const plan = planAceFp16VaeConvTranspose1d(operation.shape);
      const range = {
        base: quantum.logicalOutputBase,
        count: quantum.logicalOutputCount,
      };
      const portable = planAceFp16VaeConvTranspose1dRange(plan, range);
      const congruent = planAceFp16VaeConvTranspose1dCongruentRange(
        plan,
        range,
      );
      const portableCount = portable.workgroupsX * portable.workgroupsY;
      const congruentCount = congruent.workgroupsX *
        congruent.workgroupsY * congruent.workgroupsZ;
      portableWorkgroups += portableCount;
      congruentWorkgroups += congruentCount;
      portableBarrierInstances += portableCount * plan.kernelSize *
        plan.inputChannelChunkCount * 2;
      congruentBarrierInstances += congruentCount * 2 *
        plan.inputChannelChunkCount * 2;
      portableStagedFp16Writes += portableCount * plan.kernelSize *
        plan.inputChannels * (plan.tileFrames + plan.tileChannels);
      congruentStagedFp16Writes += congruentCount * 2 *
        plan.inputChannels * (plan.tileFrames + plan.tileChannels);
    }
    expect(transposeQuanta).toBe(322);
    expect(portableWorkgroups).toBe(945_024);
    expect(congruentWorkgroups).toBe(998_144);
    expect(portableBarrierInstances).toBe(92_897_280);
    expect(congruentBarrierInstances).toBe(24_395_776);
    expect(portableStagedFp16Writes).toBe(71_345_111_040);
    expect(congruentStagedFp16Writes).toBe(18_735_955_968);
  });

  it("compiles once and dispatches stride phases through workgroup Z", async () => {
    const device = fakeDevice();
    const kernel = AceFp16VaeConvTranspose1dKernel.createCongruent(device);
    const shape = transposeOperations()[0]!.shape;
    const control = fakeBuffer(1_024);
    const dispatch = await kernel.createDispatch(
      "congruent",
      shape,
      bindingsFor(shape),
      rangeBinding(control, 256, 0, 56 * shape.outputChannels),
    );
    expect(dispatch.kernelId).toBe(
      ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID,
    );
    expect(dispatch.outputRange).toMatchObject({
      firstOutputTime: 0,
      outputRowCount: 56,
      workgroupsX: 1,
      workgroupsY: 128,
      workgroupsZ: 10,
    });
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    const descriptor = device.createShaderModule.mock.calls[0]?.[0] as
      GPUShaderModuleDescriptor;
    expect(descriptor.code).toContain(
      `// kernel-id: ${ACE_FP16_VAE_CONV_TRANSPOSE1D_CONGRUENT_KERNEL_ID}`,
    );
    const pass = fakePass();
    dispatch.encode(pass);
    expect(pass.dispatchWorkgroups).toHaveBeenCalledWith(1, 128, 10);
  });

  it("fails closed when the stride-phase dispatch exceeds device limits", async () => {
    const device = fakeDevice({ maximumDispatch: 9 });
    const shape = tailShape(64, 8, 10);
    await expect(AceFp16VaeConvTranspose1dKernel.createCongruent(device)
      .createDispatch(
        "congruent-z-limit",
        shape,
        bindingsFor(shape),
        fullRange(fakeBuffer(256), 0, shape),
      )).rejects.toThrow(/dispatch dimension/);
    expect(device.createShaderModule).not.toHaveBeenCalled();
  });
});

expect(ACE_FP16_VAE_CONV_TRANSPOSE1D_SUPPORTED_STRIDES)
  .toEqual([2, 4, 6, 10]);
expect(ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_CHANNEL_CHUNK).toBe(64);
expect(ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_FRAMES).toBe(16);
expect(ACE_FP16_VAE_CONV_TRANSPOSE1D_TILE_CHANNELS).toBe(8);
expect(ACE_FP16_VAE_CONV_TRANSPOSE1D_INPUT_TILE_STRIDE).toBe(17);
expect(ACE_FP16_VAE_CONV_TRANSPOSE1D_WEIGHT_TILE_STRIDE).toBe(65);
expect(ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_X).toBe(16);
expect(ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE_Y).toBe(8);
expect(ACE_FP16_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE).toBe(128);

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

function transposeOperations(): readonly AceVaeDecoderConvTransposeOperation[] {
  return planAceVaeDecoder(256).operations.filter(
    (operation): operation is AceVaeDecoderConvTransposeOperation =>
      operation.kind === "conv-transpose1d",
  );
}

function tailShape(
  inputChannels: number,
  outputChannels: number,
  stride: 2 | 4 | 6 | 10,
  batch = 1,
): AceVaeConvTranspose1dShape {
  return {
    batch,
    inputFrames: 17,
    inputChannels,
    outputChannels,
    kernelSize: 2 * stride,
    stride,
    dilation: 1,
    padding: Math.ceil(stride / 2),
    outputPadding: 0,
  };
}

function bindingsFor(
  shape: AceVaeConvTranspose1dShape,
): AceFp16VaeConvTranspose1dBindings {
  const plan = planAceFp16VaeConvTranspose1d(shape);
  return {
    input: fakeBinding(plan.inputBindingBytes),
    weight: fakeBinding(plan.weightBindingBytes),
    bias: fakeBinding(plan.biasBindingBytes),
    output: fakeBinding(plan.outputBindingBytes),
  };
}

function fullRange(
  buffer: GPUBuffer,
  offset: number,
  shape: AceVaeConvTranspose1dShape,
): AceVaeOutputRangeBinding {
  const plan = planAceFp16VaeConvTranspose1d(shape);
  return rangeBinding(buffer, offset, 0, plan.outputElements);
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
  readonly maximumUniformBinding?: number;
  readonly maximumBuffer?: number;
  readonly uniformAlignment?: number;
  readonly storageAlignment?: number;
  readonly compilationMessageBatches?: readonly (
    readonly Partial<GPUCompilationMessage>[]
  )[];
  readonly pipelineGate?: Promise<void>;
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
      maxUniformBufferBindingSize:
        options.maximumUniformBinding ?? 65_536,
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
    createComputePipelineAsync: vi.fn(async () => {
      await options.pipelineGate;
      return fakePipeline();
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

function deferred<T>(): {
  readonly promise: Promise<T>;
  resolve(value: T): void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((accept) => {
    resolve = accept;
  });
  return { promise, resolve };
}
