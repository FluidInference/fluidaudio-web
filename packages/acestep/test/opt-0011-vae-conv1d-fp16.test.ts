import { describe, expect, it, vi } from "vitest";
import type { AceVaeConv1dShape } from
  "../src/webgpu/kernels/vae-primitives.js";

import {
  ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK,
  ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_TILE_STRIDE,
  ACE_OPT_0011_VAE_CONV1D_FP16_KERNEL_SIZE,
  ACE_OPT_0011_VAE_CONV1D_FP16_PORTABLE_WORKGROUP_ID,
  ACE_OPT_0011_VAE_CONV1D_FP16_SCALAR_ORACLE_ID,
  ACE_OPT_0011_VAE_CONV1D_FP16_SUPPORTED_DILATIONS,
  ACE_OPT_0011_VAE_CONV1D_FP16_TILE_CHANNELS,
  ACE_OPT_0011_VAE_CONV1D_FP16_TILE_FRAMES,
  ACE_OPT_0011_VAE_CONV1D_FP16_WEIGHT_TILE_STRIDE,
  ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE,
  ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_X,
  ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_Y,
  AceOpt0011VaeConv1dFp16PortableWorkgroupKernel,
  AceOpt0011VaeConv1dFp16ScalarOracleKernel,
  aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl,
  aceOpt0011VaeConv1dFp16ScalarOracleWgsl,
  planAceOpt0011VaeConv1dFp16,
  type AceOpt0011VaeConv1dFp16Bindings,
  type AceOpt0011VaeConv1dFp16OutputStorage,
} from "../benchmark/opt-0011-vae-conv1d-fp16.js";

describe("OPT-0011 FP16-storage VAE K7 Conv1D kernel slice", () => {
  it("plans the production C1024 operation in exact 32-row ranges", () => {
    const plan = planAceOpt0011VaeConv1dFp16(
      principalShape(),
      "float16",
    );
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
      outputStorage: "float16",
      inputElements: 2_621_440,
      weightElements: 7_340_032,
      outputElements: 2_621_440,
      inputStorageBytes: 5_242_880,
      inputBindingBytes: 5_242_880,
      weightStorageBytes: 14_680_064,
      weightBindingBytes: 14_680_064,
      biasStorageBytes: 2_048,
      biasBindingBytes: 2_048,
      outputStorageBytes: 5_242_880,
      outputBindingBytes: 5_242_880,
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
      inputTileBytes: 2_176,
      weightTileBytes: 1_040,
      workgroupStorageBytes: 3_216,
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
  ] as const)(
    "accepts only same-length K7 dilation %i with padding %i",
    (dilation, padding) => {
      const shape = { ...tailShape(65), dilation, padding };
      const plan = planAceOpt0011VaeConv1dFp16(shape, "float16");
      expect(plan.outputFrames).toBe(shape.inputFrames);
      expect(plan.inputChannelChunkCount).toBe(2);
      expect(plan.workgroupStorageBytes).toBe(3_216);
      for (const source of [
        aceOpt0011VaeConv1dFp16ScalarOracleWgsl(
          shape,
          true,
          "float16",
        ),
        aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl(
          shape,
          true,
          "float16",
        ),
      ]) {
        expect(source).toContain(`const DILATION: u32 = ${dilation}u;`);
        expect(source).toContain(`const PADDING: u32 = ${padding}u;`);
        expect(source).toContain("var kernel = 0u;");
        expect(source).toContain("var input_channel_chunk = 0u;");
      }
    },
  );

  it("covers IC63, IC64, and IC65 with one fixed FP16 tile geometry", () => {
    for (const [inputChannels, chunks] of [
      [63, 1],
      [64, 1],
      [65, 2],
    ] as const) {
      const plan = planAceOpt0011VaeConv1dFp16(
        tailShape(inputChannels),
        "float16",
      );
      expect(plan.inputChannelChunkCount).toBe(chunks);
      expect(plan.inputTileElements).toBe(64 * 17);
      expect(plan.weightTileElements).toBe(8 * 65);
      expect(plan.workgroupStorageBytes).toBe(3_216);
    }
    const source = aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl(
      tailShape(65),
      false,
      "float16",
    );
    expect(source).toContain("const INPUT_CHANNEL_CHUNKS: u32 = 2u;");
    expect(source).toContain("input_channel < INPUT_CHANNELS");
    expect(source).toContain("INPUT_CHANNELS - chunk_first_channel");
  });

  it("keeps frame, Cout, range, and batch tails on complete rows", () => {
    const plan = planAceOpt0011VaeConv1dFp16({
      ...tailShape(65),
      batch: 2,
      inputFrames: 33,
      outputChannels: 10,
      dilation: 9,
      padding: 27,
    }, "float16");
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
      expect(range.workgroupsX).toBe(Math.ceil(range.outputRowCount / 16));
      expect(range.workgroupsY).toBe(2);
    }
  });

  it("pins odd FP16 payload bytes separately from WebGPU binding padding", () => {
    const f16 = planAceOpt0011VaeConv1dFp16(tailShape(63), "float16");
    expect(f16).toMatchObject({
      inputStorageBytes: 2_142,
      inputBindingBytes: 2_144,
      weightStorageBytes: 7_938,
      weightBindingBytes: 7_940,
      biasStorageBytes: 18,
      biasBindingBytes: 20,
      outputStorageBytes: 306,
      outputBindingBytes: 308,
    });
    const f32 = planAceOpt0011VaeConv1dFp16(tailShape(63), "float32");
    expect(f32.outputStorageBytes).toBe(612);
    expect(f32.outputBindingBytes).toBe(612);
    expect(f32.inputStorageBytes).toBe(f16.inputStorageBytes);
    expect(f32.weightStorageBytes).toBe(f16.weightStorageBytes);
  });

  it("emits FP16 portable tiles with explicit FP32 operands and source order", () => {
    const source = aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl(
      tailShape(65),
      true,
      "float16",
    );
    expect(source).toContain(
      `// kernel-id: ${ACE_OPT_0011_VAE_CONV1D_FP16_PORTABLE_WORKGROUP_ID}`,
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
    expect(source).toContain("let input_operand = f32(input_tile[");
    expect(source).toContain("let weight_operand = f32(");
    expect(source).toContain("sum = sum + input_operand * weight_operand;");
    expect(source).toContain(
      "(weight_output_channel * 7u + kernel) * INPUT_CHANNELS",
    );
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);

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
    expect(source.match(/output\[output_row \* OUTPUT_CHANNELS/g))
      .toHaveLength(1);
    expect(source.match(/=\s*f16\(sum\);/g)).toHaveLength(1);
    expectForbiddenMathAbsent(source);
  });

  it("emits an independent scalar authority with the identical contract", () => {
    const scalar = aceOpt0011VaeConv1dFp16ScalarOracleWgsl(
      tailShape(65),
      true,
      "float16",
    );
    const portable = aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl(
      tailShape(65),
      true,
      "float16",
    );
    expect(ACE_OPT_0011_VAE_CONV1D_FP16_SCALAR_ORACLE_ID)
      .not.toBe(ACE_OPT_0011_VAE_CONV1D_FP16_PORTABLE_WORKGROUP_ID);
    expect(scalar).not.toBe(portable);
    expect(scalar).toContain(
      `// kernel-id: ${ACE_OPT_0011_VAE_CONV1D_FP16_SCALAR_ORACLE_ID}`,
    );
    expect(scalar).not.toContain("var<workgroup>");
    expect(scalar).not.toContain("workgroupBarrier");
    expect(scalar).toContain("let input_operand = f32(input[");
    expect(scalar).toContain("let weight_operand = f32(weight[");
    expect(scalar).toContain(
      "(output_channel * 7u + kernel) * INPUT_CHANNELS",
    );
    expect(scalar).toContain("sum = sum + input_operand * weight_operand;");
    const kernel = scalar.indexOf("var kernel = 0u;");
    const paddingGuard = scalar.indexOf("if (padded_time >= PADDING)", kernel);
    const chunk = scalar.indexOf("var input_channel_chunk = 0u;", paddingGuard);
    const channel = scalar.indexOf("var chunk_channel = 0u;", chunk);
    expect(kernel).toBeLessThan(paddingGuard);
    expect(paddingGuard).toBeLessThan(chunk);
    expect(chunk).toBeLessThan(channel);
    expect(scalar.match(/=\s*f16\(sum\);/g)).toHaveLength(1);
    expectForbiddenMathAbsent(scalar);
  });

  it("keeps the final no-bias raw waveform output in FP32", () => {
    for (const shader of [
      aceOpt0011VaeConv1dFp16ScalarOracleWgsl,
      aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl,
    ]) {
      const source = shader(tailShape(63), false, "float32");
      expect(source).toContain("input: array<f16>;");
      expect(source).toContain("weight: array<f16>;");
      expect(source).toContain("output: array<f32>;");
      expect(source).toContain("if (output_active) { sum = 0.0; }");
      expect(source).toContain(
        "output[output_row * OUTPUT_CHANNELS + output_channel] =\n" +
          "      select(sum, bitcast<f32>(0u), " +
          "(bitcast<u32>(sum) & 0x7fffffffu) == 0u);",
      );
      expect(source).not.toContain("bias: array<f16>");
      expect(source).not.toContain("f16(sum)");
      expectForbiddenMathAbsent(source);
    }
  });

  it("rejects biased FP32 output before shader-module creation", async () => {
    const shape = tailShape(63);
    for (const shader of [
      aceOpt0011VaeConv1dFp16ScalarOracleWgsl,
      aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl,
    ]) {
      expect(() => shader(shape, true, "float32"))
        .toThrow(/final no-bias raw-waveform boundary/);
    }
    for (const create of [
      AceOpt0011VaeConv1dFp16ScalarOracleKernel.create,
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create,
    ]) {
      const device = fakeDevice();
      const kernel = create(device);
      await expect(kernel.createDispatch(
        "biased-f32",
        shape,
        bindingsFor(shape, "float32", true),
        "float32",
      )).rejects.toThrow(/final no-bias raw-waveform boundary/);
      expect(device.createShaderModule).not.toHaveBeenCalled();
      expect(device.createComputePipelineAsync).not.toHaveBeenCalled();
      expect(device.createdBuffers).toHaveLength(0);
    }
  });

  it.each([
    [{ ...tailShape(65), kernelSize: 5 }, /requires K7/],
    [{ ...tailShape(65), stride: 2 }, /stride one/],
    [{ ...tailShape(65), dilation: 2, padding: 6 }, /dilation 1\/3\/9/],
    [{ ...tailShape(65), padding: 4 }, /padding dilation\*3/],
    [{ ...tailShape(65), batch: 0 }, /positive safe integer/],
    [{
      ...tailShape(65),
      inputFrames: 7,
      outputChannels: 8 * 65_535 + 1,
    }, /dispatch domain/],
    [{
      ...tailShape(1),
      inputFrames: 65_536,
      outputChannels: 65_536,
    }, /u32 indexing domain/],
    [{
      ...tailShape(4_000_000),
      inputFrames: 7,
    }, /complete-channel output row/],
    [{
      ...tailShape(1),
      batch: 65_536,
      inputFrames: 1,
      outputChannels: 1,
    }, /output range count/],
  ])("rejects unsupported or unbounded geometry", (shape, reason) => {
    expect(() => planAceOpt0011VaeConv1dFp16(shape, "float16"))
      .toThrow(reason);
  });

  it("rejects an undeclared output storage contract", () => {
    expect(() => planAceOpt0011VaeConv1dFp16(
      tailShape(1),
      "float64" as AceOpt0011VaeConv1dFp16OutputStorage,
    )).toThrow(/unknown output storage float64/);
  });

  it("requires shader-f16 and the declared 16x8 workgroup", () => {
    for (const create of [
      AceOpt0011VaeConv1dFp16ScalarOracleKernel.create,
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create,
    ]) {
      expect(() => create(fakeDevice({ shaderF16: false })))
        .toThrow(/requires WebGPU shader-f16/);
      expect(() => create(fakeDevice({ maximumWorkgroupSizeX: 8 })))
        .toThrow(/16x8/);
      expect(() => create(fakeDevice({ maximumWorkgroupSizeY: 4 })))
        .toThrow(/16x8/);
      expect(() => create(fakeDevice({ maximumInvocations: 64 })))
        .toThrow(/128-lane/);
    }
  });

  it("fails closed on workgroup-storage and dispatch limits", async () => {
    const storageDevice = fakeDevice({ maximumWorkgroupStorage: 3_215 });
    const storage =
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(storageDevice);
    const principal = principalShape();
    await expect(storage.createDispatch(
      "storage",
      principal,
      bindingsFor(principal, "float16"),
      "float16",
    )).rejects.toThrow(/3216 workgroup-storage bytes/);
    expect(storageDevice.createShaderModule).not.toHaveBeenCalled();

    const scalarDevice = fakeDevice({ maximumWorkgroupStorage: 0 });
    const scalar =
      AceOpt0011VaeConv1dFp16ScalarOracleKernel.create(scalarDevice);
    await expect(scalar.createDispatch(
      "scalar-no-shared",
      tailShape(63),
      bindingsFor(tailShape(63), "float16"),
      "float16",
    )).resolves.toMatchObject({
      kernelId: ACE_OPT_0011_VAE_CONV1D_FP16_SCALAR_ORACLE_ID,
    });

    const dispatchDevice = fakeDevice({ maximumDispatch: 1 });
    const dispatch =
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(dispatchDevice);
    await expect(dispatch.createDispatch(
      "dispatch",
      principal,
      bindingsFor(principal, "float16"),
      "float16",
    )).rejects.toThrow(/dispatch dimension/);
    expect(dispatchDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("fails closed on bind sizes, storage alignment, and aliases", async () => {
    const shape = tailShape(63);
    const plan = planAceOpt0011VaeConv1dFp16(shape, "float16");
    const device = fakeDevice();
    const kernel =
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(device);
    const bindings = bindingsFor(shape, "float16", true);

    await expect(kernel.createDispatch("short", shape, {
      ...bindings,
      weight: fakeBinding(plan.weightBindingBytes - 4),
    }, "float16")).rejects.toThrow(/weight binding does not expose/);
    await expect(kernel.createDispatch("offset", shape, {
      ...bindings,
      input: {
        buffer: fakeBuffer(plan.inputBindingBytes + 256),
        offset: 2,
        size: plan.inputBindingBytes,
      },
    }, "float16")).rejects.toThrow(/aligned .* storage payload/);
    await expect(kernel.createDispatch("odd-size", shape, {
      ...bindings,
      input: {
        buffer: fakeBuffer(plan.inputBindingBytes + 4),
        offset: 0,
        size: plan.inputBindingBytes + 1,
      },
    }, "float16")).rejects.toThrow(/aligned .* storage payload/);

    const shared = fakeBuffer(
      Math.max(plan.inputBindingBytes, plan.outputBindingBytes) + 256,
    );
    await expect(kernel.createDispatch("alias", shape, {
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
    }, "float16")).rejects.toThrow(/bindings must not overlap/);
    expect(device.createShaderModule).not.toHaveBeenCalled();

    const invalidAlignment = fakeDevice({ storageAlignment: 12 });
    const invalidKernel =
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(invalidAlignment);
    await expect(invalidKernel.createDispatch(
      "alignment",
      shape,
      bindingsFor(shape, "float16"),
      "float16",
    )).rejects.toThrow(/invalid storage alignment/);
  });

  it("fails closed on binding, buffer, control, and dynamic-offset limits", async () => {
    const shape = tailShape(63);
    const plan = planAceOpt0011VaeConv1dFp16(shape, "float16");
    const bindingDevice = fakeDevice({
      maximumStorageBinding: plan.weightBindingBytes - 4,
    });
    const bindingKernel =
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(bindingDevice);
    await expect(bindingKernel.createDispatch(
      "binding-limit",
      shape,
      bindingsFor(shape, "float16"),
      "float16",
    )).rejects.toThrow(/weight exceeds the device storage binding limit/);
    expect(bindingDevice.createShaderModule).not.toHaveBeenCalled();

    const oversizedDevice = fakeDevice({ maximumBuffer: 8_000 });
    const oversizedKernel =
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(oversizedDevice);
    const oversized = bindingsFor(shape, "float16");
    await expect(oversizedKernel.createDispatch(
      "oversized-buffer",
      shape,
      {
        ...oversized,
        input: {
          buffer: fakeBuffer(8_004),
          offset: 0,
          size: plan.inputBindingBytes,
        },
      },
      "float16",
    )).rejects.toThrow(/buffer exceeds the device buffer limit/);

    const controlDevice = fakeDevice({
      uniformAlignment: 8_192,
      maximumBuffer: 8_000,
    });
    const controlKernel =
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(controlDevice);
    await expect(controlKernel.createDispatch(
      "control-limit",
      shape,
      bindingsFor(shape, "float16"),
      "float16",
    )).rejects.toThrow(/range controls exceed/);
    expect(controlDevice.createShaderModule).not.toHaveBeenCalled();

    const dynamicShape = { ...principalShape(), inputFrames: 65 };
    const dynamicDevice = fakeDevice({
      uniformAlignment: 2_147_483_648,
      maximumBuffer: 7_000_000_000,
    });
    const dynamicKernel =
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(dynamicDevice);
    await expect(dynamicKernel.createDispatch(
      "dynamic-limit",
      dynamicShape,
      bindingsFor(dynamicShape, "float16"),
      "float16",
    )).rejects.toThrow(/dynamic-offset limit/);
    expect(dynamicDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("compiles once, binds exact bytes, and encodes production ranges", async () => {
    const device = fakeDevice();
    const kernel =
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(device);
    const shape = { ...principalShape(), inputFrames: 65 };
    const bindings = bindingsFor(shape, "float16", true);
    const dispatch = await kernel.createDispatch(
      "candidate",
      shape,
      bindings,
      "float16",
    );
    const reused = await kernel.createDispatch(
      "reuse",
      shape,
      bindings,
      "float16",
    );
    expect(dispatch).toMatchObject({
      kernelId: ACE_OPT_0011_VAE_CONV1D_FP16_PORTABLE_WORKGROUP_ID,
      outputStorage: "float16",
      rangeCount: 3,
    });
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();

    const controls = device.createdBuffers[0]!;
    expect(controls.size).toBe(3 * 256);
    expect(Array.from(new Uint32Array(controls.mapped, 0, 2))).toEqual([0, 32]);
    expect(Array.from(new Uint32Array(controls.mapped, 256, 2)))
      .toEqual([32, 32]);
    expect(Array.from(new Uint32Array(controls.mapped, 512, 2)))
      .toEqual([64, 1]);

    const layout = device.createBindGroupLayout.mock.calls[0]?.[0] as
      GPUBindGroupLayoutDescriptor;
    const entries = Array.from(layout.entries);
    expect(entries.map(({ buffer }) => buffer?.minBindingSize)).toEqual([
      dispatch.plan.inputBindingBytes,
      dispatch.plan.weightBindingBytes,
      dispatch.plan.biasBindingBytes,
      dispatch.plan.outputBindingBytes,
      16,
    ]);
    expect(entries.at(-1)?.buffer).toEqual({
      type: "uniform",
      hasDynamicOffset: true,
      minBindingSize: 16,
    });
    const group = device.createBindGroup.mock.calls[0]?.[0] as
      GPUBindGroupDescriptor;
    expect(Array.from(group.entries).map(({ resource }) =>
      (resource as GPUBufferBinding).size
    )).toEqual([
      dispatch.plan.inputBindingBytes,
      dispatch.plan.weightBindingBytes,
      dispatch.plan.biasBindingBytes,
      dispatch.plan.outputBindingBytes,
      16,
    ]);

    const pass = fakePass();
    dispatch.encodeRange(pass, 2);
    expect(pass.setBindGroup).toHaveBeenLastCalledWith(
      0,
      expect.anything(),
      [512],
    );
    expect(pass.dispatchWorkgroups).toHaveBeenLastCalledWith(1, 128, 1);
    expect(() => dispatch.encodeRange(pass, -1)).toThrow(/non-negative/);
    expect(() => dispatch.encodeRange(pass, 1.5)).toThrow(/safe integer/);
    expect(() => dispatch.encodeRange(pass, 3)).toThrow(/outside \[0, 3\)/);
    reused.encode(pass);
    expect(pass.dispatchWorkgroups).toHaveBeenCalledTimes(4);

    kernel.destroy();
    kernel.destroy();
    await Promise.resolve();
    expect(controls.destroy).toHaveBeenCalledOnce();
    expect(() => dispatch.encode(fakePass())).toThrow(/was destroyed/);
    await expect(kernel.createDispatch(
      "after-destroy",
      shape,
      bindings,
      "float16",
    )).rejects.toThrow(/was destroyed/);
  });

  it("keys bias and output-storage layouts independently", async () => {
    const device = fakeDevice();
    const kernel =
      AceOpt0011VaeConv1dFp16ScalarOracleKernel.create(device);
    const shape = tailShape(65);
    const f16 = bindingsFor(shape, "float16");
    const f16Bias = bindingsFor(shape, "float16", true);
    const f32 = bindingsFor(shape, "float32");
    await kernel.createDispatch("f16", shape, f16, "float16");
    await kernel.createDispatch("f16-bias", shape, f16Bias, "float16");
    await kernel.createDispatch("f32", shape, f32, "float32");
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(3);
    expect(device.createdBuffers).toHaveLength(3);
    expect(device.createBindGroupLayout.mock.calls.map(([descriptor]) =>
      Array.from(
        (descriptor as GPUBindGroupLayoutDescriptor).entries,
      ).length
    )).toEqual([4, 5, 4]);
    const shaderSources = device.createShaderModule.mock.calls.map(
      ([descriptor]) => (descriptor as GPUShaderModuleDescriptor).code,
    );
    expect(shaderSources[0]).toContain("output: array<f16>;");
    expect(shaderSources[1]).toContain("bias: array<f16>;");
    expect(shaderSources[2]).toContain("output: array<f32>;");
  });

  it("reports compile diagnostics, evicts the failure, and retries", async () => {
    const device = fakeDevice({
      compilationMessageBatches: [
        [{
          message: "synthetic FP16 diagnostic",
          type: "error",
          lineNum: 42,
          linePos: 7,
        }],
        [],
      ],
    });
    const kernel =
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(device);
    const shape = tailShape(65);
    const bindings = bindingsFor(shape, "float16");
    await expect(kernel.createDispatch(
      "diagnostic",
      shape,
      bindings,
      "float16",
    )).rejects.toThrow(/42:7 synthetic FP16 diagnostic/);
    expect(device.createComputePipelineAsync).not.toHaveBeenCalled();
    expect(device.createdBuffers).toHaveLength(0);
    await expect(kernel.createDispatch(
      "retry",
      shape,
      bindings,
      "float16",
    )).resolves.toMatchObject({ rangeCount: 1 });
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
  });

  it("evicts pipeline and scoped-allocation failures for clean retry", async () => {
    const pipelineFailure = new Error("synthetic pipeline failure");
    const pipelineDevice = fakeDevice({
      pipelineResults: [
        Promise.reject(pipelineFailure),
        Promise.resolve(fakePipeline()),
      ],
    });
    const pipelineKernel =
      AceOpt0011VaeConv1dFp16ScalarOracleKernel.create(pipelineDevice);
    const shape = tailShape(65);
    const bindings = bindingsFor(shape, "float16");
    await expect(pipelineKernel.createDispatch(
      "pipeline-failure",
      shape,
      bindings,
      "float16",
    )).rejects.toBe(pipelineFailure);
    await expect(pipelineKernel.createDispatch(
      "pipeline-retry",
      shape,
      bindings,
      "float16",
    )).resolves.toMatchObject({ rangeCount: 1 });
    expect(pipelineDevice.createComputePipelineAsync).toHaveBeenCalledTimes(2);

    const allocationFailure = {
      message: "synthetic control allocation failure",
    } as GPUError;
    const allocationDevice = fakeDevice({
      scopeResults: [
        allocationFailure,
        null,
        null,
        null,
        null,
        null,
      ],
    });
    const allocationKernel =
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(allocationDevice);
    await expect(allocationKernel.createDispatch(
      "allocation-failure",
      shape,
      bindings,
      "float16",
    )).rejects.toThrow(/control allocation failure/);
    expect(allocationDevice.createdBuffers[0]?.destroy).toHaveBeenCalledOnce();
    await expect(allocationKernel.createDispatch(
      "allocation-retry",
      shape,
      bindings,
      "float16",
    )).resolves.toMatchObject({ rangeCount: 1 });
    expect(allocationDevice.pushErrorScope).toHaveBeenCalledTimes(6);
    expect(allocationDevice.popErrorScope).toHaveBeenCalledTimes(6);
  });

  it("does not allocate after destruction wins a pipeline race", async () => {
    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pending = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const device = fakeDevice({ pipelineResults: [pending] });
    const kernel =
      AceOpt0011VaeConv1dFp16PortableWorkgroupKernel.create(device);
    const shape = tailShape(65);
    const dispatch = kernel.createDispatch(
      "pending-pipeline",
      shape,
      bindingsFor(shape, "float16"),
      "float16",
    );
    await Promise.resolve();
    kernel.destroy();
    resolvePipeline(fakePipeline());
    await expect(dispatch).rejects.toThrow(/was destroyed/);
    expect(device.createdBuffers).toHaveLength(0);
  });

  it("destroys a control buffer when destruction wins allocation", async () => {
    let releaseValidation!: (error: GPUError | null) => void;
    const pendingValidation = new Promise<GPUError | null>((resolve) => {
      releaseValidation = resolve;
    });
    const device = fakeDevice({
      scopeResults: [pendingValidation, null, null],
    });
    const kernel =
      AceOpt0011VaeConv1dFp16ScalarOracleKernel.create(device);
    const shape = tailShape(65);
    const dispatch = kernel.createDispatch(
      "pending-allocation",
      shape,
      bindingsFor(shape, "float16"),
      "float16",
    );
    await waitFor(() => device.createdBuffers.length === 1);
    kernel.destroy();
    releaseValidation(null);
    await expect(dispatch).rejects.toThrow(/was destroyed/);
    expect(device.createdBuffers[0]?.destroy).toHaveBeenCalledOnce();
  });
});

expect(ACE_OPT_0011_VAE_CONV1D_FP16_KERNEL_SIZE).toBe(7);
expect(ACE_OPT_0011_VAE_CONV1D_FP16_SUPPORTED_DILATIONS).toEqual([1, 3, 9]);
expect(ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_CHANNEL_CHUNK).toBe(64);
expect(ACE_OPT_0011_VAE_CONV1D_FP16_TILE_FRAMES).toBe(16);
expect(ACE_OPT_0011_VAE_CONV1D_FP16_TILE_CHANNELS).toBe(8);
expect(ACE_OPT_0011_VAE_CONV1D_FP16_INPUT_TILE_STRIDE).toBe(17);
expect(ACE_OPT_0011_VAE_CONV1D_FP16_WEIGHT_TILE_STRIDE).toBe(65);
expect(ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_X).toBe(16);
expect(ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE_Y).toBe(8);
expect(ACE_OPT_0011_VAE_CONV1D_FP16_WORKGROUP_SIZE).toBe(128);

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

function bindingsFor(
  shape: AceVaeConv1dShape,
  outputStorage: AceOpt0011VaeConv1dFp16OutputStorage,
  hasBias = false,
): AceOpt0011VaeConv1dFp16Bindings {
  const plan = planAceOpt0011VaeConv1dFp16(shape, outputStorage);
  const base = {
    input: fakeBinding(plan.inputBindingBytes),
    weight: fakeBinding(plan.weightBindingBytes),
    output: fakeBinding(plan.outputBindingBytes),
  };
  return hasBias
    ? { ...base, bias: fakeBinding(plan.biasBindingBytes) }
    : base;
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
  readonly pushErrorScope: ReturnType<typeof vi.fn>;
  readonly popErrorScope: ReturnType<typeof vi.fn>;
  readonly createdBuffers: ReturnType<typeof fakeMappedBuffer>[];
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
  readonly pipelineResults?: readonly Promise<GPUComputePipeline>[];
  readonly scopeResults?: readonly (
    GPUError | null | Promise<GPUError | null>
  )[];
  readonly compilationMessageBatches?: readonly (
    readonly Partial<GPUCompilationMessage>[]
  )[];
} = {}): FakeDevice {
  const createdBuffers: ReturnType<typeof fakeMappedBuffer>[] = [];
  const pipelineResults = options.pipelineResults === undefined
    ? undefined
    : [...options.pipelineResults];
  const scopeResults = [...(options.scopeResults ?? [null, null, null])];
  const compilationMessageBatches = [
    ...(options.compilationMessageBatches ?? [[]]),
  ];
  return {
    features: new Set(
      options.shaderF16 === false ? [] : ["shader-f16"],
    ),
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
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(() =>
      Promise.resolve(scopeResults.shift() ?? null)
    ),
    createShaderModule: vi.fn(() => {
      const messages = compilationMessageBatches.shift() ?? [];
      return {
        label: "module",
        getCompilationInfo: vi.fn(async () => ({ messages })),
      };
    }),
    createBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    createPipelineLayout: vi.fn(() => ({ label: "pipeline-layout" })),
    createComputePipelineAsync: vi.fn(() => {
      if (pipelineResults === undefined) {
        return Promise.resolve(fakePipeline());
      }
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

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (predicate()) return;
    await Promise.resolve();
  }
  throw new Error("condition did not become true");
}
