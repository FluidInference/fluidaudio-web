import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0017_VAE_CONV1D_ACCUMULATORS_PER_THREAD,
  ACE_OPT_0017_VAE_CONV1D_BARRIERS_PER_REDUCTION_TILE,
  ACE_OPT_0017_VAE_CONV1D_CHANNELS_PER_THREAD,
  ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID,
  ACE_OPT_0017_VAE_CONV1D_REDUCTION_SEMANTICS,
  ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE,
  ACE_OPT_0017_VAE_CONV1D_ROWS_PER_THREAD,
  ACE_OPT_0017_VAE_CONV1D_WEIGHT_PANEL_STRIDE,
  ACE_OPT_0017_VAE_CONV1D_WORKGROUP_SIZE,
  AceOpt0017VaeConv1dCooperativeDot4Kernel,
  aceOpt0017VaeConv1dCooperativeDot4Wgsl,
  planAceOpt0017VaeConv1dCooperativeDot4,
  planAceOpt0017VaeConv1dCooperativeDot4Range,
} from "../src/webgpu/kernels/vae-conv1d-fp16-cooperative-dot4.js";
import {
  planAceOpt0014VaeConv1dPackedKioWeight,
} from
  "../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-subgroup.js";
import {
  planAceFp16VaeConv1d,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
} from "../src/webgpu/vae-decoder.js";

describe("OPT-0017 VAE K7 cooperative dot4", () => {
  it("pins the sole reordered-rounding WG256 owner and adaptive resources", () => {
    expect(ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID).toBe(
      "ace-vae-fp16-opt-0017-reordered-dot4-packed-kio-k7-conv1d-v1",
    );
    expect(ACE_OPT_0017_VAE_CONV1D_REDUCTION_SEMANTICS).toBe(
      "reordered-rounding-dot4",
    );
    expect(ACE_OPT_0017_VAE_CONV1D_WORKGROUP_SIZE).toBe(256);
    expect(ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE).toBe(32);
    expect(ACE_OPT_0017_VAE_CONV1D_WEIGHT_PANEL_STRIDE).toBe(33);
    expect(ACE_OPT_0017_VAE_CONV1D_ROWS_PER_THREAD).toBe(4);
    expect(ACE_OPT_0017_VAE_CONV1D_CHANNELS_PER_THREAD).toBe(4);
    expect(ACE_OPT_0017_VAE_CONV1D_ACCUMULATORS_PER_THREAD).toBe(16);
    expect(ACE_OPT_0017_VAE_CONV1D_BARRIERS_PER_REDUCTION_TILE).toBe(2);

    expect(planSummary(k7Shape(1_024, 1_024, 17))).toEqual({
      tile: "m32n128",
      tileRows: 32,
      tileChannels: 128,
      rowBlocks: 8,
      channelBlocks: 32,
      inputPanelElements: 1_024,
      weightTileElements: 4_096,
      weightPanelElements: 4_224,
      workgroupStorageBytes: 10_496,
      reductionTileCount: 224,
      barriersPerWorkgroup: 448,
      estimatedGlobalOperandBytesPerWorkgroup: 2_293_760,
    });
    for (const [inputChannels, outputChannels] of [
      [64, 2_048],
      [512, 512],
      [256, 256],
      [128, 128],
    ] as const) {
      const summary = planSummary(
        k7Shape(inputChannels, outputChannels, 17),
      );
      expect(summary).toMatchObject({
        tile: "m64n64",
        tileRows: 64,
        tileChannels: 64,
        rowBlocks: 16,
        channelBlocks: 16,
        inputPanelElements: 2_048,
        weightTileElements: 2_048,
        weightPanelElements: 2_112,
        workgroupStorageBytes: 8_320,
      });
      expect(summary.reductionTileCount).toBe(7 * inputChannels / 32);
      expect(summary.barriersPerWorkgroup).toBe(
        14 * inputChannels / 32,
      );
      expect(summary.estimatedGlobalOperandBytesPerWorkgroup).toBe(
        (7 * inputChannels / 32) * 8_192,
      );
    }
  });

  it("assigns every adaptive tile output to exactly one 4x4 owner", () => {
    for (const shape of [
      k7Shape(1_024, 1_024, 17),
      k7Shape(128, 128, 17),
    ]) {
      const plan = planAceOpt0017VaeConv1dCooperativeDot4(shape);
      const owners = new Uint8Array(plan.tileRows * plan.tileChannels);
      for (
        let localIndex = 0;
        localIndex < ACE_OPT_0017_VAE_CONV1D_WORKGROUP_SIZE;
        localIndex += 1
      ) {
        const rowBlock = Math.floor(localIndex / plan.channelBlocks);
        const channelBlock = localIndex % plan.channelBlocks;
        for (let row = 0; row < 4; row += 1) {
          for (let channel = 0; channel < 4; channel += 1) {
            const tileRow = rowBlock * 4 + row;
            const tileChannel = channelBlock * 4 + channel;
            const index = tileRow * plan.tileChannels + tileChannel;
            owners[index] = owners[index]! + 1;
          }
        }
      }
      expect([...owners].every((count) => count === 1), plan.tile).toBe(true);
      expect(owners).toHaveLength(plan.tileRows * plan.tileChannels);
    }
  });

  it("uses OPT-0014 KIO bytes and transposes each R32 panel output-major", () => {
    const shape = k7Shape(128, 128, 17);
    const plan = planAceOpt0017VaeConv1dCooperativeDot4(shape);
    expect(plan.packedWeightPlan).toEqual(
      planAceOpt0014VaeConv1dPackedKioWeight(shape),
    );

    const native = Uint16Array.from(
      { length: shape.outputChannels * 7 * shape.inputChannels },
      (_, index) => [
        0x0000,
        0x8000,
        0x0001,
        0x03ff,
        0x7bff,
        0x7c00,
        0xfc00,
        0x7e01,
        index & 0xffff,
      ][index % 9]!,
    );
    const packed = packKioReference(
      native,
      shape.inputChannels,
      shape.outputChannels,
    );
    for (const kernel of [0, 3, 6]) {
      for (const inputChannel of [0, 1, 30, 31, 32, 95, 127]) {
        for (const outputChannel of [0, 1, 3, 4, 63, 64, 127]) {
          const scalarIndex =
            (kernel * shape.inputChannels + inputChannel) *
              shape.outputChannels +
            outputChannel;
          const nativeIndex =
            (outputChannel * 7 + kernel) * shape.inputChannels +
            inputChannel;
          const panelIndex =
            (outputChannel % plan.tileChannels) *
              ACE_OPT_0017_VAE_CONV1D_WEIGHT_PANEL_STRIDE +
            (inputChannel % ACE_OPT_0017_VAE_CONV1D_REDUCTION_TILE);
          expect(panelIndex % 33).toBe(inputChannel % 32);
          expect(packed[scalarIndex]).toBe(native[nativeIndex]);
          expect(packed[scalarIndex]).toBe(
            packed[
              (kernel * shape.inputChannels +
                Math.floor(inputChannel / 32) * 32 + panelIndex % 33) *
                shape.outputChannels +
              Math.floor(outputChannel / plan.tileChannels) *
                plan.tileChannels +
              Math.floor(panelIndex / 33)
            ],
          );
        }
      }
    }
  });

  it("generates dense panels, explicit FP32 dot4 groups, and uniform barriers", () => {
    for (const shape of [
      k7Shape(1_024, 1_024, 17),
      k7Shape(128, 128, 17, 9),
    ]) {
      const plan = planAceOpt0017VaeConv1dCooperativeDot4(shape);
      const source = aceOpt0017VaeConv1dCooperativeDot4Wgsl(
        shape,
        true,
        "float16",
      );
      expect(source).toContain(
        `// kernel-id: ${ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID}`,
      );
      expect(source).toContain(
        `// reduction-semantics: ${ACE_OPT_0017_VAE_CONV1D_REDUCTION_SEMANTICS}`,
      );
      expect(source).toContain("enable f16;");
      expect(source).not.toContain("enable subgroups;");
      expect(source).not.toContain("subgroup");
      expect(source).toContain("@compute @workgroup_size(256, 1, 1)");
      expect(source).toContain(
        `var<workgroup> input_panel: array<f16, ${plan.inputPanelElements}>`,
      );
      expect(source).toContain(
        `var<workgroup> weight_panel: array<f16, ${plan.weightPanelElements}>`,
      );
      expect(source).toContain("const WEIGHT_PANEL_STRIDE: u32 =\n  33u;");
      expect(source).toContain("var initial_sum = vec4<f32>(0.0);");
      expect(source.match(/var sum[0-3] = initial_sum;/g)).toHaveLength(4);
      expect(source).toContain("for (var kernel = 0u; kernel < 7u;");
      expect(source).toContain("input_channel_base += REDUCTION_TILE");
      expect(source).toContain("let panel_r = load_index / TILE_CHANNELS;");
      expect(source).toContain(
        "let panel_channel = load_index % TILE_CHANNELS;",
      );
      expect(source).toContain(
        "panel_channel * WEIGHT_PANEL_STRIDE + panel_r",
      );
      expect(source).toContain(
        "(kernel * INPUT_CHANNELS + input_channel_base + panel_r) *",
      );
      expect(source).toContain("let input_operand0 = vec4<f32>(");
      expect(source).toContain("let weight_operand00 = vec4<f32>(");
      expect(source.match(/dot\(input_operand[0-3], weight_operand[0-3][0-3]\)/g))
        .toHaveLength(16);
      expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
      expect(source).toContain(
        "if (row_valid0 && output_channel_base + 3u < OUTPUT_CHANNELS)",
      );
      expect(source).toContain("var value = f16(0.0);");
      expect(source).toContain("output[store_base0] = f16(sum0.x);");
      expect(source).not.toMatch(/\bfma\s*\(/);
      expect(source).not.toMatch(/\batomic/);
      expect(source).not.toContain("array<f32, 16>");
      expect(source).not.toContain("output: array<f32>");
      expect(source).not.toContain("return;");
      for (const barrierIndex of indicesOf(source, "workgroupBarrier();")) {
        expect(braceDepthAt(source, barrierIndex)).toBe(3);
      }
      const invalidGuard = source.indexOf("if (row_valid0 &&");
      const firstDot = source.indexOf("dot(input_operand0", invalidGuard);
      expect(invalidGuard).toBeGreaterThan(source.indexOf("workgroupBarrier();"));
      expect(firstDot).toBeGreaterThan(invalidGuard);
      expect(firstDot).toBeLessThan(source.lastIndexOf("workgroupBarrier();"));
    }
  });

  it("explicitly widens every f16 scalar in FP32 vector constructors", () => {
    const source = aceOpt0017VaeConv1dCooperativeDot4Wgsl(
      k7Shape(1_024, 1_024, 17),
      true,
      "float16",
    );
    const constructors = parenthesizedBodies(source, "vec4<f32>(").filter(
      (body) => /(?:bias|input_panel|weight_panel)\[/.test(body),
    );
    expect(constructors).toHaveLength(21);
    let scalarReferenceCount = 0;
    for (const body of constructors) {
      const scalarReferences = body.match(
        /(?:bias|input_panel|weight_panel)\[[^\]]+\]/g,
      ) ?? [];
      const explicitConversions = body.match(
        /f32\(\s*(?:bias|input_panel|weight_panel)\[[^\]]+\]\s*\)/g,
      ) ?? [];
      expect(scalarReferences).toHaveLength(4);
      expect(explicitConversions).toHaveLength(4);
      scalarReferenceCount += scalarReferences.length;
    }
    expect(scalarReferenceCount).toBe(84);
    expect(source).not.toMatch(
      /vec4<f32>\(\s*(?:bias|input_panel|weight_panel)\[/,
    );
  });

  it.each([
    [
      256,
      16,
      2_041,
      88_448,
      7_314_944,
      30_842_814_464,
      479_392_169_984,
    ],
    [
      300,
      16,
      2_399,
      103_684,
      8_579_200,
      36_163_764_224,
      561_787_699_200,
    ],
    [
      512,
      16,
      4_082,
      176_896,
      14_629_888,
      61_685_628_928,
      958_784_339_968,
    ],
  ] as const)(
    "pins the complete 16-operation candidate and accounting at C%i",
    (
      frames,
      expectedOperations,
      expectedRanges,
      expectedWorkgroups,
      expectedBarriers,
      expectedGlobalBytes,
      expectedMacs,
    ) => {
      const graph = planAceVaeDecoder(frames);
      const cooperative = planAceVaeDecoderQuanta(graph);
      const operationCoverage = new Map<number, number>();
      const candidateOperations = new Set<number>();
      let rangeCount = 0;
      let physicalWorkgroups = 0;
      let barrierEvents = 0;
      let estimatedGlobalOperandBytes = 0;
      let logicalMacs = 0;
      for (const quantum of cooperative.quanta) {
        const operation = graph.operations[quantum.operationIndex]!;
        if (
          operation.kind !== "conv1d" ||
          operation.shape.kernelSize !== 7 ||
          operation.bias === undefined
        ) continue;
        expect(quantum.primitives).toHaveLength(1);
        const primitive = quantum.primitives[0]!;
        const plan = planAceFp16VaeConv1d(operation.shape, "float16");
        const range = planAceOpt0017VaeConv1dCooperativeDot4Range(
          plan,
          { base: primitive.outputBase, count: primitive.outputCount },
        );
        expect(range.count).toBe(quantum.logicalOutputCount);
        expect(range.outputRowCount * operation.shape.outputChannels)
          .toBe(range.count);
        rangeCount += 1;
        physicalWorkgroups += range.physicalWorkgroups;
        barrierEvents += range.barrierEvents;
        estimatedGlobalOperandBytes += range.estimatedGlobalOperandBytes;
        logicalMacs += range.count * 7 * operation.shape.inputChannels;
        candidateOperations.add(quantum.operationIndex);
        operationCoverage.set(
          quantum.operationIndex,
          (operationCoverage.get(quantum.operationIndex) ?? 0) + range.count,
        );
      }
      expect(candidateOperations.size).toBe(expectedOperations);
      for (const [operationIndex, covered] of operationCoverage) {
        const operation = graph.operations[operationIndex]!;
        expect(operation.kind).toBe("conv1d");
        if (operation.kind !== "conv1d") continue;
        expect(covered).toBe(
          operation.shape.batch * operation.shape.inputFrames *
            operation.shape.outputChannels,
        );
      }
      expect({
        rangeCount,
        physicalWorkgroups,
        barrierEvents,
        estimatedGlobalOperandBytes,
        logicalMacs,
      }).toEqual({
        rangeCount: expectedRanges,
        physicalWorkgroups: expectedWorkgroups,
        barrierEvents: expectedBarriers,
        estimatedGlobalOperandBytes: expectedGlobalBytes,
        logicalMacs: expectedMacs,
      });

      let packedBytes = 0;
      let repackWorkgroups = 0;
      for (const operation of graph.operations) {
        if (
          operation.kind !== "conv1d" ||
          operation.shape.kernelSize !== 7 ||
          operation.bias === undefined
        ) continue;
        const packed = planAceOpt0014VaeConv1dPackedKioWeight(
          operation.shape,
        );
        packedBytes += packed.packedStorageBytes;
        repackWorkgroups += packed.repackWorkgroups;
      }
      expect(packedBytes).toBe(61_014_016);
      expect(repackWorkgroups).toBe(59_584);

      const final = graph.operations.at(-1)!;
      expect(final).toMatchObject({ kind: "conv1d", label: "conv2" });
      if (final.kind !== "conv1d") throw new Error("expected final Conv1D");
      expect(final.bias).toBeUndefined();
      expect(() => planAceOpt0017VaeConv1dCooperativeDot4(final.shape))
        .toThrow(/16 biased production K7 channel shapes/);
    },
  );

  it("plans complete-row ranges and excludes padded tail rows from A traffic", () => {
    const shape = {
      ...k7Shape(128, 128, 65),
      batch: 2,
    };
    const plan = planAceFp16VaeConv1d(shape, "float16");
    const range = planAceOpt0017VaeConv1dCooperativeDot4Range(
      plan,
      { base: 65 * 128, count: 33 * 128 },
    );
    expect(range).toMatchObject({
      base: 65 * 128,
      count: 33 * 128,
      batch: 1,
      firstOutputTime: 0,
      firstOutputRow: 65,
      outputRowCount: 33,
      tile: "m64n64",
      tileRows: 64,
      tileChannels: 64,
      workgroupsX: 1,
      workgroupsY: 2,
      physicalWorkgroups: 2,
      barrierEvents: 112,
      estimatedGlobalOperandBytes: 347_648,
    });
    const fullPaddedEstimate = range.physicalWorkgroups *
      planAceOpt0017VaeConv1dCooperativeDot4(shape)
        .estimatedGlobalOperandBytesPerWorkgroup;
    expect(range.estimatedGlobalOperandBytes).toBeLessThan(fullPaddedEstimate);
  });

  it("caches pipelines/bind groups, binds dynamic ranges, and encodes geometry", async () => {
    const device = fakeDevice();
    const owner = AceOpt0017VaeConv1dCooperativeDot4Kernel.create(device);
    const shape = k7Shape(1_024, 1_024, 65);
    const bindings = bindingsFor(shape);
    const control = fakeBuffer(1_024);
    const first = await owner.createDispatch(
      "first",
      shape,
      bindings,
      "float16",
      rangeBinding(control, 256, 0, 33 * shape.outputChannels),
    );
    const second = await owner.createDispatch(
      "second",
      shape,
      bindings,
      "float16",
      rangeBinding(
        control,
        512,
        33 * shape.outputChannels,
        32 * shape.outputChannels,
      ),
    );
    const smallShape = k7Shape(128, 128, 65);
    const small = await owner.createDispatch(
      "small",
      smallShape,
      bindingsFor(smallShape),
      "float16",
      rangeBinding(
        fakeBuffer(256),
        0,
        0,
        smallShape.inputFrames * smallShape.outputChannels,
      ),
    );

    expect(first).toMatchObject({
      kernelId: ACE_OPT_0017_VAE_CONV1D_COOPERATIVE_DOT4_KERNEL_ID,
      cooperativePlan: { tile: "m32n128" },
      outputRange: { workgroupsX: 2, workgroupsY: 8 },
    });
    expect(second.outputRange).toMatchObject({
      workgroupsX: 1,
      workgroupsY: 8,
    });
    expect(small.outputRange).toMatchObject({
      workgroupsX: 2,
      workgroupsY: 2,
    });
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createBindGroup).toHaveBeenCalledTimes(2);
    expect(device.createBindGroupLayout).toHaveBeenCalledTimes(2);
    for (const [descriptor] of device.createBindGroupLayout.mock.calls) {
      expect(descriptor.entries.map((entry: GPUBindGroupLayoutEntry) =>
        entry.buffer?.minBindingSize
      )).toHaveLength(5);
      expect(descriptor.entries[4]).toMatchObject({
        binding: 4,
        buffer: {
          type: "uniform",
          hasDynamicOffset: true,
          minBindingSize: 16,
        },
      });
    }

    const pass = fakePass();
    first.encode(pass);
    second.encode(pass);
    small.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([[256], [512], [0]]);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [2, 8, 1],
      [1, 8, 1],
      [2, 2, 1],
    ]);

    owner.destroy();
    owner.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
    await expect(owner.createDispatch(
      "after-destroy",
      shape,
      bindings,
      "float16",
      rangeBinding(control, 0, 0, shape.outputChannels),
    )).rejects.toThrow(/was destroyed/);
  });

  it("rejects final FP32 conv2 and any missing-bias route before compilation", async () => {
    const device = fakeDevice();
    const owner = AceOpt0017VaeConv1dCooperativeDot4Kernel.create(device);
    const finalShape = k7Shape(128, 2, 17);
    const finalPlan = planAceFp16VaeConv1d(finalShape, "float32");
    const finalBindings = {
      input: fakeBinding(finalPlan.inputBindingBytes),
      packedWeight: fakeBinding(finalPlan.weightBindingBytes),
      bias: undefined as unknown as GPUBufferBinding,
      output: fakeBinding(finalPlan.outputBindingBytes),
    };
    await expect(owner.createDispatch(
      "final",
      finalShape,
      finalBindings,
      "float32",
      rangeBinding(
        fakeBuffer(256),
        0,
        0,
        finalShape.inputFrames * finalShape.outputChannels,
      ),
    )).rejects.toThrow(/final FP32 conv2 remains shipped fixed32/);
    const internalShape = k7Shape(128, 128, 17);
    await expect(owner.createDispatch(
      "missing-bias",
      internalShape,
      {
        ...bindingsFor(internalShape),
        bias: undefined as unknown as GPUBufferBinding,
      },
      "float16",
      rangeBinding(fakeBuffer(256), 0, 0, internalShape.outputChannels),
    )).rejects.toThrow(/only biased FP16-output/);
    expect(() => aceOpt0017VaeConv1dCooperativeDot4Wgsl(
      finalShape,
      false,
      "float32",
    )).toThrow(/final FP32 conv2 remains shipped fixed32/);
    expect(device.createShaderModule).not.toHaveBeenCalled();
  });

  it("fails closed on capabilities, shapes, ranges, and device limits", async () => {
    expect(() => AceOpt0017VaeConv1dCooperativeDot4Kernel.create(
      fakeDevice({ shaderF16: false }),
    )).toThrow(/requires WebGPU shader-f16/);
    expect(() => AceOpt0017VaeConv1dCooperativeDot4Kernel.create(
      fakeDevice({ maximumInvocations: 255 }),
    )).toThrow(/requires WG256/);
    expect(() => AceOpt0017VaeConv1dCooperativeDot4Kernel.create(
      fakeDevice({ maximumWorkgroupSizeX: 255 }),
    )).toThrow(/requires WG256/);
    expect(() => planAceOpt0017VaeConv1dCooperativeDot4({
      ...k7Shape(128, 128, 17),
      kernelSize: 1,
      padding: 0,
    })).toThrow(/supports only K7/);
    expect(() => planAceOpt0017VaeConv1dCooperativeDot4(
      k7Shape(65, 128, 17),
    )).toThrow(/divisible by R32/);
    expect(() => planAceOpt0017VaeConv1dCooperativeDot4(
      k7Shape(64, 64, 17),
    )).toThrow(/16 biased production K7 channel shapes/);

    const plan = planAceFp16VaeConv1d(
      k7Shape(128, 128, 65),
      "float16",
    );
    expect(() => planAceOpt0017VaeConv1dCooperativeDot4Range(
      plan,
      { base: 1, count: 128 },
    )).toThrow(/complete in-bounds NLC rows/);
    const batchedPlan = planAceFp16VaeConv1d(
      { ...k7Shape(128, 128, 65), batch: 2 },
      "float16",
    );
    expect(() => planAceOpt0017VaeConv1dCooperativeDot4Range(
      batchedPlan,
      { base: 64 * 128, count: 2 * 128 },
    )).toThrow(/must not cross a batch boundary/);

    const shape = k7Shape(1_024, 1_024, 65);
    const range = rangeBinding(
      fakeBuffer(256),
      0,
      0,
      shape.inputFrames * shape.outputChannels,
    );
    const storageLimited =
      AceOpt0017VaeConv1dCooperativeDot4Kernel.create(
        fakeDevice({ maximumWorkgroupStorage: 10_495 }),
      );
    await expect(storageLimited.createDispatch(
      "storage-limit",
      shape,
      bindingsFor(shape),
      "float16",
      range,
    )).rejects.toThrow(/requires 10496 workgroup-storage bytes/);
    const dispatchLimited =
      AceOpt0017VaeConv1dCooperativeDot4Kernel.create(
        fakeDevice({ maximumDispatch: 1 }),
      );
    await expect(dispatchLimited.createDispatch(
      "dispatch-limit",
      shape,
      bindingsFor(shape),
      "float16",
      range,
    )).rejects.toThrow(/exceeds the device dispatch dimension/);
    const bufferLimited =
      AceOpt0017VaeConv1dCooperativeDot4Kernel.create(
        fakeDevice({ maximumStorageBinding: 1 }),
      );
    await expect(bufferLimited.createDispatch(
      "buffer-limit",
      shape,
      bindingsFor(shape),
      "float16",
      range,
    )).rejects.toThrow(/exceeds the device buffer limits/);
  });

  it("rejects malformed or overlapping bindings before shader compilation", async () => {
    const shape = k7Shape(128, 128, 17);
    const plan = planAceFp16VaeConv1d(shape, "float16");
    const valid = bindingsFor(shape);
    const range = rangeBinding(fakeBuffer(256), 0, 0, shape.outputChannels);

    const tooSmallDevice = fakeDevice();
    const tooSmall =
      AceOpt0017VaeConv1dCooperativeDot4Kernel.create(tooSmallDevice);
    await expect(tooSmall.createDispatch(
      "too-small",
      shape,
      { ...valid, input: fakeBinding(plan.inputBindingBytes - 4) },
      "float16",
      range,
    )).rejects.toThrow(/does not expose an aligned/);
    expect(tooSmallDevice.createShaderModule).not.toHaveBeenCalled();

    const misalignedDevice = fakeDevice();
    const misaligned =
      AceOpt0017VaeConv1dCooperativeDot4Kernel.create(misalignedDevice);
    await expect(misaligned.createDispatch(
      "misaligned",
      shape,
      {
        ...valid,
        input: {
          buffer: fakeBuffer(plan.inputBindingBytes + 256),
          offset: 4,
          size: plan.inputBindingBytes,
        },
      },
      "float16",
      range,
    )).rejects.toThrow(/does not expose an aligned/);
    expect(misalignedDevice.createShaderModule).not.toHaveBeenCalled();

    const badControlDevice = fakeDevice();
    const badControl =
      AceOpt0017VaeConv1dCooperativeDot4Kernel.create(badControlDevice);
    await expect(badControl.createDispatch(
      "bad-control",
      shape,
      valid,
      "float16",
      rangeBinding(fakeBuffer(256), 4, 0, shape.outputChannels),
    )).rejects.toThrow(/aligned 16-byte immutable record/);
    expect(badControlDevice.createShaderModule).not.toHaveBeenCalled();

    const aliasDevice = fakeDevice();
    const alias = AceOpt0017VaeConv1dCooperativeDot4Kernel.create(aliasDevice);
    const shared = fakeBuffer(Math.max(
      plan.inputBindingBytes,
      plan.outputBindingBytes,
    ));
    await expect(alias.createDispatch(
      "alias",
      shape,
      {
        ...valid,
        input: { buffer: shared, offset: 0, size: plan.inputBindingBytes },
        output: { buffer: shared, offset: 0, size: plan.outputBindingBytes },
      },
      "float16",
      range,
    )).rejects.toThrow(/input and output bindings must not overlap/);
    expect(aliasDevice.createShaderModule).not.toHaveBeenCalled();
  });

  it("evicts failed compilations and rejects destruction during compilation", async () => {
    const shape = k7Shape(128, 128, 17);
    const bindings = bindingsFor(shape);
    const range = rangeBinding(fakeBuffer(256), 0, 0, shape.outputChannels);

    const failedDevice = fakeDevice({ compilationError: "dot4 rejected" });
    const failed =
      AceOpt0017VaeConv1dCooperativeDot4Kernel.create(failedDevice);
    await expect(failed.createDispatch(
      "failed-first",
      shape,
      bindings,
      "float16",
      range,
    )).rejects.toThrow(/dot4 rejected/);
    await Promise.resolve();
    await expect(failed.createDispatch(
      "failed-retry",
      shape,
      bindings,
      "float16",
      range,
    )).rejects.toThrow(/dot4 rejected/);
    expect(failedDevice.createShaderModule).toHaveBeenCalledTimes(2);
    expect(failedDevice.createComputePipelineAsync).not.toHaveBeenCalled();
    expect(failedDevice.createBindGroup).not.toHaveBeenCalled();

    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pipeline = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const pendingDevice = fakeDevice({ pipeline });
    const pendingOwner =
      AceOpt0017VaeConv1dCooperativeDot4Kernel.create(pendingDevice);
    const pending = pendingOwner.createDispatch(
      "pending",
      shape,
      bindings,
      "float16",
      range,
    );
    await vi.waitFor(() => {
      expect(pendingDevice.createComputePipelineAsync).toHaveBeenCalledOnce();
    });
    pendingOwner.destroy();
    resolvePipeline({ label: "late" } as GPUComputePipeline);
    await expect(pending).rejects.toThrow(/was destroyed/);
    expect(pendingDevice.createBindGroup).not.toHaveBeenCalled();
  });
});

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

function planSummary(shape: AceVaeConv1dShape) {
  const plan = planAceOpt0017VaeConv1dCooperativeDot4(shape);
  return {
    tile: plan.tile,
    tileRows: plan.tileRows,
    tileChannels: plan.tileChannels,
    rowBlocks: plan.rowBlocks,
    channelBlocks: plan.channelBlocks,
    inputPanelElements: plan.inputPanelElements,
    weightTileElements: plan.weightTileElements,
    weightPanelElements: plan.weightPanelElements,
    workgroupStorageBytes: plan.workgroupStorageBytes,
    reductionTileCount: plan.reductionTileCount,
    barriersPerWorkgroup: plan.barriersPerWorkgroup,
    estimatedGlobalOperandBytesPerWorkgroup:
      plan.estimatedGlobalOperandBytesPerWorkgroup,
  };
}

function k7Shape(
  inputChannels: number,
  outputChannels: number,
  inputFrames: number,
  dilation = 1,
): AceVaeConv1dShape {
  return {
    batch: 1,
    inputFrames,
    inputChannels,
    outputChannels,
    kernelSize: 7,
    stride: 1,
    dilation,
    padding: 3 * dilation,
  };
}

function packKioReference(
  native: Uint16Array,
  inputChannels: number,
  outputChannels: number,
): Uint16Array {
  const packed = new Uint16Array(native.length);
  for (let kernel = 0; kernel < 7; kernel += 1) {
    for (let inputChannel = 0; inputChannel < inputChannels; inputChannel += 1) {
      for (
        let outputChannel = 0;
        outputChannel < outputChannels;
        outputChannel += 1
      ) {
        packed[
          (kernel * inputChannels + inputChannel) * outputChannels +
          outputChannel
        ] = native[
          (outputChannel * 7 + kernel) * inputChannels + inputChannel
        ]!;
      }
    }
  }
  return packed;
}

function bindingsFor(shape: AceVaeConv1dShape) {
  const plan = planAceFp16VaeConv1d(shape, "float16");
  return {
    input: fakeBinding(plan.inputBindingBytes),
    packedWeight: fakeBinding(
      planAceOpt0014VaeConv1dPackedKioWeight(shape).packedBindingBytes,
    ),
    bias: fakeBinding(plan.biasBindingBytes),
    output: fakeBinding(plan.outputBindingBytes),
  };
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

function indicesOf(source: string, needle: string): number[] {
  const result: number[] = [];
  let offset = 0;
  while (true) {
    const index = source.indexOf(needle, offset);
    if (index < 0) return result;
    result.push(index);
    offset = index + needle.length;
  }
}

function parenthesizedBodies(source: string, prefix: string): string[] {
  const result: string[] = [];
  for (const prefixIndex of indicesOf(source, prefix)) {
    const bodyStart = prefixIndex + prefix.length;
    let depth = 1;
    for (let index = bodyStart; index < source.length; index += 1) {
      const character = source[index];
      if (character === "(") depth += 1;
      if (character === ")") depth -= 1;
      if (depth === 0) {
        result.push(source.slice(bodyStart, index));
        break;
      }
    }
    if (depth !== 0) throw new Error(`unclosed ${prefix} constructor`);
  }
  return result;
}

function braceDepthAt(source: string, index: number): number {
  let depth = 0;
  for (const character of source.slice(0, index)) {
    if (character === "{") depth += 1;
    if (character === "}") depth -= 1;
  }
  return depth;
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
  readonly maximumWorkgroupStorage?: number;
  readonly maximumDispatch?: number;
  readonly maximumStorageBinding?: number;
  readonly maximumBuffer?: number;
  readonly uniformAlignment?: number;
  readonly storageAlignment?: number;
  readonly compilationError?: string;
  readonly pipeline?: Promise<GPUComputePipeline>;
} = {}): FakeDevice {
  return {
    features: new Set(options.shaderF16 === false ? [] : ["shader-f16"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: options.maximumInvocations ?? 256,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSizeX ?? 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupStorageSize:
        options.maximumWorkgroupStorage ?? 16_384,
      maxComputeWorkgroupsPerDimension: options.maximumDispatch ?? 65_535,
      maxStorageBufferBindingSize:
        options.maximumStorageBinding ?? 1_073_741_824,
      maxBufferSize: options.maximumBuffer ?? 1_073_741_824,
      minUniformBufferOffsetAlignment: options.uniformAlignment ?? 256,
      minStorageBufferOffsetAlignment: options.storageAlignment ?? 256,
    },
    createShaderModule: vi.fn(() => ({
      label: "module",
      getCompilationInfo: vi.fn(async () => ({
        messages: options.compilationError === undefined
          ? []
          : [{
              type: "error",
              lineNum: 1,
              linePos: 1,
              message: options.compilationError,
            }],
      })),
    })),
    createBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    createPipelineLayout: vi.fn(() => ({ label: "pipeline-layout" })),
    createComputePipelineAsync: vi.fn(() =>
      options.pipeline ?? Promise.resolve({ label: "pipeline" })
    ),
    createBindGroup: vi.fn(() => ({ label: "bind-group" })),
  } as unknown as FakeDevice;
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
