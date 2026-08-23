import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0016_VAE_CONV1D_MAX_CHANNEL_BANDS,
  ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_8X32_KERNEL_ID,
  ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_8X64_KERNEL_ID,
  ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_16X32_KERNEL_ID,
  ACE_OPT_0016_VAE_CONV1D_PRIMARY_VARIANT,
  ACE_OPT_0016_VAE_CONV1D_SUBGROUP_SIZE,
  ACE_OPT_0016_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP,
  ACE_OPT_0016_VAE_CONV1D_VARIANTS,
  ACE_OPT_0016_VAE_CONV1D_WORKGROUP_SIZE,
  AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel,
  aceOpt0016VaeConv1dPackedKioMicrotileWgsl,
  planAceOpt0016VaeConv1dPackedKioMicrotile,
  planAceOpt0016VaeConv1dPackedKioMicrotileRange,
  type AceOpt0016VaeConv1dMicrotileVariant,
} from
  "../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-microtile-subgroup.js";
import {
  planAceOpt0014VaeConv1dPackedKioWeight,
} from
  "../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-subgroup.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dOutputStorage,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
} from "../src/webgpu/vae-decoder.js";

describe("OPT-0016 packed-KIO fixed32 VAE K7 microtiles", () => {
  it("pins the three explicit fixed32 variants and their adaptive bands", () => {
    expect(ACE_OPT_0016_VAE_CONV1D_SUBGROUP_SIZE).toBe(32);
    expect(ACE_OPT_0016_VAE_CONV1D_WORKGROUP_SIZE).toBe(128);
    expect(ACE_OPT_0016_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP).toBe(4);
    expect(ACE_OPT_0016_VAE_CONV1D_MAX_CHANNEL_BANDS).toBe(4);
    expect(ACE_OPT_0016_VAE_CONV1D_PRIMARY_VARIANT).toBe("16x32");
    expect(ACE_OPT_0016_VAE_CONV1D_VARIANTS).toEqual([
      "8x64",
      "16x32",
      "8x32",
    ]);

    expect(variantSummary("8x64")).toEqual({
      variant: "8x64",
      kernelId: ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_8X64_KERNEL_ID,
      rowsPerSubgroup: 8,
      outputsPerLane: 2,
      channelsPerSubgroup: 64,
      accumulatorsPerLane: 16,
      weightView: "vec2-f16",
      geometries: [
        [2, 1, 4, 32, 64],
        [64, 1, 4, 32, 64],
        [128, 2, 2, 16, 128],
        [256, 4, 1, 8, 256],
        [1_024, 4, 1, 8, 256],
      ],
    });
    expect(variantSummary("16x32")).toEqual({
      variant: "16x32",
      kernelId: ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_16X32_KERNEL_ID,
      rowsPerSubgroup: 16,
      outputsPerLane: 1,
      channelsPerSubgroup: 32,
      accumulatorsPerLane: 16,
      weightView: "scalar-f16",
      geometries: [
        [2, 1, 4, 64, 32],
        [64, 2, 2, 32, 64],
        [128, 4, 1, 16, 128],
        [256, 4, 1, 16, 128],
        [1_024, 4, 1, 16, 128],
      ],
    });
    expect(variantSummary("8x32")).toEqual({
      variant: "8x32",
      kernelId: ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_8X32_KERNEL_ID,
      rowsPerSubgroup: 8,
      outputsPerLane: 1,
      channelsPerSubgroup: 32,
      accumulatorsPerLane: 8,
      weightView: "scalar-f16",
      geometries: [
        [2, 1, 4, 32, 32],
        [64, 2, 2, 16, 64],
        [128, 4, 1, 8, 128],
        [256, 4, 1, 8, 128],
        [1_024, 4, 1, 8, 128],
      ],
    });
  });

  it.each(ACE_OPT_0016_VAE_CONV1D_VARIANTS)(
    "%s assigns every row/channel in each adaptive tile exactly once",
    (variant) => {
      for (const outputChannels of [2, 64, 128, 256, 1_024]) {
        const microtile = planAceOpt0016VaeConv1dPackedKioMicrotile(
          k7Shape(128, outputChannels, 17),
          variant,
        );
        const activeChannels = Math.min(
          outputChannels,
          microtile.tileChannels,
        );
        const owners = new Uint8Array(
          microtile.tileRows * activeChannels,
        );
        for (
          let subgroup = 0;
          subgroup < ACE_OPT_0016_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP;
          subgroup += 1
        ) {
          const channelBand = subgroup % microtile.channelBands;
          const rowBand = Math.floor(subgroup / microtile.channelBands);
          for (
            let row = 0;
            row < microtile.rowsPerSubgroup;
            row += 1
          ) {
            for (
              let lane = 0;
              lane < ACE_OPT_0016_VAE_CONV1D_SUBGROUP_SIZE;
              lane += 1
            ) {
              for (
                let component = 0;
                component < microtile.outputsPerLane;
                component += 1
              ) {
                const outputRow =
                  rowBand * microtile.rowsPerSubgroup + row;
                const outputChannel =
                  channelBand * microtile.channelsPerSubgroup +
                  lane * microtile.outputsPerLane + component;
                if (outputChannel < activeChannels) {
                  const index = outputRow * activeChannels + outputChannel;
                  owners[index] = owners[index]! + 1;
                }
              }
            }
          }
        }
        expect(
          [...owners].every((count) => count === 1),
          `${variant} Cout=${outputChannels}`,
        ).toBe(true);
      }
    },
  );

  it.each(ACE_OPT_0016_VAE_CONV1D_VARIANTS)(
    "%s reuses the exact OPT-0014 packed payload and source-order arithmetic",
    (variant) => {
      const shape = k7Shape(128, 256, 17);
      const microtile = planAceOpt0016VaeConv1dPackedKioMicrotile(
        shape,
        variant,
      );
      expect(microtile.packedWeightPlan).toEqual(
        planAceOpt0014VaeConv1dPackedKioWeight(shape),
      );
      const source = aceOpt0016VaeConv1dPackedKioMicrotileWgsl(
        shape,
        variant,
        true,
        "float16",
      );
      expect(source).toContain(`// kernel-id: ${microtile.kernelId}`);
      expect(source).toContain("enable f16;");
      expect(source).toContain("enable subgroups;");
      expect(source).toContain("@compute @workgroup_size(128, 1, 1)");
      expect(source).toContain("if (subgroup_size == 32u)");
      expect(source).toContain("let channel_band = subgroup % CHANNEL_BANDS");
      expect(source).toContain("let row_band = subgroup / CHANNEL_BANDS");
      expect(source).toContain(
        `row_band * ${microtile.rowsPerSubgroup}u`,
      );
      expect(source).toContain(
        `channel_band * ${microtile.channelsPerSubgroup}u`,
      );
      expect(source).toContain(
        `subgroup_lane * ${microtile.outputsPerLane}u`,
      );
      expect(source).toContain("for (var kernel = 0u; kernel < 7u;");
      expect(source).toContain("var input_channel = 0u;");
      expect(source).not.toContain("array<u32>");
      expect(source).not.toContain("unpack2x16float");
      expect(source).not.toContain("var<workgroup>");
      expect(source).not.toContain("workgroupBarrier");
      expect(source).not.toMatch(/\bfma\s*\(/);
      expect(source).not.toMatch(/\bdot\s*\(/);

      if (variant === "8x64") {
        expect(source).toContain("packed_weight: array<vec2<f16>>");
        expect(source).toContain(
          "weight_operands = vec2<f32>(packed_weight[packed_index])",
        );
        expect(source).toContain("output_channel_base / 2u");
      } else {
        expect(source).toContain("packed_weight: array<f16>");
        expect(source).toContain(
          "weight_operands = f32(packed_weight[packed_index])",
        );
        expect(source).toContain(
          "(kernel * INPUT_CHANNELS + input_channel) * OUTPUT_CHANNELS",
        );
      }
      for (let row = 0; row < microtile.rowsPerSubgroup; row += 1) {
        const accumulator = microtile.outputsPerLane === 2
          ? "vec2<f32>"
          : "f32";
        const operand = microtile.outputsPerLane === 2
          ? "vec2<f32>"
          : "f32";
        expect(source).toContain(
          `var sum${row}: ${accumulator} = initial_sum;`,
        );
        expect(source).toContain(
          `let input_operand${row} = subgroupBroadcast(lane_input, ${row}u);`,
        );
        expect(source).toContain(
          `sum${row} = sum${row} +\n` +
            `          ${operand}(input_operand${row}) * weight_operands;`,
        );
      }
      const kernel = source.indexOf("var kernel = 0u;");
      const inputChannel = source.indexOf("var input_channel = 0u;", kernel);
      const add = source.indexOf("sum0 = sum0 +", inputChannel);
      expect(kernel).toBeLessThan(inputChannel);
      expect(inputChannel).toBeLessThan(add);
    },
  );

  it("maps scalar and vec2 typed loads to the same packed KIO U16 bytes", () => {
    const inputChannels = 3;
    const outputChannels = 64;
    const native = Uint16Array.from(
      { length: outputChannels * 7 * inputChannels },
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
      inputChannels,
      outputChannels,
    );
    const packedBytes = new Uint8Array(packed.buffer);
    for (let kernel = 0; kernel < 7; kernel += 1) {
      for (
        let inputChannel = 0;
        inputChannel < inputChannels;
        inputChannel += 1
      ) {
        for (
          let outputChannel = 0;
          outputChannel < outputChannels;
          outputChannel += 1
        ) {
          const scalarIndex =
            (kernel * inputChannels + inputChannel) * outputChannels +
            outputChannel;
          const pairIndex =
            (kernel * inputChannels + inputChannel) *
              (outputChannels / 2) +
            Math.floor(outputChannel / 2);
          const component = outputChannel % 2;
          const pairScalarIndex = pairIndex * 2 + component;
          const scalarByteOffset = scalarIndex * Uint16Array.BYTES_PER_ELEMENT;
          const pairByteOffset = pairIndex * Uint32Array.BYTES_PER_ELEMENT +
            component * Uint16Array.BYTES_PER_ELEMENT;
          expect(pairScalarIndex).toBe(scalarIndex);
          expect(pairByteOffset).toBe(scalarByteOffset);
          expect(packed[pairScalarIndex]).toBe(packed[scalarIndex]);
          expect(
            packedBytes.slice(pairByteOffset, pairByteOffset + 2),
          ).toEqual(
            packedBytes.slice(scalarByteOffset, scalarByteOffset + 2),
          );
          expect(packed[scalarIndex]).toBe(
            native[
              (outputChannel * 7 + kernel) * inputChannels + inputChannel
            ],
          );
        }
      }
    }
  });

  it.each(ACE_OPT_0016_VAE_CONV1D_VARIANTS)(
    "%s preserves the final FP32 signed-zero boundary",
    (variant) => {
      const source = aceOpt0016VaeConv1dPackedKioMicrotileWgsl(
        k7Shape(128, 2, 17),
        variant,
        false,
        "float32",
      );
      expect(source).not.toContain("bias: array<f16>");
      expect(source).toContain("output: array<f32>");
      if (variant === "8x64") {
        expect(source).toContain("let initial_sum = vec2<f32>(0.0);");
        expect(source).toContain(
          "(bitcast<u32>(sum0.x) & 0x7fffffffu) == 0u",
        );
      } else {
        expect(source).toContain("let initial_sum: f32 = 0.0;");
        expect(source).toContain(
          "(bitcast<u32>(sum0) & 0x7fffffffu) == 0u",
        );
      }
    },
  );

  it.each([
    [256, 2_045, 363_266_048, [192_256, 184_576, 369_152]],
    [300, 2_404, 425_702_400, [225_304, 216_316, 432_608]],
    [512, 4_090, 726_532_096, [384_512, 369_152, 738_304]],
  ] as const)(
    "pins all 17 K7 operations and microtile workgroups at C%i",
    (frames, expectedQuanta, expectedOutputs, expectedWorkgroups) => {
      const graph = planAceVaeDecoder(frames);
      const cooperative = planAceVaeDecoderQuanta(graph);
      for (
        let variantIndex = 0;
        variantIndex < ACE_OPT_0016_VAE_CONV1D_VARIANTS.length;
        variantIndex += 1
      ) {
        const variant = ACE_OPT_0016_VAE_CONV1D_VARIANTS[variantIndex]!;
        const operationCoverage = new Map<number, number>();
        let quantumCount = 0;
        let logicalOutputCount = 0;
        let physicalWorkgroupCount = 0;
        for (const quantum of cooperative.quanta) {
          const operation = graph.operations[quantum.operationIndex]!;
          if (
            operation.kind !== "conv1d" ||
            operation.shape.kernelSize !== 7
          ) continue;
          expect(quantum.primitives).toHaveLength(1);
          const primitive = quantum.primitives[0]!;
          const outputStorage = operation.bias === undefined
            ? "float32"
            : "float16";
          const plan = planAceFp16VaeConv1d(operation.shape, outputStorage);
          const range = planAceOpt0016VaeConv1dPackedKioMicrotileRange(
            plan,
            variant,
            { base: primitive.outputBase, count: primitive.outputCount },
          );
          expect(range.count).toBe(quantum.logicalOutputCount);
          expect(range.outputRowCount * operation.shape.outputChannels)
            .toBe(range.count);
          expect(range.workgroupsX).toBeGreaterThan(0);
          expect(range.workgroupsY).toBeGreaterThan(0);
          quantumCount += 1;
          logicalOutputCount += range.count;
          physicalWorkgroupCount += range.workgroupsX * range.workgroupsY;
          operationCoverage.set(
            quantum.operationIndex,
            (operationCoverage.get(quantum.operationIndex) ?? 0) + range.count,
          );
        }
        expect(operationCoverage.size).toBe(17);
        for (const [operationIndex, covered] of operationCoverage) {
          const operation = graph.operations[operationIndex];
          expect(operation?.kind).toBe("conv1d");
          if (operation?.kind !== "conv1d") continue;
          expect(covered).toBe(
            operation.shape.batch * operation.shape.inputFrames *
              operation.shape.outputChannels,
          );
        }
        expect({
          variant,
          quantumCount,
          logicalOutputCount,
          physicalWorkgroupCount,
        }).toEqual({
          variant,
          quantumCount: expectedQuanta,
          logicalOutputCount: expectedOutputs,
          physicalWorkgroupCount: expectedWorkgroups[variantIndex],
        });
      }
    },
  );

  it("plans complete-row ranges with variant-specific physical geometry", () => {
    const plan = planAceFp16VaeConv1d({
      batch: 2,
      inputFrames: 65,
      inputChannels: 128,
      outputChannels: 128,
      kernelSize: 7,
      stride: 1,
      dilation: 9,
      padding: 27,
    }, "float16");
    expect(ACE_OPT_0016_VAE_CONV1D_VARIANTS.map((variant) => {
      const range = planAceOpt0016VaeConv1dPackedKioMicrotileRange(
        plan,
        variant,
        { base: 65 * 128, count: 33 * 128 },
      );
      return {
        variant,
        batch: range.batch,
        firstOutputTime: range.firstOutputTime,
        firstOutputRow: range.firstOutputRow,
        outputRowCount: range.outputRowCount,
        tileRows: range.tileRows,
        tileChannels: range.tileChannels,
        workgroupsX: range.workgroupsX,
        workgroupsY: range.workgroupsY,
      };
    })).toEqual([
      {
        variant: "8x64",
        batch: 1,
        firstOutputTime: 0,
        firstOutputRow: 65,
        outputRowCount: 33,
        tileRows: 16,
        tileChannels: 128,
        workgroupsX: 3,
        workgroupsY: 1,
      },
      {
        variant: "16x32",
        batch: 1,
        firstOutputTime: 0,
        firstOutputRow: 65,
        outputRowCount: 33,
        tileRows: 16,
        tileChannels: 128,
        workgroupsX: 3,
        workgroupsY: 1,
      },
      {
        variant: "8x32",
        batch: 1,
        firstOutputTime: 0,
        firstOutputRow: 65,
        outputRowCount: 33,
        tileRows: 8,
        tileChannels: 128,
        workgroupsX: 5,
        workgroupsY: 1,
      },
    ]);
  });

  it("owns explicit dispatches and caches each immutable variant separately", async () => {
    const device = fakeDevice();
    const owner = AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
      device,
      FIXED_32_CAPABILITY,
    );
    const shape = k7Shape(128, 256, 33);
    const plan = planAceFp16VaeConv1d(shape, "float16");
    const bindings = bindingsFor(shape, "float16", true);
    const control = fakeBuffer(1_024);
    const primaryFirst = await owner.createDispatch(
      "primary-first",
      "16x32",
      shape,
      bindings,
      "float16",
      rangeBinding(control, 256, 0, 17 * shape.outputChannels),
    );
    const primarySecond = await owner.createDispatch(
      "primary-second",
      "16x32",
      shape,
      bindings,
      "float16",
      rangeBinding(
        control,
        512,
        17 * shape.outputChannels,
        16 * shape.outputChannels,
      ),
    );
    const wide = await owner.createDispatch(
      "wide",
      "8x64",
      shape,
      bindings,
      "float16",
      rangeBinding(control, 768, 0, 17 * shape.outputChannels),
    );
    const small = await owner.createDispatch(
      "small",
      "8x32",
      shape,
      bindings,
      "float16",
      rangeBinding(control, 0, 0, 17 * shape.outputChannels),
    );

    expect(primaryFirst).toMatchObject({
      variant: "16x32",
      kernelId: ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_16X32_KERNEL_ID,
      microtilePlan: {
        rowsPerSubgroup: 16,
        outputsPerLane: 1,
        accumulatorsPerLane: 16,
        weightView: "scalar-f16",
      },
      outputRange: {
        firstOutputRow: 0,
        outputRowCount: 17,
        workgroupsX: 2,
        workgroupsY: 2,
      },
    });
    expect(primarySecond.outputRange).toMatchObject({
      firstOutputRow: 17,
      outputRowCount: 16,
      workgroupsX: 1,
      workgroupsY: 2,
    });
    expect(wide).toMatchObject({
      variant: "8x64",
      kernelId: ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_8X64_KERNEL_ID,
      outputRange: { workgroupsX: 3, workgroupsY: 1 },
    });
    expect(small).toMatchObject({
      variant: "8x32",
      kernelId: ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_8X32_KERNEL_ID,
      outputRange: { workgroupsX: 3, workgroupsY: 2 },
    });
    expect(device.createShaderModule).toHaveBeenCalledTimes(3);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(3);
    expect(device.createBindGroup).toHaveBeenCalledTimes(3);

    for (const call of device.createBindGroupLayout.mock.calls) {
      const layout = call[0] as GPUBindGroupLayoutDescriptor;
      expect(Array.from(layout.entries).map(({ buffer }) =>
        buffer?.minBindingSize
      )).toEqual([
        plan.inputBindingBytes,
        planAceOpt0014VaeConv1dPackedKioWeight(shape).packedBindingBytes,
        plan.biasBindingBytes,
        plan.outputBindingBytes,
        16,
      ]);
    }

    const pass = fakePass();
    primaryFirst.encode(pass);
    primarySecond.encode(pass);
    wide.encode(pass);
    small.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([[256], [512], [768], [0]]);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [2, 2, 1],
      [1, 2, 1],
      [3, 1, 1],
      [3, 2, 1],
    ]);

    owner.destroy();
    owner.destroy();
    expect(() => primaryFirst.encode(pass)).toThrow(/was destroyed/);
    await expect(owner.createDispatch(
      "after-destroy",
      "16x32",
      shape,
      bindings,
      "float16",
      rangeBinding(control, 0, 0, shape.outputChannels),
    )).rejects.toThrow(/was destroyed/);
  });

  it("admits only the final no-bias FP32 boundary", async () => {
    const device = fakeDevice();
    const owner = AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
      device,
      FIXED_32_CAPABILITY,
    );
    const shape = k7Shape(128, 2, 17);
    const plan = planAceFp16VaeConv1d(shape, "float32");
    const bindings = bindingsFor(shape, "float32", false);
    const range = rangeBinding(
      fakeBuffer(256),
      0,
      0,
      shape.inputFrames * shape.outputChannels,
    );
    const dispatch = await owner.createDispatch(
      "final",
      "16x32",
      shape,
      bindings,
      "float32",
      range,
    );
    expect(dispatch).toMatchObject({
      kernelId: ACE_OPT_0016_VAE_CONV1D_PACKED_KIO_16X32_KERNEL_ID,
      plan: { outputStorage: "float32", outputBindingBytes: 136 },
      outputRange: {
        channelBands: 1,
        rowBands: 4,
        tileRows: 64,
        tileChannels: 32,
        workgroupsX: 1,
        workgroupsY: 1,
      },
    });
    await expect(owner.createDispatch(
      "missing-bias",
      "16x32",
      shape,
      bindings,
      "float16",
      range,
    )).rejects.toThrow(/bias may be omitted only/);
    await expect(owner.createDispatch(
      "unexpected-bias",
      "16x32",
      shape,
      { ...bindings, bias: fakeBinding(plan.biasBindingBytes) },
      "float32",
      range,
    )).rejects.toThrow(/reserved for the final no-bias boundary/);
  });

  it("does not publish a dispatch when destroyed during compilation", async () => {
    let resolvePipeline!: (pipeline: GPUComputePipeline) => void;
    const pipeline = new Promise<GPUComputePipeline>((resolve) => {
      resolvePipeline = resolve;
    });
    const device = fakeDevice({ pipeline });
    const owner = AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
      device,
      FIXED_32_CAPABILITY,
    );
    const shape = k7Shape(128, 256, 17);
    const pending = owner.createDispatch(
      "in-flight",
      "16x32",
      shape,
      bindingsFor(shape, "float16", true),
      "float16",
      rangeBinding(fakeBuffer(256), 0, 0, shape.outputChannels),
    );
    await vi.waitFor(() => {
      expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    });
    owner.destroy();
    resolvePipeline({ label: "late-pipeline" } as GPUComputePipeline);
    await expect(pending).rejects.toThrow(/was destroyed/);
    expect(device.createBindGroup).not.toHaveBeenCalled();
  });

  it("fails closed on capabilities, variants, shapes, ranges, and limits", async () => {
    expect(() => AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
      fakeDevice({ shaderF16: false }),
      FIXED_32_CAPABILITY,
    )).toThrow(/requires WebGPU shader-f16/);
    expect(() => AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
      fakeDevice({ subgroups: false }),
      FIXED_32_CAPABILITY,
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
      fakeDevice({ maximumInvocations: 127 }),
      FIXED_32_CAPABILITY,
    )).toThrow(/requires WG128/);
    expect(() => planAceOpt0016VaeConv1dPackedKioMicrotile(
      { ...k7Shape(64, 64, 17), kernelSize: 1, padding: 0 },
      "16x32",
    )).toThrow(/support only K7/);
    expect(() => planAceOpt0016VaeConv1dPackedKioMicrotile(
      k7Shape(64, 65, 17),
      "16x32",
    )).toThrow(/output-channel pairs/);
    expect(() => planAceOpt0016VaeConv1dPackedKioMicrotile(
      k7Shape(64, 96, 17),
      "16x32",
    )).toThrow(/requires 1, 2, or 4 compile-time channel bands/);
    expect(() => planAceOpt0016VaeConv1dPackedKioMicrotile(
      k7Shape(64, 64, 17),
      "forged" as AceOpt0016VaeConv1dMicrotileVariant,
    )).toThrow(/unknown variant forged/);

    const plan = planAceFp16VaeConv1d(k7Shape(128, 256, 65), "float16");
    expect(() => planAceOpt0016VaeConv1dPackedKioMicrotileRange(
      plan,
      "16x32",
      { base: 1, count: 256 },
    )).toThrow(/complete in-bounds NLC rows/);
    const batchedPlan = planAceFp16VaeConv1d({
      ...k7Shape(128, 256, 65),
      batch: 2,
    }, "float16");
    expect(() => planAceOpt0016VaeConv1dPackedKioMicrotileRange(
      batchedPlan,
      "16x32",
      { base: 64 * 256, count: 2 * 256 },
    )).toThrow(/must not cross a batch boundary/);

    const shape = k7Shape(128, 256, 65);
    const owner = AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
      fakeDevice({ maximumDispatch: 1 }),
      FIXED_32_CAPABILITY,
    );
    await expect(owner.createDispatch(
      "dispatch-limit",
      "8x32",
      shape,
      bindingsFor(shape, "float16", true),
      "float16",
      rangeBinding(
        fakeBuffer(256),
        0,
        0,
        shape.inputFrames * shape.outputChannels,
      ),
    )).rejects.toThrow(/exceeds the device dispatch dimension/);

    const limited = AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
      fakeDevice({ maximumStorageBinding: 1 }),
      FIXED_32_CAPABILITY,
    );
    await expect(limited.createDispatch(
      "buffer-limit",
      "16x32",
      shape,
      bindingsFor(shape, "float16", true),
      "float16",
      rangeBinding(fakeBuffer(256), 0, 0, shape.outputChannels),
    )).rejects.toThrow(/exceeds the device buffer limits/);
  });

  it("rejects malformed and overlapping bindings before compilation", async () => {
    const shape = k7Shape(128, 256, 17);
    const plan = planAceFp16VaeConv1d(shape, "float16");
    const valid = bindingsFor(shape, "float16", true);
    const range = rangeBinding(fakeBuffer(256), 0, 0, shape.outputChannels);

    const tooSmallDevice = fakeDevice();
    const tooSmall =
      AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
        tooSmallDevice,
        FIXED_32_CAPABILITY,
      );
    await expect(tooSmall.createDispatch(
      "too-small",
      "16x32",
      shape,
      { ...valid, input: fakeBinding(plan.inputBindingBytes - 4) },
      "float16",
      range,
    )).rejects.toThrow(/does not expose an aligned/);
    expect(tooSmallDevice.createShaderModule).not.toHaveBeenCalled();

    const misalignedDevice = fakeDevice();
    const misaligned =
      AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
        misalignedDevice,
        FIXED_32_CAPABILITY,
      );
    await expect(misaligned.createDispatch(
      "misaligned",
      "16x32",
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

    const controlDevice = fakeDevice();
    const badControl =
      AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
        controlDevice,
        FIXED_32_CAPABILITY,
      );
    await expect(badControl.createDispatch(
      "bad-control",
      "16x32",
      shape,
      valid,
      "float16",
      rangeBinding(fakeBuffer(256), 4, 0, shape.outputChannels),
    )).rejects.toThrow(/aligned 16-byte immutable record/);
    expect(controlDevice.createShaderModule).not.toHaveBeenCalled();

    const aliasDevice = fakeDevice();
    const aliasOwner =
      AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
        aliasDevice,
        FIXED_32_CAPABILITY,
      );
    const shared = fakeBuffer(Math.max(
      plan.inputBindingBytes,
      plan.outputBindingBytes,
    ));
    await expect(aliasOwner.createDispatch(
      "aliased",
      "16x32",
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

  it("evicts a failed shader compilation instead of poisoning the cache", async () => {
    const device = fakeDevice({ compilationError: "typed f16 rejected" });
    const owner = AceOpt0016VaeConv1dPackedKioMicrotileSubgroupKernel.create(
      device,
      FIXED_32_CAPABILITY,
    );
    const shape = k7Shape(128, 256, 17);
    const bindings = bindingsFor(shape, "float16", true);
    const range = rangeBinding(fakeBuffer(256), 0, 0, shape.outputChannels);
    await expect(owner.createDispatch(
      "first",
      "16x32",
      shape,
      bindings,
      "float16",
      range,
    )).rejects.toThrow(/typed f16 rejected/);
    await Promise.resolve();
    await expect(owner.createDispatch(
      "retry",
      "16x32",
      shape,
      bindings,
      "float16",
      range,
    )).rejects.toThrow(/typed f16 rejected/);
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).not.toHaveBeenCalled();
    expect(device.createBindGroup).not.toHaveBeenCalled();
  });
});

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const FIXED_32_CAPABILITY = Object.freeze({
  subgroupMinSize: 32,
  subgroupMaxSize: 32,
});

function variantSummary(variant: AceOpt0016VaeConv1dMicrotileVariant) {
  const base = planAceOpt0016VaeConv1dPackedKioMicrotile(
    k7Shape(128, 256, 17),
    variant,
  );
  return {
    variant: base.variant,
    kernelId: base.kernelId,
    rowsPerSubgroup: base.rowsPerSubgroup,
    outputsPerLane: base.outputsPerLane,
    channelsPerSubgroup: base.channelsPerSubgroup,
    accumulatorsPerLane: base.accumulatorsPerLane,
    weightView: base.weightView,
    geometries: [2, 64, 128, 256, 1_024].map((outputChannels) => {
      const plan = planAceOpt0016VaeConv1dPackedKioMicrotile(
        k7Shape(128, outputChannels, 17),
        variant,
      );
      return [
        outputChannels,
        plan.channelBands,
        plan.rowBands,
        plan.tileRows,
        plan.tileChannels,
      ];
    }),
  };
}

function k7Shape(
  inputChannels: number,
  outputChannels: number,
  inputFrames: number,
): AceVaeConv1dShape {
  return {
    batch: 1,
    inputFrames,
    inputChannels,
    outputChannels,
    kernelSize: 7,
    stride: 1,
    dilation: 1,
    padding: 3,
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

function bindingsFor(
  shape: AceVaeConv1dShape,
  outputStorage: AceFp16VaeConv1dOutputStorage,
  hasBias: boolean,
) {
  const plan = planAceFp16VaeConv1d(shape, outputStorage);
  return {
    input: fakeBinding(plan.inputBindingBytes),
    packedWeight: fakeBinding(
      planAceOpt0014VaeConv1dPackedKioWeight(shape).packedBindingBytes,
    ),
    ...(hasBias ? { bias: fakeBinding(plan.biasBindingBytes) } : {}),
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
  readonly subgroups?: boolean;
  readonly maximumInvocations?: number;
  readonly maximumWorkgroupSizeX?: number;
  readonly maximumDispatch?: number;
  readonly maximumStorageBinding?: number;
  readonly maximumBuffer?: number;
  readonly uniformAlignment?: number;
  readonly storageAlignment?: number;
  readonly compilationError?: string;
  readonly pipeline?: Promise<GPUComputePipeline>;
} = {}): FakeDevice {
  return {
    features: new Set([
      ...(options.shaderF16 === false ? [] : ["shader-f16"]),
      ...(options.subgroups === false ? [] : ["subgroups"]),
    ]),
    limits: {
      maxComputeInvocationsPerWorkgroup: options.maximumInvocations ?? 128,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSizeX ?? 128,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupStorageSize: 32_768,
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
