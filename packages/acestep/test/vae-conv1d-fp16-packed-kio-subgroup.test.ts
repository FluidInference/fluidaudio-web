import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0014_VAE_CONV1D_CHANNELS_PER_SUBGROUP,
  ACE_OPT_0014_VAE_CONV1D_OUTPUTS_PER_LANE,
  ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_KERNEL_ID,
  ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_REPACK_KERNEL_ID,
  ACE_OPT_0014_VAE_CONV1D_MAX_CHANNEL_BANDS,
  ACE_OPT_0014_VAE_CONV1D_REPACK_WORKGROUP_SIZE,
  ACE_OPT_0014_VAE_CONV1D_ROWS_PER_SUBGROUP,
  ACE_OPT_0014_VAE_CONV1D_SUBGROUP_SIZE,
  ACE_OPT_0014_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP,
  ACE_OPT_0014_VAE_CONV1D_WORKGROUP_SIZE,
  AceOpt0014VaeConv1dPackedKioSubgroupKernel,
  aceOpt0014VaeConv1dPackedKioRepackWgsl,
  aceOpt0014VaeConv1dPackedKioWgsl,
  planAceOpt0014VaeConv1dPackedKioRange,
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

describe("OPT-0014 packed-KIO fixed32 VAE K7 Conv1D", () => {
  it("pins 16x64 subgroup tiles with reciprocal compile-time row/channel bands", () => {
    expect(ACE_OPT_0014_VAE_CONV1D_SUBGROUP_SIZE).toBe(32);
    expect(ACE_OPT_0014_VAE_CONV1D_WORKGROUP_SIZE).toBe(128);
    expect(ACE_OPT_0014_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP).toBe(4);
    expect(ACE_OPT_0014_VAE_CONV1D_ROWS_PER_SUBGROUP).toBe(16);
    expect(ACE_OPT_0014_VAE_CONV1D_OUTPUTS_PER_LANE).toBe(2);
    expect(ACE_OPT_0014_VAE_CONV1D_CHANNELS_PER_SUBGROUP).toBe(64);
    expect(ACE_OPT_0014_VAE_CONV1D_MAX_CHANNEL_BANDS).toBe(4);
    expect(ACE_OPT_0014_VAE_CONV1D_REPACK_WORKGROUP_SIZE).toBe(256);
    expect(
      ACE_OPT_0014_VAE_CONV1D_SUBGROUP_SIZE *
        ACE_OPT_0014_VAE_CONV1D_OUTPUTS_PER_LANE,
    ).toBe(ACE_OPT_0014_VAE_CONV1D_CHANNELS_PER_SUBGROUP);
    expect(
      ACE_OPT_0014_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP *
        ACE_OPT_0014_VAE_CONV1D_CHANNELS_PER_SUBGROUP,
    ).toBe(256);

    expect([2, 64, 128, 256, 1_024].map((outputChannels) => {
      const plan = planAceOpt0014VaeConv1dPackedKioWeight(
        k7Shape(128, outputChannels, 17),
      );
      return {
        outputChannels,
        channelBands: plan.channelBands,
        rowBands: plan.rowBands,
        tileRows: plan.tileRows,
        tileChannels: plan.tileChannels,
        occupiedSubgroups: plan.channelBands * plan.rowBands,
      };
    })).toEqual([
      {
        outputChannels: 2,
        channelBands: 1,
        rowBands: 4,
        tileRows: 64,
        tileChannels: 64,
        occupiedSubgroups: 4,
      },
      {
        outputChannels: 64,
        channelBands: 1,
        rowBands: 4,
        tileRows: 64,
        tileChannels: 64,
        occupiedSubgroups: 4,
      },
      {
        outputChannels: 128,
        channelBands: 2,
        rowBands: 2,
        tileRows: 32,
        tileChannels: 128,
        occupiedSubgroups: 4,
      },
      {
        outputChannels: 256,
        channelBands: 4,
        rowBands: 1,
        tileRows: 16,
        tileChannels: 256,
        occupiedSubgroups: 4,
      },
      {
        outputChannels: 1_024,
        channelBands: 4,
        rowBands: 1,
        tileRows: 16,
        tileChannels: 256,
        occupiedSubgroups: 4,
      },
    ]);

    for (const outputChannels of [2, 64, 128, 256, 1_024]) {
      const plan = planAceOpt0014VaeConv1dPackedKioWeight(
        k7Shape(128, outputChannels, 17),
      );
      const activeChannels = Math.min(outputChannels, plan.tileChannels);
      const owners = new Uint8Array(plan.tileRows * activeChannels);
      for (
        let subgroup = 0;
        subgroup < ACE_OPT_0014_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP;
        subgroup += 1
      ) {
        const channelBand = subgroup % plan.channelBands;
        const rowBand = Math.floor(subgroup / plan.channelBands);
        for (
          let row = 0;
          row < ACE_OPT_0014_VAE_CONV1D_ROWS_PER_SUBGROUP;
          row += 1
        ) {
          for (
            let lane = 0;
            lane < ACE_OPT_0014_VAE_CONV1D_SUBGROUP_SIZE;
            lane += 1
          ) {
            for (
              let component = 0;
              component < ACE_OPT_0014_VAE_CONV1D_OUTPUTS_PER_LANE;
              component += 1
            ) {
              const outputRow =
                rowBand * ACE_OPT_0014_VAE_CONV1D_ROWS_PER_SUBGROUP + row;
              const outputChannel =
                channelBand *
                  ACE_OPT_0014_VAE_CONV1D_CHANNELS_PER_SUBGROUP +
                lane * ACE_OPT_0014_VAE_CONV1D_OUTPUTS_PER_LANE + component;
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
        `Cout=${outputChannels}`,
      ).toBe(true);
    }
  });

  it("plans a same-size raw-U16 O-K-I to K-I-O pair buffer", () => {
    const shape = k7Shape(3, 4, 17);
    const plan = planAceOpt0014VaeConv1dPackedKioWeight(shape);
    expect(plan).toEqual({
      inputChannels: 3,
      outputChannels: 4,
      kernelSize: 7,
      outputChannelPairs: 2,
      channelBands: 1,
      rowBands: 4,
      tileRows: 64,
      tileChannels: 64,
      packedWordCount: 42,
      nativeStorageBytes: 168,
      nativeBindingBytes: 168,
      packedStorageBytes: 168,
      packedBindingBytes: 168,
      repackWorkgroupSize: 256,
      repackWorkgroups: 1,
    });

    const source = Uint16Array.from(
      { length: 4 * 7 * 3 },
      (_, index) => [0x0000, 0x8000, 0x7e01, 0x7c00, 0xfc00, index][
        index % 6
      ]!,
    );
    const packed = packKioReference(source, 3, 4);
    expect(packed).toHaveLength(source.length);
    for (let kernel = 0; kernel < 7; kernel += 1) {
      for (let inputChannel = 0; inputChannel < 3; inputChannel += 1) {
        for (let outputChannel = 0; outputChannel < 4; outputChannel += 1) {
          expect(
            packed[(kernel * 3 + inputChannel) * 4 + outputChannel],
          ).toBe(
            source[(outputChannel * 7 + kernel) * 3 + inputChannel],
          );
        }
      }
    }
    expect([...packed].sort((a, b) => a - b)).toEqual(
      [...source].sort((a, b) => a - b),
    );
  });

  it("emits a conversion-free raw-U16 GPU repack", () => {
    const source = aceOpt0014VaeConv1dPackedKioRepackWgsl(
      k7Shape(128, 256, 17),
    );
    expect(source).toContain(
      `// kernel-id: ${ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_REPACK_KERNEL_ID}`,
    );
    expect(source).toContain("native_weight: array<u32>");
    expect(source).toContain("packed_weight: array<u32>");
    expect(source).toContain("(word >> shift) & 0xffffu");
    expect(source).toContain("packed_weight[packed_word] = low | (high << 16u)");
    expect(source).toContain(
      "(output_channel * 7u + kernel) * INPUT_CHANNELS + input_channel",
    );
    expect(source).not.toContain("enable f16");
    expect(source).not.toMatch(/\bf16\s*\(/);
    expect(source).not.toMatch(/:\s*f16\b/);
  });

  it("emits contiguous packed pairs and exact K-then-Cin FP32 arithmetic", () => {
    const source = aceOpt0014VaeConv1dPackedKioWgsl(
      k7Shape(128, 256, 17),
      true,
      "float16",
    );
    expect(source).toContain(
      `// kernel-id: ${ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_KERNEL_ID}`,
    );
    expect(source).toContain("enable f16;");
    expect(source).toContain("enable subgroups;");
    expect(source).toContain("@compute @workgroup_size(128, 1, 1)");
    expect(source).toContain("let channel_band = subgroup % CHANNEL_BANDS");
    expect(source).toContain("let row_band = subgroup / CHANNEL_BANDS");
    expect(source).toContain("channel_band * 64u");
    expect(source).toContain("row_band * 16u");
    expect(source).toContain("subgroup_lane * 2u");
    expect(source).toContain(
      "weight_operands = unpack2x16float(packed_weight[packed_index])",
    );
    expect(source).toContain(
      "(kernel * INPUT_CHANNELS + input_channel) *",
    );
    for (let row = 0; row < 16; row += 1) {
      expect(source).toContain(`var sum${row}: vec2<f32> = initial_sum;`);
      expect(source).toContain(
        `let input_operand${row} = subgroupBroadcast(lane_input, ${row}u);`,
      );
      expect(source).toContain(
        `sum${row} = sum${row} +\n          vec2<f32>(input_operand${row}) * weight_operands;`,
      );
    }
    expect(source).not.toContain("var<workgroup>");
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toMatch(/\bdot\s*\(/);
    const kernel = source.indexOf("var kernel = 0u;");
    const inputChannel = source.indexOf("var input_channel = 0u;", kernel);
    const add = source.indexOf("sum0 = sum0 +", inputChannel);
    expect(kernel).toBeLessThan(inputChannel);
    expect(inputChannel).toBeLessThan(add);
  });

  it("preserves the final FP32 signed-zero canonicalization", () => {
    const source = aceOpt0014VaeConv1dPackedKioWgsl(
      k7Shape(128, 2, 17),
      false,
      "float32",
    );
    expect(source).not.toContain("bias: array<f16>");
    expect(source).toContain("let initial_sum = vec2<f32>(0.0);");
    expect(source).toContain(
      "(bitcast<u32>(sum0.x) & 0x7fffffffu) == 0u",
    );
    expect(source).toContain("output: array<f32>");
  });

  it.each([256, 512])(
    "covers every production K7 operation and quantum at %i frames",
    (frames) => {
      const graph = planAceVaeDecoder(frames);
      const cooperative = planAceVaeDecoderQuanta(graph);
      const operationIndices = new Set<number>();
      let quantumCount = 0;
      let logicalOutputCount = 0;
      let physicalWorkgroupCount = 0;
      for (const quantum of cooperative.quanta) {
        const operation = graph.operations[quantum.operationIndex]!;
        if (
          operation.kind !== "conv1d" ||
          operation.shape.kernelSize !== 7
        ) continue;
        const outputStorage = operation.bias === undefined
          ? "float32"
          : "float16";
        const plan = planAceFp16VaeConv1d(operation.shape, outputStorage);
        const primitive = quantum.primitives[0]!;
        const range = planAceOpt0014VaeConv1dPackedKioRange(plan, {
          base: primitive.outputBase,
          count: primitive.outputCount,
        });
        const weight = planAceOpt0014VaeConv1dPackedKioWeight(
          operation.shape,
        );
        expect(weight.outputChannels % 2).toBe(0);
        expect(range.count).toBe(quantum.logicalOutputCount);
        quantumCount += 1;
        logicalOutputCount += range.count;
        physicalWorkgroupCount += range.workgroupsX * range.workgroupsY;
        operationIndices.add(quantum.operationIndex);
      }
      expect(operationIndices.size).toBe(17);
      expect({
        quantumCount,
        logicalOutputCount,
        physicalWorkgroupCount,
      }).toEqual(frames === 256
        ? {
            quantumCount: 2_045,
            logicalOutputCount: 363_266_048,
            physicalWorkgroupCount: 96_128,
          }
        : {
            quantumCount: 4_090,
            logicalOutputCount: 726_532_096,
            physicalWorkgroupCount: 192_256,
          });
    },
  );

  it("is a benchmark-ready two-pass owner with explicit dispatch identities", async () => {
    const device = fakeDevice();
    const owner = AceOpt0014VaeConv1dPackedKioSubgroupKernel.create(
      device,
      FIXED_32_CAPABILITY,
    );
    const shape = k7Shape(128, 256, 33);
    const convPlan = planAceFp16VaeConv1d(shape, "float16");
    const weightPlan = planAceOpt0014VaeConv1dPackedKioWeight(shape);
    const nativeWeight = fakeBinding(weightPlan.nativeBindingBytes);
    const packedWeight = fakeBinding(weightPlan.packedBindingBytes);
    const repack = await owner.createRepackDispatch("repack", shape, {
      nativeWeight,
      packedWeight,
    });
    const bindings = {
      input: fakeBinding(convPlan.inputBindingBytes),
      packedWeight,
      bias: fakeBinding(convPlan.biasBindingBytes),
      output: fakeBinding(convPlan.outputBindingBytes),
    };
    const control = fakeBuffer(1_024);
    const first = await owner.createDispatch(
      "first",
      shape,
      bindings,
      "float16",
      rangeBinding(control, 256, 0, 17 * shape.outputChannels),
    );
    const second = await owner.createDispatch(
      "second",
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

    expect(repack).toMatchObject({
      kernelId: ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_REPACK_KERNEL_ID,
      plan: weightPlan,
    });
    expect(first).toMatchObject({
      kernelId: ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_KERNEL_ID,
      packedWeightPlan: weightPlan,
      outputRange: {
        firstOutputRow: 0,
        outputRowCount: 17,
        workgroupsX: 2,
        workgroupsY: 1,
      },
    });
    expect(second.outputRange).toMatchObject({
      firstOutputRow: 17,
      outputRowCount: 16,
      workgroupsX: 1,
      workgroupsY: 1,
    });
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createBindGroup).toHaveBeenCalledTimes(2);

    const repackLayout = device.createBindGroupLayout.mock.calls[0]?.[0] as
      GPUBindGroupLayoutDescriptor;
    expect(Array.from(repackLayout.entries).map(({ buffer }) =>
      buffer?.minBindingSize
    )).toEqual([
      weightPlan.nativeBindingBytes,
      weightPlan.packedBindingBytes,
    ]);
    const convLayout = device.createBindGroupLayout.mock.calls[1]?.[0] as
      GPUBindGroupLayoutDescriptor;
    expect(Array.from(convLayout.entries).map(({ buffer }) =>
      buffer?.minBindingSize
    )).toEqual([
      convPlan.inputBindingBytes,
      weightPlan.packedBindingBytes,
      convPlan.biasBindingBytes,
      convPlan.outputBindingBytes,
      16,
    ]);

    const pass = fakePass();
    repack.encode(pass);
    first.encode(pass);
    second.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([undefined, [256], [512]]);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [weightPlan.repackWorkgroups, 1, 1],
      [2, 1, 1],
      [1, 1, 1],
    ]);

    owner.destroy();
    owner.destroy();
    expect(() => repack.encode(pass)).toThrow(/was destroyed/);
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
  });

  it("admits the final no-bias FP32 boundary for U32 output comparison", async () => {
    const device = fakeDevice();
    const owner = AceOpt0014VaeConv1dPackedKioSubgroupKernel.create(
      device,
      FIXED_32_CAPABILITY,
    );
    const shape = k7Shape(128, 2, 17);
    const plan = planAceFp16VaeConv1d(shape, "float32");
    const packedWeight = fakeBinding(
      planAceOpt0014VaeConv1dPackedKioWeight(shape).packedBindingBytes,
    );
    const bindings = {
      input: fakeBinding(plan.inputBindingBytes),
      packedWeight,
      output: fakeBinding(plan.outputBindingBytes),
    };
    const dispatch = await owner.createDispatch(
      "final-fp32",
      shape,
      bindings,
      "float32",
      rangeBinding(
        fakeBuffer(256),
        0,
        0,
        shape.inputFrames * shape.outputChannels,
      ),
    );
    expect(dispatch).toMatchObject({
      kernelId: ACE_OPT_0014_VAE_CONV1D_PACKED_KIO_KERNEL_ID,
      plan: { outputStorage: "float32", outputBindingBytes: 136 },
      outputRange: {
        channelBands: 1,
        rowBands: 4,
        tileRows: 64,
        tileChannels: 64,
        workgroupsX: 1,
        workgroupsY: 1,
      },
    });
    await expect(owner.createDispatch(
      "missing-bias",
      shape,
      bindings,
      "float16",
      rangeBinding(
        fakeBuffer(256),
        0,
        0,
        shape.inputFrames * shape.outputChannels,
      ),
    )).rejects.toThrow(/bias may be omitted only/);
    await expect(owner.createDispatch(
      "unexpected-bias",
      shape,
      { ...bindings, bias: fakeBinding(plan.biasBindingBytes) },
      "float32",
      rangeBinding(
        fakeBuffer(256),
        0,
        0,
        shape.inputFrames * shape.outputChannels,
      ),
    )).rejects.toThrow(/reserved for the final no-bias boundary/);
  });

  it("fails closed on capabilities, unsupported shapes, and repack aliasing", async () => {
    expect(() => AceOpt0014VaeConv1dPackedKioSubgroupKernel.create(
      fakeDevice({ shaderF16: false }),
      FIXED_32_CAPABILITY,
    )).toThrow(/requires WebGPU shader-f16/);
    expect(() => AceOpt0014VaeConv1dPackedKioSubgroupKernel.create(
      fakeDevice({ subgroups: false }),
      FIXED_32_CAPABILITY,
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceOpt0014VaeConv1dPackedKioSubgroupKernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceOpt0014VaeConv1dPackedKioSubgroupKernel.create(
      fakeDevice({ maximumInvocations: 255 }),
      FIXED_32_CAPABILITY,
    )).toThrow(/requires WG256/);
    expect(() => planAceOpt0014VaeConv1dPackedKioWeight({
      ...k7Shape(64, 64, 17),
      kernelSize: 1,
      padding: 0,
    })).toThrow(/supports only K7/);
    expect(() => planAceOpt0014VaeConv1dPackedKioWeight(
      k7Shape(64, 129, 17),
    )).toThrow(/requires output-channel pairs/);

    const device = fakeDevice();
    const owner = AceOpt0014VaeConv1dPackedKioSubgroupKernel.create(
      device,
      FIXED_32_CAPABILITY,
    );
    const shape = k7Shape(64, 64, 17);
    const bytes = planAceOpt0014VaeConv1dPackedKioWeight(shape)
      .nativeBindingBytes;
    const aliased = fakeBinding(bytes);
    await expect(owner.createRepackDispatch("aliased", shape, {
      nativeWeight: aliased,
      packedWeight: aliased,
    })).rejects.toThrow(/must not overlap/);
    expect(device.createShaderModule).not.toHaveBeenCalled();
  });
});

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const FIXED_32_CAPABILITY = Object.freeze({
  subgroupMinSize: 32,
  subgroupMaxSize: 32,
});

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
} = {}): FakeDevice {
  return {
    features: new Set([
      ...(options.shaderF16 === false ? [] : ["shader-f16"]),
      ...(options.subgroups === false ? [] : ["subgroups"]),
    ]),
    limits: {
      maxComputeInvocationsPerWorkgroup: options.maximumInvocations ?? 256,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSizeX ?? 256,
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
      getCompilationInfo: vi.fn(async () => ({ messages: [] })),
    })),
    createBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    createPipelineLayout: vi.fn(() => ({ label: "pipeline-layout" })),
    createComputePipelineAsync: vi.fn(async () => ({ label: "pipeline" })),
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
