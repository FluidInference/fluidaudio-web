import { describe, expect, it, vi } from "vitest";

import {
  ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
  ACE_FP16_VAE_CONV1D_SUBGROUP_OUTPUTS_PER_LANE,
  ACE_FP16_VAE_CONV1D_SUBGROUP_ROWS,
  ACE_FP16_VAE_CONV1D_SUBGROUP_SIZE,
  ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_CHANNELS,
  ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_ROWS,
  ACE_FP16_VAE_CONV1D_SUBGROUP_WORKGROUP_SIZE,
  ACE_FP16_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP,
  AceFp16VaeConv1dSubgroupKernel,
  aceFp16VaeConv1dSubgroupWgsl,
  planAceFp16VaeConv1dSubgroupRange,
} from "../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import {
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

describe("fixed-32 subgroup FP16 VAE K7 Conv1D", () => {
  it("pins one barrier-free 8x128 implicit-im2col subgroup tile", () => {
    expect(ACE_FP16_VAE_CONV1D_SUBGROUP_SIZE).toBe(32);
    expect(ACE_FP16_VAE_CONV1D_SUBGROUP_WORKGROUP_SIZE).toBe(128);
    expect(ACE_FP16_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP).toBe(4);
    expect(ACE_FP16_VAE_CONV1D_SUBGROUP_ROWS).toBe(8);
    expect(ACE_FP16_VAE_CONV1D_SUBGROUP_OUTPUTS_PER_LANE).toBe(4);
    expect(ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_ROWS).toBe(32);
    expect(ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_CHANNELS).toBe(128);
    expect(
      ACE_FP16_VAE_CONV1D_SUBGROUP_SIZE *
        ACE_FP16_VAE_CONV1D_SUBGROUP_OUTPUTS_PER_LANE,
    ).toBe(ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_CHANNELS);
    expect(
      ACE_FP16_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP *
        ACE_FP16_VAE_CONV1D_SUBGROUP_ROWS,
    ).toBe(ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_ROWS);
  });

  it.each([256, 512])(
    "covers every exact K7 operation and cooperative quantum at %i frames",
    (frames) => {
      const graph = planAceVaeDecoder(frames);
      const cooperative = planAceVaeDecoderQuanta(graph);
      const operations = graph.operations.filter(
        (operation): operation is AceVaeDecoderConvOperation =>
          operation.kind === "conv1d" && operation.shape.kernelSize === 7,
      );
      expect(operations).toHaveLength(17);

      let quantumCount = 0;
      let logicalOutputCount = 0;
      let physicalWorkgroupCount = 0;
      let portablePhysicalWorkgroupCount = 0;
      let k7MultiplyAccumulates = 0;
      const operationCoverage = new Map<number, number>();
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
        const range = planAceFp16VaeConv1dSubgroupRange(plan, {
          base: primitive.outputBase,
          count: primitive.outputCount,
        });
        const portableRange = planAceFp16VaeConv1dRange(plan, {
          base: primitive.outputBase,
          count: primitive.outputCount,
        });
        expect(range.count).toBe(quantum.logicalOutputCount);
        expect(range.outputRowCount * operation.shape.outputChannels)
          .toBe(range.count);
        expect(range.workgroupsX).toBeGreaterThan(0);
        expect(range.workgroupsY).toBeGreaterThan(0);
        quantumCount += 1;
        logicalOutputCount += range.count;
        physicalWorkgroupCount += range.workgroupsX * range.workgroupsY;
        portablePhysicalWorkgroupCount +=
          portableRange.workgroupsX * portableRange.workgroupsY;
        k7MultiplyAccumulates +=
          quantum.estimatedMaximumMultiplyAccumulates;
        operationCoverage.set(
          quantum.operationIndex,
          (operationCoverage.get(quantum.operationIndex) ?? 0) + range.count,
        );
      }

      expect(operationCoverage).toHaveLength(17);
      for (const [operationIndex, covered] of operationCoverage) {
        const operation = graph.operations[operationIndex];
        expect(operation?.kind).toBe("conv1d");
        if (operation?.kind !== "conv1d") continue;
        expect(covered).toBe(
          operation.shape.batch * operation.shape.inputFrames *
            operation.shape.outputChannels,
        );
      }
      const graphMultiplyAccumulates = cooperative.quanta.reduce(
        (total, quantum) =>
          total + quantum.estimatedMaximumMultiplyAccumulates,
        0,
      );
      expect({
        frames,
        quantumCount,
        logicalOutputCount,
        physicalWorkgroupCount,
        portablePhysicalWorkgroupCount,
        k7MultiplyAccumulates,
        graphMultiplyAccumulates,
      })
        .toEqual(frames === 256
          ? {
              frames: 256,
              quantumCount: 2_045,
              logicalOutputCount: 363_266_048,
              physicalWorkgroupCount: 103_808,
              portablePhysicalWorkgroupCount: 2_861_056,
              k7MultiplyAccumulates: 480_272_973_824,
              graphMultiplyAccumulates: 623_885_942_784,
            }
          : {
              frames: 512,
              quantumCount: 4_090,
              logicalOutputCount: 726_532_096,
              physicalWorkgroupCount: 207_616,
              portablePhysicalWorkgroupCount: 5_722_112,
              k7MultiplyAccumulates: 960_545_947_648,
              graphMultiplyAccumulates: 1_247_771_885_568,
            });
    },
  );

  it("plans exact complete-row ranges in 32-row by 128-channel tiles", () => {
    const plan = planAceFp16VaeConv1d({
      batch: 2,
      inputFrames: 65,
      inputChannels: 65,
      outputChannels: 129,
      kernelSize: 7,
      stride: 1,
      dilation: 9,
      padding: 27,
    }, "float16");
    expect(planAceFp16VaeConv1dSubgroupRange(plan, {
      base: 65 * 129,
      count: 33 * 129,
    })).toEqual({
      base: 8_385,
      count: 4_257,
      batch: 1,
      firstOutputTime: 0,
      firstOutputRow: 65,
      outputRowCount: 33,
      workgroupsX: 2,
      workgroupsY: 2,
    });
  });

  it("assigns every row/channel in a partial tile exactly once", () => {
    const rowCount = 33;
    const channelCount = 129;
    const counts = new Uint8Array(rowCount * channelCount);
    const workgroupsX = Math.ceil(
      rowCount / ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_ROWS,
    );
    const workgroupsY = Math.ceil(
      channelCount / ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_CHANNELS,
    );
    for (let groupX = 0; groupX < workgroupsX; groupX += 1) {
      for (let groupY = 0; groupY < workgroupsY; groupY += 1) {
        for (
          let subgroup = 0;
          subgroup < ACE_FP16_VAE_CONV1D_SUBGROUPS_PER_WORKGROUP;
          subgroup += 1
        ) {
          for (
            let lane = 0;
            lane < ACE_FP16_VAE_CONV1D_SUBGROUP_SIZE;
            lane += 1
          ) {
            for (
              let rowInSubgroup = 0;
              rowInSubgroup < ACE_FP16_VAE_CONV1D_SUBGROUP_ROWS;
              rowInSubgroup += 1
            ) {
              const row =
                groupX * ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_ROWS +
                subgroup * ACE_FP16_VAE_CONV1D_SUBGROUP_ROWS +
                rowInSubgroup;
              for (
                let component = 0;
                component < ACE_FP16_VAE_CONV1D_SUBGROUP_OUTPUTS_PER_LANE;
                component += 1
              ) {
                const channel =
                  groupY * ACE_FP16_VAE_CONV1D_SUBGROUP_TILE_CHANNELS +
                  lane * ACE_FP16_VAE_CONV1D_SUBGROUP_OUTPUTS_PER_LANE +
                  component;
                if (row < rowCount && channel < channelCount) {
                  const index = row * channelCount + channel;
                  counts[index] = counts[index]! + 1;
                }
              }
            }
          }
        }
      }
    }
    expect([...counts].every((count) => count === 1)).toBe(true);
  });

  it("emits direct native weights, subgroup broadcasts, and source-order FP32 math", () => {
    const source = aceFp16VaeConv1dSubgroupWgsl(
      k7Shape(65, 129, 17),
      true,
      "float16",
    );
    expect(source).toContain(
      `// kernel-id: ${ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID}`,
    );
    expect(source).toContain("enable f16;");
    expect(source).toContain("enable subgroups;");
    expect(source).toContain("@compute @workgroup_size(\n  128,\n  1,\n  1,");
    expect(source).toContain("@builtin(subgroup_invocation_id) subgroup_lane");
    expect(source).toContain("@builtin(subgroup_size) subgroup_size");
    expect(source).toContain("if (subgroup_size == 32u)");
    expect(source).toContain("for (var kernel = 0u; kernel < 7u;");
    expect(source).toContain("var input_channel = 0u;");
    expect(source).toContain(
      "(output_channel_base * 7u + kernel) * INPUT_CHANNELS +",
    );
    for (let row = 0; row < 8; row += 1) {
      expect(source).toContain(
        `let input_operand${row} = subgroupBroadcast(lane_input, ${row}u);`,
      );
      expect(source).toContain(
        `sum${row} = sum${row} +\n          vec4<f32>(input_operand${row}) * weight_operands;`,
      );
    }
    expect(source).not.toContain("var<workgroup>");
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toMatch(/\bfma\s*\(/);
    expect(source).not.toMatch(/\bdot\s*\(/);
    const kernel = source.indexOf("var kernel = 0u;");
    const channel = source.indexOf("var input_channel = 0u;", kernel);
    const add = source.indexOf("sum0 = sum0 +", channel);
    expect(kernel).toBeLessThan(channel);
    expect(channel).toBeLessThan(add);
  });

  it("preserves final FP32 signed-zero canonicalization", () => {
    const source = aceFp16VaeConv1dSubgroupWgsl(
      k7Shape(64, 2, 17),
      false,
      "float32",
    );
    expect(source).not.toContain("bias: array<f16>");
    expect(source).toContain("let initial_sum = vec4<f32>(0.0);");
    expect(source).toContain(
      "(bitcast<u32>(sum0.x) & 0x7fffffffu) == 0u",
    );
    expect(source).toContain("output: array<f32>;");
  });

  it("is a benchmark-ready owner with explicit kernel identity and dynamic controls", async () => {
    const device = fakeDevice();
    const kernel = AceFp16VaeConv1dSubgroupKernel.create(
      device,
      FIXED_32_CAPABILITY,
    );
    const shape = k7Shape(1_024, 1_024, 256);
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
      rangeBinding(
        control,
        512,
        32 * shape.outputChannels,
        16 * shape.outputChannels,
      ),
    );
    expect(first).toMatchObject({
      kernelId: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
      outputRange: {
        firstOutputRow: 0,
        outputRowCount: 32,
        workgroupsX: 1,
        workgroupsY: 8,
      },
    });
    expect(second.outputRange).toMatchObject({
      firstOutputRow: 32,
      outputRowCount: 16,
      workgroupsX: 1,
      workgroupsY: 8,
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
    const pass = fakePass();
    first.encode(pass);
    second.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([[256], [512]]);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [1, 8, 1],
      [1, 8, 1],
    ]);

    kernel.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
  });

  it("fails closed on non-fixed subgroups and never accepts K1", async () => {
    expect(() => AceFp16VaeConv1dSubgroupKernel.create(
      fakeDevice({ shaderF16: false }),
      FIXED_32_CAPABILITY,
    )).toThrow(/requires WebGPU shader-f16/);
    expect(() => AceFp16VaeConv1dSubgroupKernel.create(
      fakeDevice({ subgroups: false }),
      FIXED_32_CAPABILITY,
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceFp16VaeConv1dSubgroupKernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceFp16VaeConv1dSubgroupKernel.create(
      fakeDevice({ maximumInvocations: 127 }),
      FIXED_32_CAPABILITY,
    )).toThrow(/WG128/);

    const kernel = AceFp16VaeConv1dSubgroupKernel.create(
      fakeDevice(),
      FIXED_32_CAPABILITY,
    );
    const shape: AceVaeConv1dShape = {
      batch: 1,
      inputFrames: 17,
      inputChannels: 64,
      outputChannels: 64,
      kernelSize: 1,
      stride: 1,
      dilation: 1,
      padding: 0,
    };
    await expect(kernel.createDispatch(
      "k1",
      shape,
      bindingsFor(shape, "float16", true),
      "float16",
      fullRange(fakeBuffer(256), 0, shape),
    )).rejects.toThrow(/supports only K7/);
  });

  it("reports compilation diagnostics and evicts the failed pipeline", async () => {
    const device = fakeDevice({
      compilationMessageBatches: [[{
        message: "synthetic subgroup diagnostic",
        type: "error",
        lineNum: 73,
        linePos: 4,
      }], []],
    });
    const kernel = AceFp16VaeConv1dSubgroupKernel.create(
      device,
      FIXED_32_CAPABILITY,
    );
    const shape = k7Shape(64, 64, 17);
    const bindings = bindingsFor(shape, "float16", true);
    const range = fullRange(fakeBuffer(256), 0, shape);
    await expect(kernel.createDispatch(
      "diagnostic",
      shape,
      bindings,
      "float16",
      range,
    )).rejects.toThrow(/73:4 synthetic subgroup diagnostic/);
    await expect(kernel.createDispatch(
      "retry",
      shape,
      bindings,
      "float16",
      range,
    )).resolves.toMatchObject({
      kernelId: ACE_FP16_VAE_CONV1D_SUBGROUP_KERNEL_ID,
    });
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
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
  readonly compilationMessageBatches?: readonly (
    readonly Partial<GPUCompilationMessage>[]
  )[];
} = {}): FakeDevice {
  const features = [
    ...(options.shaderF16 === false ? [] : ["shader-f16"]),
    ...(options.subgroups === false ? [] : ["subgroups"]),
  ];
  const compilationMessageBatches = [
    ...(options.compilationMessageBatches ?? [[]]),
  ];
  return {
    features: new Set(features),
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
