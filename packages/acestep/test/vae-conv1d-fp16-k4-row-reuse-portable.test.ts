import { describe, expect, it, vi } from "vitest";

import {
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS,
} from "../src/model/manifest.js";
import {
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_LAYOUT,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_REDUCTION_SEMANTICS,
  aceOpt0051VaeK7NativeWeightIndex,
  aceOpt0051VaeK7PackedWeightCoordinate,
  aceOpt0051VaeK7PackedWeightIndex,
  packAceOpt0051VaeK7WeightU16,
  planAceOpt0051VaeConv1dK4RowReuse16x64Range,
  unpackAceOpt0051VaeK7WeightU16,
} from "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-16x64.js";
import {
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_CHANNELS_PER_SLICE,
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_CIN_VECTOR_WIDTH,
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_KERNEL_ID,
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_LAYOUT,
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_OUTPUTS_PER_LANE,
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_REDUCTION_SEMANTICS,
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_ROWS,
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_SLICE_LANES,
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_SLICES,
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_STAGED_VEC4S,
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_SIZE,
  ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_STORAGE_BYTES,
  AceOpt0088VaeConv1dK4RowReusePortableKernel,
  aceOpt0088VaeConv1dK4RowReusePortableWgsl,
  aceOpt0088VaeK7NativeWeightIndex,
  aceOpt0088VaeK7PackedWeightCoordinate,
  aceOpt0088VaeK7PackedWeightIndex,
  packAceOpt0088VaeK7WeightU16,
  planAceOpt0088VaeConv1dK4RowReusePortableRange,
  unpackAceOpt0088VaeK7WeightU16,
} from "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-portable.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dBindings,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";

const CONTRACT_SHAPES = ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS.map(
  (contract) => ({
    contract,
    shape: shape(
      contract.channels,
      contract.channels,
      512,
      contract.dilation as 1 | 3 | 9,
    ),
  }),
);

describe("OPT-0088 K7 K4 row-reuse portable geometry", () => {
  it("pins the WG128 four-slice ownership within portable limits", () => {
    expect(ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_KERNEL_ID).toBe(
      "opt-0088-vae-conv1d-k4-row-reuse-portable-v1",
    );
    expect(ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_LAYOUT).toBe(
      ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_LAYOUT,
    );
    expect(ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_LAYOUT).toBe(
      "k7-cin4-cout-band64-lane32-output2-cin-element4",
    );
    expect(
      ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_REDUCTION_SEMANTICS,
    ).toBe(ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_REDUCTION_SEMANTICS);
    expect({
      sliceLanes: ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_SLICE_LANES,
      workgroupSize:
        ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_SIZE,
      slices: ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_SLICES,
      rows: ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_ROWS,
      outputsPerLane:
        ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_OUTPUTS_PER_LANE,
      channelsPerSlice:
        ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_CHANNELS_PER_SLICE,
      cinVector:
        ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_CIN_VECTOR_WIDTH,
      stagedVec4s:
        ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_STAGED_VEC4S,
      workgroupStorageBytes:
        ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_STORAGE_BYTES,
    }).toEqual({
      sliceLanes: 32,
      workgroupSize: 128,
      slices: 4,
      rows: 16,
      outputsPerLane: 2,
      channelsPerSlice: 64,
      cinVector: 4,
      stagedVec4s: 64,
      workgroupStorageBytes: 512,
    });
    expect(ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_SIZE)
      .toBeLessThanOrEqual(256);
    expect(
      ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_WORKGROUP_STORAGE_BYTES,
    ).toBeLessThanOrEqual(16_384);
  });

  it("covers all 12 revision-7 contract shapes across C1024/C512/C128", () => {
    expect(CONTRACT_SHAPES).toHaveLength(12);
    expect(new Set(CONTRACT_SHAPES.map(({ shape }) => shape.inputChannels)))
      .toEqual(new Set([1_024, 512, 128]));
    expect(new Set(CONTRACT_SHAPES.map(({ shape }) => shape.dilation)))
      .toEqual(new Set([1, 3, 9]));
    expect(CONTRACT_SHAPES.every(({ shape }) =>
      shape.inputChannels === shape.outputChannels &&
      shape.kernelSize === 7 &&
      shape.padding === 3 * shape.dilation
    )).toBe(true);
  });

  it("shares the OPT-0051 packed-weight bijection identically", () => {
    expect(aceOpt0088VaeK7PackedWeightIndex)
      .toBe(aceOpt0051VaeK7PackedWeightIndex);
    expect(aceOpt0088VaeK7PackedWeightCoordinate)
      .toBe(aceOpt0051VaeK7PackedWeightCoordinate);
    expect(aceOpt0088VaeK7NativeWeightIndex)
      .toBe(aceOpt0051VaeK7NativeWeightIndex);
    expect(packAceOpt0088VaeK7WeightU16).toBe(packAceOpt0051VaeK7WeightU16);
    expect(unpackAceOpt0088VaeK7WeightU16)
      .toBe(unpackAceOpt0051VaeK7WeightU16);
    for (const [inputChannels, outputChannels, stride] of [
      [128, 128, 1],
      [512, 512, 251],
      [1_024, 1_024, 1_021],
    ] as const) {
      const elementCount = inputChannels * outputChannels * 7;
      for (let index = 0; index < elementCount; index += stride) {
        const coordinate = aceOpt0051VaeK7PackedWeightCoordinate(
          inputChannels,
          outputChannels,
          index,
        );
        expect(aceOpt0088VaeK7PackedWeightIndex(
          inputChannels,
          outputChannels,
          coordinate,
        )).toBe(index);
      }
    }
    const native = new Uint16Array(128 * 128 * 7);
    for (let index = 0; index < native.length; index += 1) {
      native[index] = ((index * 40503) ^ (index >>> 7) ^ 0xa55a) & 0xffff;
    }
    const packed = packAceOpt0088VaeK7WeightU16(native, 128, 128);
    expect(packed).toEqual(packAceOpt0051VaeK7WeightU16(native, 128, 128));
    expect(unpackAceOpt0088VaeK7WeightU16(packed, 128, 128)).toEqual(native);
  }, 60_000);

  it("emits barrier-staged 16-row x two-output K4 math with no hardware-lane builtins", () => {
    for (const { contract, shape: candidateShape } of CONTRACT_SHAPES) {
      const source = aceOpt0088VaeConv1dK4RowReusePortableWgsl(
        candidateShape,
        true,
        "float16",
      );
      expect(source.toLowerCase()).not.toContain("subgroup");
      expect(source).toContain("enable f16;");
      expect(source).toContain(`// kernel-id: ${
        ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_KERNEL_ID
      }`);
      expect(source).toContain(`// weight-layout: ${
        ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_LAYOUT
      }`);
      expect(source).toContain(`// reduction-semantics: ${
        ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_REDUCTION_SEMANTICS
      }`);
      expect(source).toContain("fn main(");
      expect(source).toContain("@compute @workgroup_size(\n  128,");
      expect(source).toContain("@builtin(local_invocation_index) local_index");
      expect(source).toContain(
        "var<workgroup> staged_input: array<\n  vec4<f16>,\n  64\n>;",
      );
      expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
      expect(source).toContain("let lane = local_index %\n    32u;");
      expect(source).toContain("let slice = local_index /\n    32u;");
      expect(source).toContain("let stage_base = slice *\n    16u;");
      expect(source).toContain("staged_input[stage_base + lane] = lane_input;");
      expect(source).toContain(
        "let channel_band = slice % CHANNEL_BANDS_PER_WORKGROUP;",
      );
      expect(source).toContain(
        "let row_band = slice / CHANNEL_BANDS_PER_WORKGROUP;",
      );
      // Identical revision-7 packed index math as OPT-0051, lane-relabeled.
      expect(source).toContain("32u + lane) * 2u);");
      expect(source).toContain(
        "let packed_weight_base = ((((kernel * INPUT_CHANNEL_VEC4S +",
      );
      expect(source.match(/let weight[01] = weight\[packed_weight_base/g))
        .toHaveLength(2);
      expect(source.match(
        /let input_operand(?:[0-9]|1[0-5]) = staged_input\[stage_base \+ (?:[0-9]|1[0-5])u\];/g,
      )).toHaveLength(16);
      expect(source.match(/\bdot\(input_operand(?:[0-9]|1[0-5]), weight[01]\)/g))
        .toHaveLength(32);
      expect(source.match(/var sum(?:[0-9]|1[0-5]): vec2<f32> = initial_sum/g))
        .toHaveLength(16);
      expect(source.match(/let partial(?:[0-9]|1[0-5]) = vec2<f16>/g))
        .toHaveLength(16);
      expect(source.match(/\+ vec2<f32>\(partial(?:[0-9]|1[0-5])\)/g))
        .toHaveLength(16);
      expect(source.match(/\] = f16\(sum(?:[0-9]|1[0-5])\.[xy]\);/g))
        .toHaveLength(32);
      expect(source).toContain("for (var kernel = 0u; kernel < 7u;");
      expect(source).toContain("var input_channel4 = 0u;");
      expect(source).not.toMatch(/\bfma\s*\(/);
      // Shape constants stay authenticated per contract.
      expect(source).toContain(
        `const OUTPUT_CHANNEL_BANDS: u32 = ${contract.channels / 64}u;`,
      );
      expect(source).toContain(`const DILATION: u32 = ${contract.dilation}u;`);
      expect(source).toContain(
        `const PADDING: u32 = ${contract.dilation * 3}u;`,
      );
      expect(source).toContain(
        `const CHANNEL_BANDS_PER_WORKGROUP: u32 = ${
          contract.channels === 128 ? 2 : 4
        }u;`,
      );
      // Stage-write happens between the two barriers and strictly before the
      // weight loads; rows are consumed one at a time in ascending order.
      const firstBarrier = source.indexOf("workgroupBarrier();");
      const stageWrite = source.indexOf(
        "staged_input[stage_base + lane] = lane_input;",
      );
      const secondBarrier = source.indexOf("workgroupBarrier();", stageWrite);
      const firstWeight = source.indexOf("let weight0 = weight[");
      expect(firstBarrier).toBeGreaterThan(-1);
      expect(firstBarrier).toBeLessThan(stageWrite);
      expect(stageWrite).toBeLessThan(secondBarrier);
      expect(secondBarrier).toBeLessThan(firstWeight);
      const lastWeight = source.indexOf("let weight1 = weight[");
      for (let row = 0; row < 15; row += 1) {
        const staged = source.indexOf(
          `let input_operand${row} = staged_input`,
          lastWeight,
        );
        const partial = source.indexOf(
          `let partial${row} = vec2<f16>`,
          staged,
        );
        const nextStaged = source.indexOf(
          `let input_operand${row + 1} = staged_input`,
          partial,
        );
        expect(lastWeight).toBeLessThan(staged);
        expect(staged).toBeLessThan(partial);
        expect(partial).toBeLessThan(nextStaged);
      }
    }
  });

  it("keeps the dilated-tap addressing of the shared K4 arithmetic", () => {
    const source = aceOpt0088VaeConv1dK4RowReusePortableWgsl(
      shape(512, 512, 512, 9),
      true,
      "float16",
    );
    expect(source).toContain("const DILATION: u32 = 9u;");
    expect(source).toContain("const PADDING: u32 = 27u;");
    expect(source.match(/\+ kernel \* DILATION;/g)).toHaveLength(17);
    expect(source.match(/padded_time(?:[0-9]|1[0-5]) - PADDING < INPUT_FRAMES/g))
      .toHaveLength(16);
    expect(source).toContain("let lane_input_time = lane_padded_time - PADDING;");
  });

  it("reproduces the OPT-0051 tile geometry for every contract shape", () => {
    for (const { shape: candidateShape } of CONTRACT_SHAPES) {
      const plan = planAceFp16VaeConv1d(candidateShape, "float16");
      const full = { base: 0, count: plan.outputElements };
      expect(planAceOpt0088VaeConv1dK4RowReusePortableRange(plan, full))
        .toEqual(planAceOpt0051VaeConv1dK4RowReuse16x64Range(plan, full));
    }
    const highPlan = planAceFp16VaeConv1d(
      shape(1_024, 1_024, 512, 1),
      "float16",
    );
    expect(planAceOpt0088VaeConv1dK4RowReusePortableRange(
      highPlan,
      { base: 0, count: highPlan.outputElements },
    )).toMatchObject({
      channelBands: 4,
      rowBands: 1,
      tileRows: 16,
      tileChannels: 256,
      workgroupsX: 32,
      workgroupsY: 4,
    });
    const lowPlan = planAceFp16VaeConv1d(
      shape(128, 128, 35, 3),
      "float16",
    );
    expect(planAceOpt0088VaeConv1dK4RowReusePortableRange(
      lowPlan,
      { base: 0, count: lowPlan.outputElements },
    )).toMatchObject({
      channelBands: 2,
      rowBands: 2,
      tileRows: 32,
      tileChannels: 128,
      workgroupsX: 2,
      workgroupsY: 1,
    });
  });

  it("owns cached allocation-free dispatches on a no-hardware-lane device", async () => {
    const device = fakeDevice();
    expect(device.features.has("subgroups")).toBe(false);
    const kernel = AceOpt0088VaeConv1dK4RowReusePortableKernel.create(device);
    const candidateShape = shape(128, 128, 33, 3);
    const plan = planAceFp16VaeConv1d(candidateShape, "float16");
    const bindings = bindingsFor(candidateShape);
    const control = fakeBuffer(512);
    const first = await kernel.createDispatch(
      "first",
      candidateShape,
      bindings,
      "float16",
      range(control, 256, 0, 32 * 128),
    );
    const tail = await kernel.createDispatch(
      "tail",
      candidateShape,
      bindings,
      "float16",
      range(control, 0, 32 * 128, 128),
    );
    expect(first.kernelId).toBe(
      ACE_OPT_0088_VAE_CONV1D_K4_ROW_REUSE_PORTABLE_KERNEL_ID,
    );
    expect(device.createBuffer).not.toHaveBeenCalled();
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();
    const pass = fakePass();
    first.encode(pass);
    tail.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([[256], [0]]);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([[1, 1, 1], [1, 1, 1]]);
    expect(plan.weightBindingBytes).toBe(128 * 128 * 7 * 2);
    kernel.destroy();
    kernel.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
    await expect(kernel.createDispatch(
      "post-destroy",
      candidateShape,
      bindings,
      "float16",
      range(control, 0, 0, plan.outputElements),
    )).rejects.toThrow(/was destroyed/);
  });

  it("fails closed outside the frozen K7/bias/FP16/tile boundary", async () => {
    expect(() => AceOpt0088VaeConv1dK4RowReusePortableKernel.create(
      fakeDevice({ shaderF16: false }),
    )).toThrow(/shader-f16/);
    expect(() => AceOpt0088VaeConv1dK4RowReusePortableKernel.create(
      fakeDevice({ maxComputeWorkgroupStorageSize: 256 }),
    )).toThrow(/workgroup bytes/);
    expect(() => AceOpt0088VaeConv1dK4RowReusePortableKernel.create(
      fakeDevice({ maxComputeInvocationsPerWorkgroup: 64 }),
    )).toThrow(/WG128/);
    const kernel = AceOpt0088VaeConv1dK4RowReusePortableKernel.create(
      fakeDevice(),
    );
    const candidateShape = shape(128, 128, 33, 1);
    const control = fakeBuffer(256);
    await expect(kernel.createDispatch(
      "no-bias",
      candidateShape,
      bindingsFor(candidateShape, false),
      "float16",
      range(control, 0, 0, 33 * 128),
    )).rejects.toThrow(/requires bias/);
    const invalidCin = shape(130, 128, 33, 1);
    await expect(kernel.createDispatch(
      "invalid-cin",
      invalidCin,
      bindingsFor(invalidCin),
      "float16",
      range(control, 0, 0, 33 * 128),
    )).rejects.toThrow(/Cin divisible by 4/);
    const unsafeCout = shape(128, 384, 33, 1);
    await expect(kernel.createDispatch(
      "unsafe-cout-tail",
      unsafeCout,
      bindingsFor(unsafeCout),
      "float16",
      range(control, 0, 0, 33 * 384),
    )).rejects.toThrow(/Cout 128 or divisible by 256/);
    await expect(kernel.createDispatch(
      "fp32-output",
      candidateShape,
      bindingsFor(candidateShape),
      "float32",
      range(control, 0, 0, 33 * 128),
    )).rejects.toThrow(/FP16 internal output/);
  });
});

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

function shape(
  inputChannels: number,
  outputChannels: number,
  inputFrames: number,
  dilation: 1 | 3 | 9,
): AceVaeConv1dShape {
  return {
    batch: 1,
    inputFrames,
    inputChannels,
    outputChannels,
    kernelSize: 7,
    stride: 1,
    dilation,
    padding: dilation * 3,
  };
}

function bindingsFor(
  candidateShape: AceVaeConv1dShape,
  hasBias = true,
): AceFp16VaeConv1dBindings {
  const plan = planAceFp16VaeConv1d(candidateShape, "float16");
  const common = {
    input: fakeBinding(plan.inputBindingBytes),
    weight: fakeBinding(plan.weightBindingBytes),
    output: fakeBinding(plan.outputBindingBytes),
  };
  return hasBias
    ? { ...common, bias: fakeBinding(plan.biasBindingBytes) }
    : common;
}

function range(
  buffer: GPUBuffer,
  offset: number,
  base: number,
  count: number,
): AceVaeOutputRangeBinding {
  return { base, count, control: { buffer, offset, size: 16 } };
}

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: fakeBuffer(size), offset: 0, size };
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function fakeDevice(options: {
  readonly shaderF16?: boolean;
  readonly maxComputeInvocationsPerWorkgroup?: number;
  readonly maxComputeWorkgroupStorageSize?: number;
} = {}): GPUDevice & {
  readonly createBuffer: ReturnType<typeof vi.fn>;
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
} {
  // Deliberately no "subgroups" feature: the portable owner must stand up on
  // Safari/Firefox-class devices that only report shader-f16.
  return {
    features: new Set(options.shaderF16 === false ? [] : ["shader-f16"]),
    limits: {
      maxComputeInvocationsPerWorkgroup:
        options.maxComputeInvocationsPerWorkgroup ?? 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize:
        options.maxComputeWorkgroupStorageSize ?? 16_384,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxStorageBufferBindingSize: 1_073_741_824,
      maxBufferSize: 1_073_741_824,
      minUniformBufferOffsetAlignment: 256,
      minStorageBufferOffsetAlignment: 256,
    },
    createShaderModule: vi.fn(() => ({
      getCompilationInfo: vi.fn(async () => ({ messages: [] })),
    })),
    createBindGroupLayout: vi.fn(() => ({})),
    createPipelineLayout: vi.fn(() => ({})),
    createComputePipelineAsync: vi.fn(async () => ({})),
    createBindGroup: vi.fn(() => ({})),
    createBuffer: vi.fn(() => {
      throw new Error("OPT-0088 owner must not allocate");
    }),
  } as unknown as GPUDevice & {
    readonly createBuffer: ReturnType<typeof vi.fn>;
    readonly createShaderModule: ReturnType<typeof vi.fn>;
    readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
    readonly createBindGroup: ReturnType<typeof vi.fn>;
  };
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
