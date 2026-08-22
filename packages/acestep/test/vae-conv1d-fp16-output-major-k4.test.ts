import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_CIN_VECTOR_WIDTH,
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_KERNEL_ID,
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_LAYOUT,
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_OUTPUTS_PER_LANE,
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_REDUCTION_SEMANTICS,
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_ROWS,
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_SIZE,
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_SUBGROUPS_PER_WORKGROUP,
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_TILE_CHANNELS,
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_TILE_ROWS,
  ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_WORKGROUP_SIZE,
  AceOpt0047VaeConv1dOutputMajorK4Kernel,
  aceOpt0047VaeK7NativeWeightIndex,
  aceOpt0047VaeK7PackedWeightCoordinate,
  aceOpt0047VaeK7PackedWeightIndex,
  aceOpt0047VaeConv1dOutputMajorK4Wgsl,
  packAceOpt0047VaeK7WeightU16,
  unpackAceOpt0047VaeK7WeightU16,
} from "../src/webgpu/kernels/vae-conv1d-fp16-output-major-k4.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dBindings,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";
import {
  buildOpt0047Cases,
  parseOpt0047ThermalGate,
  summarizeOpt0047Timing,
} from "./browser/opt-0047-vae-k7-output-major-k4.js";
import BROWSER_SOURCE from
  "./browser/opt-0047-vae-k7-output-major-k4.ts?raw";
import BROWSER_HTML from
  "./browser/opt-0047-vae-k7-output-major-k4.html?raw";

describe("OPT-0047 K7 output-major K4 weight layout", () => {
  it("pins only the OPT-0024 ownership and physical weight ordering", () => {
    expect(ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_KERNEL_ID).toBe(
      "ace-vae-fp16-opt-0047-output-major-k4-fp16-dot4-k7-conv1d-v1",
    );
    expect(ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_LAYOUT).toBe(
      "k7-cin4-cout-tile128-lane32-output4-cin-element4",
    );
    expect(ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_REDUCTION_SEMANTICS).toBe(
      "increasing-k-cin4-fp16-dot4-partials-fp32-accumulator",
    );
    expect({
      subgroupSize: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_SIZE,
      workgroupSize: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_WORKGROUP_SIZE,
      subgroups: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_SUBGROUPS_PER_WORKGROUP,
      rows: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_ROWS,
      outputsPerLane:
        ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_OUTPUTS_PER_LANE,
      tileRows: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_TILE_ROWS,
      tileChannels: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_TILE_CHANNELS,
      cinVector: ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_CIN_VECTOR_WIDTH,
    }).toEqual({
      subgroupSize: 32,
      workgroupSize: 128,
      subgroups: 4,
      rows: 8,
      outputsPerLane: 4,
      tileRows: 32,
      tileChannels: 128,
      cinVector: 4,
    });
  });

  it("proves every production-layout address and U16 pack inverse", () => {
    const dimensions = [
      [64, 2_048],
      [1_024, 1_024],
      [512, 512],
      [256, 256],
      [128, 128],
    ] as const;
    let exhaustiveAddressCount = 0;
    let exhaustiveU16Count = 0;
    let forwardInverseMismatchCount = 0;
    let duplicateNativeAddressCount = 0;
    for (const [inputChannels, outputChannels] of dimensions) {
      const elementCount = inputChannels * outputChannels * 7;
      const seenNative = new Uint8Array(elementCount);
      const native = new Uint16Array(elementCount);
      for (let index = 0; index < elementCount; index += 1) {
        native[index] = ((index * 40503) ^ (index >>> 7) ^ 0xa55a) & 0xffff;
        const coordinate = aceOpt0047VaeK7PackedWeightCoordinate(
          inputChannels,
          outputChannels,
          index,
        );
        if (aceOpt0047VaeK7PackedWeightIndex(
          inputChannels,
          outputChannels,
          coordinate,
        ) !== index) forwardInverseMismatchCount += 1;
        const nativeIndex = aceOpt0047VaeK7NativeWeightIndex(
          inputChannels,
          outputChannels,
          coordinate,
        );
        if (seenNative[nativeIndex] !== 0) duplicateNativeAddressCount += 1;
        seenNative[nativeIndex] = 1;
      }
      expect(seenNative.every((value) => value === 1)).toBe(true);
      const packed = packAceOpt0047VaeK7WeightU16(
        native,
        inputChannels,
        outputChannels,
      );
      const inverse = unpackAceOpt0047VaeK7WeightU16(
        packed,
        inputChannels,
        outputChannels,
      );
      expect(inverse).toEqual(native);
      exhaustiveAddressCount += elementCount;
      exhaustiveU16Count += elementCount;
    }
    expect(forwardInverseMismatchCount).toBe(0);
    expect(duplicateNativeAddressCount).toBe(0);
    expect(exhaustiveAddressCount).toBe(10_665_984);
    expect(exhaustiveU16Count).toBe(10_665_984);
  }, 60_000);

  it("emits OPT-0024 math with only four adjacent packed-weight loads", () => {
    const source = aceOpt0047VaeConv1dOutputMajorK4Wgsl(
      shape(1_024, 1_024, 512, 9),
      true,
      "float16",
    );
    expect(source).toContain(`// weight-layout: ${
      ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_LAYOUT
    }`);
    expect(source).toContain("const OUTPUT_CHANNEL_TILES: u32 = 8u;");
    expect(source).toContain("let packed_weight_base = ((((kernel * INPUT_CHANNEL_VEC4S +");
    expect(source).toContain("input_channel4) * OUTPUT_CHANNEL_TILES + group.y)");
    expect(source.match(/let weight[0-3] = weight\[packed_weight_base/g))
      .toHaveLength(4);
    expect(source.match(/\bdot\(input_operand[0-7], weight[0-3]\)/g))
      .toHaveLength(32);
    expect(source.match(/sum[0-7] = sum[0-7] \+ vec4<f32>\(partial[0-7]\)/g))
      .toHaveLength(8);
    expect(source.match(/if \(input_valid[0-7]\) \{/g)).toHaveLength(8);
    expect(source).toContain("for (var kernel = 0u; kernel < 7u;");
    expect(source).toContain("var input_channel4 = 0u;");
    expect(source).not.toContain("output_channel_base * 7u + kernel");
    expect(source).not.toContain("var<workgroup>");
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toMatch(/\bfma\s*\(/);
  });

  it("owns one cached, allocation-free dispatch and rejects after destroy", async () => {
    const device = fakeDevice();
    const kernel = AceOpt0047VaeConv1dOutputMajorK4Kernel.create(
      device,
      FIXED_32,
    );
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
      ACE_OPT_0047_VAE_CONV1D_OUTPUT_MAJOR_K4_KERNEL_ID,
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

  it("fails closed outside the exact K7/bias/FP16/divisibility boundary", async () => {
    expect(() => AceOpt0047VaeConv1dOutputMajorK4Kernel.create(
      fakeDevice({ shaderF16: false }),
      FIXED_32,
    )).toThrow(/shader-f16/);
    expect(() => AceOpt0047VaeConv1dOutputMajorK4Kernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32-lane/);
    const kernel = AceOpt0047VaeConv1dOutputMajorK4Kernel.create(
      fakeDevice(),
      FIXED_32,
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
    expect(() => packAceOpt0047VaeK7WeightU16(
      new Uint16Array(4),
      4,
      128,
    )).toThrow(/length changed/);
    expect(() => aceOpt0047VaeK7PackedWeightCoordinate(4, 128, -1))
      .toThrow(/out of bounds/);
  });

  it("pins all 16 production K7 occurrences and the four weighted tiers", () => {
    const cases = buildOpt0047Cases();
    expect(cases).toHaveLength(16);
    expect(cases[0]).toMatchObject({
      id: "conv1",
      tier: "conv1",
      dilation: 1,
      timingWeight: 0,
      shape: { inputFrames: 512, inputChannels: 64, outputChannels: 2_048 },
    });
    expect(cases.filter(({ timingWeight }) => timingWeight > 0).map((item) =>
      [item.id, item.timingWeight]
    )).toEqual([
      ["block-0-res-1-conv1", 282],
      ["block-1-res-2-conv1", 423],
      ["block-2-res-1-conv1", 423],
      ["block-4-res-3-conv1", 1_269],
    ]);
    expect(new Set(cases.map(({ dilation }) => dilation))).toEqual(
      new Set([1, 3, 9]),
    );
    expect(cases.every(({ shape }) => shape.inputFrames === 512)).toBe(true);
  });

  it("requires every tier non-slower and the frozen 1.20x weighted gate", () => {
    const passing = summarizeOpt0047Timing(timingInput(120, 90));
    expect(passing).toMatchObject({
      weightTotal: 2_397,
      weightedSpeedup: 120 / 90,
      everyTierNonSlower: true,
      requiredWeightedSpeedup: 1.20,
      passed: true,
    });
    const oneRegression = timingInput(120, 90).map((tier, index) =>
      index === 0
        ? { ...tier, outputMajorSamplesMilliseconds: [121, 121, 121, 121] }
        : tier
    );
    expect(summarizeOpt0047Timing(oneRegression)).toMatchObject({
      everyTierNonSlower: false,
      passed: false,
    });
  });

  it("accepts exactly one level-0 check after at least 30 seconds", () => {
    const parameters = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalCommand: "notifyutil -g com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: "2000",
      thermalCheckedAtEpochMilliseconds: "32055",
      thermalObservations: "1",
      thermalObservedLevel: "0",
      thermalMaximumObservationGapMilliseconds: "30055",
    });
    expect(parseOpt0047ThermalGate(parameters, 1_000, 32_100)).toMatchObject({
      durationMilliseconds: 30_055,
      observationCount: 1,
      observedLevel: 0,
      launchDelayMilliseconds: 45,
    });
    parameters.set("thermalObservations", "2");
    expect(() => parseOpt0047ThermalGate(parameters, 1_000, 32_100))
      .toThrow(/one truthful level-0/);
  });

  it("pins the lean button gate, exactness, canaries, determinism, and cleanup", () => {
    for (const token of [
      "packAceOpt0047VaeK7WeightU16",
      "unpackAceOpt0047VaeK7WeightU16",
      "packInverseMismatchCount",
      "rawU16Exact",
      "deterministicBothArms",
      "OUTPUT_PREFILL_QNAN_F16",
      "allOutputsFinite",
      "guardsUntouched",
      "tracker.destroyAll()",
      "weightedSpeedup",
      "REQUIRED_WEIGHTED_SPEEDUP = 1.20",
      "window.__ACE_OPT0047_RESULT__ = receipt",
    ]) expect(BROWSER_SOURCE).toContain(token);
    expect(BROWSER_SOURCE).not.toContain("workgroupBarrier");
    expect(BROWSER_HTML).toContain('id="run" type="button" disabled');
    expect(BROWSER_HTML).toContain(
      "notifyutil -g com.apple.system.thermalpressurelevel",
    );
    expect(BROWSER_HTML).toContain("exactly one external");
    expect(BROWSER_HTML).toContain("1.20×");
  });
});

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const FIXED_32 = Object.freeze({ subgroupMinSize: 32, subgroupMaxSize: 32 });

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
  return hasBias ? { ...common, bias: fakeBinding(plan.biasBindingBytes) } : common;
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

function fakeDevice(options: { readonly shaderF16?: boolean } = {}): GPUDevice & {
  readonly createBuffer: ReturnType<typeof vi.fn>;
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
} {
  return {
    features: new Set([
      ...(options.shaderF16 === false ? [] : ["shader-f16"]),
      "subgroups",
    ]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
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
      throw new Error("OPT-0047 owner must not allocate");
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

function timingInput(
  control: number,
  candidate: number,
): readonly {
  id: string;
  weight: number;
  opt0024SamplesMilliseconds: readonly number[];
  outputMajorSamplesMilliseconds: readonly number[];
}[] {
  return [282, 423, 423, 1_269].map((weight, index) => ({
    id: `tier-${index}`,
    weight,
    opt0024SamplesMilliseconds: [control, control, control, control],
    outputMajorSamplesMilliseconds: [candidate, candidate, candidate, candidate],
  }));
}
