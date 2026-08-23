import { createHash } from "node:crypto";

import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_CHANNELS_PER_SUBGROUP,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_CIN_VECTOR_WIDTH,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_LAYOUT,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_MAX_CHANNEL_BANDS,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_OUTPUTS_PER_LANE,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_REDUCTION_SEMANTICS,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_ROWS,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_SIZE,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_SUBGROUPS_PER_WORKGROUP,
  ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_WORKGROUP_SIZE,
  AceOpt0051VaeConv1dK4RowReuse16x64Kernel,
  aceOpt0051VaeK7NativeWeightIndex,
  aceOpt0051VaeK7PackedWeightCoordinate,
  aceOpt0051VaeK7PackedWeightIndex,
  aceOpt0051VaeConv1dK4RowReuse16x64Wgsl,
  packAceOpt0051VaeK7WeightU16,
  planAceOpt0051VaeConv1dK4RowReuse16x64Range,
  unpackAceOpt0051VaeK7WeightU16,
} from "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-16x64.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dBindings,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";
import {
  buildOpt0051Cases,
  buildOpt0051GeneratedShaderIdentityPayload,
  parseOpt0051ThermalGate,
  summarizeOpt0051Timing,
} from "./browser/opt-0051-vae-k7-k4-row-reuse-16x64.js";
import OPT0024_CORE_SOURCE from
  "../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.ts?raw";
import CANDIDATE_CORE_SOURCE from
  "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-16x64.ts?raw";
import DECODER_CORE_SOURCE from "../src/webgpu/vae-decoder.ts?raw";
import BROWSER_SOURCE from
  "./browser/opt-0051-vae-k7-k4-row-reuse-16x64.ts?raw";
import BROWSER_HTML from
  "./browser/opt-0051-vae-k7-k4-row-reuse-16x64.html?raw";

describe("OPT-0051 K7 K4 row-reuse 16x64 geometry", () => {
  it("pins the fixed32 WG128 ownership and unchanged K4 reduction", () => {
    expect(ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID).toBe(
      "ace-vae-fp16-opt-0051-k4-row-reuse-16x64-fp16-dot4-k7-conv1d-v1",
    );
    expect(ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_LAYOUT).toBe(
      "k7-cin4-cout-band64-lane32-output2-cin-element4",
    );
    expect(ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_REDUCTION_SEMANTICS).toBe(
      "increasing-k-cin4-fp16-dot4-partials-fp32-accumulator",
    );
    expect({
      subgroupSize: ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_SIZE,
      workgroupSize: ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_WORKGROUP_SIZE,
      subgroups:
        ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_SUBGROUPS_PER_WORKGROUP,
      rows: ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_ROWS,
      outputsPerLane:
        ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_OUTPUTS_PER_LANE,
      channelsPerSubgroup:
        ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_CHANNELS_PER_SUBGROUP,
      maximumChannelBands:
        ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_MAX_CHANNEL_BANDS,
      cinVector:
        ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_CIN_VECTOR_WIDTH,
      fp32AccumulatorsPerLane:
        ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_ROWS *
        ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_OUTPUTS_PER_LANE,
    }).toEqual({
      subgroupSize: 32,
      workgroupSize: 128,
      subgroups: 4,
      rows: 16,
      outputsPerLane: 2,
      channelsPerSubgroup: 64,
      maximumChannelBands: 4,
      cinVector: 4,
      fp32AccumulatorsPerLane: 32,
    });
  });

  it("proves every four-tier/adversarial address and U16 pack inverse", () => {
    const dimensions = [
      [1_024, 1_024],
      [512, 512],
      [256, 256],
      [128, 128],
      [64, 128],
      [68, 128],
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
        const coordinate = aceOpt0051VaeK7PackedWeightCoordinate(
          inputChannels,
          outputChannels,
          index,
        );
        if (aceOpt0051VaeK7PackedWeightIndex(
          inputChannels,
          outputChannels,
          coordinate,
        ) !== index) forwardInverseMismatchCount += 1;
        const nativeIndex = aceOpt0051VaeK7NativeWeightIndex(
          inputChannels,
          outputChannels,
          coordinate,
        );
        if (seenNative[nativeIndex] !== 0) duplicateNativeAddressCount += 1;
        seenNative[nativeIndex] = 1;
      }
      expect(seenNative.every((value) => value === 1)).toBe(true);
      const packed = packAceOpt0051VaeK7WeightU16(
        native,
        inputChannels,
        outputChannels,
      );
      expect(unpackAceOpt0051VaeK7WeightU16(
        packed,
        inputChannels,
        outputChannels,
      )).toEqual(native);
      exhaustiveAddressCount += elementCount;
      exhaustiveU16Count += elementCount;
    }
    expect(forwardInverseMismatchCount).toBe(0);
    expect(duplicateNativeAddressCount).toBe(0);
    expect(exhaustiveAddressCount).toBe(9_866_752);
    expect(exhaustiveU16Count).toBe(9_866_752);
  }, 60_000);

  it("emits 16-row x two-output K4 math without barriers", () => {
    const source = aceOpt0051VaeConv1dK4RowReuse16x64Wgsl(
      shape(1_024, 1_024, 512, 9),
      true,
      "float16",
    );
    expect(source).toContain(`// weight-layout: ${
      ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_LAYOUT
    }`);
    expect(source).toContain("const OUTPUT_CHANNEL_BANDS: u32 = 16u;");
    expect(source).toContain("const CHANNEL_BANDS_PER_WORKGROUP: u32 = 4u;");
    expect(source).toContain("group.x * 16u");
    expect(source).toContain("let channel_band = subgroup % CHANNEL_BANDS_PER_WORKGROUP;");
    expect(source).toContain("let row_band = subgroup / CHANNEL_BANDS_PER_WORKGROUP;");
    expect(source).toContain("32u + subgroup_lane) * 2u);");
    expect(source.match(/let weight[01] = weight\[packed_weight_base/g))
      .toHaveLength(2);
    expect(source.match(/subgroupBroadcast\(lane_input, (?:[0-9]|1[0-5])u\)/g))
      .toHaveLength(16);
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
    const firstWeight = source.indexOf("let weight1 = weight[");
    for (let row = 0; row < 15; row += 1) {
      const broadcast = source.indexOf(
        `let input_operand${row} = subgroupBroadcast`,
        firstWeight,
      );
      const partial = source.indexOf(`let partial${row} = vec2<f16>`, broadcast);
      const nextBroadcast = source.indexOf(
        `let input_operand${row + 1} = subgroupBroadcast`,
        partial,
      );
      expect(firstWeight).toBeLessThan(broadcast);
      expect(broadcast).toBeLessThan(partial);
      expect(partial).toBeLessThan(nextBroadcast);
    }
    expect(source).toContain("for (var kernel = 0u; kernel < 7u;");
    expect(source).toContain("var input_channel4 = 0u;");
    expect(source).not.toContain("var<workgroup>");
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toMatch(/\bfma\s*\(/);
  });

  it("maps Cout128 across two row bands while high tiers use four channels", () => {
    const c128 = aceOpt0051VaeConv1dK4RowReuse16x64Wgsl(
      shape(128, 128, 35, 3),
      true,
      "float16",
    );
    expect(c128).toContain("const OUTPUT_CHANNEL_BANDS: u32 = 2u;");
    expect(c128).toContain("const CHANNEL_BANDS_PER_WORKGROUP: u32 = 2u;");
    expect(c128).toContain("group.x * 32u");
    expect(c128).toContain("row_band * 16u");

    const highPlan = planAceFp16VaeConv1d(
      shape(1_024, 1_024, 512, 1),
      "float16",
    );
    expect(planAceOpt0051VaeConv1dK4RowReuse16x64Range(
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
    expect(planAceOpt0051VaeConv1dK4RowReuse16x64Range(
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

  it("owns cached allocation-free dispatches and rejects after destroy", async () => {
    const device = fakeDevice();
    const kernel = AceOpt0051VaeConv1dK4RowReuse16x64Kernel.create(
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
      ACE_OPT_0051_VAE_CONV1D_ROW_REUSE_16X64_KERNEL_ID,
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
    expect(() => AceOpt0051VaeConv1dK4RowReuse16x64Kernel.create(
      fakeDevice({ shaderF16: false }),
      FIXED_32,
    )).toThrow(/shader-f16/);
    expect(() => AceOpt0051VaeConv1dK4RowReuse16x64Kernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32-lane/);
    const kernel = AceOpt0051VaeConv1dK4RowReuse16x64Kernel.create(
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
    expect(() => packAceOpt0051VaeK7WeightU16(
      new Uint16Array(4),
      4,
      128,
    )).toThrow(/length changed/);
    expect(() => aceOpt0051VaeK7PackedWeightCoordinate(4, 128, -1))
      .toThrow(/out of bounds/);
  });

  it("pins 12 four-tier/dilation cases and four adversarial cases", () => {
    const cases = buildOpt0051Cases();
    expect(cases).toHaveLength(16);
    const production = cases.filter(({ kind }) => kind === "production");
    const adversarial = cases.filter(({ kind }) => kind !== "production");
    expect(production).toHaveLength(12);
    expect(adversarial.map(({ kind }) => kind)).toEqual([
      "signed-zero",
      "cancellation",
      "finite-range",
      "tail-cin",
    ]);
    expect(new Set(production.map(({ shape }) => shape.inputChannels)))
      .toEqual(new Set([1_024, 512, 256, 128]));
    expect(new Set(production.map(({ dilation }) => dilation)))
      .toEqual(new Set([1, 3, 9]));
    expect(production.every(({ shape }) =>
      shape.inputFrames === 512 &&
      shape.inputChannels === shape.outputChannels
    )).toBe(true);
    expect(cases.filter(({ timingWeight }) => timingWeight > 0).map((item) =>
      [item.id, item.timingWeight]
    )).toEqual([
      ["c1024-d1", 282],
      ["c512-d3", 423],
      ["c256-d1", 423],
      ["c128-d9", 1_269],
    ]);
    expect(adversarial.find(({ kind }) => kind === "tail-cin")?.shape)
      .toMatchObject({ inputFrames: 39, inputChannels: 68, outputChannels: 128 });
  });

  it("freezes the control, candidate, decoder, and generated shader identities", () => {
    const identities = [
      sha256Text(OPT0024_CORE_SOURCE),
      sha256Text(CANDIDATE_CORE_SOURCE),
      sha256Text(DECODER_CORE_SOURCE),
      sha256Text(buildOpt0051GeneratedShaderIdentityPayload()),
    ];
    expect(identities).toEqual([
      "fe3bf8110cef1a3bb791006e9d376fe549e9f00fe30e4738d7429cb0daf65841",
      "59e144c1316d642d362d206222888177cd4e792743b3e23631ca415e923d770a",
      "07f294e2aadd615c0a8b840884f43205bc00c146362f54048a39a85440da1d3e",
      "4418f590a9407f1f2385d4435ee425d78db29442e928897b84dd082b0f92ff0f",
    ]);
    for (const identity of identities) expect(BROWSER_SOURCE).toContain(identity);
  });

  it("requires every tier non-slower and the frozen 1.25x weighted gate", () => {
    expect(summarizeOpt0051Timing(timingInput(125, 100))).toMatchObject({
      weightTotal: 2_397,
      weightedSpeedup: 1.25,
      everyTierNonSlower: true,
      requiredWeightedSpeedup: 1.25,
      passed: true,
    });
    const oneRegression = timingInput(125, 100).map((tier, index) =>
      index === 0
        ? { ...tier, rowReuseSamplesMilliseconds: [126, 126, 126, 126] }
        : tier
    );
    expect(summarizeOpt0051Timing(oneRegression)).toMatchObject({
      everyTierNonSlower: false,
      passed: false,
    });
    expect(summarizeOpt0051Timing(timingInput(125, 124))).toMatchObject({
      everyTierNonSlower: true,
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
    expect(parseOpt0051ThermalGate(parameters, 1_000, 32_100)).toMatchObject({
      durationMilliseconds: 30_055,
      observationCount: 1,
      observedLevel: 0,
      launchDelayMilliseconds: 45,
    });
    parameters.set("thermalObservations", "2");
    expect(() => parseOpt0051ThermalGate(parameters, 1_000, 32_100))
      .toThrow(/one truthful level-0/);
  });

  it("pins the lean exact/adversarial button gate and cleanup receipt", () => {
    for (const token of [
      "packAceOpt0051VaeK7WeightU16",
      "unpackAceOpt0051VaeK7WeightU16",
      "SIGNED_ZERO_PATTERN",
      "CANCELLATION_PATTERN",
      "FINITE_RANGE_PATTERN",
      'adversarialCase("tail-cin"',
      "packInverseMismatchCount",
      "rawU16Exact",
      "deterministicBothArms",
      "OUTPUT_PREFILL_QNAN_F16",
      "allOutputsFinite",
      "guardsUntouched",
      "tracker.destroyAll()",
      "weightedSpeedup",
      "REQUIRED_WEIGHTED_SPEEDUP = 1.25",
      "window.__ACE_OPT0051_RESULT__ = receipt",
    ]) expect(BROWSER_SOURCE).toContain(token);
    expect(BROWSER_SOURCE).not.toContain("workgroupBarrier");
    expect(BROWSER_HTML).toContain('id="run" type="button" disabled');
    expect(BROWSER_HTML).toContain(
      "notifyutil -g com.apple.system.thermalpressurelevel",
    );
    expect(BROWSER_HTML).toContain("exactly one external");
    expect(BROWSER_HTML).toContain("all four production channel tiers");
    expect(BROWSER_HTML).toContain("1.25×");
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
      throw new Error("OPT-0051 owner must not allocate");
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
  rowReuseSamplesMilliseconds: readonly number[];
}[] {
  return [282, 423, 423, 1_269].map((weight, index) => ({
    id: `tier-${index}`,
    weight,
    opt0024SamplesMilliseconds: [control, control, control, control],
    rowReuseSamplesMilliseconds: [candidate, candidate, candidate, candidate],
  }));
}

function sha256Text(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
