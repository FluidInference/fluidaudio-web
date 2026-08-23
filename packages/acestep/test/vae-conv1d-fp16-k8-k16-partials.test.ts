import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0041_VAE_K7_K8_PARTIALS_KERNEL_ID,
  ACE_OPT_0041_VAE_K7_K16_PARTIALS_KERNEL_ID,
  ACE_OPT_0041_VAE_K7_NATIVE_WEIGHT_LAYOUT,
  AceOpt0041VaeK7BoundedPartialsKernel,
  aceOpt0041VaeK7BoundedPartialsWgsl,
  planAceOpt0041VaeK7BoundedPartials,
  planAceOpt0041VaeK7BoundedPartialsRange,
} from "../src/webgpu/kernels/vae-conv1d-fp16-k8-k16-partials.js";
import { planAceFp16VaeConv1d } from
  "../src/webgpu/kernels/vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";
import {
  buildOpt0041Cases,
  parseThermalGate,
} from "./browser/opt-0041-vae-k7-fp16-k8-k16-partials.js";

const BROWSER_SOURCE = readFileSync(new URL(
  "./browser/opt-0041-vae-k7-fp16-k8-k16-partials.ts",
  import.meta.url,
), "utf8");
const BROWSER_HTML = readFileSync(new URL(
  "./browser/opt-0041-vae-k7-fp16-k8-k16-partials.html",
  import.meta.url,
), "utf8");

describe("OPT-0041 VAE K7 bounded FP16 K8/K16 partials", () => {
  it.each([
    ["k8" as const, 8, 2, 128],
    ["k16" as const, 16, 4, 64],
  ])(
    "plans %s over OPT-0024's native production geometry",
    (variant, partialInner, groups, blocks) => {
      const plan = planAceOpt0041VaeK7BoundedPartials(
        shape(1_024, 1_024, 300, 1),
        "float16",
        variant,
      );
      expect(plan).toMatchObject({
        family: "k7",
        variant,
        inputChannels: 1_024,
        outputChannels: 1_024,
        partialInner,
        cin4GroupsPerPartial: groups,
        inputChannelPartialBlocks: blocks,
        finalPartialCin4Groups: groups,
        weightLayout: ACE_OPT_0041_VAE_K7_NATIVE_WEIGHT_LAYOUT,
      });
    },
  );

  it("plans and guards final short Cin groups without changing native bytes", () => {
    const k8 = planAceOpt0041VaeK7BoundedPartials(
      shape(68, 128, 33, 3),
      "float16",
      "k8",
    );
    const k16 = planAceOpt0041VaeK7BoundedPartials(
      shape(68, 128, 33, 3),
      "float16",
      "k16",
    );
    expect(k8).toMatchObject({
      inputChannelPartialBlocks: 9,
      finalPartialCin4Groups: 1,
      weightStorageBytes: 68 * 128 * 7 * 2,
    });
    expect(k16).toMatchObject({
      inputChannelPartialBlocks: 5,
      finalPartialCin4Groups: 1,
      weightStorageBytes: 68 * 128 * 7 * 2,
    });
    for (const variant of ["k8", "k16"] as const) {
      const source = aceOpt0041VaeK7BoundedPartialsWgsl(
        shape(68, 128, 33, 3),
        true,
        "float16",
        variant,
      );
      expect(source).toContain("const INPUT_CHANNEL_VEC4S: u32 = 17u;");
      expect(source).toContain("< INPUT_CHANNEL_VEC4S) {");
      expect(source).not.toContain("packed_weight");
      expect(source).not.toContain("repack");
    }
  });

  it("emits K8 local FP16 pairs and one FP32 update per bounded block", () => {
    const source = aceOpt0041VaeK7BoundedPartialsWgsl(
      shape(512, 512, 300, 3),
      true,
      "float16",
      "k8",
    );
    expect(ACE_OPT_0041_VAE_K7_K8_PARTIALS_KERNEL_ID).toMatch(
      /native-oki-fp16-k8-partials-k7-conv1d-v1$/,
    );
    expect(source).toContain("const INPUT_CHANNEL_PARTIAL_BLOCKS: u32 = 64u;");
    expect(source).toContain("const CIN4_GROUPS_PER_PARTIAL: u32 = 2u;");
    expect(source.match(/var partial[0-7]: vec4<f16>;/g)).toHaveLength(8);
    expect(source.match(/partial0 = vec4<f16>\(/g)).toHaveLength(1);
    expect(source.match(/partial0 = partial0 \+ vec4<f16>\(/g)).toHaveLength(1);
    expect(source.match(/sum0 = sum0 \+ vec4<f32>\(partial0\);/g))
      .toHaveLength(1);
    expect(source.match(/subgroupBroadcast\(lane_input_[01], 0u\)/g))
      .toHaveLength(2);
    expect(source).toContain("var sum0: vec4<f32> = initial_sum;");
    expect(source).not.toContain("var<workgroup>");
    expect(source).not.toContain("workgroupBarrier");
  });

  it("emits K16 local FP16 quartets and no FP16 inter-block accumulator", () => {
    const source = aceOpt0041VaeK7BoundedPartialsWgsl(
      shape(256, 256, 300, 9),
      true,
      "float16",
      "k16",
    );
    expect(ACE_OPT_0041_VAE_K7_K16_PARTIALS_KERNEL_ID).toMatch(
      /native-oki-fp16-k16-partials-k7-conv1d-v1$/,
    );
    expect(source).toContain("const CIN4_GROUPS_PER_PARTIAL: u32 = 4u;");
    expect(source.match(/partial7 = vec4<f16>\(/g)).toHaveLength(1);
    expect(source.match(/partial7 = partial7 \+ vec4<f16>\(/g))
      .toHaveLength(3);
    expect(source.match(/subgroupBroadcast\(lane_input_[0-3], 7u\)/g))
      .toHaveLength(4);
    expect(source.match(/sum7 = sum7 \+ vec4<f32>\(partial7\);/g))
      .toHaveLength(1);
    expect(source).not.toMatch(/var sum\d+: vec4<f16>/);
  });

  it("reuses OPT-0024's exact range ownership", () => {
    const base = planAceFp16VaeConv1d(
      shape(128, 128, 300, 9),
      "float16",
    );
    expect(planAceOpt0041VaeK7BoundedPartialsRange(base, {
      base: 128 * 64,
      count: 128 * 32,
    })).toMatchObject({
      base: 128 * 64,
      count: 128 * 32,
      firstOutputRow: 64,
      outputRowCount: 32,
      workgroupsX: 1,
      workgroupsY: 1,
    });
  });

  it("creates allocation-free separated owners, caches, and rejects after destroy", async () => {
    for (const variant of ["k8", "k16"] as const) {
      const device = fakeDevice();
      const kernel = AceOpt0041VaeK7BoundedPartialsKernel.create(
        device,
        FIXED_32,
        variant,
      );
      const candidateShape = shape(128, 128, 33, 1);
      const plan = planAceFp16VaeConv1d(candidateShape, "float16");
      const bindings = bindingsFor(candidateShape);
      const control = fakeBuffer(512);
      const first = await kernel.createDispatch(
        "first",
        candidateShape,
        bindings,
        "float16",
        range(control, 256, 0, 128 * 32),
      );
      const tail = await kernel.createDispatch(
        "tail",
        candidateShape,
        bindings,
        "float16",
        range(control, 0, 128 * 32, 128),
      );
      expect(first.kernelId).toBe(variant === "k8"
        ? ACE_OPT_0041_VAE_K7_K8_PARTIALS_KERNEL_ID
        : ACE_OPT_0041_VAE_K7_K16_PARTIALS_KERNEL_ID);
      expect(device.createBuffer).not.toHaveBeenCalled();
      expect(device.createShaderModule).toHaveBeenCalledOnce();
      expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
      expect(device.createBindGroup).toHaveBeenCalledOnce();
      const pass = fakePass();
      first.encode(pass);
      tail.encode(pass);
      expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
        .toEqual([[256], [0]]);
      expect(pass.dispatchWorkgroups.mock.calls).toEqual([
        [1, 1, 1],
        [1, 1, 1],
      ]);
      const layout = device.createBindGroupLayout.mock.calls[0]![0] as
        GPUBindGroupLayoutDescriptor;
      expect(Array.from(layout.entries).map(({ buffer }) => ({
        type: buffer?.type,
        dynamic: buffer?.hasDynamicOffset ?? false,
        bytes: buffer?.minBindingSize,
      }))).toEqual([
        { type: "read-only-storage", dynamic: false, bytes: plan.inputBindingBytes },
        { type: "read-only-storage", dynamic: false, bytes: plan.weightBindingBytes },
        { type: "read-only-storage", dynamic: false, bytes: plan.biasBindingBytes },
        { type: "storage", dynamic: false, bytes: plan.outputBindingBytes },
        { type: "uniform", dynamic: true, bytes: 16 },
      ]);
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
    }
  });

  it("evicts failed compilation and fails closed outside the frozen boundary", async () => {
    const device = fakeDevice({
      compilationMessages: [[{
        type: "error",
        lineNum: 55,
        linePos: 9,
        message: "synthetic bounded-partial diagnostic",
      }], []],
    });
    const kernel = AceOpt0041VaeK7BoundedPartialsKernel.create(
      device,
      FIXED_32,
      "k8",
    );
    const candidateShape = shape(128, 128, 33, 1);
    const bindings = bindingsFor(candidateShape);
    const dispatchRange = range(
      fakeBuffer(256),
      0,
      0,
      33 * 128,
    );
    await expect(kernel.createDispatch(
      "first",
      candidateShape,
      bindings,
      "float16",
      dispatchRange,
    )).rejects.toThrow(/55:9 synthetic bounded-partial diagnostic/);
    await expect(kernel.createDispatch(
      "second",
      candidateShape,
      bindings,
      "float16",
      dispatchRange,
    )).resolves.toBeDefined();
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);

    expect(() => AceOpt0041VaeK7BoundedPartialsKernel.create(
      fakeDevice({ shaderF16: false }),
      FIXED_32,
      "k8",
    )).toThrow(/shader-f16/);
    expect(() => AceOpt0041VaeK7BoundedPartialsKernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
      "k8",
    )).toThrow(/fixed 32-lane/);
    expect(() => planAceOpt0041VaeK7BoundedPartials(
      { ...candidateShape, kernelSize: 1, padding: 0 },
      "float16",
      "k8",
    )).toThrow(/only K7/);
    expect(() => planAceOpt0041VaeK7BoundedPartials(
      shape(130, 128, 33, 1),
      "float16",
      "k8",
    )).toThrow(/Cin divisible by 4/);
    expect(() => planAceOpt0041VaeK7BoundedPartials(
      shape(128, 192, 33, 1),
      "float16",
      "k8",
    )).toThrow(/Cout divisible by 128/);
    expect(() => planAceOpt0041VaeK7BoundedPartials(
      candidateShape,
      "float32",
      "k8",
    )).toThrow(/FP16 internal output/);
    expect(() => planAceOpt0041VaeK7BoundedPartials(
      candidateShape,
      "float16",
      "k32" as never,
    )).toThrow(/unknown/);
  });

  it("pins a three-arm full/adversarial gate and the simple thermal form", () => {
    expect(BROWSER_SOURCE).toContain('type Arm = "k4" | "k8" | "k16"');
    for (const tier of [
      '"c1024"',
      '"c512"',
      '"c256"',
      '"c128"',
    ]) expect(BROWSER_SOURCE).toContain(tier);
    for (const fixture of [
      '"signed-zero"',
      '"cancellation"',
      '"finite-range"',
      '"tail-cin"',
    ]) expect(BROWSER_SOURCE).toContain(fixture);
    expect(BROWSER_SOURCE).toContain("OUTPUT_PREFILL_QNAN_F16");
    expect(BROWSER_SOURCE).toContain("deterministicRawU16");
    expect(BROWSER_SOURCE).toContain("completeWrites");
    expect(BROWSER_SOURCE).toContain("allOutputsFinite");
    expect(BROWSER_SOURCE).toContain("nrmse");
    expect(BROWSER_SOURCE).toContain("snrDecibels");
    expect(BROWSER_SOURCE).toContain("pearsonCorrelation");
    expect(BROWSER_SOURCE).toContain("REQUIRED_SPEEDUP_OVER_K4 = 1.15");
    expect(BROWSER_SOURCE).toContain("weightedSpeedupOverK4");
    expect(BROWSER_SOURCE).toContain("window.__ACE_OPT0041_RESULT__ = receipt");
    expect(BROWSER_SOURCE).not.toContain("workgroupBarrier");
    expect(BROWSER_HTML).toContain('id="run" type="button" disabled');
    expect(BROWSER_HTML).toContain("Wait exactly 30 seconds");
    expect(BROWSER_HTML).toContain(
      "notifyutil -g com.apple.system.thermalpressurelevel",
    );
    expect(BROWSER_HTML).toContain("one level-0 check");
    expect(BROWSER_HTML).toContain(
      "opt-0041-vae-k7-fp16-k8-k16-partials.ts",
    );
  });

  it("binds the exact four production tiers and four adversarial fixtures", () => {
    const cases = buildOpt0041Cases();
    expect(cases.map(({ id, kind, tier, timingWeight, shape, probes }) => ({
      id,
      kind,
      tier,
      timingWeight,
      frames: shape.inputFrames,
      channels: shape.inputChannels,
      dilation: shape.dilation,
      probes: probes.map(({ id }) => id),
    }))).toEqual([
      {
        id: "block-0-res-1-conv1",
        kind: "production",
        tier: "c1024",
        timingWeight: 282,
        frames: 3_000,
        channels: 1_024,
        dilation: 1,
        probes: ["first", "interior", "tail"],
      },
      {
        id: "block-1-res-2-conv1",
        kind: "production",
        tier: "c512",
        timingWeight: 423,
        frames: 18_000,
        channels: 512,
        dilation: 3,
        probes: ["first", "interior", "tail"],
      },
      {
        id: "block-2-res-1-conv1",
        kind: "production",
        tier: "c256",
        timingWeight: 423,
        frames: 72_000,
        channels: 256,
        dilation: 1,
        probes: ["first", "interior", "tail"],
      },
      {
        id: "block-4-res-3-conv1",
        kind: "production",
        tier: "c128",
        timingWeight: 1_269,
        frames: 576_000,
        channels: 128,
        dilation: 9,
        probes: ["first", "interior", "tail"],
      },
      {
        id: "signed-zero",
        kind: "signed-zero",
        tier: undefined,
        timingWeight: undefined,
        frames: 33,
        channels: 64,
        dilation: 1,
        probes: ["full"],
      },
      {
        id: "cancellation",
        kind: "cancellation",
        tier: undefined,
        timingWeight: undefined,
        frames: 35,
        channels: 128,
        dilation: 3,
        probes: ["full"],
      },
      {
        id: "finite-range",
        kind: "finite-range",
        tier: undefined,
        timingWeight: undefined,
        frames: 37,
        channels: 256,
        dilation: 9,
        probes: ["full"],
      },
      {
        id: "tail-cin",
        kind: "tail-cin",
        tier: undefined,
        timingWeight: undefined,
        frames: 39,
        channels: 68,
        dilation: 3,
        probes: ["full"],
      },
    ]);
  });

  it("accepts only one level-0 check after a fresh 30-second wait", () => {
    const valid = new URLSearchParams({
      thermalCommand: "notifyutil -g com.apple.system.thermalpressurelevel",
      waitStartedAtEpochMilliseconds: "1100",
      checkedAtEpochMilliseconds: "31100",
      checkCount: "1",
      thermalLevel: "0",
    });
    expect(parseThermalGate(valid, 1_000, 32_000)).toEqual({
      command: "notifyutil -g com.apple.system.thermalpressurelevel",
      waitStartedAtEpochMilliseconds: 1_100,
      checkedAtEpochMilliseconds: 31_100,
      waitDurationMilliseconds: 30_000,
      checkCount: 1,
      thermalLevel: 0,
      launchDelayMilliseconds: 900,
    });
    for (const override of [
      { checkedAtEpochMilliseconds: "31099" },
      { checkCount: "2" },
      { thermalLevel: "1" },
      { thermalCommand: "polling logger" },
    ]) {
      const rejected = new URLSearchParams(valid);
      for (const [key, value] of Object.entries(override)) {
        rejected.set(key, value);
      }
      expect(() => parseThermalGate(rejected, 1_000, 32_000))
        .toThrow(/exactly one level-0 notifyutil check/);
    }
  });
});

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const FIXED_32 = Object.freeze({
  subgroupMinSize: 32,
  subgroupMaxSize: 32,
});

function shape(
  inputChannels: number,
  outputChannels: number,
  frames: number,
  dilation: 1 | 3 | 9,
): AceVaeConv1dShape {
  return {
    batch: 1,
    inputFrames: frames,
    inputChannels,
    outputChannels,
    kernelSize: 7,
    stride: 1,
    dilation,
    padding: dilation * 3,
  };
}

function bindingsFor(candidateShape: AceVaeConv1dShape) {
  const plan = planAceFp16VaeConv1d(candidateShape, "float16");
  return {
    input: fakeBinding(plan.inputBindingBytes),
    weight: fakeBinding(plan.weightBindingBytes),
    bias: fakeBinding(plan.biasBindingBytes),
    output: fakeBinding(plan.outputBindingBytes),
  };
}

function range(
  buffer: GPUBuffer,
  offset: number,
  base: number,
  count: number,
): AceVaeOutputRangeBinding {
  return { base, count, control: { buffer, offset, size: 16 } };
}

type FakeDevice = GPUDevice & Readonly<{
  createShaderModule: ReturnType<typeof vi.fn>;
  createBindGroupLayout: ReturnType<typeof vi.fn>;
  createPipelineLayout: ReturnType<typeof vi.fn>;
  createComputePipelineAsync: ReturnType<typeof vi.fn>;
  createBindGroup: ReturnType<typeof vi.fn>;
  createBuffer: ReturnType<typeof vi.fn>;
}>;

function fakeDevice(options: Readonly<{
  shaderF16?: boolean;
  subgroups?: boolean;
  maximumInvocations?: number;
  maximumWorkgroupSizeX?: number;
  compilationMessages?: readonly (readonly Partial<GPUCompilationMessage>[])[];
}> = {}): FakeDevice {
  const messages = [...(options.compilationMessages ?? [[]])];
  return {
    features: new Set([
      ...(options.shaderF16 === false ? [] : ["shader-f16"]),
      ...(options.subgroups === false ? [] : ["subgroups"]),
    ]),
    limits: {
      maxComputeInvocationsPerWorkgroup: options.maximumInvocations ?? 256,
      maxComputeWorkgroupSizeX: options.maximumWorkgroupSizeX ?? 256,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxStorageBufferBindingSize: 1_073_741_824,
      maxBufferSize: 1_073_741_824,
      minStorageBufferOffsetAlignment: 256,
      minUniformBufferOffsetAlignment: 256,
    },
    createShaderModule: vi.fn(() => ({
      getCompilationInfo: vi.fn(async () => ({
        messages: messages.shift() ?? [],
      })),
    })),
    createBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    createPipelineLayout: vi.fn(() => ({ label: "pipeline-layout" })),
    createComputePipelineAsync: vi.fn(async () => ({ label: "pipeline" })),
    createBindGroup: vi.fn(() => ({ label: "bind-group" })),
    createBuffer: vi.fn(() => {
      throw new Error("OPT-0041 candidate must not allocate");
    }),
  } as unknown as FakeDevice;
}

function fakePass(): GPUComputePassEncoder & Readonly<{
  setBindGroup: ReturnType<typeof vi.fn>;
  dispatchWorkgroups: ReturnType<typeof vi.fn>;
}> {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
  } as unknown as GPUComputePassEncoder & Readonly<{
    setBindGroup: ReturnType<typeof vi.fn>;
    dispatchWorkgroups: ReturnType<typeof vi.fn>;
  }>;
}

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: fakeBuffer(size), offset: 0, size };
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}
