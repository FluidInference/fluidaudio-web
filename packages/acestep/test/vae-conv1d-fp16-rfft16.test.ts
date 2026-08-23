import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE,
  ACE_OPT_0077_RFFT16_REDUCTION_SEMANTICS,
  ACE_OPT_0077_RFFT16_STAGE_DISPATCH_COUNT,
  ACE_OPT_0077_RFFT16_SUBGROUP_SIZE,
  ACE_OPT_0077_RFFT16_TILES_PER_DOMAIN_SUBGROUP,
  ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES,
  ACE_OPT_0077_RFFT16_WORKGROUP_SIZE,
  ACE_OPT_0077_VAE_K7_RFFT16_KERNEL_ID,
  AceOpt0077VaeK7Rfft16Kernel,
  aceOpt0077Rfft16NativeWeightIndex,
  aceOpt0077Rfft16PackedWeightCoordinate,
  aceOpt0077Rfft16PackedWeightIndex,
  aceOpt0077VaeK7Rfft16Wgsl,
  planAceOpt0077VaeK7Rfft16Range,
  transformAceOpt0077VaeK7WeightU16,
  type AceOpt0077VaeK7Rfft16Bindings,
  type AceOpt0077VaeK7Rfft16ScratchBindings,
} from "../src/webgpu/kernels/vae-conv1d-fp16-rfft16.js";
import {
  ACE_OPT_0077_RFFT16_COORDINATE_ORDER,
  ACE_OPT_0077_RFFT16_WEIGHT_LAYOUT,
  aceOpt0077NumberToFloat16Bits,
} from "../src/webgpu/kernels/vae-conv1d-fp16-rfft16-math.js";
import {
  planAceFp16VaeConv1d,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";

describe("OPT-0077 K7 RFFT16 GPU owner", () => {
  it("pins the real-basis coordinate, reduction, and three-stage geometry", () => {
    expect(ACE_OPT_0077_VAE_K7_RFFT16_KERNEL_ID).toBe(
      "ace-opt-0077-vae-k7-rfft16-real-basis-overlap-save-v1",
    );
    expect(ACE_OPT_0077_RFFT16_REDUCTION_SEMANTICS).toBe(
      "real-basis-rfft16-fp16-dot4-partials-fp32-domain-and-inverse",
    );
    expect(ACE_OPT_0077_RFFT16_COORDINATE_ORDER).toEqual([
      "dc", "nyquist",
      "cos1", "sin1", "cos2", "sin2", "cos3", "sin3",
      "cos4", "sin4", "cos5", "sin5", "cos6", "sin6",
      "cos7", "sin7",
    ]);
    expect(ACE_OPT_0077_RFFT16_WEIGHT_LAYOUT).toBe(
      "coord16-cin4-cout-band128-subgroup4-lane32-cin-element4",
    );
    expect({
      coordinates: ACE_OPT_0077_RFFT16_WEIGHT_COORDINATES,
      outputs: ACE_OPT_0077_RFFT16_OUTPUTS_PER_TILE,
      subgroup: ACE_OPT_0077_RFFT16_SUBGROUP_SIZE,
      workgroup: ACE_OPT_0077_RFFT16_WORKGROUP_SIZE,
      tilesPerSubgroup: ACE_OPT_0077_RFFT16_TILES_PER_DOMAIN_SUBGROUP,
      stages: ACE_OPT_0077_RFFT16_STAGE_DISPATCH_COUNT,
      realDot4PerTenOutputs: 2 + 7 * 4,
    }).toEqual({
      coordinates: 16,
      outputs: 10,
      subgroup: 32,
      workgroup: 128,
      tilesPerSubgroup: 16,
      stages: 3,
      realDot4PerTenOutputs: 30,
    });
  });

  it("emits fixed32 register-only butterflies and a uniform-z contraction", () => {
    const source = aceOpt0077VaeK7Rfft16Wgsl(shape(1_024, 1_024, 512, 9));
    expect(source.match(/fn (?:forward|domain|inverse)_main\(/g)).toHaveLength(3);
    expect(source).toContain("enable f16;");
    expect(source).toContain("enable subgroups;");
    expect(source).toContain("const ENDPOINT_SCALE: f32 = 0.25;");
    expect(source).toContain("const PAIR_SCALE: f32 = 0.3535533845424652;");
    expect(source).toContain(
      "case 1u: { return vec2<f32>(0.9238795042037964, -0.3826834261417389); }",
    );
    expect(source).toContain("value.re * ENDPOINT_SCALE");
    expect(source).toContain("value.re * PAIR_SCALE");
    expect(source).toContain("-value.im * PAIR_SCALE");
    expect(source.match(/subgroupShuffleXor\(/g)).toHaveLength(4);
    expect(source.match(/var sum(?:[0-9]|1[0-5]): f32 = 0\.0;/g))
      .toHaveLength(16);
    expect(source.match(/subgroupBroadcast\(lane_x0, (?:[0-9]|1[0-5])u\)/g))
      .toHaveLength(16);
    expect(source.match(/subgroupBroadcast\(lane_x1, (?:[0-9]|1[0-5])u\)/g))
      .toHaveLength(16);
    expect(source).toContain("sum0 += f32(dot(x0_0, weight0));");
    expect(source).toContain("sum0 += f32(dot(x0_0, weight1));");
    expect(source).toContain("sum0 -= f32(dot(x1_0, weight0));");
    expect(source).toContain("value.re = 4.0 * contraction4(0u");
    expect(source).toContain("value.re = 2.0 * contraction4(coord");
    expect(source).toContain("if (frequency > 8u) { value.im = -value.im; }");
    expect(source).toContain("if (is_full_output_range()) {");
    expect(source).toContain("var remaining = tile;");
    expect(source).toContain("local_lane < 10u");
    expect(source).not.toContain("var<workgroup>");
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toMatch(/\bfma\s*\(/);
  });

  it("plans only intersecting globally anchored residue tiles and exact scratch", () => {
    for (const [dilation, expectedTiles] of [
      [1, 52], [3, 53], [9, 54],
    ] as const) {
      const plan = planAceFp16VaeConv1d(
        shape(1_024, 1_024, 512, dilation),
        "float16",
      );
      const range = planAceOpt0077VaeK7Rfft16Range(plan, {
        base: 0,
        count: plan.outputElements,
      });
      expect(range.tileCount).toBe(expectedTiles);
      expect(range.forwardWorkgroupsX).toBe(expectedTiles);
      expect(range.forwardWorkgroupsY).toBe(32);
      expect(range.domainWorkgroupsX).toBe(Math.ceil(expectedTiles / 16));
      expect(range.domainWorkgroupsY).toBe(8);
      expect(range.domainWorkgroupsZ).toBe(16);
      expect(range.inverseWorkgroupsX).toBe(expectedTiles);
      expect(range.inverseWorkgroupsY).toBe(32);
      expect(range.scratchBytes).toEqual({
        inputSpectrum: 16 * expectedTiles * 1_024 * 2,
        contractionSpectrum: 16 * expectedTiles * 1_024 * 4,
        total: 16 * expectedTiles * 1_024 * 6,
      });
    }

    const d9 = planAceFp16VaeConv1d(shape(128, 128, 512, 9), "float16");
    expect(planAceOpt0077VaeK7Rfft16Range(d9, {
      base: 95 * 128,
      count: 10 * 128,
    })).toMatchObject({
      firstOutputTime: 95,
      outputRowCount: 10,
      firstTileGroup: 1,
      tileGroupCount: 1,
      tileCount: 9,
    });

    const d3 = planAceFp16VaeConv1d(shape(128, 128, 512, 3), "float16");
    expect(planAceOpt0077VaeK7Rfft16Range(d3, {
      base: 2 * 128,
      count: 128,
    })).toMatchObject({
      firstOutputTime: 2,
      outputRowCount: 1,
      tileCount: 1,
      forwardWorkgroupsX: 1,
      domainWorkgroupsX: 1,
      inverseWorkgroupsX: 1,
    });
  });

  it("keeps the transformed layout bijective and finite", () => {
    const cin = 4;
    const cout = 128;
    const native = new Uint16Array(cin * cout * 7);
    const pattern = [0, 0.125, -0.125, 0.5, -0.5, 1, -1];
    for (let index = 0; index < native.length; index++) {
      native[index] = aceOpt0077NumberToFloat16Bits(pattern[index % pattern.length]!);
    }
    const packed = transformAceOpt0077VaeK7WeightU16(native, cin, cout);
    expect(packed).toHaveLength(cin * cout * 16);
    const seen = new Uint8Array(packed.length);
    for (let index = 0; index < packed.length; index++) {
      const coordinate = aceOpt0077Rfft16PackedWeightCoordinate(cin, cout, index);
      expect(aceOpt0077Rfft16PackedWeightIndex(cin, cout, coordinate)).toBe(index);
      seen[index] = 1;
    }
    expect(seen.every((value) => value === 1)).toBe(true);
    expect(aceOpt0077Rfft16NativeWeightIndex(cin, cout, {
      outputChannel: 127,
      kernel: 6,
      inputChannel: 3,
    })).toBe(native.length - 1);
  });

  it("caches three pipelines, dispatches exactly three stages, and owns no buffers", async () => {
    const device = fakeDevice();
    const kernel = AceOpt0077VaeK7Rfft16Kernel.create(device, FIXED_32);
    const candidateShape = shape(128, 128, 33, 3);
    const plan = planAceFp16VaeConv1d(candidateShape, "float16");
    const plannedRange = planAceOpt0077VaeK7Rfft16Range(plan, {
      base: 0,
      count: plan.outputElements,
    });
    const resources = resourcesFor(candidateShape, plannedRange.scratchBytes);
    const control = fakeBuffer(512);
    const first = await kernel.createDispatch(
      "first",
      candidateShape,
      resources.bindings,
      "float16",
      outputRange(control, 256, 0, plan.outputElements),
      resources.scratch,
    );
    const second = await kernel.createDispatch(
      "second",
      candidateShape,
      resources.bindings,
      "float16",
      outputRange(control, 0, 0, plan.outputElements),
      resources.scratch,
    );
    expect(first).toMatchObject({
      kernelId: ACE_OPT_0077_VAE_K7_RFFT16_KERNEL_ID,
      stageDispatchCount: 3,
      scratchBytes: plannedRange.scratchBytes,
    });
    expect(device.createBuffer).not.toHaveBeenCalled();
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(3);
    expect(device.createBindGroup).toHaveBeenCalledOnce();

    const pass = fakePass();
    first.encode(pass);
    second.encode(pass);
    expect(pass.setPipeline).toHaveBeenCalledTimes(6);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([[256], [0]]);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [6, 4, 1], [1, 1, 16], [6, 4, 1],
      [6, 4, 1], [1, 1, 16], [6, 4, 1],
    ]);
    kernel.destroy();
    kernel.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
    await expect(kernel.createDispatch(
      "post-destroy",
      candidateShape,
      resources.bindings,
      "float16",
      outputRange(control, 0, 0, plan.outputElements),
      resources.scratch,
    )).rejects.toThrow(/was destroyed/);
  });

  it("evicts a rejected three-pipeline compilation so the shape can retry", async () => {
    const device = fakeDevice({ rejectFirstPipeline: true });
    const kernel = AceOpt0077VaeK7Rfft16Kernel.create(device, FIXED_32);
    const candidateShape = shape(128, 128, 33, 3);
    const plan = planAceFp16VaeConv1d(candidateShape, "float16");
    const planned = planAceOpt0077VaeK7Rfft16Range(plan, {
      base: 0,
      count: plan.outputElements,
    });
    const resources = resourcesFor(candidateShape, planned.scratchBytes);
    const control = fakeBuffer(256);
    const create = () => kernel.createDispatch(
      "retry",
      candidateShape,
      resources.bindings,
      "float16",
      outputRange(control, 0, 0, plan.outputElements),
      resources.scratch,
    );

    await expect(create()).rejects.toThrow(/synthetic pipeline rejection/);
    expect(device.createBuffer).not.toHaveBeenCalled();
    expect(device.createBindGroup).not.toHaveBeenCalled();
    await expect(create()).resolves.toMatchObject({ stageDispatchCount: 3 });
    expect(device.createBuffer).not.toHaveBeenCalled();
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(6);
  });

  it("fails closed outside the selected fixed32 K7 boundary", async () => {
    expect(() => AceOpt0077VaeK7Rfft16Kernel.create(
      fakeDevice({ shaderF16: false }),
      FIXED_32,
    )).toThrow(/shader-f16/);
    expect(() => AceOpt0077VaeK7Rfft16Kernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32-lane/);
    const kernel = AceOpt0077VaeK7Rfft16Kernel.create(fakeDevice(), FIXED_32);
    const control = fakeBuffer(256);

    const invalidCout = shape(128, 256 + 64, 33, 1);
    const invalidPlan = planAceFp16VaeConv1d(invalidCout, "float16");
    const loose = resourcesFor(invalidCout, {
      inputSpectrum: 1 << 20,
      contractionSpectrum: 1 << 20,
    });
    await expect(kernel.createDispatch(
      "invalid-cout",
      invalidCout,
      loose.bindings,
      "float16",
      outputRange(control, 0, 0, invalidPlan.outputElements),
      loose.scratch,
    )).rejects.toThrow(/Cout divisible by 128/);

    const valid = shape(128, 128, 33, 1);
    const plan = planAceFp16VaeConv1d(valid, "float16");
    const planned = planAceOpt0077VaeK7Rfft16Range(plan, {
      base: 0,
      count: plan.outputElements,
    });
    const resources = resourcesFor(valid, planned.scratchBytes);
    await expect(kernel.createDispatch(
      "fp32-output",
      valid,
      resources.bindings,
      "float32",
      outputRange(control, 0, 0, plan.outputElements),
      resources.scratch,
    )).rejects.toThrow(/FP16 output/);
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
    padding: 3 * dilation,
  };
}

function resourcesFor(
  candidateShape: AceVaeConv1dShape,
  scratchBytes: Readonly<{
    inputSpectrum: number;
    contractionSpectrum: number;
  }>,
): Readonly<{
  bindings: AceOpt0077VaeK7Rfft16Bindings;
  scratch: AceOpt0077VaeK7Rfft16ScratchBindings;
}> {
  const plan = planAceFp16VaeConv1d(candidateShape, "float16");
  return {
    bindings: {
      input: fakeBinding(plan.inputBindingBytes),
      transformedWeight: fakeBinding(
        16 * plan.inputChannels * plan.outputChannels * 2,
      ),
      bias: fakeBinding(plan.biasBindingBytes),
      output: fakeBinding(plan.outputBindingBytes),
    },
    scratch: {
      inputSpectrum: fakeBinding(scratchBytes.inputSpectrum),
      contractionSpectrum: fakeBinding(scratchBytes.contractionSpectrum),
    },
  };
}

function outputRange(
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
  readonly rejectFirstPipeline?: boolean;
} = {}): GPUDevice & {
  readonly createBuffer: ReturnType<typeof vi.fn>;
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
} {
  let pipelineCalls = 0;
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
    createComputePipelineAsync: vi.fn(async () => {
      pipelineCalls += 1;
      if (options.rejectFirstPipeline === true && pipelineCalls === 1) {
        throw new Error("synthetic pipeline rejection");
      }
      return {};
    }),
    createBindGroup: vi.fn(() => ({})),
    createBuffer: vi.fn(() => {
      throw new Error("OPT-0077 owner must not allocate");
    }),
  } as unknown as ReturnType<typeof fakeDevice>;
}

function fakePass(): GPUComputePassEncoder & {
  readonly setPipeline: ReturnType<typeof vi.fn>;
  readonly setBindGroup: ReturnType<typeof vi.fn>;
  readonly dispatchWorkgroups: ReturnType<typeof vi.fn>;
} {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
  } as unknown as ReturnType<typeof fakePass>;
}
