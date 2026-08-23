import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_CIN_VECTOR_WIDTH,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_OUTPUTS_PER_LANE,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_REDUCTION_SEMANTICS,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_ROWS,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_SIZE,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_CHANNELS,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_ROWS,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_WORKGROUP_SIZE,
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUPS_PER_WORKGROUP,
  AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
  aceOpt0024VaeConv1dDirectDot4SubgroupWgsl,
  planAceOpt0024VaeConv1dDirectDot4SubgroupRange,
} from "../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dBindings,
  type AceFp16VaeConv1dOutputStorage,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import { planAceFp16VaeConv1dSubgroupRange } from
  "../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderConvOperation,
} from "../src/webgpu/vae-decoder.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";

describe("OPT-0024 direct-native-OKI FP16-dot4 subgroup K7 Conv1D", () => {
  it("pins the shipped fixed32 ownership with one Cin4 reduction geometry", () => {
    expect(ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID).toBe(
      "ace-vae-fp16-opt-0024-direct-native-oki-fp16-dot4-k7-conv1d-v1",
    );
    expect(
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_REDUCTION_SEMANTICS,
    ).toBe("increasing-k-cin4-fp16-dot4-partials-fp32-accumulator");
    expect(ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_SIZE).toBe(32);
    expect(ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_WORKGROUP_SIZE)
      .toBe(128);
    expect(ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUPS_PER_WORKGROUP)
      .toBe(4);
    expect(ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_ROWS).toBe(8);
    expect(ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_OUTPUTS_PER_LANE)
      .toBe(4);
    expect(ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_ROWS).toBe(32);
    expect(ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_CHANNELS)
      .toBe(128);
    expect(ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_CIN_VECTOR_WIDTH)
      .toBe(4);
    expect(
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_SIZE *
        ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_OUTPUTS_PER_LANE,
    ).toBe(ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_CHANNELS);
    expect(
      ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUPS_PER_WORKGROUP *
        ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_ROWS,
    ).toBe(ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_TILE_ROWS);
  });

  it("emits only the frozen native vec4, broadcast, FP16-dot4 mechanism", () => {
    const source = aceOpt0024VaeConv1dDirectDot4SubgroupWgsl(
      k7Shape(1_024, 1_024, 300, 9),
      true,
      "float16",
    );
    expect(source).toContain(
      `// kernel-id: ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID}`,
    );
    expect(source).toContain(
      `// reduction-semantics: ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_REDUCTION_SEMANTICS}`,
    );
    expect(source).toContain("enable f16;");
    expect(source).toContain("enable subgroups;");
    expect(source).toContain("const INPUT_CHANNEL_VEC4S: u32 = 256u;");
    expect(source).toContain(
      "@group(0) @binding(0) var<storage, read> input: array<vec4<f16>>;",
    );
    expect(source).toContain(
      "@group(0) @binding(1) var<storage, read> weight: array<vec4<f16>>;",
    );
    expect(source).toContain(
      "@group(0) @binding(3) var<storage, read_write> output: array<f16>;",
    );
    expect(source).toContain("@compute @workgroup_size(\n  128,\n  1,\n  1,");
    expect(source).toContain("if (subgroup_size == 32u)");
    expect(source).toContain("for (var kernel = 0u; kernel < 7u;");
    expect(source).toContain("var input_channel4 = 0u;");
    expect(source).toContain("input_channel4 < INPUT_CHANNEL_VEC4S;");
    expect(source).toContain(
      "(batch * INPUT_FRAMES + lane_input_time) *\n                  INPUT_CHANNEL_VEC4S + input_channel4",
    );
    expect(source).toContain(
      "(output_channel_base * 7u + kernel) * INPUT_CHANNEL_VEC4S +",
    );

    const broadcasts = source.match(
      /subgroupBroadcast\(lane_input, [0-7]u\)/g,
    ) ?? [];
    expect(broadcasts).toEqual(Array.from(
      { length: 8 },
      (_, row) => `subgroupBroadcast(lane_input, ${row}u)`,
    ));
    expect(source.match(/let weight[0-3] = weight\[/g)).toHaveLength(4);
    expect(source.match(/\bdot\(input_operand[0-7], weight[0-3]\)/g))
      .toHaveLength(32);
    expect(source.match(/let partial[0-7] = vec4<f16>\(/g)).toHaveLength(8);
    expect(source.match(/vec4<f32>\(partial[0-7]\)/g)).toHaveLength(8);
    expect(source.match(/sum[0-7] = sum[0-7] \+ vec4<f32>\(partial[0-7]\)/g))
      .toHaveLength(8);
    expect(source.match(/if \(input_valid[0-7]\) \{/g)).toHaveLength(8);
    expect(source.match(/= f16\(sum[0-7]\.[xyzw]\);/g)).toHaveLength(32);

    const cin4 = source.indexOf("var input_channel4 = 0u;");
    const broadcast0 = source.indexOf(
      "subgroupBroadcast(lane_input, 0u)",
      cin4,
    );
    const weight0 = source.indexOf("let weight0 = weight[", cin4);
    const partial0 = source.indexOf("let partial0 = vec4<f16>(", cin4);
    const add0 = source.indexOf(
      "sum0 = sum0 + vec4<f32>(partial0);",
      cin4,
    );
    expect(cin4).toBeLessThan(broadcast0);
    expect(broadcast0).toBeLessThan(weight0);
    expect(weight0).toBeLessThan(partial0);
    expect(partial0).toBeLessThan(add0);
    expect(source).not.toContain("var<workgroup>");
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toContain("packed_weight");
    expect(source).not.toContain("repack");
    expect(source).not.toMatch(/\bfma\s*\(/);
  });

  it("matches the shipped range planner and pins complete registered topology", () => {
    const expectedByFrames = new Map([
      [300, {
        operations: 16,
        ranges: 2_399,
        physicalWorkgroups: 103_672,
        logicalMultiplyAccumulates: 561_787_699_200,
      }],
      [512, {
        operations: 16,
        ranges: 4_082,
        physicalWorkgroups: 176_896,
        logicalMultiplyAccumulates: 958_784_339_968,
      }],
      [448, {
        operations: 16,
        ranges: 3_572,
        physicalWorkgroups: 154_784,
        logicalMultiplyAccumulates: 838_936_297_472,
      }],
      [340, {
        operations: 16,
        ranges: 2_720,
        physicalWorkgroups: 117_500,
        logicalMultiplyAccumulates: 636_692_725_760,
      }],
    ] as const);
    const summaries = new Map<number, CandidateScope>();
    for (const [frames, expected] of expectedByFrames) {
      const summary = candidateScope(frames);
      summaries.set(frames, summary);
      expect(pickTopology(summary)).toEqual(expected);
    }

    const long = addScopes(
      summaries.get(448)!,
      scaleScope(summaries.get(512)!, 10),
      summaries.get(340)!,
    );
    expect(long).toEqual({
      operations: 192,
      ranges: 47_112,
      physicalWorkgroups: 2_041_244,
      logicalMultiplyAccumulates: 11_063_472_422_912,
      workgroupKcin4Instances: 675_299_072,
      validInputVec4Loads: 21_603_155_968,
      nativeWeightVec4Loads: 345_753_124_864,
      subgroupBroadcastCollectives: 21_609_570_304,
      broadcastInvocationCalls: 691_506_249_728,
      fp16Dot4Calls: 2_765_203_963_904,
      fp32Vec4AccumulatorAdds: 691_300_990_976,
    });
    expect(long.validInputVec4Loads * 8).toBe(172_825_247_744);
    expect(long.nativeWeightVec4Loads * 8).toBe(2_766_024_998_912);
    expect(
      (long.validInputVec4Loads + long.nativeWeightVec4Loads) * 8,
    ).toBe(2_938_850_246_656);
  });

  it("creates one allocation-free owner with cached pipelines, bindings, and dynamic controls", async () => {
    const device = fakeDevice();
    const kernel = AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      device,
      FIXED_32_CAPABILITY,
    );
    const shape = k7Shape(1_024, 1_024, 300, 3);
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
      label: "first",
      kernelId: ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
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
    expect(device.createBuffer).not.toHaveBeenCalled();
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();

    const layout = device.createBindGroupLayout.mock.calls[0]?.[0] as
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
    const bindGroup = device.createBindGroup.mock.calls[0]?.[0] as
      GPUBindGroupDescriptor;
    expect(Array.from(bindGroup.entries)[4]?.resource).toMatchObject({
      buffer: control,
      offset: 0,
      size: 16,
    });

    const pass = fakePass();
    first.encode(pass);
    second.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([[256], [512]]);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [1, 8, 1],
      [1, 8, 1],
    ]);

    const otherBindings = bindingsFor(shape, "float16", true);
    await kernel.createDispatch(
      "other-bindings",
      shape,
      otherBindings,
      "float16",
      fullRange(fakeBuffer(256), 0, shape),
    );
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledTimes(2);

    const otherShape = k7Shape(512, 512, 300, 1);
    await kernel.createDispatch(
      "other-shape",
      otherShape,
      bindingsFor(otherShape, "float16", true),
      "float16",
      fullRange(fakeBuffer(256), 0, otherShape),
    );
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);

    kernel.destroy();
    kernel.destroy();
    expect(() => first.encode(pass)).toThrow(/was destroyed/);
    await expect(kernel.createDispatch(
      "post-destroy",
      shape,
      bindings,
      "float16",
      fullRange(control, 0, shape),
    )).rejects.toThrow(/was destroyed/);
  });

  it("evicts rejected compilation promises and reports exact diagnostics", async () => {
    const device = fakeDevice({
      compilationMessageBatches: [[{
        message: "synthetic direct-dot4 diagnostic",
        type: "error",
        lineNum: 91,
        linePos: 7,
      }], []],
    });
    const kernel = AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      device,
      FIXED_32_CAPABILITY,
    );
    const shape = k7Shape(128, 128, 17, 1);
    const bindings = bindingsFor(shape, "float16", true);
    const range = fullRange(fakeBuffer(256), 0, shape);
    await expect(kernel.createDispatch(
      "diagnostic",
      shape,
      bindings,
      "float16",
      range,
    )).rejects.toThrow(/91:7 synthetic direct-dot4 diagnostic/);
    await expect(kernel.createDispatch(
      "retry",
      shape,
      bindings,
      "float16",
      range,
    )).resolves.toMatchObject({
      kernelId: ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
    });
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
  });

  it("fails closed outside the exact capability and candidate shape boundary", async () => {
    expect(() => AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      fakeDevice({ shaderF16: false }),
      FIXED_32_CAPABILITY,
    )).toThrow(/requires WebGPU shader-f16/);
    expect(() => AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      fakeDevice({ subgroups: false }),
      FIXED_32_CAPABILITY,
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32-lane subgroups/);
    expect(() => AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      fakeDevice({ maximumInvocations: 127 }),
      FIXED_32_CAPABILITY,
    )).toThrow(/WG128/);
    expect(() => AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      fakeDevice({ maximumWorkgroupSizeX: 127 }),
      FIXED_32_CAPABILITY,
    )).toThrow(/WG128/);

    const kernel = AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      fakeDevice(),
      FIXED_32_CAPABILITY,
    );
    const k1: AceVaeConv1dShape = {
      batch: 1,
      inputFrames: 17,
      inputChannels: 128,
      outputChannels: 128,
      kernelSize: 1,
      stride: 1,
      dilation: 1,
      padding: 0,
    };
    await expect(kernel.createDispatch(
      "k1",
      k1,
      bindingsFor(k1, "float16", true),
      "float16",
      fullRange(fakeBuffer(256), 0, k1),
    )).rejects.toThrow(/supports only K7/);

    const candidate = k7Shape(128, 128, 17, 1);
    await expect(kernel.createDispatch(
      "no-bias",
      candidate,
      bindingsFor(candidate, "float32", false),
      "float32",
      fullRange(fakeBuffer(256), 0, candidate),
    )).rejects.toThrow(/requires bias/);
    await expect(kernel.createDispatch(
      "fp32",
      candidate,
      bindingsFor(candidate, "float32", true),
      "float32",
      fullRange(fakeBuffer(256), 0, candidate),
    )).rejects.toThrow(/requires FP16 internal output/);

    const invalidCin = k7Shape(130, 128, 17, 1);
    await expect(kernel.createDispatch(
      "cin",
      invalidCin,
      bindingsFor(invalidCin, "float16", true),
      "float16",
      fullRange(fakeBuffer(256), 0, invalidCin),
    )).rejects.toThrow(/Cin divisible by 4/);
    const invalidCout = k7Shape(128, 256 + 64, 17, 1);
    await expect(kernel.createDispatch(
      "cout",
      invalidCout,
      bindingsFor(invalidCout, "float16", true),
      "float16",
      fullRange(fakeBuffer(256), 0, invalidCout),
    )).rejects.toThrow(/Cout divisible by 128/);

    const plan = planAceFp16VaeConv1d(candidate, "float16");
    expect(() => planAceOpt0024VaeConv1dDirectDot4SubgroupRange(plan, {
      base: 1,
      count: candidate.outputChannels,
    })).toThrow(/complete in-bounds NLC rows/);
    expect(() => aceOpt0024VaeConv1dDirectDot4SubgroupWgsl(
      candidate,
      false,
      "float16",
    )).toThrow(/requires bias/);
  });

  it("rejects dispatch, vector-view binding, size, and overlap violations", async () => {
    const shape = k7Shape(128, 128, 65, 1);
    const ordinaryBindings = bindingsFor(shape, "float16", true);
    const range = fullRange(fakeBuffer(256), 0, shape);
    const dispatchLimited = AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      fakeDevice({ maximumDispatch: 1 }),
      FIXED_32_CAPABILITY,
    );
    await expect(dispatchLimited.createDispatch(
      "dispatch-limit",
      shape,
      ordinaryBindings,
      "float16",
      range,
    )).rejects.toThrow(/dispatch dimension/);

    const plan = planAceFp16VaeConv1d(shape, "float16");
    const bufferLimited = AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      fakeDevice({ maximumStorageBinding: plan.weightBindingBytes - 4 }),
      FIXED_32_CAPABILITY,
    );
    await expect(bufferLimited.createDispatch(
      "binding-limit",
      shape,
      ordinaryBindings,
      "float16",
      range,
    )).rejects.toThrow(/weight exceeds the device storage binding limit/);

    const vectorAlignmentDevice = fakeDevice({ storageAlignment: 4 });
    const vectorAlignment =
      AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
        vectorAlignmentDevice,
        FIXED_32_CAPABILITY,
      );
    const misalignedInput = {
      ...ordinaryBindings,
      input: {
        buffer: fakeBuffer(plan.inputBindingBytes + 4),
        offset: 4,
        size: plan.inputBindingBytes,
      },
    };
    await expect(vectorAlignment.createDispatch(
      "input-vec4-alignment",
      shape,
      misalignedInput,
      "float16",
      range,
    )).rejects.toThrow(/input binding does not expose an aligned/);
    const misalignedWeight = {
      ...ordinaryBindings,
      weight: {
        buffer: fakeBuffer(plan.weightBindingBytes + 4),
        offset: 4,
        size: plan.weightBindingBytes,
      },
    };
    await expect(vectorAlignment.createDispatch(
      "weight-vec4-alignment",
      shape,
      misalignedWeight,
      "float16",
      range,
    )).rejects.toThrow(/weight binding does not expose an aligned/);

    const ordinary = AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
      fakeDevice(),
      FIXED_32_CAPABILITY,
    );
    await expect(ordinary.createDispatch(
      "undersized",
      shape,
      {
        ...ordinaryBindings,
        weight: fakeBinding(plan.weightBindingBytes - 4),
      },
      "float16",
      range,
    )).rejects.toThrow(/weight binding does not expose an aligned/);

    const shared = fakeBuffer(plan.weightBindingBytes);
    await expect(ordinary.createDispatch(
      "overlap",
      shape,
      {
        ...ordinaryBindings,
        input: {
          buffer: shared,
          offset: 0,
          size: plan.inputBindingBytes,
        },
        weight: {
          buffer: shared,
          offset: 0,
          size: plan.weightBindingBytes,
        },
      },
      "float16",
      range,
    )).rejects.toThrow(/input and weight bindings must not overlap/);
  });
});

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const FIXED_32_CAPABILITY = Object.freeze({
  subgroupMinSize: 32,
  subgroupMaxSize: 32,
});

interface CandidateScope {
  readonly operations: number;
  readonly ranges: number;
  readonly physicalWorkgroups: number;
  readonly logicalMultiplyAccumulates: number;
  readonly workgroupKcin4Instances: number;
  readonly validInputVec4Loads: number;
  readonly nativeWeightVec4Loads: number;
  readonly subgroupBroadcastCollectives: number;
  readonly broadcastInvocationCalls: number;
  readonly fp16Dot4Calls: number;
  readonly fp32Vec4AccumulatorAdds: number;
}

function candidateScope(frames: number): CandidateScope {
  const graph = planAceVaeDecoder(frames);
  const cooperative = planAceVaeDecoderQuanta(graph);
  const operationIndices = new Set<number>();
  const scope = mutableZeroScope();
  for (const quantum of cooperative.quanta) {
    const operation = graph.operations[quantum.operationIndex]!;
    if (!isCandidateOperation(operation)) continue;
    operationIndices.add(quantum.operationIndex);
    expect(quantum.primitives).toHaveLength(1);
    const primitive = quantum.primitives[0]!;
    const plan = planAceFp16VaeConv1d(operation.shape, "float16");
    const range = planAceOpt0024VaeConv1dDirectDot4SubgroupRange(plan, {
      base: primitive.outputBase,
      count: primitive.outputCount,
    });
    expect(range).toEqual(planAceFp16VaeConv1dSubgroupRange(plan, {
      base: primitive.outputBase,
      count: primitive.outputCount,
    }));
    const physicalWorkgroups = range.workgroupsX * range.workgroupsY;
    const cin4 = plan.inputChannels / 4;
    const workgroupKcin4Instances = physicalWorkgroups * 7 * cin4;
    const validRowTaps = countValidRowTaps(plan, range);
    const validInputVec4Loads =
      validRowTaps * (plan.outputChannels / 128) * cin4;
    const fp16Dot4Calls = validRowTaps * plan.outputChannels * cin4;
    scope.ranges += 1;
    scope.physicalWorkgroups += physicalWorkgroups;
    scope.logicalMultiplyAccumulates +=
      quantum.estimatedMaximumMultiplyAccumulates;
    scope.workgroupKcin4Instances += workgroupKcin4Instances;
    scope.validInputVec4Loads += validInputVec4Loads;
    scope.nativeWeightVec4Loads += workgroupKcin4Instances * 128 * 4;
    scope.subgroupBroadcastCollectives +=
      workgroupKcin4Instances * 4 * 8;
    scope.broadcastInvocationCalls += workgroupKcin4Instances * 4 * 8 * 32;
    scope.fp16Dot4Calls += fp16Dot4Calls;
    scope.fp32Vec4AccumulatorAdds += fp16Dot4Calls / 4;
  }
  return Object.freeze({ ...scope, operations: operationIndices.size });
}

function isCandidateOperation(
  operation: ReturnType<typeof planAceVaeDecoder>["operations"][number],
): operation is AceVaeDecoderConvOperation {
  return operation.kind === "conv1d" &&
    operation.shape.kernelSize === 7 &&
    operation.bias !== undefined;
}

function countValidRowTaps(
  plan: ReturnType<typeof planAceFp16VaeConv1d>,
  range: ReturnType<
    typeof planAceOpt0024VaeConv1dDirectDot4SubgroupRange
  >,
): number {
  let count = 0;
  const end = range.firstOutputTime + range.outputRowCount;
  for (let kernel = 0; kernel < 7; kernel += 1) {
    for (
      let outputTime = range.firstOutputTime;
      outputTime < end;
      outputTime += 1
    ) {
      const paddedTime = outputTime + kernel * plan.dilation;
      if (
        paddedTime >= plan.padding &&
        paddedTime - plan.padding < plan.inputFrames
      ) count += 1;
    }
  }
  return count;
}

function mutableZeroScope(): { -readonly [K in keyof CandidateScope]: number } {
  return {
    operations: 0,
    ranges: 0,
    physicalWorkgroups: 0,
    logicalMultiplyAccumulates: 0,
    workgroupKcin4Instances: 0,
    validInputVec4Loads: 0,
    nativeWeightVec4Loads: 0,
    subgroupBroadcastCollectives: 0,
    broadcastInvocationCalls: 0,
    fp16Dot4Calls: 0,
    fp32Vec4AccumulatorAdds: 0,
  };
}

function pickTopology(scope: CandidateScope): Pick<
  CandidateScope,
  "operations" | "ranges" | "physicalWorkgroups" |
    "logicalMultiplyAccumulates"
> {
  return {
    operations: scope.operations,
    ranges: scope.ranges,
    physicalWorkgroups: scope.physicalWorkgroups,
    logicalMultiplyAccumulates: scope.logicalMultiplyAccumulates,
  };
}

function scaleScope(scope: CandidateScope, factor: number): CandidateScope {
  return Object.freeze(Object.fromEntries(
    Object.entries(scope).map(([key, value]) => [key, value * factor]),
  )) as unknown as CandidateScope;
}

function addScopes(...scopes: readonly CandidateScope[]): CandidateScope {
  const total = mutableZeroScope();
  for (const scope of scopes) {
    for (const key of Object.keys(total) as (keyof CandidateScope)[]) {
      total[key] += scope[key];
    }
  }
  return Object.freeze(total);
}

function k7Shape(
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
  readonly createBuffer: ReturnType<typeof vi.fn>;
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
    createBuffer: vi.fn(() => {
      throw new Error("OPT-0024 candidate must not allocate");
    }),
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
