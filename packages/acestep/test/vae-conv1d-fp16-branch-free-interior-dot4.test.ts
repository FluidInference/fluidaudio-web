import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0046_VAE_CONV1D_BRANCH_FREE_INTERIOR_KERNEL_ID,
  ACE_OPT_0046_VAE_CONV1D_BRANCH_FREE_INTERIOR_SEMANTICS,
  AceOpt0046VaeConv1dBranchFreeInteriorKernel,
  aceOpt0046VaeConv1dBranchFreeInteriorWgsl,
  planAceOpt0046VaeConv1dBranchFreeInteriorRange,
  planAceOpt0046VaeConv1dBranchFreePartition,
} from
  "../src/webgpu/kernels/vae-conv1d-fp16-branch-free-interior-dot4.js";
import {
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
} from
  "../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dBindings,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import {
  planAceVaeDecoder,
  type AceVaeDecoderConvOperation,
} from "../src/webgpu/vae-decoder.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../src/webgpu/kernels/vae-primitives.js";

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

describe("OPT-0046 branch-free interior FP16-dot4 K7 Conv1D", () => {
  it("keeps OPT-0024 arithmetic and removes tap/row validity from the interior", () => {
    const source = aceOpt0046VaeConv1dBranchFreeInteriorWgsl(
      shape(1_024, 1_024, 512, 9),
      true,
      "float16",
    );
    expect(ACE_OPT_0046_VAE_CONV1D_BRANCH_FREE_INTERIOR_KERNEL_ID).toBe(
      "ace-vae-fp16-opt-0046-branch-free-interior-native-oki-fp16-dot4-k7-conv1d-v1",
    );
    expect(ACE_OPT_0046_VAE_CONV1D_BRANCH_FREE_INTERIOR_SEMANTICS).toBe(
      "opt-0024-k-cin4-order-with-all-seven-taps-proven-valid",
    );
    expect(source).toContain(
      `// boundary-owner: ${ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID}`,
    );
    expect(source).toContain("@compute @workgroup_size(\n  128, 1, 1,");
    expect(source).toContain("for (var kernel = 0u; kernel < 7u;");
    expect(source).toContain("var input_channel4 = 0u;");
    expect(source.match(/subgroupBroadcast\(lane_input, [0-7]u\)/g))
      .toHaveLength(8);
    expect(source.match(/\bdot\(input_operand[0-7], weight[0-3]\)/g))
      .toHaveLength(32);
    expect(source.match(/let partial[0-7] = vec4<f16>\(/g)).toHaveLength(8);
    expect(source.match(/sum[0-7] = sum[0-7] \+ vec4<f32>\(partial[0-7]\)/g))
      .toHaveLength(8);
    expect(source.match(/= f16\(sum[0-7]\.[xyzw]\);/g)).toHaveLength(32);
    expect(source).not.toContain("input_valid");
    expect(source).not.toContain("range_end_time");
    expect(source).not.toContain("lane_padded_time");
    expect(source).not.toContain("vec4<f16>(0.0h)");
    expect(source).not.toContain("if (output_time");
    expect(source).not.toContain("var<workgroup>");
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toContain("packed_weight");

    const cin4 = source.indexOf("var input_channel4 = 0u;");
    const input = source.indexOf("lane_input = input[", cin4);
    const broadcast = source.indexOf("subgroupBroadcast(lane_input, 0u)", cin4);
    const weight = source.indexOf("let weight0 = weight[", cin4);
    const partial = source.indexOf("let partial0 = vec4<f16>(", cin4);
    const add = source.indexOf("sum0 = sum0 + vec4<f32>(partial0);", cin4);
    expect(cin4).toBeLessThan(input);
    expect(input).toBeLessThan(broadcast);
    expect(broadcast).toBeLessThan(weight);
    expect(weight).toBeLessThan(partial);
    expect(partial).toBeLessThan(add);
  });

  it("partitions C512 d1/d3/d9 into disjoint complete boundary/interior coverage", () => {
    const expected = new Map([
      [1, { prefix: 3, interior: 480, suffix: 29 }],
      [3, { prefix: 9, interior: 480, suffix: 23 }],
      [9, { prefix: 27, interior: 448, suffix: 37 }],
    ] as const);
    for (const dilation of [1, 3, 9] as const) {
      const plan = planAceFp16VaeConv1d(
        shape(512, 512, 512, dilation),
        "float16",
      );
      const partition = planAceOpt0046VaeConv1dBranchFreePartition(plan, {
        base: 0,
        count: plan.outputElements,
      });
      const rows = expected.get(dilation)!;
      expect(partition.segments.map(({ kind, outputRowCount }) => ({
        kind,
        outputRowCount,
      }))).toEqual([
        { kind: "prefix-boundary", outputRowCount: rows.prefix },
        { kind: "interior", outputRowCount: rows.interior },
        { kind: "suffix-boundary", outputRowCount: rows.suffix },
      ]);
      assertCoverage(partition.segments, 0, plan.outputElements);
      expect(partition.interior!.outputRowCount % 32).toBe(0);
      expect(() => planAceOpt0046VaeConv1dBranchFreeInteriorRange(
        plan,
        partition.interior!,
      )).not.toThrow();
    }
  });

  it("handles clipped, boundary-only, and batched ranges without empty dispatches", () => {
    const plan = planAceFp16VaeConv1d(shape(128, 128, 512, 9, 2), "float16");
    const base = (512 + 20) * 128;
    const count = 400 * 128;
    const clipped = planAceOpt0046VaeConv1dBranchFreePartition(plan, {
      base,
      count,
    });
    expect(clipped.batch).toBe(1);
    expect(clipped.segments.map(({ kind, outputRowCount }) => ({
      kind,
      outputRowCount,
    }))).toEqual([
      { kind: "prefix-boundary", outputRowCount: 7 },
      { kind: "interior", outputRowCount: 384 },
      { kind: "suffix-boundary", outputRowCount: 9 },
    ]);
    assertCoverage(clipped.segments, base, count);

    const tinyPlan = planAceFp16VaeConv1d(shape(128, 128, 55, 9), "float16");
    const tiny = planAceOpt0046VaeConv1dBranchFreePartition(tinyPlan, {
      base: 0,
      count: tinyPlan.outputElements,
    });
    expect(tiny.interior).toBeUndefined();
    expect(tiny.segments.every(({ count: segmentCount }) => segmentCount > 0))
      .toBe(true);
    assertCoverage(tiny.segments, 0, tinyPlan.outputElements);

    const before = planAceOpt0046VaeConv1dBranchFreePartition(plan, {
      base: 0,
      count: 20 * 128,
    });
    expect(before.segments).toEqual([
      expect.objectContaining({
        kind: "prefix-boundary",
        base: 0,
        count: 20 * 128,
      }),
    ]);
    const afterBase = (512 + 490) * 128;
    const after = planAceOpt0046VaeConv1dBranchFreePartition(plan, {
      base: afterBase,
      count: 20 * 128,
    });
    expect(after.segments).toEqual([
      expect.objectContaining({
        kind: "suffix-boundary",
        base: afterBase,
        count: 20 * 128,
      }),
    ]);
  });

  it("covers every biased production K7 shape and all residual channel/dilation tiers", () => {
    const operations = planAceVaeDecoder(512).operations.filter(
      (operation): operation is AceVaeDecoderConvOperation =>
        operation.kind === "conv1d" && operation.shape.kernelSize === 7 &&
        operation.bias !== undefined,
    );
    expect(operations).toHaveLength(16);
    const observed = new Set<string>();
    for (const operation of operations) {
      const plan = planAceFp16VaeConv1d(operation.shape, "float16");
      const partition = planAceOpt0046VaeConv1dBranchFreePartition(plan, {
        base: 0,
        count: plan.outputElements,
      });
      assertCoverage(partition.segments, 0, plan.outputElements);
      if (operation.label !== "conv1") {
        observed.add(`${operation.shape.inputChannels}/d${operation.shape.dilation}`);
      }
      if (partition.interior !== undefined) {
        planAceOpt0046VaeConv1dBranchFreeInteriorRange(plan, partition.interior);
      }
    }
    expect(observed).toEqual(new Set([
      "1024/d1", "1024/d3", "1024/d9",
      "512/d1", "512/d3", "512/d9",
      "256/d1", "256/d3", "256/d9",
      "128/d1", "128/d3", "128/d9",
    ]));
  });

  it("creates one cached allocation-free interior owner and rejects after destroy", async () => {
    const device = fakeDevice();
    const kernel = AceOpt0046VaeConv1dBranchFreeInteriorKernel.create(
      device,
      FIXED32,
    );
    const candidateShape = shape(512, 512, 512, 9);
    const plan = planAceFp16VaeConv1d(candidateShape, "float16");
    const partition = planAceOpt0046VaeConv1dBranchFreePartition(plan, {
      base: 0,
      count: plan.outputElements,
    });
    const controls = fakeBuffer(512);
    const bindings = bindingsFor(plan);
    const first = await kernel.createDispatch(
      "first",
      candidateShape,
      bindings,
      "float16",
      rangeBinding(controls, 256, partition.interior!),
    );
    const second = await kernel.createDispatch(
      "second",
      candidateShape,
      bindings,
      "float16",
      rangeBinding(controls, 0, partition.interior!),
    );
    expect(first).toMatchObject({
      kernelId: ACE_OPT_0046_VAE_CONV1D_BRANCH_FREE_INTERIOR_KERNEL_ID,
      outputRange: { outputRowCount: 448, workgroupsX: 14, workgroupsY: 4 },
    });
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();
    expect(device.createBuffer).not.toHaveBeenCalled();
    const pass = fakePass();
    first.encode(pass);
    second.encode(pass);
    expect(pass.setBindGroup.mock.calls.map((call) => call[2]))
      .toEqual([[256], [0]]);
    expect(pass.dispatchWorkgroups.mock.calls).toEqual([
      [14, 4, 1],
      [14, 4, 1],
    ]);
    kernel.destroy();
    kernel.destroy();
    expect(() => first.encode(pass)).toThrow(/destroyed/);
    await expect(kernel.createDispatch(
      "after-destroy",
      candidateShape,
      bindings,
      "float16",
      rangeBinding(controls, 0, partition.interior!),
    )).rejects.toThrow(/destroyed/);
  });

  it("fails closed for boundary rows, partial tiles, capability, and shape changes", async () => {
    expect(() => AceOpt0046VaeConv1dBranchFreeInteriorKernel.create(
      fakeDevice({ shaderF16: false }),
      FIXED32,
    )).toThrow(/shader-f16/);
    expect(() => AceOpt0046VaeConv1dBranchFreeInteriorKernel.create(
      fakeDevice(),
      { subgroupMinSize: 16, subgroupMaxSize: 32 },
    )).toThrow(/fixed 32/);
    const candidateShape = shape(128, 128, 512, 9);
    const plan = planAceFp16VaeConv1d(candidateShape, "float16");
    expect(() => planAceOpt0046VaeConv1dBranchFreeInteriorRange(plan, {
      base: 27 * 128,
      count: 31 * 128,
    })).toThrow(/complete 32-row/);
    expect(() => planAceOpt0046VaeConv1dBranchFreeInteriorRange(plan, {
      base: 26 * 128,
      count: 32 * 128,
    })).toThrow(/all-taps-valid/);
    expect(() => planAceOpt0046VaeConv1dBranchFreeInteriorRange(plan, {
      base: 453 * 128,
      count: 32 * 128,
    })).not.toThrow();
    expect(() => planAceOpt0046VaeConv1dBranchFreeInteriorRange(plan, {
      base: 454 * 128,
      count: 32 * 128,
    })).toThrow(/all-taps-valid/);

    const kernel = AceOpt0046VaeConv1dBranchFreeInteriorKernel.create(
      fakeDevice(),
      FIXED32,
    );
    const k1 = {
      ...candidateShape,
      kernelSize: 1,
      dilation: 1,
      padding: 0,
    };
    const k1Plan = planAceFp16VaeConv1d(k1, "float16");
    await expect(kernel.createDispatch(
      "k1",
      k1,
      bindingsFor(k1Plan),
      "float16",
      rangeBinding(fakeBuffer(256), 0, {
        base: 0,
        count: 32 * 128,
      }),
    )).rejects.toThrow(/only K7/);
    await expect(kernel.createDispatch(
      "fp32",
      candidateShape,
      bindingsFor(plan),
      "float32",
      rangeBinding(fakeBuffer(256), 0, {
        base: 27 * 128,
        count: 32 * 128,
      }),
    )).rejects.toThrow(/FP16 output/);
    expect(() => aceOpt0046VaeConv1dBranchFreeInteriorWgsl(
      candidateShape,
      false,
      "float16",
    )).toThrow(/requires bias/);
  });
});

const FIXED32 = Object.freeze({ subgroupMinSize: 32, subgroupMaxSize: 32 });

function shape(
  inputChannels: number,
  outputChannels: number,
  frames: number,
  dilation: 1 | 3 | 9,
  batch = 1,
): AceVaeConv1dShape {
  return {
    batch,
    inputFrames: frames,
    inputChannels,
    outputChannels,
    kernelSize: 7,
    stride: 1,
    dilation,
    padding: dilation * 3,
  };
}

function assertCoverage(
  segments: readonly Readonly<{ base: number; count: number }>[],
  base: number,
  count: number,
): void {
  expect(segments[0]?.base).toBe(base);
  expect(segments.reduce((sum, segment) => sum + segment.count, 0)).toBe(count);
  for (let index = 1; index < segments.length; index += 1) {
    expect(segments[index - 1]!.base + segments[index - 1]!.count)
      .toBe(segments[index]!.base);
  }
}

function bindingsFor(
  plan: ReturnType<typeof planAceFp16VaeConv1d>,
): AceFp16VaeConv1dBindings {
  return {
    input: binding(plan.inputBindingBytes),
    weight: binding(plan.weightBindingBytes),
    bias: binding(plan.biasBindingBytes),
    output: binding(plan.outputBindingBytes),
  };
}

function rangeBinding(
  buffer: GPUBuffer,
  offset: number,
  range: Readonly<{ base: number; count: number }>,
): AceVaeOutputRangeBinding {
  return { ...range, control: { buffer, offset, size: 16 } };
}

type FakeDevice = GPUDevice & {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
  readonly createBuffer: ReturnType<typeof vi.fn>;
};

function fakeDevice(options: {
  readonly shaderF16?: boolean;
  readonly subgroups?: boolean;
} = {}): FakeDevice {
  return {
    features: new Set([
      ...(options.shaderF16 === false ? [] : ["shader-f16"]),
      ...(options.subgroups === false ? [] : ["subgroups"]),
    ]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxStorageBufferBindingSize: 1_073_741_824,
      maxBufferSize: 1_073_741_824,
      minStorageBufferOffsetAlignment: 256,
      minUniformBufferOffsetAlignment: 256,
    },
    createShaderModule: vi.fn(() => ({
      getCompilationInfo: vi.fn(async () => ({ messages: [] })),
    })),
    createBindGroupLayout: vi.fn(() => ({ label: "layout" })),
    createPipelineLayout: vi.fn(() => ({ label: "pipeline-layout" })),
    createComputePipelineAsync: vi.fn(async () => ({ label: "pipeline" })),
    createBindGroup: vi.fn(() => ({ label: "bind-group" })),
    createBuffer: vi.fn(() => {
      throw new Error("OPT-0046 owner must not allocate");
    }),
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

function binding(size: number): GPUBufferBinding {
  return { buffer: fakeBuffer(size), offset: 0, size };
}

function fakeBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}
