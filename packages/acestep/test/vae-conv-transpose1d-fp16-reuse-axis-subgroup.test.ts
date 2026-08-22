import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_LAYOUT_ID,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP,
  ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  aceOpt0036VaeConvTranspose1dR4C8Wgsl,
  aceOpt0036VaeConvTranspose1dR8C4Wgsl,
  planAceOpt0036VaeConvTranspose1dR4C8,
  planAceOpt0036VaeConvTranspose1dR8C4,
  planAceOpt0036VaeConvTranspose1dRange,
  type AceOpt0036VaeConvTranspose1dPlan,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-reuse-axis-subgroup.js";
import type { AceVaeConvTranspose1dShape } from
  "../src/webgpu/kernels/vae-primitives.js";
import {
  ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_ROUTES,
  ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID,
  AceOpt0040VaeConvTranspose1dShapeSelectorKernel,
  selectAceOpt0040VaeConvTranspose1d,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-shape-selector.js";

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const KERNEL_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-reuse-axis-subgroup.ts",
  import.meta.url,
), "utf8");
const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0036-vae-conv-transpose1d-reuse-axis.ts",
  import.meta.url,
), "utf8");
const HARNESS_HTML = readFileSync(new URL(
  "./browser/opt-0036-vae-conv-transpose1d-reuse-axis.html",
  import.meta.url,
), "utf8");

const SHAPES = Object.freeze([
  shape(300, 2_048, 1_024, 10),
  shape(3_000, 1_024, 512, 6),
  shape(18_000, 512, 256, 4),
  shape(72_000, 256, 128, 4),
  shape(288_000, 128, 128, 2),
]);

describe("OPT-0036 ConvTranspose reuse-axis split", () => {
  it("plans only R8xC4 and R4xC8 with 32 accumulators per lane", () => {
    expect(ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUP_SIZE).toBe(32);
    expect(ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_SUBGROUPS_PER_WORKGROUP).toBe(4);
    expect(ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_WORKGROUP_SIZE).toBe(128);
    expect(ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_LAYOUT_ID).toBe(
      "conv-transpose1d-phase-tap-input-output-f16-v1",
    );
    for (const candidate of SHAPES) {
      const row = planAceOpt0036VaeConvTranspose1dR8C4(candidate);
      const channel = planAceOpt0036VaeConvTranspose1dR4C8(candidate);
      expect(row).toMatchObject({
        kernelId: ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
        rowsPerSubgroup: 8,
        rowsPerWorkgroup: 32,
        channelsPerLane: 4,
        channelsPerWorkgroup: 128,
        accumulatorCountPerLane: 32,
        workgroupStorageBytes: 0,
        workgroupBarrierCount: 0,
      });
      expect(channel).toMatchObject({
        kernelId: ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
        rowsPerSubgroup: 4,
        rowsPerWorkgroup: 16,
        channelsPerLane: 8,
        channelsPerWorkgroup: 256,
        accumulatorCountPerLane: 32,
        workgroupStorageBytes: 0,
        workgroupBarrierCount: 0,
      });
      const fullRange = { base: 0, count: row.outputElements };
      const rowRange = planAceOpt0036VaeConvTranspose1dRange(row, fullRange);
      const channelRange = planAceOpt0036VaeConvTranspose1dRange(
        channel,
        fullRange,
      );
      expect(rowRange.workgroupsY).toBe(candidate.outputChannels / 128);
      expect(channelRange.workgroupsY).toBe(
        Math.ceil(candidate.outputChannels / 256),
      );
      expect(rowRange.workgroupsZ).toBe(candidate.stride);
      expect(channelRange.workgroupsZ).toBe(candidate.stride);
    }
  });

  it.each([
    ["R8xC4", planAceOpt0036VaeConvTranspose1dR8C4],
    ["R4xC8", planAceOpt0036VaeConvTranspose1dR4C8],
  ] as const)("owns every ranged output exactly once for %s", (_, planner) => {
    const candidate = shape(17, 64, 128, 6);
    const plan = planner(candidate);
    const firstOutputTime = 5;
    const outputRows = 53;
    const range = planAceOpt0036VaeConvTranspose1dRange(plan, {
      base: firstOutputTime * candidate.outputChannels,
      count: outputRows * candidate.outputChannels,
    });
    const owners = new Uint8Array(range.count);
    markOwners(owners, plan, range.workgroupsX, range.workgroupsY);
    expect([...owners].every((owner) => owner === 1)).toBe(true);
  });

  it("emits exact tap-then-Cin R8xC4 arithmetic and FP16 stores", () => {
    const source = aceOpt0036VaeConvTranspose1dR8C4Wgsl(SHAPES[1]!);
    expect(source).toContain("group.x * 32u");
    expect(source).toContain("subgroup * 8u");
    expect(source).toContain("group.y * 128u");
    expect(source).toContain("lane * 4u");
    expect(source).toContain("polyphase_weight:\n  array<vec2<u32>>");
    expect(source).toContain("var sum7 = bias_value;");
    expect(source).toContain("for (var tap = 0u; tap < 2u; tap += 1u)");
    expect(source).toContain(
      "for (var inner = 0u; inner < INPUT_CHANNELS; inner += 1u)",
    );
    expect(source).toContain("subgroupBroadcast(lane_input, 7u)");
    expect(source).toContain(
      "sum7 = sum7 + vec4<f32>(a7) * weight_value;",
    );
    expect(source.match(/= f16\(sum/g)).toHaveLength(32);
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toContain("var<workgroup>");
  });

  it("emits exact tap-then-Cin R4xC8 arithmetic and FP16 stores", () => {
    const source = aceOpt0036VaeConvTranspose1dR4C8Wgsl(SHAPES[1]!);
    expect(source).toContain("group.x * 16u");
    expect(source).toContain("subgroup * 4u");
    expect(source).toContain("group.y * 256u");
    expect(source).toContain("lane * 8u");
    expect(source).toContain("polyphase_weight:\n  array<vec4<u32>>");
    expect(source).toContain("var sum3_1 = bias_value.high;");
    expect(source).toContain("subgroupBroadcast(lane_input, 3u)");
    expect(source).toContain(
      "sum3_1 = sum3_1 + vec4<f32>(a3) * weight_value.high;",
    );
    expect(source.match(/= f16\(sum/g)).toHaveLength(32);
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toContain("var<workgroup>");
  });

  it("rejects geometry outside the registered screen", () => {
    expect(() => planAceOpt0036VaeConvTranspose1dR8C4(
      shape(8, 64, 64, 2),
    )).toThrow(/exact production polyphase geometry/);
    const plan = planAceOpt0036VaeConvTranspose1dR4C8(SHAPES[4]!);
    expect(() => planAceOpt0036VaeConvTranspose1dRange(plan, {
      base: 1,
      count: plan.outputElements,
    })).toThrow(/complete in-bounds NLC rows/);
  });

  it("gates full C512 raw-U16 exactness before balanced A/B/C timing", () => {
    for (const operation of [
      'operation("block-0-conv-t1", 300, 2_048, 1_024, 10)',
      'operation("block-1-conv-t1", 3_000, 1_024, 512, 6)',
      'operation("block-2-conv-t1", 18_000, 512, 256, 4)',
      'operation("block-3-conv-t1", 72_000, 256, 128, 4)',
      'operation("block-4-conv-t1", 288_000, 128, 128, 2)',
    ]) expect(HARNESS_SOURCE).toContain(operation);
    expect(HARNESS_SOURCE).toContain("aceOpt0026VaeConvTranspose1dWgsl");
    expect(HARNESS_SOURCE).toContain(
      "aceOpt0036VaeConvTranspose1dR8C4Wgsl",
    );
    expect(HARNESS_SOURCE).toContain(
      "aceOpt0036VaeConvTranspose1dR4C8Wgsl",
    );
    expect(HARNESS_SOURCE).toContain("OUTPUT_PREFILL = 0x7e55");
    expect(HARNESS_SOURCE).toContain("rawU16Exact: true");
    expect(HARNESS_SOURCE).toContain("completeDeterministicWrites: true");
    expect(HARNESS_SOURCE).toContain("completedBeforeReady: true");
    expect(HARNESS_SOURCE).toContain("shared-timing-output");
    expect(HARNESS_SOURCE).toContain("TIMING_ORDERS.length");
    expect(HARNESS_SOURCE).toContain(
      "REQUIRED_SUMMED_MEDIAN_SPEEDUP = 1.15",
    );
    expect(HARNESS_SOURCE).toContain("noSlowerEveryShape");
    expect(HARNESS_SOURCE).toContain("requestedLimits");
    expect(HARNESS_SOURCE).toContain("window.__ACE_OPT0036_RESULT__ = receipt");
    expect(HARNESS_SOURCE).not.toContain("vae-fp16-decoder");
    expect(KERNEL_SOURCE).not.toContain("OPT-0029");
    expect(HARNESS_HTML).toContain('id="run" type="button" disabled');
    expect(HARNESS_HTML).toContain("leave the machine nominal and idle for 30 seconds");
    expect(HARNESS_HTML).toContain("Run balanced A/B/C timing");
    expect(HARNESS_HTML).toContain(
      "opt-0036-vae-conv-transpose1d-reuse-axis.ts",
    );
  });
});

describe("OPT-0040 production shape selector", () => {
  it("freezes the exact five-operation route and preserves revision-6 layout", () => {
    expect(ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_ROUTES.map((candidate) => [
      candidate.operationLabel,
      candidate.reuseAxis,
      candidate.kernelId,
    ])).toEqual([
      [
        "block-0-conv-t1",
        "channel",
        ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
      ],
      [
        "block-1-conv-t1",
        "channel",
        ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
      ],
      [
        "block-2-conv-t1",
        "channel",
        ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
      ],
      [
        "block-3-conv-t1",
        "row",
        ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
      ],
      [
        "block-4-conv-t1",
        "row",
        ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
      ],
    ]);
    expect(Object.isFrozen(ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_ROUTES)).toBe(true);
    expect(ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_ROUTES.every(Object.isFrozen))
      .toBe(true);
    for (const [index, candidate] of SHAPES.entries()) {
      const selected = selectAceOpt0040VaeConvTranspose1d(
        `block-${index}-conv-t1`,
        candidate,
      );
      expect(selected).toMatchObject({
        selectorKernelId:
          ACE_OPT_0040_VAE_CONV_TRANSPOSE1D_SHAPE_SELECTOR_KERNEL_ID,
        operationLabel: `block-${index}-conv-t1`,
        reuseAxis: index < 3 ? "channel" : "row",
        kernelId: index < 3
          ? ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID
          : ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
      });
      expect(selected.plan.weightLayout).toBe(
        ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_LAYOUT_ID,
      );
      expect(Object.isFrozen(selected)).toBe(true);
    }
    expect(() => selectAceOpt0040VaeConvTranspose1d(
      "block-5-conv-t1",
      SHAPES[0]!,
    )).toThrow(/no ConvTranspose1D route/);
    expect(() => selectAceOpt0040VaeConvTranspose1d(
      "block-0-conv-t1",
      SHAPES[1]!,
    )).toThrow(/changed its authenticated shape/);
  });

  it("caches one pipeline/bind group per selected shape and dies closed", async () => {
    const device = selectorDevice();
    const owner = AceOpt0040VaeConvTranspose1dShapeSelectorKernel.create(
      device,
      { subgroupMinSize: 32, subgroupMaxSize: 32 },
    );
    const channelShape = SHAPES[0]!;
    const channelPlan = planAceOpt0036VaeConvTranspose1dR4C8(channelShape);
    const channelBindings = selectorBindings(channelPlan);
    const controls = selectorBuffer(512);
    const first = await owner.createDispatch(
      "channel-first",
      "block-0-conv-t1",
      channelShape,
      channelBindings,
      {
        base: 0,
        count: channelPlan.outputElements,
        control: { buffer: controls, offset: 0, size: 16 },
      },
    );
    const reused = await owner.createDispatch(
      "channel-reused",
      "block-0-conv-t1",
      channelShape,
      channelBindings,
      {
        base: 0,
        count: channelPlan.outputElements,
        control: { buffer: controls, offset: 256, size: 16 },
      },
    );
    expect(device.createShaderModule).toHaveBeenCalledOnce();
    expect(device.createComputePipelineAsync).toHaveBeenCalledOnce();
    expect(device.createBindGroup).toHaveBeenCalledOnce();

    const rowShape = SHAPES[3]!;
    const rowPlan = planAceOpt0036VaeConvTranspose1dR8C4(rowShape);
    const row = await owner.createDispatch(
      "row",
      "block-3-conv-t1",
      rowShape,
      selectorBindings(rowPlan),
      {
        base: 0,
        count: rowPlan.outputElements,
        control: { buffer: controls, offset: 0, size: 16 },
      },
    );
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(device.createComputePipelineAsync).toHaveBeenCalledTimes(2);
    expect(device.createBindGroup).toHaveBeenCalledTimes(2);
    const sources = device.createShaderModule.mock.calls.map((call) =>
      String((call[0] as GPUShaderModuleDescriptor).code)
    );
    expect(sources[0]).toContain(
      `weight-layout: ${ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_LAYOUT_ID}`,
    );
    expect(sources[0]).toContain(
      ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R4C8_KERNEL_ID,
    );
    expect(sources[1]).toContain(
      ACE_OPT_0036_VAE_CONV_TRANSPOSE1D_R8C4_KERNEL_ID,
    );

    const firstPass = selectorPass();
    const reusedPass = selectorPass();
    first.encode(firstPass);
    reused.encode(reusedPass);
    row.encode(selectorPass());
    expect(firstPass.setBindGroup).toHaveBeenCalledWith(
      0,
      expect.anything(),
      [0],
    );
    expect(reusedPass.setBindGroup).toHaveBeenCalledWith(
      0,
      expect.anything(),
      [256],
    );

    owner.destroy();
    owner.destroy();
    expect(() => first.encode(selectorPass())).toThrow(/destroyed/);
    await expect(owner.createDispatch(
      "after-destroy",
      "block-0-conv-t1",
      channelShape,
      channelBindings,
      {
        base: 0,
        count: channelPlan.outputElements,
        control: { buffer: controls, offset: 0, size: 16 },
      },
    )).rejects.toThrow(/destroyed/);
  });
});

function markOwners(
  owners: Uint8Array,
  plan: AceOpt0036VaeConvTranspose1dPlan,
  workgroupsX: number,
  workgroupsY: number,
): void {
  const outputRows = owners.length / plan.outputChannels;
  for (let x = 0; x < workgroupsX; x += 1) {
    for (let y = 0; y < workgroupsY; y += 1) {
      for (let phase = 0; phase < plan.stride; phase += 1) {
        for (let subgroup = 0; subgroup < 4; subgroup += 1) {
          for (let lane = 0; lane < 32; lane += 1) {
            for (let row = 0; row < plan.rowsPerSubgroup; row += 1) {
              const phaseRow = x * plan.rowsPerWorkgroup +
                subgroup * plan.rowsPerSubgroup + row;
              const rowOffset = phase + phaseRow * plan.stride;
              if (rowOffset >= outputRows) continue;
              for (let member = 0; member < plan.channelsPerLane; member += 1) {
                const channel = y * plan.channelsPerWorkgroup +
                  lane * plan.channelsPerLane + member;
                if (channel >= plan.outputChannels) continue;
                const index = rowOffset * plan.outputChannels + channel;
                owners[index] = owners[index]! + 1;
              }
            }
          }
        }
      }
    }
  }
}

function shape(
  inputFrames: number,
  inputChannels: number,
  outputChannels: number,
  stride: number,
): AceVaeConvTranspose1dShape {
  return Object.freeze({
    batch: 1,
    inputFrames,
    inputChannels,
    outputChannels,
    kernelSize: stride * 2,
    stride,
    dilation: 1,
    padding: stride / 2,
    outputPadding: 0,
  });
}

type SelectorDevice = GPUDevice & {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
};

function selectorDevice(): SelectorDevice {
  return {
    features: new Set(["shader-f16", "subgroups"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupsPerDimension: 65_535,
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
  } as unknown as SelectorDevice;
}

function selectorBindings(
  plan: AceOpt0036VaeConvTranspose1dPlan,
) {
  return Object.freeze({
    input: selectorBinding(plan.inputBindingBytes),
    polyphaseWeight: selectorBinding(plan.weightBindingBytes),
    bias: selectorBinding(plan.biasBindingBytes),
    output: selectorBinding(plan.outputBindingBytes),
  });
}

function selectorBinding(size: number): GPUBufferBinding {
  return Object.freeze({ buffer: selectorBuffer(size), offset: 0, size });
}

function selectorBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function selectorPass(): GPUComputePassEncoder & {
  readonly setBindGroup: ReturnType<typeof vi.fn>;
} {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
  } as unknown as GPUComputePassEncoder & {
    readonly setBindGroup: ReturnType<typeof vi.fn>;
  };
}
