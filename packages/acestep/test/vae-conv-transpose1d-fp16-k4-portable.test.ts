import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

import { ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS } from "../src/model/manifest.js";
import {
  ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT,
  aceOpt0048VaeConvTranspose1dK4PackedWeightIndex,
  aceOpt0048VaeConvTranspose1dK4Wgsl,
  packAceOpt0048VaeConvTranspose1dK4WeightU16,
  planAceOpt0048VaeConvTranspose1dK4,
  planAceOpt0048VaeConvTranspose1dK4Range,
  planAceOpt0048VaeConvTranspose1dK4Weight,
  unpackAceOpt0048VaeConvTranspose1dK4WeightU16,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-partials.js";
import {
  ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SLICE_SIZE,
  ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SLICES,
  ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_WORKGROUP_SIZE,
  ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R4C8_K4_PORTABLE_KERNEL_ID,
  ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R8C4_K4_PORTABLE_KERNEL_ID,
  AceOpt0088VaeConvTranspose1dK4PortableKernel,
  aceOpt0088VaeConvTranspose1dK4PortableWgsl,
  planAceOpt0088VaeConvTranspose1dK4Portable,
  planAceOpt0088VaeConvTranspose1dK4PortableRange,
  type AceOpt0088VaeConvTranspose1dK4PortablePlan,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-portable.js";
import type { AceVaeConvTranspose1dShape } from
  "../src/webgpu/kernels/vae-primitives.js";

vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 << 2 });

const KERNEL_SOURCE = readFileSync(new URL(
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-portable.ts",
  import.meta.url,
), "utf8");

const SHAPES = Object.freeze([
  shape(300, 2_048, 1_024, 10),
  shape(3_000, 1_024, 512, 6),
  shape(18_000, 512, 256, 4),
  shape(72_000, 256, 128, 4),
  shape(288_000, 128, 128, 2),
]);

/** Arithmetic-critical WGSL statements that must match the subgroup owner. */
const SHARED_ARITHMETIC_PATTERNS = Object.freeze([
  /let weight_base = \(\(\(\(\(congruent_kernel[\s\S]*?OUTPUTS_PER_LANE\);/g,
  /let weight\d+ = packed_weight\[weight_base \+ \d+u\];/g,
  /if \(row_active\d+ && input_valid\d+\) \{\n        let partial[\s\S]*?\n      \}/g,
  /if \(tap == 0u\) \{[\s\S]*?input_valid\d+ = false;\n    \}/g,
  /if \(lane == \d+u && row_active\d+ && input_valid\d+\) \{[\s\S]*?\n      \}/g,
  /if \(row_active\d+\) \{\n    let output_base =[\s\S]*?\n  \}/g,
  /let bias_value\d = vec4<f32>\(\n(?:    f32\(bias\[output_channel[^\n]*\n)+  \);/g,
  /var sum\d+(?:_[01])? = bias_value\d(?:\.low|\.high)?;/g,
  /dot\(input_operand\d+, weight\d+\)/g,
] as const);

describe("OPT-0088 portable ConvTranspose FP16 K4", () => {
  it("mirrors OPT-0048's frozen plans behind portable kernel ids", () => {
    expect(ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R4C8_K4_PORTABLE_KERNEL_ID)
      .toBe("opt-0088-vae-conv-transpose1d-r4c8-k4-portable-v1");
    expect(ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R8C4_K4_PORTABLE_KERNEL_ID)
      .toBe("opt-0088-vae-conv-transpose1d-r8c4-k4-portable-v1");
    expect(ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SLICES).toBe(4);
    expect(ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SLICE_SIZE).toBe(32);
    expect(ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_WORKGROUP_SIZE)
      .toBe(128);
    for (const [index, candidate] of SHAPES.entries()) {
      const operationLabel = `block-${index}-conv-t1`;
      const portable = planAceOpt0088VaeConvTranspose1dK4Portable(
        operationLabel,
        candidate,
      );
      const sibling = planAceOpt0048VaeConvTranspose1dK4(
        operationLabel,
        candidate,
      );
      expect(portable).toMatchObject({
        operationLabel,
        reuseAxis: sibling.reuseAxis,
        kernelId: index < 3
          ? ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R4C8_K4_PORTABLE_KERNEL_ID
          : ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R8C4_K4_PORTABLE_KERNEL_ID,
        weightLayout: ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT,
        reductionSemantics:
          "increasing-tap-cin4-fp16-dot4-partials-fp32-running-state",
        rowsPerSubgroup: sibling.rowsPerSubgroup,
        channelsPerLane: sibling.channelsPerLane,
        accumulatorCountPerLane: 32,
        emulatedSliceCount: 4,
        emulatedSliceSize: 32,
        workgroupStorageBytes: 4 * sibling.rowsPerSubgroup * 8,
        workgroupBarrierCount: 4 * sibling.inputChannelK4Groups,
      });
      expect(portable.workgroupStorageBytes).toBeLessThanOrEqual(16_384);
      expect(portable.workgroupSize).toBeLessThanOrEqual(256);
      expect(portable.inputChannelK4Groups).toBe(sibling.inputChannelK4Groups);
      expect(portable.outputChannelTiles).toBe(sibling.outputChannelTiles);
      expect(portable.packedWeightStorageShape)
        .toEqual(sibling.packedWeightStorageShape);
      expect(Object.isFrozen(portable)).toBe(true);
      const fullRange = { base: 0, count: portable.outputElements };
      expect(planAceOpt0088VaeConvTranspose1dK4PortableRange(
        portable,
        fullRange,
      )).toEqual(planAceOpt0048VaeConvTranspose1dK4Range(sibling, fullRange));
    }
  });

  it("keeps both revision-7 K4 contract routes on the identical package layout", () => {
    expect(ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.map((contract) => [
      contract.operationLabel,
      contract.stride,
      contract.reuseAxis,
    ])).toEqual([
      ["block-1-conv-t1", 6, "channel"],
      ["block-2-conv-t1", 4, "channel"],
      ["block-3-conv-t1", 4, "row"],
      ["block-4-conv-t1", 2, "row"],
    ]);
    for (const contract of ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS) {
      const block = Number(contract.operationLabel.split("-")[1]);
      const plan = planAceOpt0088VaeConvTranspose1dK4Portable(
        contract.operationLabel,
        SHAPES[block]!,
      );
      expect(plan.reuseAxis).toBe(contract.reuseAxis);
      expect(plan.inputChannels).toBe(contract.inputChannels);
      expect(plan.outputChannels).toBe(contract.outputChannels);
      expect(plan.stride).toBe(contract.stride);
      expect(plan.weightLayout)
        .toBe(ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT);
      expect(plan.kernelId).toBe(contract.reuseAxis === "channel"
        ? ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R4C8_K4_PORTABLE_KERNEL_ID
        : ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R8C4_K4_PORTABLE_KERNEL_ID);
    }
  });

  it.each([
    ["channel" as const, 256],
    ["row" as const, 128],
  ])(
    "reuses OPT-0048's exact packed %s index bijection unchanged",
    (reuseAxis, outputChannels) => {
      const weight = planAceOpt0048VaeConvTranspose1dK4Weight({
        kernelSize: 4,
        stride: 2,
        dilation: 1,
        outputPadding: 0,
        inputChannels: 8,
        outputChannels,
      }, reuseAxis);
      const logical = Uint16Array.from(
        { length: weight.logicalWeightElements },
        (_, index) => (index * 4051 + 17) & 0xffff,
      );
      const packed = packAceOpt0048VaeConvTranspose1dK4WeightU16(
        logical,
        weight,
      );
      const visited = new Uint8Array(logical.length);
      for (let phase = 0; phase < weight.stride; phase += 1) {
        for (let tap = 0; tap < 2; tap += 1) {
          for (let input = 0; input < weight.inputChannels; input += 1) {
            for (let output = 0; output < weight.outputChannels; output += 1) {
              const physical = aceOpt0048VaeConvTranspose1dK4PackedWeightIndex(
                phase,
                tap,
                input,
                output,
                weight,
              );
              expect(visited[physical]).toBe(0);
              visited[physical] = 1;
              const logicalIndex = (((phase * 2 + tap) * weight.inputChannels +
                input) * weight.outputChannels + output);
              expect(packed[physical]).toBe(logical[logicalIndex]);
            }
          }
        }
      }
      expect(visited.every((entry) => entry === 1)).toBe(true);
      expect(unpackAceOpt0048VaeConvTranspose1dK4WeightU16(packed, weight))
        .toEqual(logical);
      // The portable module owns no index math of its own.
      expect(KERNEL_SOURCE).not.toContain("PackedWeightIndex");
      expect(KERNEL_SOURCE).not.toContain("visitPackedCoordinates");
    },
  );

  it.each([
    ["channel" as const, 1, 16],
    ["row" as const, 3, 32],
  ])(
    "emits %s WGSL without any subgroup mechanism",
    (_, block, stagedSlots) => {
      const operationLabel = `block-${block}-conv-t1`;
      const source = aceOpt0088VaeConvTranspose1dK4PortableWgsl(
        operationLabel,
        SHAPES[block]!,
      );
      expect(source).not.toContain("subgroup");
      expect(source).toContain("enable f16;");
      expect(source).not.toContain("enable subgroups");
      expect(source).toContain(
        `// weight-layout: ${ACE_OPT_0048_VAE_CONV_TRANSPOSE1D_K4_WEIGHT_LAYOUT}`,
      );
      expect(source).toContain(
        "@builtin(local_invocation_index) local_index: u32",
      );
      expect(source).toContain("let lane = local_index % 32u;");
      expect(source).toContain("let slice = local_index / 32u;");
      expect(source).toContain(
        `var<workgroup> staged_input:\n  array<vec4<f16>, ${stagedSlots}>;`,
      );
      expect(source).toContain("staged_input[slice_base + lane] = lane_input;");
      expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
      expect(source).toContain("input: array<vec4<f16>>");
      expect(source).toContain("packed_weight: array<vec4<f16>>");
      expect(source).toContain("\n  128, 1, 1,\n");
      expect(source.match(/dot\(input_operand\d+, weight\d+\)/g))
        .toHaveLength(32);
      expect(source.match(/let partial\d+_[01] = vec4<f16>\(/g))
        .toHaveLength(8);
      expect(source.match(/\+ vec4<f32>\(partial\d+_[01]\)/g))
        .toHaveLength(8);
      expect(source.match(/= f16\(sum/g)).toHaveLength(32);
      expect(source).not.toMatch(/var sum\d+(?:_[01])?\s*=\s*vec4<f16>/);
    },
  );

  it.each([
    ["channel" as const, 1],
    ["row" as const, 3],
  ])(
    "keeps every %s arithmetic statement byte-identical to OPT-0048",
    (_, block) => {
      const operationLabel = `block-${block}-conv-t1`;
      const portable = aceOpt0088VaeConvTranspose1dK4PortableWgsl(
        operationLabel,
        SHAPES[block]!,
      );
      const sibling = aceOpt0048VaeConvTranspose1dK4Wgsl(
        operationLabel,
        SHAPES[block]!,
      );
      for (const pattern of SHARED_ARITHMETIC_PATTERNS) {
        const portableMatches = portable.match(pattern);
        const siblingMatches = sibling.match(pattern);
        expect(portableMatches, String(pattern)).not.toBeNull();
        expect(portableMatches, String(pattern)).toEqual(siblingMatches);
      }
      // The only divergent input_operand source is the staged slice slot.
      const rows = block === 1 ? 4 : 8;
      for (let row = 0; row < rows; row += 1) {
        expect(portable).toContain(
          `let input_operand${row} = staged_input[slice_base + ${row}u];`,
        );
        expect(sibling).toContain(
          `let input_operand${row} = subgroupBroadcast(lane_input, ${row}u);`,
        );
      }
    },
  );

  it("fails closed on unregistered operations and shapes", () => {
    expect(() => planAceOpt0088VaeConvTranspose1dK4Portable(
      "block-5-conv-t1",
      SHAPES[0]!,
    )).toThrow(/no ConvTranspose1D route/);
    expect(() => planAceOpt0088VaeConvTranspose1dK4Portable(
      "block-1-conv-t1",
      SHAPES[2]!,
    )).toThrow(/authenticated shape/);
    const plan = planAceOpt0088VaeConvTranspose1dK4Portable(
      "block-4-conv-t1",
      SHAPES[4]!,
    );
    expect(() => planAceOpt0088VaeConvTranspose1dK4PortableRange(plan, {
      base: 1,
      count: plan.outputElements,
    })).toThrow(/complete in-bounds NLC rows/);
    expect(KERNEL_SOURCE).not.toContain("subgroupBroadcast");
    expect(KERNEL_SOURCE).not.toContain("enable subgroups");
  });

  it("hosts dispatches on shader-f16-only devices and dies closed", async () => {
    expect(() => AceOpt0088VaeConvTranspose1dK4PortableKernel.create(
      portableDevice(new Set(["subgroups"])),
    )).toThrow(/requires shader-f16/);

    const device = portableDevice(new Set(["shader-f16"]));
    const owner = AceOpt0088VaeConvTranspose1dK4PortableKernel.create(device);
    const channelShape = SHAPES[1]!;
    const channelPlan = planAceOpt0088VaeConvTranspose1dK4Portable(
      "block-1-conv-t1",
      channelShape,
    );
    const channelBindings = portableBindings(channelPlan);
    const controls = portableBuffer(512);
    const first = await owner.createDispatch(
      "channel-first",
      "block-1-conv-t1",
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
      "block-1-conv-t1",
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
    const rowPlan = planAceOpt0088VaeConvTranspose1dK4Portable(
      "block-3-conv-t1",
      rowShape,
    );
    const row = await owner.createDispatch(
      "row",
      "block-3-conv-t1",
      rowShape,
      portableBindings(rowPlan),
      {
        base: 0,
        count: rowPlan.outputElements,
        control: { buffer: controls, offset: 0, size: 16 },
      },
    );
    expect(device.createShaderModule).toHaveBeenCalledTimes(2);
    expect(row.kernelId).toBe(
      ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R8C4_K4_PORTABLE_KERNEL_ID,
    );
    expect(first.kernelId).toBe(
      ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R4C8_K4_PORTABLE_KERNEL_ID,
    );
    const sources = device.createShaderModule.mock.calls.map((call) =>
      String((call[0] as GPUShaderModuleDescriptor).code)
    );
    expect(sources[0]).toContain(
      ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R4C8_K4_PORTABLE_KERNEL_ID,
    );
    expect(sources[1]).toContain(
      ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_R8C4_K4_PORTABLE_KERNEL_ID,
    );
    for (const source of sources) {
      expect(source).toContain("enable f16;");
      expect(source).not.toContain("subgroup");
    }

    const firstPass = portablePass();
    const reusedPass = portablePass();
    first.encode(firstPass);
    reused.encode(reusedPass);
    row.encode(portablePass());
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
    expect(() => first.encode(portablePass())).toThrow(/destroyed/);
    await expect(owner.createDispatch(
      "after-destroy",
      "block-1-conv-t1",
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
    kernelSize: 2 * stride,
    stride,
    dilation: 1,
    padding: stride / 2,
    outputPadding: 0,
  });
}

type PortableDevice = GPUDevice & {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createComputePipelineAsync: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
};

function portableDevice(features: ReadonlySet<string>): PortableDevice {
  return {
    features,
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 16_384,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxStorageBufferBindingSize: 2_147_483_644,
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
  } as unknown as PortableDevice;
}

function portableBindings(plan: AceOpt0088VaeConvTranspose1dK4PortablePlan) {
  return Object.freeze({
    input: portableBinding(plan.inputBindingBytes),
    weight: portableBinding(plan.weightBindingBytes),
    bias: portableBinding(plan.biasBindingBytes),
    output: portableBinding(plan.outputBindingBytes),
  });
}

function portableBinding(size: number): GPUBufferBinding {
  return Object.freeze({ buffer: portableBuffer(size), offset: 0, size });
}

function portableBuffer(size: number): GPUBuffer {
  return { size } as GPUBuffer;
}

function portablePass(): GPUComputePassEncoder & {
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
