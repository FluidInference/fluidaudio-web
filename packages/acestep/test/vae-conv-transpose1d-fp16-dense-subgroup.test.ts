import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE,
  ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP,
  ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP,
  ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP,
  aceOpt0029VaeConvTranspose1dWgsl,
  planAceOpt0029VaeConvTranspose1d,
  planAceOpt0029VaeConvTranspose1dRange,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-dense-subgroup.js";
import type { AceVaeConvTranspose1dShape } from
  "../src/webgpu/kernels/vae-primitives.js";

const SHAPES = Object.freeze([
  shape(300, 2_048, 1_024, 10),
  shape(3_000, 1_024, 512, 6),
  shape(18_000, 512, 256, 4),
  shape(72_000, 256, 128, 4),
  shape(288_000, 128, 128, 2),
]);

describe("OPT-0029 dense subgroup ConvTranspose1D", () => {
  it("maps the production topology onto 32 rows by 256 channels", () => {
    expect(ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_SUBGROUP).toBe(8);
    expect(ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_ROWS_PER_WORKGROUP).toBe(32);
    expect(ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_LANE).toBe(8);
    expect(ACE_OPT_0029_VAE_CONV_TRANSPOSE1D_CHANNELS_PER_WORKGROUP).toBe(256);
    for (const candidate of SHAPES) {
      const plan = planAceOpt0029VaeConvTranspose1d(candidate);
      const range = planAceOpt0029VaeConvTranspose1dRange(plan, {
        base: 0,
        count: plan.outputElements,
      });
      expect(range.workgroupsY).toBe(Math.ceil(candidate.outputChannels / 256));
      expect(range.workgroupsZ).toBe(candidate.stride);
    }
  });

  it("owns every output coordinate once, including the 128-channel tail", () => {
    const candidate = shape(17, 64, 128, 6);
    const plan = planAceOpt0029VaeConvTranspose1d(candidate);
    const firstOutputTime = 5;
    const outputRows = 53;
    const range = planAceOpt0029VaeConvTranspose1dRange(plan, {
      base: firstOutputTime * candidate.outputChannels,
      count: outputRows * candidate.outputChannels,
    });
    const owners = new Uint8Array(range.count);
    for (let x = 0; x < range.workgroupsX; x += 1) {
      for (let y = 0; y < range.workgroupsY; y += 1) {
        for (let phase = 0; phase < range.workgroupsZ; phase += 1) {
          for (let subgroup = 0; subgroup < 4; subgroup += 1) {
            for (let lane = 0; lane < 32; lane += 1) {
              for (let row = 0; row < 8; row += 1) {
                const phaseRow = x * 32 + subgroup * 8 + row;
                const rowOffset = phase + phaseRow * candidate.stride;
                if (rowOffset >= outputRows) continue;
                for (let member = 0; member < 8; member += 1) {
                  const channel = y * 256 + lane * 8 + member;
                  if (channel >= candidate.outputChannels) continue;
                  const index = rowOffset * candidate.outputChannels + channel;
                  owners[index] = owners[index]! + 1;
                }
              }
            }
          }
        }
      }
    }
    expect([...owners].every((owner) => owner === 1)).toBe(true);
  });

  it("emits tap-major source-order FP32 accumulation without shared memory", () => {
    const source = aceOpt0029VaeConvTranspose1dWgsl(SHAPES[1]!);
    expect(source).toContain("group.y * 256u + lane * 8u");
    expect(source).toContain("var sum7_1 = bias_value.high");
    expect(source).toContain("for (var tap = 0u; tap < 2u; tap += 1u)");
    expect(source).toContain(
      "for (var inner = 0u; inner < INPUT_CHANNELS; inner += 1u)",
    );
    expect(source).toContain("subgroupBroadcast(lane_input, 7u)");
    expect(source).toContain("polyphase_weight[weight_word]");
    expect(source).not.toContain("workgroupBarrier");
    expect(source).not.toContain("var<workgroup>");
  });

  it("rejects non-production geometry and unaligned ranges", () => {
    expect(() => planAceOpt0029VaeConvTranspose1d(
      shape(8, 64, 64, 2),
    )).toThrow(/production polyphase/);
    const plan = planAceOpt0029VaeConvTranspose1d(SHAPES[4]!);
    expect(() => planAceOpt0029VaeConvTranspose1dRange(plan, {
      base: 1,
      count: plan.outputElements,
    })).toThrow(/complete in-bounds NLC rows/);
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
    kernelSize: stride * 2,
    stride,
    dilation: 1,
    padding: stride / 2,
    outputPadding: 0,
  });
}
