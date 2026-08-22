import { describe, expect, it } from "vitest";

import {
  aceCorrectnessRmsNormWgsl,
  planAceRmsNorm,
} from "../src/webgpu/kernels/rmsnorm.js";

describe("ACE correctness RMSNorm contract", () => {
  it("plans hidden and flattened head rows without dense statistics", () => {
    expect(planAceRmsNorm({ rows: 2_250, width: 2_048, epsilon: 1e-6 })).toEqual({
      rows: 2_250,
      width: 2_048,
      epsilon: 1e-6,
      elements: 4_608_000,
      workgroupsX: 2_250,
      workgroupsY: 1,
    });
    expect(planAceRmsNorm({ rows: 80_000, width: 128, epsilon: 1e-6 })).toMatchObject({
      workgroupsX: 65_535,
      workgroupsY: 2,
    });
  });

  it.each([
    { rows: 0, width: 2, epsilon: 1e-6 },
    { rows: 1, width: 0, epsilon: 1e-6 },
    { rows: 1, width: 2, epsilon: 0 },
    { rows: 1, width: 2, epsilon: Number.NaN },
  ])("rejects malformed norm geometry", (shape) => {
    expect(() => planAceRmsNorm(shape)).toThrow();
  });

  it("emits FP32 reduction state for packed BF16 weights", () => {
    const source = aceCorrectnessRmsNormWgsl(
      "reference-bf16",
      { rows: 2, width: 2_048, epsilon: 1e-6 },
    );
    expect(source).toContain("partial_squares: array<f32, 256>");
    expect(source).toContain("inverseSqrt");
    expect(source).toContain("bits16 << 16u");
    expect(source).not.toContain("enable f16");
  });

  it("rounds normalized values to FP16 before the learned FP16 scale", () => {
    const source = aceCorrectnessRmsNormWgsl(
      "raw-fp16",
      { rows: 2, width: 1_024, epsilon: 1e-6 },
    );
    expect(source).toContain("enable f16");
    expect(source).toContain("let value = f32(input[index])");
    expect(source).toContain("f16(value * inverse_rms) * weight[column]");
  });

  it("fails closed for an unknown JavaScript profile", () => {
    expect(() =>
      aceCorrectnessRmsNormWgsl(
        "unknown" as never,
        { rows: 1, width: 1, epsilon: 1e-6 },
      ),
    ).toThrow(/Unknown ACE RMSNorm model profile/);
  });
});
