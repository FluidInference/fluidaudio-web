import { describe, expect, it } from "vitest";

import {
  aceCorrectnessRopeWgsl,
  planAceRope,
} from "../src/webgpu/kernels/rope.js";

describe("ACE correctness RoPE contract", () => {
  it("plans DiT Q and GQA K tensors", () => {
    expect(planAceRope({ batch: 1, heads: 16, tokens: 2_250, headDimension: 128 })).toEqual({
      batch: 1,
      heads: 16,
      tokens: 2_250,
      headDimension: 128,
      scalarCount: 4_608_000,
      workgroupsX: 18_000,
      workgroupsY: 1,
    });
    expect(planAceRope({ batch: 1, heads: 8, tokens: 2_250, headDimension: 128 })).toMatchObject({
      scalarCount: 2_304_000,
      workgroupsX: 9_000,
    });
  });

  it.each([
    { batch: 0, heads: 1, tokens: 1, headDimension: 128 },
    { batch: 1, heads: -1, tokens: 1, headDimension: 128 },
    { batch: 1, heads: 1, tokens: 0, headDimension: 128 },
    { batch: 1, heads: 1, tokens: 1, headDimension: 127 },
  ])("rejects malformed rotary geometry", (shape) => {
    expect(() => planAceRope(shape)).toThrow();
  });

  it("emits split-half rotation and flattened head/token indexing", () => {
    const source = aceCorrectnessRopeWgsl(
      "reference-bf16",
      { batch: 1, heads: 16, tokens: 4, headDimension: 128 },
    );
    expect(source).toContain("const HALF_HEAD: u32 = 64u");
    expect(source).toContain("dimension + HALF_HEAD");
    expect(source).toContain("rotated_sign");
    expect(source).toContain("% TOKENS");
  });

  it("rounds only the final rotary output for the raw-FP16 graph", () => {
    const source = aceCorrectnessRopeWgsl(
      "raw-fp16",
      { batch: 1, heads: 8, tokens: 4, headDimension: 128 },
    );
    expect(source).toContain("enable f16");
    expect(source).toContain("f32(input[index])");
    expect(source).toContain("output[index] = f16(value)");
  });

  it("fails closed for an unknown JavaScript profile", () => {
    expect(() =>
      aceCorrectnessRopeWgsl(
        "unknown" as never,
        { batch: 1, heads: 1, tokens: 1, headDimension: 2 },
      ),
    ).toThrow(/Unknown ACE RoPE model profile/);
  });
});
