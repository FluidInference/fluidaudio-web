import { describe, expect, it } from "vitest";

import {
  aceCorrectnessBatchedRopeWgsl,
  planAceBatchedRope,
} from "../src/webgpu/kernels/batched-rope.js";

describe("ACE batched RoPE contract", () => {
  it("indexes one independent table row per batch and token", () => {
    const plan = planAceBatchedRope({
      batch: 2,
      heads: 16,
      tokens: 3,
      headDimension: 128,
    });
    expect(plan).toMatchObject({
      scalarCount: 12_288,
      tableScalarCount: 768,
      workgroupsX: 48,
      workgroupsY: 1,
    });
    const source = aceCorrectnessBatchedRopeWgsl("reference-bf16", plan);
    expect(source).toContain(
      "let table_index = (batch * TOKENS + token) * HEAD_DIMENSION + dimension;",
    );
  });

  it("fails closed for malformed shapes and profiles", () => {
    expect(() =>
      planAceBatchedRope({
        batch: 1,
        heads: 1,
        tokens: 1,
        headDimension: 3,
      }),
    ).toThrow(/must be even/);
    expect(() =>
      aceCorrectnessBatchedRopeWgsl("future" as never, {
        batch: 1,
        heads: 1,
        tokens: 1,
        headDimension: 2,
      }),
    ).toThrow(/Unknown ACE batched RoPE model profile/);
  });
});
