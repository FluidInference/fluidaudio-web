import { describe, expect, it } from "vitest";

import {
  aceOpt0030AttentionWgsl,
  planAceOpt0030Attention,
} from "../src/webgpu/kernels/attention-query32.js";

const SHAPE = Object.freeze({
  batch: 1,
  queryHeads: 16,
  keyValueHeads: 8,
  queryTokens: 2_250,
  keyValueTokens: 2_250,
  headDimension: 128,
  mode: "full" as const,
});

describe("OPT-0030 query16/query32 attention", () => {
  it("uses the stock 512/1024-thread adapter limits to reduce K/V traffic", () => {
    const query8 = planAceOpt0030Attention(SHAPE, 8);
    const query16 = planAceOpt0030Attention(SHAPE, 16);
    const query32 = planAceOpt0030Attention(SHAPE, 32);
    expect(query8.workgroupSize).toBe(256);
    expect(query8.loadReductionVersusQuery8).toBe(1);
    expect(query16.workgroupSize).toBe(512);
    expect(query16.queryTokensPerTile).toBe(8);
    expect(query16.loadReductionVersusQuery8).toBe(2);
    expect(query32.workgroupSize).toBe(1_024);
    expect(query32.queryTokensPerTile).toBe(16);
    expect(query32.loadReductionVersusQuery8).toBe(4);
    expect(query16.keyValueScalarLoads).toBeLessThan(
      query16.query8KeyValueScalarLoads,
    );
    expect(query32.keyValueScalarLoads).toBeLessThan(
      query16.keyValueScalarLoads,
    );
  });

  it("retains one subgroup stream and ascending-key online softmax", () => {
    for (const queries of [16, 32] as const) {
      const source = aceOpt0030AttentionWgsl(SHAPE, queries);
      expect(source).toContain(`@compute @workgroup_size(${queries * 32}, 1, 1)`);
      expect(source).toContain("let score = subgroupAdd(dot_partial)");
      expect(source).toContain(
        "for (var key_token = 0u; key_token < valid_key_tokens; key_token += 1u)",
      );
      expect(source.match(/workgroupBarrier\(\)/gu)).toHaveLength(2);
      expect(source).not.toContain("array<f32, 2250>");
    }
  });

  it("rejects shapes outside the integrated long full GQA2 contract", () => {
    expect(() => planAceOpt0030Attention({ ...SHAPE, queryTokens: 2_249 }, 16))
      .toThrow(/long unmasked full/);
    expect(() => planAceOpt0030Attention({
      ...SHAPE,
      mode: "sliding",
      slidingRadius: 128,
    }, 32))
      .toThrow(/long unmasked full/);
  });
});
