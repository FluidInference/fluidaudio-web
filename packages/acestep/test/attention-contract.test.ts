import { describe, expect, it, vi } from "vitest";

import {
  ACE_FIXED32_ATTENTION_GENERALIZED_MIN_QUERY_TOKENS,
  ACE_FIXED32_ATTENTION_QUERIES_PER_WORKGROUP,
  ACE_FIXED32_ATTENTION_WORKGROUP_BYTES,
  ACE_FIXED32_ATTENTION_WORKGROUP_SIZE,
  AceCorrectnessAttentionKernel,
  aceCorrectnessAttentionWgsl,
  aceFixed32TiledFullAttentionWgsl,
  isAceFixed32TiledFullAttentionShape,
  planAceAttention,
  planAceFixed32TiledFullAttention,
} from "../src/webgpu/kernels/attention.js";

describe("ACE online attention contract", () => {
  it("plans three-minute GQA without a score allocation", () => {
    expect(
      planAceAttention({
        batch: 1,
        queryHeads: 16,
        keyValueHeads: 8,
        queryTokens: 2_250,
        keyValueTokens: 2_250,
        headDimension: 128,
        mode: "full",
      }),
    ).toMatchObject({
      queryHeadsPerKeyValueHead: 2,
      queryElements: 4_608_000,
      keyValueElements: 2_304_000,
      outputElements: 4_608_000,
      workgroupsX: 36_000,
      workgroupsY: 1,
    });
  });

  it("plans the three-minute fixed-32 query8 tile and bounded scheduler ranges", () => {
    const plan = planAceFixed32TiledFullAttention({
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 2_250,
      keyValueTokens: 2_250,
      headDimension: 128,
      mode: "full",
    });
    expect(plan).toMatchObject({
      backend: "fixed32-subgroup-query8",
      workgroupSize: 256,
      workgroupStorageBytes: 1_024,
      queryTokensPerTile: 4,
      queriesPerWorkgroup: 8,
      queryTokenTiles: 563,
      workgroupCount: 4_504,
      outputRangeCount: 5,
      portableKeyValueScalarLoads: 20_736_000_000,
      tiledKeyValueScalarLoads: 2_594_304_000,
      portableBarriersPerKey: 10,
      tiledBarriersPerKey: 2,
    });
    expect(plan.keyValueLoadReduction).toBeCloseTo(7.9928952043, 9);
    expect(plan.outputRanges.map((range) => range.workgroupCount)).toEqual([
      932,
      932,
      932,
      932,
      776,
    ]);
    expect(
      plan.outputRanges.reduce((total, range) => total + range.outputCount, 0),
    ).toBe(plan.outputElements);
    expect(
      plan.outputRanges.every(
        (range) => range.multiplyAdds <= 2 * 1024 * 1024 * 1024,
      ),
    ).toBe(true);
  });

  it("emits eight subgroup queries over one staged K/V tile", () => {
    const source = aceFixed32TiledFullAttentionWgsl({
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 2_250,
      keyValueTokens: 2_250,
      headDimension: 128,
      mode: "full",
    });
    expect(ACE_FIXED32_ATTENTION_WORKGROUP_SIZE).toBe(256);
    expect(ACE_FIXED32_ATTENTION_QUERIES_PER_WORKGROUP).toBe(8);
    expect(ACE_FIXED32_ATTENTION_WORKGROUP_BYTES).toBe(1_024);
    expect(source).toContain("enable subgroups;");
    expect(source).toContain("@compute @workgroup_size(256, 1, 1)");
    expect(source).toContain("if (subgroup_size != 32u)");
    expect(source).toContain("subgroup / QUERY_TOKENS_PER_TILE");
    expect(source).toContain("subgroup % QUERY_TOKENS_PER_TILE");
    expect(source).toContain("var<workgroup> key_tile: array<f32, 128>");
    expect(source).toContain("var<workgroup> value_tile: array<f32, 128>");
    expect(source).toContain("let score = subgroupAdd(dot_partial) * ATTENTION_SCALE");
    expect(source).toContain(
      "online_denominator * online_alpha + online_beta",
    );
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    expect(source).not.toContain("array<f32, 2250>");
    expect(source).not.toContain("dot_partial: array");
  });

  it("plans production sliding and non-square cross attention with one range", () => {
    const sliding = planAceFixed32TiledFullAttention({
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 2_250,
      keyValueTokens: 2_250,
      headDimension: 128,
      mode: "sliding",
      slidingRadius: 128,
    });
    expect(sliding).toMatchObject({
      backend: "fixed32-subgroup-query8",
      workgroupCount: 4_504,
      outputRangeCount: 1,
      portableKeyValueScalarLoads: 2_300_878_848,
      tiledKeyValueScalarLoads: 291_000_320,
    });
    expect(sliding.keyValueLoadReduction).toBeCloseTo(7.9067914702, 9);
    expect(sliding.outputRanges).toHaveLength(1);
    expect(sliding.outputRanges[0]).toMatchObject({
      workgroupCount: 4_504,
      multiplyAdds: 1_198_080_000,
    });

    const cross = planAceFixed32TiledFullAttention({
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 2_250,
      keyValueTokens: 97,
      headDimension: 128,
      mode: "full",
    });
    expect(cross).toMatchObject({
      backend: "fixed32-subgroup-query8",
      workgroupCount: 4_504,
      outputRangeCount: 1,
      portableKeyValueScalarLoads: 893_952_000,
      tiledKeyValueScalarLoads: 111_843_328,
    });
    expect(cross.keyValueLoadReduction).toBeCloseTo(7.9928952043, 9);
    expect(cross.outputRanges[0]).toMatchObject({
      workgroupCount: 4_504,
      multiplyAdds: 446_976_000,
    });
  });

  it("gates only generalized query8 shapes at the M2250 production boundary", () => {
    expect(ACE_FIXED32_ATTENTION_GENERALIZED_MIN_QUERY_TOKENS).toBe(2_250);
    const shape = (
      queryTokens: number,
      mode: "full" | "sliding",
      keyValueTokens: number,
    ) => ({
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens,
      keyValueTokens,
      headDimension: 128,
      mode,
      ...(mode === "sliding" ? { slidingRadius: 128 } : {}),
    });

    expect(isAceFixed32TiledFullAttentionShape(shape(150, "full", 150)))
      .toBe(true);
    for (const queryTokens of [2_249, 2_250]) {
      const expected = queryTokens === 2_250;
      expect(isAceFixed32TiledFullAttentionShape(
        shape(queryTokens, "sliding", queryTokens),
      )).toBe(expected);
      expect(isAceFixed32TiledFullAttentionShape(
        shape(queryTokens, "full", 97),
      )).toBe(expected);
    }
    expect(() =>
      planAceFixed32TiledFullAttention(shape(2_249, "sliding", 2_249))
    ).toThrow(/at least 2250 query tokens/);
    expect(() =>
      planAceFixed32TiledFullAttention(shape(2_249, "full", 97))
    ).toThrow(/at least 2250 query tokens/);
  });

  it("keeps causal, masked, and unsupported geometry on the portable oracle", () => {
    for (const shape of [
      {
        batch: 1,
        queryHeads: 2,
        keyValueHeads: 1,
        queryTokens: 4,
        keyValueTokens: 4,
        headDimension: 128,
        mode: "causal" as const,
        queryPositionOffset: 0,
      },
      {
        batch: 1,
        queryHeads: 16,
        keyValueHeads: 8,
        queryTokens: 2_250,
        keyValueTokens: 2_250,
        headDimension: 128,
        mode: "causal" as const,
        keyValidity: "causal-per-key" as const,
      },
      {
        batch: 1,
        queryHeads: 4,
        keyValueHeads: 1,
        queryTokens: 4,
        keyValueTokens: 1,
        headDimension: 128,
        mode: "full" as const,
      },
    ]) {
      expect(() => planAceFixed32TiledFullAttention(shape)).toThrow(
        /requires square unmasked full GQA2/,
      );
    }
  });

  it("pins the inclusive 257-key local window", () => {
    const plan = planAceAttention({
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 2_250,
      keyValueTokens: 2_250,
      headDimension: 128,
      mode: "sliding",
      slidingRadius: 128,
    });
    expect(plan.slidingRadius).toBe(128);
    const source = aceCorrectnessAttentionWgsl("reference-bf16", plan);
    expect(source).toContain("query_token + SLIDING_RADIUS + 1u");
    expect(source).toContain("online_denominator");
    expect(source).not.toContain("array<f32, 2250>");
  });

  it("emits one sliding tile union with subgroup-local key admission", () => {
    const source = aceFixed32TiledFullAttentionWgsl({
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 2_250,
      keyValueTokens: 2_250,
      headDimension: 128,
      mode: "sliding",
      slidingRadius: 128,
    });
    expect(source).toContain("const SLIDING_RADIUS: u32 = 128u");
    expect(source).toContain("let tile_first_query = query_tile * QUERY_TOKENS_PER_TILE");
    expect(source).toContain("tile_last_query + SLIDING_RADIUS + 1u");
    expect(source).toContain("key_token >= query_key_start");
    expect(source).toContain("key_token < query_key_end");
    expect(source).toContain("var key_token = tile_key_start");
    expect(source).toContain("key_token < tile_key_end");
    expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
  });

  it("emits causal bounds for planner attention", () => {
    const source = aceCorrectnessAttentionWgsl("raw-fp16", {
      batch: 2,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 16,
      keyValueTokens: 16,
      headDimension: 128,
      mode: "causal",
      queryPositionOffset: 0,
    });
    expect(source).toContain(
      "min(valid_key_tokens, QUERY_POSITION_OFFSET + query_token + 1u)",
    );
    expect(source).toContain("var weighted_value = 0.0");
    expect(source).toContain("f16(weighted_value / online_denominator)");
  });

  it("positions an incremental planner query at the end of its KV cache", () => {
    const source = aceCorrectnessAttentionWgsl("reference-bf16", {
      batch: 2,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 1,
      keyValueTokens: 137,
      headDimension: 128,
      mode: "causal",
      queryPositionOffset: 136,
    });
    expect(source).toContain("const QUERY_POSITION_OFFSET: u32 = 136u");
  });

  it("pins arbitrary planner key validity to bound physical query positions", () => {
    const plan = planAceAttention({
      batch: 2,
      queryHeads: 4,
      keyValueHeads: 2,
      queryTokens: 1,
      keyValueTokens: 7,
      headDimension: 8,
      mode: "causal",
      keyValidity: "causal-per-key",
    });
    expect(plan).toMatchObject({
      keyValidity: "causal-per-key",
      keyValidityElements: 14,
      queryPositionElements: 2,
    });
    const source = aceCorrectnessAttentionWgsl("reference-bf16", plan);
    expect(source).toContain("@binding(5) var<storage, read> key_validity: array<u32>");
    expect(source).toContain("@binding(6) var<storage, read> query_positions: array<u32>");
    expect(source).toContain("physical_query_position < valid_key_tokens");
    expect(source).toContain(
      "key_validity[batch * KV_TOKENS + key_token] == 1u",
    );
    expect(source).toContain("if (online_denominator > 0.0)");
  });

  it("cannot accidentally apply planner validity to full or sliding DiT attention", () => {
    expect(() =>
      planAceAttention({
        batch: 1,
        queryHeads: 1,
        keyValueHeads: 1,
        queryTokens: 2,
        keyValueTokens: 2,
        headDimension: 4,
        mode: "full",
        keyValidity: "causal-per-key",
      }),
    ).toThrow(/only in causal mode/);
    const source = aceCorrectnessAttentionWgsl("reference-bf16", {
      batch: 1,
      queryHeads: 1,
      keyValueHeads: 1,
      queryTokens: 2,
      keyValueTokens: 2,
      headDimension: 4,
      mode: "full",
    });
    expect(source).not.toContain("key_validity");
    expect(source).not.toContain("query_positions");
  });

  it("rejects static offsets and incomplete bindings in planner mask mode", async () => {
    const shape = {
      batch: 1,
      queryHeads: 1,
      keyValueHeads: 1,
      queryTokens: 1,
      keyValueTokens: 2,
      headDimension: 1,
      mode: "causal" as const,
      keyValidity: "causal-per-key" as const,
    };
    expect(() => planAceAttention({ ...shape, queryPositionOffset: 1 })).toThrow(
      /bound physical positions/,
    );
    const kernel = AceCorrectnessAttentionKernel.create(fakeAttentionDevice(), "reference-bf16");
    try {
      await expect(
        kernel.createDispatch("missing-mask", shape, {
          query: fakeBinding(4),
          key: fakeBinding(8),
          value: fakeBinding(8),
          validLengths: fakeBinding(8),
          output: fakeBinding(4),
        }),
      ).rejects.toThrow(/requires keyValidity and queryPositions/);
    } finally {
      kernel.destroy();
    }
  });

  it("rejects planner-only bindings from an unmasked attention dispatch", async () => {
    const kernel = AceCorrectnessAttentionKernel.create(fakeAttentionDevice(), "reference-bf16");
    try {
      await expect(
        kernel.createDispatch(
          "dit-must-stay-unmasked",
          {
            batch: 1,
            queryHeads: 1,
            keyValueHeads: 1,
            queryTokens: 1,
            keyValueTokens: 1,
            headDimension: 1,
            mode: "full",
          },
          {
            query: fakeBinding(4),
            key: fakeBinding(4),
            value: fakeBinding(4),
            validLengths: fakeBinding(8),
            output: fakeBinding(4),
            keyValidity: fakeBinding(4),
            queryPositions: fakeBinding(4),
          },
        ),
      ).rejects.toThrow(/require causal-per-key attention/);
    } finally {
      kernel.destroy();
    }
  });

  it.each([
    "query",
    "key",
    "value",
    "validLengths",
    "keyValidity",
    "queryPositions",
  ] as const)("rejects output aliasing the %s input", async (aliasedInput) => {
    const kernel = AceCorrectnessAttentionKernel.create(
      fakeAttentionDevice(),
      "reference-bf16",
    );
    const shared = { size: 128 } as GPUBuffer;
    const independent = (): GPUBufferBinding => fakeBinding(128);
    const bindings = {
      query: independent(),
      key: independent(),
      value: independent(),
      validLengths: independent(),
      output: { buffer: shared, offset: 32, size: 16 },
      keyValidity: independent(),
      queryPositions: independent(),
    } satisfies Record<string, GPUBufferBinding>;
    bindings[aliasedInput] = {
      buffer: shared,
      offset: 40,
      size: 32,
    };
    try {
      await expect(kernel.createDispatch(
        `alias-${aliasedInput}`,
        {
          batch: 1,
          queryHeads: 1,
          keyValueHeads: 1,
          queryTokens: 1,
          keyValueTokens: 1,
          headDimension: 1,
          mode: "causal",
          keyValidity: "causal-per-key",
        },
        bindings,
      )).rejects.toThrow(/output must not overlap/);
    } finally {
      kernel.destroy();
    }
  });

  it.each([
    {
      batch: 1,
      queryHeads: 15,
      keyValueHeads: 8,
      queryTokens: 1,
      keyValueTokens: 1,
      headDimension: 128,
      mode: "full" as const,
    },
    {
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 1,
      keyValueTokens: 1,
      headDimension: 256,
      mode: "full" as const,
    },
    {
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 1,
      keyValueTokens: 1,
      headDimension: 128,
      mode: "full" as const,
      slidingRadius: 1,
    },
  ])("rejects malformed attention geometry", (shape) => {
    expect(() => planAceAttention(shape)).toThrow();
  });

  it("requires authenticated fixed-32 capability for the query8 backend", () => {
    for (const options of [
      { features: [] as string[], subgroupMinSize: 32, subgroupMaxSize: 32 },
      { features: ["subgroups"], subgroupMinSize: 16, subgroupMaxSize: 32 },
      { features: ["subgroups"], subgroupMinSize: 32, subgroupMaxSize: 64 },
    ]) {
      expect(() =>
        AceCorrectnessAttentionKernel.create(
          fakeFixed32AttentionDevice(options.features),
          "reference-bf16",
          {
            backend: "fixed32-subgroup-query8",
            capability: {
              subgroupMinSize: options.subgroupMinSize,
              subgroupMaxSize: options.subgroupMaxSize,
            },
          },
        )
      ).toThrow(/reported fixed 32-lane subgroups/);
    }
    expect(() =>
      AceCorrectnessAttentionKernel.create(
        fakeFixed32AttentionDevice(["subgroups", "shader-f16"]),
        "raw-fp16",
        {
          backend: "fixed32-subgroup-query8",
          capability: { subgroupMinSize: 32, subgroupMaxSize: 32 },
        },
      )
    ).toThrow(/requires reference-bf16/);
  });

  it("encodes fixed-32 query8 attention through bounded production ranges", async () => {
    const device = fakeFixed32AttentionDevice(["subgroups"]);
    const kernel = AceCorrectnessAttentionKernel.create(
      device,
      "reference-bf16",
      {
        backend: "fixed32-subgroup-query8",
        capability: { subgroupMinSize: 32, subgroupMaxSize: 32 },
      },
    );
    try {
      const dispatch = await kernel.createDispatch(
        "fixed32-query8",
        {
          batch: 1,
          queryHeads: 2,
          keyValueHeads: 1,
          queryTokens: 5,
          keyValueTokens: 5,
          headDimension: 128,
          mode: "full",
        },
        {
          query: fakeBinding(2 * 5 * 128 * 4),
          key: fakeBinding(5 * 128 * 4),
          value: fakeBinding(5 * 128 * 4),
          validLengths: fakeBinding(8),
          output: fakeBinding(2 * 5 * 128 * 4),
        },
      );
      expect(dispatch.backend).toBe("fixed32-subgroup-query8");
      if (dispatch.backend !== "fixed32-subgroup-query8") {
        throw new Error("expected fixed32 attention dispatch");
      }
      expect(dispatch.plan).toMatchObject({
        queryTokenTiles: 2,
        workgroupCount: 2,
        outputRangeCount: 1,
      });
      const pass = fakeComputePass();
      dispatch.encodeRange(pass, 0);
      expect(pass.dispatchWorkgroups).toHaveBeenCalledWith(2, 1, 1);
      expect(() => dispatch.encodeRange(pass, 1)).toThrow(/outside/);
      expect(device.createBindGroup).toHaveBeenCalledTimes(1);
      expect(device.createShaderModule).toHaveBeenCalledWith(
        expect.objectContaining({
          code: expect.stringContaining("subgroupAdd(dot_partial)"),
        }),
      );
    } finally {
      kernel.destroy();
    }
    await Promise.resolve();
    expect(device.rangeParameters.destroy).toHaveBeenCalledTimes(1);
  });

  it("keeps short generalized attention portable and selects it at M2250", async () => {
    const device = fakeFixed32AttentionDevice(["subgroups"]);
    const kernel = AceCorrectnessAttentionKernel.create(
      device,
      "reference-bf16",
      {
        backend: "fixed32-subgroup-query8",
        capability: { subgroupMinSize: 32, subgroupMaxSize: 32 },
      },
    );
    try {
      const bindings = (queryTokens: number, keyValueTokens: number) => ({
        query: fakeBinding(2 * queryTokens * 128 * 4),
        key: fakeBinding(keyValueTokens * 128 * 4),
        value: fakeBinding(keyValueTokens * 128 * 4),
        validLengths: fakeBinding(8),
        output: fakeBinding(2 * queryTokens * 128 * 4),
      });
      const shortFull = await kernel.createDispatch(
        "fixed32-short-full",
        {
          batch: 1,
          queryHeads: 2,
          keyValueHeads: 1,
          queryTokens: 150,
          keyValueTokens: 150,
          headDimension: 128,
          mode: "full",
        },
        bindings(150, 150),
      );
      const shortSliding = await kernel.createDispatch(
        "fixed32-short-sliding",
        {
          batch: 1,
          queryHeads: 2,
          keyValueHeads: 1,
          queryTokens: 150,
          keyValueTokens: 150,
          headDimension: 128,
          mode: "sliding",
          slidingRadius: 128,
        },
        bindings(150, 150),
      );
      const shortCross = await kernel.createDispatch(
        "fixed32-short-cross",
        {
          batch: 1,
          queryHeads: 2,
          keyValueHeads: 1,
          queryTokens: 150,
          keyValueTokens: 97,
          headDimension: 128,
          mode: "full",
        },
        bindings(150, 97),
      );
      const longSliding = await kernel.createDispatch(
        "fixed32-long-sliding",
        {
          batch: 1,
          queryHeads: 2,
          keyValueHeads: 1,
          queryTokens: 2_250,
          keyValueTokens: 2_250,
          headDimension: 128,
          mode: "sliding",
          slidingRadius: 128,
        },
        bindings(2_250, 2_250),
      );
      const longCross = await kernel.createDispatch(
        "fixed32-long-cross",
        {
          batch: 1,
          queryHeads: 2,
          keyValueHeads: 1,
          queryTokens: 2_250,
          keyValueTokens: 97,
          headDimension: 128,
          mode: "full",
        },
        bindings(2_250, 97),
      );
      expect(shortFull.backend).toBe("fixed32-subgroup-query8");
      expect(shortSliding.backend).toBe("portable");
      expect(shortCross.backend).toBe("portable");
      expect(longSliding.backend).toBe("fixed32-subgroup-query8");
      expect(longCross.backend).toBe("fixed32-subgroup-query8");
      expect(device.createShaderModule).toHaveBeenCalledTimes(5);
      expect(device.createShaderModule).toHaveBeenNthCalledWith(
        4,
        expect.objectContaining({
          code: expect.stringContaining("key_token >= query_key_start"),
        }),
      );
    } finally {
      kernel.destroy();
    }
  });

  it("leaves causal attention on the portable oracle under fixed32 configuration", async () => {
    const device = fakeFixed32AttentionDevice(["subgroups"]);
    const kernel = AceCorrectnessAttentionKernel.create(
      device,
      "reference-bf16",
      {
        backend: "fixed32-subgroup-query8",
        capability: { subgroupMinSize: 32, subgroupMaxSize: 32 },
      },
    );
    try {
      const dispatch = await kernel.createDispatch(
        "fixed32-config-causal-oracle",
        {
          batch: 1,
          queryHeads: 2,
          keyValueHeads: 1,
          queryTokens: 5,
          keyValueTokens: 5,
          headDimension: 128,
          mode: "causal",
          queryPositionOffset: 0,
        },
        {
          query: fakeBinding(2 * 5 * 128 * 4),
          key: fakeBinding(5 * 128 * 4),
          value: fakeBinding(5 * 128 * 4),
          validLengths: fakeBinding(8),
          output: fakeBinding(2 * 5 * 128 * 4),
        },
      );
      expect(dispatch.backend).toBe("portable");
    } finally {
      kernel.destroy();
    }
  });

  it("fails closed for unknown modes and profiles", () => {
    expect(() =>
      planAceAttention({
        batch: 1,
        queryHeads: 1,
        keyValueHeads: 1,
        queryTokens: 1,
        keyValueTokens: 1,
        headDimension: 1,
        mode: "future" as never,
      }),
    ).toThrow(/Unknown ACE attention mode/);
    expect(() =>
      aceCorrectnessAttentionWgsl("future" as never, {
        batch: 1,
        queryHeads: 1,
        keyValueHeads: 1,
        queryTokens: 1,
        keyValueTokens: 1,
        headDimension: 1,
        mode: "full",
      }),
    ).toThrow(/Unknown ACE attention model profile/);
    expect(() =>
      planAceAttention({
        batch: 1,
        queryHeads: 1,
        keyValueHeads: 1,
        queryTokens: 1,
        keyValueTokens: 1,
        headDimension: 1,
        mode: "causal",
        keyValidity: "future" as never,
      }),
    ).toThrow(/Unknown ACE attention key-validity mode/);
  });
});

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: { size } as GPUBuffer, offset: 0, size };
}

function fakeAttentionDevice(): GPUDevice {
  return {
    features: new Set<GPUFeatureName>(),
    limits: {
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 128,
      maxComputeWorkgroupStorageSize: 528,
    },
  } as unknown as GPUDevice;
}

function fakeFixed32AttentionDevice(features: readonly string[]): GPUDevice & {
  readonly createShaderModule: ReturnType<typeof vi.fn>;
  readonly createBindGroup: ReturnType<typeof vi.fn>;
  readonly rangeParameters: ReturnType<typeof fakeMappedBuffer>;
} {
  const rangeParameters = fakeMappedBuffer(64 * 1024);
  return {
    features: new Set<GPUFeatureName>(features as GPUFeatureName[]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 16 * 1024,
      minUniformBufferOffsetAlignment: 256,
    },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => null),
    createShaderModule: vi.fn(() => ({ label: "attention-module" })),
    createComputePipelineAsync: vi.fn(async () => fakePipeline()),
    createBuffer: vi.fn(() => rangeParameters),
    createBindGroup: vi.fn(() => ({ label: "attention-bindings" })),
    rangeParameters,
  } as unknown as GPUDevice & {
    readonly createShaderModule: ReturnType<typeof vi.fn>;
    readonly createBindGroup: ReturnType<typeof vi.fn>;
    readonly rangeParameters: ReturnType<typeof fakeMappedBuffer>;
  };
}

function fakePipeline(): GPUComputePipeline {
  return {
    getBindGroupLayout: vi.fn(() => ({ label: "attention-layout" })),
  } as unknown as GPUComputePipeline;
}

function fakeMappedBuffer(size: number) {
  const mapped = new ArrayBuffer(size);
  return {
    size,
    getMappedRange: vi.fn(() => mapped),
    unmap: vi.fn(),
    destroy: vi.fn(),
  } as unknown as GPUBuffer & {
    readonly destroy: ReturnType<typeof vi.fn>;
  };
}

function fakeComputePass(): GPUComputePassEncoder & {
  readonly dispatchWorkgroups: ReturnType<typeof vi.fn>;
} {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
  } as unknown as GPUComputePassEncoder & {
    readonly dispatchWorkgroups: ReturnType<typeof vi.fn>;
  };
}

vi.stubGlobal("GPUBufferUsage", { UNIFORM: 1 << 6 });
