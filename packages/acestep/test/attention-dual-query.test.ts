import { describe, expect, it } from "vitest";

import browserSource from
  "./browser/opt-0039-dit-attention-dual-query.ts?raw";
import browserHtml from
  "./browser/opt-0039-dit-attention-dual-query.html?raw";
import { aceFixed32TiledFullAttentionWgsl } from
  "../src/webgpu/kernels/attention.js";
import {
  ACE_OPT_0039_ATTENTION_BARRIERS_PER_KEY,
  ACE_OPT_0039_ATTENTION_HEAD_DIMENSION,
  ACE_OPT_0039_ATTENTION_QUERIES_PER_WORKGROUP,
  ACE_OPT_0039_ATTENTION_QUERY_TOKENS_PER_SUBGROUP,
  ACE_OPT_0039_ATTENTION_QUERY_TOKENS_PER_TILE,
  ACE_OPT_0039_ATTENTION_SUBGROUP_SIZE,
  ACE_OPT_0039_ATTENTION_WORKGROUP_SIZE,
  ACE_OPT_0039_ATTENTION_WORKGROUP_STORAGE_BYTES,
  aceOpt0039AttentionWgsl,
  planAceOpt0039Attention,
} from "../src/webgpu/kernels/attention-dual-query.js";
import {
  buildOpt0039TimingOrders,
  parseOpt0039ThermalGate,
  summarizeOpt0039Timing,
} from "./browser/opt-0039-dit-attention-dual-query.js";

const SHAPE = Object.freeze({
  batch: 1,
  queryHeads: 16,
  keyValueHeads: 8,
  queryTokens: 2_250,
  keyValueTokens: 2_250,
  headDimension: 128,
  mode: "full" as const,
});

describe("OPT-0039 fixed-WG256 dual-query attention", () => {
  it("halves query tiles, workgroups, K/V loads, and barrier events without widening", () => {
    expect({
      workgroup: ACE_OPT_0039_ATTENTION_WORKGROUP_SIZE,
      subgroup: ACE_OPT_0039_ATTENTION_SUBGROUP_SIZE,
      headDimension: ACE_OPT_0039_ATTENTION_HEAD_DIMENSION,
      queries: ACE_OPT_0039_ATTENTION_QUERIES_PER_WORKGROUP,
      queryTokensPerTile: ACE_OPT_0039_ATTENTION_QUERY_TOKENS_PER_TILE,
      queryTokensPerSubgroup:
        ACE_OPT_0039_ATTENTION_QUERY_TOKENS_PER_SUBGROUP,
      workgroupStorageBytes:
        ACE_OPT_0039_ATTENTION_WORKGROUP_STORAGE_BYTES,
      barriersPerKey: ACE_OPT_0039_ATTENTION_BARRIERS_PER_KEY,
    }).toEqual({
      workgroup: 256,
      subgroup: 32,
      headDimension: 128,
      queries: 16,
      queryTokensPerTile: 8,
      queryTokensPerSubgroup: 2,
      workgroupStorageBytes: 1_024,
      barriersPerKey: 2,
    });
    const plan = planAceOpt0039Attention(SHAPE);
    expect(plan).toMatchObject({
      queryTokenTiles: 282,
      workgroupCount: 2_256,
      query8WorkgroupCount: 4_504,
      keyValueScalarLoads: 1_299_456_000,
      query8KeyValueScalarLoads: 2_594_304_000,
      barriersPerWorkgroup: 4_500,
      barrierEvents: 10_152_000,
      query8BarrierEvents: 20_268_000,
      outputElements: 4_608_000,
    });
    expect(plan.keyValueLoadReductionVersusQuery8).toBeCloseTo(
      1.99645390070922,
    );
    expect(plan.barrierEventReductionVersusQuery8).toBeCloseTo(
      1.99645390070922,
    );
    expect(Object.isFrozen(plan)).toBe(true);
    expect(Object.isFrozen(plan.shape)).toBe(true);
  });

  it("owns every head/token exactly once, including the two-token tail", () => {
    const ownership = new Uint8Array(SHAPE.queryHeads * SHAPE.queryTokens);
    let inTensorStreams = 0;
    let tailStreams = 0;
    for (let kvHead = 0; kvHead < SHAPE.keyValueHeads; kvHead += 1) {
      for (let tile = 0; tile < Math.ceil(SHAPE.queryTokens / 8); tile += 1) {
        for (let subgroup = 0; subgroup < 8; subgroup += 1) {
          const head = kvHead * 2 + Math.floor(subgroup / 4);
          const token0 = tile * 8 + subgroup % 4;
          const token1 = token0 + 4;
          for (const token of [token0, token1]) {
            if (token >= SHAPE.queryTokens) continue;
            const index = head * SHAPE.queryTokens + token;
            ownership[index] = ownership[index]! + 1;
            inTensorStreams += 1;
            if (tile === 281) tailStreams += 1;
          }
        }
      }
    }
    expect(inTensorStreams).toBe(SHAPE.queryHeads * SHAPE.queryTokens);
    expect([...ownership].every((count) => count === 1)).toBe(true);
    expect(tailStreams).toBe(32);
  });

  it("keeps two independent query8 reductions and ascending-key FP32 updates", () => {
    const current = aceFixed32TiledFullAttentionWgsl(SHAPE);
    const candidate = aceOpt0039AttentionWgsl(SHAPE);
    expect(candidate).toContain("@compute @workgroup_size(256, 1, 1)");
    expect(candidate).toContain("subgroup_size != 32u");
    expect(candidate).toContain(
      "let query_head = kv_head * HEADS_PER_KV + subgroup / 4u;",
    );
    expect(candidate).toContain(
      "let query_token_0 = query_tile * QUERY_TOKENS_PER_TILE + subgroup % 4u;",
    );
    expect(candidate).toContain("let query_token_1 = query_token_0 + 4u;");
    expect(candidate).toContain(
      "for (var key_token = 0u; key_token < valid_key_tokens; key_token += 1u)",
    );
    expect(candidate.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
    expect(candidate.match(/subgroupAdd\(dot_partial_[01]\)/g)).toHaveLength(2);
    expect(candidate.match(/key_tile\[local_lane\] = key\[kv_index\];/g))
      .toHaveLength(1);
    expect(candidate.match(/value_tile\[local_lane\] = value\[kv_index\];/g))
      .toHaveLength(1);
    for (const stream of [0, 1] as const) {
      expect(candidate).toContain(
        `dot_partial_${stream} + query_values_${stream}[chunk] * key_tile[dimension]`,
      );
      expect(candidate).toContain(
        `let score_${stream} = subgroupAdd(dot_partial_${stream}) * ATTENTION_SCALE;`,
      );
      expect(candidate).toContain(
        `let online_alpha_${stream} = exp(online_max_${stream} - new_max_${stream});`,
      );
      expect(candidate).toContain(
        `online_denominator_${stream} * online_alpha_${stream} + online_beta_${stream};`,
      );
      expect(candidate).toContain(
        `weighted_values_${stream}[chunk] * online_alpha_${stream} +`,
      );
    }
    for (const invariant of [
      "for (var key_token = 0u; key_token < valid_key_tokens; key_token += 1u)",
      "let online_alpha = exp(online_max - new_max);",
      "online_denominator * online_alpha + online_beta;",
      "weighted_values[chunk] = weighted_values[chunk] * online_alpha +",
    ]) expect(current).toContain(invariant);
    expect(candidate).not.toMatch(/array<f32, 2250>/);
    expect(candidate).not.toMatch(/\batomic/i);
    expect(candidate).not.toMatch(/\bdot\s*\(/);
  });

  it("rejects every shape outside the frozen M2250 full-attention screen", () => {
    expect(() => planAceOpt0039Attention({ ...SHAPE, batch: 2 }))
      .toThrow(/exact B1/);
    expect(() => planAceOpt0039Attention({ ...SHAPE, queryTokens: 2_249 }))
      .toThrow(/exact B1/);
    expect(() => planAceOpt0039Attention({
      ...SHAPE,
      mode: "sliding",
      slidingRadius: 128,
    })).toThrow(/exact B1/);
    expect(() => planAceOpt0039Attention({ ...SHAPE, keyValueHeads: 4 }))
      .toThrow();
  });

  it("pins four balanced A/B rounds, the thermal gate, and 1.25x threshold", () => {
    const orders = buildOpt0039TimingOrders();
    expect(orders).toEqual([
      ["query8", "dualQuery16"],
      ["dualQuery16", "query8"],
      ["query8", "dualQuery16"],
      ["dualQuery16", "query8"],
    ]);
    for (const arm of ["query8", "dualQuery16"] as const) {
      expect(orders.map((order) => order.indexOf(arm)).sort()).toEqual([
        0,
        0,
        1,
        1,
      ]);
    }
    expect(summarizeOpt0039Timing([
      { arm: "query8", samplesMilliseconds: [100, 100, 100, 100] },
      { arm: "dualQuery16", samplesMilliseconds: [80, 80, 80, 80] },
    ])).toMatchObject({
      speedupVersusQuery8: 1.25,
      requiredSpeedupVersusQuery8: 1.25,
      passed: true,
    });
    expect(summarizeOpt0039Timing([
      { arm: "query8", samplesMilliseconds: [99, 99, 99, 99] },
      { arm: "dualQuery16", samplesMilliseconds: [80, 80, 80, 80] },
    ])["passed"]).toBe(false);

    const preparedAt = 1_000_000;
    const parameters = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: String(preparedAt),
      thermalCheckedAtEpochMilliseconds: String(preparedAt + 30_000),
      thermalObservations: "1",
      thermalObservedLevel: "0",
      thermalMaximumObservationGapMilliseconds: "30000",
    });
    expect(parseOpt0039ThermalGate(
      parameters,
      preparedAt,
      preparedAt + 30_500,
    )).toMatchObject({
      durationMilliseconds: 30_000,
      observationCount: 1,
      observedLevel: 0,
      protocol: "wait-30s-then-one-level0-check",
    });
    parameters.set("thermalObservedLevel", "1");
    expect(() => parseOpt0039ThermalGate(
      parameters,
      preparedAt,
      preparedAt + 30_500,
    )).toThrow(/one truthful level-0 notifyutil check/);
  });

  it("keeps the browser gate full-output, deterministic, guarded, and button-timed", () => {
    expect(browserSource).toContain("__ACE_OPT0039_RESULT__");
    expect(browserSource).toContain("outputElements: query8A.words.length");
    expect(browserSource).toContain("totalComparedElements: query8A.words.length * 3");
    expect(browserSource).toContain("dualQuery16RepeatBitMismatchCount");
    expect(browserSource).toContain("query8RepeatBitMismatchCount");
    expect(browserSource).toContain("countBitMismatches");
    expect(browserSource).toContain("OUTPUT_PREFILL_QNAN_U32");
    expect(browserSource).toContain("prefixCanaryIntact");
    expect(browserSource).toContain("suffixCanaryIntact");
    expect(browserSource).toContain("requireCleanCompilation");
    expect(browserSource).toContain("timedGpuWorkBeforeButton: false");
    expect(browserSource).toContain("runButton.addEventListener(\"click\"");
    expect(browserSource).not.toContain("void runTiming(prepared);");
    expect(browserHtml).toContain("id=\"thermal-gate\"");
    expect(browserHtml).toContain("id=\"run\"");
    expect(browserHtml).toContain("4,608,000 F32 output");
    expect(browserHtml).toContain("No timed GPU work runs");
    expect(browserHtml).toContain("before the button");
    expect(browserHtml).toContain("1.25x query8");
  });
});
