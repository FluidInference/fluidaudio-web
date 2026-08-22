import { describe, expect, it } from "vitest";

import browserSource from
  "./browser/opt-0033-dit-attention-key-blocks.ts?raw";
import browserHtml from
  "./browser/opt-0033-dit-attention-key-blocks.html?raw";
import { aceFixed32TiledFullAttentionWgsl } from
  "../src/webgpu/kernels/attention.js";
import {
  ACE_OPT_0033_ATTENTION_BARRIERS_PER_KEY_BLOCK,
  ACE_OPT_0033_ATTENTION_HEAD_DIMENSION,
  ACE_OPT_0033_ATTENTION_QUERIES_PER_WORKGROUP,
  ACE_OPT_0033_ATTENTION_QUERY_TOKENS_PER_TILE,
  ACE_OPT_0033_ATTENTION_SUBGROUP_SIZE,
  ACE_OPT_0033_ATTENTION_WORKGROUP_SIZE,
  aceOpt0033AttentionWgsl,
  planAceOpt0033Attention,
} from "../src/webgpu/kernels/attention-key-blocks.js";
import {
  buildOpt0033TimingOrders,
  parseOpt0033ThermalGate,
  summarizeOpt0033Timing,
} from "./browser/opt-0033-dit-attention-key-blocks.js";

const SHAPE = Object.freeze({
  batch: 1,
  queryHeads: 16,
  keyValueHeads: 8,
  queryTokens: 2_250,
  keyValueTokens: 2_250,
  headDimension: 128,
  mode: "full" as const,
});

describe("OPT-0033 query8 K/V key blocking", () => {
  it("pins query8 ownership, unchanged K/V loads, and barrier reductions", () => {
    expect({
      workgroup: ACE_OPT_0033_ATTENTION_WORKGROUP_SIZE,
      subgroup: ACE_OPT_0033_ATTENTION_SUBGROUP_SIZE,
      queries: ACE_OPT_0033_ATTENTION_QUERIES_PER_WORKGROUP,
      queryTokens: ACE_OPT_0033_ATTENTION_QUERY_TOKENS_PER_TILE,
      dimension: ACE_OPT_0033_ATTENTION_HEAD_DIMENSION,
      barriersPerBlock: ACE_OPT_0033_ATTENTION_BARRIERS_PER_KEY_BLOCK,
    }).toEqual({
      workgroup: 256,
      subgroup: 32,
      queries: 8,
      queryTokens: 4,
      dimension: 128,
      barriersPerBlock: 2,
    });
    const block8 = planAceOpt0033Attention(SHAPE, 8);
    const block16 = planAceOpt0033Attention(SHAPE, 16);
    expect(block8).toMatchObject({
      keyBlock: 8,
      keyBlocksPerWorkgroup: 282,
      barriersPerWorkgroup: 564,
      rowStagedBarriersPerWorkgroup: 4_500,
      workgroupCount: 4_504,
      barrierEvents: 2_540_256,
      keyValueScalarLoads: 2_594_304_000,
      outputElements: 4_608_000,
      keyPanelElements: 1_024,
      valuePanelElements: 1_024,
      workgroupStorageBytes: 8_192,
    });
    expect(block16).toMatchObject({
      keyBlock: 16,
      keyBlocksPerWorkgroup: 141,
      barriersPerWorkgroup: 282,
      rowStagedBarriersPerWorkgroup: 4_500,
      workgroupCount: 4_504,
      barrierEvents: 1_270_128,
      keyValueScalarLoads: 2_594_304_000,
      outputElements: 4_608_000,
      keyPanelElements: 2_048,
      valuePanelElements: 2_048,
      workgroupStorageBytes: 16_384,
    });
    expect(block8.barrierReductionVersusRowStaged).toBeCloseTo(
      7.9787234042553195,
    );
    expect(block16.barrierReductionVersusRowStaged).toBeCloseTo(
      15.957446808510639,
    );
    expect(block8.keyValueScalarLoads).toBe(block16.keyValueScalarLoads);
    expect(Object.isFrozen(block8)).toBe(true);
    expect(Object.isFrozen(block8.shape)).toBe(true);
  });

  it("cooperatively covers each whole panel once and guards only the tail", () => {
    for (const keyBlock of [8, 16] as const) {
      const panelElements = keyBlock * 128;
      const owners = new Uint8Array(panelElements);
      for (let lane = 0; lane < 256; lane += 1) {
        for (let index = lane; index < panelElements; index += 256) {
          owners[index] = owners[index]! + 1;
        }
      }
      expect([...owners].every((count) => count === 1)).toBe(true);
      expect(panelElements / 256).toBe(keyBlock / 2);
      const lastBlockStart = Math.floor((2_250 - 1) / keyBlock) * keyBlock;
      expect(2_250 - lastBlockStart).toBe(keyBlock === 8 ? 2 : 10);
      const visited: number[] = [];
      for (let keyStart = 0; keyStart < 2_250; keyStart += keyBlock) {
        const valid = Math.min(keyBlock, 2_250 - keyStart);
        for (let key = 0; key < valid; key += 1) visited.push(keyStart + key);
      }
      expect(visited).toHaveLength(2_250);
      expect(visited.every((key, index) => key === index)).toBe(true);
    }
  });

  it("keeps current query ownership and exact ascending-key softmax math", () => {
    const current = aceFixed32TiledFullAttentionWgsl(SHAPE);
    for (const keyBlock of [8, 16] as const) {
      const source = aceOpt0033AttentionWgsl(SHAPE, keyBlock);
      expect(source).toContain("@compute @workgroup_size(256, 1, 1)");
      expect(source).toContain("subgroup_size != 32u");
      expect(source).toContain(
        "kv_head * HEADS_PER_KV + subgroup / QUERY_TOKENS_PER_TILE",
      );
      expect(source).toContain(
        "query_tile * QUERY_TOKENS_PER_TILE + subgroup % QUERY_TOKENS_PER_TILE",
      );
      expect(source).toContain(`const KEY_BLOCK: u32 = ${keyBlock}u;`);
      expect(source).toContain(
        `var<workgroup> key_tile: array<f32, ${keyBlock * 128}>;`,
      );
      expect(source).toContain(
        `var<workgroup> value_tile: array<f32, ${keyBlock * 128}>;`,
      );
      expect(source).toContain("key_block_start += KEY_BLOCK");
      expect(source).toContain("key_in_block += 1u");
      expect(source).toContain(
        "let valid_keys_in_block = min(KEY_BLOCK, valid_key_tokens - key_block_start);",
      );
      expect(source.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
      expect(source).toContain(
        "let score = subgroupAdd(dot_partial) * ATTENTION_SCALE;",
      );
      expect(source).toContain(
        "let online_alpha = exp(online_max - new_max);",
      );
      expect(source).toContain(
        "online_denominator * online_alpha + online_beta;",
      );
      expect(source).toContain(
        "weighted_values[chunk] = weighted_values[chunk] * online_alpha +",
      );
      expect(source).not.toMatch(/\batomic/i);
      expect(source).not.toMatch(/\bdot\s*\(/);
    }
    for (const invariant of [
      "let score = subgroupAdd(dot_partial) * ATTENTION_SCALE;",
      "let new_max = max(online_max, score);",
      "let online_alpha = exp(online_max - new_max);",
      "let online_beta = exp(score - new_max);",
      "online_denominator * online_alpha + online_beta;",
    ]) {
      expect(current).toContain(invariant);
    }
  });

  it("rejects all shape and mechanism drift", () => {
    expect(() => planAceOpt0033Attention({ ...SHAPE, batch: 2 }, 8))
      .toThrow(/exact B1/);
    expect(() => planAceOpt0033Attention({ ...SHAPE, queryTokens: 2_249 }, 8))
      .toThrow();
    expect(() => planAceOpt0033Attention({
      ...SHAPE,
      mode: "sliding",
      slidingRadius: 128,
    }, 16)).toThrow(/exact B1/);
    expect(() => planAceOpt0033Attention(SHAPE, 4 as 8)).toThrow(/key-block/);
  });

  it("pins balanced A/B/C timing, thermal validity, and the 1.35x gate", () => {
    const orders = buildOpt0033TimingOrders();
    expect(orders).toEqual([
      ["query8", "keyBlock8", "keyBlock16"],
      ["keyBlock8", "keyBlock16", "query8"],
      ["keyBlock16", "query8", "keyBlock8"],
    ]);
    for (const arm of ["query8", "keyBlock8", "keyBlock16"] as const) {
      expect(orders.map((order) => order.indexOf(arm)).sort()).toEqual([0, 1, 2]);
    }
    const passing = summarizeOpt0033Timing([
      { arm: "query8", samplesMilliseconds: [14, 14, 14] },
      { arm: "keyBlock8", samplesMilliseconds: [10, 10, 10] },
      { arm: "keyBlock16", samplesMilliseconds: [11, 11, 11] },
    ]);
    expect(passing).toMatchObject({
      bestCandidate: "keyBlock8",
      speedupVersusQuery8: 1.4,
      requiredSpeedupVersusQuery8: 1.35,
      passed: true,
    });
    expect(summarizeOpt0033Timing([
      { arm: "query8", samplesMilliseconds: [13, 13, 13] },
      { arm: "keyBlock8", samplesMilliseconds: [10, 10, 10] },
      { arm: "keyBlock16", samplesMilliseconds: [11, 11, 11] },
    ])["passed"]).toBe(false);

    const preparedAt = 1_000_000;
    const parameters = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: String(preparedAt),
      thermalCompletedAtEpochMilliseconds: String(preparedAt + 30_000),
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1000",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0033ThermalGate(
      parameters,
      preparedAt,
      preparedAt + 30_500,
    )).toMatchObject({ durationMilliseconds: 30_000, observationCount: 31 });
    parameters.set("thermalMaximumPollGapMilliseconds", "1251");
    expect(() => parseOpt0033ThermalGate(
      parameters,
      preparedAt,
      preparedAt + 30_500,
    )).toThrow(/thermal gate/);
  });

  it("keeps the browser gate full-output, lifecycle-clean, and button-timed", () => {
    expect(browserSource).toContain("__ACE_OPT0033_RESULT__");
    expect(browserSource).toContain("outputElements: query8.words.length");
    expect(browserSource).toContain("countBitMismatches");
    expect(browserSource).toContain("OUTPUT_PREFILL_QNAN_U32");
    expect(browserSource).toContain("prefixCanaryIntact");
    expect(browserSource).toContain("requireCleanCompilation");
    expect(browserSource).toContain("timedGpuWorkBeforeButton: false");
    expect(browserSource).toContain("runButton.addEventListener(\"click\"");
    expect(browserSource).not.toContain("void runTiming(prepared);");
    expect(browserHtml).toContain("id=\"thermal-gate\"");
    expect(browserHtml).toContain("id=\"run\"");
    expect(browserHtml).toContain("4,608,000 F32 output words");
    expect(browserHtml).toContain("No timed GPU work runs before the button");
  });
});
