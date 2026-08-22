import { describe, expect, it } from "vitest";

import browserSource from
  "./browser/opt-0061-dit-attention-multi-query.ts?raw";
import browserHtml from
  "./browser/opt-0061-dit-attention-multi-query.html?raw";
import {
  ACE_OPT_0061_ATTENTION_BARRIERS_PER_KEY,
  ACE_OPT_0061_ATTENTION_HEAD_DIMENSION,
  ACE_OPT_0061_ATTENTION_SUBGROUP_SIZE,
  ACE_OPT_0061_ATTENTION_WORKGROUP_SIZE,
  ACE_OPT_0061_ATTENTION_WORKGROUP_STORAGE_BYTES,
  aceOpt0061AttentionWgsl,
  planAceOpt0061Attention,
  type AceOpt0061StreamCount,
} from "../src/webgpu/kernels/attention-multi-query.js";
import {
  buildOpt0061TimingOrders,
  parseOpt0061ThermalGate,
  summarizeOpt0061Timing,
  type Opt0061Arm,
} from "./browser/opt-0061-dit-attention-multi-query.js";

const SHAPE = Object.freeze({
  batch: 1,
  queryHeads: 16,
  keyValueHeads: 8,
  queryTokens: 2_250,
  keyValueTokens: 2_250,
  headDimension: 128,
  mode: "full" as const,
});

const EXPECTED = Object.freeze({
  3: Object.freeze({
    queriesPerWorkgroup: 24,
    queryTokensPerTile: 12,
    queryTokenTiles: 188,
    workgroupCount: 1_504,
    keyValueScalarLoads: 866_304_000,
    barrierEvents: 6_768_000,
    retainedPrivateFp32ValuesPerLane: 30,
    tailTokens: 96,
  }),
  4: Object.freeze({
    queriesPerWorkgroup: 32,
    queryTokensPerTile: 16,
    queryTokenTiles: 141,
    workgroupCount: 1_128,
    keyValueScalarLoads: 649_728_000,
    barrierEvents: 5_076_000,
    retainedPrivateFp32ValuesPerLane: 40,
    tailTokens: 160,
  }),
});

describe("OPT-0061 fixed-WG256 multi-query attention", () => {
  it("pins the unchanged workgroup geometry and exact three/four-stream costs", () => {
    expect({
      workgroup: ACE_OPT_0061_ATTENTION_WORKGROUP_SIZE,
      subgroup: ACE_OPT_0061_ATTENTION_SUBGROUP_SIZE,
      headDimension: ACE_OPT_0061_ATTENTION_HEAD_DIMENSION,
      workgroupStorageBytes: ACE_OPT_0061_ATTENTION_WORKGROUP_STORAGE_BYTES,
      barriersPerKey: ACE_OPT_0061_ATTENTION_BARRIERS_PER_KEY,
    }).toEqual({
      workgroup: 256,
      subgroup: 32,
      headDimension: 128,
      workgroupStorageBytes: 1_024,
      barriersPerKey: 2,
    });

    for (const streamCount of [3, 4] as const) {
      const plan = planAceOpt0061Attention(SHAPE, streamCount);
      const expected = EXPECTED[streamCount];
      expect(plan).toMatchObject({
        streamCount,
        queryTokensPerSubgroup: streamCount,
        queriesPerWorkgroup: expected.queriesPerWorkgroup,
        queryTokensPerTile: expected.queryTokensPerTile,
        queryTokenTiles: expected.queryTokenTiles,
        workgroupCount: expected.workgroupCount,
        query8WorkgroupCount: 4_504,
        dualQueryWorkgroupCount: 2_256,
        keyValueScalarLoads: expected.keyValueScalarLoads,
        query8KeyValueScalarLoads: 2_594_304_000,
        dualQueryKeyValueScalarLoads: 1_299_456_000,
        barriersPerWorkgroup: 4_500,
        barrierEvents: expected.barrierEvents,
        query8BarrierEvents: 20_268_000,
        dualQueryBarrierEvents: 10_152_000,
        retainedPrivateFp32ValuesPerLane:
          expected.retainedPrivateFp32ValuesPerLane,
        outputElements: 4_608_000,
      });
      expect(plan.keyValueLoadReductionVersusQuery8).toBeCloseTo(
        plan.query8KeyValueScalarLoads / plan.keyValueScalarLoads,
      );
      expect(plan.keyValueLoadReductionVersusDualQuery).toBeCloseTo(
        plan.dualQueryKeyValueScalarLoads / plan.keyValueScalarLoads,
      );
      expect(plan.barrierEventReductionVersusQuery8).toBeCloseTo(
        plan.query8BarrierEvents / plan.barrierEvents,
      );
      expect(plan.barrierEventReductionVersusDualQuery).toBeCloseTo(
        plan.dualQueryBarrierEvents / plan.barrierEvents,
      );
      expect(Object.isFrozen(plan)).toBe(true);
      expect(Object.isFrozen(plan.shape)).toBe(true);
    }
  });

  it.each([3, 4] as const)(
    "owns every head/token once with %i streams, including the tail",
    (streamCount) => {
      const plan = planAceOpt0061Attention(SHAPE, streamCount);
      const ownership = new Uint8Array(SHAPE.queryHeads * SHAPE.queryTokens);
      let inTensorStreams = 0;
      let tailTokens = 0;
      for (let kvHead = 0; kvHead < SHAPE.keyValueHeads; kvHead += 1) {
        for (let tile = 0; tile < plan.queryTokenTiles; tile += 1) {
          for (let subgroup = 0; subgroup < 8; subgroup += 1) {
            const head = kvHead * 2 + Math.floor(subgroup / 4);
            const tokenBase =
              tile * plan.queryTokensPerTile + subgroup % 4;
            for (let stream = 0; stream < streamCount; stream += 1) {
              const token = tokenBase + stream * 4;
              if (token >= SHAPE.queryTokens) continue;
              const index = head * SHAPE.queryTokens + token;
              ownership[index] = ownership[index]! + 1;
              inTensorStreams += 1;
              if (tile === plan.queryTokenTiles - 1) tailTokens += 1;
            }
          }
        }
      }
      expect(inTensorStreams).toBe(SHAPE.queryHeads * SHAPE.queryTokens);
      expect([...ownership].every((count) => count === 1)).toBe(true);
      expect(tailTokens).toBe(EXPECTED[streamCount].tailTokens);
    },
  );

  it.each([3, 4] as const)(
    "keeps %i independent ascending-key query8 reductions",
    (streamCount) => {
      const candidate = aceOpt0061AttentionWgsl(SHAPE, streamCount);
      expect(candidate).toContain("@compute @workgroup_size(256, 1, 1)");
      expect(candidate).toContain("subgroup_size != 32u");
      expect(candidate).toContain(
        "let query_head = kv_head * HEADS_PER_KV + subgroup / 4u;",
      );
      expect(candidate).toContain(
        "query_tile * QUERY_TOKENS_PER_TILE + subgroup % 4u;",
      );
      expect(candidate).toContain(
        "for (var key_token = 0u; key_token < valid_key_tokens; key_token += 1u)",
      );
      expect(candidate.match(/workgroupBarrier\(\);/g)).toHaveLength(2);
      expect(candidate.match(/subgroupAdd\(dot_partial_\d\)/g)).toHaveLength(
        streamCount,
      );
      expect(candidate.match(/key_tile\[local_lane\] = key\[kv_index\];/g))
        .toHaveLength(1);
      expect(candidate.match(/value_tile\[local_lane\] = value\[kv_index\];/g))
        .toHaveLength(1);
      for (let stream = 0; stream < streamCount; stream += 1) {
        expect(candidate).toContain(
          `let query_token_${stream} = query_token_base + ${stream * 4}u;`,
        );
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
      expect(candidate).not.toMatch(/array<f32, 2250>/);
      expect(candidate).not.toMatch(/\batomic/i);
      expect(candidate).not.toMatch(/\bdot\s*\(/);
    },
  );

  it("rejects any arm or shape outside the frozen exact production screen", () => {
    expect(() => planAceOpt0061Attention(
      SHAPE,
      2 as AceOpt0061StreamCount,
    )).toThrow(/exactly three or four streams/);
    expect(() => planAceOpt0061Attention({ ...SHAPE, batch: 2 }, 3))
      .toThrow(/exact B1/);
    expect(() => planAceOpt0061Attention({
      ...SHAPE,
      mode: "sliding",
      slidingRadius: 128,
    }, 4)).toThrow(/exact B1/);
    expect(() => planAceOpt0061Attention({
      ...SHAPE,
      keyValidity: "causal-per-key",
    }, 3)).toThrow(/exact B1/);
  });

  it("pins eight forward/reverse Latin rounds and both winner thresholds", () => {
    const orders = buildOpt0061TimingOrders();
    expect(orders).toHaveLength(8);
    const arms = [
      "query8",
      "dualQuery16",
      "tripleQuery24",
      "quadQuery32",
    ] as const;
    for (const arm of arms) {
      expect(orders.map((order) => order.indexOf(arm)).sort()).toEqual([
        0, 0, 1, 1, 2, 2, 3, 3,
      ]);
    }
    const timing = (values: Readonly<Record<Opt0061Arm, number>>) =>
      summarizeOpt0061Timing(arms.map((arm) => Object.freeze({
        arm,
        samplesMilliseconds: Object.freeze(Array(8).fill(values[arm])),
      })));
    expect(timing({
      query8: 150,
      dualQuery16: 100,
      tripleQuery24: 88,
      quadQuery32: 92,
    })).toMatchObject({
      samplesMilliseconds: {
        query8: Array(8).fill(150),
        dualQuery16: Array(8).fill(100),
        tripleQuery24: Array(8).fill(88),
        quadQuery32: Array(8).fill(92),
      },
      winner: "tripleQuery24",
      passed: true,
      requiredSpeedupVersusDual: 1.12,
      requiredSpeedupVersusQuery8: 1.5,
    });
    expect(timing({
      query8: 150,
      dualQuery16: 100,
      tripleQuery24: 90,
      quadQuery32: 95,
    })).toMatchObject({
      winner: null,
      passed: false,
      decision: "negative-stop-wider-stream-counts",
    });
  });

  it("requires one level-0 observation after a fresh 30-second wait", () => {
    const preparedAt = 1_000_000;
    const parameters = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: String(preparedAt),
      thermalCheckedAtEpochMilliseconds: String(preparedAt + 30_000),
      thermalObservations: "1",
      thermalObservedLevel: "0",
      thermalMaximumObservationGapMilliseconds: "30000",
    });
    expect(parseOpt0061ThermalGate(
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
    expect(() => parseOpt0061ThermalGate(
      parameters,
      preparedAt,
      preparedAt + 30_500,
    )).toThrow(/one truthful level-0 notifyutil check/);
  });

  it("keeps the browser screen full-output, guarded, and button-timed", () => {
    expect(browserSource).toContain("__ACE_OPT0061_RESULT__");
    expect(browserSource).toContain("totalComparedElements: query8A.words.length * 7");
    expect(browserSource).toContain("completeWritesAcrossEightRuns: true");
    expect(browserSource).toContain("canariesUntouchedAcrossEightRuns: true");
    expect(browserSource).toContain("OUTPUT_PREFILL_QNAN_U32");
    expect(browserSource).toContain("countBitMismatches");
    expect(browserSource).toContain("timedGpuWorkBeforeButton: false");
    expect(browserSource).toContain("balancedForwardReverseLatinOrders");
    expect(browserSource).toContain('runButton.addEventListener("click"');
    expect(browserSource).not.toContain("void runTiming(prepared);");
    expect(browserHtml).toContain('id="thermal-gate"');
    expect(browserHtml).toContain('id="run"');
    expect(browserHtml).toContain("4,608,000 F32 words");
    expect(browserHtml).toContain("No timed GPU work runs before the button");
    expect(browserHtml).toContain("1.12x dual-query");
    expect(browserHtml).toContain("1.50x query8");
  });
});
