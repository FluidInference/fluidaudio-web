import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import plannerSamplingSource from "../src/runtime/planner-sampling.ts?raw";
import {
  OPT_0084_CANDIDATE_SEAM_COMMIT,
  OPT_0084_DENSE_STATE_IDS,
  OPT_0084_PAIR_ORDERS,
  OPT_0084_STATE_IDS,
  OPT_0084_TIMING_ROUND_COUNT,
  evaluateOpt0084Timing,
  validateOpt0084RunIdentity,
  type Opt0084TimingSample,
} from "./browser/opt-0084-planner-fused-candidate-radix-sampling-contract.js";
import {
  parseOpt0084ThermalCompletion,
  parseOpt0084ThermalLaunch,
  parseOpt0084RunIdentity,
} from "./browser/opt-0084-planner-fused-candidate-radix-sampling.js";
import contractSource from
  "./browser/opt-0084-planner-fused-candidate-radix-sampling-contract.ts?raw";
import htmlSource from
  "./browser/opt-0084-planner-fused-candidate-radix-sampling.html?raw";
import pageSource from
  "./browser/opt-0084-planner-fused-candidate-radix-sampling.ts?raw";
import workerSource from
  "./browser/opt-0084-planner-fused-candidate-radix-sampling-worker.ts?raw";

describe("OPT-0084 actual-Chrome sampler harness contract", () => {
  it("pins the committed benchmark seam and keeps production unchanged", () => {
    expect(OPT_0084_CANDIDATE_SEAM_COMMIT).toBe(
      "245b5fe3347c370390eff990aa1ed45cb2b869ba",
    );
    expect(createHash("sha256").update(plannerSamplingSource).digest("hex")).toBe(
      "67055acfbb96e10682092e5d0ccfa9a5d822fd708a091fe46f7f47458226d0f3",
    );
    expect(workerSource).toContain('modelProfile: "reference-bf16"');
    expect(workerSource).toContain("AcePlannerGpuExecutor.create");
    expect(workerSource).toContain("createAcePlannerFilteredLogits");
    expect(workerSource).toContain("createAcePlannerBrowserSamplingWeights");
    expect(workerSource).toContain("fixture.workspace.sample(input)");
    expect(workerSource).toContain("sampleAcePlannerToken(input)");
    expect(workerSource).toContain("sampleOpt0084(cursorInput)");
    expect(workerSource).not.toContain("sampleCompactOpt0084");
    expect(workerSource).toContain("productionIntegrationAuthorized: false");
    expect(workerSource).toContain("const cleanupFailures: unknown[] = []");
    expect(workerSource).toMatch(
      /finally \{\s+try \{\s+context\?\.destroy\(\)/,
    );
    expect(workerSource).toContain("cleanup also ");
    expect(workerSource).toContain("{ cause: error }");
  });

  it("requires and independently validates a complete clone-safe run identity", () => {
    const parameters = new URLSearchParams({
      harnessCommit: "a".repeat(40),
      machineModel: "MacBook Air Mac15,12; Apple M3",
      osVersion: "macOS 26.5.2",
      osBuild: "25F84",
      browserVersion: "Google Chrome 151.0.7922.138",
      gpuCoreCount: "10",
      memoryBytes: "17179869184",
    });
    const identity = parseOpt0084RunIdentity(parameters);
    expect(identity).toEqual({
      harnessCommit: "a".repeat(40),
      machineModel: "MacBook Air Mac15,12; Apple M3",
      osVersion: "macOS 26.5.2",
      osBuild: "25F84",
      browserVersion: "Google Chrome 151.0.7922.138",
      gpuCoreCount: 10,
      memoryBytes: 17_179_869_184,
    });
    expect(validateOpt0084RunIdentity(structuredClone(identity))).toEqual(identity);
    expect(pageSource).toContain(
      'active.postMessage({ type: "prepare", identity: runIdentity })',
    );
    expect(workerSource).toContain(
      "validateOpt0084RunIdentity(message.identity)",
    );
    expect(workerSource).toContain("...prepared.runIdentity");

    parameters.delete("osBuild");
    expect(() => parseOpt0084RunIdentity(parameters)).toThrow(/osBuild is missing/);
    expect(() => validateOpt0084RunIdentity({
      ...identity,
      harnessCommit: "A".repeat(40),
    })).toThrow(/full lowercase commit/);
    expect(() => validateOpt0084RunIdentity({
      ...identity,
      memoryBytes: 1024,
    })).toThrow(/physical range/);
    expect(() => validateOpt0084RunIdentity({
      ...identity,
      unexpected: true,
    })).toThrow(/incomplete or unknown/);
  });

  it("covers semantic prefill plus three timed positions and all four CoT states", () => {
    expect(OPT_0084_STATE_IDS).toEqual([
      "semantic-early",
      "semantic-middle",
      "semantic-late",
      "cot-singleton",
      "cot-small",
      "cot-caption",
      "cot-all",
    ]);
    for (const stateId of OPT_0084_STATE_IDS) {
      expect(workerSource).toContain(`id: "${stateId}"`);
    }
    expect(workerSource).toContain('id: "semantic-prefill"');
    expect(workerSource).toContain(
      'executionKind: "actual-semantic-prefill-return-correctness-only"',
    );
    expect(workerSource).toContain("actualDecodeRowsRetainedBeforeTiming: true");
    expect(workerSource).toContain("actualPrefillRowsRetainedForCorrectness: true");
    expect(workerSource).toContain("initialRawLogitSha256");
    expect(workerSource).toContain("finalRawLogitSha256");
    expect(workerSource).toContain("raw-u32-over-complete-logical-vocabulary");
    expect(workerSource).toContain('forcedTerminalState: "cot-singleton"');
    expect(workerSource).toContain(
      'semanticProductionBf16Positions: Object.freeze([',
    );
    expect(OPT_0084_DENSE_STATE_IDS).toEqual([
      "semantic-early",
      "semantic-middle",
      "semantic-late",
      "cot-caption",
      "cot-all",
    ]);
  });

  it("freezes sixteen balanced pair orders and accepts a genuine exact win", () => {
    expect(OPT_0084_PAIR_ORDERS).toHaveLength(OPT_0084_TIMING_ROUND_COUNT);
    expect(OPT_0084_PAIR_ORDERS.filter((order) => order[0] === "A")).toHaveLength(8);
    expect(OPT_0084_PAIR_ORDERS.filter((order) => order[0] === "B")).toHaveLength(8);
    const samples = createTimingSamples(() => ({ a: 30, b: 10 }));
    const decision = evaluateOpt0084Timing(samples);
    expect(decision.passed).toBe(true);
    expect(decision.pairWinCount).toBe(16);
    expect(decision.aggregateSamplerSpeedup).toBe(3);
    expect(decision.projectedDefaultSemanticSavingSeconds).toBe(18);
    expect(decision.noRegressingDenseStateMedian).toBe(true);
    expect(decision.everyPairedSampleExact).toBe(true);
    expect(decision.armPositionCounts).toEqual({ A: [56, 56], B: [56, 56] });
  });

  it("stops for a dense-state regression, pair mismatch, or weak projection", () => {
    const denseRegression = evaluateOpt0084Timing(createTimingSamples(
      (stateId) => stateId === "cot-all"
        ? { a: 30, b: 31 }
        : { a: 30, b: 10 },
    ));
    expect(denseRegression.passed).toBe(false);
    expect(denseRegression.noRegressingDenseStateMedian).toBe(false);

    const mismatch = createTimingSamples(() => ({ a: 30, b: 10 }));
    mismatch[1] = Object.freeze({ ...mismatch[1]!, tokenId: 99 });
    const mismatchDecision = evaluateOpt0084Timing(mismatch);
    expect(mismatchDecision.passed).toBe(false);
    expect(mismatchDecision.everyPairedSampleExact).toBe(false);

    const weak = evaluateOpt0084Timing(createTimingSamples(
      (stateId) => stateId.startsWith("semantic")
        ? { a: 10, b: 9.995 }
        : { a: 30, b: 10 },
    ));
    expect(weak.passed).toBe(false);
    expect(weak.projectedDefaultSemanticSavingSeconds).toBeCloseTo(0.0045);
  });

  it("requires a fresh 30-second nominal gate and the same trace through cleanup", () => {
    const launchParameters = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalCommand: "notifyutil -g com.apple.system.thermalpressurelevel",
      thermalTraceStartedAtEpochMilliseconds: "90000",
      thermalGateStartedAtEpochMilliseconds: "100000",
      thermalGateCompletedAtEpochMilliseconds: "130000",
      thermalGateObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalGateMaximumPollGapMilliseconds: "1000",
      thermalGateNonNominalObservations: "0",
      thermalGateMissingObservations: "0",
    });
    const launch = parseOpt0084ThermalLaunch(
      launchParameters,
      100_000,
      130_500,
    );
    expect(launch.readyToGateDelayMilliseconds).toBe(0);
    expect(launch.launchDelayMilliseconds).toBe(500);

    const completion = parseOpt0084ThermalCompletion(new URLSearchParams({
      thermalTraceSchema:
        "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
      thermalTraceSha256: "a".repeat(64),
      thermalTraceByteLength: "4096",
      thermalTraceCompletedAtEpochMilliseconds: "140000",
      thermalTraceObservations: "51",
      thermalTraceMaximumPollGapMilliseconds: "1000",
      thermalTraceNonNominalObservations: "0",
      thermalTraceMissingObservations: "0",
      thermalTraceInitialLevel: "0",
      thermalTraceFinalLevel: "0",
      thermalTraceTransitionsJson: "[]",
    }), launch, 139_000);
    expect(completion["coversCleanup"]).toBe(true);

    const impossibleSubsetGap = new URLSearchParams({
      thermalTraceSchema:
        "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
      thermalTraceSha256: "a".repeat(64),
      thermalTraceByteLength: "4096",
      thermalTraceCompletedAtEpochMilliseconds: "140000",
      thermalTraceObservations: "51",
      thermalTraceMaximumPollGapMilliseconds: "999",
      thermalTraceNonNominalObservations: "0",
      thermalTraceMissingObservations: "0",
      thermalTraceInitialLevel: "0",
      thermalTraceFinalLevel: "0",
      thermalTraceTransitionsJson: "[]",
    });
    expect(() => parseOpt0084ThermalCompletion(
      impossibleSubsetGap,
      launch,
      139_000,
    )).toThrow(/through-cleanup thermal trace failed/);

    const shortFullCountLaunch = Object.freeze({
      ...launch,
      observationCount: 52,
    });
    impossibleSubsetGap.set("thermalTraceMaximumPollGapMilliseconds", "1000");
    expect(() => parseOpt0084ThermalCompletion(
      impossibleSubsetGap,
      shortFullCountLaunch,
      139_000,
    )).toThrow(/through-cleanup thermal trace failed/);

    launchParameters.set("thermalGateCompletedAtEpochMilliseconds", "129999");
    expect(() => parseOpt0084ThermalLaunch(
      launchParameters,
      100_000,
      130_500,
    )).toThrow(/thermal launch gate failed/);
  });

  it("keeps the clone-safe download and manual thermal protocol visible", () => {
    expect(htmlSource).toContain('id="thermal-gate"');
    expect(htmlSource).toContain('id="thermal-completion"');
    expect(htmlSource).toContain('id="download"');
    expect(pageSource).toContain("window.__ACE_OPT0084_RESULT__ = receipt");
    expect(pageSource).toContain("JSON.stringify(receipt, null, 2)");
    expect(pageSource).toContain("URL.createObjectURL");
    expect(pageSource).toContain("through-cleanup");
    expect(contractSource).toContain("OPT_0084_MINIMUM_PAIR_WINS = 14");
    expect(contractSource).toContain("OPT_0084_MINIMUM_SAMPLER_SPEEDUP = 1.5");
    expect(contractSource).toContain(
      "OPT_0084_MINIMUM_PROJECTED_SAVING_SECONDS = 10",
    );
  });
});

function createTimingSamples(
  times: (stateId: (typeof OPT_0084_STATE_IDS)[number]) => Readonly<{
    readonly a: number;
    readonly b: number;
  }>,
): Opt0084TimingSample[] {
  const samples: Opt0084TimingSample[] = [];
  for (let roundIndex = 0; roundIndex < OPT_0084_TIMING_ROUND_COUNT;
    roundIndex += 1) {
    const order = OPT_0084_PAIR_ORDERS[roundIndex]!;
    for (let armPosition = 0; armPosition < 2; armPosition += 1) {
      const arm = order[armPosition]!;
      for (let statePosition = 0; statePosition < OPT_0084_STATE_IDS.length;
        statePosition += 1) {
        const stateId = OPT_0084_STATE_IDS[statePosition]!;
        const wall = times(stateId);
        samples.push(Object.freeze({
          roundIndex,
          stateId,
          arm,
          armPosition: armPosition as 0 | 1,
          statePosition,
          wallMilliseconds: arm === "A" ? wall.a : wall.b,
          tokenId: 42 + statePosition,
          word: 0x1234_5678 + roundIndex,
          positiveCandidateCount: 1 + statePosition,
        }));
      }
    }
  }
  return samples;
}
