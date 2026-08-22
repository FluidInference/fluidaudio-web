import { describe, expect, it } from "vitest";

import {
  ACE_BROWSER_SOFTMAX_V1,
  AcePlannerSamplingCursor,
} from "../src/runtime/planner-sampling.js";
import { canonicalizeSeed } from "../src/runtime/seed.js";

import {
  OPT_0084_COMPOUND_ARM_ORDERS,
  OPT_0084_COMPOUND_MINIMUM_PAIR_WINS,
  OPT_0084_COMPOUND_MINIMUM_PROJECTED_SAVING_SECONDS,
  OPT_0084_COMPOUND_MINIMUM_SPEEDUP,
  OPT_0084_COMPOUND_POSITION_IDS,
  OPT_0084_COMPOUND_TIMING_ROUND_COUNT,
  createOpt0084CompoundAbortingLogitRow,
  evaluateOpt0084CompoundTiming,
  type Opt0084CompoundTimingSample,
} from "./browser/opt-0084-planner-compact-head-fused-sampling-contract.js";
import {
  parseOpt0084CompoundRunIdentity,
  parseOpt0084CompoundThermalCompletion,
  parseOpt0084CompoundThermalLaunch,
} from "./browser/opt-0084-planner-compact-head-fused-sampling.js";
import contractSource from
  "./browser/opt-0084-planner-compact-head-fused-sampling-contract.ts?raw";
import htmlSource from
  "./browser/opt-0084-planner-compact-head-fused-sampling.html?raw";
import pageSource from
  "./browser/opt-0084-planner-compact-head-fused-sampling.ts?raw";
import workerSource from
  "./browser/opt-0084-planner-compact-head-fused-sampling-worker.ts?raw";

describe("OPT-0084 package-native compound browser harness", () => {
  it("freezes the registered three-arm gate and accepts only a material win", () => {
    expect(OPT_0084_COMPOUND_TIMING_ROUND_COUNT).toBe(16);
    expect(OPT_0084_COMPOUND_MINIMUM_PAIR_WINS).toBe(14);
    expect(OPT_0084_COMPOUND_MINIMUM_SPEEDUP).toBe(1.15);
    expect(OPT_0084_COMPOUND_MINIMUM_PROJECTED_SAVING_SECONDS).toBe(40);
    expect(OPT_0084_COMPOUND_POSITION_IDS).toEqual([
      "semantic-early", "semantic-middle", "semantic-late",
    ]);
    expect(OPT_0084_COMPOUND_ARM_ORDERS).toHaveLength(16);
    expect(OPT_0084_COMPOUND_ARM_ORDERS.filter((order) =>
      order.indexOf("A") < order.indexOf("C"))).toHaveLength(8);
    expect(OPT_0084_COMPOUND_ARM_ORDERS.filter((order) =>
      order.indexOf("C") < order.indexOf("A"))).toHaveLength(8);
    expect(OPT_0084_COMPOUND_ARM_ORDERS.every((order) =>
      order[1] === "B")).toBe(true);

    const decision = evaluateOpt0084CompoundTiming(createSamples({
      A: 400, B: 350, C: 300,
    }));
    expect(decision.passed).toBe(true);
    expect(decision.pairWinCount).toBe(16);
    expect(decision.aggregateCompleteTokenSpeedup).toBeCloseTo(4 / 3);
    expect(decision.projected900TokenSavingSeconds).toBe(90);
    expect(decision.everyPositionCMedianBelowA).toBe(true);
    expect(decision.armPositionCounts).toEqual({
      A: [24, 0, 24], B: [0, 48, 0], C: [24, 0, 24],
    });

    const weak = evaluateOpt0084CompoundTiming(createSamples({
      A: 400, B: 350, C: 370,
    }));
    expect(weak.passed).toBe(false);
    expect(weak.aggregateCompleteTokenSpeedup).toBeLessThan(1.15);
    expect(weak.projected900TokenSavingSeconds).toBe(27);

    // Median(sum(position walls)) is deliberately nonlinear here. The frozen
    // projection is mean(position median A-C saving), matching the standalone
    // OPT-0084 method, so it must report 120 s rather than 151.5 s.
    const nonlinear = createSamples({ A: 400, B: 350, C: 300 });
    const nonlinearC = [
      [100, 300, 390, 200, 100, 100, 300, 390, 100, 390, 390, 300, 100, 200, 300, 390],
      [300, 390, 300, 390, 390, 200, 200, 100, 390, 200, 300, 300, 390, 100, 200, 300],
      [200, 390, 200, 200, 100, 100, 100, 100, 300, 200, 200, 200, 100, 100, 100, 200],
    ] as const;
    for (let index = 0; index < nonlinear.length; index += 1) {
      const sample = nonlinear[index]!;
      if (sample.arm !== "C") continue;
      const value = nonlinearC[sample.positionOrder]![sample.roundIndex]!;
      nonlinear[index] = Object.freeze({
        ...sample,
        completeTokenWallMilliseconds: value,
        modelWallMilliseconds: value - 10,
      });
    }
    const nonlinearDecision = evaluateOpt0084CompoundTiming(nonlinear);
    expect(nonlinearDecision.meanPositionMedianSavingMilliseconds).toBeCloseTo(
      400 / 3,
    );
    expect(nonlinearDecision.projected900TokenSavingSeconds).toBeCloseTo(120);
    expect((1_200 - 695) / 3 * 900 / 1_000).toBe(151.5);

    // This distribution is the actual false-pass adversary: median(round
    // sums) would project 59.25 s and pass every other gate, while the frozen
    // mean of per-position median savings is only 36 s and must stop.
    const projectionFalsePass = createSamples({ A: 400, B: 350, C: 300 });
    const falsePassC = [
      [380, 300, 300, 300, 300, 380, 350, 250, 395, 350, 380, 380, 350, 350, 250, 350],
      [250, 380, 350, 300, 350, 380, 250, 300, 300, 250, 380, 395, 350, 395, 390, 300],
      [390, 380, 250, 380, 300, 250, 300, 350, 300, 395, 380, 395, 380, 390, 350, 380],
    ] as const;
    for (let index = 0; index < projectionFalsePass.length; index += 1) {
      const sample = projectionFalsePass[index]!;
      if (sample.arm !== "C") continue;
      const value = falsePassC[sample.positionOrder]![sample.roundIndex]!;
      projectionFalsePass[index] = Object.freeze({
        ...sample,
        completeTokenWallMilliseconds: value,
        modelWallMilliseconds: value - 10,
      });
    }
    const falsePassDecision = evaluateOpt0084CompoundTiming(projectionFalsePass);
    expect(falsePassDecision.pairWinCount).toBe(16);
    expect(falsePassDecision.aggregateCompleteTokenSpeedup).toBeGreaterThan(1.15);
    expect(falsePassDecision.everyPositionCMedianBelowA).toBe(true);
    expect(falsePassDecision.meanPositionMedianSavingMilliseconds).toBe(40);
    expect(falsePassDecision.projected900TokenSavingSeconds).toBe(36);
    expect((1_200 - 1_002.5) / 3 * 900 / 1_000).toBeCloseTo(59.25);
    expect(falsePassDecision.passed).toBe(false);

    const positionRegression = createSamples({ A: 400, B: 350, C: 300 });
    for (let index = 0; index < positionRegression.length; index += 1) {
      const sample = positionRegression[index]!;
      if (sample.positionId === "semantic-late" && sample.arm === "C") {
        positionRegression[index] = Object.freeze({
          ...sample,
          completeTokenWallMilliseconds: 410,
          modelWallMilliseconds: 400,
        });
      }
    }
    expect(evaluateOpt0084CompoundTiming(positionRegression).passed).toBe(false);
  });

  it("uses one authenticated owner and actual recurring sequential tokens", () => {
    expect(workerSource).toContain("AcePlannerGpuExecutor.create");
    expect(workerSource).toContain('modelProfile: "reference-bf16"');
    expect(workerSource).toContain("oneAuthenticatedPlannerOwner: true");
    expect(workerSource).toContain("actualSequentialM2Tokens: true");
    expect(workerSource).toContain("await executor.decode(batch, REGULAR_RANGE)");
    expect(workerSource).toContain("await executor.decode(batch)");
    expect(workerSource).toContain("sampleAcePlannerToken(fullSampleInput");
    expect(workerSource).toContain("sampleAcePlannerTokenOpt0084(");
    expect(workerSource).toContain("sampleAcePlannerCompactTokenOpt0084(");
    expect(workerSource).toContain(
      "recurringTokenWallExcludesOneTimeDispatchConstruction: true",
    );
    expect(workerSource).toContain(
      "untimedDecodeAndReplayBeforeEachPositionTiming: true",
    );
    expect(workerSource).toContain(
      "sameStateOppositeHeadReplayExcludedFromCompleteTokenWall: true",
    );
    expect(workerSource).toContain("noCandidateStorageAllocationDuringTiming");
  });

  it("keeps raw exactness, cache/write status, cancellation, and cleanup mandatory", () => {
    expect(workerSource).toContain("requireExactSlice(");
    expect(workerSource).toContain("compareFilteredPaths(");
    expect(workerSource).toContain("requireWorkspaceExact(");
    expect(workerSource).toContain("filteredRawU32MismatchCount: 0");
    expect(workerSource).toContain("weightRawU32MismatchCount: 0");
    expect(workerSource).toContain("captureTimedPrimaryWorkspace(");
    expect(workerSource).toContain(
      "capturedBeforeAnyReplayOrDiagnosticRerun: true",
    );
    expect(workerSource.indexOf("const timedPrimaryWorkspace =")).toBeLessThan(
      workerSource.indexOf("// Replay only the opposite tied head"),
    );
    expect(workerSource).toContain("requireSameCursorSample(");
    expect(workerSource).toContain("cacheAppendGeometryValidatedByExecutor: true");
    expect(workerSource).toContain("mappedWriteStatusValidatedByExecutor: true");
    expect(workerSource).toContain("runCancellationCheckpointProof");
    expect(workerSource).toContain("cursors.A.sample(withoutWord");
    expect(workerSource).toContain("cursors.B.sampleOpt0084(withoutWord");
    expect(workerSource).toContain("cursors.C.sampleCompactOpt0084(");
    expect(workerSource).toContain(
      "abortRaisedInsideActualLogitAccessBeforeCursorCommit: true",
    );
    expect(workerSource).toContain("activeExecutorCancellationProofClaimed: false");
    expect(workerSource).not.toContain("activeExecutorUsesTheSameAbortSignal");
    expect(workerSource).toContain("runSameStateDispatchAndCacheProof(");
    expect(workerSource).toContain(
      "duplicate-equivalent-prefill-ordinary-vs-compact-dispatch-plus-next-full-logit-cache-witness",
    );
    expect(workerSource).toContain(
      "cacheAppendExactNextTokenFullLogitWitness: cacheWitnessExact",
    );
    expect(workerSource).toContain("await ownedExecutor.destroy(reason)");
    expect(workerSource).toContain("ownedContext.destroy()");
    expect(workerSource).toContain("cleanupSecond");
    expect(workerSource).toContain("productionIntegrationAuthorized: false");
  });

  it("enters every real sampler API under abort without committing a cursor", () => {
    const seed = canonicalizeSeed("badc0de");
    for (const arm of ["A", "B", "C"] as const) {
      const controller = new AbortController();
      const length = arm === "C" ? 8 : 16;
      const row = createOpt0084CompoundAbortingLogitRow(
        length,
        controller,
        `abort ${arm}`,
      );
      const cursor = new AcePlannerSamplingCursor(seed, 41);
      const before = cursor.consumed;
      const parameters = Object.freeze({
        temperature: 1,
        guidanceScale: 1,
        topK: 0,
        topP: 1,
        repetitionPenalty: 1,
      });
      let error: unknown;
      try {
        if (arm === "A") {
          cursor.sample({
            conditionalLogits: row,
            seenTokenIds: [],
            allowedTokens: { kind: "all" },
            parameters,
            softmax: ACE_BROWSER_SOFTMAX_V1,
          });
        } else if (arm === "B") {
          cursor.sampleOpt0084({
            conditionalLogits: row,
            seenTokenIds: [],
            allowedTokens: { kind: "all" },
            parameters,
            softmax: ACE_BROWSER_SOFTMAX_V1,
          });
        } else {
          cursor.sampleCompactOpt0084({
            firstTokenId: 4,
            vocabularySize: 16,
            conditionalLogits: row,
            seenTokenIds: [],
            parameters,
            softmax: ACE_BROWSER_SOFTMAX_V1,
          });
        }
      } catch (caught) {
        error = caught;
      }
      expect(error).toBeInstanceOf(DOMException);
      expect((error as DOMException).name).toBe("AbortError");
      expect(controller.signal.aborted).toBe(true);
      expect(cursor.consumed).toBe(before);
    }
  });

  it("does not misidentify copied evidence as OPT-0082", () => {
    expect(workerSource).toContain("reusedExactMechanism:");
    expect(workerSource).toContain("replayTiedHeadForOpt0082");
    const withoutExplicitInternalApi = workerSource.replaceAll(
      "replayTiedHeadForOpt0082",
      "replayTiedHeadForPreviousExactMechanism",
    );
    expect(withoutExplicitInternalApi).not.toMatch(/OPT-0082|Opt0082/);
  });

  it("requires complete run identity and continuous thermal evidence through cleanup", () => {
    const identity = parseOpt0084CompoundRunIdentity(new URLSearchParams({
      harnessCommit: "a".repeat(40),
      machineModel: "Mac15,12",
      osVersion: "26.5.2",
      osBuild: "25F84",
      browserVersion: "151.0.7922.173",
      gpuCoreCount: "10",
      memoryBytes: "17179869184",
    }));
    expect(identity.harnessCommit).toBe("a".repeat(40));

    const launch = parseOpt0084CompoundThermalLaunch(new URLSearchParams({
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
    }), 100_000, 130_250);
    const completionParameters = new URLSearchParams({
        thermalTraceSchema:
          "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
        thermalTraceSha256: "b".repeat(64),
        thermalTraceByteLength: "4096",
        thermalTraceCompletedAtEpochMilliseconds: "180000",
        thermalTraceObservations: "91",
        thermalTraceMaximumPollGapMilliseconds: "1000",
        thermalTraceNonNominalObservations: "0",
        thermalTraceMissingObservations: "0",
        thermalTraceInitialLevel: "0",
        thermalTraceFinalLevel: "0",
        thermalTraceTransitionsJson: "[]",
      });
    const completion = parseOpt0084CompoundThermalCompletion(
      completionParameters,
      launch,
      179_000,
      133_000,
    );
    expect(completion["coversCleanup"]).toBe(true);
    expect(completion["nominalThroughActualMeasurementStart"]).toBe(true);
    expect(completion["workerLaunchGapMilliseconds"]).toBe(3_000);

    const pressureInLaunchGap = new URLSearchParams(completionParameters);
    pressureInLaunchGap.set("thermalTraceNonNominalObservations", "49");
    pressureInLaunchGap.set("thermalTraceFinalLevel", "1");
    pressureInLaunchGap.set("thermalTraceTransitionsJson", JSON.stringify([
      { atEpochMilliseconds: 131_000, level: 1 },
    ]));
    expect(() => parseOpt0084CompoundThermalCompletion(
      pressureInLaunchGap,
      launch,
      179_000,
      133_000,
    )).toThrow(/through-cleanup thermal trace failed/);
    expect(() => parseOpt0084CompoundThermalCompletion(
      completionParameters,
      launch,
      179_000,
      135_001,
    )).toThrow(/through-cleanup thermal trace failed/);
    expect(pageSource).toContain(
      "window.__ACE_OPT0084_COMPOUND_RESULT__ = receipt",
    );
    expect(pageSource).toContain("validateOpt0084RunIdentity");
    expect(workerSource).toContain("validateOpt0084RunIdentity(message.identity)");
    expect(workerSource).toContain("validateThermalLaunch(thermalLaunch");
    expect(workerSource).toContain(
      "worker-clock timing launch gap exceeded 5 seconds",
    );
    expect(workerSource.indexOf(
      "const firstPositionRecurringWarmup = await prepareRecurringPhase",
    )).toBeLessThan(workerSource.indexOf(
      "const readyAtEpochMilliseconds = Date.now()",
    ));
    expect(workerSource).toContain(
      "firstPositionRecurringWarmupCompletedBeforeReady: true",
    );
    expect(pageSource).toContain("nominalThroughActualMeasurementStart");
    expect(htmlSource).toContain('id="thermal-gate"');
    expect(htmlSource).toContain('id="thermal-completion"');
    expect(htmlSource).toContain('name="thermalTraceTransitionsJson"');
    expect(htmlSource).toContain('id="download"');
    expect(contractSource).toContain(
      "OPT_0084_COMPOUND_MINIMUM_PROJECTED_SAVING_SECONDS = 40",
    );
  });
});

function createSamples(wall: Readonly<Record<"A" | "B" | "C", number>>):
  Opt0084CompoundTimingSample[] {
  const samples: Opt0084CompoundTimingSample[] = [];
  for (let roundIndex = 0; roundIndex <
    OPT_0084_COMPOUND_TIMING_ROUND_COUNT; roundIndex += 1) {
    const order = OPT_0084_COMPOUND_ARM_ORDERS[roundIndex]!;
    for (let positionOrder = 0; positionOrder <
      OPT_0084_COMPOUND_POSITION_IDS.length; positionOrder += 1) {
      const positionId = OPT_0084_COMPOUND_POSITION_IDS[positionOrder]!;
      for (let armPosition = 0; armPosition < order.length; armPosition += 1) {
        const arm = order[armPosition]!;
        samples.push(Object.freeze({
          roundIndex,
          positionId,
          arm,
          armPosition: armPosition as 0 | 1 | 2,
          positionOrder,
          cachedTokensBeforeAppend: 300 + roundIndex * 3 + armPosition,
          completeTokenWallMilliseconds: wall[arm],
          modelWallMilliseconds: wall[arm] - 10,
          samplingWallMilliseconds: 10,
          tokenId: 15_000 + roundIndex + armPosition,
          word: 0x1234_0000 + roundIndex * 3 + armPosition,
          drawIndex: 100 + roundIndex * 3 + armPosition,
          cursorEnd: 101 + roundIndex * 3 + armPosition,
          sameStateReplayExact: true,
          cacheWriteStatusValidated: true,
        }));
      }
    }
  }
  return samples;
}
