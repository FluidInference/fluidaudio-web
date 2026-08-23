import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const RESULT_PATH = new URL(
  "../optimization/results/OPT-0079/result.json",
  import.meta.url,
);
const RECORD_PATH = new URL(
  "../optimization/experiments/OPT-0079-dit-dense-decoded-half-tile-multicast.md",
  import.meta.url,
);
const LEDGER_PATH = new URL("../optimization/LEDGER.md", import.meta.url);

const result = JSON.parse(readFileSync(RESULT_PATH, "utf8")) as Record<
  string,
  any
>;
const record = readFileSync(RECORD_PATH, "utf8");
const ledger = readFileSync(LEDGER_PATH, "utf8");

function sha256(path: string): string {
  return createHash("sha256").update(readFileSync(
    new URL(path, import.meta.url),
  )).digest("hex");
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]): number {
  const ordered = [...values].sort((left, right) => left - right);
  return (ordered[3]! + ordered[4]!) / 2;
}

describe("OPT-0079 closed result contract", () => {
  it("retains exact candidate and artifact identity", () => {
    expect(result).toMatchObject({
      schema: "ace-opt-0079-dense-decoded-half-tile-result-v1",
      experiment: "OPT-0079",
      status: "inconclusive",
      disposition: "benchmark-only",
      passed: false,
      identity: {
        allocationBaselineCommit:
          "4084610c5e43dff2d388361965750b0f603400dd",
        registrationCommit: "6313bfbaa61dee91f065d1652ee8eb1446f987b8",
        candidateAndHarnessCommit:
          "aade2c0223383bab99cf477e7697e2394c05a380",
        generatedShaderAggregateSha256:
          "6817dd152bed6475509c1a84c42ccd4554957f0c31749f3186d62f9ca8d8d58c",
      },
      artifacts: {
        authoritativeBrowserReceipt: {
          committed: false,
          sha256:
            "ed4033518620efc5404ef0416238de84bf16631193f330f16e391e5f2b4a1842",
          byteLength: 133_315,
        },
        externalThermalTrace: {
          committed: false,
          sha256:
            "499c3ebe469567c81232b2d30707dca65fc550d794eca9d34dc41ccd82a3b9ea",
          byteLength: 17_276,
        },
      },
    });
    expect(result.identity.sourceSha256).toMatchObject({
      candidateKernel: sha256(
        "../src/webgpu/kernels/dit-dense-fp16-decoded-half-tile.ts",
      ),
      kernelContractTest: sha256(
        "./dit-dense-fp16-decoded-half-tile.test.ts",
      ),
      browserHarness: sha256(
        "./browser/opt-0079-dit-dense-decoded-half-tile.ts",
      ),
      browserHtml: sha256(
        "./browser/opt-0079-dit-dense-decoded-half-tile.html",
      ),
      browserContractTest: sha256(
        "./opt-0079-dit-dense-decoded-half-tile-browser.test.ts",
      ),
    });
  });

  it("retains the exact raw-U32 and complete-write evidence", () => {
    expect(result.correctness).toMatchObject({
      passed: true,
      shapeCount: 4,
      comparedU32CountPerArmComparison: 25_344_000,
      armComparisonCount: 3,
      perShapeArmComparisonCount: 12,
      totalComparedU32Count: 76_032_000,
      currentCandidateDifferingU32Count: 0,
      currentRerunDifferingU32Count: 0,
      candidateRerunDifferingU32Count: 0,
      boundedAdversarialFixtureCount: 5,
      everyOutputRawU32Exact: true,
      finiteClassIdentity: true,
      allOutputsFiniteAndComplete: true,
      qnanPrefillCompletelyOverwritten: true,
      guardsAndPartialMTailsIntact: true,
      malformedBindingSetupFailureRejectedBeforeAllocationOrSubmission: true,
      uncapturedGpuErrorCount: 0,
      deviceLossCount: 0,
    });
    expect(result.correctness.resultSha256ByShape).toEqual({
      "h-h": "06fd9f24b337d2b8031d1ed56d48c896a9f280c0063ae9645b11eb84406edf33",
      "h-1024": "86e0b44db6903280534062590dc72b612ce43ff7aff5d7d1a160913971b6d994",
      "h-6144": "e28ca0f527929fbc2e0bd83ea1cc4d338c83132d7f8fe63974ba1e6dc3b8e1d2",
      "6144-h": "09652e3e5b5f505e5f391651c4c2e5ce1cbc7ec7958dfcc09bf49cc2007ce81f",
    });
  });

  it("retains all weighted observations and their derived statistics", () => {
    const timing = result.timing;
    expect(timing.shapeSummaries).toHaveLength(4);
    expect(timing.shapeSummaries.find((shape: any) => shape.id === "h-h")
      .meanGpuSpeedup).toBeLessThan(1);
    expect(timing.shapeSummaries.find((shape: any) => shape.id === "h-1024")
      .medianGpuSpeedup).toBeLessThan(1);

    for (const score of [
      timing.completeWeightedScore,
      timing.feedForwardWeightedScore,
    ]) {
      for (const samples of Object.values(score.rawSamples) as number[][]) {
        expect(samples).toHaveLength(8);
      }
    }

    const complete = timing.completeWeightedScore;
    expect(mean(complete.rawSamples.currentGpuMilliseconds)).toBeCloseTo(
      complete.mean.currentGpuMilliseconds,
      12,
    );
    expect(mean(complete.rawSamples.candidateGpuMilliseconds)).toBeCloseTo(
      complete.mean.candidateGpuMilliseconds,
      12,
    );
    expect(mean(complete.rawSamples.currentWallMilliseconds)).toBeCloseTo(
      complete.mean.currentWallMilliseconds,
      12,
    );
    expect(mean(complete.rawSamples.candidateWallMilliseconds)).toBeCloseTo(
      complete.mean.candidateWallMilliseconds,
      12,
    );
    expect(median(complete.rawSamples.currentGpuMilliseconds)).toBeCloseTo(
      complete.median.currentGpuMilliseconds,
      12,
    );
    expect(median(complete.rawSamples.candidateGpuMilliseconds)).toBeCloseTo(
      complete.median.candidateGpuMilliseconds,
      12,
    );
    expect(median(complete.rawSamples.currentWallMilliseconds)).toBeCloseTo(
      complete.median.currentWallMilliseconds,
      12,
    );
    expect(median(complete.rawSamples.candidateWallMilliseconds)).toBeCloseTo(
      complete.median.candidateWallMilliseconds,
      12,
    );

    const gpuWins = complete.rawSamples.currentGpuMilliseconds.filter(
      (current: number, index: number) =>
        complete.rawSamples.candidateGpuMilliseconds[index] < current,
    ).length;
    const wallWins = complete.rawSamples.currentWallMilliseconds.filter(
      (current: number, index: number) =>
        complete.rawSamples.candidateWallMilliseconds[index] < current,
    ).length;
    expect(gpuWins).toBe(6);
    expect(wallWins).toBe(6);
    expect(complete).toMatchObject({
      mean: {
        gpuSpeedup: 1.0498897132544616,
        gpuSavingMilliseconds: 10.190847999999988,
        wallSpeedup: 1.073385404930294,
        wallSavingMilliseconds: 15.737500086426735,
      },
      median: {
        gpuSpeedup: 1.100942587832048,
        gpuSavingMilliseconds: 19.300352000000004,
        wallSpeedup: 1.105759683309193,
        wallSavingMilliseconds: 21.300000190734863,
      },
    });
    expect(timing.feedForwardWeightedScore).toMatchObject({
      mean: {
        gpuSpeedup: 1.067670669408877,
        gpuSavingMilliseconds: 8.35584,
        wallSpeedup: 1.0778825381308181,
        wallSavingMilliseconds: 9.912500038743019,
      },
      median: {
        gpuSpeedup: 1.100688924218336,
        gpuSavingMilliseconds: 12.45183999999999,
        wallSpeedup: 1.1139240504657777,
        wallSavingMilliseconds: 14.399999976158142,
      },
    });
  });

  it("retains every literal primitive-gate failure", () => {
    const gate = result.timing.frozenGate;
    const complete = result.timing.completeWeightedScore;
    expect(gate.projectedMeanGpuSavingMillisecondsAcross24LayersAnd8Evaluations)
      .toBeCloseTo(complete.mean.gpuSavingMilliseconds * 24 * 8, 12);
    expect(gate.projectedMedianGpuSavingMillisecondsAcross24LayersAnd8Evaluations)
      .toBeCloseTo(complete.median.gpuSavingMilliseconds * 24 * 8, 12);
    expect(gate.projectedMeanWallSavingMillisecondsAcross24LayersAnd8Evaluations)
      .toBeCloseTo(complete.mean.wallSavingMilliseconds * 24 * 8, 12);
    expect(gate.projectedMedianWallSavingMillisecondsAcross24LayersAnd8Evaluations)
      .toBeCloseTo(complete.median.wallSavingMilliseconds * 24 * 8, 12);
    expect(gate.meanWallGpuSavingRatio).toBeCloseTo(
      complete.mean.wallSavingMilliseconds /
        complete.mean.gpuSavingMilliseconds,
      12,
    );
    expect(gate.medianWallGpuSavingRatio).toBeCloseTo(
      complete.median.wallSavingMilliseconds /
        complete.median.gpuSavingMilliseconds,
      12,
    );
    expect(gate).toEqual({
      everyShapeMeanAndMedianFaster: false,
      requiredWeightedPairWins: 7,
      gpuWeightedPairWins: 6,
      wallWeightedPairWins: 6,
      gpuWeightedPairWinsPassed: false,
      wallWeightedPairWinsPassed: false,
      requiredMeanAndMedianSpeedup: 1.15,
      meanGpuSpeedupPassed: false,
      medianGpuSpeedupPassed: false,
      meanWallSpeedupPassed: false,
      medianWallSpeedupPassed: false,
      requiredMeanAndMedianSavingMilliseconds: 25,
      meanGpuSavingPassed: false,
      medianGpuSavingPassed: false,
      meanWallSavingPassed: false,
      medianWallSavingPassed: false,
      meanWallGpuSavingRatio: 1.5442777761405875,
      medianWallGpuSavingRatio: 1.1036068249291442,
      wallGpuSavingsAgree: false,
      projectedMeanGpuSavingMillisecondsAcross24LayersAnd8Evaluations:
        1956.6428159999978,
      projectedMedianGpuSavingMillisecondsAcross24LayersAnd8Evaluations:
        3705.6675840000007,
      projectedMeanWallSavingMillisecondsAcross24LayersAnd8Evaluations:
        3021.600016593933,
      projectedMedianWallSavingMillisecondsAcross24LayersAnd8Evaluations:
        4089.6000366210938,
      requiredProjectedSavingMilliseconds: 4800,
      projectedSavingPassed: false,
      passed: false,
    });
  });

  it("retains nominal thermal and balanced lifecycle evidence", () => {
    expect(result.thermal).toMatchObject({
      passed: true,
      gateObservationCount: 32,
      traceObservationCount: 67,
      maximumPollGapMilliseconds: 1_010,
      nonNominalObservationCount: 0,
      missingObservationCount: 0,
      launchDelayMilliseconds: 499,
      coversCleanup: true,
    });
    expect(result.lifecycle).toMatchObject({
      passed: true,
      createdBufferCount: 22,
      destroyedBufferCount: 22,
      maximumLiveBytes: 293_317_664,
      liveBufferCountAfterCleanup: 0,
      liveBytesAfterCleanup: 0,
      mapCount: 92,
      unmapCount: 92,
      activeMapCountAfterCleanup: 0,
      idempotentDestroy: true,
      postDestroyRejected: true,
    });
  });

  it("keeps evidence, disposition, and authorization synchronized", () => {
    expect(result.decision).toMatchObject({
      evidence: "inconclusive",
      disposition: "benchmark-only",
      diagnosticProfileFollowUpAuthorized: false,
      productionIntegrationAuthorized: false,
      packageChangeAuthorized: false,
      trajectoryOrListeningClaim: false,
      unchangedTimingRetryAuthorized: false,
    });
    expect(ledger).toContain("Next available ID: `OPT-0089`");
    const row = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0079 |"));
    expect(row).toContain("| inconclusive | benchmark-only |");
    expect(row).toContain("[result](results/OPT-0079/result.json)");
    expect(record).toContain("- Evidence: `inconclusive`");
    expect(record).toContain("- Disposition: `benchmark-only`");
    expect(record).toMatch(
      /literal page decision was inconclusive with no follow-up\s+authorization/,
    );
  });
});
