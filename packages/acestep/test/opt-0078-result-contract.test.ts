import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const RESULT_PATH = new URL(
  "../optimization/results/OPT-0078/result.json",
  import.meta.url,
);
const RECORD_PATH = new URL(
  "../optimization/experiments/OPT-0078-dit-dense-weight-tile-multicast.md",
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

describe("OPT-0078 closed result contract", () => {
  it("retains exact candidate and artifact identity", () => {
    expect(result).toMatchObject({
      schema: "ace-opt-0078-dense-weight-multicast-result-v1",
      experiment: "OPT-0078",
      status: "inconclusive",
      disposition: "benchmark-only",
      passed: false,
      identity: {
        allocationBaselineCommit:
          "9a5f54719208b46f869b69f11271a045ae929047",
        registrationCommit: "23e09472eb997c3543d98f07767d08182b572938",
        candidateAndHarnessCommit:
          "bfd286002654dc67b85d2986686ad917e497d073",
        generatedShaderAggregateSha256:
          "0554a6bb3c1b8cce63e167f9bcdec31c64f4cc439c405c1e2c1f3bb16fb3d9ff",
      },
      artifacts: {
        authoritativeBrowserReceipt: {
          committed: false,
          sha256:
            "27a4b899d469bfb47cf1ff42ec8c95a7623085722b623bb238f86aed4c0b5ffe",
          byteLength: 124_834,
        },
        externalThermalTrace: {
          committed: false,
          sha256:
            "ea21d97f514bb2b934acb677f59cf25b74eefd058c696af8947eba3f0d5d0e76",
          byteLength: 21_662,
        },
      },
    });
    expect(result.identity.sourceSha256).toMatchObject({
      candidateKernel: sha256(
        "../src/webgpu/kernels/dit-dense-fp16-weight-multicast.ts",
      ),
      kernelContractTest: sha256(
        "./dit-dense-fp16-weight-multicast.test.ts",
      ),
      browserHarness: sha256(
        "./browser/opt-0078-dit-dense-weight-multicast.ts",
      ),
      browserHtml: sha256(
        "./browser/opt-0078-dit-dense-weight-multicast.html",
      ),
      browserContractTest: sha256(
        "./opt-0078-dit-dense-weight-multicast-browser.test.ts",
      ),
    });
  });

  it("retains the exact raw-U32 and complete-write evidence", () => {
    expect(result.correctness).toMatchObject({
      passed: true,
      shapeCount: 4,
      comparedU32CountPerArmComparison: 25_344_000,
      armComparisonCount: 3,
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
      uncapturedGpuErrorCount: 0,
      deviceLossCount: 0,
    });
    expect(Object.keys(result.correctness.resultSha256ByShape)).toEqual([
      "h-h", "h-1024", "h-6144", "6144-h",
    ]);
  });

  it("retains the directional samples and every literal gate outcome", () => {
    const timing = result.timing;
    expect(timing.shapeSummaries).toHaveLength(4);
    expect(timing.shapeSummaries.every((shape: any) =>
      shape.meanGpuSpeedup > 1 && shape.medianGpuSpeedup > 1 &&
      shape.meanWallSpeedup > 1 && shape.medianWallSpeedup > 1
    )).toBe(true);
    for (const samples of Object.values(
      timing.completeWeightedScore.rawSamples,
    ) as unknown[][]) expect(samples).toHaveLength(8);
    for (const samples of Object.values(
      timing.feedForwardWeightedScore.rawSamples,
    ) as unknown[][]) expect(samples).toHaveLength(8);
    expect(timing.completeWeightedScore).toMatchObject({
      mean: {
        gpuSpeedup: 1.1083899645253663,
        gpuSavingMilliseconds: 20.774911999999972,
        wallSpeedup: 1.1108710419291574,
        wallSavingMilliseconds: 22.4500000923872,
      },
      median: {
        gpuSpeedup: 1.1896041435442102,
        gpuSavingMilliseconds: 33.587199999999996,
        wallSpeedup: 1.1780090763923512,
        wallSavingMilliseconds: 33.35000038146973,
      },
    });
    expect(timing.frozenGate).toEqual({
      everyShapeMeanAndMedianFaster: true,
      requiredWeightedPairWins: 7,
      gpuWeightedPairWins: 8,
      wallWeightedPairWins: 8,
      requiredMeanAndMedianSpeedup: 1.12,
      meanGpuSpeedupPassed: false,
      medianGpuSpeedupPassed: true,
      meanWallSpeedupPassed: false,
      medianWallSpeedupPassed: true,
      requiredMeanAndMedianSavingMilliseconds: 20.8334,
      meanGpuSavingPassed: false,
      medianGpuSavingPassed: true,
      meanWallSavingPassed: true,
      medianWallSavingPassed: true,
      meanWallGpuSavingRatio: 1.0806303339521837,
      medianWallGpuSavingRatio: 0.9929377971807632,
      wallGpuSavingsAgree: true,
      projectedMeanGpuSavingMillisecondsAcross24LayersAnd8Evaluations:
        3988.7831039999946,
      projectedMedianGpuSavingMillisecondsAcross24LayersAnd8Evaluations:
        6448.742399999999,
      projectedMeanWallSavingMillisecondsAcross24LayersAnd8Evaluations:
        4310.400017738342,
      projectedMedianWallSavingMillisecondsAcross24LayersAnd8Evaluations:
        6403.2000732421875,
      requiredProjectedSavingMilliseconds: 4000,
      projectedMeanGpuSavingPassed: false,
      passed: false,
    });
  });

  it("retains nominal thermal and balanced lifecycle evidence", () => {
    expect(result.thermal).toMatchObject({
      passed: true,
      gateObservationCount: 32,
      traceObservationCount: 84,
      maximumPollGapMilliseconds: 1_016,
      nonNominalObservationCount: 0,
      missingObservationCount: 0,
      launchDelayMilliseconds: 1_054,
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
    expect(ledger).toContain("Next available ID: `OPT-0088`");
    const row = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0078 |"));
    expect(row).toContain("| inconclusive | benchmark-only |");
    expect(row).toContain("[result](results/OPT-0078/result.json)");
    expect(record).toContain("- Evidence: `inconclusive`");
    expect(record).toContain("- Disposition: `benchmark-only`");
    expect(record).toContain(
      "decision was therefore inconclusive with no follow-up authorization",
    );
  });
});
