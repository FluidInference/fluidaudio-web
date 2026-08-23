import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const result = JSON.parse(readFileSync(new URL(
  "../optimization/results/OPT-0081/representative-layers.json",
  import.meta.url,
), "utf8")) as Record<string, any>;
const record = readFileSync(new URL(
  "../optimization/experiments/OPT-0081-dit-f16-dense-input-multicast.md",
  import.meta.url,
), "utf8");
const ledger = readFileSync(new URL(
  "../optimization/LEDGER.md",
  import.meta.url,
), "utf8");

const mean = (values: readonly number[]): number =>
  values.reduce((sum, value) => sum + value, 0) / values.length;

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return (sorted[sorted.length / 2 - 1]! + sorted[sorted.length / 2]!) / 2;
}

function sampleStandardDeviation(values: readonly number[]): number {
  const average = mean(values);
  return Math.sqrt(values.reduce((sum, value) =>
    sum + (value - average) ** 2, 0) / (values.length - 1));
}

describe("OPT-0081 representative-layer closed result", () => {
  it("retains exact identity, correctness, arena, and lifecycle evidence", () => {
    expect(result).toMatchObject({
      schemaVersion: 2,
      experimentId: "OPT-0081",
      stage: "representative-two-layer-graph-prefix",
      status: "completed-non-pass",
      identity: {
        coreCommit: "608ae288bb3cb57b0d95a9697990bd050424a965",
        harnessCommit: "608ae288bb3cb57b0d95a9697990bd050424a965",
        storageProfile: "opt-0081-six-dense-input-f16-storage-v1",
        tokens: 2250,
        conditionTokens: 98,
      },
      correctness: {
        passed: true,
        candidateBoundaryWordsPerRun: 73_728_000,
        denseOutputWordsPerComparison: 110_592_000,
        layerOutputWordsPerComparison: 9_216_000,
        totalComparedWordInstances: 654_336_000,
        rawMismatchCount: 0,
        unwrittenWordCount: 0,
      },
      arena: {
        controlBytes: 674_815_488,
        candidateBytes: 601_087_488,
        savingBytes: 73_728_000,
        castOrCopyBufferCountInTimedPath: 0,
        passed: true,
      },
      lifecycle: {
        createdBufferCount: 507,
        destroyedBufferCount: 507,
        mappedRangeCount: 128,
        unmappedRangeCount: 128,
        runtimeOwnerCount: 66,
        destroyedRuntimeOwnerCount: 66,
        liveBufferCountAfterCleanup: 0,
        liveByteCountAfterCleanup: 0,
        passed: true,
      },
    });
  });

  it("recomputes every decisive timing statistic from the raw paired walls", () => {
    const timing = result.timing;
    const a = timing.aWallMilliseconds as number[];
    const b = timing.bWallMilliseconds as number[];
    const savings = a.map((value, index) => value - b[index]!);
    expect(savings).toEqual(timing.pairedSavingsMilliseconds);
    expect(mean(a)).toBeCloseTo(timing.aMeanWallMilliseconds, 12);
    expect(mean(b)).toBeCloseTo(timing.bMeanWallMilliseconds, 12);
    expect(mean(a) / mean(b)).toBeCloseTo(timing.meanSpeedup, 12);
    expect(median(a)).toBeCloseTo(timing.aMedianWallMilliseconds, 12);
    expect(median(b)).toBeCloseTo(timing.bMedianWallMilliseconds, 12);

    const forward = [savings[0]!, savings[3]!, savings[4]!, savings[7]!];
    const reverse = [savings[1]!, savings[2]!, savings[5]!, savings[6]!];
    const t95 = 3.182446305284263;
    const forwardLower = mean(forward) -
      t95 * sampleStandardDeviation(forward) / Math.sqrt(4);
    const reverseLower = mean(reverse) -
      t95 * sampleStandardDeviation(reverse) / Math.sqrt(4);
    expect(mean(forward)).toBeCloseTo(
      timing.forwardMeanPairedSavingMilliseconds,
      12,
    );
    expect(mean(reverse)).toBeCloseTo(
      timing.reverseMeanPairedSavingMilliseconds,
      12,
    );
    expect(forwardLower).toBeCloseTo(timing.forwardLower95Milliseconds, 12);
    expect(reverseLower).toBeCloseTo(timing.reverseLower95Milliseconds, 12);
    expect(savings.filter((value) => value > 0)).toHaveLength(6);
    expect(forward.filter((value) => value > 0)).toHaveLength(4);
    expect(reverse.filter((value) => value > 0)).toHaveLength(2);
    expect(timing.passed).toBe(false);
  });

  it("retains the truthful thermal transition and launch evidence", () => {
    expect(result.thermal).toMatchObject({
      gateObservationCount: 87,
      gateMaximumPollGapMilliseconds: 1_010,
      launchDelayMilliseconds: 898,
      observationCount: 346,
      maximumPollGapMilliseconds: 1_013,
      nonNominalObservationCount: 159,
      missingObservationCount: 0,
      initialLevel: 0,
      finalLevel: 0,
      transitions: [
        { atEpochMilliseconds: 1_787_336_437_169, level: 1 },
        { atEpochMilliseconds: 1_787_336_451_169, level: 2 },
        { atEpochMilliseconds: 1_787_336_569_169, level: 1 },
        { atEpochMilliseconds: 1_787_336_596_169, level: 0 },
      ],
      passed: true,
    });
    expect(result.artifacts).toMatchObject({
      browserReceipt: {
        downloadBlobSha256:
          "eddb46919d5f281d64c3babe4f4de7d68eabb70d955e19d0f690ba10739270de",
        downloadBlobByteLength: 208_588,
      },
      thermalTrace: {
        sha256:
          "55b2e0099c6aa8cf880a929ce35e8a6848f609caca9423f7d1aa616112a0807d",
        byteLength: 37_258,
      },
    });
  });

  it("keeps the non-pass and production disposition synchronized", () => {
    expect(result.decision).toEqual({
      disposition: "inconclusive-directional-or-material-wall-evidence",
      passed: false,
      completeEvaluationFollowUpAuthorized: false,
      fullTrajectoryAuthorized: false,
      productGateAuthorized: false,
      productionIntegrationAuthorized: false,
      packageChangeAuthorized: false,
      unchangedTimingRetryPerformed: false,
      productionRemains:
        "OPT-0009 FP32 activation storage with OPT-0080 scheduling",
    });
    const row = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0081 |"));
    expect(row).toContain("| inconclusive | benchmark-only |");
    expect(row).toContain(
      "[representative result](results/OPT-0081/representative-layers.json)",
    );
    expect(record).toContain("- Evidence: `inconclusive` overall");
    expect(record).toContain(
      "`inconclusive-directional-or-material-wall-evidence`",
    );
    expect(record).toMatch(/No\s+unchanged timing retry is authorized\./);
  });
});
