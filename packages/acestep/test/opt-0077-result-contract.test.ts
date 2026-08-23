import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const RESULT_PATH = new URL(
  "../optimization/results/OPT-0077/result.json",
  import.meta.url,
);
const RECORD_PATH = new URL(
  "../optimization/experiments/OPT-0077-vae-k7-rfft16-transform-domain.md",
  import.meta.url,
);
const LEDGER_PATH = new URL("../optimization/LEDGER.md", import.meta.url);

const result = JSON.parse(readFileSync(RESULT_PATH, "utf8")) as Record<
  string,
  any
>;
const record = readFileSync(RECORD_PATH, "utf8");
const ledger = readFileSync(LEDGER_PATH, "utf8");

describe("OPT-0077 closed result contract", () => {
  it("closes the exact RFFT16 mechanism as negative without escalation", () => {
    expect(result).toMatchObject({
      schema: "ace-opt-0077-vae-k7-rfft16-result-v1",
      experiment: "OPT-0077",
      status: "negative",
      disposition: "abandoned",
      passed: false,
      identity: {
        registrationCommit: "3539a87",
        candidateAndHarnessCommit:
          "f78d0d24c9b205a9614703e17b7e870793d6107e",
      },
      decision: {
        disposition: "negative-stop-exact-rfft16-mechanism",
        packageC512FollowupAuthorized: false,
        productionIntegrationAuthorized: false,
        packageChangeAuthorized: false,
        trajectoryOrListeningRunAuthorized: false,
        productSpeedClaim: false,
        qualityOrListeningClaim: false,
      },
    });
  });

  it("retains the authoritative artifact hashes without committing raw JSONL", () => {
    expect(result.artifacts).toMatchObject({
      authoritativeBrowserReceipt: {
        committed: false,
        sha256:
          "724b9983e314ff68ec4060254914feaf5b9e8005e3ba2af49a00fe352dd35e4a",
        byteLength: 898_231,
      },
      externalThermalTrace: {
        committed: false,
        sha256:
          "1e6c451123330b377fb7d44ba3c3f4e3b7183dc3eb9c9b67a19703d63cf8f21f",
        byteLength: 66_632,
      },
    });
  });

  it("separates the rejected operator preflight from experiment evidence", () => {
    expect(result.setupOnlyRejectedPreflight).toMatchObject({
      classification: "operator-setup-preflight-not-experiment-evidence",
      receiptPersisted: false,
      receiptSha256: null,
      stage: "thermal-launch",
      timingDispatchCount: 0,
      timingSampleCount: 0,
      unchangedTimingRetryPerformed: false,
      enteredGate: {
        observationCount: 98,
        maximumPollGapMilliseconds: 1_010,
        nonNominalObservationCount: 0,
        missingObservationCount: 0,
      },
      cleanup: {
        createdBufferCount: 717,
        destroyedBufferCount: 717,
        liveBufferCount: 0,
        liveBytes: 0,
        mapCount: 628,
        unmapCount: 628,
        passed: true,
      },
    });
  });

  it("retains the passing numerical and guarded-write evidence", () => {
    expect(result.correctness).toMatchObject({
      passed: true,
      caseCount: 29,
      productionStratumCount: 9,
      adversarialCaseCount: 20,
      productionProbeCount: 27,
      totalProbeCount: 47,
      candidateScalarComparedU16Count: 888_448,
      candidateRepeatComparedU16Count: 888_448,
      candidateRepeatDifferingU16Count: 0,
      writesAndGuards: {
        snapshotCount: 188,
        nonFiniteOutputCount: 0,
        qNaNPrefillRemainingCount: 0,
        outOfRangeWriteCount: 0,
        passed: true,
      },
      uncapturedGpuErrorCount: 0,
      deviceLossCount: 0,
    });
    expect(result.correctness.numericalEnvelope).toMatchObject({
      candidateVersusScalarOracleAggregate: {
        fp16UlpDistributionCount: 888_448,
        nrmse: 0.000078497127375363,
        snrDb: 82.10292472187346,
        pearson: 0.999999996937326,
        relativeMaximumAbsoluteError: 0.0008223684210526315,
      },
      everyProbePassed: true,
      passed: true,
    });
    const ulpCount = Object.values(
      result.correctness.numericalEnvelope
        .candidateVersusScalarOracleAggregate.fp16UlpDistribution,
    ).reduce((sum: number, count) => sum + Number(count), 0);
    expect(ulpCount).toBe(888_448);
  });

  it("retains all samples and every literal failed performance gate", () => {
    expect(result.timing).toMatchObject({
      sampleCountPerArm: 6,
      currentAggregateDispatches: 42,
      candidateAggregateDispatches: 126,
      speedup: {
        meanGpu: 0.9990321800145174,
        medianGpu: 0.9307858687815429,
        meanWall: 1.0007230672645717,
        medianWall: 0.9308855294329339,
      },
      gates: {
        everyStratumMedianGpuNonSlower: false,
        requiredPairWins: 5,
        gpuPairWins: 1,
        wallPairWins: 1,
        gpuPairWinsPassed: false,
        wallPairWinsPassed: false,
        meanGpuSpeedupPassed: false,
        medianGpuSpeedupPassed: false,
        meanWallSpeedupPassed: false,
        medianWallSpeedupPassed: false,
        projectedSavingMilliseconds: -1197.2306213278046,
        projectedSavingPassed: false,
      },
      passed: false,
    });
    expect(result.timing.strata).toHaveLength(9);
    expect(result.timing.arms.current.samples).toHaveLength(6);
    expect(result.timing.arms.candidate.samples).toHaveLength(6);
    expect(result.timing.pairedRounds).toHaveLength(6);
    expect(result.timing.strata.filter((entry: any) =>
      !entry.nonSlower).map((entry: any) => entry.id)).toEqual([
      "c1024-d1", "c1024-d9", "c512-d1",
    ]);
  });

  it("retains resource, lifecycle, and all-nominal thermal facts", () => {
    expect(result.resources).toMatchObject({
      persistentPayload: {
        routeCount: 12,
        currentBytes: 56_426_496,
        candidateBytes: 128_974_848,
        increaseBytes: 72_548_352,
      },
      scratch: {
        maximumTotalScratchBytes: 5_308_416,
        maximumScratchBindingBytes: 3_538_944,
        maximumScratchBindingCount: 2,
        passed: true,
      },
    });
    expect(result.memoryAndLifecycle).toMatchObject({
      cleanup: {
        createdBufferCount: 717,
        destroyedBufferCount: 717,
        liveBufferCount: 0,
        liveBytes: 0,
        maximumLiveBytes: 313_105_520,
        mapCount: 748,
        unmapCount: 748,
        activeMapCount: 0,
        queueDrained: true,
        ownersDestroyedIdempotently: true,
        postDestroyRejected: true,
      },
      passed: true,
    });
    expect(result.thermal).toMatchObject({
      completion: {
        observationCount: 302,
        maximumPollGapMilliseconds: 1_010,
        nonNominalObservationCount: 0,
        missingObservationCount: 0,
        coversCleanup: true,
      },
      passed: true,
    });
  });

  it("keeps the ledger ID and written disposition synchronized", () => {
    expect(ledger).toContain("Next available ID: `OPT-0089`");
    const row = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0077 |"));
    expect(row).toContain("| negative | abandoned |");
    expect(row).toContain("[result](results/OPT-0077/result.json)");
    expect(record).toContain("- Evidence: `negative`");
    expect(record).toContain("- Disposition: `abandoned`");
    expect(record).toContain(
      "No C512 package experiment, decoder profile, waveform, listening run",
    );
  });
});
