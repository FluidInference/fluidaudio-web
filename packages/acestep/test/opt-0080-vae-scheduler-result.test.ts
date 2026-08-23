import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const RESULT = record(JSON.parse(source(
  "../optimization/results/OPT-0080/vae-scheduler.json",
)));
const RECORD = source(
  "../optimization/experiments/OPT-0080-dit-depth2-completion-epochs.md",
);
const LEDGER = source("../optimization/LEDGER.md");

describe("OPT-0080 C2314 VAE scheduler result", () => {
  it("binds the exact actual-browser run and every ignored artifact", () => {
    expect(RESULT).toMatchObject({
      schema: "ace-opt-0080-vae-scheduler-evidence-v1",
      experiment: "OPT-0080",
      stage: "c2314-vae-scheduler-screen",
      status: "passed",
      disposition: "production-vae-scheduler-integration-authorized",
      passed: true,
    });
    expect(record(RESULT.identity)).toMatchObject({
      run: {
        coreCommit: "f8fde273ad57d21c8410710518c63e501acaa1ec",
        harnessCommit: "f8fde273ad57d21c8410710518c63e501acaa1ec",
        machineModel: "Mac15,12",
        browserVersion: "Codex in-app Chromium 151.0.0.0",
        gpuCoreCount: 10,
      },
      fixtureSha256:
        "01ec291963276b4784ec0ae3f6b3d7ed80bffd657dfd3b14125729260918783d",
      waveformSha256:
        "2a16f0fc4b07661e21628e0b5574c2feeab3882ecef169da52a671c937f36f0c",
      package: {
        revision: 7,
        physicalOptimizationIdentity: "OPT-0066",
        physicalPackageIdentity: "OPT-0066-revision7",
        manifestSha256:
          "36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7",
        authenticatedAndAcquiredBeforeDeviceRequest: true,
      },
      runtimeProfile: "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1",
    });

    const artifacts = record(RESULT.artifacts);
    const files = record(artifacts.files);
    expect(Object.keys(files)).toHaveLength(21);
    expect(files["raw-receipt.json"]).toEqual({
      byteLength: 3_038_276,
      sha256:
        "6a3d97aa4b0360c268415884d5621a0a20d69bbed85548e19187a1d5e4a6ea64",
    });
    for (const armId of ["A1", "B1", "B2", "A2"] as const) {
      const thermal = record(files[`${armId}-thermal.jsonl`]);
      expect(files[`${armId}-selected.jsonl`]).toEqual(thermal);
      expect(record(files[`${armId}-index.json`]).byteLength).toBe(105);
      expect(record(files[`${armId}-gate.json`]).sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(record(files[`${armId}-trace.json`]).sha256).toMatch(/^[0-9a-f]{64}$/);
    }
  });

  it("retains complete raw-U32 identity and both physical guard boundaries", () => {
    expect(record(RESULT.geometry)).toEqual({
      latentFrames: 2_314,
      maximumWindowFrames: 2_378,
      overlapFrames: 64,
      quantaPerCommandBuffer: 64,
      decoderQuantumCount: 35_498,
      decoderCommandBufferCount: 555,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 556,
      completionFencesPerArm: 556,
      projectedProductionWindowCount: 2,
      outputU32WordCount: 8_885_760,
      outputByteLength: 35_543_040,
    });
    const correctness = record(RESULT.correctness);
    expect(correctness).toMatchObject({
      passed: true,
      excludedFromTiming: true,
      untimedCorrectnessExecutionCount: 3,
      comparedU32WordCountPerPair: 8_885_760,
      controlCandidateU32MismatchCount: 0,
      controlProbeU32MismatchCount: 0,
      controlNonFiniteCount: 0,
      controlNonzeroCount: 8_885_760,
      boundedComparisonCanariesPassed: true,
      topologyExact: true,
      allTimedArmOutputsMatchedAuthority: true,
    });
    expect(record(correctness.gpuGuards)).toEqual({
      scheme: "c2314-active-end-and-c2378-allocation-end-guards-v1",
      guardedBufferCount: 6,
      guardedRegionCount: 12,
      activeEndGuardRegionCount: 6,
      allocationEndGuardRegionCount: 6,
      guardBytesPerRegion: 256,
      checkedBytesPerExecution: 3_072,
      patternU32: 2_774_139_008,
      expectedSha256:
        "183dead51e555d79ec074ad8acfe08c5f4dffce8392ccadc46bb9da2d5aa413d",
      diagnosticCommandBufferCount: 1,
      diagnosticCompletionFenceCount: 1,
      initializationQueueFenceCount: 1,
      excludedFromSchedulerTopologyAndTiming: true,
      passed: true,
    });
  });

  it("retains every fence while reducing only drains and cooperative idles", () => {
    const topology = record(RESULT.topology);
    expect(record(topology.common)).toEqual({
      phaseCommandBufferCounts: [556],
      commandBuffersSubmitted: 556,
      completionFenceRequestedCount: 556,
      completionFenceSettledCount: 556,
      completionFenceRejectedCount: 0,
    });
    expect(record(topology.control)).toEqual({
      schedulingProfile: "depth1-epoch1",
      trueQueueDrainCount: 556,
      completionEpochCount: 556,
      cooperativeIdleTurns: 555,
      requestedCooperativeIdleMs: 555,
      maximumOutstandingCommandBuffers: 1,
    });
    expect(record(topology.candidate)).toEqual({
      schedulingProfile: "depth2-phase-epoch4",
      trueQueueDrainCount: 139,
      completionEpochCount: 139,
      cooperativeIdleTurns: 138,
      requestedCooperativeIdleMs: 138,
      maximumOutstandingCommandBuffers: 2,
    });
  });

  it("recomputes the ABBA win from authoritative complete-window walls", () => {
    const performance = record(RESULT.performance);
    const samples = arrayOfRecords(performance.samples);
    expect(samples.map((sample) => sample.armId)).toEqual([
      "A1", "B1", "B2", "A2",
    ]);
    for (const sample of samples) {
      expect(sample.outputSha256).toBe(
        "2a16f0fc4b07661e21628e0b5574c2feeab3882ecef169da52a671c937f36f0c",
      );
      expect(sample.rawU32MismatchCount).toBe(0);
      expect(number(sample.epochWallSumMs)).toBeLessThanOrEqual(
        number(sample.schedulingWallMs),
      );
      expect(
        number(sample.preSchedulingWallMs) +
          number(sample.schedulingWallMs) +
          number(sample.mapAndDetachWallMs),
      ).toBeCloseTo(number(sample.decodeWindowWallMs), 9);
    }
    const [a1, b1, b2, a2] = samples;
    const forward = number(a1!.decodeWindowWallMs) -
      number(b1!.decodeWindowWallMs);
    const reverse = number(a2!.decodeWindowWallMs) -
      number(b2!.decodeWindowWallMs);
    const aggregateSpeedup = (
      number(a1!.decodeWindowWallMs) + number(a2!.decodeWindowWallMs)
    ) / (
      number(b1!.decodeWindowWallMs) + number(b2!.decodeWindowWallMs)
    );
    expect(forward).toBeCloseTo(
      number(performance.forwardDecodeSavingMs),
      12,
    );
    expect(reverse).toBeCloseTo(
      number(performance.reverseDecodeSavingMs),
      12,
    );
    expect(forward + reverse).toBeCloseTo(
      number(performance.projectedTwoWindowSavingMs),
      12,
    );
    expect(aggregateSpeedup).toBeCloseTo(
      number(performance.aggregateDecodeSpeedup),
      12,
    );
    expect(performance).toMatchObject({
      forwardSchedulingImproved: true,
      reverseSchedulingImproved: true,
      forwardDecodeImproved: true,
      reverseDecodeImproved: true,
      requiredProjectedTwoWindowSavingMs: 800,
      projectedTwoWindowSavingPassed: true,
      wallBoundaryConsistent: true,
      passed: true,
    });
    expect(number(performance.projectedTwoWindowSavingMs)).toBeGreaterThan(800);
  });

  it("retains nominal thermal, responsiveness, cancellation, and lifecycle evidence", () => {
    const thermal = record(RESULT.thermal);
    expect(thermal).toMatchObject({
      passed: true,
      independentNominalStartGatePerArm: true,
      continuousThroughSettlementTracePerArm: true,
      allGateAndTraceObservationsNominal: true,
    });
    const thermalArms = arrayOfRecords(thermal.arms);
    expect(thermalArms.map((arm) => arm.armId)).toEqual([
      "A1", "B1", "B2", "A2",
    ]);
    expect(thermalArms.every((arm) => arm.nonNominalObservationCount === 0))
      .toBe(true);
    expect(Math.max(...thermalArms.map((arm) =>
      number(arm.traceMaximumPollGapMs)
    ))).toBe(935);

    expect(record(RESULT.responsiveness)).toEqual({
      heartbeatIntervalMs: 50,
      maximumGapMs: 52.200000047683716,
      maximumP99GapMs: 52.10000014305115,
      absoluteGatePassed: true,
      relativeGatePassed: true,
      passed: true,
    });
    expect(record(RESULT.cancellation)).toMatchObject({
      scope: "actual-c2314-vae-window",
      schedulingProfile: "depth2-phase-epoch4",
      abortedFromFirstProgressCallback: true,
      progressEventCountAtAbort: 1,
      progressEventCountAfterAbort: 0,
      schedulingEvidenceCallbackCount: 0,
      rejectedWithExactAbortReason: true,
      queueDrainedBeforeRejection: true,
      outputPublished: false,
      unhandledRejectionCount: 0,
      abortThroughRejectionMs: 114.09999990463257,
      gpuGuardsPassed: true,
      postCancellationExactProbePassed: true,
    });
    expect(record(RESULT.lifecycle)).toMatchObject({
      passed: true,
      persistentDeviceBackendAndPackageOwnerCount: 1,
      reusedAcrossCorrectnessCancellationAndTiming: true,
      fifoBackendUseWasSequential: true,
      queueDrainedBeforeBackendDestroy: true,
      backendDestroyPromiseIdempotent: true,
      createdBufferCount: 16,
      destroyedBufferCount: 16,
      liveBufferCountBeforeDeviceDestroy: 0,
      liveBufferBytesBeforeDeviceDestroy: 0,
      mapCallCount: 15,
      unmapCallCount: 15,
      mappedOrPendingBufferCountBeforeDeviceDestroy: 0,
      everyBufferDestroyedExactlyOnce: true,
      deviceContextDestroyed: true,
      runtimeEventCount: 0,
      rejectedSetupAttemptCount: 0,
    });
  });

  it("keeps the historical screen distinct from the later integration evidence", () => {
    expect(record(RESULT.scope)).toMatchObject({
      productionDefaultChanged: false,
      productionVaeSchedulerIntegrated: false,
      productIntegrationEvidence: false,
      packageBytesChanged: false,
      modelMathChanged: false,
      commandBufferContentsChanged: false,
      commandBufferCoalescing: false,
      ditSelectionRetained: true,
      plannerExecuted: false,
      fullProductTimingClaim: false,
      listeningClaim: false,
      underOneMinuteClaim: false,
    });
    expect(record(RESULT.decision)).toEqual({
      literalBrowserDecision:
        "positive-authorize-production-vae-scheduler-selection",
      evidence: "positive",
      productionVaeSchedulerIntegrationAuthorized: true,
      productionVaeSchedulerIntegrated: false,
      postIntegrationProductGateRequired: true,
      productIntegrationEvidence: false,
      plannerVaeSchedulerSelectionAuthorized: false,
      packageOrModelMathChangeAuthorized: false,
    });
    expect(RECORD).toContain(
      "integration authorization, not product\nintegration evidence",
    );
    const row = LEDGER.split("\n").find((line) =>
      line.startsWith("| OPT-0080 |")
    );
    expect(row).toContain("[VAE scheduler result](results/OPT-0080/vae-scheduler.json)");
    expect(row).toContain(
      "[VAE product-selector result](results/OPT-0080/vae-product-selector.json)",
    );
    expect(row).toContain("VAE production integration/product-exact");
    expect(row).not.toContain("VAE production integration pending");
  });
});

function source(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), "utf8");
}

function record(value: unknown): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("Expected an OPT-0080 VAE scheduler result record");
  }
  return value as Readonly<Record<string, unknown>>;
}

function arrayOfRecords(value: unknown): ReadonlyArray<Readonly<Record<string, unknown>>> {
  if (!Array.isArray(value)) {
    throw new TypeError("Expected an OPT-0080 VAE scheduler result array");
  }
  return value.map(record);
}

function number(value: unknown): number {
  if (typeof value !== "number") {
    throw new TypeError("Expected an OPT-0080 VAE scheduler result number");
  }
  return value;
}
