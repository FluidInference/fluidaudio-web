import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const RESULT = record(JSON.parse(source(
  "../optimization/results/OPT-0080/vae-product-selector.json",
)));
const RECORD = source(
  "../optimization/experiments/OPT-0080-dit-depth2-completion-epochs.md",
);
const LEDGER = source("../optimization/LEDGER.md");

describe("OPT-0080 VAE product-selector result", () => {
  it("binds the exact pushed browser receipt and literal authority", () => {
    expect(RESULT).toMatchObject({
      schema: "ace-opt-0080-vae-product-selector-result-v1",
      experimentId: "OPT-0080",
      stage: "post-integration-vae-product-selector",
      status: "passed",
      disposition: "integrated-product-exact",
    });
    expect(record(RESULT.identity)).toEqual({
      coreCommit: "8d443f6aff4f6a9b06df6db77b3f88b9401123a7",
      harnessCommit: "8d443f6aff4f6a9b06df6db77b3f88b9401123a7",
      machineModel: "Mac15,12",
      osVersion: "26.5.2",
      osBuild: "25F84",
      browserVersion: "Codex in-app Chromium 151.0.0.0",
      gpuCoreCount: 10,
      memoryBytes: 17_179_869_184,
    });
    expect(record(RESULT.artifact)).toEqual({
      path: "optimization/artifacts/OPT-0080/vae-product-selector/raw-receipt.json",
      byteLength: 61_752,
      sha256:
        "bc741387b20be1715eaf80af1661392add8b32318258acce5ff9a8e18cb09225",
    });
    expect(record(RESULT.authority)).toEqual({
      scope:
        "post-integration direct 96-second VAE selector exactness/topology plus ordinary product cancellation",
      timingClaim: false,
      thermalClaim: false,
      plannerClaim: false,
      plannerExecuted: false,
      selectorNegativeCasesCoveredByRuntimeUnitTests: true,
      ordinaryProductionArmWasSeamFree: true,
      unchangedRetryPerformed: false,
    });
  });

  it("pins the three success arms and seam-free ordinary selection", () => {
    expect(RESULT.armOrder).toEqual([
      "control",
      "candidate",
      "production",
      "cancellation",
    ]);
    const selection = record(RESULT.selection);
    expect(selection.productionPolicy).toBe(
      "opt-0080-c2314-depth2-phase-epoch4",
    );
    expect(record(selection.control)).toEqual({
      evidenceMode: "forced-retained-output",
      selectedProductionPolicy: null,
      benchmarkPolicyOverride: "depth1-epoch1",
      windowSchedulingProfiles: ["depth1-epoch1", "depth1-epoch1"],
    });
    expect(record(selection.candidate)).toEqual({
      evidenceMode: "forced-retained-output",
      selectedProductionPolicy: null,
      benchmarkPolicyOverride: "opt-0080-c2314-depth2-phase-epoch4",
      windowSchedulingProfiles: [
        "depth2-phase-epoch4",
        "depth1-epoch1",
      ],
    });
    expect(record(selection.production)).toEqual({
      evidenceMode: "seam-free-ordinary",
      selectedProductionPolicy: "opt-0080-c2314-depth2-phase-epoch4",
      benchmarkPolicyOverride: null,
      windowSchedulingProfiles: [
        "depth2-phase-epoch4",
        "depth1-epoch1",
      ],
      retainedFinalLatentOrRaw: false,
    });
  });

  it("preserves singleton command buffers and fences per exact window", () => {
    const topology = record(RESULT.topology);
    const windows = arrayOfRecords(topology.windows);
    expect(windows).toHaveLength(2);
    expect(topology).toMatchObject({
      schema: "ace-opt-0080-vae-product-selector-topology-v1",
      ordinaryReceiptSchema: "ace-vae-window-scheduling-receipt-v1",
    });
    const full = windows[0]!;
    const remainder = windows[1]!;
    expect(full).toMatchObject({
      windowIndex: 0,
      latentWindowFrames: 2_314,
      decoderQuantumCount: 35_498,
      decoderCommandBufferCount: 555,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 556,
      commandBuffersSubmitted: 556,
      completionFenceRequestedCountInForcedArms: 556,
      completionFenceSettledCountInForcedArms: 556,
      completionFenceRejectedCountInForcedArms: 0,
    });
    expect(record(full.control)).toEqual({
      schedulingProfile: "depth1-epoch1",
      queueDrains: 556,
      completionEpochs: 556,
      cooperativeIdleTurns: 555,
      maximumOutstandingCommandBuffers: 1,
    });
    expect(record(full.candidate)).toEqual({
      schedulingProfile: "depth2-phase-epoch4",
      queueDrains: 139,
      completionEpochs: 139,
      cooperativeIdleTurns: 138,
      maximumOutstandingCommandBuffers: 2,
    });
    expect(record(full.production)).toEqual({
      schedulingProfile: "depth2-phase-epoch4",
      queueDrains: 139,
      cooperativeIdleTurns: 138,
      maximumOutstandingCommandBuffers: 2,
    });

    expect(remainder).toMatchObject({
      windowIndex: 1,
      latentWindowFrames: 214,
      decoderQuantumCount: 3_342,
      decoderCommandBufferCount: 53,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 54,
      commandBuffersSubmitted: 54,
      completionFenceRequestedCountInForcedArms: 54,
      completionFenceSettledCountInForcedArms: 54,
      completionFenceRejectedCountInForcedArms: 0,
    });
    for (const arm of ["control", "candidate", "production"] as const) {
      expect(record(remainder[arm])).toMatchObject({
        schedulingProfile: "depth1-epoch1",
        queueDrains: 54,
        cooperativeIdleTurns: 53,
        maximumOutstandingCommandBuffers: 1,
      });
    }
    expect(topology).toMatchObject({
      forcedArmCommandCompletionOrderExact: true,
      forcedArmCompletionEpochTopologyExact: true,
      vaeQueueDrains: {
        control: 610,
        candidate: 193,
        production: 193,
        controlToCandidateReduction: 417,
      },
      wholeProductQueueDrains: {
        control: 1_243,
        candidate: 826,
        production: 826,
        countsAreTopologyNotTimingEvidence: true,
      },
    });
  });

  it("records forced latent/raw/seam exactness without overstating ordinary retention", () => {
    const correctness = record(RESULT.correctness);
    const forced = record(correctness.controlCandidate);
    expect(record(forced.finalLatent)).toEqual({
      wordCount: 153_600,
      byteLength: 614_400,
      exactU32MismatchCount: 0,
      sha256:
        "527cdc7e560691f21383f3b06a4a85f7f41ba92e93e6357b7be75f115a5c9e07",
    });
    expect(record(forced.fullRawWaveform)).toMatchObject({
      sampleCount: 9_216_000,
      byteLength: 36_864_000,
      exactU32MismatchCount: 0,
      nonFiniteCountPerArm: 0,
      maximumAbsoluteDifference: 0,
      meanAbsoluteDifference: 0,
      rootMeanSquareDifference: 0,
      sha256:
        "c4152aa56bcf81236b60cb2dbea3976b4a7f4d800af001b9bfbbbd52dda6e82b",
    });
    expect(record(forced.seamRawWaveform)).toMatchObject({
      seamLatentFrame: 2_250,
      seamAudioFrame: 4_320_000,
      radiusLatentFrames: 64,
      startAudioFrame: 4_197_120,
      endAudioFrame: 4_442_880,
      sampleCount: 491_520,
      byteLength: 1_966_080,
      exactU32MismatchCount: 0,
      nonFiniteCountPerArm: 0,
      sha256:
        "17540647f319a947860bf7402721e3e942f089cfc90a8d8aa20d8191ff889830",
    });
    const ordinary = record(correctness.candidateProduction);
    expect(ordinary.ordinaryRetainedFinalLatentOrRaw).toBe(false);
    expect(record(ordinary.normalizedWav)).toEqual({
      byteLength: 36_864_044,
      exactByteMismatchCount: 0,
      sha256:
        "c088385a6b4dabc30215d122b3a4da8406611f1a7d6d1255eba2846aa7e24e4a",
    });
    expect(ordinary.stableRequestMetadataRuntimePlanAndDiagnosticsExact)
      .toBe(true);
    expect(correctness.passed).toBe(true);
  });

  it("retains responsiveness, cancellation, and bounded lifecycle evidence", () => {
    expect(record(RESULT.responsiveness)).toEqual({
      heartbeatIntervalMilliseconds: 50,
      maximumAllowedGapMilliseconds: 500,
      maximumObservedGapMilliseconds: 52.90000009536743,
      maximumObservedP99GapMilliseconds: 52.200000047683716,
      allFourArmsPassed: true,
    });
    expect(record(RESULT.cancellation)).toMatchObject({
      ordinaryProductionSelector: true,
      opt0080ProductRunOmitted: true,
      triggerCount: 1,
      publicProgressAfterAbortCount: 0,
      abortReasonIdentityPreserved: true,
      postDitStageCount: 0,
      outputCount: 0,
      evidenceCount: 0,
      fatalDiagnosticCount: 0,
      unhandledRejectionCount: 0,
      workerReadyToTerminate: true,
      passed: true,
    });
    expect(record(RESULT.lifecycle)).toEqual({
      freshWorkerCount: 4,
      terminatedWorkerCount: 4,
      maximumLargeArmPayloadsRetained: 2,
      explicitReleaseAfterComparison: true,
      largePayloadsReleasedBeforeCancellation: true,
      allSuccessArmsReleasedBeforeDispose: true,
      automaticRetryCount: 0,
      workerOrderWasOneWay: true,
      passed: true,
    });
  });

  it("closes the pending integration state without timing, thermal, or planner claims", () => {
    expect(record(RESULT.decision)).toEqual({
      evidence: "positive",
      productionVaeSchedulerIntegrated: true,
      postIntegrationProductGatePassed: true,
      directProductExact: true,
      ordinarySelectorTopologyObserved: true,
      productTimingEvidence: false,
      productThermalEvidence: false,
      plannerProductEvidence: false,
      packageOrModelMathChangeAuthorized: false,
    });
    expect(RECORD).toContain(
      "## VAE production integration and product-exact gate — passed 2026-08-21",
    );
    expect(RECORD).toContain(
      "made no product timing comparison or timing claim",
    );
    expect(RECORD).toContain(
      "captured no\nthermal trace and makes no thermal claim",
    );
    expect(RECORD).toContain(
      "the planner was never executed and no planner claim follows",
    );
    const row = LEDGER.split("\n").find((line) =>
      line.startsWith("| OPT-0080 |")
    );
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
    throw new TypeError("Expected an OPT-0080 VAE selector result record");
  }
  return value as Readonly<Record<string, unknown>>;
}

function arrayOfRecords(
  value: unknown,
): ReadonlyArray<Readonly<Record<string, unknown>>> {
  if (!Array.isArray(value)) {
    throw new TypeError("Expected an OPT-0080 VAE selector result array");
  }
  return value.map(record);
}
