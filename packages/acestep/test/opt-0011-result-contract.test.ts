import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

const RAW_ARTIFACTS = [
  [
    "conv1d-fp16-correctness.json",
    "5dffca9c0f76012fa07305d1ff11eac32f56206d9280434bb0a1c639fd13e2d7",
  ],
  [
    "production-conv1d-fp16-correctness.json",
    "58c15ecf91926ed089f7c866217d46a5bc05a5971949ec7ffd2d48d248253593",
  ],
  [
    "pointwise-fp16-correctness.json",
    "22ff18f77d0ec154a45ca2a9dca39e8a4694b94662a176c145d8370307bd9d5c",
  ],
  [
    "conv-transpose1d-fp16-correctness.json",
    "3c06879410036b42b70e3da408a1376900316defadf4515cbf58651a68962d68",
  ],
  [
    "snake-fp16-correctness.json",
    "214767002480be3a6fe63e72fb5caaf5bff36f9ffb6cbf15d2f80687ad9190d0",
  ],
] as const;

interface Opt0011Candidate {
  readonly completedPrimitiveEvidence: {
    readonly convTranspose1d: Record<string, unknown>;
    readonly k7Conv1d: Record<string, unknown>;
    readonly pointwise: {
      readonly ingressCaseCount: number;
      readonly ingressElementsPerExecution: number;
      readonly addCaseCount: number;
      readonly addProductionOperationCount: number;
      readonly addProductionRangeCount: number;
      readonly addProductionElementCountPerExecution: number;
      readonly addArithmeticElementCount: number;
      readonly totalRangeCount: number;
      readonly elementComparisonsPerExecution: number;
      readonly totalRawBitComparisons: number;
      readonly mismatchCount: number;
      readonly deterministicRerunHashes: boolean;
      readonly allRangeGuardsPassed: boolean;
      readonly completeFiniteWrites: boolean;
      readonly independentAuditDecision: string;
      readonly rawArtifactByteLength: number;
      readonly executionCounts: Record<string, number>;
      readonly cancellation: Record<string, unknown>;
      readonly lifecycle: Record<string, unknown>;
    };
    readonly productionConv1d: Record<string, unknown>;
    readonly snake: Record<string, unknown>;
  };
  readonly scope: Record<string, boolean>;
}

describe("OPT-0011 partial optimization result", () => {
  it("freezes the positive primitive receipts without closing the experiment", async () => {
    const text = await readFile(new URL(
      "../optimization/results/OPT-0011/result.json",
      import.meta.url,
    ), "utf8");
    const committed = parseOptimizationResultJson(text);
    const candidate = committed.metrics.candidate as unknown as Opt0011Candidate;

    expect(committed).toMatchObject({
      experimentId: "OPT-0011",
      riskClass: "approximate",
      baselineCommit: "f07afbeb425157ca9c1eb4a9bc2102365ab7f616",
      candidateCommit: "59d94643def58a96c1cc081c177c91bd308fd83b",
      identity: {
        benchmarkHarnessCommit:
          "59d94643def58a96c1cc081c177c91bd308fd83b",
        convTranspose1dCoreCommit:
          "d2bf0819d0460f6bd60ebe0457eb091b45e7bf6a",
        convTranspose1dCoreSourceSha256:
          "ecad5f7e981c7310d73565cb15a95123d32725ed6bb41342f484235db3caadd5",
        convTranspose1dHarnessCommit:
          "356f49b20841d0f051fcae3825a87c645c88c386",
        pointwiseCoreCommit:
          "dd36a04960f846e53c2fd948d67b9aa9ddced4f2",
        pointwiseCoreSourceSha256:
          "c801eb209132ed2705a3b7e7b742afd2a6b17855d257938b5df515b6285f3eab",
        pointwiseIngressShaderSha256:
          "750bdf07e86c2cfd639eb1217f11d35408d444c8dc5460ca067a6c6d656f7d16",
        pointwiseAddShaderSha256:
          "9998dbcc049a1795a0fb6df16e6d404f541d5cbc5d486515b869b4337a528eb5",
        pointwiseHarnessCommit:
          "1ab637aa3b174dcf3593beaa56fba6ce8ab4cd44",
        productionConv1dAuditedCoreCommit:
          "82f0fa4b3d5e676ec9dc967c3563dc9650cc59bd",
        productionConv1dAuditedCoreSourceSha256:
          "bdb1ce2732d8617f61132401ab01155163a4f4197e7c7b01eb550b8408553ceb",
        productionConv1dCoreCommit:
          "75f70f12bdb43ae33b9bd37391b7d49be5aa1704",
        productionConv1dCoreSourceSha256:
          "fd14f625e3efeba3277bd9c4e8aa052af92a2b44c078108303173c9bb42a4310",
        productionConv1dHarnessCommit:
          "1320051a2413e1f187143ac0f79958df9218b54f",
        snakeCoreCommit:
          "ae2106c9d5834a3cd5cb836cad484665752230e3",
        snakeCoreSourceSha256:
          "0e0cc8d1974e6f36942a98777e43c6b48b27c00a8cb0d912ff1f510be426601f",
        snakeGraphTopologySha256:
          "ec79060be88fba5d0a2579826f1ca50730dfba16410da09ffc048963f2623bf3",
        snakeHarnessCommit:
          "59d94643def58a96c1cc081c177c91bd308fd83b",
        candidatePackageManifestSha256:
          "5644bcca87678b4f654b9541459355a73ef136c6bb601aa783b6f50fe2f6dba3",
        productionIntegrationPerformed: false,
        webgpuAdapter: { features: ["shader-f16"] },
      },
      protocol: {
        samples: 1,
        correctnessExecutionsPerRange: 2,
        oneOutstandingCommandBuffer: true,
        drainAndRealQueueEmptyTurnAfterEveryExecution: true,
        fullSelectedRangeRawBitComparison: true,
        performanceTimingPerformed: false,
        thermalGateApplied: false,
      },
      correctness: {
        passed: true,
        scope: "completed-package-and-primitive-checkpoints-only",
        convTranspose1dPrimitivePassed: true,
        pointwisePrimitivePassed: true,
        productionConv1dSelectedSyntheticPrimitivePassed: true,
        snakePrimitivePassed: true,
        completeExperimentCorrectnessEstablished: false,
        listeningRequired: true,
        listeningDecision: null,
      },
      evidence: { conclusion: "inconclusive" },
      disposition: { state: "benchmark-only" },
    });

    expect(candidate.completedPrimitiveEvidence.k7Conv1d).toMatchObject({
      passed: true,
      outputsPerExecution: 3_682_122,
      fullProductionRangeCount: 80,
    });
    expect(candidate.completedPrimitiveEvidence.productionConv1d).toEqual({
      allSelectedRangeGuardsPassed: true,
      artifactSha256:
        "58c15ecf91926ed089f7c866217d46a5bc05a5971949ec7ffd2d48d248253593",
      biasedK1OperationCount: 15,
      cancellation: {
        drainedRangeCount: 1,
        laterEncodingPrevented: true,
        laterReadbackPrevented: true,
        laterSubmissionPrevented: true,
        plannedRangeCount: 2,
        realQueueEmptyIdleTurnDelivered: true,
      },
      classification: "selected-synthetic-correctness-only",
      completeFiniteWrites: true,
      deterministicRerunHashes: true,
      distinctBiasedK1ShapeCount: 5,
      exactB256GraphShapesAndSelectedQuantumCoordinatesAuthenticated: true,
      executionCounts: {
        authorityDrains: 20,
        authorityEncodedCommandBuffers: 20,
        authorityQueueEmptyIdleTurns: 20,
        authoritySubmissions: 20,
        productionDrains: 20,
        productionEncodedCommandBuffers: 20,
        productionQueueEmptyIdleTurns: 20,
        productionSubmissions: 20,
      },
      fullSelectedRangeRawU16Comparison: true,
      independentAuditDecision: "GO",
      k1CpuComparisonCountPerArm: 2_392_217,
      k1SelectedCaseCount: 6,
      k7SelectedCaseCount: 4,
      lifecycle: {
        createdBufferCount: 104,
        destroyedBufferCount: 104,
        deviceDestroyed: true,
        heartbeatCoveredCleanup: true,
        idempotentDestroy: true,
        intentionalDeviceLossReason: "destroyed",
        liveBufferCount: 0,
        maximumAnimationFrameGapMilliseconds: 55_119.100000000006,
        maximumLiveBufferCount: 7,
        maximumTimerGapMilliseconds: 24_751.399999976158,
        responsivenessClaim: null,
        responsivenessClaimSupported: false,
        runtimeErrorCount: 0,
        uncapturedErrorCount: 0,
      },
      mismatchCount: 0,
      passed: true,
      productionAuthorityRawU16ComparisonCount: 5_374_258,
      rawArtifactByteLength: 58_686,
      selectedOutputCountPerExecution: 2_687_129,
      selectedProductionGraphCaseCount: 9,
      selectedSyntheticCaseCount: 10,
    });
    expect(candidate.completedPrimitiveEvidence.convTranspose1d).toEqual({
      allFiveCanonicalB256OperationsAuthenticated: true,
      allRangeGuardsPassed: true,
      arithmeticOutputCountPerExecution: 918,
      artifactSha256:
        "3c06879410036b42b70e3da408a1376900316defadf4515cbf58651a68962d68",
      cancellation: {
        drainedRangeCount: 1,
        laterEncodingPrevented: true,
        laterReadbackPrevented: true,
        laterSubmissionPrevented: true,
        plannedRangeCount: 2,
        realQueueEmptyIdleTurnDelivered: true,
      },
      canonicalOperationCount: 5,
      completeFiniteWrites: true,
      deterministicRerunHashes: true,
      exactB256GraphQuantumCount: 322,
      executionCounts: {
        drains: 12,
        encodedCommandBuffers: 12,
        queueEmptyIdleTurns: 12,
        readbackCommandBuffers: 12,
        submissions: 12,
      },
      fullSelectedRangeRawU16Comparison: true,
      independentAuditDecision: "GO",
      lifecycle: {
        createdBufferCount: 54,
        destroyedBufferCount: 54,
        deviceDestroyed: true,
        heartbeatCoveredCleanup: true,
        idempotentDestroy: true,
        intentionalDeviceLossReason: "destroyed",
        liveBufferCount: 0,
        maximumAnimationFrameGapMilliseconds: 50.099999999999454,
        maximumLiveBufferCount: 7,
        maximumTimerGapMilliseconds: 65.30000007152557,
        runtimeErrorCount: 0,
        uncapturedErrorCount: 0,
      },
      mismatchCount: 0,
      passed: true,
      productionOperationRangeCounts: [46, 69, 69, 69, 69],
      rawArtifactByteLength: 66_364,
      selectedCaseCount: 6,
      selectedOutputCountPerExecution: 5_782,
      selectedProductionOutputCountPerExecution: 4_864,
      totalRawBitComparisons: 11_564,
    });
    expect(candidate.completedPrimitiveEvidence.pointwise).toEqual({
      addArithmeticElementCount: 257,
      addCaseCount: 16,
      addProductionElementCountPerExecution: 361_758_720,
      addProductionOperationCount: 15,
      addProductionRangeCount: 348,
      addShapeCount: 5,
      allRangeGuardsPassed: true,
      cancellation: {
        drainedRangeCount: 1,
        laterEncodingPrevented: true,
        laterReadbackPrevented: true,
        laterSubmissionPrevented: true,
        plannedRangeCount: 2,
        realQueueEmptyIdleTurnDelivered: true,
      },
      completeFiniteWrites: true,
      deterministicRerunHashes: true,
      elementComparisonsPerExecution: 361_775_618,
      executionCounts: {
        drains: 706,
        encodedCommandBuffers: 706,
        queueEmptyIdleTurns: 706,
        readbackCommandBuffers: 706,
        submissions: 706,
      },
      independentAuditDecision: "GO",
      ingressCaseCount: 2,
      ingressElementsPerExecution: 16_641,
      lifecycle: {
        createdBufferCount: 749,
        destroyedBufferCount: 749,
        deviceDestroyed: true,
        heartbeatCoveredCleanup: true,
        idempotentDestroy: true,
        intentionalDeviceLossReason: "destroyed",
        liveBufferCount: 0,
        maximumAnimationFrameGapMilliseconds: 100,
        maximumLiveBufferCount: 6,
        maximumTimerGapMilliseconds: 113.20000004768372,
        runtimeErrorCount: 0,
        uncapturedErrorCount: 0,
      },
      mismatchCount: 0,
      rawArtifactByteLength: 723_693,
      totalRangeCount: 353,
      totalRawBitComparisons: 723_551_236,
    });
    expect(candidate.completedPrimitiveEvidence.snake).toEqual({
      allThirtySixB256SnakeOperationsAuthenticated: true,
      allRangeGuardsPassed: true,
      arithmeticCaseCount: 2,
      artifactSha256:
        "214767002480be3a6fe63e72fb5caaf5bff36f9ffb6cbf15d2f80687ad9190d0",
      cancellation: {
        drainedRangeCount: 1,
        laterEncodingPrevented: true,
        laterReadbackPrevented: true,
        laterSubmissionPrevented: true,
        plannedRangeCount: 2,
        realQueueEmptyIdleTurnDelivered: true,
      },
      caseCount: 8,
      completeFiniteWrites: true,
      deterministicRerunHashes: true,
      exactB256GraphQuantumCount: 813,
      exactB256GraphTopologyElementCount: 844_627_968,
      fullSelectedRangeRawU16Comparison: true,
      independentAuditDecision: "GO",
      lifecycle: {
        createdBufferCount: 85,
        destroyedBufferCount: 85,
        deviceDestroyed: true,
        heartbeatCoveredCleanup: true,
        heartbeatLivenessOnly: true,
        idempotentDestroy: true,
        intentionalDeviceLossReason: "destroyed",
        liveBufferCount: 0,
        maximumLiveBufferCount: 8,
        responsivenessClaim: null,
        responsivenessClaimSupported: false,
        runtimeErrorCount: 0,
        uncapturedErrorCount: 0,
      },
      mismatchCount: 0,
      passed: true,
      rawArtifactByteLength: 137_318,
      selectedGraphCaseCount: 6,
      selectedOutputCountPerExecution: 6_816_035,
      selectedRangeCount: 11,
      totalRawBitComparisons: 13_632_070,
    });
    expect(candidate.scope).toEqual({
      completeFp16256WindowExecuted: false,
      completeFp16512WindowExecuted: false,
      listeningGatePerformed: false,
      performanceTimingExecuted: false,
      productionConv1dSelectedSyntheticCorrectnessExecuted: true,
      productionIntegrationPerformed: false,
      productionPackageLoadedByPrimitiveHarnesses: false,
      productionSnakeSelectedRangeCorrectnessExecuted: true,
      responsivenessEstablished: false,
      selectorIntegrationPerformed: false,
      thermalEvidenceCollected: false,
      waveformGatePerformed: false,
    });
    expect(committed.metrics.delta).toEqual({
      convTranspose1dPrimitiveGate: "passed",
      experimentConclusionReady: false,
      performanceClaim: null,
      pointwisePrimitiveGate: "passed",
      productionConv1dSelectedSyntheticPrimitiveGate: "passed",
      productionSelectorClaim: null,
      qualityClaim: null,
      snakePrimitiveGate: "passed",
      thermalClaim: null,
    });
    expect(committed.artifacts).toEqual(RAW_ARTIFACTS.map(([name, sha256]) => ({
      location: `optimization/artifacts/OPT-0011/raw/${name}`,
      sha256,
    })));
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("keeps the record and ledger inconclusive and superseded", async () => {
    const [record, ledger] = await Promise.all([
      readFile(new URL(
        "../optimization/experiments/OPT-0011-fp16-vae-storage-and-window.md",
        import.meta.url,
      ), "utf8"),
      readFile(new URL("../optimization/LEDGER.md", import.meta.url), "utf8"),
    ]);

    expect(record).toContain("- Evidence: `inconclusive`");
    expect(record).toContain("- Disposition: `superseded`");
    expect(record).toContain("723,551,236 complete raw-bit");
    expect(record).toContain("exact 322-quantum B256 graph topology");
    expect(record).toContain("11,564 complete raw-U16 CPU comparisons");
    expect(record).toContain("exact 813-quanta/844,627,968-element B256 topology");
    expect(record).toContain("13,632,070");
    expect(record).toContain(
      "first/rerun raw-U16 CPU comparisons over 6,816,035 selected outputs",
    );
    expect(record).toContain("liveness-only and support no responsiveness claim");
    expect(record).toContain("selected synthetic correctness gate");
    expect(record).toContain("5,374,258 raw-U16 comparisons");
    expect(record).toContain("55,119.100000000006 ms (55.119 s)");
    expect(record).toContain("explicitly do **not** support a responsiveness claim");
    expect(record).toContain("independent receipt audit returned `GO`");
    expect(record).toContain("no timing, thermal, waveform, listening");
    expect(record).toContain("../results/OPT-0011/result.json");
    expect(record).not.toContain("- Result JSON: pending");

    const ledgerRow = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0011 |"),
    );
    expect(ledgerRow).toContain("| inconclusive | superseded |");
    expect(ledgerRow).toContain("723,551,236 first/rerun raw-bit comparisons");
    expect(ledgerRow).toContain("11,564 first/rerun raw-U16 comparisons");
    expect(ledgerRow).toContain("13,632,070 first/rerun raw-U16 comparisons");
    expect(ledgerRow).toContain("844,627,968-element topology");
    expect(ledgerRow).toContain("liveness only and supports no responsiveness claim");
    expect(ledgerRow).toContain("5,374,258 first/rerun raw-U16 comparisons");
    expect(ledgerRow).toContain("55.119 s rAF / 24.751 s timer gaps do not support responsiveness");
    expect(ledgerRow).toContain("[partial result](results/OPT-0011/result.json)");
    expect(ledgerRow).toContain("timing, waveforms, selector integration, and listening remain pending");
  });
});
