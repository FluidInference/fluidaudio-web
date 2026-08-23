import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import {
  assertOptimizationResult,
  benchmarkIdentitySha256,
  createBenchmarkProtocol,
  evaluateThermalGate,
  parseOptimizationResultJson,
  stringifyOptimizationResult,
  summarizeBenchmarkSamples,
  type OptimizationResult,
  type ThermalObservation,
} from "../benchmark/result.js";

const SHA = "a".repeat(64);
const COMMIT = "b".repeat(40);

function result(
  overrides: Partial<OptimizationResult> = {},
): OptimizationResult {
  return {
    schemaVersion: 2,
    experimentId: "OPT-0001",
    hypothesis: "Exact-shape profiling localizes the dominant bottleneck.",
    riskClass: "exact",
    baselineCommit: COMMIT,
    candidateCommit: COMMIT,
    identity: {
      modelManifestSha256: SHA,
      fixtureManifestSha256: SHA,
      productionBundleSha256: SHA,
      benchmarkHarnessCommit: COMMIT,
      browserVersion: "151.0.7922.138",
      browserCommandLine: ["Google Chrome", "--user-data-dir=/tmp/isolated"],
      osBuild: "macOS 26.5.2 (25F84)",
      machineModel: "Mac15,12",
      gpuCores: 10,
      memoryBytes: 16 * 1024 ** 3,
      executionProfile: "reference-bf16-subgroups",
      webgpuAdapter: {
        info: { device: "Apple M3", backend: "Metal" },
        features: ["shader-f16", "subgroups"],
        limits: { maxBufferSize: 256 * 1024 ** 2 },
      },
    },
    protocol: createBenchmarkProtocol({
      samples: 3,
      pairedOrder: ["A", "B", "B", "A"],
      memorySamplingEnabled: true,
    }),
    metrics: {
      baseline: { wallMilliseconds: summarizeBenchmarkSamples([12, 11, 13]) },
      candidate: { wallMilliseconds: summarizeBenchmarkSamples([8, 9, 7]) },
      delta: { medianMilliseconds: -4 },
      logicalGpuPeakBytes: 1024,
      physicalTreePeakBytes: 2048,
    },
    correctness: {
      passed: true,
      oracleManifestSha256: SHA,
      listeningRequired: false,
      listeningDecision: null,
    },
    evidence: {
      conclusion: "positive",
      rationale: "Measurement-only infrastructure passed its declared gate.",
    },
    disposition: {
      state: "integrated",
      rationale: "The measurement infrastructure is part of the repository.",
      revisitWhen: [],
    },
    artifacts: [{ sha256: SHA, location: "artifacts/OPT-0001/raw.json" }],
    ...overrides,
  };
}

describe("benchmark sample aggregation", () => {
  it("retains every sample and computes odd and even medians without mutation", () => {
    const odd = [9, 3, 6];
    expect(summarizeBenchmarkSamples(odd)).toEqual({
      samples: [9, 3, 6],
      sampleCount: 3,
      min: 3,
      median: 6,
      max: 9,
      range: 6,
    });
    expect(odd).toEqual([9, 3, 6]);
    expect(summarizeBenchmarkSamples([1, 7, 3, 5]).median).toBe(4);
  });

  it("rejects empty, negative, and non-finite timing samples", () => {
    expect(() => summarizeBenchmarkSamples([])).toThrow(/must not be empty/);
    expect(() => summarizeBenchmarkSamples([1, -1])).toThrow(/non-negative/);
    expect(() => summarizeBenchmarkSamples([1, Number.NaN])).toThrow(/finite/);
  });
});

describe("thermal pre-gate", () => {
  const protocol = createBenchmarkProtocol({ samples: 3 });

  it("requires the latest continuous nominal window to span 30 seconds", () => {
    const observations: ThermalObservation[] = Array.from(
      { length: 31 },
      (_, index) => ({ monotonicMilliseconds: index * 1_000, state: "nominal" }),
    );
    expect(evaluateThermalGate(observations.slice(0, -1), protocol)).toMatchObject({
      passed: false,
      reason: "nominal-window-too-short",
      continuousNominalMilliseconds: 29_000,
    });
    expect(evaluateThermalGate(observations, protocol)).toMatchObject({
      passed: true,
      reason: "passed",
      requiredNominalMilliseconds: 30_000,
      maximumPollGapMilliseconds: 1_250,
      continuousNominalMilliseconds: 30_000,
    });
  });

  it("resets on a non-nominal state or an over-tolerance polling gap", () => {
    const fairReset: ThermalObservation[] = [
      { monotonicMilliseconds: 0, state: "nominal" },
      { monotonicMilliseconds: 30_000, state: "fair" },
      { monotonicMilliseconds: 31_000, state: "nominal" },
    ];
    expect(evaluateThermalGate(fairReset, protocol)).toMatchObject({
      passed: false,
      reason: "nominal-window-too-short",
      continuousNominalMilliseconds: 0,
      nonNominalObservationCount: 1,
    });

    const gapReset: ThermalObservation[] = [
      { monotonicMilliseconds: 0, state: "nominal" },
      { monotonicMilliseconds: 1_251, state: "nominal" },
    ];
    expect(evaluateThermalGate(gapReset, protocol)).toMatchObject({
      passed: false,
      reason: "poll-gap",
      nominalWindowStartMilliseconds: 1_251,
    });
  });

  it("rejects out-of-order observations and sub-protocol gates", () => {
    expect(() =>
      evaluateThermalGate(
        [
          { monotonicMilliseconds: 1, state: "nominal" },
          { monotonicMilliseconds: 1, state: "nominal" },
        ],
        protocol,
      ),
    ).toThrow(/strictly increasing/);
    expect(() =>
      createBenchmarkProtocol({ samples: 3, thermalGateSeconds: 29 }),
    ).toThrow(/at least 30/);
  });
});

describe("optimization result contract", () => {
  it("validates the committed OPT-0001 result as canonical schema-v2 JSON", async () => {
    const text = await readFile(
      new URL(
        "../optimization/results/OPT-0001/result.json",
        import.meta.url,
      ),
      "utf8",
    );
    const committed = parseOptimizationResultJson(text);
    expect(committed.experimentId).toBe("OPT-0001");
    expect(committed.evidence.conclusion).toBe("positive");
    expect(committed.disposition.state).toBe("integrated");
    expect(committed.candidateCommit).toBe(
      "bbc961121b379b314c8929b16ae37eb292cde3cb",
    );
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("validates the integrated OPT-0002 result and scoped M3 evidence", async () => {
    const text = await readFile(
      new URL(
        "../optimization/results/OPT-0002/result.json",
        import.meta.url,
      ),
      "utf8",
    );
    const committed = parseOptimizationResultJson(text);
    expect(committed).toMatchObject({
      experimentId: "OPT-0002",
      evidence: { conclusion: "positive" },
      disposition: { state: "integrated" },
      baselineCommit: "6281ca0000fa513d001252c4d4aee937bdbb007c",
      candidateCommit: "80f3c8bf550bd16fe18c64992627027972be18a7",
      identity: {
        benchmarkHarnessCommit:
          "80f3c8bf550bd16fe18c64992627027972be18a7",
      },
      metrics: {
        candidate: {
          productionWindow: {
            decoderQuantumCount: 3_942,
            primitiveDispatchCount: 3_988,
            outputBudgetViolationCount: 0,
            convolutionMacBudgetViolationCount: 0,
          },
          responsiveness: {
            authoritativeScope:
              "warmed balanced paired representative-shape intervals",
            maximumAnimationFrameGapMilliseconds: 41.5,
            maximumTimerGapMilliseconds: 11.400000035762787,
            wholeHarnessDiagnostic: {
              maximumAnimationFrameGapMilliseconds: 135.10000000000036,
              maximumTimerGapMilliseconds: 217.69999998807907,
            },
          },
          cancellation: {
            directBrowserProbePerformed: false,
            classification: "inferred-not-directly-measured",
            maximumObservedCandidateDrainMilliseconds: 10.399999976158142,
            inferredBelow500Milliseconds: true,
          },
          fullProductionWindowExecuted: false,
          fullSongExecuted: false,
        },
        delta: {
          decoderQuantumReduction: 58_680,
          decoderQuantumReductionRatio: 15.885844748858448,
          logicalHighWaterIncreased: false,
          toyBitMismatchCount: 0,
        },
      },
      correctness: { passed: true, listeningRequired: false },
      artifacts: [{
        location: "optimization/artifacts/OPT-0002/raw/paired-ab.json",
        sha256:
          "082c5cca3f425c4b51995659468ef6b4de68a051be04b92669a4bba451182e54",
      }],
    });
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("validates the positive integrated OPT-0003 result", async () => {
    const text = await readFile(
      new URL(
        "../optimization/results/OPT-0003/result.json",
        import.meta.url,
      ),
      "utf8",
    );
    const committed = parseOptimizationResultJson(text);
    expect(committed).toMatchObject({
      experimentId: "OPT-0003",
      evidence: { conclusion: "positive" },
      disposition: { state: "integrated", revisitWhen: [] },
      riskClass: "exact",
      baselineCommit: "16e680b2a92459a9ed6d7c4677cc6fc617914222",
      candidateCommit: "0fb193becf1ec359213bbc5f50ad7a9d04c272f8",
      identity: {
        benchmarkHarnessCommit:
          "0fb193becf1ec359213bbc5f50ad7a9d04c272f8",
        integratedBrowser: {
          actualLocalChromeVersion: "151.0.7922.138",
          rawReducedUserAgentChromeVersion: "151.0.0.0",
        },
        integratedPackageIdentityCommit:
          "68d7795c616c1520b1d97ddef9f9d3147ab3973e",
        integratedPackageManifestSha256:
          "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6",
        integratedRuntimeCommit:
          "a4e4ce4d2a2a74b7d9d0b1a05e7fd25343e9d404",
        integrationHarnessCommit:
          "bf6da81647814737e88c8e881da72e04887cba07",
      },
      protocol: {
        samples: 4,
        pairedOrder: ["A", "B", "B", "A", "B", "A", "A", "B"],
      },
      metrics: {
        baseline: {
          aggregateLogicalTflops: { median: 0.48804781571224265 },
        },
        candidate: {
          aggregateLogicalTflops: { median: 0.9251608162146812 },
          maximumCandidateDrainMilliseconds: 58.60000002384186,
          subgroupRoundWins: 4,
          thermallyIndependentPageRuns: 1,
          correctness: {
            productionShapeBitMismatchCounts: [0, 0, 0, 0],
            adversarialPreflight: {
              bitMismatchCount: 0,
              contractedCpuMismatchCount: 0,
              separatelyRoundedCpuMismatchCount: 1_088,
              sourceOrderCancellation: 0.5,
              reassociatedCancellation: 1.5,
            },
          },
          fullProductionIntegrationPerformed: true,
          fullSongExecuted: false,
          productionFullPath: {
            acceptedStage1Candidate: "A",
            combinedIntegratedStackComparison: {
              attributableSolelyToOpt0003: false,
              baselineCooperativeIdleMs: 132_323,
              baselineQueueDrains: 132_326,
              baselineTotalMilliseconds: 720_489.5,
              candidateCooperativeIdleMs: 2_197,
              candidateQueueDrains: 2_200,
              candidateTotalMilliseconds: 64_822.1,
              directionalSpeedupApproximate: 11.115,
              thermalGatePerformed: false,
              timingClassification: "nonthermal-directional",
            },
            durationSeconds: 12,
            executed: true,
            executionProfile: "reference-bf16-subgroups",
            frameCount: 576_000,
            fullSongDurationSeconds: 180,
            fullSongExecuted: false,
            manifestSha256:
              "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6",
            peakTrackedGpuBytes: 3_214_388_992,
            peakTrackedGpuBytesUnchanged: true,
            receiptBodySha256:
              "efdbce5e02cc148873fc33309032097edc28c3189a8d67e83116282ebac9e117",
            schedulingProfile: "cooperative",
            stage1BaselinePeakTrackedGpuBytes: 3_214_388_992,
            stageMilliseconds: {
              conditionEncoder: 7_025.6,
              ditDenoise: 8_066.8,
              ditLoad: 14_239.3,
              textEncoder: 6_145.4,
              vaeDecode: 26_208.3,
              vaeLoad: 3_094.8,
              wavEncode: 25.2,
            },
            wavExactAcceptedStage1Candidate: true,
            wavSha256:
              "d085b6907c9872667412d6dcecfeee47b76c8038eb2bfbec615931b2d7365477",
          },
          productionIntegration: {
            cancellation: {
              classification: "compositional",
              directPackageNativeCancellationPerformed: false,
              existingActualChromeRangeCancellationPassed: true,
              productionBackendLifecycleContractsPassed: true,
            },
            finalLatent: {
              bitMismatchCount: 0,
              byteLength: 33_024,
              elementCount: 8_256,
              finite: true,
              portableSha256:
                "71f98633ada680853ad9ef6ee3fccb40e7664da7f5795f6b0a68142803072bb7",
              subgroupSha256:
                "71f98633ada680853ad9ef6ee3fccb40e7664da7f5795f6b0a68142803072bb7",
            },
            memory: {
              accountedGpuBytes: 3_170_141_952,
              allocatedArenaBytes: 19_191_040,
              boundedCpuBytes: 173_876,
              persistentLayoutReplacementDeltaBytes: 0,
              readbackBufferBytes: 33_024,
              residentWeightBytes: 3_150_917_888,
              simultaneousHeavyweightPhaseCount: 1,
            },
            package: {
              converterRevision: 4,
              ditResidentFileBytes: 3_150_917_888,
              ditTensorPayloadBytes: 3_150_917_760,
              ditTileMajorGemmCount: 271,
              ditTensorCount: 476,
              ditWeightFileCount: 50,
              manifestSha256:
                "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6",
            },
            performanceClaim: false,
            productionKernel: "AceSubgroupGemmKernel",
            runtimeEventCount: 0,
            scheduling: {
              portable: {
                commandBuffersSubmitted: 634,
                compileProgressEvents: 249,
                completedEvaluations: 8,
                cooperativeIdleMs: 633,
                denoiseProgressEvents: 633,
                layerPhysicalProgressEvents: 576,
                queueDrains: 634,
                readbackProgressEvents: 1,
              },
              subgroup: {
                commandBuffersSubmitted: 826,
                compileProgressEvents: 249,
                completedEvaluations: 8,
                cooperativeIdleMs: 825,
                denoiseProgressEvents: 825,
                layerPhysicalProgressEvents: 768,
                queueDrains: 826,
                readbackProgressEvents: 1,
              },
            },
            scope: {
              batch: 1,
              conditionTokens: 1,
              denoisingEvaluations: 8,
              latentFrames: 129,
              layerCount: 24,
              logicalGraphQuanta: 249,
            },
          },
        },
        delta: {
          medianActiveWallSpeedup: 1.8938948759997831,
          productionIntegrationPerformed: true,
        },
      },
      correctness: { passed: true, listeningRequired: false },
    });
    expect(committed.metrics.delta).not.toHaveProperty("acceptanceGates");
    expect(committed.artifacts).toEqual(
      expect.arrayContaining([
        {
          location: "optimization/artifacts/OPT-0003/raw/diagnostic-1.json",
          sha256:
            "7fd4d64a7ba573a3c93bd8b97190dc2ec0fe54bbce234227d1edcbe1dccf8c4b",
        },
        {
          location:
            "optimization/artifacts/OPT-0003/raw/diagnostic-1-thermal.jsonl",
          sha256:
            "0238fb4b3c41d106e920da368142ad28c455c0fd864db0b11bf9212eb4bb6690",
        },
        {
          location:
            "optimization/artifacts/OPT-0003/raw/package-native-dit-integration.json",
          sha256:
            "b15cb76304e65881b6dedd633b0d58dbb155efd873abd1c76339690213f36161",
        },
        {
          location:
            "optimization/artifacts/OPT-0003/raw/post-integration-direct-instrumental-12s-receipt.json",
          sha256:
            "1cb73d06818062972d6b597ee61d40c25be19a7646378065fce721a7ca835d2d",
        },
      ]),
    );
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("validates the scoped positive integrated OPT-0004 result", async () => {
    const text = await readFile(
      new URL(
        "../optimization/results/OPT-0004/result.json",
        import.meta.url,
      ),
      "utf8",
    );
    const committed = parseOptimizationResultJson(text);
    expect(committed).toMatchObject({
      experimentId: "OPT-0004",
      evidence: { conclusion: "positive" },
      disposition: { state: "integrated" },
      riskClass: "exact",
      baselineCommit: "4c43b1434fa286251538fc44098b691b05d15da2",
      candidateCommit: "48148ad1c791d26653d95418f55720240bafddff",
      identity: {
        benchmarkHarnessCommit:
          "eab519804278a0ff1a69616efed0191254983471",
        productionIntegrationCommit:
          "48148ad1c791d26653d95418f55720240bafddff",
      },
      protocol: {
        samples: 4,
        pairedOrder: ["A", "B", "B", "A", "B", "A", "A", "B"],
      },
      metrics: {
        baseline: {
          representativeActiveWallMilliseconds: {
            median: 632.0000001192093,
          },
        },
        candidate: {
          benchmarkScope: expect.stringContaining("65,536-frame"),
          representative: {
            shape: {
              inputFrames: 65_536,
              inputChannels: 128,
              outputChannels: 128,
              kernelSize: 7,
            },
            outputRangeCount: 32,
          },
          representativeActiveWallMilliseconds: {
            median: 211.5000001192093,
          },
          maximumSingleDrainMilliseconds: 20,
          tiledRoundWins: 4,
          thermallyIndependentPageRuns: 1,
          responsiveness: {
            maximumAnimationFrameGapMilliseconds: 7.400000000000546,
            maximumTimerGapMilliseconds: 12.100000023841858,
          },
          correctness: {
            manageablePreflightCaseBitMismatchCounts: [0, 0, 0],
            productionRepresentativeComparedElements: 786_432,
            productionRepresentativeBitMismatchCount: 0,
            productionRepresentativeCpuSentinelsBitExact: true,
            contractedDiscriminantBits: "0xbe7d3830",
            sourceOrderCancellation: 0.5,
            reassociatedCancellation: 1.5,
          },
          cancellation: {
            rangesSubmitted: 1,
            activeRangeDrained: true,
            laterRangeSubmissionPrevented: true,
          },
          externalThermalCoverage: {
            observationCount: 92,
            nonNominalObservationCount: 0,
          },
          completeProductionOperationExecuted: false,
          fullVaeWindowExecuted: false,
          fullSongExecuted: false,
          productionIntegrationPerformed: true,
          productionIntegration: {
            commit: "48148ad1c791d26653d95418f55720240bafddff",
            requiredWorkgroupStorageBytes: 16_384,
            canonical256FrameGraph: {
              tiledOperationLabels: [
                "conv1",
                "block-3-res-1-conv1",
                "block-4-res-1-conv1",
                "conv2",
              ],
              tiledQuantumCount: 365,
              fallbackOperationFamilies: [
                "K1",
                "K7 dilation 3",
                "K7 dilation 9",
              ],
            },
            fifoScheduling: {
              oneCommandBufferOutstanding: true,
              queueDrainAfterEveryQuantum: true,
              realQueueEmptyIntervalAfterEveryQuantum: true,
            },
            existingBackendCancellationTestsPassed: true,
          },
          integratedDecoderCorrectness: {
            finalOutputValuesChecked: 12,
            finalOutputBitMismatchCount: 0,
            maximumCpuAbsoluteError: 0.000885009765625,
            tiledOperationLabels: [
              "conv1",
              "block-0-res-1-conv1",
              "conv2",
            ],
            portableFallbackOperationLabels: [
              "block-0-res-1-conv2",
              "block-0-res-2-conv1",
              "block-0-res-2-conv2",
              "block-0-res-3-conv1",
              "block-0-res-3-conv2",
            ],
            actualChromePerformed: true,
          },
        },
        delta: {
          medianActiveWallSpeedup: 2.9881796679101207,
          scalarActiveWallDrift: {
            firstMilliseconds: 721.4999999403954,
            lastMilliseconds: 456.20000010728836,
          },
          tiledRoundWins: 4,
          worstPairedRoundSpeedup: 2.1198884741038038,
          productionIntegrationPerformed: true,
        },
      },
      correctness: { passed: true, listeningRequired: false },
      artifacts: [
        {
          location:
            "optimization/artifacts/OPT-0004/raw/representative-paired-ab.json",
          sha256:
            "6582588efa752d4203238a329be10ad4d36a95503c72db133c451e63b31c211c",
        },
        {
          location:
            "optimization/artifacts/OPT-0004/raw/representative-thermal.jsonl",
          sha256:
            "317ca8806148f9ce375c26d14e98dc512a12d46fb1b4da80badc00965b32999f",
        },
        {
          location:
            "optimization/artifacts/OPT-0004/raw/integrated-decoder-correctness.json",
          sha256:
            "433473d1863359da83d4917dc8745076885491df8f1c23029aebda7866fdf9e1",
        },
      ],
    });
    expect(committed.metrics.delta).not.toHaveProperty("acceptanceGates");
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("validates the positive integrated OPT-0005 result", async () => {
    const text = await readFile(
      new URL(
        "../optimization/results/OPT-0005/result.json",
        import.meta.url,
      ),
      "utf8",
    );
    const committed = parseOptimizationResultJson(text);
    expect(committed).toMatchObject({
      experimentId: "OPT-0005",
      evidence: { conclusion: "positive" },
      disposition: { state: "integrated" },
      riskClass: "exact",
      baselineCommit: "e90f22741a1a81564a70bf73299f64157799e6c1",
      candidateCommit: "31e8ef7f385b4c3b21180b356ca2d89ec00a7099",
      identity: {
        benchmarkHarnessCommit:
          "75c810783668b2013e69c4852e1bf55349d2bbc3",
        productionIntegrationCommit:
          "31e8ef7f385b4c3b21180b356ca2d89ec00a7099",
      },
      protocol: {
        samples: 4,
        pairedOrder: ["A", "B", "B", "A", "B", "A", "A", "B"],
      },
      metrics: {
        baseline: {
          block0Dilation1ActiveWallMilliseconds: {
            median: 691.400000244379,
          },
          block0Dilation1WallMilliseconds: { median: 787.5 },
        },
        candidate: {
          block0Dilation1: {
            completeProductionOperationExecuted: true,
            outputRangeCount: 80,
            validMultiplyAdds: 18_777_899_008,
            activeWallMilliseconds: { median: 401.14999997615814 },
            wallMilliseconds: { median: 495.0499999821186 },
            chunkedRoundWins: 4,
            correctness: {
              comparedElementCount: 98_304,
              bitMismatchCount: 0,
              cpuSentinelsBitExact: true,
              allFullOperationOutputsReadBack: false,
            },
          },
          dilation9Range: {
            selectedRangeIndex: 40,
            measuredCommandBufferCount: 1,
            medianActiveWallSpeedup: 1.8913043476852442,
            chunkedRoundWins: 4,
            correctness: {
              comparedElementCount: 32_768,
              bitMismatchCount: 0,
            },
            completeProductionOperationExecuted: false,
          },
          overlapDilation1: {
            classification: "timing-ambiguous-retain-opt0004-priority",
            scalarVersusChunkedComparedElementCount: 262_144,
            scalarVersusChunkedBitMismatchCount: 0,
            scalarVersusOpt0004ComparedElementCount: 262_144,
            scalarVersusOpt0004BitMismatchCount: 0,
            timingUsedForSelectorDecision: false,
          },
          manageableCorrectnessPreflight: {
            fullDomainBitMismatchCounts: [0, 0, 0, 0],
            cpuSentinelsBitExact: [true, true, true, true],
          },
          cancellation: {
            pageRunCount: 4,
            activeRangeDrainedOnEveryRun: true,
            laterRangeSubmissionPreventedOnEveryRun: true,
          },
          externalThermalCoverage: {
            spansPreGateRunAndPostRun: true,
            runs: [
              {
                id: "screen-d1",
                durationMilliseconds: 146_742.46262502857,
                observationCount: 147,
                nonNominalObservationCount: 0,
              },
              {
                id: "block0-d1",
                durationMilliseconds: 71_528.81466702092,
                observationCount: 72,
                nonNominalObservationCount: 0,
              },
              {
                id: "screen-d9",
                durationMilliseconds: 63_863.770207972266,
                observationCount: 64,
                nonNominalObservationCount: 0,
              },
              {
                id: "overlap-d1",
                durationMilliseconds: 84_803.8889580057,
                observationCount: 85,
                nonNominalObservationCount: 0,
              },
            ],
          },
          integratedDecoderCorrectness: {
            actualChromePerformed: true,
            decoderInputChannels: 136,
            decoderChannels: 128,
            finalOutputValuesChecked: 12,
            finalOutputBitMismatchCount: 0,
            maximumCpuAbsoluteError: 0.000354766845703125,
            tiledOperationLabels: ["block-0-res-1-conv1", "conv2"],
            channelChunkedOperationLabels: [
              "conv1",
              "block-0-res-2-conv1",
              "block-0-res-3-conv1",
            ],
            portableOperationLabels: [
              "block-0-res-1-conv2",
              "block-0-res-2-conv2",
              "block-0-res-3-conv2",
            ],
            tiledQuantumCount: 7,
            channelChunkedQuantumCount: 15,
            portableQuantumCount: 18,
            forcedPortableQuantumCount: 40,
            fallbackReasons: {
              "tiled:unsupported-math;channel-chunked:unsupported-math": 18,
            },
            forcedPortableFallbackReasons: { "profile-portable": 40 },
          },
          productionIntegration: {
            commit: "31e8ef7f385b4c3b21180b356ca2d89ec00a7099",
            requiredWorkgroupStorageBytes: 16_384,
            canonical256FrameGraph: {
              tiledOperationCount: 4,
              tiledQuantumCount: 365,
              channelChunkedOperationCount: 13,
              channelChunkedQuantumCount: 1_680,
              portableOperationCount: 15,
              portableQuantumCount: 414,
              totalConv1dQuantumCount: 2_459,
            },
            fifoScheduling: {
              oneCommandBufferOutstanding: true,
              queueDrainAfterEveryQuantum: true,
              realQueueEmptyIntervalAfterEveryQuantum: true,
            },
          },
          productionIntegrationPerformed: true,
          fullVaeWindowExecuted: false,
          fullSongExecuted: false,
        },
        delta: {
          block0Dilation1MedianActiveWallSpeedup: 1.723544809386692,
          block0Dilation1MedianWallIncludingIdleSpeedup:
            1.5907484093090491,
          dilation9RangeMedianActiveWallSpeedup: 1.8913043476852442,
          productionIntegrationPerformed: true,
        },
      },
      correctness: { passed: true, listeningRequired: false },
    });
    expect(committed.metrics.delta).not.toHaveProperty("acceptanceGates");
    expect(committed.artifacts).toEqual([
      {
        location: "optimization/artifacts/OPT-0005/raw/screen-d1.json",
        sha256:
          "ef177d738f0dced36369a10d29f01e373a55911e46ae3db6cc549a03471d0b6d",
      },
      {
        location:
          "optimization/artifacts/OPT-0005/raw/screen-d1-thermal.jsonl",
        sha256:
          "79e56d163f3e8f37885186cb40b451e7cc3fa653ba3ab0223afa27f9ff7b7e73",
      },
      {
        location: "optimization/artifacts/OPT-0005/raw/block0-d1.json",
        sha256:
          "94f9a88498a242916ccf16cac7cc23511edb09e767d1eadad5bdb8c315b34e96",
      },
      {
        location:
          "optimization/artifacts/OPT-0005/raw/block0-d1-thermal.jsonl",
        sha256:
          "49eecf03e50ed1b9423fed67d4237a51f6c176eca105ce4efd16fded0e384335",
      },
      {
        location: "optimization/artifacts/OPT-0005/raw/screen-d9.json",
        sha256:
          "ed88ca5d5d54094b03d4609e91aad52d0af2b0adf42a55c59fdcc3875941b5ce",
      },
      {
        location:
          "optimization/artifacts/OPT-0005/raw/screen-d9-thermal.jsonl",
        sha256:
          "afdb24312cc1009a4a7ee62ce15a47749875aec8a3af74841b0deda6ec8e8158",
      },
      {
        location: "optimization/artifacts/OPT-0005/raw/overlap-d1.json",
        sha256:
          "990549573ca3b4c357d35d410cd7c82409c1a7b9378d39620a121ad1f02a0553",
      },
      {
        location:
          "optimization/artifacts/OPT-0005/raw/overlap-d1-thermal.jsonl",
        sha256:
          "3c6c61e1029e4d6d427f2c0d8b3e0bc5b3d597c7d228439e9d9038ab1eb1392a",
      },
      {
        location:
          "optimization/artifacts/OPT-0005/raw/integrated-decoder-correctness.json",
        sha256:
          "26c4f9f35eb8e255cd5a8fd3972d7fc46d7543bfd030f732a44a9fc05210d643",
      },
    ]);
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("round-trips deterministic compact JSON with complete provenance", () => {
    const candidate = result();
    expect(() => assertOptimizationResult(candidate)).not.toThrow();
    const encoded = stringifyOptimizationResult(candidate);
    expect(encoded).not.toContain("\n");
    expect(encoded.indexOf('"artifacts"')).toBeLessThan(
      encoded.indexOf('"baselineCommit"'),
    );
    expect(parseOptimizationResultJson(encoded)).toEqual(candidate);
    expect(stringifyOptimizationResult(parseOptimizationResultJson(encoded))).toBe(
      encoded,
    );
    expect(benchmarkIdentitySha256(candidate.identity)).toBe(
      createHash("sha256")
        .update(
          encoded.slice(
            encoded.indexOf('"identity":') + '"identity":'.length,
            encoded.indexOf(',"metrics":'),
          ),
        )
        .digest("hex"),
    );
  });

  it("rejects malformed identities, invalid dispositions, and false integrated gates", () => {
    expect(() =>
      assertOptimizationResult({
        ...result(),
        identity: { ...result().identity, modelManifestSha256: "mutable" },
      }),
    ).toThrow(/modelManifestSha256/);
    expect(() =>
      assertOptimizationResult({
        ...result(),
        disposition: { ...result().disposition, state: "accepted" },
      }),
    ).toThrow(/disposition.state/);
    expect(() =>
      assertOptimizationResult({
        ...result(),
        correctness: { passed: false },
      }),
    ).toThrow(/must pass/);
    expect(() => assertOptimizationResult({ ...result(), hidden: true })).toThrow(
      /unknown field/,
    );
  });

  it("requires an explicit listening decision for an integrated change", () => {
    expect(() =>
      assertOptimizationResult({
        ...result(),
        correctness: {
          passed: true,
          listeningRequired: true,
          listeningDecision: null,
        },
      }),
    ).toThrow(/must record a listening decision/);
  });

  it("rejects duplicate JSON keys and non-JSON extension data", () => {
    const encoded = stringifyOptimizationResult(result());
    expect(() =>
      parseOptimizationResultJson(encoded.replace('"schemaVersion":2', '"schemaVersion":2,"schemaVersion":2')),
    ).toThrow(/duplicate object key/);
    expect(() =>
      assertOptimizationResult({
        ...result(),
        identity: { ...result().identity, extension: undefined },
      }),
    ).toThrow(/extension/);
  });
});
