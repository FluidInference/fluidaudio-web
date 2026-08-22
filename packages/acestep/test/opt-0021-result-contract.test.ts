import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";
import { aceOpt0009DenseGemmWgsl } from
  "../src/webgpu/kernels/dit-dense-fp16.js";
import { aceOpt0019DenseCooperativePanelsWgsl } from
  "../src/webgpu/kernels/dit-dense-fp16-cooperative-panels.js";
import { aceOpt0021DenseCooperativeVec4PanelsWgsl } from
  "../src/webgpu/kernels/dit-dense-fp16-cooperative-vec4-panels.js";
import {
  buildOpt0021FixtureDeclaration,
  buildOpt0021ShapeSpecs,
  fillOpt0021ActivationFixture,
  fillOpt0021PackedWeightFixture,
} from "./browser/opt-0021-dit-dense-vec4-panels.js";

type Arm = "A" | "B" | "C";
type OutputArm = "A" | "B" | "CFirst" | "CRerun";

interface DispatchSample {
  readonly roundIndex: number;
  readonly rotatedShapeOrder: readonly number[];
  readonly shapeIndex: number;
  readonly shapeId: string;
  readonly armOrder: readonly Arm[];
  readonly armPosition: number;
  readonly arm: Arm;
  readonly submitAtPerformanceMilliseconds: number;
  readonly fenceAtPerformanceMilliseconds: number;
  readonly wallMilliseconds: number;
  readonly commandBufferCount: number;
  readonly dispatchCount: number;
  readonly matchingQueueCompletionFenceCount: number;
}

interface TimingStratum {
  readonly id: string;
  readonly productionMultiplicity: number;
  readonly feedForwardMultiplicity: number;
  readonly samples: Readonly<Record<Arm, readonly number[]>>;
  readonly medians: Readonly<Record<Arm, number>>;
  readonly cFasterThanB: boolean;
  readonly bToCSpeedup: number;
  readonly aToCSpeedup: number;
  readonly aToCSavingMilliseconds: number;
}

interface CorrectnessCase {
  readonly id: string;
  readonly outputU32Count: number;
  readonly executionOrder: readonly string[];
  readonly executionCount: number;
  readonly exactComparisonCount: number;
  readonly comparisonsPerOutputWord: number;
  readonly comparedU32Count: number;
  readonly mismatchCount: number;
  readonly hashes: Readonly<Record<OutputArm, string>>;
  readonly comparisons: Readonly<Record<string, {
    readonly comparedU32Count: number;
    readonly mismatchCount: number;
    readonly firstMismatch: unknown;
    readonly rawU32Exact: boolean;
  }>>;
  readonly aVersusBRawU32Exact: boolean;
  readonly aVersusCFirstRawU32Exact: boolean;
  readonly cFirstVersusRerunRawU32Exact: boolean;
  readonly allFourExecutionsShareShapeHash: boolean;
  readonly nonFiniteCount: number;
  readonly qNaNPrefillCount: number;
  readonly prefixCanaryIntact: boolean;
  readonly suffixCanaryIntact: boolean;
  readonly tailRowWritten: boolean;
  readonly finiteCompleteWrites: boolean;
  readonly passed: boolean;
  readonly fixture: {
    readonly generatorVersion: string;
    readonly activationSeed: number;
    readonly weightSeed: number;
    readonly activationSha256: string;
    readonly packedWeightSha256: string;
    readonly activationBytes: number;
    readonly packedWeightBytes: number;
  };
}

interface Opt0021Receipt {
  readonly schema: string;
  readonly experimentId: string;
  readonly status: string;
  readonly classification: string;
  readonly recordedAt: string;
  readonly identity: {
    readonly registrationCommit: string;
    readonly currentCoreSourceSha256: string;
    readonly exactCoreSourceSha256: string;
    readonly vec4CoreSourceSha256: string;
    readonly exactKernelSetId: string;
    readonly vec4KernelSetId: string;
    readonly fixtureManifestProvenanceSha256: string;
    readonly syntheticFixtureDeclarationSha256: string;
    readonly payloadAuthority: {
      readonly modelPackageLoaded: boolean;
      readonly syntheticFixture: boolean;
      readonly persistentWeightLayout: string;
      readonly converterOrManifestChange: boolean;
      readonly productionRuntimeSelected: boolean;
    };
    readonly generatedShaders: readonly {
      readonly id: string;
      readonly A: string;
      readonly B: string;
      readonly C: string;
    }[];
  };
  readonly fixture: {
    readonly version: string;
    readonly shapes: readonly {
      readonly id: string;
      readonly fixture: {
        readonly ordinal: number;
        readonly activationSeed: number;
        readonly weightSeed: number;
        readonly activationSha256: string;
        readonly packedWeightSha256: string;
      };
    }[];
  };
  readonly environment: {
    readonly userAgent: string;
    readonly adapterInfo: Readonly<Record<string, unknown>>;
    readonly features: readonly string[];
    readonly limits: Readonly<Record<string, number>>;
  };
  readonly preparation: {
    readonly completedAtEpochMilliseconds: number;
    readonly memoryHighWaterBytes: number;
  };
  readonly protocol: {
    readonly thermal: {
      readonly source: string;
      readonly startedAtEpochMilliseconds: number;
      readonly completedAtEpochMilliseconds: number;
      readonly durationMilliseconds: number;
      readonly observationCount: number;
      readonly pollMilliseconds: number;
      readonly maximumPollGapMilliseconds: number;
      readonly nonNominalObservationCount: number;
      readonly launchDelayMilliseconds: number;
    };
    readonly fullOutputCorrectnessCompletedBeforeThermalGate: boolean;
    readonly allThreeKernelsCompiledAndWarmedBeforeThermalGate: boolean;
    readonly timingRounds: number;
    readonly continuousExternalThermalTraceRequiredThroughCleanup: boolean;
    readonly thermalTraceRequiredThroughEpochMilliseconds: number;
    readonly oneThirtySecondNominalGate: boolean;
    readonly unchangedThermalRetryPerformed: boolean;
  };
  readonly correctness: {
    readonly shapeCount: number;
    readonly executionCount: number;
    readonly exactComparisonCount: number;
    readonly comparisonsPerOutputWord: number;
    readonly outputU32Count: number;
    readonly comparedU32Count: number;
    readonly mismatchCount: number;
    readonly aVersusBRawU32Exact: boolean;
    readonly aVersusCFirstRawU32Exact: boolean;
    readonly cFirstVersusRerunRawU32Exact: boolean;
    readonly allFourExecutionsShareShapeHash: boolean;
    readonly qNaNPrefillCompleteWrites: boolean;
    readonly canariesUntouched: boolean;
    readonly finiteOutputs: boolean;
    readonly tailRowsWritten: boolean;
    readonly uncapturedGpuErrorCount: number;
    readonly aggregateOutputHashManifestSha256: {
      readonly algorithm: string;
      readonly shapeOrder: readonly string[];
      readonly hashes: Readonly<Record<OutputArm, string>>;
      readonly allFourExecutionsShareAggregateHash: boolean;
    };
    readonly cases: readonly CorrectnessCase[];
    readonly passed: boolean;
  };
  readonly timing: {
    readonly samplesPerArmPerShape: number;
    readonly completeDense: {
      readonly milliseconds: Readonly<Record<Arm, number>>;
      readonly bToCSpeedup: number;
      readonly bToCThreshold: number;
      readonly aToCSavingMilliseconds: number;
      readonly aToCSavingThresholdMilliseconds: number;
    };
    readonly feedForward: {
      readonly milliseconds: Readonly<Record<Arm, number>>;
      readonly bToCSpeedup: number;
      readonly aToCSavingMilliseconds: number;
    };
    readonly everyShapeCFasterThanB: boolean;
    readonly strata: readonly TimingStratum[];
    readonly passed: boolean;
    readonly decision: string;
    readonly timingStartedAtEpochMilliseconds: number;
    readonly timingCompletedAtEpochMilliseconds: number;
    readonly dispatchSampleCount: number;
    readonly dispatchSamples: readonly DispatchSample[];
  };
  readonly memory: {
    readonly createdBufferCount: number;
    readonly destroyedBufferCount: number;
    readonly liveBufferCount: number;
    readonly liveBytes: number;
    readonly maximumLiveBytes: number;
  };
  readonly cleanup: {
    readonly startedAtEpochMilliseconds: number;
    readonly completedAtEpochMilliseconds: number;
    readonly firstCall: {
      readonly createdBufferCount: number;
      readonly destroyedBufferCount: number;
      readonly liveBufferCount: number;
      readonly liveBytes: number;
      readonly maximumLiveBytes: number;
      readonly idempotent: boolean;
      readonly repeatedCall: boolean;
      readonly drainBeforeRelease: boolean;
      readonly deviceDestroyed: boolean;
    };
    readonly secondCall: {
      readonly createdBufferCount: number;
      readonly destroyedBufferCount: number;
      readonly liveBufferCount: number;
      readonly liveBytes: number;
      readonly maximumLiveBytes: number;
      readonly idempotent: boolean;
      readonly repeatedCall: boolean;
      readonly drainBeforeRelease: boolean;
    };
    readonly idempotent: boolean;
    readonly zeroLiveResources: boolean;
  };
  readonly decision: {
    readonly disposition: string;
    readonly packageNativeLayerGateAuthorized: boolean;
    readonly productionIntegrationAuthorized: boolean;
    readonly c98TrajectoryRunAuthorized: boolean;
    readonly m2250ProductRunAuthorized: boolean;
  };
}

interface ThermalObservation {
  readonly index: number;
  readonly epochMilliseconds: number;
  readonly level: number;
}

interface ResultView {
  readonly identity: {
    readonly currentCoreSourceSha256: string;
    readonly exactCoreSourceSha256: string;
    readonly candidateCoreSourceSha256: string;
    readonly candidateCoreContractSha256: string;
    readonly harnessSourceSha256: string;
    readonly harnessHtmlSha256: string;
    readonly preExecutionContractSha256: string;
    readonly syntheticFixtureDeclarationSha256: string;
    readonly fixtureHashes: Readonly<Record<string, {
      readonly activationSha256: string;
      readonly packedWeightSha256: string;
    }>>;
    readonly generatedShaderHashes: readonly {
      readonly id: string;
      readonly A: string;
      readonly B: string;
      readonly C: string;
    }[];
    readonly rawArtifactBytes: {
      readonly receipt: number;
      readonly thermal: number;
    };
    readonly productionPackageLoaded: boolean;
    readonly productionRuntimeSelected: boolean;
    readonly productionIntegrationPerformed: boolean;
  };
  readonly protocol: {
    readonly authoritativeTimedRunCount: number;
    readonly dispatchSampleCount: number;
    readonly unchangedThermalRetryPerformed: boolean;
    readonly thermal: {
      readonly observationCount: number;
      readonly maximumPollGapMilliseconds: number;
      readonly totalTraceObservationCount: number;
      readonly totalTraceMaximumPollGapMilliseconds: number;
      readonly totalTraceLastEpochMilliseconds: number;
      readonly cleanupCompletedAtEpochMilliseconds: number;
      readonly postCleanupCoverageMilliseconds: number;
    };
  };
  readonly metrics: {
    readonly baseline: {
      readonly completeDenseMilliseconds: number;
      readonly feedForwardMilliseconds: number;
      readonly shapeMedianMilliseconds: Readonly<Record<string, number>>;
    };
    readonly candidate: {
      readonly completeDenseMilliseconds: number;
      readonly feedForwardMilliseconds: number;
      readonly shapeMedianMilliseconds: Readonly<Record<string, number>>;
      readonly everyShapeFasterThanBaseline: boolean;
      readonly qualified: boolean;
      readonly packageNativeLayerGatePerformed: boolean;
      readonly productionIntegrationPerformed: boolean;
      readonly c98TrajectoryRunPerformed: boolean;
      readonly m2250ProductRunPerformed: boolean;
    };
    readonly delta: {
      readonly currentArmA: {
        readonly completeDenseMilliseconds: number;
        readonly feedForwardMilliseconds: number;
        readonly shapeMedianMilliseconds: Readonly<Record<string, number>>;
      };
      readonly perShapeCFasterThanB: Readonly<Record<string, boolean>>;
      readonly everyShapeFasterConditionPassed: boolean;
      readonly bToCCompleteSpeedup: number;
      readonly bToCCompleteThreshold: number;
      readonly bToCCompleteConditionPassed: boolean;
      readonly aToCCompleteSavingMilliseconds: number;
      readonly aToCCompleteSavingThresholdMilliseconds: number;
      readonly aToCCompleteSavingConditionPassed: boolean;
      readonly projectedDenseSavingAcross192LayerEvaluationsMilliseconds: number;
      readonly projectedDenseSavingThresholdMilliseconds: number;
      readonly bToCFeedForwardSpeedup: number;
      readonly stopRuleFired: boolean;
      readonly observedIntegratedDitSpeedup: null;
      readonly observed180SecondProductSpeedup: null;
      readonly under60SecondProjection: null;
    };
  };
  readonly correctness: {
    readonly primitive: {
      readonly comparedU32Count: number;
      readonly mismatchCount: number;
      readonly aggregateOutputHashManifestSha256:
        Readonly<Record<OutputArm, string>>;
      readonly caseOutputSha256: Readonly<Record<string, string>>;
    };
    readonly cleanup: {
      readonly createdBufferCount: number;
      readonly destroyedBufferCount: number;
      readonly liveBufferCount: number;
      readonly liveBytes: number;
      readonly maximumLiveBytes: number;
      readonly idempotent: boolean;
      readonly drainBeforeRelease: boolean;
      readonly deviceDestroyed: boolean;
    };
  };
}

const RESULT_URL = new URL(
  "../optimization/results/OPT-0021/result.json",
  import.meta.url,
);
const RECEIPT_URL = new URL(
  "../optimization/artifacts/OPT-0021/raw/dense-vec4-panels-abc.json",
  import.meta.url,
);
const THERMAL_URL = new URL(
  "../optimization/artifacts/OPT-0021/raw/dense-vec4-panels-thermal.jsonl",
  import.meta.url,
);
const RECORD_URL = new URL(
  "../optimization/experiments/OPT-0021-dit-dense-vec4-panels.md",
  import.meta.url,
);
const LEDGER_URL = new URL("../optimization/LEDGER.md", import.meta.url);

const SOURCE_IDENTITIES = [
  [
    "../src/webgpu/kernels/dit-dense-fp16.ts",
    "a238f67da07c6ba1097da9d9e9e97960ae97d2e1d5c129fcbabf69e962cbb6b3",
  ],
  [
    "../src/webgpu/kernels/dit-dense-fp16-cooperative-panels.ts",
    "b5dad12724882d3fc942c7df7b10c7b7b89a4bed595125ff11a5905c03152a37",
  ],
  [
    "../src/webgpu/kernels/dit-dense-fp16-cooperative-vec4-panels.ts",
    "2229c55f8b7fe66d3770ef7683de68322632d17749fe2d3085d5d46dcdc22df1",
  ],
  [
    "browser/opt-0021-dit-dense-vec4-panels.ts",
    "e0a019be08bc6222594de5e5083cc8cef1395d8395ed003661bc1e71c8cea055",
  ],
  [
    "browser/opt-0021-dit-dense-vec4-panels.html",
    "84fb23733bc30a6d37d0e40418f9fee22a37c0a87c13be3e15e81c75f1690ffa",
  ],
  [
    "opt-0021-dit-dense-vec4-panels-contract.test.ts",
    "f93bfea55377acce21e264e120a45bca5b2a867fa54a75b6a9fe0dd098ac9563",
  ],
  [
    "dit-dense-fp16-cooperative-vec4-panels.test.ts",
    "61505a4c520b9107a7f529192f58013e48ac6c95317fb98dbd98abfffd0a5806",
  ],
] as const;

const EXPECTED_ORDERS = [
  ["A", "B", "C"],
  ["A", "C", "B"],
  ["B", "A", "C"],
  ["B", "C", "A"],
  ["C", "A", "B"],
  ["C", "B", "A"],
] as const;
const SHAPE_IDS = ["h-h", "h-1024", "h-6144", "6144-h"] as const;
const ARMS = ["A", "B", "C"] as const;

const sha256 = (bytes: Uint8Array | string): string =>
  createHash("sha256").update(bytes).digest("hex");

describe("OPT-0021 optimization result", () => {
  it("commits canonical exact-but-negative evidence and every stop boundary", async () => {
    const text = await readFile(RESULT_URL, "utf8");
    const committed = parseOptimizationResultJson(text);
    const view = committed as unknown as ResultView;

    expect(committed).toMatchObject({
      experimentId: "OPT-0021",
      riskClass: "exact",
      baselineCommit: "bc2484096ff5622bc21d03aad80d08f12719989e",
      candidateCommit: "fa446366fa404e5ce00cf7350c206f7a63ba791b",
      evidence: { conclusion: "negative" },
      disposition: { state: "abandoned" },
      identity: {
        browserVersion:
          "Google Chrome 151.0.7922.138; reduced user agent 151.0.0.0",
        osBuild: "macOS 26.5.2 build 25F84",
        benchmarkHarnessCommit:
          "fa446366fa404e5ce00cf7350c206f7a63ba791b",
        productionPackageLoaded: false,
        productionRuntimeSelected: false,
        productionIntegrationPerformed: false,
      },
      correctness: {
        passed: true,
        listeningRequired: false,
        listeningDecision: null,
      },
    });
    expect(committed.artifacts).toEqual([
      {
        location:
          "optimization/artifacts/OPT-0021/raw/dense-vec4-panels-abc.json",
        sha256:
          "607e3173586ed46023d1f55fb7060bed755ab273930701e4010c7e64c15a08b6",
      },
      {
        location:
          "optimization/artifacts/OPT-0021/raw/dense-vec4-panels-thermal.jsonl",
        sha256:
          "fe482a89246148be4fe76af169f472b4fe4af42f5ee74d7a5925a7f4d12f761f",
      },
    ]);
    expect(view.correctness.primitive).toMatchObject({
      comparedU32Count: 76_032_000,
      mismatchCount: 0,
    });
    expect(view.metrics.delta).toMatchObject({
      everyShapeFasterConditionPassed: false,
      bToCCompleteSpeedup: 0.991159923845376,
      bToCCompleteThreshold: 1.075,
      bToCCompleteConditionPassed: false,
      aToCCompleteSavingMilliseconds: 40.94999998807907,
      aToCCompleteSavingThresholdMilliseconds: 52.0834,
      aToCCompleteSavingConditionPassed: false,
      stopRuleFired: true,
      observedIntegratedDitSpeedup: null,
      observed180SecondProductSpeedup: null,
      under60SecondProjection: null,
    });
    expect(view.metrics.candidate).toMatchObject({
      everyShapeFasterThanBaseline: false,
      qualified: false,
      packageNativeLayerGatePerformed: false,
      productionIntegrationPerformed: false,
      c98TrajectoryRunPerformed: false,
      m2250ProductRunPerformed: false,
    });
    expect(view.protocol).toMatchObject({
      authoritativeTimedRunCount: 1,
      dispatchSampleCount: 72,
      unchangedThermalRetryPerformed: false,
    });
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("recomputes all 72 samples, medians, scores, thresholds, and exactness", async () => {
    const [resultText, receiptBytes] = await Promise.all([
      readFile(RESULT_URL, "utf8"),
      readFile(RECEIPT_URL),
    ]);
    expect(receiptBytes.byteLength).toBe(72_064);
    expect(sha256(receiptBytes)).toBe(
      "607e3173586ed46023d1f55fb7060bed755ab273930701e4010c7e64c15a08b6",
    );
    const result = parseOptimizationResultJson(resultText) as unknown as ResultView;
    const receipt = JSON.parse(receiptBytes.toString("utf8")) as Opt0021Receipt;

    expect(receipt).toMatchObject({
      schema: "ace-opt-0021-dit-dense-vec4-panels-abc-v1",
      experimentId: "OPT-0021",
      status: "passed",
      classification: "bounded-exact-primitive-decision-gate-not-integrated",
      identity: {
        registrationCommit: "7dcbe50c7f04d1e07f6b30657da96372ca8574d1",
        currentCoreSourceSha256:
          "a238f67da07c6ba1097da9d9e9e97960ae97d2e1d5c129fcbabf69e962cbb6b3",
        exactCoreSourceSha256:
          "b5dad12724882d3fc942c7df7b10c7b7b89a4bed595125ff11a5905c03152a37",
        vec4CoreSourceSha256:
          "2229c55f8b7fe66d3770ef7683de68322632d17749fe2d3085d5d46dcdc22df1",
        payloadAuthority: {
          modelPackageLoaded: false,
          syntheticFixture: true,
          persistentWeightLayout: "dit-gemm-n256-k32-tile-major-v1",
          converterOrManifestChange: false,
          productionRuntimeSelected: false,
        },
      },
      correctness: {
        shapeCount: 4,
        executionCount: 16,
        exactComparisonCount: 12,
        comparisonsPerOutputWord: 3,
        outputU32Count: 25_344_000,
        comparedU32Count: 76_032_000,
        mismatchCount: 0,
        aVersusBRawU32Exact: true,
        aVersusCFirstRawU32Exact: true,
        cFirstVersusRerunRawU32Exact: true,
        allFourExecutionsShareShapeHash: true,
        qNaNPrefillCompleteWrites: true,
        canariesUntouched: true,
        finiteOutputs: true,
        tailRowsWritten: true,
        uncapturedGpuErrorCount: 0,
        passed: true,
      },
      timing: {
        samplesPerArmPerShape: 6,
        dispatchSampleCount: 72,
        passed: false,
        decision: "negative-stop-primitive-gate",
      },
      decision: {
        disposition: "negative-stop-primitive-gate",
        packageNativeLayerGateAuthorized: false,
        productionIntegrationAuthorized: false,
        c98TrajectoryRunAuthorized: false,
        m2250ProductRunAuthorized: false,
      },
    });

    expect(receipt.correctness.cases).toHaveLength(4);
    expect(receipt.correctness.cases.reduce(
      (sum, entry) => sum + entry.outputU32Count,
      0,
    )).toBe(receipt.correctness.outputU32Count);
    expect(receipt.correctness.cases.reduce(
      (sum, entry) => sum + entry.comparedU32Count,
      0,
    )).toBe(receipt.correctness.comparedU32Count);
    expect(receipt.correctness.cases.every((entry) =>
      entry.executionOrder.join("") === "ABCC" &&
      entry.executionCount === 4 &&
      entry.exactComparisonCount === 3 &&
      entry.comparisonsPerOutputWord === 3 &&
      entry.comparedU32Count === entry.outputU32Count * 3 &&
      entry.mismatchCount === 0 &&
      Object.values(entry.comparisons).every((comparison) =>
        comparison.comparedU32Count === entry.outputU32Count &&
        comparison.mismatchCount === 0 &&
        comparison.firstMismatch === null &&
        comparison.rawU32Exact
      ) &&
      entry.aVersusBRawU32Exact && entry.aVersusCFirstRawU32Exact &&
      entry.cFirstVersusRerunRawU32Exact &&
      entry.allFourExecutionsShareShapeHash &&
      new Set(Object.values(entry.hashes)).size === 1 &&
      entry.nonFiniteCount === 0 && entry.qNaNPrefillCount === 0 &&
      entry.prefixCanaryIntact && entry.suffixCanaryIntact &&
      entry.tailRowWritten && entry.finiteCompleteWrites && entry.passed
    )).toBe(true);

    const aggregate = receipt.correctness.aggregateOutputHashManifestSha256;
    expect(aggregate.algorithm).toBe(
      "sha256-utf8-of-ordered-shape-id-colon-raw-output-sha256-lines-v1",
    );
    for (const arm of ["A", "B", "CFirst", "CRerun"] as const) {
      expect(sha256(receipt.correctness.cases.map((entry) =>
        `${entry.id}:${entry.hashes[arm]}`
      ).join("\n"))).toBe(aggregate.hashes[arm]);
    }
    expect(new Set(Object.values(aggregate.hashes)).size).toBe(1);
    expect(aggregate.allFourExecutionsShareAggregateHash).toBe(true);
    expect(result.correctness.primitive.aggregateOutputHashManifestSha256)
      .toEqual(aggregate.hashes);
    expect(result.correctness.primitive.caseOutputSha256).toEqual(
      Object.fromEntries(receipt.correctness.cases.map((entry) =>
        [entry.id, entry.hashes.A]
      )),
    );

    const collected = Object.fromEntries(SHAPE_IDS.map((id) => [id, {
      A: [] as number[],
      B: [] as number[],
      C: [] as number[],
    }])) as Record<string, Record<Arm, number[]>>;
    let sampleIndex = 0;
    for (let roundIndex = 0; roundIndex < 6; roundIndex += 1) {
      const shapeOrder = [0, 1, 2, 3].map((index) =>
        (index + roundIndex % 4) % 4
      );
      for (const shapeIndex of shapeOrder) {
        for (let armPosition = 0; armPosition < 3; armPosition += 1) {
          const sample = receipt.timing.dispatchSamples[sampleIndex++]!;
          const arm = EXPECTED_ORDERS[roundIndex]![armPosition]!;
          expect(sample).toMatchObject({
            roundIndex,
            rotatedShapeOrder: shapeOrder,
            shapeIndex,
            shapeId: SHAPE_IDS[shapeIndex],
            armOrder: EXPECTED_ORDERS[roundIndex],
            armPosition,
            arm,
            commandBufferCount: 1,
            dispatchCount: 1,
            matchingQueueCompletionFenceCount: 1,
          });
          expect(sample.wallMilliseconds).toBeGreaterThan(0);
          expect(sample.fenceAtPerformanceMilliseconds -
            sample.submitAtPerformanceMilliseconds).toBe(
              sample.wallMilliseconds,
            );
          collected[sample.shapeId]![arm].push(sample.wallMilliseconds);
        }
      }
    }
    expect(sampleIndex).toBe(72);

    const complete = { A: 0, B: 0, C: 0 };
    const feedForward = { A: 0, B: 0, C: 0 };
    const perShapeCFasterThanB: Record<string, boolean> = {};
    for (const [index, stratum] of receipt.timing.strata.entries()) {
      expect(stratum.id).toBe(SHAPE_IDS[index]);
      for (const arm of ARMS) {
        expect(stratum.samples[arm]).toEqual(collected[stratum.id]![arm]);
        const median = median6(collected[stratum.id]![arm]);
        expect(stratum.medians[arm]).toBe(median);
        complete[arm] += median * stratum.productionMultiplicity;
        feedForward[arm] += median * stratum.feedForwardMultiplicity;
      }
      expect(stratum.cFasterThanB).toBe(
        stratum.medians.C < stratum.medians.B,
      );
      expect(stratum.bToCSpeedup).toBe(
        stratum.medians.B / stratum.medians.C,
      );
      expect(stratum.aToCSpeedup).toBe(
        stratum.medians.A / stratum.medians.C,
      );
      expect(stratum.aToCSavingMilliseconds).toBe(
        stratum.medians.A - stratum.medians.C,
      );
      perShapeCFasterThanB[stratum.id] = stratum.cFasterThanB;
    }
    expect(complete).toEqual({
      A: 227.60000002384186,
      B: 184.99999982118607,
      C: 186.6500000357628,
    });
    expect(feedForward).toEqual({
      A: 145.10000002384186,
      B: 115.69999986886978,
      C: 116.3500000834465,
    });
    expect(receipt.timing.completeDense.milliseconds).toEqual(complete);
    expect(receipt.timing.feedForward.milliseconds).toEqual(feedForward);
    expect(receipt.timing.completeDense.bToCSpeedup).toBe(
      complete.B / complete.C,
    );
    expect(receipt.timing.feedForward.bToCSpeedup).toBe(
      feedForward.B / feedForward.C,
    );
    expect(receipt.timing.completeDense.aToCSavingMilliseconds).toBe(
      complete.A - complete.C,
    );
    expect(receipt.timing.feedForward.aToCSavingMilliseconds).toBe(
      feedForward.A - feedForward.C,
    );
    expect(perShapeCFasterThanB).toEqual({
      "h-h": false,
      "h-1024": true,
      "h-6144": false,
      "6144-h": true,
    });
    expect(receipt.timing.everyShapeCFasterThanB).toBe(false);
    expect(receipt.timing.completeDense.bToCSpeedup).toBeLessThan(
      receipt.timing.completeDense.bToCThreshold,
    );
    expect(receipt.timing.completeDense.aToCSavingMilliseconds).toBeLessThan(
      receipt.timing.completeDense.aToCSavingThresholdMilliseconds,
    );
    expect(result.metrics.baseline).toMatchObject({
      completeDenseMilliseconds: complete.B,
      feedForwardMilliseconds: feedForward.B,
    });
    expect(result.metrics.candidate).toMatchObject({
      completeDenseMilliseconds: complete.C,
      feedForwardMilliseconds: feedForward.C,
    });
    expect(result.metrics.delta.currentArmA).toMatchObject({
      completeDenseMilliseconds: complete.A,
      feedForwardMilliseconds: feedForward.A,
    });
    expect(result.metrics.delta.perShapeCFasterThanB).toEqual(
      perShapeCFasterThanB,
    );
    expect(result.metrics.delta.projectedDenseSavingAcross192LayerEvaluationsMilliseconds)
      .toBe((complete.A - complete.C) * 192);
    expect(result.metrics.delta.projectedDenseSavingThresholdMilliseconds)
      .toBe(52.0834 * 192);
  });

  it("recomputes source, fixture, shader, environment, thermal, and cleanup identity", {
    timeout: 120_000,
  }, async () => {
    const [resultText, receiptBytes, thermalBytes, record, ledger] =
      await Promise.all([
        readFile(RESULT_URL, "utf8"),
        readFile(RECEIPT_URL),
        readFile(THERMAL_URL),
        readFile(RECORD_URL, "utf8"),
        readFile(LEDGER_URL, "utf8"),
      ]);
    const result = parseOptimizationResultJson(resultText) as unknown as ResultView;
    const receipt = JSON.parse(receiptBytes.toString("utf8")) as Opt0021Receipt;

    expect(thermalBytes.byteLength).toBe(4_322);
    expect(sha256(thermalBytes)).toBe(
      "fe482a89246148be4fe76af169f472b4fe4af42f5ee74d7a5925a7f4d12f761f",
    );
    expect(result.identity.rawArtifactBytes).toEqual({
      receipt: receiptBytes.byteLength,
      thermal: thermalBytes.byteLength,
    });

    for (const [path, expected] of SOURCE_IDENTITIES) {
      expect(sha256(await readFile(new URL(path, import.meta.url)))).toBe(expected);
    }
    expect(result.identity).toMatchObject({
      currentCoreSourceSha256: SOURCE_IDENTITIES[0][1],
      exactCoreSourceSha256: SOURCE_IDENTITIES[1][1],
      candidateCoreSourceSha256: SOURCE_IDENTITIES[2][1],
      harnessSourceSha256: SOURCE_IDENTITIES[3][1],
      harnessHtmlSha256: SOURCE_IDENTITIES[4][1],
      preExecutionContractSha256: SOURCE_IDENTITIES[5][1],
      candidateCoreContractSha256: SOURCE_IDENTITIES[6][1],
    });
    expect(receipt.identity).toMatchObject({
      currentCoreSourceSha256: SOURCE_IDENTITIES[0][1],
      exactCoreSourceSha256: SOURCE_IDENTITIES[1][1],
      vec4CoreSourceSha256: SOURCE_IDENTITIES[2][1],
    });

    const fixtureDeclaration = buildOpt0021FixtureDeclaration();
    expect(sha256(canonicalJson(fixtureDeclaration))).toBe(
      "954aff0a07dcc2946ac8191c054e2ecf63473d05ed9ebf81dc7db6d535f80f0c",
    );
    expect(result.identity.syntheticFixtureDeclarationSha256).toBe(
      receipt.identity.syntheticFixtureDeclarationSha256,
    );
    const fixtureHashes: Record<string, {
      activationSha256: string;
      packedWeightSha256: string;
    }> = {};
    for (const [shapeIndex, spec] of buildOpt0021ShapeSpecs().entries()) {
      const activation = new Float32Array(spec.shape.rows * spec.shape.inner);
      fillOpt0021ActivationFixture(activation, shapeIndex);
      const weight = new Uint16Array(spec.shape.inner * spec.shape.columns);
      fillOpt0021PackedWeightFixture(weight, shapeIndex);
      fixtureHashes[spec.id] = {
        activationSha256: sha256(new Uint8Array(activation.buffer)),
        packedWeightSha256: sha256(new Uint8Array(weight.buffer)),
      };
    }
    expect(fixtureHashes).toEqual(result.identity.fixtureHashes);
    expect(fixtureHashes).toEqual(Object.fromEntries(receipt.fixture.shapes.map(
      (entry) => [entry.id, {
        activationSha256: entry.fixture.activationSha256,
        packedWeightSha256: entry.fixture.packedWeightSha256,
      }],
    )));

    const generatedShaderHashes = buildOpt0021ShapeSpecs().map((spec) => ({
      id: spec.id,
      A: sha256(aceOpt0009DenseGemmWgsl(spec.shape)),
      B: sha256(aceOpt0019DenseCooperativePanelsWgsl(spec.shape)),
      C: sha256(aceOpt0021DenseCooperativeVec4PanelsWgsl(spec.shape)),
    }));
    expect(generatedShaderHashes).toEqual(receipt.identity.generatedShaders);
    expect(generatedShaderHashes).toEqual(result.identity.generatedShaderHashes);

    expect(receipt.environment).toMatchObject({
      userAgent:
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
      adapterInfo: {
        vendor: "apple",
        architecture: "metal-3",
        subgroupMinSize: 32,
        subgroupMaxSize: 32,
        isFallbackAdapter: false,
      },
      features: ["core-features-and-limits", "shader-f16", "subgroups"],
      limits: {
        maxBufferSize: 268_435_456,
        maxStorageBufferBindingSize: 134_217_728,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupSizeY: 256,
        maxComputeWorkgroupStorageSize: 16_384,
      },
    });

    const thermal = thermalBytes.toString("utf8").trim().split("\n")
      .map((line) => JSON.parse(line) as ThermalObservation);
    expect(thermal).toHaveLength(76);
    expect(thermal.every((entry, index) =>
      entry.index === index && entry.level === 0 &&
      (index === 0 || entry.epochMilliseconds >
        thermal[index - 1]!.epochMilliseconds)
    )).toBe(true);
    const totalMaximumGap = maximumGap(thermal);
    expect(totalMaximumGap).toBe(1_013);
    const gate = thermal.filter((entry) =>
      entry.epochMilliseconds >=
        receipt.protocol.thermal.startedAtEpochMilliseconds &&
      entry.epochMilliseconds <=
        receipt.protocol.thermal.completedAtEpochMilliseconds
    );
    expect(gate).toHaveLength(60);
    expect(gate[0]!.epochMilliseconds).toBe(
      receipt.protocol.thermal.startedAtEpochMilliseconds,
    );
    expect(gate.at(-1)!.epochMilliseconds).toBe(
      receipt.protocol.thermal.completedAtEpochMilliseconds,
    );
    expect(gate.at(-1)!.epochMilliseconds - gate[0]!.epochMilliseconds).toBe(
      receipt.protocol.thermal.durationMilliseconds,
    );
    expect(maximumGap(gate)).toBe(
      receipt.protocol.thermal.maximumPollGapMilliseconds,
    );
    expect(receipt.protocol.thermal).toMatchObject({
      source: "notifyutil-com.apple.system.thermalpressurelevel",
      durationMilliseconds: 59_507,
      observationCount: 60,
      pollMilliseconds: 1_000,
      maximumPollGapMilliseconds: 1_013,
      nonNominalObservationCount: 0,
      launchDelayMilliseconds: 233,
    });
    expect(receipt.protocol).toMatchObject({
      fullOutputCorrectnessCompletedBeforeThermalGate: true,
      allThreeKernelsCompiledAndWarmedBeforeThermalGate: true,
      timingRounds: 6,
      continuousExternalThermalTraceRequiredThroughCleanup: true,
      oneThirtySecondNominalGate: true,
      unchangedThermalRetryPerformed: false,
    });
    expect(receipt.preparation.completedAtEpochMilliseconds).toBeLessThanOrEqual(
      receipt.protocol.thermal.startedAtEpochMilliseconds,
    );
    expect(receipt.timing.timingStartedAtEpochMilliseconds).toBeGreaterThanOrEqual(
      receipt.protocol.thermal.completedAtEpochMilliseconds,
    );
    expect(thermal.at(-1)!.epochMilliseconds).toBeGreaterThanOrEqual(
      receipt.protocol.thermalTraceRequiredThroughEpochMilliseconds,
    );
    expect(thermal.at(-1)!.epochMilliseconds).toBeGreaterThanOrEqual(
      receipt.cleanup.completedAtEpochMilliseconds,
    );
    expect(result.protocol.thermal).toEqual({
      source: receipt.protocol.thermal.source,
      startedAtEpochMilliseconds:
        receipt.protocol.thermal.startedAtEpochMilliseconds,
      completedAtEpochMilliseconds:
        receipt.protocol.thermal.completedAtEpochMilliseconds,
      durationMilliseconds: receipt.protocol.thermal.durationMilliseconds,
      observationCount: receipt.protocol.thermal.observationCount,
      maximumPollGapMilliseconds:
        receipt.protocol.thermal.maximumPollGapMilliseconds,
      nonNominalObservationCount:
        receipt.protocol.thermal.nonNominalObservationCount,
      launchDelayMilliseconds: receipt.protocol.thermal.launchDelayMilliseconds,
      totalTraceObservationCount: thermal.length,
      totalTraceMaximumPollGapMilliseconds: totalMaximumGap,
      totalTraceLastEpochMilliseconds: thermal.at(-1)!.epochMilliseconds,
      cleanupCompletedAtEpochMilliseconds:
        receipt.cleanup.completedAtEpochMilliseconds,
      postCleanupCoverageMilliseconds: thermal.at(-1)!.epochMilliseconds -
        receipt.cleanup.completedAtEpochMilliseconds,
    });

    expect(receipt.memory.maximumLiveBytes).toBe(311_749_632);
    expect(receipt.memory.maximumLiveBytes).toBe(
      receipt.preparation.memoryHighWaterBytes,
    );
    expect(receipt.cleanup).toMatchObject({
      firstCall: {
        createdBufferCount: 20,
        destroyedBufferCount: 20,
        liveBufferCount: 0,
        liveBytes: 0,
        maximumLiveBytes: 311_749_632,
        idempotent: true,
        repeatedCall: false,
        drainBeforeRelease: true,
        deviceDestroyed: true,
      },
      secondCall: {
        createdBufferCount: 20,
        destroyedBufferCount: 20,
        liveBufferCount: 0,
        liveBytes: 0,
        maximumLiveBytes: 311_749_632,
        idempotent: true,
        repeatedCall: true,
        drainBeforeRelease: true,
      },
      idempotent: true,
      zeroLiveResources: true,
    });
    expect(result.correctness.cleanup).toMatchObject({
      createdBufferCount: 20,
      destroyedBufferCount: 20,
      liveBufferCount: 0,
      liveBytes: 0,
      maximumLiveBytes: 311_749_632,
      idempotent: true,
      drainBeforeRelease: true,
      deviceDestroyed: true,
    });

    expect(record).toContain("- Evidence: `negative`");
    expect(record).toContain("- Disposition: `abandoned`");
    expect(record).toContain("All three frozen performance conditions failed");
    expect(record).toContain("There was no unchanged-candidate retry");
    expect(ledger).toContain(
      "| OPT-0021 | DiT dense GEMM | Reorienting OPT-0019's two cooperative panels",
    );
    expect(ledger).toContain("| negative | abandoned |");
  });
});

function median6(samples: readonly number[]): number {
  expect(samples).toHaveLength(6);
  const sorted = [...samples].sort((left, right) => left - right);
  return (sorted[2]! + sorted[3]!) / 2;
}

function maximumGap(observations: readonly ThermalObservation[]): number {
  return Math.max(...observations.slice(1).map((entry, index) =>
    entry.epochMilliseconds - observations[index]!.epochMilliseconds
  ));
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(sortJsonValue(value));
}

function sortJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortJsonValue);
  if (value !== null && typeof value === "object") {
    const record = value as Readonly<Record<string, unknown>>;
    return Object.fromEntries(Object.keys(record).sort().map((key) =>
      [key, sortJsonValue(record[key])]
    ));
  }
  return value;
}
