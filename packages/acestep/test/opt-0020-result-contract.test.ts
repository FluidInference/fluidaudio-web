import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

type Arm = "A" | "B" | "C";
type OutputArm = "A" | "B" | "CFirst" | "CRerun";
type JsonValue =
  | null
  | boolean
  | number
  | string
  | readonly JsonValue[]
  | { readonly [key: string]: JsonValue };

interface NumericalThresholds {
  readonly nrmseMaximum: number;
  readonly snrDecibelsMinimum: number;
  readonly pearsonCorrelationMinimum: number;
  readonly relativeMaximumErrorMaximum: number;
  readonly maximumAbsoluteErrorMaximum: number;
}

interface UlpBin {
  readonly label: string;
  readonly minimum: number;
  readonly maximum: number;
  readonly count: number;
}

interface DifferenceLocation {
  readonly shapeId: string;
  readonly linearIndex: number;
  readonly absoluteError: number;
}

interface NumericalSummary {
  readonly count: number;
  readonly differingCount: number;
  readonly ranges: {
    readonly control: readonly [number, number];
    readonly candidate: readonly [number, number];
    readonly error: readonly [number, number];
  };
  readonly signedMeanError: number;
  readonly meanAbsoluteError: number;
  readonly rmsError: number;
  readonly controlRms: number;
  readonly nrmse: number;
  readonly snrDecibels: number;
  readonly pearsonCorrelation: number;
  readonly maximumAbsoluteControl: number;
  readonly maximumAbsoluteError: number;
  readonly relativeMaximumError: number;
  readonly firstDifference: DifferenceLocation;
  readonly worstDifference: DifferenceLocation;
  readonly signedZeroDifferenceCount: number;
  readonly signAwareF32UlpDistribution: readonly UlpBin[];
  readonly thresholds: NumericalThresholds;
  readonly passed: boolean;
}

interface CorrectnessCase {
  readonly id: string;
  readonly shape: {
    readonly rows: number;
    readonly inner: number;
    readonly columns: number;
  };
  readonly outputU32Count: number;
  readonly executionOrder: readonly string[];
  readonly executionCount: number;
  readonly exactComparisonCount: number;
  readonly aVersusBComparedU32Count: number;
  readonly cFirstVersusRerunComparedU32Count: number;
  readonly candidateVersusControlStatCount: number;
  readonly hashes: Readonly<Record<OutputArm, string>>;
  readonly aVersusBRawU32Exact: boolean;
  readonly cFirstVersusRerunRawU32Exact: boolean;
  readonly nonFiniteCount: number;
  readonly qNaNPrefillCount: number;
  readonly prefixCanaryIntact: boolean;
  readonly suffixCanaryIntact: boolean;
  readonly tailRowWritten: boolean;
  readonly numerics: NumericalSummary;
  readonly passed: boolean;
  readonly fixture: {
    readonly generatorVersion: string;
    readonly activationSeed: number;
    readonly weightSeed: number;
    readonly activationSha256: string;
    readonly packedWeightSha256: string;
  };
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
}

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

interface Opt0020Receipt {
  readonly schema: string;
  readonly experimentId: string;
  readonly status: string;
  readonly classification: string;
  readonly recordedAt: string;
  readonly identity: {
    readonly registrationCommit: string;
    readonly currentCoreSourceSha256: string;
    readonly exactCoreSourceSha256: string;
    readonly dot4CoreSourceSha256: string;
    readonly exactKernelSetId: string;
    readonly dot4KernelSetId: string;
    readonly payloadAuthority: {
      readonly modelPackageLoaded: boolean;
      readonly syntheticFixture: boolean;
      readonly persistentWeightLayout: string;
      readonly converterOrManifestChange: boolean;
    };
    readonly generatedShaders: readonly JsonValue[];
  };
  readonly fixture: {
    readonly version: string;
    readonly activationSeeds: readonly number[];
    readonly weightSeeds: readonly number[];
    readonly finiteNonPowerFp16MagnitudeBits: readonly number[];
    readonly constants: JsonValue;
    readonly shapes: readonly {
      readonly id: string;
      readonly shape: CorrectnessCase["shape"];
      readonly fixture: CorrectnessCase["fixture"] & { readonly ordinal: number };
    }[];
  };
  readonly environment: {
    readonly userAgent: string;
    readonly adapterInfo: JsonValue;
    readonly features: readonly string[];
    readonly limits: JsonValue;
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
    readonly timingOrder: string;
    readonly sampleTopology: string;
    readonly continuousExternalThermalTraceRequiredThroughCleanup: boolean;
    readonly thermalTraceRequiredThroughEpochMilliseconds: number;
    readonly oneThirtySecondNominalGate: boolean;
    readonly unchangedThermalRetryPerformed: boolean;
  };
  readonly correctness: {
    readonly shapeCount: number;
    readonly executionCount: number;
    readonly exactComparisonCount: number;
    readonly aVersusBComparedU32Count: number;
    readonly cFirstVersusRerunComparedU32Count: number;
    readonly candidateVersusControlStatCount: number;
    readonly qNaNPrefillCompleteWrites: boolean;
    readonly canariesUntouched: boolean;
    readonly finiteOutputs: boolean;
    readonly cDeterministicReruns: boolean;
    readonly aVersusBRawU32Exact: boolean;
    readonly uncapturedGpuErrorCount: number;
    readonly aggregateOutputHashManifestSha256: {
      readonly shapeOrder: readonly string[];
      readonly hashes: Readonly<Record<OutputArm, string>>;
    };
    readonly aggregateNumerics: NumericalSummary;
    readonly cases: readonly CorrectnessCase[];
    readonly passed: boolean;
  };
  readonly timing: {
    readonly samplesPerArmPerShape: number;
    readonly dispatchSampleCount: number;
    readonly dispatchSamples: readonly DispatchSample[];
    readonly strata: readonly TimingStratum[];
    readonly completeDense: {
      readonly multiplicities: string;
      readonly milliseconds: Readonly<Record<Arm, number>>;
      readonly bToCSpeedup: number;
      readonly bToCThreshold: number;
      readonly aToCSpeedup: number;
      readonly aToCThreshold: number;
    };
    readonly feedForward: {
      readonly multiplicities: string;
      readonly milliseconds: Readonly<Record<Arm, number>>;
      readonly bToCSpeedup: number;
      readonly bToCThreshold: number;
    };
    readonly everyShapeCFasterThanB: boolean;
    readonly passed: boolean;
    readonly decision: string;
    readonly timingStartedAtEpochMilliseconds: number;
    readonly timingCompletedAtEpochMilliseconds: number;
  };
  readonly memory: {
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
    readonly dot4CoreSourceSha256: string;
    readonly harnessSourceSha256: string;
    readonly harnessHtmlSha256: string;
    readonly preExecutionContractSha256: string;
    readonly dot4CoreContractSha256: string;
    readonly syntheticFixtureDeclarationSha256: string;
    readonly generatedShaderManifestSha256: string;
    readonly outputCaseHashManifestSha256: string;
    readonly fixtureHashes: Readonly<
      Record<
        string,
        {
          readonly activationSha256: string;
          readonly packedWeightSha256: string;
        }
      >
    >;
    readonly rawArtifactBytes: { readonly receipt: number; readonly thermal: number };
    readonly productionPackageLoaded: boolean;
    readonly productionRuntimeSelected: boolean;
    readonly productionIntegrationPerformed: boolean;
  };
  readonly protocol: {
    readonly samples: number;
    readonly armOrder: readonly string[];
    readonly dispatchSampleCount: number;
    readonly authoritativeTimedRunCount: number;
    readonly unchangedThermalRetryPerformed: boolean;
    readonly thermal: {
      readonly observationCount: number;
      readonly maximumPollGapMilliseconds: number;
      readonly totalTraceObservationCount: number;
      readonly totalTraceMaximumPollGapMilliseconds: number;
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
      readonly qualified: boolean;
      readonly packageNativeLayerGatePerformed: boolean;
      readonly c98TrajectoryRunPerformed: boolean;
      readonly m2250ProductRunPerformed: boolean;
      readonly productionIntegrationPerformed: boolean;
    };
    readonly delta: {
      readonly bToCCompleteSpeedup: number;
      readonly aToCCompleteSpeedup: number;
      readonly bToCFeedForwardSpeedup: number;
      readonly everyShapeFasterConditionPassed: boolean;
      readonly bToCCompleteConditionPassed: boolean;
      readonly aToCCompleteConditionPassed: boolean;
      readonly bToCFeedForwardConditionPassed: boolean;
      readonly stopRuleFired: boolean;
      readonly observedIntegratedDitSpeedup: null;
      readonly observed180SecondProductSpeedup: null;
      readonly under60SecondProjection: null;
    };
  };
  readonly correctness: {
    readonly primitive: {
      readonly aVersusBComparedU32Count: number;
      readonly cFirstVersusRerunComparedU32Count: number;
      readonly candidateVersusControlStatCount: number;
      readonly aVersusBRawU32Exact: boolean;
      readonly cDeterministicReruns: boolean;
      readonly aggregateOutputHashManifestSha256: Readonly<Record<OutputArm, string>>;
    };
    readonly numerical: NumericalSummary & {
      readonly allFourShapesPassed: boolean;
      readonly signAwareF32UlpBinCounts: Readonly<Record<string, number>>;
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
  "../optimization/results/OPT-0020/result.json",
  import.meta.url,
);
const RECEIPT_URL = new URL(
  "../optimization/artifacts/OPT-0020/raw/dense-cooperative-dot4-abc.json",
  import.meta.url,
);
const THERMAL_URL = new URL(
  "../optimization/artifacts/OPT-0020/raw/dense-cooperative-dot4-thermal.jsonl",
  import.meta.url,
);
const RECORD_URL = new URL(
  "../optimization/experiments/OPT-0020-dit-dense-cooperative-dot4.md",
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
    "../src/webgpu/kernels/dit-dense-fp16-cooperative-dot4.ts",
    "466cf7b4c8f860ff55a89b03a5bb2ead99c0d849420918e11d6888e11482d28e",
  ],
  [
    "browser/opt-0020-dit-dense-cooperative-dot4.ts",
    "f2a7f78ef112481b9c4423ca88366a449c02eaf925524121a6f32d782bedda72",
  ],
  [
    "browser/opt-0020-dit-dense-cooperative-dot4.html",
    "19b226945d8c02329f5ee0e34a39fa1d393f3b8eb20f8e77bf4b8f4992c79b20",
  ],
  [
    "opt-0020-dit-dense-cooperative-dot4-contract.test.ts",
    "b2e2970c4f74abffb2c9b72e9e2eb5264cd4d16b6fba855a7d1d45aee61d7205",
  ],
  [
    "dit-dense-fp16-cooperative-dot4.test.ts",
    "491733c0740fc18079870b6114f64f152e245df19cb10ae6d8d8ec66e6b0e745",
  ],
] as const;

const SHAPE_IDS = ["h-h", "h-1024", "h-6144", "6144-h"] as const;
const ARM_ORDERS = [
  ["A", "B", "C"],
  ["A", "C", "B"],
  ["B", "A", "C"],
  ["B", "C", "A"],
  ["C", "A", "B"],
  ["C", "B", "A"],
] as const satisfies readonly (readonly Arm[])[];

const sha256 = (bytes: Uint8Array | string): string =>
  createHash("sha256").update(bytes).digest("hex");

const canonicalJson = (value: JsonValue): string => {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalJson(entry)).join(",")}]`;
  }
  const object = value as { readonly [key: string]: JsonValue };
  return `{${Object.keys(object)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key]!)}`)
    .join(",")}}`;
};

const median6 = (samples: readonly number[]): number => {
  expect(samples).toHaveLength(6);
  const sorted = [...samples].sort((left, right) => left - right);
  return (sorted[2]! + sorted[3]!) / 2;
};

const weightedScore = (
  strata: readonly TimingStratum[],
  arm: Arm,
  feedForward: boolean,
): number => strata.reduce(
  (sum, stratum) =>
    sum +
    stratum.medians[arm] *
      (feedForward
        ? stratum.feedForwardMultiplicity
        : stratum.productionMultiplicity),
  0,
);

describe("OPT-0020 optimization result", () => {
  it("commits canonical numerically-valid but performance-negative evidence", async () => {
    const text = await readFile(RESULT_URL, "utf8");
    const committed = parseOptimizationResultJson(text);
    const view = committed as unknown as ResultView;

    expect(committed).toMatchObject({
      experimentId: "OPT-0020",
      riskClass: "reordered-rounding",
      baselineCommit: "7f8db6c5cfb17b2c508669e1b136d1bb02bab239",
      candidateCommit: "1d825a1399fdcc23c4ef3ce18151a2efa2415626",
      evidence: { conclusion: "negative" },
      disposition: { state: "abandoned" },
      identity: {
        browserVersion:
          "Google Chrome 151.0.7922.138; reduced user agent 151.0.0.0",
        benchmarkHarnessCommit:
          "1d825a1399fdcc23c4ef3ce18151a2efa2415626",
        registrationCommit:
          "fce77739841572942eca4e96cc6a9f48eb02a971",
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
          "optimization/artifacts/OPT-0020/raw/dense-cooperative-dot4-abc.json",
        sha256:
          "8c168677ee630a4ee337e7b3e7e2c0d8c1c2b53493ce6eebdc48765cdb554679",
      },
      {
        location:
          "optimization/artifacts/OPT-0020/raw/dense-cooperative-dot4-thermal.jsonl",
        sha256:
          "cd5461cb760b9e6022a0f8b78105661d1cb17d7e7dd525f415fc7e60298de960",
      },
    ]);
    expect(view.identity.rawArtifactBytes).toEqual({
      receipt: 90_410,
      thermal: 6_212,
    });
    expect(view.protocol).toMatchObject({
      samples: 6,
      armOrder: ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"],
      dispatchSampleCount: 72,
      authoritativeTimedRunCount: 1,
      unchangedThermalRetryPerformed: false,
      thermal: {
        observationCount: 93,
        maximumPollGapMilliseconds: 1_031,
        totalTraceObservationCount: 109,
        totalTraceMaximumPollGapMilliseconds: 1_089,
      },
    });
    expect(view.metrics.baseline).toMatchObject({
      completeDenseMilliseconds: 180.00000029802322,
      feedForwardMilliseconds: 106.6000000834465,
    });
    expect(view.metrics.candidate).toMatchObject({
      completeDenseMilliseconds: 302.6499999165535,
      feedForwardMilliseconds: 192.94999998807907,
      qualified: false,
      packageNativeLayerGatePerformed: false,
      c98TrajectoryRunPerformed: false,
      m2250ProductRunPerformed: false,
      productionIntegrationPerformed: false,
    });
    expect(view.metrics.delta).toMatchObject({
      bToCCompleteSpeedup: 0.5947464078891549,
      aToCCompleteSpeedup: 0.7203039805137254,
      bToCFeedForwardSpeedup: 0.5524747348537575,
      everyShapeFasterConditionPassed: false,
      bToCCompleteConditionPassed: false,
      aToCCompleteConditionPassed: false,
      bToCFeedForwardConditionPassed: false,
      stopRuleFired: true,
      observedIntegratedDitSpeedup: null,
      observed180SecondProductSpeedup: null,
      under60SecondProjection: null,
    });
    expect(view.correctness.primitive).toMatchObject({
      aVersusBComparedU32Count: 25_344_000,
      cFirstVersusRerunComparedU32Count: 25_344_000,
      candidateVersusControlStatCount: 25_344_000,
      aVersusBRawU32Exact: true,
      cDeterministicReruns: true,
    });
    expect(view.correctness.numerical).toMatchObject({
      count: 25_344_000,
      differingCount: 24_304_155,
      nrmse: 9.336650318270838e-7,
      snrDecibels: 120.59617812682795,
      pearsonCorrelation: 0.9999999999994078,
      relativeMaximumError: 0.000002798660623218186,
      maximumAbsoluteError: 0.0000858306884765625,
      allFourShapesPassed: true,
      passed: true,
    });
    expect(view.correctness.cleanup).toMatchObject({
      createdBufferCount: 20,
      destroyedBufferCount: 20,
      liveBufferCount: 0,
      liveBytes: 0,
      maximumLiveBytes: 311_749_632,
      idempotent: true,
      drainBeforeRelease: true,
      deviceDestroyed: true,
    });
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("recomputes source, fixture, output, numerical, timing, thermal, and stop contracts", async () => {
    const [
      resultText,
      receiptBytes,
      thermalBytes,
      record,
      ledger,
      ...sourceBytes
    ] = await Promise.all([
      readFile(RESULT_URL, "utf8"),
      readFile(RECEIPT_URL),
      readFile(THERMAL_URL),
      readFile(RECORD_URL, "utf8"),
      readFile(LEDGER_URL, "utf8"),
      ...SOURCE_IDENTITIES.map(([path]) =>
        readFile(new URL(path, import.meta.url))
      ),
    ]);
    const result = parseOptimizationResultJson(resultText) as unknown as ResultView;

    expect(receiptBytes.byteLength).toBe(90_410);
    expect(sha256(receiptBytes)).toBe(
      "8c168677ee630a4ee337e7b3e7e2c0d8c1c2b53493ce6eebdc48765cdb554679",
    );
    expect(thermalBytes.byteLength).toBe(6_212);
    expect(sha256(thermalBytes)).toBe(
      "cd5461cb760b9e6022a0f8b78105661d1cb17d7e7dd525f415fc7e60298de960",
    );
    for (const [index, [, expectedHash]] of SOURCE_IDENTITIES.entries()) {
      expect(sha256(sourceBytes[index]!)).toBe(expectedHash);
    }

    const receipt = JSON.parse(
      receiptBytes.toString("utf8"),
    ) as Opt0020Receipt;
    expect(receipt).toMatchObject({
      schema: "ace-opt-0020-dit-dense-cooperative-dot4-abc-v1",
      experimentId: "OPT-0020",
      status: "passed",
      classification:
        "bounded-reordered-rounding-primitive-decision-gate-not-integrated",
      recordedAt: "2026-08-15T05:28:48.702Z",
      identity: {
        registrationCommit:
          "fce77739841572942eca4e96cc6a9f48eb02a971",
        currentCoreSourceSha256:
          result.identity.currentCoreSourceSha256,
        exactCoreSourceSha256: result.identity.exactCoreSourceSha256,
        dot4CoreSourceSha256: result.identity.dot4CoreSourceSha256,
        exactKernelSetId:
          "opt-0019-m64-n128-k16-cooperative-fp16-fp32-v1",
        dot4KernelSetId:
          "opt-0020-m64-n128-k16-cooperative-dot4-fp16-fp32-v1",
        payloadAuthority: {
          modelPackageLoaded: false,
          syntheticFixture: true,
          persistentWeightLayout: "dit-gemm-n256-k32-tile-major-v1",
          converterOrManifestChange: false,
        },
      },
      protocol: {
        fullOutputCorrectnessCompletedBeforeThermalGate: true,
        allThreeKernelsCompiledAndWarmedBeforeThermalGate: true,
        timingRounds: 6,
        oneThirtySecondNominalGate: true,
        unchangedThermalRetryPerformed: false,
      },
      correctness: {
        shapeCount: 4,
        executionCount: 16,
        exactComparisonCount: 8,
        aVersusBComparedU32Count: 25_344_000,
        cFirstVersusRerunComparedU32Count: 25_344_000,
        candidateVersusControlStatCount: 25_344_000,
        qNaNPrefillCompleteWrites: true,
        canariesUntouched: true,
        finiteOutputs: true,
        cDeterministicReruns: true,
        aVersusBRawU32Exact: true,
        uncapturedGpuErrorCount: 0,
        passed: true,
      },
      timing: {
        samplesPerArmPerShape: 6,
        dispatchSampleCount: 72,
        everyShapeCFasterThanB: false,
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
    expect(receipt.environment.userAgent).toContain("Chrome/151.0.0.0");
    expect(receipt.environment.features).toEqual([
      "core-features-and-limits",
      "shader-f16",
      "subgroups",
    ]);

    expect(sha256(canonicalJson(receipt.fixture as unknown as JsonValue))).toBe(
      result.identity.syntheticFixtureDeclarationSha256,
    );
    expect(
      sha256(
        canonicalJson(receipt.identity.generatedShaders as unknown as JsonValue),
      ),
    ).toBe(result.identity.generatedShaderManifestSha256);
    const outputCaseManifest = receipt.correctness.cases.map(({ id, hashes }) => ({
      id,
      hashes,
    }));
    expect(sha256(canonicalJson(outputCaseManifest))).toBe(
      result.identity.outputCaseHashManifestSha256,
    );
    expect(receipt.fixture).toMatchObject({
      version: "opt-0020-finite-fp16-cancellation-v1",
      activationSeeds: [826366246, 655894552, 1779033703, 3144134277],
      weightSeeds: [1013904242, 2773480762, 1359893119, 2600822924],
      finiteNonPowerFp16MagnitudeBits: [9233, 10421, 11603, 12775, 13675, 14765],
      constants: {
        subjectMix: 2654435761,
        groupMix: 2246822507,
        offsetMix: 3266489909,
        pairedCancellationGroups:
          "g and g^1 except every eighth group's breaker",
        withinK4ProductSignPattern: "+--+",
        groupSignFlip: "activation sign xor (floor(k/4) & 1)",
      },
    });
    expect(receipt.fixture.shapes.map(({ id }) => id)).toEqual(SHAPE_IDS);
    expect(receipt.correctness.cases.map(({ id }) => id)).toEqual(SHAPE_IDS);
    for (const [index, correctnessCase] of receipt.correctness.cases.entries()) {
      const frozenFixture = receipt.fixture.shapes[index]!;
      const { ordinal, ...frozenFixtureIdentity } = frozenFixture.fixture;
      expect(ordinal).toBe(index);
      expect(correctnessCase.shape).toEqual(frozenFixture.shape);
      expect(correctnessCase.fixture).toMatchObject(frozenFixtureIdentity);
      expect(result.identity.fixtureHashes[correctnessCase.id]).toEqual({
        activationSha256: correctnessCase.fixture.activationSha256,
        packedWeightSha256: correctnessCase.fixture.packedWeightSha256,
      });
      expect(correctnessCase).toMatchObject({
        executionOrder: ["A", "B", "C", "C"],
        executionCount: 4,
        exactComparisonCount: 2,
        aVersusBRawU32Exact: true,
        cFirstVersusRerunRawU32Exact: true,
        nonFiniteCount: 0,
        qNaNPrefillCount: 0,
        prefixCanaryIntact: true,
        suffixCanaryIntact: true,
        tailRowWritten: true,
        passed: true,
      });
      expect(correctnessCase.hashes.A).toBe(correctnessCase.hashes.B);
      expect(correctnessCase.hashes.CFirst).toBe(
        correctnessCase.hashes.CRerun,
      );

      const numerics = correctnessCase.numerics;
      expect(numerics.count).toBe(correctnessCase.outputU32Count);
      expect(numerics.signAwareF32UlpDistribution.reduce(
        (sum, bin) => sum + bin.count,
        0,
      )).toBe(numerics.count);
      expect(numerics.signedZeroDifferenceCount).toBe(0);
      expect(numerics.nrmse).toBeLessThanOrEqual(
        numerics.thresholds.nrmseMaximum,
      );
      expect(numerics.snrDecibels).toBeGreaterThanOrEqual(
        numerics.thresholds.snrDecibelsMinimum,
      );
      expect(numerics.pearsonCorrelation).toBeGreaterThanOrEqual(
        numerics.thresholds.pearsonCorrelationMinimum,
      );
      expect(numerics.relativeMaximumError).toBeLessThanOrEqual(
        numerics.thresholds.relativeMaximumErrorMaximum,
      );
      expect(numerics.maximumAbsoluteError).toBeLessThanOrEqual(
        numerics.thresholds.maximumAbsoluteErrorMaximum,
      );
      expect(numerics.passed).toBe(true);
    }
    expect(receipt.correctness.cases.reduce(
      (sum, entry) => sum + entry.aVersusBComparedU32Count,
      0,
    )).toBe(receipt.correctness.aVersusBComparedU32Count);
    expect(receipt.correctness.cases.reduce(
      (sum, entry) => sum + entry.cFirstVersusRerunComparedU32Count,
      0,
    )).toBe(receipt.correctness.cFirstVersusRerunComparedU32Count);
    expect(receipt.correctness.cases.reduce(
      (sum, entry) => sum + entry.candidateVersusControlStatCount,
      0,
    )).toBe(receipt.correctness.candidateVersusControlStatCount);

    for (const arm of ["A", "B", "CFirst", "CRerun"] as const) {
      const manifest = receipt.correctness.cases
        .map((entry) => `${entry.id}:${entry.hashes[arm]}`)
        .join("\n");
      expect(sha256(manifest)).toBe(
        receipt.correctness.aggregateOutputHashManifestSha256.hashes[arm],
      );
    }
    expect(receipt.correctness.aggregateOutputHashManifestSha256.hashes).toEqual(
      result.correctness.primitive.aggregateOutputHashManifestSha256,
    );

    const aggregate = receipt.correctness.aggregateNumerics;
    const cases = receipt.correctness.cases;
    const count = cases.reduce((sum, entry) => sum + entry.numerics.count, 0);
    const weightedMean = (
      select: (entry: NumericalSummary) => number,
    ): number => cases.reduce(
      (sum, entry) => sum + select(entry.numerics) * entry.numerics.count,
      0,
    ) / count;
    const rmsError = Math.sqrt(weightedMean((entry) => entry.rmsError ** 2));
    const controlRms = Math.sqrt(
      weightedMean((entry) => entry.controlRms ** 2),
    );
    expect(aggregate.count).toBe(count);
    expect(aggregate.differingCount).toBe(cases.reduce(
      (sum, entry) => sum + entry.numerics.differingCount,
      0,
    ));
    expect(aggregate.signedMeanError).toBeCloseTo(
      weightedMean((entry) => entry.signedMeanError),
      18,
    );
    expect(aggregate.meanAbsoluteError).toBeCloseTo(
      weightedMean((entry) => entry.meanAbsoluteError),
      18,
    );
    expect(aggregate.rmsError).toBeCloseTo(rmsError, 18);
    expect(aggregate.controlRms).toBeCloseTo(controlRms, 14);
    expect(aggregate.nrmse).toBeCloseTo(rmsError / controlRms, 18);
    expect(aggregate.snrDecibels).toBeCloseTo(
      20 * Math.log10(controlRms / rmsError),
      12,
    );
    expect(aggregate.maximumAbsoluteError).toBe(Math.max(
      ...cases.map((entry) => entry.numerics.maximumAbsoluteError),
    ));
    expect(aggregate.maximumAbsoluteControl).toBe(Math.max(
      ...cases.map((entry) => entry.numerics.maximumAbsoluteControl),
    ));
    expect(aggregate.relativeMaximumError).toBe(
      aggregate.maximumAbsoluteError / aggregate.maximumAbsoluteControl,
    );
    expect(aggregate.firstDifference).toEqual(cases[0]!.numerics.firstDifference);
    expect(aggregate.worstDifference).toEqual(
      cases.reduce((worst, entry) =>
        entry.numerics.worstDifference.absoluteError > worst.absoluteError
          ? entry.numerics.worstDifference
          : worst,
      cases[0]!.numerics.worstDifference),
    );
    const aggregateBins = Object.fromEntries(
      aggregate.signAwareF32UlpDistribution.map((bin) => [bin.label, bin.count]),
    );
    for (const bin of aggregate.signAwareF32UlpDistribution) {
      expect(bin.count).toBe(cases.reduce(
        (sum, entry) =>
          sum +
          entry.numerics.signAwareF32UlpDistribution.find(
            (candidate) => candidate.label === bin.label,
          )!.count,
        0,
      ));
    }
    expect(aggregateBins).toEqual(
      result.correctness.numerical.signAwareF32UlpBinCounts,
    );
    expect(aggregate).toMatchObject({
      count: result.correctness.numerical.count,
      differingCount: result.correctness.numerical.differingCount,
      signedMeanError: result.correctness.numerical.signedMeanError,
      meanAbsoluteError: result.correctness.numerical.meanAbsoluteError,
      rmsError: result.correctness.numerical.rmsError,
      controlRms: result.correctness.numerical.controlRms,
      nrmse: result.correctness.numerical.nrmse,
      snrDecibels: result.correctness.numerical.snrDecibels,
      pearsonCorrelation: result.correctness.numerical.pearsonCorrelation,
      maximumAbsoluteError: result.correctness.numerical.maximumAbsoluteError,
      relativeMaximumError: result.correctness.numerical.relativeMaximumError,
      signedZeroDifferenceCount: result.correctness.numerical.signedZeroDifferenceCount,
      thresholds: result.correctness.numerical.thresholds,
      passed: true,
    });

    const collected = Object.fromEntries(SHAPE_IDS.map((id) => [
      id,
      { A: [] as number[], B: [] as number[], C: [] as number[] },
    ])) as Record<string, Record<Arm, number[]>>;
    let cursor = 0;
    let previousFence = -Infinity;
    for (let roundIndex = 0; roundIndex < 6; roundIndex += 1) {
      const rotatedShapeOrder = [0, 1, 2, 3].map(
        (index) => (index + roundIndex) % 4,
      );
      for (const shapeIndex of rotatedShapeOrder) {
        for (const [armPosition, arm] of ARM_ORDERS[roundIndex]!.entries()) {
          const sample = receipt.timing.dispatchSamples[cursor++]!;
          expect(sample).toMatchObject({
            roundIndex,
            rotatedShapeOrder,
            shapeIndex,
            shapeId: SHAPE_IDS[shapeIndex],
            armOrder: ARM_ORDERS[roundIndex],
            armPosition,
            arm,
            commandBufferCount: 1,
            dispatchCount: 1,
            matchingQueueCompletionFenceCount: 1,
          });
          expect(sample.submitAtPerformanceMilliseconds).toBeGreaterThanOrEqual(
            previousFence,
          );
          expect(
            sample.fenceAtPerformanceMilliseconds -
              sample.submitAtPerformanceMilliseconds,
          ).toBe(sample.wallMilliseconds);
          previousFence = sample.fenceAtPerformanceMilliseconds;
          collected[sample.shapeId]![arm].push(sample.wallMilliseconds);
        }
      }
    }
    expect(cursor).toBe(72);
    for (const stratum of receipt.timing.strata) {
      expect(stratum.samples).toEqual(collected[stratum.id]);
      for (const arm of ["A", "B", "C"] as const) {
        expect(median6(stratum.samples[arm])).toBe(stratum.medians[arm]);
      }
      expect(stratum.medians.C).toBeGreaterThan(stratum.medians.B);
      expect(stratum.cFasterThanB).toBe(false);
      expect(stratum.medians.B / stratum.medians.C).toBe(stratum.bToCSpeedup);
      expect(stratum.medians.A / stratum.medians.C).toBe(stratum.aToCSpeedup);
      expect(result.metrics.baseline.shapeMedianMilliseconds[stratum.id]).toBe(
        stratum.medians.B,
      );
      expect(result.metrics.candidate.shapeMedianMilliseconds[stratum.id]).toBe(
        stratum.medians.C,
      );
    }
    for (const arm of ["A", "B", "C"] as const) {
      expect(weightedScore(receipt.timing.strata, arm, false)).toBe(
        receipt.timing.completeDense.milliseconds[arm],
      );
      expect(weightedScore(receipt.timing.strata, arm, true)).toBe(
        receipt.timing.feedForward.milliseconds[arm],
      );
    }
    expect(receipt.timing.completeDense).toMatchObject({
      multiplicities: "4/2/2/1",
      bToCSpeedup: 0.5947464078891549,
      bToCThreshold: 1.25,
      aToCSpeedup: 0.7203039805137254,
      aToCThreshold: 1.55,
    });
    expect(receipt.timing.feedForward).toMatchObject({
      multiplicities: "0/0/2/1",
      bToCSpeedup: 0.5524747348537575,
      bToCThreshold: 1.25,
    });
    expect(receipt.timing.completeDense.bToCSpeedup).toBeLessThan(
      receipt.timing.completeDense.bToCThreshold,
    );
    expect(receipt.timing.completeDense.aToCSpeedup).toBeLessThan(
      receipt.timing.completeDense.aToCThreshold,
    );
    expect(receipt.timing.feedForward.bToCSpeedup).toBeLessThan(
      receipt.timing.feedForward.bToCThreshold,
    );

    const thermal = thermalBytes.toString("utf8").trim().split("\n")
      .map((line) => JSON.parse(line) as ThermalObservation);
    expect(thermal).toHaveLength(109);
    expect(thermal.every((entry, index) =>
      entry.index === index &&
      entry.level === 0 &&
      (index === 0 ||
        entry.epochMilliseconds > thermal[index - 1]!.epochMilliseconds)
    )).toBe(true);
    const gate = thermal.filter((entry) =>
      entry.epochMilliseconds >=
        receipt.protocol.thermal.startedAtEpochMilliseconds &&
      entry.epochMilliseconds <=
        receipt.protocol.thermal.completedAtEpochMilliseconds
    );
    expect(gate).toHaveLength(93);
    const maximumPollGap = Math.max(...gate.slice(1).map((entry, index) =>
      entry.epochMilliseconds - gate[index]!.epochMilliseconds
    ));
    const maximumTracePollGap = Math.max(...thermal.slice(1).map(
      (entry, index) =>
        entry.epochMilliseconds - thermal[index]!.epochMilliseconds,
    ));
    expect(receipt.protocol.thermal).toMatchObject({
      source: "notifyutil-com.apple.system.thermalpressurelevel",
      durationMilliseconds: 92_864,
      observationCount: 93,
      pollMilliseconds: 1_000,
      maximumPollGapMilliseconds: 1_031,
      nonNominalObservationCount: 0,
      launchDelayMilliseconds: 752,
    });
    expect(
      receipt.protocol.thermal.completedAtEpochMilliseconds -
        receipt.protocol.thermal.startedAtEpochMilliseconds,
    ).toBe(receipt.protocol.thermal.durationMilliseconds);
    expect(maximumPollGap).toBe(
      receipt.protocol.thermal.maximumPollGapMilliseconds,
    );
    expect(maximumTracePollGap).toBe(1_089);
    expect(maximumTracePollGap).toBeLessThanOrEqual(1_250);
    expect(
      receipt.timing.timingStartedAtEpochMilliseconds -
        receipt.protocol.thermal.completedAtEpochMilliseconds,
    ).toBe(receipt.protocol.thermal.launchDelayMilliseconds);
    expect(receipt.timing.timingCompletedAtEpochMilliseconds).toBeLessThanOrEqual(
      receipt.cleanup.startedAtEpochMilliseconds,
    );
    expect(receipt.protocol.thermalTraceRequiredThroughEpochMilliseconds).toBe(
      receipt.cleanup.completedAtEpochMilliseconds,
    );
    expect(thermal.at(-1)!.epochMilliseconds).toBeGreaterThanOrEqual(
      receipt.cleanup.completedAtEpochMilliseconds,
    );
    expect(thermal.at(-1)!.epochMilliseconds).toBeGreaterThan(
      Date.parse(receipt.recordedAt),
    );

    expect(receipt.memory.maximumLiveBytes).toBe(311_749_632);
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

    expect(record).toContain("- Evidence: `negative`");
    expect(record).toContain("- Disposition: `abandoned`");
    expect(record).toContain("C was slower than B on every shape");
    expect(record).toContain("There was no unchanged-candidate retry");
    expect(record).toContain("no package-native layer gate");
    expect(ledger).toContain(
      "| OPT-0020 | DiT dense GEMM | Transposing OPT-0019's cooperative weight panel",
    );
    expect(ledger).toContain("| negative | abandoned |");
  });
});
