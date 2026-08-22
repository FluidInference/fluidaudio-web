import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

const RAW_ARTIFACTS = [
  [
    "compact-semantic-head-timing.json",
    "b261b892021328debfc9da1d26687844ba7f859944c7126e682ec7ffa0e0dddf",
  ],
  [
    "compact-semantic-head-timing-thermal.jsonl",
    "982951ebe45a4320307f73a954cdfcdf12a2421d8375f5226d1d8be3f9b539f8",
  ],
  [
    "compact-semantic-head-trajectory.json",
    "6468fb1d71e1d3f8e7309aa2217fc30dc2264b12f3f21a8b4c59d31bbc9486a8",
  ],
  [
    "corrective-failed-unstable-nan-oracle.json",
    "24ef921ae175859e3e2ef2447ee701b1ab768c9f7e3e6412b137bb595ad3debc",
  ],
  [
    "corrective-failed-nonnominal-thermal.jsonl",
    "94999454c7918aee5b8ac67c8c19147bf255950bf26ee7c280dead53a0641e12",
  ],
  [
    "corrective-failed-nonnominal-timing.json",
    "8a3bb770ad5f05f6b3f4cde9a2fdb4a3bf2373930d3d97a93a05ee15e6d119ad",
  ],
  [
    "corrective-failed-nonnominal-timing-thermal.jsonl",
    "09e2ed9492cce7279e5b415cace2fe64ee93c629714daed77aed2c817860d492",
  ],
  [
    "corrective-failed-nonnominal-rerun-timing.json",
    "089a3bae9f0a09cdc88bdcc75f1d073e6b54d3640fc32971826b432084d09415",
  ],
  [
    "corrective-failed-nonnominal-rerun-timing-thermal.jsonl",
    "f83abcda338261aad2822b6cc3401b2dff70162c444cebb5d7f876a709de6004",
  ],
  [
    "compact-semantic-head-corrective-trajectory.json",
    "03246b9f8c8654cf5f22159b5b8bbd45689e1d9cfc3a2bb6dec6c422af492c4c",
  ],
] as const;

interface TimingCase {
  readonly id: string;
  readonly AmedianMilliseconds: number;
  readonly BmedianMilliseconds: number;
  readonly CmedianMilliseconds: number;
  readonly CversusAPairedWins: number;
  readonly CversusBPairedWins: number;
}

interface TimingRun {
  readonly receipt: string;
  readonly thermalReceipt: string;
  readonly cases: readonly TimingCase[];
  readonly thermal: Record<string, unknown>;
  readonly correctness: Record<string, unknown>;
  readonly lifecycle: Record<string, unknown>;
}

interface Opt0012Candidate {
  readonly timingRuns: readonly TimingRun[];
  readonly sameImmutableByteReplay: {
    readonly pairCount: number;
    readonly candidateWins: number;
    readonly ties: number;
    readonly everyBeforeAfterByteSha256Equal: boolean;
    readonly medianMilliseconds: readonly Record<string, number | string>[];
    readonly minimumMedianSavingPercent: number;
    readonly maximumMedianSavingPercent: number;
  };
  readonly fp16Converter: Record<string, unknown>;
  readonly trajectory: Record<string, unknown>;
  readonly trajectoryCancellation: Record<string, unknown>;
  readonly trajectoryLifecycle: Record<string, unknown>;
  readonly scope: Record<string, unknown>;
}

describe("OPT-0012 final optimization result", () => {
  it("freezes positive benchmark-only evidence under the disclosed thermal rule", async () => {
    const text = await readFile(new URL(
      "../optimization/results/OPT-0012/result.json",
      import.meta.url,
    ), "utf8");
    const committed = parseOptimizationResultJson(text);
    const candidate = committed.metrics.candidate as unknown as Opt0012Candidate;

    expect(committed).toMatchObject({
      experimentId: "OPT-0012",
      riskClass: "exact",
      baselineCommit: "f5e8e5db0b88a9a44dc96b73319183114daf136a",
      candidateCommit: "f73380bceebdd5568d93908a67ff33cea2b7d8f0",
      identity: {
        modelManifestSha256:
          "c5b547cd08aa5e6d2971b2c9c84940b8af193f2e230ce689258ca81fcd292a3b",
        benchmarkHarnessCommit:
          "eb6d8cc4c1d1db8f4fc75f0cd836ceed8a12daa8",
        correctiveCoreSourceSha256:
          "53fb06aa1ec54c7dc4003731c7d360aac06110715e601451bee8a22256236fdf",
        thermalDecisionRuleCommit:
          "8f0d13ba8ec75eb892b30d0652ca39e2798b099f",
        browserVersion: "151.0.7922.138",
        osBuild: "macOS 26.5.2 build 25F84",
        machineModel: "Mac15,12",
        gpuCores: 10,
        memoryBytes: 17_179_869_184,
        productionIntegrationPerformed: false,
        webgpuAdapter: {
          info: {
            vendor: "apple",
            architecture: "metal-3",
            subgroupMinSize: 32,
            subgroupMaxSize: 32,
          },
          features: expect.arrayContaining(["shader-f16", "subgroups"]),
          limits: {
            maxBufferSize: 268_435_456,
            maxStorageBufferBindingSize: 268_435_456,
            maxComputeInvocationsPerWorkgroup: 256,
          },
        },
      },
      protocol: {
        thermalGateSeconds: 30,
        thermalPollMilliseconds: 1000,
        thermalPollToleranceMilliseconds: 250,
        samples: 12,
        balancedRunCount: 2,
        samplesPerArmPerCasePerRun: 6,
        trajectoryExecutionsPerArm: 2,
        oneOutstandingCommandBuffer: true,
        productionIntegrationPerformed: false,
      },
      correctness: {
        passed: true,
        listeningRequired: false,
        allRetainedFp16BitsExact: true,
        allReconstructedFullU32VectorsExact: true,
        allSamplingAndCursorBoundariesExact: true,
        allSixFinalTrajectoryReceiptsExact: true,
        everyArmTrajectorySelfRepeatExact: true,
        allActualPackageReadbacksHaveZeroFp16NaNs: true,
        exhaustiveFp16ConverterGatePassed: true,
        cancellationPassed: true,
        lifecyclePassed: true,
        runtimeEventCount: 0,
      },
      evidence: { conclusion: "positive" },
      disposition: { state: "benchmark-only" },
    });

    expect(candidate.timingRuns.map((run) => ({
      receipt: run.receipt,
      thermalReceipt: run.thermalReceipt,
      medians: run.cases.map((entry) => [
        entry.id,
        entry.AmedianMilliseconds,
        entry.BmedianMilliseconds,
        entry.CmedianMilliseconds,
      ]),
      wins: run.cases.map((entry) => [
        entry.CversusAPairedWins,
        entry.CversusBPairedWins,
      ]),
    }))).toEqual([
      {
        receipt: "corrective-failed-nonnominal-timing.json",
        thermalReceipt: "corrective-failed-nonnominal-timing-thermal.jsonl",
        medians: [
          ["semantic-m2-short", 294.94999998807907, 278.44999998807907, 269.80000001192093],
          ["semantic-m2-mid", 300.94999998807907, 302.60000002384186, 286.10000002384186],
          ["semantic-m2-long", 551.9499999880791, 520.5999999642372, 420.10000002384186],
        ],
        wins: [[6, 4], [5, 4], [6, 4]],
      },
      {
        receipt: "corrective-failed-nonnominal-rerun-timing.json",
        thermalReceipt:
          "corrective-failed-nonnominal-rerun-timing-thermal.jsonl",
        medians: [
          ["semantic-m2-short", 287.69999998807907, 262.94999998807907, 238.50000005960464],
          ["semantic-m2-mid", 357.94999998807907, 347.39999997615814, 320.25],
          ["semantic-m2-long", 384.80000001192093, 374.3499999642372, 334.39999997615814],
        ],
        wins: [[6, 6], [4, 4], [5, 5]],
      },
    ]);

    expect(candidate.timingRuns.map((run) => run.thermal)).toEqual([
      expect.objectContaining({
        nominalPreGateMilliseconds: 45_005,
        nominalPreGateObservationCount: 46,
        externalMaximumPollGapMilliseconds: 1050.9930829284713,
        firstNonNominalMillisecondsAfterTimedStart: 15_625,
        throughCleanupObservationCount: 309,
        throughCleanupNonNominalObservationCount: 237,
        allNominal: false,
        decisionUsefulUnderOwnerRule: true,
        releaseQualityThermalComparison: false,
      }),
      expect.objectContaining({
        nominalPreGateMilliseconds: 45_005,
        nominalPreGateObservationCount: 46,
        externalMaximumPollGapMilliseconds: 1032.6179170515388,
        firstNonNominalMillisecondsAfterTimedStart: 32_005,
        throughCleanupObservationCount: 273,
        throughCleanupNonNominalObservationCount: 189,
        allNominal: false,
        decisionUsefulUnderOwnerRule: true,
        releaseQualityThermalComparison: false,
      }),
    ]);
    for (const run of candidate.timingRuns) {
      expect(run.correctness).toEqual({
        actualPackageFp16NaNCensusCount: 54,
        actualPackageFp16NaNCount: 0,
        actualPackageFp16WordCount: 12_427_344,
        mismatchCount: 0,
        reconstructedFullU32ComparisonsPerCandidateArm: 1_737_632,
        retainedFp16ComparisonsPerCandidateArm: 384_002,
      });
      expect(run.lifecycle).toEqual({
        createdBufferCount: 7997,
        destroyedBufferCount: 7997,
        executorDestroyIdempotent: true,
        liveTrackedBufferCount: 0,
        mapCallCount: 362,
        runtimeEventCount: 0,
        unmapCallCount: 362,
      });
    }

    expect(candidate.sameImmutableByteReplay).toMatchObject({
      pairCount: 36,
      candidateWins: 36,
      ties: 0,
      everyBeforeAfterByteSha256Equal: true,
      minimumMedianSavingPercent: 41.79254779488781,
      maximumMedianSavingPercent: 45.05672612202935,
    });
    expect(candidate.sameImmutableByteReplay.medianMilliseconds).toHaveLength(6);
    expect(candidate.sameImmutableByteReplay.medianMilliseconds.map((entry) =>
      entry.savingPercent
    )).toEqual([
      44.93201480216297,
      41.79254779488781,
      44.37467288091364,
      42.87661912898973,
      45.05672612202935,
      43.421052686124156,
    ]);

    expect(candidate.fp16Converter).toMatchObject({
      completeDomainWordCount: 65_536,
      nonNaNWordCount: 63_490,
      nanWordCount: 2046,
      candidateU32LeSha256:
        "b636c5716ff84d972782faf02d0194cb8951526bea4cc487082feb47b1860ddf",
      candidateCrossPassMismatchCount: 0,
      acceptedStableNanEnvelope: true,
      pairCount: 12,
      candidateWins: 12,
      minimumMedianSavingPercent: 98.90109863958031,
      maximumMedianSavingPercent: 99.08675804679238,
    });
    expect(candidate.trajectory).toEqual({
      actualPackageFp16NaNCensusCount: 906,
      actualPackageFp16NaNCount: 0,
      actualPackageFp16WordCount: 207_991_224,
      allSixPerDrawReceiptsExact: true,
      armCount: 3,
      codeCount: 150,
      everyArmSelfRepeatExact: true,
      executionsPerArm: 2,
      finalDrawEnd: "260",
      firstDrawIndex: "109",
      receipt: "compact-semantic-head-corrective-trajectory.json",
      semanticCodeSha256:
        "08c69f3d598bea591754948b831684e67e879fca730b11ba796e7547a0f798fb",
      serializedAudioCodeTextSha256:
        "f9aec6e269424585028aae47ad5582fbda5139ef9961642774f0ce0b4c58f62e",
      terminalEosCount: 1,
      timingComparisonAuthority: false,
    });
    expect(candidate.trajectoryCancellation).toEqual({
      callbackCountAfter: 9,
      callbackCountBefore: 9,
      cursorAfter: "118",
      cursorBefore: "118",
      finalizationCount: 0,
      noLaterSubmitDrainMapAllocationCursorCallbackOrFinalization: true,
      rejectedInvocationCacheNotPublished: true,
      rejectionName: "AbortError",
      sampleCallCountDuringCancelledInvocation: 0,
    });
    expect(candidate.trajectoryLifecycle).toEqual({
      createdBufferCount: 4954,
      destroyedBufferCount: 4954,
      executorDestroyIdempotent: true,
      liveTrackedBufferCount: 0,
      mapCallCount: 2869,
      postDestroyPrefillAndDecodeRejected: true,
      runtimeEventCount: 0,
      unmapCallCount: 2869,
    });
    expect(candidate.scope).toEqual({
      directGenerationTargetMeasured: false,
      fullSongExecuted: false,
      listeningRequiredForExactBenchmarkMechanism: false,
      plannerOptimizationParkedUntilDirectTargetPriorityAllows: true,
      productionIntegrationPerformed: false,
    });

    expect(committed.metrics.delta).toEqual({
      AversusBConclusion: "marginal-and-noisy",
      AversusBMedianComparisonCount: 6,
      AversusBMedianWinsForB: 5,
      AversusBPairedComparisonCount: 36,
      AversusBPairedWinsForB: 24,
      attributableCversusBMedianComparisonCount: 6,
      attributableCversusBMedianWins: 6,
      attributableCversusBPairedComparisonCount: 36,
      attributableCversusBPairedWins: 27,
      combinedCversusAMedianComparisonCount: 6,
      combinedCversusAMedianWins: 6,
      combinedCversusAPairedComparisonCount: 36,
      combinedCversusAPairedWins: 32,
      directGenerationTargetContributionMeasured: false,
      productSpeedupClaim: null,
      productionIntegrationPerformed: false,
      sameImmutableByteReplayComparisonCount: 36,
      sameImmutableByteReplayWins: 36,
      thermalCaveat: expect.stringContaining("neither continuously all-nominal"),
    });

    expect(committed.artifacts).toEqual(RAW_ARTIFACTS.map(([name, sha256]) => ({
      location: `optimization/artifacts/OPT-0012/raw/${name}`,
      sha256,
    })));
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("closes the record and ledger without claiming integration or release thermals", async () => {
    const [record, ledger] = await Promise.all([
      readFile(new URL(
        "../optimization/experiments/OPT-0012-compact-semantic-head-sampling.md",
        import.meta.url,
      ), "utf8"),
      readFile(new URL("../optimization/LEDGER.md", import.meta.url), "utf8"),
    ]);

    expect(record).toContain("- Evidence: `positive`");
    expect(record).toContain("- Disposition: `benchmark-only`");
    expect(record).toContain("384,002 retained FP16");
    expect(record).toContain("207,991,224 binary16 words with zero");
    expect(record).toContain("32/36 same-round comparisons");
    expect(record).toContain("27/36 same-round comparisons");
    expect(record).toContain("all 36/36 immutable-byte pairs");
    expect(record).toContain("41.8–45.1%");
    expect(record).toContain("all 12/12 late balanced pairs");
    expect(record).toContain("represented as\ncontinuously all-nominal or release-quality");
    expect(record).toContain("no production path was changed");
    expect(record).toContain("park this planner integration");
    expect(record).toContain("../results/OPT-0012/result.json");

    const ledgerRow = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0012 |"),
    );
    expect(ledgerRow).toContain("| positive | benchmark-only |");
    expect(ledgerRow).toContain("all six medians and 32/36");
    expect(ledgerRow).toContain("all six medians and 27/36");
    expect(ledgerRow).toContain("36/36 with 41.8–45.1%");
    expect(ledgerRow).toContain("not continuously all-nominal or release-quality");
    expect(ledgerRow).toContain("No production integration occurred");
    expect(ledgerRow).toContain("[result](results/OPT-0012/result.json)");
  });
});
