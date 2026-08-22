import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";
import { aceFp16VaeConv1dSubgroupWgsl } from
  "../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import { aceOpt0014VaeConv1dPackedKioRepackWgsl } from
  "../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-subgroup.js";
import { aceOpt0017VaeConv1dCooperativeDot4Wgsl } from
  "../src/webgpu/kernels/vae-conv1d-fp16-cooperative-dot4.js";
import { buildOpt0017C300Topology } from
  "./browser/opt-0017-vae-k7-cooperative-dot4-ab.js";

interface NumericRange {
  readonly minimum: number;
  readonly maximum: number;
}

interface NumericalSummary {
  readonly comparedValueCount: number;
  readonly maximumAbsoluteError: number;
  readonly meanError: number;
  readonly rmsError: number;
  readonly nrmse: number;
  readonly snrDb: number | "Infinity";
  readonly pearson: number;
  readonly relativeMaximumAbsoluteError: number;
  readonly signedZeroDifferenceCount: number;
  readonly numericOutputRanges: Readonly<{
    control: NumericRange;
    candidate: NumericRange;
  }>;
  readonly fp16UlpDistribution: Readonly<Record<string, number>>;
}

interface ArtifactProbe {
  readonly count: number;
  readonly armSha256: Readonly<Record<"fixed32" | "cooperativeDot4", string>>;
  readonly candidateExecutionSha256: readonly string[];
  readonly comparedCandidateControlValues: number;
  readonly deterministicCandidateRerun: boolean;
  readonly numerical: NumericalSummary;
}

interface RepackCase {
  readonly label: string;
  readonly uniqueU16WordCount: number;
  readonly comparedU16WordCount: number;
  readonly packedBytes: number;
  readonly repackWorkgroups: number;
  readonly sha256: string;
  readonly mismatchCount: number;
  readonly qNaNPrefillCount: number;
  readonly redzonesUntouched: boolean;
  readonly deterministicRerun: boolean;
}

interface TimingStratum {
  readonly weight: number;
  readonly samples: Readonly<Record<
    "fixed32" | "cooperativeDot4",
    readonly number[]
  >>;
  readonly medians: Readonly<Record<"fixed32" | "cooperativeDot4", number>>;
}

interface Opt0017Artifact {
  readonly schema: string;
  readonly status: string;
  readonly experimentId: string;
  readonly identity: {
    readonly sourceAuthority: Readonly<{
      coreCommit: string;
      fixed32CoreSourceSha256: string;
      packed16x64CoreSourceSha256: string;
      cooperativeCoreSourceSha256: string;
      generatedShaderCountPerArm: number;
      fixed32GeneratedAggregateSha256: string;
      repackGeneratedAggregateSha256: string;
      cooperativeGeneratedAggregateSha256: string;
    }>;
  };
  readonly topology: Readonly<{
    operationCount: number;
    exactRangeCount: number;
    outputWordCount: number;
    representativeRangeWeight: number;
    omittedConv1RangeWeight: number;
    omittedShippedFinalConv2RangeWeight: number;
  }>;
  readonly correctness: Readonly<{
    probeCount: number;
    executionsPerProbe: number;
    comparedU16WordCount: number;
    deterministicCandidateRerunHashes: boolean;
    qNaNPrefillCompleteWrites: boolean;
    guardsAndAdjacentCanariesUntouched: boolean;
    thresholds: Readonly<{
      nrmseMaximum: number;
      snrMinimumDb: number;
      pearsonMinimum: number;
      relativeMaximumAbsoluteErrorMaximum: number;
    }>;
    cases: readonly Readonly<{ probes: readonly ArtifactProbe[] }>[];
    aggregate: NumericalSummary;
  }>;
  readonly repack: Readonly<{
    packedPayloadBytes: number;
    incrementalMemoryCostIfIntegratedBytes: number;
    all16Correctness: Readonly<{
      operationCount: number;
      selectedOperationCount: number;
      selectedUniqueU16WordCount: number;
      selectedRepackWorkgroupCount: number;
      uniqueU16WordCount: number;
      comparedU16WordCount: number;
      repackWorkgroupCount: number;
      mismatchCount: number;
      qNaNPrefillCompleteWrites: boolean;
      redzonesUntouched: boolean;
      deterministicRerunHashes: boolean;
      cases: readonly RepackCase[];
    }>;
  }>;
  readonly representativeTiming: Readonly<{
    representedRangeWeight: number;
    omittedConv1RangeWeight: number;
    omittedFinalConv2RangeWeight: number;
    fixed32WeightedMilliseconds: number;
    candidateWeightedMilliseconds: Readonly<{ cooperativeDot4: number }>;
    speedups: Readonly<{ cooperativeDot4: number }>;
    strata: readonly TimingStratum[];
  }>;
  readonly selection: Readonly<{
    status: string;
    selectedWinner: string | null;
    qualifyingArms: readonly string[];
    threshold: number;
  }>;
  readonly fullSequenceAllocationPerformed: boolean;
  readonly fullSequence: unknown;
  readonly decision: Readonly<{
    disposition: string;
    productionIntegrationAuthorized: boolean;
  }>;
  readonly protocol: Readonly<{
    thermal: Readonly<{
      startedAtEpochMilliseconds: number;
      completedAtEpochMilliseconds: number;
      durationMilliseconds: number;
      observationCount: number;
      maximumPollGapMilliseconds: number;
      nonNominalObservationCount: number;
      launchDelayMilliseconds: number;
    }>;
    numericalAndAll16RepackCompletedBeforeThermalGate: boolean;
    representativeSamplesPerArmPerTier: number;
    representativeQualifyingSpeedup: number;
    fullSequenceQualifyingSpeedup: number;
    unchangedThermalRetryPerformed: boolean;
  }>;
  readonly cleanup: Readonly<{
    createdBufferCount: number;
    destroyedBufferCount: number;
    liveBufferCount: number;
    liveBytes: number;
    maximumLiveBytes: number;
    idempotent: boolean;
    deviceDestroyed: boolean;
  }>;
}

const ARTIFACT_URL = new URL(
  "../optimization/artifacts/OPT-0017/raw/cooperative-dot4-k7-ab.json",
  import.meta.url,
);
const RESULT_URL = new URL(
  "../optimization/results/OPT-0017/result.json",
  import.meta.url,
);
const sha256 = (bytes: Uint8Array | string): string =>
  createHash("sha256").update(bytes).digest("hex");
const median4 = (samples: readonly number[]): number => {
  expect(samples).toHaveLength(4);
  const sorted = [...samples].sort((left, right) => left - right);
  return (sorted[1]! + sorted[2]!) / 2;
};

function assertNumericalThresholds(
  summary: NumericalSummary,
  thresholds: Opt0017Artifact["correctness"]["thresholds"],
): void {
  for (const range of Object.values(summary.numericOutputRanges)) {
    expect(Number.isFinite(range.minimum)).toBe(true);
    expect(Number.isFinite(range.maximum)).toBe(true);
    expect(range.minimum).toBeLessThanOrEqual(range.maximum);
  }
  expect(summary.nrmse).toBeLessThanOrEqual(thresholds.nrmseMaximum);
  if (summary.snrDb === "Infinity") {
    expect(summary.rmsError).toBe(0);
  } else {
    expect(summary.snrDb).toBeGreaterThanOrEqual(thresholds.snrMinimumDb);
  }
  expect(summary.pearson).toBeGreaterThanOrEqual(thresholds.pearsonMinimum);
  expect(summary.relativeMaximumAbsoluteError).toBeLessThanOrEqual(
    thresholds.relativeMaximumAbsoluteErrorMaximum,
  );
}

describe("OPT-0017 optimization result", () => {
  it("commits canonical reordered-rounding but negative evidence", async () => {
    const text = await readFile(RESULT_URL, "utf8");
    const committed = parseOptimizationResultJson(text);

    expect(committed).toMatchObject({
      experimentId: "OPT-0017",
      riskClass: "reordered-rounding",
      baselineCommit: "36608b857827b2b1d31ac91bf5cca9639fb0b9ed",
      candidateCommit: "b83f4fe94d56787ddb980629ea6f41804543ca69",
      identity: {
        benchmarkHarnessCommit: "c34efbb67017c679c7932eed1df783254af17631",
        coreCommit: "b83f4fe94d56787ddb980629ea6f41804543ca69",
        cooperativeCoreSourceSha256:
          "83987aa9b16e05a5b6f45c25ebfe33ed08bfb82ae94bd9dfb4fb624c625407b8",
        productionPackageLoaded: false,
        productionRuntimeSelected: false,
        productionIntegrationPerformed: false,
      },
      protocol: {
        samples: 4,
        thermalGateSeconds: 30,
        thermal: {
          durationMilliseconds: 30_198,
          observationCount: 31,
          maximumPollGapMilliseconds: 1_015,
          nonNominalObservationCount: 0,
          launchDelayMilliseconds: 81,
        },
        unchangedThermalRetryPerformed: false,
        representativeQualificationThreshold: 1.75,
        fullSequenceQualificationThreshold: 2,
      },
      metrics: {
        baseline: {
          exactC300OperationCount: 16,
          exactC300RangeCount: 2_399,
          representativeRangeWeight: 2_397,
          weightedRepresentativeMilliseconds: 13_761.599894106388,
        },
        candidate: {
          weightedRepresentativeMilliseconds: 26_183.699956297874,
          qualificationThreshold: 1.75,
          qualified: false,
          fullPhaseExecuted: false,
          fullSequenceAllocationPerformed: false,
          productionIntegrationPerformed: false,
        },
        delta: {
          representativeSpeedup: 0.5255788875168637,
          candidateSlowdownFactor: 1.9026639458913086,
          representativeSavingMilliseconds: -12_422.100062191486,
          stopRuleFired: true,
          selectedWinner: null,
          fullSequenceSpeedup: null,
          observed12SecondProductSpeedup: null,
          observed180SecondProductSpeedup: null,
          observedIntegratedDecoderSpeedup: null,
          under60SecondProjection: null,
          productionIntegrationPerformed: false,
        },
      },
      correctness: {
        passed: true,
        listeningRequired: false,
        listeningDecision: null,
        numerical: {
          passed: true,
          probeCount: 12,
          executionsPerProbe: 2,
          aggregate: {
            comparedValueCount: 2_392_064,
            nrmse: 0.000009066605130144652,
            snrDb: 100.85110596605332,
            pearson: 0.9999999999588972,
          },
        },
        repack: {
          operationCount: 16,
          uniqueU16WordCount: 30_507_008,
          comparedU16WordCount: 40_255_488,
          mismatchCount: 0,
        },
        fullPhase: {
          executed: false,
          fullSequenceAllocationPerformed: false,
        },
        cleanup: {
          createdBufferCount: 83,
          destroyedBufferCount: 83,
          liveBufferCount: 0,
          liveBytes: 0,
          deviceDestroyed: true,
        },
      },
      evidence: { conclusion: "negative" },
      disposition: { state: "abandoned" },
    });
    expect(committed.artifacts).toEqual([{
      location:
        "optimization/artifacts/OPT-0017/raw/cooperative-dot4-k7-ab.json",
      sha256:
        "903d810d5c0ea4f0c411587cdeffbc90a462690bb9cd570a1665b619ec7eebb2",
    }]);
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("recomputes source, numerical, repack, timing, stop, and cleanup facts", async () => {
    const bytes = await readFile(ARTIFACT_URL);
    expect(bytes.byteLength).toBe(52_616);
    expect(sha256(bytes)).toBe(
      "903d810d5c0ea4f0c411587cdeffbc90a462690bb9cd570a1665b619ec7eebb2",
    );
    const artifact = JSON.parse(bytes.toString("utf8")) as Opt0017Artifact;
    expect(artifact).toMatchObject({
      schema: "ace-opt-0017-vae-k7-cooperative-dot4-ab-v1",
      status: "passed",
      experimentId: "OPT-0017",
      topology: {
        operationCount: 16,
        exactRangeCount: 2_399,
        outputWordCount: 424_550_400,
        representativeRangeWeight: 2_397,
        omittedConv1RangeWeight: 2,
        omittedShippedFinalConv2RangeWeight: 5,
      },
    });

    const authority = artifact.identity.sourceAuthority;
    const [fixed32Source, packedSource, cooperativeSource] = await Promise.all([
      readFile(new URL(
        "../src/webgpu/kernels/vae-conv1d-fp16-subgroup.ts",
        import.meta.url,
      )),
      readFile(new URL(
        "../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-subgroup.ts",
        import.meta.url,
      )),
      readFile(new URL(
        "../src/webgpu/kernels/vae-conv1d-fp16-cooperative-dot4.ts",
        import.meta.url,
      )),
    ]);
    expect(sha256(fixed32Source)).toBe(authority.fixed32CoreSourceSha256);
    expect(sha256(packedSource)).toBe(authority.packed16x64CoreSourceSha256);
    expect(sha256(cooperativeSource)).toBe(
      authority.cooperativeCoreSourceSha256,
    );
    expect(authority.coreCommit).toBe(
      "b83f4fe94d56787ddb980629ea6f41804543ca69",
    );

    const topology = buildOpt0017C300Topology();
    expect(topology).toHaveLength(authority.generatedShaderCountPerArm);
    const generatedHash = (shaders: readonly string[]): string =>
      sha256(shaders.join("\n\u0000\n"));
    expect(generatedHash(topology.map((operation) =>
      `${operation.label}\n${aceFp16VaeConv1dSubgroupWgsl(
        operation.shape,
        true,
        operation.outputStorage,
      )}`
    ))).toBe(authority.fixed32GeneratedAggregateSha256);
    expect(generatedHash(topology.map((operation) =>
      `${operation.label}\n${aceOpt0014VaeConv1dPackedKioRepackWgsl(
        operation.shape,
      )}`
    ))).toBe(authority.repackGeneratedAggregateSha256);
    expect(generatedHash(topology.map((operation) =>
      `${operation.label}\n${aceOpt0017VaeConv1dCooperativeDot4Wgsl(
        operation.shape,
        true,
        operation.outputStorage,
      )}`
    ))).toBe(authority.cooperativeGeneratedAggregateSha256);

    const correctness = artifact.correctness;
    const probes = correctness.cases.flatMap(({ probes }) => probes);
    expect(probes).toHaveLength(correctness.probeCount);
    for (const probe of probes) {
      expect(probe.candidateExecutionSha256).toHaveLength(
        correctness.executionsPerProbe,
      );
      expect(new Set(probe.candidateExecutionSha256)).toEqual(
        new Set([probe.armSha256.cooperativeDot4]),
      );
      expect(probe.deterministicCandidateRerun).toBe(true);
      expect(probe.comparedCandidateControlValues).toBe(
        probe.count * correctness.executionsPerProbe,
      );
      expect(probe.numerical.comparedValueCount).toBe(
        probe.comparedCandidateControlValues,
      );
      assertNumericalThresholds(probe.numerical, correctness.thresholds);
    }
    expect(probes.reduce(
      (sum, probe) => sum + probe.comparedCandidateControlValues,
      0,
    )).toBe(correctness.comparedU16WordCount);
    expect(correctness.comparedU16WordCount).toBe(2_392_064);
    expect(correctness).toMatchObject({
      deterministicCandidateRerunHashes: true,
      qNaNPrefillCompleteWrites: true,
      guardsAndAdjacentCanariesUntouched: true,
    });
    assertNumericalThresholds(correctness.aggregate, correctness.thresholds);
    expect(correctness.aggregate).toMatchObject({
      comparedValueCount: 2_392_064,
      maximumAbsoluteError: 0.000030517578125,
      meanError: -2.667685007446841e-10,
      rmsError: 1.6688909076031648e-7,
      nrmse: 0.000009066605130144652,
      snrDb: 100.85110596605332,
      pearson: 0.9999999999588972,
      relativeMaximumAbsoluteError: 0.0005431830526887561,
      signedZeroDifferenceCount: 0,
      numericOutputRanges: {
        control: { minimum: -0.0537109375, maximum: 0.056182861328125 },
        candidate: { minimum: -0.0537109375, maximum: 0.056182861328125 },
      },
      fp16UlpDistribution: { "0": 2_390_380, "1": 1_684 },
    });
    for (const arm of ["control", "candidate"] as const) {
      expect(correctness.aggregate.numericOutputRanges[arm]).toEqual({
        minimum: Math.min(...probes.map((probe) =>
          probe.numerical.numericOutputRanges[arm].minimum
        )),
        maximum: Math.max(...probes.map((probe) =>
          probe.numerical.numericOutputRanges[arm].maximum
        )),
      });
    }
    const ulpTotals = new Map<string, number>();
    for (const probe of probes) {
      for (const [ulp, count] of Object.entries(
        probe.numerical.fp16UlpDistribution,
      )) ulpTotals.set(ulp, (ulpTotals.get(ulp) ?? 0) + count);
    }
    expect(Object.fromEntries(ulpTotals)).toEqual(
      correctness.aggregate.fp16UlpDistribution,
    );
    expect(probes.reduce((sum, probe) =>
      sum + probe.numerical.signedZeroDifferenceCount, 0))
      .toBe(correctness.aggregate.signedZeroDifferenceCount);

    const repack = artifact.repack.all16Correctness;
    const sumRepack = (field: keyof Pick<
      RepackCase,
      | "uniqueU16WordCount"
      | "comparedU16WordCount"
      | "packedBytes"
      | "repackWorkgroups"
    >): number => repack.cases.reduce((sum, entry) => sum + entry[field], 0);
    expect(repack.cases).toHaveLength(repack.operationCount);
    expect(sumRepack("uniqueU16WordCount")).toBe(repack.uniqueU16WordCount);
    expect(sumRepack("comparedU16WordCount")).toBe(
      repack.comparedU16WordCount,
    );
    expect(sumRepack("repackWorkgroups")).toBe(repack.repackWorkgroupCount);
    expect(sumRepack("packedBytes")).toBe(artifact.repack.packedPayloadBytes);
    const reruns = repack.cases.filter(({ deterministicRerun }) =>
      deterministicRerun
    );
    expect(reruns).toHaveLength(repack.selectedOperationCount);
    expect(reruns.reduce((sum, entry) =>
      sum + entry.uniqueU16WordCount, 0)).toBe(repack.selectedUniqueU16WordCount);
    expect(reruns.reduce((sum, entry) =>
      sum + entry.repackWorkgroups, 0)).toBe(
        repack.selectedRepackWorkgroupCount,
      );
    expect(Object.fromEntries(reruns.map(({ label, sha256 }) =>
      [label, sha256]
    ))).toEqual({
      "block-0-res-1-conv1":
        "a5b263af7413fdcddb14d0cf4468360074e614f8cab1be22b4ce39e12633059b",
      "block-1-res-2-conv1":
        "ae8c34cb62c1b3c6a512328640377b7980971d5b8a508bc7bbb7403c6ed11564",
      "block-2-res-1-conv1":
        "baa781e62b263e89486a05b566635652db893da0064e6d9e02f9d01b97705c19",
      "block-4-res-3-conv1":
        "fd9eb42c7e1a36829ea952af49de1c20e2755354e9263bab53a04233aa2c8f3f",
    });
    expect(repack).toMatchObject({
      uniqueU16WordCount: 30_507_008,
      comparedU16WordCount: 40_255_488,
      repackWorkgroupCount: 59_584,
      mismatchCount: 0,
      qNaNPrefillCompleteWrites: true,
      redzonesUntouched: true,
      deterministicRerunHashes: true,
    });
    expect(repack.cases.every((entry) => entry.mismatchCount === 0 &&
      entry.qNaNPrefillCount === 0 && entry.redzonesUntouched)).toBe(true);

    const timing = artifact.representativeTiming;
    let fixedWeighted = 0;
    let candidateWeighted = 0;
    for (const stratum of timing.strata) {
      expect(median4(stratum.samples.fixed32)).toBe(stratum.medians.fixed32);
      expect(median4(stratum.samples.cooperativeDot4)).toBe(
        stratum.medians.cooperativeDot4,
      );
      fixedWeighted += stratum.medians.fixed32 * stratum.weight;
      candidateWeighted += stratum.medians.cooperativeDot4 * stratum.weight;
    }
    expect(timing.strata.reduce((sum, entry) => sum + entry.weight, 0)).toBe(
      timing.representedRangeWeight,
    );
    expect(fixedWeighted).toBe(timing.fixed32WeightedMilliseconds);
    expect(candidateWeighted).toBe(
      timing.candidateWeightedMilliseconds.cooperativeDot4,
    );
    expect(fixedWeighted / candidateWeighted).toBe(
      timing.speedups.cooperativeDot4,
    );
    expect(timing.speedups.cooperativeDot4).toBeLessThan(
      artifact.protocol.representativeQualifyingSpeedup,
    );

    expect(artifact.selection).toEqual({
      status: "negative-stop-no-representative-qualifier",
      selectedWinner: null,
      qualifyingArms: [],
      threshold: 1.75,
    });
    expect(artifact.fullSequenceAllocationPerformed).toBe(false);
    expect(artifact.fullSequence).toBeNull();
    expect(artifact.decision).toEqual({
      disposition: "negative-stop-no-representative-qualifier",
      productionIntegrationAuthorized: false,
    });
    expect(artifact.protocol).toMatchObject({
      numericalAndAll16RepackCompletedBeforeThermalGate: true,
      representativeSamplesPerArmPerTier: 4,
      representativeQualifyingSpeedup: 1.75,
      fullSequenceQualifyingSpeedup: 2,
      unchangedThermalRetryPerformed: false,
    });
    expect(artifact.protocol.thermal.durationMilliseconds).toBe(
      artifact.protocol.thermal.completedAtEpochMilliseconds -
        artifact.protocol.thermal.startedAtEpochMilliseconds,
    );
    expect(artifact.protocol.thermal).toMatchObject({
      durationMilliseconds: 30_198,
      observationCount: 31,
      maximumPollGapMilliseconds: 1_015,
      nonNominalObservationCount: 0,
      launchDelayMilliseconds: 81,
    });
    expect(artifact.cleanup).toEqual({
      createdBufferCount: 83,
      destroyedBufferCount: 83,
      liveBufferCount: 0,
      liveBytes: 0,
      maximumLiveBytes: 580_667_392,
      idempotent: true,
      deviceDestroyed: true,
    });
  });

  it("closes the experiment without a full-sequence or product claim", async () => {
    const [record, ledger] = await Promise.all([
      readFile(new URL(
        "../optimization/experiments/OPT-0017-vae-k7-cooperative-dot4.md",
        import.meta.url,
      ), "utf8"),
      readFile(new URL("../optimization/LEDGER.md", import.meta.url), "utf8"),
    ]);
    const row = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0017 |")
    );
    expect(row).toContain("| negative | abandoned |");
    expect(row).toContain("[result](results/OPT-0017/result.json)");
    expect(row).toContain("0.52558x");
    expect(row).toContain("full phase was skipped");
    expect(row).toContain("not integrated");

    expect(record).toContain("- Evidence: `negative`");
    expect(record).toContain("- Disposition: `abandoned`");
    expect(record).toContain("`40,255,488` raw-U16 comparisons");
    expect(record).toContain("`2,392,064`\nU16 values");
    expect(record).toContain("only\n`0.5255788875168637x`");
    expect(record).toContain("full-sequence controls/dispatch allocation was not performed");
    expect(record).toContain("all-16 weight buffers had already");
    expect(record).toContain("Production integration was not authorized or performed");
    expect(record).toContain("No full-C300 timing, 12-second production generation");
    expect(record).toContain("[result.json](../results/OPT-0017/result.json)");
  });
});
