import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  buildOpt0079ShapeSpecs,
  buildOpt0079TimingRounds,
  classifyOpt0079PrimitiveDisposition,
  parseOpt0079ThermalCompletion,
  parseOpt0079ThermalLaunchGate,
  summarizeOpt0079Timing,
  type Opt0079TimestampSample,
  type Opt0079TimingInput,
} from "./browser/opt-0079-dit-dense-decoded-half-tile.js";
import { planAceOpt0009DenseGemm } from
  "../src/webgpu/kernels/dit-dense-fp16.js";
import { planAceOpt0079DenseDecodedHalfTile } from
  "../src/webgpu/kernels/dit-dense-fp16-decoded-half-tile.js";

const HARNESS_SOURCE = source(
  "./browser/opt-0079-dit-dense-decoded-half-tile.ts",
);
const HTML_SOURCE = source(
  "./browser/opt-0079-dit-dense-decoded-half-tile.html",
);

describe("OPT-0079 target-browser dense decoded-half-tile contract", () => {
  it("pins every exact M2250 shape, 4/2/2/1 and FF scores, and edge families", () => {
    expect(buildOpt0079ShapeSpecs()).toEqual([
      { id: "h-h", shape: { rows: 2_250, inner: 2_048, columns: 2_048 },
        productionMultiplicity: 4, feedForwardMultiplicity: 0,
        fixtureIds: ["signed-zero", "subnormal-normal-boundary"], ordinal: 0 },
      { id: "h-1024", shape: { rows: 2_250, inner: 2_048, columns: 1_024 },
        productionMultiplicity: 2, feedForwardMultiplicity: 0,
        fixtureIds: ["alternating-cancellation"], ordinal: 1 },
      { id: "h-6144", shape: { rows: 2_250, inner: 2_048, columns: 6_144 },
        productionMultiplicity: 2, feedForwardMultiplicity: 2,
        fixtureIds: ["maximum-finite-fp16-bounded"], ordinal: 2 },
      { id: "6144-h", shape: { rows: 2_250, inner: 6_144, columns: 2_048 },
        productionMultiplicity: 1, feedForwardMultiplicity: 1,
        fixtureIds: ["activation-rounding-boundary"], ordinal: 3 },
    ]);
  });

  it("reconciles M64/N128/K32 decoded-panel work and frozen totals", () => {
    let currentWorkgroups = 0;
    let candidateWorkgroups = 0;
    let currentMacs = 0;
    let candidateMacs = 0;
    let currentActivationBytes = 0;
    let currentWeightBytes = 0;
    let candidateActivationBytes = 0;
    let candidateWeightBytes = 0;
    let candidateOperandBytes = 0;
    for (const spec of buildOpt0079ShapeSpecs()) {
      const current = planAceOpt0009DenseGemm(spec.shape);
      const candidate = planAceOpt0079DenseDecodedHalfTile(spec.shape);
      expect(current).toMatchObject({ tileRows: 32, tileColumns: 256,
        tileInner: 32, workgroupSize: 128 });
      expect(candidate).toMatchObject({ tileRows: 64, tileColumns: 128,
        tileInner: 32, workgroupSize: 256, subgroupsPerWorkgroup: 8,
        rowsPerSubgroup: 8, columnsPerLane: 4, accumulatorsPerLane: 32,
        packedRecordsPerInnerTile: 512, packedRecordsPerLane: 2,
        decodedVectorsPerInnerTile: 1_024,
        decodedVectorsPerPackedRecord: 2,
        workgroupStorageBytes: 8_192, scheduledRows: 2_304 });
      expect(candidate.rowTiles).toBe(36);
      expect(candidate.columnTiles).toBe(current.columnTiles * 2);
      expect(candidate.physicalColumnTiles).toBe(current.columnTiles);
      expect(candidate.innerTiles).toBe(current.innerTiles);
      expect(candidate.validMultiplyAdds).toBe(
        spec.shape.rows * spec.shape.inner * spec.shape.columns,
      );
      expect(candidate.outputRanges[0]).toMatchObject({ firstOutput: 0,
        outputCount: spec.shape.rows * spec.shape.columns,
        firstWorkgroup: 0, workgroupCount: candidate.workgroupCount,
        multiplyAdds: candidate.scheduledMultiplyAdds });
      expect(candidate.packedWeightStorageShape)
        .toEqual(current.packedWeightStorageShape);
      currentWorkgroups += current.workgroupCount * spec.productionMultiplicity;
      candidateWorkgroups += candidate.workgroupCount *
        spec.productionMultiplicity;
      currentMacs += current.outputRanges[0]!.multiplyAdds *
        spec.productionMultiplicity;
      candidateMacs += candidate.scheduledMultiplyAdds *
        spec.productionMultiplicity;
      candidateActivationBytes += candidate.estimatedGlobalActivationBytes *
        spec.productionMultiplicity;
      candidateWeightBytes += candidate.estimatedGlobalWeightBytes *
        spec.productionMultiplicity;
      candidateOperandBytes += candidate.estimatedGlobalOperandBytes *
        spec.productionMultiplicity;
      currentActivationBytes += current.rowTiles * current.tileRows *
        current.inner * current.columnTiles * 4 *
        spec.productionMultiplicity;
      currentWeightBytes += current.weightElements * 2 * current.rowTiles *
        (current.workgroupSize / current.subgroupSize) *
        spec.productionMultiplicity;
    }
    expect(currentWorkgroups).toBe(6_816);
    expect(candidateWorkgroups).toBe(6_912);
    expect(currentMacs).toBe(133_412_421_632);
    expect(candidateMacs).toBe(135_291_469_824);
    expect(currentActivationBytes).toBe(2_084_569_088);
    expect(currentWeightBytes).toBe(33_353_105_408);
    expect(currentActivationBytes + currentWeightBytes)
      .toBe(35_437_674_496);
    expect(candidateActivationBytes).toBe(4_227_858_432);
    expect(candidateWeightBytes).toBe(4_227_858_432);
    expect(candidateOperandBytes).toBe(8_455_716_864);
  });

  it("uses eight alternating arm orders and two balanced shape rotations", () => {
    const rounds = buildOpt0079TimingRounds();
    expect(rounds).toHaveLength(8);
    expect(rounds.map(({ shapeOrder }) => shapeOrder)).toEqual([
      [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
      [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
    ]);
    expect(rounds.map(({ armOrder }) => armOrder)).toEqual([
      ["current", "candidate"], ["candidate", "current"],
      ["current", "candidate"], ["candidate", "current"],
      ["current", "candidate"], ["candidate", "current"],
      ["current", "candidate"], ["candidate", "current"],
    ]);
  });

  it("passes only both-clock 1.15x, 25ms, seven-pair, shape-safe evidence", () => {
    const passing = timingInput(10, 7, 10, 7);
    const result = summarizeOpt0079Timing(passing);
    expect(result).toMatchObject({ samplesPerArmPerShape: 8,
      multiplicities: { complete: "4/2/2/1", feedForward: "0/0/2/1" },
      complete: {
        mean: { gpuSpeedup: 10 / 7, wallSpeedup: 10 / 7,
          gpuSavingMilliseconds: 27, wallSavingMilliseconds: 27 },
        median: { gpuSpeedup: 10 / 7, wallSpeedup: 10 / 7,
          gpuSavingMilliseconds: 27, wallSavingMilliseconds: 27 },
      },
      gates: { gpuPairWins: 8, wallPairWins: 8,
        requiredMeanAndMedianSpeedup: 1.15,
        requiredMeanAndMedianSavingMilliseconds: 25,
        requiredProjectedSavingMilliseconds: 4_800,
        everyShapeMeanAndMedianFaster: true,
        meanGpuSpeedupPassed: true, medianGpuSpeedupPassed: true,
        meanWallSpeedupPassed: true, medianWallSpeedupPassed: true,
        meanGpuSavingPassed: true, medianGpuSavingPassed: true,
        meanWallSavingPassed: true, medianWallSavingPassed: true,
        wallGpuSavingsAgree: true, projectedSavingPassed: true },
      passed: true });
    expect(classifyOpt0079PrimitiveDisposition({ ...receipt(result),
      inPagePassed: true })).toBe(
        "positive-primitive-diagnostic-profile-authorized",
      );

    expect(summarizeOpt0079Timing(
      timingInput(10, 7.23, 10, 7.23),
    )).toMatchObject({ gates: { meanGpuSavingPassed: false,
      medianGpuSavingPassed: false, meanWallSavingPassed: false,
      medianWallSavingPassed: false, projectedSavingPassed: false },
    passed: false });

    const twoLosses = timingInput(10, 5, 10, 5);
    for (const input of twoLosses) {
      const candidate = [...input.samples.candidate];
      candidate[0] = sample(11, 12);
      candidate[1] = sample(11, 12);
      input.samples = { ...input.samples, candidate };
    }
    expect(summarizeOpt0079Timing(twoLosses)).toMatchObject({
      gates: { gpuPairWins: 6, wallPairWins: 6,
        gpuPairWinsPassed: false, wallPairWinsPassed: false }, passed: false,
    });
  });

  it("classifies frozen negative and inconclusive boundaries without overclaiming", () => {
    const subTwoSecond = summarizeOpt0079Timing(
      timingInput(5, 4, 5.5, 4.4),
    );
    expect(classifyOpt0079PrimitiveDisposition(receipt(subTwoSecond)))
      .toBe("negative-stop-exact-decoded-half-tile-mechanism");
    const regressingShapeInputs = timingInput(10, 7, 10, 7);
    regressingShapeInputs[2]!.samples = {
      current: eight(sample(10, 10)), candidate: eight(sample(11, 11)),
    };
    expect(classifyOpt0079PrimitiveDisposition(receipt(
      summarizeOpt0079Timing(regressingShapeInputs),
    ))).toBe("negative-stop-exact-decoded-half-tile-mechanism");

    const disagreement = summarizeOpt0079Timing(
      timingInput(10, 7.5, 15, 8.25),
    );
    expect(disagreement).toMatchObject({
      gates: { wallGpuSavingsAgree: false }, passed: false,
    });
    expect(classifyOpt0079PrimitiveDisposition(receipt(disagreement)))
      .toBe("inconclusive-directional-or-wall-gpu-evidence");
    const mixedClock = summarizeOpt0079Timing(
      timingInput(10, 7.5, 10, 10.5),
    );
    expect(mixedClock).toMatchObject({ passed: false });
    expect((mixedClock["strata"] as readonly Readonly<
      Record<string, unknown>
    >[])[0]).toMatchObject({ gates: {
      meanGpuFaster: true, meanWallFaster: false,
    } });
    expect(classifyOpt0079PrimitiveDisposition(receipt(mixedClock)))
      .toBe("inconclusive-directional-or-wall-gpu-evidence");
    expect(classifyOpt0079PrimitiveDisposition({ ...receipt(disagreement),
      correctness: { passed: false } }))
      .toBe("inconclusive-invalid-correctness-or-lifecycle-evidence");
    expect(classifyOpt0079PrimitiveDisposition({ ...receipt(disagreement),
      inPagePassed: true, cleanup: { passed: false,
        zeroPostCleanupDeviceLosses: false } }))
      .toBe("inconclusive-invalid-correctness-or-lifecycle-evidence");
    expect(classifyOpt0079PrimitiveDisposition({ inPagePassed: false,
      correctness: { passed: true }, cleanup: { passed: true } }))
      .toBe("inconclusive-infrastructure-or-missing-evidence");
    expect(classifyOpt0079PrimitiveDisposition({ ...receipt(disagreement),
      timing: { ...disagreement, strata: [{ gates: undefined }] } }))
      .toBe("inconclusive-infrastructure-or-missing-evidence");
  });

  it("parses one continuous nominal launch slice and trace through cleanup", () => {
    const launch = parseOpt0079ThermalLaunchGate(new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalCommand: "notifyutil -g com.apple.system.thermalpressurelevel",
      thermalTraceStartedAtEpochMilliseconds: "9000",
      thermalGateStartedAtEpochMilliseconds: "11000",
      thermalGateCompletedAtEpochMilliseconds: "41000",
      thermalGateObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalGateMaximumPollGapMilliseconds: "1004",
      thermalGateNonNominalObservations: "0",
      thermalGateMissingObservations: "0",
    }), 10_000, 42_000);
    expect(launch).toMatchObject({ observationCount: 31,
      readyToGateDelayMilliseconds: 1_000,
      launchDelayMilliseconds: 1_000 });
    expect(parseOpt0079ThermalCompletion(new URLSearchParams({
      thermalTraceSchema:
        "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
      thermalTraceSha256: "a".repeat(64),
      thermalTraceByteLength: "8192",
      thermalTraceCompletedAtEpochMilliseconds: "70000",
      thermalTraceObservations: "62",
      thermalTraceMaximumPollGapMilliseconds: "1008",
      thermalTraceNonNominalObservations: "0",
      thermalTraceMissingObservations: "0",
      thermalTraceInitialLevel: "0",
      thermalTraceFinalLevel: "0",
    }), launch, 69_000)).toMatchObject({ coversCleanup: true,
      initialLevel: 0, finalLevel: 0 });

    expect(() => parseOpt0079ThermalLaunchGate(new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalCommand: "notifyutil -g com.apple.system.thermalpressurelevel",
      thermalTraceStartedAtEpochMilliseconds: "10001",
      thermalGateStartedAtEpochMilliseconds: "11000",
      thermalGateCompletedAtEpochMilliseconds: "41000",
      thermalGateObservations: "31", thermalPollMilliseconds: "1000",
      thermalGateMaximumPollGapMilliseconds: "1000",
      thermalGateNonNominalObservations: "0",
      thermalGateMissingObservations: "0",
    }), 10_000, 42_000)).toThrow(/thermal launch gate/u);
    expect(() => parseOpt0079ThermalCompletion(new URLSearchParams({
      thermalTraceSchema:
        "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
      thermalTraceSha256: "b".repeat(64), thermalTraceByteLength: "8192",
      thermalTraceCompletedAtEpochMilliseconds: "68000",
      thermalTraceObservations: "62",
      thermalTraceMaximumPollGapMilliseconds: "1000",
      thermalTraceNonNominalObservations: "0",
      thermalTraceMissingObservations: "0", thermalTraceInitialLevel: "0",
      thermalTraceFinalLevel: "0",
    }), launch, 69_000)).toThrow(/through-cleanup thermal trace/u);
  });

  it("freezes full raw-U32 reruns, guards, timestamps, lifecycle, and download", () => {
    for (const token of [
      '"6313bfbaa61dee91f065d1652ee8eb1446f987b8"',
      "comparisonCount: 12",
      "comparisonsPerOutputWord: 3",
      "currentCandidateRawU32Exact",
      "currentRerunRawU32Exact",
      "candidateRerunRawU32Exact",
      "verifyMalformedSetupRejected",
      "setupFailureRejected",
      "targetBrowserMalformedBindingSetupFailureRejected",
      "existing-h-h-activation-aliased-as-output",
      "allocatedBufferCount: 0",
      "submittedCommandBufferCount: 0",
      "sameLogicalOutputInterval",
      "sameNativePackedWeightStorageShape",
      "candidateEstimatedGlobalOperandBytes",
      "sourceSha256",
      "generatedShaderAggregateSha256",
      '"4084610c5e43dff2d388361965750b0f603400dd"',
      "decodedVectorsPerInnerTile",
      "decodedVectorWritesPerWorkgroup",
      "physicalColumnTiles",
      "OUTPUT_PREFILL_QNAN_U32",
      "prefixGuardIntact",
      "suffixGuardIntact",
      "partialMTailWritten",
      "fullOutputRawU32AndCurrentCandidateReruns",
      "timestampWrites",
      "onePassOneCommandBufferOneSubmitOneDrainPerSample",
      "createdEqualsDestroyed",
      "mapsBalanced",
      "postDestroyRejected",
      "zeroPostCleanupDeviceLosses",
      "inconclusive-infrastructure-or-missing-evidence",
      "partialCurrent?.destroy()",
      "partialCandidate?.destroy()",
      "partialQuerySet?.destroy()",
      "tracker.destroyAll()",
      'if (buffer.mapState === "mapped") this.unmap(buffer)',
      "unchangedTimingRetryPerformed: false",
      "productionIntegrationAuthorized: false",
      "__ACE_OPT0079_RESULT__",
      "opt-0079-dense-decoded-half-tile-receipt.json",
    ]) expect(HARNESS_SOURCE).toContain(token);
    expect(HARNESS_SOURCE).not.toContain("Math.random");
    expect(HTML_SOURCE).toContain("current and candidate reruns");
    expect(HTML_SOURCE).toContain("Eight rounds alternate arm order");
    expect(HTML_SOURCE).toContain("at least 1.15x and 25.0 ms");
    expect(HTML_SOURCE).toContain("a 4,800 ms projection");
    expect(HTML_SOURCE).toContain("Download compact JSON receipt");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0079-dit-dense-decoded-half-tile.ts"',
    );
  });
});

function timingInput(
  currentGpu: number,
  candidateGpu: number,
  currentWall: number,
  candidateWall: number,
): MutableTimingInput[] {
  return buildOpt0079ShapeSpecs().map((spec) => ({ id: spec.id,
    samples: { current: eight(sample(currentGpu, currentWall)),
      candidate: eight(sample(candidateGpu, candidateWall)) } }));
}

type MutableTimingInput = {
  id: Opt0079TimingInput["id"];
  samples: {
    current: Opt0079TimestampSample[];
    candidate: Opt0079TimestampSample[];
  };
};

function sample(
  gpuMilliseconds: number,
  wallMilliseconds: number,
): Opt0079TimestampSample {
  const gpuElapsedNanoseconds = Math.round(gpuMilliseconds * 1_000_000);
  return { submitAtPerformanceMilliseconds: 1,
    fenceAtPerformanceMilliseconds: 1 + wallMilliseconds,
    submitAtEpochMilliseconds: 1_000, fenceAtEpochMilliseconds: 1_001,
    wallMilliseconds, timestampBeginNanoseconds: "1",
    timestampEndNanoseconds: String(1 + gpuElapsedNanoseconds),
    gpuElapsedNanoseconds, gpuMilliseconds,
    gpuToWallRatio: gpuMilliseconds / wallMilliseconds,
    validMultiplyAdds: 1, scheduledMultiplyAdds: 1,
    validGpuTflops: 1, scheduledGpuTflops: 1,
    validWallTflops: 1, scheduledWallTflops: 1,
    commandBufferCount: 1, queueDrainCount: 1,
    timestampResolveCount: 1, timestampCopyCount: 1 };
}

function eight<T>(value: T): T[] {
  return Array.from({ length: 8 }, () => value);
}

function receipt(timing: Readonly<Record<string, unknown>>):
Readonly<Record<string, unknown>> {
  return { inPagePassed: false, correctness: { passed: true },
    cleanup: { passed: true }, timing };
}

function source(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}
