import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  buildOpt0078ShapeSpecs,
  buildOpt0078TimingRounds,
  classifyOpt0078PrimitiveDisposition,
  parseOpt0078ThermalCompletion,
  parseOpt0078ThermalLaunchGate,
  summarizeOpt0078Timing,
  type Opt0078TimestampSample,
  type Opt0078TimingInput,
} from "./browser/opt-0078-dit-dense-weight-multicast.js";
import { planAceOpt0009DenseGemm } from
  "../src/webgpu/kernels/dit-dense-fp16.js";
import { planAceOpt0078DenseWeightMulticast } from
  "../src/webgpu/kernels/dit-dense-fp16-weight-multicast.js";

const HARNESS_SOURCE = source(
  "./browser/opt-0078-dit-dense-weight-multicast.ts",
);
const HTML_SOURCE = source(
  "./browser/opt-0078-dit-dense-weight-multicast.html",
);

describe("OPT-0078 target-browser dense weight-multicast contract", () => {
  it("pins every exact M2250 shape, 4/2/2/1 and FF scores, and edge families", () => {
    expect(buildOpt0078ShapeSpecs()).toEqual([
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

  it("reconciles identical M32/N256/K32 work with WG256 multicast", () => {
    let currentWorkgroups = 0;
    let candidateWorkgroups = 0;
    let currentMacs = 0;
    let candidateMacs = 0;
    for (const spec of buildOpt0078ShapeSpecs()) {
      const current = planAceOpt0009DenseGemm(spec.shape);
      const candidate = planAceOpt0078DenseWeightMulticast(spec.shape);
      expect(current).toMatchObject({ tileRows: 32, tileColumns: 256,
        tileInner: 32, workgroupSize: 128 });
      expect(candidate).toMatchObject({ tileRows: 32, tileColumns: 256,
        tileInner: 32, workgroupSize: 256, subgroupsPerWorkgroup: 8,
        rowsPerSubgroup: 4, accumulatorsPerLane: 32,
        packedRecordsPerInnerTile: 1_024,
        workgroupStorageBytes: 16_384 });
      expect(candidate.rowTiles).toBe(current.rowTiles);
      expect(candidate.columnTiles).toBe(current.columnTiles);
      expect(candidate.innerTiles).toBe(current.innerTiles);
      expect(candidate.outputRanges).toEqual(current.outputRanges);
      expect(candidate.packedWeightStorageShape)
        .toEqual(current.packedWeightStorageShape);
      currentWorkgroups += current.workgroupCount * spec.productionMultiplicity;
      candidateWorkgroups += candidate.workgroupCount *
        spec.productionMultiplicity;
      currentMacs += current.outputRanges[0]!.multiplyAdds *
        spec.productionMultiplicity;
      candidateMacs += candidate.outputRanges[0]!.multiplyAdds *
        spec.productionMultiplicity;
    }
    expect(currentWorkgroups).toBe(6_816);
    expect(candidateWorkgroups).toBe(6_816);
    expect(currentMacs).toBe(133_412_421_632);
    expect(candidateMacs).toBe(currentMacs);
  });

  it("uses eight alternating arm orders and two balanced shape rotations", () => {
    const rounds = buildOpt0078TimingRounds();
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

  it("passes only both-clock 1.12x, 20.8334ms, seven-pair, shape-safe evidence", () => {
    const passing = timingInput(10, 7.5, 11, 8.25);
    const result = summarizeOpt0078Timing(passing);
    expect(result).toMatchObject({ samplesPerArmPerShape: 8,
      multiplicities: { complete: "4/2/2/1", feedForward: "0/0/2/1" },
      complete: {
        mean: { gpuSpeedup: 4 / 3, wallSpeedup: 4 / 3,
          gpuSavingMilliseconds: 22.5, wallSavingMilliseconds: 24.75 },
        median: { gpuSpeedup: 4 / 3, wallSpeedup: 4 / 3,
          gpuSavingMilliseconds: 22.5, wallSavingMilliseconds: 24.75 },
      },
      gates: { gpuPairWins: 8, wallPairWins: 8,
        everyShapeMeanAndMedianFaster: true,
        meanGpuSpeedupPassed: true, medianGpuSpeedupPassed: true,
        meanWallSpeedupPassed: true, medianWallSpeedupPassed: true,
        meanGpuSavingPassed: true, medianGpuSavingPassed: true,
        meanWallSavingPassed: true, medianWallSavingPassed: true,
        wallGpuSavingsAgree: true, projectedSavingPassed: true },
      passed: true });

    const twoLosses = timingInput(10, 7, 11, 7.7);
    for (const input of twoLosses) {
      const candidate = [...input.samples.candidate];
      candidate[0] = sample(11, 12);
      candidate[1] = sample(11, 12);
      input.samples = { ...input.samples, candidate };
    }
    expect(summarizeOpt0078Timing(twoLosses)).toMatchObject({
      gates: { gpuPairWins: 6, wallPairWins: 6,
        gpuPairWinsPassed: false, wallPairWinsPassed: false }, passed: false,
    });
  });

  it("classifies frozen negative and inconclusive boundaries without overclaiming", () => {
    const subTwoSecond = summarizeOpt0078Timing(
      timingInput(5, 4, 5.5, 4.4),
    );
    expect(classifyOpt0078PrimitiveDisposition(receipt(subTwoSecond)))
      .toBe("negative-stop-exact-weight-multicast-mechanism");

    const disagreement = summarizeOpt0078Timing(
      timingInput(10, 7.5, 15, 8.25),
    );
    expect(disagreement).toMatchObject({
      gates: { wallGpuSavingsAgree: false }, passed: false,
    });
    expect(classifyOpt0078PrimitiveDisposition(receipt(disagreement)))
      .toBe("inconclusive-directional-or-wall-gpu-evidence");
    const mixedClock = summarizeOpt0078Timing(
      timingInput(10, 7.5, 10, 10.5),
    );
    expect(mixedClock).toMatchObject({ passed: false });
    expect((mixedClock["strata"] as readonly Readonly<
      Record<string, unknown>
    >[])[0]).toMatchObject({ gates: {
      meanGpuFaster: true, meanWallFaster: false,
    } });
    expect(classifyOpt0078PrimitiveDisposition(receipt(mixedClock)))
      .toBe("inconclusive-directional-or-wall-gpu-evidence");
    expect(classifyOpt0078PrimitiveDisposition({ ...receipt(disagreement),
      correctness: { passed: false } }))
      .toBe("inconclusive-invalid-correctness-or-lifecycle-evidence");
  });

  it("parses one continuous nominal launch slice and trace through cleanup", () => {
    const launch = parseOpt0078ThermalLaunchGate(new URLSearchParams({
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
    expect(parseOpt0078ThermalCompletion(new URLSearchParams({
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
  });

  it("freezes full raw-U32 reruns, guards, timestamps, lifecycle, and download", () => {
    for (const token of [
      'const REGISTRATION_COMMIT = "23e0947"',
      "comparisonCount: 12",
      "comparisonsPerOutputWord: 3",
      "currentCandidateRawU32Exact",
      "currentRerunRawU32Exact",
      "candidateRerunRawU32Exact",
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
      "unchangedTimingRetryPerformed: false",
      "productionIntegrationAuthorized: false",
      "__ACE_OPT0078_RESULT__",
      "opt-0078-dense-weight-multicast-receipt.json",
    ]) expect(HARNESS_SOURCE).toContain(token);
    expect(HARNESS_SOURCE).not.toContain("Math.random");
    expect(HTML_SOURCE).toContain("current and candidate reruns");
    expect(HTML_SOURCE).toContain("Eight rounds alternate arm order");
    expect(HTML_SOURCE).toContain("Download compact JSON receipt");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0078-dit-dense-weight-multicast.ts"',
    );
  });
});

function timingInput(
  currentGpu: number,
  candidateGpu: number,
  currentWall: number,
  candidateWall: number,
): MutableTimingInput[] {
  return buildOpt0078ShapeSpecs().map((spec) => ({ id: spec.id,
    samples: { current: eight(sample(currentGpu, currentWall)),
      candidate: eight(sample(candidateGpu, candidateWall)) } }));
}

type MutableTimingInput = {
  id: Opt0078TimingInput["id"];
  samples: {
    current: Opt0078TimestampSample[];
    candidate: Opt0078TimestampSample[];
  };
};

function sample(
  gpuMilliseconds: number,
  wallMilliseconds: number,
): Opt0078TimestampSample {
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
