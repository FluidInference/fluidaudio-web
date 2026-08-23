import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  buildOpt0081ShapeSpecs,
  buildOpt0081TimingRounds,
  classifyOpt0081PrimitiveDisposition,
  numberToFloat16Bits,
  parseOpt0081ThermalCompletion,
  parseOpt0081ThermalLaunchGate,
  summarizeOpt0081Timing,
  type Opt0081Arm,
  type Opt0081TimestampSample,
  type Opt0081TimingInput,
} from "./browser/opt-0081-dit-f16-dense-input-multicast.js";
import { planAceOpt0009DenseGemm } from
  "../src/webgpu/kernels/dit-dense-fp16.js";
import { planAceOpt0081DenseF16Input } from
  "../src/webgpu/kernels/dit-dense-f16-input.js";
import { planAceOpt0081DenseF16InputWeightMulticast } from
  "../src/webgpu/kernels/dit-dense-f16-input-weight-multicast.js";

const HARNESS_SOURCE = source(
  "./browser/opt-0081-dit-f16-dense-input-multicast.ts",
);
const HTML_SOURCE = source(
  "./browser/opt-0081-dit-f16-dense-input-multicast.html",
);

describe("OPT-0081 target-browser typed-F16 dense multicast contract", () => {
  it("pins every full M2250 shape to its audited real producer output", () => {
    expect(buildOpt0081ShapeSpecs()).toEqual([
      { id: "h-h", shape: { rows: 2_250, inner: 2_048, columns: 2_048 },
        productionMultiplicity: 4, feedForwardMultiplicity: 0,
        retainedProducerRole: "selfMergedAttention",
        fixtureIds: ["signed-zero", "subnormal-normal-boundary"], ordinal: 0 },
      { id: "h-1024", shape: { rows: 2_250, inner: 2_048,
        columns: 1_024 }, productionMultiplicity: 2,
        feedForwardMultiplicity: 0, retainedProducerRole: "selfModulated",
        fixtureIds: ["alternating-cancellation"], ordinal: 1 },
      { id: "h-6144", shape: { rows: 2_250, inner: 2_048,
        columns: 6_144 }, productionMultiplicity: 2,
        feedForwardMultiplicity: 2, retainedProducerRole: "mlpModulated",
        fixtureIds: ["maximum-finite-fp16-bounded"], ordinal: 2 },
      { id: "6144-h", shape: { rows: 2_250, inner: 6_144,
        columns: 2_048 }, productionMultiplicity: 1,
        feedForwardMultiplicity: 1, retainedProducerRole: "gatedActivation",
        fixtureIds: ["activation-rounding-boundary"], ordinal: 3 },
    ]);
  });

  it("reconciles A/B/C work, packed weights, and typed-F16 input bytes", () => {
    let aWorkgroups = 0;
    let cWorkgroups = 0;
    let aMacs = 0;
    let cMacs = 0;
    let bActivationRequests = 0;
    let bWeightRequests = 0;
    let cActivationRequests = 0;
    let cWeightRequests = 0;
    for (const spec of buildOpt0081ShapeSpecs()) {
      const A = planAceOpt0009DenseGemm(spec.shape);
      const B = planAceOpt0081DenseF16Input(spec.shape);
      const C = planAceOpt0081DenseF16InputWeightMulticast(spec.shape);
      expect(A).toMatchObject({ tileRows: 32, tileColumns: 256,
        tileInner: 32, workgroupSize: 128 });
      expect(B).toMatchObject({ tileRows: 32, tileColumns: 256,
        tileInner: 32, workgroupSize: 128, activationElementBytes: 2 });
      expect(C).toMatchObject({ tileRows: 32, tileColumns: 256,
        tileInner: 32, workgroupSize: 256, subgroupsPerWorkgroup: 8,
        rowsPerSubgroup: 4, accumulatorsPerLane: 32,
        packedRecordsPerInnerTile: 1_024, workgroupStorageBytes: 16_384,
        activationStorage: "scalar-f16",
        activationElementBytes: 2, outputElementBytes: 4 });
      expect(B.outputRanges).toEqual(A.outputRanges);
      expect(C.outputRanges).toEqual(A.outputRanges);
      expect(B.packedWeightStorageShape).toEqual(A.packedWeightStorageShape);
      expect(C.packedWeightStorageShape).toEqual(A.packedWeightStorageShape);
      expect(B.activationElements * B.activationElementBytes)
        .toBe(spec.shape.rows * spec.shape.inner * 2);
      expect(C.activationBytes).toBe(spec.shape.rows * spec.shape.inner * 2);
      expect(C.barriersPerWorkgroup).toBe(C.innerTiles * 2);
      expect(C.barrierEvents).toBe(C.workgroupCount * C.innerTiles * 2);
      aWorkgroups += A.workgroupCount * spec.productionMultiplicity;
      cWorkgroups += C.workgroupCount * spec.productionMultiplicity;
      aMacs += A.outputRanges[0]!.multiplyAdds * spec.productionMultiplicity;
      cMacs += C.outputRanges[0]!.multiplyAdds * spec.productionMultiplicity;
      bActivationRequests += B.estimatedGlobalActivationBytes *
        spec.productionMultiplicity;
      bWeightRequests += B.estimatedGlobalWeightBytes *
        spec.productionMultiplicity;
      cActivationRequests += C.estimatedGlobalActivationBytes *
        spec.productionMultiplicity;
      cWeightRequests += C.estimatedGlobalWeightBytes *
        spec.productionMultiplicity;
    }
    expect(aWorkgroups).toBe(6_816);
    expect(cWorkgroups).toBe(aWorkgroups);
    expect(aMacs).toBe(133_412_421_632);
    expect(cMacs).toBe(aMacs);
    expect({
      aActivationRequests: bActivationRequests * 2,
      aWeightRequests: bWeightRequests,
      aOperandRequests: bActivationRequests * 2 + bWeightRequests,
      bActivationRequests, bWeightRequests,
      bOperandRequests: bActivationRequests + bWeightRequests,
      cActivationRequests, cWeightRequests,
      cOperandRequests: cActivationRequests + cWeightRequests,
    }).toEqual({
      aActivationRequests: 2_084_569_088,
      aWeightRequests: 33_353_105_408,
      aOperandRequests: 35_437_674_496,
      bActivationRequests: 1_042_284_544,
      bWeightRequests: 33_353_105_408,
      bOperandRequests: 34_395_389_952,
      cActivationRequests: 1_042_284_544,
      cWeightRequests: 8_338_276_352,
      cOperandRequests: 9_380_560_896,
    });
  });

  it("freezes the eight registered three-arm orders and shape rotations", () => {
    const rounds = buildOpt0081TimingRounds();
    expect(rounds.map(({ shapeOrder }) => shapeOrder)).toEqual([
      [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
      [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
    ]);
    expect(rounds.map(({ armOrder }) => armOrder)).toEqual([
      ["A", "B", "C"], ["C", "B", "A"], ["B", "C", "A"],
      ["A", "C", "B"], ["C", "A", "B"], ["B", "A", "C"],
      ["A", "B", "C"], ["C", "B", "A"],
    ]);
  });

  it("applies the independent B/A, C/A, and C/B registered gates", () => {
    const result = summarizeOpt0081Timing(timingInput(
      { gpu: 10, wall: 11 }, { gpu: 8, wall: 8.8 },
      { gpu: 6, wall: 6.6 },
    ));
    expect(result).toMatchObject({ samplesPerArmPerShape: 8,
      multiplicities: { complete: "4/2/2/1", feedForward: "0/0/2/1" },
      comparisons: {
        bOverA: { gpu: { pairWins: 8, meanSpeedup: 1.25,
          meanSavingMilliseconds: 18 }, wall: { pairWins: 8,
          meanSpeedup: 1.25 },
          wallGpuSavingsAgree: true },
        cOverA: { gpu: { pairWins: 8, meanSavingMilliseconds: 36 },
          wall: { pairWins: 8 } },
        cOverB: { gpu: { pairWins: 8, meanSavingMilliseconds: 18 },
          wall: { pairWins: 8 } },
      }, gates: { bStandalone: { passed: true }, c: { passed: true } },
      bStandalonePassed: true, cPassed: true, passed: true });
    const comparisons = result["comparisons"] as Record<string,
      { wall: { meanSavingMilliseconds: number } }>;
    expect(comparisons["bOverA"]!.wall.meanSavingMilliseconds)
      .toBeCloseTo(19.8);
    expect(comparisons["cOverA"]!.wall.meanSavingMilliseconds)
      .toBeCloseTo(39.6);
    expect(comparisons["cOverB"]!.wall.meanSavingMilliseconds)
      .toBeCloseTo(19.8);
    expect(classifyOpt0081PrimitiveDisposition({ ...receipt(result),
      inPagePassed: true })).toBe(
      "positive-C-primitive-diagnostic-profile-authorized",
    );
  });

  it("keeps B classification nonblocking while enforcing C evidence", () => {
    const result = summarizeOpt0081Timing(timingInput(
      { gpu: 10, wall: 11 }, { gpu: 9.2, wall: 10.12 },
      { gpu: 6, wall: 6.6 },
    ));
    expect(result).toMatchObject({ bStandalonePassed: false,
      cPassed: true, selectedArm: "C", passed: true });

    const bAlone = summarizeOpt0081Timing(timingInput(
      { gpu: 10, wall: 11 }, { gpu: 8, wall: 8.8 },
      { gpu: 11, wall: 12.1 },
    ));
    expect(bAlone).toMatchObject({ bStandalonePassed: true, cPassed: false,
      selectedArm: "B", passed: true });
    expect(classifyOpt0081PrimitiveDisposition({ ...receipt(bAlone),
      inPagePassed: true })).toBe(
      "positive-B-standalone-primitive-diagnostic-profile-authorized",
    );

    const pairLosses = timingInput(
      { gpu: 10, wall: 11 }, { gpu: 8, wall: 8.8 },
      { gpu: 5, wall: 5.5 },
    );
    for (const input of pairLosses) {
      input.samples.C[0] = sample(8.1, 8.91);
      input.samples.C[1] = sample(8.1, 8.91);
    }
    expect(summarizeOpt0081Timing(pairLosses)).toMatchObject({
      comparisons: { cOverB: { gpu: { pairWins: 6 },
        wall: { pairWins: 6 } } }, cPassed: false,
      bStandalonePassed: true, selectedArm: "B", passed: true,
    });

    const shapeRegression = timingInput(
      { gpu: 10, wall: 11 }, { gpu: 8, wall: 8.8 },
      { gpu: 6, wall: 6.6 },
    );
    shapeRegression[0]!.samples.C = eight(sample(10.5, 11.55));
    expect(summarizeOpt0081Timing(shapeRegression)).toMatchObject({
      cPassed: false, bStandalonePassed: true,
      selectedArm: "B", passed: true,
    });
  });

  it("reports median saving agreement without adding it to the mean gate", () => {
    const inputs = timingInput(
      { gpu: 10, wall: 11 }, { gpu: 9.2, wall: 10.12 },
      { gpu: 5, wall: 5.5 },
    );
    const cGpu = [1, 1, 1, 1, 9, 9, 9, 9];
    const cWall = [1, 1, 1, 8, 8, 8, 8, 10];
    for (const input of inputs) {
      input.samples.C = cGpu.map((gpu, index) =>
        sample(gpu, cWall[index]!));
    }
    const result = summarizeOpt0081Timing(inputs);
    expect(result).toMatchObject({ cPassed: true, selectedArm: "C",
      comparisons: {
        cOverA: { wallGpuSavingsAgree: true },
        cOverB: { wallGpuSavingsAgree: true },
      } });
    const comparison = (result["comparisons"] as Record<string,
      { meanWallGpuSavingRatio: number;
        medianWallGpuSavingRatio: number }>)["cOverA"]!;
    expect(comparison.meanWallGpuSavingRatio).toBeCloseTo(1.075);
    expect(comparison.medianWallGpuSavingRatio).toBeCloseTo(0.6);
  });

  it("classifies valid negative, directional, and lifecycle evidence", () => {
    const negative = summarizeOpt0081Timing(timingInput(
      { gpu: 10, wall: 11 }, { gpu: 9.2, wall: 10.12 },
      { gpu: 11, wall: 12.1 },
    ));
    expect(classifyOpt0081PrimitiveDisposition(receipt(negative)))
      .toBe("negative-stop-typed-f16-weight-multicast-mechanism");

    const disagreement = summarizeOpt0081Timing(timingInput(
      { gpu: 10, wall: 11 }, { gpu: 9.2, wall: 10.12 },
      { gpu: 6, wall: 5.5 },
    ));
    expect(disagreement).toMatchObject({ comparisons: {
      cOverA: { wallGpuSavingsAgree: false } }, passed: false });
    expect(classifyOpt0081PrimitiveDisposition(receipt(disagreement)))
      .toBe("inconclusive-directional-or-wall-gpu-evidence");
    const mixedB = summarizeOpt0081Timing(timingInput(
      { gpu: 10, wall: 11 }, { gpu: 8, wall: 12 },
      { gpu: 11, wall: 12.1 },
    ));
    expect(classifyOpt0081PrimitiveDisposition(receipt(mixedB)))
      .toBe("inconclusive-directional-or-wall-gpu-evidence");
    const crossShape = timingInput(
      { gpu: 10, wall: 11 }, { gpu: 9.2, wall: 10.12 },
      { gpu: 11, wall: 12.1 },
    );
    crossShape[0]!.samples.C = eight(sample(6, 6.6));
    const crossShapeResult = summarizeOpt0081Timing(crossShape);
    expect(crossShapeResult).toMatchObject({ selectedArm: null, passed: false });
    expect(classifyOpt0081PrimitiveDisposition(receipt(crossShapeResult)))
      .toBe("inconclusive-directional-or-wall-gpu-evidence");
    const pairClockConflict = timingInput(
      { gpu: 10, wall: 11 }, { gpu: 9.2, wall: 10.12 },
      { gpu: 6, wall: 6.6 },
    );
    for (const input of pairClockConflict) {
      input.samples.C[0] = sample(11, 6.6);
      input.samples.C[1] = sample(11, 6.6);
    }
    const pairClockResult = summarizeOpt0081Timing(pairClockConflict);
    expect(pairClockResult).toMatchObject({ selectedArm: null, passed: false,
      comparisons: { cOverA: { gpu: { pairWins: 6 },
        wall: { pairWins: 8 } } } });
    expect(classifyOpt0081PrimitiveDisposition(receipt(pairClockResult)))
      .toBe("inconclusive-directional-or-wall-gpu-evidence");
    const partialPairs = timingInput(
      { gpu: 10, wall: 11 }, { gpu: 9.2, wall: 10.12 },
      { gpu: 11, wall: 12.1 },
    );
    for (const input of partialPairs) {
      for (let round = 0; round < 6; round += 1) {
        input.samples.C[round] = sample(6, 6.6);
      }
    }
    const partialPairResult = summarizeOpt0081Timing(partialPairs);
    expect(partialPairResult).toMatchObject({ selectedArm: null, passed: false,
      comparisons: { cOverA: { gpu: { pairWins: 6 },
        wall: { pairWins: 6 } } } });
    expect(classifyOpt0081PrimitiveDisposition(receipt(partialPairResult)))
      .toBe("inconclusive-directional-or-wall-gpu-evidence");
    expect(classifyOpt0081PrimitiveDisposition({ ...receipt(disagreement),
      correctness: { passed: false } }))
      .toBe("inconclusive-invalid-correctness-or-lifecycle-evidence");
    expect(classifyOpt0081PrimitiveDisposition({ ...receipt(negative),
      correctness: observedMismatchCorrectness() }))
      .toBe("negative-stop-observed-raw-bit-correctness-mismatch");
    expect(classifyOpt0081PrimitiveDisposition({ ...receipt(negative),
      correctness: { ...observedMismatchCorrectness(),
        uncapturedGpuErrorCount: 1 } }))
      .toBe("inconclusive-invalid-correctness-or-lifecycle-evidence");
    expect(classifyOpt0081PrimitiveDisposition({ ...receipt(negative),
      cleanup: { passed: false } }))
      .toBe("inconclusive-invalid-correctness-or-lifecycle-evidence");
  });

  it("implements independent round-to-nearest-even scalar F16 semantics", () => {
    expect([
      numberToFloat16Bits(0), numberToFloat16Bits(-0),
      numberToFloat16Bits(2 ** -25), numberToFloat16Bits(2 ** -24),
      numberToFloat16Bits(3 * 2 ** -25),
      numberToFloat16Bits(2 ** -14 - 2 ** -24),
      numberToFloat16Bits(2 ** -14),
      numberToFloat16Bits(1 + 2 ** -11),
      numberToFloat16Bits(1 + 3 * 2 ** -11),
      numberToFloat16Bits(65_504), numberToFloat16Bits(65_520),
    ]).toEqual([
      0x0000, 0x8000, 0x0000, 0x0001, 0x0002, 0x03ff, 0x0400,
      0x3c00, 0x3c02, 0x7bff, 0x7c00,
    ]);
  });

  it("parses one continuous nominal launch slice and trace through cleanup", () => {
    const launch = parseOpt0081ThermalLaunchGate(new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalCommand: "notifyutil -g com.apple.system.thermalpressurelevel",
      thermalTraceStartedAtEpochMilliseconds: "9000",
      thermalGateStartedAtEpochMilliseconds: "11000",
      thermalGateCompletedAtEpochMilliseconds: "41000",
      thermalGateObservations: "31", thermalPollMilliseconds: "1000",
      thermalGateMaximumPollGapMilliseconds: "1400",
      thermalGateNonNominalObservations: "0",
      thermalGateMissingObservations: "0",
    }), 10_000, 42_000);
    expect(launch).toMatchObject({ observationCount: 31,
      maximumPollGapMilliseconds: 1_400,
      readyToGateDelayMilliseconds: 1_000, launchDelayMilliseconds: 1_000 });
    expect(parseOpt0081ThermalCompletion(new URLSearchParams({
      thermalTraceSchema:
        "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
      thermalTraceSha256: "a".repeat(64), thermalTraceByteLength: "8192",
      thermalTraceCompletedAtEpochMilliseconds: "70000",
      thermalTraceObservations: "62",
      thermalTraceMaximumPollGapMilliseconds: "1400",
      thermalTraceNonNominalObservations: "0",
      thermalTraceMissingObservations: "0", thermalTraceInitialLevel: "0",
      thermalTraceFinalLevel: "0",
    }), launch, 69_000)).toMatchObject({ coversCleanup: true,
      initialLevel: 0, finalLevel: 0 });
  });

  it("freezes full producer, correctness, timing, and lifecycle mechanics", () => {
    for (const token of [
      "70a5e4a29c5455ec00a4b757dcdf5cdcc70a5e91",
      "bbe180bf7feb59272a5d5f7afbafb3877afee416",
      "312d67024978a64b77d2563dd9386b4328f17d33",
      "createSelfModulatedDispatch", "createSelfMergedAttentionDispatch",
      "createCrossNormalizedDispatch", "createCrossMergedAttentionDispatch",
      "createMlpModulatedDispatch", "createGatedActivationDispatch",
      "candidateU16ComparedToIndependentCast: true",
      "gpuConversionDispatchCount: 0", "OUTPUT_PREFILL_QNAN_U16",
      "OUTPUT_PREFILL_QNAN_U32", "STORAGE_GUARD_U16", "STORAGE_GUARD_U32",
      'executionOrder: Object.freeze(["A", "A", "B", "B", "C", "C"])',
      "directBCRawU32Exact", "partialMTailWritten", "timestampWrites",
      "onePassOneCommandBufferOneSubmitOneDrainPerSample",
      "createdEqualsDestroyed", "mapsBalanced", "producerPostDestroyRejected",
      "armBPostDestroyRejected", "armCPostDestroyRejected",
      "nonFiniteF16Count", "bothSignedZerosObservedAndExact",
      "Opt0081CorrectnessGateFailure", "stoppedBeforeDenseAndTiming: true",
      "negative-stop-observed-raw-bit-correctness-mismatch",
      "inconclusive-invalid-thermal-or-protocol-evidence",
      "unchangedTimingRetryPerformed: false",
      "productionIntegrationAuthorized: false", "__ACE_OPT0081_RESULT__",
      "opt-0081-dit-f16-dense-input-multicast-receipt.json",
    ]) expect(HARNESS_SOURCE).toContain(token);
    expect(HARNESS_SOURCE).not.toContain("Math.random");
    expect(HTML_SOURCE).toContain("all six actual M2250 producers");
    expect(HTML_SOURCE).toContain("A,A,B,B,C,C");
    expect(HTML_SOURCE).toContain("ABC, CBA, BCA, ACB, CAB, BAC, ABC, CBA");
    expect(HTML_SOURCE).toContain("Download compact JSON receipt");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0081-dit-f16-dense-input-multicast.ts"',
    );
  });
});

function timingInput(A: Metric, B: Metric, C: Metric): MutableTimingInput[] {
  return buildOpt0081ShapeSpecs().map((spec) => ({ id: spec.id,
    samples: { A: eight(sample(A.gpu, A.wall)),
      B: eight(sample(B.gpu, B.wall)),
      C: eight(sample(C.gpu, C.wall)) } }));
}

interface Metric { readonly gpu: number; readonly wall: number }

type MutableTimingInput = {
  id: Opt0081TimingInput["id"];
  samples: Record<Opt0081Arm, Opt0081TimestampSample[]>;
};

function sample(gpuMilliseconds: number,
  wallMilliseconds: number): Opt0081TimestampSample {
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

function observedMismatchCorrectness(): Readonly<Record<string, unknown>> {
  const producerCases = Array.from({ length: 6 }, (_, index) => ({
    differingU16Count: index === 0 ? 1 : 0, nonFiniteF32Count: 0,
    nonFiniteF16Count: 0, qNaNPrefillF32Count: 0, qNaNPrefillF16Count: 0,
    bothSignedZerosObservedAndExact: index !== 0,
    guards: { f32: { passed: true }, f16: { passed: true } },
  }));
  const denseCases = Array.from({ length: 4 }, () => ({
    allSixRawU32Exact: true, everyArmRerunExact: true,
    directBCRawU32Exact: true, outputsFiniteAndComplete: true,
    finiteClassIdentity: true, guardsAndPartialMTailsIntact: true,
  }));
  return { passed: false, completedBeforeReady: true,
    uncapturedGpuErrorCount: 0, deviceLossCount: 0,
    producerAudit: { cases: producerCases }, cases: denseCases };
}

function source(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}
