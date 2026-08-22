import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  buildOpt0077AdversarialCases,
  buildOpt0077CorrectnessCases,
  buildOpt0077MathProof,
  buildOpt0077PersistentPayloadAccounting,
  buildOpt0077ProductionStrata,
  buildOpt0077ProductionWeightBindingPlan,
  buildOpt0077ScratchAccounting,
  parseOpt0077ThermalCompletion,
  parseOpt0077ThermalLaunchGate,
  planOpt0077ScratchBytes,
  planOpt0077TileCount,
  summarizeOpt0077Timing,
  type Opt0077TimestampSample,
  type Opt0077TimingInput,
} from "./browser/opt-0077-vae-k7-rfft16.js";

const HARNESS_SOURCE = source("./browser/opt-0077-vae-k7-rfft16.ts");
const HTML_SOURCE = source("./browser/opt-0077-vae-k7-rfft16.html");
const RECORD_SOURCE = source(
  "../optimization/experiments/OPT-0077-vae-k7-rfft16-transform-domain.md",
);

describe("OPT-0077 RFFT16 browser primitive contract", () => {
  it("pins nine production strata, twelve represented routes, and exact weights", () => {
    const strata = buildOpt0077ProductionStrata();
    expect(strata).toHaveLength(9);
    expect(strata.map(({ id, timingMultiplicity }) =>
      [id, timingMultiplicity])).toEqual([
      ["c1024-d1", 2], ["c1024-d3", 2], ["c1024-d9", 2],
      ["c512-d1", 3], ["c512-d3", 3], ["c512-d9", 3],
      ["c128-d1", 9], ["c128-d3", 9], ["c128-d9", 9],
    ]);
    expect(strata.reduce((sum, item) =>
      sum + item.timingMultiplicity, 0)).toBe(42);
    expect(new Set(strata.flatMap(({ representedOperationLabels }) =>
      representedOperationLabels))).toHaveLength(12);
    const weightBindings = buildOpt0077ProductionWeightBindingPlan();
    expect(weightBindings).toHaveLength(12);
    expect(weightBindings.map(({ operationLabel, multiplicity }) =>
      [operationLabel, multiplicity])).toEqual([
      ["block-0-res-1-conv1", 2], ["block-0-res-2-conv1", 2],
      ["block-0-res-3-conv1", 2], ["block-1-res-1-conv1", 3],
      ["block-1-res-2-conv1", 3], ["block-1-res-3-conv1", 3],
      ["block-3-res-1-conv1", 3], ["block-4-res-1-conv1", 6],
      ["block-3-res-2-conv1", 3], ["block-4-res-2-conv1", 6],
      ["block-3-res-3-conv1", 3], ["block-4-res-3-conv1", 6],
    ]);
    expect(new Set(weightBindings.map(({ operationLabel }) =>
      operationLabel))).toHaveLength(12);
    expect(weightBindings.reduce((sum, { multiplicity }) =>
      sum + multiplicity, 0)).toBe(42);
    expect(strata.every(({ shape }) => shape.inputFrames === 512)).toBe(true);
    expect(RECORD_SOURCE).toContain(
      "C1024 has multiplicity 2, C512 has 3, and C128 has 9",
    );
  });

  it("uses unaligned first/interior/tail probes over all dilation residues", () => {
    const strata = buildOpt0077ProductionStrata();
    for (const spec of strata) {
      expect(spec.probes.map(({ id }) => id)).toEqual([
        "first", "interior", "tail",
      ]);
      const interior = spec.probes[1]!;
      expect(interior.firstOutputRow % 10).not.toBe(0);
      expect(interior.outputRowCount % 10).not.toBe(0);
      expect(interior.outputRowCount).toBeGreaterThanOrEqual(spec.dilation);
      expect(spec.probes[0]!.firstOutputRow).toBe(0);
      expect(spec.probes[2]!.firstOutputRow +
        spec.probes[2]!.outputRowCount).toBe(512);
    }
    expect(planOpt0077TileCount(512, 1)).toBe(52);
    expect(planOpt0077TileCount(512, 3)).toBe(53);
    expect(planOpt0077TileCount(512, 9)).toBe(54);
  });

  it("pins all four general and sixteen spectral adversarial cases", () => {
    const adversarial = buildOpt0077AdversarialCases();
    expect(adversarial).toHaveLength(20);
    expect(adversarial.slice(0, 6).map(({ kind }) => kind)).toEqual([
      "signed-zero-subnormal",
      "alternating-cancellation",
      "finite-transform-amplification",
      "partial-tile",
      "dc",
      "nyquist",
    ]);
    expect(adversarial.filter(({ kind }) =>
      /^cos[1-7]$|^sin[1-7]$/u.test(kind))).toHaveLength(14);
    expect(buildOpt0077CorrectnessCases()).toHaveLength(29);
    expect(adversarial.every(({ probes }) =>
      probes.length === 1 && probes[0]!.id === "full")).toBe(true);
  });

  it("passes the independent radix-2/direct-DFT basis and range proof", () => {
    const proof = buildOpt0077MathProof();
    expect(proof).toMatchObject({
      comparedForwardCoordinates: 256,
      comparedWeightCoordinates: 112,
      comparedCorrelationOutputs: 1_120,
      noExtraOverallScaleFactor: true,
      passed: true,
      residueAndRangeMapping: {
        plannerDomains: 192,
        exactOnceCoverage: true,
        partialFinalTilesCovered: true,
        passed: true,
      },
    });
  });

  it("derives bounded full-range scratch from the candidate planner", () => {
    const strata = buildOpt0077ProductionStrata();
    const d1 = planOpt0077ScratchBytes(strata[0]!.shape);
    const d3 = planOpt0077ScratchBytes(strata[1]!.shape);
    const d9 = planOpt0077ScratchBytes(strata[2]!.shape);
    expect(d1.inputSpectrum).toBe(52 * 1_024 * 16 * 2);
    expect(d1.contractionSpectrum).toBe(52 * 1_024 * 16 * 4);
    expect(d3.inputSpectrum).toBe(53 * 1_024 * 16 * 2);
    expect(d9.inputSpectrum).toBe(54 * 1_024 * 16 * 2);
    expect(d1.total).toBe(d1.inputSpectrum + d1.contractionSpectrum);
  });

  it("accounts for all twelve persistent routes and bounded scratch", () => {
    const persistent = buildOpt0077PersistentPayloadAccounting();
    expect(persistent).toMatchObject({ routeCount: 12,
      currentBytes: 56_426_496, candidateBytes: 128_974_848,
      increaseBytes: 72_548_352,
      nativeAndTransformedDuplicateSurvives: false });
    const scratch = buildOpt0077ScratchAccounting({
      maxStorageBufferBindingSize: 256 * 1_024 * 1_024,
      maxBufferSize: 256 * 1_024 * 1_024,
    });
    expect(scratch).toMatchObject({
      maximumInputSpectrumBytes: 54 * 1_024 * 16 * 2,
      maximumContractionSpectrumBytes: 54 * 1_024 * 16 * 4,
      maximumTotalScratchBytes: 54 * 1_024 * 16 * 6,
      maximumScratchBindingCount: 2,
      authenticatedLimits: {
        everyScratchBindingWithinAuthenticatedLimits: true,
      },
      passed: true,
    });
  });

  it("requires scalar-oracle numerics, deterministic reruns, and guarded scratch", () => {
    for (const token of [
      'type ExecutionArm = Opt0077Arm | "scalar-oracle"',
      '"current", "candidate", "scalar-oracle", "candidate"',
      "candidateRepeatComparison.differingU16Count === 0",
      "numericalPassed(candidateScalarMetrics)",
      "const NRMSE_MAXIMUM = 0.001",
      "const SNR_MINIMUM_DB = 60",
      "const PEARSON_MINIMUM = 0.99999",
      "const RELATIVE_MAXIMUM_ABSOLUTE_ERROR_MAXIMUM = 0.01",
      "outOfRangeWriteCount",
      "adjacentBeforeIntact",
      "adjacentAfterIntact",
      "inputSpectrum: inputCheck",
      "contractionSpectrum: contractionCheck",
      "unusedTailIntact",
      "finiteClassTransitionDistribution",
      "transformedMaximumAbsoluteValue",
      "finiteTransformAmplificationStressPassed",
      "candidateVersusCurrent: Object.freeze",
      "perStratum: Object.freeze",
      "allCaseAggregate",
    ]) expect(HARNESS_SOURCE).toContain(token);
    expect(RECORD_SOURCE).toContain(
      "Run the current OPT-0051 row-reuse K4 control once, the candidate twice",
    );
  });

  it("passes only a literal six-round 1.35x/five-pair/four-second timing win", () => {
    const result = summarizeOpt0077Timing(timingInput(20, 10));
    expect(result).toMatchObject({
      sampleCountPerArm: 6,
      currentAggregateDispatches: 42,
      candidateAggregateDispatches: 126,
      speedup: { meanGpu: 2, medianGpu: 2, meanWall: 2, medianWall: 2 },
      gates: {
        everyStratumMedianGpuNonSlower: true,
        gpuPairWins: 6,
        wallPairWins: 6,
        meanGpuSpeedupPassed: true,
        medianGpuSpeedupPassed: true,
        meanWallSpeedupPassed: true,
        medianWallSpeedupPassed: true,
        projectedSavingPassed: true,
      },
      passed: true,
    });
    expect(Number((result.gates as Record<string, unknown>)[
      "projectedSavingMilliseconds"
    ])).toBeCloseTo(8_062.6);
  });

  it("fails on two paired losses, one slower stratum, or a sub-1.35x score", () => {
    const pairs = timingInput(20, 10);
    const candidates = [...pairs.candidate];
    candidates[0] = sample(21, 21, 126);
    candidates[1] = sample(21, 21, 126);
    expect(summarizeOpt0077Timing({ ...pairs,
      candidate: candidates })).toMatchObject({
      gates: { gpuPairWins: 4, wallPairWins: 4,
        gpuPairWinsPassed: false, wallPairWinsPassed: false },
      passed: false,
    });
    const slowStratum = timingInput(20, 10);
    const strata = [...slowStratum.strata];
    strata[4] = { ...strata[4]!,
      candidateGpuMilliseconds: six(21) };
    expect(summarizeOpt0077Timing({ ...slowStratum, strata })).toMatchObject({
      gates: { everyStratumMedianGpuNonSlower: false }, passed: false,
    });
    expect(summarizeOpt0077Timing(timingInput(20, 15))).toMatchObject({
      gates: { meanGpuSpeedupPassed: false, medianGpuSpeedupPassed: false,
        meanWallSpeedupPassed: false, medianWallSpeedupPassed: false },
      passed: false,
    });
  });

  it("parses a continuous launch slice and a trace covering cleanup", () => {
    const ready = 10_000;
    const launch = parseOpt0077ThermalLaunchGate(new URLSearchParams({
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
    }), ready, 42_000);
    expect(launch).toMatchObject({ observationCount: 31,
      maximumPollGapMilliseconds: 1004, launchDelayMilliseconds: 1000 });
    const completion = parseOpt0077ThermalCompletion(new URLSearchParams({
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
    }), launch, 69_000);
    expect(completion).toMatchObject({ coversCleanup: true,
      initialLevel: 0, finalLevel: 0 });
  });

  it("rejects stale/non-nominal launch gates and traces ending before cleanup", () => {
    const base = {
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalCommand: "notifyutil -g com.apple.system.thermalpressurelevel",
      thermalTraceStartedAtEpochMilliseconds: "9000",
      thermalGateStartedAtEpochMilliseconds: "11000",
      thermalGateCompletedAtEpochMilliseconds: "41000",
      thermalGateObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalGateMaximumPollGapMilliseconds: "1000",
      thermalGateNonNominalObservations: "0",
      thermalGateMissingObservations: "0",
    };
    expect(() => parseOpt0077ThermalLaunchGate(
      new URLSearchParams({ ...base, thermalGateNonNominalObservations: "1" }),
      10_000, 42_000,
    )).toThrow(/nominal thermal launch gate failed/u);
    const launch = parseOpt0077ThermalLaunchGate(new URLSearchParams(base),
      10_000, 42_000);
    expect(() => parseOpt0077ThermalCompletion(new URLSearchParams({
      thermalTraceSchema:
        "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
      thermalTraceSha256: "b".repeat(64), thermalTraceByteLength: "100",
      thermalTraceCompletedAtEpochMilliseconds: "60000",
      thermalTraceObservations: "52",
      thermalTraceMaximumPollGapMilliseconds: "1000",
      thermalTraceNonNominalObservations: "0",
      thermalTraceMissingObservations: "0",
      thermalTraceInitialLevel: "0", thermalTraceFinalLevel: "0",
    }), launch, 61_000)).toThrow(/through-cleanup thermal trace failed/u);
  });

  it("keeps timing disabled until correctness/warmup and reconciles lifecycle", () => {
    expect(HTML_SOURCE).toContain('id="run" type="button" disabled');
    expect(HTML_SOURCE).toContain('id="finalize" type="button" disabled');
    for (const token of [
      "completedBeforeReady: true",
      "symmetric: true, completedBeforeReady: true",
      "readyAtEpochMilliseconds = Date.now()",
      "querySet?.destroy()",
      "postDestroyRejected",
      "candidateKernel?.destroy()",
      "tracker.destroyAll()",
      "zeroLiveBuffers",
      "zeroLiveBytes",
      "createdEqualsDestroyed",
      "mapsBalanced",
      "transformedInsideTiming: false",
      "unchangedTimingRetryPerformed: false",
      "buildPreparedProductionWorkingSetReceipt",
      "distinctWeightKeyCount !== 12",
      "distinctNativeBufferCount !== 12",
      "distinctCurrentPackedBufferCount !== 12",
      "distinctCandidateTransformedBufferCount !== 12",
      "distinctNativeSha256Count !== 12",
      "distinctCurrentPackedSha256Count !== 12",
      "distinctCandidateTransformedSha256Count !== 12",
      "c128OperationMultiplicitySplit",
      "productionOperationWeightBindingCount: 12",
      "distinctNativeCurrentAndTransformedBuffersPerOperation: true",
      "stableStringHash32(operationLabel)",
      "perStratumInstancesPerSample: 1",
      "perStratumCurrentDispatches: 1",
      "perStratumCandidateDispatches: 3",
    ]) expect(HARNESS_SOURCE).toContain(token);
    expect(HTML_SOURCE).toContain("Keep the trace running through cleanup");
  });

  it("self-authenticates all sources and leaves production unchanged", () => {
    for (const raw of [
      "vae-conv1d-fp16-k4-row-reuse-shape-selector.ts?raw",
      "vae-conv1d-fp16-k4-row-reuse-16x64.ts?raw",
      "vae-conv1d-fp16-subgroup.ts?raw",
      "vae-conv1d-fp16-rfft16.ts?raw",
      "vae-conv1d-fp16-rfft16-math.ts?raw",
      "OPT-0077-vae-k7-rfft16-transform-domain.md?raw",
      "opt-0077-vae-k7-rfft16.html?raw",
      "opt-0077-vae-k7-rfft16.ts?raw",
    ]) expect(HARNESS_SOURCE).toContain(raw);
    for (const generator of [
      "aceOpt0051VaeConv1dK4RowReuse16x64Wgsl",
      "aceOpt0077VaeK7Rfft16Wgsl",
      "aceFp16VaeConv1dSubgroupWgsl",
      "currentSha256",
      "candidateSha256",
      "scalarOracleSha256",
    ]) expect(HARNESS_SOURCE).toContain(generator);
    expect(HARNESS_SOURCE).toContain(
      "productionIntegrationAuthorized: false",
    );
    expect(HARNESS_SOURCE).toContain("qualityOrListeningClaim: false");
  });
});

function timingInput(current: number, candidate: number): Opt0077TimingInput {
  return {
    current: sixSamples(current, 42),
    candidate: sixSamples(candidate, 126),
    strata: buildOpt0077ProductionStrata().map(({ id }) => ({ id,
      currentGpuMilliseconds: six(current / 4),
      candidateGpuMilliseconds: six(candidate / 4) })),
  };
}

function sixSamples(value: number, dispatches: number): Opt0077TimestampSample[] {
  return Array.from({ length: 6 }, () => sample(value, value, dispatches));
}

function six(value: number): number[] {
  return Array.from({ length: 6 }, (_, index) => value + index * 0.001);
}

function sample(
  gpuMilliseconds: number,
  wallMilliseconds: number,
  dispatchCount: number,
): Opt0077TimestampSample {
  return { gpuMilliseconds, wallMilliseconds,
    gpuElapsedNanoseconds: Math.round(gpuMilliseconds * 1_000_000),
    dispatchCount, commandBufferCount: 1, queueDrainCount: 1 };
}

function source(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), "utf8");
}
