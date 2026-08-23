import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0081_REPRESENTATIVE_DENSE_TAPS,
  AceOpt0081RepresentativeCheckpointExecutionError,
} from "../src/webgpu/dit-backend.js";
import {
  serializeOpt0018Failure,
} from "./browser/opt-0018-dit-m2250-production-family-profile.js";
import {
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
} from "../src/webgpu/dit-attention-profile.js";
import {
  OPT_0081_REPRESENTATIVE_BOUNDARY_ROLES,
  OPT_0081_REPRESENTATIVE_CONDITION_SHA256,
  OPT_0081_REPRESENTATIVE_CONTEXT_SHA256,
  OPT_0081_REPRESENTATIVE_DENSE_OUTPUT_ROLES,
  OPT_0081_REPRESENTATIVE_EPOCHS,
  OPT_0081_REPRESENTATIVE_LYRIC_TOKEN_SHA256,
  OPT_0081_REPRESENTATIVE_TEXT_TOKEN_SHA256,
  buildOpt0081RepresentativeRounds,
  inspectOpt0081RepresentativeCorrectness,
  numberToOpt0081Float16Bits,
  opt0081RepresentativeThermalTimingPassed,
  parseOpt0081RepresentativeThermalCompletion,
  parseOpt0081RepresentativeThermalLaunch,
  requireOpt0081RepresentativeReceivedThermalLaunch,
  requireOpt0081RepresentativeTopology,
  summarizeOpt0081RepresentativeTiming,
  type Opt0081RepresentativeCancellationEvidence,
  type Opt0081RepresentativeCorrectness,
  type Opt0081RepresentativeHeartbeat,
  type Opt0081RepresentativeLifecycleEvidence,
  type Opt0081RepresentativeRawCheckpoint,
  type Opt0081RepresentativeTimingSample,
  type Opt0081RepresentativeTopology,
} from "./browser/opt-0081-dit-f16-representative-layers-contract.js";
import {
  classifyOpt0081RepresentativeDisposition,
  createOpt0081RepresentativeConditioningEvidence,
  createOpt0081RepresentativeIdentity,
  type Opt0081RepresentativeArenaEvidence,
  type Opt0081RepresentativeConditioningAuthorityEvidence,
  type Opt0081RepresentativePreparationEvidence,
} from "./browser/opt-0081-dit-f16-representative-layers-result.js";

const CONTRACT_SOURCE = source(
  "./browser/opt-0081-dit-f16-representative-layers-contract.ts",
);
const RESULT_SOURCE = source(
  "./browser/opt-0081-dit-f16-representative-layers-result.ts",
);
const WORKER_SOURCE = source(
  "./browser/opt-0081-dit-f16-representative-layers-worker.ts",
);
const PAGE_SOURCE = source(
  "./browser/opt-0081-dit-f16-representative-layers.ts",
);
const HTML_SOURCE = source(
  "./browser/opt-0081-dit-f16-representative-layers.html",
);

describe("OPT-0081 representative two-layer browser gate", () => {
  it("retains bounded first-checkpoint context and the direct root cause", () => {
    const failure = new AceOpt0081RepresentativeCheckpointExecutionError(
      "A",
      0,
      "selfModulated",
      "prefill-layer-0-target",
      new DOMException(
        "WriteBuffer data size exceeds typed-array elements",
        "OperationError",
      ),
    );
    const serialized = serializeOpt0018Failure(failure);
    expect(serialized).toMatchObject({
      name: "AceOpt0081RepresentativeCheckpointExecutionError",
      message:
        "OPT-0081 A layer 0 selfModulated checkpoint failed during prefill-layer-0-target",
      ownFields: {
        arm: "A",
        layer: 0,
        tap: "selfModulated",
        substage: "prefill-layer-0-target",
        causeName: "OperationError",
        causeMessage: "WriteBuffer data size exceeds typed-array elements",
        causeCode: 0,
      },
    });
  });

  it("freezes the eight paired orders and cross-layer fourth epoch", () => {
    const rounds = buildOpt0081RepresentativeRounds();
    expect(rounds.map(({ armOrder }) => armOrder.join(""))).toEqual([
      "AB", "BA", "BA", "AB", "AB", "BA", "BA", "AB",
    ]);
    expect(rounds.filter(({ direction }) => direction === "forward"))
      .toHaveLength(4);
    expect(rounds.filter(({ direction }) => direction === "reverse"))
      .toHaveLength(4);
    expect(OPT_0081_REPRESENTATIVE_EPOCHS).toEqual([
      { epochIndex: 0, firstCommandIndex: 25, lastCommandIndex: 28,
        commandCount: 4 },
      { epochIndex: 1, firstCommandIndex: 29, lastCommandIndex: 32,
        commandCount: 4 },
      { epochIndex: 2, firstCommandIndex: 33, lastCommandIndex: 36,
        commandCount: 4 },
      { epochIndex: 3, firstCommandIndex: 37, lastCommandIndex: 40,
        commandCount: 4 },
      { epochIndex: 4, firstCommandIndex: 41, lastCommandIndex: 44,
        commandCount: 4 },
      { epochIndex: 5, firstCommandIndex: 45, lastCommandIndex: 48,
        commandCount: 4 },
      { epochIndex: 6, firstCommandIndex: 49, lastCommandIndex: 52,
        commandCount: 4 },
    ]);
    const changedRoute = Object.freeze({
      ...topology(),
      layerAttentionRoutes: Object.freeze([
        Object.freeze({ layer: 0 as const, self: "quad-query32-full" as const,
          cross: "query8-cross" as const }),
        topology().layerAttentionRoutes[1],
      ]),
    }) as unknown as Opt0081RepresentativeTopology;
    expect(() => requireOpt0081RepresentativeTopology(changedRoute))
      .toThrow(/command topology/u);
  });

  it("requires both directional material floors and the 7/8 pair gate", () => {
    const passing = summarizeOpt0081RepresentativeTiming(timingSamples(
      () => ({ A: 100, B: 60 }),
    ));
    expect(passing).toMatchObject({
      aMeanWallMilliseconds: 100,
      bMeanWallMilliseconds: 60,
      aMedianWallMilliseconds: 100,
      bMedianWallMilliseconds: 60,
      pairedWins: 8,
      forwardPairedWins: 4,
      reversePairedWins: 4,
      forwardMeanPairedSavingMilliseconds: 40,
      reverseMeanPairedSavingMilliseconds: 40,
      forwardPairedSavingSampleStandardDeviationMilliseconds: 0,
      reversePairedSavingSampleStandardDeviationMilliseconds: 0,
      forwardPairedSavingLower95Milliseconds: 40,
      reversePairedSavingLower95Milliseconds: 40,
      forwardProjectedEightEvaluationSavingMilliseconds: 3_840,
      reverseProjectedEightEvaluationSavingMilliseconds: 3_840,
      passed: true,
    });

    const directionalMiss = summarizeOpt0081RepresentativeTiming(
      timingSamples((direction) => direction === "forward"
        ? { A: 100, B: 60 } : { A: 100, B: 75 }),
    );
    expect(directionalMiss).toMatchObject({
      pairedWins: 8,
      forwardMeanPairedSavingMilliseconds: 40,
      reverseMeanPairedSavingMilliseconds: 25,
      passed: false,
    });
    expect(classifyOpt0081RepresentativeDisposition(receiptBase(
      directionalMiss,
      correctness(),
    ))).toBe("inconclusive-directional-or-material-wall-evidence");
  });

  it("rejects directional variance whose 95% lower bound overlaps the floor", () => {
    let forwardCandidateIndex = 0;
    const samples = timingSamples(() => ({ A: 300, B: 100 })).map((sample) => {
      if (sample.direction !== "forward" || sample.arm !== "B") return sample;
      const wall = forwardCandidateIndex++ === 3 ? 400 : 100;
      return Object.freeze({
        ...sample,
        sliceDrainedAtPerformanceMilliseconds:
          sample.sliceStartedAtPerformanceMilliseconds + wall,
        sliceWallMilliseconds: wall,
      });
    });
    const summary = summarizeOpt0081RepresentativeTiming(samples);
    expect(summary).toMatchObject({
      pairedWins: 7,
      forwardPairedWins: 3,
      reversePairedWins: 4,
      forwardMeanPairedSavingMilliseconds: 125,
      reverseMeanPairedSavingMilliseconds: 200,
      passed: false,
    });
    expect(summary.forwardPairedSavingLower95Milliseconds).toBeLessThan(31.25);
    expect(classifyOpt0081RepresentativeDisposition(
      receiptBase(summary, correctness()),
    )).toBe("inconclusive-directional-or-material-wall-evidence");
  });

  it("authenticates the complete A/A, A/B, B/B raw inventory", () => {
    const value = correctness();
    expect(inspectOpt0081RepresentativeCorrectness(value)).toEqual({
      structurallyValid: true,
      observedRawMismatch: false,
      passed: true,
    });
    expect(value.boundaryRuns).toHaveLength(2);
    expect([value.controlBoundaryRepeat, value.candidateBoundaryRepeat]
      .every((checkpoints) => checkpoints.length === 12 &&
        checkpoints.reduce((sum, checkpoint) =>
          sum + checkpoint.comparedWords, 0) === 73_728_000
      )).toBe(true);
    expect(value.boundaryRuns.every(({ checkpoints }) =>
      checkpoints.length === 12 && checkpoints.reduce((sum, checkpoint) =>
        sum + checkpoint.comparedWords, 0) === 73_728_000
    )).toBe(true);
    expect(value.comparisons.every(({ denseOutputs, layerOutputs }) =>
      denseOutputs.length === 18 && layerOutputs.length === 2 &&
      denseOutputs.reduce((sum, checkpoint) =>
        sum + checkpoint.comparedWords, 0) === 110_592_000 &&
      layerOutputs.reduce((sum, checkpoint) =>
        sum + checkpoint.comparedWords, 0) === 9_216_000
    )).toBe(true);
    expect(inspectOpt0081RepresentativeCorrectness(Object.freeze({
      ...value,
      passed: false,
      controlBoundaryRepeat: value.controlBoundaryRepeat.slice(1),
    }))).toEqual({
      structurallyValid: false,
      observedRawMismatch: false,
      passed: false,
    });
  });

  it("retains identical canonical setup/main conditioning authority", () => {
    const expected = conditioningAuthority();
    expect(createOpt0081RepresentativeConditioningEvidence(
      expected,
      expected,
    )).toMatchObject({
      setupAndMainExact: true,
      capturedBeforeReady: true,
      setup: expected,
      main: expected,
    });
    expect(() => createOpt0081RepresentativeConditioningEvidence(
      expected,
      Object.freeze({ ...expected, conditionSha256: "0".repeat(64) }),
    )).toThrow(/canonical conditioning authority/u);
    expect(() => createOpt0081RepresentativeConditioningEvidence(
      Object.freeze({ ...expected, textTokenCount: 81 }) as unknown as
        Opt0081RepresentativeConditioningAuthorityEvidence,
      expected,
    )).toThrow(/canonical conditioning authority/u);
    const receipt = receiptBase(
      summarizeOpt0081RepresentativeTiming(timingSamples(
        () => ({ A: 100, B: 60 }),
      )),
      correctness(),
    );
    expect(classifyOpt0081RepresentativeDisposition(Object.freeze({
      ...receipt,
      conditioning: Object.freeze({
        ...receipt["conditioning"] as Readonly<Record<string, unknown>>,
        main: Object.freeze({
          ...expected,
          contextSha256: "f".repeat(64),
        }),
      }),
    }))).toBe(
      "inconclusive-invalid-correctness-topology-or-lifecycle-evidence",
    );
  });

  it("maps every receipt-facing dense tap to the exact core tap once", () => {
    const mappings = [
      ["selfQuery", "selfQueryFlat"],
      ["selfKey", "selfKeyFlat"],
      ["selfValue", "selfValueFlat"],
      ["selfOutput", "selfProjectedAttention"],
      ["crossQuery", "crossQueryFlat"],
      ["crossOutput", "crossProjectedAttention"],
      ["mlpGate", "gate"],
      ["mlpUp", "up"],
      ["mlpDown", "projectedMlp"],
    ] as const;
    expect(mappings.map(([, core]) => core))
      .toEqual([...ACE_OPT_0081_REPRESENTATIVE_DENSE_TAPS]);
    expect(mappings.map(([receipt]) => receipt))
      .toEqual(OPT_0081_REPRESENTATIVE_DENSE_OUTPUT_ROLES.map(({ tap }) => tap));
    for (const [receipt, core] of mappings) {
      expect(WORKER_SOURCE).toContain(
        `{ receipt: "${receipt}" as const,`,
      );
      expect(WORKER_SOURCE).toContain(`core: "${core}" as const`);
    }
  });

  it("uses exact binary16 ties-to-even for every typed producer boundary", () => {
    expect([
      0, -0, 2 ** -25, 3 * 2 ** -25, 2 ** -14,
      1.00048828125, 1.00146484375, 65_504, 65_520,
    ].map(numberToOpt0081Float16Bits)).toEqual([
      0x0000, 0x8000, 0x0000, 0x0002, 0x0400,
      0x3c00, 0x3c02, 0x7bff, 0x7c00,
    ]);
  });

  it("classifies only a structurally observed bit mismatch as negative", () => {
    const value = correctness();
    const firstRun = value.boundaryRuns[0]!;
    const mismatchCheckpoint = Object.freeze({
      ...firstRun.checkpoints[0]!,
      differingWordCount: 1,
      exact: false,
    });
    const mismatch = Object.freeze({
      ...value,
      passed: false,
      boundaryRuns: Object.freeze([
        Object.freeze({ ...firstRun, checkpoints: Object.freeze([
          mismatchCheckpoint,
          ...firstRun.checkpoints.slice(1),
        ]) }),
        value.boundaryRuns[1]!,
      ]),
    });
    expect(inspectOpt0081RepresentativeCorrectness(mismatch)).toEqual({
      structurallyValid: true,
      observedRawMismatch: true,
      passed: false,
    });
    const timing = summarizeOpt0081RepresentativeTiming(timingSamples(
      () => ({ A: 100, B: 60 }),
    ));
    expect(classifyOpt0081RepresentativeDisposition(
      receiptBase(timing, mismatch),
    )).toBe("negative-stop-observed-raw-bit-correctness-mismatch");
    expect(classifyOpt0081RepresentativeDisposition(receiptBase(timing,
      Object.freeze({ ...mismatch, uncapturedGpuErrorCount: 1 }),
    ))).toBe(
      "inconclusive-invalid-correctness-topology-or-lifecycle-evidence",
    );
    expect(classifyOpt0081RepresentativeDisposition({
      ...receiptBase(timing, value),
      correctness: { passed: false },
    })).toBe(
      "inconclusive-invalid-correctness-topology-or-lifecycle-evidence",
    );
  });

  it("requires the 30-second external gate and trace through cleanup", () => {
    const launch = parseOpt0081RepresentativeThermalLaunch(
      new URLSearchParams({
        thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
        thermalCommand:
          "notifyutil -g com.apple.system.thermalpressurelevel",
        thermalTraceStartedAtEpochMilliseconds: "999000",
        thermalGateStartedAtEpochMilliseconds: "1005000",
        thermalGateCompletedAtEpochMilliseconds: "1035000",
        thermalGateObservations: "31",
        thermalPollMilliseconds: "1000",
        thermalGateMaximumPollGapMilliseconds: "1004",
        thermalGateNonNominalObservations: "0",
        thermalGateMissingObservations: "0",
      }),
      1_000_000,
      1_036_000,
    );
    expect(launch).toMatchObject({
      observationCount: 31,
      maximumPollGapMilliseconds: 1_004,
      launchDelayMilliseconds: 1_000,
    });
    expect(requireOpt0081RepresentativeReceivedThermalLaunch(
      launch,
      1_000_000,
      1_036_250,
    )).toBe(launch);
    expect(() => requireOpt0081RepresentativeReceivedThermalLaunch(
      launch,
      1_000_000,
      1_041_001,
    )).toThrow(/delivery stale/u);
    expect(parseOpt0081RepresentativeThermalCompletion(
      new URLSearchParams({
        thermalTraceSchema:
          "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
        thermalTraceSha256: "a".repeat(64),
        thermalTraceByteLength: "8192",
        thermalTraceCompletedAtEpochMilliseconds: "1050000",
        thermalTraceObservations: "52",
        thermalTraceMaximumPollGapMilliseconds: "1004",
        thermalTraceNonNominalObservations: "0",
        thermalTraceMissingObservations: "0",
        thermalTraceInitialLevel: "0",
        thermalTraceFinalLevel: "0",
        thermalTraceTransitionsJson: "[]",
      }),
      launch,
      1_049_000,
    )).toMatchObject({ coversCleanup: true, initialLevel: 0, finalLevel: 0,
      transitions: [] });
    const transitionedParameters = new URLSearchParams({
        thermalTraceSchema:
          "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
        thermalTraceSha256: "b".repeat(64),
        thermalTraceByteLength: "9000",
        thermalTraceCompletedAtEpochMilliseconds: "1050000",
        thermalTraceObservations: "52",
        thermalTraceMaximumPollGapMilliseconds: "1004",
        thermalTraceNonNominalObservations: "5",
        thermalTraceMissingObservations: "0",
        thermalTraceInitialLevel: "0",
        thermalTraceFinalLevel: "2",
        thermalTraceTransitionsJson: JSON.stringify([
          { atEpochMilliseconds: 1_040_000, level: 2 },
        ]),
      });
    const transitioned = parseOpt0081RepresentativeThermalCompletion(
      transitionedParameters,
      launch,
      1_049_000,
    );
    expect(transitioned).toMatchObject({
      nonNominalObservationCount: 5,
      finalLevel: 2,
      transitions: [{ atEpochMilliseconds: 1_040_000, level: 2 }],
    });
    expect(opt0081RepresentativeThermalTimingPassed(
      transitioned,
      launch,
      1_049_000,
    )).toBe(true);
    const missingDisclosure = new URLSearchParams(transitionedParameters);
    missingDisclosure.set("thermalTraceTransitionsJson", "[]");
    expect(() => parseOpt0081RepresentativeThermalCompletion(
      missingDisclosure,
      launch,
      1_049_000,
    )).toThrow(/through-cleanup trace/u);
    const contradictoryCount = new URLSearchParams(transitionedParameters);
    contradictoryCount.set("thermalTraceNonNominalObservations", "0");
    expect(() => parseOpt0081RepresentativeThermalCompletion(
      contradictoryCount,
      launch,
      1_049_000,
    )).toThrow(/through-cleanup trace/u);
  });

  it("freezes the worker-only graph, thermal, and receipt protocol", () => {
    for (const token of [
      "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6",
      "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f",
      "OPT_0018_CANONICAL_REQUEST_SHA256",
      "runPrecompute", "snapshotCheckpoint", "runCancellationPreflight",
      "warmup(\"A\")", "warmup(\"B\")", "runTimedSlice",
      "OPT_0081_REPRESENTATIVE_EPOCHS", "pendingDescriptorCountAfterRun",
      "Opt0081RepresentativeCorrectnessFailure",
    ]) expect(WORKER_SOURCE).toContain(token);
    for (const hash of [
      OPT_0081_REPRESENTATIVE_TEXT_TOKEN_SHA256,
      OPT_0081_REPRESENTATIVE_LYRIC_TOKEN_SHA256,
      OPT_0081_REPRESENTATIVE_CONDITION_SHA256,
      OPT_0081_REPRESENTATIVE_CONTEXT_SHA256,
    ]) expect(CONTRACT_SOURCE).toContain(hash);
    expect(CONTRACT_SOURCE).toContain(
      "031e418ac5db37355fe5e265a005cb280e02ce418e560312ac89fa184bb8862f",
    );
    expect(RESULT_SOURCE).toContain(
      "negative-stop-observed-raw-bit-correctness-mismatch",
    );
    expect(WORKER_SOURCE).not.toContain("Math.random");
    expect(WORKER_SOURCE).not.toContain("runTimedSlice(\"C\"");
    expect(WORKER_SOURCE).not.toContain("timestampWrites");
    expect(PAGE_SOURCE).toContain("new Worker(");
    expect(PAGE_SOURCE).not.toContain("createAceWebGpuPipelineBackend");
    expect(PAGE_SOURCE).toContain(
      "OPT_0081_REPRESENTATIVE_HEARTBEAT_INTERVAL_MS",
    );
    expect(PAGE_SOURCE).toContain("parseOpt0081RepresentativeThermalLaunch");
    expect(PAGE_SOURCE).toContain(
      "parseOpt0081RepresentativeThermalCompletion",
    );
    expect(HTML_SOURCE).toContain("AB, BA, BA, AB, AB, BA, BA, AB");
    expect(HTML_SOURCE).toContain("at least 30 seconds");
    expect(HTML_SOURCE).toContain("production remains unchanged");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0081-dit-f16-representative-layers.ts"',
    );
  });

  it("keeps the stable result B-only and benchmark-only", () => {
    for (const token of [
      'selectedDiagnosticArm: passed ? "B" : null',
      "completeEvaluationFollowUpAuthorized: passed",
      "productionIntegrationAuthorized: false",
      "packageChangeAuthorized: false",
      "fullTrajectoryAuthorized: false",
      "productGateAuthorized: false",
      "unchangedTimingRetryPerformed: false",
    ]) expect(RESULT_SOURCE).toContain(token);
    expect(CONTRACT_SOURCE).toContain(
      'Object.freeze(["A", "B"] as const)',
    );
    expect(CONTRACT_SOURCE).not.toContain('readonly selectedDiagnosticArm: "C"');
  });
});

function topology(): Opt0081RepresentativeTopology {
  const labels = Object.freeze(Array.from({ length: 28 }, (_, index) =>
    `physical-${index + 25}`));
  return Object.freeze({
    firstCommandIndex: 25,
    lastCommandIndex: 52,
    commandBufferCount: 28,
    completionFenceRequestedCount: 28,
    completionFenceSettledCount: 28,
    completionFenceRejectedCount: 0,
    completionEpochCount: 7,
    trueQueueDrainCount: 7,
    cooperativeIdleTurnCount: 6,
    requestedCooperativeIdleMilliseconds: 6,
    maximumOutstandingCommandBuffers: 2,
    maximumPendingDescriptorCount: 2,
    pendingDescriptorCountAfterRun: 0,
    timestepCommandCount: 1,
    inputProjectionCommandCount: 1,
    slidingLayerCommandCount: 11,
    fullLayerCommandCount: 15,
    producerStoreCount: 12,
    denseConsumerCount: 18,
    layerAttentionRoutes: Object.freeze([
      Object.freeze({ layer: 0 as const, self: "query8-sliding" as const,
        cross: "query8-cross" as const }),
      Object.freeze({ layer: 1 as const, self: "quad-query32-full" as const,
        cross: "query8-cross" as const }),
    ] as const),
    descriptorOrTapCommandCount: 0,
    timestampQueryCount: 0,
    measurementReadbackCount: 0,
    measurementMapCount: 0,
    attentionRuntimeProfile:
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    epochs: OPT_0081_REPRESENTATIVE_EPOCHS,
    physicalCommandLabels: labels,
    progressLabels: labels,
  });
}

function timingSamples(
  values: (direction: "forward" | "reverse") =>
    Readonly<Record<"A" | "B", number>>,
): readonly Opt0081RepresentativeTimingSample[] {
  const samples: Opt0081RepresentativeTimingSample[] = [];
  for (const round of buildOpt0081RepresentativeRounds()) {
    for (let occurrenceIndex = 0; occurrenceIndex < 2; occurrenceIndex += 1) {
      const arm = round.armOrder[occurrenceIndex]!;
      const wall = values(round.direction)[arm];
      const started = 1_000 + round.roundIndex * 1_000 + occurrenceIndex * 200;
      samples.push(Object.freeze({
        roundIndex: round.roundIndex,
        occurrenceIndex: occurrenceIndex as 0 | 1,
        arm,
        direction: round.direction,
        sliceStartedAtPerformanceMilliseconds: started,
        sliceDrainedAtPerformanceMilliseconds: started + wall,
        sliceWallMilliseconds: wall,
        epochs: Object.freeze(OPT_0081_REPRESENTATIVE_EPOCHS.map((epoch) =>
          Object.freeze({
            epochIndex: epoch.epochIndex,
            firstCommandIndex: epoch.firstCommandIndex,
            lastCommandIndex: epoch.lastCommandIndex,
            submitThroughTrueDrainMilliseconds: 8,
          })
        )),
        completionFenceLatenciesMilliseconds: Object.freeze(
          Array.from({ length: 28 }, () => 2),
        ),
        topology: topology(),
      }));
    }
  }
  return Object.freeze(samples);
}

function correctness(): Opt0081RepresentativeCorrectness {
  const boundary = (run: "B1" | "B2") => Object.freeze({
    run,
    checkpoints: checkpoints(OPT_0081_REPRESENTATIVE_BOUNDARY_ROLES),
  });
  const comparison = (id: "A/A" | "A/B" | "B/B") => Object.freeze({
    comparison: id,
    denseOutputs: checkpoints(OPT_0081_REPRESENTATIVE_DENSE_OUTPUT_ROLES),
    layerOutputs: checkpoints([{ tap: "layerOutput", words: 4_608_000 }]),
  });
  return Object.freeze({
    completedBeforeReady: true,
    runOrder: Object.freeze(["A", "A", "B", "B"] as const),
    controlBoundaryRepeat:
      checkpoints(OPT_0081_REPRESENTATIVE_BOUNDARY_ROLES),
    boundaryRuns: Object.freeze([boundary("B1"), boundary("B2")]),
    candidateBoundaryRepeat:
      checkpoints(OPT_0081_REPRESENTATIVE_BOUNDARY_ROLES),
    comparisons: Object.freeze([
      comparison("A/A"), comparison("A/B"), comparison("B/B"),
    ]),
    boundaryWordsPerCandidateRun: 73_728_000,
    denseOutputWordsPerComparison: 110_592_000,
    layerOutputWordsPerComparison: 9_216_000,
    uncapturedGpuErrorCount: 0,
    validationErrorCount: 0,
    deviceLossCount: 0,
    passed: true,
  });
}

function checkpoints(
  roles: readonly Readonly<{ readonly tap: string; readonly words: number }>[],
): readonly Opt0081RepresentativeRawCheckpoint[] {
  return Object.freeze([0, 1].flatMap((layer) => roles.map(({ tap, words }) =>
    Object.freeze({
      layer: layer as 0 | 1,
      tap,
      comparedWords: words,
      differingWordCount: 0,
      unwrittenWordCount: 0,
      exact: true,
      signedZeroExact: true,
      finite: true,
      qNaNPrefillOverwritten: true,
      firstWordCovered: true,
      lastWordCovered: true,
      tailRows2240Through2249Covered: true,
      prefixGuardIntact: true,
      suffixGuardIntact: true,
      adjacentGuardsIntact: true,
      sha256: "a".repeat(64),
    })
  )));
}

function receiptBase(
  timing: ReturnType<typeof summarizeOpt0081RepresentativeTiming>,
  correctnessValue: Opt0081RepresentativeCorrectness,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    identity: createOpt0081RepresentativeIdentity(Object.freeze({
      coreCommit: "1".repeat(40),
      harnessCommit: "2".repeat(40),
      machineModel: "test-machine",
      osVersion: "test-os",
      osBuild: "test-build",
      browserVersion: "test-browser",
      gpuCoreCount: 32,
      memoryBytes: 64 * 1024 ** 3,
    })),
    correctness: correctnessValue,
    conditioning: createOpt0081RepresentativeConditioningEvidence(
      conditioningAuthority(),
      conditioningAuthority(),
    ),
    arena: arena(),
    preparation: preparation(),
    cancellation: cancellation(),
    cleanup: lifecycle(),
    timing,
    heartbeat: heartbeat(),
    cleanupCompletedAtEpochMilliseconds: 40_000,
    thermal: thermalReceipt(),
  });
}

function arena(): Opt0081RepresentativeArenaEvidence {
  return Object.freeze({
    controlArenaBytes: 674_815_488,
    candidateArenaBytes: 601_087_488,
    arenaSavingBytes: 73_728_000,
    controlRoleCount: 123,
    candidateRoleCount: 123,
    controlSlotCount: 98,
    candidateSlotCount: 98,
    candidateSelectedDenseInputSlotIndices:
      Object.freeze([61, 69, 72, 77, 81, 84] as const),
    controlLargestArenaBindingBytes: 55_296_000,
    candidateLargestArenaBindingBytes: 55_296_000,
    largestArenaBindingIncreaseBytes: 0,
    castOrCopyBufferCount: 0,
    passed: true,
  });
}

function preparation(): Opt0081RepresentativePreparationEvidence {
  return Object.freeze({
    correctnessCompletedBeforeReady: true,
    precomputeCompletedBeforeReady: true,
    precomputeCommandBufferCountPerArm: 25,
    warmupArmOrder: Object.freeze(["A", "B"] as const),
    warmupSelectedSliceOccurrencesPerArm: 1,
    everyWarmupTerminallyDrained: true,
    conditioningAuthorityCapturedBeforeReady: true,
    setupAndMainConditioningAuthorityExact: true,
    readyAtEpochMilliseconds: 1_000,
    passed: true,
  });
}

function conditioningAuthority():
  Opt0081RepresentativeConditioningAuthorityEvidence {
  return Object.freeze({
    schema: "ace-opt-0081-representative-conditioning-authority-v1",
    textTokenCount: 82,
    textTokenSha256: OPT_0081_REPRESENTATIVE_TEXT_TOKEN_SHA256,
    lyricTokenCount: 15,
    lyricTokenSha256: OPT_0081_REPRESENTATIVE_LYRIC_TOKEN_SHA256,
    conditionTokens: 98,
    conditionElementCount: 200_704,
    conditionSha256: OPT_0081_REPRESENTATIVE_CONDITION_SHA256,
    contextElementCount: 576_000,
    contextSha256: OPT_0081_REPRESENTATIVE_CONTEXT_SHA256,
  });
}

function cancellation(): Opt0081RepresentativeCancellationEvidence {
  return Object.freeze({
    arm: "B",
    residentArmReused: true,
    successorSubmittedBeforeObservation: true,
    outstandingSuccessorCountAtObservation: 1,
    backfillAfterObservationCount: 0,
    progressAfterObservationCount: 0,
    allSubmittedFencesSettledBeforeRelease: true,
    originalErrorPreserved: true,
    cleanupMilliseconds: 10,
    pendingDescriptorCountAfterCleanup: 0,
    temporaryCreatedBufferCount: 0,
    temporaryDestroyedBufferCount: 0,
    temporaryLiveBufferCountAfterCleanup: 0,
    temporaryLiveByteCountAfterCleanup: 0,
    temporaryRuntimeOwnerCount: 0,
    temporaryDestroyedRuntimeOwnerCount: 0,
    passed: true,
  });
}

function lifecycle(): Opt0081RepresentativeLifecycleEvidence {
  return Object.freeze({
    createdBufferCount: 10,
    destroyedBufferCount: 10,
    maximumLiveByteCount: 100,
    mappedRangeCount: 4,
    unmappedRangeCount: 4,
    liveBufferCount: 0,
    liveByteCount: 0,
    liveMapCount: 0,
    pendingDescriptorCount: 0,
    maximumPendingDescriptorCount: 2,
    callbackCount: 0,
    leaseCount: 0,
    maximumActiveLeaseCount: 1,
    correctnessTargetCount: 0,
    maximumCorrectnessTargetCount: 1,
    correctnessRuntimeCount: 0,
    maximumCorrectnessRuntimeCount: 1,
    maximumCorrectnessTargetBytes: 612,
    correctnessTargetCompilationCount: 64,
    correctnessTargetReuseCount: 64,
    checkpointSnapshotCount: 128,
    maximumDetachedCheckpointBytes: 100,
    resetOrPrefillCount: 100,
    profileSwitchCount: 10,
    snapshotMapCount: 128,
    guardedTargetReleaseCount: 64,
    armReleaseCount: 2,
    drainOrderViolationCount: 0,
    profileSwitchWhilePendingCount: 0,
    runtimeOwnerCount: 66,
    destroyedRuntimeOwnerCount: 66,
    residentModelDestroyed: true,
    precomputeCompleted: true,
    destroyCallCount: 2,
    postDestroyRejectedOperationCount: 1,
    drainBeforeEveryResetSwitchMapAndRelease: true,
    profileSwitchOnlyAfterTerminalDrain: true,
    postDestroyRejected: true,
    idempotentDestroy: true,
    setupFailureCleanupPassed: true,
    deviceLossCleanupPassed: true,
    deviceDestroyed: true,
    setupFailureCleanup: Object.freeze({
      schema: "ace-opt-0081-representative-setup-cleanup-v1",
      createdBufferCount: 5,
      destroyedBufferCount: 5,
      liveBufferCount: 0,
      liveByteCount: 0,
      runtimeOwnerCount: 2,
      destroyedRuntimeOwnerCount: 2,
      residentModelDestroyed: true,
      mappedRangeCount: 0,
      unmappedRangeCount: 0,
      liveMapCount: 0,
      pendingDescriptorCount: 0,
      callbackCount: 0,
      leaseCount: 0,
      armReleaseCount: 2,
      drainOrderViolationCount: 0,
      passed: true,
    }),
    deviceLossCleanup: Object.freeze({
      schema: "ace-opt-0081-representative-device-loss-cleanup-v1",
      deviceLossInduced: true,
      deviceLossObserved: true,
      ownerDestroyedAfterLoss: true,
      liveBufferCount: 0,
      liveByteCount: 0,
      liveMapCount: 0,
      pendingDescriptorCount: 0,
      callbackCount: 0,
      leaseCount: 0,
      idempotentDestroyVerified: true,
      postDestroyRejected: true,
      passed: true,
    }),
    passed: true,
  });
}

function heartbeat(): Opt0081RepresentativeHeartbeat {
  return Object.freeze({
    intervalMilliseconds: 50,
    startedAtEpochMilliseconds: 33_000,
    completedAtEpochMilliseconds: 41_000,
    gapsMilliseconds: Object.freeze([50, 50, 50]),
    maximumGapMilliseconds: 50,
    p99GapMilliseconds: 50,
  });
}

function thermalReceipt(): Readonly<Record<string, unknown>> {
  return Object.freeze({
    launch: Object.freeze({
      source: "notifyutil-com.apple.system.thermalpressurelevel",
      command: "notifyutil -g com.apple.system.thermalpressurelevel",
      traceStartedAtEpochMilliseconds: 500,
      gateStartedAtEpochMilliseconds: 2_000,
      gateCompletedAtEpochMilliseconds: 32_000,
      observationCount: 31,
      pollMilliseconds: 1_000,
      maximumPollGapMilliseconds: 1_000,
      nonNominalObservationCount: 0,
      missingObservationCount: 0,
      readyToGateDelayMilliseconds: 1_000,
      launchDelayMilliseconds: 1_000,
    }),
    completion: Object.freeze({
      schema: "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
      sha256: "c".repeat(64),
      byteLength: 8_192,
      completedAtEpochMilliseconds: 50_000,
      observationCount: 50,
      maximumPollGapMilliseconds: 1_000,
      nonNominalObservationCount: 0,
      missingObservationCount: 0,
      initialLevel: 0,
      finalLevel: 0,
      transitions: Object.freeze([]),
      coversCleanup: true,
    }),
    passed: true,
  });
}

function source(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}
