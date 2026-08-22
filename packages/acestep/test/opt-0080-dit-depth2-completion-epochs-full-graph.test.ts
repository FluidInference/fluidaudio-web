import { describe, expect, it } from "vitest";
import experimentSource from
  "../optimization/experiments/OPT-0080-dit-depth2-completion-epochs.md?raw";
import workerSource from
  "./browser/opt-0080-dit-depth2-completion-epochs-full-graph-worker.ts?raw";
import pageSource from
  "./browser/opt-0080-dit-depth2-completion-epochs-full-graph.ts?raw";
import htmlSource from
  "./browser/opt-0080-dit-depth2-completion-epochs-full-graph.html?raw";
import {
  OPT_0080_CANDIDATE_PROFILE,
  OPT_0080_CONTROL_PROFILE,
  OPT_0080_EVALUATION_SLICE_EVIDENCE_SHA256,
  OPT_0080_FULL_ARM_ORDER,
  OPT_0080_FULL_EVALUATION_SHA256,
  OPT_0080_FULL_FINAL_LATENT_SHA256,
  OPT_0080_FULL_PHASE_COMMAND_BUFFER_COUNTS,
  exactOpt0080FullTensorIdentity,
  requireOpt0080FullCancellation,
  requireOpt0080FullTopology,
  summarizeOpt0080FullPerformance,
  type Opt0080FullCancellationEvidence,
  type Opt0080FullHeartbeatCapture,
  type Opt0080FullSchedulingProfile,
  type Opt0080FullThermalGate,
  type Opt0080FullThermalTrace,
  type Opt0080FullTimingSample,
  type Opt0080FullTopologyEvidence,
} from
  "./browser/opt-0080-dit-depth2-completion-epochs-full-graph-contract.js";
import { planOpt0080CompletionEpochs } from
  "./browser/opt-0080-dit-depth2-completion-epochs-contract.js";
import {
  buildOpt0080FullResult,
  type Opt0080FullCorrectnessArm,
  type Opt0080FullCorrectnessTap,
  type Opt0080FullTimedArmResult,
} from
  "./browser/opt-0080-dit-depth2-completion-epochs-full-graph-result.js";

describe("OPT-0080 full M2250 completion-epoch confirmation", () => {
  it("freezes the separately authorized A/B order and accepted trajectory", () => {
    expect(OPT_0080_FULL_ARM_ORDER).toEqual([
      { armId: "A", order: 0, schedulingProfile: "depth1-epoch1" },
      {
        armId: "B",
        order: 1,
        schedulingProfile: "opt-0080-depth2-epoch4",
      },
    ]);
    expect(OPT_0080_FULL_EVALUATION_SHA256).toEqual([
      "d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286",
      "29b113de627c68cfe67cd1d3e2fe3ef3279d968e1584d16ffb3e5f217e220331",
      "d4ef5e04ceaa6b20e8dd0bd87afbb47a61a8c3f2e4e936d74b50998f40d5a1ca",
      "d1cff5e3804105a21a9188ac784a9730fcbd4e589aaade20b5bfefeee4837543",
      "67539d0c653881306821bffa0934c0770a8eaaa96f003cecf5cc6f0d9005bbb2",
      "060c796e1a625b114bc40fc29e310e8c0cce568164b71f776dc0a42c41c0f85e",
      "73c345cb357c9081c33565d61356e2c07090b6c8011d552ddfec6243f862b237",
      "1812a085f48b7879212633c7193dda08ec2854852a492ce661262c5e6be98f4c",
    ]);
    expect(OPT_0080_FULL_FINAL_LATENT_SHA256).toBe(
      OPT_0080_FULL_EVALUATION_SHA256[7],
    );
    expect(OPT_0080_EVALUATION_SLICE_EVIDENCE_SHA256).toBe(
      "a2ad4320f22be728898668d76b84524cd2957cc1e08a588f36a0f3714b183fd9",
    );
    expect(experimentSource).toContain(
      "Only a passing evaluation slice may run one separately cooled full-M2250 A/B",
    );
  });

  it("reconciles the complete nine-phase topology without crossing evaluations", () => {
    expect(OPT_0080_FULL_PHASE_COMMAND_BUFFER_COUNTS).toEqual([
      25, 316, 316, 316, 316, 316, 316, 316, 316,
    ]);
    const candidatePlan = planOpt0080CompletionEpochs(
      OPT_0080_FULL_PHASE_COMMAND_BUFFER_COUNTS,
      4,
    );
    expect(candidatePlan).toHaveLength(639);
    expect(candidatePlan[6]).toMatchObject({
      phaseIndex: 0,
      firstCommandBufferIndex: 24,
      lastCommandBufferIndex: 24,
      commandBufferCount: 1,
    });
    expect(candidatePlan[7]).toMatchObject({
      phaseIndex: 1,
      firstCommandBufferIndex: 25,
      lastCommandBufferIndex: 28,
      commandBufferCount: 4,
    });
    expect(candidatePlan[85]).toMatchObject({
      phaseIndex: 1,
      firstCommandBufferIndex: 337,
      lastCommandBufferIndex: 340,
      commandBufferCount: 4,
    });
    const control = topology(OPT_0080_CONTROL_PROFILE);
    const candidate = topology(OPT_0080_CANDIDATE_PROFILE);
    expect(requireOpt0080FullTopology(control)).toBe(control);
    expect(requireOpt0080FullTopology(candidate)).toBe(candidate);
    expect(control).toMatchObject({
      graphTrueQueueDrainCount: 2_553,
      totalTrueQueueDrainCount: 2_554,
      graphCooperativeIdleTurns: 2_552,
      totalCooperativeIdleTurns: 2_553,
      maximumOutstandingCommandBuffers: 1,
    });
    expect(candidate).toMatchObject({
      graphTrueQueueDrainCount: 639,
      totalTrueQueueDrainCount: 640,
      graphCooperativeIdleTurns: 638,
      totalCooperativeIdleTurns: 639,
      maximumOutstandingCommandBuffers: 2,
      maximumPendingDescriptorCount: 2,
    });
    expect(() => requireOpt0080FullTopology({
      ...candidate,
      graphTrueQueueDrainCount: 640 as 639,
    })).toThrow(/full command\/fence\/epoch topology/);
  });

  it("uses the authoritative full graph wall for both material gates", () => {
    const control = sample(
      "A",
      0,
      OPT_0080_CONTROL_PROFILE,
      1_000,
      5_000,
      41_000,
      41_200,
    );
    const candidate = sample(
      "B",
      1,
      OPT_0080_CANDIDATE_PROFILE,
      700,
      4_500,
      36_700,
      36_900,
    );
    expect(summarizeOpt0080FullPerformance([control, candidate])).toMatchObject({
      graphImproved: true,
      backendImproved: true,
      graphSpeedup: 41_000 / 36_700,
      graphSavingMs: 4_300,
      classification: "passed",
      passed: true,
    });
    expect(summarizeOpt0080FullPerformance([
      control,
      sample(
        "B",
        1,
        OPT_0080_CANDIDATE_PROFILE,
        900,
        4_700,
        38_500,
        38_700,
      ),
    ])).toMatchObject({
      graphSavingMs: 2_500,
      graphSpeedup: 41_000 / 38_500,
      classification: "passed",
      passed: true,
    });
    expect(summarizeOpt0080FullPerformance([
      control,
      sample(
        "B",
        1,
        OPT_0080_CANDIDATE_PROFILE,
        900,
        4_725,
        38_700,
        38_900,
      ),
    ])).toMatchObject({ classification: "inconclusive", passed: false });
    expect(workerSource).toContain(
      "Never sum submitThroughCompletionFenceMs",
    );
    expect(workerSource).not.toContain(
      "submitThroughCompletionFenceMs.reduce",
    );
  });

  it("keeps correctness taps out of timed arms and requires all raw words", () => {
    expect(workerSource).toContain("captureEvaluationTaps: true as const");
    expect(workerSource).toContain(
      "? { captureEvaluationTaps: true as const }",
    );
    expect(workerSource).toContain(
      "timedOrdinaryFinalReadbackOnly",
    );
    expect(workerSource).toContain("checkpoint.evaluationTaps === undefined");
    expect(workerSource).toContain("rawU32Equal(");
    expect(workerSource).toContain("exactOpt0080FullTensorIdentity(");
    expect(workerSource).toContain(
      '"eight-in-command-evaluation-tap-copies"',
    );
    expect(workerSource).not.toContain(
      "physicalCommandBufferContentsUnchanged",
    );
    const left = tensor();
    const right = left.slice();
    expect(exactOpt0080FullTensorIdentity(left, right)).toBe(true);
    const rightWords = new Uint32Array(right.buffer);
    rightWords[287_999] = rightWords[287_999]! ^ 1;
    expect(exactOpt0080FullTensorIdentity(left, right)).toBe(false);
  });

  it("runs an actual full candidate graph cancellation before timing", () => {
    const evidence = cancellationEvidence();
    expect(requireOpt0080FullCancellation(evidence)).toBe(evidence);
    expect(() => requireOpt0080FullCancellation({
      ...evidence,
      completionFenceRequestedCountAtAbort: 3 as 2,
    })).toThrow(/cancellation/);
    expect(workerSource).toContain('scope: "actual-dit-full-graph"');
    expect(workerSource).toContain("opt0080FullDitRun");
    expect(workerSource).toContain("onCommandBufferCompleted(completion)");
    expect(workerSource).toContain("controller.abort(stop)");
    expect(workerSource).toContain("await backend.dispose()");
    expect(workerSource).toContain(
      "await new Promise<void>((resolve) => setTimeout(resolve, 0))",
    );
  });

  it("accepts a nominal start while retaining a later thermal transition", () => {
    const controlTensor = tensor();
    const candidateTensor = controlTensor.slice();
    const conditioning = Object.freeze({
      textTokenSha256: "a".repeat(64),
      lyricTokenSha256: "b".repeat(64),
    });
    const correctness = Object.freeze({
      control: correctnessArm(
        OPT_0080_CONTROL_PROFILE,
        controlTensor,
        conditioning,
      ),
      candidate: correctnessArm(
        OPT_0080_CANDIDATE_PROFILE,
        candidateTensor,
        conditioning,
      ),
    });
    const gate = thermalGate();
    const arms = Object.freeze([
      timedArm(
        sample(
          "A",
          0,
          OPT_0080_CONTROL_PROFILE,
          1_000,
          5_000,
          41_000,
          41_200,
          timedHeartbeat(),
        ),
        controlTensor,
        conditioning,
        gate,
        thermalTrace("a", true),
        5,
      ),
      timedArm(
        sample(
          "B",
          1,
          OPT_0080_CANDIDATE_PROFILE,
          700,
          4_500,
          36_700,
          36_900,
          timedHeartbeat(),
        ),
        controlTensor.slice(),
        conditioning,
        gate,
        thermalTrace("b", false),
        7,
      ),
    ]);
    const receipt = buildOpt0080FullResult({
      runIdentity: {} as never,
      requestCanonicalJson: "{}",
      requestSha256: "c".repeat(64),
      requestByteLength: 2,
      mainManifestSha256: "d".repeat(64),
      denseManifestSha256: "e".repeat(64),
      denseRuntimeProfile: "opt-0009-fp16-fp32-dense-v1",
      attentionRuntimeProfile:
        "opt-0070-fixed32-quad-query32-full-self-production-v1",
      evaluationSliceResultSha256:
        OPT_0080_EVALUATION_SLICE_EVIDENCE_SHA256,
      correctness,
      cancellation: cancellationEvidence(),
      arms,
      rejectedSetupAttempts: Object.freeze([]),
    });
    expect(receipt).toMatchObject({
      schema:
        "ace-opt-0080-dit-depth2-completion-epochs-full-graph-gate-v1",
      stage: "full-m2250-confirmation",
      status: "passed",
      passed: true,
      thermalStartPassed: true,
      correctness: {
        evaluationRawU32MismatchCount: 0,
        finalLatentRawU32MismatchCount: 0,
        evaluationTapExtraCommandBufferCount: 0,
      },
      throughCleanupThermalTransitions: {
        A: { nonNominalObservationCount: 3 },
      },
      scope: {
        productionIntegrationPerformed: false,
        timedCommandBufferContentsChanged: false,
        untimedCorrectnessCommandBuffersAddedEvaluationTapCopies: true,
        untimedEvaluationTapInCommandCopyCountPerArm: 8,
      },
      decision: "full-graph-pass-authorize-production-dit-integration",
    });
  });

  it("binds fresh ownership, distinct traces, heartbeat, and manual order", () => {
    expect(pageSource).toContain(
      "OPT_0080_HEARTBEAT_INTERVAL_MILLISECONDS",
    );
    expect(pageSource).toContain("ready-for-arm");
    expect(workerSource).toContain("accepted.some((prior)");
    expect(workerSource).toContain("submittedEvidence");
    expect(workerSource).toContain("cleanupCompletedOrdinal");
    expect(workerSource).toContain("backendCreatedOrdinal");
    expect(htmlSource).toContain(
      "Accepted timing is exactly one separately cooled A/B pair",
    );
    expect(htmlSource).toContain("B is not prepared until A's distinct");
    expect(htmlSource).toContain("ordinary final readback");
    expect(pageSource).toContain("publishFailure(error)");
  });
});

function topology(
  schedulingProfile: Opt0080FullSchedulingProfile,
): Opt0080FullTopologyEvidence {
  const candidate = schedulingProfile === OPT_0080_CANDIDATE_PROFILE;
  const plan = planOpt0080CompletionEpochs(
    OPT_0080_FULL_PHASE_COMMAND_BUFFER_COUNTS,
    candidate ? 4 : 1,
  );
  const graphDrains = candidate ? 639 : 2_553;
  return Object.freeze({
    schedulingProfile,
    descriptorTableMemberCount: 6_833,
    graphCommandBufferCount: 2_553,
    readbackCommandBufferCount: 1,
    totalCommandBufferCount: 2_554,
    commandBuffersSubmitted: 2_554,
    completionFenceRequestedCount: 2_554,
    completionFenceSettledCount: 2_554,
    completionFenceRejectedCount: 0,
    graphTrueQueueDrainCount: graphDrains as 639 | 2_553,
    totalTrueQueueDrainCount: (graphDrains + 1) as 640 | 2_554,
    graphCompletionEpochCount: graphDrains as 639 | 2_553,
    graphCooperativeIdleTurns: (graphDrains - 1) as 638 | 2_552,
    totalCooperativeIdleTurns: graphDrains as 639 | 2_553,
    graphRequestedCooperativeIdleMs: (graphDrains - 1) as 638 | 2_552,
    totalRequestedCooperativeIdleMs: graphDrains as 639 | 2_553,
    maximumOutstandingCommandBuffers: candidate ? 2 : 1,
    maximumPendingDescriptorCount: candidate ? 2 : 1,
    pendingDescriptorCountAfterCleanup: 0,
    graphCompletionEpochs: Object.freeze(plan.map((epoch) => Object.freeze({
      ...epoch,
      submitThroughTrueDrainMs: 10,
    }))),
    submitThroughCompletionFenceMs: Object.freeze(Array(2_553).fill(10)),
    graphToReadbackRequestedIdleMs: 1,
    readbackSubmitThroughCompletionFenceMs: 2,
  });
}

function sample(
  armId: "A" | "B",
  order: 0 | 1,
  schedulingProfile: Opt0080FullSchedulingProfile,
  precomputeWallMs: number,
  evaluationWallEachMs: number,
  graphWallMs: number,
  backendWallMs: number,
  heartbeatCapture = heartbeat(),
): Opt0080FullTimingSample {
  const topologyEvidence = topology(schedulingProfile);
  return Object.freeze({
    armId,
    order,
    schedulingProfile,
    precomputeWallMs,
    evaluationWallMs: Object.freeze(Array(8).fill(evaluationWallEachMs)) as
      unknown as Opt0080FullTimingSample["evaluationWallMs"],
    graphWallMs,
    graphToReadbackObservedIdleMs: 1,
    readbackFenceMs: 2,
    readbackMapMs: 1,
    backendWallMs,
    graphEpochWallSumMs:
      topologyEvidence.graphCompletionEpochs.length * 10,
    topology: topologyEvidence,
    heartbeat: heartbeatCapture,
  });
}

function tensor(): Float32Array<ArrayBuffer> {
  const value = new Float32Array(288_000);
  value[0] = 1;
  value[287_999] = -0;
  return value;
}

function tap(
  evaluation: number,
  value: Float32Array<ArrayBuffer>,
): Opt0080FullCorrectnessTap {
  return Object.freeze({
    evaluation,
    result: value,
    rawU32: new Uint32Array(
      value.buffer,
      value.byteOffset,
      value.length,
    ),
    sha256: OPT_0080_FULL_EVALUATION_SHA256[evaluation]!,
    nonFiniteCount: 0,
    nonzeroCount: 1,
    maxAbs: 1,
  });
}

function correctnessArm(
  schedulingProfile: Opt0080FullSchedulingProfile,
  value: Float32Array<ArrayBuffer>,
  conditioningAuthority: Readonly<Record<string, unknown>>,
): Opt0080FullCorrectnessArm {
  return Object.freeze({
    schedulingProfile,
    captureEvaluationTaps: true,
    evaluationTaps: Object.freeze(Array.from({ length: 8 }, (_, evaluation) =>
      tap(evaluation, value)
    )),
    finalLatent: value,
    finalLatentRawU32: new Uint32Array(
      value.buffer,
      value.byteOffset,
      value.length,
    ),
    finalLatentSha256: OPT_0080_FULL_FINAL_LATENT_SHA256,
    finalLatentNonFiniteCount: 0,
    finalLatentNonzeroCount: 1,
    finalLatentMaxAbs: 1,
    descriptorTableSha256: "f".repeat(64),
    descriptorTableMemberCount: 6_833,
    topology: topology(schedulingProfile),
    evaluationTapInCommandCopyCount: 8,
    evaluationTapExtraCommandBufferCount: 0,
    evaluationTapExtraQueueDrainCount: 0,
    uncapturedWebGpuErrorCount: 0,
    deviceLost: false,
    conditioningAuthority,
    receipt: Object.freeze({ schedulingProfile }),
  });
}

function timedArm(
  timing: Opt0080FullTimingSample,
  value: Float32Array<ArrayBuffer>,
  conditioningAuthority: Readonly<Record<string, unknown>>,
  gate: Opt0080FullThermalGate,
  trace: Opt0080FullThermalTrace,
  backendCreatedOrdinal: number,
): Opt0080FullTimedArmResult {
  return Object.freeze({
    armId: timing.armId,
    schedulingProfile: timing.schedulingProfile,
    captureEvaluationTaps: false,
    evaluationTapCount: 0,
    finalLatent: value,
    finalLatentRawU32: new Uint32Array(
      value.buffer,
      value.byteOffset,
      value.length,
    ),
    finalLatentSha256: OPT_0080_FULL_FINAL_LATENT_SHA256,
    descriptorTableSha256: "f".repeat(64),
    conditioningAuthority,
    sample: timing,
    gate,
    trace,
    receipt: Object.freeze({ armId: timing.armId }),
    backendCreatedOrdinal,
    cleanupCompletedOrdinal: backendCreatedOrdinal + 1,
    cleanupCompletedAtEpochMilliseconds: 79_000,
    timingReadyAtEpochMilliseconds: 1_000,
  });
}

function heartbeat(gaps = [50, 50, 50, 50]): Opt0080FullHeartbeatCapture {
  const sorted = [...gaps].sort((left, right) => left - right);
  return Object.freeze({
    intervalMilliseconds: 50,
    startedAtEpochMilliseconds: 34_000,
    completedAtEpochMilliseconds: 80_000,
    gapsMilliseconds: Object.freeze([...gaps]),
    maximumGapMilliseconds: sorted.at(-1)!,
    p99GapMilliseconds:
      sorted[Math.max(0, Math.ceil(sorted.length * 0.99) - 1)]!,
  });
}

function timedHeartbeat(): Opt0080FullHeartbeatCapture {
  return heartbeat(Array(920).fill(50));
}

function thermalGate(): Opt0080FullThermalGate {
  return Object.freeze({
    source: "notifyutil-com.apple.system.thermalpressurelevel",
    command: "notifyutil -g com.apple.system.thermalpressurelevel",
    startedAtEpochMilliseconds: 2_000,
    completedAtEpochMilliseconds: 33_000,
    observationCount: 32,
    maximumPollGapMilliseconds: 1_000,
    nonNominalObservationCount: 0,
    observations: Object.freeze(Array.from({ length: 32 }, (_, index) =>
      Object.freeze({
        atEpochMilliseconds: 2_000 + index * 1_000,
        level: 0,
        rawValue: "0",
      })
    )),
  });
}

function thermalTrace(
  identity: string,
  laterTransition: boolean,
): Opt0080FullThermalTrace {
  return Object.freeze({
    source: "notifyutil-com.apple.system.thermalpressurelevel",
    command: "notifyutil -g com.apple.system.thermalpressurelevel",
    rawTraceSha256: identity.repeat(64),
    completedAtEpochMilliseconds: 80_000,
    observationCount: 79,
    maximumPollGapMilliseconds: 1_000,
    nonNominalObservationCount: laterTransition ? 3 : 0,
    observations: Object.freeze(Array.from({ length: 79 }, (_, index) => {
      const level = laterTransition && index >= 68 && index <= 70 ? 1 : 0;
      return Object.freeze({
        atEpochMilliseconds: 2_000 + index * 1_000,
        level,
        rawValue: String(level),
      });
    })),
    transitions: laterTransition
      ? Object.freeze([
          Object.freeze({ atEpochMilliseconds: 70_000, level: 1 }),
          Object.freeze({ atEpochMilliseconds: 73_000, level: 0 }),
        ])
      : Object.freeze([]),
  });
}

function cancellationEvidence(): Opt0080FullCancellationEvidence {
  return Object.freeze({
    scope: "actual-dit-full-graph",
    schedulingProfile: OPT_0080_CANDIDATE_PROFILE,
    abortedCommandBufferIndex: 0,
    abortObservedFromCompletionCallback: true,
    outstandingSuccessorsAtAbort: 1,
    prefetchedTailCommandCount: 1,
    backfillAfterAbortCount: 0,
    publicProgressAfterAbortCount: 0,
    specializedCompletionCallbackCount: 1,
    specializedCompletionCallbackCountAfterAbort: 0,
    completedCommandBufferCountAtAbort: 1,
    completionFenceRequestedCountAtAbort: 2,
    checkpointCount: 0,
    unhandledRejectionCount: 0,
    unhandledRejectionListenerRemoved: true,
    generationRejectedWithAbortReason: true,
    terminalSettlementBeforeGenerationRejection: true,
    graphAndModelDestroyAwaitedBeforeGenerationRejection: true,
    backendDisposeAwaitedAfterGenerationRejection: true,
    abortObservationThroughRejectionAndCleanupMs: 40,
  });
}
