import { describe, expect, it } from "vitest";
import experimentSource from
  "../optimization/experiments/OPT-0080-dit-depth2-completion-epochs.md?raw";
import schedulerSource from "../src/runtime/scheduler.ts?raw";
import graphSource from "../src/webgpu/dit-graph.ts?raw";
import graphTestSource from "./dit-graph.test.ts?raw";
import workerSource from
  "./browser/opt-0080-dit-depth2-completion-epochs-worker.ts?raw";
import pageSource from
  "./browser/opt-0080-dit-depth2-completion-epochs.ts?raw";
import htmlSource from
  "./browser/opt-0080-dit-depth2-completion-epochs.html?raw";
import {
  OPT_0080_ARM_ORDER,
  OPT_0080_CANDIDATE_PROFILE,
  OPT_0080_CONTROL_PROFILE,
  OPT_0080_EVALUATION0_SHA256,
  exactOpt0080ResultIdentity,
  planOpt0080CompletionEpochs,
  requireOpt0080Cancellation,
  requireOpt0080Heartbeat,
  requireOpt0080Topology,
  summarizeOpt0080Performance,
  type Opt0080CancellationEvidence,
  type Opt0080HeartbeatCapture,
  type Opt0080SchedulingProfile,
  type Opt0080TimingSample,
  type Opt0080ThermalGate,
  type Opt0080ThermalTrace,
  type Opt0080TopologyEvidence,
} from "./browser/opt-0080-dit-depth2-completion-epochs-contract.js";
import {
  buildOpt0080Result,
  type Opt0080CorrectnessArm,
  type Opt0080TimedArmResult,
} from "./browser/opt-0080-dit-depth2-completion-epochs-result.js";

describe("OPT-0080 DiT depth-two completion epochs", () => {
  it("freezes ABBA ownership and the inherited exact evaluation identity", () => {
    expect(OPT_0080_ARM_ORDER).toEqual([
      { armId: "A1", order: 0, schedulingProfile: "depth1-epoch1" },
      {
        armId: "B1",
        order: 1,
        schedulingProfile: "opt-0080-depth2-epoch4",
      },
      {
        armId: "B2",
        order: 2,
        schedulingProfile: "opt-0080-depth2-epoch4",
      },
      { armId: "A2", order: 3, schedulingProfile: "depth1-epoch1" },
    ]);
    expect(OPT_0080_EVALUATION0_SHA256).toBe(
      "d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286",
    );
    expect(experimentSource).toContain("`6,833` graph members");
    expect(experimentSource).toContain("`288,000`-word evaluation result");
    expect(experimentSource).toContain("maximum outstanding two");
  });

  it("plans phase-aligned epochs without crossing tails", () => {
    for (const count of [1, 2, 3, 4, 5, 10]) {
      const plan = planOpt0080CompletionEpochs([count], 4);
      expect(plan).toHaveLength(Math.ceil(count / 4));
      expect(plan[0]?.firstCommandBufferIndex).toBe(0);
      expect(plan.at(-1)?.lastCommandBufferIndex).toBe(count - 1);
    }
    const evaluation0 = planOpt0080CompletionEpochs([25, 316], 4);
    expect(evaluation0).toHaveLength(7 + 79);
    expect(evaluation0[6]).toMatchObject({
      phaseIndex: 0,
      firstCommandBufferIndex: 24,
      lastCommandBufferIndex: 24,
      commandBufferCount: 1,
    });
    expect(evaluation0[7]).toMatchObject({
      phaseIndex: 1,
      firstCommandBufferIndex: 25,
      lastCommandBufferIndex: 28,
      commandBufferCount: 4,
    });
    const full = planOpt0080CompletionEpochs(
      [25, 316, 316, 316, 316, 316, 316, 316, 316],
      4,
    );
    expect(full).toHaveLength(7 + 8 * 79);
    expect(() => planOpt0080CompletionEpochs([0], 4)).toThrow(/phase/);
  });

  it("reconciles every command, fence, true drain, idle, and epoch", () => {
    const control = topology(OPT_0080_CONTROL_PROFILE);
    const candidate = topology(OPT_0080_CANDIDATE_PROFILE);
    expect(requireOpt0080Topology(control)).toBe(control);
    expect(requireOpt0080Topology(candidate)).toBe(candidate);
    expect(candidate).toMatchObject({
      descriptorTableMemberCount: 6_833,
      commandBuffersSubmitted: 342,
      completionFenceRequestedCount: 342,
      completionFenceSettledCount: 342,
      graphTrueQueueDrainCount: 86,
      totalTrueQueueDrainCount: 87,
      graphCompletionEpochCount: 86,
      graphCooperativeIdleTurns: 85,
      totalCooperativeIdleTurns: 86,
      maximumOutstandingCommandBuffers: 2,
      maximumPendingDescriptorCount: 2,
    });
    expect(() => requireOpt0080Topology({
      ...candidate,
      completionFenceRequestedCount: 341 as 342,
    })).toThrow(/topology/);
    expect(() => requireOpt0080Topology({
      ...candidate,
      graphCompletionEpochs: candidate.graphCompletionEpochs.slice(1),
    })).toThrow(/topology/);
  });

  it("uses only authoritative walls and disjoint epochs for its material gate", () => {
    const passing = [
      sample("A1", 0, OPT_0080_CONTROL_PROFILE, 1_000, 5_000, 6_000, 6_100),
      sample("B1", 1, OPT_0080_CANDIDATE_PROFILE, 700, 4_500, 5_200, 5_300),
      sample("B2", 2, OPT_0080_CANDIDATE_PROFILE, 710, 4_550, 5_260, 5_360),
      sample("A2", 3, OPT_0080_CONTROL_PROFILE, 1_010, 5_100, 6_110, 6_210),
    ] as const;
    const summary = summarizeOpt0080Performance(passing);
    expect(summary).toMatchObject({
      forwardEvaluationImproved: true,
      reverseEvaluationImproved: true,
      forwardGraphImproved: true,
      reverseGraphImproved: true,
      forwardBackendImproved: true,
      reverseBackendImproved: true,
      forwardEvaluationSavingMs: 500,
      reverseEvaluationSavingMs: 550,
      classification: "passed",
      passed: true,
    });
    expect(summary.aggregateEvaluationSpeedup).toBeGreaterThan(1.04);
    expect(summary.projectedFullGraphSavingMs).toBeGreaterThan(2_500);
    expect(workerSource).toContain(
      "Epoch timings are disjoint. Never sum submitThroughCompletionFenceMs.",
    );
    expect(workerSource).not.toContain(
      "submitThroughCompletionFenceMs.reduce",
    );
    const cleanThresholdMiss = summarizeOpt0080Performance([
      passing[0],
      {
        ...passing[1],
        evaluationWallMs: 4_800,
        graphWallMs: 5_500,
        backendWallMs: 5_600,
      },
      passing[2],
      passing[3],
    ]);
    expect(cleanThresholdMiss).toMatchObject({
      cleanReproducibleThresholdMiss: true,
      classification: "failed",
      passed: false,
    });
  });

  it("classifies mixed or wall-inconsistent pairs as inconclusive", () => {
    const a1 = sample(
      "A1", 0, OPT_0080_CONTROL_PROFILE, 1_000, 5_000, 6_000, 6_100,
    );
    const a2 = sample(
      "A2", 3, OPT_0080_CONTROL_PROFILE, 1_010, 5_100, 6_110, 6_210,
    );
    const mixed = summarizeOpt0080Performance([
      a1,
      sample(
        "B1", 1, OPT_0080_CANDIDATE_PROFILE, 700, 5_100, 5_800, 5_900,
      ),
      sample(
        "B2", 2, OPT_0080_CANDIDATE_PROFILE, 710, 4_550, 5_260, 5_360,
      ),
      a2,
    ]);
    expect(mixed).toMatchObject({
      mixedPairedDirections: true,
      classification: "inconclusive",
      passed: false,
    });
    const wallInconsistent = summarizeOpt0080Performance([
      a1,
      sample(
        "B1", 1, OPT_0080_CANDIDATE_PROFILE, 2_100, 4_500, 6_600, 6_700,
      ),
      sample(
        "B2", 2, OPT_0080_CANDIDATE_PROFILE, 2_110, 4_550, 6_660, 6_760,
      ),
      a2,
    ]);
    expect(wallInconsistent).toMatchObject({
      mixedPairedDirections: false,
      crossMetricDirectionConsistent: false,
      wallBoundaryConsistent: true,
      classification: "inconclusive",
      passed: false,
    });
    const reproducibleRegression = summarizeOpt0080Performance([
      a1,
      sample(
        "B1", 1, OPT_0080_CANDIDATE_PROFILE, 1_100, 5_200, 6_300, 6_400,
      ),
      sample(
        "B2", 2, OPT_0080_CANDIDATE_PROFILE, 1_110, 5_250, 6_360, 6_460,
      ),
      a2,
    ]);
    expect(reproducibleRegression).toMatchObject({
      cleanReproducibleThresholdMiss: true,
      classification: "failed",
      passed: false,
    });
    const impossibleWalls = summarizeOpt0080Performance([
      { ...a1, graphEpochWallSumMs: a1.graphWallMs + 1 },
      sample(
        "B1", 1, OPT_0080_CANDIDATE_PROFILE, 700, 4_500, 5_200, 5_300,
      ),
      sample(
        "B2", 2, OPT_0080_CANDIDATE_PROFILE, 710, 4_550, 5_260, 5_360,
      ),
      a2,
    ]);
    expect(impossibleWalls).toMatchObject({
      wallBoundaryConsistent: false,
      classification: "inconclusive",
      passed: false,
    });
  });

  it("enforces the 50 ms heartbeat absolute and paired relative gates", () => {
    const valid = heartbeat([50, 51, 49, 50]);
    expect(requireOpt0080Heartbeat(valid)).toBe(valid);
    expect(() => requireOpt0080Heartbeat({
      ...valid,
      maximumGapMilliseconds: 500,
    })).toThrow(/heartbeat/);
    const samples = [
      sample("A1", 0, OPT_0080_CONTROL_PROFILE, 1_000, 5_000, 6_000, 6_100),
      sample(
        "B1",
        1,
        OPT_0080_CANDIDATE_PROFILE,
        700,
        4_500,
        5_200,
        5_300,
        heartbeat([150, 150, 150, 150]),
      ),
      sample("B2", 2, OPT_0080_CANDIDATE_PROFILE, 710, 4_550, 5_260, 5_360),
      sample("A2", 3, OPT_0080_CONTROL_PROFILE, 1_010, 5_100, 6_110, 6_210),
    ] as const;
    const summary = summarizeOpt0080Performance(samples);
    expect(summary.heartbeatAbsolutePassed).toBe(true);
    expect(summary.heartbeatRelativePassed).toBe(false);
    expect(summary.passed).toBe(false);
    expect(pageSource).toContain(
      "OPT_0080_HEARTBEAT_INTERVAL_MILLISECONDS",
    );
  });

  it("requires actual-DiT one-tail abort and statically binds graph cleanup", () => {
    const evidence = cancellationEvidence();
    expect(requireOpt0080Cancellation(evidence)).toBe(evidence);
    expect(() => requireOpt0080Cancellation({
      ...evidence,
      backfillAfterAbortCount: 1 as 0,
    })).toThrow(/cancellation/);
    expect(workerSource).toContain("scope: \"actual-dit-evaluation0-graph\"");
    expect(workerSource).toContain("onCommandBufferCompleted(completion)");
    expect(workerSource).toContain("controller.abort(stop)");
    expect(workerSource).toContain("await backend.generate(request, context)");
    expect(workerSource).toContain("await backend.dispose()");
    expect(workerSource).not.toContain("AceCooperativeGpuScheduler");
    expect(graphTestSource).toContain(
      "retains the prefetched OPT-0080 descriptor until abort settlement",
    );
    expect(graphTestSource).toContain("expect(fixture.commandBuffers.live).toBe(0)");
    expect(graphTestSource).toContain("expect(runtimeState.destroys).toBe(1)");
    expect(graphTestSource).toContain("expect(modelState.destroys).toBe(1)");
    expect(graphSource).toContain("pending.clear()");
  });

  it("publishes finalization failures instead of swallowing them", () => {
    const finalize = workerSource.indexOf("const result = finalizeGate()");
    const settled = workerSource.indexOf('state = "settled";', finalize);
    const publish = workerSource.indexOf(
      'self.postMessage({ type: "gate-complete", result })',
      settled,
    );
    expect(finalize).toBeGreaterThan(0);
    expect(settled).toBeGreaterThan(finalize);
    expect(publish).toBeGreaterThan(settled);
    expect(workerSource).toContain("} catch (error) {\n        fail(error);");
  });

  it("retains a later thermal transition without failing a valid nominal start", () => {
    const result = new Float32Array(288_000);
    result[0] = 1;
    const conditioning = Object.freeze({
      textTokenSha256: "a".repeat(64),
      lyricTokenSha256: "b".repeat(64),
    });
    const correctnessArm = (
      schedulingProfile: Opt0080SchedulingProfile,
    ): Opt0080CorrectnessArm => Object.freeze({
      schedulingProfile,
      result,
      resultSha256: OPT_0080_EVALUATION0_SHA256,
      descriptorTableSha256: "c".repeat(64),
      descriptorTableMemberCount: 6_833,
      graphCommandBufferCount: 341,
      resultNonFiniteCount: 0,
      resultNonzeroCount: 1,
      uncapturedWebGpuErrorCount: 0,
      deviceLost: false,
      conditioningAuthority: conditioning,
      receipt: Object.freeze({ schedulingProfile }),
    });
    const gate = thermalGate();
    const arms = [
      timedArm(
        sample(
          "A1",
          0,
          OPT_0080_CONTROL_PROFILE,
          1_000,
          5_000,
          6_000,
          6_100,
          timedHeartbeat(),
        ),
        result,
        conditioning,
        gate,
        thermalTrace("1", true),
        1,
      ),
      timedArm(
        sample(
          "B1",
          1,
          OPT_0080_CANDIDATE_PROFILE,
          700,
          4_500,
          5_200,
          5_300,
          timedHeartbeat(),
        ),
        result,
        conditioning,
        gate,
        thermalTrace("2", false),
        3,
      ),
      timedArm(
        sample(
          "B2",
          2,
          OPT_0080_CANDIDATE_PROFILE,
          710,
          4_550,
          5_260,
          5_360,
          timedHeartbeat(),
        ),
        result,
        conditioning,
        gate,
        thermalTrace("3", false),
        5,
      ),
      timedArm(
        sample(
          "A2",
          3,
          OPT_0080_CONTROL_PROFILE,
          1_010,
          5_100,
          6_110,
          6_210,
          timedHeartbeat(),
        ),
        result,
        conditioning,
        gate,
        thermalTrace("4", false),
        7,
      ),
    ] as const;
    const resultInput = {
      runIdentity: {} as never,
      requestCanonicalJson: "{}",
      requestSha256: "d".repeat(64),
      requestByteLength: 2,
      mainManifestSha256: "e".repeat(64),
      denseManifestSha256: "f".repeat(64),
      denseRuntimeProfile: "opt-0009-fp16-fp32-dense-v1",
      attentionRuntimeProfile:
        "opt-0070-fixed32-quad-query32-full-self-production-v1",
      correctness: Object.freeze({
        controlFirst: correctnessArm(OPT_0080_CONTROL_PROFILE),
        controlRepeat: correctnessArm(OPT_0080_CONTROL_PROFILE),
        candidateFirst: correctnessArm(OPT_0080_CANDIDATE_PROFILE),
        candidateRepeat: correctnessArm(OPT_0080_CANDIDATE_PROFILE),
      }),
      cancellation: cancellationEvidence(),
      arms,
      rejectedSetupAttempts: Object.freeze([]),
    } as const;
    const receipt = buildOpt0080Result(resultInput);
    expect(receipt).toMatchObject({
      status: "passed",
      passed: true,
      thermalStartPassed: true,
      throughCleanupThermalTransitions: {
        A1: { nonNominalObservationCount: 3 },
      },
    });
    const mixedReceipt = buildOpt0080Result({
      ...resultInput,
      arms: Object.freeze([
        arms[0],
        Object.freeze({
          ...arms[1],
          sample: Object.freeze({
            ...arms[1].sample,
            evaluationWallMs: 5_100,
            graphWallMs: 5_800,
            backendWallMs: 5_900,
          }),
        }),
        arms[2],
        arms[3],
      ]),
    });
    expect(mixedReceipt).toMatchObject({
      status: "inconclusive",
      passed: false,
      performance: { classification: "inconclusive" },
    });
    const cleanMissReceipt = buildOpt0080Result({
      ...resultInput,
      arms: Object.freeze([
        arms[0],
        Object.freeze({
          ...arms[1],
          sample: Object.freeze({
            ...arms[1].sample,
            evaluationWallMs: 4_800,
            graphWallMs: 5_500,
            backendWallMs: 5_600,
          }),
        }),
        Object.freeze({
          ...arms[2],
          sample: Object.freeze({
            ...arms[2].sample,
            evaluationWallMs: 4_900,
            graphWallMs: 5_610,
            backendWallMs: 5_710,
          }),
        }),
        arms[3],
      ]),
    });
    expect(cleanMissReceipt).toMatchObject({
      status: "failed",
      passed: false,
      performance: { classification: "failed" },
    });
  });

  it("compares all 288,000 evaluation words as raw U32", () => {
    const left = new Float32Array(288_000);
    left[0] = 1;
    left[287_999] = -0;
    const right = left.slice();
    expect(exactOpt0080ResultIdentity(left, right)).toBe(true);
    new Uint32Array(right.buffer)[287_999] = 0;
    expect(exactOpt0080ResultIdentity(left, right)).toBe(false);
    expect(exactOpt0080ResultIdentity(left, new Float32Array(1))).toBe(false);
  });

  it("keeps the candidate diagnostic-only and preserves singleton submissions", () => {
    expect(schedulerSource).toContain(
      "submitAceCommandBufferFactoriesDepth2Epoch4",
    );
    expect(schedulerSource).toContain("pending.length < 2");
    expect(schedulerSource).toContain("options.queue.submit([commandBuffer])");
    expect(graphSource).toContain('"opt-0080-depth2-epoch4"');
    expect(workerSource).toContain("opt0080DitRun");
    expect(workerSource).toContain("control-first");
    expect(workerSource).toContain("candidate-repeat");
    expect(workerSource).not.toContain("captureActualLayerIdentity");
    expect(workerSource).not.toContain("resultGuardMismatchCount");
    expect(workerSource).toContain("descriptorTableMemberCount");
    expect(workerSource).toContain(
      "ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE",
    );
    expect(workerSource).toContain(
      "/model/files-fp16-vae-revision7-experimental/manifest.json",
    );
    expect(workerSource).toContain(
      "ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE",
    );
    expect(workerSource).not.toContain(
      "opt-0062-fixed32-quad-query32-full-self-v1",
    );
    expect(workerSource).not.toContain("files-fp16-vae-experimental");
    expect(htmlSource).toContain("A1 depth1, B1 depth2/epoch4");
    expect(htmlSource).toContain("The next arm is not");
  });
});

function topology(
  schedulingProfile: Opt0080SchedulingProfile,
): Opt0080TopologyEvidence {
  const candidate = schedulingProfile === OPT_0080_CANDIDATE_PROFILE;
  const plan = planOpt0080CompletionEpochs([25, 316], candidate ? 4 : 1);
  const graphDrains = candidate ? 86 : 341;
  return Object.freeze({
    schedulingProfile,
    descriptorTableMemberCount: 6_833,
    graphCommandBufferCount: 341,
    readbackCommandBufferCount: 1,
    totalCommandBufferCount: 342,
    commandBuffersSubmitted: 342,
    completionFenceRequestedCount: 342,
    completionFenceSettledCount: 342,
    completionFenceRejectedCount: 0,
    graphTrueQueueDrainCount: graphDrains,
    totalTrueQueueDrainCount: graphDrains + 1,
    graphCompletionEpochCount: graphDrains,
    graphCooperativeIdleTurns: graphDrains - 1,
    totalCooperativeIdleTurns: graphDrains,
    graphRequestedCooperativeIdleMs: graphDrains - 1,
    totalRequestedCooperativeIdleMs: graphDrains,
    maximumOutstandingCommandBuffers: candidate ? 2 : 1,
    maximumPendingDescriptorCount: candidate ? 2 : 1,
    pendingDescriptorCountAfterCleanup: 0,
    graphCompletionEpochs: Object.freeze(plan.map((epoch) => Object.freeze({
      ...epoch,
      submitThroughTrueDrainMs: 10,
    }))),
    submitThroughCompletionFenceMs: Object.freeze(Array(341).fill(10)),
    graphToReadbackRequestedIdleMs: 1,
    readbackSubmitThroughCompletionFenceMs: 2,
  });
}

function heartbeat(gaps = [50, 50, 50, 50]): Opt0080HeartbeatCapture {
  const sorted = [...gaps].sort((left, right) => left - right);
  return Object.freeze({
    intervalMilliseconds: 50,
    startedAtEpochMilliseconds: 1_000,
    completedAtEpochMilliseconds: 2_000,
    gapsMilliseconds: Object.freeze([...gaps]),
    maximumGapMilliseconds: sorted.at(-1)!,
    p99GapMilliseconds:
      sorted[Math.max(0, Math.ceil(sorted.length * 0.99) - 1)]!,
  });
}

function timedHeartbeat(): Opt0080HeartbeatCapture {
  const gaps = Object.freeze(Array(10).fill(50) as number[]);
  return Object.freeze({
    intervalMilliseconds: 50,
    startedAtEpochMilliseconds: 34_000,
    completedAtEpochMilliseconds: 39_500,
    gapsMilliseconds: gaps,
    maximumGapMilliseconds: 50,
    p99GapMilliseconds: 50,
  });
}

function thermalGate(): Opt0080ThermalGate {
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
): Opt0080ThermalTrace {
  return Object.freeze({
    source: "notifyutil-com.apple.system.thermalpressurelevel",
    command: "notifyutil -g com.apple.system.thermalpressurelevel",
    rawTraceSha256: identity.repeat(64),
    completedAtEpochMilliseconds: 40_000,
    observationCount: 39,
    maximumPollGapMilliseconds: 1_000,
    nonNominalObservationCount: laterTransition ? 3 : 0,
    observations: Object.freeze(Array.from({ length: 39 }, (_, index) => {
      const level = laterTransition && index >= 33 && index <= 35 ? 1 : 0;
      return Object.freeze({
        atEpochMilliseconds: 2_000 + index * 1_000,
        level,
        rawValue: String(level),
      });
    })),
    transitions: laterTransition
      ? Object.freeze([
          Object.freeze({ atEpochMilliseconds: 35_000, level: 1 }),
          Object.freeze({ atEpochMilliseconds: 38_000, level: 0 }),
        ])
      : Object.freeze([]),
  });
}

function cancellationEvidence(): Opt0080CancellationEvidence {
  return Object.freeze({
    scope: "actual-dit-evaluation0-graph",
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

function timedArm(
  timing: Opt0080TimingSample,
  result: Float32Array<ArrayBuffer>,
  conditioningAuthority: Readonly<Record<string, unknown>>,
  gate: Opt0080ThermalGate,
  trace: Opt0080ThermalTrace,
  backendCreatedOrdinal: number,
): Opt0080TimedArmResult {
  return Object.freeze({
    armId: timing.armId,
    schedulingProfile: timing.schedulingProfile,
    result,
    resultSha256: OPT_0080_EVALUATION0_SHA256,
    descriptorTableSha256: "c".repeat(64),
    conditioningAuthority,
    sample: timing,
    gate,
    trace,
    receipt: Object.freeze({ armId: timing.armId }),
    backendCreatedOrdinal,
    cleanupCompletedOrdinal: backendCreatedOrdinal + 1,
    cleanupCompletedAtEpochMilliseconds: 39_500,
    timingReadyAtEpochMilliseconds: 1_000,
  });
}

function sample(
  armId: "A1" | "B1" | "B2" | "A2",
  order: 0 | 1 | 2 | 3,
  schedulingProfile: Opt0080SchedulingProfile,
  precomputeWallMs: number,
  evaluationWallMs: number,
  graphWallMs: number,
  backendWallMs: number,
  heartbeatCapture = heartbeat(),
): Opt0080TimingSample {
  const topologyEvidence = topology(schedulingProfile);
  return Object.freeze({
    armId,
    order,
    schedulingProfile,
    precomputeWallMs,
    evaluationWallMs,
    graphWallMs,
    graphToReadbackObservedIdleMs: 1,
    readbackFenceMs: 2,
    readbackMapMs: 1,
    backendWallMs,
    graphEpochWallSumMs: topologyEvidence.graphCompletionEpochs.length * 10,
    topology: topologyEvidence,
    heartbeat: heartbeatCapture,
  });
}
