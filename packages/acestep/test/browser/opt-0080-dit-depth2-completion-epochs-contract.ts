import {
  OPT_0067_EVALUATION0_SHA256,
  OPT_0067_GRAPH_COMMAND_BUFFERS,
  OPT_0067_TOTAL_COMMAND_BUFFERS,
  exactOpt0067ResultIdentity,
  requireOpt0067ThermalGate,
  requireOpt0067ThermalTrace,
  type Opt0067ThermalGate,
  type Opt0067ThermalTrace,
} from "./opt-0067-dit-quad-query-evaluation-slice-contract.js";

export const OPT_0080_CONTROL_PROFILE = "depth1-epoch1" as const;
export const OPT_0080_CANDIDATE_PROFILE =
  "opt-0080-depth2-epoch4" as const;
export const OPT_0080_EVALUATION0_SHA256 = OPT_0067_EVALUATION0_SHA256;
export const OPT_0080_RESULT_ELEMENTS = 288_000 as const;
export const OPT_0080_DESCRIPTOR_TABLE_MEMBER_COUNT = 6_833 as const;
export const OPT_0080_GRAPH_COMMAND_BUFFERS =
  OPT_0067_GRAPH_COMMAND_BUFFERS;
export const OPT_0080_TOTAL_COMMAND_BUFFERS =
  OPT_0067_TOTAL_COMMAND_BUFFERS;
export const OPT_0080_PRECOMPUTE_COMMAND_BUFFERS = 25 as const;
export const OPT_0080_EVALUATION_COMMAND_BUFFERS = 316 as const;
export const OPT_0080_HEARTBEAT_INTERVAL_MILLISECONDS = 50 as const;
export const OPT_0080_MAXIMUM_HEARTBEAT_GAP_MILLISECONDS = 500 as const;

export type Opt0080ThermalGate = Opt0067ThermalGate;
export type Opt0080ThermalTrace = Opt0067ThermalTrace;
export type Opt0080ArmId = "A1" | "B1" | "B2" | "A2";
export type Opt0080SchedulingProfile =
  | typeof OPT_0080_CONTROL_PROFILE
  | typeof OPT_0080_CANDIDATE_PROFILE;

export const OPT_0080_ARM_ORDER = Object.freeze([
  Object.freeze({
    armId: "A1" as const,
    order: 0 as const,
    schedulingProfile: OPT_0080_CONTROL_PROFILE,
  }),
  Object.freeze({
    armId: "B1" as const,
    order: 1 as const,
    schedulingProfile: OPT_0080_CANDIDATE_PROFILE,
  }),
  Object.freeze({
    armId: "B2" as const,
    order: 2 as const,
    schedulingProfile: OPT_0080_CANDIDATE_PROFILE,
  }),
  Object.freeze({
    armId: "A2" as const,
    order: 3 as const,
    schedulingProfile: OPT_0080_CONTROL_PROFILE,
  }),
]);

export interface Opt0080CompletionEpochPlanEntry {
  readonly completionEpochIndex: number;
  readonly phaseIndex: number;
  readonly firstCommandBufferIndex: number;
  readonly lastCommandBufferIndex: number;
  readonly commandBufferCount: number;
}

export interface Opt0080CompletionEpochTiming
  extends Opt0080CompletionEpochPlanEntry {
  /** Disjoint timing; these are the only per-epoch values that may be summed. */
  readonly submitThroughTrueDrainMs: number;
}

export interface Opt0080TopologyEvidence {
  readonly schedulingProfile: Opt0080SchedulingProfile;
  /** Member count for the complete compiled 2,553-command descriptor table. */
  readonly descriptorTableMemberCount: 6_833;
  readonly graphCommandBufferCount: 341;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: 342;
  readonly commandBuffersSubmitted: 342;
  readonly completionFenceRequestedCount: 342;
  readonly completionFenceSettledCount: 342;
  readonly completionFenceRejectedCount: 0;
  readonly graphTrueQueueDrainCount: number;
  readonly totalTrueQueueDrainCount: number;
  readonly graphCompletionEpochCount: number;
  readonly graphCooperativeIdleTurns: number;
  readonly totalCooperativeIdleTurns: number;
  readonly graphRequestedCooperativeIdleMs: number;
  readonly totalRequestedCooperativeIdleMs: number;
  readonly maximumOutstandingCommandBuffers: 1 | 2;
  readonly maximumPendingDescriptorCount: 1 | 2;
  readonly pendingDescriptorCountAfterCleanup: 0;
  readonly graphCompletionEpochs:
    readonly Opt0080CompletionEpochTiming[];
  /** Cumulative and overlapping in the candidate. It is deliberately non-additive. */
  readonly submitThroughCompletionFenceMs: readonly number[];
  readonly graphToReadbackRequestedIdleMs: 1;
  readonly readbackSubmitThroughCompletionFenceMs: number;
}

export interface Opt0080HeartbeatCapture {
  readonly intervalMilliseconds: 50;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly gapsMilliseconds: readonly number[];
  readonly maximumGapMilliseconds: number;
  readonly p99GapMilliseconds: number;
}

export interface Opt0080TimingSample {
  readonly armId: Opt0080ArmId;
  readonly order: 0 | 1 | 2 | 3;
  readonly schedulingProfile: Opt0080SchedulingProfile;
  readonly precomputeWallMs: number;
  readonly evaluationWallMs: number;
  readonly graphWallMs: number;
  readonly graphToReadbackObservedIdleMs: number;
  readonly readbackFenceMs: number;
  readonly readbackMapMs: number;
  readonly backendWallMs: number;
  readonly graphEpochWallSumMs: number;
  readonly topology: Opt0080TopologyEvidence;
  readonly heartbeat: Opt0080HeartbeatCapture;
}

export interface Opt0080CancellationEvidence {
  readonly scope: "actual-dit-evaluation0-graph";
  readonly schedulingProfile: typeof OPT_0080_CANDIDATE_PROFILE;
  readonly abortedCommandBufferIndex: 0;
  readonly abortObservedFromCompletionCallback: true;
  readonly outstandingSuccessorsAtAbort: 1;
  readonly prefetchedTailCommandCount: 1;
  readonly backfillAfterAbortCount: 0;
  readonly publicProgressAfterAbortCount: 0;
  readonly specializedCompletionCallbackCount: 1;
  readonly specializedCompletionCallbackCountAfterAbort: 0;
  readonly completedCommandBufferCountAtAbort: 1;
  readonly completionFenceRequestedCountAtAbort: 2;
  readonly checkpointCount: 0;
  readonly unhandledRejectionCount: 0;
  readonly unhandledRejectionListenerRemoved: true;
  readonly generationRejectedWithAbortReason: true;
  readonly terminalSettlementBeforeGenerationRejection: true;
  readonly graphAndModelDestroyAwaitedBeforeGenerationRejection: true;
  readonly backendDisposeAwaitedAfterGenerationRejection: true;
  readonly abortObservationThroughRejectionAndCleanupMs: number;
}

export type Opt0080PerformanceClassification =
  | "passed"
  | "failed"
  | "inconclusive";

export interface Opt0080PerformanceSummary {
  readonly fixedOrder: readonly [
    "A1-depth1-epoch1",
    "B1-opt-0080-depth2-epoch4",
    "B2-opt-0080-depth2-epoch4",
    "A2-depth1-epoch1",
  ];
  readonly samples: readonly Opt0080TimingSample[];
  readonly forwardEvaluationImproved: boolean;
  readonly reverseEvaluationImproved: boolean;
  readonly forwardGraphImproved: boolean;
  readonly reverseGraphImproved: boolean;
  readonly forwardBackendImproved: boolean;
  readonly reverseBackendImproved: boolean;
  readonly aggregateEvaluationSpeedup: number;
  readonly forwardEvaluationSavingMs: number;
  readonly reverseEvaluationSavingMs: number;
  readonly projectedFullGraphSavingMs: number;
  readonly heartbeatAbsolutePassed: boolean;
  readonly heartbeatRelativePassed: boolean;
  readonly mixedPairedDirections: boolean;
  readonly crossMetricDirectionConsistent: boolean;
  readonly wallBoundaryConsistent: boolean;
  readonly cleanReproducibleThresholdMiss: boolean;
  readonly classification: Opt0080PerformanceClassification;
  readonly passed: boolean;
}

export function planOpt0080CompletionEpochs(
  phaseCommandBufferCounts: readonly number[],
  maximumCommandsPerEpoch: 1 | 4,
): readonly Opt0080CompletionEpochPlanEntry[] {
  if (
    phaseCommandBufferCounts.length === 0 ||
    phaseCommandBufferCounts.some((count) =>
      !Number.isSafeInteger(count) || count <= 0
    )
  ) throw new Error("OPT-0080 phase plan is invalid");
  const epochs: Opt0080CompletionEpochPlanEntry[] = [];
  let commandBufferIndex = 0;
  for (
    let phaseIndex = 0;
    phaseIndex < phaseCommandBufferCounts.length;
    phaseIndex += 1
  ) {
    const phaseCount = phaseCommandBufferCounts[phaseIndex]!;
    const phaseEnd = commandBufferIndex + phaseCount;
    while (commandBufferIndex < phaseEnd) {
      const commandBufferCount = Math.min(
        maximumCommandsPerEpoch,
        phaseEnd - commandBufferIndex,
      );
      epochs.push(Object.freeze({
        completionEpochIndex: epochs.length,
        phaseIndex,
        firstCommandBufferIndex: commandBufferIndex,
        lastCommandBufferIndex: commandBufferIndex + commandBufferCount - 1,
        commandBufferCount,
      }));
      commandBufferIndex += commandBufferCount;
    }
  }
  return Object.freeze(epochs);
}

export function requireOpt0080ThermalGate(
  value: Opt0080ThermalGate,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0080ThermalGate {
  return requireOpt0067ThermalGate(
    value,
    readyAtEpochMilliseconds,
    nowEpochMilliseconds,
  );
}

export function requireOpt0080ThermalTrace(
  value: Opt0080ThermalTrace,
  gate: Opt0080ThermalGate,
  cleanupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0080ThermalTrace {
  return requireOpt0067ThermalTrace(
    value,
    gate,
    cleanupCompletedAtEpochMilliseconds,
    nowEpochMilliseconds,
  );
}

export function exactOpt0080ResultIdentity(
  left: Float32Array,
  right: Float32Array,
): boolean {
  return exactOpt0067ResultIdentity(left, right);
}

export function requireOpt0080Topology(
  value: Opt0080TopologyEvidence,
): Opt0080TopologyEvidence {
  const candidate = value.schedulingProfile === OPT_0080_CANDIDATE_PROFILE;
  const expectedEpochs = planOpt0080CompletionEpochs(
    [OPT_0080_PRECOMPUTE_COMMAND_BUFFERS, OPT_0080_EVALUATION_COMMAND_BUFFERS],
    candidate ? 4 : 1,
  );
  const expectedGraphDrains = candidate ? 86 : 341;
  const expectedGraphIdles = expectedGraphDrains - 1;
  const expectedMaximum = candidate ? 2 : 1;
  if (
    value.descriptorTableMemberCount !==
      OPT_0080_DESCRIPTOR_TABLE_MEMBER_COUNT ||
    value.graphCommandBufferCount !== OPT_0080_GRAPH_COMMAND_BUFFERS ||
    value.readbackCommandBufferCount !== 1 ||
    value.totalCommandBufferCount !== OPT_0080_TOTAL_COMMAND_BUFFERS ||
    value.commandBuffersSubmitted !== OPT_0080_TOTAL_COMMAND_BUFFERS ||
    value.completionFenceRequestedCount !== OPT_0080_TOTAL_COMMAND_BUFFERS ||
    value.completionFenceSettledCount !== OPT_0080_TOTAL_COMMAND_BUFFERS ||
    value.completionFenceRejectedCount !== 0 ||
    value.graphTrueQueueDrainCount !== expectedGraphDrains ||
    value.totalTrueQueueDrainCount !== expectedGraphDrains + 1 ||
    value.graphCompletionEpochCount !== expectedGraphDrains ||
    value.graphCooperativeIdleTurns !== expectedGraphIdles ||
    value.totalCooperativeIdleTurns !== expectedGraphIdles + 1 ||
    value.graphRequestedCooperativeIdleMs !== expectedGraphIdles ||
    value.totalRequestedCooperativeIdleMs !== expectedGraphIdles + 1 ||
    value.maximumOutstandingCommandBuffers !== expectedMaximum ||
    value.maximumPendingDescriptorCount !== expectedMaximum ||
    value.pendingDescriptorCountAfterCleanup !== 0 ||
    value.graphToReadbackRequestedIdleMs !== 1 ||
    value.submitThroughCompletionFenceMs.length !==
      OPT_0080_GRAPH_COMMAND_BUFFERS ||
    value.submitThroughCompletionFenceMs.some(invalidDuration) ||
    invalidDuration(value.readbackSubmitThroughCompletionFenceMs) ||
    value.graphCompletionEpochs.length !== expectedEpochs.length ||
    value.graphCompletionEpochs.some((epoch, index) => {
      const expected = expectedEpochs[index]!;
      return epoch.completionEpochIndex !== expected.completionEpochIndex ||
        epoch.phaseIndex !== expected.phaseIndex ||
        epoch.firstCommandBufferIndex !== expected.firstCommandBufferIndex ||
        epoch.lastCommandBufferIndex !== expected.lastCommandBufferIndex ||
        epoch.commandBufferCount !== expected.commandBufferCount ||
        invalidDuration(epoch.submitThroughTrueDrainMs);
    })
  ) throw new Error("OPT-0080 command/fence/epoch topology changed");
  return value;
}

export function requireOpt0080Heartbeat(
  value: Opt0080HeartbeatCapture,
): Opt0080HeartbeatCapture {
  const sorted = [...value.gapsMilliseconds].sort((left, right) => left - right);
  const expectedMaximum = sorted.at(-1);
  const expectedP99 = sorted[Math.max(0, Math.ceil(sorted.length * 0.99) - 1)];
  if (
    value.intervalMilliseconds !== OPT_0080_HEARTBEAT_INTERVAL_MILLISECONDS ||
    !Number.isSafeInteger(value.startedAtEpochMilliseconds) ||
    !Number.isSafeInteger(value.completedAtEpochMilliseconds) ||
    value.completedAtEpochMilliseconds <= value.startedAtEpochMilliseconds ||
    value.gapsMilliseconds.length === 0 ||
    value.gapsMilliseconds.some((gap) => !Number.isFinite(gap) || gap <= 0) ||
    value.maximumGapMilliseconds !== expectedMaximum ||
    value.p99GapMilliseconds !== expectedP99
  ) throw new Error("OPT-0080 heartbeat capture changed");
  return value;
}

export function requireOpt0080Cancellation(
  value: Opt0080CancellationEvidence,
): Opt0080CancellationEvidence {
  if (
    value.scope !== "actual-dit-evaluation0-graph" ||
    value.schedulingProfile !== OPT_0080_CANDIDATE_PROFILE ||
    value.abortedCommandBufferIndex !== 0 ||
    value.abortObservedFromCompletionCallback !== true ||
    value.outstandingSuccessorsAtAbort !== 1 ||
    value.prefetchedTailCommandCount !== 1 ||
    value.backfillAfterAbortCount !== 0 ||
    value.publicProgressAfterAbortCount !== 0 ||
    value.specializedCompletionCallbackCount !== 1 ||
    value.specializedCompletionCallbackCountAfterAbort !== 0 ||
    value.completedCommandBufferCountAtAbort !== 1 ||
    value.completionFenceRequestedCountAtAbort !== 2 ||
    value.checkpointCount !== 0 ||
    value.unhandledRejectionCount !== 0 ||
    value.unhandledRejectionListenerRemoved !== true ||
    value.generationRejectedWithAbortReason !== true ||
    value.terminalSettlementBeforeGenerationRejection !== true ||
    value.graphAndModelDestroyAwaitedBeforeGenerationRejection !== true ||
    value.backendDisposeAwaitedAfterGenerationRejection !== true ||
    !Number.isFinite(value.abortObservationThroughRejectionAndCleanupMs) ||
    value.abortObservationThroughRejectionAndCleanupMs < 0 ||
    value.abortObservationThroughRejectionAndCleanupMs > 1_000
  ) throw new Error("OPT-0080 cancellation/terminal-settlement preflight failed");
  return value;
}

export function summarizeOpt0080Performance(
  samples: readonly Opt0080TimingSample[],
): Opt0080PerformanceSummary {
  if (
    samples.length !== OPT_0080_ARM_ORDER.length ||
    samples.some((sample, index) => {
      const expected = OPT_0080_ARM_ORDER[index]!;
      return sample.armId !== expected.armId || sample.order !== expected.order ||
        sample.schedulingProfile !== expected.schedulingProfile ||
        numericTimingFields(sample).some(invalidDuration) ||
        requireWithoutThrow(() => requireOpt0080Topology(sample.topology)) ===
          false ||
        requireWithoutThrow(() => requireOpt0080Heartbeat(sample.heartbeat)) ===
          false;
    })
  ) throw new Error("OPT-0080 ABBA sample inventory changed");
  const [a1, b1, b2, a2] = samples as readonly [
    Opt0080TimingSample,
    Opt0080TimingSample,
    Opt0080TimingSample,
    Opt0080TimingSample,
  ];
  const forwardEvaluationImproved = b1.evaluationWallMs < a1.evaluationWallMs;
  const reverseEvaluationImproved = b2.evaluationWallMs < a2.evaluationWallMs;
  const forwardGraphImproved = b1.graphWallMs < a1.graphWallMs;
  const reverseGraphImproved = b2.graphWallMs < a2.graphWallMs;
  const forwardBackendImproved = b1.backendWallMs < a1.backendWallMs;
  const reverseBackendImproved = b2.backendWallMs < a2.backendWallMs;
  const aggregateEvaluationSpeedup = ratio(
    a1.evaluationWallMs + a2.evaluationWallMs,
    b1.evaluationWallMs + b2.evaluationWallMs,
  );
  const forwardEvaluationSavingMs = a1.evaluationWallMs - b1.evaluationWallMs;
  const reverseEvaluationSavingMs = a2.evaluationWallMs - b2.evaluationWallMs;
  const meanPrecomputeSavingMs = ((a1.precomputeWallMs - b1.precomputeWallMs) +
    (a2.precomputeWallMs - b2.precomputeWallMs)) / 2;
  const meanEvaluationSavingMs =
    (forwardEvaluationSavingMs + reverseEvaluationSavingMs) / 2;
  const projectedFullGraphSavingMs = meanPrecomputeSavingMs +
    8 * meanEvaluationSavingMs;
  const heartbeatAbsolutePassed = samples.every((sample) =>
    sample.heartbeat.maximumGapMilliseconds <=
      OPT_0080_MAXIMUM_HEARTBEAT_GAP_MILLISECONDS
  );
  const heartbeatRelativePassed = heartbeatPairPassed(a1, b1) &&
    heartbeatPairPassed(a2, b2);
  const pairedDirections = [
    [forwardEvaluationImproved, reverseEvaluationImproved],
    [forwardGraphImproved, reverseGraphImproved],
    [forwardBackendImproved, reverseBackendImproved],
  ] as const;
  const mixedPairedDirections = pairedDirections.some(
    ([forward, reverse]) => forward !== reverse,
  );
  const directionValues = pairedDirections.flat();
  const crossMetricDirectionConsistent = directionValues.every((value) =>
    value === directionValues[0]
  );
  const wallBoundaryConsistent = samples.every((sample) =>
    Math.abs(
      sample.precomputeWallMs + sample.evaluationWallMs - sample.graphWallMs,
    ) <= 1e-6 && sample.graphEpochWallSumMs <= sample.graphWallMs + 1e-6
  );
  const passed = forwardEvaluationImproved && reverseEvaluationImproved &&
    forwardGraphImproved && reverseGraphImproved &&
    forwardBackendImproved && reverseBackendImproved &&
    aggregateEvaluationSpeedup >= 1.04 &&
    forwardEvaluationSavingMs >= 250 && reverseEvaluationSavingMs >= 250 &&
    projectedFullGraphSavingMs >= 2_500 && heartbeatAbsolutePassed &&
    heartbeatRelativePassed && wallBoundaryConsistent;
  const cleanReproducibleThresholdMiss = !passed &&
    !mixedPairedDirections && crossMetricDirectionConsistent &&
    wallBoundaryConsistent;
  const classification: Opt0080PerformanceClassification = passed
    ? "passed"
    : cleanReproducibleThresholdMiss
      ? "failed"
      : "inconclusive";
  return Object.freeze({
    fixedOrder: Object.freeze([
      "A1-depth1-epoch1",
      "B1-opt-0080-depth2-epoch4",
      "B2-opt-0080-depth2-epoch4",
      "A2-depth1-epoch1",
    ]) as Opt0080PerformanceSummary["fixedOrder"],
    samples: Object.freeze([...samples]),
    forwardEvaluationImproved,
    reverseEvaluationImproved,
    forwardGraphImproved,
    reverseGraphImproved,
    forwardBackendImproved,
    reverseBackendImproved,
    aggregateEvaluationSpeedup,
    forwardEvaluationSavingMs,
    reverseEvaluationSavingMs,
    projectedFullGraphSavingMs,
    heartbeatAbsolutePassed,
    heartbeatRelativePassed,
    mixedPairedDirections,
    crossMetricDirectionConsistent,
    wallBoundaryConsistent,
    cleanReproducibleThresholdMiss,
    classification,
    passed,
  });
}

function heartbeatPairPassed(
  control: Opt0080TimingSample,
  candidate: Opt0080TimingSample,
): boolean {
  return candidate.heartbeat.p99GapMilliseconds <= Math.max(
    100,
    1.25 * control.heartbeat.p99GapMilliseconds,
  ) && candidate.heartbeat.maximumGapMilliseconds <= Math.max(
    500,
    1.25 * control.heartbeat.maximumGapMilliseconds,
  );
}

function numericTimingFields(sample: Opt0080TimingSample): readonly number[] {
  return [
    sample.precomputeWallMs,
    sample.evaluationWallMs,
    sample.graphWallMs,
    sample.graphToReadbackObservedIdleMs,
    sample.readbackFenceMs,
    sample.readbackMapMs,
    sample.backendWallMs,
    sample.graphEpochWallSumMs,
  ];
}

function invalidDuration(value: number): boolean {
  return !Number.isFinite(value) || value < 0;
}

function ratio(control: number, candidate: number): number {
  return candidate === 0
    ? control === 0 ? 1 : Number.POSITIVE_INFINITY
    : control / candidate;
}

function requireWithoutThrow(run: () => void): boolean {
  try {
    run();
    return true;
  } catch {
    return false;
  }
}
