import {
  OPT_0080_CANDIDATE_PROFILE,
  OPT_0080_CONTROL_PROFILE,
  OPT_0080_DESCRIPTOR_TABLE_MEMBER_COUNT,
  OPT_0080_HEARTBEAT_INTERVAL_MILLISECONDS,
  OPT_0080_MAXIMUM_HEARTBEAT_GAP_MILLISECONDS,
  planOpt0080CompletionEpochs,
  requireOpt0080Heartbeat,
  requireOpt0080ThermalGate,
  requireOpt0080ThermalTrace,
  type Opt0080CompletionEpochTiming,
  type Opt0080HeartbeatCapture,
  type Opt0080SchedulingProfile,
  type Opt0080ThermalGate,
  type Opt0080ThermalTrace,
} from "./opt-0080-dit-depth2-completion-epochs-contract.js";

export {
  OPT_0080_CANDIDATE_PROFILE,
  OPT_0080_CONTROL_PROFILE,
  OPT_0080_HEARTBEAT_INTERVAL_MILLISECONDS,
  type Opt0080HeartbeatCapture as Opt0080FullHeartbeatCapture,
  type Opt0080SchedulingProfile as Opt0080FullSchedulingProfile,
  type Opt0080ThermalGate as Opt0080FullThermalGate,
  type Opt0080ThermalTrace as Opt0080FullThermalTrace,
};

export const OPT_0080_FULL_GRAPH_COMMAND_BUFFERS = 2_553 as const;
export const OPT_0080_FULL_READBACK_COMMAND_BUFFERS = 1 as const;
export const OPT_0080_FULL_TOTAL_COMMAND_BUFFERS = 2_554 as const;
export const OPT_0080_FULL_PRECOMPUTE_COMMAND_BUFFERS = 25 as const;
export const OPT_0080_FULL_EVALUATION_COMMAND_BUFFERS = 316 as const;
export const OPT_0080_FULL_EVALUATION_COUNT = 8 as const;
export const OPT_0080_FULL_TENSOR_ELEMENTS = 288_000 as const;
export const OPT_0080_FULL_TENSOR_BYTES = 1_152_000 as const;
export const OPT_0080_FULL_DESCRIPTOR_TABLE_MEMBER_COUNT =
  OPT_0080_DESCRIPTOR_TABLE_MEMBER_COUNT;
export const OPT_0080_EVALUATION_SLICE_EVIDENCE_SHA256 =
  "a2ad4320f22be728898668d76b84524cd2957cc1e08a588f36a0f3714b183fd9" as const;

/**
 * Accepted OPT-0062 full-trajectory hashes under the same authenticated
 * revision-7 dense package and now-production OPT-0070 attention arithmetic.
 */
export const OPT_0080_FULL_EVALUATION_SHA256 = Object.freeze([
  "d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286",
  "29b113de627c68cfe67cd1d3e2fe3ef3279d968e1584d16ffb3e5f217e220331",
  "d4ef5e04ceaa6b20e8dd0bd87afbb47a61a8c3f2e4e936d74b50998f40d5a1ca",
  "d1cff5e3804105a21a9188ac784a9730fcbd4e589aaade20b5bfefeee4837543",
  "67539d0c653881306821bffa0934c0770a8eaaa96f003cecf5cc6f0d9005bbb2",
  "060c796e1a625b114bc40fc29e310e8c0cce568164b71f776dc0a42c41c0f85e",
  "73c345cb357c9081c33565d61356e2c07090b6c8011d552ddfec6243f862b237",
  "1812a085f48b7879212633c7193dda08ec2854852a492ce661262c5e6be98f4c",
] as const);
export const OPT_0080_FULL_FINAL_LATENT_SHA256 =
  OPT_0080_FULL_EVALUATION_SHA256[7];

export type Opt0080FullArmId = "A" | "B";

export const OPT_0080_FULL_ARM_ORDER = Object.freeze([
  Object.freeze({
    armId: "A" as const,
    order: 0 as const,
    schedulingProfile: OPT_0080_CONTROL_PROFILE,
  }),
  Object.freeze({
    armId: "B" as const,
    order: 1 as const,
    schedulingProfile: OPT_0080_CANDIDATE_PROFILE,
  }),
]);

export const OPT_0080_FULL_PHASE_COMMAND_BUFFER_COUNTS = Object.freeze([
  OPT_0080_FULL_PRECOMPUTE_COMMAND_BUFFERS,
  OPT_0080_FULL_EVALUATION_COMMAND_BUFFERS,
  OPT_0080_FULL_EVALUATION_COMMAND_BUFFERS,
  OPT_0080_FULL_EVALUATION_COMMAND_BUFFERS,
  OPT_0080_FULL_EVALUATION_COMMAND_BUFFERS,
  OPT_0080_FULL_EVALUATION_COMMAND_BUFFERS,
  OPT_0080_FULL_EVALUATION_COMMAND_BUFFERS,
  OPT_0080_FULL_EVALUATION_COMMAND_BUFFERS,
  OPT_0080_FULL_EVALUATION_COMMAND_BUFFERS,
] as const);

export interface Opt0080FullTopologyEvidence {
  readonly schedulingProfile: Opt0080SchedulingProfile;
  readonly descriptorTableMemberCount: 6_833;
  readonly graphCommandBufferCount: 2_553;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: 2_554;
  readonly commandBuffersSubmitted: 2_554;
  readonly completionFenceRequestedCount: 2_554;
  readonly completionFenceSettledCount: 2_554;
  readonly completionFenceRejectedCount: 0;
  readonly graphTrueQueueDrainCount: 2_553 | 639;
  readonly totalTrueQueueDrainCount: 2_554 | 640;
  readonly graphCompletionEpochCount: 2_553 | 639;
  readonly graphCooperativeIdleTurns: 2_552 | 638;
  readonly totalCooperativeIdleTurns: 2_553 | 639;
  readonly graphRequestedCooperativeIdleMs: 2_552 | 638;
  readonly totalRequestedCooperativeIdleMs: 2_553 | 639;
  readonly maximumOutstandingCommandBuffers: 1 | 2;
  readonly maximumPendingDescriptorCount: 1 | 2;
  readonly pendingDescriptorCountAfterCleanup: 0;
  readonly graphCompletionEpochs:
    readonly Opt0080CompletionEpochTiming[];
  /** Cumulative and overlapping under depth two; never additive. */
  readonly submitThroughCompletionFenceMs: readonly number[];
  readonly graphToReadbackRequestedIdleMs: 1;
  readonly readbackSubmitThroughCompletionFenceMs: number;
}

export interface Opt0080FullTimingSample {
  readonly armId: Opt0080FullArmId;
  readonly order: 0 | 1;
  readonly schedulingProfile: Opt0080SchedulingProfile;
  readonly precomputeWallMs: number;
  readonly evaluationWallMs: readonly [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  readonly graphWallMs: number;
  readonly graphToReadbackObservedIdleMs: number;
  readonly readbackFenceMs: number;
  readonly readbackMapMs: number;
  readonly backendWallMs: number;
  readonly graphEpochWallSumMs: number;
  readonly topology: Opt0080FullTopologyEvidence;
  readonly heartbeat: Opt0080HeartbeatCapture;
}

export interface Opt0080FullCancellationEvidence {
  readonly scope: "actual-dit-full-graph";
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

export type Opt0080FullPerformanceClassification =
  | "passed"
  | "failed"
  | "inconclusive";

export interface Opt0080FullPerformanceSummary {
  readonly fixedOrder: readonly [
    "A-depth1-epoch1",
    "B-opt-0080-depth2-epoch4",
  ];
  readonly samples: readonly [
    Opt0080FullTimingSample,
    Opt0080FullTimingSample,
  ];
  readonly graphImproved: boolean;
  readonly backendImproved: boolean;
  readonly graphSpeedup: number;
  readonly graphSavingMs: number;
  readonly heartbeatAbsolutePassed: boolean;
  readonly heartbeatRelativePassed: boolean;
  readonly wallBoundaryConsistent: boolean;
  readonly classification: Opt0080FullPerformanceClassification;
  readonly passed: boolean;
}

export function requireOpt0080FullTopology(
  value: Opt0080FullTopologyEvidence,
): Opt0080FullTopologyEvidence {
  const candidate = value.schedulingProfile === OPT_0080_CANDIDATE_PROFILE;
  const expectedEpochs = planOpt0080CompletionEpochs(
    OPT_0080_FULL_PHASE_COMMAND_BUFFER_COUNTS,
    candidate ? 4 : 1,
  );
  const expectedGraphDrains = candidate ? 639 : 2_553;
  const expectedGraphIdles = expectedGraphDrains - 1;
  const expectedMaximum = candidate ? 2 : 1;
  if (
    value.descriptorTableMemberCount !==
      OPT_0080_FULL_DESCRIPTOR_TABLE_MEMBER_COUNT ||
    value.graphCommandBufferCount !== OPT_0080_FULL_GRAPH_COMMAND_BUFFERS ||
    value.readbackCommandBufferCount !== OPT_0080_FULL_READBACK_COMMAND_BUFFERS ||
    value.totalCommandBufferCount !== OPT_0080_FULL_TOTAL_COMMAND_BUFFERS ||
    value.commandBuffersSubmitted !== OPT_0080_FULL_TOTAL_COMMAND_BUFFERS ||
    value.completionFenceRequestedCount !==
      OPT_0080_FULL_TOTAL_COMMAND_BUFFERS ||
    value.completionFenceSettledCount !==
      OPT_0080_FULL_TOTAL_COMMAND_BUFFERS ||
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
      OPT_0080_FULL_GRAPH_COMMAND_BUFFERS ||
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
  ) throw new Error("OPT-0080 full command/fence/epoch topology changed");
  return value;
}

export function requireOpt0080FullHeartbeat(
  value: Opt0080HeartbeatCapture,
): Opt0080HeartbeatCapture {
  return requireOpt0080Heartbeat(value);
}

export function requireOpt0080FullThermalGate(
  value: Opt0080ThermalGate,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0080ThermalGate {
  return requireOpt0080ThermalGate(
    value,
    readyAtEpochMilliseconds,
    nowEpochMilliseconds,
  );
}

export function requireOpt0080FullThermalTrace(
  value: Opt0080ThermalTrace,
  gate: Opt0080ThermalGate,
  cleanupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0080ThermalTrace {
  return requireOpt0080ThermalTrace(
    value,
    gate,
    cleanupCompletedAtEpochMilliseconds,
    nowEpochMilliseconds,
  );
}

export function requireOpt0080FullCancellation(
  value: Opt0080FullCancellationEvidence,
): Opt0080FullCancellationEvidence {
  if (
    value.scope !== "actual-dit-full-graph" ||
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
  ) throw new Error("OPT-0080 full cancellation/settlement preflight failed");
  return value;
}

export function exactOpt0080FullTensorIdentity(
  left: Float32Array,
  right: Float32Array,
): boolean {
  if (
    left.length !== OPT_0080_FULL_TENSOR_ELEMENTS ||
    right.length !== OPT_0080_FULL_TENSOR_ELEMENTS
  ) return false;
  const a = new Uint32Array(left.buffer, left.byteOffset, left.length);
  const b = new Uint32Array(right.buffer, right.byteOffset, right.length);
  return a.every((word, index) => word === b[index]);
}

export function summarizeOpt0080FullPerformance(
  samples: readonly Opt0080FullTimingSample[],
): Opt0080FullPerformanceSummary {
  if (
    samples.length !== OPT_0080_FULL_ARM_ORDER.length ||
    samples.some((sample, index) => {
      const expected = OPT_0080_FULL_ARM_ORDER[index]!;
      return sample.armId !== expected.armId || sample.order !== expected.order ||
        sample.schedulingProfile !== expected.schedulingProfile ||
        numericTimingFields(sample).some(invalidDuration) ||
        requireWithoutThrow(() => requireOpt0080FullTopology(sample.topology)) ===
          false ||
        requireWithoutThrow(() => requireOpt0080FullHeartbeat(sample.heartbeat)) ===
          false;
    })
  ) throw new Error("OPT-0080 full A/B sample inventory changed");
  const [control, candidate] = samples as readonly [
    Opt0080FullTimingSample,
    Opt0080FullTimingSample,
  ];
  const graphImproved = candidate.graphWallMs < control.graphWallMs;
  const backendImproved = candidate.backendWallMs < control.backendWallMs;
  const graphSpeedup = ratio(control.graphWallMs, candidate.graphWallMs);
  const graphSavingMs = control.graphWallMs - candidate.graphWallMs;
  const heartbeatAbsolutePassed = samples.every((sample) =>
    sample.heartbeat.maximumGapMilliseconds <=
      OPT_0080_MAXIMUM_HEARTBEAT_GAP_MILLISECONDS
  );
  const heartbeatRelativePassed =
    candidate.heartbeat.p99GapMilliseconds <= Math.max(
      100,
      1.25 * control.heartbeat.p99GapMilliseconds,
    ) && candidate.heartbeat.maximumGapMilliseconds <= Math.max(
      500,
      1.25 * control.heartbeat.maximumGapMilliseconds,
    );
  const wallBoundaryConsistent = samples.every((sample) =>
    Math.abs(
      sample.precomputeWallMs +
        sample.evaluationWallMs.reduce((sum, value) => sum + value, 0) -
        sample.graphWallMs,
    ) <= 1e-6 && sample.graphEpochWallSumMs <= sample.graphWallMs + 1e-6 &&
    sample.backendWallMs + 1e-6 >= sample.graphWallMs
  );
  const passed = graphImproved && backendImproved && graphSpeedup >= 1.04 &&
    graphSavingMs >= 2_500 && heartbeatAbsolutePassed &&
    heartbeatRelativePassed && wallBoundaryConsistent;
  const classification: Opt0080FullPerformanceClassification = passed
    ? "passed"
    : !wallBoundaryConsistent
      ? "inconclusive"
      : !heartbeatAbsolutePassed || !heartbeatRelativePassed
        ? "failed"
        : !graphImproved && !backendImproved
          ? "failed"
          : "inconclusive";
  return Object.freeze({
    fixedOrder: Object.freeze([
      "A-depth1-epoch1",
      "B-opt-0080-depth2-epoch4",
    ]) as Opt0080FullPerformanceSummary["fixedOrder"],
    samples: Object.freeze([...samples]) as unknown as
      Opt0080FullPerformanceSummary["samples"],
    graphImproved,
    backendImproved,
    graphSpeedup,
    graphSavingMs,
    heartbeatAbsolutePassed,
    heartbeatRelativePassed,
    wallBoundaryConsistent,
    classification,
    passed,
  });
}

function numericTimingFields(sample: Opt0080FullTimingSample): readonly number[] {
  return [
    sample.precomputeWallMs,
    ...sample.evaluationWallMs,
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
