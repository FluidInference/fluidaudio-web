import type { Opt0018RunIdentity } from
  "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0080_ARM_ORDER,
  OPT_0080_CANDIDATE_PROFILE,
  OPT_0080_DESCRIPTOR_TABLE_MEMBER_COUNT,
  OPT_0080_EVALUATION0_SHA256,
  OPT_0080_GRAPH_COMMAND_BUFFERS,
  exactOpt0080ResultIdentity,
  requireOpt0080Cancellation,
  requireOpt0080Heartbeat,
  requireOpt0080ThermalGate,
  requireOpt0080ThermalTrace,
  requireOpt0080Topology,
  summarizeOpt0080Performance,
  type Opt0080ArmId,
  type Opt0080CancellationEvidence,
  type Opt0080SchedulingProfile,
  type Opt0080ThermalGate,
  type Opt0080ThermalTrace,
  type Opt0080TimingSample,
} from "./opt-0080-dit-depth2-completion-epochs-contract.js";

export interface Opt0080CorrectnessArm {
  readonly schedulingProfile: Opt0080SchedulingProfile;
  readonly result: Float32Array<ArrayBuffer>;
  readonly resultSha256: string;
  readonly descriptorTableSha256: string;
  readonly descriptorTableMemberCount: number;
  readonly graphCommandBufferCount: number;
  readonly resultNonFiniteCount: number;
  readonly resultNonzeroCount: number;
  readonly uncapturedWebGpuErrorCount: number;
  readonly deviceLost: boolean;
  readonly conditioningAuthority: Readonly<Record<string, unknown>>;
  readonly receipt: Readonly<Record<string, unknown>>;
}

export interface Opt0080CorrectnessPreflight {
  readonly controlFirst: Opt0080CorrectnessArm;
  readonly controlRepeat: Opt0080CorrectnessArm;
  readonly candidateFirst: Opt0080CorrectnessArm;
  readonly candidateRepeat: Opt0080CorrectnessArm;
}

export interface Opt0080TimedArmResult {
  readonly armId: Opt0080ArmId;
  readonly schedulingProfile: Opt0080SchedulingProfile;
  readonly result: Float32Array<ArrayBuffer>;
  readonly resultSha256: string;
  readonly descriptorTableSha256: string;
  readonly conditioningAuthority: Readonly<Record<string, unknown>>;
  readonly sample: Opt0080TimingSample;
  readonly gate: Opt0080ThermalGate;
  readonly trace: Opt0080ThermalTrace;
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly backendCreatedOrdinal: number;
  readonly cleanupCompletedOrdinal: number;
  readonly cleanupCompletedAtEpochMilliseconds: number;
  readonly timingReadyAtEpochMilliseconds: number;
}

export interface Opt0080ResultInput {
  readonly runIdentity: Opt0018RunIdentity;
  readonly requestCanonicalJson: string;
  readonly requestSha256: string;
  readonly requestByteLength: number;
  readonly mainManifestSha256: string;
  readonly denseManifestSha256: string;
  readonly denseRuntimeProfile: string;
  readonly attentionRuntimeProfile: string;
  readonly correctness: Opt0080CorrectnessPreflight;
  readonly cancellation: Opt0080CancellationEvidence;
  readonly arms: readonly Opt0080TimedArmResult[];
  readonly rejectedSetupAttempts:
    readonly Readonly<Record<string, unknown>>[];
}

export function buildOpt0080Result(
  input: Opt0080ResultInput,
): Readonly<Record<string, unknown>> {
  const correctness = requireCorrectness(input.correctness);
  const cancellation = requireOpt0080Cancellation(input.cancellation);
  const timedArms = requireTimedArms(input.arms, correctness.controlFirst);
  const samples = Object.freeze(timedArms.map((arm) => arm.sample));
  const performance = summarizeOpt0080Performance(samples);
  const thermalStartPassed = timedArms.every((arm) =>
    arm.gate.nonNominalObservationCount === 0
  );
  const passed = performance.passed && thermalStartPassed;
  const status = !thermalStartPassed
    ? "inconclusive"
    : performance.classification;
  return Object.freeze({
    schema: "ace-opt-0080-dit-depth2-completion-epochs-gate-v1",
    experimentId: "OPT-0080",
    status,
    passed,
    identity: Object.freeze({
      run: input.runIdentity,
      requestCanonicalJson: input.requestCanonicalJson,
      requestSha256: input.requestSha256,
      requestByteLength: input.requestByteLength,
      mainManifestSha256: input.mainManifestSha256,
      denseManifestSha256: input.denseManifestSha256,
      denseRuntimeProfile: input.denseRuntimeProfile,
      attentionRuntimeProfile: input.attentionRuntimeProfile,
      expectedEvaluation0Sha256: OPT_0080_EVALUATION0_SHA256,
    }),
    correctness: Object.freeze({
      passed: true,
      excludedFromTiming: true,
      resultElementCount: 288_000,
      rawU32MismatchCount: 0,
      deterministicRepeatMismatchCount: 0,
      descriptorTableMemberCount: OPT_0080_DESCRIPTOR_TABLE_MEMBER_COUNT,
      descriptorSequenceExact: true,
      graphCommandBufferCount: OPT_0080_GRAPH_COMMAND_BUFFERS,
      controlFirst: correctness.controlFirst.receipt,
      controlRepeat: correctness.controlRepeat.receipt,
      candidateFirst: correctness.candidateFirst.receipt,
      candidateRepeat: correctness.candidateRepeat.receipt,
    }),
    cancellation,
    performance,
    thermalStartPassed,
    throughCleanupThermalTransitions: Object.freeze(Object.fromEntries(
      timedArms.map((arm) => [
        arm.armId,
        Object.freeze({
          nonNominalObservationCount: arm.trace.nonNominalObservationCount,
          transitions: arm.trace.transitions,
        }),
      ]),
    )),
    arms: Object.freeze(Object.fromEntries(timedArms.map((arm) => [
      arm.armId,
      Object.freeze({
        gate: arm.gate,
        thermalTrace: arm.trace,
        heartbeat: arm.sample.heartbeat,
        receipt: arm.receipt,
      }),
    ]))),
    rejectedSetupAttempts: Object.freeze([...input.rejectedSetupAttempts]),
    lifecycle: Object.freeze({
      oneFifoGraphOwnerPerArm: true,
      sequentialNonOverlappingPackageOwnership: true,
      noCrossArmBackendOrDeviceReuse: true,
      graphCompiledBeforeEveryThermalGate: true,
      distinctThermalTracePerArm: true,
      oneCompletionFencePerPhysicalCommandBuffer: true,
      terminalSettlementBeforeRelease: true,
      allBackendsAndDevicesDisposed: true,
      executedEvaluationCountPerArm: 1,
      laterEvaluationEncodeCount: 0,
    }),
    metrics: Object.freeze({
      overlappingFenceLatenciesAreNonAdditive: true,
      performanceClaimsUseAuthoritativeWalls: true,
      graphEpochWallsAreDisjoint: true,
      controlGraphTrueDrains: 341,
      candidateGraphTrueDrains: 86,
      controlGraphIdleTurns: 340,
      candidateGraphIdleTurns: 85,
      maximumCandidateOutstandingCommandBuffers: 2,
    }),
    scope: Object.freeze({
      productionDefaultChanged: false,
      packageBytesChanged: false,
      modelMathChanged: false,
      commandBufferContentsChanged: false,
      commandBufferCoalescing: false,
      uniformPoolUsedByCandidate: false,
      vaeExecuted: false,
      plannerExecuted: false,
      laterEvaluationsExecuted: false,
      fullGraphClaim: false,
      listeningClaim: false,
      underOneMinuteClaim: false,
    }),
    decision: passed
      ? "evaluation-slice-pass-authorize-separate-full-graph-confirmation"
      : status === "inconclusive"
        ? "evaluation-slice-inconclusive-keep-depth1-production-default"
        : "evaluation-slice-failed-keep-depth1-production-default",
  });
}

function requireCorrectness(
  value: Opt0080CorrectnessPreflight,
): Opt0080CorrectnessPreflight {
  const arms = [
    value.controlFirst,
    value.controlRepeat,
    value.candidateFirst,
    value.candidateRepeat,
  ] as const;
  if (
    value.controlFirst.schedulingProfile !== "depth1-epoch1" ||
    value.controlRepeat.schedulingProfile !== "depth1-epoch1" ||
    value.candidateFirst.schedulingProfile !== OPT_0080_CANDIDATE_PROFILE ||
    value.candidateRepeat.schedulingProfile !== OPT_0080_CANDIDATE_PROFILE ||
    arms.some((arm) =>
      arm.resultSha256 !== OPT_0080_EVALUATION0_SHA256 ||
      arm.descriptorTableMemberCount !==
        OPT_0080_DESCRIPTOR_TABLE_MEMBER_COUNT ||
      arm.graphCommandBufferCount !== OPT_0080_GRAPH_COMMAND_BUFFERS ||
      arm.resultNonFiniteCount !== 0 || arm.resultNonzeroCount <= 0 ||
      arm.uncapturedWebGpuErrorCount !== 0 ||
      arm.deviceLost
    ) ||
    arms.some((arm) =>
      arm.descriptorTableSha256 !== value.controlFirst.descriptorTableSha256 ||
      !sameJson(
        arm.conditioningAuthority,
        value.controlFirst.conditioningAuthority,
      ) ||
      !exactOpt0080ResultIdentity(arm.result, value.controlFirst.result)
    )
  ) throw new Error("OPT-0080 untimed correctness/repeat identity failed");
  return value;
}

function requireTimedArms(
  arms: readonly Opt0080TimedArmResult[],
  correctness: Opt0080CorrectnessArm,
): readonly Opt0080TimedArmResult[] {
  if (
    arms.length !== OPT_0080_ARM_ORDER.length ||
    arms.some((arm, index) => {
      const expected = OPT_0080_ARM_ORDER[index]!;
      return arm.armId !== expected.armId ||
        arm.schedulingProfile !== expected.schedulingProfile ||
        arm.sample.armId !== expected.armId || arm.sample.order !== index ||
        arm.sample.schedulingProfile !== expected.schedulingProfile ||
        arm.resultSha256 !== OPT_0080_EVALUATION0_SHA256 ||
        arm.descriptorTableSha256 !== correctness.descriptorTableSha256 ||
        !sameJson(arm.conditioningAuthority, correctness.conditioningAuthority) ||
        !exactOpt0080ResultIdentity(arm.result, correctness.result) ||
        !Number.isSafeInteger(arm.backendCreatedOrdinal) ||
        !Number.isSafeInteger(arm.cleanupCompletedOrdinal) ||
        !Number.isSafeInteger(arm.cleanupCompletedAtEpochMilliseconds) ||
        !Number.isSafeInteger(arm.timingReadyAtEpochMilliseconds) ||
        arm.cleanupCompletedOrdinal <= arm.backendCreatedOrdinal ||
        arm.sample.heartbeat.startedAtEpochMilliseconds <
          arm.gate.completedAtEpochMilliseconds ||
        arm.sample.heartbeat.completedAtEpochMilliseconds <
          arm.cleanupCompletedAtEpochMilliseconds ||
        requireFailed(() => requireOpt0080Topology(arm.sample.topology)) ||
        requireFailed(() => requireOpt0080Heartbeat(arm.sample.heartbeat)) ||
        requireFailed(() => requireOpt0080ThermalGate(
          arm.gate,
          arm.timingReadyAtEpochMilliseconds,
          arm.gate.completedAtEpochMilliseconds,
        ));
    }) ||
    arms.some((arm, index) =>
      index > 0 &&
      arms[index - 1]!.cleanupCompletedOrdinal >= arm.backendCreatedOrdinal
    ) ||
    new Set(arms.map((arm) => arm.trace.rawTraceSha256)).size !== arms.length ||
    arms.some((arm) =>
      requireFailed(() => requireOpt0080ThermalTrace(
        arm.trace,
        arm.gate,
        arm.cleanupCompletedAtEpochMilliseconds,
        arm.trace.completedAtEpochMilliseconds,
      ))
    )
  ) throw new Error("OPT-0080 timed arm identity/lifecycle changed");
  return Object.freeze([...arms]);
}

function requireFailed(run: () => void): boolean {
  try {
    run();
    return false;
  } catch {
    return true;
  }
}

function sameJson(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}
