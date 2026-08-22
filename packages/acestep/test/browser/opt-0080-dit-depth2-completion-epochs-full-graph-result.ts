import type { Opt0018RunIdentity } from
  "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0080_CANDIDATE_PROFILE,
  OPT_0080_CONTROL_PROFILE,
  OPT_0080_FULL_ARM_ORDER,
  OPT_0080_FULL_DESCRIPTOR_TABLE_MEMBER_COUNT,
  OPT_0080_FULL_EVALUATION_COUNT,
  OPT_0080_FULL_EVALUATION_SHA256,
  OPT_0080_FULL_FINAL_LATENT_SHA256,
  OPT_0080_FULL_GRAPH_COMMAND_BUFFERS,
  OPT_0080_FULL_TENSOR_ELEMENTS,
  exactOpt0080FullTensorIdentity,
  requireOpt0080FullCancellation,
  requireOpt0080FullHeartbeat,
  requireOpt0080FullThermalGate,
  requireOpt0080FullThermalTrace,
  requireOpt0080FullTopology,
  summarizeOpt0080FullPerformance,
  type Opt0080FullArmId,
  type Opt0080FullCancellationEvidence,
  type Opt0080FullSchedulingProfile,
  type Opt0080FullThermalGate,
  type Opt0080FullThermalTrace,
  type Opt0080FullTimingSample,
  type Opt0080FullTopologyEvidence,
} from
  "./opt-0080-dit-depth2-completion-epochs-full-graph-contract.js";

export interface Opt0080FullCorrectnessTap {
  readonly evaluation: number;
  readonly result: Float32Array<ArrayBuffer>;
  /** Same-buffer raw-bit view supplied by the runtime checkpoint. */
  readonly rawU32: Uint32Array<ArrayBuffer>;
  readonly sha256: string;
  readonly nonFiniteCount: 0;
  readonly nonzeroCount: number;
  readonly maxAbs: number;
}

export interface Opt0080FullCorrectnessArm {
  readonly schedulingProfile: Opt0080FullSchedulingProfile;
  readonly captureEvaluationTaps: true;
  readonly evaluationTaps: readonly Opt0080FullCorrectnessTap[];
  readonly finalLatent: Float32Array<ArrayBuffer>;
  /** Same-buffer raw-bit view supplied by the runtime checkpoint. */
  readonly finalLatentRawU32: Uint32Array<ArrayBuffer>;
  readonly finalLatentSha256: string;
  readonly finalLatentNonFiniteCount: 0;
  readonly finalLatentNonzeroCount: number;
  readonly finalLatentMaxAbs: number;
  readonly descriptorTableSha256: string;
  readonly descriptorTableMemberCount: number;
  readonly topology: Opt0080FullTopologyEvidence;
  readonly evaluationTapInCommandCopyCount: 8;
  readonly evaluationTapExtraCommandBufferCount: 0;
  readonly evaluationTapExtraQueueDrainCount: 0;
  readonly uncapturedWebGpuErrorCount: 0;
  readonly deviceLost: false;
  readonly conditioningAuthority: Readonly<Record<string, unknown>>;
  readonly receipt: Readonly<Record<string, unknown>>;
}

export interface Opt0080FullCorrectnessPreflight {
  readonly control: Opt0080FullCorrectnessArm;
  readonly candidate: Opt0080FullCorrectnessArm;
}

export interface Opt0080FullTimedArmResult {
  readonly armId: Opt0080FullArmId;
  readonly schedulingProfile: Opt0080FullSchedulingProfile;
  readonly captureEvaluationTaps: false;
  readonly evaluationTapCount: 0;
  readonly finalLatent: Float32Array<ArrayBuffer>;
  readonly finalLatentRawU32: Uint32Array<ArrayBuffer>;
  readonly finalLatentSha256: string;
  readonly descriptorTableSha256: string;
  readonly conditioningAuthority: Readonly<Record<string, unknown>>;
  readonly sample: Opt0080FullTimingSample;
  readonly gate: Opt0080FullThermalGate;
  readonly trace: Opt0080FullThermalTrace;
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly backendCreatedOrdinal: number;
  readonly cleanupCompletedOrdinal: number;
  readonly cleanupCompletedAtEpochMilliseconds: number;
  readonly timingReadyAtEpochMilliseconds: number;
}

export interface Opt0080FullResultInput {
  readonly runIdentity: Opt0018RunIdentity;
  readonly requestCanonicalJson: string;
  readonly requestSha256: string;
  readonly requestByteLength: number;
  readonly mainManifestSha256: string;
  readonly denseManifestSha256: string;
  readonly denseRuntimeProfile: string;
  readonly attentionRuntimeProfile: string;
  readonly evaluationSliceResultSha256: string;
  readonly correctness: Opt0080FullCorrectnessPreflight;
  readonly cancellation: Opt0080FullCancellationEvidence;
  readonly arms: readonly Opt0080FullTimedArmResult[];
  readonly rejectedSetupAttempts:
    readonly Readonly<Record<string, unknown>>[];
}

export function buildOpt0080FullResult(
  input: Opt0080FullResultInput,
): Readonly<Record<string, unknown>> {
  const correctness = requireCorrectness(input.correctness);
  const cancellation = requireOpt0080FullCancellation(input.cancellation);
  const arms = requireTimedArms(input.arms, correctness.control);
  const performance = summarizeOpt0080FullPerformance(
    arms.map((arm) => arm.sample),
  );
  const thermalStartPassed = arms.every((arm) =>
    arm.gate.nonNominalObservationCount === 0
  );
  const passed = performance.passed && thermalStartPassed;
  const status = thermalStartPassed
    ? performance.classification
    : "inconclusive";
  return Object.freeze({
    schema: "ace-opt-0080-dit-depth2-completion-epochs-full-graph-gate-v1",
    experimentId: "OPT-0080",
    stage: "full-m2250-confirmation",
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
      evaluationSliceResultSha256: input.evaluationSliceResultSha256,
      expectedEvaluationSha256: OPT_0080_FULL_EVALUATION_SHA256,
      expectedFinalLatentSha256: OPT_0080_FULL_FINAL_LATENT_SHA256,
    }),
    correctness: Object.freeze({
      passed: true,
      excludedFromTiming: true,
      control: correctness.control.receipt,
      candidate: correctness.candidate.receipt,
      evaluationCount: OPT_0080_FULL_EVALUATION_COUNT,
      evaluationElementsPerTap: OPT_0080_FULL_TENSOR_ELEMENTS,
      comparedEvaluationRawU32Words:
        OPT_0080_FULL_EVALUATION_COUNT * OPT_0080_FULL_TENSOR_ELEMENTS,
      evaluationRawU32MismatchCount: 0,
      finalLatentRawU32MismatchCount: 0,
      finalTapToFinalReadbackRawU32MismatchCount: 0,
      everyTapMatchedAcceptedHash: true,
      descriptorTableMemberCount:
        OPT_0080_FULL_DESCRIPTOR_TABLE_MEMBER_COUNT,
      descriptorSequenceExact: true,
      graphCommandBufferCount: OPT_0080_FULL_GRAPH_COMMAND_BUFFERS,
      evaluationTapInCommandCopyCountPerArm: 8,
      evaluationTapExtraCommandBufferCount: 0,
      evaluationTapExtraQueueDrainCount: 0,
    }),
    cancellation,
    performance,
    thermalStartPassed,
    throughCleanupThermalTransitions: Object.freeze(Object.fromEntries(
      arms.map((arm) => [
        arm.armId,
        Object.freeze({
          nonNominalObservationCount: arm.trace.nonNominalObservationCount,
          transitions: arm.trace.transitions,
        }),
      ]),
    )),
    arms: Object.freeze(Object.fromEntries(arms.map((arm) => [
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
      executedEvaluationCountPerArm: 8,
      timedEvaluationSnapshotCount: 0,
      timedReadbackCount: 1,
    }),
    metrics: Object.freeze({
      performanceClaimsUseAuthoritativeGraphWall: true,
      overlappingFenceLatenciesAreNonAdditive: true,
      graphEpochWallsAreDisjoint: true,
      controlGraphTrueDrains: 2_553,
      candidateGraphTrueDrains: 639,
      controlGraphIdleTurns: 2_552,
      candidateGraphIdleTurns: 638,
      maximumCandidateOutstandingCommandBuffers: 2,
    }),
    scope: Object.freeze({
      productionDefaultChanged: false,
      productionIntegrationPerformed: false,
      packageBytesChanged: false,
      modelMathChanged: false,
      timedCommandBufferContentsChanged: false,
      untimedCorrectnessCommandBuffersAddedEvaluationTapCopies: true,
      untimedEvaluationTapInCommandCopyCountPerArm: 8,
      commandBufferCoalescing: false,
      uniformPoolUsedByCandidate: false,
      vaeExecuted: false,
      plannerExecuted: false,
      fullGraphClaim: true,
      listeningClaim: false,
      underOneMinuteClaim: false,
    }),
    decision: passed
      ? "full-graph-pass-authorize-production-dit-integration"
      : status === "inconclusive"
        ? "full-graph-inconclusive-keep-depth1-production-default"
        : "full-graph-failed-keep-depth1-production-default",
  });
}

function requireCorrectness(
  value: Opt0080FullCorrectnessPreflight,
): Opt0080FullCorrectnessPreflight {
  const arms = [value.control, value.candidate] as const;
  if (
    value.control.schedulingProfile !== OPT_0080_CONTROL_PROFILE ||
    value.candidate.schedulingProfile !== OPT_0080_CANDIDATE_PROFILE ||
    arms.some((arm) =>
      arm.captureEvaluationTaps !== true ||
      arm.evaluationTaps.length !== OPT_0080_FULL_EVALUATION_COUNT ||
      arm.finalLatentSha256 !== OPT_0080_FULL_FINAL_LATENT_SHA256 ||
      !sameBufferViews(arm.finalLatent, arm.finalLatentRawU32) ||
      arm.finalLatentNonFiniteCount !== 0 ||
      arm.finalLatentNonzeroCount <= 0 || !(arm.finalLatentMaxAbs > 0) ||
      !/^[0-9a-f]{64}$/u.test(arm.descriptorTableSha256) ||
      arm.descriptorTableMemberCount !==
        OPT_0080_FULL_DESCRIPTOR_TABLE_MEMBER_COUNT ||
      arm.evaluationTapInCommandCopyCount !== 8 ||
      arm.evaluationTapExtraCommandBufferCount !== 0 ||
      arm.evaluationTapExtraQueueDrainCount !== 0 ||
      arm.uncapturedWebGpuErrorCount !== 0 || arm.deviceLost !== false ||
      requireFailed(() => requireOpt0080FullTopology(arm.topology)) ||
      arm.evaluationTaps.some((tap, evaluation) =>
        tap.evaluation !== evaluation ||
        tap.sha256 !== OPT_0080_FULL_EVALUATION_SHA256[evaluation] ||
        !sameBufferViews(tap.result, tap.rawU32) ||
        tap.nonFiniteCount !== 0 || tap.nonzeroCount <= 0 || !(tap.maxAbs > 0)
      ) ||
      !rawU32Equal(
        arm.evaluationTaps[7]!.rawU32,
        arm.finalLatentRawU32,
      )
    ) ||
    value.candidate.descriptorTableSha256 !==
      value.control.descriptorTableSha256 ||
    !sameJson(
      value.candidate.conditioningAuthority,
      value.control.conditioningAuthority,
    ) ||
    value.control.evaluationTaps.some((tap, evaluation) =>
      !exactOpt0080FullTensorIdentity(
        tap.result,
        value.candidate.evaluationTaps[evaluation]!.result,
      )
    ) ||
    !exactOpt0080FullTensorIdentity(
      value.control.finalLatent,
      value.candidate.finalLatent,
    )
  ) throw new Error("OPT-0080 full untimed trajectory identity failed");
  return value;
}

function requireTimedArms(
  arms: readonly Opt0080FullTimedArmResult[],
  correctness: Opt0080FullCorrectnessArm,
): readonly Opt0080FullTimedArmResult[] {
  if (
    arms.length !== OPT_0080_FULL_ARM_ORDER.length ||
    arms.some((arm, index) => {
      const expected = OPT_0080_FULL_ARM_ORDER[index]!;
      return arm.armId !== expected.armId ||
        arm.schedulingProfile !== expected.schedulingProfile ||
        arm.sample.armId !== expected.armId || arm.sample.order !== index ||
        arm.sample.schedulingProfile !== expected.schedulingProfile ||
        arm.captureEvaluationTaps !== false || arm.evaluationTapCount !== 0 ||
        arm.finalLatentSha256 !== OPT_0080_FULL_FINAL_LATENT_SHA256 ||
        !sameBufferViews(arm.finalLatent, arm.finalLatentRawU32) ||
        arm.descriptorTableSha256 !== correctness.descriptorTableSha256 ||
        !sameJson(arm.conditioningAuthority, correctness.conditioningAuthority) ||
        !exactOpt0080FullTensorIdentity(
          arm.finalLatent,
          correctness.finalLatent,
        ) ||
        !Number.isSafeInteger(arm.backendCreatedOrdinal) ||
        !Number.isSafeInteger(arm.cleanupCompletedOrdinal) ||
        !Number.isSafeInteger(arm.cleanupCompletedAtEpochMilliseconds) ||
        !Number.isSafeInteger(arm.timingReadyAtEpochMilliseconds) ||
        arm.cleanupCompletedOrdinal <= arm.backendCreatedOrdinal ||
        arm.sample.heartbeat.startedAtEpochMilliseconds <
          arm.gate.completedAtEpochMilliseconds ||
        arm.sample.heartbeat.completedAtEpochMilliseconds <
          arm.cleanupCompletedAtEpochMilliseconds ||
        requireFailed(() => requireOpt0080FullTopology(arm.sample.topology)) ||
        requireFailed(() => requireOpt0080FullHeartbeat(arm.sample.heartbeat)) ||
        requireFailed(() => requireOpt0080FullThermalGate(
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
      requireFailed(() => requireOpt0080FullThermalTrace(
        arm.trace,
        arm.gate,
        arm.cleanupCompletedAtEpochMilliseconds,
        arm.trace.completedAtEpochMilliseconds,
      ))
    )
  ) throw new Error("OPT-0080 full timed arm identity/lifecycle changed");
  return Object.freeze([...arms]);
}

function sameBufferViews(
  floats: Float32Array,
  words: Uint32Array,
): boolean {
  return floats.length === OPT_0080_FULL_TENSOR_ELEMENTS &&
    words.length === OPT_0080_FULL_TENSOR_ELEMENTS &&
    floats.buffer === words.buffer && floats.byteOffset === words.byteOffset &&
    floats.byteLength === words.byteLength;
}

function rawU32Equal(left: Uint32Array, right: Uint32Array): boolean {
  return left.length === OPT_0080_FULL_TENSOR_ELEMENTS &&
    right.length === OPT_0080_FULL_TENSOR_ELEMENTS &&
    left.every((word, index) => word === right[index]);
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
