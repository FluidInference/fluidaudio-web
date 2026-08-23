import type { Opt0018RunIdentity } from
  "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0080_VAE_ARM_ORDER,
  OPT_0080_VAE_EXPERIMENT_ID,
  OPT_0080_VAE_FIXTURE_SHA256,
  OPT_0080_VAE_OUTPUT_ELEMENTS,
  OPT_0080_VAE_SCHEMA,
  OPT_0080_VAE_WAVEFORM_SHA256,
  requireOpt0080VaeCancellation,
  requireOpt0080VaeThermalGate,
  requireOpt0080VaeThermalTrace,
  summarizeOpt0080VaePerformance,
  type Opt0080VaeArmId,
  type Opt0080VaeCancellationEvidence,
  type Opt0080VaeThermalGate,
  type Opt0080VaeThermalTrace,
  type Opt0080VaeTimingSample,
} from "./opt-0080-vae-depth2-completion-epochs-contract.js";

export interface Opt0080VaeCorrectnessReceipt {
  readonly controlOutputSha256: string;
  readonly candidateOutputSha256: string;
  readonly postCancellationProbeSha256: string;
  readonly comparedU32WordCount: 8_885_760;
  readonly controlCandidateU32MismatchCount: 0;
  readonly controlProbeU32MismatchCount: 0;
  readonly controlNonFiniteCount: 0;
  readonly controlNonzeroCount: number;
  readonly boundedComparisonCanariesPassed: true;
  readonly gpuGuardCanariesPassed: true;
  readonly gpuGuardExpectedSha256:
    "183dead51e555d79ec074ad8acfe08c5f4dffce8392ccadc46bb9da2d5aa413d";
  readonly gpuGuardedBufferCount: 6;
  readonly gpuGuardedRegionCount: 12;
  readonly gpuGuardCheckedBytesPerExecution: 3_072;
  readonly topologyExact: true;
  readonly excludedFromTiming: true;
}

export interface Opt0080VaeTimedArmResult {
  readonly armId: Opt0080VaeArmId;
  readonly order: 0 | 1 | 2 | 3;
  readonly readyAtEpochMilliseconds: number;
  readonly launchedAtEpochMilliseconds: number;
  readonly settledAtEpochMilliseconds: number;
  readonly sample: Opt0080VaeTimingSample;
  readonly gate: Opt0080VaeThermalGate;
  readonly trace: Opt0080VaeThermalTrace;
  readonly guardEvidence: Readonly<Record<string, unknown>>;
}

export interface Opt0080VaeResultInput {
  readonly runIdentity: Opt0018RunIdentity;
  readonly package: Readonly<Record<string, unknown>>;
  readonly device: Readonly<Record<string, unknown>>;
  readonly dispatchTopology: unknown;
  readonly memory: unknown;
  readonly correctness: Opt0080VaeCorrectnessReceipt;
  readonly preflightEvidence: Readonly<Record<string, unknown>>;
  readonly cancellation: Opt0080VaeCancellationEvidence;
  readonly arms: readonly Opt0080VaeTimedArmResult[];
  readonly lifecycle: Readonly<Record<string, unknown>>;
  readonly runtimeEvents: readonly Readonly<Record<string, unknown>>[];
  readonly rejectedSetupAttempts: readonly Readonly<Record<string, unknown>>[];
}

export function buildOpt0080VaeResult(
  input: Opt0080VaeResultInput,
): Readonly<Record<string, unknown>> {
  const correctness = requireCorrectness(input.correctness);
  const cancellation = requireOpt0080VaeCancellation(input.cancellation);
  const arms = requireTimedArms(input.arms, correctness.controlOutputSha256);
  const performance = summarizeOpt0080VaePerformance(
    arms.map((arm) => arm.sample),
  );
  const thermalStartPassed = arms.every((arm) =>
    arm.gate.nonNominalObservationCount === 0
  );
  const throughSettlementThermalTransitions = Object.freeze(
    Object.fromEntries(arms.map((arm) => [arm.armId, Object.freeze({
      nonNominalObservationCount: arm.trace.nonNominalObservationCount,
      transitions: arm.trace.transitions,
    })])),
  );
  // As in the parent OPT-0080 DiT gate, a nominal start is authoritative.
  // Later transitions remain visible in the continuous trace without silently
  // rewriting an otherwise valid paired comparison.
  const thermalAttributionPassed = thermalStartPassed;
  const lifecyclePassed = input.lifecycle["passed"] === true;
  const runtimePassed = input.runtimeEvents.length === 0;
  const passed = performance.passed && thermalAttributionPassed &&
    lifecyclePassed && runtimePassed;
  const status = !thermalAttributionPassed ||
      performance.classification === "inconclusive"
    ? "inconclusive"
    : passed ? "passed" : "failed";
  return Object.freeze({
    schema: OPT_0080_VAE_SCHEMA,
    experimentId: OPT_0080_VAE_EXPERIMENT_ID,
    status,
    passed,
    identity: Object.freeze({
      run: input.runIdentity,
      fixtureSha256: OPT_0080_VAE_FIXTURE_SHA256,
      outputSha256: correctness.controlOutputSha256,
      package: input.package,
      device: input.device,
      runtimeProfile:
        "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1",
      physicalPackageIdentity: "OPT-0066-revision7",
    }),
    geometry: Object.freeze({
      latentFrames: 2_314,
      maximumWindowFrames: 2_378,
      overlapFrames: 64,
      quantaPerCommandBuffer: 64,
      decoderCommandBuffers: 555,
      readbackCommandBuffers: 1,
      completionFencesPerArm: 556,
      controlTrueDrains: 556,
      controlIdleTurns: 555,
      candidateTrueDrains: 139,
      candidateIdleTurns: 138,
      candidateMaximumOutstandingCommandBuffers: 2,
      projectedProductionWindowCount: 2,
    }),
    memory: input.memory,
    dispatchTopology: input.dispatchTopology,
    correctness,
    preflightEvidence: input.preflightEvidence,
    cancellation,
    performance,
    thermalStartPassed,
    thermalAttributionPassed,
    throughSettlementThermalTransitions,
    arms: Object.freeze(Object.fromEntries(arms.map((arm) => [
      arm.armId,
      Object.freeze({
        gate: arm.gate,
        readyAtEpochMilliseconds: arm.readyAtEpochMilliseconds,
        launchedAtEpochMilliseconds: arm.launchedAtEpochMilliseconds,
        settledAtEpochMilliseconds: arm.settledAtEpochMilliseconds,
        trace: arm.trace,
        heartbeat: arm.sample.heartbeat,
        sample: arm.sample,
        guardEvidence: arm.guardEvidence,
      }),
    ]))),
    lifecycle: input.lifecycle,
    runtimeEvents: Object.freeze([...input.runtimeEvents]),
    rejectedSetupAttempts: Object.freeze([...input.rejectedSetupAttempts]),
    scope: Object.freeze({
      samePersistentDeviceBackendAndPackageOwner: true,
      productionDefaultChanged: false,
      packageBytesChanged: false,
      modelMathChanged: false,
      commandBufferContentsChanged: false,
      commandBufferCoalescing: false,
      ditSelectionRetainedRegardlessOfVaeDecision: true,
      plannerExecuted: false,
      fullProductTimingClaim: false,
      listeningClaim: false,
      underOneMinuteClaim: false,
    }),
    decision: passed
      ? "positive-authorize-production-vae-scheduler-selection"
      : status === "inconclusive"
        ? "inconclusive-retain-depth1-vae-production-scheduler"
        : "negative-retain-depth1-vae-production-scheduler",
  });
}

function requireCorrectness(
  value: Opt0080VaeCorrectnessReceipt,
): Opt0080VaeCorrectnessReceipt {
  if (
    value.controlOutputSha256 !== OPT_0080_VAE_WAVEFORM_SHA256 ||
    value.candidateOutputSha256 !== value.controlOutputSha256 ||
    value.postCancellationProbeSha256 !== value.controlOutputSha256 ||
    value.comparedU32WordCount !== OPT_0080_VAE_OUTPUT_ELEMENTS ||
    value.controlCandidateU32MismatchCount !== 0 ||
    value.controlProbeU32MismatchCount !== 0 ||
    value.controlNonFiniteCount !== 0 || value.controlNonzeroCount <= 0 ||
    value.boundedComparisonCanariesPassed !== true ||
    value.gpuGuardCanariesPassed !== true ||
    value.gpuGuardExpectedSha256 !==
      "183dead51e555d79ec074ad8acfe08c5f4dffce8392ccadc46bb9da2d5aa413d" ||
    value.gpuGuardedBufferCount !== 6 ||
    value.gpuGuardedRegionCount !== 12 ||
    value.gpuGuardCheckedBytesPerExecution !== 3_072 ||
    value.topologyExact !== true || value.excludedFromTiming !== true
  ) throw new Error("OPT-0080 C2314 VAE correctness receipt changed");
  return Object.freeze({ ...value });
}

function requireTimedArms(
  values: readonly Opt0080VaeTimedArmResult[],
  outputSha256: string,
): readonly Opt0080VaeTimedArmResult[] {
  if (
    values.length !== OPT_0080_VAE_ARM_ORDER.length ||
    values.some((arm, index) => {
      const expected = OPT_0080_VAE_ARM_ORDER[index]!;
      return arm.armId !== expected.armId || arm.order !== expected.order ||
        arm.sample.armId !== expected.armId ||
        arm.sample.order !== expected.order ||
        arm.sample.schedulingProfile !== expected.schedulingProfile ||
        arm.sample.outputSha256 !== outputSha256 ||
        arm.guardEvidence["passed"] !== true ||
        arm.guardEvidence["sha256"] !==
          "183dead51e555d79ec074ad8acfe08c5f4dffce8392ccadc46bb9da2d5aa413d" ||
        arm.launchedAtEpochMilliseconds < arm.readyAtEpochMilliseconds ||
        arm.settledAtEpochMilliseconds <= arm.readyAtEpochMilliseconds ||
        requireWithoutThrow(() => requireOpt0080VaeThermalGate(
          arm.gate,
          arm.readyAtEpochMilliseconds,
          arm.launchedAtEpochMilliseconds,
        )) === false ||
        requireWithoutThrow(() => requireOpt0080VaeThermalTrace(
          arm.trace,
          arm.gate,
          arm.settledAtEpochMilliseconds,
          arm.trace.completedAtEpochMilliseconds,
        )) === false;
    }) || new Set(values.map((arm) => arm.trace.rawTraceSha256)).size !==
      OPT_0080_VAE_ARM_ORDER.length
  ) throw new Error("OPT-0080 C2314 VAE timed arm provenance changed");
  return Object.freeze([...values]);
}

function requireWithoutThrow(run: () => unknown): boolean {
  try {
    run();
    return true;
  } catch {
    return false;
  }
}
