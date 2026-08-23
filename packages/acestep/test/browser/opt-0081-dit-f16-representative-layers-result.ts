import {
  validateOpt0018RunIdentity,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0081_REPRESENTATIVE_ARENA_SAVING_BYTES,
  OPT_0081_REPRESENTATIVE_CANDIDATE_ARENA_BYTES,
  OPT_0081_REPRESENTATIVE_CONDITION_SHA256,
  OPT_0081_REPRESENTATIVE_CONTEXT_SHA256,
  OPT_0081_REPRESENTATIVE_CONTROL_ARENA_BYTES,
  OPT_0081_REPRESENTATIVE_DENSE_MANIFEST_SHA256,
  OPT_0081_REPRESENTATIVE_EXPERIMENT,
  OPT_0081_REPRESENTATIVE_GRAPH_PROFILE_COMMIT,
  OPT_0081_REPRESENTATIVE_MAIN_MANIFEST_SHA256,
  OPT_0081_REPRESENTATIVE_LYRIC_TOKEN_SHA256,
  OPT_0081_REPRESENTATIVE_RECEIPT_SCHEMA,
  OPT_0081_REPRESENTATIVE_REGISTRATION_COMMIT,
  OPT_0081_REPRESENTATIVE_REQUEST_SHA256,
  OPT_0081_REPRESENTATIVE_SETUP_CLARIFICATION_COMMIT,
  OPT_0081_REPRESENTATIVE_STORAGE_PROFILE,
  OPT_0081_REPRESENTATIVE_TEXT_TOKEN_SHA256,
  inspectOpt0081RepresentativeCorrectness,
  opt0081RepresentativeThermalTimingPassed,
  requireOpt0081RepresentativeCancellation,
  requireOpt0081RepresentativeHeartbeat,
  requireOpt0081RepresentativeLifecycle,
  requireOpt0081RepresentativeThermalCompletion,
  requireOpt0081RepresentativeThermalLaunch,
  summarizeOpt0081RepresentativeTiming,
  type Opt0081RepresentativeCancellationEvidence,
  type Opt0081RepresentativeCorrectness,
  type Opt0081RepresentativeDisposition,
  type Opt0081RepresentativeHeartbeat,
  type Opt0081RepresentativeLifecycleEvidence,
  type Opt0081RepresentativeThermalCompletion,
  type Opt0081RepresentativeThermalLaunch,
  type Opt0081RepresentativeTimingSample,
  type Opt0081RepresentativeTimingSummary,
} from "./opt-0081-dit-f16-representative-layers-contract.js";

export interface Opt0081RepresentativeIdentityEvidence {
  readonly run: Opt0018RunIdentity;
  readonly registrationCommit:
    typeof OPT_0081_REPRESENTATIVE_REGISTRATION_COMMIT;
  readonly setupClarificationCommit:
    typeof OPT_0081_REPRESENTATIVE_SETUP_CLARIFICATION_COMMIT;
  readonly graphProfileCommit:
    typeof OPT_0081_REPRESENTATIVE_GRAPH_PROFILE_COMMIT;
  readonly requestSha256: typeof OPT_0081_REPRESENTATIVE_REQUEST_SHA256;
  readonly mainManifestSha256:
    typeof OPT_0081_REPRESENTATIVE_MAIN_MANIFEST_SHA256;
  readonly denseManifestSha256:
    typeof OPT_0081_REPRESENTATIVE_DENSE_MANIFEST_SHA256;
  readonly storageProfile: typeof OPT_0081_REPRESENTATIVE_STORAGE_PROFILE;
  readonly requiredConditioningAuthority:
    Opt0081RepresentativeConditioningAuthorityEvidence;
  readonly shape: Readonly<{
    readonly batch: 1;
    readonly latentFrames: 4_500;
    readonly tokens: 2_250;
    readonly conditionTokens: 98;
    readonly evaluation: 0;
  }>;
  readonly conditionerExecutedOnlyInExcludedSetup: true;
  readonly plannerExecuted: false;
  readonly vaeExecuted: false;
  readonly samplerExecuted: false;
}

export interface Opt0081RepresentativeConditioningAuthorityEvidence {
  readonly schema: "ace-opt-0081-representative-conditioning-authority-v1";
  readonly textTokenCount: number;
  readonly textTokenSha256: string;
  readonly lyricTokenCount: number;
  readonly lyricTokenSha256: string;
  readonly conditionTokens: number;
  readonly conditionElementCount: number;
  readonly conditionSha256: string;
  readonly contextElementCount: number;
  readonly contextSha256: string;
}

export interface Opt0081RepresentativeConditioningEvidence {
  readonly setup: Opt0081RepresentativeConditioningAuthorityEvidence;
  readonly main: Opt0081RepresentativeConditioningAuthorityEvidence;
  readonly setupAndMainExact: true;
  readonly capturedBeforeReady: true;
}

export interface Opt0081RepresentativeArenaEvidence {
  readonly controlArenaBytes: number;
  readonly candidateArenaBytes: number;
  readonly arenaSavingBytes: number;
  readonly controlRoleCount: number;
  readonly candidateRoleCount: number;
  readonly controlSlotCount: number;
  readonly candidateSlotCount: number;
  readonly candidateSelectedDenseInputSlotIndices: readonly number[];
  readonly controlLargestArenaBindingBytes: number;
  readonly candidateLargestArenaBindingBytes: number;
  readonly largestArenaBindingIncreaseBytes: number;
  readonly castOrCopyBufferCount: number;
  readonly passed: boolean;
}

export interface Opt0081RepresentativePreparationEvidence {
  readonly correctnessCompletedBeforeReady: true;
  readonly precomputeCompletedBeforeReady: true;
  readonly precomputeCommandBufferCountPerArm: 25;
  readonly warmupArmOrder: readonly ["A", "B"];
  readonly warmupSelectedSliceOccurrencesPerArm: 1;
  readonly everyWarmupTerminallyDrained: true;
  readonly conditioningAuthorityCapturedBeforeReady: true;
  readonly setupAndMainConditioningAuthorityExact: true;
  readonly readyAtEpochMilliseconds: number;
  readonly passed: boolean;
}

export interface Opt0081RepresentativeRunEvidence {
  readonly identity: Opt0081RepresentativeIdentityEvidence;
  readonly conditioning: Opt0081RepresentativeConditioningEvidence;
  readonly arena: Opt0081RepresentativeArenaEvidence;
  readonly preparation: Opt0081RepresentativePreparationEvidence;
  readonly correctness: Opt0081RepresentativeCorrectness;
  readonly cancellation: Opt0081RepresentativeCancellationEvidence;
  readonly timingSamples: readonly Opt0081RepresentativeTimingSample[];
  readonly cleanup: Opt0081RepresentativeLifecycleEvidence;
  readonly cleanupCompletedAtEpochMilliseconds: number;
  readonly thermalLaunch: Opt0081RepresentativeThermalLaunch;
}

export interface Opt0081RepresentativeReceipt {
  readonly schema: typeof OPT_0081_REPRESENTATIVE_RECEIPT_SCHEMA;
  readonly experiment: typeof OPT_0081_REPRESENTATIVE_EXPERIMENT;
  readonly status: "completed";
  readonly identity: Opt0081RepresentativeIdentityEvidence;
  readonly conditioning: Opt0081RepresentativeConditioningEvidence;
  readonly arena: Opt0081RepresentativeArenaEvidence;
  readonly preparation: Opt0081RepresentativePreparationEvidence;
  readonly correctness: Opt0081RepresentativeCorrectness;
  readonly cancellation: Opt0081RepresentativeCancellationEvidence;
  readonly timing: Opt0081RepresentativeTimingSummary;
  readonly heartbeat: Opt0081RepresentativeHeartbeat;
  readonly thermal: Readonly<{
    readonly launch: Opt0081RepresentativeThermalLaunch;
    readonly completion: Opt0081RepresentativeThermalCompletion;
    readonly passed: boolean;
  }>;
  readonly cleanup: Opt0081RepresentativeLifecycleEvidence;
  readonly cleanupCompletedAtEpochMilliseconds: number;
  readonly decision: Readonly<{
    readonly disposition: Opt0081RepresentativeDisposition;
    readonly passed: boolean;
    readonly selectedDiagnosticArm: "B" | null;
    readonly completeEvaluationFollowUpAuthorized: boolean;
    readonly productionIntegrationAuthorized: false;
    readonly packageChangeAuthorized: false;
    readonly fullTrajectoryAuthorized: false;
    readonly productGateAuthorized: false;
    readonly unchangedTimingRetryPerformed: false;
  }>;
}

export function createOpt0081RepresentativeIdentity(
  run: Opt0018RunIdentity,
): Opt0081RepresentativeIdentityEvidence {
  const validatedRun = validateOpt0018RunIdentity(run);
  return Object.freeze({
    run: validatedRun,
    registrationCommit: OPT_0081_REPRESENTATIVE_REGISTRATION_COMMIT,
    setupClarificationCommit:
      OPT_0081_REPRESENTATIVE_SETUP_CLARIFICATION_COMMIT,
    graphProfileCommit: OPT_0081_REPRESENTATIVE_GRAPH_PROFILE_COMMIT,
    requestSha256: OPT_0081_REPRESENTATIVE_REQUEST_SHA256,
    mainManifestSha256: OPT_0081_REPRESENTATIVE_MAIN_MANIFEST_SHA256,
    denseManifestSha256: OPT_0081_REPRESENTATIVE_DENSE_MANIFEST_SHA256,
    storageProfile: OPT_0081_REPRESENTATIVE_STORAGE_PROFILE,
    requiredConditioningAuthority: canonicalConditioningAuthority(),
    shape: Object.freeze({ batch: 1, latentFrames: 4_500, tokens: 2_250,
      conditionTokens: 98, evaluation: 0 }),
    conditionerExecutedOnlyInExcludedSetup: true,
    plannerExecuted: false,
    vaeExecuted: false,
    samplerExecuted: false,
  });
}

export function createOpt0081RepresentativeConditioningEvidence(
  setup: Opt0081RepresentativeConditioningAuthorityEvidence,
  main: Opt0081RepresentativeConditioningAuthorityEvidence,
): Opt0081RepresentativeConditioningEvidence {
  const acceptedSetup = requireOpt0081RepresentativeConditioningAuthority(
    setup,
  );
  const acceptedMain = requireOpt0081RepresentativeConditioningAuthority(main);
  if (conditioningAuthorityKey(acceptedSetup) !==
      conditioningAuthorityKey(acceptedMain)) {
    throw new Error("OPT-0081 setup/main conditioning authority changed");
  }
  return Object.freeze({
    setup: acceptedSetup,
    main: acceptedMain,
    setupAndMainExact: true,
    capturedBeforeReady: true,
  });
}

export function requireOpt0081RepresentativeConditioningAuthority(
  value: Opt0081RepresentativeConditioningAuthorityEvidence,
): Opt0081RepresentativeConditioningAuthorityEvidence {
  if (conditioningAuthorityKey(value) !==
      conditioningAuthorityKey(canonicalConditioningAuthority())) {
    throw new Error("OPT-0081 canonical conditioning authority changed");
  }
  return value;
}

export function finalizeOpt0081RepresentativeReceipt(
  evidence: Opt0081RepresentativeRunEvidence,
  heartbeat: Opt0081RepresentativeHeartbeat,
  thermalCompletion: Opt0081RepresentativeThermalCompletion,
): Opt0081RepresentativeReceipt {
  const timing = summarizeOpt0081RepresentativeTiming(evidence.timingSamples);
  const acceptedHeartbeat = requireOpt0081RepresentativeHeartbeat(
    heartbeat,
    evidence.cleanupCompletedAtEpochMilliseconds,
  );
  requireOpt0081RepresentativeThermalCompletion(
    thermalCompletion,
    evidence.thermalLaunch,
    evidence.cleanupCompletedAtEpochMilliseconds,
  );
  const thermalPassed = opt0081RepresentativeThermalTimingPassed(
    thermalCompletion,
    evidence.thermalLaunch,
    evidence.cleanupCompletedAtEpochMilliseconds,
  );
  const candidate = Object.freeze({
    schema: OPT_0081_REPRESENTATIVE_RECEIPT_SCHEMA,
    experiment: OPT_0081_REPRESENTATIVE_EXPERIMENT,
    status: "completed" as const,
    identity: evidence.identity,
    conditioning: evidence.conditioning,
    arena: evidence.arena,
    preparation: evidence.preparation,
    correctness: evidence.correctness,
    cancellation: evidence.cancellation,
    timing,
    heartbeat: acceptedHeartbeat,
    thermal: Object.freeze({
      launch: evidence.thermalLaunch,
      completion: thermalCompletion,
      passed: thermalPassed,
    }),
    cleanup: evidence.cleanup,
    cleanupCompletedAtEpochMilliseconds:
      evidence.cleanupCompletedAtEpochMilliseconds,
  });
  const disposition = classifyOpt0081RepresentativeDisposition(candidate);
  const passed = disposition ===
    "positive-B-representative-layers-authorize-complete-evaluation";
  return Object.freeze({
    ...candidate,
    decision: Object.freeze({
      disposition,
      passed,
      selectedDiagnosticArm: passed ? "B" : null,
      completeEvaluationFollowUpAuthorized: passed,
      productionIntegrationAuthorized: false,
      packageChangeAuthorized: false,
      fullTrajectoryAuthorized: false,
      productGateAuthorized: false,
      unchangedTimingRetryPerformed: false,
    }),
  });
}

export function classifyOpt0081RepresentativeDisposition(
  receipt: Readonly<Record<string, unknown>>,
): Opt0081RepresentativeDisposition {
  const identity = receipt["identity"] as
    Opt0081RepresentativeIdentityEvidence | undefined;
  if (identity === undefined || !identityPassed(identity)) {
    return "inconclusive-invalid-correctness-topology-or-lifecycle-evidence";
  }
  const conditioning = receipt["conditioning"] as
    Opt0081RepresentativeConditioningEvidence | undefined;
  if (conditioning === undefined || !conditioningPassed(conditioning)) {
    return "inconclusive-invalid-correctness-topology-or-lifecycle-evidence";
  }
  const correctness = receipt["correctness"] as
    Opt0081RepresentativeCorrectness | undefined;
  if (correctness === undefined) {
    return "inconclusive-invalid-correctness-topology-or-lifecycle-evidence";
  }
  const inspection = inspectOpt0081RepresentativeCorrectness(correctness);
  if (
    inspection.structurallyValid && inspection.observedRawMismatch &&
    correctness.uncapturedGpuErrorCount === 0 &&
    correctness.validationErrorCount === 0 &&
    correctness.deviceLossCount === 0
  ) {
    return "negative-stop-observed-raw-bit-correctness-mismatch";
  }
  const arena = receipt["arena"] as Opt0081RepresentativeArenaEvidence |
    undefined;
  const preparation = receipt["preparation"] as
    Opt0081RepresentativePreparationEvidence | undefined;
  const cancellation = receipt["cancellation"] as
    Opt0081RepresentativeCancellationEvidence | undefined;
  const cleanup = receipt["cleanup"] as
    Opt0081RepresentativeLifecycleEvidence | undefined;
  const timing = receipt["timing"] as Opt0081RepresentativeTimingSummary |
    undefined;
  if (
    !inspection.structurallyValid || !inspection.passed ||
    arena === undefined || !arenaPassed(arena) ||
    preparation === undefined || !preparationPassed(preparation) ||
    cancellation === undefined || requireFailed(() =>
      requireOpt0081RepresentativeCancellation(cancellation)) ||
    cleanup === undefined || requireFailed(() =>
      requireOpt0081RepresentativeLifecycle(cleanup)) ||
    timing === undefined || !timing.topologyIdenticalAcrossArms ||
    !timing.wallBoundaryConsistent
  ) {
    return "inconclusive-invalid-correctness-topology-or-lifecycle-evidence";
  }
  const thermal = receipt["thermal"] as Readonly<Record<string, unknown>> |
    undefined;
  const heartbeat = receipt["heartbeat"] as Opt0081RepresentativeHeartbeat |
    undefined;
  const cleanupCompletedAtEpochMilliseconds = Number(
    receipt["cleanupCompletedAtEpochMilliseconds"],
  );
  const thermalLaunch = thermal?.["launch"] as
    Opt0081RepresentativeThermalLaunch | undefined;
  const thermalCompletion = thermal?.["completion"] as
    Opt0081RepresentativeThermalCompletion | undefined;
  const thermalStructurallyValid = thermalLaunch !== undefined &&
    thermalCompletion !== undefined &&
    !requireFailed(() => requireOpt0081RepresentativeThermalLaunch(
      thermalLaunch,
      preparation.readyAtEpochMilliseconds,
      thermalLaunch.gateCompletedAtEpochMilliseconds +
        thermalLaunch.launchDelayMilliseconds,
    )) &&
    !requireFailed(() => requireOpt0081RepresentativeThermalCompletion(
      thermalCompletion,
      thermalLaunch,
      cleanupCompletedAtEpochMilliseconds,
    )) && thermal?.["passed"] === opt0081RepresentativeThermalTimingPassed(
      thermalCompletion,
      thermalLaunch,
      cleanupCompletedAtEpochMilliseconds,
    );
  if (
    !thermalStructurallyValid || thermal?.["passed"] !== true ||
    heartbeat === undefined ||
    !Number.isSafeInteger(cleanupCompletedAtEpochMilliseconds) ||
    requireFailed(() => requireOpt0081RepresentativeHeartbeat(
      heartbeat,
      cleanupCompletedAtEpochMilliseconds,
    ))
  ) return "inconclusive-invalid-thermal-or-heartbeat-provenance";
  return timing.passed
    ? "positive-B-representative-layers-authorize-complete-evaluation"
    : "inconclusive-directional-or-material-wall-evidence";
}

function identityPassed(value: Opt0081RepresentativeIdentityEvidence): boolean {
  const runValid = !requireFailed(() => validateOpt0018RunIdentity(value.run));
  return runValid &&
    value.registrationCommit === OPT_0081_REPRESENTATIVE_REGISTRATION_COMMIT &&
    value.setupClarificationCommit ===
      OPT_0081_REPRESENTATIVE_SETUP_CLARIFICATION_COMMIT &&
    value.graphProfileCommit === OPT_0081_REPRESENTATIVE_GRAPH_PROFILE_COMMIT &&
    value.requestSha256 === OPT_0081_REPRESENTATIVE_REQUEST_SHA256 &&
    value.mainManifestSha256 ===
      OPT_0081_REPRESENTATIVE_MAIN_MANIFEST_SHA256 &&
    value.denseManifestSha256 ===
      OPT_0081_REPRESENTATIVE_DENSE_MANIFEST_SHA256 &&
    value.storageProfile === OPT_0081_REPRESENTATIVE_STORAGE_PROFILE &&
    conditioningAuthorityKey(value.requiredConditioningAuthority) ===
      conditioningAuthorityKey(canonicalConditioningAuthority()) &&
    value.shape.batch === 1 && value.shape.latentFrames === 4_500 &&
    value.shape.tokens === 2_250 && value.shape.conditionTokens === 98 &&
    value.shape.evaluation === 0 &&
    value.conditionerExecutedOnlyInExcludedSetup === true &&
    value.plannerExecuted === false && value.vaeExecuted === false &&
    value.samplerExecuted === false;
}

function arenaPassed(value: Opt0081RepresentativeArenaEvidence): boolean {
  const passed = value.controlArenaBytes ===
      OPT_0081_REPRESENTATIVE_CONTROL_ARENA_BYTES &&
    value.candidateArenaBytes ===
      OPT_0081_REPRESENTATIVE_CANDIDATE_ARENA_BYTES &&
    value.arenaSavingBytes === OPT_0081_REPRESENTATIVE_ARENA_SAVING_BYTES &&
    value.controlArenaBytes - value.candidateArenaBytes ===
      value.arenaSavingBytes && value.controlRoleCount === 123 &&
    value.candidateRoleCount === 123 && value.controlSlotCount === 98 &&
    value.candidateSlotCount === 98 &&
    JSON.stringify(value.candidateSelectedDenseInputSlotIndices) ===
      "[61,69,72,77,81,84]" &&
    value.controlLargestArenaBindingBytes > 0 &&
    value.candidateLargestArenaBindingBytes > 0 &&
    value.candidateLargestArenaBindingBytes <=
      value.controlLargestArenaBindingBytes &&
    value.largestArenaBindingIncreaseBytes === 0 &&
    value.castOrCopyBufferCount === 0;
  return passed && value.passed === passed;
}

function preparationPassed(
  value: Opt0081RepresentativePreparationEvidence,
): boolean {
  const passed = value.correctnessCompletedBeforeReady === true &&
    value.precomputeCompletedBeforeReady === true &&
    value.precomputeCommandBufferCountPerArm === 25 &&
    value.warmupArmOrder[0] === "A" && value.warmupArmOrder[1] === "B" &&
    value.warmupSelectedSliceOccurrencesPerArm === 1 &&
    value.everyWarmupTerminallyDrained === true &&
    value.conditioningAuthorityCapturedBeforeReady === true &&
    value.setupAndMainConditioningAuthorityExact === true &&
    Number.isSafeInteger(value.readyAtEpochMilliseconds) &&
    value.readyAtEpochMilliseconds > 0;
  return passed && value.passed === passed;
}

function conditioningPassed(
  value: Opt0081RepresentativeConditioningEvidence,
): boolean {
  return value.setupAndMainExact === true && value.capturedBeforeReady === true &&
    !requireFailed(() => requireOpt0081RepresentativeConditioningAuthority(
      value.setup,
    )) &&
    !requireFailed(() => requireOpt0081RepresentativeConditioningAuthority(
      value.main,
    )) && conditioningAuthorityKey(value.setup) ===
      conditioningAuthorityKey(value.main);
}

function canonicalConditioningAuthority():
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

function conditioningAuthorityKey(
  value: Opt0081RepresentativeConditioningAuthorityEvidence,
): string {
  return JSON.stringify([
    value.schema,
    value.textTokenCount,
    value.textTokenSha256,
    value.lyricTokenCount,
    value.lyricTokenSha256,
    value.conditionTokens,
    value.conditionElementCount,
    value.conditionSha256,
    value.contextElementCount,
    value.contextSha256,
  ]);
}

function requireFailed(run: () => void): boolean {
  try {
    run();
    return false;
  } catch {
    return true;
  }
}
