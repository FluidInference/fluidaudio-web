import type {
  AcePlannerOpt0085SchedulingDiagnostics,
  AcePlannerOpt0085SchedulingProfile,
} from "../../src/webgpu/planner-executor.js";

export const OPT_0085_SCHEMA =
  "ace-opt-0085-planner-depth2-completion-epochs-v1" as const;
export const OPT_0085_THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
export const OPT_0085_THERMAL_POLL_MILLISECONDS = 1_000;
export const OPT_0085_THERMAL_POLL_TOLERANCE_MILLISECONDS = 250;
export const OPT_0085_MINIMUM_NOMINAL_MILLISECONDS = 30_000;
export const OPT_0085_PAIR_COUNT_PER_PATH = 4;
export const OPT_0085_TOTAL_PAIR_COUNT = 16;
export const OPT_0085_REQUIRED_CANDIDATE_WINS = 14;
export const OPT_0085_REQUIRED_MEDIAN_SAVING_MILLISECONDS = 20;
export const OPT_0085_PROJECTION_DRAW_COUNT = 1_010;
export const OPT_0085_REQUIRED_PROJECTED_SAVING_SECONDS = 20;

export const OPT_0085_CONTROL_PROFILE = "depth1-epoch1" as const;
export const OPT_0085_CANDIDATE_PROFILE =
  "opt-0085-depth2-epoch4" as const;

export type Opt0085Arm = "control" | "candidate";
export type Opt0085PathId =
  | "cot-m1-middle-full"
  | "semantic-m2-middle-full"
  | "semantic-m2-middle-compact"
  | "semantic-m2-middle-forced-eos";

export const OPT_0085_PATH_IDS = Object.freeze([
  "cot-m1-middle-full",
  "semantic-m2-middle-full",
  "semantic-m2-middle-compact",
  "semantic-m2-middle-forced-eos",
] as const satisfies readonly Opt0085PathId[]);

/** Four balanced orders per path; across four paths each arm runs first eight times. */
export const OPT_0085_PAIR_ORDERS = Object.freeze([
  Object.freeze(["control", "candidate"] as const),
  Object.freeze(["candidate", "control"] as const),
  Object.freeze(["candidate", "control"] as const),
  Object.freeze(["control", "candidate"] as const),
]);

export interface Opt0085RunIdentity {
  readonly implementationCommit: string;
  readonly harnessCommit: string;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly gpuCoreCount: number;
  readonly memoryBytes: number;
}

export interface Opt0085ThermalTraceMetadata {
  readonly source: typeof OPT_0085_THERMAL_SOURCE;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: typeof OPT_0085_THERMAL_POLL_MILLISECONDS;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
}

export interface Opt0085PathTimingInput {
  readonly id: Opt0085PathId;
  readonly controlCompleteWallMilliseconds: readonly number[];
  readonly candidateCompleteWallMilliseconds: readonly number[];
}

export interface Opt0085TimingGateReceipt {
  readonly everyPathCandidateMedianBelowControl: boolean;
  readonly candidateWins: number;
  readonly requiredCandidateWins: typeof OPT_0085_REQUIRED_CANDIDATE_WINS;
  readonly aggregateControlMedianMilliseconds: number;
  readonly aggregateCandidateMedianMilliseconds: number;
  readonly aggregateMedianSavingMilliseconds: number;
  readonly requiredMedianSavingMilliseconds:
    typeof OPT_0085_REQUIRED_MEDIAN_SAVING_MILLISECONDS;
  readonly projected1010DrawSavingSeconds: number;
  readonly requiredProjectedSavingSeconds:
    typeof OPT_0085_REQUIRED_PROJECTED_SAVING_SECONDS;
  readonly passed: boolean;
}

export function opt0085ProfileForArm(
  arm: Opt0085Arm,
): AcePlannerOpt0085SchedulingProfile {
  return arm === "control"
    ? OPT_0085_CONTROL_PROFILE
    : OPT_0085_CANDIDATE_PROFILE;
}

export function validateOpt0085Topology(
  diagnostics: AcePlannerOpt0085SchedulingDiagnostics,
  expectedProfile: AcePlannerOpt0085SchedulingProfile,
  expectedTotalCommandBuffers: 33 | 34,
): void {
  const candidate = expectedProfile === OPT_0085_CANDIDATE_PROFILE;
  const expectedDrains = candidate
    ? Math.ceil(expectedTotalCommandBuffers / 4)
    : expectedTotalCommandBuffers;
  const expectedIdleTurns = candidate
    ? expectedDrains - 1
    : expectedTotalCommandBuffers;
  if (
    diagnostics.schema !== "ace-opt-0085-planner-scheduling-v1" ||
    diagnostics.schedulingProfile !== expectedProfile ||
    diagnostics.phaseKind !== "decode" ||
    diagnostics.totalCommandBuffers !== expectedTotalCommandBuffers ||
    diagnostics.commandBuffersSubmitted !== expectedTotalCommandBuffers ||
    diagnostics.completionFenceRequestedCount !== expectedTotalCommandBuffers ||
    diagnostics.completionFenceSettledCount !== expectedTotalCommandBuffers ||
    diagnostics.completionFenceRejectedCount !== 0 ||
    diagnostics.trueQueueDrainCount !== expectedDrains ||
    diagnostics.completionEpochCount !== expectedDrains ||
    diagnostics.cooperativeIdleTurns !== expectedIdleTurns ||
    diagnostics.requestedCooperativeIdleMs !== expectedIdleTurns ||
    diagnostics.maximumOutstandingCommandBuffers !== (candidate ? 2 : 1)
  ) {
    throw new Error(
      `OPT-0085 ${expectedProfile} ${expectedTotalCommandBuffers}-command ` +
        "topology changed",
    );
  }
}

export function evaluateOpt0085TimingGate(
  paths: readonly Opt0085PathTimingInput[],
): Opt0085TimingGateReceipt {
  if (
    paths.length !== OPT_0085_PATH_IDS.length ||
    paths.some((path, index) => path.id !== OPT_0085_PATH_IDS[index])
  ) throw new Error("OPT-0085 timing paths are incomplete or out of order");
  const control: number[] = [];
  const candidate: number[] = [];
  let candidateWins = 0;
  let everyPathCandidateMedianBelowControl = true;
  for (const path of paths) {
    validateTimingSamples(path.controlCompleteWallMilliseconds);
    validateTimingSamples(path.candidateCompleteWallMilliseconds);
    control.push(...path.controlCompleteWallMilliseconds);
    candidate.push(...path.candidateCompleteWallMilliseconds);
    everyPathCandidateMedianBelowControl &&=
      median(path.candidateCompleteWallMilliseconds) <
        median(path.controlCompleteWallMilliseconds);
    for (let index = 0; index < OPT_0085_PAIR_COUNT_PER_PATH; index += 1) {
      if (
        path.candidateCompleteWallMilliseconds[index]! <
          path.controlCompleteWallMilliseconds[index]!
      ) candidateWins += 1;
    }
  }
  if (
    control.length !== OPT_0085_TOTAL_PAIR_COUNT ||
    candidate.length !== OPT_0085_TOTAL_PAIR_COUNT
  ) throw new Error("OPT-0085 aggregate timing pair count changed");
  const aggregateControlMedianMilliseconds = median(control);
  const aggregateCandidateMedianMilliseconds = median(candidate);
  const aggregateMedianSavingMilliseconds =
    aggregateControlMedianMilliseconds - aggregateCandidateMedianMilliseconds;
  const projected1010DrawSavingSeconds =
    aggregateMedianSavingMilliseconds * OPT_0085_PROJECTION_DRAW_COUNT / 1_000;
  const passed = everyPathCandidateMedianBelowControl &&
    candidateWins >= OPT_0085_REQUIRED_CANDIDATE_WINS &&
    aggregateMedianSavingMilliseconds >=
      OPT_0085_REQUIRED_MEDIAN_SAVING_MILLISECONDS &&
    projected1010DrawSavingSeconds >=
      OPT_0085_REQUIRED_PROJECTED_SAVING_SECONDS;
  return Object.freeze({
    everyPathCandidateMedianBelowControl,
    candidateWins,
    requiredCandidateWins: OPT_0085_REQUIRED_CANDIDATE_WINS,
    aggregateControlMedianMilliseconds,
    aggregateCandidateMedianMilliseconds,
    aggregateMedianSavingMilliseconds,
    requiredMedianSavingMilliseconds:
      OPT_0085_REQUIRED_MEDIAN_SAVING_MILLISECONDS,
    projected1010DrawSavingSeconds,
    requiredProjectedSavingSeconds:
      OPT_0085_REQUIRED_PROJECTED_SAVING_SECONDS,
    passed,
  });
}

export function validateOpt0085RunIdentity(
  identity: unknown,
): Opt0085RunIdentity {
  if (typeof identity !== "object" || identity === null) {
    throw new Error("OPT-0085 requires a run identity");
  }
  const candidate = identity as Readonly<Record<string, unknown>>;
  const requiredString = (name: string): string => {
    const value = candidate[name];
    if (
      typeof value !== "string" ||
      value.trim() === "" ||
      value !== value.trim()
    ) throw new Error(`OPT-0085 requires run identity ${name}`);
    return value;
  };
  const requiredPositiveInteger = (name: string): number => {
    const value = candidate[name];
    if (!Number.isSafeInteger(value) || (value as number) <= 0) {
      throw new Error(`OPT-0085 run identity ${name} must be positive`);
    }
    return value as number;
  };
  const implementationCommit = requiredString("implementationCommit");
  const harnessCommit = requiredString("harnessCommit");
  if (!/^[0-9a-f]{40}$/.test(implementationCommit)) {
    throw new Error("OPT-0085 implementationCommit must be 40 lowercase hex");
  }
  if (!/^[0-9a-f]{40}$/.test(harnessCommit)) {
    throw new Error("OPT-0085 harnessCommit must be 40 lowercase hex");
  }
  return Object.freeze({
    implementationCommit,
    harnessCommit,
    machineModel: requiredString("machineModel"),
    osVersion: requiredString("osVersion"),
    osBuild: requiredString("osBuild"),
    browserVersion: requiredString("browserVersion"),
    gpuCoreCount: requiredPositiveInteger("gpuCoreCount"),
    memoryBytes: requiredPositiveInteger("memoryBytes"),
  });
}

export function validateOpt0085PreGate(
  thermal: Opt0085ThermalTraceMetadata,
  warmupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds = Date.now(),
): void {
  validateThermalShape(thermal, nowEpochMilliseconds);
  if (
    thermal.startedAtEpochMilliseconds < warmupCompletedAtEpochMilliseconds ||
    thermal.durationMilliseconds < OPT_0085_MINIMUM_NOMINAL_MILLISECONDS
  ) throw new Error("OPT-0085 requires 30 nominal seconds after warmup");
}

export function validateOpt0085ThroughCleanupTrace(
  thermal: Opt0085ThermalTraceMetadata,
  preGate: Opt0085ThermalTraceMetadata,
  cleanupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds = Date.now(),
): void {
  validateThermalShape(thermal, nowEpochMilliseconds);
  if (
    thermal.startedAtEpochMilliseconds !== preGate.startedAtEpochMilliseconds ||
    thermal.completedAtEpochMilliseconds <
      cleanupCompletedAtEpochMilliseconds ||
    thermal.completedAtEpochMilliseconds < preGate.completedAtEpochMilliseconds ||
    thermal.observationCount < preGate.observationCount ||
    thermal.maximumPollGapMilliseconds < preGate.maximumPollGapMilliseconds
  ) throw new Error("OPT-0085 final thermal trace does not cover cleanup");
}

export function median(values: readonly number[]): number {
  if (values.length === 0) throw new Error("OPT-0085 cannot summarize no values");
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function validateTimingSamples(values: readonly number[]): void {
  if (
    values.length !== OPT_0085_PAIR_COUNT_PER_PATH ||
    values.some((value) => !Number.isFinite(value) || value < 0)
  ) throw new Error("OPT-0085 timing samples are invalid");
}

function validateThermalShape(
  thermal: Opt0085ThermalTraceMetadata,
  nowEpochMilliseconds: number,
): void {
  if (
    thermal.source !== OPT_0085_THERMAL_SOURCE ||
    thermal.completedAtEpochMilliseconds < thermal.startedAtEpochMilliseconds ||
    thermal.completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    thermal.durationMilliseconds !==
      thermal.completedAtEpochMilliseconds - thermal.startedAtEpochMilliseconds ||
    !Number.isSafeInteger(thermal.observationCount) ||
    thermal.observationCount <
      Math.floor(thermal.durationMilliseconds /
        OPT_0085_THERMAL_POLL_MILLISECONDS) + 1 ||
    thermal.pollMilliseconds !== OPT_0085_THERMAL_POLL_MILLISECONDS ||
    !Number.isFinite(thermal.maximumPollGapMilliseconds) ||
    thermal.maximumPollGapMilliseconds < 0 ||
    thermal.maximumPollGapMilliseconds >
      OPT_0085_THERMAL_POLL_MILLISECONDS +
        OPT_0085_THERMAL_POLL_TOLERANCE_MILLISECONDS ||
    thermal.nonNominalObservationCount !== 0
  ) throw new Error("OPT-0085 thermal trace metadata is invalid");
}
