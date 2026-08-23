export const OPT_0067_RUNTIME_PROFILE =
  "opt-0062-fixed32-quad-query32-full-self-v1" as const;
export const OPT_0067_KERNEL_SET_ID =
  "opt-0062-query8-plus-quad-query32-full-self-v1" as const;
export const OPT_0067_KERNEL_ID =
  "opt-0062-fixed32-quad-query32-full-self-v1" as const;
export const OPT_0067_WGSL_SHA256 =
  "7b9af88e0f24f96da54dd525850da2432158fb4a7cdaccab1633b961f10911e6" as const;
export const OPT_0067_EVALUATION0_SHA256 =
  "d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286" as const;
export const OPT_0067_THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
export const OPT_0067_THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
export const OPT_0067_THERMAL_POLL_MILLISECONDS = 1_000 as const;
export const OPT_0067_MINIMUM_NOMINAL_MILLISECONDS = 30_000 as const;
export const OPT_0067_MAXIMUM_POLL_GAP_MILLISECONDS = 1_250 as const;
export const OPT_0067_MAXIMUM_GATE_HANDOFF_MILLISECONDS = 5_000 as const;
export const OPT_0067_RESULT_ELEMENTS = 288_000 as const;
export const OPT_0067_GRAPH_COMMAND_BUFFERS = 341 as const;
export const OPT_0067_TOTAL_COMMAND_BUFFERS = 342 as const;

export type Opt0067Owner = "query8" | "quad";
export type Opt0067ArmId = "A1" | "B1" | "B2" | "A2";

export const OPT_0067_ARM_ORDER = Object.freeze([
  Object.freeze({ armId: "A1" as const, order: 0 as const, owner: "query8" as const }),
  Object.freeze({ armId: "B1" as const, order: 1 as const, owner: "quad" as const }),
  Object.freeze({ armId: "B2" as const, order: 2 as const, owner: "quad" as const }),
  Object.freeze({ armId: "A2" as const, order: 3 as const, owner: "query8" as const }),
]);

export interface Opt0067ThermalObservation {
  readonly atEpochMilliseconds: number;
  readonly level: number;
  /** Exact trimmed stdout from the accepted notifyutil command. */
  readonly rawValue: string;
}

export interface Opt0067ThermalGate {
  readonly source: typeof OPT_0067_THERMAL_SOURCE;
  readonly command: typeof OPT_0067_THERMAL_COMMAND;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly observations: readonly Opt0067ThermalObservation[];
}

export interface Opt0067ThermalTrace {
  readonly source: typeof OPT_0067_THERMAL_SOURCE;
  readonly command: typeof OPT_0067_THERMAL_COMMAND;
  readonly rawTraceSha256: string;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: number;
  readonly observations: readonly Opt0067ThermalObservation[];
  readonly transitions: readonly Readonly<{
    readonly atEpochMilliseconds: number;
    readonly level: number;
  }>[];
}

export interface Opt0067TimingSample {
  readonly armId: Opt0067ArmId;
  readonly order: 0 | 1 | 2 | 3;
  readonly owner: Opt0067Owner;
  readonly fullSelfMs: number;
  readonly evaluationWallMs: number;
  readonly nonFullSelfEvaluationWallMs: number;
  readonly graphWallMs: number;
  readonly commandDrainMs: number;
  readonly requestedIdleMs: number;
  readonly readbackMs: number;
  readonly residualMs: number;
  readonly familyMs: Readonly<Record<string, number>>;
}

export interface Opt0067PerformanceSummary {
  readonly fixedOrder: readonly ["A1-query8", "B1-quad", "B2-quad", "A2-query8"];
  readonly samples: readonly Opt0067TimingSample[];
  readonly forwardFullSelfImproved: boolean;
  readonly reverseFullSelfImproved: boolean;
  readonly forwardEvaluationImproved: boolean;
  readonly reverseEvaluationImproved: boolean;
  readonly aggregateFullSelfSpeedup: number;
  readonly projectedEightEvaluationSavingMs: number;
  readonly nonFullSelfRegressions: Readonly<Record<string, number>>;
  readonly nonFullSelfAbsoluteDeltasMs: Readonly<Record<string, number>>;
  readonly maximumNonFullSelfRegression: number;
  readonly passed: boolean;
}

export function requireOpt0067ThermalGate(
  value: Opt0067ThermalGate,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0067ThermalGate {
  const duration = value.completedAtEpochMilliseconds -
    value.startedAtEpochMilliseconds;
  const observation = inspectObservations(
    value.observations,
    value.startedAtEpochMilliseconds,
    value.completedAtEpochMilliseconds,
  );
  if (
    value.source !== OPT_0067_THERMAL_SOURCE ||
    value.command !== OPT_0067_THERMAL_COMMAND ||
    !Number.isFinite(readyAtEpochMilliseconds) ||
    value.startedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    value.completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    nowEpochMilliseconds - value.completedAtEpochMilliseconds >
      OPT_0067_MAXIMUM_GATE_HANDOFF_MILLISECONDS ||
    duration < OPT_0067_MINIMUM_NOMINAL_MILLISECONDS ||
    !Number.isSafeInteger(value.observationCount) ||
    value.observationCount <
      Math.floor(duration / OPT_0067_THERMAL_POLL_MILLISECONDS) + 1 ||
    !Number.isFinite(value.maximumPollGapMilliseconds) ||
    value.maximumPollGapMilliseconds < 0 ||
    value.maximumPollGapMilliseconds >
      OPT_0067_MAXIMUM_POLL_GAP_MILLISECONDS ||
    value.maximumPollGapMilliseconds !== observation.maximumPollGapMilliseconds ||
    value.nonNominalObservationCount !== 0 ||
    observation.nonNominalObservationCount !== 0 ||
    value.observations.length !== value.observationCount
  ) throw new Error("OPT-0067 rejected the arm thermal gate");
  return Object.freeze({
    ...value,
    observations: freezeObservations(value.observations),
  });
}

export function requireOpt0067ThermalTrace(
  value: Opt0067ThermalTrace,
  gate: Opt0067ThermalGate,
  cleanupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0067ThermalTrace {
  const duration = value.completedAtEpochMilliseconds -
    gate.startedAtEpochMilliseconds;
  const observation = inspectObservations(
    value.observations,
    gate.startedAtEpochMilliseconds,
    value.completedAtEpochMilliseconds,
  );
  if (
    value.source !== OPT_0067_THERMAL_SOURCE ||
    value.command !== OPT_0067_THERMAL_COMMAND ||
    !/^[0-9a-f]{64}$/u.test(value.rawTraceSha256) ||
    value.completedAtEpochMilliseconds < cleanupCompletedAtEpochMilliseconds ||
    value.completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    !Number.isSafeInteger(value.observationCount) ||
    value.observationCount <
      Math.floor(duration / OPT_0067_THERMAL_POLL_MILLISECONDS) + 1 ||
    value.observationCount < gate.observationCount ||
    !Number.isFinite(value.maximumPollGapMilliseconds) ||
    value.maximumPollGapMilliseconds < 0 ||
    value.maximumPollGapMilliseconds >
      OPT_0067_MAXIMUM_POLL_GAP_MILLISECONDS ||
    value.maximumPollGapMilliseconds !== observation.maximumPollGapMilliseconds ||
    !Number.isSafeInteger(value.nonNominalObservationCount) ||
    value.nonNominalObservationCount < 0 ||
    value.nonNominalObservationCount !== observation.nonNominalObservationCount ||
    value.observations.length !== value.observationCount ||
    gate.observations.some((item, index) =>
      !sameObservation(item, value.observations[index])
    ) ||
    value.transitions.some(({ atEpochMilliseconds, level }) =>
      !Number.isFinite(atEpochMilliseconds) ||
      atEpochMilliseconds < gate.completedAtEpochMilliseconds ||
      atEpochMilliseconds > value.completedAtEpochMilliseconds ||
      !Number.isSafeInteger(level) || level < 0
    ) ||
    (value.nonNominalObservationCount === 0 &&
      value.transitions.some(({ level }) => level !== 0)) ||
    (value.nonNominalObservationCount > 0 &&
      value.transitions.every(({ level }) => level === 0))
  ) throw new Error("OPT-0067 rejected the through-cleanup thermal trace");
  return Object.freeze({
    ...value,
    observations: freezeObservations(value.observations),
    transitions: Object.freeze(value.transitions.map((item) =>
      Object.freeze({ ...item })
    )),
  });
}

export function exactOpt0067ResultIdentity(
  left: Float32Array,
  right: Float32Array,
): boolean {
  if (
    left.length !== OPT_0067_RESULT_ELEMENTS ||
    right.length !== OPT_0067_RESULT_ELEMENTS
  ) return false;
  const a = new Uint32Array(left.buffer, left.byteOffset, left.length);
  const b = new Uint32Array(right.buffer, right.byteOffset, right.length);
  return a.every((word, index) => word === b[index]);
}

export function summarizeOpt0067Performance(
  samples: readonly Opt0067TimingSample[],
): Opt0067PerformanceSummary {
  if (
    samples.length !== OPT_0067_ARM_ORDER.length ||
    samples.some((sample, index) => {
      const expected = OPT_0067_ARM_ORDER[index]!;
      return sample.armId !== expected.armId || sample.order !== expected.order ||
        sample.owner !== expected.owner ||
        numericFields(sample).some((value) => !Number.isFinite(value) || value < 0);
    })
  ) throw new Error("OPT-0067 ABBA sample inventory changed");
  const [a1, b1, b2, a2] = samples as readonly [
    Opt0067TimingSample,
    Opt0067TimingSample,
    Opt0067TimingSample,
    Opt0067TimingSample,
  ];
  const forwardFullSelfImproved = b1.fullSelfMs < a1.fullSelfMs;
  const reverseFullSelfImproved = b2.fullSelfMs < a2.fullSelfMs;
  const forwardEvaluationImproved = b1.evaluationWallMs < a1.evaluationWallMs;
  const reverseEvaluationImproved = b2.evaluationWallMs < a2.evaluationWallMs;
  const aggregateFullSelfSpeedup = ratio(
    a1.fullSelfMs + a2.fullSelfMs,
    b1.fullSelfMs + b2.fullSelfMs,
  );
  const projectedEightEvaluationSavingMs = 8 *
    (((a1.evaluationWallMs - b1.evaluationWallMs) +
      (a2.evaluationWallMs - b2.evaluationWallMs)) / 2);
  const nonFullSelfRegressions: Record<string, number> = Object.create(null) as
    Record<string, number>;
  const nonFullSelfAbsoluteDeltasMs: Record<string, number> =
    Object.create(null) as Record<string, number>;
  const familyNames = new Set([
    ...Object.keys(a1.familyMs),
    ...Object.keys(b1.familyMs),
    ...Object.keys(b2.familyMs),
    ...Object.keys(a2.familyMs),
  ]);
  familyNames.delete("self-full");
  for (const [direction, control, quad] of [
    ["forward", a1, b1],
    ["reverse", a2, b2],
  ] as const) {
    for (const family of [...familyNames].sort()) {
      const controlMs = control.familyMs[family] ?? 0;
      const quadMs = quad.familyMs[family] ?? 0;
      const key = `${direction}:${family}`;
      nonFullSelfRegressions[key] = regression(controlMs, quadMs);
      nonFullSelfAbsoluteDeltasMs[key] = quadMs - controlMs;
    }
    const aggregateKey = `${direction}:aggregate-non-full-self`;
    nonFullSelfRegressions[aggregateKey] = regression(
      control.nonFullSelfEvaluationWallMs,
      quad.nonFullSelfEvaluationWallMs,
    );
    nonFullSelfAbsoluteDeltasMs[aggregateKey] =
      quad.nonFullSelfEvaluationWallMs - control.nonFullSelfEvaluationWallMs;
  }
  const maximumNonFullSelfRegression = Math.max(
    0,
    ...Object.values(nonFullSelfRegressions),
  );
  const passed = forwardFullSelfImproved && reverseFullSelfImproved &&
    forwardEvaluationImproved && reverseEvaluationImproved &&
    aggregateFullSelfSpeedup >= 1.30 &&
    projectedEightEvaluationSavingMs >= 3_000 &&
    maximumNonFullSelfRegression <= 0.02;
  return Object.freeze({
    fixedOrder: Object.freeze([
      "A1-query8",
      "B1-quad",
      "B2-quad",
      "A2-query8",
    ]) as Opt0067PerformanceSummary["fixedOrder"],
    samples: Object.freeze([...samples]),
    forwardFullSelfImproved,
    reverseFullSelfImproved,
    forwardEvaluationImproved,
    reverseEvaluationImproved,
    aggregateFullSelfSpeedup,
    projectedEightEvaluationSavingMs,
    nonFullSelfRegressions: Object.freeze(nonFullSelfRegressions),
    nonFullSelfAbsoluteDeltasMs: Object.freeze(nonFullSelfAbsoluteDeltasMs),
    maximumNonFullSelfRegression,
    passed,
  });
}

function numericFields(sample: Opt0067TimingSample): readonly number[] {
  return [
    sample.fullSelfMs,
    sample.evaluationWallMs,
    sample.nonFullSelfEvaluationWallMs,
    sample.graphWallMs,
    sample.commandDrainMs,
    sample.requestedIdleMs,
    sample.readbackMs,
    sample.residualMs,
    ...Object.values(sample.familyMs),
  ];
}

function ratio(control: number, candidate: number): number {
  return candidate === 0
    ? control === 0 ? 1 : Number.POSITIVE_INFINITY
    : control / candidate;
}

function regression(control: number, candidate: number): number {
  return control === 0
    ? candidate === 0 ? 0 : Number.POSITIVE_INFINITY
    : candidate / control - 1;
}

function inspectObservations(
  observations: readonly Opt0067ThermalObservation[],
  startedAtEpochMilliseconds: number,
  completedAtEpochMilliseconds: number,
): Readonly<{
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: number;
}> {
  if (
    observations.length === 0 ||
    observations[0]?.atEpochMilliseconds !== startedAtEpochMilliseconds ||
    observations.at(-1)?.atEpochMilliseconds !== completedAtEpochMilliseconds
  ) throw new Error("OPT-0067 thermal observation boundary changed");
  let maximumPollGapMilliseconds = 0;
  let nonNominalObservationCount = 0;
  for (let index = 0; index < observations.length; index += 1) {
    const item = observations[index]!;
    const previous = observations[index - 1];
    if (
      !Number.isSafeInteger(item.atEpochMilliseconds) ||
      !Number.isSafeInteger(item.level) || item.level < 0 ||
      item.rawValue !== String(item.level) ||
      (previous !== undefined &&
        item.atEpochMilliseconds <= previous.atEpochMilliseconds)
    ) throw new Error("OPT-0067 thermal observation is invalid");
    if (previous !== undefined) {
      maximumPollGapMilliseconds = Math.max(
        maximumPollGapMilliseconds,
        item.atEpochMilliseconds - previous.atEpochMilliseconds,
      );
    }
    if (item.level !== 0) nonNominalObservationCount += 1;
  }
  return Object.freeze({
    maximumPollGapMilliseconds,
    nonNominalObservationCount,
  });
}

function sameObservation(
  left: Opt0067ThermalObservation,
  right: Opt0067ThermalObservation | undefined,
): boolean {
  return right !== undefined &&
    left.atEpochMilliseconds === right.atEpochMilliseconds &&
    left.level === right.level && left.rawValue === right.rawValue;
}

function freezeObservations(
  observations: readonly Opt0067ThermalObservation[],
): readonly Opt0067ThermalObservation[] {
  return Object.freeze(observations.map((item) => Object.freeze({ ...item })));
}
