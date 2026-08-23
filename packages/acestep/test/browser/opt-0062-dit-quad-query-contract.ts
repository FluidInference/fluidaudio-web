export const OPT_0062_RUNTIME_PROFILE =
  "opt-0062-fixed32-quad-query32-full-self-v1" as const;
export const OPT_0062_KERNEL_SET_ID =
  "opt-0062-query8-plus-quad-query32-full-self-v1" as const;
export const OPT_0062_KERNEL_ID =
  "opt-0062-fixed32-quad-query32-full-self-v1" as const;
export const OPT_0062_WGSL_SHA256 =
  "7b9af88e0f24f96da54dd525850da2432158fb4a7cdaccab1633b961f10911e6" as const;
export const OPT_0062_THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
export const OPT_0062_THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
export const OPT_0062_THERMAL_POLL_MILLISECONDS = 1_000 as const;
export const OPT_0062_MINIMUM_NOMINAL_MILLISECONDS = 30_000 as const;
export const OPT_0062_MAXIMUM_POLL_GAP_MILLISECONDS = 1_250 as const;
export const OPT_0062_MAXIMUM_GATE_HANDOFF_MILLISECONDS = 5_000 as const;
export const OPT_0062_LATENT_ELEMENTS = 288_000 as const;
export const OPT_0062_GRAPH_COMMAND_BUFFERS = 2_553 as const;
export const OPT_0062_TOTAL_COMMAND_BUFFERS = 2_554 as const;

export type Opt0062Direction = "forward" | "reverse";
export type Opt0062Arm = "query8" | "quad";

export interface Opt0062ThermalGate {
  readonly source: typeof OPT_0062_THERMAL_SOURCE;
  readonly command: typeof OPT_0062_THERMAL_COMMAND;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
}

export interface Opt0062ThermalTrace {
  readonly source: typeof OPT_0062_THERMAL_SOURCE;
  readonly command: typeof OPT_0062_THERMAL_COMMAND;
  readonly rawTraceSha256: string;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: number;
  readonly transitions: readonly Readonly<{
    readonly atEpochMilliseconds: number;
    readonly level: number;
  }>[];
}

export interface Opt0062TimingSample {
  readonly direction: Opt0062Direction;
  readonly order: 0 | 1;
  readonly arm: Opt0062Arm;
  readonly fullSelfMs: number;
  readonly sliceEvaluation0FullSelfMs: number;
  readonly graphWallMs: number;
  readonly ditStageWallMs: number;
  readonly commandDrainMs: number;
  readonly requestedIdleMs: number;
  readonly readbackMs: number;
  readonly residualMs: number;
  readonly familyMs: Readonly<Record<string, number>>;
}

export interface Opt0062BalancedSummary {
  readonly fixedOrder: readonly [
    "forward-query8",
    "forward-quad",
    "reverse-quad",
    "reverse-query8",
  ];
  readonly samples: readonly Opt0062TimingSample[];
  readonly forwardGraphImproved: boolean;
  readonly reverseGraphImproved: boolean;
  readonly forwardStageImproved: boolean;
  readonly reverseStageImproved: boolean;
  readonly forwardFullSelfImproved: boolean;
  readonly reverseFullSelfImproved: boolean;
  readonly forwardSliceImproved: boolean;
  readonly reverseSliceImproved: boolean;
  readonly aggregateFullSelfSpeedup: number;
  readonly medianSliceFullSelfSpeedup: number;
  readonly aggregateGraphSavingsMs: number;
  readonly nonAttentionRegressions: Readonly<Record<string, number>>;
  readonly maximumNonAttentionRegression: number;
  readonly passed: boolean;
}

export function requireOpt0062ThermalGate(
  value: Opt0062ThermalGate,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0062ThermalGate {
  const duration = value.completedAtEpochMilliseconds -
    value.startedAtEpochMilliseconds;
  if (
    value.source !== OPT_0062_THERMAL_SOURCE ||
    value.command !== OPT_0062_THERMAL_COMMAND ||
    !Number.isFinite(readyAtEpochMilliseconds) ||
    value.startedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    value.completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    nowEpochMilliseconds - value.completedAtEpochMilliseconds >
      OPT_0062_MAXIMUM_GATE_HANDOFF_MILLISECONDS ||
    duration < OPT_0062_MINIMUM_NOMINAL_MILLISECONDS ||
    !Number.isSafeInteger(value.observationCount) ||
    value.observationCount <
      Math.floor(duration / OPT_0062_THERMAL_POLL_MILLISECONDS) + 1 ||
    !Number.isFinite(value.maximumPollGapMilliseconds) ||
    value.maximumPollGapMilliseconds < 0 ||
    value.maximumPollGapMilliseconds >
      OPT_0062_MAXIMUM_POLL_GAP_MILLISECONDS ||
    value.nonNominalObservationCount !== 0
  ) throw new Error("OPT-0062 rejected the direction thermal gate");
  return Object.freeze({ ...value });
}

export function requireOpt0062ThermalTrace(
  value: Opt0062ThermalTrace,
  gate: Opt0062ThermalGate,
  cleanupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0062ThermalTrace {
  const duration = value.completedAtEpochMilliseconds -
    gate.startedAtEpochMilliseconds;
  if (
    value.source !== OPT_0062_THERMAL_SOURCE ||
    value.command !== OPT_0062_THERMAL_COMMAND ||
    !/^[0-9a-f]{64}$/u.test(value.rawTraceSha256) ||
    value.completedAtEpochMilliseconds < cleanupCompletedAtEpochMilliseconds ||
    value.completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    !Number.isSafeInteger(value.observationCount) ||
    value.observationCount <
      Math.floor(duration / OPT_0062_THERMAL_POLL_MILLISECONDS) + 1 ||
    value.observationCount < gate.observationCount ||
    !Number.isFinite(value.maximumPollGapMilliseconds) ||
    value.maximumPollGapMilliseconds < 0 ||
    value.maximumPollGapMilliseconds >
      OPT_0062_MAXIMUM_POLL_GAP_MILLISECONDS ||
    !Number.isSafeInteger(value.nonNominalObservationCount) ||
    value.nonNominalObservationCount < 0 ||
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
  ) throw new Error("OPT-0062 rejected the through-cleanup thermal trace");
  return Object.freeze({
    ...value,
    transitions: Object.freeze(value.transitions.map((item) =>
      Object.freeze({ ...item })
    )),
  });
}

export function exactOpt0062TrajectoryIdentity(
  left: readonly Float32Array[],
  right: readonly Float32Array[],
): boolean {
  if (left.length !== 8 || right.length !== 8) return false;
  return left.every((values, evaluation) => {
    const other = right[evaluation];
    if (
      other === undefined ||
      values.length !== OPT_0062_LATENT_ELEMENTS ||
      other.length !== values.length
    ) return false;
    const a = new Uint32Array(
      values.buffer,
      values.byteOffset,
      values.length,
    );
    const b = new Uint32Array(
      other.buffer,
      other.byteOffset,
      other.length,
    );
    return a.every((word, index) => word === b[index]);
  });
}

export function summarizeOpt0062BalancedGate(
  samples: readonly Opt0062TimingSample[],
): Opt0062BalancedSummary {
  const expected = [
    ["forward", 0, "query8"],
    ["forward", 1, "quad"],
    ["reverse", 0, "quad"],
    ["reverse", 1, "query8"],
  ] as const;
  if (
    samples.length !== expected.length ||
    samples.some((sample, index) => {
      const row = expected[index]!;
      return sample.direction !== row[0] || sample.order !== row[1] ||
        sample.arm !== row[2] || numericSampleFields(sample).some((value) =>
          !Number.isFinite(value) || value < 0
        );
    })
  ) throw new Error("OPT-0062 balanced sample inventory changed");
  const [forwardControl, forwardQuad, reverseQuad, reverseControl] = samples as
    readonly [
      Opt0062TimingSample,
      Opt0062TimingSample,
      Opt0062TimingSample,
      Opt0062TimingSample,
    ];
  const aggregateFullSelfSpeedup = ratio(
    forwardControl.fullSelfMs + reverseControl.fullSelfMs,
    forwardQuad.fullSelfMs + reverseQuad.fullSelfMs,
  );
  const sliceSpeedups = [
    ratio(
      forwardControl.sliceEvaluation0FullSelfMs,
      forwardQuad.sliceEvaluation0FullSelfMs,
    ),
    ratio(
      reverseControl.sliceEvaluation0FullSelfMs,
      reverseQuad.sliceEvaluation0FullSelfMs,
    ),
  ].sort((left, right) => left - right);
  const medianSliceFullSelfSpeedup =
    (sliceSpeedups[0]! + sliceSpeedups[1]!) / 2;
  const aggregateGraphSavingsMs =
    forwardControl.graphWallMs + reverseControl.graphWallMs -
    forwardQuad.graphWallMs - reverseQuad.graphWallMs;
  const familyNames = new Set([
    ...Object.keys(forwardControl.familyMs),
    ...Object.keys(forwardQuad.familyMs),
    ...Object.keys(reverseQuad.familyMs),
    ...Object.keys(reverseControl.familyMs),
  ]);
  familyNames.delete("self-full");
  const nonAttentionRegressions: Record<string, number> = Object.create(null) as
    Record<string, number>;
  for (const family of [...familyNames].sort()) {
    for (const [direction, control, quad] of [
      ["forward", forwardControl, forwardQuad],
      ["reverse", reverseControl, reverseQuad],
    ] as const) {
      const controlMs = control.familyMs[family] ?? 0;
      const quadMs = quad.familyMs[family] ?? 0;
      nonAttentionRegressions[`${direction}:${family}`] = controlMs === 0
        ? quadMs === 0 ? 0 : Number.POSITIVE_INFINITY
        : quadMs / controlMs - 1;
    }
  }
  const maximumNonAttentionRegression = Math.max(
    0,
    ...Object.values(nonAttentionRegressions),
  );
  const forwardGraphImproved =
    forwardQuad.graphWallMs < forwardControl.graphWallMs;
  const reverseGraphImproved =
    reverseQuad.graphWallMs < reverseControl.graphWallMs;
  const forwardStageImproved =
    forwardQuad.ditStageWallMs < forwardControl.ditStageWallMs;
  const reverseStageImproved =
    reverseQuad.ditStageWallMs < reverseControl.ditStageWallMs;
  const forwardFullSelfImproved =
    forwardQuad.fullSelfMs < forwardControl.fullSelfMs;
  const reverseFullSelfImproved =
    reverseQuad.fullSelfMs < reverseControl.fullSelfMs;
  const forwardSliceImproved =
    forwardQuad.sliceEvaluation0FullSelfMs <
      forwardControl.sliceEvaluation0FullSelfMs;
  const reverseSliceImproved =
    reverseQuad.sliceEvaluation0FullSelfMs <
      reverseControl.sliceEvaluation0FullSelfMs;
  const passed = forwardGraphImproved && reverseGraphImproved &&
    forwardStageImproved && reverseStageImproved &&
    forwardFullSelfImproved && reverseFullSelfImproved &&
    forwardSliceImproved && reverseSliceImproved && aggregateFullSelfSpeedup >= 1.30 &&
    medianSliceFullSelfSpeedup >= 1.30 && aggregateGraphSavingsMs >= 3_000 &&
    maximumNonAttentionRegression <= 0.02;
  return Object.freeze({
    fixedOrder: Object.freeze([
      "forward-query8",
      "forward-quad",
      "reverse-quad",
      "reverse-query8",
    ]) as Opt0062BalancedSummary["fixedOrder"],
    samples: Object.freeze([...samples]),
    forwardGraphImproved,
    reverseGraphImproved,
    forwardStageImproved,
    reverseStageImproved,
    forwardFullSelfImproved,
    reverseFullSelfImproved,
    forwardSliceImproved,
    reverseSliceImproved,
    aggregateFullSelfSpeedup,
    medianSliceFullSelfSpeedup,
    aggregateGraphSavingsMs,
    nonAttentionRegressions: Object.freeze(nonAttentionRegressions),
    maximumNonAttentionRegression,
    passed,
  });
}

function numericSampleFields(sample: Opt0062TimingSample): readonly number[] {
  return [
    sample.fullSelfMs,
    sample.sliceEvaluation0FullSelfMs,
    sample.graphWallMs,
    sample.ditStageWallMs,
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
