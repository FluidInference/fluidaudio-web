import {
  OPT_0084_CANDIDATE_SEAM_COMMIT,
  OPT_0084_DEFAULT_SEMANTIC_DRAW_COUNT,
  OPT_0084_EXPERIMENT_ID,
  OPT_0084_MINIMUM_NOMINAL_MILLISECONDS,
  OPT_0084_THERMAL_COMMAND,
  OPT_0084_THERMAL_POLL_MILLISECONDS,
  OPT_0084_THERMAL_SOURCE,
  validateOpt0084RunIdentity,
  type Opt0084RunIdentity,
  type Opt0084ThermalLaunch,
} from "./opt-0084-planner-fused-candidate-radix-sampling-contract.js";

export {
  OPT_0084_CANDIDATE_SEAM_COMMIT,
  OPT_0084_DEFAULT_SEMANTIC_DRAW_COUNT,
  OPT_0084_EXPERIMENT_ID,
  OPT_0084_MINIMUM_NOMINAL_MILLISECONDS,
  OPT_0084_THERMAL_COMMAND,
  OPT_0084_THERMAL_POLL_MILLISECONDS,
  OPT_0084_THERMAL_SOURCE,
  validateOpt0084RunIdentity,
};
export type { Opt0084RunIdentity, Opt0084ThermalLaunch };

export const OPT_0084_COMPOUND_RECEIPT_SCHEMA =
  "ace-opt-0084-planner-compact-head-fused-sampling-compound-v1" as const;
export const OPT_0084_COMPOUND_TIMING_ROUND_COUNT = 16 as const;
export const OPT_0084_COMPOUND_MINIMUM_PAIR_WINS = 14 as const;
export const OPT_0084_COMPOUND_MINIMUM_SPEEDUP = 1.15 as const;
export const OPT_0084_COMPOUND_MINIMUM_PROJECTED_SAVING_SECONDS = 40 as const;

export const OPT_0084_COMPOUND_POSITION_IDS = Object.freeze([
  "semantic-early",
  "semantic-middle",
  "semantic-late",
] as const);
export type Opt0084CompoundPositionId =
  (typeof OPT_0084_COMPOUND_POSITION_IDS)[number];
export type Opt0084CompoundArm = "A" | "B" | "C";

/**
 * B remains between the complete-token A/C observations. Reversing A/C in
 * exactly half the rounds balances cache-age and first/last-position effects;
 * B is the same-full-head causal sampler arm in every round.
 */
export const OPT_0084_COMPOUND_ARM_ORDERS: readonly (
  readonly [Opt0084CompoundArm, Opt0084CompoundArm, Opt0084CompoundArm]
)[] = Object.freeze([
  Object.freeze(["A", "B", "C"] as const),
  Object.freeze(["C", "B", "A"] as const),
  Object.freeze(["C", "B", "A"] as const),
  Object.freeze(["A", "B", "C"] as const),
  Object.freeze(["A", "B", "C"] as const),
  Object.freeze(["C", "B", "A"] as const),
  Object.freeze(["A", "B", "C"] as const),
  Object.freeze(["C", "B", "A"] as const),
  Object.freeze(["C", "B", "A"] as const),
  Object.freeze(["A", "B", "C"] as const),
  Object.freeze(["C", "B", "A"] as const),
  Object.freeze(["A", "B", "C"] as const),
  Object.freeze(["A", "B", "C"] as const),
  Object.freeze(["C", "B", "A"] as const),
  Object.freeze(["C", "B", "A"] as const),
  Object.freeze(["A", "B", "C"] as const),
]);

export interface Opt0084CompoundTimingSample {
  readonly roundIndex: number;
  readonly positionId: Opt0084CompoundPositionId;
  readonly arm: Opt0084CompoundArm;
  readonly armPosition: 0 | 1 | 2;
  readonly positionOrder: number;
  readonly cachedTokensBeforeAppend: number;
  readonly completeTokenWallMilliseconds: number;
  readonly modelWallMilliseconds: number;
  readonly samplingWallMilliseconds: number;
  readonly tokenId: number;
  readonly word: number;
  readonly drawIndex: number;
  readonly cursorEnd: number;
  readonly sameStateReplayExact: true;
  readonly cacheWriteStatusValidated: true;
}

export interface Opt0084CompoundSummary {
  readonly count: number;
  readonly minimum: number;
  readonly median: number;
  readonly maximum: number;
}

export interface Opt0084CompoundDecision {
  readonly passed: boolean;
  readonly pairWinCount: number;
  readonly requiredPairWinCount: typeof OPT_0084_COMPOUND_MINIMUM_PAIR_WINS;
  readonly aggregateCompleteTokenSpeedup: number;
  readonly requiredCompleteTokenSpeedup:
    typeof OPT_0084_COMPOUND_MINIMUM_SPEEDUP;
  readonly projected900TokenSavingSeconds: number;
  readonly meanPositionMedianSavingMilliseconds: number;
  readonly requiredProjectedSavingSeconds:
    typeof OPT_0084_COMPOUND_MINIMUM_PROJECTED_SAVING_SECONDS;
  readonly everyPositionCMedianBelowA: boolean;
  readonly everySampleSameStateExact: boolean;
  readonly armPositionCounts: Readonly<Record<
    Opt0084CompoundArm,
    readonly [number, number, number]
  >>;
  readonly aggregateRoundWallMilliseconds: Readonly<Record<
    Opt0084CompoundArm,
    Opt0084CompoundSummary
  >>;
  readonly positionTiming: Readonly<Record<
    Opt0084CompoundPositionId,
    Readonly<Record<Opt0084CompoundArm, Opt0084CompoundSummary>>
  >>;
}

/**
 * Test/worker cancellation sentinel. The real sampler API is entered; its
 * first logit-length access raises AbortError before a cursor can commit.
 */
export function createOpt0084CompoundAbortingLogitRow(
  length: number,
  controller: AbortController,
  message: string,
): ArrayLike<number> {
  if (!Number.isSafeInteger(length) || length <= 0) {
    throw new RangeError("OPT-0084 compound aborting row length is invalid");
  }
  const target: ArrayLike<number> = Object.freeze({ length });
  return new Proxy(target, {
    get(object, property, receiver) {
      if (property === "length") {
        controller.abort(new DOMException(message, "AbortError"));
        controller.signal.throwIfAborted();
      }
      return Reflect.get(object, property, receiver);
    },
  });
}

export function evaluateOpt0084CompoundTiming(
  samples: readonly Opt0084CompoundTimingSample[],
): Opt0084CompoundDecision {
  const expectedCount = OPT_0084_COMPOUND_TIMING_ROUND_COUNT *
    OPT_0084_COMPOUND_POSITION_IDS.length * 3;
  if (samples.length !== expectedCount) {
    throw new Error(
      `OPT-0084 compound requires ${expectedCount} samples, got ${samples.length}`,
    );
  }
  const roundWalls: Record<Opt0084CompoundArm, number[]> = {
    A: Array<number>(OPT_0084_COMPOUND_TIMING_ROUND_COUNT).fill(0),
    B: Array<number>(OPT_0084_COMPOUND_TIMING_ROUND_COUNT).fill(0),
    C: Array<number>(OPT_0084_COMPOUND_TIMING_ROUND_COUNT).fill(0),
  };
  const byPosition = Object.fromEntries(OPT_0084_COMPOUND_POSITION_IDS.map(
    (positionId) => [positionId, { A: [] as number[], B: [] as number[],
      C: [] as number[] }],
  )) as Record<Opt0084CompoundPositionId,
    Record<Opt0084CompoundArm, number[]>>;
  const armPositionCounts: Record<Opt0084CompoundArm, [number, number, number]> =
    { A: [0, 0, 0], B: [0, 0, 0], C: [0, 0, 0] };
  const seen = new Set<string>();
  let everySampleSameStateExact = true;

  for (const sample of samples) {
    if (
      !Number.isSafeInteger(sample.roundIndex) || sample.roundIndex < 0 ||
      sample.roundIndex >= OPT_0084_COMPOUND_TIMING_ROUND_COUNT ||
      !OPT_0084_COMPOUND_POSITION_IDS.includes(sample.positionId) ||
      !(["A", "B", "C"] as const).includes(sample.arm) ||
      !Number.isSafeInteger(sample.armPosition) || sample.armPosition < 0 ||
      sample.armPosition > 2 || !Number.isSafeInteger(sample.positionOrder) ||
      sample.positionOrder < 0 || sample.positionOrder > 2 ||
      OPT_0084_COMPOUND_POSITION_IDS[sample.positionOrder] !==
        sample.positionId ||
      !Number.isSafeInteger(sample.cachedTokensBeforeAppend) ||
      sample.cachedTokensBeforeAppend < 1 ||
      !Number.isFinite(sample.completeTokenWallMilliseconds) ||
      sample.completeTokenWallMilliseconds <= 0 ||
      !Number.isFinite(sample.modelWallMilliseconds) ||
      sample.modelWallMilliseconds <= 0 ||
      !Number.isFinite(sample.samplingWallMilliseconds) ||
      sample.samplingWallMilliseconds < 0 ||
      sample.samplingWallMilliseconds > sample.completeTokenWallMilliseconds ||
      Math.abs(sample.modelWallMilliseconds + sample.samplingWallMilliseconds -
        sample.completeTokenWallMilliseconds) > 1e-6 ||
      !Number.isSafeInteger(sample.tokenId) || sample.tokenId < 0 ||
      !Number.isSafeInteger(sample.word) || sample.word < 0 ||
      sample.word > 0xffff_ffff || !Number.isSafeInteger(sample.drawIndex) ||
      sample.drawIndex < 0 || sample.cursorEnd !== sample.drawIndex + 1 ||
      sample.sameStateReplayExact !== true ||
      sample.cacheWriteStatusValidated !== true
    ) {
      throw new Error("OPT-0084 compound timing sample is invalid");
    }
    const order = OPT_0084_COMPOUND_ARM_ORDERS[sample.roundIndex]!;
    if (order[sample.armPosition] !== sample.arm) {
      throw new Error("OPT-0084 compound timing sample violates arm order");
    }
    const key = `${sample.roundIndex}:${sample.positionId}:${sample.arm}`;
    if (seen.has(key)) throw new Error("OPT-0084 compound sample is duplicated");
    seen.add(key);
    roundWalls[sample.arm][sample.roundIndex] =
      roundWalls[sample.arm][sample.roundIndex]! +
        sample.completeTokenWallMilliseconds;
    byPosition[sample.positionId][sample.arm].push(
      sample.completeTokenWallMilliseconds,
    );
    armPositionCounts[sample.arm][sample.armPosition] =
      armPositionCounts[sample.arm][sample.armPosition]! + 1;
    everySampleSameStateExact &&= sample.sameStateReplayExact === true &&
      sample.cacheWriteStatusValidated === true;
  }

  const aggregateRoundWallMilliseconds = Object.freeze({
    A: summarize(roundWalls.A),
    B: summarize(roundWalls.B),
    C: summarize(roundWalls.C),
  });
  let pairWinCount = 0;
  for (let round = 0; round < OPT_0084_COMPOUND_TIMING_ROUND_COUNT; round += 1) {
    if (roundWalls.C[round]! < roundWalls.A[round]!) pairWinCount += 1;
  }
  const positionTiming = {} as Record<Opt0084CompoundPositionId,
    Record<Opt0084CompoundArm, Opt0084CompoundSummary>>;
  let everyPositionCMedianBelowA = true;
  for (const positionId of OPT_0084_COMPOUND_POSITION_IDS) {
    const values = byPosition[positionId];
    const timing = {
      A: summarize(values.A), B: summarize(values.B), C: summarize(values.C),
    };
    positionTiming[positionId] = timing;
    everyPositionCMedianBelowA &&= timing.C.median < timing.A.median;
  }
  const aggregateCompleteTokenSpeedup =
    aggregateRoundWallMilliseconds.A.median /
    aggregateRoundWallMilliseconds.C.median;
  const meanPositionMedianSavingMilliseconds =
    OPT_0084_COMPOUND_POSITION_IDS.reduce((sum, positionId) =>
      sum + positionTiming[positionId].A.median -
        positionTiming[positionId].C.median, 0) /
      OPT_0084_COMPOUND_POSITION_IDS.length;
  const projected900TokenSavingSeconds =
    meanPositionMedianSavingMilliseconds *
      OPT_0084_DEFAULT_SEMANTIC_DRAW_COUNT / 1_000;
  const passed = pairWinCount >= OPT_0084_COMPOUND_MINIMUM_PAIR_WINS &&
    aggregateCompleteTokenSpeedup >= OPT_0084_COMPOUND_MINIMUM_SPEEDUP &&
    projected900TokenSavingSeconds >=
      OPT_0084_COMPOUND_MINIMUM_PROJECTED_SAVING_SECONDS &&
    everyPositionCMedianBelowA && everySampleSameStateExact;
  return Object.freeze({
    passed,
    pairWinCount,
    requiredPairWinCount: OPT_0084_COMPOUND_MINIMUM_PAIR_WINS,
    aggregateCompleteTokenSpeedup,
    requiredCompleteTokenSpeedup: OPT_0084_COMPOUND_MINIMUM_SPEEDUP,
    projected900TokenSavingSeconds,
    meanPositionMedianSavingMilliseconds,
    requiredProjectedSavingSeconds:
      OPT_0084_COMPOUND_MINIMUM_PROJECTED_SAVING_SECONDS,
    everyPositionCMedianBelowA,
    everySampleSameStateExact,
    armPositionCounts: Object.freeze({
      A: Object.freeze([...armPositionCounts.A]) as readonly [number, number, number],
      B: Object.freeze([...armPositionCounts.B]) as readonly [number, number, number],
      C: Object.freeze([...armPositionCounts.C]) as readonly [number, number, number],
    }),
    aggregateRoundWallMilliseconds,
    positionTiming: Object.freeze(positionTiming),
  });
}

function summarize(values: readonly number[]): Opt0084CompoundSummary {
  if (values.length === 0 || values.some((value) =>
    !Number.isFinite(value) || value <= 0)) {
    throw new Error("OPT-0084 compound timing values are invalid");
  }
  return Object.freeze({
    count: values.length,
    minimum: Math.min(...values),
    median: median(values),
    maximum: Math.max(...values),
  });
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}
