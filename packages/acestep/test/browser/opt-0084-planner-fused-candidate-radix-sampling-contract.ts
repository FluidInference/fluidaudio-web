export const OPT_0084_EXPERIMENT_ID = "OPT-0084" as const;
export const OPT_0084_RECEIPT_SCHEMA =
  "ace-opt-0084-planner-fused-candidate-radix-sampling-v1" as const;
export const OPT_0084_CANDIDATE_SEAM_COMMIT =
  "245b5fe3347c370390eff990aa1ed45cb2b869ba" as const;
export const OPT_0084_THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
export const OPT_0084_THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
export const OPT_0084_THERMAL_POLL_MILLISECONDS = 1_000 as const;
export const OPT_0084_MINIMUM_NOMINAL_MILLISECONDS = 30_000 as const;
export const OPT_0084_TIMING_ROUND_COUNT = 16 as const;
export const OPT_0084_MINIMUM_PAIR_WINS = 14 as const;
export const OPT_0084_MINIMUM_SAMPLER_SPEEDUP = 1.5 as const;
export const OPT_0084_MINIMUM_PROJECTED_SAVING_SECONDS = 10 as const;
export const OPT_0084_DEFAULT_SEMANTIC_DRAW_COUNT = 900 as const;

export const OPT_0084_STATE_IDS = Object.freeze([
  "semantic-early",
  "semantic-middle",
  "semantic-late",
  "cot-singleton",
  "cot-small",
  "cot-caption",
  "cot-all",
] as const);

export type Opt0084StateId = (typeof OPT_0084_STATE_IDS)[number];
export type Opt0084Arm = "A" | "B";

export interface Opt0084RunIdentity {
  readonly harnessCommit: string;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly gpuCoreCount: number;
  readonly memoryBytes: number;
}

export const OPT_0084_DENSE_STATE_IDS: readonly Opt0084StateId[] =
  Object.freeze([
    "semantic-early",
    "semantic-middle",
    "semantic-late",
    "cot-caption",
    "cot-all",
  ]);

export const OPT_0084_SEMANTIC_STATE_IDS: readonly Opt0084StateId[] =
  Object.freeze([
    "semantic-early",
    "semantic-middle",
    "semantic-late",
  ]);

export const OPT_0084_PAIR_ORDERS: readonly (readonly [Opt0084Arm, Opt0084Arm])[] =
  Object.freeze([
    Object.freeze(["A", "B"] as const),
    Object.freeze(["B", "A"] as const),
    Object.freeze(["B", "A"] as const),
    Object.freeze(["A", "B"] as const),
    Object.freeze(["A", "B"] as const),
    Object.freeze(["B", "A"] as const),
    Object.freeze(["A", "B"] as const),
    Object.freeze(["B", "A"] as const),
    Object.freeze(["B", "A"] as const),
    Object.freeze(["A", "B"] as const),
    Object.freeze(["B", "A"] as const),
    Object.freeze(["A", "B"] as const),
    Object.freeze(["A", "B"] as const),
    Object.freeze(["B", "A"] as const),
    Object.freeze(["B", "A"] as const),
    Object.freeze(["A", "B"] as const),
  ]);

export interface Opt0084ThermalLaunch {
  readonly source: typeof OPT_0084_THERMAL_SOURCE;
  readonly command: typeof OPT_0084_THERMAL_COMMAND;
  readonly traceStartedAtEpochMilliseconds: number;
  readonly gateStartedAtEpochMilliseconds: number;
  readonly gateCompletedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: typeof OPT_0084_THERMAL_POLL_MILLISECONDS;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly missingObservationCount: 0;
  readonly readyToGateDelayMilliseconds: number;
  readonly launchDelayMilliseconds: number;
}

export interface Opt0084TimingSample {
  readonly roundIndex: number;
  readonly stateId: Opt0084StateId;
  readonly arm: Opt0084Arm;
  readonly armPosition: 0 | 1;
  readonly statePosition: number;
  readonly wallMilliseconds: number;
  readonly tokenId: number;
  readonly word: number;
  readonly positiveCandidateCount: number;
}

export interface Opt0084Summary {
  readonly count: number;
  readonly minimum: number;
  readonly median: number;
  readonly maximum: number;
}

export interface Opt0084TimingDecision {
  readonly passed: boolean;
  readonly pairWinCount: number;
  readonly requiredPairWinCount: typeof OPT_0084_MINIMUM_PAIR_WINS;
  readonly aggregateSamplerSpeedup: number;
  readonly requiredSamplerSpeedup: typeof OPT_0084_MINIMUM_SAMPLER_SPEEDUP;
  readonly projectedDefaultSemanticSavingSeconds: number;
  readonly requiredProjectedSavingSeconds:
    typeof OPT_0084_MINIMUM_PROJECTED_SAVING_SECONDS;
  readonly noRegressingDenseStateMedian: boolean;
  readonly everyPairedSampleExact: boolean;
  readonly armPositionCounts: Readonly<Record<Opt0084Arm, readonly [number, number]>>;
  readonly aggregateRoundWallMilliseconds: Readonly<{
    readonly A: Opt0084Summary;
    readonly B: Opt0084Summary;
  }>;
  readonly stateTiming: Readonly<Record<Opt0084StateId, Readonly<{
    readonly A: Opt0084Summary;
    readonly B: Opt0084Summary;
    readonly medianSpeedup: number;
    readonly medianSavingMilliseconds: number;
    readonly denseGateState: boolean;
  }>>>;
}

/** Independently applied at the page and worker structured-clone boundaries. */
export function validateOpt0084RunIdentity(
  value: unknown,
): Opt0084RunIdentity {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("OPT-0084 run identity must be an object");
  }
  const record = value as Record<string, unknown>;
  const expectedKeys = [
    "browserVersion",
    "gpuCoreCount",
    "harnessCommit",
    "machineModel",
    "memoryBytes",
    "osBuild",
    "osVersion",
  ] as const;
  const actualKeys = Object.keys(record).sort();
  if (
    actualKeys.length !== expectedKeys.length ||
    actualKeys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new TypeError("OPT-0084 run identity keys are incomplete or unknown");
  }
  const harnessCommit = requireIdentityString(record, "harnessCommit");
  if (!/^[0-9a-f]{40}$/.test(harnessCommit)) {
    throw new TypeError("OPT-0084 harnessCommit must be a full lowercase commit");
  }
  const gpuCoreCount = requireIdentityInteger(record, "gpuCoreCount");
  const memoryBytes = requireIdentityInteger(record, "memoryBytes");
  if (gpuCoreCount < 1 || gpuCoreCount > 256) {
    throw new RangeError("OPT-0084 gpuCoreCount is outside the physical range");
  }
  if (memoryBytes < 4 * 1024 ** 3 || memoryBytes > Number.MAX_SAFE_INTEGER) {
    throw new RangeError("OPT-0084 memoryBytes is outside the physical range");
  }
  return Object.freeze({
    harnessCommit,
    machineModel: requireIdentityString(record, "machineModel"),
    osVersion: requireIdentityString(record, "osVersion"),
    osBuild: requireIdentityString(record, "osBuild"),
    browserVersion: requireIdentityString(record, "browserVersion"),
    gpuCoreCount,
    memoryBytes,
  });
}

/** Pure, source-testable evaluator for the registered browser timing gate. */
export function evaluateOpt0084Timing(
  samples: readonly Opt0084TimingSample[],
): Opt0084TimingDecision {
  const expectedSamples = OPT_0084_TIMING_ROUND_COUNT *
    OPT_0084_STATE_IDS.length * 2;
  if (samples.length !== expectedSamples) {
    throw new Error(
      `OPT-0084 requires ${expectedSamples} timing samples, got ${samples.length}`,
    );
  }
  const aggregate = {
    A: Array<number>(OPT_0084_TIMING_ROUND_COUNT).fill(0),
    B: Array<number>(OPT_0084_TIMING_ROUND_COUNT).fill(0),
  };
  const stateValues = new Map<Opt0084StateId, {
    readonly A: number[];
    readonly B: number[];
  }>(OPT_0084_STATE_IDS.map((stateId) => [stateId, { A: [], B: [] }]));
  const paired = new Map<string, Partial<Record<Opt0084Arm, Opt0084TimingSample>>>();
  const armPositionCounts: Record<Opt0084Arm, [number, number]> = {
    A: [0, 0],
    B: [0, 0],
  };

  for (const sample of samples) {
    if (
      !Number.isSafeInteger(sample.roundIndex) || sample.roundIndex < 0 ||
      sample.roundIndex >= OPT_0084_TIMING_ROUND_COUNT ||
      !OPT_0084_STATE_IDS.includes(sample.stateId) ||
      (sample.arm !== "A" && sample.arm !== "B") ||
      (sample.armPosition !== 0 && sample.armPosition !== 1) ||
      !Number.isSafeInteger(sample.statePosition) || sample.statePosition < 0 ||
      sample.statePosition >= OPT_0084_STATE_IDS.length ||
      !Number.isFinite(sample.wallMilliseconds) || sample.wallMilliseconds <= 0 ||
      !Number.isSafeInteger(sample.tokenId) || sample.tokenId < 0 ||
      !Number.isSafeInteger(sample.word) || sample.word < 0 ||
      sample.word > 0xffff_ffff ||
      !Number.isSafeInteger(sample.positiveCandidateCount) ||
      sample.positiveCandidateCount <= 0
    ) {
      throw new Error("OPT-0084 timing sample is invalid");
    }
    if (OPT_0084_PAIR_ORDERS[sample.roundIndex]![sample.armPosition] !== sample.arm) {
      throw new Error("OPT-0084 timing sample violates its frozen arm order");
    }
    const values = stateValues.get(sample.stateId)!;
    values[sample.arm].push(sample.wallMilliseconds);
    aggregate[sample.arm][sample.roundIndex] =
      aggregate[sample.arm][sample.roundIndex]! + sample.wallMilliseconds;
    armPositionCounts[sample.arm][sample.armPosition] += 1;
    const key = `${sample.roundIndex}:${sample.stateId}`;
    const pair = paired.get(key) ?? {};
    if (pair[sample.arm] !== undefined) {
      throw new Error("OPT-0084 timing sample duplicated an arm/state/round");
    }
    pair[sample.arm] = sample;
    paired.set(key, pair);
  }

  let everyPairedSampleExact = paired.size ===
    OPT_0084_TIMING_ROUND_COUNT * OPT_0084_STATE_IDS.length;
  for (const pair of paired.values()) {
    const a = pair.A;
    const b = pair.B;
    if (
      a === undefined || b === undefined ||
      a.tokenId !== b.tokenId || a.word !== b.word ||
      a.positiveCandidateCount !== b.positiveCandidateCount
    ) {
      everyPairedSampleExact = false;
    }
  }

  const denseIds = new Set(OPT_0084_DENSE_STATE_IDS);
  const stateTiming = {} as Record<Opt0084StateId, {
    readonly A: Opt0084Summary;
    readonly B: Opt0084Summary;
    readonly medianSpeedup: number;
    readonly medianSavingMilliseconds: number;
    readonly denseGateState: boolean;
  }>;
  let noRegressingDenseStateMedian = true;
  for (const stateId of OPT_0084_STATE_IDS) {
    const values = stateValues.get(stateId)!;
    const a = summarizeOpt0084(values.A, OPT_0084_TIMING_ROUND_COUNT);
    const b = summarizeOpt0084(values.B, OPT_0084_TIMING_ROUND_COUNT);
    const denseGateState = denseIds.has(stateId);
    if (denseGateState && b.median > a.median) {
      noRegressingDenseStateMedian = false;
    }
    stateTiming[stateId] = Object.freeze({
      A: a,
      B: b,
      medianSpeedup: a.median / b.median,
      medianSavingMilliseconds: a.median - b.median,
      denseGateState,
    });
  }

  let pairWinCount = 0;
  for (let round = 0; round < OPT_0084_TIMING_ROUND_COUNT; round += 1) {
    if (aggregate.B[round]! < aggregate.A[round]!) pairWinCount += 1;
  }
  const aggregateA = summarizeOpt0084(
    aggregate.A,
    OPT_0084_TIMING_ROUND_COUNT,
  );
  const aggregateB = summarizeOpt0084(
    aggregate.B,
    OPT_0084_TIMING_ROUND_COUNT,
  );
  const semanticMedianSaving = OPT_0084_SEMANTIC_STATE_IDS.reduce(
    (total, stateId) => total + stateTiming[stateId].medianSavingMilliseconds,
    0,
  ) / OPT_0084_SEMANTIC_STATE_IDS.length;
  const projectedDefaultSemanticSavingSeconds = semanticMedianSaving *
    OPT_0084_DEFAULT_SEMANTIC_DRAW_COUNT / 1_000;
  const aggregateSamplerSpeedup = aggregateA.median / aggregateB.median;
  const passed = everyPairedSampleExact && noRegressingDenseStateMedian &&
    pairWinCount >= OPT_0084_MINIMUM_PAIR_WINS &&
    aggregateSamplerSpeedup >= OPT_0084_MINIMUM_SAMPLER_SPEEDUP &&
    projectedDefaultSemanticSavingSeconds >=
      OPT_0084_MINIMUM_PROJECTED_SAVING_SECONDS;

  return Object.freeze({
    passed,
    pairWinCount,
    requiredPairWinCount: OPT_0084_MINIMUM_PAIR_WINS,
    aggregateSamplerSpeedup,
    requiredSamplerSpeedup: OPT_0084_MINIMUM_SAMPLER_SPEEDUP,
    projectedDefaultSemanticSavingSeconds,
    requiredProjectedSavingSeconds:
      OPT_0084_MINIMUM_PROJECTED_SAVING_SECONDS,
    noRegressingDenseStateMedian,
    everyPairedSampleExact,
    armPositionCounts: Object.freeze({
      A: Object.freeze([...armPositionCounts.A]) as readonly [number, number],
      B: Object.freeze([...armPositionCounts.B]) as readonly [number, number],
    }),
    aggregateRoundWallMilliseconds: Object.freeze({ A: aggregateA, B: aggregateB }),
    stateTiming: Object.freeze(stateTiming),
  });
}

export function summarizeOpt0084(
  values: readonly number[],
  expectedCount: number,
): Opt0084Summary {
  if (
    values.length !== expectedCount ||
    values.some((value) => !Number.isFinite(value) || value <= 0)
  ) {
    throw new Error("OPT-0084 timing sample set is invalid");
  }
  return Object.freeze({
    count: values.length,
    minimum: Math.min(...values),
    median: opt0084Median(values),
    maximum: Math.max(...values),
  });
}

export function opt0084Median(values: readonly number[]): number {
  if (values.length === 0) throw new Error("OPT-0084 cannot summarize no values");
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function requireIdentityString(
  record: Readonly<Record<string, unknown>>,
  name: string,
): string {
  const value = record[name];
  if (
    typeof value !== "string" || value !== value.trim() || value.length === 0 ||
    value.length > 256 || /[\u0000-\u001f\u007f]/.test(value)
  ) {
    throw new TypeError(`OPT-0084 run identity ${name} is invalid`);
  }
  return value;
}

function requireIdentityInteger(
  record: Readonly<Record<string, unknown>>,
  name: string,
): number {
  const value = record[name];
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new TypeError(`OPT-0084 run identity ${name} must be an integer`);
  }
  return value;
}
