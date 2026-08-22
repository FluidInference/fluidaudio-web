import {
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
} from "../../src/webgpu/dit-attention-profile.js";

export const OPT_0081_REPRESENTATIVE_EXPERIMENT = "OPT-0081" as const;
export const OPT_0081_REPRESENTATIVE_RECEIPT_SCHEMA =
  "ace-opt-0081-f16-representative-layers-v1" as const;
export const OPT_0081_REPRESENTATIVE_STORAGE_PROFILE =
  "opt-0081-six-dense-input-f16-storage-v1" as const;
export const OPT_0081_REPRESENTATIVE_REGISTRATION_COMMIT =
  "70bd6d023b9cf9d2f3b8adb7d34122435e3a5fbd" as const;
export const OPT_0081_REPRESENTATIVE_SETUP_CLARIFICATION_COMMIT =
  "9d275757d235465a2a05be89b0e053af9dda5360" as const;
export const OPT_0081_REPRESENTATIVE_GRAPH_PROFILE_COMMIT =
  "79fa9f68755b9ec1537e045fad9331de95c1b62f" as const;
export const OPT_0081_REPRESENTATIVE_REQUEST_SHA256 =
  "031e418ac5db37355fe5e265a005cb280e02ce418e560312ac89fa184bb8862f" as const;
export const OPT_0081_REPRESENTATIVE_MAIN_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6" as const;
export const OPT_0081_REPRESENTATIVE_DENSE_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f" as const;
export const OPT_0081_REPRESENTATIVE_TEXT_TOKEN_SHA256 =
  "8067ee5c606e45e54d991364aa82a0ef7303e2a4e98831a01bb974236cafb3b2" as const;
export const OPT_0081_REPRESENTATIVE_LYRIC_TOKEN_SHA256 =
  "b4b58cd318163b4dfaa02b7ddbf46b18d84a415909c7662f9538c0b9053f3764" as const;
export const OPT_0081_REPRESENTATIVE_CONDITION_SHA256 =
  "102308c377139c80b034acd38d90d3c81c7272eb9aa077cd9fbd66f47100c49b" as const;
export const OPT_0081_REPRESENTATIVE_CONTEXT_SHA256 =
  "22c66fd3f3c80d1cd4c6c7ffbf3f20f65a5fc822d89aa0ed4b5729a92c1a66c0" as const;

export const OPT_0081_REPRESENTATIVE_CONTROL_ARENA_BYTES =
  674_815_488 as const;
export const OPT_0081_REPRESENTATIVE_CANDIDATE_ARENA_BYTES =
  601_087_488 as const;
export const OPT_0081_REPRESENTATIVE_ARENA_SAVING_BYTES =
  73_728_000 as const;
export const OPT_0081_REPRESENTATIVE_FIRST_COMMAND = 25 as const;
export const OPT_0081_REPRESENTATIVE_LAST_COMMAND = 52 as const;
export const OPT_0081_REPRESENTATIVE_COMMAND_COUNT = 28 as const;
export const OPT_0081_REPRESENTATIVE_EPOCH_COUNT = 7 as const;
export const OPT_0081_REPRESENTATIVE_IDLE_TURNS = 6 as const;
export const OPT_0081_REPRESENTATIVE_REQUIRED_PAIR_WINS = 7 as const;
export const OPT_0081_REPRESENTATIVE_REQUIRED_DIRECTION_WINS = 3 as const;
export const OPT_0081_REPRESENTATIVE_DIRECTIONAL_SAVING_MS = 31.25 as const;
export const OPT_0081_REPRESENTATIVE_DIRECTIONAL_T_95 =
  3.182446305284263 as const;
export const OPT_0081_REPRESENTATIVE_PROJECTION_MULTIPLIER = 96 as const;
export const OPT_0081_REPRESENTATIVE_HEARTBEAT_INTERVAL_MS = 50 as const;
export const OPT_0081_REPRESENTATIVE_MAX_HEARTBEAT_GAP_MS = 500 as const;
export const OPT_0081_REPRESENTATIVE_THERMAL_POLL_MS = 1_000 as const;
export const OPT_0081_REPRESENTATIVE_MINIMUM_NOMINAL_MS = 30_000 as const;
export const OPT_0081_REPRESENTATIVE_MAXIMUM_THERMAL_GAP_MS = 1_500 as const;
export const OPT_0081_REPRESENTATIVE_MAXIMUM_LAUNCH_DELAY_MS = 5_000 as const;

const F16_CAST_SCRATCH = new ArrayBuffer(4);
const F16_CAST_FLOAT = new Float32Array(F16_CAST_SCRATCH);
const F16_CAST_WORD = new Uint32Array(F16_CAST_SCRATCH);

/** Exact IEEE-754 binary32 to binary16 round-to-nearest, ties-to-even cast. */
export function numberToOpt0081Float16Bits(value: number): number {
  F16_CAST_FLOAT[0] = value;
  const word = F16_CAST_WORD[0]!;
  const sign = (word >>> 16) & 0x8000;
  const exponent = (word >>> 23) & 0xff;
  const mantissa = word & 0x007f_ffff;
  if (exponent === 0xff) {
    return mantissa === 0 ? sign | 0x7c00 : sign | 0x7e00;
  }
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 31) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const significand = mantissa | 0x0080_0000;
    const shift = 14 - halfExponent;
    let rounded = significand >>> shift;
    const mask = 2 ** shift - 1;
    const remainder = significand & mask;
    const halfway = 2 ** (shift - 1);
    if (
      remainder > halfway ||
      (remainder === halfway && (rounded & 1) !== 0)
    ) rounded += 1;
    return sign | rounded;
  }
  let roundedMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (
    remainder > 0x1000 ||
    (remainder === 0x1000 && (roundedMantissa & 1) !== 0)
  ) roundedMantissa += 1;
  return sign | ((halfExponent << 10) + roundedMantissa);
}

export type Opt0081RepresentativeArm = "A" | "B";
export type Opt0081RepresentativeRoundIndex =
  0 | 1 | 2 | 3 | 4 | 5 | 6 | 7;
export type Opt0081RepresentativeDirection = "forward" | "reverse";
export type Opt0081RepresentativeDisposition =
  | "positive-B-representative-layers-authorize-complete-evaluation"
  | "negative-stop-observed-raw-bit-correctness-mismatch"
  | "inconclusive-invalid-correctness-topology-or-lifecycle-evidence"
  | "inconclusive-invalid-thermal-or-heartbeat-provenance"
  | "inconclusive-directional-or-material-wall-evidence";

export const OPT_0081_REPRESENTATIVE_ROUND_ORDERS = Object.freeze([
  Object.freeze(["A", "B"] as const),
  Object.freeze(["B", "A"] as const),
  Object.freeze(["B", "A"] as const),
  Object.freeze(["A", "B"] as const),
  Object.freeze(["A", "B"] as const),
  Object.freeze(["B", "A"] as const),
  Object.freeze(["B", "A"] as const),
  Object.freeze(["A", "B"] as const),
] as const);

export const OPT_0081_REPRESENTATIVE_BOUNDARY_ROLES = Object.freeze([
  Object.freeze({ tap: "selfModulated" as const, words: 4_608_000 }),
  Object.freeze({ tap: "selfMergedAttention" as const, words: 4_608_000 }),
  Object.freeze({ tap: "crossNormalized" as const, words: 4_608_000 }),
  Object.freeze({ tap: "crossMergedAttention" as const, words: 4_608_000 }),
  Object.freeze({ tap: "mlpModulated" as const, words: 4_608_000 }),
  Object.freeze({ tap: "gatedActivation" as const, words: 13_824_000 }),
] as const);

export const OPT_0081_REPRESENTATIVE_DENSE_OUTPUT_ROLES = Object.freeze([
  Object.freeze({ tap: "selfQuery" as const, words: 4_608_000 }),
  Object.freeze({ tap: "selfKey" as const, words: 2_304_000 }),
  Object.freeze({ tap: "selfValue" as const, words: 2_304_000 }),
  Object.freeze({ tap: "selfOutput" as const, words: 4_608_000 }),
  Object.freeze({ tap: "crossQuery" as const, words: 4_608_000 }),
  Object.freeze({ tap: "crossOutput" as const, words: 4_608_000 }),
  Object.freeze({ tap: "mlpGate" as const, words: 13_824_000 }),
  Object.freeze({ tap: "mlpUp" as const, words: 13_824_000 }),
  Object.freeze({ tap: "mlpDown" as const, words: 4_608_000 }),
] as const);

export type Opt0081RepresentativeBoundaryTap =
  typeof OPT_0081_REPRESENTATIVE_BOUNDARY_ROLES[number]["tap"];
export type Opt0081RepresentativeDenseOutputTap =
  typeof OPT_0081_REPRESENTATIVE_DENSE_OUTPUT_ROLES[number]["tap"];

export interface Opt0081RepresentativeRoundPlan {
  readonly roundIndex: Opt0081RepresentativeRoundIndex;
  readonly direction: Opt0081RepresentativeDirection;
  readonly armOrder: readonly [
    Opt0081RepresentativeArm,
    Opt0081RepresentativeArm,
  ];
}

export function buildOpt0081RepresentativeRounds():
  readonly Opt0081RepresentativeRoundPlan[] {
  return Object.freeze(OPT_0081_REPRESENTATIVE_ROUND_ORDERS.map(
    (armOrder, roundIndex) => Object.freeze({
      roundIndex: roundIndex as Opt0081RepresentativeRoundIndex,
      direction: armOrder[0] === "A" ? "forward" : "reverse",
      armOrder,
    }),
  ));
}

export interface Opt0081RepresentativeEpochTopology {
  readonly epochIndex: 0 | 1 | 2 | 3 | 4 | 5 | 6;
  readonly firstCommandIndex: number;
  readonly lastCommandIndex: number;
  readonly commandCount: 4;
}

export const OPT_0081_REPRESENTATIVE_EPOCHS = Object.freeze(
  Array.from({ length: OPT_0081_REPRESENTATIVE_EPOCH_COUNT }, (_, index) =>
    Object.freeze({
      epochIndex: index as Opt0081RepresentativeEpochTopology["epochIndex"],
      firstCommandIndex: OPT_0081_REPRESENTATIVE_FIRST_COMMAND + index * 4,
      lastCommandIndex: OPT_0081_REPRESENTATIVE_FIRST_COMMAND + index * 4 + 3,
      commandCount: 4 as const,
    })
  ),
);

export interface Opt0081RepresentativeTopology {
  readonly firstCommandIndex: 25;
  readonly lastCommandIndex: 52;
  readonly commandBufferCount: 28;
  readonly completionFenceRequestedCount: 28;
  readonly completionFenceSettledCount: 28;
  readonly completionFenceRejectedCount: 0;
  readonly completionEpochCount: 7;
  readonly trueQueueDrainCount: 7;
  readonly cooperativeIdleTurnCount: 6;
  readonly requestedCooperativeIdleMilliseconds: 6;
  readonly maximumOutstandingCommandBuffers: 2;
  readonly maximumPendingDescriptorCount: 2;
  readonly pendingDescriptorCountAfterRun: 0;
  readonly timestepCommandCount: 1;
  readonly inputProjectionCommandCount: 1;
  readonly slidingLayerCommandCount: 11;
  readonly fullLayerCommandCount: 15;
  readonly producerStoreCount: 12;
  readonly denseConsumerCount: 18;
  readonly layerAttentionRoutes: readonly [
    Readonly<{ readonly layer: 0; readonly self: "query8-sliding";
      readonly cross: "query8-cross" }>,
    Readonly<{ readonly layer: 1; readonly self: "quad-query32-full";
      readonly cross: "query8-cross" }>,
  ];
  readonly descriptorOrTapCommandCount: 0;
  readonly timestampQueryCount: 0;
  readonly measurementReadbackCount: 0;
  readonly measurementMapCount: 0;
  readonly attentionRuntimeProfile:
    typeof ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE;
  readonly epochs: readonly Opt0081RepresentativeEpochTopology[];
  readonly physicalCommandLabels: readonly string[];
  readonly progressLabels: readonly string[];
}

export function requireOpt0081RepresentativeTopology(
  value: Opt0081RepresentativeTopology,
): Opt0081RepresentativeTopology {
  if (
    value.firstCommandIndex !== OPT_0081_REPRESENTATIVE_FIRST_COMMAND ||
    value.lastCommandIndex !== OPT_0081_REPRESENTATIVE_LAST_COMMAND ||
    value.commandBufferCount !== OPT_0081_REPRESENTATIVE_COMMAND_COUNT ||
    value.completionFenceRequestedCount !==
      OPT_0081_REPRESENTATIVE_COMMAND_COUNT ||
    value.completionFenceSettledCount !==
      OPT_0081_REPRESENTATIVE_COMMAND_COUNT ||
    value.completionFenceRejectedCount !== 0 ||
    value.completionEpochCount !== OPT_0081_REPRESENTATIVE_EPOCH_COUNT ||
    value.trueQueueDrainCount !== OPT_0081_REPRESENTATIVE_EPOCH_COUNT ||
    value.cooperativeIdleTurnCount !== OPT_0081_REPRESENTATIVE_IDLE_TURNS ||
    value.requestedCooperativeIdleMilliseconds !==
      OPT_0081_REPRESENTATIVE_IDLE_TURNS ||
    value.maximumOutstandingCommandBuffers !== 2 ||
    value.maximumPendingDescriptorCount !== 2 ||
    value.pendingDescriptorCountAfterRun !== 0 ||
    value.timestepCommandCount !== 1 ||
    value.inputProjectionCommandCount !== 1 ||
    value.slidingLayerCommandCount !== 11 ||
    value.fullLayerCommandCount !== 15 ||
    value.producerStoreCount !== 12 || value.denseConsumerCount !== 18 ||
    value.layerAttentionRoutes.length !== 2 ||
    value.layerAttentionRoutes[0]?.layer !== 0 ||
    value.layerAttentionRoutes[0].self !== "query8-sliding" ||
    value.layerAttentionRoutes[0].cross !== "query8-cross" ||
    value.layerAttentionRoutes[1]?.layer !== 1 ||
    value.layerAttentionRoutes[1].self !== "quad-query32-full" ||
    value.layerAttentionRoutes[1].cross !== "query8-cross" ||
    value.descriptorOrTapCommandCount !== 0 ||
    value.timestampQueryCount !== 0 ||
    value.measurementReadbackCount !== 0 || value.measurementMapCount !== 0 ||
    value.attentionRuntimeProfile !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
    value.epochs.length !== OPT_0081_REPRESENTATIVE_EPOCH_COUNT ||
    value.epochs.some((epoch, index) => {
      const expected = OPT_0081_REPRESENTATIVE_EPOCHS[index]!;
      return epoch.epochIndex !== expected.epochIndex ||
        epoch.firstCommandIndex !== expected.firstCommandIndex ||
        epoch.lastCommandIndex !== expected.lastCommandIndex ||
        epoch.commandCount !== expected.commandCount;
    }) ||
    value.physicalCommandLabels.length !==
      OPT_0081_REPRESENTATIVE_COMMAND_COUNT ||
    value.progressLabels.length !== OPT_0081_REPRESENTATIVE_COMMAND_COUNT ||
    value.physicalCommandLabels.some((label, index) =>
      !nonempty(label) || label !== value.progressLabels[index]
    )
  ) throw new Error("OPT-0081 representative command topology changed");
  return value;
}

export interface Opt0081RepresentativeEpochTiming {
  readonly epochIndex: 0 | 1 | 2 | 3 | 4 | 5 | 6;
  readonly firstCommandIndex: number;
  readonly lastCommandIndex: number;
  readonly submitThroughTrueDrainMilliseconds: number;
}

export interface Opt0081RepresentativeTimingSample {
  readonly roundIndex: Opt0081RepresentativeRoundIndex;
  readonly occurrenceIndex: 0 | 1;
  readonly arm: Opt0081RepresentativeArm;
  readonly direction: Opt0081RepresentativeDirection;
  readonly sliceStartedAtPerformanceMilliseconds: number;
  readonly sliceDrainedAtPerformanceMilliseconds: number;
  readonly sliceWallMilliseconds: number;
  readonly epochs: readonly Opt0081RepresentativeEpochTiming[];
  /** Overlapping and non-additive; retained individually, never summed. */
  readonly completionFenceLatenciesMilliseconds: readonly number[];
  readonly topology: Opt0081RepresentativeTopology;
}

export interface Opt0081RepresentativeTimingSummary {
  readonly fixedRoundOrders: readonly string[];
  readonly samples: readonly Opt0081RepresentativeTimingSample[];
  readonly aWallMilliseconds: readonly number[];
  readonly bWallMilliseconds: readonly number[];
  readonly pairedSavingsMilliseconds: readonly number[];
  readonly aMeanWallMilliseconds: number;
  readonly bMeanWallMilliseconds: number;
  readonly aMedianWallMilliseconds: number;
  readonly bMedianWallMilliseconds: number;
  readonly bOverallMeanFaster: boolean;
  readonly bOverallMedianFaster: boolean;
  readonly pairedWins: number;
  readonly forwardPairedWins: number;
  readonly reversePairedWins: number;
  readonly forwardMeanPairedSavingMilliseconds: number;
  readonly reverseMeanPairedSavingMilliseconds: number;
  readonly forwardPairedSavingSampleStandardDeviationMilliseconds: number;
  readonly reversePairedSavingSampleStandardDeviationMilliseconds: number;
  readonly forwardPairedSavingLower95Milliseconds: number;
  readonly reversePairedSavingLower95Milliseconds: number;
  readonly forwardProjectedEightEvaluationSavingMilliseconds: number;
  readonly reverseProjectedEightEvaluationSavingMilliseconds: number;
  readonly topologyIdenticalAcrossArms: boolean;
  readonly wallBoundaryConsistent: boolean;
  readonly passed: boolean;
}

export function summarizeOpt0081RepresentativeTiming(
  samples: readonly Opt0081RepresentativeTimingSample[],
): Opt0081RepresentativeTimingSummary {
  const rounds = buildOpt0081RepresentativeRounds();
  if (samples.length !== rounds.length * 2) {
    throw new Error("OPT-0081 representative timing inventory changed");
  }
  const referenceTopology = topologyKey(samples[0]!.topology);
  let wallBoundaryConsistent = true;
  let topologyIdenticalAcrossArms = true;
  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index]!;
    const round = rounds[Math.floor(index / 2)]!;
    const occurrenceIndex = index % 2 as 0 | 1;
    const elapsed = sample.sliceDrainedAtPerformanceMilliseconds -
      sample.sliceStartedAtPerformanceMilliseconds;
    const epochWall = sample.epochs.reduce((sum, epoch) =>
      sum + epoch.submitThroughTrueDrainMilliseconds, 0);
    if (
      sample.roundIndex !== round.roundIndex ||
      sample.occurrenceIndex !== occurrenceIndex ||
      sample.arm !== round.armOrder[occurrenceIndex] ||
      sample.direction !== round.direction ||
      invalidDuration(sample.sliceWallMilliseconds) ||
      invalidDuration(sample.sliceStartedAtPerformanceMilliseconds) ||
      invalidDuration(sample.sliceDrainedAtPerformanceMilliseconds) ||
      sample.sliceDrainedAtPerformanceMilliseconds <
        sample.sliceStartedAtPerformanceMilliseconds ||
      Math.abs(elapsed - sample.sliceWallMilliseconds) > 1e-6 ||
      sample.epochs.length !== OPT_0081_REPRESENTATIVE_EPOCH_COUNT ||
      sample.epochs.some((epoch, epochIndex) => {
        const expected = OPT_0081_REPRESENTATIVE_EPOCHS[epochIndex]!;
        return epoch.epochIndex !== expected.epochIndex ||
          epoch.firstCommandIndex !== expected.firstCommandIndex ||
          epoch.lastCommandIndex !== expected.lastCommandIndex ||
          invalidDuration(epoch.submitThroughTrueDrainMilliseconds);
      }) ||
      sample.completionFenceLatenciesMilliseconds.length !==
        OPT_0081_REPRESENTATIVE_COMMAND_COUNT ||
      sample.completionFenceLatenciesMilliseconds.some(invalidDuration) ||
      requireWithoutThrow(() =>
        requireOpt0081RepresentativeTopology(sample.topology)
      ) === false
    ) throw new Error("OPT-0081 representative timing sample changed");
    wallBoundaryConsistent &&= epochWall <= sample.sliceWallMilliseconds + 1e-6;
    topologyIdenticalAcrossArms &&=
      topologyKey(sample.topology) === referenceTopology;
  }
  const aWallMilliseconds: number[] = [];
  const bWallMilliseconds: number[] = [];
  const pairedSavingsMilliseconds: number[] = [];
  const forwardSavings: number[] = [];
  const reverseSavings: number[] = [];
  for (let roundIndex = 0; roundIndex < rounds.length; roundIndex += 1) {
    const left = samples[roundIndex * 2]!;
    const right = samples[roundIndex * 2 + 1]!;
    const a = left.arm === "A" ? left : right;
    const b = left.arm === "B" ? left : right;
    aWallMilliseconds.push(a.sliceWallMilliseconds);
    bWallMilliseconds.push(b.sliceWallMilliseconds);
    const saving = a.sliceWallMilliseconds - b.sliceWallMilliseconds;
    pairedSavingsMilliseconds.push(saving);
    (rounds[roundIndex]!.direction === "forward"
      ? forwardSavings : reverseSavings).push(saving);
  }
  const aMeanWallMilliseconds = mean(aWallMilliseconds);
  const bMeanWallMilliseconds = mean(bWallMilliseconds);
  const aMedianWallMilliseconds = median(aWallMilliseconds);
  const bMedianWallMilliseconds = median(bWallMilliseconds);
  const pairedWins = pairedSavingsMilliseconds.filter((value) => value > 0)
    .length;
  const forwardPairedWins = forwardSavings.filter((value) => value > 0).length;
  const reversePairedWins = reverseSavings.filter((value) => value > 0).length;
  const forwardMeanPairedSavingMilliseconds = mean(forwardSavings);
  const reverseMeanPairedSavingMilliseconds = mean(reverseSavings);
  const forwardPairedSavingSampleStandardDeviationMilliseconds =
    sampleStandardDeviation(forwardSavings);
  const reversePairedSavingSampleStandardDeviationMilliseconds =
    sampleStandardDeviation(reverseSavings);
  const forwardPairedSavingLower95Milliseconds =
    forwardMeanPairedSavingMilliseconds -
      OPT_0081_REPRESENTATIVE_DIRECTIONAL_T_95 *
      forwardPairedSavingSampleStandardDeviationMilliseconds /
      Math.sqrt(forwardSavings.length);
  const reversePairedSavingLower95Milliseconds =
    reverseMeanPairedSavingMilliseconds -
      OPT_0081_REPRESENTATIVE_DIRECTIONAL_T_95 *
      reversePairedSavingSampleStandardDeviationMilliseconds /
      Math.sqrt(reverseSavings.length);
  const bOverallMeanFaster = bMeanWallMilliseconds < aMeanWallMilliseconds;
  const bOverallMedianFaster = bMedianWallMilliseconds < aMedianWallMilliseconds;
  const passed = bOverallMeanFaster && bOverallMedianFaster &&
    pairedWins >= OPT_0081_REPRESENTATIVE_REQUIRED_PAIR_WINS &&
    forwardPairedWins >= OPT_0081_REPRESENTATIVE_REQUIRED_DIRECTION_WINS &&
    reversePairedWins >= OPT_0081_REPRESENTATIVE_REQUIRED_DIRECTION_WINS &&
    forwardMeanPairedSavingMilliseconds >=
      OPT_0081_REPRESENTATIVE_DIRECTIONAL_SAVING_MS &&
    reverseMeanPairedSavingMilliseconds >=
      OPT_0081_REPRESENTATIVE_DIRECTIONAL_SAVING_MS &&
    forwardPairedSavingLower95Milliseconds >=
      OPT_0081_REPRESENTATIVE_DIRECTIONAL_SAVING_MS &&
    reversePairedSavingLower95Milliseconds >=
      OPT_0081_REPRESENTATIVE_DIRECTIONAL_SAVING_MS &&
    topologyIdenticalAcrossArms && wallBoundaryConsistent;
  return Object.freeze({
    fixedRoundOrders: Object.freeze(rounds.map(({ armOrder }) =>
      armOrder.join(""))),
    samples: Object.freeze([...samples]),
    aWallMilliseconds: Object.freeze(aWallMilliseconds),
    bWallMilliseconds: Object.freeze(bWallMilliseconds),
    pairedSavingsMilliseconds: Object.freeze(pairedSavingsMilliseconds),
    aMeanWallMilliseconds,
    bMeanWallMilliseconds,
    aMedianWallMilliseconds,
    bMedianWallMilliseconds,
    bOverallMeanFaster,
    bOverallMedianFaster,
    pairedWins,
    forwardPairedWins,
    reversePairedWins,
    forwardMeanPairedSavingMilliseconds,
    reverseMeanPairedSavingMilliseconds,
    forwardPairedSavingSampleStandardDeviationMilliseconds,
    reversePairedSavingSampleStandardDeviationMilliseconds,
    forwardPairedSavingLower95Milliseconds,
    reversePairedSavingLower95Milliseconds,
    forwardProjectedEightEvaluationSavingMilliseconds:
      forwardMeanPairedSavingMilliseconds *
        OPT_0081_REPRESENTATIVE_PROJECTION_MULTIPLIER,
    reverseProjectedEightEvaluationSavingMilliseconds:
      reverseMeanPairedSavingMilliseconds *
        OPT_0081_REPRESENTATIVE_PROJECTION_MULTIPLIER,
    topologyIdenticalAcrossArms,
    wallBoundaryConsistent,
    passed,
  });
}

export interface Opt0081RepresentativeRawCheckpoint {
  readonly layer: 0 | 1;
  readonly tap: string;
  readonly comparedWords: number;
  readonly differingWordCount: number;
  readonly unwrittenWordCount: number;
  readonly exact: boolean;
  readonly signedZeroExact: boolean;
  readonly finite: boolean;
  readonly qNaNPrefillOverwritten: boolean;
  readonly firstWordCovered: boolean;
  readonly lastWordCovered: boolean;
  readonly tailRows2240Through2249Covered: boolean;
  readonly prefixGuardIntact: boolean;
  readonly suffixGuardIntact: boolean;
  readonly adjacentGuardsIntact: boolean;
  readonly sha256: string;
}

export interface Opt0081RepresentativeBoundaryRun {
  readonly run: "B1" | "B2";
  readonly checkpoints: readonly Opt0081RepresentativeRawCheckpoint[];
}

export interface Opt0081RepresentativeComparison {
  readonly comparison: "A/A" | "A/B" | "B/B";
  readonly denseOutputs: readonly Opt0081RepresentativeRawCheckpoint[];
  readonly layerOutputs: readonly Opt0081RepresentativeRawCheckpoint[];
}

export interface Opt0081RepresentativeCorrectness {
  readonly completedBeforeReady: true;
  readonly runOrder: readonly ["A", "A", "B", "B"];
  /** A/A current-producer raw-U32 deterministic repeat. */
  readonly controlBoundaryRepeat:
    readonly Opt0081RepresentativeRawCheckpoint[];
  readonly boundaryRuns: readonly Opt0081RepresentativeBoundaryRun[];
  /** B/B typed-boundary raw-U16 deterministic repeat. */
  readonly candidateBoundaryRepeat:
    readonly Opt0081RepresentativeRawCheckpoint[];
  readonly comparisons: readonly Opt0081RepresentativeComparison[];
  readonly boundaryWordsPerCandidateRun: 73_728_000;
  readonly denseOutputWordsPerComparison: 110_592_000;
  readonly layerOutputWordsPerComparison: 9_216_000;
  readonly uncapturedGpuErrorCount: number;
  readonly validationErrorCount: number;
  readonly deviceLossCount: number;
  readonly passed: boolean;
}

export interface Opt0081RepresentativeCorrectnessInspection {
  readonly structurallyValid: boolean;
  readonly observedRawMismatch: boolean;
  readonly passed: boolean;
}

export function inspectOpt0081RepresentativeCorrectness(
  value: Opt0081RepresentativeCorrectness,
): Opt0081RepresentativeCorrectnessInspection {
  const structural = value.completedBeforeReady === true &&
    sameStrings(value.runOrder, ["A", "A", "B", "B"]) &&
    value.boundaryWordsPerCandidateRun === 73_728_000 &&
    value.denseOutputWordsPerComparison === 110_592_000 &&
    value.layerOutputWordsPerComparison === 9_216_000 &&
    value.boundaryRuns.length === 2 &&
    value.boundaryRuns[0]?.run === "B1" &&
    value.boundaryRuns[1]?.run === "B2" &&
    checkpointInventory(
      value.controlBoundaryRepeat,
      OPT_0081_REPRESENTATIVE_BOUNDARY_ROLES,
      73_728_000,
    ) &&
    value.boundaryRuns.every((run) => checkpointInventory(
      run.checkpoints,
      OPT_0081_REPRESENTATIVE_BOUNDARY_ROLES,
      73_728_000,
    )) &&
    checkpointInventory(
      value.candidateBoundaryRepeat,
      OPT_0081_REPRESENTATIVE_BOUNDARY_ROLES,
      73_728_000,
    ) &&
    value.comparisons.length === 3 &&
    value.comparisons.every((comparison, index) =>
      comparison.comparison === (["A/A", "A/B", "B/B"] as const)[index] &&
      checkpointInventory(
        comparison.denseOutputs,
        OPT_0081_REPRESENTATIVE_DENSE_OUTPUT_ROLES,
        110_592_000,
      ) && checkpointInventory(
        comparison.layerOutputs,
        [{ tap: "layerOutput", words: 4_608_000 }] as const,
        9_216_000,
      )
    ) &&
    nonNegativeInteger(value.uncapturedGpuErrorCount) &&
    nonNegativeInteger(value.validationErrorCount) &&
    nonNegativeInteger(value.deviceLossCount);
  if (!structural) return Object.freeze({
    structurallyValid: false, observedRawMismatch: false, passed: false,
  });
  const checkpoints = [
    ...value.controlBoundaryRepeat,
    ...value.boundaryRuns.flatMap(({ checkpoints }) => checkpoints),
    ...value.candidateBoundaryRepeat,
    ...value.comparisons.flatMap(({ denseOutputs, layerOutputs }) =>
      [...denseOutputs, ...layerOutputs]),
  ];
  const observedRawMismatch = checkpoints.some((checkpoint) =>
    checkpoint.differingWordCount > 0 || checkpoint.exact === false
  );
  const checkpointGatesPassed = checkpoints.every(checkpointPassed);
  const passed = !observedRawMismatch && checkpointGatesPassed &&
    value.uncapturedGpuErrorCount === 0 && value.validationErrorCount === 0 &&
    value.deviceLossCount === 0;
  return Object.freeze({
    structurallyValid: value.passed === passed,
    observedRawMismatch,
    passed: value.passed === passed && passed,
  });
}

export interface Opt0081RepresentativeCancellationEvidence {
  readonly arm: "B";
  readonly residentArmReused: true;
  readonly successorSubmittedBeforeObservation: true;
  readonly outstandingSuccessorCountAtObservation: 1;
  readonly backfillAfterObservationCount: 0;
  readonly progressAfterObservationCount: 0;
  readonly allSubmittedFencesSettledBeforeRelease: true;
  readonly originalErrorPreserved: true;
  readonly cleanupMilliseconds: number;
  readonly pendingDescriptorCountAfterCleanup: 0;
  readonly temporaryCreatedBufferCount: 0;
  readonly temporaryDestroyedBufferCount: 0;
  readonly temporaryLiveBufferCountAfterCleanup: 0;
  readonly temporaryLiveByteCountAfterCleanup: 0;
  readonly temporaryRuntimeOwnerCount: 0;
  readonly temporaryDestroyedRuntimeOwnerCount: 0;
  readonly passed: boolean;
}

export function requireOpt0081RepresentativeCancellation(
  value: Opt0081RepresentativeCancellationEvidence,
): Opt0081RepresentativeCancellationEvidence {
  const passed = value.arm === "B" &&
    value.residentArmReused === true &&
    value.successorSubmittedBeforeObservation === true &&
    value.outstandingSuccessorCountAtObservation === 1 &&
    value.backfillAfterObservationCount === 0 &&
    value.progressAfterObservationCount === 0 &&
    value.allSubmittedFencesSettledBeforeRelease === true &&
    value.originalErrorPreserved === true &&
    Number.isFinite(value.cleanupMilliseconds) &&
    value.cleanupMilliseconds >= 0 && value.cleanupMilliseconds <= 1_000 &&
    value.pendingDescriptorCountAfterCleanup === 0 &&
    value.temporaryCreatedBufferCount === 0 &&
    value.temporaryDestroyedBufferCount === 0 &&
    value.temporaryLiveBufferCountAfterCleanup === 0 &&
    value.temporaryLiveByteCountAfterCleanup === 0 &&
    value.temporaryRuntimeOwnerCount === 0 &&
    value.temporaryDestroyedRuntimeOwnerCount === 0;
  if (!passed || value.passed !== true) {
    throw new Error("OPT-0081 representative cancellation preflight failed");
  }
  return value;
}

export interface Opt0081RepresentativeSetupCleanupEvidence {
  readonly schema: "ace-opt-0081-representative-setup-cleanup-v1";
  readonly createdBufferCount: number;
  readonly destroyedBufferCount: number;
  readonly liveBufferCount: number;
  readonly liveByteCount: number;
  readonly runtimeOwnerCount: number;
  readonly destroyedRuntimeOwnerCount: number;
  readonly residentModelDestroyed: boolean;
  readonly mappedRangeCount: number;
  readonly unmappedRangeCount: number;
  readonly liveMapCount: number;
  readonly pendingDescriptorCount: number;
  readonly callbackCount: number;
  readonly leaseCount: number;
  readonly armReleaseCount: number;
  readonly drainOrderViolationCount: number;
  readonly passed: boolean;
}

export interface Opt0081RepresentativeDeviceLossCleanupEvidence {
  readonly schema: "ace-opt-0081-representative-device-loss-cleanup-v1";
  readonly deviceLossInduced: boolean;
  readonly deviceLossObserved: boolean;
  readonly ownerDestroyedAfterLoss: boolean;
  readonly liveBufferCount: number;
  readonly liveByteCount: number;
  readonly liveMapCount: number;
  readonly pendingDescriptorCount: number;
  readonly callbackCount: number;
  readonly leaseCount: number;
  readonly idempotentDestroyVerified: boolean;
  readonly postDestroyRejected: boolean;
  readonly passed: boolean;
}

export interface Opt0081RepresentativeLifecycleEvidence {
  readonly createdBufferCount: number;
  readonly destroyedBufferCount: number;
  readonly maximumLiveByteCount: number;
  readonly mappedRangeCount: number;
  readonly unmappedRangeCount: number;
  readonly liveBufferCount: number;
  readonly liveByteCount: number;
  readonly liveMapCount: number;
  readonly pendingDescriptorCount: number;
  readonly maximumPendingDescriptorCount: number;
  readonly callbackCount: number;
  readonly leaseCount: number;
  readonly maximumActiveLeaseCount: number;
  readonly correctnessTargetCount: number;
  readonly maximumCorrectnessTargetCount: number;
  readonly correctnessRuntimeCount: number;
  readonly maximumCorrectnessRuntimeCount: number;
  readonly maximumCorrectnessTargetBytes: number;
  readonly correctnessTargetCompilationCount: number;
  readonly correctnessTargetReuseCount: number;
  readonly checkpointSnapshotCount: number;
  readonly maximumDetachedCheckpointBytes: number;
  readonly resetOrPrefillCount: number;
  readonly profileSwitchCount: number;
  readonly snapshotMapCount: number;
  readonly guardedTargetReleaseCount: number;
  readonly armReleaseCount: number;
  readonly drainOrderViolationCount: number;
  readonly profileSwitchWhilePendingCount: number;
  readonly runtimeOwnerCount: number;
  readonly destroyedRuntimeOwnerCount: number;
  readonly residentModelDestroyed: boolean;
  readonly precomputeCompleted: boolean;
  readonly destroyCallCount: number;
  readonly postDestroyRejectedOperationCount: number;
  readonly drainBeforeEveryResetSwitchMapAndRelease: boolean;
  readonly profileSwitchOnlyAfterTerminalDrain: boolean;
  readonly postDestroyRejected: boolean;
  readonly idempotentDestroy: boolean;
  readonly setupFailureCleanupPassed: boolean;
  readonly deviceLossCleanupPassed: boolean;
  readonly deviceDestroyed: boolean;
  readonly setupFailureCleanup: Opt0081RepresentativeSetupCleanupEvidence;
  readonly deviceLossCleanup: Opt0081RepresentativeDeviceLossCleanupEvidence;
  readonly passed: boolean;
}

export function requireOpt0081RepresentativeLifecycle(
  value: Opt0081RepresentativeLifecycleEvidence,
): Opt0081RepresentativeLifecycleEvidence {
  const passed = positiveInteger(value.createdBufferCount) &&
    value.createdBufferCount === value.destroyedBufferCount &&
    positiveInteger(value.maximumLiveByteCount) &&
    nonNegativeInteger(value.mappedRangeCount) &&
    value.mappedRangeCount === value.unmappedRangeCount &&
    value.liveBufferCount === 0 && value.liveByteCount === 0 &&
    value.liveMapCount === 0 && value.pendingDescriptorCount === 0 &&
    value.maximumPendingDescriptorCount === 2 &&
    value.callbackCount === 0 && value.leaseCount === 0 &&
    value.maximumActiveLeaseCount === 1 &&
    value.correctnessTargetCount === 0 &&
    value.maximumCorrectnessTargetCount === 1 &&
    value.correctnessRuntimeCount === 0 &&
    value.maximumCorrectnessRuntimeCount === 1 &&
    positiveInteger(value.maximumCorrectnessTargetBytes) &&
    value.correctnessTargetCompilationCount === 64 &&
    value.correctnessTargetReuseCount === 64 &&
    value.checkpointSnapshotCount === 128 &&
    positiveInteger(value.maximumDetachedCheckpointBytes) &&
    value.maximumCorrectnessTargetBytes ===
      value.maximumDetachedCheckpointBytes + 512 &&
    positiveInteger(value.resetOrPrefillCount) &&
    positiveInteger(value.profileSwitchCount) &&
    value.snapshotMapCount === 128 &&
    value.guardedTargetReleaseCount === 64 &&
    value.armReleaseCount === 2 &&
    value.drainOrderViolationCount === 0 &&
    value.profileSwitchWhilePendingCount === 0 &&
    value.runtimeOwnerCount === 66 &&
    value.runtimeOwnerCount === value.destroyedRuntimeOwnerCount &&
    value.residentModelDestroyed === true &&
    value.precomputeCompleted === true &&
    value.destroyCallCount >= 2 &&
    value.postDestroyRejectedOperationCount >= 1 &&
    value.drainBeforeEveryResetSwitchMapAndRelease === true &&
    value.profileSwitchOnlyAfterTerminalDrain === true &&
    value.postDestroyRejected === true && value.idempotentDestroy === true &&
    value.setupFailureCleanupPassed === true &&
    value.deviceLossCleanupPassed === true && value.deviceDestroyed === true &&
    setupCleanupPassed(value.setupFailureCleanup) &&
    deviceLossCleanupPassed(value.deviceLossCleanup);
  if (!passed || value.passed !== true) {
    throw new Error("OPT-0081 representative lifecycle evidence failed");
  }
  return value;
}

function setupCleanupPassed(
  value: Opt0081RepresentativeSetupCleanupEvidence,
): boolean {
  const passed = value.schema ===
      "ace-opt-0081-representative-setup-cleanup-v1" &&
    positiveInteger(value.createdBufferCount) &&
    value.createdBufferCount === value.destroyedBufferCount &&
    value.liveBufferCount === 0 && value.liveByteCount === 0 &&
    value.runtimeOwnerCount === 2 &&
    value.destroyedRuntimeOwnerCount === 2 &&
    value.residentModelDestroyed === true &&
    value.mappedRangeCount === 0 && value.unmappedRangeCount === 0 &&
    value.liveMapCount === 0 && value.pendingDescriptorCount === 0 &&
    value.callbackCount === 0 && value.leaseCount === 0 &&
    positiveInteger(value.armReleaseCount) &&
    value.drainOrderViolationCount === 0;
  return passed && value.passed === passed;
}

function deviceLossCleanupPassed(
  value: Opt0081RepresentativeDeviceLossCleanupEvidence,
): boolean {
  const passed = value.schema ===
      "ace-opt-0081-representative-device-loss-cleanup-v1" &&
    value.deviceLossInduced === true && value.deviceLossObserved === true &&
    value.ownerDestroyedAfterLoss === true && value.liveBufferCount === 0 &&
    value.liveByteCount === 0 && value.liveMapCount === 0 &&
    value.pendingDescriptorCount === 0 && value.callbackCount === 0 &&
    value.leaseCount === 0 && value.idempotentDestroyVerified === true &&
    value.postDestroyRejected === true;
  return passed && value.passed === passed;
}

export interface Opt0081RepresentativeHeartbeat {
  readonly intervalMilliseconds: 50;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly gapsMilliseconds: readonly number[];
  readonly maximumGapMilliseconds: number;
  readonly p99GapMilliseconds: number;
}

export function requireOpt0081RepresentativeHeartbeat(
  value: Opt0081RepresentativeHeartbeat,
  cleanupCompletedAtEpochMilliseconds?: number,
): Opt0081RepresentativeHeartbeat {
  const sorted = [...value.gapsMilliseconds].sort((left, right) => left - right);
  const maximum = sorted.at(-1);
  const p99 = sorted[Math.max(0, Math.ceil(sorted.length * 0.99) - 1)];
  if (
    value.intervalMilliseconds !==
      OPT_0081_REPRESENTATIVE_HEARTBEAT_INTERVAL_MS ||
    !Number.isSafeInteger(value.startedAtEpochMilliseconds) ||
    !Number.isSafeInteger(value.completedAtEpochMilliseconds) ||
    value.completedAtEpochMilliseconds <= value.startedAtEpochMilliseconds ||
    value.gapsMilliseconds.length === 0 ||
    value.gapsMilliseconds.some((gap) => !Number.isFinite(gap) || gap <= 0) ||
    value.maximumGapMilliseconds !== maximum ||
    value.p99GapMilliseconds !== p99 ||
    value.maximumGapMilliseconds >
      OPT_0081_REPRESENTATIVE_MAX_HEARTBEAT_GAP_MS ||
    (cleanupCompletedAtEpochMilliseconds !== undefined &&
      value.completedAtEpochMilliseconds < cleanupCompletedAtEpochMilliseconds)
  ) throw new Error("OPT-0081 representative heartbeat evidence failed");
  return value;
}

export interface Opt0081RepresentativeThermalLaunch {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly command: "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly traceStartedAtEpochMilliseconds: number;
  readonly gateStartedAtEpochMilliseconds: number;
  readonly gateCompletedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: 1_000;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly missingObservationCount: 0;
  readonly readyToGateDelayMilliseconds: number;
  readonly launchDelayMilliseconds: number;
}

export interface Opt0081RepresentativeThermalCompletion {
  readonly schema:
    "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1";
  readonly sha256: string;
  readonly byteLength: number;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: number;
  readonly missingObservationCount: number;
  readonly initialLevel: number;
  readonly finalLevel: number;
  readonly transitions: readonly Readonly<{
    readonly atEpochMilliseconds: number;
    readonly level: number;
  }>[];
  readonly coversCleanup: true;
}

export function parseOpt0081RepresentativeThermalLaunch(
  parameters: URLSearchParams,
  readyAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0081RepresentativeThermalLaunch {
  const source = requiredParameter(parameters, "thermalSource");
  const command = requiredParameter(parameters, "thermalCommand");
  const traceStartedAtEpochMilliseconds = requiredFiniteParameter(
    parameters, "thermalTraceStartedAtEpochMilliseconds");
  const gateStartedAtEpochMilliseconds = requiredFiniteParameter(
    parameters, "thermalGateStartedAtEpochMilliseconds");
  const gateCompletedAtEpochMilliseconds = requiredFiniteParameter(
    parameters, "thermalGateCompletedAtEpochMilliseconds");
  const observationCount = requiredIntegerParameter(
    parameters, "thermalGateObservations");
  const pollMilliseconds = requiredIntegerParameter(
    parameters, "thermalPollMilliseconds");
  const maximumPollGapMilliseconds = requiredFiniteParameter(
    parameters, "thermalGateMaximumPollGapMilliseconds");
  const nonNominalObservationCount = requiredIntegerParameter(
    parameters, "thermalGateNonNominalObservations");
  const missingObservationCount = requiredIntegerParameter(
    parameters, "thermalGateMissingObservations");
  const readyToGateDelayMilliseconds = gateStartedAtEpochMilliseconds -
    readyAtEpochMilliseconds;
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    gateCompletedAtEpochMilliseconds;
  const value = Object.freeze({
    source,
    command,
    traceStartedAtEpochMilliseconds,
    gateStartedAtEpochMilliseconds,
    gateCompletedAtEpochMilliseconds,
    observationCount,
    pollMilliseconds,
    maximumPollGapMilliseconds,
    nonNominalObservationCount,
    missingObservationCount,
    readyToGateDelayMilliseconds,
    launchDelayMilliseconds,
  }) as Opt0081RepresentativeThermalLaunch;
  return requireOpt0081RepresentativeThermalLaunch(
    value,
    readyAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
}

export function requireOpt0081RepresentativeThermalLaunch(
  value: Opt0081RepresentativeThermalLaunch,
  readyAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0081RepresentativeThermalLaunch {
  const duration = value.gateCompletedAtEpochMilliseconds -
    value.gateStartedAtEpochMilliseconds;
  const minimumObservations = Math.floor(duration /
    OPT_0081_REPRESENTATIVE_THERMAL_POLL_MS) + 1;
  if (
    value.source !== "notifyutil-com.apple.system.thermalpressurelevel" ||
    value.command !== "notifyutil -g com.apple.system.thermalpressurelevel" ||
    value.pollMilliseconds !== OPT_0081_REPRESENTATIVE_THERMAL_POLL_MS ||
    value.traceStartedAtEpochMilliseconds > readyAtEpochMilliseconds ||
    value.readyToGateDelayMilliseconds !==
      value.gateStartedAtEpochMilliseconds - readyAtEpochMilliseconds ||
    value.readyToGateDelayMilliseconds < 0 ||
    duration < OPT_0081_REPRESENTATIVE_MINIMUM_NOMINAL_MS ||
    value.observationCount < minimumObservations ||
    value.maximumPollGapMilliseconds < 0 ||
    value.maximumPollGapMilliseconds >
      OPT_0081_REPRESENTATIVE_MAXIMUM_THERMAL_GAP_MS ||
    value.nonNominalObservationCount !== 0 ||
    value.missingObservationCount !== 0 ||
    value.launchDelayMilliseconds !== launchedAtEpochMilliseconds -
      value.gateCompletedAtEpochMilliseconds ||
    value.launchDelayMilliseconds < 0 ||
    value.launchDelayMilliseconds >
      OPT_0081_REPRESENTATIVE_MAXIMUM_LAUNCH_DELAY_MS
  ) throw new Error("OPT-0081 representative thermal launch gate failed");
  return value;
}

/** Revalidates page evidence without redefining its launch time on receipt. */
export function requireOpt0081RepresentativeReceivedThermalLaunch(
  value: Opt0081RepresentativeThermalLaunch,
  readyAtEpochMilliseconds: number,
  receivedAtEpochMilliseconds: number,
): Opt0081RepresentativeThermalLaunch {
  const launchedAtEpochMilliseconds =
    value.gateCompletedAtEpochMilliseconds + value.launchDelayMilliseconds;
  requireOpt0081RepresentativeThermalLaunch(
    value,
    readyAtEpochMilliseconds,
    launchedAtEpochMilliseconds,
  );
  const deliveryMilliseconds = receivedAtEpochMilliseconds -
    launchedAtEpochMilliseconds;
  if (
    !Number.isSafeInteger(receivedAtEpochMilliseconds) ||
    !Number.isSafeInteger(launchedAtEpochMilliseconds) ||
    deliveryMilliseconds < 0 ||
    deliveryMilliseconds > OPT_0081_REPRESENTATIVE_MAXIMUM_LAUNCH_DELAY_MS
  ) throw new Error("OPT-0081 representative thermal launch delivery stale");
  return value;
}

export function parseOpt0081RepresentativeThermalCompletion(
  parameters: URLSearchParams,
  launch: Opt0081RepresentativeThermalLaunch,
  cleanupCompletedAtEpochMilliseconds: number,
): Opt0081RepresentativeThermalCompletion {
  const schema = requiredParameter(parameters, "thermalTraceSchema");
  const sha256 = requiredParameter(parameters, "thermalTraceSha256");
  const byteLength = requiredIntegerParameter(parameters,
    "thermalTraceByteLength");
  const completedAtEpochMilliseconds = requiredFiniteParameter(
    parameters, "thermalTraceCompletedAtEpochMilliseconds");
  const observationCount = requiredIntegerParameter(
    parameters, "thermalTraceObservations");
  const maximumPollGapMilliseconds = requiredFiniteParameter(
    parameters, "thermalTraceMaximumPollGapMilliseconds");
  const nonNominalObservationCount = requiredIntegerParameter(
    parameters, "thermalTraceNonNominalObservations");
  const missingObservationCount = requiredIntegerParameter(
    parameters, "thermalTraceMissingObservations");
  const initialLevel = requiredIntegerParameter(
    parameters, "thermalTraceInitialLevel");
  const finalLevel = requiredIntegerParameter(
    parameters, "thermalTraceFinalLevel");
  const transitions = parseThermalTransitions(requiredParameter(
    parameters,
    "thermalTraceTransitionsJson",
  ));
  const value = Object.freeze({
    schema,
    sha256,
    byteLength,
    completedAtEpochMilliseconds,
    observationCount,
    maximumPollGapMilliseconds,
    nonNominalObservationCount,
    missingObservationCount,
    initialLevel,
    finalLevel,
    transitions,
    coversCleanup: true,
  }) as Opt0081RepresentativeThermalCompletion;
  return requireOpt0081RepresentativeThermalCompletion(
    value,
    launch,
    cleanupCompletedAtEpochMilliseconds,
  );
}

export function requireOpt0081RepresentativeThermalCompletion(
  value: Opt0081RepresentativeThermalCompletion,
  launch: Opt0081RepresentativeThermalLaunch,
  cleanupCompletedAtEpochMilliseconds: number,
): Opt0081RepresentativeThermalCompletion {
  const duration = value.completedAtEpochMilliseconds -
    launch.traceStartedAtEpochMilliseconds;
  const minimumObservations = Math.floor(duration /
    OPT_0081_REPRESENTATIVE_THERMAL_POLL_MS) + 1;
  let currentLevel = value.initialLevel;
  let previousTransitionAt = launch.traceStartedAtEpochMilliseconds - 1;
  const transitionsValid = value.transitions.length <=
      Math.max(0, value.observationCount - 1) &&
    value.transitions.every((transition) => {
      const valid = Number.isSafeInteger(transition.atEpochMilliseconds) &&
        transition.atEpochMilliseconds >=
          launch.traceStartedAtEpochMilliseconds &&
        transition.atEpochMilliseconds <= value.completedAtEpochMilliseconds &&
        transition.atEpochMilliseconds > previousTransitionAt &&
        thermalLevel(transition.level) && transition.level !== currentLevel;
      previousTransitionAt = transition.atEpochMilliseconds;
      currentLevel = transition.level;
      return valid;
    }) && currentLevel === value.finalLevel;
  const gateStartedLevel = thermalLevelAt(
    value.initialLevel,
    value.transitions,
    launch.gateStartedAtEpochMilliseconds,
  );
  const gateTraceConsistent = gateStartedLevel === 0 &&
    value.transitions.every((transition) =>
      transition.atEpochMilliseconds < launch.gateStartedAtEpochMilliseconds ||
      transition.atEpochMilliseconds > launch.gateCompletedAtEpochMilliseconds
    );
  const hasDisclosedNonNominalLevel = value.initialLevel !== 0 ||
    value.transitions.some((transition) => transition.level !== 0);
  const nonNominalCountConsistent = value.nonNominalObservationCount === 0
    ? !hasDisclosedNonNominalLevel
    : hasDisclosedNonNominalLevel;
  if (
    value.schema !==
      "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1" ||
    !/^[0-9a-f]{64}$/u.test(value.sha256) || value.byteLength <= 0 ||
    value.completedAtEpochMilliseconds < cleanupCompletedAtEpochMilliseconds ||
    value.observationCount < minimumObservations ||
    value.maximumPollGapMilliseconds < 0 ||
    value.maximumPollGapMilliseconds >
      OPT_0081_REPRESENTATIVE_MAXIMUM_THERMAL_GAP_MS ||
    !nonNegativeInteger(value.nonNominalObservationCount) ||
    value.nonNominalObservationCount > value.observationCount ||
    value.missingObservationCount !== 0 ||
    !thermalLevel(value.initialLevel) || !thermalLevel(value.finalLevel) ||
    !transitionsValid || !gateTraceConsistent ||
    !nonNominalCountConsistent || value.coversCleanup !== true
  ) throw new Error("OPT-0081 representative through-cleanup trace failed");
  return value;
}

export function opt0081RepresentativeThermalTimingPassed(
  value: Opt0081RepresentativeThermalCompletion,
  launch: Opt0081RepresentativeThermalLaunch,
  cleanupCompletedAtEpochMilliseconds: number,
): boolean {
  requireOpt0081RepresentativeThermalCompletion(
    value,
    launch,
    cleanupCompletedAtEpochMilliseconds,
  );
  return true;
}

function parseThermalTransitions(value: string): readonly Readonly<{
  readonly atEpochMilliseconds: number;
  readonly level: number;
}>[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error("OPT-0081 thermal transitions must be valid JSON");
  }
  if (!Array.isArray(parsed)) {
    throw new Error("OPT-0081 thermal transitions must be a JSON array");
  }
  return Object.freeze(parsed.map((entry) => {
    if (
      typeof entry !== "object" || entry === null ||
      !("atEpochMilliseconds" in entry) || !("level" in entry)
    ) throw new Error("OPT-0081 thermal transition entry changed");
    return Object.freeze({
      atEpochMilliseconds: Number(entry.atEpochMilliseconds),
      level: Number(entry.level),
    });
  }));
}

function thermalLevel(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 0 && value <= 3;
}

function thermalLevelAt(
  initialLevel: number,
  transitions: readonly Readonly<{
    readonly atEpochMilliseconds: number;
    readonly level: number;
  }>[],
  atEpochMilliseconds: number,
): number {
  let level = initialLevel;
  for (const transition of transitions) {
    if (transition.atEpochMilliseconds > atEpochMilliseconds) break;
    level = transition.level;
  }
  return level;
}

function checkpointInventory(
  checkpoints: readonly Opt0081RepresentativeRawCheckpoint[],
  roles: readonly Readonly<{ readonly tap: string; readonly words: number }>[],
  expectedTotalWords: number,
): boolean {
  if (checkpoints.length !== roles.length * 2) return false;
  let totalWords = 0;
  for (let layer = 0; layer < 2; layer += 1) {
    for (let roleIndex = 0; roleIndex < roles.length; roleIndex += 1) {
      const checkpoint = checkpoints[layer * roles.length + roleIndex];
      const role = roles[roleIndex]!;
      if (
        checkpoint === undefined || checkpoint.layer !== layer ||
        checkpoint.tap !== role.tap || checkpoint.comparedWords !== role.words ||
        !nonNegativeInteger(checkpoint.differingWordCount) ||
        !nonNegativeInteger(checkpoint.unwrittenWordCount) ||
        checkpoint.exact !== (checkpoint.differingWordCount === 0) ||
        !/^[0-9a-f]{64}$/u.test(checkpoint.sha256)
      ) return false;
      totalWords += checkpoint.comparedWords;
    }
  }
  return totalWords === expectedTotalWords;
}

function checkpointPassed(value: Opt0081RepresentativeRawCheckpoint): boolean {
  return value.exact && value.differingWordCount === 0 &&
    value.unwrittenWordCount === 0 && value.signedZeroExact && value.finite &&
    value.qNaNPrefillOverwritten && value.firstWordCovered &&
    value.lastWordCovered && value.tailRows2240Through2249Covered &&
    value.prefixGuardIntact && value.suffixGuardIntact &&
    value.adjacentGuardsIntact;
}

function topologyKey(value: Opt0081RepresentativeTopology): string {
  return JSON.stringify(value);
}

function mean(values: readonly number[]): number {
  if (values.length === 0) throw new Error("Cannot average an empty sample");
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: readonly number[]): number {
  if (values.length === 0) throw new Error("Cannot median an empty sample");
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function sampleStandardDeviation(values: readonly number[]): number {
  if (values.length < 2) {
    throw new Error("Sample standard deviation requires at least two values");
  }
  const average = mean(values);
  const squaredDeviationSum = values.reduce((sum, value) =>
    sum + (value - average) ** 2, 0);
  return Math.sqrt(squaredDeviationSum / (values.length - 1));
}

function invalidDuration(value: number): boolean {
  return !Number.isFinite(value) || value < 0;
}

function nonempty(value: string): boolean {
  return value.trim().length > 0;
}

function nonNegativeInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 0;
}

function positiveInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function sameStrings(
  left: readonly string[],
  right: readonly string[],
): boolean {
  return left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function requireWithoutThrow(run: () => void): boolean {
  try {
    run();
    return true;
  } catch {
    return false;
  }
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value === "") {
    throw new Error(`Missing OPT-0081 representative parameter ${name}`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`Invalid OPT-0081 representative parameter ${name}`);
  }
  return value;
}

function requiredIntegerParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = requiredFiniteParameter(parameters, name);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`Invalid OPT-0081 representative integer ${name}`);
  }
  return value;
}
