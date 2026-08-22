import type {
  AcePlannerOpt0087InvocationDiagnostics,
} from "../../src/webgpu/planner-executor.js";
import type {
  AceOpt0087PlannerDenseArm,
} from "../../src/webgpu/planner-dense-owner.js";

export const OPT_0087_SCHEMA =
  "ace-opt-0087-planner-package-native-low-row-gemv-v1" as const;
export const OPT_0087_THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
export const OPT_0087_THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
export const OPT_0087_THERMAL_TRACE_SCHEMA =
  "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1" as const;
export const OPT_0087_THERMAL_POLL_MILLISECONDS = 1_000;
export const OPT_0087_THERMAL_POLL_TOLERANCE_MILLISECONDS = 250;
export const OPT_0087_MINIMUM_NOMINAL_MILLISECONDS = 30_000;
export const OPT_0087_PAIR_COUNT_PER_PATH = 8;
export const OPT_0087_TOTAL_PAIR_COUNT = 16;
export const OPT_0087_REQUIRED_CANDIDATE_WINS = 14;
export const OPT_0087_REQUIRED_LAYER_SPEEDUP = 1.5;
export const OPT_0087_REQUIRED_MEDIAN_SAVING_MILLISECONDS = 60;
export const OPT_0087_PROJECTION_DRAW_COUNT = 1_010;
export const OPT_0087_REQUIRED_PROJECTED_SAVING_SECONDS = 60;
export const OPT_0087_COLD_GENERIC_CONTROL_BUFFER_COUNT = 5 as const;
export const OPT_0087_COMPILE_CACHE_CONTROL_BUFFER_BYTES = 256 as const;
export const OPT_0087_COLD_GENERIC_CONTROL_TOTAL_BYTES = 1_280 as const;
export const OPT_0087_COLD_GENERIC_ARM_MAP_COUNT = 7 as const;
export const OPT_0087_NO_NEW_CONTROL_ARM_MAP_COUNT = 2 as const;

export type Opt0087Arm = "control" | "candidate";
export type Opt0087PathId = "cot-m1-middle-full" |
  "semantic-m2-middle-full";

export const OPT_0087_PATH_IDS = Object.freeze([
  "cot-m1-middle-full",
  "semantic-m2-middle-full",
] as const satisfies readonly Opt0087PathId[]);

/** Eight balanced orders per path; each arm runs first four times. */
export const OPT_0087_PAIR_ORDERS = Object.freeze([
  Object.freeze(["control", "candidate"] as const),
  Object.freeze(["candidate", "control"] as const),
  Object.freeze(["candidate", "control"] as const),
  Object.freeze(["control", "candidate"] as const),
  Object.freeze(["candidate", "control"] as const),
  Object.freeze(["control", "candidate"] as const),
  Object.freeze(["control", "candidate"] as const),
  Object.freeze(["candidate", "control"] as const),
]);

export interface Opt0087RunIdentity {
  readonly implementationCommit: string;
  readonly harnessCommit: string;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly gpuCoreCount: number;
  readonly memoryBytes: number;
}

export interface Opt0087ThermalLaunch {
  readonly source: typeof OPT_0087_THERMAL_SOURCE;
  readonly command: typeof OPT_0087_THERMAL_COMMAND;
  readonly traceStartedAtEpochMilliseconds: number;
  readonly gateStartedAtEpochMilliseconds: number;
  readonly gateCompletedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: typeof OPT_0087_THERMAL_POLL_MILLISECONDS;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly missingObservationCount: 0;
  readonly readyToGateDelayMilliseconds: number;
  readonly launchDelayMilliseconds: number;
}

export interface Opt0087ThermalTransition {
  readonly atEpochMilliseconds: number;
  readonly level: 0 | 1 | 2 | 3;
}

export interface Opt0087ThermalCompletion {
  readonly schema: typeof OPT_0087_THERMAL_TRACE_SCHEMA;
  readonly sha256: string;
  readonly byteLength: number;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: number;
  readonly missingObservationCount: 0;
  readonly initialLevel: 0;
  readonly finalLevel: 0 | 1 | 2 | 3;
  readonly transitions: readonly Opt0087ThermalTransition[];
  readonly coversCleanup: true;
  readonly laterNonNominalDisclosed: boolean;
}

export interface Opt0087ArmTimingSamples {
  readonly transformerLayerWallMilliseconds: readonly number[];
  readonly tiedHeadWallMilliseconds: readonly number[];
  readonly modelThroughReadbackWallMilliseconds: readonly number[];
  readonly completeTokenWallMilliseconds: readonly number[];
}

export interface Opt0087PathTimingInput {
  readonly id: Opt0087PathId;
  readonly control: Opt0087ArmTimingSamples;
  readonly candidate: Opt0087ArmTimingSamples;
}

export interface Opt0087TimingGateReceipt {
  readonly everyPathLayerMedianBelowControl: boolean;
  readonly everyPathHeadMedianBelowControl: boolean;
  readonly everyPathModelMedianBelowControl: boolean;
  readonly everyPathCompleteMedianBelowControl: boolean;
  readonly aggregateLayerSpeedup: number;
  readonly requiredLayerSpeedup: typeof OPT_0087_REQUIRED_LAYER_SPEEDUP;
  readonly candidateModelWins: number;
  readonly requiredCandidateWins: typeof OPT_0087_REQUIRED_CANDIDATE_WINS;
  readonly aggregateControlModelMedianMilliseconds: number;
  readonly aggregateCandidateModelMedianMilliseconds: number;
  readonly aggregateModelMedianSavingMilliseconds: number;
  readonly requiredMedianSavingMilliseconds:
    typeof OPT_0087_REQUIRED_MEDIAN_SAVING_MILLISECONDS;
  readonly projected1010DrawSavingSeconds: number;
  readonly requiredProjectedSavingSeconds:
    typeof OPT_0087_REQUIRED_PROJECTED_SAVING_SECONDS;
  readonly passed: boolean;
}

export interface Opt0087ExplicitArmResourceDelta {
  readonly createdBufferCount: number;
  readonly destroyedBufferCount: number;
  readonly createdByteLength: number;
  readonly destroyedByteLength: number;
  readonly successfulMapCount: number;
  readonly failedMapCount: number;
  readonly unmapCount: number;
  readonly destroyedWhileMappedCount: number;
  readonly repeatedDestroyCallCount: number;
  readonly liveBufferCountBefore: number;
  readonly liveBufferCountAfter: number;
  readonly liveByteLengthBefore: number;
  readonly liveByteLengthAfter: number;
  readonly activeMapCountBefore: number;
  readonly activeMapCountAfter: number;
}

export type Opt0087ExplicitArmResourceExpectation =
  | "cold-generic-a-compile-cache"
  | "no-new-compile-cache-buffer";

export interface Opt0087ResourceTopologyDiagnostic {
  readonly schema: "ace-opt-0087-explicit-arm-resource-mismatch-v1";
  readonly expectation: Opt0087ExplicitArmResourceExpectation;
  readonly expected: Readonly<Record<string, number | boolean | string>>;
  readonly actual: Opt0087ExplicitArmResourceDelta;
}

export class Opt0087ResourceTopologyError extends Error {
  readonly diagnostic: Opt0087ResourceTopologyDiagnostic;

  constructor(
    expectation: Opt0087ExplicitArmResourceExpectation,
    actual: Opt0087ExplicitArmResourceDelta,
  ) {
    const coldGeneric = expectation === "cold-generic-a-compile-cache";
    const expected = Object.freeze({
      runtimeCompileCacheControlBufferCount: coldGeneric
        ? OPT_0087_COLD_GENERIC_CONTROL_BUFFER_COUNT
        : 0,
      bytesPerCompileCacheControlBuffer:
        OPT_0087_COMPILE_CACHE_CONTROL_BUFFER_BYTES,
      createdBufferCount: coldGeneric
        ? OPT_0087_COLD_GENERIC_CONTROL_BUFFER_COUNT
        : 0,
      destroyedBufferCount: 0,
      createdByteLength: coldGeneric
        ? OPT_0087_COLD_GENERIC_CONTROL_TOTAL_BYTES
        : 0,
      destroyedByteLength: 0,
      successfulMapCount: coldGeneric
        ? OPT_0087_COLD_GENERIC_ARM_MAP_COUNT
        : OPT_0087_NO_NEW_CONTROL_ARM_MAP_COUNT,
      unmapCount: coldGeneric
        ? OPT_0087_COLD_GENERIC_ARM_MAP_COUNT
        : OPT_0087_NO_NEW_CONTROL_ARM_MAP_COUNT,
      evidenceReadbackMapCount: 2,
      mappedAtCreationCompileCacheControlCount: coldGeneric
        ? OPT_0087_COLD_GENERIC_CONTROL_BUFFER_COUNT
        : 0,
      liveBufferCountDelta: coldGeneric
        ? OPT_0087_COLD_GENERIC_CONTROL_BUFFER_COUNT
        : 0,
      liveByteLengthDelta: coldGeneric
        ? OPT_0087_COLD_GENERIC_CONTROL_TOTAL_BYTES
        : 0,
      resourceDeltaCompletesBeforeAuthoritativeModelWall: true,
      allocationOwner: coldGeneric
        ? "planner-runtime shape compile cache through owner cleanup"
        : "none; compile cache already warm or direct-B uses no such buffers",
    });
    const diagnostic = Object.freeze({
      schema: "ace-opt-0087-explicit-arm-resource-mismatch-v1" as const,
      expectation,
      expected,
      actual: Object.freeze({ ...actual }),
    });
    super(
      "OPT-0087 explicit-arm GPU resource topology changed: " +
        JSON.stringify(diagnostic),
    );
    this.name = "Opt0087ResourceTopologyError";
    this.diagnostic = diagnostic;
  }
}

export class Opt0087ResourcePairTopologyError extends Error {
  readonly diagnostic: Readonly<{
    schema: "ace-opt-0087-resource-pair-mismatch-v1";
    kind: "cold-warmup-a-to-direct-b" | "warmed-timed-pair";
    control: Opt0087ExplicitArmResourceDelta;
    candidate: Opt0087ExplicitArmResourceDelta;
  }>;

  constructor(
    kind: "cold-warmup-a-to-direct-b" | "warmed-timed-pair",
    control: Opt0087ExplicitArmResourceDelta,
    candidate: Opt0087ExplicitArmResourceDelta,
  ) {
    const diagnostic = Object.freeze({
      schema: "ace-opt-0087-resource-pair-mismatch-v1" as const,
      kind,
      control: Object.freeze({ ...control }),
      candidate: Object.freeze({ ...candidate }),
    });
    super(
      "OPT-0087 resource pair topology changed: " +
        JSON.stringify(diagnostic),
    );
    this.name = "Opt0087ResourcePairTopologyError";
    this.diagnostic = diagnostic;
  }
}

export function validateOpt0087ExplicitArmResources(
  receipt: Opt0087ExplicitArmResourceDelta,
  expectation: Opt0087ExplicitArmResourceExpectation,
): void {
  if (
    expectation !== "cold-generic-a-compile-cache" &&
    expectation !== "no-new-compile-cache-buffer"
  ) throw new TypeError("OPT-0087 resource expectation is invalid");
  const coldGeneric = expectation === "cold-generic-a-compile-cache";
  const createdBufferCount = coldGeneric
    ? OPT_0087_COLD_GENERIC_CONTROL_BUFFER_COUNT
    : 0;
  const createdByteLength = coldGeneric
    ? OPT_0087_COLD_GENERIC_CONTROL_TOTAL_BYTES
    : 0;
  const mapCount = coldGeneric
    ? OPT_0087_COLD_GENERIC_ARM_MAP_COUNT
    : OPT_0087_NO_NEW_CONTROL_ARM_MAP_COUNT;
  if (
    Object.values(receipt).some((value) => !Number.isSafeInteger(value)) ||
    receipt.createdBufferCount !== createdBufferCount ||
    receipt.destroyedBufferCount !== 0 ||
    receipt.createdByteLength !== createdByteLength ||
    receipt.destroyedByteLength !== 0 ||
    receipt.successfulMapCount !== mapCount ||
    receipt.failedMapCount !== 0 ||
    receipt.unmapCount !== mapCount ||
    receipt.destroyedWhileMappedCount !== 0 ||
    receipt.repeatedDestroyCallCount !== 0 ||
    receipt.liveBufferCountBefore <= 0 ||
    receipt.liveBufferCountAfter - receipt.liveBufferCountBefore !==
      createdBufferCount ||
    receipt.liveByteLengthBefore <= 0 ||
    receipt.liveByteLengthAfter - receipt.liveByteLengthBefore !==
      createdByteLength ||
    receipt.activeMapCountBefore !== 0 ||
    receipt.activeMapCountAfter !== 0
  ) throw new Opt0087ResourceTopologyError(expectation, receipt);
}

export function validateOpt0087ResourcePair(
  control: Opt0087ExplicitArmResourceDelta,
  candidate: Opt0087ExplicitArmResourceDelta,
  kind: "cold-warmup-a-to-direct-b" | "warmed-timed-pair",
): void {
  if (kind === "cold-warmup-a-to-direct-b") {
    validateOpt0087ExplicitArmResources(
      control,
      "cold-generic-a-compile-cache",
    );
    validateOpt0087ExplicitArmResources(
      candidate,
      "no-new-compile-cache-buffer",
    );
    if (
      candidate.liveBufferCountBefore !== control.liveBufferCountAfter ||
      candidate.liveByteLengthBefore !== control.liveByteLengthAfter
    ) throw new Opt0087ResourcePairTopologyError(kind, control, candidate);
    return;
  }
  if (kind !== "warmed-timed-pair") {
    throw new TypeError("OPT-0087 resource pair kind is invalid");
  }
  validateOpt0087ExplicitArmResources(
    control,
    "no-new-compile-cache-buffer",
  );
  validateOpt0087ExplicitArmResources(
    candidate,
    "no-new-compile-cache-buffer",
  );
  if (JSON.stringify(control) !== JSON.stringify(candidate)) {
    throw new Opt0087ResourcePairTopologyError(kind, control, candidate);
  }
}

export function opt0087DenseArmForArm(
  arm: Opt0087Arm,
): AceOpt0087PlannerDenseArm {
  return arm === "control" ? "generic-a" : "direct-b";
}

export function validateOpt0087Topology(
  diagnostics: AcePlannerOpt0087InvocationDiagnostics,
  expectedArm: AceOpt0087PlannerDenseArm,
  rows: 1 | 2,
): void {
  const expectedReason = expectedArm === "generic-a"
    ? "control-requested"
    : "direct-selected";
  const layerTimings = diagnostics.quantumTimings.filter(
    ({ kind }) => kind === "layer",
  );
  const headTimings = diagnostics.quantumTimings.filter(
    ({ kind }) => kind === "tied-lm-head",
  );
  if (
    diagnostics.schema !== "ace-opt-0087-planner-package-invocation-v1" ||
    diagnostics.arm !== expectedArm ||
    diagnostics.phaseKind !== "decode" ||
    diagnostics.modelQuantumCount !== 33 ||
    diagnostics.totalCommandBuffers !== 34 ||
    diagnostics.commandBuffersSubmitted !== 34 ||
    diagnostics.trueQueueDrainCount !== 34 ||
    diagnostics.cooperativeIdleTurns !== 34 ||
    diagnostics.requestedCooperativeIdleMs !== 34 ||
    diagnostics.maximumOutstandingCommandBuffers !== 1 ||
    diagnostics.readbackMapCount !== 2 ||
    diagnostics.readbackShardCount !== 5 ||
    !Number.isSafeInteger(diagnostics.readbackByteLength) ||
    diagnostics.readbackByteLength <= 0 ||
    diagnostics.cacheAppendReadbackByteLength !==
      (rows === 1 ? 229_632 : 459_008) ||
    diagnostics.cacheAppendLogicalByteLength !==
      (rows === 1 ? 229_380 : 458_760) ||
    diagnostics.cacheAppendCopyCount !== (rows === 1 ? 449 : 898) ||
    diagnostics.cacheAppendKeyValueWordCount !==
      (rows === 1 ? 57_344 : 114_688) ||
    diagnostics.cacheAppendValidityWordCount !== rows ||
    diagnostics.cacheAppendWords.length !==
      diagnostics.cacheAppendKeyValueWordCount + rows ||
    [...diagnostics.cacheAppendWords.subarray(
      diagnostics.cacheAppendKeyValueWordCount,
    )].some((word) => word !== 1) ||
    diagnostics.logitRows !== rows ||
    diagnostics.logitTokenCount !== 217_204 ||
    !Number.isSafeInteger(diagnostics.accountedGpuBytes) ||
    diagnostics.accountedGpuBytes <= 0 ||
    !Number.isSafeInteger(diagnostics.arenaBufferCount) ||
    diagnostics.arenaBufferCount <= 0 ||
    diagnostics.layerQuantumCount !== 28 ||
    diagnostics.tiedHeadQuantumCount !== 2 ||
    diagnostics.quantumTimings.length !== 33 ||
    layerTimings.length !== 28 ||
    layerTimings.some(({ layer }, index) => layer !== index) ||
    headTimings.length !== 2 ||
    diagnostics.denseSelections.length !== 28 * 7 + 5 ||
    diagnostics.denseSelections.some((selection) =>
      selection.requestedArm !== expectedArm ||
      selection.selectedArm !== expectedArm ||
      selection.reason !== expectedReason ||
      selection.rows !== rows
    ) ||
    !sameNestedNumbers(diagnostics.headQuantumSliceFirstRows, [
      [0, 49_152],
      [98_304, 147_456, 196_608],
    ]) ||
    diagnostics.writeStatusWords.length !== rows ||
    diagnostics.writeStatusWords.some((word) => word !== 1) ||
    !finitePositive([
      diagnostics.transformerLayerWallMilliseconds,
      diagnostics.tiedHeadWallMilliseconds,
      diagnostics.readbackWallMilliseconds,
      diagnostics.modelThroughReadbackWallMilliseconds,
      ...diagnostics.quantumTimings.map(
        ({ submitThroughDrainWallMilliseconds }) =>
          submitThroughDrainWallMilliseconds,
      ),
    ])
  ) throw new Error(`OPT-0087 ${expectedArm} M${rows} topology changed`);
}

export function evaluateOpt0087TimingGate(
  paths: readonly Opt0087PathTimingInput[],
): Opt0087TimingGateReceipt {
  if (
    paths.length !== OPT_0087_PATH_IDS.length ||
    paths.some((path, index) => path.id !== OPT_0087_PATH_IDS[index])
  ) throw new Error("OPT-0087 timing paths are incomplete or out of order");
  const controlLayers: number[] = [];
  const candidateLayers: number[] = [];
  const controlModel: number[] = [];
  const candidateModel: number[] = [];
  let candidateModelWins = 0;
  let everyPathLayerMedianBelowControl = true;
  let everyPathHeadMedianBelowControl = true;
  let everyPathModelMedianBelowControl = true;
  let everyPathCompleteMedianBelowControl = true;
  for (const path of paths) {
    validateArmSamples(path.control);
    validateArmSamples(path.candidate);
    const control = path.control;
    const candidate = path.candidate;
    controlLayers.push(...control.transformerLayerWallMilliseconds);
    candidateLayers.push(...candidate.transformerLayerWallMilliseconds);
    controlModel.push(...control.modelThroughReadbackWallMilliseconds);
    candidateModel.push(...candidate.modelThroughReadbackWallMilliseconds);
    everyPathLayerMedianBelowControl &&=
      median(candidate.transformerLayerWallMilliseconds) <
        median(control.transformerLayerWallMilliseconds);
    everyPathHeadMedianBelowControl &&=
      median(candidate.tiedHeadWallMilliseconds) <
        median(control.tiedHeadWallMilliseconds);
    everyPathModelMedianBelowControl &&=
      median(candidate.modelThroughReadbackWallMilliseconds) <
        median(control.modelThroughReadbackWallMilliseconds);
    everyPathCompleteMedianBelowControl &&=
      median(candidate.completeTokenWallMilliseconds) <
        median(control.completeTokenWallMilliseconds);
    for (let index = 0; index < OPT_0087_PAIR_COUNT_PER_PATH; index += 1) {
      if (
        candidate.modelThroughReadbackWallMilliseconds[index]! <
          control.modelThroughReadbackWallMilliseconds[index]!
      ) candidateModelWins += 1;
    }
  }
  if (
    controlModel.length !== OPT_0087_TOTAL_PAIR_COUNT ||
    candidateModel.length !== OPT_0087_TOTAL_PAIR_COUNT
  ) throw new Error("OPT-0087 aggregate timing pair count changed");
  const aggregateLayerSpeedup = median(controlLayers) / median(candidateLayers);
  const aggregateControlModelMedianMilliseconds = median(controlModel);
  const aggregateCandidateModelMedianMilliseconds = median(candidateModel);
  const aggregateModelMedianSavingMilliseconds =
    aggregateControlModelMedianMilliseconds -
      aggregateCandidateModelMedianMilliseconds;
  const projected1010DrawSavingSeconds =
    aggregateModelMedianSavingMilliseconds * OPT_0087_PROJECTION_DRAW_COUNT /
      1_000;
  const passed = everyPathLayerMedianBelowControl &&
    everyPathHeadMedianBelowControl &&
    everyPathModelMedianBelowControl &&
    everyPathCompleteMedianBelowControl &&
    aggregateLayerSpeedup >= OPT_0087_REQUIRED_LAYER_SPEEDUP &&
    candidateModelWins >= OPT_0087_REQUIRED_CANDIDATE_WINS &&
    aggregateModelMedianSavingMilliseconds >=
      OPT_0087_REQUIRED_MEDIAN_SAVING_MILLISECONDS &&
    projected1010DrawSavingSeconds >=
      OPT_0087_REQUIRED_PROJECTED_SAVING_SECONDS;
  return Object.freeze({
    everyPathLayerMedianBelowControl,
    everyPathHeadMedianBelowControl,
    everyPathModelMedianBelowControl,
    everyPathCompleteMedianBelowControl,
    aggregateLayerSpeedup,
    requiredLayerSpeedup: OPT_0087_REQUIRED_LAYER_SPEEDUP,
    candidateModelWins,
    requiredCandidateWins: OPT_0087_REQUIRED_CANDIDATE_WINS,
    aggregateControlModelMedianMilliseconds,
    aggregateCandidateModelMedianMilliseconds,
    aggregateModelMedianSavingMilliseconds,
    requiredMedianSavingMilliseconds:
      OPT_0087_REQUIRED_MEDIAN_SAVING_MILLISECONDS,
    projected1010DrawSavingSeconds,
    requiredProjectedSavingSeconds:
      OPT_0087_REQUIRED_PROJECTED_SAVING_SECONDS,
    passed,
  });
}

export function validateOpt0087RunIdentity(identity: unknown): Opt0087RunIdentity {
  if (typeof identity !== "object" || identity === null) {
    throw new Error("OPT-0087 requires a run identity");
  }
  const candidate = identity as Readonly<Record<string, unknown>>;
  const requiredString = (name: string): string => {
    const value = candidate[name];
    if (typeof value !== "string" || value.trim() === "" || value !== value.trim()) {
      throw new Error(`OPT-0087 requires run identity ${name}`);
    }
    return value;
  };
  const requiredPositiveInteger = (name: string): number => {
    const value = candidate[name];
    if (!Number.isSafeInteger(value) || (value as number) <= 0) {
      throw new Error(`OPT-0087 run identity ${name} must be positive`);
    }
    return value as number;
  };
  const implementationCommit = requiredString("implementationCommit");
  const harnessCommit = requiredString("harnessCommit");
  if (!/^[0-9a-f]{40}$/.test(implementationCommit)) {
    throw new Error("OPT-0087 implementationCommit must be 40 lowercase hex");
  }
  if (!/^[0-9a-f]{40}$/.test(harnessCommit)) {
    throw new Error("OPT-0087 harnessCommit must be 40 lowercase hex");
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

export function validateOpt0087ThermalLaunch(
  thermal: Opt0087ThermalLaunch,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds = Date.now(),
): void {
  const duration = thermal.gateCompletedAtEpochMilliseconds -
    thermal.gateStartedAtEpochMilliseconds;
  if (
    !Number.isSafeInteger(readyAtEpochMilliseconds) ||
    readyAtEpochMilliseconds <= 0 ||
    !Number.isSafeInteger(nowEpochMilliseconds) || nowEpochMilliseconds <= 0 ||
    !Number.isSafeInteger(thermal.traceStartedAtEpochMilliseconds) ||
    thermal.traceStartedAtEpochMilliseconds <= 0 ||
    !Number.isSafeInteger(thermal.gateStartedAtEpochMilliseconds) ||
    thermal.gateStartedAtEpochMilliseconds <= 0 ||
    !Number.isSafeInteger(thermal.gateCompletedAtEpochMilliseconds) ||
    thermal.gateCompletedAtEpochMilliseconds <= 0 ||
    !Number.isSafeInteger(thermal.readyToGateDelayMilliseconds) ||
    !Number.isSafeInteger(thermal.launchDelayMilliseconds) ||
    thermal.source !== OPT_0087_THERMAL_SOURCE ||
    thermal.command !== OPT_0087_THERMAL_COMMAND ||
    thermal.pollMilliseconds !== OPT_0087_THERMAL_POLL_MILLISECONDS ||
    thermal.traceStartedAtEpochMilliseconds > readyAtEpochMilliseconds ||
    thermal.gateStartedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    thermal.gateCompletedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    duration < OPT_0087_MINIMUM_NOMINAL_MILLISECONDS ||
    !Number.isSafeInteger(thermal.observationCount) ||
    thermal.observationCount < Math.floor(duration /
      OPT_0087_THERMAL_POLL_MILLISECONDS) + 1 ||
    !Number.isFinite(thermal.maximumPollGapMilliseconds) ||
    thermal.maximumPollGapMilliseconds < 0 ||
    thermal.maximumPollGapMilliseconds >
      OPT_0087_THERMAL_POLL_MILLISECONDS +
        OPT_0087_THERMAL_POLL_TOLERANCE_MILLISECONDS ||
    thermal.nonNominalObservationCount !== 0 ||
    thermal.missingObservationCount !== 0 ||
    thermal.readyToGateDelayMilliseconds !==
      thermal.gateStartedAtEpochMilliseconds - readyAtEpochMilliseconds ||
    thermal.launchDelayMilliseconds < 0 ||
    thermal.launchDelayMilliseconds > 5_000
  ) throw new Error("OPT-0087 fresh continuous nominal launch gate failed");
}

export function validateOpt0087ThermalCompletion(
  thermal: Opt0087ThermalCompletion,
  launch: Opt0087ThermalLaunch,
  cleanupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds = Date.now(),
): void {
  const minimumObservations = Math.floor(
    (thermal.completedAtEpochMilliseconds -
      launch.traceStartedAtEpochMilliseconds) /
        OPT_0087_THERMAL_POLL_MILLISECONDS,
  ) + 1;
  let currentLevel: 0 | 1 | 2 | 3 = thermal.initialLevel;
  let previousTransitionAt = launch.traceStartedAtEpochMilliseconds - 1;
  let transitionsValid = true;
  for (const transition of thermal.transitions) {
    if (
      !Number.isSafeInteger(transition.atEpochMilliseconds) ||
      transition.atEpochMilliseconds <= previousTransitionAt ||
      transition.atEpochMilliseconds < launch.traceStartedAtEpochMilliseconds ||
      transition.atEpochMilliseconds > thermal.completedAtEpochMilliseconds ||
      !Number.isSafeInteger(transition.level) ||
      transition.level < 0 || transition.level > 3 ||
      transition.level === currentLevel
    ) {
      transitionsValid = false;
      break;
    }
    previousTransitionAt = transition.atEpochMilliseconds;
    currentLevel = transition.level;
  }
  const gateStartedLevel = thermalLevelAt(
    thermal.initialLevel,
    thermal.transitions,
    launch.gateStartedAtEpochMilliseconds,
  );
  const gateTraceConsistent = gateStartedLevel === 0 &&
    thermal.transitions.every((transition) =>
      transition.atEpochMilliseconds < launch.gateStartedAtEpochMilliseconds ||
      transition.atEpochMilliseconds > launch.gateCompletedAtEpochMilliseconds
    );
  const disclosedNonNominal = thermal.initialLevel !== 0 ||
    thermal.transitions.some(({ level }) => level !== 0);
  if (
    !Number.isSafeInteger(cleanupCompletedAtEpochMilliseconds) ||
    cleanupCompletedAtEpochMilliseconds <= 0 ||
    !Number.isSafeInteger(nowEpochMilliseconds) || nowEpochMilliseconds <= 0 ||
    !Number.isSafeInteger(thermal.completedAtEpochMilliseconds) ||
    thermal.completedAtEpochMilliseconds <= 0 ||
    thermal.schema !== OPT_0087_THERMAL_TRACE_SCHEMA ||
    !/^[0-9a-f]{64}$/.test(thermal.sha256) ||
    !Number.isSafeInteger(thermal.byteLength) || thermal.byteLength <= 0 ||
    thermal.completedAtEpochMilliseconds < cleanupCompletedAtEpochMilliseconds ||
    thermal.completedAtEpochMilliseconds < launch.gateCompletedAtEpochMilliseconds ||
    thermal.completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    !Number.isSafeInteger(thermal.observationCount) ||
    thermal.observationCount < minimumObservations ||
    thermal.observationCount < launch.observationCount ||
    !Number.isFinite(thermal.maximumPollGapMilliseconds) ||
    thermal.maximumPollGapMilliseconds < launch.maximumPollGapMilliseconds ||
    thermal.maximumPollGapMilliseconds < 0 ||
    thermal.maximumPollGapMilliseconds >
      OPT_0087_THERMAL_POLL_MILLISECONDS +
        OPT_0087_THERMAL_POLL_TOLERANCE_MILLISECONDS ||
    !Number.isSafeInteger(thermal.nonNominalObservationCount) ||
    thermal.nonNominalObservationCount < 0 ||
    thermal.nonNominalObservationCount > thermal.observationCount ||
    thermal.missingObservationCount !== 0 ||
    thermal.initialLevel !== 0 ||
    !Number.isSafeInteger(thermal.finalLevel) ||
    thermal.finalLevel < 0 || thermal.finalLevel > 3 ||
    !transitionsValid || currentLevel !== thermal.finalLevel ||
    thermal.transitions.length > Math.max(0, thermal.observationCount - 1) ||
    !gateTraceConsistent ||
    (thermal.nonNominalObservationCount === 0) === disclosedNonNominal ||
    thermal.coversCleanup !== true ||
    thermal.laterNonNominalDisclosed !==
      (thermal.nonNominalObservationCount > 0)
  ) throw new Error("OPT-0087 complete through-cleanup thermal trace failed");
}

export function median(values: readonly number[]): number {
  if (values.length === 0) throw new Error("OPT-0087 cannot summarize no values");
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function validateArmSamples(samples: Opt0087ArmTimingSamples): void {
  for (const values of [
    samples.transformerLayerWallMilliseconds,
    samples.tiedHeadWallMilliseconds,
    samples.modelThroughReadbackWallMilliseconds,
    samples.completeTokenWallMilliseconds,
  ]) {
    if (
      values.length !== OPT_0087_PAIR_COUNT_PER_PATH ||
      !finitePositive(values)
    ) throw new Error("OPT-0087 timing samples are invalid");
  }
}

function finitePositive(values: readonly number[]): boolean {
  return values.every((value) => Number.isFinite(value) && value > 0);
}

function thermalLevelAt(
  initialLevel: 0 | 1 | 2 | 3,
  transitions: readonly Opt0087ThermalTransition[],
  atEpochMilliseconds: number,
): 0 | 1 | 2 | 3 {
  let level = initialLevel;
  for (const transition of transitions) {
    if (transition.atEpochMilliseconds > atEpochMilliseconds) break;
    level = transition.level;
  }
  return level;
}

function sameNestedNumbers(
  actual: readonly (readonly number[])[],
  expected: readonly (readonly number[])[],
): boolean {
  return actual.length === expected.length && actual.every((row, index) =>
    row.length === expected[index]!.length && row.every(
      (value, column) => value === expected[index]![column],
    )
  );
}
