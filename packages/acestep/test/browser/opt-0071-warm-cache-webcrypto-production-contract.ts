export const OPT_0071_EXPERIMENT_ID = "OPT-0071" as const;

export const OPT_0071_MAIN_MANIFEST_PATH =
  "/model/files-reference/manifest.json" as const;
export const OPT_0071_MAIN_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6" as const;
export const OPT_0071_DENSE_MANIFEST_PATH =
  "/model/files-fp16-dit-rev7-oracle/manifest.json" as const;
export const OPT_0071_DENSE_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f" as const;
export const OPT_0071_VAE_MANIFEST_PATH =
  "/model/files-fp16-vae-revision7-experimental/manifest.json" as const;
export const OPT_0071_VAE_MANIFEST_SHA256 =
  "36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7" as const;

export const OPT_0071_FULL_LOGICAL_RECORDS = 158 as const;
export const OPT_0071_FULL_UNIQUE_DIGESTS = 156 as const;
export const OPT_0071_FULL_LOGICAL_BYTES = 7_330_447_819 as const;
export const OPT_0071_FULL_PHYSICAL_BYTES = 7_325_999_133 as const;
export const OPT_0071_TIMED_LOGICAL_RECORDS = 151 as const;
export const OPT_0071_TIMED_UNIQUE_DIGESTS = 149 as const;
export const OPT_0071_TIMED_LOGICAL_BYTES = 7_161_656_267 as const;
export const OPT_0071_TIMED_PHYSICAL_BYTES = 7_157_207_581 as const;
export const OPT_0071_LARGEST_FILE_BYTES = 121_668_608 as const;
export const OPT_0071_SCALAR_HASH_CHUNK_BYTES = 4 * 1024 * 1024;
export const OPT_0071_MAXIMUM_FILE_BYTES = 128 * 1024 * 1024;
export const OPT_0071_CONSERVATIVE_TRANSIENT_BYTES =
  OPT_0071_LARGEST_FILE_BYTES * 3;
export const OPT_0071_MAXIMUM_TRANSIENT_BYTES = 384 * 1024 * 1024;
export const OPT_0071_UPLOAD_SUBSET_FILES = 102 as const;
export const OPT_0071_UPLOAD_SUBSET_BYTES = 5_731_837_696 as const;
export const OPT_0071_INVENTORY_FINGERPRINT =
  "5ca8331a5f99bd9825bcf28f446549e8b30000e3316264025a1a7bfa93ab94a2" as const;

export const OPT_0071_MINIMUM_READY_SAVING_MS = 15_000 as const;
export const OPT_0071_MINIMUM_AUTHENTICATION_SAVING_MS = 15_000 as const;
export const OPT_0071_MAXIMUM_CANDIDATE_AUTHENTICATION_MEDIAN_MS = 8_000 as const;
export const OPT_0071_MINIMUM_AUTHENTICATION_THROUGHPUT_BYTES_PER_SECOND =
  915_749_892 as const;
export const OPT_0071_MAXIMUM_UNRELATED_REGRESSION = 0.02 as const;

export const OPT_0071_THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
export const OPT_0071_THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
export const OPT_0071_MINIMUM_NOMINAL_MILLISECONDS = 30_000 as const;
export const OPT_0071_MAXIMUM_POLL_GAP_MILLISECONDS = 1_250 as const;
export const OPT_0071_MAXIMUM_GATE_HANDOFF_MILLISECONDS = 5_000 as const;

export type Opt0071Owner = "scalar-stream" | "webcrypto-whole-file";
export type Opt0071ArmId = "A1" | "B1" | "B2" | "A2";

export const OPT_0071_ARM_ORDER = Object.freeze([
  Object.freeze({
    armId: "A1" as const,
    order: 0 as const,
    owner: "scalar-stream" as const,
  }),
  Object.freeze({
    armId: "B1" as const,
    order: 1 as const,
    owner: "webcrypto-whole-file" as const,
  }),
  Object.freeze({
    armId: "B2" as const,
    order: 2 as const,
    owner: "webcrypto-whole-file" as const,
  }),
  Object.freeze({
    armId: "A2" as const,
    order: 3 as const,
    owner: "scalar-stream" as const,
  }),
]);

export const OPT_0071_UNRELATED_STAGE_NAMES = Object.freeze([
  "device-request",
  "opfs-open",
  "stale-audio-recovery",
  "main-manifest-authentication",
  "dit-dense-manifest-authentication",
  "vae-manifest-authentication",
  "text-tokenizer-load",
  "planner-tokenizer-load",
  "ready-publication-and-residual",
] as const);

export type Opt0071UnrelatedStageName =
  (typeof OPT_0071_UNRELATED_STAGE_NAMES)[number];

export interface Opt0071RunIdentity {
  readonly coreCommit: string;
  readonly harnessCommit: string;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly cpuCoreCount: number;
  readonly memoryBytes: number;
}

export function parseOpt0071RunIdentity(
  parameters: URLSearchParams,
): Opt0071RunIdentity {
  return validateOpt0071RunIdentity({
    coreCommit: requiredString(parameters, "coreCommit"),
    harnessCommit: requiredString(parameters, "harnessCommit"),
    machineModel: requiredString(parameters, "machineModel"),
    osVersion: requiredString(parameters, "osVersion"),
    osBuild: requiredString(parameters, "osBuild"),
    browserVersion: requiredString(parameters, "browserVersion"),
    cpuCoreCount: requiredInteger(parameters, "cpuCoreCount"),
    memoryBytes: requiredInteger(parameters, "memoryBytes"),
  });
}

export function validateOpt0071RunIdentity(
  value: Opt0071RunIdentity,
): Opt0071RunIdentity {
  if (
    !isRecord(value) || !isCommit(value.coreCommit) ||
    !isCommit(value.harnessCommit) || !isNonempty(value.machineModel) ||
    !isNonempty(value.osVersion) || !isNonempty(value.osBuild) ||
    !isNonempty(value.browserVersion) ||
    !Number.isSafeInteger(value.cpuCoreCount) || value.cpuCoreCount < 1 ||
    !Number.isSafeInteger(value.memoryBytes) || value.memoryBytes < 1
  ) throw new Error("OPT-0071 requires a complete frozen run identity");
  return Object.freeze({ ...value });
}

export interface Opt0071ThermalObservation {
  readonly atEpochMilliseconds: number;
  readonly level: number;
  readonly rawValue: string;
}

export interface Opt0071ThermalGate {
  readonly source: typeof OPT_0071_THERMAL_SOURCE;
  readonly command: typeof OPT_0071_THERMAL_COMMAND;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly observations: readonly Opt0071ThermalObservation[];
}

export interface Opt0071ThermalTrace {
  readonly source: typeof OPT_0071_THERMAL_SOURCE;
  readonly command: typeof OPT_0071_THERMAL_COMMAND;
  readonly rawTraceSha256: string;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: number;
  readonly observations: readonly Opt0071ThermalObservation[];
  readonly transitions: readonly Readonly<{
    readonly atEpochMilliseconds: number;
    readonly level: number;
  }>[];
}

export function requireOpt0071ThermalGate(
  value: Opt0071ThermalGate,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0071ThermalGate {
  const duration = value.completedAtEpochMilliseconds -
    value.startedAtEpochMilliseconds;
  const inspection = inspectObservations(
    value.observations,
    value.startedAtEpochMilliseconds,
    value.completedAtEpochMilliseconds,
  );
  if (
    value.source !== OPT_0071_THERMAL_SOURCE ||
    value.command !== OPT_0071_THERMAL_COMMAND ||
    !Number.isFinite(readyAtEpochMilliseconds) ||
    value.startedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    value.completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    nowEpochMilliseconds - value.completedAtEpochMilliseconds >
      OPT_0071_MAXIMUM_GATE_HANDOFF_MILLISECONDS ||
    duration < OPT_0071_MINIMUM_NOMINAL_MILLISECONDS ||
    value.observationCount !== value.observations.length ||
    value.observationCount < Math.floor(duration / 1_000) + 1 ||
    value.maximumPollGapMilliseconds !== inspection.maximumPollGapMilliseconds ||
    value.maximumPollGapMilliseconds > OPT_0071_MAXIMUM_POLL_GAP_MILLISECONDS ||
    value.nonNominalObservationCount !== 0 ||
    inspection.nonNominalObservationCount !== 0
  ) throw new Error("OPT-0071 rejected the arm thermal gate");
  return Object.freeze({
    ...value,
    observations: freezeObservations(value.observations),
  });
}

export function requireOpt0071ThermalTrace(
  value: Opt0071ThermalTrace,
  gate: Opt0071ThermalGate,
  workerTerminatedAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0071ThermalTrace {
  const inspection = inspectObservations(
    value.observations,
    gate.startedAtEpochMilliseconds,
    value.completedAtEpochMilliseconds,
  );
  if (
    value.source !== OPT_0071_THERMAL_SOURCE ||
    value.command !== OPT_0071_THERMAL_COMMAND ||
    !/^[0-9a-f]{64}$/u.test(value.rawTraceSha256) ||
    value.completedAtEpochMilliseconds < workerTerminatedAtEpochMilliseconds ||
    value.completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    value.observationCount !== value.observations.length ||
    value.observationCount < gate.observationCount ||
    value.maximumPollGapMilliseconds !== inspection.maximumPollGapMilliseconds ||
    value.maximumPollGapMilliseconds > OPT_0071_MAXIMUM_POLL_GAP_MILLISECONDS ||
    value.nonNominalObservationCount !== inspection.nonNominalObservationCount ||
    gate.observations.some((item, index) =>
      !sameObservation(item, value.observations[index])
    ) ||
    value.transitions.length === 0 ||
    value.transitions.some((item) =>
      !Number.isSafeInteger(item.atEpochMilliseconds) ||
      !Number.isSafeInteger(item.level) || item.level < 0 ||
      item.atEpochMilliseconds < gate.startedAtEpochMilliseconds ||
      item.atEpochMilliseconds > value.completedAtEpochMilliseconds
    )
  ) throw new Error("OPT-0071 rejected the through-termination thermal trace");
  return Object.freeze({
    ...value,
    observations: freezeObservations(value.observations),
    transitions: Object.freeze(value.transitions.map((item) =>
      Object.freeze({ ...item })
    )),
  });
}

export type Opt0071UnrelatedStageWalls = Readonly<Record<
  Opt0071UnrelatedStageName,
  number
>>;

export interface Opt0071ArmSample {
  readonly armId: Opt0071ArmId;
  readonly order: number;
  readonly owner: Opt0071Owner;
  readonly readyWallMs: number;
  readonly authenticationWallMs: number;
  readonly authenticationThroughputBytesPerSecond: number;
  readonly unrelatedStageWalls: Opt0071UnrelatedStageWalls;
  readonly aggregateUnrelatedWallMs: number;
  readonly timedLogicalRecords: number;
  readonly timedUniqueDigests: number;
  readonly timedLogicalBytes: number;
  readonly timedPhysicalBytes: number;
  readonly timedInventoryFingerprint: string;
  readonly fullLogicalRecordsProven: number;
  readonly fullUniqueDigestsProven: number;
  readonly fullLogicalBytesProven: number;
  readonly fullPhysicalBytesProven: number;
  readonly inventoryFingerprint: string;
  readonly maximumExplicitLivePayloadBytes: number;
  readonly maximumExplicitLivePayloadCount: number;
  readonly conservativeTransientBytes: number;
  readonly downloadCount: number;
  readonly downloadBytes: number;
  readonly cacheMutationCount: number;
  readonly exactDigestsPassed: boolean;
  readonly memoryPassed: boolean;
  readonly cancellationPassed: boolean;
  readonly lifecyclePassed: boolean;
  readonly thermalNonNominalObservations: number;
}

export interface Opt0071StageComparison {
  readonly pair: "A1-to-B1" | "A2-to-B2";
  readonly stage: Opt0071UnrelatedStageName | "aggregate-unrelated";
  readonly controlMs: number;
  readonly candidateMs: number;
  readonly absoluteDeltaMs: number;
  readonly regressionRatio: number;
  readonly passed: boolean;
}

export interface Opt0071PerformanceSummary {
  readonly candidateAuthenticationMedianMs: number;
  readonly candidateAuthenticationThroughputBytesPerSecond: number;
  readonly a1MinusB1ReadySavingMs: number;
  readonly a2MinusB2ReadySavingMs: number;
  readonly a1MinusB1AuthenticationSavingMs: number;
  readonly a2MinusB2AuthenticationSavingMs: number;
  readonly unrelatedStageComparisons: readonly Opt0071StageComparison[];
  readonly maximumUnrelatedRegression: number;
  readonly allSamplesExactAndSafe: boolean;
  readonly allSamplesNominal: boolean;
  readonly passed: boolean;
}

export function summarizeOpt0071Performance(
  samples: readonly Opt0071ArmSample[],
): Opt0071PerformanceSummary {
  if (
    samples.length !== OPT_0071_ARM_ORDER.length ||
    samples.some((sample, index) => {
      const expected = OPT_0071_ARM_ORDER[index]!;
      return sample.armId !== expected.armId || sample.order !== expected.order ||
        sample.owner !== expected.owner ||
        numericSampleFields(sample).some((value) =>
          !Number.isFinite(value) || value < 0
        ) ||
        Object.keys(sample.unrelatedStageWalls).length !==
          OPT_0071_UNRELATED_STAGE_NAMES.length ||
        OPT_0071_UNRELATED_STAGE_NAMES.some((stage) =>
          !Number.isFinite(sample.unrelatedStageWalls[stage]) ||
          sample.unrelatedStageWalls[stage] < 0
        ) || !/^[0-9a-f]{64}$/u.test(sample.timedInventoryFingerprint);
    })
  ) throw new Error("OPT-0071 balanced sample inventory changed");
  const byId = new Map(samples.map((sample) => [sample.armId, sample]));
  const a1 = byId.get("A1")!;
  const b1 = byId.get("B1")!;
  const b2 = byId.get("B2")!;
  const a2 = byId.get("A2")!;
  const candidateAuthenticationMedianMs = median([
    b1.authenticationWallMs,
    b2.authenticationWallMs,
  ]);
  const candidateAuthenticationThroughputBytesPerSecond =
    OPT_0071_TIMED_PHYSICAL_BYTES /
    (candidateAuthenticationMedianMs / 1_000);
  const comparisons = Object.freeze([
    ...stageComparisons("A1-to-B1", a1, b1),
    ...stageComparisons("A2-to-B2", a2, b2),
  ]);
  const maximumUnrelatedRegression = Math.max(
    0,
    ...comparisons.map((item) => item.regressionRatio),
  );
  const allSamplesExactAndSafe = samples.every((sample) =>
    sample.timedLogicalRecords === OPT_0071_TIMED_LOGICAL_RECORDS &&
    sample.timedUniqueDigests === OPT_0071_TIMED_UNIQUE_DIGESTS &&
    sample.timedLogicalBytes === OPT_0071_TIMED_LOGICAL_BYTES &&
    sample.timedPhysicalBytes === OPT_0071_TIMED_PHYSICAL_BYTES &&
    sample.timedInventoryFingerprint === samples[0]!.timedInventoryFingerprint &&
    sample.fullLogicalRecordsProven === OPT_0071_FULL_LOGICAL_RECORDS &&
    sample.fullUniqueDigestsProven === OPT_0071_FULL_UNIQUE_DIGESTS &&
    sample.fullLogicalBytesProven === OPT_0071_FULL_LOGICAL_BYTES &&
    sample.fullPhysicalBytesProven === OPT_0071_FULL_PHYSICAL_BYTES &&
    sample.inventoryFingerprint === OPT_0071_INVENTORY_FINGERPRINT &&
    (sample.owner === "scalar-stream"
      ? sample.maximumExplicitLivePayloadBytes <=
          OPT_0071_SCALAR_HASH_CHUNK_BYTES &&
        sample.conservativeTransientBytes ===
          OPT_0071_SCALAR_HASH_CHUNK_BYTES
      : sample.maximumExplicitLivePayloadBytes ===
          OPT_0071_LARGEST_FILE_BYTES &&
        sample.conservativeTransientBytes ===
          OPT_0071_CONSERVATIVE_TRANSIENT_BYTES) &&
    sample.maximumExplicitLivePayloadCount === 1 &&
    sample.conservativeTransientBytes < OPT_0071_MAXIMUM_TRANSIENT_BYTES &&
    sample.downloadCount === 0 && sample.downloadBytes === 0 &&
    sample.cacheMutationCount === 0 && sample.exactDigestsPassed &&
    sample.memoryPassed && sample.cancellationPassed && sample.lifecyclePassed
  );
  const allSamplesNominal = samples.every((sample) =>
    sample.thermalNonNominalObservations === 0
  );
  const a1MinusB1ReadySavingMs = a1.readyWallMs - b1.readyWallMs;
  const a2MinusB2ReadySavingMs = a2.readyWallMs - b2.readyWallMs;
  const a1MinusB1AuthenticationSavingMs =
    a1.authenticationWallMs - b1.authenticationWallMs;
  const a2MinusB2AuthenticationSavingMs =
    a2.authenticationWallMs - b2.authenticationWallMs;
  const passed = allSamplesExactAndSafe && allSamplesNominal &&
    a1MinusB1ReadySavingMs >= OPT_0071_MINIMUM_READY_SAVING_MS &&
    a2MinusB2ReadySavingMs >= OPT_0071_MINIMUM_READY_SAVING_MS &&
    a1MinusB1AuthenticationSavingMs >=
      OPT_0071_MINIMUM_AUTHENTICATION_SAVING_MS &&
    a2MinusB2AuthenticationSavingMs >=
      OPT_0071_MINIMUM_AUTHENTICATION_SAVING_MS &&
    candidateAuthenticationMedianMs <=
      OPT_0071_MAXIMUM_CANDIDATE_AUTHENTICATION_MEDIAN_MS &&
    candidateAuthenticationThroughputBytesPerSecond >=
      OPT_0071_MINIMUM_AUTHENTICATION_THROUGHPUT_BYTES_PER_SECOND &&
    comparisons.every((item) => item.passed);
  return Object.freeze({
    candidateAuthenticationMedianMs,
    candidateAuthenticationThroughputBytesPerSecond,
    a1MinusB1ReadySavingMs,
    a2MinusB2ReadySavingMs,
    a1MinusB1AuthenticationSavingMs,
    a2MinusB2AuthenticationSavingMs,
    unrelatedStageComparisons: comparisons,
    maximumUnrelatedRegression,
    allSamplesExactAndSafe,
    allSamplesNominal,
    passed,
  });
}

export function checkedOpt0071ByteAdd(left: number, right: number): number {
  const value = left + right;
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError("OPT-0071 byte accounting exceeds the safe range");
  }
  return value;
}

/** Timed capture walls are finite fractional milliseconds, not byte counts. */
export function checkedOpt0071DurationAdd(
  left: number,
  right: number,
): number {
  const value = left + right;
  if (
    !Number.isFinite(left) || left < 0 ||
    !Number.isFinite(right) || right < 0 ||
    !Number.isFinite(value) || value < 0
  ) {
    throw new RangeError("OPT-0071 duration accounting is invalid");
  }
  return value;
}

export function serializeOpt0071Failure(
  error: unknown,
): Readonly<Record<string, unknown>> {
  const value = error as { name?: unknown; message?: unknown; stack?: unknown };
  return Object.freeze({
    name: boundedText(typeof value?.name === "string" ? value.name : "Error", 512),
    message: boundedText(
      typeof value?.message === "string" ? value.message : String(error),
      4_096,
    ),
    ...(typeof value?.stack === "string"
      ? { stack: boundedText(value.stack, 8_192) }
      : {}),
  });
}

function stageComparisons(
  pair: Opt0071StageComparison["pair"],
  control: Opt0071ArmSample,
  candidate: Opt0071ArmSample,
): readonly Opt0071StageComparison[] {
  return Object.freeze([
    ...OPT_0071_UNRELATED_STAGE_NAMES.map((stage) => comparison(
      pair,
      stage,
      control.unrelatedStageWalls[stage],
      candidate.unrelatedStageWalls[stage],
    )),
    comparison(
      pair,
      "aggregate-unrelated",
      control.aggregateUnrelatedWallMs,
      candidate.aggregateUnrelatedWallMs,
    ),
  ]);
}

function comparison(
  pair: Opt0071StageComparison["pair"],
  stage: Opt0071StageComparison["stage"],
  controlMs: number,
  candidateMs: number,
): Opt0071StageComparison {
  const regressionRatio = regression(controlMs, candidateMs);
  return Object.freeze({
    pair,
    stage,
    controlMs,
    candidateMs,
    absoluteDeltaMs: candidateMs - controlMs,
    regressionRatio,
    passed: regressionRatio <= OPT_0071_MAXIMUM_UNRELATED_REGRESSION,
  });
}

function numericSampleFields(sample: Opt0071ArmSample): readonly number[] {
  return [
    sample.readyWallMs,
    sample.authenticationWallMs,
    sample.authenticationThroughputBytesPerSecond,
    sample.aggregateUnrelatedWallMs,
    sample.timedLogicalRecords,
    sample.timedUniqueDigests,
    sample.timedLogicalBytes,
    sample.timedPhysicalBytes,
    sample.fullLogicalRecordsProven,
    sample.fullUniqueDigestsProven,
    sample.fullLogicalBytesProven,
    sample.fullPhysicalBytesProven,
    sample.maximumExplicitLivePayloadBytes,
    sample.maximumExplicitLivePayloadCount,
    sample.conservativeTransientBytes,
    sample.downloadCount,
    sample.downloadBytes,
    sample.cacheMutationCount,
    sample.thermalNonNominalObservations,
  ];
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  if (sorted.length === 0) throw new Error("OPT-0071 median requires samples");
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function regression(control: number, candidate: number): number {
  return control === 0
    ? candidate === 0 ? 0 : Number.POSITIVE_INFINITY
    : candidate / control - 1;
}

function inspectObservations(
  observations: readonly Opt0071ThermalObservation[],
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
  ) throw new Error("OPT-0071 thermal observation boundary changed");
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
    ) throw new Error("OPT-0071 thermal observation is invalid");
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

function freezeObservations(
  observations: readonly Opt0071ThermalObservation[],
): readonly Opt0071ThermalObservation[] {
  return Object.freeze(observations.map((item) => Object.freeze({ ...item })));
}

function sameObservation(
  left: Opt0071ThermalObservation,
  right: Opt0071ThermalObservation | undefined,
): boolean {
  return right !== undefined &&
    left.atEpochMilliseconds === right.atEpochMilliseconds &&
    left.level === right.level && left.rawValue === right.rawValue;
}

function requiredString(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name)?.trim();
  if (value === undefined || value === "") {
    throw new Error(`OPT-0071 requires URL parameter ${name}`);
  }
  return value;
}

function requiredInteger(parameters: URLSearchParams, name: string): number {
  const text = requiredString(parameters, name);
  if (!/^[0-9]+$/u.test(text)) {
    throw new Error(`OPT-0071 URL parameter ${name} must be an integer`);
  }
  const value = Number(text);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`OPT-0071 URL parameter ${name} is outside the safe range`);
  }
  return value;
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isCommit(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{40}$/u.test(value);
}

function isNonempty(value: unknown): value is string {
  return typeof value === "string" && value.trim() !== "";
}

function boundedText(value: string, limit: number): string {
  return value.length <= limit ? value : `${value.slice(0, limit)}…`;
}
