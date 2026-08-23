export const OPT_0069_EXPERIMENT_ID = "OPT-0069" as const;

export const OPT_0069_MAIN_MANIFEST_PATH =
  "/model/files-reference/manifest.json" as const;
export const OPT_0069_MAIN_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6" as const;
export const OPT_0069_DENSE_MANIFEST_PATH =
  "/model/files-fp16-dit-rev7-oracle/manifest.json" as const;
export const OPT_0069_DENSE_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f" as const;
export const OPT_0069_VAE_MANIFEST_PATH =
  "/model/files-fp16-vae-experimental/manifest.json" as const;
export const OPT_0069_VAE_MANIFEST_SHA256 =
  "94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949" as const;

export const OPT_0069_HASH_CHUNK_BYTES = 4 * 1024 * 1024;
export const OPT_0069_WEBCRYPTO_MAXIMUM_FILE_BYTES = 128 * 1024 * 1024;
export const OPT_0069_WEBCRYPTO_TRANSIENT_MULTIPLIER = 3 as const;
export const OPT_0069_WEBCRYPTO_MAXIMUM_TRANSIENT_BYTES = 384 * 1024 * 1024;
export const OPT_0069_COMPLETE_LOGICAL_RECORDS = 158 as const;
export const OPT_0069_COMPLETE_UNIQUE_DIGESTS = 156 as const;
export const OPT_0069_COMPLETE_LOGICAL_BYTES = 7_330_447_819 as const;
export const OPT_0069_COMPLETE_PHYSICAL_BYTES = 7_325_999_133 as const;
export const OPT_0069_LARGEST_FILE_BYTES = 121_668_608 as const;
export const OPT_0069_UPLOAD_SUBSET_FILES = 102 as const;
export const OPT_0069_UPLOAD_SUBSET_BYTES = 5_731_837_696 as const;
export const OPT_0069_UPLOAD_SUBSET_REPORTED_GB = "5.7318 GB" as const;
export const OPT_0069_MAXIMUM_CANDIDATE_MEDIAN_MS = 8_000 as const;
export const OPT_0069_MINIMUM_DIRECTIONAL_SAVING_MS = 15_000 as const;
export const OPT_0069_MINIMUM_THROUGHPUT_BYTES_PER_SECOND =
  915_749_892 as const;
export const OPT_0069_MAXIMUM_READ_COPY_REGRESSION = 0.20 as const;
export const OPT_0069_MAXIMUM_RESPONSIVENESS_REGRESSION = 0.20 as const;

export const OPT_0069_THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
export const OPT_0069_THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
export const OPT_0069_THERMAL_POLL_MILLISECONDS = 1_000 as const;
export const OPT_0069_MINIMUM_NOMINAL_MILLISECONDS = 30_000 as const;
export const OPT_0069_MAXIMUM_POLL_GAP_MILLISECONDS = 1_250 as const;
export const OPT_0069_MAXIMUM_GATE_HANDOFF_MILLISECONDS = 5_000 as const;

export type Opt0069Owner = "scalar-stream" | "webcrypto-whole-file";
export type Opt0069ArmId =
  | "A1"
  | "B1"
  | "B2"
  | "A2"
  | "B3"
  | "A3"
  | "A4"
  | "B4";

export const OPT_0069_ARM_ORDER = Object.freeze([
  Object.freeze({ armId: "A1" as const, order: 0 as const,
    owner: "scalar-stream" as const }),
  Object.freeze({ armId: "B1" as const, order: 1 as const,
    owner: "webcrypto-whole-file" as const }),
  Object.freeze({ armId: "B2" as const, order: 2 as const,
    owner: "webcrypto-whole-file" as const }),
  Object.freeze({ armId: "A2" as const, order: 3 as const,
    owner: "scalar-stream" as const }),
  Object.freeze({ armId: "B3" as const, order: 4 as const,
    owner: "webcrypto-whole-file" as const }),
  Object.freeze({ armId: "A3" as const, order: 5 as const,
    owner: "scalar-stream" as const }),
  Object.freeze({ armId: "A4" as const, order: 6 as const,
    owner: "scalar-stream" as const }),
  Object.freeze({ armId: "B4" as const, order: 7 as const,
    owner: "webcrypto-whole-file" as const }),
]);

export interface Opt0069RunIdentity {
  readonly coreCommit: string;
  readonly harnessCommit: string;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly cpuCoreCount: number;
  readonly memoryBytes: number;
}

export function parseOpt0069RunIdentity(
  parameters: URLSearchParams,
): Opt0069RunIdentity {
  return validateOpt0069RunIdentity({
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

export function validateOpt0069RunIdentity(
  value: Opt0069RunIdentity,
): Opt0069RunIdentity {
  if (
    !isRecord(value) || !isCommit(value.coreCommit) ||
    !isCommit(value.harnessCommit) || !isNonempty(value.machineModel) ||
    !isNonempty(value.osVersion) || !isNonempty(value.osBuild) ||
    !isNonempty(value.browserVersion) ||
    !Number.isSafeInteger(value.cpuCoreCount) || value.cpuCoreCount < 1 ||
    !Number.isSafeInteger(value.memoryBytes) || value.memoryBytes < 1
  ) throw new Error("OPT-0069 requires a complete frozen run identity");
  return Object.freeze({ ...value });
}

export interface Opt0069PackageInventory {
  readonly packageKind: "main" | "dit-dense" | "vae";
  readonly logicalRecords: number;
  readonly uniqueDigests: number;
  readonly logicalBytes: number;
  readonly physicalBytes: number;
}

export interface Opt0069Inventory {
  readonly packages: readonly Opt0069PackageInventory[];
  readonly logicalRecords: number;
  readonly uniqueDigests: number;
  readonly logicalBytes: number;
  readonly physicalBytes: number;
  readonly largestFileBytes: number;
  readonly uploadSubsetFiles: number;
  readonly uploadSubsetBytes: number;
}

const EXPECTED_PACKAGES = Object.freeze([
  Object.freeze({
    packageKind: "main" as const,
    logicalRecords: 103,
    uniqueDigests: 101,
    logicalBytes: 4_140_848_075,
    physicalBytes: 4_136_399_389,
  }),
  Object.freeze({
    packageKind: "dit-dense" as const,
    logicalRecords: 48,
    uniqueDigests: 48,
    logicalBytes: 3_020_808_192,
    physicalBytes: 3_020_808_192,
  }),
  Object.freeze({
    packageKind: "vae" as const,
    logicalRecords: 7,
    uniqueDigests: 7,
    logicalBytes: 168_791_552,
    physicalBytes: 168_791_552,
  }),
]);

export function requireOpt0069Inventory(
  value: Opt0069Inventory,
): Opt0069Inventory {
  if (
    value.logicalRecords !== OPT_0069_COMPLETE_LOGICAL_RECORDS ||
    value.uniqueDigests !== OPT_0069_COMPLETE_UNIQUE_DIGESTS ||
    value.logicalBytes !== OPT_0069_COMPLETE_LOGICAL_BYTES ||
    value.physicalBytes !== OPT_0069_COMPLETE_PHYSICAL_BYTES ||
    value.largestFileBytes !== OPT_0069_LARGEST_FILE_BYTES ||
    value.uploadSubsetFiles !== OPT_0069_UPLOAD_SUBSET_FILES ||
    value.uploadSubsetBytes !== OPT_0069_UPLOAD_SUBSET_BYTES ||
    value.packages.length !== EXPECTED_PACKAGES.length ||
    value.packages.some((item, index) =>
      !samePackageInventory(item, EXPECTED_PACKAGES[index]!)
    )
  ) throw new Error("OPT-0069 authenticated inventory changed");
  return Object.freeze({
    ...value,
    packages: Object.freeze(value.packages.map((item) =>
      Object.freeze({ ...item })
    )),
  });
}

export interface Opt0069ThermalObservation {
  readonly atEpochMilliseconds: number;
  readonly level: number;
  readonly rawValue: string;
}

export interface Opt0069ThermalGate {
  readonly source: typeof OPT_0069_THERMAL_SOURCE;
  readonly command: typeof OPT_0069_THERMAL_COMMAND;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly observations: readonly Opt0069ThermalObservation[];
}

export interface Opt0069ThermalTrace {
  readonly source: typeof OPT_0069_THERMAL_SOURCE;
  readonly command: typeof OPT_0069_THERMAL_COMMAND;
  readonly rawTraceSha256: string;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: number;
  readonly observations: readonly Opt0069ThermalObservation[];
  readonly transitions: readonly Readonly<{
    readonly atEpochMilliseconds: number;
    readonly level: number;
  }>[];
}

export function requireOpt0069ThermalGate(
  value: Opt0069ThermalGate,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0069ThermalGate {
  const duration = value.completedAtEpochMilliseconds -
    value.startedAtEpochMilliseconds;
  const inspection = inspectObservations(
    value.observations,
    value.startedAtEpochMilliseconds,
    value.completedAtEpochMilliseconds,
  );
  if (
    value.source !== OPT_0069_THERMAL_SOURCE ||
    value.command !== OPT_0069_THERMAL_COMMAND ||
    !Number.isFinite(readyAtEpochMilliseconds) ||
    value.startedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    value.completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    nowEpochMilliseconds - value.completedAtEpochMilliseconds >
      OPT_0069_MAXIMUM_GATE_HANDOFF_MILLISECONDS ||
    duration < OPT_0069_MINIMUM_NOMINAL_MILLISECONDS ||
    !Number.isSafeInteger(value.observationCount) ||
    value.observationCount <
      Math.floor(duration / OPT_0069_THERMAL_POLL_MILLISECONDS) + 1 ||
    value.observations.length !== value.observationCount ||
    value.maximumPollGapMilliseconds !== inspection.maximumPollGapMilliseconds ||
    value.maximumPollGapMilliseconds > OPT_0069_MAXIMUM_POLL_GAP_MILLISECONDS ||
    value.nonNominalObservationCount !== 0 ||
    inspection.nonNominalObservationCount !== 0
  ) throw new Error("OPT-0069 rejected the arm thermal gate");
  return Object.freeze({
    ...value,
    observations: freezeObservations(value.observations),
  });
}

export function requireOpt0069ThermalTrace(
  value: Opt0069ThermalTrace,
  gate: Opt0069ThermalGate,
  cleanupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0069ThermalTrace {
  const inspection = inspectObservations(
    value.observations,
    gate.startedAtEpochMilliseconds,
    value.completedAtEpochMilliseconds,
  );
  if (
    value.source !== OPT_0069_THERMAL_SOURCE ||
    value.command !== OPT_0069_THERMAL_COMMAND ||
    !/^[0-9a-f]{64}$/u.test(value.rawTraceSha256) ||
    value.completedAtEpochMilliseconds < cleanupCompletedAtEpochMilliseconds ||
    value.completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    value.observationCount !== value.observations.length ||
    value.observationCount < gate.observationCount ||
    value.maximumPollGapMilliseconds !== inspection.maximumPollGapMilliseconds ||
    value.maximumPollGapMilliseconds > OPT_0069_MAXIMUM_POLL_GAP_MILLISECONDS ||
    value.nonNominalObservationCount !== inspection.nonNominalObservationCount ||
    gate.observations.some((item, index) =>
      !sameObservation(item, value.observations[index])
    ) ||
    value.transitions.some((item) =>
      !Number.isSafeInteger(item.atEpochMilliseconds) ||
      !Number.isSafeInteger(item.level) || item.level < 0 ||
      item.atEpochMilliseconds < gate.completedAtEpochMilliseconds ||
      item.atEpochMilliseconds > value.completedAtEpochMilliseconds
    )
  ) throw new Error("OPT-0069 rejected the through-cleanup thermal trace");
  return Object.freeze({
    ...value,
    observations: freezeObservations(value.observations),
    transitions: Object.freeze(value.transitions.map((item) =>
      Object.freeze({ ...item })
    )),
  });
}

export interface Opt0069TimingSample {
  readonly armId: Opt0069ArmId;
  readonly order: number;
  readonly owner: Opt0069Owner;
  readonly wallMs: number;
  readonly readCopyMs: number;
  readonly hashMs: number;
  readonly finalizationAndComparisonMs: number;
  readonly cleanupMs: number;
  readonly uploadSubsetWallMs: number;
  readonly matchedUniqueDigests: number;
  readonly logicalRecordsCovered: number;
  readonly physicalBytes: number;
  readonly logicalBytes: number;
  readonly uploadSubsetFiles: number;
  readonly uploadSubsetBytes: number;
  readonly maximumExplicitLivePayloadBytes: number;
  readonly maximumExplicitLivePayloadCount: number;
  readonly conservativeTransientBytes: number;
  readonly maximumWorkerHeartbeatGapMs: number;
  readonly maximumPageHeartbeatGapMs: number;
  readonly inventoryFingerprint: string;
  readonly correctnessPassed: boolean;
  readonly boundedMemoryPassed: boolean;
  readonly abortPassed: boolean;
  readonly cleanupPassed: boolean;
  readonly thermalNonNominalObservations: number;
}

export interface Opt0069PerformanceSummary {
  readonly candidateMedianMs: number;
  readonly candidateThroughputBytesPerSecond: number;
  readonly forwardMedianSavingMs: number;
  readonly reverseMedianSavingMs: number;
  readonly maximumReadCopyRegression: number;
  readonly maximumResponsivenessRegression: number;
  readonly allSamplesExact: boolean;
  readonly allSamplesNominal: boolean;
  readonly passed: boolean;
}

export function summarizeOpt0069Performance(
  samples: readonly Opt0069TimingSample[],
): Opt0069PerformanceSummary {
  if (
    samples.length !== OPT_0069_ARM_ORDER.length ||
    samples.some((sample, index) => {
      const expected = OPT_0069_ARM_ORDER[index]!;
      return sample.armId !== expected.armId || sample.order !== expected.order ||
        sample.owner !== expected.owner || numericSampleFields(sample).some(
          (value) => !Number.isFinite(value) || value < 0,
        ) || !/^[0-9a-f]{64}$/u.test(sample.inventoryFingerprint);
    })
  ) throw new Error("OPT-0069 balanced sample inventory changed");
  const byId = new Map(samples.map((sample) => [sample.armId, sample]));
  const a1 = byId.get("A1")!;
  const b1 = byId.get("B1")!;
  const a2 = byId.get("A2")!;
  const b2 = byId.get("B2")!;
  const a3 = byId.get("A3")!;
  const b3 = byId.get("B3")!;
  const a4 = byId.get("A4")!;
  const b4 = byId.get("B4")!;
  const candidateMedianMs = median([b1.wallMs, b2.wallMs, b3.wallMs, b4.wallMs]);
  const candidateThroughputBytesPerSecond =
    OPT_0069_COMPLETE_PHYSICAL_BYTES / (candidateMedianMs / 1_000);
  const forwardMedianSavingMs = median([
    a1.wallMs - b1.wallMs,
    a4.wallMs - b4.wallMs,
  ]);
  const reverseMedianSavingMs = median([
    a2.wallMs - b2.wallMs,
    a3.wallMs - b3.wallMs,
  ]);
  const pairs = [[a1, b1], [a2, b2], [a3, b3], [a4, b4]] as const;
  const maximumReadCopyRegression = Math.max(
    0,
    ...pairs.map(([control, candidate]) =>
      regression(control.readCopyMs, candidate.readCopyMs)
    ),
  );
  const maximumResponsivenessRegression = Math.max(
    0,
    ...pairs.flatMap(([control, candidate]) => [
      regression(
        control.maximumWorkerHeartbeatGapMs,
        candidate.maximumWorkerHeartbeatGapMs,
      ),
      regression(
        control.maximumPageHeartbeatGapMs,
        candidate.maximumPageHeartbeatGapMs,
      ),
    ]),
  );
  const fingerprint = samples[0]!.inventoryFingerprint;
  const allSamplesExact = samples.every((sample) =>
    sample.correctnessPassed && sample.boundedMemoryPassed &&
    sample.abortPassed && sample.cleanupPassed &&
    sample.matchedUniqueDigests === OPT_0069_COMPLETE_UNIQUE_DIGESTS &&
    sample.logicalRecordsCovered === OPT_0069_COMPLETE_LOGICAL_RECORDS &&
    sample.physicalBytes === OPT_0069_COMPLETE_PHYSICAL_BYTES &&
    sample.logicalBytes === OPT_0069_COMPLETE_LOGICAL_BYTES &&
    sample.uploadSubsetFiles === OPT_0069_UPLOAD_SUBSET_FILES &&
    sample.uploadSubsetBytes === OPT_0069_UPLOAD_SUBSET_BYTES &&
    sample.inventoryFingerprint === fingerprint
  );
  const allSamplesNominal = samples.every((sample) =>
    sample.thermalNonNominalObservations === 0
  );
  const passed = allSamplesExact && allSamplesNominal &&
    candidateMedianMs <= OPT_0069_MAXIMUM_CANDIDATE_MEDIAN_MS &&
    candidateThroughputBytesPerSecond >=
      OPT_0069_MINIMUM_THROUGHPUT_BYTES_PER_SECOND &&
    forwardMedianSavingMs >= OPT_0069_MINIMUM_DIRECTIONAL_SAVING_MS &&
    reverseMedianSavingMs >= OPT_0069_MINIMUM_DIRECTIONAL_SAVING_MS &&
    maximumReadCopyRegression <= OPT_0069_MAXIMUM_READ_COPY_REGRESSION &&
    maximumResponsivenessRegression <=
      OPT_0069_MAXIMUM_RESPONSIVENESS_REGRESSION;
  return Object.freeze({
    candidateMedianMs,
    candidateThroughputBytesPerSecond,
    forwardMedianSavingMs,
    reverseMedianSavingMs,
    maximumReadCopyRegression,
    maximumResponsivenessRegression,
    allSamplesExact,
    allSamplesNominal,
    passed,
  });
}

export function checkedOpt0069ByteAdd(left: number, right: number): number {
  const value = left + right;
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError("OPT-0069 byte accounting exceeds the safe range");
  }
  return value;
}

export function serializeOpt0069Failure(
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

function requiredString(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name)?.trim();
  if (value === undefined || value === "") {
    throw new Error(`OPT-0069 requires URL parameter ${name}`);
  }
  return value;
}

function requiredInteger(parameters: URLSearchParams, name: string): number {
  const text = requiredString(parameters, name);
  if (!/^[0-9]+$/u.test(text)) {
    throw new Error(`OPT-0069 URL parameter ${name} must be an integer`);
  }
  const value = Number(text);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`OPT-0069 URL parameter ${name} is outside the safe range`);
  }
  return value;
}

function samePackageInventory(
  left: Opt0069PackageInventory,
  right: Opt0069PackageInventory,
): boolean {
  return left.packageKind === right.packageKind &&
    left.logicalRecords === right.logicalRecords &&
    left.uniqueDigests === right.uniqueDigests &&
    left.logicalBytes === right.logicalBytes &&
    left.physicalBytes === right.physicalBytes;
}

function inspectObservations(
  observations: readonly Opt0069ThermalObservation[],
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
  ) throw new Error("OPT-0069 thermal observation boundary changed");
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
    ) throw new Error("OPT-0069 thermal observation is invalid");
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
  observations: readonly Opt0069ThermalObservation[],
): readonly Opt0069ThermalObservation[] {
  return Object.freeze(observations.map((item) => Object.freeze({ ...item })));
}

function sameObservation(
  left: Opt0069ThermalObservation,
  right: Opt0069ThermalObservation | undefined,
): boolean {
  return right !== undefined &&
    left.atEpochMilliseconds === right.atEpochMilliseconds &&
    left.level === right.level && left.rawValue === right.rawValue;
}

function numericSampleFields(sample: Opt0069TimingSample): readonly number[] {
  return [
    sample.wallMs,
    sample.readCopyMs,
    sample.hashMs,
    sample.finalizationAndComparisonMs,
    sample.cleanupMs,
    sample.uploadSubsetWallMs,
    sample.matchedUniqueDigests,
    sample.logicalRecordsCovered,
    sample.physicalBytes,
    sample.logicalBytes,
    sample.uploadSubsetFiles,
    sample.uploadSubsetBytes,
    sample.maximumExplicitLivePayloadBytes,
    sample.maximumExplicitLivePayloadCount,
    sample.conservativeTransientBytes,
    sample.maximumWorkerHeartbeatGapMs,
    sample.maximumPageHeartbeatGapMs,
    sample.thermalNonNominalObservations,
  ];
}

function median(values: readonly number[]): number {
  if (values.length === 0) throw new Error("OPT-0069 median requires samples");
  const sorted = [...values].sort((left, right) => left - right);
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
