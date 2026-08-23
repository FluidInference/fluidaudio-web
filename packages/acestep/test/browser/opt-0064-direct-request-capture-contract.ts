export const OPT_0064_THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
export const OPT_0064_THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
export const OPT_0064_THERMAL_POLL_MILLISECONDS = 1_000 as const;
export const OPT_0064_MINIMUM_NOMINAL_MILLISECONDS = 30_000 as const;
export const OPT_0064_MAXIMUM_POLL_GAP_MILLISECONDS = 1_250 as const;
export const OPT_0064_MAXIMUM_GATE_HANDOFF_MILLISECONDS = 5_000 as const;
export const OPT_0064_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024;
export const OPT_0064_REQUEST_CANONICAL_JSON =
  '{"generationProfile":"ace-turbo-v1-correctness","prompt":"Warm analog synth arpeggios over a restrained breakbeat, rounded electric bass, airy pads, instrumental, detailed stereo production.","lyrics":"","instrumental":true,"durationSeconds":12,"seed":"0000000000c0ffee","planner":{"mode":"disabled"},"metadata":{"bpm":104,"keyScale":"D minor","timeSignature":"4"}}' as const;
export const OPT_0064_REQUEST_BYTE_LENGTH = 365 as const;
export const OPT_0064_REQUEST_SHA256 =
  "9ebde5a84b47167f5ed08005892adb694d1840b8848eb3f05368bb531dc30887" as const;
export const OPT_0064_ACCEPTED_WAV_SHA256 =
  "d085b6907c9872667412d6dcecfeee47b76c8038eb2bfbec615931b2d7365477" as const;
export const OPT_0064_ACCEPTED_WAV_AUTHORITY =
  "owner-approved-stage1-packed-bf16-fp32-candidate-a" as const;
export const OPT_0064_OUTPUT_FRAMES = 576_000 as const;
export const OPT_0064_OUTPUT_DATA_BYTES = 4_608_000 as const;
export const OPT_0064_OUTPUT_WAV_BYTES = 4_608_044 as const;

export type Opt0064UploadPhaseId =
  | "text"
  | "conditioner-constants"
  | "dit"
  | "vae";

export const OPT_0064_UPLOAD_PHASES = Object.freeze([
  Object.freeze({
    id: "text" as const,
    fileCount: 31,
    bytes: 1_191_553_024,
    queueDrains: 34,
    queueEmptyGaps: 3,
  }),
  Object.freeze({
    id: "conditioner-constants" as const,
    fileCount: 14,
    bytes: 1_220_575_232,
    queueDrains: 26,
    queueEmptyGaps: 12,
  }),
  Object.freeze({
    id: "dit" as const,
    fileCount: 50,
    bytes: 3_150_917_888,
    queueDrains: 75,
    queueEmptyGaps: 25,
  }),
  Object.freeze({
    id: "vae" as const,
    fileCount: 7,
    bytes: 168_791_552,
    queueDrains: 8,
    queueEmptyGaps: 1,
  }),
]);

export const OPT_0064_TOTAL_UPLOAD_FILES = 102 as const;
export const OPT_0064_TOTAL_UPLOAD_BYTES = 5_731_837_696 as const;
export const OPT_0064_TOTAL_UPLOAD_DRAINS = 143 as const;
export const OPT_0064_TOTAL_UPLOAD_GAPS = 41 as const;

export interface Opt0064RunIdentity {
  readonly coreCommit: string;
  readonly harnessCommit: string;
  readonly machineModel: string;
  readonly osVersion: string;
  readonly osBuild: string;
  readonly browserVersion: string;
  readonly gpuCoreCount: number;
  readonly memoryBytes: number;
}

export function parseOpt0064RunIdentity(
  parameters: URLSearchParams,
): Opt0064RunIdentity {
  return validateOpt0064RunIdentity({
    coreCommit: requiredString(parameters, "coreCommit"),
    harnessCommit: requiredString(parameters, "harnessCommit"),
    machineModel: requiredString(parameters, "machineModel"),
    osVersion: requiredString(parameters, "osVersion"),
    osBuild: requiredString(parameters, "osBuild"),
    browserVersion: requiredString(parameters, "browserVersion"),
    gpuCoreCount: requiredInteger(parameters, "gpuCoreCount"),
    memoryBytes: requiredInteger(parameters, "memoryBytes"),
  });
}

export function validateOpt0064RunIdentity(
  value: Opt0064RunIdentity,
): Opt0064RunIdentity {
  if (
    !isRecord(value) || !isCommit(value.coreCommit) ||
    !isCommit(value.harnessCommit) || !isNonempty(value.machineModel) ||
    !isNonempty(value.osVersion) || !isNonempty(value.osBuild) ||
    !isNonempty(value.browserVersion) ||
    !Number.isSafeInteger(value.gpuCoreCount) || value.gpuCoreCount < 1 ||
    !Number.isSafeInteger(value.memoryBytes) || value.memoryBytes < 1
  ) throw new Error("OPT-0064 requires a complete frozen run identity");
  return Object.freeze({ ...value });
}

export function serializeOpt0064Failure(
  error: unknown,
): Readonly<Record<string, unknown>> {
  return serializeFailure(error, 0, new Set<object>());
}

const MAXIMUM_SERIALIZED_FAILURE_DEPTH = 4;
const MAXIMUM_SERIALIZED_FAILURE_CHILDREN = 8;

function serializeFailure(
  error: unknown,
  depth: number,
  seen: Set<object>,
): Readonly<Record<string, unknown>> {
  const object = error !== null &&
      (typeof error === "object" || typeof error === "function")
    ? error as object
    : undefined;
  if (object !== undefined && seen.has(object)) {
    return Object.freeze({
      name: safeErrorText(error, "name", "Error", 1_024),
      message: "Cyclic nested failure omitted",
      cyclic: true,
    });
  }
  if (object !== undefined) seen.add(object);
  const name = safeErrorText(error, "name", "Error", 1_024);
  const message = safeErrorText(
    error,
    "message",
    safeString(error, "Unknown OPT-0064 failure"),
    4_096,
  );
  const stack = safeRead(error, "stack");
  const code = safeRead(error, "code");
  const errors = safeRead(error, "errors");
  const cause = safeRead(error, "cause");
  const mayDescend = depth < MAXIMUM_SERIALIZED_FAILURE_DEPTH;
  const nestedErrors = Array.isArray(errors)
    ? (() => {
        const children = errors.slice(0, MAXIMUM_SERIALIZED_FAILURE_CHILDREN);
        return Object.freeze({
          errorCount: errors.length,
          errors: Object.freeze(mayDescend
            ? children.map((child) =>
                serializeFailure(child, depth + 1, seen)
              )
            : []),
          ...(errors.length > children.length
            ? { errorsTruncated: true }
            : {}),
          ...(!mayDescend && errors.length !== 0
            ? { nestedErrorsOmitted: true }
            : {}),
        });
      })()
    : undefined;
  return Object.freeze({
    name,
    message,
    ...(code === undefined
      ? {}
      : { code: boundedText(String(code), 1_024) }),
    ...(typeof stack === "string" && stack !== ""
      ? { stack: boundedText(stack, 8_192) }
      : {}),
    ...(nestedErrors === undefined ? {} : nestedErrors),
    ...(cause === undefined
      ? {}
      : mayDescend
        ? { cause: serializeFailure(cause, depth + 1, seen) }
        : { nestedCauseOmitted: true }),
  });
}

export interface Opt0064ThermalObservation {
  readonly atEpochMilliseconds: number;
  readonly level: number;
  /** Exact trimmed stdout from the accepted notifyutil command. */
  readonly rawValue: string;
}

export interface Opt0064ThermalGate {
  readonly source: typeof OPT_0064_THERMAL_SOURCE;
  readonly command: typeof OPT_0064_THERMAL_COMMAND;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly observations: readonly Opt0064ThermalObservation[];
}

export interface Opt0064ThermalTrace {
  readonly source: typeof OPT_0064_THERMAL_SOURCE;
  readonly command: typeof OPT_0064_THERMAL_COMMAND;
  readonly rawTraceSha256: string;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: number;
  readonly observations: readonly Opt0064ThermalObservation[];
  readonly transitions: readonly Readonly<{
    readonly atEpochMilliseconds: number;
    readonly level: number;
  }>[];
}

export function requireOpt0064ThermalGate(
  value: Opt0064ThermalGate,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0064ThermalGate {
  const duration = value.completedAtEpochMilliseconds -
    value.startedAtEpochMilliseconds;
  const inspected = inspectObservations(
    value.observations,
    value.startedAtEpochMilliseconds,
    value.completedAtEpochMilliseconds,
  );
  if (
    value.source !== OPT_0064_THERMAL_SOURCE ||
    value.command !== OPT_0064_THERMAL_COMMAND ||
    !Number.isFinite(readyAtEpochMilliseconds) ||
    value.startedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    value.completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    nowEpochMilliseconds - value.completedAtEpochMilliseconds >
      OPT_0064_MAXIMUM_GATE_HANDOFF_MILLISECONDS ||
    duration < OPT_0064_MINIMUM_NOMINAL_MILLISECONDS ||
    !Number.isSafeInteger(value.observationCount) ||
    value.observationCount <
      Math.floor(duration / OPT_0064_THERMAL_POLL_MILLISECONDS) + 1 ||
    value.observationCount !== value.observations.length ||
    !Number.isFinite(value.maximumPollGapMilliseconds) ||
    value.maximumPollGapMilliseconds < 0 ||
    value.maximumPollGapMilliseconds >
      OPT_0064_MAXIMUM_POLL_GAP_MILLISECONDS ||
    value.maximumPollGapMilliseconds !== inspected.maximumPollGapMilliseconds ||
    value.nonNominalObservationCount !== 0 ||
    inspected.nonNominalObservationCount !== 0
  ) throw new Error("OPT-0064 rejected the direct-request thermal gate");
  return Object.freeze({
    ...value,
    observations: freezeObservations(value.observations),
  });
}

export function requireOpt0064ThermalTrace(
  value: Opt0064ThermalTrace,
  gate: Opt0064ThermalGate,
  cleanupCompletedAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0064ThermalTrace {
  const inspected = inspectObservations(
    value.observations,
    gate.startedAtEpochMilliseconds,
    value.completedAtEpochMilliseconds,
  );
  const transitions = transitionsFor(value.observations);
  if (
    value.source !== OPT_0064_THERMAL_SOURCE ||
    value.command !== OPT_0064_THERMAL_COMMAND ||
    !/^[0-9a-f]{64}$/u.test(value.rawTraceSha256) ||
    value.completedAtEpochMilliseconds < cleanupCompletedAtEpochMilliseconds ||
    value.completedAtEpochMilliseconds > nowEpochMilliseconds + 1_000 ||
    !Number.isSafeInteger(value.observationCount) ||
    value.observationCount !== value.observations.length ||
    value.observationCount < gate.observationCount ||
    !Number.isFinite(value.maximumPollGapMilliseconds) ||
    value.maximumPollGapMilliseconds < 0 ||
    value.maximumPollGapMilliseconds >
      OPT_0064_MAXIMUM_POLL_GAP_MILLISECONDS ||
    value.maximumPollGapMilliseconds !== inspected.maximumPollGapMilliseconds ||
    !Number.isSafeInteger(value.nonNominalObservationCount) ||
    value.nonNominalObservationCount < 0 ||
    value.nonNominalObservationCount !== inspected.nonNominalObservationCount ||
    gate.observations.some((item, index) =>
      !sameObservation(item, value.observations[index])
    ) ||
    transitions.length !== value.transitions.length ||
    transitions.some((transition, index) => {
      const supplied = value.transitions[index];
      return supplied === undefined ||
        transition.atEpochMilliseconds !== supplied.atEpochMilliseconds ||
        transition.level !== supplied.level;
    })
  ) throw new Error("OPT-0064 rejected the through-cleanup thermal trace");
  return Object.freeze({
    ...value,
    observations: freezeObservations(value.observations),
    transitions: Object.freeze(value.transitions.map((item) =>
      Object.freeze({ ...item })
    )),
  });
}

function inspectObservations(
  observations: readonly Opt0064ThermalObservation[],
  startedAtEpochMilliseconds: number,
  completedAtEpochMilliseconds: number,
): Readonly<{
  maximumPollGapMilliseconds: number;
  nonNominalObservationCount: number;
}> {
  if (
    !Array.isArray(observations) || observations.length === 0 ||
    !Number.isFinite(startedAtEpochMilliseconds) ||
    !Number.isFinite(completedAtEpochMilliseconds) ||
    completedAtEpochMilliseconds < startedAtEpochMilliseconds
  ) throw new Error("OPT-0064 thermal observations are malformed");
  let maximumPollGapMilliseconds = 0;
  let nonNominalObservationCount = 0;
  for (let index = 0; index < observations.length; index += 1) {
    const observation = observations[index]!;
    const prior = observations[index - 1];
    if (
      !Number.isFinite(observation.atEpochMilliseconds) ||
      observation.atEpochMilliseconds < startedAtEpochMilliseconds ||
      observation.atEpochMilliseconds > completedAtEpochMilliseconds ||
      (prior !== undefined &&
        observation.atEpochMilliseconds <= prior.atEpochMilliseconds) ||
      !Number.isSafeInteger(observation.level) || observation.level < 0 ||
      observation.rawValue.trim() !== String(observation.level)
    ) throw new Error("OPT-0064 thermal observation inventory changed");
    if (prior !== undefined) {
      maximumPollGapMilliseconds = Math.max(
        maximumPollGapMilliseconds,
        observation.atEpochMilliseconds - prior.atEpochMilliseconds,
      );
    }
    if (observation.level !== 0) nonNominalObservationCount += 1;
  }
  return Object.freeze({
    maximumPollGapMilliseconds,
    nonNominalObservationCount,
  });
}

function transitionsFor(
  observations: readonly Opt0064ThermalObservation[],
): readonly Readonly<{ atEpochMilliseconds: number; level: number }>[] {
  const transitions: Readonly<{
    atEpochMilliseconds: number;
    level: number;
  }>[] = [];
  for (let index = 0; index < observations.length; index += 1) {
    const observation = observations[index]!;
    if (index === 0 || observation.level !== observations[index - 1]!.level) {
      transitions.push(Object.freeze({
        atEpochMilliseconds: observation.atEpochMilliseconds,
        level: observation.level,
      }));
    }
  }
  return Object.freeze(transitions);
}

function freezeObservations(
  observations: readonly Opt0064ThermalObservation[],
): readonly Opt0064ThermalObservation[] {
  return Object.freeze(observations.map((item) => Object.freeze({ ...item })));
}

function sameObservation(
  left: Opt0064ThermalObservation,
  right: Opt0064ThermalObservation | undefined,
): boolean {
  return right !== undefined &&
    left.atEpochMilliseconds === right.atEpochMilliseconds &&
    left.level === right.level && left.rawValue === right.rawValue;
}

function requiredString(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name)?.trim();
  if (value === undefined || value === "") {
    throw new Error(`OPT-0064 identity field ${name} is missing`);
  }
  return value;
}

function requiredInteger(parameters: URLSearchParams, name: string): number {
  const value = Number(requiredString(parameters, name));
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`OPT-0064 identity field ${name} must be a positive integer`);
  }
  return value;
}

function isCommit(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{40}$/u.test(value);
}

function isNonempty(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function safeErrorText(
  value: unknown,
  property: string,
  fallback: string,
  limit: number,
): string {
  const read = safeRead(value, property);
  return boundedText(typeof read === "string" && read !== "" ? read : fallback, limit);
}

function safeRead(value: unknown, property: string): unknown {
  if ((typeof value !== "object" || value === null) && typeof value !== "function") {
    return undefined;
  }
  try {
    return (value as Record<string, unknown>)[property];
  } catch {
    return undefined;
  }
}

function safeString(value: unknown, fallback: string): string {
  try {
    return String(value);
  } catch {
    return fallback;
  }
}

function boundedText(value: string, limit: number): string {
  return value.length <= limit ? value : `${value.slice(0, limit)}…`;
}
